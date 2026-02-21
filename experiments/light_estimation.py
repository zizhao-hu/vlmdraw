"""
Light Source Estimation via Diffuse Reflection Model

Uses depth maps to compute surface normals, then applies the Lambertian
(diffuse) reflection model to estimate the dominant light direction from
observed brightness. In real images, the estimated light direction should
be globally consistent. AI-generated images may have physically impossible
lighting — different regions implying contradictory light sources.

Physics:
    B(x,y) = ρ(x,y) · max(n⃗(x,y) · l⃗, 0)

    B = observed brightness (luminance)
    ρ = surface albedo/reflectance
    n⃗ = surface normal (from depth gradient)
    l⃗ = light direction (unit vector, what we solve for)

Features extracted:
1. Global light direction (least-squares fit)
2. Per-patch light direction variance (consistency)
3. Lambertian model residual (how well physics fits)
4. Fraction of anomalous patches (contradictory lighting)
5. Light direction angular dispersion

Usage:
    python experiments/light_estimation.py \
        --real-dir data/aigenbench/real \
        --fake-dir data/aigenbench/fake \
        --output-dir results/light_estimation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats


# ─── Shared Utilities ──────────────────────────────────────────────

def load_depth_model(model_name: str, device: str):
    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def estimate_depth(img: Image.Image, processor, model, device: str) -> np.ndarray:
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


def rgb_to_luminance(img_array: np.ndarray) -> np.ndarray:
    """Convert RGB to luminance (BT.601). Returns HxW float32 in [0,1]."""
    return (
        0.299 * img_array[:, :, 0].astype(np.float32) +
        0.587 * img_array[:, :, 1].astype(np.float32) +
        0.114 * img_array[:, :, 2].astype(np.float32)
    ) / 255.0


# ─── Surface Normals from Depth ───────────────────────────────────

def depth_to_normals(depth: np.ndarray) -> np.ndarray:
    """
    Compute surface normals from a depth map.

    Assumes an orthographic camera for simplicity.
    Returns HxWx3 array of unit normal vectors (nx, ny, nz).
    nz is positive (pointing toward the camera).
    """
    # Depth gradients → partial derivatives of surface height
    # dz/dx and dz/dy
    dzdx = np.gradient(depth, axis=1)  # horizontal
    dzdy = np.gradient(depth, axis=0)  # vertical

    # Normal vector: n = (-dz/dx, -dz/dy, 1), then normalize
    # We negate because depth increases away from camera,
    # but surface rises toward camera
    normals = np.stack([-dzdx, -dzdy, np.ones_like(depth)], axis=-1)

    # Normalize to unit vectors
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = normals / norms

    return normals.astype(np.float32)


# ─── Light Direction Estimation ───────────────────────────────────

def estimate_light_direction(
    brightness: np.ndarray,
    normals: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Estimate light direction using least-squares on the Lambertian model.

    B(x,y) ≈ n⃗(x,y) · l⃗   (ignoring albedo, clamping, ambient)

    We solve: min ||N @ l - b||² for l ∈ R³

    Args:
        brightness: HxW luminance array
        normals: HxWx3 surface normals
        mask: Optional HxW boolean mask (True = valid pixels)

    Returns:
        (light_dir, residual): unit light direction and normalized residual
    """
    h, w = brightness.shape

    if mask is None:
        # Exclude very dark and very bright pixels (saturated / shadow)
        mask = (brightness > 0.05) & (brightness < 0.95)

    # Flatten
    b = brightness[mask].ravel()
    N = normals[mask].reshape(-1, 3)

    if len(b) < 10:
        return np.array([0.0, 0.0, 1.0]), 1.0

    # Subsample for speed if too many pixels
    if len(b) > 100000:
        idx = np.random.choice(len(b), 100000, replace=False)
        b = b[idx]
        N = N[idx]

    # Least-squares solve: N @ l ≈ b
    # Using pseudoinverse: l = (N^T N)^{-1} N^T b
    try:
        result = np.linalg.lstsq(N, b, rcond=None)
        l = result[0]
        residuals = result[1]

        # Normalize to unit vector
        l_norm = np.linalg.norm(l)
        if l_norm < 1e-8:
            return np.array([0.0, 0.0, 1.0]), 1.0
        l_unit = l / l_norm

        # Compute normalized residual
        prediction = N @ l
        total_var = np.var(b) + 1e-8
        residual_var = np.var(b - prediction)
        r_squared = 1.0 - residual_var / total_var
        normalized_residual = float(residual_var / total_var)

        return l_unit, normalized_residual

    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0, 1.0]), 1.0


def estimate_light_per_patch(
    brightness: np.ndarray,
    normals: np.ndarray,
    block_size: int = 64,
    min_valid_fraction: float = 0.3,
) -> list[dict]:
    """
    Estimate light direction in each local patch.

    Returns list of patch results with:
    - light_dir: estimated light direction
    - residual: Lambertian fit residual
    - center: (cx, cy) patch center coordinates
    """
    h, w = brightness.shape
    patches = []

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            b_patch = brightness[y:y+block_size, x:x+block_size]
            n_patch = normals[y:y+block_size, x:x+block_size]

            # Need enough valid (non-saturated) pixels
            mask = (b_patch > 0.05) & (b_patch < 0.95)
            if mask.sum() < block_size * block_size * min_valid_fraction:
                continue

            # Need enough surface normal variation
            n_flat = n_patch[mask].reshape(-1, 3)
            if n_flat.std(axis=0).max() < 0.01:
                continue  # flat surface, can't estimate light

            light_dir, residual = estimate_light_direction(b_patch, n_patch, mask)

            patches.append({
                "light_dir": light_dir,
                "residual": residual,
                "center": (x + block_size // 2, y + block_size // 2),
            })

    return patches


# ─── Feature Extraction ───────────────────────────────────────────

def extract_light_features(
    brightness: np.ndarray,
    depth: np.ndarray,
    block_size: int = 64,
) -> dict:
    """
    Extract all light-estimation features for one image.

    Returns dict of scalar features for real/fake comparison.
    """
    normals = depth_to_normals(depth)

    # Global light estimation
    global_light, global_residual = estimate_light_direction(brightness, normals)

    # Per-patch light estimation
    patches = estimate_light_per_patch(brightness, normals, block_size)

    if len(patches) < 3:
        return {
            "global_light_x": float(global_light[0]),
            "global_light_y": float(global_light[1]),
            "global_light_z": float(global_light[2]),
            "global_residual": float(global_residual),
            "n_valid_patches": len(patches),
            "light_dir_std_x": 0.0,
            "light_dir_std_y": 0.0,
            "light_dir_std_z": 0.0,
            "light_dir_angular_dispersion": 0.0,
            "light_dir_consistency": 0.0,
            "mean_patch_residual": 0.0,
            "std_patch_residual": 0.0,
            "frac_anomalous_patches": 0.0,
            "max_angular_deviation": 0.0,
            "median_angular_deviation": 0.0,
        }

    # Collect patch light directions
    light_dirs = np.array([p["light_dir"] for p in patches])
    patch_residuals = np.array([p["residual"] for p in patches])

    # Light direction variance (should be low for real images)
    light_std = light_dirs.std(axis=0)

    # Mean patch light direction
    mean_light = light_dirs.mean(axis=0)
    mean_light_norm = np.linalg.norm(mean_light)
    if mean_light_norm > 1e-8:
        mean_light = mean_light / mean_light_norm

    # Angular deviation of each patch from the global estimate
    # cos(θ) = l_patch · l_global
    cos_angles = np.clip(np.dot(light_dirs, global_light), -1.0, 1.0)
    angular_deviations = np.arccos(cos_angles)  # in radians
    angular_deviations_deg = np.degrees(angular_deviations)

    # Angular dispersion (circular variance)
    # R = ||mean of unit vectors|| → dispersion = 1 - R
    resultant_length = np.linalg.norm(light_dirs.mean(axis=0))
    angular_dispersion = 1.0 - resultant_length

    # Consistency: how well patches agree with global estimate
    light_consistency = float(cos_angles.mean())

    # Anomalous patches: patches where light direction deviates > 45° from global
    anomaly_threshold = 45.0  # degrees
    frac_anomalous = float((angular_deviations_deg > anomaly_threshold).mean())

    return {
        "global_light_x": float(global_light[0]),
        "global_light_y": float(global_light[1]),
        "global_light_z": float(global_light[2]),
        "global_residual": float(global_residual),
        "n_valid_patches": len(patches),
        "light_dir_std_x": float(light_std[0]),
        "light_dir_std_y": float(light_std[1]),
        "light_dir_std_z": float(light_std[2]),
        "light_dir_angular_dispersion": float(angular_dispersion),
        "light_dir_consistency": float(light_consistency),
        "mean_patch_residual": float(patch_residuals.mean()),
        "std_patch_residual": float(patch_residuals.std()),
        "frac_anomalous_patches": float(frac_anomalous),
        "max_angular_deviation": float(angular_deviations_deg.max()),
        "median_angular_deviation": float(np.median(angular_deviations_deg)),
    }


# ─── Visualization ────────────────────────────────────────────────

def save_visualization(
    img_array, brightness, depth, normals, patches,
    global_light, feats, viz_dir, stem, label,
):
    """Save a 6-panel visualization showing the light estimation."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        f"{label}: {stem}\n"
        f"Global light=({feats['global_light_x']:.2f}, {feats['global_light_y']:.2f}, {feats['global_light_z']:.2f})  "
        f"Residual={feats['global_residual']:.3f}  "
        f"Anomalous={feats['frac_anomalous_patches']:.1%}  "
        f"Consistency={feats['light_dir_consistency']:.3f}",
        fontsize=12, fontweight="bold",
    )

    # (0,0) Original image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # (0,1) Depth map
    im = axes[0, 1].imshow(depth, cmap="inferno")
    axes[0, 1].set_title("Depth Map")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

    # (0,2) Surface normals (RGB encoding: xyz → rgb)
    normal_vis = (normals + 1.0) / 2.0  # map [-1,1] to [0,1]
    axes[0, 2].imshow(normal_vis)
    axes[0, 2].set_title("Surface Normals (RGB = XYZ)")
    axes[0, 2].axis("off")

    # (1,0) Brightness with light direction arrows overlaid
    axes[1, 0].imshow(brightness, cmap="gray")
    axes[1, 0].set_title("Brightness + Patch Light Dirs")

    # Draw light direction arrows per patch
    for p in patches:
        cx, cy = p["center"]
        lx, ly = p["light_dir"][0], p["light_dir"][1]
        # Color by agreement with global direction
        cos_a = np.clip(np.dot(p["light_dir"], global_light), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_a))
        color = "lime" if angle_deg < 30 else ("yellow" if angle_deg < 60 else "red")
        scale = 25
        axes[1, 0].arrow(cx, cy, lx * scale, ly * scale,
                         head_width=5, head_length=3, fc=color, ec=color, alpha=0.8)
    axes[1, 0].axis("off")

    # (1,1) Lambertian reconstruction
    prediction = np.clip(np.sum(normals * global_light, axis=-1), 0, 1)
    axes[1, 1].imshow(prediction, cmap="gray")
    axes[1, 1].set_title("Lambertian Reconstruction")
    axes[1, 1].axis("off")

    # (1,2) Residual map (|observed - predicted|)
    residual_map = np.abs(brightness - prediction)
    im = axes[1, 2].imshow(residual_map, cmap="hot", vmin=0, vmax=0.5)
    axes[1, 2].set_title("Residual |B - N·L|")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(viz_dir / f"{stem}_light.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── Comparison Plots ─────────────────────────────────────────────

def create_comparison_plots(real_stats: list[dict], fake_stats: list[dict], output_dir: Path):
    """Create box plots comparing real vs fake light features."""
    scalar_keys = [
        "global_residual",
        "light_dir_angular_dispersion",
        "light_dir_consistency",
        "mean_patch_residual",
        "std_patch_residual",
        "frac_anomalous_patches",
        "max_angular_deviation",
        "median_angular_deviation",
        "light_dir_std_x",
        "light_dir_std_y",
        "light_dir_std_z",
        "n_valid_patches",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    fig.suptitle("Light Estimation Features: Real vs. AI-Generated", fontsize=16, fontweight="bold")

    for idx, key in enumerate(scalar_keys):
        ax = axes[idx // 4, idx % 4]
        real_vals = [s[key] for s in real_stats if key in s]
        fake_vals = [s[key] for s in fake_stats if key in s]

        if not real_vals and not fake_vals:
            ax.set_visible(False)
            continue

        data = []
        labels = []
        if real_vals:
            data.append(real_vals)
            labels.append("Real")
        if fake_vals:
            data.append(fake_vals)
            labels.append("Fake")

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        colors = ["#4CAF50", "#F44336"]
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(colors[i] if len(data) > 1 else colors[0])

        ax.set_title(key.replace("_", " ").title(), fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "light_comparison_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Angular deviation distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Light Direction Analysis: Real vs. Fake", fontsize=14)

    # Dispersion
    ax = axes[0]
    real_vals = [s["light_dir_angular_dispersion"] for s in real_stats]
    fake_vals = [s["light_dir_angular_dispersion"] for s in fake_stats]
    if real_vals:
        ax.hist(real_vals, bins=15, alpha=0.6, color="#4CAF50", label="Real", density=True)
    if fake_vals:
        ax.hist(fake_vals, bins=15, alpha=0.6, color="#F44336", label="Fake", density=True)
    ax.set_xlabel("Angular Dispersion (1 - R)")
    ax.set_ylabel("Density")
    ax.set_title("Light Direction Dispersion")
    ax.legend()
    ax.grid(alpha=0.3)

    # Anomalous fraction
    ax = axes[1]
    real_vals = [s["frac_anomalous_patches"] for s in real_stats]
    fake_vals = [s["frac_anomalous_patches"] for s in fake_stats]
    if real_vals:
        ax.hist(real_vals, bins=15, alpha=0.6, color="#4CAF50", label="Real", density=True)
    if fake_vals:
        ax.hist(fake_vals, bins=15, alpha=0.6, color="#F44336", label="Fake", density=True)
    ax.set_xlabel("Fraction of Anomalous Patches")
    ax.set_ylabel("Density")
    ax.set_title("Patches with Contradictory Lighting (>45°)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Global residual
    ax = axes[2]
    real_vals = [s["global_residual"] for s in real_stats]
    fake_vals = [s["global_residual"] for s in fake_stats]
    if real_vals:
        ax.hist(real_vals, bins=15, alpha=0.6, color="#4CAF50", label="Real", density=True)
    if fake_vals:
        ax.hist(fake_vals, bins=15, alpha=0.6, color="#F44336", label="Fake", density=True)
    ax.set_xlabel("Lambertian Residual (1 - R²)")
    ax.set_ylabel("Density")
    ax.set_title("How Well Lambertian Model Fits")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "light_distributions.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"📊 Comparison plots saved to {output_dir}")


# ─── Summary ──────────────────────────────────────────────────────

def print_summary(real_stats, fake_stats, output_dir):
    keys = [
        "global_residual",
        "light_dir_angular_dispersion",
        "light_dir_consistency",
        "mean_patch_residual",
        "frac_anomalous_patches",
        "max_angular_deviation",
        "median_angular_deviation",
        "light_dir_std_x",
        "light_dir_std_y",
    ]

    lines = []
    print("\n" + "=" * 80)
    print("LIGHT ESTIMATION ANALYSIS: DIFFUSE REFLECTION MODEL")
    print("=" * 80)
    print(f"\n{'Feature':<35} {'Real (mean±std)':<22} {'Fake (mean±std)':<22} {'Δ':>8}")
    print("-" * 90)
    lines.append(f"{'Feature':<35} | {'Real':<10} | {'Fake':<10} | {'Δ':<10}")

    for key in keys:
        real_vals = np.array([s[key] for s in real_stats if key in s])
        fake_vals = np.array([s[key] for s in fake_stats if key in s])
        if len(real_vals) > 0 and len(fake_vals) > 0:
            real_str = f"{real_vals.mean():.4f} ± {real_vals.std():.4f}"
            fake_str = f"{fake_vals.mean():.4f} ± {fake_vals.std():.4f}"
            delta = fake_vals.mean() - real_vals.mean()
            print(f"{key:<35} {real_str:<22} {fake_str:<22} {delta:>+.4f}")
            lines.append(f"{key:<35} | R={real_vals.mean():.4f} | F={fake_vals.mean():.4f} | D={delta:.4f}")
        elif len(real_vals) > 0:
            real_str = f"{real_vals.mean():.4f} ± {real_vals.std():.4f}"
            print(f"{key:<35} {real_str:<22} {'(no data)':<22}")
        elif len(fake_vals) > 0:
            fake_str = f"{fake_vals.mean():.4f} ± {fake_vals.std():.4f}"
            print(f"{key:<35} {'(no data)':<22} {fake_str:<22}")

    print()

    with open(output_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines))


# ─── Main Pipeline ────────────────────────────────────────────────

def process_directory(
    img_dir: Path, processor, depth_model, output_dir: Path,
    label: str, device: str, max_images: int = 50, block_size: int = 64,
) -> list[dict]:
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in extensions
    ])[:max_images]

    print(f"\nProcessing {len(image_files)} {label} images...")
    viz_dir = output_dir / f"{label}_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, img_path in enumerate(image_files):
        print(f"  [{i+1}/{len(image_files)}] {img_path.name}...", end=" ", flush=True)
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)

            brightness = rgb_to_luminance(img_array)
            depth = estimate_depth(img, processor, depth_model, device)
            normals = depth_to_normals(depth)

            feats = extract_light_features(brightness, depth, block_size)
            feats["filename"] = img_path.name
            feats["label"] = label

            # Visualize first 10 images
            if i < 10:
                patches = estimate_light_per_patch(brightness, normals, block_size)
                global_light = np.array([feats["global_light_x"], feats["global_light_y"], feats["global_light_z"]])
                save_visualization(
                    img_array, brightness, depth, normals, patches,
                    global_light, feats, viz_dir, img_path.stem, label,
                )

            results.append(feats)
            print(
                f"✓ (residual={feats['global_residual']:.3f}, "
                f"anomalous={feats['frac_anomalous_patches']:.1%}, "
                f"consistency={feats['light_dir_consistency']:.3f})"
            )

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Light source estimation via diffuse reflection model"
    )
    parser.add_argument("--real-dir", type=str, required=True)
    parser.add_argument("--fake-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/light_estimation")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=64,
                        help="Patch size for local light estimation")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    processor, depth_model = load_depth_model(args.model, device)

    real_stats = process_directory(
        Path(args.real_dir), processor, depth_model, output_dir,
        "real", device, args.max_images, args.block_size,
    )
    fake_stats = process_directory(
        Path(args.fake_dir), processor, depth_model, output_dir,
        "fake", device, args.max_images, args.block_size,
    )

    print_summary(real_stats, fake_stats, output_dir)

    if real_stats or fake_stats:
        create_comparison_plots(real_stats, fake_stats, output_dir)

    # Save raw stats
    all_stats = {"real": real_stats, "fake": fake_stats}
    with open(output_dir / "light_statistics.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"📁 Statistics saved to {output_dir / 'light_statistics.json'}")


if __name__ == "__main__":
    main()
