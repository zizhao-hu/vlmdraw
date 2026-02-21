"""
Brightness-Depth Consistency Analysis

Combines per-pixel brightness (luminance) with per-pixel depth to extract
features that capture the physical relationship between lighting and geometry.

Key insight: In real images, brightness and depth are physically linked
(inverse-square law, atmospheric scattering, shadow geometry). AI-generated
images may not maintain these physical consistencies.

Features extracted:
1. Global brightness-depth correlation
2. Depth-conditioned brightness statistics (brightness at each depth level)
3. Brightness gradient vs depth gradient consistency
4. Joint brightness-depth histogram (2D)
5. Local brightness-depth coherence (per-patch analysis)

Usage:
    python experiments/brightness_depth.py \
        --real-dir data/aigenbench/real \
        --fake-dir data/aigenbench/fake \
        --output-dir results/brightness_depth
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


# ─── Feature Extraction ────────────────────────────────────────────

def brightness_depth_correlation(brightness: np.ndarray, depth: np.ndarray) -> dict:
    """Global correlation between brightness and depth."""
    b_flat = brightness.ravel()
    d_flat = depth.ravel()

    # Pearson correlation
    pearson_r, pearson_p = sp_stats.pearsonr(b_flat, d_flat)

    # Spearman rank correlation (more robust)
    # Subsample for speed
    n = len(b_flat)
    if n > 50000:
        idx = np.random.choice(n, 50000, replace=False)
        b_sub, d_sub = b_flat[idx], d_flat[idx]
    else:
        b_sub, d_sub = b_flat, d_flat
    spearman_r, spearman_p = sp_stats.spearmanr(b_sub, d_sub)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


def depth_conditioned_brightness(brightness: np.ndarray, depth: np.ndarray, n_bins: int = 10) -> dict:
    """
    Statistics of brightness at different depth levels.
    Split depth into bins and compute brightness stats per bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_stds = []
    bin_skews = []

    for i in range(n_bins):
        mask = (depth >= bin_edges[i]) & (depth < bin_edges[i + 1])
        if mask.sum() < 10:
            bin_means.append(0.0)
            bin_stds.append(0.0)
            bin_skews.append(0.0)
            continue
        b_in_bin = brightness[mask]
        bin_means.append(float(b_in_bin.mean()))
        bin_stds.append(float(b_in_bin.std()))
        if b_in_bin.std() > 1e-6:
            centered = b_in_bin - b_in_bin.mean()
            bin_skews.append(float((centered ** 3).mean() / (b_in_bin.std() ** 3)))
        else:
            bin_skews.append(0.0)

    # How much does brightness change across depth bins?
    brightness_range_across_depth = max(bin_means) - min(bin_means) if bin_means else 0.0
    brightness_trend = np.polyfit(np.arange(len(bin_means)), bin_means, 1)[0] if len(bin_means) > 1 else 0.0

    return {
        "bin_means": bin_means,
        "bin_stds": bin_stds,
        "bin_skews": bin_skews,
        "brightness_range_across_depth": float(brightness_range_across_depth),
        "brightness_depth_slope": float(brightness_trend),
        "bin_std_mean": float(np.mean(bin_stds)),
        "bin_std_variation": float(np.std(bin_stds)),
    }


def gradient_consistency(brightness: np.ndarray, depth: np.ndarray) -> dict:
    """
    Compare brightness gradients with depth gradients.
    In real images, strong depth edges often (but not always) correspond to
    brightness edges. The consistency of this relationship may differ.
    """
    # Compute gradients
    b_gx = np.diff(brightness, axis=1)
    b_gy = np.diff(brightness, axis=0)
    d_gx = np.diff(depth, axis=1)
    d_gy = np.diff(depth, axis=0)

    # Align sizes
    h = min(b_gx.shape[0], b_gy.shape[0], d_gx.shape[0], d_gy.shape[0])
    w = min(b_gx.shape[1], b_gy.shape[1], d_gx.shape[1], d_gy.shape[1])

    b_gx, b_gy = b_gx[:h, :w], b_gy[:h, :w]
    d_gx, d_gy = d_gx[:h, :w], d_gy[:h, :w]

    b_mag = np.sqrt(b_gx ** 2 + b_gy ** 2)
    d_mag = np.sqrt(d_gx ** 2 + d_gy ** 2)

    # Correlation between gradient magnitudes
    b_flat = b_mag.ravel()
    d_flat = d_mag.ravel()
    if len(b_flat) > 50000:
        idx = np.random.choice(len(b_flat), 50000, replace=False)
        b_flat, d_flat = b_flat[idx], d_flat[idx]

    grad_corr, _ = sp_stats.pearsonr(b_flat, d_flat)

    # Gradient direction consistency (where both have significant gradients)
    b_angle = np.arctan2(b_gy, b_gx)
    d_angle = np.arctan2(d_gy, d_gx)

    # Only look at pixels with significant gradients
    threshold_b = np.percentile(b_mag, 75)
    threshold_d = np.percentile(d_mag, 75)
    strong_mask = (b_mag > threshold_b) & (d_mag > threshold_d)

    if strong_mask.sum() > 10:
        angle_diff = np.abs(b_angle[strong_mask] - d_angle[strong_mask])
        # Wrap to [0, pi]
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        angle_diff = np.minimum(angle_diff, np.pi)
        direction_consistency = float(np.cos(angle_diff).mean())
    else:
        direction_consistency = 0.0

    # Ratio statistics: where depth changes a lot, how much does brightness change?
    high_depth_grad = d_mag > np.percentile(d_mag, 90)
    if high_depth_grad.sum() > 10:
        brightness_at_depth_edges = float(b_mag[high_depth_grad].mean())
    else:
        brightness_at_depth_edges = 0.0

    return {
        "grad_magnitude_correlation": float(grad_corr),
        "grad_direction_consistency": float(direction_consistency),
        "brightness_at_depth_edges": float(brightness_at_depth_edges),
        "mean_brightness_grad": float(b_mag.mean()),
        "mean_depth_grad": float(d_mag.mean()),
    }


def joint_histogram_features(brightness: np.ndarray, depth: np.ndarray, n_bins: int = 16) -> dict:
    """
    Compute features from the 2D joint brightness-depth histogram.
    """
    hist2d, _, _ = np.histogram2d(
        brightness.ravel(), depth.ravel(),
        bins=n_bins, range=[[0, 1], [0, 1]]
    )
    hist2d = hist2d / (hist2d.sum() + 1e-8)

    # Joint entropy
    h = hist2d[hist2d > 0]
    joint_entropy = -np.sum(h * np.log2(h))

    # Mutual information (approximate)
    b_marginal = hist2d.sum(axis=1)
    d_marginal = hist2d.sum(axis=0)
    outer = np.outer(b_marginal, d_marginal) + 1e-12
    mi_terms = hist2d[hist2d > 0] * np.log2(hist2d[hist2d > 0] / outer[hist2d > 0])
    mutual_info = float(np.sum(mi_terms))

    # Concentration: how spread out is the joint distribution?
    max_prob = float(hist2d.max())
    n_occupied_bins = int((hist2d > 0.001).sum())

    return {
        "joint_entropy": float(joint_entropy),
        "mutual_information": float(mutual_info),
        "max_joint_prob": max_prob,
        "n_occupied_bins": n_occupied_bins,
    }


def local_coherence(brightness: np.ndarray, depth: np.ndarray, block_size: int = 32) -> dict:
    """
    Compute local brightness-depth correlation in patches.
    Real images should have more consistent local correlations.
    """
    h, w = brightness.shape
    correlations = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            b_block = brightness[y:y+block_size, x:x+block_size].ravel()
            d_block = depth[y:y+block_size, x:x+block_size].ravel()
            if b_block.std() > 1e-6 and d_block.std() > 1e-6:
                r, _ = sp_stats.pearsonr(b_block, d_block)
                correlations.append(r)

    if not correlations:
        return {
            "local_corr_mean": 0.0,
            "local_corr_std": 0.0,
            "local_corr_abs_mean": 0.0,
            "frac_positive_corr": 0.0,
            "frac_strong_corr": 0.0,
        }

    corrs = np.array(correlations)
    return {
        "local_corr_mean": float(corrs.mean()),
        "local_corr_std": float(corrs.std()),
        "local_corr_abs_mean": float(np.abs(corrs).mean()),
        "frac_positive_corr": float((corrs > 0).mean()),
        "frac_strong_corr": float((np.abs(corrs) > 0.5).mean()),
    }


# ─── Main Pipeline ─────────────────────────────────────────────────

def extract_all_features(brightness: np.ndarray, depth: np.ndarray) -> dict:
    """Extract all brightness-depth features for one image."""
    feats = {}
    feats.update(brightness_depth_correlation(brightness, depth))
    feats.update(depth_conditioned_brightness(brightness, depth))
    feats.update(gradient_consistency(brightness, depth))
    feats.update(joint_histogram_features(brightness, depth))
    feats.update(local_coherence(brightness, depth))
    return feats


def process_directory(
    img_dir: Path, processor, model, output_dir: Path,
    label: str, device: str, max_images: int = 50,
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
            depth = estimate_depth(img, processor, model, device)

            feats = extract_all_features(brightness, depth)
            feats["filename"] = img_path.name
            feats["label"] = label

            # Save visualization (first 10 images)
            if i < 10:
                save_visualization(img_array, brightness, depth, feats, viz_dir, img_path.stem, label)

            results.append(feats)
            print(f"✓ (pearson={feats['pearson_r']:.3f}, grad_corr={feats['grad_magnitude_correlation']:.3f}, MI={feats['mutual_information']:.3f})")

        except Exception as e:
            print(f"✗ Error: {e}")

    return results


def save_visualization(img_array, brightness, depth, feats, viz_dir, stem, label):
    """Save a 4-panel visualization: image, brightness, depth, scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"{label}: {stem}\nPearson r={feats['pearson_r']:.3f}, Grad corr={feats['grad_magnitude_correlation']:.3f}, MI={feats['mutual_information']:.3f}",
                 fontsize=13, fontweight="bold")

    # Original image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Brightness
    axes[0, 1].imshow(brightness, cmap="gray")
    axes[0, 1].set_title("Brightness (Luminance)")
    axes[0, 1].axis("off")

    # Depth
    im = axes[1, 0].imshow(depth, cmap="inferno")
    axes[1, 0].set_title("Depth Map")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

    # Brightness vs Depth scatter (subsampled)
    b_flat = brightness.ravel()
    d_flat = depth.ravel()
    n = len(b_flat)
    idx = np.random.choice(n, min(5000, n), replace=False)
    axes[1, 1].scatter(d_flat[idx], b_flat[idx], alpha=0.15, s=2, c="steelblue")
    axes[1, 1].set_xlabel("Depth")
    axes[1, 1].set_ylabel("Brightness")
    axes[1, 1].set_title(f"Brightness vs Depth (r={feats['pearson_r']:.3f})")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / f"{stem}_bd.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_comparison_plots(real_stats: list[dict], fake_stats: list[dict], output_dir: Path):
    """Create comparison plots between real and fake brightness-depth features."""

    # --- Box plots for key scalar features ---
    scalar_keys = [
        "pearson_r", "spearman_r",
        "grad_magnitude_correlation", "grad_direction_consistency",
        "brightness_at_depth_edges",
        "mutual_information", "joint_entropy",
        "local_corr_mean", "local_corr_abs_mean", "frac_strong_corr",
        "brightness_range_across_depth", "brightness_depth_slope",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle("Brightness-Depth Features: Real vs. AI-Generated", fontsize=16, fontweight="bold")

    for idx, key in enumerate(scalar_keys):
        ax = axes[idx // 4, idx % 4]
        real_vals = [s[key] for s in real_stats if key in s]
        fake_vals = [s[key] for s in fake_stats if key in s]

        bp = ax.boxplot([real_vals, fake_vals], labels=["Real", "Fake"],
                       patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")
        ax.set_title(key.replace("_", " ").title(), fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bd_comparison_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Depth-conditioned brightness curves ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Depth-Conditioned Brightness: Real vs. Fake", fontsize=14)
    x = np.linspace(0.05, 0.95, 10)  # depth bin centers

    # Mean brightness per depth bin
    real_curves = np.array([s["bin_means"] for s in real_stats])
    fake_curves = np.array([s["bin_means"] for s in fake_stats])

    ax = axes[0]
    for curve in real_curves:
        ax.plot(x, curve, alpha=0.2, color="#4CAF50")
    for curve in fake_curves:
        ax.plot(x, curve, alpha=0.2, color="#F44336")
    ax.plot(x, real_curves.mean(axis=0), color="#2E7D32", linewidth=2.5, label="Real (mean)")
    ax.plot(x, fake_curves.mean(axis=0), color="#C62828", linewidth=2.5, label="Fake (mean)")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Mean Brightness")
    ax.set_title("Brightness vs Depth")
    ax.legend()
    ax.grid(alpha=0.3)

    # Std of brightness per depth bin
    real_std_curves = np.array([s["bin_stds"] for s in real_stats])
    fake_std_curves = np.array([s["bin_stds"] for s in fake_stats])

    ax = axes[1]
    for curve in real_std_curves:
        ax.plot(x, curve, alpha=0.2, color="#4CAF50")
    for curve in fake_std_curves:
        ax.plot(x, curve, alpha=0.2, color="#F44336")
    ax.plot(x, real_std_curves.mean(axis=0), color="#2E7D32", linewidth=2.5, label="Real (mean)")
    ax.plot(x, fake_std_curves.mean(axis=0), color="#C62828", linewidth=2.5, label="Fake (mean)")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Brightness Std")
    ax.set_title("Brightness Variation vs Depth")
    ax.legend()
    ax.grid(alpha=0.3)

    # Distribution of global correlations
    ax = axes[2]
    real_corrs = [s["pearson_r"] for s in real_stats]
    fake_corrs = [s["pearson_r"] for s in fake_stats]
    ax.hist(real_corrs, bins=12, alpha=0.6, color="#4CAF50", label="Real", density=True)
    ax.hist(fake_corrs, bins=12, alpha=0.6, color="#F44336", label="Fake", density=True)
    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("Density")
    ax.set_title("Brightness-Depth Correlation Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bd_depth_conditioned.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"📊 Comparison plots saved to {output_dir}")


def print_summary(real_stats, fake_stats):
    print("\n" + "=" * 70)
    print("BRIGHTNESS-DEPTH ANALYSIS SUMMARY")
    print("=" * 70)

    keys = [
        "pearson_r", "spearman_r",
        "grad_magnitude_correlation", "grad_direction_consistency",
        "mutual_information", "joint_entropy",
        "local_corr_mean", "local_corr_abs_mean", "frac_strong_corr",
        "brightness_range_across_depth", "brightness_depth_slope",
    ]

    print(f"\n{'Feature':<32} {'Real (mean±std)':<22} {'Fake (mean±std)':<22} {'Δ':>8}")
    print("-" * 86)

    for key in keys:
        real_vals = np.array([s[key] for s in real_stats if key in s])
        fake_vals = np.array([s[key] for s in fake_stats if key in s])
        if len(real_vals) > 0 and len(fake_vals) > 0:
            real_str = f"{real_vals.mean():.4f} ± {real_vals.std():.4f}"
            fake_str = f"{fake_vals.mean():.4f} ± {fake_vals.std():.4f}"
            delta = fake_vals.mean() - real_vals.mean()
            print(f"{key:<32} {real_str:<22} {fake_str:<22} {delta:>+.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Brightness-Depth consistency analysis")
    parser.add_argument("--real-dir", type=str, required=True)
    parser.add_argument("--fake-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/brightness_depth")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    processor, model = load_depth_model(args.model, device)

    real_stats = process_directory(
        Path(args.real_dir), processor, model, output_dir, "real", device, args.max_images
    )
    fake_stats = process_directory(
        Path(args.fake_dir), processor, model, output_dir, "fake", device, args.max_images
    )

    print_summary(real_stats, fake_stats)
    create_comparison_plots(real_stats, fake_stats, output_dir)

    # Save raw stats (exclude non-serializable)
    all_stats = {"real": real_stats, "fake": fake_stats}
    with open(output_dir / "bd_statistics.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"📁 Statistics saved to {output_dir / 'bd_statistics.json'}")


if __name__ == "__main__":
    main()
