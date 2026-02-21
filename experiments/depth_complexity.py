"""
Depth & Brightness Complexity Analysis

Quantifies "how many levels" of depth and brightness exist in an image.
Real images should have more complex, multi-layered depth/brightness structure
because real scenes contain many objects at various distances with diverse lighting.

Metrics:
1. Histogram entropy (depth, brightness, joint) - higher = more levels
2. Number of histogram modes/peaks - distinct depth planes / brightness zones
3. Effective number of clusters (k-means on depth, brightness, joint)
4. Quantization error - how well N discrete levels can represent the distribution
5. Depth plane count - distinct depth layers via peak detection
6. Unique gradient levels - edge complexity at different scales
7. Spatial fragmentation - how many distinct depth/brightness regions

Usage:
    python experiments/depth_complexity.py \
        --real-dir data/aigenbench/real \
        --fake-dir data/aigenbench/fake \
        --output-dir results/depth_complexity
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
from scipy.signal import find_peaks
from scipy.ndimage import label as ndimage_label


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
    return (
        0.299 * img_array[:, :, 0].astype(np.float32) +
        0.587 * img_array[:, :, 1].astype(np.float32) +
        0.114 * img_array[:, :, 2].astype(np.float32)
    ) / 255.0


# ─── Complexity Metrics ────────────────────────────────────────────

def histogram_entropy(values: np.ndarray, n_bins: int = 256) -> float:
    """Shannon entropy of the histogram — measures spread of values."""
    hist, _ = np.histogram(values.ravel(), bins=n_bins, range=(0, 1))
    hist = hist / (hist.sum() + 1e-12)
    h = hist[hist > 0]
    return -float(np.sum(h * np.log2(h)))


def count_histogram_modes(values: np.ndarray, n_bins: int = 128, min_prominence: float = 0.005) -> dict:
    """
    Count distinct peaks in the histogram.
    More peaks = more distinct levels.
    """
    hist, bin_edges = np.histogram(values.ravel(), bins=n_bins, range=(0, 1))
    hist_smooth = np.convolve(hist / hist.sum(), np.ones(5)/5, mode='same')

    peaks, properties = find_peaks(hist_smooth, prominence=min_prominence, distance=3)

    # Also count with stricter threshold
    peaks_strict, _ = find_peaks(hist_smooth, prominence=min_prominence * 2, distance=5)

    return {
        "n_modes": len(peaks),
        "n_modes_strict": len(peaks_strict),
        "peak_prominences": [float(p) for p in properties.get("prominences", [])],
    }


def effective_n_levels(values: np.ndarray, max_k: int = 20) -> dict:
    """
    Use k-means to find the effective number of distinct levels.
    Measure reconstruction error for different k values.
    The 'elbow' indicates the true number of levels.
    """
    flat = values.ravel()
    # Subsample for speed
    if len(flat) > 50000:
        flat = np.random.choice(flat, 50000, replace=False)

    errors = []
    for k in range(1, max_k + 1):
        # Simple 1D k-means
        centers = np.linspace(flat.min(), flat.max(), k)
        for _ in range(20):  # iterations
            # Assign
            dists = np.abs(flat[:, None] - centers[None, :])
            labels = dists.argmin(axis=1)
            # Update
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    centers[c] = flat[mask].mean()
        # Reconstruction error
        reconstructed = centers[labels]
        mse = float(np.mean((flat - reconstructed) ** 2))
        errors.append(mse)

    errors = np.array(errors)

    # Find elbow: largest drop in error ratio
    if len(errors) > 2:
        ratios = errors[:-1] / (errors[1:] + 1e-12)
        elbow_k = int(np.argmax(ratios)) + 2  # +2 because index 0 = k=1→k=2
    else:
        elbow_k = 1

    # Effective levels: how many k needed to get < 5% of k=1 error
    threshold = errors[0] * 0.05
    effective_k = 1
    for i, e in enumerate(errors):
        if e < threshold:
            effective_k = i + 1
            break
    else:
        effective_k = max_k

    return {
        "elbow_k": elbow_k,
        "effective_k": effective_k,
        "quantization_errors": [float(e) for e in errors],
        "error_at_k5": float(errors[4]) if len(errors) > 4 else 0,
        "error_at_k10": float(errors[9]) if len(errors) > 9 else 0,
        "error_ratio_k1_k10": float(errors[0] / (errors[9] + 1e-12)) if len(errors) > 9 else 0,
    }


def depth_plane_analysis(depth: np.ndarray, n_bins: int = 100) -> dict:
    """
    Detect distinct depth planes by finding stable depth regions.
    Real images should have more distinct depth planes.
    """
    hist, bin_edges = np.histogram(depth.ravel(), bins=n_bins, range=(0, 1))
    hist_norm = hist / hist.sum()

    # Smooth and find peaks (each peak = a depth plane)
    kernel = np.ones(3) / 3
    hist_smooth = np.convolve(hist_norm, kernel, mode='same')

    peaks, props = find_peaks(hist_smooth, prominence=0.003, distance=2, width=1)

    # Measure the "spread" of depth: what fraction of [0,1] range is actually used
    occupied_bins = (hist > 0).sum()
    depth_coverage = occupied_bins / n_bins

    # Depth range used (95th percentile - 5th percentile)
    d_flat = depth.ravel()
    p5, p95 = np.percentile(d_flat, [5, 95])
    effective_range = p95 - p5

    # Uniformity: how uniform is the depth distribution? Real scenes may be less uniform
    if hist_norm.sum() > 0:
        uniform = np.ones_like(hist_norm) / len(hist_norm)
        kl_from_uniform = float(np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0] / uniform[hist_norm > 0])))
    else:
        kl_from_uniform = 0

    return {
        "n_depth_planes": len(peaks),
        "depth_coverage": float(depth_coverage),
        "effective_depth_range": float(effective_range),
        "depth_kl_from_uniform": float(kl_from_uniform),
    }


def spatial_fragmentation(depth: np.ndarray, brightness: np.ndarray, n_levels: int = 8) -> dict:
    """
    Quantize depth/brightness into levels and count connected components.
    More fragments = more complex spatial structure.
    """
    # Depth fragmentation
    depth_q = np.clip((depth * n_levels).astype(int), 0, n_levels - 1)
    depth_fragments = 0
    for level in range(n_levels):
        mask = (depth_q == level).astype(np.int32)
        if mask.sum() > 0:
            labeled, n_components = ndimage_label(mask)
            # Only count components larger than 0.5% of image
            min_size = mask.size * 0.005
            significant = sum(1 for c in range(1, n_components + 1) if (labeled == c).sum() > min_size)
            depth_fragments += significant

    # Brightness fragmentation
    brightness_q = np.clip((brightness * n_levels).astype(int), 0, n_levels - 1)
    brightness_fragments = 0
    for level in range(n_levels):
        mask = (brightness_q == level).astype(np.int32)
        if mask.sum() > 0:
            labeled, n_components = ndimage_label(mask)
            min_size = mask.size * 0.005
            significant = sum(1 for c in range(1, n_components + 1) if (labeled == c).sum() > min_size)
            brightness_fragments += significant

    # Joint fragmentation: quantize both and count unique regions
    joint_q = depth_q * n_levels + brightness_q
    unique_joint_levels = len(np.unique(joint_q))

    return {
        "depth_fragments": depth_fragments,
        "brightness_fragments": brightness_fragments,
        "joint_unique_levels": unique_joint_levels,
        "depth_fragments_per_level": float(depth_fragments / n_levels),
        "brightness_fragments_per_level": float(brightness_fragments / n_levels),
    }


def gradient_level_complexity(arr: np.ndarray, n_bins: int = 64) -> dict:
    """
    Analyze the complexity of gradient magnitudes.
    More diverse gradients = more distinct transitions = more "levels".
    """
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    h = min(gx.shape[0], gy.shape[0])
    w = min(gx.shape[1], gy.shape[1])
    mag = np.sqrt(gx[:h, :w] ** 2 + gy[:h, :w] ** 2)

    # Entropy of gradient magnitudes
    hist, _ = np.histogram(mag.ravel(), bins=n_bins)
    hist_n = hist / (hist.sum() + 1e-12)
    h_vals = hist_n[hist_n > 0]
    grad_entropy = -float(np.sum(h_vals * np.log2(h_vals)))

    # Number of distinct gradient levels (non-zero bins)
    n_grad_levels = int((hist > 0).sum())

    # Edge density: fraction of pixels with significant gradient
    threshold = np.percentile(mag, 90)
    edge_density = float((mag > threshold * 0.5).mean())

    return {
        "grad_entropy": float(grad_entropy),
        "n_grad_levels": n_grad_levels,
        "edge_density": float(edge_density),
    }


def joint_complexity(brightness: np.ndarray, depth: np.ndarray, n_bins: int = 32) -> dict:
    """
    Analyze the 2D joint brightness-depth distribution complexity.
    """
    hist2d, _, _ = np.histogram2d(
        brightness.ravel(), depth.ravel(),
        bins=n_bins, range=[[0, 1], [0, 1]]
    )
    hist2d_norm = hist2d / (hist2d.sum() + 1e-12)

    # Number of occupied cells
    n_occupied = int((hist2d > 0).sum())
    total_cells = n_bins * n_bins
    occupancy_ratio = n_occupied / total_cells

    # 2D entropy
    h = hist2d_norm[hist2d_norm > 0]
    joint_entropy = -float(np.sum(h * np.log2(h)))

    # Effective dimensionality via SVD of the 2D histogram
    U, S, Vt = np.linalg.svd(hist2d_norm)
    S_norm = S / (S.sum() + 1e-12)
    # Effective rank (entropy-based)
    s_nonzero = S_norm[S_norm > 1e-10]
    spectral_entropy = -float(np.sum(s_nonzero * np.log2(s_nonzero)))
    effective_rank = float(2 ** spectral_entropy)

    # Concentration: how many cells hold 90% of mass
    sorted_vals = np.sort(hist2d_norm.ravel())[::-1]
    cumsum = np.cumsum(sorted_vals)
    n_for_90 = int(np.searchsorted(cumsum, 0.9) + 1)

    return {
        "joint_n_occupied": n_occupied,
        "joint_occupancy_ratio": float(occupancy_ratio),
        "joint_entropy_2d": float(joint_entropy),
        "joint_effective_rank": float(effective_rank),
        "joint_n_for_90pct": n_for_90,
    }


# ─── Main Pipeline ─────────────────────────────────────────────────

def extract_complexity(brightness: np.ndarray, depth: np.ndarray) -> dict:
    feats = {}

    # Depth complexity
    feats["depth_entropy"] = histogram_entropy(depth, 256)
    depth_modes = count_histogram_modes(depth)
    feats["depth_n_modes"] = depth_modes["n_modes"]
    feats["depth_n_modes_strict"] = depth_modes["n_modes_strict"]

    depth_levels = effective_n_levels(depth)
    feats["depth_elbow_k"] = depth_levels["elbow_k"]
    feats["depth_effective_k"] = depth_levels["effective_k"]
    feats["depth_error_ratio_k1_k10"] = depth_levels["error_ratio_k1_k10"]
    feats["depth_quant_errors"] = depth_levels["quantization_errors"]

    depth_planes = depth_plane_analysis(depth)
    feats.update({f"depth_{k}": v for k, v in depth_planes.items() if k != "n_depth_planes"})
    feats["n_depth_planes"] = depth_planes["n_depth_planes"]

    # Brightness complexity
    feats["brightness_entropy"] = histogram_entropy(brightness, 256)
    bright_modes = count_histogram_modes(brightness)
    feats["brightness_n_modes"] = bright_modes["n_modes"]
    feats["brightness_n_modes_strict"] = bright_modes["n_modes_strict"]

    bright_levels = effective_n_levels(brightness)
    feats["brightness_elbow_k"] = bright_levels["elbow_k"]
    feats["brightness_effective_k"] = bright_levels["effective_k"]
    feats["brightness_error_ratio_k1_k10"] = bright_levels["error_ratio_k1_k10"]
    feats["brightness_quant_errors"] = bright_levels["quantization_errors"]

    # Spatial fragmentation
    frag = spatial_fragmentation(depth, brightness)
    feats.update(frag)

    # Gradient complexity
    d_grad = gradient_level_complexity(depth)
    feats.update({f"depth_{k}": v for k, v in d_grad.items()})
    b_grad = gradient_level_complexity(brightness)
    feats.update({f"brightness_{k}": v for k, v in b_grad.items()})

    # Joint complexity
    jc = joint_complexity(brightness, depth)
    feats.update(jc)

    return feats


def process_directory(img_dir, processor, model, output_dir, label, device, max_images=50):
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])[:max_images]

    print(f"\nProcessing {len(image_files)} {label} images...")
    results = []
    for i, img_path in enumerate(image_files):
        print(f"  [{i+1}/{len(image_files)}] {img_path.name}...", end=" ", flush=True)
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            brightness = rgb_to_luminance(img_array)
            depth = estimate_depth(img, processor, model, device)
            feats = extract_complexity(brightness, depth)
            feats["filename"] = img_path.name
            feats["label"] = label
            results.append(feats)
            print(f"✓ (depth_planes={feats['n_depth_planes']}, depth_modes={feats['depth_n_modes']}, "
                  f"joint_occupied={feats['joint_n_occupied']}, d_ent={feats['depth_entropy']:.2f}, b_ent={feats['brightness_entropy']:.2f})")
        except Exception as e:
            print(f"✗ {e}")
    return results


def create_plots(real_stats, fake_stats, output_dir):
    # ─── Box plots ───
    scalar_keys = [
        "depth_entropy", "brightness_entropy",
        "depth_n_modes", "brightness_n_modes",
        "n_depth_planes",
        "depth_elbow_k", "brightness_elbow_k",
        "depth_effective_k", "brightness_effective_k",
        "depth_fragments", "brightness_fragments",
        "joint_n_occupied", "joint_occupancy_ratio",
        "joint_entropy_2d", "joint_effective_rank", "joint_n_for_90pct",
        "joint_unique_levels",
        "depth_grad_entropy", "brightness_grad_entropy",
        "depth_coverage", "effective_depth_range",
    ]

    n_plots = len(scalar_keys)
    cols = 4
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 4))
    fig.suptitle("Depth & Brightness Complexity: Real vs. AI-Generated", fontsize=16, fontweight="bold")

    for idx, key in enumerate(scalar_keys):
        ax = axes[idx // cols, idx % cols]
        real_vals = [s[key] for s in real_stats if key in s]
        fake_vals = [s[key] for s in fake_stats if key in s]
        if not real_vals or not fake_vals:
            ax.set_visible(False)
            continue

        bp = ax.boxplot([real_vals, fake_vals], labels=["Real", "Fake"],
                       patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")
        ax.set_title(key.replace("_", " ").title(), fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Add mean values as text
        rm = np.mean(real_vals)
        fm = np.mean(fake_vals)
        ax.annotate(f"R={rm:.2f}", xy=(1, rm), fontsize=7, color="green", ha="center")
        ax.annotate(f"F={fm:.2f}", xy=(2, fm), fontsize=7, color="red", ha="center")

    # Hide unused
    for idx in range(n_plots, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "complexity_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ─── Quantization error curves ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Quantization Error: How Many Levels Needed?", fontsize=14)

    for ax, channel, color_r, color_f in [
        (axes[0], "depth", "#2E7D32", "#C62828"),
        (axes[1], "brightness", "#1565C0", "#E65100"),
    ]:
        for s in real_stats:
            if f"{channel}_quant_errors" in s:
                ax.plot(range(1, len(s[f"{channel}_quant_errors"]) + 1),
                       s[f"{channel}_quant_errors"], alpha=0.15, color=color_r)
        for s in fake_stats:
            if f"{channel}_quant_errors" in s:
                ax.plot(range(1, len(s[f"{channel}_quant_errors"]) + 1),
                       s[f"{channel}_quant_errors"], alpha=0.15, color=color_f)

        # Mean curves
        real_errs = np.array([s[f"{channel}_quant_errors"] for s in real_stats if f"{channel}_quant_errors" in s])
        fake_errs = np.array([s[f"{channel}_quant_errors"] for s in fake_stats if f"{channel}_quant_errors" in s])
        if len(real_errs) > 0:
            ax.plot(range(1, real_errs.shape[1] + 1), real_errs.mean(axis=0),
                   color=color_r, linewidth=2.5, label="Real (mean)")
        if len(fake_errs) > 0:
            ax.plot(range(1, fake_errs.shape[1] + 1), fake_errs.mean(axis=0),
                   color=color_f, linewidth=2.5, label="Fake (mean)")
        ax.set_xlabel("Number of Levels (k)")
        ax.set_ylabel("Quantization MSE")
        ax.set_title(f"{channel.title()} Quantization Error")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "quantization_curves.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"📊 Plots saved to {output_dir}")


def print_summary(real_stats, fake_stats):
    print("\n" + "=" * 80)
    print("DEPTH & BRIGHTNESS COMPLEXITY SUMMARY")
    print("=" * 80)

    keys = [
        "depth_entropy", "brightness_entropy",
        "depth_n_modes", "brightness_n_modes",
        "n_depth_planes",
        "depth_elbow_k", "brightness_elbow_k",
        "depth_fragments", "brightness_fragments",
        "joint_n_occupied", "joint_occupancy_ratio",
        "joint_entropy_2d", "joint_effective_rank", "joint_n_for_90pct",
        "joint_unique_levels",
        "depth_coverage", "effective_depth_range",
        "depth_error_ratio_k1_k10", "brightness_error_ratio_k1_k10",
    ]

    print(f"\n{'Feature':<35} {'Real (mean±std)':<22} {'Fake (mean±std)':<22} {'Δ':>8}")
    print("-" * 90)

    for key in keys:
        rv = np.array([s[key] for s in real_stats if key in s])
        fv = np.array([s[key] for s in fake_stats if key in s])
        if len(rv) > 0 and len(fv) > 0:
            real_str = f"{rv.mean():.3f} ± {rv.std():.3f}"
            fake_str = f"{fv.mean():.3f} ± {fv.std():.3f}"
            delta = fv.mean() - rv.mean()
            print(f"{key:<35} {real_str:<22} {fake_str:<22} {delta:>+.3f}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=str, required=True)
    parser.add_argument("--fake-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/depth_complexity")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    processor, model = load_depth_model(args.model, device)

    real_stats = process_directory(Path(args.real_dir), processor, model, output_dir, "real", device, args.max_images)
    fake_stats = process_directory(Path(args.fake_dir), processor, model, output_dir, "fake", device, args.max_images)

    print_summary(real_stats, fake_stats)
    create_plots(real_stats, fake_stats, output_dir)

    # Save (strip quantization error arrays for cleaner JSON summary)
    save_stats = {"real": [], "fake": []}
    for label in ["real", "fake"]:
        src = real_stats if label == "real" else fake_stats
        for s in src:
            s_copy = {k: v for k, v in s.items() if not k.endswith("_quant_errors")}
            save_stats[label].append(s_copy)

    with open(output_dir / "complexity_stats.json", "w") as f:
        json.dump(save_stats, f, indent=2)
    print(f"📁 Stats saved to {output_dir / 'complexity_stats.json'}")


if __name__ == "__main__":
    main()
