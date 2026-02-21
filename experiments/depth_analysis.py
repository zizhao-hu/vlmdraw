"""
Depth Analysis Experiment: Real vs. AI-Generated Images

Uses Depth Anything V2 to estimate depth maps for real and fake images,
then compares the results visually and statistically.

Hypothesis: AI-generated images may produce inconsistent or different
depth maps because generative models don't correctly understand 3D
scene geometry, lighting, and diffuse reflection.

Usage:
    python experiments/depth_analysis.py \
        --real-dir data/samples/real \
        --fake-dir data/samples/fake \
        --output-dir results/depth_analysis
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt
import matplotlib


def load_depth_model(model_name: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = "cuda"):
    """Load Depth Anything V2 model and processor."""
    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return processor, model


def estimate_depth(
    img: Image.Image,
    processor,
    model,
    device: str = "cuda",
) -> np.ndarray:
    """
    Estimate depth map for a single image.

    Returns:
        HxW float32 depth map (normalized to [0, 1]).
    """
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img.size[::-1],  # (H, W)
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.cpu().numpy()
    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


def compute_depth_statistics(depth: np.ndarray) -> dict:
    """Compute statistical features from a depth map."""
    stats = {
        "mean": float(depth.mean()),
        "std": float(depth.std()),
        "median": float(np.median(depth)),
        "min": float(depth.min()),
        "max": float(depth.max()),
        "range": float(depth.max() - depth.min()),
        "iqr": float(np.percentile(depth, 75) - np.percentile(depth, 25)),
    }

    # Higher-order statistics
    if depth.std() > 1e-8:
        centered = depth - depth.mean()
        stats["skewness"] = float((centered ** 3).mean() / (depth.std() ** 3))
        stats["kurtosis"] = float((centered ** 4).mean() / (depth.std() ** 4) - 3)
    else:
        stats["skewness"] = 0.0
        stats["kurtosis"] = 0.0

    # Gradient statistics (depth smoothness)
    gx = np.diff(depth, axis=1)
    gy = np.diff(depth, axis=0)
    stats["grad_mean"] = float((np.abs(gx).mean() + np.abs(gy).mean()) / 2)
    stats["grad_std"] = float((gx.std() + gy.std()) / 2)

    # Histogram entropy
    hist, _ = np.histogram(depth, bins=64, range=(0, 1))
    hist = hist.astype(np.float32) / hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    stats["entropy"] = float(entropy)

    return stats


def process_directory(
    img_dir: Path,
    processor,
    model,
    output_dir: Path,
    label: str,
    device: str = "cuda",
    max_images: int = 50,
) -> list[dict]:
    """Process all images in a directory and save depth maps."""
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in extensions
    ])[:max_images]

    print(f"\nProcessing {len(image_files)} {label} images from {img_dir}")

    depth_dir = output_dir / f"{label}_depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, img_path in enumerate(image_files):
        print(f"  [{i+1}/{len(image_files)}] {img_path.name}...", end=" ", flush=True)
        try:
            img = Image.open(img_path).convert("RGB")
            depth = estimate_depth(img, processor, model, device)

            # Save depth visualization
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"{label}: {img_path.name}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(depth, cmap="inferno")
            plt.title("Depth Map")
            plt.colorbar(shrink=0.8)
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(depth_dir / f"{img_path.stem}_depth.png", dpi=150, bbox_inches="tight")
            plt.close()

            # Save raw depth as npy
            np.save(depth_dir / f"{img_path.stem}_depth.npy", depth)

            # Compute statistics
            stats = compute_depth_statistics(depth)
            stats["filename"] = img_path.name
            stats["label"] = label
            results.append(stats)

            print(f"✓ (mean={stats['mean']:.3f}, std={stats['std']:.3f}, entropy={stats['entropy']:.2f})")

        except Exception as e:
            print(f"✗ Error: {e}")

    return results


def create_comparison_plots(
    real_stats: list[dict],
    fake_stats: list[dict],
    output_dir: Path,
):
    """Create side-by-side comparison plots of depth statistics."""
    matplotlib.use("Agg")

    metrics = ["mean", "std", "skewness", "kurtosis", "grad_mean", "grad_std", "entropy", "iqr"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Depth Map Statistics: Real vs. AI-Generated", fontsize=16, fontweight="bold")

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 4, idx % 4]

        real_vals = [s[metric] for s in real_stats if metric in s]
        fake_vals = [s[metric] for s in fake_stats if metric in s]

        # Box plot comparison
        bp = ax.boxplot(
            [real_vals, fake_vals],
            labels=["Real", "Fake"],
            patch_artist=True,
            widths=0.6,
        )
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")

        ax.set_title(metric.replace("_", " ").title(), fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        # Add mean line
        if real_vals and fake_vals:
            ax.axhline(np.mean(real_vals), color="#4CAF50", linestyle="--", alpha=0.5, linewidth=1)
            ax.axhline(np.mean(fake_vals), color="#F44336", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / "depth_comparison_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Histogram comparison for key metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Distribution Comparison: Real vs. Fake Depth Features", fontsize=14)

    for idx, metric in enumerate(["mean", "std", "entropy"]):
        ax = axes[idx]
        real_vals = [s[metric] for s in real_stats if metric in s]
        fake_vals = [s[metric] for s in fake_stats if metric in s]

        ax.hist(real_vals, bins=15, alpha=0.6, color="#4CAF50", label="Real", density=True)
        ax.hist(fake_vals, bins=15, alpha=0.6, color="#F44336", label="Fake", density=True)
        ax.set_title(f"Depth {metric.title()}", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "depth_comparison_histograms.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n📊 Comparison plots saved to {output_dir}")


def print_summary(real_stats: list[dict], fake_stats: list[dict]):
    """Print a text summary comparing real and fake depth statistics."""
    print("\n" + "=" * 60)
    print("DEPTH ANALYSIS SUMMARY")
    print("=" * 60)

    metrics = ["mean", "std", "skewness", "kurtosis", "grad_mean", "entropy"]

    print(f"\n{'Metric':<15} {'Real (mean±std)':<25} {'Fake (mean±std)':<25} {'Δ':>8}")
    print("-" * 75)

    for metric in metrics:
        real_vals = np.array([s[metric] for s in real_stats if metric in s])
        fake_vals = np.array([s[metric] for s in fake_stats if metric in s])

        if len(real_vals) > 0 and len(fake_vals) > 0:
            real_str = f"{real_vals.mean():.4f} ± {real_vals.std():.4f}"
            fake_str = f"{fake_vals.mean():.4f} ± {fake_vals.std():.4f}"
            delta = fake_vals.mean() - real_vals.mean()
            print(f"{metric:<15} {real_str:<25} {fake_str:<25} {delta:>+.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Depth analysis: Real vs. AI-generated images")
    parser.add_argument("--real-dir", type=str, required=True, help="Directory of real images")
    parser.add_argument("--fake-dir", type=str, required=True, help="Directory of AI-generated images")
    parser.add_argument("--output-dir", type=str, default="results/depth_analysis", help="Output directory")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="Depth model HuggingFace ID")
    parser.add_argument("--max-images", type=int, default=50, help="Max images per category")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load depth model
    device = args.device if torch.cuda.is_available() else "cpu"
    processor, model = load_depth_model(args.model, device)

    # Process real images
    real_stats = process_directory(
        Path(args.real_dir), processor, model, output_dir, "real", device, args.max_images
    )

    # Process fake images
    fake_stats = process_directory(
        Path(args.fake_dir), processor, model, output_dir, "fake", device, args.max_images
    )

    # Print summary
    print_summary(real_stats, fake_stats)

    # Create comparison plots
    create_comparison_plots(real_stats, fake_stats, output_dir)

    # Save all statistics
    all_stats = {"real": real_stats, "fake": fake_stats}
    with open(output_dir / "depth_statistics.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"📁 Statistics saved to {output_dir / 'depth_statistics.json'}")


if __name__ == "__main__":
    main()
