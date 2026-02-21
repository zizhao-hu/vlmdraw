"""
Generate a large combined figure showing real vs fake images with their depth maps.

Layout:
- Top section: Real images (original + depth map side by side)
- Bottom section: Fake images (original + depth map side by side)
- Grid format: N rows x 2 columns per image (original | depth)

Usage:
    python experiments/make_gallery.py \
        --real-dir data/aigenbench/real \
        --fake-dir data/aigenbench/fake \
        --output-dir results/gallery \
        --n-images 15
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def load_images(img_dir: Path, max_images: int):
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])[:max_images]
    return files


def make_combined_gallery(real_dir, fake_dir, processor, model, device, output_dir, n_images=15):
    """
    Big figure: Real vs Fake, each row = one image.
    Columns: Original | Depth Map | Brightness
    Two halves side by side: Real (left 3 cols) | Fake (right 3 cols)
    """
    real_files = load_images(real_dir, n_images)
    fake_files = load_images(fake_dir, n_images)
    n = min(len(real_files), len(fake_files), n_images)

    print(f"Creating gallery with {n} images per category...")

    # ─── Figure 1: Side-by-side Real vs Fake (Image + Depth) ───
    fig = plt.figure(figsize=(28, 3.2 * n + 2))
    gs = gridspec.GridSpec(n + 1, 4, figure=fig, hspace=0.25, wspace=0.05,
                           height_ratios=[0.3] + [1] * n)

    # Headers
    header_style = dict(fontsize=18, fontweight="bold", ha="center", va="center")
    ax_h1 = fig.add_subplot(gs[0, 0:2])
    ax_h1.text(0.5, 0.5, "REAL IMAGES (COCO)", color="#2E7D32", **header_style, transform=ax_h1.transAxes)
    ax_h1.set_facecolor("#E8F5E9")
    ax_h1.set_xticks([]); ax_h1.set_yticks([])
    for spine in ax_h1.spines.values(): spine.set_visible(False)

    ax_h2 = fig.add_subplot(gs[0, 2:4])
    ax_h2.text(0.5, 0.5, "FAKE IMAGES (AI-GenBench)", color="#C62828", **header_style, transform=ax_h2.transAxes)
    ax_h2.set_facecolor("#FFEBEE")
    ax_h2.set_xticks([]); ax_h2.set_yticks([])
    for spine in ax_h2.spines.values(): spine.set_visible(False)

    for i in range(n):
        # Real
        print(f"  [{i+1}/{n}] Real: {real_files[i].name}, Fake: {fake_files[i].name}", flush=True)

        real_img = Image.open(real_files[i]).convert("RGB")
        real_arr = np.array(real_img)
        real_depth = estimate_depth(real_img, processor, model, device)

        ax_ri = fig.add_subplot(gs[i + 1, 0])
        ax_ri.imshow(real_arr)
        ax_ri.set_title(real_files[i].stem, fontsize=7, color="#2E7D32")
        ax_ri.axis("off")

        ax_rd = fig.add_subplot(gs[i + 1, 1])
        ax_rd.imshow(real_depth, cmap="inferno", vmin=0, vmax=1)
        ax_rd.set_title("Depth", fontsize=7)
        ax_rd.axis("off")

        # Fake
        fake_img = Image.open(fake_files[i]).convert("RGB")
        fake_arr = np.array(fake_img)
        fake_depth = estimate_depth(fake_img, processor, model, device)

        ax_fi = fig.add_subplot(gs[i + 1, 2])
        ax_fi.imshow(fake_arr)
        ax_fi.set_title(fake_files[i].stem, fontsize=7, color="#C62828")
        ax_fi.axis("off")

        ax_fd = fig.add_subplot(gs[i + 1, 3])
        ax_fd.imshow(fake_depth, cmap="inferno", vmin=0, vmax=1)
        ax_fd.set_title("Depth", fontsize=7)
        ax_fd.axis("off")

    plt.savefig(output_dir / "gallery_depth_sidebyside.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  ✅ Saved gallery_depth_sidebyside.png")

    # ─── Figure 2: Image + Depth + Brightness + Scatter (detailed, fewer images) ───
    n_detail = min(8, n)
    fig2 = plt.figure(figsize=(32, 4 * n_detail + 2))
    gs2 = gridspec.GridSpec(n_detail * 2 + 1, 4, figure=fig2, hspace=0.35, wspace=0.12,
                            height_ratios=[0.3] + [1] * (n_detail * 2))

    # Header
    ax_h = fig2.add_subplot(gs2[0, :])
    ax_h.text(0.5, 0.5, "Detailed Comparison: Original → Brightness → Depth → Brightness vs Depth Scatter",
              fontsize=16, fontweight="bold", ha="center", va="center", transform=ax_h.transAxes)
    ax_h.set_facecolor("#F5F5F5")
    ax_h.set_xticks([]); ax_h.set_yticks([])
    for spine in ax_h.spines.values(): spine.set_visible(False)

    row = 1
    for i in range(n_detail):
        # Real row
        real_img = Image.open(real_files[i]).convert("RGB")
        real_arr = np.array(real_img)
        real_brightness = rgb_to_luminance(real_arr)
        real_depth = estimate_depth(real_img, processor, model, device)

        ax = fig2.add_subplot(gs2[row, 0])
        ax.imshow(real_arr)
        ax.set_ylabel(f"REAL #{i}", fontsize=9, color="#2E7D32", fontweight="bold")
        ax.set_title("Original", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig2.add_subplot(gs2[row, 1])
        ax.imshow(real_brightness, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Brightness", fontsize=8)
        ax.axis("off")

        ax = fig2.add_subplot(gs2[row, 2])
        ax.imshow(real_depth, cmap="inferno", vmin=0, vmax=1)
        ax.set_title("Depth", fontsize=8)
        ax.axis("off")

        ax = fig2.add_subplot(gs2[row, 3])
        b_flat = real_brightness.ravel()
        d_flat = real_depth.ravel()
        idx = np.random.choice(len(b_flat), min(3000, len(b_flat)), replace=False)
        ax.scatter(d_flat[idx], b_flat[idx], alpha=0.1, s=1, c="#2E7D32")
        from scipy.stats import pearsonr
        r_val, _ = pearsonr(b_flat, d_flat)
        ax.set_title(f"B vs D (r={r_val:.3f})", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Depth", fontsize=7); ax.set_ylabel("Brightness", fontsize=7)
        ax.grid(alpha=0.2)
        row += 1

        # Fake row
        fake_img = Image.open(fake_files[i]).convert("RGB")
        fake_arr = np.array(fake_img)
        fake_brightness = rgb_to_luminance(fake_arr)
        fake_depth = estimate_depth(fake_img, processor, model, device)

        ax = fig2.add_subplot(gs2[row, 0])
        ax.imshow(fake_arr)
        ax.set_ylabel(f"FAKE #{i}", fontsize=9, color="#C62828", fontweight="bold")
        ax.set_title("Original", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig2.add_subplot(gs2[row, 1])
        ax.imshow(fake_brightness, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Brightness", fontsize=8)
        ax.axis("off")

        ax = fig2.add_subplot(gs2[row, 2])
        ax.imshow(fake_depth, cmap="inferno", vmin=0, vmax=1)
        ax.set_title("Depth", fontsize=8)
        ax.axis("off")

        ax = fig2.add_subplot(gs2[row, 3])
        b_flat = fake_brightness.ravel()
        d_flat = fake_depth.ravel()
        idx = np.random.choice(len(b_flat), min(3000, len(b_flat)), replace=False)
        ax.scatter(d_flat[idx], b_flat[idx], alpha=0.1, s=1, c="#C62828")
        r_val, _ = pearsonr(b_flat, d_flat)
        ax.set_title(f"B vs D (r={r_val:.3f})", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Depth", fontsize=7); ax.set_ylabel("Brightness", fontsize=7)
        ax.grid(alpha=0.2)
        row += 1

    plt.savefig(output_dir / "gallery_detailed_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  ✅ Saved gallery_detailed_comparison.png")

    # ─── Figure 3: Depth histograms grid ───
    n_hist = min(10, n)
    fig3, axes3 = plt.subplots(n_hist, 2, figsize=(12, 2.5 * n_hist))
    fig3.suptitle("Depth Distributions: Real (left) vs Fake (right)", fontsize=14, fontweight="bold")

    for i in range(n_hist):
        real_img = Image.open(real_files[i]).convert("RGB")
        real_depth = estimate_depth(real_img, processor, model, device)
        fake_img = Image.open(fake_files[i]).convert("RGB")
        fake_depth = estimate_depth(fake_img, processor, model, device)

        ax = axes3[i, 0]
        ax.hist(real_depth.ravel(), bins=80, range=(0, 1), color="#4CAF50", alpha=0.8, density=True)
        ax.set_title(f"Real: {real_files[i].stem}", fontsize=8, color="#2E7D32")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
        if i == n_hist - 1:
            ax.set_xlabel("Depth")

        ax = axes3[i, 1]
        ax.hist(fake_depth.ravel(), bins=80, range=(0, 1), color="#F44336", alpha=0.8, density=True)
        ax.set_title(f"Fake: {fake_files[i].stem}", fontsize=8, color="#C62828")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
        if i == n_hist - 1:
            ax.set_xlabel("Depth")

    plt.tight_layout()
    plt.savefig(output_dir / "gallery_depth_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved gallery_depth_histograms.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=str, required=True)
    parser.add_argument("--fake-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/gallery")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--n-images", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    processor, model = load_depth_model(args.model, device)

    make_combined_gallery(
        Path(args.real_dir), Path(args.fake_dir),
        processor, model, device, output_dir, args.n_images,
    )
    print("\n✅ All gallery figures generated!")


if __name__ == "__main__":
    main()
