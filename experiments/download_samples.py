"""
Download sample images for the depth analysis experiment.

Downloads a small set of real and AI-generated images for quick testing.
Uses publicly available sources.

Usage:
    python experiments/download_samples.py --output-dir data/samples --n-images 20
"""

import argparse
import os
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def download_from_huggingface(output_dir: Path, n_images: int = 20):
    """
    Download real and fake image samples from HuggingFace datasets.

    Uses publicly available AI-generated image detection datasets.
    """
    if not HAS_DATASETS:
        print("❌ 'datasets' library not installed. Run: pip install datasets")
        return

    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Try to load a public real/fake dataset
    # Option 1: Use "poloclub/diffusiondb" for AI-generated
    # Option 2: Use a detection benchmark dataset
    print("Downloading AI-generated images from DiffusionDB...")
    try:
        ds = load_dataset(
            "poloclub/diffusiondb",
            "2m_random_1k",
            split="train",
        )
        for i, sample in enumerate(ds):
            if i >= n_images:
                break
            img = sample["image"]
            img.save(fake_dir / f"diffusiondb_{i:04d}.png")
        print(f"  ✅ Downloaded {min(n_images, len(ds))} fake images")
    except Exception as e:
        print(f"  ⚠️ Could not download DiffusionDB: {e}")

    # Download real images from COCO or similar
    print("Downloading real images from COCO subset...")
    try:
        ds = load_dataset(
            "detection-datasets/coco",
            split="val",
        )
        for i, sample in enumerate(ds):
            if i >= n_images:
                break
            img = sample["image"]
            img = img.convert("RGB")
            img.save(real_dir / f"coco_{i:04d}.png")
        print(f"  ✅ Downloaded {min(n_images, len(ds))} real images")
    except Exception as e:
        print(f"  ⚠️ Could not download COCO: {e}")
        print("  Trying alternative source...")
        try:
            ds = load_dataset(
                "jmhessel/newyorker_caption_contest",
                "explanation",
                split="validation",
            )
            for i, sample in enumerate(ds):
                if i >= n_images:
                    break
                img = sample["image"]
                img = img.convert("RGB")
                img.save(real_dir / f"real_{i:04d}.png")
            print(f"  ✅ Downloaded {min(n_images, len(ds))} real images")
        except Exception as e2:
            print(f"  ❌ Could not download alternative: {e2}")

    # Summary
    n_real = len(list(real_dir.glob("*.png")))
    n_fake = len(list(fake_dir.glob("*.png")))
    print(f"\n📁 Data saved to {output_dir}")
    print(f"   Real images: {n_real}")
    print(f"   Fake images: {n_fake}")


def main():
    parser = argparse.ArgumentParser(description="Download sample real/fake images")
    parser.add_argument("--output-dir", type=str, default="data/samples", help="Output directory")
    parser.add_argument("--n-images", type=int, default=20, help="Number of images per category")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_from_huggingface(output_dir, args.n_images)


if __name__ == "__main__":
    main()
