"""
Download sample images for the depth analysis experiment.

Downloads real and AI-generated images from working HuggingFace datasets.

Usage:
    python experiments/download_samples.py --output-dir data/samples --n-images 30
"""

import argparse
from pathlib import Path
from datasets import load_dataset


def download_samples(output_dir: Path, n_images: int = 30):
    """Download real and fake image samples from HuggingFace."""

    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # --- Download AI-generated images ---
    # Use MJHQ (Midjourney HQ) or similar working dataset
    fake_sources = [
        ("UCSC-VLAA/Recap-DataComp-1B", "default", "train", "url"),  # fallback
        ("fantasyfish/laion-art", "default", "train", "URL"),
    ]

    print("Downloading AI-generated images...")

    # Try to use a Stable Diffusion output dataset
    try:
        # Use Gustavosta/Stable-Diffusion-Prompts with image gen
        ds = load_dataset("Falah/Cifake_Dataset", split="train", streaming=True)
        count = 0
        for sample in ds:
            if count >= n_images:
                break
            # This dataset has "label" column: 0=real, 1=fake
            if sample.get("label", -1) == 1:
                img = sample["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(fake_dir / f"fake_{count:04d}.png")
                count += 1
                if count % 10 == 0:
                    print(f"  Downloaded {count} fake images...")
        print(f"  ✅ Downloaded {count} fake images from Cifake")
    except Exception as e:
        print(f"  ⚠️ Cifake failed: {e}")
        print("  Trying alternative: AI-generated art dataset...")
        try:
            ds = load_dataset("daspartho/stable-diffusion-prompts", split="train")
            # This is text-only, won't work. Try another.
            raise Exception("Text-only dataset")
        except:
            pass

        # Fallback: Use images from a working AI image dataset
        try:
            ds = load_dataset(
                "ChristophSchuhmann/improved_aesthetics_6.5plus",
                split="train",
                streaming=True,
            )
            count = 0
            for sample in ds:
                if count >= n_images:
                    break
                try:
                    img = sample["image"]
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(fake_dir / f"fake_{count:04d}.png")
                    count += 1
                except:
                    continue
            print(f"  ✅ Downloaded {count} images from aesthetics dataset")
        except Exception as e2:
            print(f"  ❌ All fake image sources failed: {e2}")

    # --- Download real images (COCO) ---
    n_existing_real = len(list(real_dir.glob("*.png")))
    if n_existing_real >= n_images:
        print(f"Already have {n_existing_real} real images, skipping download.")
    else:
        print("Downloading real images from COCO...")
        try:
            ds = load_dataset("detection-datasets/coco", split="val")
            count = 0
            for sample in ds:
                if count >= n_images:
                    break
                img = sample["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(real_dir / f"coco_{count:04d}.png")
                count += 1
            print(f"  ✅ Downloaded {count} real images from COCO")
        except Exception as e:
            print(f"  ❌ COCO download failed: {e}")

    # Summary
    n_real = len(list(real_dir.glob("*.png")))
    n_fake = len(list(fake_dir.glob("*.png")))
    print(f"\n📁 Data saved to {output_dir}")
    print(f"   Real images: {n_real}")
    print(f"   Fake images: {n_fake}")


def main():
    parser = argparse.ArgumentParser(description="Download sample real/fake images")
    parser.add_argument("--output-dir", type=str, default="data/samples", help="Output directory")
    parser.add_argument("--n-images", type=int, default=30, help="Number of images per category")
    args = parser.parse_args()

    download_samples(Path(args.output_dir), args.n_images)


if __name__ == "__main__":
    main()
