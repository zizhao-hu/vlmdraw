"""
Download samples from AI-GenBench for depth analysis experiment.

Uses:
- Fake images: HuggingFace dataset lrzpellegrini/AI-GenBench-fake_part
- Real images: COCO validation (already downloaded) or from HuggingFace

Usage:
    python experiments/download_aigenbench.py --output-dir data/aigenbench --n-images 30
"""

import argparse
from pathlib import Path
from datasets import load_dataset


def download_aigenbench_samples(output_dir: Path, n_images: int = 30):
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # --- Download fake images from AI-GenBench HuggingFace ---
    print("Downloading AI-generated images from AI-GenBench (HuggingFace)...")
    try:
        ds = load_dataset(
            "lrzpellegrini/AI-GenBench-fake_part",
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
                # Resize to reasonable size for depth estimation
                if max(img.size) > 1024:
                    ratio = 1024 / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size)
                img.save(fake_dir / f"aigenbench_fake_{count:04d}.png")
                count += 1
                if count % 10 == 0:
                    print(f"  Downloaded {count}/{n_images} fake images...", flush=True)
            except Exception as e:
                print(f"  ⚠️ Skipping sample: {e}")
                continue
        print(f"  ✅ Downloaded {count} fake images from AI-GenBench")
    except Exception as e:
        print(f"  ❌ AI-GenBench fake download failed: {e}")

    # --- Download real images (COCO) ---
    n_existing_real = len(list(real_dir.glob("*.png")))
    if n_existing_real >= n_images:
        print(f"Already have {n_existing_real} real images, skipping.")
    else:
        print("Downloading real images from COCO validation...")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/aigenbench")
    parser.add_argument("--n-images", type=int, default=30)
    args = parser.parse_args()
    download_aigenbench_samples(Path(args.output_dir), args.n_images)
