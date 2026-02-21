"""
Image and video compression pipeline for AIGC detection benchmark.

Applies realistic compression transforms to simulate real-world distribution:
- JPEG at various quality levels
- WebP compression
- H.264/H.265 video encoding
- Resize + recompression
"""

import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────
# Compression Configs
# ──────────────────────────────────────────────────────────────────

COMPRESSION_PRESETS = {
    # JPEG quality levels
    "jpeg_95": {"format": "jpeg", "quality": 95},
    "jpeg_85": {"format": "jpeg", "quality": 85},
    "jpeg_75": {"format": "jpeg", "quality": 75},
    "jpeg_50": {"format": "jpeg", "quality": 50},
    # WebP quality levels
    "webp_90": {"format": "webp", "quality": 90},
    "webp_75": {"format": "webp", "quality": 75},
    "webp_50": {"format": "webp", "quality": 50},
    # Resize + JPEG (simulates mobile sharing)
    "resize50_jpeg75": {"format": "jpeg", "quality": 75, "resize": 0.5},
    "resize75_jpeg75": {"format": "jpeg", "quality": 75, "resize": 0.75},
    # Screenshot simulation (PNG re-encode, lossless but strips metadata)
    "screenshot": {"format": "png"},
}


def compress_image(
    img: Image.Image,
    format: str = "jpeg",
    quality: int = 75,
    resize: Optional[float] = None,
) -> Image.Image:
    """
    Apply compression to a PIL Image and return the result.

    The image is saved to a buffer with the specified format/quality,
    then re-loaded — simulating a real save-load cycle.

    Args:
        img: Input PIL Image (RGB).
        format: Output format ('jpeg', 'webp', 'png').
        quality: Compression quality (1-100, ignored for PNG).
        resize: Optional resize factor (e.g., 0.5 = half resolution).

    Returns:
        Compressed PIL Image.
    """
    from io import BytesIO

    # Convert to RGB if needed (JPEG doesn't support RGBA)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Optional resize
    if resize is not None and resize != 1.0:
        new_w = max(1, int(img.width * resize))
        new_h = max(1, int(img.height * resize))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Compress via save/load cycle
    buffer = BytesIO()
    save_kwargs = {}
    if format.lower() == "jpeg":
        save_format = "JPEG"
        save_kwargs["quality"] = quality
        save_kwargs["subsampling"] = 0  # 4:4:4 chroma subsampling
    elif format.lower() == "webp":
        save_format = "WEBP"
        save_kwargs["quality"] = quality
    elif format.lower() == "png":
        save_format = "PNG"
    else:
        raise ValueError(f"Unsupported format: {format}")

    img.save(buffer, format=save_format, **save_kwargs)
    buffer.seek(0)
    return Image.open(buffer).copy()


def compress_image_file(
    input_path: str | Path,
    output_path: str | Path,
    preset: str = "jpeg_75",
) -> None:
    """Compress a single image file using a named preset."""
    config = COMPRESSION_PRESETS[preset]
    img = Image.open(input_path)
    compressed = compress_image(img, **config)
    compressed.save(str(output_path))


def compress_video_ffmpeg(
    input_path: str | Path,
    output_path: str | Path,
    codec: str = "libx264",
    crf: int = 23,
) -> None:
    """
    Compress a video file using FFmpeg.

    Args:
        input_path: Path to input video.
        output_path: Path to output compressed video.
        codec: Video codec ('libx264' for H.264, 'libx265' for H.265).
        crf: Constant Rate Factor (lower = higher quality).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def compress_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    presets: list[str],
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp"),
) -> dict[str, int]:
    """
    Compress all images in a directory with multiple presets.

    Creates subdirectories for each preset:
        output_dir/
            jpeg_75/
            jpeg_50/
            webp_75/
            ...

    Returns:
        Dict mapping preset name to number of images processed.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    stats = {}

    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.rglob(f"*{ext}"))

    print(f"Found {len(image_files)} images in {input_dir}")

    for preset in presets:
        if preset not in COMPRESSION_PRESETS:
            print(f"⚠️  Unknown preset '{preset}', skipping.")
            continue

        preset_dir = output_dir / preset
        count = 0

        for img_path in image_files:
            # Preserve relative directory structure
            rel_path = img_path.relative_to(input_dir)
            out_path = preset_dir / rel_path.with_suffix(
                ".jpg" if "jpeg" in preset else
                ".webp" if "webp" in preset else
                ".png"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                compress_image_file(img_path, out_path, preset)
                count += 1
            except Exception as e:
                print(f"  ❌ Error compressing {img_path}: {e}")

        stats[preset] = count
        print(f"  ✅ {preset}: {count} images compressed")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compress images for AIGC detection benchmark")
    parser.add_argument("--input", type=str, required=True, help="Input directory of images")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: <input>_compressed)")
    parser.add_argument(
        "--compressions", type=str, default="jpeg_75,jpeg_50,webp_75",
        help="Comma-separated list of compression presets"
    )
    parser.add_argument("--list-presets", action="store_true", help="List available compression presets")
    args = parser.parse_args()

    if args.list_presets:
        print("Available compression presets:")
        for name, config in COMPRESSION_PRESETS.items():
            print(f"  {name:20s} → {config}")
        return

    output_dir = args.output or f"{args.input}_compressed"
    presets = [p.strip() for p in args.compressions.split(",")]

    print(f"Compressing images from: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Presets: {presets}")
    print()

    stats = compress_dataset(args.input, output_dir, presets)

    print(f"\n{'='*40}")
    print("Compression Summary:")
    for preset, count in stats.items():
        print(f"  {preset}: {count} images")


if __name__ == "__main__":
    main()
