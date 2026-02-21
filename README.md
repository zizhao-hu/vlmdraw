# Compressed AIGC Detection Benchmark (CAB)

> **Do AI-Generated Content Detectors Survive Compression?**

## Motivation

Current benchmarks for AI-generated image/video detection evaluate detectors on **raw, uncompressed** outputs from generative models (Stable Diffusion, DALL·E, Midjourney, Sora, etc.). However, in real-world deployment scenarios, AI-generated content is almost always **compressed** before distribution:

- **Images**: JPEG compression (social media uploads, messaging apps)
- **Videos**: H.264/H.265 encoding (YouTube, TikTok, Twitter)
- **Screenshots**: PNG re-encoding, resolution downscaling

This compression destroys many of the subtle artifacts that existing detectors rely on (frequency-domain fingerprints, GAN checkerboard patterns, diffusion noise signatures), causing a **significant drop in detection accuracy** that existing benchmarks fail to capture.

## Contributions

### 1. Compressed AIGC Detection Benchmark (CAB)

We enhance existing AIGC detection benchmarks by applying **realistic compression pipelines**:

| Compression Type | Parameters | Real-World Analogy |
|-----------------|------------|-------------------|
| JPEG | Quality 50, 70, 85, 95 | Social media upload |
| WebP | Quality 50, 75, 90 | Web distribution |
| H.264 (video) | CRF 18, 23, 28, 35 | YouTube/TikTok |
| H.265 (video) | CRF 20, 28, 35 | Streaming platforms |
| Resize + JPEG | 0.5x, 0.75x + Q75 | Mobile sharing |
| Screenshot | PNG re-encode | Screen capture |

We show that **state-of-the-art detectors experience 15-40% accuracy drops** under standard compression, revealing a critical gap in current evaluation methodology.

### 2. Brightness-Based Lightweight Detector (BLD)

We propose a novel, compression-robust detection method based on the observation that:

> **AI-generated images do not correctly model diffuse reflection (Lambertian reflectance) as it occurs in the real world.**

Real photographs exhibit physically consistent brightness patterns governed by Lambert's cosine law, surface albedo, and light source geometry. Generative models approximate these patterns but introduce subtle, systematic deviations in:

- **Brightness gradients** across surfaces
- **Shadow-to-highlight transitions**
- **Specular vs. diffuse reflection ratios**
- **Global illumination consistency**

Our method:
1. Extracts the **luminance channel** (Y from YCbCr or V from HSV)
2. Computes **brightness distribution features** (histogram, gradient statistics, local contrast)
3. Trains a **lightweight classifier** (small MLP or linear probe) on these features
4. Achieves competitive detection with **orders of magnitude fewer parameters** than CNN/ViT-based detectors

Because brightness patterns survive compression far better than high-frequency artifacts, BLD maintains strong performance even on heavily compressed content.

## Project Structure

```
├── data/                    # Data preparation and compression
│   ├── compress.py          # Image/video compression pipeline
│   ├── datasets.py          # Dataset loaders (GenImage, DiffusionDB, etc.)
│   └── download.py          # Dataset download scripts
├── detectors/               # Existing detector wrappers
│   ├── base.py              # Base detector interface
│   ├── cnn_based.py         # CNNDetect, Wang2020, etc.
│   ├── clip_based.py        # DIRE, UnivFD, etc.
│   └── frequency.py         # Frequency-domain methods (F3Net, etc.)
├── bld/                     # Brightness-based Lightweight Detector
│   ├── features.py          # Brightness feature extraction
│   ├── model.py             # Lightweight classifier
│   └── train.py             # Training pipeline
├── benchmark/               # Benchmark evaluation
│   ├── evaluate.py          # Run detectors on compressed data
│   ├── metrics.py           # Accuracy, AUC, robustness metrics
│   └── visualize.py         # Result plotting
├── configs/                 # Experiment configs
├── scripts/                 # SLURM job scripts
└── paper/                   # LaTeX paper draft
```

## Existing Benchmarks to Enhance

- **GenImage** (Zhu et al., 2023) — 1M+ images from 8 generators
- **DiffusionForensics** (Wang et al., 2023) — Diffusion model forensics
- **AIGCDetect** — Multi-source AI-generated content
- **FakeAVCeleb / FaceForensics++** — Deepfake video datasets

## Existing Detectors to Evaluate

| Detector | Type | Year |
|----------|------|------|
| CNNDetect (Wang et al.) | CNN Binary Classifier | 2020 |
| DIRE | Diffusion Reconstruction Error | 2023 |
| UnivFD | CLIP-based Universal | 2023 |
| F3Net | Frequency Domain | 2020 |
| NPR | Neighboring Pixel Relationships | 2024 |
| DRCT | Diffusion Reconstruction Contrastive | 2024 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download and compress a benchmark dataset
python data/download.py --dataset genimage --split test
python data/compress.py --input data/genimage/test --compressions jpeg_75,jpeg_50,webp_75

# Evaluate existing detectors on compressed data
python benchmark/evaluate.py --detector cnndetect --data data/genimage/test_compressed

# Train BLD (our method)
python bld/train.py --data data/genimage/train --features luminance_hist,gradient_stats

# Evaluate BLD on compressed data
python benchmark/evaluate.py --detector bld --data data/genimage/test_compressed
```

## License

MIT
