"""
Brightness-based feature extraction for AI-generated content detection.

Core insight: AI-generated images do not correctly model diffuse (Lambertian)
reflection as it occurs in the real world. This manifests as subtle but
systematic deviations in brightness distribution, gradients, and local contrast.

We extract features from the luminance channel that capture these deviations
and survive compression far better than high-frequency artifacts.
"""

import numpy as np
from PIL import Image
from typing import Optional


def rgb_to_luminance(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to luminance (Y channel from BT.601).

    Args:
        img: HxWx3 uint8 RGB array.

    Returns:
        HxW float32 luminance array in [0, 1].
    """
    # BT.601 luminance coefficients
    return (
        0.299 * img[:, :, 0].astype(np.float32) +
        0.587 * img[:, :, 1].astype(np.float32) +
        0.114 * img[:, :, 2].astype(np.float32)
    ) / 255.0


def brightness_histogram(luminance: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """
    Compute normalized brightness histogram.

    Real photos exhibit characteristic brightness distributions governed by
    scene illumination. AI-generated images often have subtly different
    distributions due to imperfect radiance modeling.

    Args:
        luminance: HxW float luminance in [0, 1].
        n_bins: Number of histogram bins.

    Returns:
        Normalized histogram of shape (n_bins,).
    """
    hist, _ = np.histogram(luminance, bins=n_bins, range=(0, 1))
    return hist.astype(np.float32) / hist.sum()


def gradient_statistics(luminance: np.ndarray) -> np.ndarray:
    """
    Compute statistics of brightness gradients.

    Diffuse reflection in real scenes creates smooth, physically-consistent
    brightness gradients. AI models often produce gradients that are too
    smooth or have incorrect local structure.

    Returns:
        Feature vector of gradient statistics:
        [mean_gx, std_gx, mean_gy, std_gy, mean_mag, std_mag,
         skew_mag, kurtosis_mag, gradient_entropy]
    """
    # Sobel-like gradients
    gx = np.diff(luminance, axis=1)  # horizontal gradient
    gy = np.diff(luminance, axis=0)  # vertical gradient

    # Gradient magnitude (on overlapping region)
    min_h = min(gx.shape[0], gy.shape[0])
    min_w = min(gx.shape[1], gy.shape[1])
    mag = np.sqrt(gx[:min_h, :min_w] ** 2 + gy[:min_h, :min_w] ** 2)

    features = [
        gx.mean(), gx.std(),
        gy.mean(), gy.std(),
        mag.mean(), mag.std(),
    ]

    # Higher-order statistics of gradient magnitude
    if mag.std() > 1e-8:
        skew = ((mag - mag.mean()) ** 3).mean() / (mag.std() ** 3)
        kurtosis = ((mag - mag.mean()) ** 4).mean() / (mag.std() ** 4) - 3
    else:
        skew = 0.0
        kurtosis = 0.0

    features.extend([skew, kurtosis])

    # Gradient magnitude entropy (binned)
    hist, _ = np.histogram(mag, bins=32, range=(0, mag.max() + 1e-8))
    hist = hist.astype(np.float32)
    hist = hist / hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    features.append(entropy)

    return np.array(features, dtype=np.float32)


def local_contrast_features(
    luminance: np.ndarray,
    block_size: int = 16,
) -> np.ndarray:
    """
    Compute local contrast statistics over non-overlapping blocks.

    Real scenes have physically motivated local contrast patterns (e.g.,
    soft shadows from area lights, inverse-square falloff). AI images
    often have statistically different local contrast distributions.

    Args:
        luminance: HxW float luminance.
        block_size: Size of non-overlapping blocks.

    Returns:
        Feature vector of local contrast statistics:
        [mean_lc, std_lc, skew_lc, kurtosis_lc, min_lc, max_lc,
         lc_range, lc_iqr]
    """
    h, w = luminance.shape
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size

    local_contrasts = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = luminance[
                i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size
            ]
            # Local contrast = std of luminance in block
            lc = block.std()
            local_contrasts.append(lc)

    lc = np.array(local_contrasts, dtype=np.float32)

    if len(lc) == 0 or lc.std() < 1e-8:
        return np.zeros(8, dtype=np.float32)

    skew = ((lc - lc.mean()) ** 3).mean() / (lc.std() ** 3 + 1e-8)
    kurtosis = ((lc - lc.mean()) ** 4).mean() / (lc.std() ** 4 + 1e-8) - 3
    iqr = np.percentile(lc, 75) - np.percentile(lc, 25)

    return np.array([
        lc.mean(), lc.std(), skew, kurtosis,
        lc.min(), lc.max(), lc.max() - lc.min(), iqr,
    ], dtype=np.float32)


def shadow_highlight_transition(luminance: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """
    Analyze shadow-to-highlight transitions.

    In real photos, the transition from shadows to highlights follows
    physically consistent patterns based on light geometry. AI models
    often produce sharper or smoother transitions than physically accurate.

    Returns:
        Feature vector characterizing transition patterns.
    """
    # Classify pixels into shadow/midtone/highlight
    shadow_mask = luminance < 0.3
    highlight_mask = luminance > 0.7
    midtone_mask = ~shadow_mask & ~highlight_mask

    total = luminance.size
    features = [
        shadow_mask.sum() / total,      # shadow ratio
        midtone_mask.sum() / total,     # midtone ratio
        highlight_mask.sum() / total,   # highlight ratio
    ]

    # Mean brightness in each region
    for mask, name in [(shadow_mask, "shadow"), (midtone_mask, "mid"), (highlight_mask, "high")]:
        if mask.sum() > 0:
            features.append(luminance[mask].mean())
            features.append(luminance[mask].std())
        else:
            features.extend([0.0, 0.0])

    # Transition sharpness: histogram of gradient magnitudes at shadow-highlight boundaries
    gx = np.abs(np.diff(luminance, axis=1))
    gy = np.abs(np.diff(luminance, axis=0))
    avg_gradient = (gx.mean() + gy.mean()) / 2
    features.append(avg_gradient)

    return np.array(features, dtype=np.float32)


def extract_features(
    img: Image.Image | np.ndarray,
    histogram_bins: int = 64,
    block_size: int = 16,
) -> np.ndarray:
    """
    Extract full brightness-based feature vector from an image.

    Args:
        img: PIL Image or HxWx3 uint8 numpy array.
        histogram_bins: Number of bins for brightness histogram.
        block_size: Block size for local contrast computation.

    Returns:
        1D feature vector (float32) combining all brightness features.
        Total dimension: histogram_bins + 9 + 8 + 10 = ~91 features.
    """
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))

    luminance = rgb_to_luminance(img)

    # Extract all feature groups
    hist_feats = brightness_histogram(luminance, n_bins=histogram_bins)
    grad_feats = gradient_statistics(luminance)
    contrast_feats = local_contrast_features(luminance, block_size=block_size)
    transition_feats = shadow_highlight_transition(luminance)

    # Concatenate
    return np.concatenate([hist_feats, grad_feats, contrast_feats, transition_feats])


def extract_features_batch(
    image_paths: list[str],
    histogram_bins: int = 64,
    block_size: int = 16,
) -> np.ndarray:
    """
    Extract features from a batch of images.

    Returns:
        NxD feature matrix where N = number of images, D = feature dimension.
    """
    features = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            feat = extract_features(img, histogram_bins, block_size)
            features.append(feat)
        except Exception as e:
            print(f"⚠️  Error processing {path}: {e}")
            continue

    return np.array(features, dtype=np.float32)
