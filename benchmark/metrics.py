"""
Evaluation metrics for AIGC detection benchmark.

Measures detector performance across compression levels with
standard metrics and compression-robustness metrics.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    """Results from evaluating a detector on a specific dataset/compression."""
    detector_name: str
    dataset_name: str
    compression: str  # e.g., "raw", "jpeg_75", "jpeg_50"
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    n_samples: int
    # Per-class accuracy
    real_acc: float = 0.0
    fake_acc: float = 0.0


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute standard binary classification metrics.

    Args:
        labels: Ground-truth binary labels (0 = real, 1 = fake).
        predictions: Predicted binary labels.
        probabilities: Predicted probabilities for the positive class.

    Returns:
        Dict with accuracy, AUC, precision, recall, F1, per-class accuracy.
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score,
        precision_score, recall_score, f1_score,
    )

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "n_samples": len(labels),
    }

    if probabilities is not None:
        try:
            metrics["auc"] = roc_auc_score(labels, probabilities)
        except ValueError:
            metrics["auc"] = 0.5  # Only one class present
    else:
        metrics["auc"] = 0.0

    # Per-class accuracy
    real_mask = labels == 0
    fake_mask = labels == 1
    if real_mask.sum() > 0:
        metrics["real_acc"] = accuracy_score(labels[real_mask], predictions[real_mask])
    if fake_mask.sum() > 0:
        metrics["fake_acc"] = accuracy_score(labels[fake_mask], predictions[fake_mask])

    return metrics


def compression_robustness_score(
    raw_metrics: dict,
    compressed_metrics: dict,
    metric_key: str = "accuracy",
) -> float:
    """
    Compute compression robustness score (CRS).

    CRS = compressed_metric / raw_metric

    A score of 1.0 means no degradation, < 1.0 means the detector
    performs worse on compressed data.

    Args:
        raw_metrics: Metrics on raw (uncompressed) data.
        compressed_metrics: Metrics on compressed data.
        metric_key: Which metric to compare.

    Returns:
        Robustness score in [0, ∞).
    """
    raw_val = raw_metrics.get(metric_key, 0)
    if raw_val == 0:
        return 0.0
    return compressed_metrics.get(metric_key, 0) / raw_val


def compute_robustness_profile(
    results: list[DetectionResult],
    detector_name: str,
) -> dict:
    """
    Compute a full robustness profile for a detector across all compressions.

    Returns:
        Dict mapping compression name to robustness scores.
    """
    detector_results = [r for r in results if r.detector_name == detector_name]

    # Find raw (uncompressed) baseline
    raw = [r for r in detector_results if r.compression == "raw"]
    if not raw:
        return {}

    raw_metrics = {"accuracy": raw[0].accuracy, "auc": raw[0].auc, "f1": raw[0].f1}

    profile = {"raw": {"accuracy": 1.0, "auc": 1.0, "f1": 1.0}}
    for r in detector_results:
        if r.compression == "raw":
            continue
        compressed = {"accuracy": r.accuracy, "auc": r.auc, "f1": r.f1}
        profile[r.compression] = {
            "accuracy_crs": compression_robustness_score(raw_metrics, compressed, "accuracy"),
            "auc_crs": compression_robustness_score(raw_metrics, compressed, "auc"),
            "f1_crs": compression_robustness_score(raw_metrics, compressed, "f1"),
            "raw_accuracy": r.accuracy,
            "raw_auc": r.auc,
        }

    return profile


def format_results_table(results: list[DetectionResult]) -> str:
    """Format results as a readable table string."""
    header = f"{'Detector':<15} {'Dataset':<15} {'Compression':<15} {'Acc':>6} {'AUC':>6} {'F1':>6}"
    sep = "-" * len(header)
    lines = [header, sep]

    for r in sorted(results, key=lambda x: (x.detector_name, x.compression)):
        line = f"{r.detector_name:<15} {r.dataset_name:<15} {r.compression:<15} {r.accuracy:>6.3f} {r.auc:>6.3f} {r.f1:>6.3f}"
        lines.append(line)

    return "\n".join(lines)
