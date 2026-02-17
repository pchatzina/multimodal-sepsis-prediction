"""Shared evaluation utilities for all modality classifiers.

Provides:
- ``load_embeddings(path)`` — load a ``.pt`` embedding file into numpy arrays.
- ``compute_metrics(y_true, y_prob, threshold)`` — return a dict of standard
  binary-classification metrics.
- ``print_metrics(metrics, name)`` — pretty-print a metrics dict.
- ``log_metrics_to_tensorboard(writer, metrics, step, prefix)`` — write
  scalars to a ``SummaryWriter``.
- ``save_predictions(path, sample_ids, y_true, y_prob, y_pred)`` — dump a CSV.
- ``save_metrics(path, metrics)`` — dump metrics to JSON for reproducibility.

All classifier scripts (LR, XGBoost, MLP) across all modalities should import
from here instead of duplicating helpers.

Usage:
    from src.utils.evaluation import (
        load_embeddings, compute_metrics, print_metrics,
        log_metrics_to_tensorboard, save_predictions, save_metrics,
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ==========================================
# DATA LOADING
# ==========================================


def load_embeddings(
    filepath: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a ``.pt`` embedding file.

    Expected format (produced by every extract_embeddings script)::

        {
            "embeddings": Tensor[N, D],
            "labels":     List[str],       # "0" / "1"
            "sample_ids": List[str],
        }

    Returns:
        X:          np.ndarray of shape (N, D), float32
        y:          np.ndarray of shape (N,), int
        sample_ids: list of str
    """
    filepath = Path(filepath)
    logger.info("Loading embeddings from %s", filepath)
    data = torch.load(filepath, map_location="cpu", weights_only=False)

    X = data["embeddings"].cpu().numpy().astype(np.float32)
    y = np.array([int(label) for label in data["labels"]], dtype=int)
    sample_ids = list(data["sample_ids"])

    logger.info("  → %d samples, dim %d", X.shape[0], X.shape[1])
    return X, y, sample_ids


# ==========================================
# METRICS
# ==========================================


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard binary-classification metrics.

    Returns a flat dict suitable for JSON serialisation and TensorBoard logging.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }


def print_metrics(metrics: Dict[str, float], name: str = "") -> None:
    """Pretty-print a metrics dict to stdout."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}  (PPV)")
    print(f"  Recall:      {metrics['recall']:.4f}  (Sensitivity)")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(
        f"  Samples:     {metrics['n_samples']}  "
        f"(+{metrics['n_positive']} / -{metrics['n_negative']})"
    )
    print(
        f"  Confusion:   TP={metrics['tp']}  FP={metrics['fp']}  "
        f"FN={metrics['fn']}  TN={metrics['tn']}"
    )
    print(f"{'=' * 50}")


# ==========================================
# TENSORBOARD
# ==========================================


def log_metrics_to_tensorboard(
    writer,  # torch.utils.tensorboard.SummaryWriter
    metrics: Dict[str, float],
    global_step: int = 0,
    prefix: str = "",
) -> None:
    """Write scalar metrics to an open TensorBoard ``SummaryWriter``.

    Only numeric values from *metrics* are logged (confusion counts, etc.
    are included).  The *prefix* is prepended as ``prefix/metric_name``.
    """
    tag_prefix = f"{prefix}/" if prefix else ""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{tag_prefix}{key}", value, global_step)


# ==========================================
# PERSISTENCE
# ==========================================


def save_predictions(
    path: Union[str, Path],
    sample_ids: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> None:
    """Save per-sample predictions as a CSV file."""
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": y_true,
            "probability": y_prob,
            "prediction": y_pred,
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Predictions saved → %s  (%d rows)", path, len(df))


def save_metrics(
    path: Union[str, Path],
    metrics: Dict[str, float],
) -> None:
    """Save metrics dict as a JSON file for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", path)
