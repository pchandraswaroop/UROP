"""
metrics.py — Per-Class and Aggregate Evaluation Metrics
========================================================
Computes AUC-ROC, AUC-PR (Average Precision), F1, Precision, Recall
for all 14 disease classes and macro aggregates.

Handles edge cases:
  - Classes with no positive examples in a split.
  - Classes with all-positive predictions.
  - NaN metrics (reported separately).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.dataset.label_encoder import DISEASE_LABEL_NAMES

logger = logging.getLogger(__name__)


# ── Metric Computation ────────────────────────────────────────────────────────

def compute_metrics(
    all_probs: np.ndarray,    # [N, 14] predicted probabilities
    all_targets: np.ndarray,  # [N, 14] binary ground truth
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute full suite of metrics.

    Args:
        all_probs:   [N, 14] float probabilities in [0, 1].
        all_targets: [N, 14] binary float labels.
        threshold:   Classification threshold for binary predictions.
        label_names: List of class names for per-class reporting.

    Returns:
        Dict of metric_name → value.
    """
    if label_names is None:
        label_names = DISEASE_LABEL_NAMES

    num_classes = all_probs.shape[1]
    all_preds = (all_probs >= threshold).astype(np.float32)

    metrics: Dict[str, float] = {}

    # Per-class metrics
    auc_list, ap_list, f1_list = [], [], []

    for c in range(num_classes):
        name = label_names[c] if c < len(label_names) else f"class_{c}"
        probs_c   = all_probs[:, c]
        targets_c = all_targets[:, c]
        preds_c   = all_preds[:, c]

        # AUC-ROC (requires at least one positive)
        if targets_c.sum() > 0 and targets_c.sum() < len(targets_c):
            auc = roc_auc_score(targets_c, probs_c)
            ap  = average_precision_score(targets_c, probs_c)
        else:
            auc = float("nan")
            ap  = float("nan")
            logger.debug("Class '%s': no positive examples, AUC set to NaN.", name)

        # F1 / Precision / Recall
        f1   = f1_score(targets_c, preds_c, zero_division=0)
        prec = precision_score(targets_c, preds_c, zero_division=0)
        rec  = recall_score(targets_c, preds_c, zero_division=0)

        metrics[f"{name}/auc"]       = auc
        metrics[f"{name}/ap"]        = ap
        metrics[f"{name}/f1"]        = f1
        metrics[f"{name}/precision"] = prec
        metrics[f"{name}/recall"]    = rec

        if not np.isnan(auc):
            auc_list.append(auc)
        if not np.isnan(ap):
            ap_list.append(ap)
        f1_list.append(f1)

    # Aggregate metrics (skip NaN classes)
    metrics["auc_mean"]     = float(np.mean(auc_list)) if auc_list else float("nan")
    metrics["ap_mean"]      = float(np.mean(ap_list))  if ap_list  else float("nan")
    metrics["f1_macro"]     = float(np.mean(f1_list))  if f1_list  else float("nan")
    metrics["f1_micro"]     = float(f1_score(all_targets, all_preds, average="micro", zero_division=0))
    metrics["n_valid_auc"]  = len(auc_list)
    metrics["threshold"]    = threshold

    return metrics


def format_metrics_table(metrics: Dict[str, float], label_names=None) -> str:
    """Return a human-readable table of per-class metrics."""
    if label_names is None:
        label_names = DISEASE_LABEL_NAMES

    lines = ["=" * 72]
    lines.append(f"{'Disease':<25} {'AUC':>8} {'AP':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    lines.append("-" * 72)

    for name in label_names:
        auc  = metrics.get(f"{name}/auc", float("nan"))
        ap   = metrics.get(f"{name}/ap", float("nan"))
        f1   = metrics.get(f"{name}/f1", float("nan"))
        prec = metrics.get(f"{name}/precision", float("nan"))
        rec  = metrics.get(f"{name}/recall", float("nan"))
        lines.append(
            f"{name:<25} {auc:>8.4f} {ap:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}"
        )

    lines.append("=" * 72)
    lines.append(
        f"{'MACRO AVG':<25} {metrics.get('auc_mean', float('nan')):>8.4f} "
        f"{metrics.get('ap_mean', float('nan')):>8.4f} "
        f"{metrics.get('f1_macro', float('nan')):>8.4f}"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


# ── Accumulator ───────────────────────────────────────────────────────────────

class MetricAccumulator:
    """Collects batch predictions / targets; computes epoch-level metrics."""

    def __init__(self) -> None:
        self._probs:   List[np.ndarray] = []
        self._targets: List[np.ndarray] = []

    def update(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._probs.append(probs.detach().cpu().numpy())
        self._targets.append(targets.detach().cpu().numpy())

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        all_probs   = np.concatenate(self._probs,   axis=0)
        all_targets = np.concatenate(self._targets, axis=0)
        return compute_metrics(all_probs, all_targets, threshold=threshold)

    def reset(self) -> None:
        self._probs.clear()
        self._targets.clear()
