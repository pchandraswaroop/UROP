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

def find_optimal_thresholds(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    label_names: Optional[List[str]] = None,
    min_positives: int = 1,
    fallback: float = 0.5,
    candidates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Find per-class F1-optimal decision thresholds using validation data only.

    CRITICAL CLINICAL & METHODOLOGICAL INVARIANT:
      Threshold optimization MUST be performed strictly on validation predictions
      and validation labels. Held-out test labels are never accessed or observed.

    Args:
        val_probs:     [N, C] predicted probabilities on the validation set.
        val_targets:   [N, C] binary ground-truth labels on the validation set.
        label_names:   Optional list of class names.
        min_positives: Minimum positive examples required in validation set to tune.
        fallback:      Default threshold when class support is insufficient (default 0.50).
        candidates:    Array of threshold search values (default: 0.05 to 0.90 in steps of 0.01).

    Returns:
        [C] array of per-class optimal thresholds in [0, 1].
    """
    if candidates is None:
        candidates = np.arange(0.05, 0.91, 0.01)

    num_classes = val_probs.shape[1]
    best_thresholds = np.full(num_classes, fallback, dtype=np.float32)

    for c in range(num_classes):
        targets_c = val_targets[:, c]
        probs_c = val_probs[:, c]

        if targets_c.sum() < min_positives:
            best_thresholds[c] = fallback
            continue

        best_f1 = -1.0
        best_t = fallback

        for t in candidates:
            preds_c = (probs_c >= t).astype(np.float32)
            f1 = f1_score(targets_c, preds_c, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
            elif f1 == best_f1 and abs(t - 0.50) < abs(best_t - 0.50):
                best_t = t

        best_thresholds[c] = float(best_t) if best_f1 > 0.0 else fallback

    return best_thresholds


def compute_metrics(
    all_probs: np.ndarray,    # [N, 14] predicted probabilities
    all_targets: np.ndarray,  # [N, 14] binary ground truth
    threshold: float | np.ndarray = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute full suite of metrics.

    Args:
        all_probs:   [N, 14] float probabilities in [0, 1].
        all_targets: [N, 14] binary float labels.
        threshold:   Classification threshold for binary predictions (scalar or [14] array).
        label_names: List of class names for per-class reporting.

    Returns:
        Dict of metric_name → value.
    """
    if label_names is None:
        label_names = DISEASE_LABEL_NAMES

    num_classes = all_probs.shape[1]
    if isinstance(threshold, (int, float)):
        all_preds = (all_probs >= threshold).astype(np.float32)
        thresh_arr = np.full(num_classes, float(threshold), dtype=np.float32)
    else:
        thresh_arr = np.asarray(threshold, dtype=np.float32)
        all_preds = (all_probs >= thresh_arr[None, :]).astype(np.float32)

    metrics: Dict[str, float] = {}

    # Per-class metrics
    auc_list, ap_list, f1_list = [], [], []
    prec_list, rec_list, spec_list, acc_list, bal_acc_list = [], [], [], [], []

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

        # F1 / Precision / Recall (Sensitivity)
        f1   = f1_score(targets_c, preds_c, zero_division=0)
        prec = precision_score(targets_c, preds_c, zero_division=0)
        rec  = recall_score(targets_c, preds_c, zero_division=0)

        # Confusion components for class c
        tp = float(np.sum((targets_c == 1) & (preds_c == 1)))
        tn = float(np.sum((targets_c == 0) & (preds_c == 0)))
        fp = float(np.sum((targets_c == 0) & (preds_c == 1)))
        fn = float(np.sum((targets_c == 1) & (preds_c == 0)))
        total_c = len(targets_c)

        acc  = (tp + tn) / total_c if total_c > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens = rec  # Clinical standard: Sensitivity is True Positive Rate (Recall)
        bal_acc = (sens + spec) / 2.0

        metrics[f"{name}/auc"]               = auc
        metrics[f"{name}/ap"]                = ap
        metrics[f"{name}/f1"]                = f1
        metrics[f"{name}/precision"]         = prec
        metrics[f"{name}/recall"]            = rec
        metrics[f"{name}/sensitivity"]       = sens
        metrics[f"{name}/specificity"]       = spec
        metrics[f"{name}/accuracy"]          = acc
        metrics[f"{name}/balanced_accuracy"] = bal_acc
        metrics[f"{name}/threshold"]         = float(thresh_arr[c])

        if not np.isnan(auc):
            auc_list.append(auc)
        if not np.isnan(ap):
            ap_list.append(ap)
        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)
        spec_list.append(spec)
        acc_list.append(acc)
        bal_acc_list.append(bal_acc)

    # Aggregate metrics (skip NaN classes)
    metrics["auc_mean"]                 = float(np.mean(auc_list)) if auc_list else float("nan")
    metrics["ap_mean"]                  = float(np.mean(ap_list))  if ap_list  else float("nan")
    metrics["f1_macro"]                 = float(np.mean(f1_list))  if f1_list  else float("nan")
    metrics["f1_micro"]                 = float(f1_score(all_targets, all_preds, average="micro", zero_division=0))
    metrics["precision_macro"]          = float(np.mean(prec_list)) if prec_list else float("nan")
    metrics["recall_macro"]             = float(np.mean(rec_list))  if rec_list  else float("nan")
    metrics["sensitivity_macro"]        = metrics["recall_macro"]
    metrics["specificity_macro"]        = float(np.mean(spec_list)) if spec_list else float("nan")
    metrics["accuracy_macro"]           = float(np.mean(acc_list))  if acc_list  else float("nan")
    metrics["balanced_accuracy_macro"]  = float(np.mean(bal_acc_list)) if bal_acc_list else float("nan")

    # Global multi-label accuracy metrics:
    metrics["subset_accuracy"]          = float(np.mean(np.all(all_targets == all_preds, axis=1))) if len(all_targets) > 0 else float("nan")
    metrics["hamming_accuracy"]         = float(np.mean(all_targets == all_preds)) if len(all_targets) > 0 else float("nan")

    # Micro aggregates
    metrics["precision_micro"]          = float(precision_score(all_targets, all_preds, average="micro", zero_division=0))
    metrics["recall_micro"]             = float(recall_score(all_targets, all_preds, average="micro", zero_division=0))
    metrics["sensitivity_micro"]        = metrics["recall_micro"]
    total_tn = float(np.sum((all_targets == 0) & (all_preds == 0)))
    total_fp = float(np.sum((all_targets == 0) & (all_preds == 1)))
    metrics["specificity_micro"]        = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0

    metrics["n_valid_auc"]              = len(auc_list)
    metrics["threshold"]                = float(np.mean(thresh_arr))
    metrics["thresholds"]               = thresh_arr.tolist()

    return metrics


def format_metrics_table(metrics: Dict[str, float], label_names=None) -> str:
    """Return a human-readable table of per-class metrics."""
    if label_names is None:
        label_names = DISEASE_LABEL_NAMES

    lines = ["=" * 88]
    lines.append(f"{'Disease':<22} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Sens/Rec':>9} {'Spec':>7} {'Acc':>7} {'BalAcc':>7}")
    lines.append("-" * 88)

    for name in label_names:
        auc  = metrics.get(f"{name}/auc", float("nan"))
        f1   = metrics.get(f"{name}/f1", float("nan"))
        prec = metrics.get(f"{name}/precision", float("nan"))
        rec  = metrics.get(f"{name}/recall", float("nan"))
        spec = metrics.get(f"{name}/specificity", float("nan"))
        acc  = metrics.get(f"{name}/accuracy", float("nan"))
        bal  = metrics.get(f"{name}/balanced_accuracy", float("nan"))
        lines.append(
            f"{name:<22} {auc:>7.4f} {f1:>7.4f} {prec:>7.4f} {rec:>9.4f} {spec:>7.4f} {acc:>7.4f} {bal:>7.4f}"
        )

    lines.append("=" * 88)
    lines.append(
        f"{'MACRO AVERAGE':<22} "
        f"{metrics.get('auc_mean', float('nan')):>7.4f} "
        f"{metrics.get('f1_macro', float('nan')):>7.4f} "
        f"{metrics.get('precision_macro', float('nan')):>7.4f} "
        f"{metrics.get('recall_macro', float('nan')):>9.4f} "
        f"{metrics.get('specificity_macro', float('nan')):>7.4f} "
        f"{metrics.get('accuracy_macro', float('nan')):>7.4f} "
        f"{metrics.get('balanced_accuracy_macro', float('nan')):>7.4f}"
    )
    lines.append("=" * 88)
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
        all_probs   = np.concatenate(self._probs,   axis=0) if self._probs else np.empty((0, 14))
        all_targets = np.concatenate(self._targets, axis=0) if self._targets else np.empty((0, 14))
        return compute_metrics(all_probs, all_targets, threshold=threshold)

    def get_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return concatenated raw probabilities and binary targets."""
        all_probs   = np.concatenate(self._probs,   axis=0) if self._probs else np.empty((0, 14))
        all_targets = np.concatenate(self._targets, axis=0) if self._targets else np.empty((0, 14))
        return all_probs, all_targets

    def reset(self) -> None:
        self._probs.clear()
        self._targets.clear()
