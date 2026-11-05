"""
losses.py — Weighted Binary Cross-Entropy for Multi-Label Classification
=========================================================================
NIH ChestX-ray14 has severe class imbalance (e.g., Hernia: ~0.2%,
Infiltration: ~17%). We use per-class positive weighting.

pos_weight[i] = (N - N_pos[i]) / N_pos[i]  (capped at pos_weight_cap)

This is the standard approach from PyTorch BCEWithLogitsLoss documentation.

Note: The classifier outputs probabilities (after sigmoid). We re-compute
logits by inverting sigmoid here, OR the classifier should return logits.
We configure MultiModalCXRModel to return logits during training
(sigmoid applied only for inference).
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Loss ──────────────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """BCEWithLogitsLoss with per-class positive weights.

    Args:
        pos_weights:  [14] tensor of positive class weights, or None for uniform.
        pos_weight_cap: Maximum pos_weight value (avoids extreme weights).
        reduction:    "mean" or "sum".
    """

    def __init__(
        self,
        pos_weights: Optional[torch.Tensor] = None,
        pos_weight_cap: float = 10.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if pos_weights is not None:
            pos_weights = torch.clamp(pos_weights, max=pos_weight_cap)
            logger.info(
                "WeightedBCELoss: pos_weights min=%.2f, max=%.2f",
                pos_weights.min().item(), pos_weights.max().item()
            )

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weights,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 14] raw logits (pre-sigmoid).
            targets: [B, 14] float binary targets.

        Returns:
            Scalar loss.
        """
        return self.loss_fn(logits, targets)


# ── Pos-Weight Computation ────────────────────────────────────────────────────

def compute_pos_weights(
    train_df: pd.DataFrame,
    num_classes: int = 14,
    cap: float = 10.0,
) -> torch.Tensor:
    """Compute per-class positive weights from training labels.

    Args:
        train_df:    Training split DataFrame with 'labels_raw' column.
        num_classes: Number of disease classes (14).
        cap:         Maximum weight value.

    Returns:
        [14] float tensor of positive weights.
    """
    from src.dataset.label_encoder import safe_parse_labels

    N = len(train_df)
    pos_counts = np.zeros(num_classes)

    label_col = "labels_raw" if "labels_raw" in train_df.columns else "labels"
    for raw_item in train_df[label_col]:
        parsed = safe_parse_labels(raw_item)
        for nih_idx in parsed:
            if nih_idx == 0:
                continue  # No Finding
            disease_idx = nih_idx - 1
            if 0 <= disease_idx < num_classes:
                pos_counts[disease_idx] += 1

    neg_counts = N - pos_counts
    # Avoid division by zero
    pos_counts_safe = np.where(pos_counts == 0, 1, pos_counts)
    weights = neg_counts / pos_counts_safe
    weights = np.clip(weights, 1.0, cap)

    logger.info(
        "Pos weights: min=%.2f, max=%.2f, mean=%.2f",
        weights.min(), weights.max(), weights.mean()
    )
    return torch.tensor(weights, dtype=torch.float32)
