"""
mlp_baseline.py — MLP Baseline on Metadata Only
================================================
A simple MLP that classifies all 14 diseases from metadata alone (age, sex).

Purpose:
  - Establishes a lower bound for the GNN branch.
  - If the GNN does not significantly outperform this MLP, the graph signal
    is insufficient and the GNN branch should be dropped.
  - Also used as the fallback clinical branch for val/test patients who are
    not present in the training graph.

Input:  [B, 2]  (age_norm, sex_binary)
Output: [B, 14] (logits, apply sigmoid for probabilities)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLPBaseline(nn.Module):
    """Metadata-only MLP for 14-class multi-label classification.

    Args:
        input_dim:  Metadata feature dimension (2 for age+sex).
        hidden_dim: Hidden layer dimension.
        num_classes: Number of disease classes (14).
        dropout:    Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_classes: int = 14,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        logger.info(
            "MLPBaseline: input_dim=%d, hidden_dim=%d, num_classes=%d",
            input_dim, hidden_dim, num_classes
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            meta: [B, input_dim] metadata features.

        Returns:
            [B, num_classes] logits (pre-sigmoid).
        """
        return self.net(meta)


class MetadataEmbedder(nn.Module):
    """Project metadata to a richer embedding for use in fusion.

    This differs from MLPBaseline in that it produces an embedding vector
    (rather than classification logits) to be fused with image features.

    Input:  [B, 2]
    Output: [B, output_dim]
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = output_dim

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)
