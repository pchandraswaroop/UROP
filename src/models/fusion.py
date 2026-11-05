"""
fusion.py — Disease-Specific Multimodal Fusion
===============================================
Fuses image features and clinical (GNN/MLP) features.

Strategy: Disease-Specific Fusion
----------------------------------
Rather than a single global fusion, we apply a per-disease gating mechanism:
  - For each of the 14 diseases, a learned attention gate determines how
    much to weight the image branch vs. the clinical branch.
  - This allows some diseases to be driven more by imaging (e.g., Pneumothorax)
    while others may benefit from demographic information.

Input:
  - image_features:    [B, image_dim]
  - clinical_features: [B, clinical_dim]

Output:
  - fused_features: [B, 14, fused_dim]  — one fused vector per disease
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DiseaseSpecificFusion(nn.Module):
    """Per-disease attention-gated multimodal fusion.

    For each disease class d in {0..13}:
      gate_d = sigmoid(W_d @ [image_feat; clinical_feat] + b_d)
      fused_d = gate_d * image_feat + (1 - gate_d) * clinical_feat
      → project to fused_dim

    Args:
        image_dim:    Dimension of image branch output.
        clinical_dim: Dimension of clinical branch output (GNN or MLP).
        fused_dim:    Dimension of each per-disease fused vector.
        num_classes:  Number of disease classes (14).
        dropout:      Dropout rate.
    """

    def __init__(
        self,
        image_dim: int,
        clinical_dim: int,
        fused_dim: int = 256,
        num_classes: int = 14,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.num_classes  = num_classes
        self.fused_dim    = fused_dim
        concat_dim        = image_dim + clinical_dim

        # Per-disease gating: shared input, independent gate per disease
        # Shape: [14, concat_dim] — one gate vector per disease
        self.gate_weights = nn.Parameter(
            torch.randn(num_classes, concat_dim) * 0.01
        )
        self.gate_biases  = nn.Parameter(torch.zeros(num_classes))

        # Projection: fused vector → fused_dim (shared across diseases)
        self.proj = nn.Sequential(
            nn.Linear(image_dim, fused_dim),  # project image features to fused_dim
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        logger.info(
            "DiseaseSpecificFusion: image_dim=%d, clinical_dim=%d, fused_dim=%d, classes=%d",
            image_dim, clinical_dim, fused_dim, num_classes
        )

    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse image and clinical features with per-disease gating.

        Args:
            image_features:    [B, image_dim]
            clinical_features: [B, clinical_dim]

        Returns:
            fused: [B, num_classes, fused_dim]
        """
        B = image_features.shape[0]

        # Concatenate for gate computation
        concat = torch.cat([image_features, clinical_features], dim=-1)  # [B, concat_dim]

        # Per-disease gate: [B, num_classes]
        # gate_weights: [num_classes, concat_dim]
        gates = torch.sigmoid(
            torch.einsum("bd,cd->bc", concat, self.gate_weights) + self.gate_biases
        )  # [B, 14]

        # Project both modalities to fused_dim
        img_proj  = self.proj(image_features)       # [B, fused_dim]
        clin_proj = self.clinical_proj(clinical_features)  # [B, fused_dim]

        # Gate: gates[:, :, None] is [B, 14, 1]
        img_expanded  = img_proj.unsqueeze(1).expand(B, self.num_classes, -1)   # [B, 14, fused_dim]
        clin_expanded = clin_proj.unsqueeze(1).expand(B, self.num_classes, -1)  # [B, 14, fused_dim]

        gates_expanded = gates.unsqueeze(-1)  # [B, 14, 1]
        fused = gates_expanded * img_expanded + (1 - gates_expanded) * clin_expanded  # [B, 14, fused_dim]

        return fused


class SimpleConcatFusion(nn.Module):
    """Simple concatenation + projection fusion (baseline alternative).

    Input:  [B, image_dim] + [B, clinical_dim]
    Output: [B, fused_dim]
    """

    def __init__(
        self,
        image_dim: int,
        clinical_dim: int,
        fused_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim + clinical_dim, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
        )
        self.fused_dim = fused_dim

    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([image_features, clinical_features], dim=-1))
