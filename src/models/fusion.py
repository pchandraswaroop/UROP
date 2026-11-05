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

    Supports both 3-modality fusion (Quantum Image, GNN Scan Graph, Clinical MLP)
    and 2-modality fusion (Image, Clinical).

    In 3-Modality Mode:
      For each disease class d in {0..13}:
        gate_d = Softmax(W_d @ [image_feat; clinical_feat; gnn_feat] + b_d)  (3 normalized weights)
        fused_d = gate_d[0]*proj_img + gate_d[1]*proj_clin + gate_d[2]*proj_gnn

    In 2-Modality Mode:
      gate_d = Sigmoid(W_d @ [image_feat; clinical_feat] + b_d)
      fused_d = gate_d * proj_img + (1 - gate_d) * proj_clin

    Args:
        image_dim:    Dimension of image branch output (e.g. 64).
        clinical_dim: Dimension of clinical MLP branch output (e.g. 64).
        fused_dim:    Dimension of each per-disease fused vector (e.g. 128).
        num_classes:  Number of disease classes (14).
        gnn_dim:      Optional dimension of GNN patient graph output (e.g. 64).
        dropout:      Dropout rate.
    """

    def __init__(
        self,
        image_dim: int,
        clinical_dim: int,
        fused_dim: int = 256,
        num_classes: int = 14,
        gnn_dim: Optional[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.num_classes  = num_classes
        self.fused_dim    = fused_dim
        self.image_dim    = image_dim
        self.clinical_dim = clinical_dim
        self.gnn_dim      = gnn_dim
        self.has_gnn      = (gnn_dim is not None)

        concat_dim = image_dim + clinical_dim + (gnn_dim if self.has_gnn else 0)
        self.concat_dim = concat_dim

        # Projection heads for each modality into shared fused_dim
        self.proj = nn.Sequential(
            nn.Linear(image_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if self.has_gnn:
            self.gnn_proj = nn.Sequential(
                nn.Linear(gnn_dim, fused_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            # 3-modality gating: [num_classes, 3, concat_dim]
            self.gate_proj = nn.Linear(concat_dim, num_classes * 3)
        else:
            # 2-modality gating: [num_classes, concat_dim]
            self.gate_weights = nn.Parameter(
                torch.randn(num_classes, concat_dim) * 0.01
            )
            self.gate_biases = nn.Parameter(torch.zeros(num_classes))

        logger.info(
            "DiseaseSpecificFusion: image_dim=%d, clinical_dim=%d, gnn_dim=%s, fused_dim=%d, classes=%d",
            image_dim, clinical_dim, str(gnn_dim), fused_dim, num_classes
        )

    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: torch.Tensor,
        gnn_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse multimodal representations with per-disease attention gating.

        Args:
            image_features:    [B, image_dim]
            clinical_features: [B, clinical_dim]
            gnn_features:      [B, gnn_dim] (optional)

        Returns:
            fused: [B, num_classes, fused_dim]
        """
        B = image_features.shape[0]

        img_proj  = self.proj(image_features)          # [B, fused_dim]
        clin_proj = self.clinical_proj(clinical_features) # [B, fused_dim]

        if self.has_gnn and gnn_features is not None:
            gnn_proj = self.gnn_proj(gnn_features)    # [B, fused_dim]
            concat = torch.cat([image_features, clinical_features, gnn_features], dim=-1)  # [B, concat_dim]

            # Compute 3-way softmax gating across [img, clin, gnn] per disease
            # Shape: [B, num_classes, 3]
            gate_logits = self.gate_proj(concat).view(B, self.num_classes, 3)
            gate_weights = torch.softmax(gate_logits, dim=-1)

            g_img  = gate_weights[:, :, 0:1]  # [B, num_classes, 1]
            g_clin = gate_weights[:, :, 1:2]  # [B, num_classes, 1]
            g_gnn  = gate_weights[:, :, 2:3]  # [B, num_classes, 1]

            img_expanded  = img_proj.unsqueeze(1).expand(B, self.num_classes, -1)
            clin_expanded = clin_proj.unsqueeze(1).expand(B, self.num_classes, -1)
            gnn_expanded  = gnn_proj.unsqueeze(1).expand(B, self.num_classes, -1)

            fused = g_img * img_expanded + g_clin * clin_expanded + g_gnn * gnn_expanded
            return fused
        else:
            # 2-modality gating (Image + Clinical)
            concat = torch.cat([image_features, clinical_features], dim=-1)  # [B, concat_dim]

            if hasattr(self, "gate_weights"):
                gates = torch.sigmoid(
                    torch.einsum("bd,cd->bc", concat, self.gate_weights) + self.gate_biases
                )  # [B, 14]
            else:
                gates = torch.sigmoid(self.gate_proj(concat).view(B, self.num_classes, -1)[:, :, 0])

            img_expanded  = img_proj.unsqueeze(1).expand(B, self.num_classes, -1)
            clin_expanded = clin_proj.unsqueeze(1).expand(B, self.num_classes, -1)

            gates_expanded = gates.unsqueeze(-1)  # [B, 14, 1]
            fused = gates_expanded * img_expanded + (1.0 - gates_expanded) * clin_expanded
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
