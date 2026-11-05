"""
classifier.py — 14-Head Multi-Label Classifier
===============================================
Takes fused features and produces 14 independent sigmoid outputs.

Input: Either
  - [B, 14, fused_dim] (disease-specific fusion output)
  - [B, fused_dim]     (concatenation fusion output)

Output: [B, 14] probabilities (sigmoid applied)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DiseaseSpecificClassifier(nn.Module):
    """Per-disease classification heads for disease-specific fusion output.

    Operates on [B, 14, fused_dim] → produces [B, 14] logits.
    Each disease has its own classification head.

    Args:
        fused_dim:   Input feature dimension per disease.
        hidden_dim:  Hidden layer dimension.
        num_classes: Number of disease classes (14).
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        fused_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 14,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Shared + per-disease head
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Per-disease output projection: [14, hidden_dim → 1]
        self.per_disease_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])

        logger.info(
            "DiseaseSpecificClassifier: fused_dim=%d, hidden_dim=%d, classes=%d",
            fused_dim, hidden_dim, num_classes
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: [B, 14, fused_dim]

        Returns:
            [B, 14] logits (pre-sigmoid).
        """
        B = fused.shape[0]
        logits = []

        for d in range(self.num_classes):
            feat_d = fused[:, d, :]        # [B, fused_dim]
            hidden = self.shared(feat_d)   # [B, hidden_dim]
            logit  = self.per_disease_heads[d](hidden)  # [B, 1]
            logits.append(logit)

        return torch.cat(logits, dim=-1)   # [B, 14]


class SharedClassifier(nn.Module):
    """Shared MLP classifier for concatenation fusion.

    Input:  [B, fused_dim]
    Output: [B, 14] logits
    """

    def __init__(
        self,
        fused_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 14,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)  # [B, 14]


class MultiModalCXRModel(nn.Module):
    """Full end-to-end model combining all branches.

    This is the top-level model class used for training and inference.

    When the clinical branch is a GNN, a ``clinical_fallback`` (typically a
    MetadataEmbedder) is required for inference on patients who are NOT in
    the training graph.  The GNN path is only used when ``graph_x`` and
    ``graph_edge_index`` are explicitly provided.
    """

    def __init__(
        self,
        image_branch: nn.Module,
        clinical_branch: nn.Module,      # GNN or MLP
        quantum_layer: nn.Module,        # QuantumLayer or Identity
        fusion: nn.Module,
        classifier: nn.Module,
        use_disease_specific_fusion: bool = True,
        clinical_fallback: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.image_branch    = image_branch
        self.clinical_branch = clinical_branch
        self.quantum_layer   = quantum_layer
        self.fusion          = fusion
        self.classifier      = classifier
        self.use_ds_fusion   = use_disease_specific_fusion
        # Fallback for when clinical_branch is a GNN but no graph is provided
        self.clinical_fallback = clinical_fallback

    def forward(
        self,
        images: torch.Tensor,
        meta: torch.Tensor,
        graph_x: torch.Tensor = None,
        graph_edge_index: torch.Tensor = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            images:           [B, 3, H, W]
            meta:             [B, 2]  (age_norm, sex_binary)
            graph_x:          [N, node_feat] or None
            graph_edge_index: [2, E] or None
            return_logits:    If True, return raw logits (pre-sigmoid).
                              Use during training to avoid logit → sigmoid → logit round-trip.
                              If False (default), apply sigmoid and return probabilities.

        Returns:
            [B, 14] logits (if return_logits=True) or probabilities (if False)
        """
        # 1. Image branch
        img_feats    = self.image_branch(images)      # [B, image_dim]
        img_feats    = self.quantum_layer(img_feats)  # [B, quantum_out]

        # 2. Clinical branch
        if graph_x is not None and graph_edge_index is not None:
            # GNN path — real graph message passing if clinical branch accepts edge_index
            try:
                clin_feats = self.clinical_branch(graph_x, graph_edge_index)
            except TypeError:
                clin_feats = self.clinical_branch(meta)
        elif self.clinical_fallback is not None:
            # GNN model but no graph data → use metadata-only fallback
            clin_feats = self.clinical_fallback(meta)
        else:
            # clinical_branch IS the MLP/MetadataEmbedder (non-GNN experiments)
            clin_feats = self.clinical_branch(meta)

        # 3. Fusion
        fused = self.fusion(img_feats, clin_feats)

        # 4. Classify
        logits = self.classifier(fused)   # [B, 14]

        if return_logits:
            return logits                 # [B, 14] raw logits
        return torch.sigmoid(logits)      # [B, 14] probabilities
