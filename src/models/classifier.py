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

    Supports:
      - 3-Branch Proposed Model (Quantum Image + Scan Graph GNN + Clinical MLP)
      - 2-Branch Baseline Models (Image-Only, Classical Multimodal, Quantum-CNN)

    In the 3-Branch Architecture (Quantum-GNN Multimodal):
      1. Quantum Image Branch: EfficientNet-B0 + PennyLane 4-qubit VQC -> [B, 64]
      2. Patient Scan Graph GNN: GraphSAGE operates over scan nodes containing
         [img_feats, meta] (dimension 66) connected by patient-scan edges -> [B, 64]
      3. Clinical MLP: Normalized Age and Sex processed through MetadataEmbedder -> [B, 64]
      4. Disease-Specific Fusion: 14 per-disease 3-way softmax gating heads -> [B, 14, 128]
      5. Final Classifier: 14 independent sigmoid multi-label classification heads -> [B, 14]
    """

    def __init__(
        self,
        image_branch: nn.Module,
        clinical_branch: nn.Module,      # Clinical MLP or MetadataEmbedder
        quantum_layer: nn.Module,        # QuantumLayer or Identity
        fusion: nn.Module,               # DiseaseSpecificFusion or SimpleConcatFusion
        classifier: nn.Module,           # DiseaseSpecificClassifier or SharedClassifier
        gnn_branch: nn.Module | None = None,  # Dedicated GNN branch
        use_disease_specific_fusion: bool = True,
        clinical_fallback: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.image_branch    = image_branch
        self.clinical_branch = clinical_branch
        self.quantum_layer   = quantum_layer
        self.fusion          = fusion
        self.classifier      = classifier
        self.gnn_branch      = gnn_branch
        self.use_ds_fusion   = use_disease_specific_fusion
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
            graph_x:          Optional pre-computed node feature tensor
            graph_edge_index: Optional [2, E] edge list connecting scans of the same patient
            return_logits:    If True, return raw logits (pre-sigmoid).

        Returns:
            [B, 14] logits (if return_logits=True) or probabilities (if False)
        """
        B = images.shape[0]

        # 1. Image branch (Visual extraction + Quantum enhancement)
        img_feats = self.image_branch(images)          # [B, image_dim]
        img_feats = self.quantum_layer(img_feats)      # [B, quantum_out]

        # 2. Branch Processing
        if self.gnn_branch is not None:
            # 3-Branch Architecture:
            # Clinical MLP processes demographic metadata
            clin_feats = self.clinical_branch(meta)    # [B, clinical_out]

            # GNN Branch: Node features incorporate meaningful scan visual features + demographics
            if graph_x is not None and graph_x.shape[-1] == (img_feats.shape[-1] + meta.shape[-1]):
                node_feats = graph_x
            else:
                node_feats = torch.cat([img_feats, meta], dim=-1)  # [B, q_out + 2]

            # If no graph edges provided, default to self-loops (isolated scans)
            if graph_edge_index is None:
                graph_edge_index = torch.arange(B, device=images.device, dtype=torch.long).repeat(2, 1)

            gnn_feats = self.gnn_branch(node_feats, graph_edge_index)  # [B_nodes, gnn_out]
            if gnn_feats.shape[0] != B:
                gnn_feats = gnn_feats[:B]

            # 3. Multimodal Fusion (Image + Clinical MLP + Scan Graph GNN)
            fused = self.fusion(img_feats, clin_feats, gnn_features=gnn_feats)

        elif graph_x is not None and graph_edge_index is not None:
            # Legacy 2-branch with GNN as clinical branch
            try:
                clin_feats = self.clinical_branch(graph_x, graph_edge_index)
            except TypeError:
                clin_feats = self.clinical_branch(meta)
            fused = self.fusion(img_feats, clin_feats)

        elif self.clinical_fallback is not None:
            clin_feats = self.clinical_fallback(meta)
            fused = self.fusion(img_feats, clin_feats)

        else:
            clin_feats = self.clinical_branch(meta)
            fused = self.fusion(img_feats, clin_feats)

        # 4. Final Classification (14 independent multi-label outputs)
        logits = self.classifier(fused)   # [B, 14]

        if return_logits:
            return logits                 # [B, 14] raw logits
        return torch.sigmoid(logits)      # [B, 14] probabilities
