"""
model_factory.py — Experiment Variant Factory
==============================================
Builds any MultiModalCXRModel variant from a config dict.

Supported experiment names
--------------------------
  cnn_only               — EfficientNet-B0 + IdentityQuantumLayer + MetadataEmbedder
                           + SimpleConcatFusion + SharedClassifier
  vit_only               — ViT-Small + IdentityQuantumLayer + MetadataEmbedder
                           + SimpleConcatFusion + SharedClassifier
  quantum_cnn            — EfficientNet-B0 + QuantumLayer/Identity + MetadataEmbedder
                           + DiseaseSpecificFusion + DiseaseSpecificClassifier
  mlp_clinical_only      — ZeroImageBranch + MetadataEmbedder
                           + SimpleConcatFusion + SharedClassifier
  gnn_only               — ZeroImageBranch + GNNBranch/MetadataEmbedder
                           + SimpleConcatFusion + SharedClassifier
  classical_multimodal   — EfficientNet-B0 + IdentityQuantumLayer + MetadataEmbedder
                           + SimpleConcatFusion + SharedClassifier
  quantum_gnn_multimodal — EfficientNet-B0 + QuantumLayer/Identity + MetadataEmbedder
                           + DiseaseSpecificFusion + DiseaseSpecificClassifier
  disease_specific_fusion — alias for quantum_gnn_multimodal
  shared_fusion          — EfficientNet-B0 + QuantumLayer/Identity + MetadataEmbedder
                           + SimpleConcatFusion + SharedClassifier

Dimension reference
-------------------
  EfficientNet-B0:        1280
  ViT-Small-Patch16-224:  384
  QuantumLayer output:    configurable (default 64)
  MetadataEmbedder output: 64
  DiseaseSpecificFusion fused_dim: 128
  SharedClassifier hidden_dim:     128
  num_classes:            14
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.image_branch import ImageBranch
from src.models.quantum_layer import get_quantum_layer, IdentityQuantumLayer
from src.models.gnn_branch import get_gnn_branch
from src.models.mlp_baseline import MetadataEmbedder
from src.models.fusion import DiseaseSpecificFusion, SimpleConcatFusion
from src.models.classifier import (
    DiseaseSpecificClassifier,
    SharedClassifier,
    MultiModalCXRModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZeroImageBranch & ZeroClinicalBranch — for ablation / unimodal baselines
# ---------------------------------------------------------------------------

class ZeroImageBranch(nn.Module):
    """Dummy image branch that always returns zeros.

    Used for clinical-only or GNN-only experiments where the image modality
    is intentionally excluded so we can measure the contribution of other
    branches in isolation.

    Args:
        output_dim: Dimension of the zero tensor to output. Must match what
                    the downstream fusion layer expects as `image_dim`.
    """

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.feature_dim = output_dim   # mirrors ImageBranch.feature_dim API
        logger.info("ZeroImageBranch: will output zeros [B, %d]", output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return zero tensor matching expected image feature shape.

        Args:
            x: [B, C, H, W] — accepted but ignored.

        Returns:
            [B, output_dim] zero tensor on the same device as x.
        """
        B = x.shape[0]
        return torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)


class ZeroClinicalBranch(nn.Module):
    """Dummy clinical branch that always returns zeros.

    Used for image-only experiments (cnn_only, vit_only) to strictly guarantee
    zero demographic/metadata leakage into the image-only baseline.

    Args:
        output_dim: Dimension of the zero tensor to output. Must match what
                    the downstream fusion layer expects as `clinical_dim`.
    """

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        logger.info("ZeroClinicalBranch: will output zeros [B, %d]", output_dim)

    def forward(self, meta: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Return zero tensor matching expected clinical feature shape.

        Args:
            meta: [B, 2] — accepted but ignored.

        Returns:
            [B, output_dim] zero tensor on the same device as meta.
        """
        B = meta.shape[0]
        return torch.zeros(B, self.output_dim, device=meta.device, dtype=meta.dtype)


# ---------------------------------------------------------------------------
# Default dimension constants
# ---------------------------------------------------------------------------

_BACKBONE_DIMS = {
    "efficientnet_b0": 1280,
    "efficientnet_b2": 1408,
    "vit_small_patch16_224": 384,
    "resnet50": 2048,
    "densenet121": 1024,
}

_DEFAULTS = {
    "quantum_output_dim": 64,
    "clinical_output_dim": 64,
    "fused_dim_ds": 128,          # DiseaseSpecificFusion fused_dim
    "fused_dim_simple": 128,      # SimpleConcatFusion fused_dim (= SharedClassifier input)
    "classifier_hidden_dim": 128,
    "num_classes": 14,
    "dropout": 0.3,
    "zero_image_dim": 64,
}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_model(
    experiment: str,
    cfg: Optional[Dict[str, Any]] = None,
    pretrained: bool = False,
    device: str = "cpu",
) -> MultiModalCXRModel:
    """Build a MultiModalCXRModel for the given experiment variant.

    Args:
        experiment: One of the supported experiment names (see module docstring).
        cfg:        Optional config dict to override defaults.  Any key not
                    present falls back to ``_DEFAULTS``.
        pretrained: Whether to load ImageNet pretrained weights for the backbone.
                    Set False for tests (avoids network calls).
        device:     Target device string; used only for quantum-layer selection.

    Returns:
        Fully-constructed MultiModalCXRModel ready for training.
    """
    if cfg is None:
        cfg = {}

    exp = experiment.lower().strip()

    # Resolve per-experiment backbone
    backbone = _resolve(cfg, "image_backbone", "efficientnet_b0")
    backbone_dim = _BACKBONE_DIMS.get(backbone, 512)

    q_out       = _resolve(cfg, "quantum_output_dim",    _DEFAULTS["quantum_output_dim"])
    clin_out    = _resolve(cfg, "clinical_output_dim",   _DEFAULTS["clinical_output_dim"])
    fused_ds    = _resolve(cfg, "fused_dim",             _DEFAULTS["fused_dim_ds"])
    fused_simple= _resolve(cfg, "fused_dim",             _DEFAULTS["fused_dim_simple"])
    cls_hidden  = _resolve(cfg, "classifier_hidden_dim", _DEFAULTS["classifier_hidden_dim"])
    num_classes = _resolve(cfg, "num_classes",           _DEFAULTS["num_classes"])
    dropout     = _resolve(cfg, "dropout",               _DEFAULTS["dropout"])
    zero_dim    = _resolve(cfg, "zero_image_dim",        _DEFAULTS["zero_image_dim"])

    # Quantum enabled: if explicitly configured in cfg, respect it;
    # otherwise for quantum experiment variants, enable whenever PennyLane is available
    if "quantum_enabled" in cfg:
        quantum_enabled = bool(cfg["quantum_enabled"])
    else:
        quantum_enabled = True if "quantum" in exp else False

    # ------------------------------------------------------------------
    # cnn_only
    # ------------------------------------------------------------------
    if exp == "cnn_only":
        img  = ImageBranch(backbone="efficientnet_b0", pretrained=pretrained)
        q    = IdentityQuantumLayer(input_dim=img.feature_dim, output_dim=q_out)
        clin = ZeroClinicalBranch(output_dim=clin_out)
        fuse = SimpleConcatFusion(image_dim=q_out, clinical_dim=clin_out,
                                  fused_dim=fused_simple, dropout=dropout)
        cls  = SharedClassifier(fused_dim=fused_simple, hidden_dim=cls_hidden,
                                num_classes=num_classes, dropout=dropout)
        return MultiModalCXRModel(img, clin, q, fuse, cls,
                                  use_disease_specific_fusion=False)

    # ------------------------------------------------------------------
    # vit_only
    # ------------------------------------------------------------------
    elif exp == "vit_only":
        img  = ImageBranch(backbone="vit_small_patch16_224", pretrained=pretrained)
        q    = IdentityQuantumLayer(input_dim=img.feature_dim, output_dim=q_out)
        clin = ZeroClinicalBranch(output_dim=clin_out)
        fuse = SimpleConcatFusion(image_dim=q_out, clinical_dim=clin_out,
                                  fused_dim=fused_simple, dropout=dropout)
        cls  = SharedClassifier(fused_dim=fused_simple, hidden_dim=cls_hidden,
                                num_classes=num_classes, dropout=dropout)
        return MultiModalCXRModel(img, clin, q, fuse, cls,
                                  use_disease_specific_fusion=False)

    # ------------------------------------------------------------------
    # quantum_cnn
    # ------------------------------------------------------------------
    elif exp == "quantum_cnn":
        img  = ImageBranch(backbone="efficientnet_b0", pretrained=pretrained)
        q    = get_quantum_layer(input_dim=img.feature_dim, output_dim=q_out,
                                 enabled=quantum_enabled)
        clin = MetadataEmbedder(input_dim=2, output_dim=clin_out, dropout=dropout)
        fuse = DiseaseSpecificFusion(image_dim=q_out, clinical_dim=clin_out,
                                     fused_dim=fused_ds, num_classes=num_classes,
                                     dropout=dropout)
        cls  = DiseaseSpecificClassifier(fused_dim=fused_ds, hidden_dim=cls_hidden,
                                         num_classes=num_classes, dropout=dropout)
        return MultiModalCXRModel(img, clin, q, fuse, cls,
                                  use_disease_specific_fusion=True)

    # ------------------------------------------------------------------
    # mlp_clinical_only
    # ------------------------------------------------------------------
    elif exp == "mlp_clinical_only":
        img  = ZeroImageBranch(output_dim=zero_dim)
        q    = IdentityQuantumLayer(input_dim=zero_dim, output_dim=zero_dim)
        clin = MetadataEmbedder(input_dim=2, output_dim=clin_out, dropout=dropout)
        fuse = SimpleConcatFusion(image_dim=zero_dim, clinical_dim=clin_out,
                                  fused_dim=fused_simple, dropout=dropout)
        cls  = SharedClassifier(fused_dim=fused_simple, hidden_dim=cls_hidden,
                                num_classes=num_classes, dropout=dropout)
        return MultiModalCXRModel(img, clin, q, fuse, cls,
                                  use_disease_specific_fusion=False)

    # ------------------------------------------------------------------
    # gnn_only
    # ------------------------------------------------------------------
    elif exp == "gnn_only":
        img  = ZeroImageBranch(output_dim=zero_dim)
        q    = IdentityQuantumLayer(input_dim=zero_dim, output_dim=zero_dim)
        # Always create MetadataEmbedder as fallback for when no graph is provided
        clin_fallback = MetadataEmbedder(input_dim=2, output_dim=clin_out, dropout=dropout)
        # Try GNN; fall back to MetadataEmbedder-only if torch_geometric unavailable
        try:
            from src.models.gnn_branch import TG_AVAILABLE
            if TG_AVAILABLE:
                clin = get_gnn_branch(node_feat_dim=2, hidden_dim=64,
                                      output_dim=clin_out, n_layers=2,
                                      dropout=dropout, enabled=True)
            else:
                raise ImportError
        except (ImportError, Exception):
            logger.warning("GNN unavailable for gnn_only; using MetadataEmbedder only")
            clin = clin_fallback
            clin_fallback = None  # no separate fallback needed
        fuse = SimpleConcatFusion(image_dim=zero_dim, clinical_dim=clin_out,
                                  fused_dim=fused_simple, dropout=dropout)
        cls  = SharedClassifier(fused_dim=fused_simple, hidden_dim=cls_hidden,
                                num_classes=num_classes, dropout=dropout)
        return MultiModalCXRModel(img, clin, q, fuse, cls,
                                  use_disease_specific_fusion=False,
                                  clinical_fallback=clin_fallback)

    # ------------------------------------------------------------------
    # classical_multimodal
    # ------------------------------------------------------------------
    elif exp == "classical_multimodal":
        img  = ImageBranch(backbone=backbone, pretrained=pretrained)
        q    = IdentityQuantumLayer(input_dim=img.feature_dim, output_dim=q_out)
        clin = MetadataEmbedder(input_dim=2, output_dim=clin_out, dropout=dropout)
        fuse = SimpleConcatFusion(image_dim=q_out, clinical_dim=clin_out,
                                  fused_dim=fused_simple, dropout=dropout)
        cls  = SharedClassifier(fused_dim=fused_simple, hidden_dim=cls_hidden,
                                num_classes=num_classes, dropout=dropout)
        return MultiModalCXRModel(img, clin, q, fuse, cls,
                                  use_disease_specific_fusion=False)

    # ------------------------------------------------------------------
    # quantum_gnn_multimodal  /  disease_specific_fusion
    # ------------------------------------------------------------------
    elif exp in ("quantum_gnn_multimodal", "disease_specific_fusion"):
        img  = ImageBranch(backbone="efficientnet_b0", pretrained=pretrained)
        q    = get_quantum_layer(input_dim=img.feature_dim, output_dim=q_out,
                                 enabled=quantum_enabled)
        # Clinical MLP processes patient demographic metadata
        clin = MetadataEmbedder(input_dim=2, output_dim=clin_out, dropout=dropout)

        # GNN Branch operates on scan graph where nodes combine image representation + metadata
        gnn_node_dim = q_out + 2
        try:
            from src.models.gnn_branch import GNNBranch, TG_AVAILABLE
            if TG_AVAILABLE:
                gnn = GNNBranch(node_feat_dim=gnn_node_dim, hidden_dim=64, output_dim=clin_out,
                                n_layers=2, dropout=dropout)
            else:
                gnn = None
        except (ImportError, Exception):
            gnn = None

        # Multimodal Disease-Specific Fusion: combines Quantum Image, Clinical MLP, and Scan Graph GNN
        fuse = DiseaseSpecificFusion(image_dim=q_out, clinical_dim=clin_out,
                                     gnn_dim=clin_out if gnn is not None else None,
                                     fused_dim=fused_ds, num_classes=num_classes,
                                     dropout=dropout)
        cls  = DiseaseSpecificClassifier(fused_dim=fused_ds, hidden_dim=cls_hidden,
                                         num_classes=num_classes, dropout=dropout)
        return MultiModalCXRModel(img, clin, q, fuse, cls,
                                  gnn_branch=gnn,
                                  use_disease_specific_fusion=True)

    # ------------------------------------------------------------------
    # cnn_gnn_multimodal / classical_gnn_multimodal (Ablation: CNN + GNN + Clinical MLP)
    # ------------------------------------------------------------------
    elif exp in ("cnn_gnn_multimodal", "classical_gnn_multimodal"):
        img  = ImageBranch(backbone="efficientnet_b0", pretrained=pretrained)
        q    = IdentityQuantumLayer(input_dim=img.feature_dim, output_dim=q_out)
        clin = MetadataEmbedder(input_dim=2, output_dim=clin_out, dropout=dropout)
        gnn_node_dim = q_out + 2
        try:
            from src.models.gnn_branch import GNNBranch, TG_AVAILABLE
            if TG_AVAILABLE:
                gnn = GNNBranch(node_feat_dim=gnn_node_dim, hidden_dim=64, output_dim=clin_out,
                                n_layers=2, dropout=dropout)
            else:
                gnn = None
        except (ImportError, Exception):
            gnn = None

        fuse = DiseaseSpecificFusion(image_dim=q_out, clinical_dim=clin_out,
                                     gnn_dim=clin_out if gnn is not None else None,
                                     fused_dim=fused_ds, num_classes=num_classes,
                                     dropout=dropout)
        cls  = DiseaseSpecificClassifier(fused_dim=fused_ds, hidden_dim=cls_hidden,
                                         num_classes=num_classes, dropout=dropout)
        return MultiModalCXRModel(img, clin, q, fuse, cls,
                                  gnn_branch=gnn,
                                  use_disease_specific_fusion=True)

    # ------------------------------------------------------------------
    # shared_fusion
    # ------------------------------------------------------------------
    elif exp == "shared_fusion":
        img  = ImageBranch(backbone="efficientnet_b0", pretrained=pretrained)
        q    = get_quantum_layer(input_dim=img.feature_dim, output_dim=q_out,
                                 enabled=quantum_enabled)
        clin = MetadataEmbedder(input_dim=2, output_dim=clin_out, dropout=dropout)
        fuse = SimpleConcatFusion(image_dim=q_out, clinical_dim=clin_out,
                                  fused_dim=fused_simple, dropout=dropout)
        cls  = SharedClassifier(fused_dim=fused_simple, hidden_dim=cls_hidden,
                                num_classes=num_classes, dropout=dropout)
        return MultiModalCXRModel(img, clin, q, fuse, cls,
                                  use_disease_specific_fusion=False)

    else:
        supported = [
            "cnn_only", "vit_only", "quantum_cnn", "mlp_clinical_only",
            "gnn_only", "classical_multimodal", "quantum_gnn_multimodal",
            "cnn_gnn_multimodal", "classical_gnn_multimodal",
            "disease_specific_fusion", "shared_fusion",
        ]
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Supported: {supported}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """Return cfg[key] if present, else default."""
    return cfg.get(key, default)


def list_experiments() -> list:
    """Return a list of all supported experiment names."""
    return [
        "cnn_only",
        "vit_only",
        "quantum_cnn",
        "mlp_clinical_only",
        "gnn_only",
        "classical_multimodal",
        "quantum_gnn_multimodal",
        "cnn_gnn_multimodal",
        "classical_gnn_multimodal",
        "disease_specific_fusion",
        "shared_fusion",
    ]
