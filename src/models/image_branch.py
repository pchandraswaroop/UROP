"""
image_branch.py — CNN/ViT Image Feature Extractor
==================================================
Provides a configurable image backbone:
  - efficientnet_b0 (default, CPU-friendly)
  - vit_small_patch16_224

Output: feature vector of shape [B, feature_dim]
        where feature_dim depends on the backbone.

The backbone is loaded from timm with pretrained weights.
The classification head is removed; only features are returned.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not installed. Run: pip install timm")


# ── Image Branch ──────────────────────────────────────────────────────────────

class ImageBranch(nn.Module):
    """Feature extractor from a timm backbone.

    Args:
        backbone:     timm model name (e.g. "efficientnet_b0").
        pretrained:   Load ImageNet pretrained weights.
        freeze_until: Fraction of layers to freeze [0.0, 1.0].
                      0.0 = fine-tune all, 1.0 = freeze all.
        output_dim:   Optional projection to a fixed output dimension.
                      If None, no projection is applied.
    """

    SUPPORTED_BACKBONES = {
        "efficientnet_b0": 1280,
        "efficientnet_b2": 1408,
        "vit_small_patch16_224": 384,
        "resnet50": 2048,
        "densenet121": 1024,
    }

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_until: float = 0.0,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ImageBranch. Install: pip install timm")

        self.backbone_name = backbone
        self.feature_dim = self.SUPPORTED_BACKBONES.get(backbone, 512)

        # Load backbone without classification head
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # removes classifier head
            global_pool="avg",      # global average pooling
        )

        # Selective layer freezing
        if freeze_until > 0.0:
            self._freeze_layers(freeze_until)

        # Optional projection
        self.proj = None
        if output_dim is not None and output_dim != self.feature_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.feature_dim, output_dim),
                nn.ReLU(inplace=True),
            )
            self.feature_dim = output_dim

        logger.info(
            "ImageBranch: backbone=%s, feature_dim=%d, pretrained=%s",
            backbone, self.feature_dim, pretrained
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Feature tensor [B, feature_dim]
        """
        features = self.backbone(x)   # [B, feature_dim]
        if self.proj is not None:
            features = self.proj(features)
        return features

    def _freeze_layers(self, fraction: float) -> None:
        """Freeze the first `fraction` of backbone parameters."""
        params = list(self.backbone.parameters())
        n_freeze = int(len(params) * fraction)
        for i, p in enumerate(params):
            p.requires_grad = i >= n_freeze
        logger.info(
            "Froze %d / %d backbone parameters (fraction=%.2f)",
            n_freeze, len(params), fraction
        )

    def get_feature_dim(self) -> int:
        return self.feature_dim
