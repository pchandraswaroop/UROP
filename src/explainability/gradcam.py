"""
gradcam.py — Grad-CAM++ and ViT Token Attribution for Chest X-rays
==================================================================
Implements:
  1. GradCAMPlusPlus: Second-order gradient-weighted class activation mapping
     for CNN-based and Quantum-CNN image backbones.
  2. ViTAttribution: Gradient-weighted spatial token attribution for Vision
     Transformers (vit_small_patch16_224).
  3. Overlay and heatmap generation for multi-disease diagnosis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ── Grad-CAM++ ───────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """Grad-CAM++ implementation for CNN and Quantum-CNN backbones.

    Grad-CAM++ provides better localization of multiple occurrences of an object
    and handles complex multi-label disease manifestations on chest radiographs.

    Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual
    Explanations for Deep Convolutional Networks", WACV 2018.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            model: MultiModalCXRModel or standalone image model.
            target_layer: Conv2d layer to hook. If None, automatically finds
                          the last Conv2d layer in the image branch.
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
        if target_layer is None:
            raise ValueError("Could not automatically locate a Conv2d layer in model.")

        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._hooks = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _find_last_conv_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """Find the last Conv2d layer in the model hierarchy."""
        last_conv = None
        # Look in image_branch first if present
        root = getattr(model, "image_branch", model)
        for _, module in root.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _forward_hook(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self.activations = out.detach()

    def _backward_hook(self, module: nn.Module, grad_in: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_out[0].detach()

    def generate(
        self,
        images: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        target_class: int = 0,
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap for a specific disease class.

        Args:
            images: [B, 3, H, W] input image batch.
            meta: Optional [B, 2] metadata tensor.
            graph_x: Optional GNN node features.
            graph_edge_index: Optional GNN edge indices.
            target_class: Index of the disease class (0..13) to explain.

        Returns:
            numpy array of shape [B, H, W] with values in [0.0, 1.0].
        """
        self.model.zero_grad()
        B, C, H, W = images.shape

        if meta is None:
            meta = torch.zeros(B, 2, device=images.device)

        # Graph node alignment for visual saliency:
        # If graph_x has mismatched batch dimension (e.g., companion scan nodes while images is single primary scan),
        # isolate the primary scan representation to avoid dimension mismatch in fusion.
        if graph_x is not None and graph_x.shape[0] != B:
            graph_x = None
            graph_edge_index = None

        with torch.enable_grad():
            # Forward pass (ensure gradients can flow back to activations)
            images = images.clone().detach().requires_grad_(True)
            try:
                logits = self.model(
                    images,
                    meta,
                    graph_x=graph_x,
                    graph_edge_index=graph_edge_index,
                    return_logits=True,
                )
            except TypeError:
                try:
                    logits = self.model(images, meta, return_logits=True)
                except TypeError:
                    logits = self.model(images)

            # Score for target class
            score = logits[:, target_class].sum()
            self.model.zero_grad()
            score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks failed to capture activations or gradients.")

        # Compute Grad-CAM++ weights
        # activations: [B, C_feat, H_feat, W_feat]
        # gradients:   [B, C_feat, H_feat, W_feat]
        A = self.activations
        grad = self.gradients

        grad_2 = grad.pow(2)
        grad_3 = grad.pow(3)

        # alpha_ij = grad_2 / (2 * grad_2 + sum(A * grad_3) + eps)
        eps = 1e-7
        sum_A_grad3 = (A * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha = grad_2 / (2.0 * grad_2 + sum_A_grad3 + eps)

        # weights: sum(alpha * relu(grad))
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # [B, C_feat, 1, 1]

        # Weighted combination of feature maps
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))  # [B, 1, H_feat, W_feat]

        # Upsample to original image resolution [B, 1, H, W]
        cam_upsampled = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        # Normalize each sample in batch to [0, 1]
        cam_np = cam_upsampled.squeeze(1).cpu().numpy()
        for b in range(B):
            c_min = cam_np[b].min()
            c_max = cam_np[b].max()
            if c_max > c_min:
                cam_np[b] = (cam_np[b] - c_min) / (c_max - c_min + eps)
            else:
                cam_np[b] = np.zeros_like(cam_np[b])

        return cam_np

    def remove_hooks(self) -> None:
        """Remove forward and backward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ── ViT Token Attribution ───────────────────────────────────────────────────

class ViTAttribution:
    """Gradient-weighted spatial token attribution for Vision Transformers.

    For ViT architectures where 2D convolutions are absent, attribution is computed
    by projecting token gradients onto patch representations and reshaping the
    $14 \\times 14$ patch tokens back to 2D image coordinates.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()
        self.tokens: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Hook the output of the final transformer block
        root = getattr(model, "image_branch", model)
        backbone = getattr(root, "backbone", root)
        blocks = getattr(backbone, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError("ViT blocks not found in model.")

        self.target_block = blocks[-1]
        self._hooks = [
            self.target_block.register_forward_hook(self._forward_hook),
            self.target_block.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self.tokens = out.detach()

    def _backward_hook(self, module: nn.Module, grad_in: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_out[0].detach()

    def generate(
        self,
        images: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        target_class: int = 0,
    ) -> np.ndarray:
        """Generate ViT spatial attribution heatmap for a target disease class."""
        self.model.zero_grad()
        B, C, H, W = images.shape

        if meta is None:
            meta = torch.zeros(B, 2, device=images.device)

        images = images.clone().detach().requires_grad_(True)
        try:
            logits = self.model(
                images,
                meta,
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                return_logits=True,
            )
        except TypeError:
            try:
                logits = self.model(images, meta, return_logits=True)
            except TypeError:
                logits = self.model(images)

        score = logits[:, target_class].sum()
        self.model.zero_grad()
        score.backward(retain_graph=False)

        if self.tokens is None or self.gradients is None:
            raise RuntimeError("ViT hooks failed to capture tokens or gradients.")

        # tokens & gradients shape: [B, N_tokens, Dim] (e.g., [B, 197, 384])
        # Token 0 is CLS; tokens 1..196 are the 14x14 patches
        patch_tokens = self.tokens[:, 1:, :]      # [B, 196, Dim]
        patch_grads  = self.gradients[:, 1:, :]   # [B, 196, Dim]

        # Token attribution: positive gradient-activation product
        attribution = F.relu((patch_tokens * patch_grads).sum(dim=-1))  # [B, 196]

        # Reshape to 14x14 grid
        grid_size = int(np.sqrt(attribution.shape[1]))
        heatmap_grid = attribution.view(B, 1, grid_size, grid_size)

        # Interpolate to image size [B, 1, H, W]
        cam_upsampled = F.interpolate(heatmap_grid, size=(H, W), mode="bilinear", align_corners=False)

        cam_np = cam_upsampled.squeeze(1).cpu().numpy()
        eps = 1e-7
        for b in range(B):
            c_min = cam_np[b].min()
            c_max = cam_np[b].max()
            if c_max > c_min:
                cam_np[b] = (cam_np[b] - c_min) / (c_max - c_min + eps)
            else:
                cam_np[b] = np.zeros_like(cam_np[b])

        return cam_np

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ── Unified Factory ──────────────────────────────────────────────────────────

def get_image_explainer(
    model: nn.Module,
    target_layer: Optional[nn.Module] = None,
) -> Union[GradCAMPlusPlus, ViTAttribution]:
    """Factory: Automatically detect backbone and return Grad-CAM++ or ViTAttribution."""
    root = getattr(model, "image_branch", model)
    backbone_name = getattr(root, "backbone_name", "").lower()

    if "vit" in backbone_name:
        logger.info("Instantiating ViTAttribution for transformer backbone (%s).", backbone_name)
        return ViTAttribution(model)

    logger.info("Instantiating GradCAMPlusPlus for convolutional/quantum backbone (%s).", backbone_name)
    return GradCAMPlusPlus(model, target_layer=target_layer)
