"""
visualizer.py — Export Utilities for Explanations and Faithfulness Curves
========================================================================
Generates publication-quality figures:
  1. Original CXR vs Heatmap vs Overlay side-by-side comparisons.
  2. Multi-disease comparative attribution on the same scan.
  3. Tabular SHAP per-disease bar plots.
  4. Faithfulness deletion curves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.dataset.label_encoder import DISEASE_LABEL_NAMES

logger = logging.getLogger(__name__)


class ExplainabilityVisualizer:
    """Helper class to visualize and save explanation artifacts."""

    @staticmethod
    def overlay_heatmap_on_image(
        image_np: np.ndarray,
        heatmap_np: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Create an RGB overlay of heatmap on grayscale or RGB chest radiograph.

        Args:
            image_np: [H, W] or [H, W, 3] in [0, 1] or [0, 255].
            heatmap_np: [H, W] float array in [0, 1].
            alpha: Transparency factor for heatmap overlay.
            colormap: Matplotlib colormap name.

        Returns:
            RGB image as numpy array [H, W, 3] in [0, 1].
        """
        # Normalize image to [0, 1] float32
        img = image_np.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img_rgb = np.repeat(img, 3, axis=-1)
        else:
            img_rgb = img

        # Apply colormap to heatmap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_np)[:, :, :3]  # [H, W, 3]

        # Blend
        overlay = (1.0 - alpha) * img_rgb + alpha * heatmap_colored
        return np.clip(overlay, 0.0, 1.0)

    @classmethod
    def plot_heatmap_comparison(
        cls,
        image_tensor_or_np: Union[torch.Tensor, np.ndarray],
        heatmap: np.ndarray,
        disease_name: str,
        predicted_prob: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot side-by-side: Original Chest X-ray, Grad-CAM++ Heatmap, and Overlay."""
        if isinstance(image_tensor_or_np, torch.Tensor):
            # Convert [3, H, W] tensor with ImageNet normalization back to [0, 1]
            img = image_tensor_or_np.detach().cpu().float()
            if img.dim() == 4:
                img = img[0]
            # Unnormalize standard ImageNet mean/std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        else:
            img = image_tensor_or_np

        overlay = cls.overlay_heatmap_on_image(img, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img if img.ndim == 3 else img, cmap="gray")
        axes[0].set_title("Original Chest X-ray", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        im1 = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"Grad-CAM++ Heatmap\nTarget: {disease_name}", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        title = f"Overlay ({disease_name})"
        if predicted_prob is not None:
            title += f"\nP(disease) = {predicted_prob:.3f}"
        axes[2].set_title(title, fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved heatmap comparison to %s", save_path)

        return fig

    @classmethod
    def plot_multi_disease_heatmaps(
        cls,
        image_tensor: torch.Tensor,
        heatmaps_dict: Dict[str, np.ndarray],
        probs_dict: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot multi-disease attributions side-by-side for the same radiograph."""
        n_diseases = len(heatmaps_dict)
        ncols = n_diseases + 1
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

        # 1. Original image
        img = image_tensor.detach().cpu().float()
        if img.dim() == 4:
            img = img[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_np = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        axes[0].imshow(img_np)
        axes[0].set_title("Original Radiograph", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        for idx, (d_name, hmap) in enumerate(heatmaps_dict.items()):
            ax = axes[idx + 1]
            overlay = cls.overlay_heatmap_on_image(img_np, hmap)
            ax.imshow(overlay)
            p_str = f"\nP = {probs_dict[d_name]:.3f}" if probs_dict and d_name in probs_dict else ""
            ax.set_title(f"{d_name}{p_str}", fontsize=11, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved multi-disease heatmap comparison to %s", save_path)

        return fig

    @staticmethod
    def plot_tabular_shap_summary(
        shap_data: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot per-disease horizontal bar chart of Age vs Sex Shapley attributions."""
        age_imp = shap_data["Age"]  # [14]
        sex_imp = shap_data["Sex"]  # [14]
        diseases = DISEASE_LABEL_NAMES

        fig, ax = plt.subplots(figsize=(10, 8))
        y = np.arange(len(diseases))
        height = 0.35

        ax.barh(y - height / 2, age_imp, height, label="Age Contribution", color="#3A86FF")
        ax.barh(y + height / 2, sex_imp, height, label="Sex Contribution", color="#FF006E")

        ax.set_xlabel("Mean Absolute Shapley Value (|phi|)", fontsize=11)
        ax.set_title("Clinical Metadata Feature Importance Across 14 Diseases", fontsize=13, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(diseases, fontsize=10)
        ax.invert_yaxis()  # top-down
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved tabular SHAP summary to %s", save_path)

        return fig

    @staticmethod
    def plot_faithfulness_curves(
        faithfulness_res: Dict[str, Any],
        disease_name: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot deletion curve comparing Most Important, Random, and Least Important."""
        steps = [s * 100 for s in faithfulness_res["steps"]]
        most  = faithfulness_res["most_important_drops"]
        rand  = faithfulness_res["random_drops"]
        least = faithfulness_res["least_important_drops"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, most,  marker="o", linewidth=2.2, color="#E63946", label="Most Important Masked (Faithful Drop)")
        ax.plot(steps, rand,  marker="s", linewidth=1.8, linestyle="--", color="#457B9D", label="Random Masked (Control)")
        ax.plot(steps, least, marker="^", linewidth=1.8, linestyle=":", color="#2A9D8F", label="Least Important Masked")

        ax.set_xlabel("Percentage of Pixels Masked (%)", fontsize=11)
        ax.set_ylabel("Drop in Disease Probability (Delta P)", fontsize=11)
        ax.set_title(f"Image Faithfulness Deletion Curve — {disease_name}", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved faithfulness deletion curves to %s", save_path)

        return fig
