"""
modality_attribution.py — Multimodal Fusion Attribution (Image vs Clinical vs Graph)
=====================================================================================
Quantifies the relative contribution of each modality (Image, Clinical Metadata,
and Graph) to multi-disease predictions.

Attribution Method:
  Exact multi-modal Shapley values across the 3 modality subsets (2^3 = 8 evaluations).
  Measures the marginal contribution of each modality to each disease logit.

Scientific Disclaimer:
  Modality attribution measures model representation sensitivity and feature usage.
  It does NOT imply biological or clinical etiology (attributions are diagnostic
  evidence weights, not causal mechanisms).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.dataset.label_encoder import DISEASE_LABEL_NAMES

logger = logging.getLogger(__name__)


class MultimodalAttribution:
    """Computes representation attribution across Image, Tabular Metadata, and Graph."""

    MODALITIES = ["Image", "Clinical_Meta", "Graph"]

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 14,
    ) -> None:
        self.model = model
        self.model.eval()
        self.num_classes = num_classes

    def _eval_modality_subset(
        self,
        images: torch.Tensor,
        meta: torch.Tensor,
        graph_x: Optional[torch.Tensor],
        graph_edge_index: Optional[torch.Tensor],
        active_modalities: Tuple[str, ...],
    ) -> torch.Tensor:
        """Evaluate model logits with only a specific subset of modalities active."""
        B = images.shape[0]
        device = images.device

        # 1. Image modality: if disabled, use zeros
        if "Image" in active_modalities:
            img_in = images
        else:
            img_in = torch.zeros_like(images)

        # 2. Metadata modality: if disabled, use zeros (standardized baseline)
        if "Clinical_Meta" in active_modalities:
            meta_in = meta
        else:
            meta_in = torch.zeros_like(meta)

        # 3. Graph modality: if disabled, pass None for graph tensors (forces fallback or non-graph)
        if "Graph" in active_modalities and graph_x is not None and graph_edge_index is not None:
            gx_in = graph_x
            ge_in = graph_edge_index
        else:
            gx_in = None
            ge_in = None

        with torch.no_grad():
            try:
                logits = self.model(
                    img_in,
                    meta_in,
                    graph_x=gx_in,
                    graph_edge_index=ge_in,
                    return_logits=True,
                )
            except TypeError:
                try:
                    logits = self.model(img_in, meta_in, return_logits=True)
                except TypeError:
                    logits = self.model(img_in)

        return logits  # [B, 14]

    def attribute_instance(
        self,
        images: torch.Tensor,
        meta: torch.Tensor,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute exact 3-player Shapley attribution across modalities.

        Returns:
            Dict mapping modality ("Image", "Clinical_Meta", "Graph") to [B, 14]
            Shapley attribution arrays.
        """
        # All 2^3 = 8 modality subsets
        v0   = self._eval_modality_subset(images, meta, graph_x, graph_edge_index, ())
        v_I  = self._eval_modality_subset(images, meta, graph_x, graph_edge_index, ("Image",))
        v_M  = self._eval_modality_subset(images, meta, graph_x, graph_edge_index, ("Clinical_Meta",))
        v_G  = self._eval_modality_subset(images, meta, graph_x, graph_edge_index, ("Graph",))
        v_IM = self._eval_modality_subset(images, meta, graph_x, graph_edge_index, ("Image", "Clinical_Meta"))
        v_IG = self._eval_modality_subset(images, meta, graph_x, graph_edge_index, ("Image", "Graph"))
        v_MG = self._eval_modality_subset(images, meta, graph_x, graph_edge_index, ("Clinical_Meta", "Graph"))
        v_IMG= self._eval_modality_subset(images, meta, graph_x, graph_edge_index, ("Image", "Clinical_Meta", "Graph"))

        # Exact Shapley weights for 3 players:
        # Coalition sizes: |S|=0: 1/3; |S|=1: 1/6; |S|=2: 1/3
        # phi_i = 1/3 * [v({i}) - v0]
        #       + 1/6 * sum_{j} [v({i,j}) - v({j})]
        #       + 1/3 * [v({i,j,k}) - v({j,k})]
        phi_I = (1.0 / 3.0) * (v_I - v0) \
              + (1.0 / 6.0) * ((v_IM - v_M) + (v_IG - v_G)) \
              + (1.0 / 3.0) * (v_IMG - v_MG)

        phi_M = (1.0 / 3.0) * (v_M - v0) \
              + (1.0 / 6.0) * ((v_IM - v_I) + (v_MG - v_G)) \
              + (1.0 / 3.0) * (v_IMG - v_IG)

        phi_G = (1.0 / 3.0) * (v_G - v0) \
              + (1.0 / 6.0) * ((v_IG - v_I) + (v_MG - v_M)) \
              + (1.0 / 3.0) * (v_IMG - v_IM)

        return {
            "Image": phi_I.cpu().numpy(),
            "Clinical_Meta": phi_M.cpu().numpy(),
            "Graph": phi_G.cpu().numpy(),
            "Base_Logits": v0.cpu().numpy(),
            "Full_Logits": v_IMG.cpu().numpy(),
        }

    def attribute_dataset(
        self,
        loader: Any,
        max_samples: int = 100,
        device: str = "cpu",
    ) -> Dict[str, np.ndarray]:
        """Aggregate modality attribution across a dataset.

        Returns:
            Dict containing mean absolute contribution and relative percentages per disease:
              - mean_attribution: [3, 14]
              - relative_percentages: [3, 14]
              - modalities: List of modality names
        """
        all_I, all_M, all_G = [], [], []
        count = 0

        for batch in loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)

            graph_x, graph_edge_index = None, None
            if "patient_id" in batch:
                from src.graph.graph_builder import build_batch_scan_graph
                graph_x, graph_edge_index = build_batch_scan_graph(meta, batch["patient_id"], add_self_loops=True)

            res = self.attribute_instance(images, meta, graph_x, graph_edge_index)
            all_I.append(res["Image"])
            all_M.append(res["Clinical_Meta"])
            all_G.append(res["Graph"])

            count += images.shape[0]
            if count >= max_samples:
                break

        arr_I = np.concatenate(all_I, axis=0)[:max_samples]  # [N, 14]
        arr_M = np.concatenate(all_M, axis=0)[:max_samples]  # [N, 14]
        arr_G = np.concatenate(all_G, axis=0)[:max_samples]  # [N, 14]

        # Mean absolute contribution
        mean_I = np.abs(arr_I).mean(axis=0)  # [14]
        mean_M = np.abs(arr_M).mean(axis=0)  # [14]
        mean_G = np.abs(arr_G).mean(axis=0)  # [14]

        total = mean_I + mean_M + mean_G + 1e-7
        pct_I = 100.0 * (mean_I / total)
        pct_M = 100.0 * (mean_M / total)
        pct_G = 100.0 * (mean_G / total)

        return {
            "mean_attribution": np.stack([mean_I, mean_M, mean_G]),      # [3, 14]
            "relative_percentages": np.stack([pct_I, pct_M, pct_G]),  # [3, 14]
            "modalities": self.MODALITIES,
            "disease_names": DISEASE_LABEL_NAMES,
        }

    def plot_modality_contributions(
        self,
        attribution_data: Dict[str, Any],
        title: str = "Multimodal Contribution by Disease",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot stacked bar chart of relative modality contributions across all 14 diseases."""
        pcts = attribution_data["relative_percentages"]  # [3, 14]
        diseases = attribution_data.get("disease_names", DISEASE_LABEL_NAMES)
        n_d = len(diseases)

        fig, ax = plt.subplots(figsize=(14, 6))
        ind = np.arange(n_d)
        width = 0.65

        p1 = ax.bar(ind, pcts[0], width, label="Image", color="#2E86AB")
        p2 = ax.bar(ind, pcts[1], width, bottom=pcts[0], label="Clinical Meta", color="#A23B72")
        p3 = ax.bar(ind, pcts[2], width, bottom=pcts[0] + pcts[1], label="Graph", color="#F18F01")

        ax.set_ylabel("Contribution Percentage (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(ind)
        ax.set_xticklabels(diseases, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved modality contribution chart to %s", save_path)

        return fig
