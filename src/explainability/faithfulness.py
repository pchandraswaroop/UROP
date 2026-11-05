"""
faithfulness.py — Quantitative Faithfulness Evaluation Framework
================================================================
Evaluates whether generated explanations faithfully reflect the internal
decision-making mechanisms of the models rather than acting as visual artifacts.

Implements:
  A. Image Faithfulness:
     Deletion and insertion curves comparing top-attributed regions against
     random regions and least-attributed regions.
  B. Tabular Feature Faithfulness:
     Targeted feature shifts vs control/placebo perturbations.
  C. Graph Faithfulness:
     Removal of top-attributed edges vs random edge removal.
  D. Modality Faithfulness:
     Systematic single-modality and multi-modality ablation checks.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class FaithfulnessEvaluator:
    """Quantitative evaluation suite for explanation faithfulness."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    # ── A. Image Faithfulness ─────────────────────────────────────────────────

    def evaluate_image_faithfulness(
        self,
        image: torch.Tensor,
        heatmap: np.ndarray,
        target_class: int,
        meta: Optional[torch.Tensor] = None,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        steps: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    ) -> Dict[str, Any]:
        """Evaluate deletion curves for image explanations.

        Compares:
          1. Most Important Pixel Masking (Deletion): dropping top-k% heatmap regions.
          2. Random Pixel Masking: dropping random regions.
          3. Least Important Pixel Masking: dropping lowest heatmap regions.

        Faithful explainer property:
          Deletion of most important regions causes a sharper drop in probability
          than random or least important regions.
        """
        device = image.device
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, 3, H, W]
        B, C, H, W = image.shape
        if meta is None:
            meta = torch.zeros(B, 2, device=device)

        # Base probability
        with torch.no_grad():
            base_logits = self.model(
                image, meta, graph_x=graph_x, graph_edge_index=graph_edge_index, return_logits=True
            )
            base_prob = torch.sigmoid(base_logits)[0, target_class].item()

        flat_heatmap = heatmap.flatten()
        sorted_indices = np.argsort(flat_heatmap)  # ascending: least important first
        total_pixels = len(flat_heatmap)

        most_important_drops = []
        least_important_drops = []
        random_drops = []

        baseline_value = image.mean()  # uninformative constant replacement

        for frac in steps:
            k = int(frac * total_pixels)

            # 1. Most important: top k pixels masked
            mask_most = np.ones(total_pixels, dtype=bool)
            mask_most[sorted_indices[-k:]] = False
            img_most = image.clone().view(B, C, -1)
            img_most[:, :, ~mask_most] = baseline_value
            img_most = img_most.view(B, C, H, W)

            # 2. Least important: bottom k pixels masked
            mask_least = np.ones(total_pixels, dtype=bool)
            mask_least[sorted_indices[:k]] = False
            img_least = image.clone().view(B, C, -1)
            img_least[:, :, ~mask_least] = baseline_value
            img_least = img_least.view(B, C, H, W)

            # 3. Random: random k pixels masked
            rng = np.random.default_rng(seed=int(frac * 100))
            rand_idx = rng.choice(total_pixels, size=k, replace=False)
            mask_rand = np.ones(total_pixels, dtype=bool)
            mask_rand[rand_idx] = False
            img_rand = image.clone().view(B, C, -1)
            img_rand[:, :, ~mask_rand] = baseline_value
            img_rand = img_rand.view(B, C, H, W)

            with torch.no_grad():
                prob_most = torch.sigmoid(
                    self.model(img_most, meta, graph_x=graph_x, graph_edge_index=graph_edge_index, return_logits=True)
                )[0, target_class].item()
                prob_least = torch.sigmoid(
                    self.model(img_least, meta, graph_x=graph_x, graph_edge_index=graph_edge_index, return_logits=True)
                )[0, target_class].item()
                prob_rand = torch.sigmoid(
                    self.model(img_rand, meta, graph_x=graph_x, graph_edge_index=graph_edge_index, return_logits=True)
                )[0, target_class].item()

            most_important_drops.append(base_prob - prob_most)
            least_important_drops.append(base_prob - prob_least)
            random_drops.append(base_prob - prob_rand)

        # Faithfulness score: Average drop difference between most and random
        mean_most = float(np.mean(most_important_drops))
        mean_rand = float(np.mean(random_drops))
        mean_least = float(np.mean(least_important_drops))
        faithfulness_ratio = mean_most / (mean_rand + 1e-7)

        return {
            "steps": steps,
            "base_prob": base_prob,
            "most_important_drops": most_important_drops,
            "least_important_drops": least_important_drops,
            "random_drops": random_drops,
            "mean_most_drop": mean_most,
            "mean_random_drop": mean_rand,
            "mean_least_drop": mean_least,
            "faithfulness_ratio": faithfulness_ratio,
            "is_faithful": mean_most >= mean_rand,
        }

    # ── B. Tabular Feature Faithfulness ───────────────────────────────────────

    def evaluate_tabular_faithfulness(
        self,
        image: Optional[torch.Tensor],
        meta: torch.Tensor,
        target_class: int,
        shap_values: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Perturb the top-ranked metadata feature and evaluate prediction drop."""
        B = meta.shape[0]
        device = meta.device

        with torch.no_grad():
            base_logits = self.model(image, meta, return_logits=True)
            base_prob = torch.sigmoid(base_logits)[:, target_class].mean().item()

        # Identify top feature by mean absolute Shapley value
        age_imp = abs(shap_values["Age"][:, target_class]).mean()
        sex_imp = abs(shap_values["Sex"][:, target_class]).mean()

        top_feature = "Age" if age_imp >= sex_imp else "Sex"

        # Perturb Age (shift by 2 standard deviations)
        meta_age_perturbed = meta.clone()
        meta_age_perturbed[:, 0] += 2.0  # +2 std dev

        # Perturb Sex (flip binary flag)
        meta_sex_perturbed = meta.clone()
        meta_sex_perturbed[:, 1] = 1.0 - meta_sex_perturbed[:, 1]

        with torch.no_grad():
            prob_age_p = torch.sigmoid(self.model(image, meta_age_perturbed, return_logits=True))[:, target_class].mean().item()
            prob_sex_p = torch.sigmoid(self.model(image, meta_sex_perturbed, return_logits=True))[:, target_class].mean().item()

        delta_age = abs(base_prob - prob_age_p)
        delta_sex = abs(base_prob - prob_sex_p)

        top_delta = delta_age if top_feature == "Age" else delta_sex
        other_delta = delta_sex if top_feature == "Age" else delta_age

        return {
            "top_feature": top_feature,
            "base_prob": base_prob,
            "delta_age": delta_age,
            "delta_sex": delta_sex,
            "top_feature_delta": top_delta,
            "other_feature_delta": other_delta,
            "is_faithful": top_delta >= other_delta,
        }

    # ── C. Graph Faithfulness ─────────────────────────────────────────────────

    def evaluate_graph_faithfulness(
        self,
        images: Optional[torch.Tensor],
        meta: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        edge_importance: np.ndarray,
        target_class: int,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Compare removal of top-attributed edges vs random edge removal."""
        E = graph_edge_index.shape[1]
        device = graph_edge_index.device

        with torch.no_grad():
            base_logits = self.model(images, meta, graph_x=graph_x, graph_edge_index=graph_edge_index, return_logits=True)
            base_prob = torch.sigmoid(base_logits)[:, target_class].mean().item()

        if E <= 1:
            return {
                "base_prob": base_prob,
                "important_edges_drop": 0.0,
                "random_edges_drop": 0.0,
                "is_faithful": True,
            }

        k = min(top_k, E // 2 if E > 2 else 1)

        # 1. Drop top-k important edges
        top_edges = np.argsort(edge_importance)[-k:]
        mask_top = torch.ones(E, dtype=torch.bool, device=device)
        mask_top[top_edges] = False
        edge_index_top_dropped = graph_edge_index[:, mask_top]

        # 2. Drop random-k edges
        rng = np.random.default_rng(seed=42)
        rand_edges = rng.choice(E, size=k, replace=False)
        mask_rand = torch.ones(E, dtype=torch.bool, device=device)
        mask_rand[rand_edges] = False
        edge_index_rand_dropped = graph_edge_index[:, mask_rand]

        with torch.no_grad():
            prob_top_dropped = torch.sigmoid(
                self.model(images, meta, graph_x=graph_x, graph_edge_index=edge_index_top_dropped, return_logits=True)
            )[:, target_class].mean().item()

            prob_rand_dropped = torch.sigmoid(
                self.model(images, meta, graph_x=graph_x, graph_edge_index=edge_index_rand_dropped, return_logits=True)
            )[:, target_class].mean().item()

        drop_important = abs(base_prob - prob_top_dropped)
        drop_random = abs(base_prob - prob_rand_dropped)

        return {
            "base_prob": base_prob,
            "edges_removed": k,
            "important_edges_drop": drop_important,
            "random_edges_drop": drop_random,
            "is_faithful": drop_important >= drop_random,
        }

    # ── D. Modality Faithfulness ──────────────────────────────────────────────

    def evaluate_modality_faithfulness(
        self,
        images: torch.Tensor,
        meta: torch.Tensor,
        graph_x: Optional[torch.Tensor],
        graph_edge_index: Optional[torch.Tensor],
        modality_percentages: np.ndarray,  # [3, 14]
        target_class: int,
    ) -> Dict[str, Any]:
        """Ablate dominant modality vs subordinate modalities."""
        with torch.no_grad():
            base_prob = torch.sigmoid(
                self.model(images, meta, graph_x=graph_x, graph_edge_index=graph_edge_index, return_logits=True)
            )[:, target_class].mean().item()

        # Dominant modality for target disease
        class_pcts = modality_percentages[:, target_class]
        dominant_idx = int(np.argmax(class_pcts))  # 0: Image, 1: Meta, 2: Graph
        modalities = ["Image", "Clinical_Meta", "Graph"]
        dominant_name = modalities[dominant_idx]

        # 1. Ablate Image
        with torch.no_grad():
            prob_no_img = torch.sigmoid(
                self.model(torch.zeros_like(images), meta, graph_x=graph_x, graph_edge_index=graph_edge_index, return_logits=True)
            )[:, target_class].mean().item()

        # 2. Ablate Meta
        with torch.no_grad():
            prob_no_meta = torch.sigmoid(
                self.model(images, torch.zeros_like(meta), graph_x=graph_x, graph_edge_index=graph_edge_index, return_logits=True)
            )[:, target_class].mean().item()

        # 3. Ablate Graph
        with torch.no_grad():
            prob_no_graph = torch.sigmoid(
                self.model(images, meta, graph_x=None, graph_edge_index=None, return_logits=True)
            )[:, target_class].mean().item()

        drops = {
            "Image": abs(base_prob - prob_no_img),
            "Clinical_Meta": abs(base_prob - prob_no_meta),
            "Graph": abs(base_prob - prob_no_graph),
        }

        # Check if the modality attributed with the highest percentage caused the largest drop
        max_drop_modality = max(drops.items(), key=lambda x: x[1])[0]

        return {
            "base_prob": base_prob,
            "dominant_attributed_modality": dominant_name,
            "modality_drops": drops,
            "largest_drop_modality": max_drop_modality,
            "is_faithful": dominant_name == max_drop_modality or drops[dominant_name] >= 0.8 * drops[max_drop_modality],
        }
