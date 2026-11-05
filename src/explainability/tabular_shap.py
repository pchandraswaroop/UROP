"""
tabular_shap.py — Shapley Value Attribution for Clinical Metadata (Age & Sex)
=============================================================================
Computes exact and sampling-based Shapley value feature attributions for
clinical metadata (age and sex) across all 14 disease classification heads.

Zero label leakage:
  Only clinical inputs (age and sex) are attributed. Target disease labels
  are NEVER provided as input features.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.info("External `shap` library not installed. Using exact analytical Shapley explainer.")


class TabularFeatureExplainer:
    """Exact and sampling-based Shapley value feature explainer for tabular clinical inputs.

    Because the clinical feature space has 2 non-label dimensions (normalized age and binary sex),
    the power set of feature coalitions contains exactly 2^2 = 4 subsets. This allows computing
    the exact, unapproximated Shapley values analytically without Monte Carlo variance.
    """

    FEATURE_NAMES = ["Age", "Sex"]

    def __init__(
        self,
        model: nn.Module,
        baseline_meta: Optional[torch.Tensor] = None,
        num_classes: int = 14,
    ) -> None:
        """
        Args:
            model: Full MultiModalCXRModel or clinical baseline model.
            baseline_meta: Reference baseline [1, 2] tensor (default: [0.0, 0.0] representing
                           standardized median age and reference sex).
            num_classes: Number of disease classes (14).
        """
        self.model = model
        self.model.eval()
        self.num_classes = num_classes

        if baseline_meta is None:
            # Baseline: normalized age = 0.0 (mean age ~63), sex = 0.0 (reference)
            self.baseline = torch.zeros(1, 2)
        else:
            self.baseline = baseline_meta.detach().clone().view(1, 2)

    def _eval_coalition(
        self,
        images: Optional[torch.Tensor],
        meta: torch.Tensor,
        coalition: Tuple[int, ...],
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate model logits for a given subset of active features."""
        B = meta.shape[0]
        device = meta.device
        baseline_exp = self.baseline.to(device).expand(B, 2).clone()

        # Build feature vector where non-coalition features take baseline values
        meta_coalition = baseline_exp.clone()
        for feat_idx in coalition:
            meta_coalition[:, feat_idx] = meta[:, feat_idx]

        # If images are not provided, synthesize zero image features or dummy images
        if images is None:
            images = torch.zeros(B, 3, 224, 224, device=device)

        # Graph node features for coalition
        gx = graph_x
        if gx is not None and gx.shape[0] == B:
            gx_coalition = meta_coalition.clone()
        else:
            gx_coalition = gx

        with torch.no_grad():
            try:
                logits = self.model(
                    images,
                    meta_coalition,
                    graph_x=gx_coalition,
                    graph_edge_index=graph_edge_index,
                    return_logits=True,
                )
            except TypeError:
                try:
                    logits = self.model(images, meta_coalition, return_logits=True)
                except TypeError:
                    logits = self.model(meta_coalition)

        return logits  # [B, 14]

    def explain_instance(
        self,
        images: Optional[torch.Tensor],
        meta: torch.Tensor,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute exact Shapley values for a single patient / scan or mini-batch.

        Returns:
            Dictionary mapping feature names ("Age", "Sex") to an array of shape [B, 14],
            giving the contribution of that feature to each disease logit.
        """
        B = meta.shape[0]

        # Evaluate all 4 coalitions:
        # Coalition 0: ()
        v_empty = self._eval_coalition(images, meta, (), graph_x, graph_edge_index)
        # Coalition 1: (0,) -> Age only
        v_age = self._eval_coalition(images, meta, (0,), graph_x, graph_edge_index)
        # Coalition 2: (1,) -> Sex only
        v_sex = self._eval_coalition(images, meta, (1,), graph_x, graph_edge_index)
        # Coalition 3: (0, 1) -> Both Age & Sex
        v_both = self._eval_coalition(images, meta, (0, 1), graph_x, graph_edge_index)

        # Exact Shapley formula for 2 players:
        # phi_age = 0.5 * (v_age - v_empty) + 0.5 * (v_both - v_sex)
        # phi_sex = 0.5 * (v_sex - v_empty) + 0.5 * (v_both - v_age)
        phi_age = 0.5 * (v_age - v_empty) + 0.5 * (v_both - v_sex)
        phi_sex = 0.5 * (v_sex - v_empty) + 0.5 * (v_both - v_age)

        return {
            "Age": phi_age.cpu().numpy(),
            "Sex": phi_sex.cpu().numpy(),
            "BaseValue": v_empty.cpu().numpy(),
        }

    def explain_dataset(
        self,
        loader: Any,
        max_samples: int = 100,
        device: str = "cpu",
    ) -> Dict[str, np.ndarray]:
        """Aggregate mean absolute Shapley values across multiple batches.

        Returns:
            Dict mapping feature name to mean absolute importance per disease: [14].
        """
        age_shaps = []
        sex_shaps = []
        count = 0

        for batch in loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)

            graph_x, graph_edge_index = None, None
            if "patient_id" in batch:
                from src.graph.graph_builder import build_batch_scan_graph
                graph_x, graph_edge_index = build_batch_scan_graph(meta, batch["patient_id"], add_self_loops=True)

            res = self.explain_instance(images, meta, graph_x=graph_x, graph_edge_index=graph_edge_index)
            age_shaps.append(res["Age"])
            sex_shaps.append(res["Sex"])

            count += meta.shape[0]
            if count >= max_samples:
                break

        all_age = np.concatenate(age_shaps, axis=0)[:max_samples]  # [N, 14]
        all_sex = np.concatenate(sex_shaps, axis=0)[:max_samples]  # [N, 14]

        # Mean absolute impact per disease
        return {
            "Age": np.abs(all_age).mean(axis=0),
            "Sex": np.abs(all_sex).mean(axis=0),
            "Raw_Age": all_age,
            "Raw_Sex": all_sex,
        }
