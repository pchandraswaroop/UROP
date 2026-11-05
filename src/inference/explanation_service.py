"""
explanation_service.py — Unified Explainability Service for Frontend
=====================================================================
Orchestrates Grad-CAM++, ViT token attribution, exact analytical tabular SHAP,
GNNExplainer, and multimodal Shapley fusion attribution for inference results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dataset.label_encoder import DISEASE_LABEL_NAMES
from src.explainability.faithfulness import FaithfulnessEvaluator
from src.explainability.gnn_explainer import ScanGraphExplainer
from src.explainability.gradcam import get_image_explainer
from src.explainability.modality_attribution import MultimodalAttribution
from src.explainability.tabular_shap import TabularFeatureExplainer
from src.explainability.visualizer import ExplainabilityVisualizer
from src.inference.predictor import PredictionResult

logger = logging.getLogger(__name__)


class ExplanationService:
    """Provides high-level, crash-resilient explainability methods for UI display."""

    @staticmethod
    def get_image_explanation(
        result: PredictionResult,
        target_disease: str,
    ) -> Dict[str, Any]:
        """Generate Grad-CAM++ or ViT attribution heatmap and overlay."""
        if result.model_instance is None or result.image_tensor is None:
            return {"status": "error", "message": "Model instance or image tensor not available."}

        model = result.model_instance
        # Check if model has a functional image branch
        if hasattr(model, "image_branch") and model.image_branch.__class__.__name__ == "ZeroImageBranch":
            return {
                "status": "unavailable",
                "message": (
                    "Image explainability is not applicable for this configuration "
                    f"({result.model_name}) because the image modality is zeroed out by design."
                ),
            }

        try:
            disease_idx = DISEASE_LABEL_NAMES.index(target_disease)
        except ValueError:
            return {"status": "error", "message": f"Unknown disease target '{target_disease}'."}

        try:
            explainer = get_image_explainer(model)
            hmap = explainer.generate(
                result.image_tensor,
                result.meta_tensor,
                target_class=disease_idx,
                graph_x=result.graph_x,
                graph_edge_index=result.graph_edge_index,
            )[0]  # [H, W]

            if result.raw_rgb is not None:
                overlay = ExplainabilityVisualizer.overlay_heatmap_on_image(
                    result.raw_rgb, hmap, alpha=0.45, colormap="viridis"
                )
            else:
                overlay = None

            return {
                "status": "success",
                "heatmap": hmap,
                "overlay": overlay,
                "disease": target_disease,
                "disease_idx": disease_idx,
            }
        except Exception as e:
            logger.exception("Image explanation failed: %s", e)
            return {"status": "error", "message": f"Grad-CAM++ computation error: {str(e)}"}

    @staticmethod
    def get_tabular_explanation(
        result: PredictionResult,
        target_disease: str,
    ) -> Dict[str, Any]:
        """Compute exact analytical Shapley values for Age and Sex."""
        if result.model_instance is None or result.image_tensor is None or result.meta_tensor is None:
            return {"status": "error", "message": "Inference context missing."}

        try:
            disease_idx = DISEASE_LABEL_NAMES.index(target_disease)
        except ValueError:
            return {"status": "error", "message": f"Unknown disease target '{target_disease}'."}

        try:
            tab_explainer = TabularFeatureExplainer(result.model_instance, num_classes=14)
            shap_dict = tab_explainer.explain_instance(
                result.image_tensor,
                result.meta_tensor,
                graph_x=result.graph_x,
                graph_edge_index=result.graph_edge_index,
            )

            age_shap = float(shap_dict["Age"][0, disease_idx])
            sex_shap = float(shap_dict["Sex"][0, disease_idx])

            return {
                "status": "success",
                "disease": target_disease,
                "age_shap": age_shap,
                "sex_shap": sex_shap,
                "raw_shap": shap_dict,
            }
        except Exception as e:
            logger.exception("Tabular SHAP failed: %s", e)
            return {"status": "error", "message": f"SHAP computation error: {str(e)}"}

    @staticmethod
    def get_graph_explanation(
        result: PredictionResult,
        target_disease: str,
    ) -> Dict[str, Any]:
        """Explain scan graph relationships or document single-scan limitations."""
        if result.model_instance is None or result.graph_x is None or result.graph_edge_index is None:
            return {"status": "error", "message": "Graph data missing from inference result."}

        # Check if the model actually uses a GNN branch
        has_gnn = hasattr(result.model_instance, "clinical_branch") and "GNN" in result.model_instance.clinical_branch.__class__.__name__

        if not has_gnn:
            return {
                "status": "unavailable",
                "message": (
                    f"Graph explainability is not applicable because {result.model_name} "
                    "does not utilize a Graph Neural Network clinical branch."
                ),
            }

        if result.node_count <= 1:
            return {
                "status": "single_scan_note",
                "message": (
                    "Single-Scan Graph Note: The current inference is performed on a single radiograph. "
                    "The scan-to-scan graph connects multiple radiographs belonging to the same patient "
                    "to capture longitudinal history. With only 1 scan provided, the graph consists of an isolated "
                    "node with a self-loop. Edge ablation analysis requires longitudinal multi-scan history."
                ),
            }

        try:
            disease_idx = DISEASE_LABEL_NAMES.index(target_disease)
            explainer = ScanGraphExplainer(result.model_instance, num_classes=14)
            exp = explainer.explain(
                result.image_tensor,
                result.meta_tensor,
                result.graph_x,
                result.graph_edge_index,
                target_class=disease_idx,
            )
            fig = explainer.visualize_graph(exp, title=f"Patient Scan Graph Attribution: {target_disease}")

            return {
                "status": "success",
                "disease": target_disease,
                "figure": fig,
                "edge_importance": exp.get("edge_importance", []),
            }
        except Exception as e:
            logger.exception("Graph explanation failed: %s", e)
            return {"status": "error", "message": f"Graph explainer error: {str(e)}"}

    @staticmethod
    def get_modality_attribution(
        result: PredictionResult,
    ) -> Dict[str, Any]:
        """Compute exact 3-player Shapley attribution across Image, Clinical Meta, and Graph."""
        if result.model_instance is None or result.image_tensor is None or result.meta_tensor is None:
            return {"status": "error", "message": "Inference context missing."}

        try:
            mm = MultimodalAttribution(result.model_instance, num_classes=14)
            res = mm.attribute_instance(
                result.image_tensor,
                result.meta_tensor,
                result.graph_x,
                result.graph_edge_index,
            )

            # Average absolute contributions across samples (here batch_size=1)
            mean_I = np.abs(res["Image"]).mean(axis=0)
            mean_M = np.abs(res["Clinical_Meta"]).mean(axis=0)
            mean_G = np.abs(res["Graph"]).mean(axis=0)
            total = mean_I + mean_M + mean_G + 1e-8

            pct_I = (mean_I / total) * 100.0
            pct_M = (mean_M / total) * 100.0
            pct_G = (mean_G / total) * 100.0

            df = pd.DataFrame({
                "Disease": DISEASE_LABEL_NAMES,
                "Image (%)": np.round(pct_I, 2),
                "Clinical_Meta (%)": np.round(pct_M, 2),
                "Graph (%)": np.round(pct_G, 2),
            })

            return {
                "status": "success",
                "summary_table": df,
                "pct_image": pct_I,
                "pct_meta": pct_M,
                "pct_graph": pct_G,
                "disclaimer": "Modality attribution reflects mathematical representation sensitivity, not clinical causality.",
            }
        except Exception as e:
            logger.exception("Modality attribution failed: %s", e)
            return {"status": "error", "message": f"Modality attribution error: {str(e)}"}
