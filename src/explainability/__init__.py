"""
src/explainability — Explainability, Interpretability & Faithfulness Framework
=============================================================================
Modules:
  - gradcam: Grad-CAM++ for CNN / Quantum-CNN and ViT Token Attribution.
  - tabular_shap: Exact & sampling Shapley value feature attribution for metadata.
  - gnn_explainer: Scan-to-scan graph edge & node attribution with NetworkX visualization.
  - modality_attribution: Multimodal representation attribution (Image vs Clinical vs Graph).
  - faithfulness: Quantitative perturbation-based faithfulness evaluations.
  - visualizer: Export utilities for heatmaps, overlays, bar plots, and graph diagrams.
"""

from src.explainability.gradcam import GradCAMPlusPlus, ViTAttribution, get_image_explainer
from src.explainability.tabular_shap import TabularFeatureExplainer
from src.explainability.gnn_explainer import ScanGraphExplainer
from src.explainability.modality_attribution import MultimodalAttribution
from src.explainability.faithfulness import FaithfulnessEvaluator
from src.explainability.visualizer import ExplainabilityVisualizer

__all__ = [
    "GradCAMPlusPlus",
    "ViTAttribution",
    "get_image_explainer",
    "TabularFeatureExplainer",
    "ScanGraphExplainer",
    "MultimodalAttribution",
    "FaithfulnessEvaluator",
    "ExplainabilityVisualizer",
]
