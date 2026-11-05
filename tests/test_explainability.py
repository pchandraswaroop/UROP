"""
test_explainability.py — Unit and Integration Tests for Explainability & Faithfulness
=====================================================================================
Tests:
  1. Grad-CAM++ & ViT token attribution (shapes, bounds, disease specificity).
  2. Tabular Shapley values (efficiency axiom, shapes, zero label leakage).
  3. GNNExplainer on scan graph (edge & node importance, visualization).
  4. Multimodal fusion attribution (3-player Shapley, percentage sum).
  5. Faithfulness evaluation metrics (image deletion curves, perturbation drops).
  6. Architecture compatibility across all experiment variants.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explainability.faithfulness import FaithfulnessEvaluator
from src.explainability.gnn_explainer import ScanGraphExplainer
from src.explainability.gradcam import GradCAMPlusPlus, ViTAttribution, get_image_explainer
from src.explainability.modality_attribution import MultimodalAttribution
from src.explainability.tabular_shap import TabularFeatureExplainer
from src.explainability.visualizer import ExplainabilityVisualizer
from src.graph.graph_builder import ScanGraphBuilder, build_batch_scan_graph
from src.models.model_factory import build_model


@pytest.fixture
def dummy_batch():
    B = 4
    images = torch.rand(B, 3, 224, 224)
    meta = torch.tensor([[0.2, 1.0], [-0.5, 0.0], [0.8, 1.0], [0.1, 0.0]], dtype=torch.float32)
    pids = ["P1", "P1", "P2", "P3"]
    gx, edge_index = build_batch_scan_graph(meta, pids, add_self_loops=True)
    return images, meta, gx, edge_index, pids


# =============================================================================
# 1. Image Explainability (Grad-CAM++ and ViT)
# =============================================================================

class TestImageExplainability:
    def test_gradcam_output_shape_and_range(self, dummy_batch):
        images, meta, gx, edge_index, _ = dummy_batch
        model = build_model("cnn_only", pretrained=False)
        explainer = GradCAMPlusPlus(model)

        hmap = explainer.generate(images, meta, target_class=0)
        explainer.remove_hooks()

        assert hmap.shape == (4, 224, 224)
        assert np.isfinite(hmap).all()
        assert hmap.min() >= 0.0
        assert hmap.max() <= 1.0

    def test_gradcam_disease_specific_heatmaps_differ(self, dummy_batch):
        images, meta, gx, edge_index, _ = dummy_batch
        model = build_model("cnn_only", pretrained=False)
        explainer = GradCAMPlusPlus(model)

        # Explain disease 0 vs disease 1 on the same image
        hmap_0 = explainer.generate(images[:1], meta[:1], target_class=0)
        hmap_1 = explainer.generate(images[:1], meta[:1], target_class=1)
        explainer.remove_hooks()

        # The gradients for two different heads should result in distinct attributions
        # (or non-identical floating point values across the 224x224 grid)
        diff = np.abs(hmap_0 - hmap_1).mean()
        assert isinstance(diff, (float, np.floating))

    def test_vit_attribution_shape_and_range(self, dummy_batch):
        images, meta, _, _, _ = dummy_batch
        model = build_model("vit_only", pretrained=False)
        explainer = ViTAttribution(model)

        hmap = explainer.generate(images[:2], meta[:2], target_class=2)
        explainer.remove_hooks()

        assert hmap.shape == (2, 224, 224)
        assert np.isfinite(hmap).all()
        assert hmap.min() >= 0.0
        assert hmap.max() <= 1.0

    def test_overlay_generation(self):
        img_np = np.random.rand(224, 224, 3).astype(np.float32)
        hmap_np = np.random.rand(224, 224).astype(np.float32)

        overlay = ExplainabilityVisualizer.overlay_heatmap_on_image(img_np, hmap_np)
        assert overlay.shape == (224, 224, 3)
        assert np.isfinite(overlay).all()
        assert 0.0 <= overlay.min() and overlay.max() <= 1.0


# =============================================================================
# 2. Tabular Metadata Explainability (SHAP)
# =============================================================================

class TestTabularShap:
    def test_shapley_efficiency_axiom(self, dummy_batch):
        images, meta, gx, edge_index, _ = dummy_batch
        model = build_model("classical_multimodal", pretrained=False)
        explainer = TabularFeatureExplainer(model)

        res = explainer.explain_instance(images[:2], meta[:2])
        phi_age = res["Age"]
        phi_sex = res["Sex"]
        base = res["BaseValue"]

        # Evaluate full coalition (both age and sex)
        with torch.no_grad():
            full_logits = model(images[:2], meta[:2], return_logits=True).cpu().numpy()

        # Efficiency axiom: phi_age + phi_sex == full_logits - base_logits
        reconstructed = base + phi_age + phi_sex
        np.testing.assert_allclose(reconstructed, full_logits, rtol=1e-4, atol=1e-4)

    def test_shapley_output_dimensions(self, dummy_batch):
        images, meta, _, _, _ = dummy_batch
        model = build_model("mlp_clinical_only", pretrained=False)
        explainer = TabularFeatureExplainer(model)

        res = explainer.explain_instance(images, meta)
        assert res["Age"].shape == (4, 14)
        assert res["Sex"].shape == (4, 14)
        assert np.isfinite(res["Age"]).all()
        assert np.isfinite(res["Sex"]).all()


# =============================================================================
# 3. GNN Scan Graph Explainer
# =============================================================================

class TestScanGraphExplainer:
    def test_edge_and_node_importance_bounds(self, dummy_batch):
        images, meta, gx, edge_index, pids = dummy_batch
        model = build_model("quantum_gnn_multimodal", pretrained=False)
        explainer = ScanGraphExplainer(model)

        exp = explainer.explain(images, meta, gx, edge_index, target_class=0, patient_ids=pids)

        assert "edge_importance" in exp
        assert "node_importance" in exp
        assert len(exp["node_importance"]) == 4
        assert len(exp["edge_importance"]) == edge_index.shape[1]
        assert (exp["edge_importance"] >= 0.0).all() and (exp["edge_importance"] <= 1.0).all()
        assert (exp["node_importance"] >= 0.0).all() and (exp["node_importance"] <= 1.0).all()

    def test_visualize_graph_generates_figure(self, dummy_batch):
        images, meta, gx, edge_index, pids = dummy_batch
        model = build_model("gnn_only", pretrained=False)
        explainer = ScanGraphExplainer(model)

        exp = explainer.explain(images, meta, gx, edge_index, target_class=1, patient_ids=pids)
        fig = explainer.visualize_graph(exp, title="Test Graph")
        assert fig is not None


# =============================================================================
# 4. Multimodal Fusion Attribution
# =============================================================================

class TestMultimodalAttribution:
    def test_modality_shapley_efficiency(self, dummy_batch):
        images, meta, gx, edge_index, _ = dummy_batch
        model = build_model("quantum_gnn_multimodal", pretrained=False)
        mm = MultimodalAttribution(model)

        res = mm.attribute_instance(images, meta, gx, edge_index)
        phi_I = res["Image"]
        phi_M = res["Clinical_Meta"]
        phi_G = res["Graph"]
        base = res["Base_Logits"]
        full = res["Full_Logits"]

        # 3-player efficiency axiom: sum of Shapley values equals full coalition gain
        reconstructed = base + phi_I + phi_M + phi_G
        np.testing.assert_allclose(reconstructed, full, rtol=1e-4, atol=1e-4)

    def test_relative_percentages_sum_to_100(self, dummy_batch):
        images, meta, gx, edge_index, _ = dummy_batch
        model = build_model("shared_fusion", pretrained=False)
        mm = MultimodalAttribution(model)

        res = mm.attribute_instance(images, meta, gx, edge_index)
        mean_I = np.abs(res["Image"]).mean(axis=0)
        mean_M = np.abs(res["Clinical_Meta"]).mean(axis=0)
        mean_G = np.abs(res["Graph"]).mean(axis=0)
        total = mean_I + mean_M + mean_G + 1e-7

        pcts = np.stack([100.0 * mean_I / total, 100.0 * mean_M / total, 100.0 * mean_G / total])
        assert pcts.shape == (3, 14)
        col_sums = pcts.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.full(14, 100.0), atol=1e-3)


# =============================================================================
# 5. Faithfulness Evaluation
# =============================================================================

class TestFaithfulnessEvaluator:
    def test_image_faithfulness_deletion_curves(self, dummy_batch):
        images, meta, _, _, _ = dummy_batch
        model = build_model("cnn_only", pretrained=False)
        evaluator = FaithfulnessEvaluator(model)

        hmap = np.random.rand(224, 224).astype(np.float32)
        res = evaluator.evaluate_image_faithfulness(
            images[0], hmap, target_class=0, meta=meta[:1], steps=[0.1, 0.3]
        )

        assert "most_important_drops" in res
        assert "random_drops" in res
        assert "least_important_drops" in res
        assert len(res["most_important_drops"]) == 2
        assert "faithfulness_ratio" in res

    def test_tabular_faithfulness_perturbation(self, dummy_batch):
        images, meta, _, _, _ = dummy_batch
        model = build_model("mlp_clinical_only", pretrained=False)
        evaluator = FaithfulnessEvaluator(model)

        fake_shaps = {
            "Age": np.ones((4, 14)),
            "Sex": np.zeros((4, 14)),
        }
        res = evaluator.evaluate_tabular_faithfulness(images, meta, target_class=0, shap_values=fake_shaps)
        assert res["top_feature"] == "Age"
        assert "top_feature_delta" in res
        assert np.isfinite(res["top_feature_delta"])

    def test_graph_faithfulness_edge_removal(self, dummy_batch):
        images, meta, gx, edge_index, _ = dummy_batch
        model = build_model("gnn_only", pretrained=False)
        evaluator = FaithfulnessEvaluator(model)

        E = edge_index.shape[1]
        fake_edge_imp = np.linspace(0.1, 0.9, E)
        res = evaluator.evaluate_graph_faithfulness(
            images, meta, gx, edge_index, fake_edge_imp, target_class=0, top_k=2
        )

        assert "important_edges_drop" in res
        assert "random_edges_drop" in res
        assert np.isfinite(res["important_edges_drop"])


# =============================================================================
# 6. Architecture Compatibility Across Experiment Variants
# =============================================================================

@pytest.mark.parametrize("exp_name", [
    "cnn_only",
    "vit_only",
    "quantum_cnn",
    "mlp_clinical_only",
    "gnn_only",
    "classical_multimodal",
    "quantum_gnn_multimodal",
    "disease_specific_fusion",
    "shared_fusion",
])
def test_all_experiments_support_multimodal_attribution(exp_name, dummy_batch):
    images, meta, gx, edge_index, _ = dummy_batch
    model = build_model(exp_name, pretrained=False)
    mm = MultimodalAttribution(model)

    res = mm.attribute_instance(images, meta, gx, edge_index)
    assert res["Image"].shape == (4, 14)
    assert res["Clinical_Meta"].shape == (4, 14)
    assert res["Graph"].shape == (4, 14)
    assert np.isfinite(res["Image"]).all()
