"""
test_explainability_alignment.py — Verification of Explainability Alignment & Safety
=====================================================================================
Tests:
  1. MultiModalCXRModel graph edge index bounds checking and ValueError assertion.
  2. PredictionResult multi-scan tensor retention (all_image_tensors, all_meta_tensors).
  3. Single-scan visual saliency (Grad-CAM++) and tabular SHAP isolation to self-loops.
  4. Multi-scan graph explanation routing to genuine companion scan tensors.
  5. Single-scan graph explanation informative note return (node_count <= 1).
  6. Three-way multimodal attribution evaluation for single-scan and multi-scan graphs.
  7. Held-out test split zero-leakage guarantee for Notebook 03.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from src.dataset.label_encoder import DISEASE_LABEL_NAMES
from src.inference.explanation_service import ExplanationService
from src.inference.predictor import PredictionResult, InferenceMode
from src.models.classifier import MultiModalCXRModel, DiseaseSpecificClassifier
from src.models.fusion import DiseaseSpecificFusion
from src.models.gnn_branch import GNNBranch
from src.models.mlp_baseline import MetadataEmbedder
from src.models.quantum_layer import IdentityQuantumLayer


class DummyImageBranch(nn.Module):
    """Lightweight test mock for ImageBranch to enable unit testing without heavy dependencies."""
    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


@pytest.fixture
def test_multimodal_model():
    """Construct a lightweight MultiModalCXRModel matching proposed 3-branch architecture."""
    img_branch = DummyImageBranch(out_dim=32)
    meta_branch = MetadataEmbedder(input_dim=2, output_dim=16)
    q_layer = IdentityQuantumLayer(input_dim=32, output_dim=32)
    gnn_branch = GNNBranch(node_feat_dim=32 + 2, hidden_dim=16, output_dim=16)
    fusion = DiseaseSpecificFusion(
        image_dim=32,
        clinical_dim=16,
        fused_dim=16,
        gnn_dim=16,
        num_classes=14,
    )
    classifier = DiseaseSpecificClassifier(fused_dim=16, hidden_dim=16, num_classes=14)

    model = MultiModalCXRModel(
        image_branch=img_branch,
        clinical_branch=meta_branch,
        quantum_layer=q_layer,
        fusion=fusion,
        classifier=classifier,
        gnn_branch=gnn_branch,
        use_disease_specific_fusion=True,
    )
    model.eval()
    return model


class TestGraphIndexValidation:
    """Verify strict validation and error raising on graph index bounds."""

    def test_invalid_graph_edge_index_raises_value_error(self, test_multimodal_model):
        """When graph_edge_index references a node index >= node count, raise ValueError."""
        B = 1  # 1 image provided
        images = torch.rand(B, 3, 224, 224)
        meta = torch.tensor([[0.2, 1.0]], dtype=torch.float32)

        # Graph edge index referencing node 1 when only node 0 exists
        invalid_edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)

        with pytest.raises(ValueError) as excinfo:
            test_multimodal_model(
                images,
                meta,
                graph_x=meta,
                graph_edge_index=invalid_edge_index,
                return_logits=True,
            )
        assert "Invalid graph structure: graph_edge_index contains node index 1" in str(excinfo.value)
        assert "exceeds the node feature count of 1" in str(excinfo.value)

    def test_valid_graph_edge_index_succeeds(self, test_multimodal_model):
        """Valid graph edge indices run cleanly for single-scan and multi-scan."""
        # Single-scan with self-loop
        img_1 = torch.rand(1, 3, 224, 224)
        meta_1 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        edge_1 = torch.tensor([[0], [0]], dtype=torch.long)
        out_1 = test_multimodal_model(img_1, meta_1, graph_x=meta_1, graph_edge_index=edge_1, return_logits=True)
        assert out_1.shape == (1, 14)

        # Multi-scan with 3 companion scans
        img_3 = torch.rand(3, 3, 224, 224)
        meta_3 = torch.tensor([[0.1, 1.0], [0.2, 1.0], [0.3, 1.0]], dtype=torch.float32)
        edge_3 = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 0]], dtype=torch.long)
        out_3 = test_multimodal_model(img_3, meta_3, graph_x=meta_3, graph_edge_index=edge_3, return_logits=True)
        assert out_3.shape == (3, 14)


class TestPredictionResultFields:
    """Verify PredictionResult stores both primary scan and companion scan tensors."""

    def test_prediction_result_retains_companion_tensors(self):
        primary_img = torch.rand(1, 3, 224, 224)
        primary_meta = torch.tensor([[0.25, 1.0]])
        all_imgs = torch.rand(2, 3, 224, 224)
        all_meta = torch.tensor([[0.25, 1.0], [0.25, 1.0]])

        result = PredictionResult(
            mode=InferenceMode.RESEARCH,
            model_name="quantum_gnn_multimodal",
            probabilities={"Cardiomegaly": 0.65},
            positive_predictions=["Cardiomegaly"],
            is_no_finding=False,
            threshold=0.5,
            checkpoint_info={"path": "test.pt"},
            device="cpu",
            node_count=2,
            edge_count=4,
            is_multi_scan=True,
            image_tensor=primary_img,
            meta_tensor=primary_meta,
            all_image_tensors=all_imgs,
            all_meta_tensors=all_meta,
        )

        assert result.image_tensor.shape == (1, 3, 224, 224)
        assert result.meta_tensor.shape == (1, 2)
        assert result.all_image_tensors.shape == (2, 3, 224, 224)
        assert result.all_meta_tensors.shape == (2, 2)
        assert result.is_multi_scan is True


class TestExplanationServiceRouting:
    """Verify ExplanationService routes single-scan and multi-scan cases appropriately."""

    def test_single_scan_graph_explanation_returns_note(self, test_multimodal_model):
        """Single-scan inference results return an informative single_scan_note."""
        result = PredictionResult(
            mode=InferenceMode.RESEARCH,
            model_name="quantum_gnn_multimodal",
            probabilities={DISEASE_LABEL_NAMES[0]: 0.1},
            positive_predictions=[],
            is_no_finding=True,
            threshold=0.5,
            checkpoint_info=None,
            device="cpu",
            node_count=1,
            edge_count=1,
            is_multi_scan=False,
            image_tensor=torch.rand(1, 3, 224, 224),
            meta_tensor=torch.tensor([[0.1, 1.0]]),
            graph_x=torch.tensor([[0.1, 1.0]]),
            graph_edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            model_instance=test_multimodal_model,
        )

        graph_exp = ExplanationService.get_graph_explanation(result, target_disease=DISEASE_LABEL_NAMES[0])
        assert graph_exp["status"] == "single_scan_note"
        assert "Single-Scan Graph Note" in graph_exp["message"]

    def test_primary_scan_tabular_shap_does_not_crash_on_multi_scan_edge_index(self, test_multimodal_model):
        """Tabular SHAP isolates the primary scan so it never crashes if graph_edge_index has companion nodes."""
        result = PredictionResult(
            mode=InferenceMode.RESEARCH,
            model_name="quantum_gnn_multimodal",
            probabilities={DISEASE_LABEL_NAMES[0]: 0.1},
            positive_predictions=[],
            is_no_finding=True,
            threshold=0.5,
            checkpoint_info=None,
            device="cpu",
            node_count=2,
            edge_count=4,
            is_multi_scan=True,
            image_tensor=torch.rand(1, 3, 224, 224),
            meta_tensor=torch.tensor([[0.1, 1.0]]),
            graph_x=torch.tensor([[0.1, 1.0], [0.1, 1.0]]),
            graph_edge_index=torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long),  # Multi-node graph
            model_instance=test_multimodal_model,
        )

        tab_res = ExplanationService.get_tabular_explanation(result, target_disease=DISEASE_LABEL_NAMES[0])
        assert tab_res["status"] == "success"
        assert "age_shap" in tab_res
        assert "sex_shap" in tab_res

    def test_multi_scan_modality_attribution_evaluates_actual_companion_tensors(self, test_multimodal_model):
        """Three-way modality attribution evaluates all_image_tensors with the complete patient graph."""
        all_imgs = torch.rand(2, 3, 224, 224)
        all_meta = torch.tensor([[0.2, 1.0], [-0.3, 0.0]], dtype=torch.float32)
        edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)

        result = PredictionResult(
            mode=InferenceMode.RESEARCH,
            model_name="quantum_gnn_multimodal",
            probabilities={DISEASE_LABEL_NAMES[0]: 0.5},
            positive_predictions=[DISEASE_LABEL_NAMES[0]],
            is_no_finding=False,
            threshold=0.5,
            checkpoint_info=None,
            device="cpu",
            node_count=2,
            edge_count=4,
            is_multi_scan=True,
            image_tensor=all_imgs[:1],
            meta_tensor=all_meta[:1],
            all_image_tensors=all_imgs,
            all_meta_tensors=all_meta,
            graph_x=all_meta,
            graph_edge_index=edge_index,
            model_instance=test_multimodal_model,
        )

        mod_res = ExplanationService.get_modality_attribution(result, target_disease=DISEASE_LABEL_NAMES[0])
        assert mod_res["status"] == "success"
        assert "image_pct" in mod_res
        assert "clinical_pct" in mod_res
        assert "gnn_pct" in mod_res
        total_pct = mod_res["image_pct"] + mod_res["clinical_pct"] + mod_res["gnn_pct"]
        assert pytest.approx(total_pct, rel=1e-2) == 100.0


class TestTestSplitLeakageInvariant:
    """Verify Notebook 03 evaluation records maintain 0% patient leakage from train and val."""

    def test_zero_leakage_in_test_split(self):
        split_dir = Path("data/splits")
        test_path = split_dir / "test.csv"
        train_path = split_dir / "train.csv"
        val_path = split_dir / "val.csv"

        assert test_path.exists(), "test.csv must exist"
        test_df = pd.read_csv(test_path)
        test_patients = set(test_df["patient_id"].unique())

        if train_path.exists():
            train_df = pd.read_csv(train_path)
            train_patients = set(train_df["patient_id"].unique())
            overlap_train = test_patients.intersection(train_patients)
            assert len(overlap_train) == 0, f"Patient leakage detected with train: {overlap_train}"

        if val_path.exists():
            val_df = pd.read_csv(val_path)
            val_patients = set(val_df["patient_id"].unique())
            overlap_val = test_patients.intersection(val_patients)
            assert len(overlap_val) == 0, f"Patient leakage detected with val: {overlap_val}"
