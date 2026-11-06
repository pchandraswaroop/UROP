"""
test_inference.py — Unit and Integration Tests for Inference & Dashboard Services
==================================================================================
Tests:
  1. Input validation (image presence, age limits, sex values).
  2. Image and clinical metadata preprocessing pipelines.
  3. Checkpoint discovery, validation, and error handling.
  4. Multi-label inference output dimensions and probability ranges.
  5. Multi-label 'No Finding' logic.
  6. Strict separation between Research Mode and Demo Mode.
  7. Explanation service integration (Grad-CAM++, SHAP, Graph, Modality).
  8. Graceful error handling for incompatible architectures.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.dataset.label_encoder import DISEASE_LABEL_NAMES
from src.inference.checkpoint_loader import CheckpointError, ModelCheckpointLoader
from src.inference.explanation_service import ExplanationService
from src.inference.predictor import (
    CXRInferenceService,
    InferenceMode,
    PredictionResult,
)
from src.inference.preprocessing import InferencePreprocessor, InputValidationError
from src.models.model_factory import build_model


@pytest.fixture
def sample_pil_image():
    """Create a dummy 224x224 RGB image for testing."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# =============================================================================
# 1. Input Validation Tests
# =============================================================================

class TestInputValidation:
    def test_valid_inputs_pass(self, sample_pil_image):
        InferencePreprocessor.validate_inputs(sample_pil_image, age=45, sex="M")
        InferencePreprocessor.validate_inputs(sample_pil_image, age=0, sex="F")
        InferencePreprocessor.validate_inputs(sample_pil_image, age=120, sex="Female")

    def test_missing_image_raises_error(self):
        with pytest.raises(InputValidationError, match="No image provided"):
            InferencePreprocessor.validate_inputs(None, age=30, sex="M")

    def test_invalid_age_out_of_bounds_raises_error(self, sample_pil_image):
        with pytest.raises(InputValidationError, match="out of valid clinical range"):
            InferencePreprocessor.validate_inputs(sample_pil_image, age=-5, sex="M")

        with pytest.raises(InputValidationError, match="out of valid clinical range"):
            InferencePreprocessor.validate_inputs(sample_pil_image, age=130, sex="F")

    def test_non_integer_age_raises_error(self, sample_pil_image):
        with pytest.raises(InputValidationError, match="Invalid age value"):
            InferencePreprocessor.validate_inputs(sample_pil_image, age="invalid_age", sex="M")

    def test_invalid_sex_raises_error(self, sample_pil_image):
        with pytest.raises(InputValidationError, match="Invalid sex value"):
            InferencePreprocessor.validate_inputs(sample_pil_image, age=40, sex="Unknown")


# =============================================================================
# 2. Preprocessing Tests
# =============================================================================

class TestPreprocessing:
    def test_image_preprocessing_shapes(self, sample_pil_image):
        prep = InferencePreprocessor(image_size=224)
        tensor, raw_rgb = prep.preprocess_image(sample_pil_image)

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32
        assert raw_rgb.shape == (224, 224, 3)
        assert np.isfinite(tensor.numpy()).all()

    def test_metadata_preprocessing_shapes(self):
        prep = InferencePreprocessor()
        meta = prep.preprocess_metadata(age=50, sex="M")

        assert meta.shape == (1, 2)
        assert meta.dtype == torch.float32
        assert meta[0, 1].item() == 1.0  # M -> 1.0

        meta_f = prep.preprocess_metadata(age=30, sex="F")
        assert meta_f[0, 1].item() == 0.0  # F -> 0.0

    def test_single_scan_graph_construction(self):
        prep = InferencePreprocessor()
        meta = prep.preprocess_metadata(age=40, sex="M")
        gx, edge_index = prep.build_single_scan_graph(meta)

        assert gx.shape == (1, 2)
        assert edge_index.shape == (2, 1)
        assert edge_index[0, 0].item() == 0 and edge_index[1, 0].item() == 0  # Self-loop


# =============================================================================
# 3. Checkpoint Discovery & Loading Tests
# =============================================================================

class TestCheckpointLoader:
    def test_missing_checkpoint_returns_false(self):
        is_valid, msg, _ = ModelCheckpointLoader.validate_checkpoint("non_existent_ckpt.pt")
        assert not is_valid
        assert "does not exist" in msg

    def test_loading_missing_checkpoint_raises_checkpoint_error(self):
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        with pytest.raises(CheckpointError, match="Cannot load invalid checkpoint"):
            ModelCheckpointLoader.load_weights("ghost_path.pt", model)

    def test_saving_and_loading_valid_checkpoint(self):
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_model.pt"
            torch.save({
                "epoch": 5,
                "state_dict": model.state_dict(),
                "metrics": {"val_auc_mean": 0.825},
            }, ckpt_path)

            is_valid, msg, meta = ModelCheckpointLoader.validate_checkpoint(ckpt_path, model=model)
            assert is_valid
            assert meta["epoch"] == 5

            loaded_meta = ModelCheckpointLoader.load_weights(ckpt_path, model)
            assert loaded_meta["epoch"] == 5

    def test_validate_auroc_criteria(self):
        from src.inference.checkpoint_loader import validate_auroc
        # 1. 0.0 must be accepted
        ok, val, _ = validate_auroc(0.0)
        assert ok is True and val == 0.0

        # 2. Valid range accepted
        ok, val, _ = validate_auroc(0.785)
        assert ok is True and val == 0.785
        ok, val, _ = validate_auroc(1.0)
        assert ok is True and val == 1.0

        # 3. Non-numeric / boolean / None rejected
        assert validate_auroc(None)[0] is False
        assert validate_auroc(True)[0] is False
        assert validate_auroc(False)[0] is False
        assert validate_auroc("invalid_score")[0] is False

        # 4. NaN and infinities rejected
        assert validate_auroc(float("nan"))[0] is False
        assert validate_auroc(float("inf"))[0] is False
        assert validate_auroc(-float("inf"))[0] is False

        # 5. Out of range rejected
        assert validate_auroc(-0.01)[0] is False
        assert validate_auroc(1.01)[0] is False

    def test_find_best_checkpoint_strict_selection(self):
        from src.inference.checkpoint_loader import ModelCheckpointLoader
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exp_dir = root / "test_exp"
            exp_dir.mkdir()

            # ckpt 1: score 0.80, epoch 1
            torch.save({"epoch": 1, "metrics": {"val_auc_mean": 0.80}}, exp_dir / "ckpt_epoch1_val_auc_mean=0.8000.pt")
            # ckpt 2: score 0.92, epoch 2 (highest valid)
            torch.save({"epoch": 2, "metrics": {"val_auc_mean": 0.92}}, exp_dir / "ckpt_epoch2_val_auc_mean=0.9200.pt")
            # ckpt 3: score 0.0 (valid, but lower)
            torch.save({"epoch": 3, "metrics": {"val_auc_mean": 0.0}}, exp_dir / "ckpt_epoch3_val_auc_mean=0.0000.pt")
            # ckpt 4: score NaN (invalid, must be ignored)
            torch.save({"epoch": 4, "metrics": {"val_auc_mean": float("nan")}}, exp_dir / "ckpt_epoch4_val_auc_mean=nan.pt")
            # ckpt 5: score -0.5 (out of range, must be ignored)
            torch.save({"epoch": 5, "metrics": {"val_auc_mean": -0.5}}, exp_dir / "ckpt_epoch5_val_auc_mean=-0.5000.pt")

            best = ModelCheckpointLoader.find_best_checkpoint("test_exp", checkpoint_dir=root)
            assert best is not None
            assert best.name == "ckpt_epoch2_val_auc_mean=0.9200.pt"

    def test_validate_checkpoint_strict_key_parity(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        state = model.state_dict()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Missing key checkpoint
            incomplete_state = {k: v for k, v in state.items() if "0.weight" not in k}
            p1 = Path(tmp_dir) / "missing_key.pt"
            torch.save({"state_dict": incomplete_state}, p1)

            ok, msg, _ = ModelCheckpointLoader.validate_checkpoint(p1, model=model)
            assert ok is False
            assert "Strict key parity required" in msg

            # Shape mismatch checkpoint
            mismatched_state = {k: v.clone() for k, v in state.items()}
            mismatched_state["0.weight"] = torch.zeros(10, 4)  # expected (4, 4)
            p2 = Path(tmp_dir) / "shape_mismatch.pt"
            torch.save({"state_dict": mismatched_state}, p2)

            ok, msg, _ = ModelCheckpointLoader.validate_checkpoint(p2, model=model)
            assert ok is False
            assert "Tensor shape mismatch" in msg


# =============================================================================
# 4. Inference Service & Multi-Label Logic Tests
# =============================================================================

class TestInferenceService:
    def test_demo_mode_inference(self, sample_pil_image):
        service = CXRInferenceService(device="cpu")
        result = service.predict(
            image_input=sample_pil_image,
            age=55,
            sex="M",
            experiment_name="quantum_gnn_multimodal",
            mode=InferenceMode.DEMO,
            threshold=0.5,
        )

        assert result.is_demo is True
        assert len(result.probabilities) == 14
        for name in DISEASE_LABEL_NAMES:
            assert name in result.probabilities
            p = result.probabilities[name]
            assert 0.0 <= p <= 1.0

    def test_research_mode_blocks_when_checkpoint_missing(self, sample_pil_image, monkeypatch):
        service = CXRInferenceService(device="cpu")
        # 1. When explicit path does not exist
        with pytest.raises(CheckpointError, match="Checkpoint file does not exist"):
            service.predict(
                image_input=sample_pil_image,
                age=55,
                sex="M",
                experiment_name="quantum_gnn_multimodal",
                mode=InferenceMode.RESEARCH,
                checkpoint_path="missing_weights.pt",
            )

        # 2. When checkpoint_path is None and no checkpoints exist on disk
        monkeypatch.setattr(ModelCheckpointLoader, "list_available_checkpoints", staticmethod(lambda: {}))
        with pytest.raises(CheckpointError, match="No trained checkpoint found"):
            service.predict(
                image_input=sample_pil_image,
                age=55,
                sex="M",
                experiment_name="quantum_gnn_multimodal",
                mode=InferenceMode.RESEARCH,
                checkpoint_path=None,
            )

    def test_research_mode_succeeds_with_valid_checkpoint(self, sample_pil_image):
        model = build_model("cnn_only", pretrained=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_cnn.pt"
            torch.save({"state_dict": model.state_dict(), "epoch": 1}, ckpt_path)

            service = CXRInferenceService(device="cpu")
            result = service.predict(
                image_input=sample_pil_image,
                age=40,
                sex="F",
                experiment_name="cnn_only",
                mode=InferenceMode.RESEARCH,
                checkpoint_path=str(ckpt_path),
                threshold=0.5,
            )

            assert result.is_demo is False
            assert len(result.probabilities) == 14
            assert result.is_no_finding in [True, False]

    def test_no_finding_logic_when_all_below_threshold(self, sample_pil_image):
        service = CXRInferenceService(device="cpu")
        # Set threshold to 1.0 so no probability will exceed it
        result = service.predict(
            image_input=sample_pil_image,
            age=50,
            sex="M",
            experiment_name="cnn_only",
            mode=InferenceMode.DEMO,
            threshold=1.0,
        )

        assert result.is_no_finding is True
        assert len(result.positive_predictions) == 0

    def test_positive_finding_logic_when_exceeding_threshold(self, sample_pil_image):
        service = CXRInferenceService(device="cpu")
        # Set threshold to 0.001 so at least one probability exceeds it
        result = service.predict(
            image_input=sample_pil_image,
            age=50,
            sex="M",
            experiment_name="cnn_only",
            mode=InferenceMode.DEMO,
            threshold=0.001,
        )

        assert result.is_no_finding is False
        assert len(result.positive_predictions) > 0


# =============================================================================
# 5. Explanation Service Tests
# =============================================================================

class TestExplanationService:
    @pytest.fixture
    def demo_prediction_result(self, sample_pil_image):
        service = CXRInferenceService(device="cpu")
        return service.predict(
            image_input=sample_pil_image,
            age=60,
            sex="M",
            experiment_name="quantum_gnn_multimodal",
            mode=InferenceMode.DEMO,
            threshold=0.3,
        )

    def test_image_explanation_returns_valid_heatmap_and_overlay(self, demo_prediction_result):
        res = ExplanationService.get_image_explanation(demo_prediction_result, "Cardiomegaly")
        assert res["status"] == "success"
        assert res["heatmap"].shape == (224, 224)
        assert res["overlay"].shape == (224, 224, 3)

    def test_tabular_explanation_returns_age_and_sex_shap(self, demo_prediction_result):
        res = ExplanationService.get_tabular_explanation(demo_prediction_result, "Cardiomegaly")
        assert res["status"] == "success"
        assert np.isfinite(res["age_shap"])
        assert np.isfinite(res["sex_shap"])

    def test_graph_explanation_handles_single_scan_gracefully(self, demo_prediction_result):
        res = ExplanationService.get_graph_explanation(demo_prediction_result, "Cardiomegaly")
        # For single scan with node_count <= 1, it should provide a helpful note without crashing
        assert res["status"] in ("single_scan_note", "success")

    def test_modality_attribution_sums_to_100_percent(self, demo_prediction_result):
        res = ExplanationService.get_modality_attribution(demo_prediction_result)
        assert res["status"] == "success"
        assert "pct_image" in res
        assert "pct_meta" in res
        assert "pct_graph" in res

        total = res["pct_image"] + res["pct_meta"] + res["pct_graph"]
        np.testing.assert_allclose(total, 100.0, rtol=1e-3, atol=1e-3)

    def test_image_explanation_on_clinical_only_model_returns_unavailable(self, sample_pil_image):
        service = CXRInferenceService(device="cpu")
        res = service.predict(
            image_input=sample_pil_image,
            age=45,
            sex="F",
            experiment_name="mlp_clinical_only",
            mode=InferenceMode.DEMO,
        )
        exp_res = ExplanationService.get_image_explanation(res, "Atelectasis")
        assert exp_res["status"] == "unavailable"
        assert "zeroed out" in exp_res["message"]

    def test_multi_scan_inference(self, sample_pil_image):
        """Test inference with multiple scans belonging to the same patient."""
        service = CXRInferenceService(device="cpu")
        images = [sample_pil_image, sample_pil_image, sample_pil_image]  # 3 scans
        res = service.predict(
            image_input=images,
            age=62,
            sex="F",
            experiment_name="quantum_gnn_multimodal",
            mode=InferenceMode.DEMO,
            threshold=0.4,
            primary_scan_index=1,
        )
        assert res.is_multi_scan is True
        assert res.node_count == 3
        assert res.edge_count == 9  # 3x3 all-to-all directed edges including self loops
        assert res.primary_scan_index == 1
        assert len(res.probabilities) == 14
        assert len(res.all_probabilities) == 3
        assert len(res.all_raw_rgbs) == 3

    def test_multi_scan_patient_graph_builder(self):
        """Test patient graph construction creates isolated patient graphs with self loops."""
        prep = InferencePreprocessor()
        edge_index = prep.build_multi_scan_patient_graph(num_scans=4)
        assert edge_index.shape == (2, 16)
        # Check that self loops exist for each node
        for i in range(4):
            has_self_loop = any(edge_index[0, k] == i and edge_index[1, k] == i for k in range(16))
            assert has_self_loop
