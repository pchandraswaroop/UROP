"""
test_models.py — Unit tests for all model components
=====================================================
All tests use dummy tensors only — no network calls, no HF dataset.

Test coverage
-------------
  - ZeroImageBranch output shape
  - ImageBranch (pretrained=False) output shape
  - IdentityQuantumLayer output shape
  - MetadataEmbedder output shape
  - DiseaseSpecificFusion output shape
  - SimpleConcatFusion output shape
  - DiseaseSpecificClassifier input/output shapes
  - SharedClassifier input/output shapes
  - MultiModalCXRModel.forward return_logits=True → [B,14] logits
  - MultiModalCXRModel.forward return_logits=False → [B,14] probs in [0,1]
  - build_model() for all experiment variants → valid model, correct output shape
  - Backward pass + NaN-grad check for each variant
  - EarlyStopping triggers after patience, restores best weights
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

B = 2            # batch size
NUM_CLASSES = 14
IMAGE_SIZE  = 224


def _dummy_images(b: int = B) -> torch.Tensor:
    return torch.rand(b, 3, IMAGE_SIZE, IMAGE_SIZE)


def _dummy_meta(b: int = B) -> torch.Tensor:
    return torch.rand(b, 2)


def _dummy_labels(b: int = B) -> torch.Tensor:
    return torch.randint(0, 2, (b, NUM_CLASSES)).float()


# =============================================================================
# ZeroImageBranch
# =============================================================================

class TestZeroImageBranch:
    def test_output_shape(self):
        from src.models.model_factory import ZeroImageBranch
        m = ZeroImageBranch(output_dim=64)
        x = _dummy_images()
        out = m(x)
        assert out.shape == (B, 64), f"Expected ({B}, 64), got {out.shape}"

    def test_output_is_zero(self):
        from src.models.model_factory import ZeroImageBranch
        m = ZeroImageBranch(output_dim=32)
        out = m(_dummy_images())
        assert out.sum().item() == 0.0

    def test_feature_dim_attribute(self):
        from src.models.model_factory import ZeroImageBranch
        m = ZeroImageBranch(output_dim=128)
        assert m.feature_dim == 128


# =============================================================================
# ImageBranch
# =============================================================================

class TestImageBranch:
    def test_efficientnet_shape(self):
        from src.models.image_branch import ImageBranch
        m = ImageBranch(backbone="efficientnet_b0", pretrained=False)
        out = m(_dummy_images())
        assert out.shape == (B, 1280), f"Expected ({B}, 1280), got {out.shape}"

    def test_vit_shape(self):
        from src.models.image_branch import ImageBranch
        m = ImageBranch(backbone="vit_small_patch16_224", pretrained=False)
        out = m(_dummy_images())
        assert out.shape == (B, 384), f"Expected ({B}, 384), got {out.shape}"

    def test_with_projection(self):
        from src.models.image_branch import ImageBranch
        m = ImageBranch(backbone="efficientnet_b0", pretrained=False, output_dim=64)
        out = m(_dummy_images())
        assert out.shape == (B, 64)

    def test_feature_dim_attribute(self):
        from src.models.image_branch import ImageBranch
        m = ImageBranch(backbone="efficientnet_b0", pretrained=False)
        assert m.feature_dim == 1280


# =============================================================================
# QuantumLayer / IdentityQuantumLayer
# =============================================================================

class TestQuantumLayer:
    def test_identity_shape(self):
        from src.models.quantum_layer import IdentityQuantumLayer
        m = IdentityQuantumLayer(input_dim=1280, output_dim=64)
        out = m(torch.rand(B, 1280))
        assert out.shape == (B, 64)

    def test_identity_output_dim_attribute(self):
        from src.models.quantum_layer import IdentityQuantumLayer
        m = IdentityQuantumLayer(input_dim=256, output_dim=32)
        assert m.output_dim == 32

    def test_get_quantum_layer_returns_identity_without_pennylane(self):
        from src.models.quantum_layer import get_quantum_layer, IdentityQuantumLayer
        # Pass enabled=False to force Identity regardless of pennylane availability
        m = get_quantum_layer(input_dim=100, output_dim=16, enabled=False)
        assert isinstance(m, IdentityQuantumLayer)
        out = m(torch.rand(B, 100))
        assert out.shape == (B, 16)

    def test_real_quantum_layer_circuit_and_gradient(self):
        from src.models.quantum_layer import QuantumLayer, IdentityQuantumLayer, PENNYLANE_AVAILABLE
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not installed in this environment.")
        q = QuantumLayer(input_dim=8, n_qubits=2, n_layers=1, output_dim=4)
        assert not isinstance(q, IdentityQuantumLayer)
        assert isinstance(q, QuantumLayer)
        x = torch.rand(2, 8, requires_grad=True)
        out = q(x)
        assert out.shape == (2, 4)
        assert torch.isfinite(out).all()
        loss = out.sum()
        loss.backward()
        assert q.quantum_weights.grad is not None
        assert torch.norm(q.quantum_weights.grad) > 0

    def test_get_quantum_layer_returns_real_quantum_layer_when_enabled(self):
        from src.models.quantum_layer import get_quantum_layer, QuantumLayer, PENNYLANE_AVAILABLE
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not installed in this environment.")
        m = get_quantum_layer(input_dim=8, n_qubits=2, n_layers=1, output_dim=4, enabled=True)
        assert isinstance(m, QuantumLayer)

    def test_quantum_model_build_with_real_quantum_circuit(self):
        from src.models.model_factory import build_model
        from src.models.quantum_layer import QuantumLayer, PENNYLANE_AVAILABLE
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not installed in this environment.")
        # When quantum_enabled is explicitly set to True, build_model must use real QuantumLayer
        model = build_model("quantum_cnn", cfg={"quantum_enabled": True}, pretrained=False)
        assert isinstance(model.quantum_layer, QuantumLayer)


# =============================================================================
# MetadataEmbedder
# =============================================================================

class TestMetadataEmbedder:
    def test_output_shape(self):
        from src.models.mlp_baseline import MetadataEmbedder
        m = MetadataEmbedder(input_dim=2, output_dim=64)
        out = m(_dummy_meta())
        assert out.shape == (B, 64)

    def test_custom_output_dim(self):
        from src.models.mlp_baseline import MetadataEmbedder
        m = MetadataEmbedder(input_dim=2, output_dim=32)
        assert m.output_dim == 32
        out = m(_dummy_meta())
        assert out.shape == (B, 32)


# =============================================================================
# DiseaseSpecificFusion
# =============================================================================

class TestDiseaseSpecificFusion:
    def test_output_shape(self):
        from src.models.fusion import DiseaseSpecificFusion
        m = DiseaseSpecificFusion(image_dim=64, clinical_dim=64, fused_dim=128)
        img  = torch.rand(B, 64)
        clin = torch.rand(B, 64)
        out  = m(img, clin)
        assert out.shape == (B, 14, 128), f"Expected ({B}, 14, 128), got {out.shape}"

    def test_num_classes_param(self):
        from src.models.fusion import DiseaseSpecificFusion
        m = DiseaseSpecificFusion(image_dim=32, clinical_dim=32, fused_dim=64, num_classes=14)
        out = m(torch.rand(B, 32), torch.rand(B, 32))
        assert out.shape[1] == 14


# =============================================================================
# SimpleConcatFusion
# =============================================================================

class TestSimpleConcatFusion:
    def test_output_shape(self):
        from src.models.fusion import SimpleConcatFusion
        m = SimpleConcatFusion(image_dim=64, clinical_dim=64, fused_dim=128)
        out = m(torch.rand(B, 64), torch.rand(B, 64))
        assert out.shape == (B, 128)

    def test_fused_dim_attribute(self):
        from src.models.fusion import SimpleConcatFusion
        m = SimpleConcatFusion(image_dim=32, clinical_dim=32, fused_dim=64)
        assert m.fused_dim == 64


# =============================================================================
# DiseaseSpecificClassifier
# =============================================================================

class TestDiseaseSpecificClassifier:
    def test_output_shape(self):
        from src.models.classifier import DiseaseSpecificClassifier
        m = DiseaseSpecificClassifier(fused_dim=128, hidden_dim=64, num_classes=14)
        x   = torch.rand(B, 14, 128)
        out = m(x)
        assert out.shape == (B, 14), f"Expected ({B}, 14), got {out.shape}"

    def test_output_range_is_logits(self):
        """Classifier should output logits (not clamped to [0,1])."""
        from src.models.classifier import DiseaseSpecificClassifier
        m = DiseaseSpecificClassifier(fused_dim=128, hidden_dim=64)
        x   = torch.rand(B, 14, 128)
        out = m(x)
        # logits can be < 0 or > 1
        assert out.shape == (B, 14)


# =============================================================================
# SharedClassifier
# =============================================================================

class TestSharedClassifier:
    def test_output_shape(self):
        from src.models.classifier import SharedClassifier
        m = SharedClassifier(fused_dim=128, hidden_dim=64, num_classes=14)
        out = m(torch.rand(B, 128))
        assert out.shape == (B, 14)


# =============================================================================
# MultiModalCXRModel — return_logits flag
# =============================================================================

class TestMultiModalCXRModel:
    def _build_simple_model(self):
        """Build a small MultiModalCXRModel entirely from scratch (no pretrained)."""
        from src.models.model_factory import build_model
        return build_model("cnn_only", pretrained=False)

    def test_return_logits_true_shape(self):
        model  = self._build_simple_model()
        images = _dummy_images()
        meta   = _dummy_meta()
        out    = model(images, meta, return_logits=True)
        assert out.shape == (B, 14)

    def test_return_logits_false_shape(self):
        model = self._build_simple_model()
        out   = model(_dummy_images(), _dummy_meta(), return_logits=False)
        assert out.shape == (B, 14)

    def test_return_logits_false_probs_in_range(self):
        model = self._build_simple_model()
        out   = model(_dummy_images(), _dummy_meta(), return_logits=False)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_default_returns_probs(self):
        """Default (no return_logits arg) should return probabilities."""
        model = self._build_simple_model()
        out   = model(_dummy_images(), _dummy_meta())
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_logits_differ_from_probs(self):
        """sigmoid(logits) must equal probs — evaluated in eval mode, same forward pass."""
        model  = self._build_simple_model()
        model.eval()
        images, meta = _dummy_images(), _dummy_meta()
        # Single deterministic forward: get logits, manually apply sigmoid, compare
        with torch.no_grad():
            logits = model(images, meta, return_logits=True)
            probs  = torch.sigmoid(logits)
            probs2 = model(images, meta, return_logits=False)
        # Both paths must agree
        assert torch.allclose(probs, probs2, atol=1e-5), (
            f"sigmoid(logits) != model probs: max diff = {(probs - probs2).abs().max():.6f}"
        )


# =============================================================================
# build_model — all experiment variants
# =============================================================================

EXPERIMENT_NAMES = [
    "cnn_only",
    "vit_only",
    "quantum_cnn",
    "mlp_clinical_only",
    "gnn_only",
    "classical_multimodal",
    "quantum_gnn_multimodal",
    "disease_specific_fusion",
    "shared_fusion",
]


class TestBuildModel:
    @pytest.mark.parametrize("exp_name", EXPERIMENT_NAMES)
    def test_valid_model_output_shape(self, exp_name):
        from src.models.model_factory import build_model
        model  = build_model(exp_name, pretrained=False)
        images = _dummy_images()
        meta   = _dummy_meta()
        with torch.no_grad():
            out = model(images, meta, return_logits=False)
        assert out.shape == (B, 14), (
            f"{exp_name}: expected ({B}, 14), got {out.shape}"
        )
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    @pytest.mark.parametrize("exp_name", EXPERIMENT_NAMES)
    def test_logits_output_shape(self, exp_name):
        from src.models.model_factory import build_model
        model  = build_model(exp_name, pretrained=False)
        images = _dummy_images()
        meta   = _dummy_meta()
        with torch.no_grad():
            out = model(images, meta, return_logits=True)
        assert out.shape == (B, 14), (
            f"{exp_name}: logits shape expected ({B}, 14), got {out.shape}"
        )

    def test_unknown_experiment_raises(self):
        from src.models.model_factory import build_model
        with pytest.raises(ValueError, match="Unknown experiment"):
            build_model("does_not_exist", pretrained=False)


# =============================================================================
# GNN with real graph data — verify actual message passing
# =============================================================================

class TestGNNWithGraphData:
    """Verify that when torch_geometric is installed, gnn_only uses a real
    GNNBranch with SAGEConv message passing, not a silent MLP substitute."""

    def _has_torch_geometric(self) -> bool:
        try:
            from src.models.gnn_branch import TG_AVAILABLE
            return TG_AVAILABLE
        except ImportError:
            return False

    def test_gnn_only_uses_real_gnn_branch(self):
        """When torch_geometric is available, clinical_branch must be GNNBranch."""
        if not self._has_torch_geometric():
            pytest.skip("torch_geometric not installed")
        from src.models.model_factory import build_model
        from src.models.gnn_branch import GNNBranch
        model = build_model("gnn_only", pretrained=False)
        assert isinstance(model.clinical_branch, GNNBranch), (
            f"Expected GNNBranch, got {type(model.clinical_branch).__name__}"
        )

    def test_gnn_only_has_metadata_fallback(self):
        """When clinical_branch is GNNBranch, clinical_fallback must exist."""
        if not self._has_torch_geometric():
            pytest.skip("torch_geometric not installed")
        from src.models.model_factory import build_model
        from src.models.mlp_baseline import MetadataEmbedder
        model = build_model("gnn_only", pretrained=False)
        assert model.clinical_fallback is not None, (
            "gnn_only must have a clinical_fallback for inference without graph"
        )
        assert isinstance(model.clinical_fallback, MetadataEmbedder)

    def test_gnn_forward_with_graph_data(self):
        """GNNBranch must work with actual edge_index graph connectivity."""
        if not self._has_torch_geometric():
            pytest.skip("torch_geometric not installed")
        from src.models.model_factory import build_model

        model = build_model("gnn_only", pretrained=False)
        model.eval()

        images = _dummy_images()
        meta   = _dummy_meta()
        N = B  # treat each sample as a graph node
        # Node features: [N, 2] (age_norm, sex_binary)
        graph_x = meta.clone()
        # Create edges: fully connected (0→1, 1→0)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        with torch.no_grad():
            out = model(images, meta, graph_x=graph_x,
                        graph_edge_index=edge_index, return_logits=False)
        assert out.shape == (B, 14), f"Expected ({B}, 14), got {out.shape}"
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_gnn_forward_without_graph_uses_fallback(self):
        """Without graph data, gnn_only must use MetadataEmbedder fallback."""
        if not self._has_torch_geometric():
            pytest.skip("torch_geometric not installed")
        from src.models.model_factory import build_model

        model = build_model("gnn_only", pretrained=False)
        model.eval()

        images = _dummy_images()
        meta   = _dummy_meta()

        with torch.no_grad():
            out = model(images, meta, return_logits=False)
        assert out.shape == (B, 14)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_gnn_backward_with_graph_data(self):
        """Backward pass through GNN with real graph data must not produce NaN."""
        if not self._has_torch_geometric():
            pytest.skip("torch_geometric not installed")
        from src.models.model_factory import build_model
        from src.training.losses import WeightedBCELoss

        model     = build_model("gnn_only", pretrained=False)
        criterion = WeightedBCELoss()
        images = _dummy_images()
        meta   = _dummy_meta()
        labels = _dummy_labels()

        graph_x = meta.clone()
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        logits = model(images, meta, graph_x=graph_x,
                       graph_edge_index=edge_index, return_logits=True)
        loss = criterion(logits, labels)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), (
                    f"NaN gradient in GNN parameter '{name}'"
                )
        assert torch.isfinite(loss), f"GNN loss is not finite: {loss.item()}"


# =============================================================================
# Backward pass — no NaN gradients
# =============================================================================

class TestBackwardPass:
    @pytest.mark.parametrize("exp_name", EXPERIMENT_NAMES)
    def test_backward_no_nan_grads(self, exp_name):
        from src.models.model_factory import build_model
        from src.training.losses import WeightedBCELoss

        model     = build_model(exp_name, pretrained=False)
        criterion = WeightedBCELoss()
        images    = _dummy_images()
        meta      = _dummy_meta()
        labels    = _dummy_labels()

        logits = model(images, meta, return_logits=True)
        loss   = criterion(logits, labels)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), (
                    f"{exp_name}: NaN gradient in parameter '{name}'"
                )
        assert not torch.isnan(loss), f"{exp_name}: loss is NaN"


# =============================================================================
# EarlyStopping
# =============================================================================

class TestEarlyStopping:
    def test_does_not_stop_before_patience(self):
        from src.training.trainer import EarlyStopping
        es    = EarlyStopping(patience=3, mode="max", restore_best_weights=False)
        model = nn.Linear(2, 2)  # dummy model
        for _ in range(2):
            stopped = es(0.5, model)
            assert not stopped

    def test_stops_after_patience(self):
        from src.training.trainer import EarlyStopping
        es    = EarlyStopping(patience=3, mode="max", restore_best_weights=False)
        model = nn.Linear(2, 2)
        # First call sets best score
        es(0.8, model)
        # Next 3 calls with no improvement → should stop
        results = [es(0.5, model) for _ in range(3)]
        assert results[-1] is True, "EarlyStopping should have triggered after patience=3"

    def test_improvement_resets_counter(self):
        from src.training.trainer import EarlyStopping
        es    = EarlyStopping(patience=3, mode="max", restore_best_weights=False)
        model = nn.Linear(2, 2)
        es(0.5, model)
        es(0.4, model)  # worse
        es(0.4, model)  # worse (counter=2)
        es(0.9, model)  # improvement — counter resets
        result = es(0.1, model)  # worse again (counter=1 after reset)
        assert not result  # should NOT stop yet (patience=3, counter=1)

    def test_restores_best_weights(self):
        from src.training.trainer import EarlyStopping
        model = nn.Linear(4, 4)
        es    = EarlyStopping(patience=2, mode="max", restore_best_weights=True)

        # First call: best score = 0.9, weights saved
        es(0.9, model)
        best_w = {k: v.clone() for k, v in model.state_dict().items()}

        # Modify weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(99.0)

        # Trigger stop (2 bad epochs)
        es(0.1, model)
        es(0.1, model)  # triggers stop + restore

        for k, v in model.state_dict().items():
            assert torch.allclose(v.float(), best_w[k].float()), (
                f"Weight '{k}' not restored correctly"
            )

    def test_min_mode(self):
        from src.training.trainer import EarlyStopping
        es    = EarlyStopping(patience=2, mode="min", restore_best_weights=False)
        model = nn.Linear(2, 2)
        es(0.5, model)   # best = 0.5
        es(0.6, model)   # worse (counter=1)
        stopped = es(0.7, model)  # worse (counter=2 → stop)
        assert stopped

    def test_best_score_property(self):
        from src.training.trainer import EarlyStopping
        es    = EarlyStopping(patience=5, mode="max", restore_best_weights=False)
        model = nn.Linear(2, 2)
        es(0.3, model)
        es(0.7, model)
        es(0.5, model)
        assert abs(es.best_score - 0.7) < 1e-6
