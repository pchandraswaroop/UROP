"""
test_training.py — Integration tests for training infrastructure
================================================================
All tests use dummy tensors / loaders — no network calls, no HF dataset.

Test coverage
-------------
  - WeightedBCELoss: logits + targets → scalar finite loss
  - compute_metrics: random probs/targets → all keys present
  - CheckpointManager: save/load round-trip → state_dict matches
  - Trainer.train_epoch: 2 batches, batch_size=2 → metrics dict with 'loss'
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUM_CLASSES = 14
B = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dummy_logits(b: int = B, nc: int = NUM_CLASSES) -> torch.Tensor:
    return torch.randn(b, nc)


def _dummy_probs(b: int = B, nc: int = NUM_CLASSES) -> torch.Tensor:
    return torch.sigmoid(_dummy_logits(b, nc))


def _dummy_targets(b: int = B, nc: int = NUM_CLASSES) -> torch.Tensor:
    return torch.randint(0, 2, (b, nc)).float()


class _FakeLoader:
    """Minimal DataLoader stub returning fixed fake batches."""

    def __init__(self, n_batches: int = 2, batch_size: int = 2) -> None:
        self.n_batches  = n_batches
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            yield {
                "image": torch.zeros(self.batch_size, 3, 224, 224),
                "label": _dummy_targets(self.batch_size),
                "meta":  torch.zeros(self.batch_size, 2),
            }

    def __len__(self):
        return self.n_batches


class _TinyModel(nn.Module):
    """Tiny model that accepts (images, meta, return_logits=False)."""

    def __init__(self, num_classes: int = 14) -> None:
        super().__init__()
        self.head = nn.Linear(224 * 224 * 3 + 2, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        meta:   torch.Tensor,
        graph_x=None,
        graph_edge_index=None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        B = images.shape[0]
        flat   = images.view(B, -1)
        concat = torch.cat([flat, meta], dim=-1)
        logits = self.head(concat)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


# =============================================================================
# WeightedBCELoss
# =============================================================================

class TestWeightedBCELoss:
    def test_scalar_finite_loss(self):
        from src.training.losses import WeightedBCELoss
        loss_fn = WeightedBCELoss()
        logits  = _dummy_logits()
        targets = _dummy_targets()
        loss    = loss_fn(logits, targets)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_with_pos_weights(self):
        from src.training.losses import WeightedBCELoss
        weights = torch.ones(NUM_CLASSES) * 2.0
        loss_fn = WeightedBCELoss(pos_weights=weights)
        loss    = loss_fn(_dummy_logits(), _dummy_targets())
        assert torch.isfinite(loss)

    def test_loss_decreases_with_perfect_predictions(self):
        """Perfect predictions (very large positive logits for positive targets)
        should give a lower loss than random logits."""
        from src.training.losses import WeightedBCELoss
        loss_fn = WeightedBCELoss()
        targets = _dummy_targets()
        # Perfect: large positive where target=1, large negative where target=0
        perfect_logits = targets * 10 + (1 - targets) * (-10)
        random_logits  = _dummy_logits()
        assert loss_fn(perfect_logits, targets) < loss_fn(random_logits, targets)

    def test_pos_weight_cap(self):
        from src.training.losses import WeightedBCELoss
        huge_weights = torch.ones(NUM_CLASSES) * 1000.0
        loss_fn = WeightedBCELoss(pos_weights=huge_weights, pos_weight_cap=10.0)
        loss    = loss_fn(_dummy_logits(), _dummy_targets())
        assert torch.isfinite(loss)


# =============================================================================
# compute_metrics
# =============================================================================

class TestComputeMetrics:
    def _make_data(self, n: int = 100) -> tuple:
        rng     = np.random.default_rng(0)
        probs   = rng.uniform(0, 1, (n, NUM_CLASSES)).astype(np.float32)
        targets = rng.integers(0, 2, (n, NUM_CLASSES)).astype(np.float32)
        # Ensure at least one positive per class so AUC is computable
        targets[0, :] = 1.0
        targets[1, :] = 0.0
        return probs, targets

    def test_all_keys_present(self):
        from src.training.metrics import compute_metrics
        probs, targets = self._make_data()
        m = compute_metrics(probs, targets)
        required = ["auc_mean", "ap_mean", "f1_macro", "f1_micro",
                    "n_valid_auc", "threshold"]
        for key in required:
            assert key in m, f"Missing key: {key}"

    def test_per_class_keys_present(self):
        from src.training.metrics import compute_metrics
        from src.dataset.label_encoder import DISEASE_LABEL_NAMES
        probs, targets = self._make_data()
        m = compute_metrics(probs, targets)
        for name in DISEASE_LABEL_NAMES:
            for suffix in ("/auc", "/ap", "/f1", "/precision", "/recall"):
                assert (name + suffix) in m, f"Missing key: {name}{suffix}"

    def test_auc_mean_in_valid_range(self):
        from src.training.metrics import compute_metrics
        probs, targets = self._make_data()
        m = compute_metrics(probs, targets)
        auc = m["auc_mean"]
        if not np.isnan(auc):
            assert 0.0 <= auc <= 1.0, f"AUC out of range: {auc}"

    def test_f1_macro_in_valid_range(self):
        from src.training.metrics import compute_metrics
        probs, targets = self._make_data()
        m = compute_metrics(probs, targets)
        assert 0.0 <= m["f1_macro"] <= 1.0

    def test_all_negative_class_handled(self):
        """Classes with zero positives should produce NaN AUC without crashing."""
        from src.training.metrics import compute_metrics
        probs   = np.random.rand(50, NUM_CLASSES).astype(np.float32)
        targets = np.zeros((50, NUM_CLASSES), dtype=np.float32)
        m = compute_metrics(probs, targets)  # should not raise
        assert np.isnan(m["auc_mean"]) or m["n_valid_auc"] == 0


# =============================================================================
# CheckpointManager — save / load round-trip
# =============================================================================

class TestCheckpointManager:
    def test_save_load_roundtrip(self):
        from src.training.trainer import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr   = CheckpointManager(checkpoint_dir=tmpdir, save_top_k=5)
            model = nn.Linear(8, 4)
            opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

            # Save
            metrics = {"val_auc_mean": 0.75}
            path    = mgr.save(model, opt, sched, epoch=1, metrics=metrics)
            assert path is not None

            # Modify weights
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(0.0)

            # Load
            epoch_loaded = CheckpointManager.load(path, model, opt, sched, device="cpu")
            assert epoch_loaded == 1

            # Weights should be restored (non-zero)
            for p in model.parameters():
                assert p.abs().sum().item() != 0.0

    def test_top_k_pruning(self):
        from src.training.trainer import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr   = CheckpointManager(checkpoint_dir=tmpdir, save_top_k=2)
            model = nn.Linear(2, 2)
            opt   = torch.optim.SGD(model.parameters(), lr=1e-3)

            for i, score in enumerate([0.5, 0.6, 0.7, 0.8]):
                mgr.save(model, opt, None, epoch=i,
                         metrics={"val_auc_mean": score})

            # Only top-2 checkpoints should remain
            ckpts = list(Path(tmpdir).glob("*.pt"))
            assert len(ckpts) <= 2, f"Expected ≤2 checkpoints, found {len(ckpts)}"

    def test_load_state_dict_matches(self):
        from src.training.trainer import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr    = CheckpointManager(checkpoint_dir=tmpdir, save_top_k=1)
            model  = nn.Linear(4, 4)
            opt    = torch.optim.Adam(model.parameters())
            before = {k: v.clone() for k, v in model.state_dict().items()}

            path = mgr.save(model, opt, None, epoch=0,
                            metrics={"val_auc_mean": 0.6})

            # Zero out model
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()

            CheckpointManager.load(path, model, device="cpu")
            after = model.state_dict()

            for k in before:
                assert torch.allclose(before[k].float(), after[k].float()), (
                    f"Parameter '{k}' mismatch after checkpoint load"
                )


# =============================================================================
# Trainer.train_epoch
# =============================================================================

class TestTrainerTrainEpoch:
    def _make_trainer(self):
        from src.training.trainer import Trainer
        from src.training.losses import WeightedBCELoss

        model     = _TinyModel()
        criterion = WeightedBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device="cpu",
        )

    def test_returns_metrics_dict_with_loss(self):
        trainer = self._make_trainer()
        loader  = _FakeLoader(n_batches=2, batch_size=2)
        metrics = trainer.train_epoch(loader, epoch=0)
        assert "loss" in metrics, "train_epoch should return a dict with 'loss'"
        assert torch.isfinite(torch.tensor(metrics["loss"])), (
            f"train loss not finite: {metrics['loss']}"
        )

    def test_returns_epoch_key(self):
        trainer = self._make_trainer()
        loader  = _FakeLoader(n_batches=2, batch_size=2)
        metrics = trainer.train_epoch(loader, epoch=3)
        assert metrics.get("epoch") == 3

    def test_loss_is_scalar(self):
        trainer = self._make_trainer()
        loader  = _FakeLoader(n_batches=2, batch_size=2)
        metrics = trainer.train_epoch(loader, epoch=0)
        loss = metrics["loss"]
        assert isinstance(loss, float), f"loss should be float, got {type(loss)}"

    def test_eval_epoch_returns_loss(self):
        trainer = self._make_trainer()
        loader  = _FakeLoader(n_batches=2, batch_size=2)
        metrics = trainer.eval_epoch(loader, split="val")
        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_fit_returns_history(self):
        from src.training.trainer import Trainer
        from src.training.losses import WeightedBCELoss

        model     = _TinyModel()
        criterion = WeightedBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer   = Trainer(model=model, criterion=criterion,
                            optimizer=optimizer, scheduler=None, device="cpu")
        loader  = _FakeLoader(n_batches=1, batch_size=2)
        history = trainer.fit(loader, loader, max_epochs=2)
        assert len(history) == 2
        for entry in history:
            assert "train_loss" in entry
            assert "val_loss"   in entry

    def test_sample_weighted_loss_aggregation_unequal_batches(self):
        """Verify that epoch loss is weighted by the number of samples in each batch,
        rather than simple unweighted mean of batch losses."""
        trainer = self._make_trainer()

        # Batch 1 has 10 samples, Batch 2 has 2 samples
        b1_imgs = torch.zeros(10, 3, 224, 224)
        b1_meta = torch.zeros(10, 2)
        b1_lbls = torch.zeros(10, NUM_CLASSES)

        b2_imgs = torch.ones(2, 3, 224, 224)
        b2_meta = torch.ones(2, 2)
        b2_lbls = torch.ones(2, NUM_CLASSES)

        class _UnevenLoader:
            def __iter__(self):
                yield {"image": b1_imgs, "meta": b1_meta, "label": b1_lbls}
                yield {"image": b2_imgs, "meta": b2_meta, "label": b2_lbls}
            def __len__(self):
                return 2

        with torch.no_grad():
            out1 = trainer._model_forward(b1_imgs, b1_meta, return_logits=True)
            loss1 = trainer.criterion(out1, b1_lbls).item()
            out2 = trainer._model_forward(b2_imgs, b2_meta, return_logits=True)
            loss2 = trainer.criterion(out2, b2_lbls).item()

        expected_weighted_loss = (10 * loss1 + 2 * loss2) / 12.0
        unweighted_loss = (loss1 + loss2) / 2.0

        eval_metrics = trainer.eval_epoch(_UnevenLoader(), split="val")
        assert np.isclose(eval_metrics["loss"], expected_weighted_loss, atol=1e-5), (
            f"Expected sample-weighted loss {expected_weighted_loss:.6f}, got {eval_metrics['loss']:.6f}"
        )
        assert not np.isclose(eval_metrics["loss"], unweighted_loss, atol=1e-3), (
            f"Loss was improperly calculated using unweighted average"
        )
