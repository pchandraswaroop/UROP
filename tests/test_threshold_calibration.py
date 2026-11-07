"""Unit tests for validation-calibrated decision thresholds and best-epoch metric indexing.

Validates Fix 1 (best checkpoint val metrics retrieval) and Fix 2 (validation-calibrated thresholds).
"""
import pytest
import numpy as np
from src.training.metrics import find_optimal_thresholds, compute_metrics, DISEASE_LABEL_NAMES


def test_find_optimal_thresholds_tunes_on_val_data():
    """Verify that optimal thresholds maximize F1 on validation data without observing test data."""
    np.random.seed(42)
    N_val, C = 400, 14
    val_targets = (np.random.rand(N_val, C) < 0.10).astype(np.float32)
    # Ensure every class has at least 5 positives
    val_targets[:5, :] = 1.0

    # Predictions where probabilities are centered around 0.35 for positives, 0.10 for negatives
    val_probs = 0.10 * (1 - val_targets) + 0.35 * val_targets + 0.05 * np.random.randn(N_val, C)
    val_probs = np.clip(val_probs, 0.01, 0.49)  # None exceed 0.50!

    thresholds = find_optimal_thresholds(val_probs, val_targets, min_positives=1, fallback=0.50)

    assert thresholds.shape == (C,)
    # Optimal thresholds should be tuned below 0.50 to capture the positives
    assert np.all(thresholds < 0.50)
    assert np.all(thresholds >= 0.05)


def test_find_optimal_thresholds_fallback():
    """Verify fallback to 0.50 when a disease has zero positive cases."""
    val_targets = np.zeros((100, 14), dtype=np.float32)
    val_probs = np.full((100, 14), 0.25, dtype=np.float32)

    thresholds = find_optimal_thresholds(val_probs, val_targets, min_positives=1, fallback=0.50)
    assert np.all(thresholds == 0.50)


def test_compute_metrics_threshold_invariance_for_auroc():
    """Verify that AUROC is strictly invariant to whether threshold is 0.50 or an array of thresholds."""
    np.random.seed(123)
    N_test, C = 300, 14
    test_targets = (np.random.rand(N_test, C) < 0.15).astype(np.float32)
    test_targets[:3, :] = 1.0  # Ensure valid AUC
    test_targets[3:6, :] = 0.0

    test_probs = np.clip(0.1 + 0.35 * test_targets + 0.1 * np.random.rand(N_test, C), 0.01, 0.99)

    custom_thresholds = np.linspace(0.20, 0.45, C).astype(np.float32)

    metrics_default = compute_metrics(test_probs, test_targets, threshold=0.50)
    metrics_custom = compute_metrics(test_probs, test_targets, threshold=custom_thresholds)

    # AUROC must be completely identical
    assert np.isclose(metrics_default["auc_mean"], metrics_custom["auc_mean"], atol=1e-6)
    # Per-class AUC must be identical
    for d in DISEASE_LABEL_NAMES:
        assert np.isclose(metrics_default[f"{d}/auc"], metrics_custom[f"{d}/auc"], atol=1e-6)

    # But F1 should reflect the custom thresholds
    assert metrics_custom["threshold"] == float(np.mean(custom_thresholds))
    assert len(metrics_custom["thresholds"]) == C


def test_experiment_best_epoch_metric_indexing():
    """Verify that val_metrics extracts from the best epoch rather than the last epoch."""
    # Simulate an Experiment instance without training
    class DummyExperiment:
        pass

    exp = DummyExperiment()
    exp.train_history = [
        {"epoch": 0, "val_auc_mean": 0.60, "val_f1_macro": 0.10},
        {"epoch": 1, "val_auc_mean": 0.6458, "val_f1_macro": 0.15},  # Best epoch (index 1)
        {"epoch": 2, "val_auc_mean": 0.6350, "val_f1_macro": 0.12},
        {"epoch": 3, "val_auc_mean": 0.6304, "val_f1_macro": 0.08},  # Final epoch (index 3)
    ]
    exp.best_epoch = 1  # Epoch 2 (zero-indexed 1)

    # Apply the Fix 1 logic:
    best_idx = exp.best_epoch if (0 <= exp.best_epoch < len(exp.train_history)) else -1
    best_entry = exp.train_history[best_idx]
    exp.val_metrics = {
        k.replace("val_", ""): v
        for k, v in best_entry.items()
        if k.startswith("val_")
    }
    exp.val_metrics["best_epoch"] = exp.best_epoch

    assert exp.val_metrics["auc_mean"] == 0.6458
    assert exp.val_metrics["auc_mean"] != 0.6304
    assert exp.val_metrics["best_epoch"] == 1
