"""
test_evaluator.py — Unit Tests for 3-Model Evaluator, Plots & Visualizer
========================================================================
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.training.evaluator import (
    THREE_MODELS,
    FOUR_MODELS,
    load_experiment_metrics,
    generate_comparison_plots,
    generate_comparison_report,
    visualize_test_predictions,
    evaluate_scan_count_subgroups,
)
from src.models.model_factory import build_model


@pytest.fixture
def mock_results_dir(tmp_path):
    """Create a temporary results directory with realistic evaluation metrics."""
    res_dir = tmp_path / "results"
    res_dir.mkdir()

    for m in THREE_MODELS:
        m_dir = res_dir / m
        m_dir.mkdir()

        val_metrics = {
            "auc_mean": 0.65,
            "f1_macro": 0.20,
            "best_epoch": 2,
            "training_time_seconds": 125.4,
        }
        test_metrics = {
            "auc_mean": 0.64,
            "f1_macro": 0.19,
            "f1_micro": 0.25,
            "precision_macro": 0.22,
            "recall_macro": 0.18,
            "Atelectasis/auc": 0.68,
            "Cardiomegaly/auc": 0.72,
        }
        train_hist = [
            {"epoch": 0, "train_loss": 0.45, "val_loss": 0.40, "val_auc_mean": 0.55},
            {"epoch": 1, "train_loss": 0.38, "val_loss": 0.35, "val_auc_mean": 0.60},
            {"epoch": 2, "train_loss": 0.32, "val_loss": 0.31, "val_auc_mean": 0.65},
        ]

        (m_dir / "val_metrics.json").write_text(json.dumps(val_metrics))
        (m_dir / "test_metrics.json").write_text(json.dumps(test_metrics))
        (m_dir / "train_history.json").write_text(json.dumps(train_hist))

    return res_dir


def test_load_experiment_metrics(mock_results_dir):
    df = load_experiment_metrics(results_dir=mock_results_dir, models=THREE_MODELS)
    assert len(df) == 3
    expected_cols = [
        "Model", "Best Epoch", "Validation AUROC", "Test AUROC",
        "Macro F1", "Micro F1", "Training Time"
    ]
    for col in expected_cols:
        assert col in df.columns
    assert df.loc[0, "Best Epoch"] == 3  # 0-indexed 2 -> 1-indexed 3


def test_generate_comparison_plots(mock_results_dir, tmp_path):
    out_dir = tmp_path / "outputs"
    plots = generate_comparison_plots(results_dir=mock_results_dir, output_dir=out_dir, models=THREE_MODELS)

    assert "training_curves" in plots
    assert "validation_auc" in plots
    assert "test_auc_comparison" in plots
    assert "macro_f1_comparison" in plots
    assert "disease_wise_auc" in plots
    assert "precision_recall_comparison" in plots

    for name, p in plots.items():
        assert p.exists()
        assert p.stat().st_size > 0


def test_generate_comparison_report(mock_results_dir, tmp_path):
    df = load_experiment_metrics(results_dir=mock_results_dir, models=THREE_MODELS)
    out_file = tmp_path / "comparison_report.md"
    res = generate_comparison_report(df, out_file)
    assert res.exists()
    content = res.read_text(encoding="utf-8")
    assert "Three-Model Comparative Evaluation Report" in content
    assert "cnn_only" in content


def test_visualize_test_predictions_dummy(tmp_path):
    pytest.importorskip('timm')
    # Create small test df
    test_df = pd.DataFrame([
        {"image_index": 0, "patient_id": 101, "age": 45, "sex": "M", "labels": "[1, 3]"},
        {"image_index": 1, "patient_id": 102, "age": 60, "sex": "F", "labels": "[]"},
    ])

    models_dict = {
        "cnn_only": build_model("cnn_only", pretrained=False),
        "classical_multimodal": build_model("classical_multimodal", pretrained=False),
        "quantum_gnn_multimodal": build_model("quantum_gnn_multimodal", pretrained=False),
    }

    out_dir = tmp_path / "prediction_examples"
    saved = visualize_test_predictions(
        test_df=test_df,
        hf_dataset=None,
        models_dict=models_dict,
        output_dir=out_dir,
        device="cpu",
        n_examples=2,
    )
    assert len(saved) == 2
    for p in saved:
        assert p.exists()
        assert p.stat().st_size > 0


def test_evaluate_scan_count_subgroups(mock_results_dir):
    test_df = pd.DataFrame([
        {"image_index": 0, "patient_id": 101, "age": 45, "sex": "M", "labels": "[1, 3]"},
        {"image_index": 1, "patient_id": 101, "age": 45, "sex": "M", "labels": "[1]"},
        {"image_index": 2, "patient_id": 102, "age": 60, "sex": "F", "labels": "[]"},
    ])
    df = evaluate_scan_count_subgroups(test_df, results_dir=mock_results_dir, models=THREE_MODELS)
    assert len(df) == 3
    assert "Single-Scan Scans" in df.columns
    assert "Multi-Scan Scans" in df.columns
    assert df.loc[0, "Single-Scan Scans"] == 1
    assert df.loc[0, "Multi-Scan Scans"] == 2


def test_four_models_constant():
    """Verify FOUR_MODELS definition and compatibility."""
    assert len(FOUR_MODELS) == 4
    assert FOUR_MODELS == ["cnn_only", "classical_multimodal", "cnn_gnn_multimodal", "quantum_gnn_multimodal"]
    assert "cnn_gnn_multimodal" not in THREE_MODELS
    assert len(THREE_MODELS) == 3


def test_authentic_subgroup_auroc_with_predictions(tmp_path):
    """Verify that single-scan and multi-scan AUROCs are independently computed from authentic predictions."""
    # Cohort:
    # Patient 1: 1 scan (Single)
    # Patient 2: 1 scan (Single)
    # Patient 3: 2 scans (Multi)
    # Patient 4: 2 scans (Multi)
    # Total 6 scans (2 Single, 4 Multi)
    test_df = pd.DataFrame([
        {"patient_id": "P1", "age": 40, "sex": "M"},
        {"patient_id": "P2", "age": 50, "sex": "F"},
        {"patient_id": "P3", "age": 60, "sex": "M"},
        {"patient_id": "P3", "age": 60, "sex": "M"},
        {"patient_id": "P4", "age": 70, "sex": "F"},
        {"patient_id": "P4", "age": 70, "sex": "F"},
    ])

    # Targets: 14 classes
    targets = np.zeros((6, 14), dtype=np.float32)
    # Class 0: Single-scan (idx 0 is 1, idx 1 is 0). Multi-scan (idx 2,3 is 1, idx 4,5 is 0)
    targets[0, 0] = 1.0
    targets[2, 0] = 1.0
    targets[3, 0] = 1.0

    # Predictions:
    # For single-scan: perfect discrimination on class 0 (P1=0.9, P2=0.1 -> AUC=1.0)
    # For multi-scan: inverted discrimination on class 0 (P3=0.1, P3=0.1, P4=0.9, P4=0.9 -> AUC=0.0)
    probs = np.full((6, 14), 0.5, dtype=np.float32)
    probs[0, 0] = 0.9
    probs[1, 0] = 0.1
    probs[2, 0] = 0.1
    probs[3, 0] = 0.1
    probs[4, 0] = 0.9
    probs[5, 0] = 0.9

    predictions = {
        "quantum_gnn_multimodal": {
            "probs": probs,
            "targets": targets,
        }
    }

    df = evaluate_scan_count_subgroups(
        test_df,
        results_dir=tmp_path,
        models=["quantum_gnn_multimodal"],
        predictions=predictions,
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["Single-Scan Scans"] == 2
    assert row["Multi-Scan Scans"] == 4
    # Single-scan AUROC is 1.0, Multi-scan AUROC is 0.0
    assert row["Single-Scan AUROC"] == 1.0
    assert row["Multi-Scan AUROC"] == 0.0
    assert row["Single-Scan AUROC"] != row["Multi-Scan AUROC"], "Subgroup AUROCs must be independently computed!"


def test_subgroup_auroc_single_class_edge_case(tmp_path):
    """Verify that when a disease class has zero positive labels in a subgroup,
    it is assigned NaN, excluded from the macro mean, and counted in valid classes."""
    test_df = pd.DataFrame([
        {"patient_id": "P1", "age": 40, "sex": "M"},
        {"patient_id": "P2", "age": 50, "sex": "F"},
        {"patient_id": "P3", "age": 60, "sex": "M"},
        {"patient_id": "P3", "age": 60, "sex": "M"},
    ])

    targets = np.zeros((4, 14), dtype=np.float32)
    # Class 0: positive in single (idx 0), negative in single (idx 1)
    targets[0, 0] = 1.0
    # Class 1: all zeros in single (idx 0, 1), but positive in multi (idx 2)
    targets[2, 1] = 1.0

    probs = np.full((4, 14), 0.5, dtype=np.float32)
    probs[0, 0] = 0.8
    probs[1, 0] = 0.2
    probs[2, 1] = 0.8
    probs[3, 1] = 0.2

    # Save to test_predictions.npz
    m_dir = tmp_path / "classical_multimodal"
    m_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(m_dir / "test_predictions.npz", probs=probs, targets=targets)

    df = evaluate_scan_count_subgroups(
        test_df,
        results_dir=tmp_path,
        models=["classical_multimodal"],
    )

    assert len(df) == 1
    row = df.iloc[0]
    # For single-scan: only class 0 has both classes (1 valid class)
    assert row["Single-Scan Valid Classes"] == 1
    assert row["Single-Scan AUROC"] == 1.0
