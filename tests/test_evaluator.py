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
    load_experiment_metrics,
    generate_comparison_plots,
    generate_comparison_report,
    visualize_test_predictions,
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
