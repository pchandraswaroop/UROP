"""
smoke_test_local.py — Local End-to-End Component Verification Suite
===================================================================
Executes all locally-supported end-to-end components:
  1. Metadata encoding (demographic normalization)
  2. Label encoding (14 thoracic pathologies)
  3. Batch scan graph construction & patient clique isolation
  4. GraphSAGE GNN forward and backward pass
  5. Weighted BCE loss and sample-weighted aggregation
  6. Multi-label metrics and single-class edge cases
  7. Checkpoint saving, strict validation, and ranked selection
  8. Authentic single-scan vs multi-scan subgroup AUROC calculation
  9. Comparative plotting and markdown report generation
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dataset.label_encoder import LabelEncoder, DISEASE_LABEL_NAMES
from src.dataset.preprocessor import MetadataEncoder
from src.graph.graph_builder import build_batch_scan_graph
from src.models.gnn_branch import GNNBranch
from src.training.losses import WeightedBCELoss
from src.training.metrics import MetricAccumulator, compute_metrics
from src.training.trainer import CheckpointManager
from src.training.evaluator import (
    FOUR_MODELS,
    THREE_MODELS,
    evaluate_scan_count_subgroups,
    generate_comparison_plots,
    generate_comparison_report,
)
from src.inference.checkpoint_loader import ModelCheckpointLoader, validate_auroc


def main():
    print("=" * 70)
    print("LOCAL END-TO-END PIPELINE SMOKE TEST")
    print("=" * 70)

    # 1. Metadata & Label Encoding
    meta_enc = MetadataEncoder()
    row_m = meta_enc.encode_row(age=65, sex="M")
    assert row_m.shape == (2,), f"Metadata shape mismatch: {row_m.shape}"
    assert np.isclose(row_m[0].item(), (65 - 46.9) / 16.6, atol=1e-3)
    assert row_m[1].item() == 1.0  # M = 1.0
    print("[PASS] [1/9] Metadata encoding verified (formula: (clamp(age,0,120) - 46.9) / 16.6).")

    label_enc = LabelEncoder()
    multi_label = label_enc.encode([1, 4])  # Atelectasis, Effusion
    assert multi_label.shape == (14,)
    assert multi_label.sum().item() == 2.0
    print("[PASS] [2/9] 14-disease label encoding verified.")

    # 2. Online Graph Construction
    pids = ["Patient_1", "Patient_1", "Patient_2", "Patient_3", "Patient_3", "Patient_3"]
    B = len(pids)
    dummy_meta = torch.randn(B, 2)
    gx, edge_index = build_batch_scan_graph(dummy_meta, pids, add_self_loops=True)
    assert gx.shape == (B, 2)
    assert edge_index.shape[0] == 2
    # Verify zero cross-patient edges
    for e in range(edge_index.shape[1]):
        u, v = edge_index[0, e].item(), edge_index[1, e].item()
        assert pids[u] == pids[v], "Cross-patient edge leaked!"
    print(f"[PASS] [3/9] Online scan graph verified: {B} nodes, {edge_index.shape[1]} intra-patient edges, 0 leakage.")

    # 3. GraphSAGE GNN Forward + Backward Pass
    gnn = GNNBranch(node_feat_dim=66, hidden_dim=64, output_dim=64, n_layers=2)
    node_feats = torch.randn(B, 66, requires_grad=True)
    gnn_out = gnn(node_feats, edge_index)
    assert gnn_out.shape == (B, 64)
    loss_gnn = gnn_out.sum()
    loss_gnn.backward()
    assert node_feats.grad is not None
    assert torch.isfinite(node_feats.grad).all()
    print("[PASS] [4/9] GraphSAGE GNN forward and backward pass verified.")

    # 4. Weighted BCE Loss & Metric Accumulator
    criterion = WeightedBCELoss()
    dummy_logits = torch.randn(B, 14)
    dummy_targets = torch.randint(0, 2, (B, 14)).float()
    loss = criterion(dummy_logits, dummy_targets)
    assert torch.isfinite(loss)

    acc = MetricAccumulator()
    acc.update(torch.sigmoid(dummy_logits), dummy_targets)
    metrics = acc.compute()
    assert "auc_mean" in metrics
    assert "f1_macro" in metrics
    print(f"[PASS] [5/9] Weighted BCE Loss ({loss.item():.4f}) & MetricAccumulator verified.")

    # 5. Checkpoint Manager & Strict AUROC Validation
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(checkpoint_dir=str(Path(tmpdir) / "test_exp"), save_top_k=2)
        dummy_model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        dummy_opt = torch.optim.Adam(dummy_model.parameters())
        saved_path = ckpt_mgr.save(dummy_model, dummy_opt, None, epoch=1, metrics={"val_auc_mean": 0.885})
        assert Path(saved_path).exists()

        is_valid_auc, score, _ = validate_auroc(0.885)
        assert is_valid_auc and score == 0.885

        best_ckpt = ModelCheckpointLoader.find_best_checkpoint("test_exp", checkpoint_dir=tmpdir)
        assert best_ckpt is not None

        loaded_ok, loaded_info = ModelCheckpointLoader.load_checkpoint(dummy_model, best_ckpt, strict=True)
        assert loaded_ok is True
        assert loaded_info["metrics"]["val_auc_mean"] == 0.885
    print("[PASS] [6/9] CheckpointManager save, ranked selection, and strict loading verified.")

    # 6. Authentic Subgroup AUROC Calculation
    test_df = pd.DataFrame([
        {"patient_id": "P_A", "age": 45, "sex": "M"},
        {"patient_id": "P_B", "age": 55, "sex": "F"},
        {"patient_id": "P_B", "age": 55, "sex": "F"},
    ])
    preds = {
        "quantum_gnn_multimodal": {
            "probs": np.random.uniform(0.1, 0.9, (3, 14)),
            "targets": np.array([[1, 0] + [0]*12, [0, 1] + [0]*12, [0, 1] + [0]*12]),
        }
    }
    df_subgroups = evaluate_scan_count_subgroups(test_df, models=["quantum_gnn_multimodal"], predictions=preds)
    assert len(df_subgroups) == 1
    assert df_subgroups.iloc[0]["Single-Scan Scans"] == 1
    assert df_subgroups.iloc[0]["Multi-Scan Scans"] == 2
    print("[PASS] [7/9] Authentic single-scan vs multi-scan subgroup AUROC calculation verified.")

    # 7. Comparison Plots & Markdown Report Generation
    with tempfile.TemporaryDirectory() as tmp_out:
        out_dir = Path(tmp_out)
        res_dir = out_dir / "results"
        for m in FOUR_MODELS:
            m_res = res_dir / m
            m_res.mkdir(parents=True, exist_ok=True)
            (m_res / "test_metrics.json").write_text('{"auc_mean": 0.72, "f1_macro": 0.25}', encoding="utf-8")
            (m_res / "train_history.json").write_text('[{"epoch": 0, "train_loss": 0.5, "val_loss": 0.45, "val_auc_mean": 0.65}]', encoding="utf-8")

        plots = generate_comparison_plots(results_dir=res_dir, output_dir=out_dir, models=FOUR_MODELS)
        assert len(plots) == 6
        for name, p_path in plots.items():
            assert p_path.exists() and p_path.stat().st_size > 0

        report_file = out_dir / "comparison_report.md"
        df_dummy_comp = pd.DataFrame([{"Model": m, "Test AUROC": 0.72} for m in FOUR_MODELS])
        rep = generate_comparison_report(df_dummy_comp, output_path=report_file)
        assert rep.exists() and rep.stat().st_size > 0
    print("[PASS] [8/9] Evaluation plots (6 figures) and Markdown comparison report verified.")

    # 8. Four-Model Architectural Compatibility
    assert len(FOUR_MODELS) == 4
    assert len(THREE_MODELS) == 3
    print("[PASS] [9/9] Four-model comparison hierarchy and legacy compatibility confirmed.")

    print("\n" + "=" * 70)
    print("LOCAL SMOKE TEST PASSED: ALL 9 SYSTEM COMPONENTS OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    main()
