import pytest
import pandas as pd
import numpy as np
import torch

from src.training.execution_modes import (
    ExecutionMode,
    ModeConfig,
    get_mode_config,
    sample_patient_scans,
    select_patient_subset,
    verify_zero_patient_leakage,
    prepare_mode_datasets,
    get_hardware_info,
)
from src.graph.graph_builder import build_batch_scan_graph


@pytest.fixture
def synthetic_splits():
    # 30 patients, each with 1 to 6 scans
    # 20 train patients, 5 val patients, 5 test patients (strictly separated)
    train_rows = []
    for pid in range(1, 21):
        n_scans = (pid % 6) + 1
        for s in range(n_scans):
            train_rows.append({
                "patient_id": f"P{pid:03d}",
                "scan_id": s,
                "age": 45,
                "sex": "M",
                "hf_index": len(train_rows),
                **{f"disease_{i}": 1 if (pid + s + i) % 7 == 0 else 0 for i in range(14)}
            })
    train_df = pd.DataFrame(train_rows)

    val_rows = []
    for pid in range(21, 26):
        n_scans = (pid % 5) + 1
        for s in range(n_scans):
            val_rows.append({
                "patient_id": f"P{pid:03d}",
                "scan_id": s,
                "age": 50,
                "sex": "F",
                "hf_index": len(val_rows) + 100,
                **{f"disease_{i}": 1 if (pid + s + i) % 7 == 0 else 0 for i in range(14)}
            })
    val_df = pd.DataFrame(val_rows)

    test_rows = []
    for pid in range(26, 31):
        n_scans = (pid % 5) + 1
        for s in range(n_scans):
            test_rows.append({
                "patient_id": f"P{pid:03d}",
                "scan_id": s,
                "age": 55,
                "sex": "M",
                "hf_index": len(test_rows) + 200,
                **{f"disease_{i}": 1 if (pid + s + i) % 7 == 0 else 0 for i in range(14)}
            })
    test_df = pd.DataFrame(test_rows)

    return train_df, val_df, test_df


@pytest.fixture
def large_cohort_df():
    """Create a cohort with both small (<50) and large (>50) scan count patients."""
    rows = []
    # Patient 1: 5 scans
    for s in range(5):
        rows.append({"patient_id": "P_SMALL_1", "scan_id": s, "age": 30, "sex": "M", "hf_index": s, **{f"d_{i}": 0 for i in range(14)}})
    # Patient 2: exactly 50 scans
    for s in range(50):
        rows.append({"patient_id": "P_EXACT_50", "scan_id": s, "age": 40, "sex": "F", "hf_index": 100 + s, **{f"d_{i}": 1 if i == 0 else 0 for i in range(14)}})
    # Patient 3: 75 scans
    for s in range(75):
        rows.append({"patient_id": "P_LARGE_75", "scan_id": s, "age": 60, "sex": "M", "hf_index": 200 + s, **{f"d_{i}": 1 if s == 74 and i == 1 else 0 for i in range(14)}})
    # Patient 4: 184 scans (matching NIH max patient)
    for s in range(184):
        rows.append({"patient_id": "P_MAX_184", "scan_id": s, "age": 70, "sex": "F", "hf_index": 400 + s, **{f"d_{i}": 1 if s == 183 and i == 2 else 0 for i in range(14)}})
    return pd.DataFrame(rows)


def test_mode_config_defaults():
    smoke = get_mode_config("smoke")
    assert smoke.mode == "smoke"
    assert smoke.target_train_scans == 600
    assert smoke.target_val_scans == 200
    assert smoke.max_epochs == 2
    assert smoke.max_scans_per_patient == 2
    assert smoke.is_preliminary is True
    assert smoke.checkpoint_dir == "checkpoints/smoke"
    assert smoke.results_dir == "results/smoke"
    assert smoke.outputs_dir == "outputs/smoke/model_comparison"

    med = get_mode_config("medium")
    assert med.mode == "medium"
    assert med.target_train_scans == 4000
    assert med.max_epochs == 4
    assert med.max_scans_per_patient == 5
    assert med.is_preliminary is True
    assert med.checkpoint_dir == "checkpoints/medium"
    assert med.results_dir == "results/medium"

    full = get_mode_config("full")
    assert full.mode == "full"
    assert full.target_train_scans is None
    assert full.max_scans_per_patient is None
    assert full.max_epochs == 15
    assert full.early_stopping_patience == 4
    assert full.is_preliminary is False
    assert full.checkpoint_dir == "checkpoints/full"
    assert full.results_dir == "results/full"

    full_50 = get_mode_config("full_50")
    assert full_50.mode == "full_50"
    assert full_50.target_train_scans is None
    assert full_50.max_scans_per_patient == 50
    assert full_50.max_epochs == 15
    assert full_50.early_stopping_patience == 4
    assert full_50.is_preliminary is False
    assert full_50.checkpoint_dir == "checkpoints/full_50"
    assert full_50.results_dir == "results/full_50"
    assert full_50.outputs_dir == "outputs/full_50/model_comparison"


def test_full_50_retains_every_patient(large_cohort_df):
    """Verify that under full_50, 100% of unique patients are retained (never excluded)."""
    cfg = get_mode_config("full_50")
    subset = select_patient_subset(large_cohort_df, max_scans_per_patient=cfg.max_scans_per_patient)
    assert set(subset["patient_id"]) == set(large_cohort_df["patient_id"])
    assert subset["patient_id"].nunique() == 4


def test_full_50_patients_with_50_or_fewer_scans_retain_all_scans(large_cohort_df):
    """Verify patients with <= 50 scans retain all scans without alteration."""
    cfg = get_mode_config("full_50")
    subset = select_patient_subset(large_cohort_df, max_scans_per_patient=cfg.max_scans_per_patient)

    # P_SMALL_1 had 5 scans -> retains exactly 5
    p1 = subset[subset["patient_id"] == "P_SMALL_1"]
    assert len(p1) == 5
    assert p1["scan_id"].tolist() == list(range(5))

    # P_EXACT_50 had 50 scans -> retains exactly 50
    p2 = subset[subset["patient_id"] == "P_EXACT_50"]
    assert len(p2) == 50
    assert p2["scan_id"].tolist() == list(range(50))


def test_full_50_patients_with_over_50_scans_retain_exactly_50(large_cohort_df):
    """Verify patients with > 50 scans are capped at exactly 50 scans."""
    cfg = get_mode_config("full_50")
    subset = select_patient_subset(large_cohort_df, max_scans_per_patient=cfg.max_scans_per_patient)

    # P_LARGE_75 had 75 scans -> retains exactly 50
    p3 = subset[subset["patient_id"] == "P_LARGE_75"]
    assert len(p3) == 50

    # P_MAX_184 had 184 scans -> retains exactly 50
    p4 = subset[subset["patient_id"] == "P_MAX_184"]
    assert len(p4) == 50


def test_full_50_preserves_first_and_last_scans(large_cohort_df):
    """Verify that for patients with > 50 scans, the first and last scans are strictly preserved."""
    cfg = get_mode_config("full_50")
    subset = select_patient_subset(large_cohort_df, max_scans_per_patient=cfg.max_scans_per_patient)

    p3 = subset[subset["patient_id"] == "P_LARGE_75"]
    assert p3.iloc[0]["scan_id"] == 0
    assert p3.iloc[-1]["scan_id"] == 74

    p4 = subset[subset["patient_id"] == "P_MAX_184"]
    assert p4.iloc[0]["scan_id"] == 0
    assert p4.iloc[-1]["scan_id"] == 183
    # Check that late-emerging disease label at scan 183 is preserved
    assert p4.iloc[-1]["d_2"] == 1


def test_sampling_is_deterministic(large_cohort_df):
    """Verify that uniform sampling produces identical results across multiple runs."""
    cfg = get_mode_config("full_50")
    run1 = select_patient_subset(large_cohort_df, max_scans_per_patient=cfg.max_scans_per_patient)
    run2 = select_patient_subset(large_cohort_df, max_scans_per_patient=cfg.max_scans_per_patient)
    pd.testing.assert_frame_equal(run1, run2)


def test_selected_scans_distributed_across_sequence(large_cohort_df):
    """Verify intermediate scans are strictly increasing and distributed across the timeline."""
    p_rows = large_cohort_df[large_cohort_df["patient_id"] == "P_MAX_184"]
    sampled = sample_patient_scans(p_rows, max_scans=50)

    scan_ids = sampled["scan_id"].values
    assert len(scan_ids) == 50
    # Strictly increasing
    assert (np.diff(scan_ids) > 0).all()
    # Step sizes should be balanced (either 3 or 4 for 184 -> 50)
    diffs = np.diff(scan_ids)
    assert set(diffs).issubset({3, 4})
    # Intermediate scans exist between extremes
    assert scan_ids[1] > 0
    assert scan_ids[-2] < 183


def test_preservation_of_all_metadata_and_labels(large_cohort_df):
    """Verify patient_id, scan_id, age, sex, and all 14 labels are preserved without corruption."""
    cfg = get_mode_config("full_50")
    subset = select_patient_subset(large_cohort_df, max_scans_per_patient=cfg.max_scans_per_patient)

    for col in ["patient_id", "scan_id", "age", "sex"] + [f"d_{i}" for i in range(14)]:
        assert col in subset.columns

    p4 = subset[subset["patient_id"] == "P_MAX_184"]
    assert (p4["age"] == 70).all()
    assert (p4["sex"] == "F").all()


def test_zero_patient_leakage_in_full_50(synthetic_splits):
    """Verify zero patient leakage occurs across splits under full_50 mode."""
    train_df, val_df, test_df = synthetic_splits
    cfg = get_mode_config("full_50")

    s_train, s_val, s_test = prepare_mode_datasets(train_df, val_df, test_df, cfg)
    is_clean, msg = verify_zero_patient_leakage(s_train, s_val, s_test)
    assert is_clean is True
    assert "Zero Patient Leakage Confirmed" in msg


def test_existing_graph_construction_remains_unchanged():
    """Verify patient scan graph construction remains complete directed with self-loops (K^2 edges)."""
    # 2 patients: Patient A has 3 scans, Patient B has 2 scans
    patient_ids = ["P_A", "P_A", "P_A", "P_B", "P_B"]
    meta = torch.tensor([
        [0.45, 1.0],
        [0.45, 1.0],
        [0.45, 1.0],
        [0.60, 0.0],
        [0.60, 0.0],
    ], dtype=torch.float32)

    x, edge_index = build_batch_scan_graph(meta, patient_ids, add_self_loops=True)

    assert x.shape == (5, 2)
    # Patient A (K=3) -> 3^2 = 9 edges
    # Patient B (K=2) -> 2^2 = 4 edges
    # Total edges = 9 + 4 = 13 edges
    assert edge_index.shape[1] == 13

    # Verify no cross-patient edges:
    # Nodes 0, 1, 2 belong to P_A; Nodes 3, 4 belong to P_B
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        if s in {0, 1, 2}:
            assert d in {0, 1, 2}, f"Cross-patient edge found: {s} -> {d}"
        if s in {3, 4}:
            assert d in {3, 4}, f"Cross-patient edge found: {s} -> {d}"

    # Verify all self-loops exist
    for i in range(5):
        has_self_loop = any(s == i and d == i for s, d in zip(src, dst))
        assert has_self_loop, f"Missing self-loop for node {i}"


def test_select_patient_subset_caps_scans_per_patient(synthetic_splits):
    train_df, _, _ = synthetic_splits
    subset = select_patient_subset(train_df, target_scans=20, max_scans_per_patient=2, seed=42)

    assert len(subset) >= 20
    p_counts = subset["patient_id"].value_counts()
    assert (p_counts <= 2).all(), f"Found patient with >2 scans: {p_counts.max()}"

    for pid in subset["patient_id"].unique():
        p_scans = subset[subset["patient_id"] == pid]["scan_id"].tolist()
        assert p_scans == sorted(p_scans)
        assert len(p_scans) <= 2


def test_patients_with_many_scans_not_excluded(synthetic_splits):
    train_df, _, _ = synthetic_splits
    top_pid = train_df["patient_id"].value_counts().index[0]

    subset = select_patient_subset(train_df, target_scans=len(train_df), max_scans_per_patient=2, seed=42)
    assert top_pid in subset["patient_id"].values
    assert len(subset[subset["patient_id"] == top_pid]) == 2


def test_zero_patient_leakage_detection(synthetic_splits):
    train_df, val_df, test_df = synthetic_splits

    is_clean, msg = verify_zero_patient_leakage(train_df, val_df, test_df)
    assert is_clean is True
    assert "Zero Patient Leakage Confirmed" in msg

    leaky_val = val_df.copy()
    leaky_val = pd.concat([leaky_val, train_df.iloc[:2]], ignore_index=True)
    is_clean, msg = verify_zero_patient_leakage(train_df, leaky_val, test_df)
    assert is_clean is False
    assert "CRITICAL PATIENT LEAKAGE DETECTED" in msg


def test_prepare_mode_datasets_smoke_and_medium(synthetic_splits):
    train_df, val_df, test_df = synthetic_splits

    smoke_cfg = get_mode_config("smoke")
    smoke_cfg.target_train_scans = 15
    smoke_cfg.target_val_scans = 5
    smoke_cfg.target_test_scans = 5

    s_train, s_val, s_test = prepare_mode_datasets(train_df, val_df, test_df, smoke_cfg, seed=42)
    assert len(s_train) >= 15
    assert len(s_val) >= 5
    assert len(s_test) >= 5
    assert s_train["patient_id"].value_counts().max() <= 2
    assert s_val["patient_id"].value_counts().max() <= 2
    assert s_test["patient_id"].value_counts().max() <= 2

    is_clean, _ = verify_zero_patient_leakage(s_train, s_val, s_test)
    assert is_clean is True


def test_full_mode_returns_exact_splits(synthetic_splits):
    train_df, val_df, test_df = synthetic_splits
    full_cfg = get_mode_config("full")

    f_train, f_val, f_test = prepare_mode_datasets(train_df, val_df, test_df, full_cfg)
    assert len(f_train) == len(train_df)
    assert len(f_val) == len(val_df)
    assert len(f_test) == len(test_df)
    assert set(f_train["patient_id"]) == set(train_df["patient_id"])


def test_get_hardware_info():
    info = get_hardware_info()
    assert "device" in info
    assert "cuda_available" in info
    assert "pennylane_installed" in info
