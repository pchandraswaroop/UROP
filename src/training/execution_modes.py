"""
execution_modes.py — Unified Four-Mode Execution System
========================================================
Supports:
  1. SMOKE:   ~600 train / ~200 val / ~200 test scans (2 epochs).
              Maximum 2 scans per patient (max_scans_per_patient=2).
              Rapid pipeline verification on laptop/Colab. Clearly labeled preliminary.
  2. MEDIUM:  ~4,000 train / ~800 val / ~800 test scans (3–5 epochs).
              Maximum 5 scans per patient (max_scans_per_patient=5).
              Laptop experimentation to observe actual training dynamics and metrics.
  3. FULL:    Complete NIH ChestX-ray14 split (79,006 train / 16,460 val / 16,654 test),
              All available scans per patient (no per-patient limit, max_scans_per_patient=None).
              15 epochs with early stopping (patience 4) for official NVIDIA GPU research training.
  4. FULL_50: Full patient cohort (30,805 unique patients), maximum 50 scans per patient
              (76,532 train / 16,122 val / 16,204 test scans, 108,858 scans total)
              using deterministic uniform sampling preserving first & last scans.
              Designed for memory-constrained GPU environments (reduces peak single-patient
              graph memory by 92.6%).

Guarantees:
  - Strict patient-level grouping (companion scans of each patient kept together).
  - Deterministic subset selection and uniform scan sampling.
  - Zero patient leakage across splits (asserted).
  - Partitioned directories by mode: checkpoints/{mode}, results/{mode}, outputs/{mode}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    SMOKE = "smoke"
    MEDIUM = "medium"
    HIGH = "high"
    FULL = "full"
    FULL_50 = "full_50"


@dataclass
class ModeConfig:
    """Configuration container for execution mode parameters."""
    mode: str
    description: str
    target_train_scans: Optional[int]
    target_val_scans: Optional[int]
    target_test_scans: Optional[int]
    max_epochs: int
    early_stopping_patience: int
    checkpoint_dir: str
    results_dir: str
    outputs_dir: str
    max_scans_per_patient: Optional[int] = None
    is_preliminary: bool = True
    default_models: List[str] = field(default_factory=lambda: [
        "cnn_only",
        "classical_multimodal",
        "cnn_gnn_multimodal",
        "quantum_gnn_multimodal",
    ])


def get_mode_config(
    mode: str | ExecutionMode,
    target_train_scans: Optional[int] = None,
    target_val_scans: Optional[int] = None,
    max_epochs: Optional[int] = None,
    max_scans_per_patient: Optional[int] = None,
) -> ModeConfig:
    """Return the configuration for the requested execution mode.

    Args:
        mode: "smoke", "medium", "full", or "full_50" (case-insensitive).
        target_train_scans: Optional override for training scans.
        target_val_scans: Optional override for validation scans.
        max_epochs: Optional override for maximum epochs.
        max_scans_per_patient: Optional override for max scans per patient.

    Returns:
        ModeConfig populated with mode-specific parameters and directories.
    """
    mode_str = str(mode).lower().strip()
    if mode_str in ("smoke", "smoke_test", "test"):
        cfg = ModeConfig(
            mode="smoke",
            description="SMOKE TEST: Rapid pipeline verification (~600 train / ~200 val scans, max 2 scans/patient, 2 epochs). Preliminary results only.",
            target_train_scans=600 if target_train_scans is None else target_train_scans,
            target_val_scans=200 if target_val_scans is None else target_val_scans,
            target_test_scans=200,
            max_epochs=2 if max_epochs is None else max_epochs,
            early_stopping_patience=2,
            checkpoint_dir="checkpoints/smoke",
            results_dir="results/smoke",
            outputs_dir="outputs/smoke/model_comparison",
            max_scans_per_patient=2 if max_scans_per_patient is None else max_scans_per_patient,
            is_preliminary=True,
        )
    elif mode_str in ("medium", "med", "preliminary", "laptop"):
        train_target = 4000 if target_train_scans is None else target_train_scans
        val_target = 800 if target_val_scans is None else target_val_scans
        cfg = ModeConfig(
            mode="medium",
            description=f"MEDIUM EXPERIMENTATION: Sub-cohort training ({train_target:,} train / {val_target:,} val scans, max 5 scans/patient, 3-5 epochs) on laptop/GPU. Preliminary metrics.",
            target_train_scans=train_target,
            target_val_scans=val_target,
            target_test_scans=800,
            max_epochs=4 if max_epochs is None else max_epochs,
            early_stopping_patience=3,
            checkpoint_dir="checkpoints/medium",
            results_dir="results/medium",
            outputs_dir="outputs/medium/model_comparison",
            max_scans_per_patient=5 if max_scans_per_patient is None else max_scans_per_patient,
            is_preliminary=True,
        )
    elif mode_str in ("high", "20k"):
        train_target = 20000 if target_train_scans is None else target_train_scans
        val_target = 800 if target_val_scans is None else target_val_scans
        cfg = ModeConfig(
            mode="high",
            description=f"HIGH EXPERIMENTATION: Scaled cohort training ({train_target:,} train / {val_target:,} val / 800 test scans, max 5 scans/patient, 10 epochs) on GPU.",
            target_train_scans=train_target,
            target_val_scans=val_target,
            target_test_scans=800,
            max_epochs=10 if max_epochs is None else max_epochs,
            early_stopping_patience=3,
            checkpoint_dir="checkpoints/high",
            results_dir="results/high",
            outputs_dir="outputs/high/model_comparison",
            max_scans_per_patient=5 if max_scans_per_patient is None else max_scans_per_patient,
            is_preliminary=False,
        )
    elif mode_str in ("full", "final", "research", "gpu"):
        cfg = ModeConfig(
            mode="full",
            description="FULL RESEARCH TRAINING: Complete NIH ChestX-ray14 dataset (79,006 train / 16,460 val / 16,654 test), all scans per patient (no limit), 15 epochs with early stopping on NVIDIA GPU.",
            target_train_scans=None,
            target_val_scans=None,
            target_test_scans=None,
            max_epochs=15 if max_epochs is None else max_epochs,
            early_stopping_patience=4,
            checkpoint_dir="checkpoints/full",
            results_dir="results/full",
            outputs_dir="outputs/full/model_comparison",
            max_scans_per_patient=None,
            is_preliminary=False,
        )
    elif mode_str in ("full_50", "full50", "full-50"):
        cfg = ModeConfig(
            mode="full_50",
            description="FULL-50 RESEARCH TRAINING: Complete patient cohort (all 30,805 patients), maximum 50 scans per patient (108,858 scans total) using deterministic uniform sampling, 15 epochs with early stopping on NVIDIA GPU.",
            target_train_scans=None,
            target_val_scans=None,
            target_test_scans=None,
            max_epochs=15 if max_epochs is None else max_epochs,
            early_stopping_patience=4,
            checkpoint_dir="checkpoints/full_50",
            results_dir="results/full_50",
            outputs_dir="outputs/full_50/model_comparison",
            max_scans_per_patient=50 if max_scans_per_patient is None else max_scans_per_patient,
            is_preliminary=False,
        )
    else:
        raise ValueError(
            f"Unknown execution mode '{mode}'. Choose from: 'smoke', 'medium', 'high', 'full', 'full_50'."
        )

    return cfg


def sample_patient_scans(
    p_rows: pd.DataFrame,
    max_scans: Optional[int],
) -> pd.DataFrame:
    """Sample at most max_scans from a patient's scans using deterministic uniform sampling.

    Guarantees:
      1. Non-exclusion: All patient scans are retained if len(p_rows) <= max_scans.
      2. When len(p_rows) > max_scans:
         - The earliest scan (by scan_id or natural sequence) is preserved at index 0.
         - The latest scan is preserved at index -1.
         - Intermediate scans are sampled with uniform spacing across the remaining sequence.
         - Exactly max_scans are selected.
      3. Determinism: Strictly reproducible via deterministic equidistant spacing (no RNG).
      4. Preservation: Preserves patient_id, scan_id, image path, metadata, and all 14 disease labels.

    Args:
        p_rows: DataFrame containing all scans for a single patient.
        max_scans: Maximum number of scans to retain for this patient.

    Returns:
        DataFrame containing selected scans.
    """
    n_scans = len(p_rows)
    if max_scans is None or n_scans <= max_scans:
        return p_rows

    sort_col = "scan_id" if "scan_id" in p_rows.columns else None
    if sort_col is not None:
        p_sorted = p_rows.sort_values(sort_col).reset_index(drop=True)
    else:
        p_sorted = p_rows.reset_index(drop=True)

    if max_scans == 1:
        selected_indices = np.array([0], dtype=int)
    else:
        # np.linspace endpoints are guaranteed to be 0 and n_scans - 1.
        # Since n_scans > max_scans, the step size (n_scans - 1) / (max_scans - 1) is strictly > 1.
        # Therefore, np.round() yields strictly monotonic, unique indices.
        selected_indices = np.round(np.linspace(0, n_scans - 1, max_scans)).astype(int)

    return p_sorted.iloc[selected_indices]


def select_patient_subset(
    df: pd.DataFrame,
    target_scans: Optional[int] = None,
    max_scans_per_patient: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a deterministic subset of patients while capping scans per patient if requested.

    Guarantees:
      1. Patient grouping integrity: Scans for each patient remain strictly grouped.
      2. Non-exclusion: Patients with many scans are never excluded; they are uniformly sampled.
      3. Uniform sampling: Capped scans preserve first & last chronological scans with uniform spacing.
      4. Determinism: Fixed seed for patient selection when target_scans is specified.
      5. Zero leakage: Subsets drawn from disjoint split files preserve 0% patient overlap.

    Args:
        df: Input split DataFrame with 'patient_id' and 'scan_id'.
        target_scans: Target number of radiographs. If None, retains all patients in df.
        max_scans_per_patient: Maximum scans allowed per patient. If None, retains all scans.
        seed: Random seed for deterministic patient shuffling when target_scans is set.

    Returns:
        Filtered DataFrame containing complete or uniformly capped patient clusters.
    """
    if target_scans is None and max_scans_per_patient is None:
        return df.copy().reset_index(drop=True)

    grouped = [group for _, group in df.groupby("patient_id", sort=False)]

    if target_scans is None:
        # Full or Full-50: Retain EVERY patient in the split
        selected_dfs = [
            sample_patient_scans(grp, max_scans_per_patient)
            for grp in grouped
        ]
        subset_df = pd.concat(selected_dfs, ignore_index=True)
    else:
        # Sub-cohort modes (smoke or medium): Deterministic patient shuffling
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(len(grouped))

        selected_dfs = []
        accumulated_scans = 0
        for idx in shuffled_indices:
            p_rows = grouped[idx]
            p_sampled = sample_patient_scans(p_rows, max_scans_per_patient)
            selected_dfs.append(p_sampled)
            accumulated_scans += len(p_sampled)
            if accumulated_scans >= target_scans:
                break

        if selected_dfs:
            subset_df = pd.concat(selected_dfs, ignore_index=True)
        else:
            subset_df = df.head(0).copy()

    logger.info(
        "select_patient_subset: Target %s scans, max %s scans/pt -> Selected %d scans from %d patients",
        target_scans, max_scans_per_patient, len(subset_df), subset_df["patient_id"].nunique()
    )
    return subset_df


def verify_zero_patient_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[bool, str]:
    """Strictly assert 0% patient leakage across train, val, and test splits.

    Returns:
        Tuple (is_leakage_free: bool, message: str)
    """
    train_pids: Set[Any] = set(train_df["patient_id"])
    val_pids: Set[Any] = set(val_df["patient_id"])
    test_pids: Set[Any] = set(test_df["patient_id"])

    train_val_overlap = train_pids.intersection(val_pids)
    train_test_overlap = train_pids.intersection(test_pids)
    val_test_overlap = val_pids.intersection(test_pids)

    total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)

    if total_overlap == 0:
        msg = (
            f"Zero Patient Leakage Confirmed: {len(train_pids):,} train, "
            f"{len(val_pids):,} val, {len(test_pids):,} test unique patients. 0% overlap."
        )
        return True, msg
    else:
        err_msg = (
            f"CRITICAL PATIENT LEAKAGE DETECTED: "
            f"Train/Val overlap={len(train_val_overlap)}, "
            f"Train/Test overlap={len(train_test_overlap)}, "
            f"Val/Test overlap={len(val_test_overlap)}!"
        )
        return False, err_msg


def prepare_mode_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode_cfg: ModeConfig,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare train, validation, and test splits for the specified execution mode.

    Applies patient-level grouping, per-patient scan capping, and asserts zero patient leakage.
    """
    active_train = select_patient_subset(
        train_df,
        target_scans=mode_cfg.target_train_scans,
        max_scans_per_patient=mode_cfg.max_scans_per_patient,
        seed=seed,
    )
    active_val = select_patient_subset(
        val_df,
        target_scans=mode_cfg.target_val_scans,
        max_scans_per_patient=mode_cfg.max_scans_per_patient,
        seed=seed,
    )
    active_test = select_patient_subset(
        test_df,
        target_scans=mode_cfg.target_test_scans,
        max_scans_per_patient=mode_cfg.max_scans_per_patient,
        seed=seed,
    )

    is_clean, msg = verify_zero_patient_leakage(active_train, active_val, active_test)
    if not is_clean:
        raise ValueError(msg)

    return active_train, active_val, active_test


def get_hardware_info(device_name: Optional[str] = None) -> Dict[str, Any]:
    """Inspect local hardware environment and return diagnostic summary."""
    device = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    info: Dict[str, Any] = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_gb": 0.0,
        "compute_capability": None,
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        info["compute_capability"] = torch.cuda.get_device_capability(0)

    # Quantum status
    try:
        import pennylane as qml
        info["pennylane_installed"] = True
        info["pennylane_version"] = qml.__version__
    except ImportError:
        info["pennylane_installed"] = False
        info["pennylane_version"] = None

    return info
