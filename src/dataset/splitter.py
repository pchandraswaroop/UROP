"""
splitter.py — Patient-Level Dataset Splitting
=============================================
Prevents patient leakage by grouping all scans from the same patient
into a single split.  Uses GroupShuffleSplit from scikit-learn.

Split sizes: 70 / 15 / 15 (train / val / test).

The output CSV files contain:
  - hf_index:     row index in the original HF dataset (for fast select())
  - patient_id:   patient identifier
  - scan_id:      scan identifier
  - split:        "train" | "val" | "test"

No disease labels, ages, or sex values are used for splitting decisions
(only the group = patient_id is used for GroupShuffleSplit).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


# ── Splitter ─────────────────────────────────────────────────────────────────

class PatientSplitter:
    """Patient-level GroupShuffleSplit into train / val / test."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
            "Split ratios must sum to 1.0"
        )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign each row to train / val / test split.

        Args:
            df: DataFrame with at minimum columns: [patient_id, hf_index]
                (hf_index is the row position in the HF dataset).

        Returns:
            DataFrame with an additional ``split`` column.
        """
        if "patient_id" not in df.columns:
            raise ValueError("DataFrame must contain 'patient_id' column.")

        groups = df["patient_id"].values
        indices = np.arange(len(df))

        # Step 1: Split off test set
        # We need (val_ratio + test_ratio) as the test fraction first
        val_test_ratio = self.val_ratio + self.test_ratio

        gss_1 = GroupShuffleSplit(
            n_splits=1,
            test_size=val_test_ratio,
            random_state=self.seed,
        )
        train_idx, val_test_idx = next(gss_1.split(indices, groups=groups))

        # Step 2: From the val_test pool, split into val and test
        # Within this pool, what fraction should be val?
        val_fraction_of_val_test = self.val_ratio / val_test_ratio

        groups_val_test = groups[val_test_idx]
        indices_val_test = np.arange(len(val_test_idx))

        gss_2 = GroupShuffleSplit(
            n_splits=1,
            test_size=1.0 - val_fraction_of_val_test,
            random_state=self.seed + 1,
        )
        val_local_idx, test_local_idx = next(
            gss_2.split(indices_val_test, groups=groups_val_test)
        )
        val_idx = val_test_idx[val_local_idx]
        test_idx = val_test_idx[test_local_idx]

        # Verify no patient leakage
        self._verify_no_leakage(df, train_idx, val_idx, test_idx)

        # Assign split labels
        df = df.copy()
        df["split"] = "train"
        df.iloc[val_idx, df.columns.get_loc("split")] = "val"
        df.iloc[test_idx, df.columns.get_loc("split")] = "test"

        # Log statistics
        n_patients_train = df[df["split"] == "train"]["patient_id"].nunique()
        n_patients_val = df[df["split"] == "val"]["patient_id"].nunique()
        n_patients_test = df[df["split"] == "test"]["patient_id"].nunique()

        logger.info(
            "Split sizes — train: %d scans (%d patients) | val: %d scans "
            "(%d patients) | test: %d scans (%d patients)",
            (df["split"] == "train").sum(), n_patients_train,
            (df["split"] == "val").sum(), n_patients_val,
            (df["split"] == "test").sum(), n_patients_test,
        )

        return df

    def _verify_no_leakage(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        """Assert that no patient appears in more than one split."""
        train_patients = set(df.iloc[train_idx]["patient_id"].unique())
        val_patients = set(df.iloc[val_idx]["patient_id"].unique())
        test_patients = set(df.iloc[test_idx]["patient_id"].unique())

        tv_overlap = train_patients & val_patients
        tt_overlap = train_patients & test_patients
        vt_overlap = val_patients & test_patients

        if tv_overlap or tt_overlap or vt_overlap:
            raise AssertionError(
                f"Patient leakage detected! "
                f"train∩val={len(tv_overlap)}, "
                f"train∩test={len(tt_overlap)}, "
                f"val∩test={len(vt_overlap)}"
            )
        logger.info("Patient leakage check passed: no overlap across splits.")


# ── Utilities ────────────────────────────────────────────────────────────────

def build_index_dataframe(ds) -> pd.DataFrame:
    """Build a lightweight DataFrame from the HF dataset (no images).

    Returns columns: [hf_index, patient_id, scan_id, age, sex, labels_raw]
    where labels_raw is the list[int] as stored in HF.
    """
    logger.info("Building index DataFrame (loading metadata columns)...")

    # Remove image column before converting to pandas to save memory
    metadata_ds = ds.remove_columns(["image"])
    df = metadata_ds.to_pandas()
    df.insert(0, "hf_index", df.index)

    # Rename if needed
    if "labels" not in df.columns:
        raise ValueError("Expected 'labels' column in dataset.")

    # Convert labels from list objects to a reliable form
    df["labels_raw"] = df["labels"].apply(
        lambda x: list(x) if hasattr(x, "__iter__") else []
    )

    logger.info("Index DataFrame built: %d rows", len(df))
    return df


def save_split_csvs(df: pd.DataFrame, output_dir: str) -> None:
    """Save per-split CSVs to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name].copy()
        path = out / f"{split_name}.csv"
        subset.to_csv(path, index=False)
        logger.info("Saved %s: %d rows → %s", split_name, len(subset), path)

    # Also save the full index
    full_path = out / "full_index.csv"
    df.to_csv(full_path, index=False)
    logger.info("Saved full index: %d rows → %s", len(df), full_path)


def load_split_csvs(split_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved train / val / test CSVs.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    d = Path(split_dir)
    train_df = pd.read_csv(d / "train.csv")
    val_df = pd.read_csv(d / "val.csv")
    test_df = pd.read_csv(d / "test.csv")
    return train_df, val_df, test_df
