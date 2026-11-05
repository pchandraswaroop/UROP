"""
test_splitter.py — Unit tests for PatientSplitter
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.splitter import PatientSplitter


def make_fake_df(n_patients: int = 100, scans_per_patient: int = 3) -> pd.DataFrame:
    """Create a fake split-ready DataFrame."""
    rows = []
    for p_id in range(n_patients):
        for s_id in range(scans_per_patient):
            rows.append({
                "hf_index":   p_id * scans_per_patient + s_id,
                "patient_id": f"P{p_id:05d}",
                "scan_id":    p_id * scans_per_patient + s_id,
                "age":        30 + (p_id % 50),
                "sex":        "M" if p_id % 2 == 0 else "F",
                "labels_raw": [1] if p_id % 3 == 0 else [0],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def fake_df():
    return make_fake_df(n_patients=100)


class TestPatientSplitter:
    def test_split_creates_three_splits(self, fake_df):
        splitter = PatientSplitter()
        result = splitter.split(fake_df)
        splits = result["split"].unique()
        assert set(splits) == {"train", "val", "test"}

    def test_ratios_approximately_correct(self, fake_df):
        splitter = PatientSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        result = splitter.split(fake_df)
        total = len(result)
        for split_name, expected_ratio in [("train", 0.7), ("val", 0.15), ("test", 0.15)]:
            actual_ratio = (result["split"] == split_name).sum() / total
            assert abs(actual_ratio - expected_ratio) < 0.1, (
                f"{split_name} ratio {actual_ratio:.2f} too far from {expected_ratio}"
            )

    def test_no_patient_leakage(self, fake_df):
        splitter = PatientSplitter()
        result = splitter.split(fake_df)
        train_p = set(result[result["split"] == "train"]["patient_id"])
        val_p   = set(result[result["split"] == "val"]["patient_id"])
        test_p  = set(result[result["split"] == "test"]["patient_id"])
        assert len(train_p & val_p)  == 0, "Train-val patient overlap!"
        assert len(train_p & test_p) == 0, "Train-test patient overlap!"
        assert len(val_p & test_p)   == 0, "Val-test patient overlap!"

    def test_all_rows_assigned(self, fake_df):
        splitter = PatientSplitter()
        result = splitter.split(fake_df)
        assert len(result) == len(fake_df), "Some rows lost during splitting"
        assert result["split"].isna().sum() == 0, "NaN split assignments found"

    def test_reproducible_with_same_seed(self, fake_df):
        s1 = PatientSplitter(seed=42).split(fake_df)["split"].values
        s2 = PatientSplitter(seed=42).split(fake_df)["split"].values
        assert (s1 == s2).all(), "Splits not reproducible with same seed"

    def test_different_seeds_different_splits(self, fake_df):
        s1 = PatientSplitter(seed=1).split(fake_df)["split"].values
        s2 = PatientSplitter(seed=999).split(fake_df)["split"].values
        assert not (s1 == s2).all(), "Different seeds should (very likely) differ"

    def test_ratios_sum_to_one(self):
        with pytest.raises(AssertionError):
            PatientSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.2)

    def test_single_patient_dataset(self):
        """With only 1 unique patient sklearn GroupShuffleSplit cannot split — expect ValueError."""
        df = pd.DataFrame([
            {"hf_index": i, "patient_id": "P00001", "scan_id": i,
             "age": 40, "sex": "M", "labels_raw": [1]}
            for i in range(5)
        ])
        splitter = PatientSplitter()
        # sklearn requires at least 2 groups to form a non-empty train set
        with pytest.raises(ValueError):
            splitter.split(df)
