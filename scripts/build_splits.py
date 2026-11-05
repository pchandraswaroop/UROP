"""
build_splits.py
===============
Task 5: Create patient-level train / val / test splits.

Loads the full HF dataset metadata (no images), builds an index DataFrame,
applies GroupShuffleSplit, verifies no patient leakage, and saves CSVs.

Usage:
    cd urop
    python scripts/build_splits.py --cache_dir data/raw --split_dir data/splits

    # For quick testing on a small sample:
    python scripts/build_splits.py --sample_n 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.reproducibility import set_seed
import logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data/raw")
    parser.add_argument("--split_dir", default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--test_ratio",  type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_n", type=int, default=-1,
        help="Limit to N rows for testing. -1 = full dataset."
    )
    args = parser.parse_args()

    set_seed(args.seed)

    from src.dataset.hf_loader import load_hf_dataset
    from src.dataset.splitter import (
        build_index_dataframe, PatientSplitter, save_split_csvs
    )

    # Load dataset (metadata only for split generation)
    sample_n = args.sample_n if args.sample_n > 0 else None
    logger.info("Loading dataset (sample_n=%s)...", sample_n or "full")
    ds = load_hf_dataset(cache_dir=args.cache_dir, sample_n=sample_n)

    # Build index DataFrame
    index_df = build_index_dataframe(ds)

    # Verify required columns
    assert "patient_id" in index_df.columns, "Missing patient_id column"
    assert "labels_raw" in index_df.columns, "Missing labels_raw column"

    logger.info("Dataset: %d rows, %d unique patients",
                len(index_df), index_df["patient_id"].nunique())

    # Apply patient-level split
    splitter = PatientSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split_df = splitter.split(index_df)

    # Print split statistics
    for s in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == s]
        n_patients = sub["patient_id"].nunique()
        logger.info(
            "  %s: %d scans | %d patients (%.1f%%)",
            s, len(sub), n_patients,
            100 * len(sub) / len(split_df)
        )

    # Save CSVs
    save_split_csvs(split_df, args.split_dir)
    logger.info("✅ Splits saved to %s", args.split_dir)


if __name__ == "__main__":
    main()
