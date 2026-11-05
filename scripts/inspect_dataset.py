"""
inspect_dataset.py
==================
Task 1-2: Inspect the HuggingFace dataset schema and print a
comprehensive report on schema, sample records, and high-level statistics.

Usage:
    cd urop
    python scripts/inspect_dataset.py [--sample_n 200] [--cache_dir data/raw]
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
import logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Inspect NIH ChestX-ray HF Dataset")
    parser.add_argument("--cache_dir", default="data/raw", help="HF cache directory")
    parser.add_argument(
        "--sample_n", type=int, default=200,
        help="Number of rows to inspect (use -1 for full dataset)"
    )
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    # If full dataset is already cached locally, use offline mode to avoid remote network timeout
    import os
    raw_path = Path(args.cache_dir)
    if raw_path.exists() and (list(raw_path.rglob("*.arrow")) or list(raw_path.rglob("*.parquet"))):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    from src.dataset.hf_loader import load_hf_dataset, inspect_schema, get_dataset_stats_metadata

    logger.info("=" * 60)
    logger.info("NIH Chest X-ray Dataset Inspector")
    logger.info("=" * 60)

    # Load a small sample for schema inspection
    sample_n = args.sample_n if args.sample_n > 0 else None
    ds = load_hf_dataset(
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        sample_n=sample_n,
    )

    # Schema
    logger.info("\n--- SCHEMA ---")
    if not args.streaming:
        schema = inspect_schema(ds)
        for col, info in schema.items():
            logger.info("  %s: %s", col, info)

        # Sample records
        logger.info("\n--- SAMPLE RECORDS ---")
        for i in [0, 1, 2]:
            if i < len(ds):
                row = ds[i]
                logger.info(
                    "  Row %d: patient_id=%s | scan_id=%s | age=%s | sex=%s | labels=%s | image_size=%s",
                    i,
                    row["patient_id"],
                    row["scan_id"],
                    row["age"],
                    row["sex"],
                    row["labels"],
                    row["image"].size if row["image"] else "N/A",
                )

        # High-level statistics
        logger.info("\n--- HIGH-LEVEL STATISTICS ---")
        stats, df = get_dataset_stats_metadata(ds)
        for k, v in stats.items():
            logger.info("  %s: %s", k, v)

        # Label distribution
        logger.info("\n--- LABEL DISTRIBUTION (sample) ---")
        from src.dataset.label_encoder import NIH_LABEL_NAMES, DISEASE_LABEL_NAMES, NO_FINDING_IDX
        import ast

        label_col = "labels" if "labels" in df.columns else "labels_raw"
        label_counts = {name: 0 for name in NIH_LABEL_NAMES}
        for lbl_entry in df[label_col]:
            if isinstance(lbl_entry, str):
                try:
                    lbl_entry = ast.literal_eval(lbl_entry)
                except Exception:
                    lbl_entry = []
            elif hasattr(lbl_entry, "tolist"):
                lbl_entry = lbl_entry.tolist()
            if isinstance(lbl_entry, (list, tuple)):
                for idx in lbl_entry:
                    if 0 <= idx < len(NIH_LABEL_NAMES):
                        label_counts[NIH_LABEL_NAMES[idx]] += 1

        total = len(df)
        logger.info("  %-25s %8s %8s", "Label", "Count", "Pct%")
        logger.info("  " + "-" * 44)
        for name, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / total
            logger.info("  %-25s %8d %7.2f%%", name, cnt, pct)

        # Patient and Scan Graph Statistics
        logger.info("\n--- PATIENT & SCAN GRAPH ANALYSIS ---")
        n_unique_patients = df["patient_id"].nunique()
        p_counts = df["patient_id"].value_counts()
        multi_patients = p_counts[p_counts > 1]
        n_multi_patients = len(multi_patients)
        n_multi_scans = int(multi_patients.sum()) if n_multi_patients > 0 else 0
        max_scans = int(p_counts.max()) if len(p_counts) > 0 else 0

        # Scan-to-scan edges: connect scans belonging to same patient_id (no labels used)
        scan_edges = 0
        for count in p_counts:
            if count > 1:
                scan_edges += count * (count - 1)  # directed edges (i != j)

        logger.info("  Total scans (nodes):              %d", total)
        logger.info("  Unique patients:                  %d", n_unique_patients)
        logger.info("  Patients with multiple scans:     %d (%.1f%% of patients)",
                    n_multi_patients, 100 * n_multi_patients / max(1, n_unique_patients))
        logger.info("  Scans belonging to multi-patient: %d (%.1f%% of scans)",
                    n_multi_scans, 100 * n_multi_scans / max(1, total))
        logger.info("  Max scans for single patient:     %d", max_scans)
        logger.info("  Scan-to-scan edges (i != j):      %d directed edges", scan_edges)
        logger.info("  Scan-to-scan edges (+self-loops): %d directed edges", scan_edges + total)

        # Age anomaly report
        logger.info("\n--- AGE ANALYSIS ---")
        logger.info("  Age < 1:   %d", (df["age"] < 1).sum())
        logger.info("  Age > 100: %d", (df["age"] > 100).sum())
        logger.info(
            "  Age quantiles: 5th=%.0f, 25th=%.0f, 50th=%.0f, 75th=%.0f, 95th=%.0f",
            df["age"].quantile(0.05),
            df["age"].quantile(0.25),
            df["age"].quantile(0.50),
            df["age"].quantile(0.75),
            df["age"].quantile(0.95),
        )

        logger.info("\n[OK] Dataset inspection complete.")
    else:
        logger.info("Streaming mode: schema and sample inspection only.")
        # Just show first sample
        for i, row in enumerate(ds):
            logger.info("  Row %d: patient_id=%s | labels=%s", i, row["patient_id"], row["labels"])
            if i >= 2:
                break


if __name__ == "__main__":
    main()
