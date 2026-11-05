"""
hf_loader.py — HuggingFace Dataset Loader
==========================================
Handles streaming vs. full loading of chehablab/NIHChestXR.
Provides utilities to sample subsets for fast local development.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset, Dataset, DatasetDict

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

HF_REPO = "chehablab/NIHChestXR"
HF_SPLIT = "train"


# ── Loader ───────────────────────────────────────────────────────────────────

def load_hf_dataset(
    repo: str = HF_REPO,
    split: str = HF_SPLIT,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    sample_n: Optional[int] = None,
) -> Dataset:
    """Load the NIH Chest X-ray dataset from HuggingFace.

    Args:
        repo:       HuggingFace dataset repository id.
        split:      Dataset split (only "train" is available).
        cache_dir:  Local path to cache downloaded files.
        streaming:  If True, use streaming mode (no download). Slower for
                    random access but avoids disk usage.  Set to False for
                    full pipeline use.
        sample_n:   If provided, return only the first `sample_n` rows.
                    Useful for fast local development and smoke tests.

    Returns:
        A HuggingFace ``Dataset`` object (or ``IterableDataset`` if streaming).
    """
    cache_dir = str(Path(cache_dir).resolve()) if cache_dir else None

    logger.info(
        "Loading dataset '%s' (split=%s, streaming=%s, cache_dir=%s, sample_n=%s)",
        repo, split, streaming, cache_dir, sample_n
    )

    # 1. If explicit streaming without sample_n, return the raw IterableDataset
    if streaming and sample_n is None:
        return load_dataset(repo, split=split, cache_dir=cache_dir, streaming=True, trust_remote_code=False)

    # 2. Check if a complete dataset is already cached locally
    cached_exists = False
    if cache_dir and Path(cache_dir).exists():
        cached_files = list(Path(cache_dir).rglob("*.arrow")) + list(Path(cache_dir).rglob("*.parquet"))
        if len(cached_files) >= 5:
            cached_exists = True
            import os
            os.environ["HF_DATASETS_OFFLINE"] = "1"

    # 3. If sample_n requested and no complete local cache, use genuine streaming
    # to download ONLY sample_n records without downloading all six shards (~2.5 GB)
    if sample_n is not None and sample_n > 0 and not cached_exists and not streaming:
        logger.info(
            "Genuine streaming enabled: downloading only first %d records (skipping full ~2.5 GB download).",
            sample_n
        )
        streamed_ds = load_dataset(repo, split=split, cache_dir=cache_dir, streaming=True, trust_remote_code=False)
        sample_rows = []
        for row in streamed_ds.take(sample_n):
            sample_rows.append(row)
        ds = Dataset.from_list(sample_rows)
        logger.info("Materialized in-memory Dataset from stream: %d rows, columns: %s", len(ds), ds.column_names)
        return ds

    # 4. Standard path: load full dataset (or from cache if offline/available)
    try:
        ds = load_dataset(
            repo,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=False,
        )
    except Exception as exc:
        if cached_exists:
            logger.info("Network check failed; loading from existing offline cache: %s", exc)
            import os
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            ds = load_dataset(
                repo,
                split=split,
                cache_dir=cache_dir,
                streaming=streaming,
                trust_remote_code=False,
            )
        else:
            raise exc

    if sample_n is not None and sample_n > 0:
        logger.info("Selecting first %d samples for development subset.", sample_n)
        if streaming:
            # For streaming datasets, take() returns an IterableDataset
            ds = ds.take(sample_n)
        else:
            ds = ds.select(range(min(sample_n, len(ds))))

    if not streaming:
        logger.info("Dataset loaded: %d rows, features=%s", len(ds), list(ds.features.keys()))

    return ds


def load_full_non_streaming(
    repo: str = HF_REPO,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Convenience wrapper: full dataset, non-streaming.

    Preferred for split generation, statistics, and DataLoader construction.
    The ~2.5 GB dataset will be cached locally after the first download.
    """
    return load_hf_dataset(repo=repo, cache_dir=cache_dir, streaming=False)


def inspect_schema(ds: Dataset) -> dict:
    """Return a dict summarising the dataset's schema and a sample row.

    Does NOT print images — only metadata fields and shapes.
    """
    features = ds.features
    sample = ds[0]

    schema = {}
    for col, feat in features.items():
        if col == "image":
            img = sample["image"]
            schema[col] = {
                "type": type(img).__name__,
                "mode": getattr(img, "mode", "N/A"),
                "size": getattr(img, "size", "N/A"),
            }
        else:
            raw_val = sample[col]
            schema[col] = {
                "hf_feature_type": str(feat),
                "sample_value": raw_val,
                "python_type": type(raw_val).__name__,
            }

    return schema


def get_dataset_stats_metadata(ds: Dataset) -> dict:
    """Fast pass to collect non-image stats without loading PIL images.

    Returns dict with keys: n_rows, unique_patients, age_range, sex_values,
    label_lengths.
    """
    import pandas as pd

    # Convert to pandas (metadata columns only — skip image)
    df = ds.remove_columns(["image"]).to_pandas()
    # Provide labels_raw alias if labels is present for backward compatibility
    if "labels" in df.columns and "labels_raw" not in df.columns:
        df["labels_raw"] = df["labels"]

    stats = {
        "n_rows": len(df),
        "n_columns_non_image": len(df.columns),
        "columns": list(df.columns),
        "unique_patients": df["patient_id"].nunique(),
        "age_min": int(df["age"].min()),
        "age_max": int(df["age"].max()),
        "age_mean": float(df["age"].mean()),
        "age_std": float(df["age"].std()),
        "sex_value_counts": df["sex"].value_counts().to_dict(),
        "missing_patient_id": int(df["patient_id"].isna().sum()),
        "missing_scan_id": int(df["scan_id"].isna().sum()),
        "missing_age": int(df["age"].isna().sum()),
        "missing_sex": int(df["sex"].isna().sum()),
        "missing_labels": int(df["labels"].apply(lambda x: x is None or len(x) == 0).sum()),
        "duplicate_scan_ids": int(df["scan_id"].duplicated().sum()),
    }

    return stats, df
