"""
analyze_statistics.py
=====================
Task 4: Deep analysis of dataset statistics including:
  - Missing values
  - Duplicate scan IDs
  - Age distribution with outlier detection
  - Disease label distribution and co-occurrence
  - Patient scan count distribution
  - Sex distribution

Produces: data/stats/*.png visualizations and data/stats/dataset_report.md

Usage:
    cd urop
    python scripts/analyze_statistics.py --cache_dir data/raw --sample_n 2000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
import logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data/raw")
    parser.add_argument("--sample_n", type=int, default=2000,
                        help="Sample size for fast analysis; -1 = full dataset")
    parser.add_argument("--stats_dir", default="data/stats")
    args = parser.parse_args()

    import ast
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.dataset.hf_loader import load_hf_dataset, get_dataset_stats_metadata
    from src.dataset.label_encoder import NIH_LABEL_NAMES, DISEASE_LABEL_NAMES

    Path(args.stats_dir).mkdir(parents=True, exist_ok=True)
    sample_n = args.sample_n if args.sample_n > 0 else None
    ds = load_hf_dataset(cache_dir=args.cache_dir, sample_n=sample_n)

    logger.info("Running statistics analysis on %d rows...", len(ds))
    stats, df = get_dataset_stats_metadata(ds)

    # ── Parse labels ──────────────────────────────────────────────────────
    def parse_labels(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        if isinstance(x, (list, np.ndarray)):
            return list(x)
        return []

    label_col = "labels" if "labels" in df.columns else "labels_raw"
    df["labels_parsed"] = df[label_col].apply(parse_labels)

    # ── Missing values ──────────────────────────────────────────────────
    logger.info("\n=== MISSING VALUES ===")
    for col in ["patient_id", "scan_id", "age", "sex"]:
        n_miss = df[col].isna().sum()
        logger.info("  %s: %d missing (%.2f%%)", col, n_miss, 100 * n_miss / len(df))

    # ── Duplicates ───────────────────────────────────────────────────────
    n_dup_scans = df["scan_id"].duplicated().sum()
    logger.info("\n=== DUPLICATES ===")
    logger.info("  Duplicate scan_ids: %d", n_dup_scans)

    # ── Age distribution ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["age"].dropna(), bins=50, color="#4A90D9", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Age (years)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Patient Age Distribution — NIH ChestX-ray14", fontsize=15)
    ax.axvline(df["age"].mean(), color="#E74C3C", ls="--", label=f"Mean={df['age'].mean():.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{args.stats_dir}/age_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved age_distribution.png")

    # Age outliers
    age_outlier_low  = (df["age"] < 1).sum()
    age_outlier_high = (df["age"] > 100).sum()
    logger.info("  Age < 1:   %d rows", age_outlier_low)
    logger.info("  Age > 100: %d rows", age_outlier_high)

    # ── Sex distribution ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sex_counts = df["sex"].value_counts()
    ax.bar(sex_counts.index, sex_counts.values, color=["#4A90D9", "#E74C3C"], alpha=0.85, edgecolor="white")
    ax.set_title("Sex Distribution", fontsize=15)
    ax.set_ylabel("Count", fontsize=13)
    for i, (label, val) in enumerate(sex_counts.items()):
        ax.text(i, val + 10, f"{val}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{args.stats_dir}/sex_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved sex_distribution.png")

    # ── Label distribution ────────────────────────────────────────────────
    label_counts = {name: 0 for name in NIH_LABEL_NAMES}
    for labels in df["labels_parsed"]:
        for idx in labels:
            if 0 <= idx < len(NIH_LABEL_NAMES):
                label_counts[NIH_LABEL_NAMES[idx]] += 1

    label_series = pd.Series(label_counts).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(label_series.index, label_series.values, color="#4A90D9", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Count", fontsize=13)
    ax.set_title("Disease Label Distribution — NIH ChestX-ray14", fontsize=15)
    for bar, val in zip(bars, label_series.values):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{args.stats_dir}/label_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved label_distribution.png")

    # ── Patient scan count distribution ──────────────────────────────────
    scan_per_patient = df.groupby("patient_id").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scan_per_patient, bins=30, color="#27AE60", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Scans per patient", fontsize=13)
    ax.set_ylabel("Number of patients", fontsize=13)
    ax.set_title("Scans per Patient Distribution", fontsize=15)
    ax.axvline(scan_per_patient.mean(), color="#E74C3C", ls="--",
               label=f"Mean={scan_per_patient.mean():.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{args.stats_dir}/patient_scan_count.png", dpi=150)
    plt.close()
    logger.info("Saved patient_scan_count.png")

    # ── Label co-occurrence heatmap ───────────────────────────────────────
    # Build multi-hot matrix for disease labels only
    disease_matrix = np.zeros((len(df), 14))
    for i, labels in enumerate(df["labels_parsed"]):
        for idx in labels:
            if idx == 0:
                continue
            di = idx - 1
            if 0 <= di < 14:
                disease_matrix[i, di] = 1.0

    co_occur = (disease_matrix.T @ disease_matrix).astype(int)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        co_occur,
        xticklabels=DISEASE_LABEL_NAMES,
        yticklabels=DISEASE_LABEL_NAMES,
        annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.3, linecolor="gray",
        ax=ax,
    )
    ax.set_title("Disease Co-occurrence Matrix", fontsize=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{args.stats_dir}/cooccurrence_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved cooccurrence_matrix.png")

    # ── Write report ──────────────────────────────────────────────────────
    report_path = Path(args.stats_dir) / "dataset_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# NIH ChestX-ray14 Dataset Report\n\n")
        f.write(f"**Sample size analysed:** {len(df):,}\n\n")
        f.write("## Schema\n")
        f.write("| Field | Type | Description |\n|---|---|---|\n")
        f.write("| patient_id | string | Unique patient identifier |\n")
        f.write("| scan_id | int32 | Unique scan identifier |\n")
        f.write("| image | PIL Image (244×244) | Frontal chest X-ray |\n")
        f.write("| age | int32 | Patient age at scan time |\n")
        f.write("| sex | string | M / F |\n")
        f.write("| labels | list[int32] | NIH label indices (0=No Finding, 1-14=disease) |\n\n")
        f.write("## Statistics\n")
        for k, v in stats.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Outliers\n")
        f.write(f"- Age < 1: {age_outlier_low}\n")
        f.write(f"- Age > 100: {age_outlier_high}\n")
        f.write(f"- Duplicate scan_ids: {n_dup_scans}\n")
        f.write("\n## Label Prevalence\n")
        f.write("| Label | Count | % |\n|---|---|---|\n")
        for name, cnt in label_series.items():
            f.write(f"| {name} | {cnt} | {100*cnt/len(df):.2f}% |\n")
        f.write("\n## Figures\n")
        for fig_name in ["age_distribution", "sex_distribution", "label_distribution",
                         "patient_scan_count", "cooccurrence_matrix"]:
            f.write(f"- [{fig_name}.png]({fig_name}.png)\n")
        f.write("\n## Design Decisions\n")
        f.write("- Labels stored as list[int32]; index 0 = No Finding (excluded from targets).\n")
        f.write("- 14 disease classes predicted with independent sigmoid outputs.\n")
        f.write("- Patient IDs used ONLY for split assignment (GroupShuffleSplit).\n")
        f.write("- GNN node features limited to age_norm + sex_binary (non-leaking).\n")

    logger.info("Dataset report saved to %s", report_path)
    logger.info("✅ Statistics analysis complete.")


if __name__ == "__main__":
    main()
