"""
generate_model_comparison.py — Automated Three-Model Evaluation & Visualizer CLI
================================================================================
Generates all metric comparison tables, training/validation graphs,
test radiograph prediction examples, and markdown reports.

Usage
-----
    python scripts/generate_model_comparison.py --results_dir results --output_dir outputs/model_comparison
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import torch

from src.training.evaluator import (
    THREE_MODELS,
    load_experiment_metrics,
    generate_comparison_plots,
    visualize_test_predictions,
    generate_comparison_report,
)
from src.models.model_factory import build_model
from src.inference.checkpoint_loader import ModelCheckpointLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_model_comparison")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 3-Model Comparison & Visualizations.")
    p.add_argument("--results_dir", type=str, default="results", help="Path to results directory.")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Path to checkpoints directory.")
    p.add_argument("--output_dir", type=str, default="outputs/model_comparison", help="Output directory for reports & plots.")
    p.add_argument("--splits_dir", type=str, default="data/splits", help="Path to splits directory.")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    p.add_argument("--n_examples", type=int, default=10, help="Number of real test examples to visualize.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="'cuda' or 'cpu'.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== STEP 1: Aggregating Metrics for 3 Models ===")
    df_metrics = load_experiment_metrics(results_dir=args.results_dir, models=THREE_MODELS)
    csv_path = out_dir / "comparison_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    logger.info("Saved comparison metrics to %s\n%s", csv_path, df_metrics.to_string())

    logger.info("=== STEP 2: Generating Evaluation Comparison Graphs ===")
    plots = generate_comparison_plots(results_dir=args.results_dir, output_dir=out_dir, models=THREE_MODELS)
    for name, path in plots.items():
        logger.info("  Generated: %s -> %s", name, path)

    logger.info("=== STEP 3: Loading Models for Real Test Image Visualizations ===")
    models_dict = {}
    for m_name in THREE_MODELS:
        m = build_model(m_name, pretrained=False, device=args.device)
        # Attempt to load best checkpoint if available
        ckpt = ModelCheckpointLoader.find_best_checkpoint(m_name, checkpoint_dir=args.checkpoints_dir)
        if ckpt and ckpt.exists():
            loaded, info = ModelCheckpointLoader.load_checkpoint(m, ckpt, device=args.device)
            if loaded:
                logger.info("Loaded real checkpoint for %s: %s", m_name, ckpt.name)
        m = m.to(args.device)
        m.eval()
        models_dict[m_name] = m

    test_csv = Path(args.splits_dir) / "test.csv"
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        hf_dataset = None
        try:
            from datasets import load_dataset
            hf_dataset = load_dataset("chehablab/NIHChestXR", split="train")
            logger.info("Loaded HF dataset for authentic radiographs.")
        except Exception as e:
            logger.warning("HF dataset unavailable; test visualizer will use fallback RGB: %s", e)

        logger.info("=== STEP 4: Visualizing %d Real Test-Set Radiographs ===", args.n_examples)
        example_dir = out_dir / "prediction_examples"
        visualize_test_predictions(
            test_df=test_df,
            hf_dataset=hf_dataset,
            models_dict=models_dict,
            output_dir=example_dir,
            device=args.device,
            threshold=args.threshold,
            n_examples=args.n_examples,
        )

    logger.info("=== STEP 5: Generating Comparison Report Markdown ===")
    report_file = out_dir / "comparison_report.md"
    generate_comparison_report(df_metrics, report_file)
    logger.info("=== Evaluation Pipeline Complete! Outputs saved in %s ===", out_dir)


if __name__ == "__main__":
    main()
