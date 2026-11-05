"""
build_graph.py
==============
Task 8: Build the patient-level graph from training split metadata.

Loads the training CSV, builds the PatientGraph using Jaccard similarity,
validates the graph, and saves it.

Usage:
    cd urop
    python scripts/build_graph.py --split_dir data/splits --graph_path data/patient_graph.pt
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
    parser.add_argument("--split_dir", default="data/splits")
    parser.add_argument("--graph_path", default="data/patient_graph.pt")
    parser.add_argument("--edge_threshold", type=float, default=0.3)
    parser.add_argument(
        "--max_patients", type=int, default=500,
        help="Limit patients for testing (avoids OOM on full 30k). -1 = all."
    )
    args = parser.parse_args()

    from src.dataset.splitter import load_split_csvs
    from src.graph.graph_builder import PatientGraphBuilder, save_graph, TORCH_GEOMETRIC_AVAILABLE

    if not TORCH_GEOMETRIC_AVAILABLE:
        logger.error(
            "torch-geometric is not installed.\n"
            "The GNN branch will be disabled. Proceeding without graph.\n"
            "To enable: pip install torch-geometric"
        )
        logger.info(
            "LIMITATION: Without torch-geometric, only the MLP baseline\n"
            "(metadata-only) and image branch will be used for this run.\n"
            "The GNN ClinicalBranch is disabled via GNNDisabledStub."
        )
        return

    train_df, val_df, test_df = load_split_csvs(args.split_dir)
    logger.info("Training split: %d rows, %d patients",
                len(train_df), train_df["patient_id"].nunique())

    # Optionally limit for fast testing
    if args.max_patients > 0:
        patient_ids = train_df["patient_id"].unique()[:args.max_patients]
        train_df = train_df[train_df["patient_id"].isin(patient_ids)].copy()
        logger.info("Limited to %d patients for graph building.", len(train_df["patient_id"].unique()))

    builder = PatientGraphBuilder(
        edge_threshold=args.edge_threshold,
        num_disease_classes=14,
    )
    graph = builder.build(train_df)

    if graph is None:
        logger.error("Graph construction failed.")
        return

    logger.info("Graph summary:")
    logger.info("  Nodes: %d", graph.num_nodes)
    logger.info("  Edges: %d (undirected: %d)", graph.edge_index.shape[1],
                graph.edge_index.shape[1] // 2)
    logger.info("  Node feature dim: %s", graph.x.shape)

    # Basic validation
    assert graph.x.shape[0] == graph.num_nodes, "Node feature mismatch"
    assert graph.edge_index.max() < graph.num_nodes, "Edge index out of bounds"
    logger.info("✅ Graph validation passed.")

    # Save
    save_graph(graph, args.graph_path)
    logger.info("Graph saved to %s", args.graph_path)

    # Print GNN limitation
    logger.info("\n=== GNN DESIGN LIMITATIONS ===")
    logger.info("  Node features: age_norm + sex_binary (dim=2) only.")
    logger.info("  Edges: Jaccard similarity on TRAINING labels (not available for new patients).")
    logger.info("  This is a TRANSDUCTIVE graph: val/test patients not in the graph.")
    logger.info("  At inference: clinical branch falls back to MLPBaseline on metadata.")
    logger.info("  Recommendation: Compare GNN vs MLP AUC before deploying GNN branch.")


if __name__ == "__main__":
    main()
