"""
run_explainability.py — Full Explainability & Faithfulness Evaluation Pipeline
=============================================================================
Generates:
  1. Grad-CAM++ / ViT heatmaps and overlays for individual disease predictions.
  2. Exact tabular Shapley values for age and sex.
  3. Scan-to-scan graph explanations with GNNExplainer.
  4. Multimodal fusion attribution (Image vs Clinical Meta vs Graph).
  5. Quantitative faithfulness evaluation (deletion curves, perturbation tests).

Outputs are structured into:
  outputs/explainability/
      gradcam/
      shap/
      graph/
      modality/
      faithfulness/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Auto-detect local cache to prevent remote network timeout
raw_cache = Path(__file__).parent.parent / "data" / "raw"
if raw_cache.exists() and (list(raw_cache.rglob("*.arrow")) or list(raw_cache.rglob("*.parquet"))):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch

from src.dataset.hf_loader import load_hf_dataset
from src.dataset.label_encoder import DISEASE_LABEL_NAMES, LabelEncoder
from src.dataset.preprocessor import MetadataEncoder, get_transforms
from src.explainability.faithfulness import FaithfulnessEvaluator
from src.explainability.gnn_explainer import ScanGraphExplainer
from src.explainability.gradcam import get_image_explainer
from src.explainability.modality_attribution import MultimodalAttribution
from src.explainability.tabular_shap import TabularFeatureExplainer
from src.explainability.visualizer import ExplainabilityVisualizer
from src.graph.graph_builder import ScanGraphBuilder, build_batch_scan_graph
from src.models.model_factory import build_model
from src.utils.logger import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Explainability & Faithfulness Pipeline")
    parser.add_argument("--experiment", default="quantum_gnn_multimodal", help="Experiment variant")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint")
    parser.add_argument("--sample_n", type=int, default=20, help="Number of records to explain")
    parser.add_argument("--output_dir", default="outputs/explainability", help="Base directory for artifacts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu/cuda)")
    parser.add_argument("--cache_dir", default="data/raw", help="Path to raw dataset cache")
    parser.add_argument("--target_classes", nargs="+", type=int, default=[0, 1, 3], help="Class indices to explain")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    # Enable offline dataset loading if local cache is present
    raw_path = Path(args.cache_dir)
    if raw_path.exists() and (list(raw_path.rglob("*.arrow")) or list(raw_path.rglob("*.parquet"))):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    out_dir = Path(args.output_dir)
    for sub in ["gradcam", "shap", "graph", "modality", "faithfulness"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Explainability & Faithfulness Pipeline")
    logger.info("Experiment: %s | Device: %s | Sample size: %d", args.experiment, device, args.sample_n)
    logger.info("=" * 70)

    # 1. Build Model
    logger.info("Building model for experiment '%s'...", args.experiment)
    model = build_model(args.experiment, pretrained=False, device=device)
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info("Loading checkpoint from %s...", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
    else:
        logger.info("No checkpoint provided (or not found). Running evaluation with initialized model.")
    model.to(device)
    model.eval()

    # 2. Load Sample Data
    logger.info("Loading %d dataset samples...", args.sample_n)
    ds = load_hf_dataset(cache_dir=args.cache_dir, sample_n=args.sample_n)
    df = ds.remove_columns(["image"]).to_pandas()
    if "labels" in df.columns and "labels_raw" not in df.columns:
        df["labels_raw"] = df["labels"]

    # Preprocessing
    transform = get_transforms("val", image_size=224)
    meta_encoder = MetadataEncoder()

    images_list = []
    for i in range(len(ds)):
        pil_img = ds[i]["image"]
        images_list.append(transform(pil_img))
    images = torch.stack(images_list).to(device)  # [B, 3, 224, 224]

    meta = meta_encoder.encode_batch(df["age"].tolist(), df["sex"].tolist()).to(device)  # [B, 2]
    patient_ids = [str(pid) for pid in df["patient_id"].tolist()]

    # Graph construction
    graph_builder = ScanGraphBuilder(add_self_loops=True)
    scan_graph = graph_builder.build(df, meta_encoder=meta_encoder)
    graph_x = scan_graph.x.to(device) if scan_graph else None
    graph_edge_index = scan_graph.edge_index.to(device) if scan_graph else None

    # Model Predictions
    with torch.no_grad():
        try:
            logits = model(images, meta, graph_x=graph_x, graph_edge_index=graph_edge_index, return_logits=True)
        except TypeError:
            logits = model(images, meta, return_logits=True)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Save summary predictions CSV
    preds_df = pd.DataFrame(probs, columns=DISEASE_LABEL_NAMES)
    preds_df.insert(0, "patient_id", patient_ids)
    preds_df.insert(1, "scan_id", df["scan_id"].tolist())
    preds_df.to_csv(out_dir / "sample_predictions.csv", index=False)
    logger.info("Saved sample predictions to %s", out_dir / "sample_predictions.csv")

    # -------------------------------------------------------------------------
    # 1. Grad-CAM++ / ViT Image Explanations
    # -------------------------------------------------------------------------
    logger.info("\n--- 1. Generating Image Explanations (Grad-CAM++) ---")
    image_explainer = get_image_explainer(model)
    gradcam_records = []

    for sample_idx in range(min(5, len(df))):
        sample_img = images[sample_idx : sample_idx + 1]
        sample_meta = meta[sample_idx : sample_idx + 1]

        # Multi-disease heatmaps on the same radiograph
        heatmaps_dict = {}
        probs_dict = {}

        for c_idx in args.target_classes:
            disease_name = DISEASE_LABEL_NAMES[c_idx]
            hmap = image_explainer.generate(
                sample_img,
                sample_meta,
                graph_x=sample_img if graph_x is None else graph_x[sample_idx : sample_idx + 1],
                graph_edge_index=torch.zeros((2, 0), dtype=torch.long, device=device) if graph_edge_index is None else torch.zeros((2, 0), dtype=torch.long, device=device),
                target_class=c_idx,
            )[0]
            heatmaps_dict[disease_name] = hmap
            probs_dict[disease_name] = float(probs[sample_idx, c_idx])

            # Save individual comparison plot
            save_p = out_dir / "gradcam" / f"sample_{sample_idx:02d}_{disease_name}.png"
            ExplainabilityVisualizer.plot_heatmap_comparison(
                sample_img[0], hmap, disease_name, probs_dict[disease_name], save_path=str(save_p)
            )
            gradcam_records.append({
                "sample_idx": sample_idx,
                "patient_id": patient_ids[sample_idx],
                "disease": disease_name,
                "probability": probs_dict[disease_name],
                "heatmap_mean": float(hmap.mean()),
                "heatmap_max": float(hmap.max()),
                "file": save_p.name,
            })

        # Save multi-disease comparative figure
        multi_p = out_dir / "gradcam" / f"sample_{sample_idx:02d}_multi_disease.png"
        ExplainabilityVisualizer.plot_multi_disease_heatmaps(
            sample_img[0], heatmaps_dict, probs_dict, save_path=str(multi_p)
        )

    image_explainer.remove_hooks()
    pd.DataFrame(gradcam_records).to_csv(out_dir / "gradcam" / "gradcam_summary.csv", index=False)
    logger.info("Saved Grad-CAM++ summary CSV.")

    # -------------------------------------------------------------------------
    # 2. Tabular SHAP Explanations
    # -------------------------------------------------------------------------
    logger.info("\n--- 2. Generating Tabular Feature Explanations (SHAP) ---")
    tab_explainer = TabularFeatureExplainer(model, num_classes=14)
    tab_res = tab_explainer.explain_instance(images, meta, graph_x=graph_x, graph_edge_index=graph_edge_index)

    mean_age_imp = np.abs(tab_res["Age"]).mean(axis=0)  # [14]
    mean_sex_imp = np.abs(tab_res["Sex"]).mean(axis=0)  # [14]

    shap_summary_data = {
        "Age": mean_age_imp,
        "Sex": mean_sex_imp,
    }
    shap_fig_path = out_dir / "shap" / "tabular_feature_importance.png"
    ExplainabilityVisualizer.plot_tabular_shap_summary(shap_summary_data, save_path=str(shap_fig_path))

    # Save SHAP CSV
    shap_df = pd.DataFrame({
        "Disease": DISEASE_LABEL_NAMES,
        "Age_MeanAbsShap": mean_age_imp,
        "Sex_MeanAbsShap": mean_sex_imp,
    })
    shap_df.to_csv(out_dir / "shap" / "shap_feature_importance.csv", index=False)
    logger.info("Saved SHAP summary to %s", out_dir / "shap" / "shap_feature_importance.csv")

    # -------------------------------------------------------------------------
    # 3. GNN Scan Graph Explanations
    # -------------------------------------------------------------------------
    logger.info("\n--- 3. Generating Scan Graph Explanations (GNNExplainer) ---")
    if scan_graph is not None and scan_graph.edge_index.shape[1] > 0:
        graph_explainer = ScanGraphExplainer(model)
        for c_idx in args.target_classes[:2]:
            disease_name = DISEASE_LABEL_NAMES[c_idx]
            g_exp = graph_explainer.explain(
                images, meta, graph_x, graph_edge_index,
                target_class=c_idx, patient_ids=patient_ids
            )
            g_fig_p = out_dir / "graph" / f"graph_explanation_{disease_name}.png"
            graph_explainer.visualize_graph(
                g_exp, title=f"Scan-to-Scan Graph Explanation ({disease_name})", save_path=str(g_fig_p)
            )

            # Save edge importance scores
            edge_csv_p = out_dir / "graph" / f"edge_importance_{disease_name}.csv"
            edges_df = pd.DataFrame({
                "src_node": g_exp["edge_index"][0],
                "dst_node": g_exp["edge_index"][1],
                "edge_importance": g_exp["edge_importance"],
            })
            edges_df.to_csv(edge_csv_p, index=False)
            logger.info("Saved graph explanation for %s", disease_name)

    # -------------------------------------------------------------------------
    # 4. Multimodal Fusion Attribution
    # -------------------------------------------------------------------------
    logger.info("\n--- 4. Generating Multimodal Fusion Attribution ---")
    mm_attr = MultimodalAttribution(model, num_classes=14)
    mm_res = mm_attr.attribute_instance(images, meta, graph_x, graph_edge_index)

    # Aggregate percentages
    mean_I = np.abs(mm_res["Image"]).mean(axis=0)
    mean_M = np.abs(mm_res["Clinical_Meta"]).mean(axis=0)
    mean_G = np.abs(mm_res["Graph"]).mean(axis=0)
    total_attr = mean_I + mean_M + mean_G + 1e-7

    pcts = np.stack([100.0 * mean_I / total_attr, 100.0 * mean_M / total_attr, 100.0 * mean_G / total_attr])
    attr_data = {
        "relative_percentages": pcts,
        "disease_names": DISEASE_LABEL_NAMES,
    }
    mm_chart_p = out_dir / "modality" / "modality_contributions.png"
    mm_attr.plot_modality_contributions(attr_data, save_path=str(mm_chart_p))

    mm_df = pd.DataFrame({
        "Disease": DISEASE_LABEL_NAMES,
        "Image_Contribution_Pct": pcts[0],
        "Clinical_Meta_Contribution_Pct": pcts[1],
        "Graph_Contribution_Pct": pcts[2],
    })
    mm_df.to_csv(out_dir / "modality" / "modality_contributions.csv", index=False)
    logger.info("Saved multimodal attribution CSV to %s", out_dir / "modality" / "modality_contributions.csv")

    # -------------------------------------------------------------------------
    # 5. Faithfulness Evaluations
    # -------------------------------------------------------------------------
    logger.info("\n--- 5. Executing Quantitative Faithfulness Evaluation ---")
    evaluator = FaithfulnessEvaluator(model)

    faith_records = []
    for sample_idx in range(min(3, len(df))):
        sample_img = images[sample_idx]
        sample_meta = meta[sample_idx : sample_idx + 1]

        # Use target class 0
        c_idx = args.target_classes[0]
        disease_name = DISEASE_LABEL_NAMES[c_idx]

        # A. Image Faithfulness
        hmap = image_explainer.generate(
            sample_img.unsqueeze(0), sample_meta, target_class=c_idx
        )[0]
        img_faith = evaluator.evaluate_image_faithfulness(
            sample_img, hmap, target_class=c_idx, meta=sample_meta
        )

        curve_p = out_dir / "faithfulness" / f"deletion_curve_sample_{sample_idx:02d}_{disease_name}.png"
        ExplainabilityVisualizer.plot_faithfulness_curves(img_faith, disease_name, save_path=str(curve_p))

        # B. Tabular Faithfulness
        tab_faith = evaluator.evaluate_tabular_faithfulness(
            sample_img.unsqueeze(0), sample_meta, target_class=c_idx,
            shap_values={"Age": tab_res["Age"][sample_idx : sample_idx + 1], "Sex": tab_res["Sex"][sample_idx : sample_idx + 1]}
        )

        # C. Modality Faithfulness
        mod_faith = evaluator.evaluate_modality_faithfulness(
            sample_img.unsqueeze(0), sample_meta, graph_x=None, graph_edge_index=None,
            modality_percentages=pcts, target_class=c_idx
        )

        faith_records.append({
            "sample_idx": sample_idx,
            "patient_id": patient_ids[sample_idx],
            "disease": disease_name,
            "img_faithfulness_ratio": img_faith["faithfulness_ratio"],
            "img_is_faithful": img_faith["is_faithful"],
            "tab_top_feature": tab_faith["top_feature"],
            "tab_top_feature_delta": tab_faith["top_feature_delta"],
            "tab_is_faithful": tab_faith["is_faithful"],
            "mod_dominant_modality": mod_faith["dominant_attributed_modality"],
            "mod_is_faithful": mod_faith["is_faithful"],
        })

    pd.DataFrame(faith_records).to_csv(out_dir / "faithfulness" / "faithfulness_metrics.csv", index=False)
    logger.info("Saved faithfulness metrics CSV.")

    logger.info("\n" + "=" * 70)
    logger.info("EXPLAINABILITY & FAITHFULNESS PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("All visual and numerical artifacts written to: %s", out_dir.resolve())
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
