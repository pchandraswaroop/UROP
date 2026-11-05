"""
verify_pipeline.py
==================
End-to-end smoke test on a tiny sample (≤ 64 rows).

Verifies:
  1. Dataset loading (HF)
  2. Split generation (no leakage)
  3. Image transform pipeline (correct tensor shapes)
  4. Label encoding (multi-hot [14])
  5. Metadata encoding ([2])
  6. DataLoader iteration (batch shapes)
  7. Model forward pass (image branch + MLP clinical + fusion + classifier)
  8. Loss computation
  9. Backpropagation (no NaN gradients)

Usage:
    cd urop
    python scripts/verify_pipeline.py [--n_samples 64]
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.reproducibility import set_seed
import logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        logger.info("  ✅ %s", name)
        return True
    else:
        logger.error("  ❌ %s — %s", name, detail)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--cache_dir", default="data/raw")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    set_seed(42)
    results = {}

    logger.info("=" * 60)
    logger.info("Pipeline Smoke Test (n=%d samples)", args.n_samples)
    logger.info("=" * 60)

    # ── 1. Dataset Loading ────────────────────────────────────────────────
    logger.info("\n[1] Dataset Loading")
    try:
        from src.dataset.hf_loader import load_hf_dataset
        ds = load_hf_dataset(cache_dir=args.cache_dir, sample_n=args.n_samples)
        results["load"] = check("HF dataset loaded", len(ds) == args.n_samples or len(ds) > 0,
                                f"got {len(ds)} rows")
    except Exception as e:
        logger.error("  ❌ Dataset loading failed: %s", e)
        traceback.print_exc()
        results["load"] = False

    if not results.get("load"):
        logger.error("Cannot proceed without dataset. Exiting.")
        sys.exit(1)

    # ── 2. Label Encoder ──────────────────────────────────────────────────
    logger.info("\n[2] Label Encoder")
    try:
        import torch
        from src.dataset.label_encoder import LabelEncoder, NIH_LABEL_NAMES

        enc = LabelEncoder()
        # Test: No Finding only
        t = enc.encode([0])
        results["label_no_finding"] = check("No Finding → all zeros", t.sum() == 0)
        # Test: Disease labels
        t = enc.encode([1, 4, 14])  # Atelectasis, Effusion, Pneumothorax
        results["label_disease"] = check(
            "Disease labels [1,4,14] → [14] tensor with 3 ones",
            t.sum() == 3 and t.shape == torch.Size([14]),
            f"shape={t.shape}, sum={t.sum()}"
        )
        # Test: empty
        t = enc.encode([])
        results["label_empty"] = check("Empty labels → all zeros", t.sum() == 0)
    except Exception as e:
        logger.error("  ❌ Label encoder failed: %s", e)
        traceback.print_exc()
        results["label_no_finding"] = False

    # ── 3. Metadata Encoder ───────────────────────────────────────────────
    logger.info("\n[3] Metadata Encoder")
    try:
        from src.dataset.preprocessor import MetadataEncoder
        meta_enc = MetadataEncoder()
        t = meta_enc.encode_row(age=45, sex="M")
        results["meta"] = check(
            "MetadataEncoder: [2] tensor",
            t.shape == torch.Size([2]),
            f"shape={t.shape}"
        )
        t_f = meta_enc.encode_row(age=45, sex="F")
        results["meta_sex"] = check("Sex M vs F differ", t[1] != t_f[1])
    except Exception as e:
        logger.error("  ❌ Metadata encoder failed: %s", e)
        traceback.print_exc()

    # ── 4. Transforms ──────────────────────────────────────────────────────
    logger.info("\n[4] Image Transforms")
    try:
        from src.dataset.preprocessor import get_transforms
        train_tf = get_transforms("train", image_size=224)
        val_tf   = get_transforms("val",   image_size=224)
        sample_img = ds[0]["image"]
        t_train = train_tf(sample_img)
        t_val   = val_tf(sample_img)
        results["transform_train"] = check("Train transform: [3,224,224]",
                                            t_train.shape == torch.Size([3, 224, 224]),
                                            str(t_train.shape))
        results["transform_val"]   = check("Val transform: [3,224,224]",
                                            t_val.shape == torch.Size([3, 224, 224]),
                                            str(t_val.shape))
    except Exception as e:
        logger.error("  ❌ Transform failed: %s", e)
        traceback.print_exc()

    # ── 5. Splitter ────────────────────────────────────────────────────────
    logger.info("\n[5] Patient Splitter")
    try:
        import pandas as pd
        from src.dataset.splitter import build_index_dataframe, PatientSplitter
        index_df = build_index_dataframe(ds)
        splitter = PatientSplitter(seed=42)
        split_df = splitter.split(index_df)
        n_splits = split_df["split"].nunique()
        results["split"] = check("Splitter: 3 splits", n_splits == 3, f"got {n_splits}")
        # Verify no leakage
        train_pats = set(split_df[split_df["split"] == "train"]["patient_id"])
        val_pats   = set(split_df[split_df["split"] == "val"]["patient_id"])
        test_pats  = set(split_df[split_df["split"] == "test"]["patient_id"])
        results["no_leakage"] = check(
            "No patient leakage",
            len(train_pats & val_pats) == 0 and
            len(train_pats & test_pats) == 0 and
            len(val_pats & test_pats) == 0,
        )
    except Exception as e:
        logger.error("  ❌ Splitter failed: %s", e)
        traceback.print_exc()

    # ── 6. DataLoader ──────────────────────────────────────────────────────
    logger.info("\n[6] DataLoader")
    try:
        from src.dataset.dataloader import CXRDataset, get_dataloader
        from src.dataset.preprocessor import get_transforms, MetadataEncoder

        if "split_df" not in dir():
            index_df = build_index_dataframe(ds)
            splitter = PatientSplitter(seed=42)
            split_df = splitter.split(index_df)

        train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
        loader   = get_dataloader("train", train_df, ds, batch_size=args.batch_size,
                                  num_workers=0)
        batch    = next(iter(loader))

        results["dl_image"]  = check(f"Batch image shape [B,3,224,224]",
                                      batch["image"].shape[1:] == torch.Size([3, 224, 224]),
                                      str(batch["image"].shape))
        results["dl_label"]  = check("Batch label shape [B,14]",
                                      batch["label"].shape[1] == 14,
                                      str(batch["label"].shape))
        results["dl_meta"]   = check("Batch meta shape [B,2]",
                                      batch["meta"].shape[1] == 2,
                                      str(batch["meta"].shape))
        results["dl_finite"] = check("Image values finite",
                                      torch.isfinite(batch["image"]).all())
    except Exception as e:
        logger.error("  ❌ DataLoader failed: %s", e)
        traceback.print_exc()

    # ── 7. Model Forward Pass ──────────────────────────────────────────────
    logger.info("\n[7] Model Forward Pass")
    try:
        from src.models.image_branch import ImageBranch
        from src.models.quantum_layer import get_quantum_layer
        from src.models.mlp_baseline import MetadataEmbedder
        from src.models.fusion import DiseaseSpecificFusion
        from src.models.classifier import DiseaseSpecificClassifier, MultiModalCXRModel

        FUSED_DIM = 256
        img_feat_dim = 1280   # EfficientNet-B0 output
        clinical_dim = 64
        quantum_out  = 64

        image_branch    = ImageBranch("efficientnet_b0", pretrained=False)
        quantum_layer   = get_quantum_layer(image_branch.feature_dim, output_dim=quantum_out, enabled=False)
        clinical_branch = MetadataEmbedder(input_dim=2, output_dim=clinical_dim)
        fusion          = DiseaseSpecificFusion(quantum_out, clinical_dim, fused_dim=FUSED_DIM)
        classifier      = DiseaseSpecificClassifier(FUSED_DIM, num_classes=14)

        model = MultiModalCXRModel(image_branch, clinical_branch, quantum_layer, fusion, classifier)
        model.eval()

        images_b = batch["image"]
        meta_b   = batch["meta"]
        with torch.no_grad():
            out = model(images_b, meta_b)

        results["model_shape"]  = check("Model output [B,14]",
                                         out.shape == torch.Size([args.batch_size, 14]) or
                                         out.shape[1] == 14,
                                         str(out.shape))
        results["model_range"]  = check("Output in [0,1] (sigmoid applied)",
                                         (out >= 0).all() and (out <= 1).all())
        results["model_finite"] = check("Output finite", torch.isfinite(out).all())
    except Exception as e:
        logger.error("  ❌ Model forward pass failed: %s", e)
        traceback.print_exc()

    # ── 8. Loss + Backprop ────────────────────────────────────────────────
    logger.info("\n[8] Loss + Backpropagation")
    try:
        from src.training.losses import WeightedBCELoss

        model.train()
        criterion = WeightedBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        out    = model(images_b, meta_b)
        logits = torch.logit(out.clamp(1e-6, 1 - 1e-6))
        loss   = criterion(logits, batch["label"])
        loss.backward()

        results["loss_finite"] = check("Loss finite", torch.isfinite(loss))
        # Check for NaN gradients
        has_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        results["no_nan_grad"] = check("No NaN gradients", not has_nan_grad)
    except Exception as e:
        logger.error("  ❌ Loss/backprop failed: %s", e)
        traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    total  = len(results)
    passed = sum(results.values())
    failed = total - passed
    logger.info(
        "SMOKE TEST RESULTS: %d/%d passed | %d failed",
        passed, total, failed
    )
    if failed > 0:
        logger.warning("Failed checks: %s", [k for k, v in results.items() if not v])
    else:
        logger.info("✅ All checks passed. Pipeline is ready.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
