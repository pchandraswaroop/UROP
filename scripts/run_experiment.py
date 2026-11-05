"""
run_experiment.py — CLI for a single experiment run
====================================================
Run any experiment variant with configurable options.

Examples
--------
    python scripts/run_experiment.py --experiment cnn_only --smoke_test
    python scripts/run_experiment.py --experiment quantum_gnn_multimodal --max_epochs 50
    python scripts/run_experiment.py --experiment classical_multimodal --batch_size 8 --device cuda
    python scripts/run_experiment.py --experiment cnn_only --resume --checkpoint checkpoints/cnn_only/checkpoint_epoch004_val_auc_mean=0.5500.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a single experiment variant for the Quantum-GNN CXR project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiment", "-e",
        required=True,
        help="Experiment name. Choices: cnn_only, vit_only, quantum_cnn, "
             "mlp_clinical_only, gnn_only, classical_multimodal, "
             "quantum_gnn_multimodal, disease_specific_fusion, shared_fusion",
    )
    p.add_argument("--max_epochs",  type=int, default=50,    help="Max training epochs.")
    p.add_argument("--batch_size",  type=int, default=None,  help="Override batch size.")
    p.add_argument("--device",      type=str, default="cpu", help="'cpu' or 'cuda'.")
    p.add_argument("--smoke_test",  action="store_true",
                   help="Quick test: 100 samples, 2 epochs. Must complete in <60s on CPU.")
    p.add_argument("--resume",      action="store_true", help="Resume from checkpoint.")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to checkpoint file for --resume.")
    p.add_argument("--config",      type=str, default="config/config.yaml",
                   help="Path to config YAML.")
    p.add_argument("--output_dir",  type=str, default="results",
                   help="Directory to write results.")
    p.add_argument("--split_dir",   type=str, default=None,
                   help="Path to split CSVs. Defaults to config splits.split_dir.")
    p.add_argument("--no_hf",       action="store_true",
                   help="Skip HF dataset load (for offline / dummy-data tests).")
    return p.parse_args()


# ── Load config ───────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.warning("Config not found at %s; using defaults.", cfg_path)
        return {}
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    # Flatten nested config into a single-level dict for the factory
    flat: dict = {}
    # Training section
    flat.update(raw.get("training", {}))
    # Model section
    model_cfg = raw.get("model", {})
    flat["image_backbone"] = model_cfg.get("image_backbone", "efficientnet_b0")
    flat["pretrained"]     = model_cfg.get("pretrained", False)
    flat["num_classes"]    = raw.get("dataset", {}).get("num_classes", 14)
    flat["quantum_enabled"] = model_cfg.get("quantum", {}).get("enabled", False)
    flat["quantum_output_dim"] = model_cfg.get("quantum", {}).get("output_dim", 64)
    flat["log_dir"]        = raw.get("paths", {}).get("logs", "logs")
    flat["checkpoint_dir"] = raw.get("training", {}).get("checkpoint_dir", "checkpoints")
    flat["split_dir"]      = raw.get("splits", {}).get("split_dir", "data/splits")
    flat["image_size"]     = raw.get("image", {}).get("size", 224)
    flat["fused_dim"]      = 128    # CPU-friendly
    return flat


# ── Smoke-test dataframe ───────────────────────────────────────────────────────

def _make_smoke_df(hf_dataset, n: int = 100):
    """Build a minimal index DataFrame from the first n rows of hf_dataset."""
    import pandas as pd
    rows = []
    for i in range(min(n, len(hf_dataset))):
        row = hf_dataset[i]
        rows.append({
            "hf_index":  i,
            "patient_id": f"P{i:05d}",
            "scan_id":    i,
            "age":        50,
            "sex":        "M",
            "labels_raw": row.get("labels", [0]),
        })
    return pd.DataFrame(rows)


# ── Dummy DataLoader (for --no_hf smoke tests) ────────────────────────────────

def _make_dummy_loader(n_batches: int = 4, batch_size: int = 2):
    """Return a simple list of fake batches (no real DataLoader needed)."""
    import torch
    batches = []
    for _ in range(n_batches):
        batches.append({
            "image": torch.zeros(batch_size, 3, 224, 224),
            "label": torch.zeros(batch_size, 14),
            "meta":  torch.zeros(batch_size, 2),
        })
    return batches


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    t0     = time.time()

    # ── Config ───────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    split_dir = args.split_dir or cfg.get("split_dir", "data/splits")

    logger.info("=" * 60)
    logger.info("Experiment : %s", args.experiment)
    logger.info("Device     : %s", args.device)
    logger.info("Max epochs : %s", "2 (smoke)" if args.smoke_test else args.max_epochs)
    logger.info("=" * 60)

    # ── Import experiment runner ──────────────────────────────────────
    from src.training.experiment import Experiment

    exp = Experiment(name=args.experiment, cfg=cfg, device=args.device)

    # ── Data ─────────────────────────────────────────────────────────
    if args.no_hf:
        # Offline / CI mode: use dummy tensors wrapped in a fake loader
        logger.warning("--no_hf: using dummy data (no real images)")
        from src.training.trainer import Trainer
        from src.training.losses import WeightedBCELoss
        from src.models.model_factory import build_model
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        model = build_model(
            args.experiment, cfg=cfg, pretrained=False, device=args.device
        ).to(args.device)

        import torch
        criterion   = WeightedBCELoss()
        optimizer   = AdamW(model.parameters(), lr=1e-4)
        scheduler   = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)

        from src.training.trainer import EarlyStopping
        from pathlib import Path as _P
        log_dir  = _P(cfg.get("log_dir", "logs")) / args.experiment
        csv_path = str(log_dir / "metrics.csv")

        trainer = Trainer(
            model=model, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler,
            device=args.device, csv_path=csv_path,
        )
        dummy_loader = _make_dummy_loader(
            n_batches=2 if args.smoke_test else 4,
            batch_size=cfg.get("batch_size", 2),
        )

        n_epochs = 2 if args.smoke_test else args.max_epochs
        for epoch in range(n_epochs):
            trainer.model.train()
            for batch in dummy_loader:
                images = batch["image"].to(args.device)
                labels = batch["label"].to(args.device)
                meta   = batch["meta"].to(args.device)
                optimizer.zero_grad()
                logits = trainer._model_forward(images, meta, return_logits=True)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            logger.info("Epoch %d/%d done (dummy)", epoch + 1, n_epochs)

        logger.info("Smoke test with dummy data complete in %.1fs", time.time() - t0)
        return

    # ── Real data path ────────────────────────────────────────────────
    try:
        from datasets import load_from_disk, load_dataset  # type: ignore
    except ImportError:
        logger.error("HuggingFace 'datasets' not installed. Use --no_hf for offline mode.")
        sys.exit(1)

    import pandas as pd

    # Load split CSVs
    split_path = Path(split_dir)
    if not split_path.exists():
        logger.error(
            "Split directory '%s' not found. "
            "Run scripts/build_splits.py first, or use --no_hf for offline mode.",
            split_dir,
        )
        sys.exit(1)

    train_df = pd.read_csv(split_path / "train.csv")
    val_df   = pd.read_csv(split_path / "val.csv")
    test_df  = pd.read_csv(split_path / "test.csv")

    if args.smoke_test:
        logger.info("Smoke test: truncating splits to 100 samples each")
        train_df = train_df.head(100)
        val_df   = val_df.head(50)
        test_df  = test_df.head(50)

    # Load HF dataset
    hf_cache = cfg.get("cache_dir", "data/raw")
    hf_repo  = cfg.get("hf_repo", "chehablab/NIHChestXR")
    logger.info("Loading HF dataset (cache=%s) …", hf_cache)
    try:
        hf_dataset = load_dataset(hf_repo, split="train", cache_dir=hf_cache)
    except Exception as e:
        logger.error("Failed to load HF dataset: %s", e)
        logger.error("Use --no_hf to run with dummy data offline.")
        sys.exit(1)

    # Setup + run
    exp.setup(train_df, val_df, test_df, hf_dataset)

    # Optional: resume from checkpoint
    if args.resume and args.checkpoint:
        from src.training.trainer import CheckpointManager
        start_epoch = CheckpointManager.load(
            args.checkpoint, exp.model,
            exp.trainer.optimizer, exp.trainer.scheduler,
            device=args.device,
        )
        logger.info("Resumed from epoch %d", start_epoch)

    val_metrics  = exp.run(max_epochs=args.max_epochs, smoke_test=args.smoke_test)
    test_metrics = exp.evaluate_test()
    exp.save_results(args.output_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("DONE  (%s)  elapsed=%.1fs", args.experiment, elapsed)
    logger.info("Val  AUC=%.4f | F1=%.4f",
                val_metrics.get("auc_mean",  float("nan")),
                val_metrics.get("f1_macro",  float("nan")))
    logger.info("Test AUC=%.4f | F1=%.4f",
                test_metrics.get("auc_mean", float("nan")),
                test_metrics.get("f1_macro", float("nan")))
    logger.info("=" * 60)

    if args.smoke_test and elapsed > 60:
        logger.warning("Smoke test took %.1fs > 60s target!", elapsed)


if __name__ == "__main__":
    main()
