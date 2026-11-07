"""
experiment.py — Clean Experiment Runner
========================================
Encapsulates the full lifecycle of a single training experiment:
  setup → run → evaluate_test → save_results

Usage
-----
    exp = Experiment("cnn_only", cfg, device="cpu")
    exp.setup(train_df, val_df, test_df, hf_dataset)
    metrics = exp.run(max_epochs=50)
    test_metrics = exp.evaluate_test()
    exp.save_results("results/")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.model_factory import build_model
from src.training.losses import WeightedBCELoss, compute_pos_weights
from src.training.trainer import CheckpointManager, EarlyStopping, Trainer

logger = logging.getLogger(__name__)


class Experiment:
    """Manages the full lifecycle of one experiment variant.

    Args:
        name:   Experiment name (must match a key in model_factory).
        cfg:    Config dict (merged from config.yaml + CLI overrides).
        device: "cpu" or "cuda".
    """

    def __init__(self, name: str, cfg: Dict[str, Any], device: str = "cpu") -> None:
        self.name   = name
        self.cfg    = cfg
        self.device = device

        self.model: Optional[nn.Module] = None
        self.trainer: Optional[Trainer] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader:   Optional[DataLoader] = None
        self.test_loader:  Optional[DataLoader] = None

        self.mode: Optional[str] = cfg.get("mode", None)
        self.train_scan_count: int = 0
        self.val_scan_count: int = 0
        self.test_scan_count: int = 0
        self.train_patient_count: int = 0
        self.val_patient_count: int = 0
        self.test_patient_count: int = 0

        self.train_history: list = []
        self.val_metrics:   Dict[str, float] = {}
        self.test_metrics:  Dict[str, float] = {}
        self.start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(
        self,
        train_df,
        val_df,
        test_df,
        hf_dataset,
        mode: Optional[str] = None,
    ) -> None:
        """Build model, dataloaders, optimizer, scheduler, and trainer.

        Args:
            train_df:   Training split DataFrame (with hf_index, metadata columns).
            val_df:     Validation split DataFrame.
            test_df:    Test split DataFrame.
            hf_dataset: HuggingFace Dataset (full, used for image lookup).
        """
        from src.dataset.dataloader import get_dataloader

        logger.info("Setting up experiment '%s' on device=%s", self.name, self.device)

        if mode is not None:
            self.mode = mode
        if hasattr(train_df, "__len__"):
            self.train_scan_count = len(train_df)
            if hasattr(train_df, "columns") and "patient_id" in train_df.columns:
                self.train_patient_count = int(train_df["patient_id"].nunique())
        if hasattr(val_df, "__len__"):
            self.val_scan_count = len(val_df)
            if hasattr(val_df, "columns") and "patient_id" in val_df.columns:
                self.val_patient_count = int(val_df["patient_id"].nunique())
        if hasattr(test_df, "__len__"):
            self.test_scan_count = len(test_df)
            if hasattr(test_df, "columns") and "patient_id" in test_df.columns:
                self.test_patient_count = int(test_df["patient_id"].nunique())

        # ── DataLoaders ──────────────────────────────────────────────
        batch_size  = self.cfg.get("batch_size", 4)
        num_workers = self.cfg.get("num_workers", 0)
        image_size  = self.cfg.get("image_size", 224)
        pin_memory  = self.device == "cuda"

        self.train_loader = get_dataloader(
            split="train", index_df=train_df, hf_dataset=hf_dataset,
            batch_size=batch_size, num_workers=num_workers,
            image_size=image_size, pin_memory=pin_memory,
        )
        self.val_loader = get_dataloader(
            split="val",   index_df=val_df,   hf_dataset=hf_dataset,
            batch_size=batch_size, num_workers=num_workers,
            image_size=image_size, pin_memory=pin_memory,
        )
        self.test_loader = get_dataloader(
            split="test",  index_df=test_df,  hf_dataset=hf_dataset,
            batch_size=batch_size, num_workers=num_workers,
            image_size=image_size, pin_memory=pin_memory,
        )

        # ── Model ────────────────────────────────────────────────────
        pretrained = self.cfg.get("pretrained", False)
        self.model = build_model(
            experiment=self.name,
            cfg=self.cfg,
            pretrained=pretrained,
            device=self.device,
        )
        self.model = self.model.to(self.device)

        # ── Loss ─────────────────────────────────────────────────────
        if self.cfg.get("use_weighted_loss", True):
            pos_weights = compute_pos_weights(
                train_df,
                num_classes=self.cfg.get("num_classes", 14),
                cap=self.cfg.get("pos_weight_cap", 10.0),
            ).to(self.device)
        else:
            pos_weights = None
        criterion = WeightedBCELoss(pos_weights=pos_weights)

        # ── Optimizer & Scheduler ────────────────────────────────────
        lr           = self.cfg.get("learning_rate", 1e-4)
        weight_decay = self.cfg.get("weight_decay", 1e-5)
        max_epochs   = self.cfg.get("max_epochs", 50)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.01)

        # ── Early Stopping ───────────────────────────────────────────
        early_stopping = EarlyStopping(
            patience=self.cfg.get("early_stopping_patience", 10),
            min_delta=self.cfg.get("early_stopping_min_delta", 1e-4),
            mode="max",
            restore_best_weights=True,
        )

        # ── Checkpoint Manager ───────────────────────────────────────
        ckpt_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints")) / self.name
        ckpt_mgr = CheckpointManager(
            checkpoint_dir=str(ckpt_dir),
            save_top_k=self.cfg.get("save_top_k", 3),
            monitor=self.cfg.get("monitor_metric", "val_auc_mean"),
            mode=self.cfg.get("monitor_mode", "max"),
        )

        # ── CSV Logger path ──────────────────────────────────────────
        log_dir  = Path(self.cfg.get("log_dir", "logs")) / self.name
        csv_path = str(log_dir / "metrics.csv")

        # ── Trainer ──────────────────────────────────────────────────
        self.trainer = Trainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            use_amp=self.cfg.get("amp", False),
            ckpt_mgr=ckpt_mgr,
            csv_path=csv_path,
            early_stopping=early_stopping,
        )

        logger.info(
            "Experiment '%s' ready: %d train / %d val / %d test samples",
            self.name,
            len(self.train_loader.dataset),
            len(self.val_loader.dataset),
            len(self.test_loader.dataset),
        )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        max_epochs: int,
        smoke_test: bool = False,
    ) -> Dict[str, float]:
        """Train the model.

        Args:
            max_epochs: Maximum epochs (overrides cfg if provided).
            smoke_test: If True, run only 2 epochs regardless of max_epochs.

        Returns:
            Final validation metrics dict.
        """
        if self.trainer is None:
            raise RuntimeError("Call setup() before run().")

        actual_epochs = 2 if smoke_test else max_epochs
        self.start_time = time.time()

        logger.info(
            "Running experiment '%s' for %d epochs (smoke_test=%s)",
            self.name, actual_epochs, smoke_test,
        )

        self.train_history = self.trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            max_epochs=actual_epochs,
        )

        self.training_time_seconds = time.time() - self.start_time
        self.best_epoch = getattr(self.trainer, "best_epoch", 0)
        logger.info(
            "Experiment '%s' finished in %.1fs (best epoch: %d)",
            self.name, self.training_time_seconds, self.best_epoch + 1
        )

        # Extract best val metrics from the history entry of best_epoch
        if self.train_history:
            best_idx = self.best_epoch if (0 <= self.best_epoch < len(self.train_history)) else -1
            best_entry = self.train_history[best_idx]
            self.val_metrics = {
                k.replace("val_", ""): v
                for k, v in best_entry.items()
                if k.startswith("val_")
            }
        self.val_metrics["best_epoch"] = self.best_epoch
        self.val_metrics["training_time_seconds"] = self.training_time_seconds
        return self.val_metrics

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_checkpoint_path(self) -> Optional[str]:
        """Return the path to the best model checkpoint from training."""
        if self.trainer:
            return self.trainer.best_checkpoint_path
        return None

    # ------------------------------------------------------------------
    # evaluate_test
    # ------------------------------------------------------------------

    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate the trained model on the test split.

        Returns:
            Test metrics dict.
        """
        if self.trainer is None or self.test_loader is None:
            raise RuntimeError("Call setup() and run() before evaluate_test().")

        logger.info("Evaluating experiment '%s' on test split …", self.name)
        res = self.trainer.eval_epoch(self.test_loader, split="test", return_raw=True)
        if isinstance(res, tuple):
            self.test_metrics, self.test_probs, self.test_targets = res
        else:
            self.test_metrics = res
            self.test_probs, self.test_targets = None, None

        logger.info(
            "Test | auc_mean=%.4f | f1_macro=%.4f",
            self.test_metrics.get("auc_mean", float("nan")),
            self.test_metrics.get("f1_macro", float("nan")),
        )
        return self.test_metrics

    # ------------------------------------------------------------------
    # save_results
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str) -> None:
        """Save metrics CSV and config JSON to output_dir/experiment_name/.

        Args:
            output_dir: Root directory to write results into.
        """
        out = Path(output_dir) / self.name
        out.mkdir(parents=True, exist_ok=True)

        # Config snapshot
        cfg_path = out / "config.json"
        with open(cfg_path, "w") as f:
            json.dump({"experiment": self.name, **self.cfg}, f, indent=2, default=str)

        # Val metrics
        val_path = out / "val_metrics.json"
        with open(val_path, "w") as f:
            json.dump(self.val_metrics, f, indent=2, default=str)

        # Test metrics
        test_path = out / "test_metrics.json"
        with open(test_path, "w") as f:
            json.dump(self.test_metrics, f, indent=2, default=str)

        # Test predictions (for downstream analysis and visualizations)
        if getattr(self, "test_probs", None) is not None and getattr(self, "test_targets", None) is not None:
            np.savez_compressed(
                out / "test_predictions.npz",
                probs=self.test_probs,
                targets=self.test_targets,
            )

        # Training history
        history_path = out / "train_history.json"
        with open(history_path, "w") as f:
            json.dump(self.train_history, f, indent=2, default=str)

        # Experiment metadata snapshot
        metadata = {
            "experiment": self.name,
            "mode": getattr(self, "mode", self.cfg.get("mode", "unspecified")),
            "device": str(self.device),
            "train_scans": getattr(self, "train_scan_count", len(self.train_loader.dataset) if self.train_loader else 0),
            "val_scans": getattr(self, "val_scan_count", len(self.val_loader.dataset) if self.val_loader else 0),
            "test_scans": getattr(self, "test_scan_count", len(self.test_loader.dataset) if self.test_loader else 0),
            "train_patients": getattr(self, "train_patient_count", 0),
            "val_patients": getattr(self, "val_patient_count", 0),
            "test_patients": getattr(self, "test_patient_count", 0),
            "epochs_trained": len(self.train_history),
            "best_epoch": getattr(self, "best_epoch", 0) + 1,
            "training_time_seconds": getattr(self, "training_time_seconds", 0.0),
            "best_checkpoint_path": self.best_checkpoint_path,
            "val_auc_mean": self.val_metrics.get("auc_mean", None),
            "val_f1_macro": self.val_metrics.get("f1_macro", None),
            "test_auc_mean": self.test_metrics.get("auc_mean", None),
            "test_f1_macro": self.test_metrics.get("f1_macro", None),
        }
        meta_path = out / "experiment_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Results saved to %s", out)
