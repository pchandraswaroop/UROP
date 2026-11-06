"""
trainer.py — Training Loop with Checkpointing and Early Stopping
================================================================
Supports:
  - Mixed precision training (AMP, CUDA only)
  - Checkpoint saving (top-k by val AUC)
  - Training resumption from checkpoint
  - Per-epoch metric logging to CSV
  - Early stopping with best-weight restoration
  - GNN integration (graph passed each epoch)
  - Clean logit/prob split: model returns logits during training,
    probs during eval (no sigmoid → logit round-trip)
"""

from __future__ import annotations

import copy
import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.training.metrics import MetricAccumulator, format_metrics_table

logger = logging.getLogger(__name__)


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience:            Number of epochs to wait after last improvement.
        min_delta:           Minimum change to qualify as an improvement.
        mode:                "max" (higher is better) or "min" (lower is better).
        restore_best_weights: If True, restore model weights from the best epoch
                              when stopping.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        restore_best_weights: bool = True,
    ) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.restore_best_weights = restore_best_weights

        self._counter      = 0
        self._best_score: Optional[float] = None
        self._best_weights = None
        self.stopped_epoch = 0
        self.best_epoch = 0

        self._is_better = (lambda a, b: a - b > min_delta) if mode == "max" \
                          else (lambda a, b: b - a > min_delta)

    # ------------------------------------------------------------------

    def __call__(self, score: float, model: nn.Module, epoch: int = 0) -> bool:
        """Check whether training should stop.

        Args:
            score: Current epoch metric value (e.g. val_auc_mean).
            model: The model being trained.
            epoch: Current epoch index.

        Returns:
            True if training should stop, False otherwise.
        """
        if self._best_score is None or self._is_better(score, self._best_score):
            self._best_score   = score
            self._counter      = 0
            self.best_epoch    = epoch
            if self.restore_best_weights:
                self._best_weights = copy.deepcopy(model.state_dict())
            logger.debug("EarlyStopping: new best score=%.6f at epoch %d", score, epoch)
            return False

        self._counter += 1
        logger.debug(
            "EarlyStopping: no improvement (%.6f <= %.6f), patience %d/%d",
            score, self._best_score, self._counter, self.patience,
        )
        if self._counter >= self.patience:
            if self.restore_best_weights and self._best_weights is not None:
                model.load_state_dict(self._best_weights)
                logger.info(
                    "EarlyStopping: restored best weights (score=%.6f)", self._best_score
                )
            return True
        return False

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    def reset(self) -> None:
        self._counter      = 0
        self._best_score   = None
        self._best_weights = None


# ── Checkpoint ────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Manages saving and loading of model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str,
        save_top_k: int = 3,
        monitor: str = "val_auc_mean",
        mode: str = "max",
    ) -> None:
        self.dir        = Path(checkpoint_dir)
        self.save_top_k = save_top_k
        self.monitor    = monitor
        self.better     = (lambda a, b: a > b) if mode == "max" else (lambda a, b: a < b)
        self._history: list = []   # list of (score, path)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch: int, metrics: dict) -> Optional[str]:
        score = metrics.get(self.monitor, float("-inf") if self.better(1, 0) else float("inf"))
        path  = self.dir / f"checkpoint_epoch{epoch:03d}_{self.monitor}={score:.4f}.pt"

        torch.save({
            "epoch":      epoch,
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler else None,
            "metrics":    metrics,
        }, path)

        self._history.append((score, path))
        self._history.sort(key=lambda x: x[0], reverse=self.better(1, 0))

        # Remove worst checkpoints beyond top-k
        while len(self._history) > self.save_top_k:
            _, old_path = self._history.pop(-1)
            if old_path.exists():
                old_path.unlink()
                logger.debug("Removed old checkpoint: %s", old_path)

        logger.info("Checkpoint saved: %s (score=%.4f)", path.name, score)
        return str(path)

    @property
    def best_path(self) -> Optional[Path]:
        """Return Path to the highest-scoring checkpoint recorded."""
        if self._history:
            return self._history[0][1]
        return None

    @staticmethod
    def load(path: str, model, optimizer=None, scheduler=None, device="cpu") -> int:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        epoch = ckpt.get("epoch", 0)
        logger.info("Loaded checkpoint: %s (epoch %d)", path, epoch)
        return epoch


# ── CSV Logger ────────────────────────────────────────────────────────────────

class MetricsCSVLogger:
    """Appends per-epoch metrics to a CSV file.

    Columns: epoch, train_loss, val_loss, val_auc_mean, val_ap_mean, val_f1_macro
    """

    COLUMNS = ["epoch", "train_loss", "val_loss", "val_auc_mean", "val_ap_mean", "val_f1_macro"]

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def log(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        row = {
            "epoch":        epoch,
            "train_loss":   round(train_metrics.get("loss", float("nan")), 6),
            "val_loss":     round(val_metrics.get("loss",  float("nan")), 6),
            "val_auc_mean": round(val_metrics.get("auc_mean", float("nan")), 6),
            "val_ap_mean":  round(val_metrics.get("ap_mean",  float("nan")), 6),
            "val_f1_macro": round(val_metrics.get("f1_macro", float("nan")), 6),
        }

        write_header = not self._initialized and not self.path.exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._initialized = True


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """Training loop manager.

    Args:
        model:      The full MultiModalCXRModel (or any nn.Module).
        criterion:  Loss function (WeightedBCELoss). Expects logits.
        optimizer:  Optimizer (AdamW).
        scheduler:  LR scheduler.
        device:     "cpu" or "cuda".
        use_amp:    Use automatic mixed precision (CUDA only).
        ckpt_mgr:   CheckpointManager instance.
        csv_path:   Path to per-epoch CSV log file. None = no logging.
        early_stopping: EarlyStopping instance. None = no early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        scheduler,
        device: str = "cpu",
        use_amp: bool = False,
        ckpt_mgr: Optional[CheckpointManager] = None,
        csv_path: Optional[str] = None,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> None:
        self.model          = model.to(device)
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.device         = device
        self.use_amp        = use_amp and (device == "cuda")
        self.ckpt_mgr       = ckpt_mgr
        self.early_stopping = early_stopping
        self.csv_logger     = MetricsCSVLogger(csv_path) if csv_path else None
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.best_epoch: int = 0

    @property
    def best_checkpoint_path(self) -> Optional[str]:
        """Return the filesystem path to the best saved checkpoint, if any."""
        if self.ckpt_mgr and self.ckpt_mgr.best_path:
            return str(self.ckpt_mgr.best_path)
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _model_forward(
        self,
        images: torch.Tensor,
        meta: torch.Tensor,
        return_logits: bool = True,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Call model.forward with graph tensors and return_logits if supported."""
        try:
            return self.model(
                images,
                meta,
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                return_logits=return_logits,
            )
        except TypeError:
            try:
                return self.model(images, meta, return_logits=return_logits)
            except TypeError:
                out = self.model(images, meta)
                if return_logits:
                    return torch.logit(out.clamp(1e-6, 1 - 1e-6))
                return out

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one training epoch.

        Uses model(return_logits=True) → passes logits directly to loss.
        Uses sigmoid(logits) for metric accumulation.
        """
        self.model.train()
        acc        = MetricAccumulator()
        total_loss_sum = 0.0
        total_samples  = 0
        n_batches      = 0

        if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            meta   = batch["meta"].to(self.device)

            graph_x, graph_edge_index = None, None
            if "patient_id" in batch:
                from src.graph.graph_builder import build_batch_scan_graph
                graph_x, graph_edge_index = build_batch_scan_graph(meta, batch["patient_id"], add_self_loops=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self._model_forward(
                    images, meta, return_logits=True,
                    graph_x=graph_x, graph_edge_index=graph_edge_index,
                )
                loss   = self.criterion(logits, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            probs = torch.sigmoid(logits.detach())
            acc.update(probs, labels.detach())
            b_size = images.shape[0]
            total_loss_sum += loss.item() * b_size
            total_samples  += b_size
            n_batches      += 1

        if self.scheduler is not None:
            self.scheduler.step()

        metrics = acc.compute()
        metrics["loss"]  = total_loss_sum / max(total_samples, 1)
        metrics["epoch"] = epoch
        return metrics

    # ------------------------------------------------------------------
    # Evaluation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_epoch(
        self,
        loader: DataLoader,
        split: str = "val",
        return_raw: bool = False,
    ):
        """Run one evaluation epoch.

        Uses model(return_logits=True) → sigmoid for probs, logits for loss.
        If return_raw=True, returns (metrics, all_probs, all_targets).
        """
        self.model.eval()
        acc        = MetricAccumulator()
        total_loss_sum = 0.0
        total_samples  = 0
        n_batches      = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            meta   = batch["meta"].to(self.device)

            graph_x, graph_edge_index = None, None
            if "patient_id" in batch:
                from src.graph.graph_builder import build_batch_scan_graph
                graph_x, graph_edge_index = build_batch_scan_graph(meta, batch["patient_id"], add_self_loops=True)

            logits = self._model_forward(
                images, meta, return_logits=True,
                graph_x=graph_x, graph_edge_index=graph_edge_index,
            )
            loss   = self.criterion(logits, labels)
            probs  = torch.sigmoid(logits)

            acc.update(probs, labels)
            b_size = images.shape[0]
            total_loss_sum += loss.item() * b_size
            total_samples  += b_size
            n_batches      += 1

        metrics = acc.compute()
        metrics["loss"] = total_loss_sum / max(total_samples, 1)

        if return_raw:
            all_probs, all_targets = acc.get_raw()
            return metrics, all_probs, all_targets
        return metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        start_epoch: int = 0,
    ) -> List[Dict[str, float]]:
        """Full training loop with early stopping and CSV logging.

        Args:
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
            max_epochs:   Maximum number of epochs to train.
            start_epoch:  Epoch index to start from (for resumption).

        Returns:
            List of per-epoch metric dicts (combined train + val keys).
        """
        logger.info("Starting training: epochs %d → %d", start_epoch, max_epochs)
        history: List[Dict[str, float]] = []

        for epoch in range(start_epoch, max_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics   = self.eval_epoch(val_loader, "val")

            # Merge for history
            combined = {f"train_{k}": v for k, v in train_metrics.items()}
            combined.update({f"val_{k}": v for k, v in val_metrics.items()})
            combined["epoch"] = epoch
            history.append(combined)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | val_f1=%.4f",
                epoch + 1, max_epochs,
                train_metrics.get("loss", float("nan")),
                val_metrics.get("loss",   float("nan")),
                val_metrics.get("auc_mean", float("nan")),
                val_metrics.get("f1_macro", float("nan")),
            )

            # CSV logging
            if self.csv_logger is not None:
                self.csv_logger.log(epoch, train_metrics, val_metrics)

            # Checkpoint
            if self.ckpt_mgr is not None:
                self.ckpt_mgr.save(
                    self.model, self.optimizer, self.scheduler,
                    epoch,
                    {"val_auc_mean": val_metrics.get("auc_mean", 0.0),
                     **val_metrics},
                )

            # Early stopping
            if self.early_stopping is not None:
                monitor_val = val_metrics.get("auc_mean", float("nan"))
                if not (monitor_val != monitor_val):  # nan check
                    should_stop = self.early_stopping(monitor_val, self.model, epoch=epoch)
                    self.best_epoch = self.early_stopping.best_epoch
                    if should_stop:
                        logger.info(
                            "Early stopping triggered at epoch %d "
                            "(best val_auc_mean=%.4f at epoch %d)",
                            epoch + 1,
                            self.early_stopping.best_score or float("nan"),
                            self.best_epoch + 1,
                        )
                        break

        # Ensure model holds the best validation checkpoint weights
        if self.early_stopping and self.early_stopping.restore_best_weights and self.early_stopping._best_weights is not None:
            self.model.load_state_dict(self.early_stopping._best_weights)
            logger.info(
                "Restored best model weights from epoch %d (best val score: %.4f)",
                self.best_epoch + 1,
                self.early_stopping.best_score or float("nan"),
            )
        elif self.ckpt_mgr and self.ckpt_mgr.best_path:
            CheckpointManager.load(str(self.ckpt_mgr.best_path), self.model, device=self.device)
            logger.info("Restored best model weights from checkpoint: %s", self.ckpt_mgr.best_path.name)

        logger.info("Training complete.")
        return history
