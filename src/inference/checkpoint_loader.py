"""
checkpoint_loader.py — Robust Model Checkpoint Loader
=====================================================
Manages checkpoint discovery, validation, and weight restoration for
MultiModalCXRModel architectures.

Strict Mode Guarantees:
  - Real Inference requires a verified, compatible checkpoint.
  - Strict key and tensor-shape compatibility required (strict=True).
  - Incompatible or corrupted checkpoints raise CheckpointError with clear diagnostics.
  - Never silently falls back to random weights during real inference.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Raised when a checkpoint is missing, invalid, or architecturally incompatible."""
    pass


def validate_auroc(score: Any) -> Tuple[bool, Optional[float], str]:
    """Validate whether an AUROC score is numeric, finite, and in [0.0, 1.0].

    Accepts:
        0.0 (valid AUROC, e.g. inverted/adversarial classifier).
        Any float in [0.0, 1.0].

    Rejects:
        None, strings, bools, NaN, +inf, -inf, < 0.0, > 1.0.
    """
    if score is None or isinstance(score, bool):
        return False, None, "AUROC cannot be None or boolean"
    try:
        val = float(score)
    except (ValueError, TypeError):
        return False, None, f"AUROC value is non-numeric: {score!r}"

    if not math.isfinite(val):
        return False, None, f"AUROC value is not finite: {val}"

    if val < 0.0 or val > 1.0:
        return False, None, f"AUROC value out of range [0.0, 1.0]: {val}"

    return True, val, "Valid AUROC"


class ModelCheckpointLoader:
    """Discovers, validates, and loads trained model checkpoints."""

    @staticmethod
    def list_available_checkpoints(checkpoint_dir: str = "checkpoints") -> Dict[str, List[Path]]:
        """Scan checkpoint directory and return map of experiment_name -> list of checkpoint paths."""
        ckpt_dir = Path(checkpoint_dir)
        available: Dict[str, List[Path]] = {}
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            return available

        for path in ckpt_dir.rglob("*.pt"):
            exp_name = path.parent.name if path.parent != ckpt_dir else "general"
            available.setdefault(exp_name, []).append(path)

        for path in ckpt_dir.rglob("*.pth"):
            exp_name = path.parent.name if path.parent != ckpt_dir else "general"
            available.setdefault(exp_name, []).append(path)

        # Sort each experiment's checkpoints: prioritize 'medium' mode and highest validation score
        for exp_name in available:
            def sort_key(p: Path):
                is_med = 1 if "medium" in str(p).lower() else 0
                is_valid, score, _ = ModelCheckpointLoader.extract_checkpoint_auroc(p)
                score_val = score if is_valid and score is not None else -1.0
                return (is_med, score_val, p.stat().st_mtime if p.exists() else 0)
            available[exp_name].sort(key=sort_key, reverse=True)

        return available

    @classmethod
    def extract_checkpoint_auroc(cls, checkpoint_path: str | Path) -> Tuple[bool, Optional[float], str]:
        """Extract and validate the validation AUROC score from a checkpoint file or filename.

        Returns:
            Tuple of (is_valid, float_score_or_None, reason_string)
        """
        path = Path(checkpoint_path)
        if not path.exists():
            return False, None, f"File does not exist: {path}"

        score_candidate = None
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                metrics = ckpt.get("metrics", {})
                if isinstance(metrics, dict) and "val_auc_mean" in metrics:
                    score_candidate = metrics["val_auc_mean"]
                elif "val_auc_mean" in ckpt:
                    score_candidate = ckpt["val_auc_mean"]
        except Exception:
            pass

        if score_candidate is None:
            m = re.search(r"val_auc_mean=([0-9\.\-eE]+)", path.name)
            if m:
                score_candidate = m.group(1)

        if score_candidate is None:
            return False, None, f"No AUROC score found in checkpoint or filename: {path.name}"

        return validate_auroc(score_candidate)

    @classmethod
    def find_best_checkpoint(
        cls,
        experiment: str,
        checkpoint_dir: str | Path = "checkpoints",
    ) -> Optional[Path]:
        """Find the checkpoint with the highest valid validation AUROC for an experiment.

        Rejects any checkpoint with invalid, missing, non-finite, or out-of-range AUROC.
        Returns None if no verified checkpoint exists.
        """
        ckpt_root = Path(checkpoint_dir)
        if not ckpt_root.exists():
            return None

        # Search prioritized mode directories first (medium > direct > smoke > full)
        search_dirs = [
            ckpt_root / "medium" / experiment,
            ckpt_root / experiment,
            ckpt_root / "smoke" / experiment,
            ckpt_root / "full" / experiment,
            ckpt_root / "full_50" / experiment,
        ]

        searched_files: List[Path] = []
        for d in search_dirs:
            if d.exists() and d.is_dir():
                found = list(d.glob("*.pt")) + list(d.glob("*.pth"))
                if found:
                    searched_files.extend(found)

        # Fallback: recursive scan under checkpoint_dir matching the experiment
        if not searched_files:
            for path in list(ckpt_root.rglob("*.pt")) + list(ckpt_root.rglob("*.pth")):
                if experiment == path.parent.name or experiment in path.name:
                    searched_files.append(path)

        candidates: List[Tuple[float, int, Path]] = []
        for path in searched_files:
            is_valid, score, _ = cls.extract_checkpoint_auroc(path)
            if is_valid and score is not None:
                ep_match = re.search(r"epoch(\d+)", path.name)
                epoch = int(ep_match.group(1)) if ep_match else 0
                candidates.append((score, epoch, path))

        if not candidates:
            return None

        # Sort descending by score, then descending by epoch
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_path = candidates[0][2]
        logger.info("Found best checkpoint for %s: %s (AUROC=%.4f)", experiment, best_path.name, candidates[0][0])
        return best_path

    @classmethod
    def validate_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate whether a checkpoint file exists and is strictly structurally compatible.

        Requires exact state_dict key matching and identical tensor shapes.

        Returns:
            Tuple of (is_valid, message, checkpoint_metadata_dict)
        """
        path = Path(checkpoint_path)
        if not path.exists():
            return False, f"Checkpoint file does not exist: {path}", None
        if not path.is_file():
            return False, f"Path is not a regular file: {path}", None

        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            return False, f"Failed to parse checkpoint file: {str(e)}", None

        # Determine state_dict format
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            metrics = ckpt.get("metrics", {})
            metadata = {
                "epoch": ckpt.get("epoch", "unknown"),
                "metrics": metrics,
                "has_optimizer": "optimizer" in ckpt,
            }
            # If val_auc_mean is present in metrics, validate it
            if isinstance(metrics, dict) and "val_auc_mean" in metrics:
                auc_valid, auc_val, reason = validate_auroc(metrics["val_auc_mean"])
                if not auc_valid:
                    return False, f"Checkpoint contains invalid AUROC value: {reason}", metadata
        elif isinstance(ckpt, dict):
            state_dict = ckpt
            metadata = {"epoch": "raw_state_dict", "metrics": {}, "has_optimizer": False}
        else:
            return False, f"Unexpected checkpoint type: {type(ckpt).__name__}", None

        # Exact key and shape compatibility
        if model is not None:
            model_state = model.state_dict()
            model_keys = set(model_state.keys())
            ckpt_keys = set(state_dict.keys())

            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys

            if len(missing) > 0 or len(unexpected) > 0:
                return (
                    False,
                    f"Checkpoint architecture mismatch: {len(missing)} missing keys, "
                    f"{len(unexpected)} unexpected keys. Strict key parity required.",
                    metadata,
                )

            for k in model_keys:
                if state_dict[k].shape != model_state[k].shape:
                    return (
                        False,
                        f"Tensor shape mismatch for '{k}': expected {model_state[k].shape}, "
                        f"got {state_dict[k].shape}",
                        metadata,
                    )

        return True, "Checkpoint is valid and strictly compatible.", metadata

    @classmethod
    def load_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str | Path,
        device: str = "cpu",
        strict: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate and strictly load checkpoint weights into model.

        Returns:
            Tuple of (success_bool, info_dict)
        """
        is_valid, msg, meta = cls.validate_checkpoint(checkpoint_path, model=model, device=device)
        if not is_valid:
            logger.warning("Checkpoint validation failed for %s: %s", checkpoint_path, msg)
            return False, {"error": msg}

        path = Path(checkpoint_path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt

        try:
            model.load_state_dict(state_dict, strict=strict)
            model.eval()
            return True, meta or {}
        except Exception as e:
            logger.warning("Failed to load state_dict into model: %s", e)
            return False, {"error": str(e)}

    @classmethod
    def load_weights(
        cls,
        checkpoint_path: str | Path,
        model: nn.Module,
        device: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load weights into model strictly. Raises CheckpointError on failure."""
        success, info = cls.load_checkpoint(model, checkpoint_path, device=device, strict=strict)
        if not success:
            raise CheckpointError(f"Cannot load invalid checkpoint: {info.get('error')}")
        logger.info("Successfully loaded checkpoint weights from %s", checkpoint_path)
        return info
