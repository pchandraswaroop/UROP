"""
checkpoint_loader.py — Robust Model Checkpoint Loader
=====================================================
Manages checkpoint discovery, validation, and weight restoration for
MultiModalCXRModel architectures.

Strict Mode Guarantees:
  - Real Inference requires a verified, compatible checkpoint.
  - Incompatible or corrupted checkpoints raise CheckpointError with clear diagnostics.
  - Never silently falls back to random weights during real inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Raised when a checkpoint is missing, invalid, or architecturally incompatible."""
    pass


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

        return available

    @classmethod
    def validate_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate whether a checkpoint file exists and is structurally compatible.

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
            metadata = {
                "epoch": ckpt.get("epoch", "unknown"),
                "metrics": ckpt.get("metrics", {}),
                "has_optimizer": "optimizer" in ckpt,
            }
        elif isinstance(ckpt, dict):
            # Raw state dict
            state_dict = ckpt
            metadata = {"epoch": "raw_state_dict", "metrics": {}, "has_optimizer": False}
        else:
            return False, f"Unexpected checkpoint type: {type(ckpt).__name__}", None

        # If a reference model is provided, check key overlap
        if model is not None:
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())

            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys

            # If more than 50% of model keys are missing, reject as incompatible
            if len(missing) > len(model_keys) * 0.5:
                return (
                    False,
                    f"Checkpoint architecture mismatch: {len(missing)} missing keys out of {len(model_keys)}",
                    metadata,
                )

        return True, "Checkpoint is valid and compatible.", metadata

    @classmethod
    def load_weights(
        cls,
        checkpoint_path: str | Path,
        model: nn.Module,
        device: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load weights into model. Raises CheckpointError on failure."""
        is_valid, msg, meta = cls.validate_checkpoint(checkpoint_path, model=model, device=device)
        if not is_valid:
            raise CheckpointError(f"Cannot load invalid checkpoint: {msg}")

        path = Path(checkpoint_path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt

        try:
            model.load_state_dict(state_dict, strict=strict)
            model.eval()
            logger.info("Successfully loaded checkpoint weights from %s", path)
            return meta or {}
        except Exception as e:
            raise CheckpointError(f"Error loading state_dict into model: {str(e)}") from e
