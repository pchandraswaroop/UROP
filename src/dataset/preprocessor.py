"""
preprocessor.py — Image and Metadata Transforms
================================================
Provides separate transform pipelines for train / val / test:

  - Train: augmentation + normalize
  - Val/Test: resize + normalize only

Also provides metadata encoding for the GNN branch:
  - age → z-score normalized float
  - sex → binary float (M=1, F=0)

These are returned as a [2]-dim tensor used as per-patient node features
in the GNN (augmented at graph-build time with 14-dim disease history).
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
from torchvision import transforms
from PIL import Image


# ── Constants ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean for 3-channel
IMAGENET_STD  = [0.229, 0.224, 0.225]  # ImageNet std

# Known population stats for NIH ChestX-ray14 (used for z-score normalisation of age)
# Source: Wang et al. 2017; rough estimates for robust normalisation
NIH_AGE_MEAN = 46.9
NIH_AGE_STD  = 16.6


# ── Image Transforms ─────────────────────────────────────────────────────────

def get_train_transforms(
    image_size: int = 224,
    mean: list = IMAGENET_MEAN,
    std: list  = IMAGENET_STD,
) -> transforms.Compose:
    """Augmentation-heavy transform for training."""
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),      # slight oversample
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),                 # HF images may be L-mode
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
    ])


def get_val_transforms(
    image_size: int = 224,
    mean: list = IMAGENET_MEAN,
    std: list  = IMAGENET_STD,
) -> transforms.Compose:
    """Deterministic transform for validation and test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_transforms(
    split: str,
    image_size: int = 224,
    mean: list = IMAGENET_MEAN,
    std: list  = IMAGENET_STD,
) -> transforms.Compose:
    """Dispatch transform by split name.

    Args:
        split: One of "train", "val", "test".

    Returns:
        Appropriate torchvision Compose pipeline.

    Raises:
        ValueError: If split is not recognised.
    """
    if split == "train":
        return get_train_transforms(image_size, mean, std)
    elif split in ("val", "test"):
        return get_val_transforms(image_size, mean, std)
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")


# ── Metadata Encoding ────────────────────────────────────────────────────────

class MetadataEncoder:
    """Encode patient-level metadata into a feature vector.

    Design
    ------
    Node features for GNN:
      [0] age_normalized   : z-score of patient age
      [1] sex_binary       : 1.0 for M, 0.0 for F, 0.5 for unknown/other

    These 2 features are the only non-label, non-ID metadata available.
    The GNN is therefore given weak but clinically non-leaking node signals.

    Note
    ----
    Do NOT use disease labels as node features at ANY point.
    """

    SEX_MAP = {"M": 1.0, "m": 1.0, "MALE": 1.0,
               "F": 0.0, "f": 0.0, "FEMALE": 0.0}
    SEX_UNKNOWN = 0.5

    def __init__(
        self,
        age_mean: float = NIH_AGE_MEAN,
        age_std: float  = NIH_AGE_STD,
    ) -> None:
        self.age_mean = age_mean
        self.age_std  = age_std

    def encode_row(self, age: int, sex: str) -> torch.Tensor:
        """Encode a single metadata row.

        Args:
            age: Patient age (int). Clamped to [0, 120].
            sex: Sex string from dataset ("M", "F", etc.).

        Returns:
            Float tensor of shape [2].
        """
        age_clamp = max(0, min(120, int(age)))
        age_norm  = (age_clamp - self.age_mean) / (self.age_std + 1e-8)

        sex_clean = str(sex).strip().upper()
        sex_val   = self.SEX_MAP.get(sex_clean, self.SEX_UNKNOWN)

        return torch.tensor([age_norm, sex_val], dtype=torch.float32)

    def encode_batch(
        self,
        ages: list,
        sexes: list,
    ) -> torch.Tensor:
        """Encode a batch of metadata rows.

        Args:
            ages:  List of int ages.
            sexes: List of str sex values.

        Returns:
            Float tensor of shape [N, 2].
        """
        rows = [self.encode_row(a, s) for a, s in zip(ages, sexes)]
        return torch.stack(rows, dim=0)
