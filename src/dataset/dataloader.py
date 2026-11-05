"""
dataloader.py — PyTorch Dataset and DataLoader Factory
=======================================================
Provides:
  - CXRDataset: PyTorch Dataset wrapping HF Dataset rows.
  - get_dataloader(): Factory that builds train / val / test DataLoaders
    from split CSVs.

Design decisions:
  - Images are loaded from the HF Dataset by integer index (fast with parquet).
  - Labels are encoded to [14] float tensors via LabelEncoder.
  - Metadata (age, sex) is encoded via MetadataEncoder for the GNN branch.
  - Patient ID and scan ID are passed through for monitoring only (never as
    model inputs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from datasets import Dataset as HFDataset
except ImportError:  # pragma: no cover
    HFDataset = None  # type: ignore[assignment,misc]  # optional at import time
from PIL import Image

from src.dataset.label_encoder import LabelEncoder, encode_labels, safe_parse_labels
from src.dataset.preprocessor import get_transforms, MetadataEncoder

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────

class CXRDataset(Dataset):
    """PyTorch Dataset for NIH Chest X-ray multi-label classification.

    Args:
        index_df:       DataFrame with columns [hf_index, patient_id, scan_id,
                        age, sex, labels_raw] for this split.
        hf_dataset:     Full HuggingFace Dataset (used for image lookup by index).
        transform:      torchvision transform pipeline to apply to PIL images.
        metadata_encoder: MetadataEncoder for age/sex features.
        return_meta:    If True, include metadata dict in __getitem__ output.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        hf_dataset: HFDataset,
        transform,
        metadata_encoder: Optional[MetadataEncoder] = None,
        return_meta: bool = True,
    ) -> None:
        super().__init__()
        self.df = index_df.reset_index(drop=True)
        self.hf = hf_dataset
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.meta_encoder = metadata_encoder or MetadataEncoder()
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        hf_idx = int(row["hf_index"])

        # ── Image ──────────────────────────────────────────────────────────
        hf_row = self.hf[hf_idx]
        pil_image: Image.Image = hf_row["image"]

        # Apply transform
        image_tensor: torch.Tensor = self.transform(pil_image)  # [3, H, W]

        # ── Labels ─────────────────────────────────────────────────────────
        # labels_raw may be stored as a list of ints or an array; fallback to labels if labels_raw missing
        labels_raw = row["labels_raw"] if "labels_raw" in row else row["labels"]
        parsed_labels = safe_parse_labels(labels_raw)
        label_tensor = self.label_encoder.encode(parsed_labels)  # [14]

        # ── Metadata ───────────────────────────────────────────────────────
        meta_tensor = self.meta_encoder.encode_row(
            age=int(row["age"]),
            sex=str(row["sex"]),
        )  # [2]

        item = {
            "image":  image_tensor,   # [3, 224, 224]
            "label":  label_tensor,   # [14]
            "meta":   meta_tensor,    # [2]  — age_norm, sex_binary
        }

        if self.return_meta:
            item["patient_id"] = str(row["patient_id"])
            item["scan_id"]    = int(row["scan_id"])

        return item


# ── Patient-Grouped Batch Sampler ───────────────────────────────────────────

class PatientGroupedBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that groups all companion scans of a patient into the same batch.

    Clinical & Graph Rationale:
      - Inter-scan Graph Neural Network message passing operates over companion radiographs
        of the SAME patient to contextualize diagnostic findings.
      - Standard uniform random row sampling scatters multi-scan patient records across random
        mini-batches, rendering inter-scan message passing inactive during training.
      - PatientGroupedBatchSampler clusters dataset row indices by patient_id.
      - At each epoch, patient clusters are randomly shuffled (if shuffle=True).
      - Patients are accumulated into batches targeting `batch_size` scans.
      - If an individual patient has more scans than batch_size, their scans are divided into
        sub-batches of at most `batch_size`.
      - Cross-patient isolation is strictly preserved: different patients within the batch
        remain disconnected in the graph (0 cross-patient edges).
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Group dataset row indices [0 .. len(df)-1] by patient_id
        clean_df = index_df.reset_index(drop=True)
        groups = clean_df.groupby("patient_id").indices
        self.patient_groups: list[list[int]] = [list(indices) for indices in groups.values()]

    def __iter__(self):
        import numpy as np
        rng = np.random.RandomState(self.seed + self.epoch)
        order = list(range(len(self.patient_groups)))
        if self.shuffle:
            rng.shuffle(order)

        current_batch: list[int] = []
        for g_idx in order:
            scans = self.patient_groups[g_idx]

            # If this patient alone exceeds batch_size, yield in chunks
            if len(scans) >= self.batch_size:
                if len(current_batch) > 0:
                    yield current_batch
                    current_batch = []
                for i in range(0, len(scans), self.batch_size):
                    chunk = scans[i : i + self.batch_size]
                    yield chunk
                continue

            # If adding this patient exceeds batch_size, flush current_batch first
            if len(current_batch) + len(scans) > self.batch_size:
                if len(current_batch) > 0:
                    yield current_batch
                    current_batch = []

            current_batch.extend(scans)

        if len(current_batch) > 0:
            yield current_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        total_scans = sum(len(g) for g in self.patient_groups)
        return total_scans // self.batch_size if self.drop_last else (total_scans + self.batch_size - 1) // self.batch_size


# ── DataLoader Factory ────────────────────────────────────────────────────────

def get_dataloader(
    split: str,
    index_df: pd.DataFrame,
    hf_dataset: HFDataset,
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: int = 224,
    return_meta: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    patient_grouping: bool = True,
) -> DataLoader:
    """Build a DataLoader for the given split.

    Args:
        split:         "train", "val", or "test".
        index_df:      Split-specific DataFrame with hf_index and metadata.
        hf_dataset:    Full HF Dataset (all 112,120 rows).
        batch_size:    Mini-batch size.
        num_workers:   Number of DataLoader workers (0 = main process).
        image_size:    Target image size (default 224).
        return_meta:   Include patient_id / scan_id in batch dict.
        pin_memory:    Pin memory for GPU transfer (only useful with GPU).
        persistent_workers: Keep workers alive between epochs.
        patient_grouping: If True and patient_id is present, group companion scans
                          of each patient together using PatientGroupedBatchSampler.

    Returns:
        Configured DataLoader.
    """
    transform = get_transforms(split, image_size=image_size)
    meta_encoder = MetadataEncoder()

    dataset = CXRDataset(
        index_df=index_df,
        hf_dataset=hf_dataset,
        transform=transform,
        metadata_encoder=meta_encoder,
        return_meta=return_meta,
    )

    shuffle = split == "train"

    if patient_grouping and "patient_id" in index_df.columns:
        batch_sampler = PatientGroupedBatchSampler(
            index_df=index_df,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            collate_fn=_collate_fn,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            drop_last=shuffle,   # drop last incomplete batch in training
            collate_fn=_collate_fn,
        )

    logger.info(
        "DataLoader[%s]: %d samples, batch_size=%d, num_workers=%d, patient_grouping=%s",
        split, len(dataset), batch_size, num_workers, (patient_grouping and "patient_id" in index_df.columns)
    )
    return loader


def get_all_dataloaders(
    split_dir: str,
    hf_dataset: HFDataset,
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: int = 224,
    device: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience: load all three splits and return (train, val, test) loaders.

    Args:
        split_dir:   Path to directory with train.csv, val.csv, test.csv.
        hf_dataset:  Full HF Dataset.
        batch_size:  Base batch size.
        num_workers: Worker count.
        image_size:  Image size.
        device:      "cuda" or "cpu". If "cuda", enables pin_memory and
                     persistent_workers when num_workers > 0.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from src.dataset.splitter import load_split_csvs

    use_gpu = device == "cuda"
    train_df, val_df, test_df = load_split_csvs(split_dir)

    loaders = []
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        loader = get_dataloader(
            split=split,
            index_df=df,
            hf_dataset=hf_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            pin_memory=use_gpu,
            persistent_workers=(use_gpu and num_workers > 0),
        )
        loaders.append(loader)

    return tuple(loaders)


# ── Collate Function ──────────────────────────────────────────────────────────

def _collate_fn(batch: list) -> Dict[str, Any]:
    """Custom collate that handles string fields (patient_id, scan_id)."""
    images  = torch.stack([b["image"] for b in batch])
    labels  = torch.stack([b["label"] for b in batch])
    metas   = torch.stack([b["meta"]  for b in batch])

    out = {"image": images, "label": labels, "meta": metas}

    if "patient_id" in batch[0]:
        out["patient_id"] = [b["patient_id"] for b in batch]
        out["scan_id"]    = [b["scan_id"]    for b in batch]

    return out
