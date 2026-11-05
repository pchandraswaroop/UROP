"""
test_dataloader.py — Unit tests for CXRDataset and DataLoader (mocked)
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.dataloader import CXRDataset, _collate_fn
from src.dataset.preprocessor import get_transforms, MetadataEncoder


def make_fake_hf_dataset(n: int = 10):
    """Create a mock HF dataset that returns PIL images."""
    records = []
    for i in range(n):
        records.append({
            "image":      Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8)),
            "patient_id": f"P{i:05d}",
            "scan_id":    i,
            "age":        30 + i,
            "sex":        "M" if i % 2 == 0 else "F",
            "labels":     [1] if i % 3 == 0 else [0],
        })

    ds = MagicMock()
    ds.__getitem__ = lambda self, idx: records[idx]
    ds.__len__ = lambda self: n
    return ds


def make_fake_index_df(n: int = 10):
    import pandas as pd
    return pd.DataFrame([
        {
            "hf_index":   i,
            "patient_id": f"P{i:05d}",
            "scan_id":    i,
            "age":        30 + i,
            "sex":        "M" if i % 2 == 0 else "F",
            "labels_raw": [1] if i % 3 == 0 else [0],
            "split":      "train",
        }
        for i in range(n)
    ])


class TestCXRDataset:
    def test_len(self):
        hf = make_fake_hf_dataset(10)
        df = make_fake_index_df(10)
        tf = get_transforms("val", image_size=224)
        ds = CXRDataset(df, hf, tf)
        assert len(ds) == 10

    def test_getitem_shapes(self):
        hf = make_fake_hf_dataset(10)
        df = make_fake_index_df(10)
        tf = get_transforms("val", image_size=224)
        ds = CXRDataset(df, hf, tf)
        item = ds[0]

        assert item["image"].shape == torch.Size([3, 224, 224])
        assert item["label"].shape == torch.Size([14])
        assert item["meta"].shape  == torch.Size([2])

    def test_label_dtype(self):
        hf = make_fake_hf_dataset(10)
        df = make_fake_index_df(10)
        tf = get_transforms("val")
        ds = CXRDataset(df, hf, tf)
        item = ds[0]
        assert item["label"].dtype == torch.float32

    def test_image_dtype(self):
        hf = make_fake_hf_dataset(10)
        df = make_fake_index_df(10)
        tf = get_transforms("val")
        ds = CXRDataset(df, hf, tf)
        item = ds[0]
        assert item["image"].dtype == torch.float32

    def test_image_values_finite(self):
        hf = make_fake_hf_dataset(10)
        df = make_fake_index_df(10)
        tf = get_transforms("train")
        ds = CXRDataset(df, hf, tf)
        for i in range(5):
            item = ds[i]
            assert torch.isfinite(item["image"]).all()

    def test_meta_values_finite(self):
        hf = make_fake_hf_dataset(10)
        df = make_fake_index_df(10)
        tf = get_transforms("val")
        ds = CXRDataset(df, hf, tf)
        item = ds[0]
        assert torch.isfinite(item["meta"]).all()

    def test_no_finding_label_zero_tensor(self):
        """Row with labels=[0] should produce all-zero disease tensor."""
        hf = make_fake_hf_dataset(10)
        df = make_fake_index_df(10)
        # Force label to [0] (No Finding)
        df.at[0, "labels_raw"] = [0]
        tf = get_transforms("val")
        ds = CXRDataset(df, hf, tf)
        item = ds[0]
        assert item["label"].sum() == 0.0


class TestCollate:
    def test_collate_stacks_tensors(self):
        batch = [
            {
                "image":      torch.zeros(3, 224, 224),
                "label":      torch.zeros(14),
                "meta":       torch.zeros(2),
                "patient_id": "P00001",
                "scan_id":    1,
            },
            {
                "image":      torch.ones(3, 224, 224),
                "label":      torch.ones(14),
                "meta":       torch.ones(2),
                "patient_id": "P00002",
                "scan_id":    2,
            },
        ]
        out = _collate_fn(batch)
        assert out["image"].shape == torch.Size([2, 3, 224, 224])
        assert out["label"].shape == torch.Size([2, 14])
        assert out["meta"].shape  == torch.Size([2, 2])
        assert len(out["patient_id"]) == 2


class TestHFLoaderStreaming:
    def test_sample_n_triggers_genuine_streaming_without_cache(self, tmp_path):
        from src.dataset.hf_loader import load_hf_dataset
        with patch("src.dataset.hf_loader.load_dataset") as mock_load:
            mock_stream = MagicMock()
            mock_stream.take.return_value = [
                {"patient_id": f"P{i}", "scan_id": i, "age": 40, "sex": "M", "labels": [1]}
                for i in range(5)
            ]
            mock_load.return_value = mock_stream

            empty_cache = str(tmp_path / "empty_cache")
            ds = load_hf_dataset("chehablab/NIHChestXR", sample_n=5, cache_dir=empty_cache)

            # Confirm load_dataset was called with streaming=True
            mock_load.assert_called_once()
            _, kwargs = mock_load.call_args
            assert kwargs.get("streaming") is True
            mock_stream.take.assert_called_once_with(5)
            assert len(ds) == 5

    def test_full_experiment_loads_non_streaming(self, tmp_path):
        from src.dataset.hf_loader import load_hf_dataset
        with patch("src.dataset.hf_loader.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__len__.return_value = 100
            mock_ds.features = {"image": None}
            mock_load.return_value = mock_ds

            empty_cache = str(tmp_path / "empty_cache")
            ds = load_hf_dataset("chehablab/NIHChestXR", sample_n=None, cache_dir=empty_cache)

            mock_load.assert_called_once()
            _, kwargs = mock_load.call_args
            assert kwargs.get("streaming") is False
