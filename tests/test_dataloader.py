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


class TestPatientGroupedBatchSampler:
    def test_same_patient_scans_grouped_together_in_batch(self):
        """Prove that companion scans of the same patient are kept in the same batch
        and that build_batch_scan_graph creates intra-patient edges with zero cross-patient edges."""
        import pandas as pd
        from src.dataset.dataloader import get_dataloader, PatientGroupedBatchSampler
        from src.graph.graph_builder import build_batch_scan_graph

        # Synthetic multi-scan patient dataset:
        # P001: 3 scans, P002: 2 scans, P003: 1 scan, P004: 4 scans (total 10 scans)
        df = pd.DataFrame([
            {"hf_index": 0, "patient_id": "P001", "scan_id": 0, "age": 45, "sex": "M", "labels_raw": [1]},
            {"hf_index": 1, "patient_id": "P001", "scan_id": 1, "age": 45, "sex": "M", "labels_raw": [1]},
            {"hf_index": 2, "patient_id": "P001", "scan_id": 2, "age": 45, "sex": "M", "labels_raw": [1]},
            {"hf_index": 3, "patient_id": "P002", "scan_id": 0, "age": 52, "sex": "F", "labels_raw": [2]},
            {"hf_index": 4, "patient_id": "P002", "scan_id": 1, "age": 52, "sex": "F", "labels_raw": [2]},
            {"hf_index": 5, "patient_id": "P003", "scan_id": 0, "age": 60, "sex": "M", "labels_raw": [0]},
            {"hf_index": 6, "patient_id": "P004", "scan_id": 0, "age": 38, "sex": "F", "labels_raw": [3]},
            {"hf_index": 7, "patient_id": "P004", "scan_id": 1, "age": 38, "sex": "F", "labels_raw": [3]},
            {"hf_index": 8, "patient_id": "P004", "scan_id": 2, "age": 38, "sex": "F", "labels_raw": [3]},
            {"hf_index": 9, "patient_id": "P004", "scan_id": 3, "age": 38, "sex": "F", "labels_raw": [3]},
        ])

        hf = make_fake_hf_dataset(10)
        loader = get_dataloader(
            split="train",
            index_df=df,
            hf_dataset=hf,
            batch_size=5,
            patient_grouping=True,
        )

        total_sampled = 0
        for batch in loader:
            pids = batch["patient_id"]
            total_sampled += len(pids)

            # Verify that any patient appearing in this batch has ALL their scans present in this batch
            # (unless patient scans exceed batch_size)
            for p in set(pids):
                total_in_df = (df["patient_id"] == p).sum()
                in_batch = pids.count(p)
                assert in_batch == min(total_in_df, 5), (
                    f"Patient {p} companion scans were split across batches! "
                    f"Expected {min(total_in_df, 5)} in batch, found {in_batch}"
                )

            # Build the batch scan graph using the real training graph construction
            meta = batch["meta"]
            _, edge_index = build_batch_scan_graph(meta, pids, add_self_loops=True)

            # Verify zero cross-patient edges
            for e in range(edge_index.shape[1]):
                src, dst = edge_index[0, e].item(), edge_index[1, e].item()
                assert pids[src] == pids[dst], (
                    f"Cross-patient edge detected: {pids[src]} -> {pids[dst]}"
                )

            # Verify intra-patient connectivity:
            # For each patient p with k scans in the batch, expected directed edges = k^2
            for p in set(pids):
                k = pids.count(p)
                # Count edges where both endpoints belong to p
                p_edges = sum(
                    1 for e in range(edge_index.shape[1])
                    if pids[edge_index[0, e].item()] == p and pids[edge_index[1, e].item()] == p
                )
                assert p_edges == k * k, (
                    f"Patient {p} with {k} scans should have {k*k} edges (complete subgraph + self-loops), got {p_edges}"
                )

        assert total_sampled == len(df), f"Expected {len(df)} total scans sampled, got {total_sampled}"

    def test_sampler_exact_index_multiplicity_drop_last_false(self):
        """Verify that with drop_last=False (default), every single index appears exactly once."""
        import pandas as pd
        from src.dataset.dataloader import PatientGroupedBatchSampler

        # 25 scans across 7 patients with variable scan counts (1 to 8 scans)
        pids = ["P1"] * 8 + ["P2"] * 5 + ["P3"] * 4 + ["P4"] * 3 + ["P5"] * 2 + ["P6"] * 2 + ["P7"] * 1
        df = pd.DataFrame({"patient_id": pids, "age": [50] * len(pids), "sex": ["M"] * len(pids)})

        for b_size in [4, 8, 16, 32]:
            sampler = PatientGroupedBatchSampler(df, batch_size=b_size, shuffle=True, seed=42, drop_last=False)
            batches = list(sampler)
            assert len(sampler) == len(batches), f"Length mismatch for B={b_size}: {len(sampler)} vs {len(batches)}"

            all_indices = [idx for b in batches for idx in b]
            assert len(all_indices) == len(df), f"Total count mismatch for B={b_size}: {len(all_indices)} vs {len(df)}"
            assert sorted(all_indices) == list(range(len(df))), f"Indices do not have 1:1 multiplicity for B={b_size}"

            for b in batches:
                assert len(b) <= b_size, f"Batch exceeded batch_size {b_size}: got {len(b)}"

    def test_sampler_oversized_patient_balanced_partitioning(self):
        """Verify balanced chunk partitioning for oversized patients across B in {4, 8, 16, 32}."""
        import pandas as pd
        from src.dataset.dataloader import PatientGroupedBatchSampler, balanced_chunks

        # Verify balanced_chunks directly for critical sizes
        for K in [33, 65, 184]:
            for B in [4, 8, 16, 32]:
                scans = list(range(K))
                chunks = balanced_chunks(scans, B)
                # 1. Total elements preserved
                flattened = [x for c in chunks for x in c]
                assert flattened == scans
                # 2. Every chunk <= B
                for c in chunks:
                    assert len(c) <= B
                # 3. No chunk of size 1 when K >= 2*B
                if K >= 2 * B:
                    for c in chunks:
                        assert len(c) > 1, f"Found singleton chunk for K={K}, B={B}: sizes={[len(x) for x in chunks]}"

        # Now test within PatientGroupedBatchSampler
        pids = ["P_LARGE"] * 33 + ["P_SMALL"] * 3
        df = pd.DataFrame({"patient_id": pids, "age": [40] * len(pids), "sex": ["F"] * len(pids)})
        sampler = PatientGroupedBatchSampler(df, batch_size=32, shuffle=False, drop_last=False)
        batches = list(sampler)
        assert len(sampler) == len(batches)
        # P_LARGE should be partitioned into 17 and 16, not 32 and 1
        large_batch_sizes = [len(b) for b in batches if 0 in b or 1 in b]
        assert 1 not in large_batch_sizes, f"Oversized patient produced a singleton chunk: {large_batch_sizes}"

    def test_sampler_drop_last_true_behavior(self):
        """Verify drop_last=True strictly enforces batch_size:
        - Every emitted batch has exactly B samples (len(b) == B).
        - No batch exceeds B.
        - Number of retained and omitted indices is exact.
        - No index is duplicated.
        """
        import pandas as pd
        from src.dataset.dataloader import PatientGroupedBatchSampler

        # Deterministic setup:
        # P1: 2 scans (indices 0, 1)
        # P2: 2 scans (indices 2, 3) -> P1 + P2 = 4 (Batch 1: exact B=4)
        # P3: 3 scans (indices 4, 5, 6)
        # P4: 2 scans (indices 7, 8) -> P3 + P4 = 5 > 4. P3 closed at 3 < 4 (dropped in drop_last=True)
        # P5: 2 scans (indices 9, 10) -> P4 + P5 = 4 (Batch 2: exact B=4)
        # P6: 1 scan (index 11) -> closed at 1 < 4 (dropped in drop_last=True)
        pids = ["P1"] * 2 + ["P2"] * 2 + ["P3"] * 3 + ["P4"] * 2 + ["P5"] * 2 + ["P6"] * 1
        df = pd.DataFrame({"patient_id": pids, "age": [40] * len(pids), "sex": ["F"] * len(pids)})
        total_samples = len(df)  # 12

        B = 4
        sampler = PatientGroupedBatchSampler(df, batch_size=B, shuffle=False, drop_last=True)
        batches = list(sampler)

        # 1. Sampler length parity
        assert len(sampler) == len(batches)
        assert len(batches) == 2, f"Expected 2 complete batches, got {len(batches)}"

        # 2. Every emitted batch has exactly B samples and none exceeds B
        for b in batches:
            assert len(b) == B, f"Emitted batch has size {len(b)} != B={B}"
            assert len(b) <= B

        # 3. No duplicate indices across emitted batches
        flat_retained = [idx for b in batches for idx in b]
        assert len(flat_retained) == len(set(flat_retained)), "Duplicate index in emitted batches"
        assert set(flat_retained) == {0, 1, 2, 3, 7, 8, 9, 10}

        # 4. Correct retained and omitted counts
        retained_count = len(flat_retained)
        omitted_indices = set(range(total_samples)) - set(flat_retained)
        assert retained_count == 8, f"Expected 8 retained samples, got {retained_count}"
        assert len(omitted_indices) == 4, f"Expected 4 omitted samples, got {len(omitted_indices)}"
        assert omitted_indices == {4, 5, 6, 11}

        # Also test with shuffle=True across epochs
        for ep in range(5):
            sampler.set_epoch(ep)
            s_batches = list(sampler)
            assert len(sampler) == len(s_batches)
            for b in s_batches:
                assert len(b) == B, f"Epoch {ep}: batch size {len(b)} != {B}"
            s_retained = [idx for b in s_batches for idx in b]
            assert len(s_retained) == len(set(s_retained)), f"Epoch {ep}: duplicated indices"
            assert len(s_retained) % B == 0
            s_omitted = set(range(total_samples)) - set(s_retained)
            assert len(s_retained) + len(s_omitted) == total_samples

    def test_graph_isolation_and_complete_patient_cliques(self):
        """Verify build_batch_scan_graph() creates zero cross-patient edges
        while preserving each patient's complete clique (including self-loops)."""
        import torch
        from src.graph.graph_builder import build_batch_scan_graph

        # Multi-patient batch:
        # P_A: 2 scans, P_B: 3 scans, P_C: 1 scan (singleton), P_D: 4 scans (total 10 scans)
        pids = ["P_A", "P_A", "P_B", "P_B", "P_B", "P_C", "P_D", "P_D", "P_D", "P_D"]
        B = len(pids)
        meta_tensor = torch.randn(B, 2)

        x, edge_index = build_batch_scan_graph(meta_tensor, pids, add_self_loops=True)

        assert x.shape == (B, 2)
        assert edge_index.shape[0] == 2

        # 1. Zero cross-patient edges
        for e in range(edge_index.shape[1]):
            src = edge_index[0, e].item()
            dst = edge_index[1, e].item()
            assert pids[src] == pids[dst], (
                f"Cross-patient edge detected: node {src} ({pids[src]}) -> node {dst} ({pids[dst]})"
            )

        # 2. Complete clique preservation for each patient
        # A patient with k scans must have exactly k^2 directed edges (k*(k-1) cross-edges + k self-loops)
        from collections import Counter
        counts = Counter(pids)
        expected_total_edges = sum(k * k for k in counts.values())
        assert edge_index.shape[1] == expected_total_edges, (
            f"Expected {expected_total_edges} total edges, got {edge_index.shape[1]}"
        )

        edges_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for pid, k in counts.items():
            patient_indices = [idx for idx, p in enumerate(pids) if p == pid]
            for i in patient_indices:
                for j in patient_indices:
                    assert (i, j) in edges_set, f"Missing intra-patient edge ({i}, {j}) for {pid}"

    def test_sampler_oversized_patients_comprehensive(self):
        """Test K=33, 65, and 184 for B=4, 8, 16, and 32 through PatientGroupedBatchSampler.
        Verify capacity, exact coverage, and patient isolation on resulting graphs.
        """
        import pandas as pd
        import torch
        from src.dataset.dataloader import PatientGroupedBatchSampler
        from src.graph.graph_builder import build_batch_scan_graph

        for K in [33, 65, 184]:
            for B in [4, 8, 16, 32]:
                # Cohort with 1 oversized patient + companion smaller patients
                pids = ["P_OVERSIZED"] * K + ["P_SMALL1"] * 3 + ["P_SMALL2"] * 2 + ["P_SINGLE"] * 1
                df = pd.DataFrame({
                    "patient_id": pids,
                    "age": [55] * len(pids),
                    "sex": ["M"] * len(pids),
                    "scan_id": range(len(pids)),
                })
                N = len(df)

                sampler = PatientGroupedBatchSampler(df, batch_size=B, shuffle=True, seed=42, drop_last=False)
                batches = list(sampler)

                # 1. Exact length parity
                assert len(sampler) == len(batches), f"Len mismatch for K={K}, B={B}"

                # 2. Capacity invariant: no batch exceeds B
                for b in batches:
                    assert len(b) <= B, f"Batch exceeded B={B}: size {len(b)} (K={K})"

                # 3. Exact coverage invariant: every index appears exactly once (drop_last=False)
                flat_indices = [idx for b in batches for idx in b]
                assert len(flat_indices) == N, f"Count mismatch for K={K}, B={B}: {len(flat_indices)} vs {N}"
                assert sorted(flat_indices) == list(range(N)), f"Missing/duplicate indices for K={K}, B={B}"

                # 4. Patient isolation: verify graph builder creates 0 cross-patient edges
                for b in batches:
                    batch_pids = [df.iloc[idx]["patient_id"] for idx in b]
                    meta = torch.zeros((len(b), 2))
                    _, edge_index = build_batch_scan_graph(meta, batch_pids, add_self_loops=True)
                    for e in range(edge_index.shape[1]):
                        src, dst = edge_index[0, e].item(), edge_index[1, e].item()
                        assert batch_pids[src] == batch_pids[dst], (
                            f"Cross-patient edge in batch! {batch_pids[src]} -> {batch_pids[dst]}"
                        )

    def test_sampler_len_matches_iter_len_across_epochs(self):
        """Verify len(sampler) == len(list(sampler)) identically across multiple epochs."""
        import pandas as pd
        from src.dataset.dataloader import PatientGroupedBatchSampler

        pids = ["P1"] * 4 + ["P2"] * 3 + ["P3"] * 6 + ["P4"] * 2 + ["P5"] * 5
        df = pd.DataFrame({"patient_id": pids, "age": [45] * len(pids), "sex": ["M"] * len(pids)})

        sampler = PatientGroupedBatchSampler(df, batch_size=8, shuffle=True, seed=123, drop_last=False)
        for ep in range(5):
            sampler.set_epoch(ep)
            batches = list(sampler)
            assert len(sampler) == len(batches), f"Mismatch on epoch {ep}: {len(sampler)} vs {len(batches)}"
