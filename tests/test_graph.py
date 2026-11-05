"""
test_graph.py — Tests for Scan Graph Construction & GNN Processing
==================================================================
Tests scan-to-scan graph construction from patient_id, absence of label
leakage, batch graph builder, and GNN message-passing forward/backward passes.
"""

import pytest
import torch
import pandas as pd
from src.graph.graph_builder import (
    ScanGraphBuilder,
    PatientGraphBuilder,
    build_batch_scan_graph,
    TORCH_GEOMETRIC_AVAILABLE,
)
from src.models.gnn_branch import GNNBranch, get_gnn_branch


@pytest.fixture
def sample_scan_df():
    """Sample DataFrame with 6 scans across 3 patients:
    - Patient 1: 3 scans (indices 0, 1, 2)
    - Patient 2: 2 scans (indices 3, 4)
    - Patient 3: 1 scan  (index 5)
    """
    return pd.DataFrame({
        "patient_id": ["P001", "P001", "P001", "P002", "P002", "P003"],
        "scan_id":    ["S01", "S02", "S03", "S04", "S05", "S06"],
        "age":        [45, 46, 47, 60, 61, 30],
        "sex":        ["M", "M", "M", "F", "F", "F"],
        "labels":     [[1], [1, 4], [0], [2], [0], [3]],
    })


class TestScanGraphBuilder:
    def test_scan_graph_node_count(self, sample_scan_df):
        if not TORCH_GEOMETRIC_AVAILABLE:
            pytest.skip("torch-geometric not installed")
        builder = ScanGraphBuilder(add_self_loops=False)
        graph = builder.build(sample_scan_df)
        assert graph.num_nodes == 6
        assert graph.x.shape == (6, 2)

    def test_scan_graph_edge_connectivity_by_patient_id(self, sample_scan_df):
        """P001 (3 scans) -> 3*2 = 6 directed edges.
        P002 (2 scans) -> 2*1 = 2 directed edges.
        P003 (1 scan)  -> 0 edges (without self-loops).
        Total directed edges = 8.
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            pytest.skip("torch-geometric not installed")
        builder = ScanGraphBuilder(add_self_loops=False)
        graph = builder.build(sample_scan_df)
        edge_index = graph.edge_index
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 8

        # Verify all edges connect scans with matching patient_ids
        for col in range(edge_index.shape[1]):
            u, v = edge_index[0, col].item(), edge_index[1, col].item()
            assert u != v
            assert sample_scan_df.loc[u, "patient_id"] == sample_scan_df.loc[v, "patient_id"]

    def test_scan_graph_with_self_loops(self, sample_scan_df):
        if not TORCH_GEOMETRIC_AVAILABLE:
            pytest.skip("torch-geometric not installed")
        builder = ScanGraphBuilder(add_self_loops=True)
        graph = builder.build(sample_scan_df)
        # 8 cross-scan edges + 6 self-loops = 14 directed edges
        assert graph.edge_index.shape[1] == 14

    def test_zero_label_leakage(self, sample_scan_df):
        """Altering or removing labels must NOT change graph edge_index."""
        if not TORCH_GEOMETRIC_AVAILABLE:
            pytest.skip("torch-geometric not installed")
        builder = ScanGraphBuilder(add_self_loops=True)
        graph1 = builder.build(sample_scan_df)

        modified_df = sample_scan_df.copy()
        modified_df["labels"] = [[0], [0], [0], [0], [0], [0]]
        graph2 = builder.build(modified_df)

        assert torch.equal(graph1.edge_index, graph2.edge_index)


class TestBatchScanGraph:
    def test_batch_graph_shapes(self):
        meta = torch.rand(4, 2)
        pids = ["A", "A", "B", "C"]
        x, edge_index = build_batch_scan_graph(meta, pids, add_self_loops=True)
        assert x.shape == (4, 2)
        assert edge_index.shape[0] == 2
        # A has 2 scans -> 2 edges (0->1, 1->0) + 4 self loops = 6 edges
        assert edge_index.shape[1] == 6


class TestGNNOnScanGraph:
    def test_gnn_forward_backward_on_scan_graph(self, sample_scan_df):
        if not TORCH_GEOMETRIC_AVAILABLE:
            pytest.skip("torch-geometric not installed")
        builder = ScanGraphBuilder(add_self_loops=True)
        graph = builder.build(sample_scan_df)

        gnn = GNNBranch(node_feat_dim=2, hidden_dim=32, output_dim=64, n_layers=2)
        out = gnn(graph.x, graph.edge_index)
        assert out.shape == (6, 64)
        assert torch.isfinite(out).all()

        loss = out.sum()
        loss.backward()
        for p in gnn.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert not torch.isnan(p.grad).any()
