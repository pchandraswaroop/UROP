"""
gnn_branch.py — Graph Neural Network Clinical Branch
=====================================================
Operates on the patient-level graph built by graph_builder.py.
Outputs per-node embeddings used in the fusion layer.

Architecture:
  [N, node_feat_dim] → GCNConv × n_layers → [N, hidden_dim]

At inference time for val/test patients NOT in the training graph:
  FeaturelessGNN uses only structural signal; but since new patients
  have no graph connection, we fall back to a simple MLP on metadata.
  See mlp_baseline.py.

Note: Requires torch-geometric.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
    TG_AVAILABLE = True
except ImportError:
    TG_AVAILABLE = False
    logger.warning(
        "torch-geometric not installed. GNN branch disabled. "
        "Install: pip install torch-geometric"
    )


# ── GNN ──────────────────────────────────────────────────────────────────────

class GNNBranch(nn.Module):
    """GraphSAGE-based patient graph encoder.

    Uses GraphSAGE (SAGEConv) rather than GCN because:
    1. GraphSAGE is inductive (generalises to unseen nodes).
    2. Better suited when node degrees vary greatly.
    3. More robust with sparse features.

    Args:
        node_feat_dim: Input node feature dimension (e.g., 2 for age+sex).
        hidden_dim:    Hidden dimension per GNN layer.
        output_dim:    Output embedding dimension.
        n_layers:      Number of message-passing layers.
        dropout:       Dropout rate.
    """

    def __init__(
        self,
        node_feat_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if not TG_AVAILABLE:
            raise ImportError(
                "torch-geometric required for GNNBranch. "
                "Install: pip install torch-geometric"
            )

        self.n_layers = n_layers
        self.dropout  = dropout

        # Build SAGEConv layers
        dims = [node_feat_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1])
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1])
            for i in range(n_layers)
        ])

        logger.info(
            "GNNBranch: node_feat=%d, hidden=%d, output=%d, layers=%d",
            node_feat_dim, hidden_dim, output_dim, n_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass on graph.

        Args:
            x:          Node feature matrix [N, node_feat_dim].
            edge_index: Edge list [2, E].

        Returns:
            Node embeddings [N, output_dim].
        """
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            if i < self.n_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


# ── Disabled Stub ─────────────────────────────────────────────────────────────

class GNNDisabledStub(nn.Module):
    """Stub returned when GNN is disabled. Outputs zeros of the correct shape."""

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        logger.info("GNNDisabledStub: GNN branch disabled, output zeros [B, %d]", output_dim)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        return torch.zeros(B, self.output_dim, device=x.device)


def get_gnn_branch(
    node_feat_dim: int = 2,
    hidden_dim: int = 64,
    output_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.3,
    enabled: bool = True,
) -> nn.Module:
    """Factory: return GNNBranch or GNNDisabledStub."""
    if enabled and TG_AVAILABLE:
        return GNNBranch(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
    return GNNDisabledStub(output_dim=output_dim)
