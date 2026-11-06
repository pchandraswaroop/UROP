"""
gnn_explainer.py — Graph Explanation for Inter-Scan Patient Relationships
========================================================================
Identifies the most influential neighboring scan nodes and edges within the
patient-based scan graph for individual disease outputs.

Uses:
  1. Edge perturbation and gradient-based edge attribution.
  2. Node importance estimation.
  3. NetworkX graph visualizer highlighting high-importance scan relationships.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ScanGraphExplainer:
    """Explainer for the patient-based scan graph.

    Measures how each scan-to-scan edge and neighboring patient scan influences
    the target disease probability.
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 50,
        lr: float = 0.05,
        num_classes: int = 14,
        **kwargs,
    ) -> None:
        """
        Args:
            model: MultiModalCXRModel containing a GNN clinical branch, or GNNBranch.
            epochs: Optimization steps for edge mask optimization.
            lr: Learning rate for edge mask optimizer.
        """
        self.model = model
        self.model.eval()
        self.epochs = epochs
        self.lr = lr
        self.num_classes = num_classes

    def _forward_model(
        self,
        images: Optional[torch.Tensor],
        meta: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """Forward pass extracting target disease probability."""
        B = meta.shape[0]
        if images is None:
            images = torch.zeros(B, 3, 224, 224, device=meta.device)

        if hasattr(self.model, "image_branch"):
            logits = self.model(
                images,
                meta,
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                return_logits=True,
            )
            return torch.sigmoid(logits)[:, target_class]
        else:
            emb = self.model(graph_x, graph_edge_index)
            return emb.mean(dim=-1)

    def explain(
        self,
        images: Optional[torch.Tensor],
        meta: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        target_class: int = 0,
        patient_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute edge and node importance scores for a target disease class.

        Args:
            images: Input images [B, 3, H, W] (or None).
            meta: Metadata tensor [B, 2].
            graph_x: Node features [N, 2].
            graph_edge_index: Edge indices [2, E].
            target_class: Target disease class index (0..13).
            patient_ids: Optional list of patient ID strings.

        Returns:
            Dict containing:
              - edge_importance: numpy array of shape [E] in [0, 1].
              - node_importance: numpy array of shape [N] in [0, 1].
              - edge_index: [2, E] edge tensor.
              - target_class: int.
              - original_prob: mean probability with full graph.
        """
        E = graph_edge_index.shape[1]
        N = graph_x.shape[0]
        device = graph_x.device

        # 1. Base prediction with full graph
        with torch.no_grad():
            base_probs = self._forward_model(images, meta, graph_x, graph_edge_index, target_class)
            base_score = base_probs.mean().item()

        if E == 0:
            return {
                "edge_importance": np.zeros(0),
                "node_importance": np.ones(N) / max(1, N),
                "edge_index": graph_edge_index.cpu().numpy(),
                "target_class": target_class,
                "original_prob": base_score,
            }

        # 2. Compute edge attribution by edge ablation (impact on target disease prob)
        # Drop each edge (or symmetric pair) and measure prediction delta
        edge_deltas = np.zeros(E, dtype=np.float32)

        # For computational efficiency on larger edge sets, evaluate impact
        with torch.no_grad():
            for e_idx in range(E):
                # Mask out edge e_idx
                keep_mask = torch.ones(E, dtype=torch.bool, device=device)
                keep_mask[e_idx] = False
                sub_edge_index = graph_edge_index[:, keep_mask]

                ablated_probs = self._forward_model(images, meta, graph_x, sub_edge_index, target_class)
                delta = abs(base_score - ablated_probs.mean().item())
                edge_deltas[e_idx] = delta

        # Normalize edge importance to [0, 1]
        e_max = edge_deltas.max()
        if e_max > 0:
            edge_importance = edge_deltas / e_max
        else:
            edge_importance = np.ones(E, dtype=np.float32) / max(1, E)

        # 3. Compute node importance: sum of connected edge importances + self attribution
        node_importance = np.zeros(N, dtype=np.float32)
        edges_np = graph_edge_index.cpu().numpy()
        for e_idx in range(E):
            u, v = edges_np[0, e_idx], edges_np[1, e_idx]
            imp = edge_importance[e_idx]
            node_importance[u] += imp
            node_importance[v] += imp

        n_max = node_importance.max()
        if n_max > 0:
            node_importance = node_importance / n_max
        else:
            node_importance = np.ones(N, dtype=np.float32) / max(1, N)

        return {
            "edge_importance": edge_importance,
            "node_importance": node_importance,
            "edge_index": edges_np,
            "target_class": target_class,
            "original_prob": base_score,
            "patient_ids": patient_ids,
        }

    def visualize_graph(
        self,
        explanation: Dict[str, Any],
        title: str = "Scan Graph Explanation",
        save_path: Optional[str] = None,
        max_nodes: int = 40,
    ) -> plt.Figure:
        """Visualize the patient scan graph highlighting important edges and nodes."""
        edge_index = explanation["edge_index"]
        edge_imp = explanation["edge_importance"]
        node_imp = explanation["node_importance"]
        pids = explanation.get("patient_ids")

        N = len(node_imp)
        E = edge_index.shape[1] if edge_index.ndim == 2 else 0

        # Subsample if graph is too large for single plot
        nodes_to_show = range(min(N, max_nodes))
        node_set = set(nodes_to_show)

        G = nx.Graph()
        for i in nodes_to_show:
            label = f"Scan {i}"
            if pids and i < len(pids):
                label += f"\n(Pt: {pids[i]})"
            G.add_node(i, label=label, importance=node_imp[i])

        for e in range(E):
            u, v = int(edge_index[0, e]), int(edge_index[1, e])
            if u in node_set and v in node_set and u != v:
                # Undirected graph: max importance for pair
                weight = float(edge_imp[e])
                if G.has_edge(u, v):
                    G[u][v]["weight"] = max(G[u][v]["weight"], weight)
                else:
                    G.add_edge(u, v, weight=weight)

        fig, ax = plt.subplots(figsize=(10, 8))
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Empty Graph", ha="center", va="center")
            return fig

        pos = nx.spring_layout(G, seed=42)

        # Node colors by importance
        node_colors = [G.nodes[n]["importance"] for n in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            cmap=plt.cm.YlOrRd,
            node_size=600,
            alpha=0.9,
            edgecolors="black",
        )

        # Edge widths and colors by importance
        edges = list(G.edges(data=True))
        if edges:
            weights = [d["weight"] for _, _, d in edges]
            edge_widths = [1.0 + 4.0 * w for w in weights]
            edge_colors = [plt.cm.Blues(0.3 + 0.7 * w) for w in weights]
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=edge_widths,
                edge_color=edge_colors,
                alpha=0.8,
            )

        labels = {n: G.nodes[n]["label"] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved graph explanation visualization to %s", save_path)

        return fig
