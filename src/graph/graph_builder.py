"""
graph_builder.py — Patient Graph Construction
=============================================
Constructs a patient-level graph from training-split metadata.

Design Rationale
----------------
Available non-leaking metadata:
  - age (int)
  - sex (string)

Available (training-only, for edge construction):
  - Multi-label disease annotations per patient

Graph definition:
  - NODES: Each unique patient in the training split is a node.
  - NODE FEATURES: [age_norm, sex_binary]  (shape [N, 2])
    * Training-set disease prevalence per patient is intentionally NOT used
      as node features to prevent label leakage at inference time.
  - EDGES: Two patients are connected if their sets of observed disease
    labels have Jaccard similarity ≥ threshold (default 0.3).
    * Edges are constructed ONLY from training labels.
    * Val/test patients are NOT added to the graph (transductive setup
      would require their labels; inductive inference instead relies on
      the image branch and MLP baseline).

GNN Limitation (Documented)
----------------------------
With only age and sex as non-label node features, GNN node discriminability
is inherently limited. The graph structure (who is connected to whom) is
constructed from labels and is therefore unavailable at test time for new
patients. This limits the GNN to a transductive setting within the training
graph plus a feature-fallback for test-time inference.

We implement:
  1. Patient graph for training (transductive).
  2. Node feature inference for val/test: only [age_norm, sex_binary] — no
     disease-history features.
  3. MLP baseline on metadata only for fair comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

try:
    import torch_geometric
    from torch_geometric.data import Data as PygData
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning(
        "torch_geometric not installed. GNN branch will be disabled. "
        "Install with: pip install torch-geometric"
    )


# ── Graph Builder ─────────────────────────────────────────────────────────────

class PatientGraphBuilder:
    """Build a patient-level graph from training split metadata.

    Args:
        edge_threshold: Minimum Jaccard similarity for an edge to exist.
        num_disease_classes: Number of disease classes (14 for NIH).
    """

    def __init__(
        self,
        edge_threshold: float = 0.3,
        num_disease_classes: int = 14,
    ) -> None:
        self.edge_threshold = edge_threshold
        self.num_disease_classes = num_disease_classes

    def build(
        self,
        train_df: pd.DataFrame,
        meta_encoder=None,
    ) -> Optional["PygData"]:
        """Build a PyG Data object from the training split DataFrame.

        Args:
            train_df:     Training split DataFrame with columns:
                          [patient_id, age, sex, labels_raw]
            meta_encoder: MetadataEncoder for age/sex encoding.

        Returns:
            PyG Data object with:
              - x:          Node features [N_patients, 2]
              - edge_index: Edge list [2, E]
              - edge_attr:  Jaccard similarity for each edge [E]
              - patient_ids: List[str] mapping node index → patient_id
            Returns None if torch_geometric is not available.
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.error("Cannot build graph: torch-geometric not installed.")
            return None

        if meta_encoder is None:
            from src.dataset.preprocessor import MetadataEncoder
            meta_encoder = MetadataEncoder()

        # ── Aggregate per-patient ──────────────────────────────────────────
        patient_df = self._aggregate_patients(train_df)
        logger.info("Building graph for %d unique patients.", len(patient_df))

        # ── Node Features ──────────────────────────────────────────────────
        # Only age_norm and sex_binary — NO disease labels
        node_features = meta_encoder.encode_batch(
            ages=patient_df["age"].tolist(),
            sexes=patient_df["sex"].tolist(),
        )  # [N, 2]

        # ── Edge Construction (Jaccard on disease sets) ────────────────────
        # Build per-patient disease sets from TRAINING labels only
        disease_matrix = self._build_disease_matrix(patient_df)  # [N, 14]
        edge_index, edge_attr = self._build_jaccard_edges(disease_matrix)

        logger.info(
            "Graph built: %d nodes, %d edges (threshold=%.2f)",
            len(patient_df), edge_index.shape[1] if edge_index.numel() > 0 else 0,
            self.edge_threshold,
        )

        graph = PygData(
            x=node_features,              # [N, 2]
            edge_index=edge_index,        # [2, E]
            edge_attr=edge_attr,          # [E]
            num_nodes=len(patient_df),
        )
        # Store patient IDs as a list attribute (not tensor)
        graph.patient_ids = patient_df["patient_id"].tolist()

        return graph

    def _aggregate_patients(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Group by patient_id, keeping representative age/sex and merged labels."""
        from src.dataset.label_encoder import safe_parse_labels

        df = train_df.copy()
        label_col = "labels_raw" if "labels_raw" in df.columns else "labels"
        df["labels_parsed"] = df[label_col].apply(safe_parse_labels)

        # Per-patient: use median age, mode sex, union of all labels seen
        agg = df.groupby("patient_id").agg(
            age=("age", "median"),
            sex=("sex", lambda s: s.mode()[0] if len(s) > 0 else "M"),
            labels_union=("labels_parsed", lambda ll: list(set(
                idx for lst in ll for idx in lst
            ))),
        ).reset_index()

        return agg

    def _build_disease_matrix(self, patient_df: pd.DataFrame) -> torch.Tensor:
        """Build binary [N_patients, 14] matrix from training disease labels."""
        N = len(patient_df)
        mat = torch.zeros(N, self.num_disease_classes, dtype=torch.float32)

        for i, row in patient_df.iterrows():
            labels = row["labels_union"]
            for nih_idx in labels:
                if nih_idx == 0:
                    continue  # No Finding
                disease_idx = nih_idx - 1  # NIH 1..14 → 0..13
                if 0 <= disease_idx < self.num_disease_classes:
                    mat[i, disease_idx] = 1.0

        return mat

    def _build_jaccard_edges(
        self,
        disease_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute pairwise Jaccard similarity and return edges above threshold.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        Uses vectorised computation. For N=30,805 patients this creates a
        30,805×30,805 matrix which is ~3.6 GB float32. For large datasets
        we use chunked computation.

        Returns:
            edge_index: [2, E] long tensor
            edge_attr:  [E] float tensor (Jaccard values)
        """
        N = disease_matrix.shape[0]
        logger.info("Computing pairwise Jaccard for %d patients (chunked)...", N)

        CHUNK = 512  # rows per chunk to avoid OOM
        src_list, dst_list, sim_list = [], [], []

        for i in range(0, N, CHUNK):
            chunk_a = disease_matrix[i : i + CHUNK]  # [chunk, 14]
            for j in range(i, N, CHUNK):
                chunk_b = disease_matrix[j : j + CHUNK]  # [chunk, 14]

                # intersection: [chunk_a, chunk_b]
                intersection = torch.mm(chunk_a, chunk_b.t())
                # |A| and |B|
                count_a = chunk_a.sum(dim=1, keepdim=True)  # [chunk_a, 1]
                count_b = chunk_b.sum(dim=1, keepdim=True)  # [chunk_b, 1]
                union = count_a + count_b.t() - intersection  # [chunk_a, chunk_b]

                # Avoid div-by-zero (both have no diseases)
                safe_union = union.clone()
                safe_union[safe_union == 0] = 1.0
                jaccard = intersection / safe_union
                # Pairs where both patients have zero diseases get Jaccard=0
                jaccard[(union == 0)] = 0.0

                # Find pairs above threshold (excluding self-loops)
                mask = jaccard >= self.edge_threshold
                # For the diagonal block (i==j), exclude exact diagonal
                if i == j:
                    diag_mask = torch.eye(chunk_a.shape[0], chunk_b.shape[0], dtype=torch.bool)
                    mask = mask & ~diag_mask
                # Only upper-triangle to avoid duplicates (add both directions later)
                if i == j:
                    upper = torch.triu(mask, diagonal=1)
                else:
                    upper = mask  # off-diagonal blocks are fully valid

                rows, cols = upper.nonzero(as_tuple=True)
                if len(rows) > 0:
                    sims = jaccard[rows, cols]
                    src_list.append(rows + i)
                    dst_list.append(cols + j)
                    sim_list.append(sims)

        if not src_list:
            logger.warning("No edges found with Jaccard >= %.2f", self.edge_threshold)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros(0, dtype=torch.float32)
            return edge_index, edge_attr

        src = torch.cat(src_list)
        dst = torch.cat(dst_list)
        sims = torch.cat(sim_list)

        # Make undirected: add reverse edges
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )
        edge_attr = torch.cat([sims, sims])

        logger.info("Jaccard edges: %d undirected edges.", edge_index.shape[1] // 2)
        return edge_index, edge_attr


# ── Save / Load Graph ────────────────────────────────────────────────────────

def save_graph(graph: "PygData", path: str) -> None:
    """Save a PyG Data graph to a .pt file."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)
    logger.info("Graph saved to %s", path)


def load_graph(path: str) -> Optional["PygData"]:
    """Load a PyG Data graph from a .pt file."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    graph = torch.load(path, map_location="cpu")
    logger.info("Graph loaded from %s", path)
    return graph


# ── Defensible Scan-to-Scan Graph (Patient-Grouping, Zero Label Leakage) ──────

class ScanGraphBuilder:
    """Build a scan-level graph using defensible scan-to-scan relationships.

    Clinical Rationale:
      - NODES: Each scan (radiograph record) is a distinct node.
      - EDGES: Connect scans (i, j) if they belong to the same patient (patient_id[i] == patient_id[j]).
        This directly captures longitudinal examination history and shared patient biology.
      - ZERO LABEL LEAKAGE: Edges are built strictly from patient_id, NEVER from disease labels.
      - SELF-LOOPS: Optional self-loops (i, i) ensure scans with only a single examination in the
        cohort/batch can still aggregate their own node features through message-passing.
      - NODE FEATURES: [age_norm, sex_binary] (shape [N, 2]) from metadata.
    """

    def __init__(self, add_self_loops: bool = True) -> None:
        self.add_self_loops = add_self_loops

    def build(
        self,
        df: pd.DataFrame,
        meta_encoder=None,
    ) -> Optional["PygData"]:
        """Build a PyG Data object of scan-to-scan relationships from metadata DataFrame."""
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.error("Cannot build scan graph: torch-geometric not installed.")
            return None

        if meta_encoder is None:
            from src.dataset.preprocessor import MetadataEncoder
            meta_encoder = MetadataEncoder()

        N = len(df)
        if N == 0:
            return None

        # 1. Node features: [N, 2]
        node_features = meta_encoder.encode_batch(
            ages=df["age"].tolist(),
            sexes=df["sex"].tolist(),
        )

        # 2. Defensible scan-to-scan edges from patient_id (no labels)
        groups = df.groupby("patient_id").indices
        src_list, dst_list = [], []
        for p_id, idxs in groups.items():
            if len(idxs) > 1:
                for i in idxs:
                    for j in idxs:
                        if i != j:
                            src_list.append(i)
                            dst_list.append(j)

        if self.add_self_loops:
            for i in range(N):
                src_list.append(i)
                dst_list.append(i)

        if len(src_list) > 0:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # 3. Multi-label targets (if available, for supervision only)
        label_col = "labels" if "labels" in df.columns else ("labels_raw" if "labels_raw" in df.columns else None)
        y = None
        if label_col is not None:
            from src.dataset.label_encoder import LabelEncoder
            from src.dataset.label_encoder import safe_parse_labels
            encoder = LabelEncoder()
            y_list = []
            for item in df[label_col]:
                parsed = safe_parse_labels(item)
                y_list.append(encoder.encode(parsed))
            y = torch.stack(y_list)  # [N, 14]

        logger.info(
            "ScanGraph built: %d nodes, %d edges (%d unique patients, self_loops=%s)",
            N, edge_index.shape[1], df["patient_id"].nunique(), self.add_self_loops
        )

        graph = PygData(
            x=node_features,
            edge_index=edge_index,
            y=y,
            num_nodes=N,
        )
        graph.patient_ids = df["patient_id"].tolist()
        if "scan_id" in df.columns:
            graph.scan_ids = df["scan_id"].tolist()

        return graph


def build_batch_scan_graph(
    meta_tensor: torch.Tensor,
    patient_ids: List[Any],
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct an online scan-to-scan graph for a mini-batch of scans.

    Args:
        meta_tensor: [B, 2] metadata feature tensor.
        patient_ids: Length-B list of patient IDs corresponding to each batch item.
        add_self_loops: Whether to include self-loops (i, i).

    Returns:
        (x, edge_index): x is [B, 2] and edge_index is [2, E_batch] on meta_tensor's device.
    """
    B = meta_tensor.shape[0]
    p_to_indices: dict = {}
    for idx, pid in enumerate(patient_ids):
        p_to_indices.setdefault(pid, []).append(idx)

    src_list, dst_list = [], []
    for pid, idxs in p_to_indices.items():
        if len(idxs) > 1:
            for i in idxs:
                for j in idxs:
                    if i != j:
                        src_list.append(i)
                        dst_list.append(j)

    if add_self_loops:
        for i in range(B):
            src_list.append(i)
            dst_list.append(i)

    if len(src_list) > 0:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=meta_tensor.device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=meta_tensor.device)

    return meta_tensor, edge_index

