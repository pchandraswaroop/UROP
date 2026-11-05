"""
3_Model_Information.py — Architecture Specifications & Model Comparison
========================================================================
Comprehensive architectural breakdown of the 9 experiment variants,
quantum circuit design, GraphSAGE message passing, and gating fusion.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import torch
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.checkpoint_loader import ModelCheckpointLoader
from src.inference.predictor import InferenceMode
from src.ui.components import (
    render_disclaimer,
    render_header,
    render_mode_badge,
)

st.set_page_config(page_title="Model Information", page_icon="📐", layout="wide")

if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = InferenceMode.DEMO

render_header(
    title="Model Architecture & Component Specifications",
    subtitle="Detailed technical specifications of the 9 research architectures, quantum layers, and GNN encoders.",
)

render_disclaimer()
render_mode_badge(st.session_state["app_mode"])

# --- System & Device Status ---
device_name = "CUDA (NVIDIA GPU)" if torch.cuda.is_available() else "CPU"
vram_info = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "System RAM"

st.subheader("Runtime Environment & Inference Engine")
e1, e2, e3, e4 = st.columns(4)
e1.metric("Compute Device", device_name)
e2.metric("Available Memory", vram_info)
e3.metric("PyTorch Version", torch.__version__)
e4.metric("Disease Outputs", "14 (Sigmoid)")

st.divider()

# --- Model Variants Comparison Table ---
st.subheader("Comparison of the 9 Architecture Variants")

variants_data = [
    {
        "Variant": "cnn_only",
        "Image Backbone": "EfficientNet-B0 (1280-d)",
        "Quantum Layer": "None (Identity)",
        "Clinical Encoder": "MLP (Age, Sex)",
        "Graph Branch": "None",
        "Fusion Type": "Concatenation",
        "Classification Head": "Shared 14-head MLP",
    },
    {
        "Variant": "vit_only",
        "Image Backbone": "ViT-Small (384-d)",
        "Quantum Layer": "None (Identity)",
        "Clinical Encoder": "MLP (Age, Sex)",
        "Graph Branch": "None",
        "Fusion Type": "Concatenation",
        "Classification Head": "Shared 14-head MLP",
    },
    {
        "Variant": "quantum_cnn",
        "Image Backbone": "EfficientNet-B0",
        "Quantum Layer": "PennyLane 4-Qubit Circuit",
        "Clinical Encoder": "MLP (Age, Sex)",
        "Graph Branch": "None",
        "Fusion Type": "Disease-Specific Gating",
        "Classification Head": "Per-Disease Heads (14x)",
    },
    {
        "Variant": "mlp_clinical_only",
        "Image Backbone": "ZeroImageBranch (Ablated)",
        "Quantum Layer": "None",
        "Clinical Encoder": "MLP Metadata Embedder",
        "Graph Branch": "None",
        "Fusion Type": "Concatenation",
        "Classification Head": "Shared 14-head MLP",
    },
    {
        "Variant": "gnn_only",
        "Image Backbone": "ZeroImageBranch (Ablated)",
        "Quantum Layer": "None",
        "Clinical Encoder": "Metadata Fallback",
        "Graph Branch": "GraphSAGE (2-layer)",
        "Fusion Type": "Concatenation",
        "Classification Head": "Shared 14-head MLP",
    },
    {
        "Variant": "classical_multimodal",
        "Image Backbone": "EfficientNet-B0",
        "Quantum Layer": "None (Identity)",
        "Clinical Encoder": "MLP Metadata Embedder",
        "Graph Branch": "None",
        "Fusion Type": "Concatenation",
        "Classification Head": "Shared 14-head MLP",
    },
    {
        "Variant": "quantum_gnn_multimodal",
        "Image Backbone": "EfficientNet-B0",
        "Quantum Layer": "PennyLane 4-Qubit Circuit",
        "Clinical Encoder": "Metadata Fallback",
        "Graph Branch": "GraphSAGE (2-layer)",
        "Fusion Type": "Disease-Specific Gating",
        "Classification Head": "Per-Disease Heads (14x)",
    },
    {
        "Variant": "disease_specific_fusion",
        "Image Backbone": "EfficientNet-B0",
        "Quantum Layer": "PennyLane 4-Qubit Circuit",
        "Clinical Encoder": "Metadata Fallback",
        "Graph Branch": "GraphSAGE (2-layer)",
        "Fusion Type": "Disease-Specific Gating",
        "Classification Head": "Per-Disease Heads (14x)",
    },
    {
        "Variant": "shared_fusion",
        "Image Backbone": "EfficientNet-B0",
        "Quantum Layer": "PennyLane 4-Qubit Circuit",
        "Clinical Encoder": "MLP Metadata Embedder",
        "Graph Branch": "None",
        "Fusion Type": "Concatenation",
        "Classification Head": "Shared 14-head MLP",
    },
]

df_variants = pd.DataFrame(variants_data)
st.dataframe(df_variants, width="stretch", hide_index=True)

st.divider()

# --- Deep Component Breakdown ---
st.subheader("Architectural Subsystem Specifications")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("### 1. Quantum-Classical Variational Circuit")
    st.markdown(
        r"""
        - **Framework:** PennyLane with PyTorch QNode interface.
        - **Device:** `default.qubit` (or GPU-accelerated `lightning.qubit`).
        - **Topology:** 4 Qubits with 2 Entangling Layers.
        - **Embedding:** `qml.AngleEmbedding` mapping 4 classical features into qubit rotation angles.
        - **Entanglement:** `qml.StronglyEntanglingLayers` parameterised by learnable rotational weights.
        - **Measurement:** Expectation values $\langle Z_i \rangle$ across all 4 Pauli-Z observables.
        - **Compression:** High-dimensional classical features (1280-d) are linear-projected to 4-d, transformed in Hilbert space, and projected to a 64-d quantum representation.
        """
    )

    st.markdown("### 2. Longitudinal Scan Graph & GNN Branch")
    st.markdown(
        r"""
        - **Node Connectivity:** Edges exist strictly between radiographs of the **same patient** (`patient_id[i] == patient_id[j]`).
        - **Zero Label Leakage:** Node features contain only normalized age and binary sex. Target disease labels are **never** present in input graphs.
        - **Architecture:** 2-layer GraphSAGE (`SAGEConv`) with Mean Aggregation and Layer Normalization.
        - **Self-Loops:** Added to all nodes to preserve ego-scan features during message passing.
        """
    )

with col_right:
    st.markdown("### 3. Disease-Specific Gating Multimodal Fusion")
    st.markdown(
        r"""
        - **Mathematical Formulation:**
          For each disease head $d \in \{1, \dots, 14\}$:
          $$\alpha_d = \sigma(W_{\alpha, d} \cdot [z_{\text{img}} \parallel z_{\text{clin}}] + b_{\alpha, d})$$
          $$f_d = \alpha_d \odot W_{\text{img}, d} z_{\text{img}} + (1 - \alpha_d) \odot W_{\text{clin}, d} z_{\text{clin}}$$
        - **Interpretability Advantage:** Each disease learns independent gating weights $\alpha_d \in [0, 1]$, allowing visual features to dominate structural lung diseases (e.g. Cardiomegaly) while clinical history contributes to systemic or age-correlated conditions.
        """
    )

    st.markdown("### 4. Classification & Loss Function")
    st.markdown(
        r"""
        - **Classification Layer:** 14 independent linear heads, followed by Sigmoid activation.
        - **Loss Function:** Class-Imbalanced Weighted Binary Cross-Entropy (BCE):
          $$\mathcal{L} = - \sum_{d=1}^{14} \left[ w_d \cdot y_d \log \hat{y}_d + (1 - y_d) \log (1 - \hat{y}_d) \right]$$
        - **Positive Weighting:** Computed inversely to disease frequency: $w_d = \min\left(\frac{N - N_d}{N_d}, 10.0\right)$.
        """
    )

st.divider()

# --- Checkpoint Inventory ---
st.subheader("Trained Model Checkpoint Inventory")
ckpts = ModelCheckpointLoader.list_available_checkpoints()
if ckpts:
    for exp, files in ckpts.items():
        st.markdown(f"**Experiment:** `{exp}`")
        for f in files:
            st.markdown(f"• `{f.name}` ({f.stat().st_size / 1e6:.1f} MB)")
else:
    st.info(
        "No trained `.pt` checkpoints currently exist in `checkpoints/`. "
        "Training experiments should be executed on Google Colab GPU (`notebooks/02_model_training.ipynb`) "
        "and checkpoint files placed in the `checkpoints/<experiment_name>/` folder for real inference.",
        icon="ℹ️",
    )
