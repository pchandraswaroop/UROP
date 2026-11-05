"""
app.py — Main Entry Point for Streamlit Research Application
============================================================
Explainable Multimodal Quantum–GNN Framework for Multi-Disease Chest X-ray Diagnosis
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.label_encoder import DISEASE_LABEL_NAMES
from src.inference.checkpoint_loader import ModelCheckpointLoader
from src.inference.predictor import InferenceMode
from src.ui.components import (
    render_disclaimer,
    render_header,
    render_mode_badge,
    render_pipeline_diagram,
)

# Page configuration
st.set_page_config(
    page_title="Multimodal Quantum-GNN CXR Diagnosis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = InferenceMode.DEMO
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

# --- Sidebar Configuration ---
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/PyTorch-Dark.svg",
        width=48,
    )
    st.title("Framework Controls")

    # Mode Selector
    st.subheader("Application Mode")
    available_ckpts = ModelCheckpointLoader.list_available_checkpoints()
    has_checkpoints = len(available_ckpts) > 0

    mode_options = [InferenceMode.DEMO.value, InferenceMode.RESEARCH.value]
    mode_selection = st.radio(
        "Select Operation Mode:",
        options=mode_options,
        index=0 if st.session_state["app_mode"] == InferenceMode.DEMO else 1,
        help=(
            "RESEARCH MODE: Requires a valid trained checkpoint and runs actual neural network inference.\n\n"
            "DEMO MODE: Interface preview with synthetic preview mock outputs clearly labeled with DEMO tags."
        ),
    )
    st.session_state["app_mode"] = InferenceMode(mode_selection)

    st.divider()

    # Checkpoint Status
    st.subheader("Checkpoint Status")
    if has_checkpoints:
        st.success(f"Found {sum(len(v) for v in available_ckpts.values())} checkpoint(s)")
        for exp_name, paths in available_ckpts.items():
            st.caption(f"• **{exp_name}**: {len(paths)} checkpoint(s)")
    else:
        st.warning("No checkpoints found in `checkpoints/`.")
        if st.session_state["app_mode"] == InferenceMode.RESEARCH:
            st.error(
                "Research Mode is selected, but no trained checkpoints exist. Real inference will be "
                "disabled until a `.pt` checkpoint is provided."
            )

    st.divider()
    st.caption("UROP Research Prototype • Multi-Disease CXR Diagnosis")

# --- Main Content ---
render_header(
    title="Explainable Multimodal Quantum–GNN Framework for Multi-Disease CXR Diagnosis",
    subtitle="An end-to-end research architecture combining Deep Vision, PennyLane Quantum Circuits, Graph Neural Networks, and Axiomatic Explainability.",
)

render_disclaimer()
render_mode_badge(st.session_state["app_mode"])

# Tabs for overview
tab1, tab2, tab3 = st.tabs(["🏛️ Architecture & Objectives", "📊 14 Disease Targets", "🚀 Navigation Guide"])

with tab1:
    st.markdown("### Research Objectives")
    st.markdown(
        r"""
        Thoracic diseases frequently co-occur with complex comorbidity patterns. Standard unimodal computer vision
        models fail to exploit patient-level longitudinal history and clinical metadata. This research framework introduces:

        1. **Dual-Branch Multimodal Architecture**:
           - **Image Branch**: Pretrained CNNs (EfficientNet-B0, ResNet50, DenseNet121) or Vision Transformer (`vit_small_patch16_224`) compressed into a variational quantum circuit.
           - **Quantum Layer**: 4-qubit parameterized quantum circuit simulated using PennyLane with AngleEmbedding and StrongEntanglingLayers.
           - **Clinical Branch**: Metadata feature embedding (Age, Sex) and a GraphSAGE Graph Neural Network operating over longitudinal patient scan graphs.
        2. **Disease-Specific Gating Fusion**:
           - 14 parallel gating units enabling independent, disease-specific representations for each thoracic pathology.
        3. **Independent Multi-Label Classification**:
           - 14 independent sigmoid output units. Multi-disease cases are handled naturally.
           - **`No Finding`** is defined mathematically as the absence of all 14 diseases ($\hat{y}_c < \tau \ \forall c$), **not** as a 15th classification target.
        """
    )

    st.markdown("### Multimodal Pipeline Architecture")
    render_pipeline_diagram()

with tab2:
    st.markdown("### 14 Canonical Thoracic Disease Targets")
    st.markdown(
        "Predictions are made simultaneously across 14 independent pathology heads (NIH ChestX-ray14 dataset standard):"
    )

    cols = st.columns(3)
    for idx, disease in enumerate(DISEASE_LABEL_NAMES):
        with cols[idx % 3]:
            st.markdown(f"**{idx + 1}. {disease}**")

    st.info(
        "**Multi-Label Multi-Task Prediction:** Unlike single-label classification with Softmax where probabilities "
        "are forced to sum to 100%, each pathology here has an independent probability between 0% and 100%. "
        "A radiograph may simultaneously test positive for Cardiomegaly and Effusion.",
        icon="ℹ️",
    )

with tab3:
    st.markdown("### Dashboard Pages")
    st.markdown(
        """
        Navigate using the sidebar to explore the research system:

        - 🔬 **[1. Prediction](pages/1_Prediction.py)**: Upload a chest radiograph, enter patient age/sex, choose the model architecture, and generate 14-disease multi-label predictions with configurable thresholds.
        - 🔍 **[2. Explainability](pages/2_Explainability.py)**: Inspect Grad-CAM++ spatial heatmaps, exact analytical SHAP clinical attributions, patient scan graph connectivity, and 3-player modality Shapley reliance.
        - 📐 **[3. Model Information](pages/3_Model_Information.py)**: Compare the 9 experiment variants, model parameter counts, quantum circuit configurations, and inference devices.
        - 📈 **[4. Research Evaluation](pages/4_Research_Evaluation.py)**: View real experiment results, ROC-AUC, macro-F1, training loss curves, and quantitative faithfulness metrics.
        """
    )
