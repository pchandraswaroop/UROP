"""
2_Explainability.py — Multimodal Explainability Dashboard
==========================================================
Interactive visualization of Grad-CAM++, ViT token attribution,
exact analytical tabular SHAP, GNN graph attribution, and 3-player
modality Shapley reliance.
"""

from __future__ import annotations

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.label_encoder import DISEASE_LABEL_NAMES
from src.inference.explanation_service import ExplanationService
from src.inference.predictor import (
    CXRInferenceService,
    InferenceMode,
    PredictionResult,
)
from src.ui.components import (
    render_disclaimer,
    render_header,
    render_mode_badge,
)
from src.ui.visualizations import plot_modality_breakdown, plot_shap_bar

st.set_page_config(page_title="Explainability Dashboard", page_icon="🔍", layout="wide")

if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = InferenceMode.DEMO
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

render_header(
    title="Explainability & Interpretability Dashboard",
    subtitle="Axiomatic and gradient-based attributions decomposed across visual, tabular, and graph modalities.",
)

render_disclaimer()
render_mode_badge(st.session_state["app_mode"])

curr_result: Optional[PredictionResult] = st.session_state["last_prediction"]

# If no prediction has been run yet, offer to generate a default demonstration prediction
if curr_result is None:
    st.warning("No recent inference result found. Please run a prediction on the Prediction page, or click below to generate a demo sample.")
    if st.button("Generate Demo Sample Prediction", type="primary"):
        with st.spinner("Generating sample prediction for explainability..."):
            service = CXRInferenceService()
            # Synthetic 224x224 gradient
            w, h = 224, 224
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            xx, yy = np.meshgrid(x, y)
            sample_arr = ((np.sin(xx * 2) * np.cos(yy * 2) * 0.5 + 0.5) * 255).astype(np.uint8)
            curr_result = service.predict(
                image_input=sample_arr,
                age=62,
                sex="M",
                experiment_name="quantum_gnn_multimodal",
                mode=InferenceMode.DEMO,
                threshold=0.35,
            )
            st.session_state["last_prediction"] = curr_result
            st.rerun()
    st.stop()

# --- Prediction Summary Header ---
st.subheader("Active Inference Context")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model Architecture", curr_result.model_name)
c2.metric("Positive Findings", len(curr_result.positive_predictions))
c3.metric("Decision Threshold (τ)", f"{curr_result.threshold:.2f}")
c4.metric("Graph Nodes / Edges", f"{curr_result.node_count} nodes / {curr_result.edge_count} edges")

# Target Disease Selection for Explainability
default_target = curr_result.positive_predictions[0] if curr_result.positive_predictions else "Cardiomegaly"
target_disease = st.selectbox(
    "Select Target Disease Pathology to Explain:",
    options=DISEASE_LABEL_NAMES,
    index=DISEASE_LABEL_NAMES.index(default_target) if default_target in DISEASE_LABEL_NAMES else 0,
    help="Grad-CAM++ and SHAP attributions are conditioned on individual disease logit gradients.",
)

target_prob = curr_result.probabilities.get(target_disease, 0.0)
st.caption(f"Predicted Probability for **{target_disease}**: **{target_prob:.4f}** ({target_prob * 100:.1f}%)")

st.divider()

# --- Section Tabs ---
tab_img, tab_tab, tab_graph, tab_mod = st.tabs([
    "🖼️ Image Attribution (Grad-CAM++ / ViT)",
    "📋 Clinical Attribution (SHAP)",
    "🕸️ Graph Attribution (GNNExplainer)",
    "⚖️ Multimodal Attribution (Shapley)",
])

# 1. IMAGE EXPLAINABILITY
with tab_img:
    st.subheader(f"Visual Saliency Attribution: {target_disease}")
    with st.spinner("Computing gradient-weighted class activation maps..."):
        img_exp = ExplanationService.get_image_explanation(curr_result, target_disease)

    if img_exp["status"] == "success":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original Radiograph**")
            st.image(curr_result.raw_rgb, use_container_width=True)
        with col2:
            st.markdown(f"**Heatmap: {target_disease}**")
            st.image(img_exp["heatmap"], clamp=True, use_container_width=True)
        with col3:
            st.markdown(f"**Grad-CAM++ Overlay**")
            st.image(img_exp["overlay"], use_container_width=True)

        st.caption(
            "Grad-CAM++ weights feature maps by second-order gradients corresponding to the selected pathology output, "
            "highlighting anatomical regions that drove the network's prediction."
        )
    elif img_exp["status"] == "unavailable":
        st.info(img_exp["message"], icon="ℹ️")
    else:
        st.error(f"Image Explainability Error: {img_exp.get('message', 'Unknown error')}")

# 2. CLINICAL EXPLAINABILITY
with tab_tab:
    st.subheader(f"Clinical Metadata Feature Importance (SHAP): {target_disease}")
    with st.spinner("Calculating exact 2-player Shapley values..."):
        tab_exp = ExplanationService.get_tabular_explanation(curr_result, target_disease)

    if tab_exp["status"] == "success":
        col_s1, col_s2 = st.columns([1.2, 1])
        with col_s1:
            shap_fig = plot_shap_bar(tab_exp["age_shap"], tab_exp["sex_shap"], target_disease)
            st.plotly_chart(shap_fig, use_container_width=True)
        with col_s2:
            st.markdown("#### Shapley Attribution Analysis")
            st.markdown(f"• **Age Impact:** `{tab_exp['age_shap']:+.4f}` log-odds shift")
            st.markdown(f"• **Sex Impact:** `{tab_exp['sex_shap']:+.4f}` log-odds shift")
            st.markdown(
                r"""
                > **Methodology Note:**
                > These values represent **exact analytical Shapley values** computed across all $2^2=4$
                > metadata coalitions: $\{Age\}$, $\{Sex\}$, $\{Age, Sex\}$, and $\emptyset$.
                > Disease target labels are strictly **never** used as input features (zero label leakage).
                """
            )
    else:
        st.error(f"Tabular Explainability Error: {tab_exp.get('message')}")

# 3. GRAPH EXPLAINABILITY
with tab_graph:
    st.subheader(f"Patient Scan-to-Scan Graph Explanation: {target_disease}")
    with st.spinner("Evaluating graph edge ablation..."):
        graph_exp = ExplanationService.get_graph_explanation(curr_result, target_disease)

    if graph_exp["status"] == "success":
        st.pyplot(graph_exp["figure"])
        st.markdown("#### Influential Longitudinal Connections")
        st.write(graph_exp["edge_importance"])
    elif graph_exp["status"] == "single_scan_note":
        st.info(graph_exp["message"], icon="ℹ️")
        st.markdown(
            """
            **Clinical Graph Rationale:**
            - In the NIH ChestX-ray14 dataset, graph edges connect scans belonging to the **same patient** (`patient_id[i] == patient_id[j]`).
            - This captures longitudinal disease progression and patient biology across multiple clinical visits.
            - When predicting a single isolated scan, no prior examinations exist in the query batch, so the graph operates with a self-loop.
            """
        )
    elif graph_exp["status"] == "unavailable":
        st.info(graph_exp["message"], icon="ℹ️")
    else:
        st.error(f"Graph Explainability Error: {graph_exp.get('message')}")

# 4. MODALITY EXPLAINABILITY
with tab_mod:
    st.subheader("Disease-Specific Multimodal Reliance (Exact 3-Player Shapley Decomposition)")
    with st.spinner("Computing 3-player Shapley game across Image, Clinical Meta, and Graph..."):
        mod_exp = ExplanationService.get_modality_attribution(curr_result)

    if mod_exp["status"] == "success":
        mod_fig = plot_modality_breakdown(
            mod_exp["pct_image"],
            mod_exp["pct_meta"],
            mod_exp["pct_graph"],
            DISEASE_LABEL_NAMES,
        )
        st.plotly_chart(mod_fig, use_container_width=True)

        st.markdown("#### Relative Modality Percentage Breakdown Table")
        st.dataframe(mod_exp["summary_table"], use_container_width=True, hide_index=True)

        st.warning(f"⚠️ **Clinical Disclaimer:** {mod_exp['disclaimer']}", icon="⚠️")
    else:
        st.error(f"Modality Attribution Error: {mod_exp.get('message')}")
