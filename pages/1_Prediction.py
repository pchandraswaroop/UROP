"""
1_Prediction.py — Multi-Label CXR Disease Prediction Page
==========================================================
Enables image upload, metadata entry, model configuration selection,
and 14-disease multi-label inference with configurable thresholding.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.label_encoder import DISEASE_LABEL_NAMES
from src.inference.checkpoint_loader import CheckpointError, ModelCheckpointLoader
from src.inference.predictor import (
    CXRInferenceService,
    InferenceMode,
    InputValidationError,
    PredictionResult,
)
from src.ui.components import (
    render_disclaimer,
    render_header,
    render_mode_badge,
    render_prediction_table,
)
from src.ui.visualizations import plot_probability_chart

st.set_page_config(page_title="Disease Prediction", page_icon="🔬", layout="wide")

if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = InferenceMode.DEMO
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

render_header(
    title="Multi-Disease Chest X-ray Prediction",
    subtitle="Generate multi-label predictions across 14 thoracic disease targets using independent sigmoid outputs.",
)

render_disclaimer()
render_mode_badge(st.session_state["app_mode"])

# --- Layout: Inputs (Left) and Results (Right) ---
col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("1. Clinical Inputs")

    # Image upload: supports one or multiple scans for a patient
    uploaded_files = st.file_uploader(
        "Upload Chest Radiographs (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload one or more frontal (PA/AP) chest X-rays belonging to the same patient.",
    )

    # Demo sample button if user has no image
    use_sample = False
    if not uploaded_files:
        st.caption("Don't have an X-ray on hand?")
        use_sample = st.checkbox("Use a synthetic sample radiograph for testing", value=True)

    input_images: List[Image.Image] = []
    primary_scan_idx: int = 0
    confirm_same_patient: bool = True

    if uploaded_files:
        n_scans = len(uploaded_files)
        st.write(f"📁 **{n_scans} Radiograph{'s' if n_scans > 1 else ''} Selected**")

        # Patient confirmation requirement when multiple scans are provided
        if n_scans > 1:
            confirm_same_patient = st.checkbox(
                "✅ I confirm that all uploaded radiographs belong to the same patient",
                value=True,
                help=(
                    "For patient privacy and clinical validity, all scans uploaded in one prediction session "
                    "are treated as belonging to the same patient only after explicit confirmation. "
                    "These scans are connected into a temporary graph and never saved or mixed across patients."
                ),
            )
            if not confirm_same_patient:
                st.warning("⚠️ Scans cannot be linked into a patient graph without confirming they belong to the same patient.")

            primary_scan_idx = st.selectbox(
                "Select Primary / Index Scan to Diagnose:",
                options=list(range(n_scans)),
                format_func=lambda i: f"Scan #{i+1} — {uploaded_files[i].name} {'(Current Study)' if i == 0 else '(Comparison Study)'}",
                help="The GNN will aggregate relational features across all scans to evaluate this primary scan.",
            )

        # Load images
        for f in uploaded_files:
            try:
                img = Image.open(f).convert("RGB")
                input_images.append(img)
            except Exception as e:
                st.error(f"Failed to load {f.name}: {e}")

        # Display preview thumbnails
        if input_images:
            num_cols = min(len(input_images), 3)
            cols = st.columns(num_cols)
            for idx, img in enumerate(input_images):
                with cols[idx % num_cols]:
                    is_prim = (idx == primary_scan_idx)
                    caption = f"Scan #{idx+1} {'⭐ Primary' if is_prim else 'Prior'}"
                    st.image(img, caption=caption, width="stretch")

            if n_scans > 1 and confirm_same_patient:
                st.info(
                    f"🔗 **Multi-Scan Graph Active:** {n_scans} scans will be structured into a patient-specific graph. "
                    f"The GNN branch will learn inter-scan visual relationships to support diagnosis."
                )
            elif n_scans == 1:
                st.caption(
                    "ℹ️ **Single-Scan Mode:** A 1-node patient graph with a self-loop will be constructed. "
                    "The GNN executes self-aggregation (no companion scans to aggregate)."
                )

    elif use_sample:
        # Create a clean synthetic CXR placeholder image
        w, h = 224, 224
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        synthetic_pattern = (np.sin(xx * 3) * np.cos(yy * 3) * 0.5 + 0.5) * 200
        sample_img = Image.fromarray(synthetic_pattern.astype(np.uint8)).convert("RGB")
        input_images = [sample_img]
        primary_scan_idx = 0
        st.image(sample_img, caption="Synthetic Sample Radiograph (Testing)", width="stretch")
        st.caption("ℹ️ **Single-Scan Mode:** 1-node patient graph with self-loop.")

    st.divider()

    # Patient Metadata (provided once for the patient)
    st.subheader("2. Patient Metadata")
    st.caption("Patient demographic attributes are provided once for all scans of this patient.")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        age_input = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=54,
            step=1,
            help="Patient chronological age in years (0 - 120).",
        )
    with m_col2:
        sex_input = st.selectbox(
            "Patient Sex",
            options=["M", "F"],
            format_func=lambda x: "Male (M)" if x == "M" else "Female (F)",
            help="Biological sex recorded for patient.",
        )

    st.divider()

    # Model Configuration
    st.subheader("3. Model Configuration")
    exp_options = [
        "quantum_gnn_multimodal",
        "classical_multimodal",
        "cnn_only",
        "vit_only",
        "quantum_cnn",
        "mlp_clinical_only",
        "gnn_only",
        "disease_specific_fusion",
        "shared_fusion",
    ]
    selected_exp = st.selectbox(
        "Select Architecture Variant:",
        options=exp_options,
        index=0,
        help="Select the exact research architecture variant to run.",
    )

    threshold_val = st.slider(
        "Positive Decision Threshold (τ):",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Pathologies with predicted probability >= τ are classified as positive findings.",
    )

    # Checkpoint configuration for research mode
    custom_ckpt_path = None
    if st.session_state["app_mode"] == InferenceMode.RESEARCH:
        available_ckpts = ModelCheckpointLoader.list_available_checkpoints()
        exp_ckpts = available_ckpts.get(selected_exp, [])
        if exp_ckpts:
            custom_ckpt_path = st.selectbox(
                "Select Trained Checkpoint:",
                options=[str(p) for p in exp_ckpts],
            )
        else:
            st.error(
                f"No checkpoint found in `checkpoints/{selected_exp}/`. "
                "In Research Mode, real inference is blocked without a trained checkpoint."
            )
            custom_ckpt_path = st.text_input(
                "Or specify absolute checkpoint path (.pt):",
                value="",
            )

    can_run = (len(input_images) > 0) and confirm_same_patient
    run_btn = st.button(
        "Run Multi-Disease Inference",
        type="primary",
        width="stretch",
        disabled=not can_run,
    )

with col_output:
    st.subheader("Inference Results & Probabilities")

    if run_btn:
        if not input_images:
            st.error("Please upload or select an image first.")
        else:
            with st.spinner("Executing multi-label prediction pipeline..."):
                service = CXRInferenceService()
                try:
                    result = service.predict(
                        image_input=input_images,
                        age=age_input,
                        sex=sex_input,
                        experiment_name=selected_exp,
                        mode=st.session_state["app_mode"],
                        checkpoint_path=custom_ckpt_path if custom_ckpt_path else None,
                        threshold=threshold_val,
                        primary_scan_index=primary_scan_idx,
                    )
                    st.session_state["last_prediction"] = result
                except CheckpointError as ce:
                    st.error(f"**Checkpoint Error:** {ce}")
                    st.info("Tip: Switch to **DEMO MODE** in the sidebar to preview the dashboard interface.")
                except InputValidationError as ie:
                    st.error(f"**Input Validation Error:** {ie}")
                except Exception as ex:
                    st.error(f"**Inference Error:** {ex}")

    # Display results if available in session
    curr_result: Optional[PredictionResult] = st.session_state["last_prediction"]

    if curr_result is not None:
        # Show mode tag
        if curr_result.is_demo:
            st.warning("⚠️ **DEMO PREVIEW OUTPUT:** The probabilities below are mock demonstration values.", icon="🧪")
        else:
            st.success(f"✅ **RESEARCH PREDICTION:** Loaded checkpoint: `{curr_result.checkpoint_info.get('path', 'verified')}`", icon="🔬")

        # Graph Context Banner
        if curr_result.is_multi_scan:
            st.success(
                f"🔗 **Patient Scan Graph Analyzed:** {curr_result.node_count} patient scans connected by "
                f"{curr_result.edge_count} relational edges. Displaying primary findings for Scan #{curr_result.primary_scan_index + 1}.",
                icon="🩻",
            )
        else:
            st.info(
                f"ℹ️ **Single-Scan Analysis:** 1 radiograph analyzed in an isolated 1-node graph (with self-loop).",
                icon="🩻",
            )

        # Threshold and positive findings summary
        render_prediction_table(
            curr_result.probabilities,
            curr_result.threshold,
            curr_result.positive_predictions,
            curr_result.is_no_finding,
        )

        st.divider()

        # Probability Bar Chart
        fig = plot_probability_chart(curr_result.probabilities, threshold=curr_result.threshold)
        st.plotly_chart(fig, width="stretch")

        # If multi-scan, allow exploring all scans
        if curr_result.is_multi_scan and curr_result.all_probabilities:
            with st.expander(f"📑 View Findings Across All {curr_result.node_count} Patient Scans"):
                for k, scan_probs in enumerate(curr_result.all_probabilities):
                    st.markdown(f"**Scan #{k+1} {'(Primary Study)' if k == curr_result.primary_scan_index else '(Prior Study)'}**")
                    pos = [name for name, p in scan_probs.items() if p >= curr_result.threshold]
                    if pos:
                        st.write(f"Positive Findings ({len(pos)}): **{', '.join(pos)}**")
                    else:
                        st.write("No findings exceed threshold.")

        # Call-to-action to inspect explainability
        st.info(
            "💡 **Proceed to Explainability:** Want to understand which image regions, clinical features, "
            "or graph edges drove these predictions? Open the **Explainability Dashboard** from the sidebar.",
            icon="🔍",
        )
    else:
        st.info("Upload one or more radiographs and click **'Run Multi-Disease Inference'** to generate predictions.")
