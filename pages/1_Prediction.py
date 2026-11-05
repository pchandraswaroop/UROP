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

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Chest Radiograph (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload a frontal (PA/AP) chest X-ray image for diagnosis.",
    )

    # Demo sample button if user has no image
    use_sample = False
    if uploaded_file is None:
        st.caption("Don't have an X-ray on hand?")
        use_sample = st.checkbox("Use a synthetic sample radiograph for testing", value=True)

    # Display image preview
    preview_image: Optional[Image.Image] = None
    if uploaded_file is not None:
        try:
            preview_image = Image.open(uploaded_file).convert("RGB")
            st.image(preview_image, caption="Uploaded Radiograph", width="stretch")
        except Exception as e:
            st.error(f"Failed to read image: {e}")
    elif use_sample:
        # Create a clean synthetic synthetic CXR placeholder image
        w, h = 224, 224
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        synthetic_pattern = (np.sin(xx * 3) * np.cos(yy * 3) * 0.5 + 0.5) * 200
        preview_image = Image.fromarray(synthetic_pattern.astype(np.uint8)).convert("RGB")
        st.image(preview_image, caption="Synthetic Sample Radiograph (Testing)", width="stretch")

    st.divider()

    # Patient Metadata
    st.subheader("2. Patient Metadata")
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
            help="Biological sex recorded at scan time.",
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
        value=0.35,
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

    run_btn = st.button("Run Multi-Disease Inference", type="primary", width="stretch")

with col_output:
    st.subheader("Inference Results & Probabilities")

    if run_btn:
        if preview_image is None:
            st.error("Please upload or select an image first.")
        else:
            with st.spinner("Executing multi-label prediction pipeline..."):
                service = CXRInferenceService()
                try:
                    result = service.predict(
                        image_input=preview_image,
                        age=age_input,
                        sex=sex_input,
                        experiment_name=selected_exp,
                        mode=st.session_state["app_mode"],
                        checkpoint_path=custom_ckpt_path if custom_ckpt_path else None,
                        threshold=threshold_val,
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

        # Call-to-action to inspect explainability
        st.info(
            "💡 **Proceed to Explainability:** Want to understand which image regions, clinical features, "
            "or graph edges drove these predictions? Open the **Explainability Dashboard** from the sidebar.",
            icon="🔍",
        )
    else:
        st.info("Upload an image and click **'Run Multi-Disease Inference'** to generate predictions.")
