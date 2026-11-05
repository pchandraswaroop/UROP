"""
components.py — Reusable UI Components for Streamlit Dashboard
==============================================================
Provides styled headers, medical disclaimer banners, mode badges,
pipeline diagrams, and prediction summary tables.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import streamlit as st

from src.inference.predictor import InferenceMode, MEDICAL_DISCLAIMER


def render_disclaimer() -> None:
    """Render mandatory medical disclaimer banner."""
    st.warning(f"**Medical Research Disclaimer:** {MEDICAL_DISCLAIMER}", icon="⚠️")


def render_header(title: str, subtitle: Optional[str] = None) -> None:
    """Render consistent clean research title banner."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()


def render_mode_badge(mode: InferenceMode) -> None:
    """Render visible status badge distinguishing Research from Demo mode."""
    if mode == InferenceMode.RESEARCH:
        st.success(
            "🔬 **RESEARCH INFERENCE MODE** — Model loaded with verified trained checkpoint weights.",
            icon="✅",
        )
    else:
        st.info(
            "🧪 **DEMO / INTERFACE PREVIEW MODE** — Interface and workflow demonstration with "
            "clearly labeled preview mock predictions. Never presented as real medical AI results.",
            icon="ℹ️",
        )


def render_pipeline_diagram() -> None:
    """Render multimodal architecture flow description."""
    st.markdown(
        """
        ```
        ┌─────────────────────────┐          ┌───────────────────────────────────┐
        │   Chest Radiograph      │          │   Clinical Metadata (Age, Sex)    │
        │      [B, 3, 224, 224]   │          │          [B, 2]                   │
        └────────────┬────────────┘          └─────────────────┬─────────────────┘
                     │                                         │
                     ▼                                         ▼
        ┌─────────────────────────┐          ┌───────────────────────────────────┐
        │   Image Branch (CNN/ViT)│          │  Clinical Branch (MLP / GraphSAGE)│
        │   + PennyLane Quantum   │          │  Patient Scan-to-Scan Graph       │
        └────────────┬────────────┘          └─────────────────┬─────────────────┘
                     │ [B, 64]                                 │ [B, 64]
                     └───────────────────┬─────────────────────┘
                                         ▼
                     ┌─────────────────────────────────────────┐
                     │  Disease-Specific Gating Multimodal     │
                     │  Fusion (14 Parallel Channels)          │
                     └───────────────────┬─────────────────────┘
                                         ▼
                     ┌─────────────────────────────────────────┐
                     │  14 Independent Sigmoid Output Heads    │
                     │  (Multi-Label Binary Classification)    │
                     └───────────────────┬─────────────────────┘
                                         ▼
                     ┌─────────────────────────────────────────┐
                     │  Explainability Suite                   │
                     │  • Grad-CAM++ / ViT Attribution         │
                     │  • Tabular SHAP Feature Importance      │
                     │  • GNN Edge Ablation Attribution        │
                     │  • Exact 3-Player Modality Shapley      │
                     └─────────────────────────────────────────┘
        ```
        """
    )


def render_prediction_table(
    probabilities: Dict[str, float],
    threshold: float,
    positive_predictions: List[str],
    is_no_finding: bool,
) -> None:
    """Render tabular view of 14 multi-label disease predictions."""
    if is_no_finding:
        st.success(
            "🟢 **Predicted Label: No Finding** — All 14 thoracic disease probabilities "
            f"are below the threshold ({threshold:.2f}).",
            icon="🩺",
        )
    else:
        st.error(
            f"🔴 **Positive Findings ({len(positive_predictions)}):** "
            + ", ".join(f"**{p}**" for p in positive_predictions),
            icon="🚨",
        )

    # Format table data
    table_data = []
    for disease, prob in probabilities.items():
        is_pos = prob >= threshold
        table_data.append({
            "Disease Target": disease,
            "Probability": f"{prob:.4f} ({prob * 100:.1f}%)",
            "Threshold": f"{threshold:.2f}",
            "Status": "POSITIVE" if is_pos else "Negative",
        })

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
    )
