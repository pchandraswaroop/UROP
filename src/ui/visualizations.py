"""
visualizations.py — Interactive Plotly & Matplotlib Charts
==========================================================
Produces interactive probability charts, SHAP attribution bars,
modality breakdown graphs, and image overlays.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def plot_probability_chart(probabilities: Dict[str, float], threshold: float = 0.5) -> go.Figure:
    """Generate an interactive Plotly horizontal bar chart for the 14 multi-label outputs."""
    diseases = list(probabilities.keys())
    probs = [probabilities[d] for d in diseases]

    colors = ["#e74c3c" if p >= threshold else "#2980b9" for p in probs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=probs,
        y=diseases,
        orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="auto",
        hoverinfo="y+x",
    ))

    # Add threshold vertical dashed reference line
    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dash",
        line_color="#e67e22",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="top right",
    )

    fig.update_layout(
        title="<b>14-Disease Multi-Label Predicted Probabilities</b> (Independent Sigmoid)",
        xaxis_title="Predicted Probability",
        yaxis_title="Thoracic Pathology",
        xaxis=dict(range=[0.0, 1.05], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        height=520,
        margin=dict(l=10, r=20, t=40, b=30),
        template="plotly_white",
    )
    return fig


def plot_shap_bar(age_shap: float, sex_shap: float, disease_name: str) -> go.Figure:
    """Generate an interactive Plotly bar chart for Age and Sex SHAP values."""
    features = ["Age", "Sex"]
    values = [age_shap, sex_shap]
    colors = ["#27ae60" if v >= 0 else "#c0392b" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=features,
        y=values,
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"<b>Clinical Metadata SHAP Feature Attribution</b> ({disease_name})",
        yaxis_title="Shapley Value (Log-Odds Impact)",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_modality_breakdown(
    pct_image: np.ndarray,
    pct_meta: np.ndarray,
    pct_graph: np.ndarray,
    disease_names: List[str],
) -> go.Figure:
    """Generate stacked horizontal bar chart for 3-modality Shapley reliance."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Image",
        y=disease_names,
        x=pct_image,
        orientation="h",
        marker_color="#3498db",
    ))
    fig.add_trace(go.Bar(
        name="Clinical Metadata",
        y=disease_names,
        x=pct_meta,
        orientation="h",
        marker_color="#2ecc71",
    ))
    fig.add_trace(go.Bar(
        name="Scan Graph",
        y=disease_names,
        x=pct_graph,
        orientation="h",
        marker_color="#9b59b6",
    ))

    fig.update_layout(
        barmode="stack",
        title="<b>Relative Modality Reliance by Disease</b> (Exact 3-Player Shapley Decomposition)",
        xaxis_title="Percentage Contribution (%)",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 100]),
        height=540,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=20, t=50, b=30),
    )
    return fig
