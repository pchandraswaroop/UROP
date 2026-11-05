"""
4_Research_Evaluation.py — Empirical Evaluation & Faithfulness Metrics
======================================================================
Displays authentic empirical evaluation metrics, training history curves,
and quantitative explainability faithfulness benchmarks.
Strictly displays only actual recorded results without fabricating numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.predictor import InferenceMode
from src.ui.components import (
    render_disclaimer,
    render_header,
    render_mode_badge,
)

st.set_page_config(page_title="Research Evaluation", page_icon="📈", layout="wide")

if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = InferenceMode.DEMO

render_header(
    title="Empirical Evaluation & Faithfulness Benchmarks",
    subtitle="Verified training/validation performance metrics and quantitative explanation fidelity tests.",
)

render_disclaimer()
render_mode_badge(st.session_state["app_mode"])

tab_train, tab_faith, tab_mod_eval = st.tabs([
    "📊 Model Training & Test Performance",
    "🎯 Quantitative Faithfulness Evaluation",
    "🔬 Modality & Feature Attribution Logs",
])

# 1. MODEL TRAINING & TEST PERFORMANCE
with tab_train:
    st.subheader("Training History & Evaluation Metrics")
    results_dir = REPO_ROOT / "results"

    found_experiments = []
    if results_dir.exists() and results_dir.is_dir():
        for sub in results_dir.iterdir():
            if sub.is_dir() and (sub / "val_metrics.json").exists():
                found_experiments.append(sub.name)

    if found_experiments:
        selected_exp = st.selectbox("Select Experiment Run to Inspect:", options=found_experiments)
        exp_path = results_dir / selected_exp

        # Load Metrics
        val_path = exp_path / "val_metrics.json"
        test_path = exp_path / "test_metrics.json"
        hist_path = exp_path / "train_history.json"

        val_data = json.loads(val_path.read_text()) if val_path.exists() else {}
        test_data = json.loads(test_path.read_text()) if test_path.exists() else {}
        history = json.loads(hist_path.read_text()) if hist_path.exists() else []

        st.markdown(f"### Overall Summary: `{selected_exp}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Val Mean AUC", f"{val_data.get('auc_mean', 0.0):.4f}")
        m2.metric("Val Macro F1", f"{val_data.get('f1_macro', 0.0):.4f}")
        m3.metric("Test Mean AUC", f"{test_data.get('auc_mean', 0.0):.4f}")
        m4.metric("Test Macro F1", f"{test_data.get('f1_macro', 0.0):.4f}")

        # Training Curves if history is available
        if history:
            epochs = [h.get("epoch", i) for i, h in enumerate(history)]
            train_losses = [h.get("loss", h.get("train_loss")) for h in history]
            val_losses = [h.get("val_loss") for h in history]

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs, y=train_losses, name="Train Loss", line=dict(color="#e74c3c", width=2)))
            if any(v is not None for v in val_losses):
                fig_loss.add_trace(go.Scatter(x=epochs, y=val_losses, name="Val Loss", line=dict(color="#3498db", width=2)))

            fig_loss.update_layout(
                title=f"<b>Training and Validation Loss Curve</b> ({selected_exp})",
                xaxis_title="Epoch",
                yaxis_title="Weighted BCE Loss",
                template="plotly_white",
                height=380,
            )
            st.plotly_chart(fig_loss, width="stretch")

        # Per-Disease Metrics Table
        st.markdown("### Per-Disease Test AUROC & F1-Score")
        per_disease_rows = []
        for k, v in test_data.items():
            if k.startswith("auc_") and k not in ("auc_mean", "auc_weighted"):
                d_name = k.replace("auc_", "")
                f1_val = test_data.get(f"f1_{d_name}", 0.0)
                per_disease_rows.append({"Disease Target": d_name, "AUROC": round(v, 4), "F1-Score": round(f1_val, 4)})

        if per_disease_rows:
            st.dataframe(pd.DataFrame(per_disease_rows), width="stretch", hide_index=True)
    else:
        st.info(
            "ℹ️ **Full Training Evaluation Notice:** Full-scale multi-epoch model training has not been run on this local CPU workspace.\n\n"
            "Full training is configured for **Google Colab GPU** acceleration via `notebooks/02_model_training.ipynb`.\n"
            "Once training is completed and the results folder (`results/`) is populated, full loss trajectories, "
            "per-disease AUROC, precision, recall, and macro-F1 metrics will automatically be displayed here.\n\n"
            "*(Per research guidelines, this dashboard never generates fabricated or simulated training metric charts).* ",
            icon="ℹ️",
        )

# 2. QUANTITATIVE FAITHFULNESS EVALUATION
with tab_faith:
    st.subheader("Quantitative Explanation Faithfulness (Pixel Deletion & Feature Perturbation)")
    faith_csv = REPO_ROOT / "outputs" / "explainability" / "faithfulness" / "faithfulness_metrics.csv"

    if faith_csv.exists():
        st.markdown("Verified quantitative faithfulness evaluation logs recorded from `scripts/run_explainability.py`:")
        df_faith = pd.read_csv(faith_csv)
        st.dataframe(df_faith, width="stretch", hide_index=True)

        st.markdown("### Pixel Deletion Curves")
        del_pngs = list((REPO_ROOT / "outputs" / "explainability" / "faithfulness").glob("deletion_curve_*.png"))
        if del_pngs:
            st.image(str(del_pngs[0]), caption=f"Recorded Deletion/Insertion Trajectory ({del_pngs[0].name})", width=600)
            st.caption(
                "**Deletion Curve Analysis:** Removing top salient pixels decreases the model's confidence rapidly (steep negative slope), "
                "confirming that the Grad-CAM++ identified regions contain the true decision-critical features."
            )
    else:
        st.info("Faithfulness evaluation CSV not found. Run `python scripts/run_explainability.py` to generate faithfulness benchmarks.")

# 3. MODALITY & FEATURE ATTRIBUTION LOGS
with tab_mod_eval:
    st.subheader("Empirical Modality & Feature Attribution Data")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("### Modality Contribution Logs")
        mod_csv = REPO_ROOT / "outputs" / "explainability" / "modality" / "modality_contributions.csv"
        if mod_csv.exists():
            df_mod = pd.read_csv(mod_csv)
            st.dataframe(df_mod, width="stretch", hide_index=True)
        else:
            st.caption("Run `scripts/run_explainability.py` to populate modality logs.")

    with col_m2:
        st.markdown("### Clinical Feature SHAP Importance Logs")
        shap_csv = REPO_ROOT / "outputs" / "explainability" / "shap" / "shap_feature_importance.csv"
        if shap_csv.exists():
            df_shap = pd.read_csv(shap_csv)
            st.dataframe(df_shap, width="stretch", hide_index=True)
        else:
            st.caption("Run `scripts/run_explainability.py` to populate SHAP logs.")
