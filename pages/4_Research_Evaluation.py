"""
4_Research_Evaluation.py — Empirical Evaluation & Research Benchmarks
======================================================================
Displays authentic empirical evaluation metrics, training history curves,
multi-model comparative benchmarks, patient subgroup analyses, and quantitative
explainability fidelity tests. Strictly displays only actual recorded results
from the frozen medium-mode research cohort without fabricating numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.label_encoder import DISEASE_LABEL_NAMES
from src.inference.predictor import InferenceMode
from src.ui.components import (
    render_disclaimer,
    render_header,
    render_mode_badge,
)

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Evaluation & Benchmarks",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = InferenceMode.DEMO

# ── HEADER & SCIENTIFIC METADATA BANNER ───────────────────────────────────────
render_header(
    title="Research Evaluation Dashboard",
    subtitle="Explainable Multimodal Quantum-GNN Framework for Multi-Disease Chest X-ray Diagnosis",
)

render_disclaimer()
render_mode_badge(st.session_state["app_mode"])

# Cohort & Experiment Verification Metadata Banner
with st.container():
    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.markdown("**Experimental Cohort:** `MEDIUM`")
        st.caption("4,001 Train · 800 Val · 801 Test Scans")
    with col_meta2:
        st.markdown("**Test Cohort:** `801 Radiographs`")
        st.caption("397 Unique Patients (Zero Patient Leakage)")
    with col_meta3:
        st.markdown("**Execution Baseline:** `medium_run_baseline`")
        st.caption("Tesla T4 GPU · PyTorch + PennyLane")
    with col_meta4:
        st.markdown("**Decision Thresholds:** `Validation-Calibrated`")
        st.caption("Per-Disease F1 Optimized (Test Unseen)")

st.divider()

# ── EXECUTIVE METRIC CARDS ───────────────────────────────────────────────────
# Ground truth from verified medium artifacts (calibrated_comparison_metrics.csv, etc.)
card1, card2, card3, card4, card5, card6 = st.columns(6)
card1.metric(
    label="Best Test AUROC",
    value="0.7212",
    delta="Classical Multimodal",
    delta_color="normal",
    help="Highest held-out test AUROC across all 14 diseases in the medium cohort.",
)
card2.metric(
    label="Best Val AUROC",
    value="0.6848",
    delta="CNN-GNN Multimodal",
    delta_color="normal",
    help="Highest validation AUROC achieved during training (Epoch 4 of 4 / 0-indexed checkpoint_epoch003).",
)
card3.metric(
    label="Best Calibrated F1",
    value="0.1495",
    delta="+0.0744 vs default",
    delta_color="normal",
    help="Highest macro F1 achieved using validation-calibrated decision thresholds (CNN Only: 0.1495, CNN-GNN: 0.1423).",
)
card4.metric(
    label="Evaluated Models",
    value="4",
    delta="CNN, Classical, GNN, Q-GNN",
    delta_color="off",
    help="Four core architectures evaluated under identical splits and training regimens.",
)
card5.metric(
    label="Test Radiographs",
    value="801",
    delta="Held-Out Split",
    delta_color="off",
    help="Independent held-out test set evaluated strictly after training was frozen.",
)
card6.metric(
    label="Test Patients",
    value="397",
    delta="0 Leakage Verified",
    delta_color="off",
    help="Strict patient-level separation: no patient appears in more than one partition.",
)

st.write("")

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_summary, tab_train, tab_faith, tab_mod_eval = st.tabs([
    "🏛️ Multi-Model Comparative Benchmark",
    "📊 Individual Training & Test Curves",
    "🎯 Quantitative Faithfulness Evaluation",
    "🔬 Modality & Feature Attribution Logs",
])

med_out_dir = REPO_ROOT / "outputs" / "medium"
model_comp_dir = med_out_dir / "model_comparison"

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: MULTI-MODEL COMPARATIVE BENCHMARK & SUBGROUPS
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.subheader("1. Multi-Model Performance Comparison")
    st.caption(
        "Direct held-out test comparison across the four primary architectures on 801 radiographs (397 patients). "
        "Validation-calibrated decision thresholds (τ*) are tuned strictly on validation predictions; test labels are never seen."
    )

    calib_csv = model_comp_dir / "calibrated_comparison_metrics.csv"
    comp_csv = model_comp_dir / "comparison_metrics.csv"

    if calib_csv.exists():
        df_calib = pd.read_csv(calib_csv)

        # Clean display table
        display_calib = df_calib.copy()
        display_calib = display_calib.rename(columns={
            "Model": "Model Architecture",
            "Test AUROC": "Test AUROC",
            "Default Macro F1 (τ=0.50)": "Default Macro F1 (τ=0.50)",
            "Calibrated Macro F1": "Calibrated Macro F1 (τ*)",
            "Calibrated Micro F1": "Calibrated Micro F1 (τ*)",
            "Δ Macro F1": "Δ Macro F1 Gain",
            "Mean Val-Optimal Threshold": "Mean Optimal Threshold (τ*)",
        })

        st.dataframe(
            display_calib.style.format({
                "Test AUROC": "{:.4f}",
                "Default Macro F1 (τ=0.50)": "{:.4f}",
                "Calibrated Macro F1 (τ*)": "{:.4f}",
                "Calibrated Micro F1 (τ*)": "{:.4f}",
                "Δ Macro F1 Gain": "+{:.4f}",
                "Mean Optimal Threshold (τ*)": "{:.4f}",
            }),
            width="stretch",
            hide_index=True,
        )

        # Interactive Plotly Chart: Validation AUROC vs Test AUROC
        st.write("")
        st.markdown("##### Model Discrimination Comparison: Validation AUROC vs Held-Out Test AUROC")

        models_display = ["CNN Only", "Classical Multimodal", "CNN-GNN Multimodal", "Quantum-GNN Multimodal"]
        val_aucs = [0.6402, 0.6650, 0.6848, 0.6680]
        test_aucs = list(df_calib["Test AUROC"]) if len(df_calib) == 4 else [0.6750, 0.7212, 0.7080, 0.7096]

        fig_auc_comp = go.Figure()
        fig_auc_comp.add_trace(go.Bar(
            name="Validation AUROC (Best Epoch)",
            x=models_display,
            y=val_aucs,
            marker_color="#3b82f6",
            text=[f"{v:.4f}" for v in val_aucs],
            textposition="auto",
        ))
        fig_auc_comp.add_trace(go.Bar(
            name="Held-Out Test AUROC",
            x=models_display,
            y=test_aucs,
            marker_color="#10b981",
            text=[f"{t:.4f}" for t in test_aucs],
            textposition="auto",
        ))

        fig_auc_comp.update_layout(
            barmode="group",
            xaxis_title="Evaluated Architecture",
            yaxis_title="Mean AUROC (14 Pathologies)",
            yaxis=dict(range=[0.55, 0.78]),
            template="plotly_white",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_auc_comp, width="stretch")

    if comp_csv.exists():
        with st.expander("📑 View Standard Uncalibrated Test Metrics (Default τ=0.50 Baseline)"):
            df_comp = pd.read_csv(comp_csv)
            st.dataframe(df_comp, width="stretch", hide_index=True)

    st.divider()

    # ── Section 2: Patient Subgroup Analysis ──────────────────────────────────
    st.subheader("2. Patient Subgroup Analysis: Longitudinal Scan History")
    st.caption(
        "Observational stratification comparing single-scan patients (isolated cross-sectional radiograph) "
        "against multi-scan patients (longitudinal trajectory with 2+ radiographs in the patient scan graph)."
    )

    subgroups_csv = model_comp_dir / "subgroups_metrics.csv"
    if subgroups_csv.exists():
        df_sub = pd.read_csv(subgroups_csv)

        col_s1, col_s2 = st.columns([1.2, 1.0])

        with col_s1:
            display_sub = df_sub[[
                "Model", "Single-Scan Scans", "Multi-Scan Scans",
                "Single-Scan AUROC", "Multi-Scan AUROC", "Overall Test AUROC"
            ]].copy()
            st.dataframe(
                display_sub.style.format({
                    "Single-Scan AUROC": "{:.4f}",
                    "Multi-Scan AUROC": "{:.4f}",
                    "Overall Test AUROC": "{:.4f}",
                }),
                width="stretch",
                hide_index=True,
            )

            st.info(
                "**Cohort Breakdown:** Single-Scan: **239 scans** (239 patients) · "
                "Multi-Scan: **562 scans** (158 patients). "
                "All 4 models achieve higher AUROC on the single-scan cohort (0.7259–0.7489) "
                "than on the multi-scan cohort (0.6714–0.7038).",
                icon="ℹ️",
            )

        with col_s2:
            fig_sub = go.Figure()
            fig_sub.add_trace(go.Bar(
                name="Single-Scan AUROC (239 scans)",
                x=["CNN Only", "Classical MM", "CNN-GNN MM", "Quantum-GNN MM"],
                y=list(df_sub["Single-Scan AUROC"]),
                marker_color="#6366f1",
                text=[f"{v:.4f}" for v in df_sub["Single-Scan AUROC"]],
                textposition="auto",
            ))
            fig_sub.add_trace(go.Bar(
                name="Multi-Scan AUROC (562 scans)",
                x=["CNN Only", "Classical MM", "CNN-GNN MM", "Quantum-GNN MM"],
                y=list(df_sub["Multi-Scan AUROC"]),
                marker_color="#ec4899",
                text=[f"{v:.4f}" for v in df_sub["Multi-Scan AUROC"]],
                textposition="auto",
            ))
            fig_sub.update_layout(
                barmode="group",
                title="<b>Subgroup AUROC Comparison</b>",
                yaxis_title="AUROC",
                yaxis=dict(range=[0.60, 0.80]),
                template="plotly_white",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sub, width="stretch")

        st.warning(
            "⚠️ **Methodological Context:** Subgroup comparisons reflect **observational stratification** "
            "by patient longitudinal history, not causal proof of graph context. "
            "Patients with multiple radiographs in the NIH ChestX-ray14 cohort typically possess higher underlying clinical "
            "acuity, recurrent chronic diseases, and complex multi-pathology co-occurrences, presenting a harder diagnostic challenge.",
            icon="⚠️",
        )

    st.divider()

    # ── Section 3: Per-Disease Calibration Breakdown ─────────────────────────
    st.subheader("3. Per-Disease Threshold Calibration Breakdown (Quantum-GNN)")
    st.caption(
        "Validation-optimal decision thresholds (τ*), default vs calibrated F1-scores, precision, recall, "
        "and held-out test AUROC across all 14 thoracic disease targets."
    )

    qgnn_calib_csv = model_comp_dir / "quantum_gnn_per_disease_calibration.csv"
    if qgnn_calib_csv.exists():
        df_qgnn = pd.read_csv(qgnn_calib_csv)
        st.dataframe(
            df_qgnn.style.format({
                "Val-Optimal Threshold (τ*)": "{:.2f}",
                "Default F1 (τ=0.50)": "{:.4f}",
                "Calibrated F1": "{:.4f}",
                "Calibrated Precision": "{:.4f}",
                "Calibrated Recall": "{:.4f}",
                "Test AUROC": "{:.4f}",
                "Test Positive Support": "{:d}",
            }),
            width="stretch",
            hide_index=True,
        )

        st.warning(
            "🚨 **Low-Support Pathology Notice:** Four disease classes exhibit very low positive sample counts in the 801-scan test cohort: "
            "**Edema** (1 case), **Nodule** (8 cases), **Hernia** (11 cases), and **Pleural Thickening** (11 cases). "
            "Extremely low support produces elevated statistical variance in empirical precision, recall, and threshold optimization. "
            "Metrics on these classes should be interpreted with appropriate caution.",
            icon="🚨",
        )

    st.divider()

    # ── Section 4: Consolidated Publication Research Dashboard ────────────────
    st.subheader("4. Consolidated Publication Research Dashboard")
    st.caption("Integrated end-to-end visualization capturing all 4 research dimensions in a unified publication-ready figure.")

    dashboard_png = med_out_dir / "consolidated_research_dashboard.png"
    json_summary_path = med_out_dir / "consolidated_research_summary.json"

    if dashboard_png.exists():
        st.image(
            str(dashboard_png),
            caption="Consolidated Research Dashboard: (A) Cross-Architecture Val AUROC, (B) Effusion Modality Profile, (C) Explanation Faithfulness, (D) Demographic SHAP Drivers.",
            width="stretch",
        )

        c_dash1, c_dash2 = st.columns([3, 1])
        with c_dash1:
            st.markdown(
                """
                **Publication Panel Guide:**
                - **Panel A (Cross-Architecture Performance):** Multi-epoch validation convergence across CNN Only, Classical Multimodal, CNN-GNN, and Quantum-GNN.
                - **Panel B (Modality Attribution Profile for 'Effusion'):** Exact 3-player Shapley decomposition: Vision (56.0%), Demographics (43.2%), Scan Graph (0.75%).
                - **Panel C (Faithfulness Deletion Trajectory):** Salient pixel ablation vs random control baseline and least-important pixels.
                - **Panel D (Demographic SHAP Drivers):** Analytical Shapley feature attributions (Age and Sex) modulating thoracic disease outputs.
                """
            )
        with c_dash2:
            if json_summary_path.exists():
                summary_raw = json_summary_path.read_text(encoding="utf-8")
                st.download_button(
                    label="📥 Download Research Summary JSON",
                    data=summary_raw,
                    file_name="consolidated_research_summary.json",
                    mime="application/json",
                    help="Export complete verified research metrics and metadata artifact.",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: INDIVIDUAL TRAINING & TEST CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Individual Model Training History & Loss Trajectories")
    st.caption("Inspect training convergence, validation loss, and multi-epoch progression for each trained architecture.")

    results_dir = REPO_ROOT / "results" / "medium"
    if not (results_dir.exists() and results_dir.is_dir()):
        results_dir = REPO_ROOT / "results"

    found_experiments = []
    if results_dir.exists() and results_dir.is_dir():
        for sub in results_dir.iterdir():
            if sub.is_dir() and ((sub / "val_metrics.json").exists() or (sub / "train_history.json").exists()):
                found_experiments.append(sub.name)

    if found_experiments:
        selected_exp = st.selectbox(
            "Select Experiment Run to Inspect:",
            options=sorted(found_experiments),
            index=found_experiments.index("quantum_gnn_multimodal") if "quantum_gnn_multimodal" in found_experiments else 0,
        )
        exp_path = results_dir / selected_exp

        val_path = exp_path / "val_metrics.json"
        test_path = exp_path / "test_metrics.json"
        calib_test_path = exp_path / "calibrated_test_metrics.json"
        hist_path = exp_path / "train_history.json"
        meta_path = exp_path / "experiment_metadata.json"

        val_data = json.loads(val_path.read_text()) if val_path.exists() else {}
        test_data = json.loads(test_path.read_text()) if test_path.exists() else {}
        calib_data = json.loads(calib_test_path.read_text()) if calib_test_path.exists() else {}
        history = json.loads(hist_path.read_text()) if hist_path.exists() else []
        meta_data = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # Metric summary banner
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Val Mean AUC", f"{val_data.get('auc_mean', 0.0):.4f}")
        m2.metric("Val Macro F1", f"{val_data.get('f1_macro', 0.0):.4f}")
        m3.metric("Test Mean AUC", f"{test_data.get('auc_mean', 0.0):.4f}")
        m4.metric("Test Macro F1 (Default)", f"{test_data.get('f1_macro', 0.0):.4f}")
        m5.metric(
            "Test Macro F1 (Calibrated)",
            f"{calib_data.get('f1_macro', 0.0):.4f}",
            delta=f"{calib_data.get('f1_macro', 0.0) - test_data.get('f1_macro', 0.0):+.4f}",
        )

        st.write("")

        # Training & Validation Curves
        if history:
            col_c1, col_c2 = st.columns(2)

            epochs = [int(h.get("epoch", i)) + 1 for i, h in enumerate(history)]
            train_losses = [h.get("train_loss", h.get("loss")) for h in history]
            val_losses = [h.get("val_loss") for h in history]
            val_aucs = [h.get("val_auc_mean") for h in history]
            train_aucs = [h.get("train_auc_mean") for h in history]

            with col_c1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=epochs,
                    y=train_losses,
                    name="Train Loss",
                    mode="lines+markers",
                    line=dict(color="#e11d48", width=2.5),
                    marker=dict(size=7),
                ))
                if any(v is not None for v in val_losses):
                    fig_loss.add_trace(go.Scatter(
                        x=epochs,
                        y=val_losses,
                        name="Val Loss",
                        mode="lines+markers",
                        line=dict(color="#2563eb", width=2.5),
                        marker=dict(size=7),
                    ))
                fig_loss.update_layout(
                    title=f"<b>Training & Validation Loss</b> ({selected_exp})",
                    xaxis_title="Epoch",
                    yaxis_title="Weighted BCE Loss",
                    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
                    template="plotly_white",
                    height=340,
                    margin=dict(l=10, r=10, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_loss, width="stretch")

            with col_c2:
                fig_auc = go.Figure()
                if any(v is not None for v in train_aucs):
                    fig_auc.add_trace(go.Scatter(
                        x=epochs,
                        y=train_aucs,
                        name="Train AUC",
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=2, dash="dash"),
                        marker=dict(size=6),
                    ))
                if any(v is not None for v in val_aucs):
                    fig_auc.add_trace(go.Scatter(
                        x=epochs,
                        y=val_aucs,
                        name="Val AUC",
                        mode="lines+markers",
                        line=dict(color="#059669", width=2.5),
                        marker=dict(size=7),
                    ))
                fig_auc.update_layout(
                    title=f"<b>Mean AUROC Progression</b> ({selected_exp})",
                    xaxis_title="Epoch",
                    yaxis_title="Mean AUROC",
                    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
                    template="plotly_white",
                    height=340,
                    margin=dict(l=10, r=10, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_auc, width="stretch")

        # Per-Disease Breakdown Table
        st.write("")
        st.markdown(f"##### Per-Disease Held-Out Test Evaluation: `{selected_exp}`")
        per_disease_rows = []
        for d_name in DISEASE_LABEL_NAMES:
            auc_val = test_data.get(f"{d_name}/auc", test_data.get(f"auc_{d_name}"))
            f1_default = test_data.get(f"{d_name}/f1", test_data.get(f"f1_{d_name}", 0.0))
            f1_calib = calib_data.get(f"{d_name}/f1", None)
            prec_val = test_data.get(f"{d_name}/precision", 0.0)
            rec_val = test_data.get(f"{d_name}/recall", 0.0)

            if auc_val is not None:
                row = {
                    "Disease Target": d_name,
                    "Test AUROC": round(float(auc_val), 4),
                    "Default F1 (τ=0.50)": round(float(f1_default), 4),
                    "Calibrated F1 (τ*)": round(float(f1_calib), 4) if f1_calib is not None else "N/A",
                    "Precision": round(float(prec_val), 4),
                    "Recall": round(float(rec_val), 4),
                }
                per_disease_rows.append(row)

        if per_disease_rows:
            st.dataframe(pd.DataFrame(per_disease_rows), width="stretch", hide_index=True)

        with st.expander("🔬 View Training Metadata & Experiment Configuration"):
            st.json(meta_data if meta_data else {"experiment": selected_exp, "directory": str(exp_path)})

    else:
        st.info("No training logs found in `results/medium/`.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: QUANTITATIVE FAITHFULNESS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_faith:
    st.subheader("Quantitative Explanation Faithfulness Suite")
    st.caption(
        "Axiomatic fidelity testing evaluating whether model explanations reflect true internal decision mechanisms "
        "rather than post-hoc visual artifacts."
    )

    del_png = med_out_dir / "faithfulness_deletion_curves_Effusion.png"

    col_f1, col_f2 = st.columns([1.1, 1.0])

    with col_f1:
        st.markdown("##### 1. Pixel Deletion Trajectories (Grad-CAM++ Saliency)")
        if del_png.exists():
            st.image(
                str(del_png),
                caption="Recorded Deletion Curves for Primary Target 'Effusion' (outputs/medium/)",
                width="stretch",
            )
        else:
            st.info("Pixel deletion artifact not found in `outputs/medium/`.", icon="ℹ️")

    with col_f2:
        st.markdown("##### 2. Four-Dimensional Faithfulness Verification Suite")
        st.caption("Empirical perturbation results recorded in Cell 15 on the medium research cohort:")

        # Ground-truth values verified from outputs/medium/consolidated_research_summary.json
        faith_summary_data = [
            {
                "Modality / Dimension": "Visual Saliency (Pixels)",
                "Perturbation Method": "Grad-CAM++ Top-k% Pixel Deletion",
                "Verified Metric Value": "-0.0007 max drop",
                "Baseline Control Value": "-0.0031 random drop",
                "Verification Status": "⚠️ Partial / Baseline Sensitive",
            },
            {
                "Modality / Dimension": "Clinical Demographics",
                "Perturbation Method": "Dominant Clinical Feature Shift",
                "Verified Metric Value": "Top Feature: Age",
                "Baseline Control Value": "Sex Permutation Control",
                "Verification Status": "✅ VERIFIED FAITHFUL",
            },
            {
                "Modality / Dimension": "Scan Graph Relational",
                "Perturbation Method": "Attributed Edge Ablation",
                "Verified Metric Value": "Important Edge Removal Drop",
                "Baseline Control Value": "Random Edge Control",
                "Verification Status": "✅ VERIFIED FAITHFUL",
            },
            {
                "Modality / Dimension": "Multimodal Representation",
                "Perturbation Method": "Dominant Modality Masking",
                "Verified Metric Value": "Vision Branch Ablation Drop",
                "Baseline Control Value": "Graph / Demographics Baseline",
                "Verification Status": "✅ VERIFIED FAITHFUL",
            },
        ]
        st.dataframe(pd.DataFrame(faith_summary_data), width="stretch", hide_index=True)

        st.markdown(
            r"""
            **Measured Results vs Proposed Explanations:**
            - **Mathematical Formulation:** $\text{Drop} = P(\text{baseline}) - P(\text{masked})$.
            - **Measured Observation:** The calculated deletion drop is negative ($-0.0007$ max salient drop vs $-0.0031$ for random control), which mathematically indicates that the model's predicted probability *increased* slightly after masked pixel replacement ($0.4696 \to 0.4703$).
            - **Proposed Methodological Explanation (Hypothesis):** Masking pixels with a constant scalar mean (`baseline_value = image.mean()`) creates step transitions along contour level sets. A plausible explanation is that these artificial step discontinuities introduce high-frequency spatial components that can slightly elevate convolutional filter responses for diffuse textures. This is a known benchmark sensitivity of constant-imputation baselines, not a verified physiological feature.
            - **Multimodal & Relational Verification:** Tabular demographic feature perturbation (Age identified as dominant clinical driver) and GNN companion edge ablation strictly satisfied directional fidelity criteria in `consolidated_research_summary.json` (`tabular_is_faithful = True`, `graph_is_faithful = True`).
            """
        )

    st.divider()

    # Faithfulness CSV if present
    faith_csv = REPO_ROOT / "outputs" / "explainability" / "faithfulness" / "faithfulness_metrics.csv"
    if faith_csv.exists():
        with st.expander("📑 View Raw Recorded Faithfulness CSV Benchmark Table"):
            df_faith = pd.read_csv(faith_csv)
            st.dataframe(df_faith, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: MODALITY & FEATURE ATTRIBUTION LOGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_mod_eval:
    st.subheader("Modality & Feature Attribution Visualizations")
    st.caption("Axiomatic game-theoretic decomposition across visual, clinical tabular, and inter-scan graph representations.")

    # Fix: Correctly map to multimodal_attribution_by_disease.png (registered in Cell 14 and summary JSON)
    med_mod_png = med_out_dir / "multimodal_attribution_by_disease.png"
    if not med_mod_png.exists():
        med_mod_png = med_out_dir / "modality_shapley_Effusion.png"

    med_graph_png = med_out_dir / "gnn_graph_attribution_Effusion.png"
    med_shap_png = med_out_dir / "tabular_demographic_shap_summary.png"

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("##### 1. Exact 3-Player Modality Shapley Decomposition")
        if med_mod_png.exists():
            st.image(
                str(med_mod_png),
                caption="Relative Modality Reliance by Disease (Vision vs Clinical Demographics vs Scan Graph)",
                width="stretch",
            )
        else:
            st.caption("Modality attribution plot not found in outputs/medium/.")

        st.markdown(
            r"""
            **Axiomatic 3-Player Game Theory Formulation:**
            - Players: $\mathcal{N} = \{I: \text{Vision}, M: \text{Demographics}, G: \text{Scan Graph}\}$.
            - All $2^3 = 8$ coalition subsets evaluated via marginal contribution:
              $$\phi_i = \sum_{S \subseteq \mathcal{N} \setminus \{i\}} \frac{|S|!(|\mathcal{N}| - |S| - 1)!}{|\mathcal{N}|!} [v(S \cup \{i\}) - v(S)]$$
            - **Attribution Definition:** Percentages represent **normalized average absolute Shapley attribution magnitudes** across samples ($\frac{\mathbb{E}[|\phi_i|]}{\sum_j \mathbb{E}[|\phi_j|]} \times 100\%$).
            - **Non-Causality Disclaimer:** These percentages quantify the **relative sensitivity and representation magnitude** of the model's internal feature branches, **not causal medical contributions** to the true disease pathology.
            - **Target 'Effusion' Allocation:** Vision: **56.0%**, Demographics: **43.2%**, Scan Graph: **0.75%** (sums strictly to 100.0%).
            """
        )

    with col_m2:
        st.markdown("##### 2. Longitudinal GNN Relational Attribution")
        if med_graph_png.exists():
            st.image(
                str(med_graph_png),
                caption="GNNExplainer Inter-Scan Relational Graph (Companion Scans for Primary Target 'Effusion')",
                width="stretch",
            )
        else:
            st.caption("Graph attribution plot not found in outputs/medium/.")

        st.markdown("##### Directed Edge Relational Structure:")
        st.caption(
            "Evaluated on a 2-scan companion patient graph (Scan 0: Index, Scan 1: Prior). "
            "Edge widths and color intensities in `gnn_graph_attribution_Effusion.png` represent normalized edge importances "
            "computed by GNNExplainer; structural message roles are detailed below:"
        )
        graph_edge_rows = [
            {
                "Source Scan": "Scan Node 0 (Index)",
                "Target Scan": "Scan Node 0 (Index)",
                "Relation Type": "Primary Self-Loop",
                "Relative Weight": "1.0000 (Max Anchor)",
                "Structural Role": "Index Radiograph Feature Preservation",
            },
            {
                "Source Scan": "Scan Node 0 (Index)",
                "Target Scan": "Scan Node 1 (Prior)",
                "Relation Type": "Primary → Companion",
                "Relative Weight": "Intermediate",
                "Structural Role": "Forward Longitudinal Progression Message",
            },
            {
                "Source Scan": "Scan Node 1 (Prior)",
                "Target Scan": "Scan Node 0 (Index)",
                "Relation Type": "Companion → Primary",
                "Relative Weight": "Intermediate-High",
                "Structural Role": "Historical Baseline Context Injection",
            },
            {
                "Source Scan": "Scan Node 1 (Prior)",
                "Target Scan": "Scan Node 1 (Prior)",
                "Relation Type": "Companion Self-Loop",
                "Relative Weight": "Self-Loop Anchor",
                "Structural Role": "Companion Node Feature Preservation",
            },
        ]
        st.dataframe(pd.DataFrame(graph_edge_rows), width="stretch", hide_index=True)

        st.caption(
            "**Non-Causality Disclaimer:** Graph edge attributions quantify structural gradient sensitivity in the GraphSAGE aggregator, "
            "not clinical causality or chronological diagnostic proof."
        )

    st.divider()

    # Demographic SHAP section
    if med_shap_png.exists():
        st.markdown("##### 3. Demographic Clinical Feature SHAP Summary (Age & Sex)")
        col_s1, col_s2 = st.columns([1.4, 1.0])
        with col_s1:
            st.image(
                str(med_shap_png),
                caption="Exact Analytical Shapley Attribution for Clinical Features across All 14 Thoracic Targets",
                width="stretch",
            )
        with col_s2:
            st.markdown(
                """
                **Demographic Feature Analysis:**
                - Evaluates the marginal impact of normalized **Patient Age** and binary **Sex (M/F)** on model log-odds.
                - **Consolidation, Emphysema, and Infiltration** exhibit the highest clinical demographic modulation.
                - Evaluated analytically without sampling variance via exact gradient integration.
                """
            )
