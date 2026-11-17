import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config.settings import EXPERIMENT_RESULTS_DIR
from src.ui_theme import (
    apply_cyberpunk_theme,
    render_hud_header,
    render_section_header,
    render_metric_card,
)

st.set_page_config(
    page_title="Model Evaluation & Experiments", page_icon="📊", layout="wide"
)

apply_cyberpunk_theme("EVAL_BENCH // 04")

render_hud_header(
    title="📊 Model Evaluation & Experimental Results",
    subtitle="Comprehensive empirical evaluation across 6 experimental configurations (E1 - E6) using writer-disjoint splits and the CEDAR Signature Dataset (55 writers, 2,640 genuine and forged signatures).",
    sys_code="MOD.EVAL // 04",
    status_tag="CEDAR BENCHMARK // 55 WRITERS",
)

json_path = EXPERIMENT_RESULTS_DIR / "experiment_metrics.json"

if not json_path.exists():
    st.info(
        "⏳ Experiments are running or metrics file not found yet. Please refresh shortly."
    )
    st.stop()

with open(json_path, "r") as f:
    metrics = json.load(f)

# Display Key Metric Highlights from Best Model (E5/E6)
best_model_key = "E5" if "E5" in metrics else list(metrics.keys())[0]
best = metrics[best_model_key]

render_section_header(
    "CHAMPION.ARCH // 01", f"🏆 Best Validated Architecture: {best['name']}"
)

m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
with m_col1:
    render_metric_card(
        label="Test Accuracy",
        value=f"{best['test_accuracy']*100:.1f}%",
        subtext="UNSEEN WRITERS",
        accent_hex="#00FF9D",
        code="KPI-01",
    )
with m_col2:
    render_metric_card(
        label="Equal Error Rate (EER)",
        value=f"{best['test_eer']*100:.1f}%",
        subtext="FAR = FRR EQUILIBRIUM",
        accent_hex="#FCEE0A",
        code="KPI-02",
    )
with m_col3:
    render_metric_card(
        label="ROC AUC",
        value=f"{best['test_auc']:.3f}",
        subtext="SEPARABILITY AREA",
        accent_hex="#00F0FF",
        code="KPI-03",
    )
with m_col4:
    render_metric_card(
        label="Top-1 ID Accuracy",
        value=f"{best['top1_id_accuracy']*100:.1f}%",
        subtext="1:N RANK-1 HIT RATE",
        accent_hex="#38BDF8",
        code="KPI-04",
    )
with m_col5:
    render_metric_card(
        label="False Accept Rate (FAR)",
        value=f"{best['test_far']*100:.1f}%",
        subtext="FORGERY SPOOF RATE",
        accent_hex="#FF0055",
        code="KPI-05",
    )
with m_col6:
    render_metric_card(
        label="Mean Latency",
        value=f"{best['inference_time_ms']:.1f} ms",
        subtext="INFERENCE TIME",
        accent_hex="#B829FF",
        code="KPI-06",
    )

st.divider()

# Experiments Comparison Table
render_section_header(
    "MATRIX.E1-E6 // 02", "📋 Comprehensive Experimental Benchmark (E1 - E6)"
)

table_rows = []
for k, v in metrics.items():
    table_rows.append(
        {
            "Exp": k,
            "Configuration": v["name"],
            "Val Acc (%)": f"{v.get('val_accuracy', 0)*100:.1f}%",
            "Test Acc (%)": f"{v['test_accuracy']*100:.1f}%",
            "Precision (%)": f"{v['test_precision']*100:.1f}%",
            "Recall (%)": f"{v['test_recall']*100:.1f}%",
            "F1-Score": f"{v['test_f1']:.3f}",
            "FAR (%)": f"{v['test_far']*100:.1f}%",
            "FRR (%)": f"{v['test_frr']*100:.1f}%",
            "EER (%)": f"{v['test_eer']*100:.1f}%",
            "ROC AUC": f"{v['test_auc']:.3f}",
            "Top-1 ID Acc (%)": f"{v['top1_id_accuracy']*100:.1f}%",
            "Latency (ms)": f"{v['inference_time_ms']:.1f}",
        }
    )

df_table = pd.DataFrame(table_rows)
st.dataframe(df_table, width='stretch', hide_index=True)

# Visual Charts Section
st.divider()
render_section_header("TELEMETRY.PLOTS // 03", "📈 Experimental Visualizations")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    comp_chart_path = EXPERIMENT_RESULTS_DIR / "experiment_comparison_chart.png"
    if comp_chart_path.exists():
        st.image(
            str(comp_chart_path),
            caption="Comparison of Accuracy, Top-1 ID, and EER across E1-E6",
            width='stretch',
        )

    roc_chart_path = EXPERIMENT_RESULTS_DIR / "roc_curves.png"
    if roc_chart_path.exists():
        st.image(
            str(roc_chart_path),
            caption="Receiver Operating Characteristic (ROC) on Unseen Test Writers",
            width='stretch',
        )

with chart_col2:
    cm_path = EXPERIMENT_RESULTS_DIR / "best_model_confusion_matrix.png"
    if cm_path.exists():
        st.image(
            str(cm_path),
            caption="Confusion Matrix: Best Performing Model (Test Set)",
            width='stretch',
        )

    st.markdown(
        """
        <div class="cp-panel">
            <div style="font-family:'Orbitron',sans-serif; font-size:1.05rem; font-weight:700; color:#00F0FF; letter-spacing:0.06em; margin-bottom:10px;">
                🔬 ACADEMIC FINDINGS &amp; DISCUSSION
            </div>
            <ol style="margin-bottom:0; padding-left:1.2rem;">
                <li style="margin-bottom:8px;">
                    <strong style="color:#00FF9D;">Effect of Feature Fusion (HOG vs Hu vs Geometry):</strong>
                    <ul>
                        <li><strong>HOG alone (E1)</strong> provides strong local orientation information (edge gradients and line crossings).</li>
                        <li>Adding <strong>Hu Invariant Moments and Geometric Features (E2, E3)</strong> improves discrimination against skilled forgers who mimic overall letter flow but fail on global moment ratios and stroke density.</li>
                    </ul>
                </li>
                <li style="margin-bottom:8px;">
                    <strong style="color:#00F0FF;">Pairwise SVM vs Cosine Baseline:</strong>
                    <ul>
                        <li>Cosine similarity on normalized descriptors provides a mathematically robust, non-parametric distance metric that generalizes reliably to unseen writers without risk of overfitting.</li>
                        <li>Pairwise SVM learns non-linear combinations of descriptor differences, but requires careful regularization to avoid memorizing training writer quirks.</li>
                    </ul>
                </li>
                <li style="margin-bottom:8px;">
                    <strong style="color:#FCEE0A;">Why One-Shot Registration is Inherently Difficult:</strong>
                    <ul>
                        <li>With only <strong>1 template per user</strong>, the intra-writer variance cannot be modeled from user data.</li>
                        <li>A writer's natural daily handwriting variability (pen pressure, slant, speed) can easily trigger false rejection (FRR), while closely drawn skilled forgeries can trigger false acceptance (FAR).</li>
                    </ul>
                </li>
                <li>
                    <strong style="color:#FF0055;">Uncertain / Unknown Decision Thresholding:</strong>
                    <ul>
                        <li>A hard binary decision forces borderline signatures into false matches.</li>
                        <li>Incorporating an <strong>indecision margin</strong> significantly reduces security vulnerability in practical deployments.</li>
                    </ul>
                </li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
