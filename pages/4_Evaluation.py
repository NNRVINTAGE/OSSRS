import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config.settings import EXPERIMENT_RESULTS_DIR

st.set_page_config(page_title="Model Evaluation & Experiments", page_icon="📊", layout="wide")

st.title("📊 Model Evaluation & Experimental Results")
st.markdown("""
Comprehensive empirical evaluation across **6 experimental configurations (E1 - E6)** using **writer-disjoint** splits
and the **CEDAR Signature Dataset** (55 writers, 2,640 genuine and forged signatures).
""")

json_path = EXPERIMENT_RESULTS_DIR / "experiment_metrics.json"

if not json_path.exists():
    st.info("⏳ Experiments are running or metrics file not found yet. Please refresh shortly.")
    st.stop()

with open(json_path, "r") as f:
    metrics = json.load(f)

# Display Key Metric Highlights from Best Model (E5/E6)
best_model_key = "E5" if "E5" in metrics else list(metrics.keys())[0]
best = metrics[best_model_key]

st.subheader(f"🏆 Best Validated Architecture: {best['name']}")

m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
with m_col1:
    st.metric("Test Accuracy", f"{best['test_accuracy']*100:.1f}%")
with m_col2:
    st.metric("Equal Error Rate (EER)", f"{best['test_eer']*100:.1f}%")
with m_col3:
    st.metric("ROC AUC", f"{best['test_auc']:.3f}")
with m_col4:
    st.metric("Top-1 ID Accuracy", f"{best['top1_id_accuracy']*100:.1f}%")
with m_col5:
    st.metric("False Accept Rate (FAR)", f"{best['test_far']*100:.1f}%")
with m_col6:
    st.metric("Mean Latency", f"{best['inference_time_ms']:.1f} ms")

st.divider()

# Experiments Comparison Table
st.subheader("📋 Comprehensive Experimental Benchmark (E1 - E6)")

table_rows = []
for k, v in metrics.items():
    table_rows.append({
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
        "Latency (ms)": f"{v['inference_time_ms']:.1f}"
    })

df_table = pd.DataFrame(table_rows)
st.dataframe(df_table, use_container_width=True, hide_index=True)

# Visual Charts Section
st.divider()
st.subheader("📈 Experimental Visualizations")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    comp_chart_path = EXPERIMENT_RESULTS_DIR / "experiment_comparison_chart.png"
    if comp_chart_path.exists():
        st.image(str(comp_chart_path), caption="Comparison of Accuracy, Top-1 ID, and EER across E1-E6", use_container_width=True)
        
    roc_chart_path = EXPERIMENT_RESULTS_DIR / "roc_curves.png"
    if roc_chart_path.exists():
        st.image(str(roc_chart_path), caption="Receiver Operating Characteristic (ROC) on Unseen Test Writers", use_container_width=True)

with chart_col2:
    cm_path = EXPERIMENT_RESULTS_DIR / "best_model_confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix: Best Performing Model (Test Set)", use_container_width=True)
        
    st.markdown("""
    ### 🔬 Academic Findings & Discussion
    
    1. **Effect of Feature Fusion (HOG vs Hu vs Geometry)**:
       - **HOG alone (E1)** provides strong local orientation information (edge gradients and line crossings).
       - Adding **Hu Invariant Moments and Geometric Features (E2, E3)** improves discrimination against skilled forgers who mimic overall letter flow but fail on global moment ratios and stroke density.
       
    2. **Pairwise SVM vs Cosine Baseline**:
       - Cosine similarity on normalized descriptors provides a mathematically robust, non-parametric distance metric that generalizes reliably to unseen writers without risk of overfitting.
       - Pairwise SVM learns non-linear combinations of descriptor differences, but requires careful regularization to avoid memorizing training writer quirks.
       
    3. **Why One-Shot Registration is Inherently Difficult**:
       - With only **1 template per user**, the intra-writer variance cannot be modeled from user data.
       - A writer's natural daily handwriting variability (pen pressure, slant, speed) can easily trigger false rejection (FRR), while closely drawn skilled forgeries can trigger false acceptance (FAR).
       
    4. **Uncertain / Unknown Decision Thresholding**:
       - A hard binary decision forces borderline signatures into false matches.
       - Incorporating an **indecision margin** significantly reduces security vulnerability in practical deployments.
    """)
