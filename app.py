import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import PROJECT_ROOT, EXPERIMENT_RESULTS_DIR
from src.database import init_db, get_dashboard_stats

# Streamlit Page Configuration
st.set_page_config(
    page_title="OSSRS",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Custom CSS for polished academic appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .notice-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 16px 0;
        font-size: 0.9rem;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<div class='main-header'>✍️ One-Shot Offline Signature Recognition System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Course Project: Digital Image Processing (Pengolahan Citra Digital) | Classical Machine Learning & Computer Vision</div>", unsafe_allow_html=True)

# Fetch stats from database
stats = get_dashboard_stats()

# Summary Metric Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val'>{stats['total_registered']}</div>
        <div class='metric-label'>Registered Users</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val'>{stats['total_attempts']}</div>
        <div class='metric-label'>Total Attempts</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val' style='color: #16A34A;'>{stats['matched_count']}</div>
        <div class='metric-label'>Matches</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val' style='color: #DC2626;'>{stats['rejected_count']}</div>
        <div class='metric-label'>Rejections / Unknown</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val' style='color: #D97706;'>{stats['uncertain_count']}</div>
        <div class='metric-label'>Uncertain</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class='notice-box'>
    <strong>Academic Disclaimer:</strong> This system uses classical computer vision (HOG, Hu Moments, Geometric Descriptors)
    and pairwise classification for educational purposes. Visual similarity between offline stroke patterns indicates visual resemblance,
    not certified legal or forensic authenticity.
</div>
""", unsafe_allow_html=True)

# Main Two Column Layout
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📋 System Architecture & Workflow")
    st.markdown("""
    This application implements an offline **One-Shot Signature Recognition and Verification** system:
    
    1. **One-Shot Constraint**: Each registered user provides **exactly one** signature template image during enrollment.
    2. **Multi-Stage Preprocessing**: Grayscale conversion, Gaussian denoising, Otsu binarization, bounding box cropping, minor skew correction, and aspect-ratio preserving padding to **128 × 128**.
    3. **Hybrid Feature Vector (8,119 dimensions)**:
       - **HOG (8,100 dims)**: Captures local stroke direction, gradient orientation, and edge junctions.
       - **Hu Invariant Moments (7 dims)**: Translation-, scale-, and rotation-invariant global shape descriptors with logarithmic scaling.
       - **Geometric Features (12 dims)**: Aspect ratio, ink occupancy, contour area ratio, stroke density, centroid, and spatial stroke balance.
    4. **Two Recognition Modes**:
       - **1:N Identification**: Compares an unknown probe against all enrolled templates; applies dynamic confidence margin filtering to report the owner or `Unknown`.
       - **1:1 Verification**: Compares a probe against a claimed user's single template; reports `Match`, `Not Match`, or `Uncertain`.
    """)
    
    st.subheader("🔍 Quick Navigation")
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
    with c_btn1:
        if st.button("➕ Register User", use_container_width=True):
            st.switch_page("pages/1_Register.py")
    with c_btn2:
        if st.button("🔎 1:N Identify", use_container_width=True):
            st.switch_page("pages/2_Identify.py")
    with c_btn3:
        if st.button("🛡️ 1:1 Verify", use_container_width=True):
            st.switch_page("pages/3_Verify.py")
    with c_btn4:
        if st.button("📊 Evaluation", use_container_width=True):
            st.switch_page("pages/4_Evaluation.py")

with right_col:
    st.subheader("🕒 Recent Verification Activity")
    recent_logs = stats["recent_logs"]
    if recent_logs:
        log_df = pd.DataFrame(recent_logs)
        display_df = pd.DataFrame({
            "Time": log_df["created_at"],
            "Op": log_df["operation_type"],
            "Claimed": log_df["claimed_name"].fillna("-"),
            "Predicted": log_df["predicted_name"].fillna("-"),
            "Score": log_df["similarity_score"].apply(lambda s: f"{s:.3f}"),
            "Decision": log_df["decision"]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No verification attempts logged yet. Try verifying or identifying a signature in the pages menu!")

    # Evaluation Summary Preview
    summary_chart = EXPERIMENT_RESULTS_DIR / "experiment_comparison_chart.png"
    if summary_chart.exists():
        st.subheader("📈 Benchmark Preview (E1 - E6)")
        st.image(str(summary_chart), caption="One-Shot Performance Across Experiments", use_container_width=True)
