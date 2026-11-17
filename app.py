import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import PROJECT_ROOT, EXPERIMENT_RESULTS_DIR
from src.database import init_db, get_dashboard_stats
from src.ui_theme import (
    apply_cyberpunk_theme,
    render_hud_header,
    render_section_header,
    render_metric_card,
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="OSSRS",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database
init_db()

# Apply Cyberpunk UI Theme
apply_cyberpunk_theme("CORE_DASHBOARD // 00")

# App Header
render_hud_header(
    title="✍️ One-Shot Offline Signature Recognition System",
    subtitle="Course Project: Digital Image Processing (Pengolahan Citra Digital) | Classical Machine Learning & Computer Vision",
    sys_code="SYS.CORE // OSSRS-HUD",
    status_tag="ONE-SHOT BIOMETRIC PROTOCOL",
)

# Fetch stats from database
stats = get_dashboard_stats()

# Summary Metric Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric_card(
        label="Registered Users",
        value=str(stats["total_registered"]),
        subtext="ENROLLED 1-SHOT TEMPLATES",
        accent_hex="#00F0FF",
        code="USR-01",
    )

with col2:
    render_metric_card(
        label="Total Attempts",
        value=str(stats["total_attempts"]),
        subtext="LOGGED BIOMETRIC SCANS",
        accent_hex="#38BDF8",
        code="SCN-02",
    )

with col3:
    render_metric_card(
        label="Matches",
        value=str(stats["matched_count"]),
        subtext="VERIFIED IDENTITIES",
        accent_hex="#00FF9D",
        code="MCH-03",
    )

with col4:
    render_metric_card(
        label="Rejections / Unknown",
        value=str(stats["rejected_count"]),
        subtext="BLOCKED / UNRECOGNIZED",
        accent_hex="#FF0055",
        code="REJ-04",
    )

with col5:
    render_metric_card(
        label="Uncertain",
        value=str(stats["uncertain_count"]),
        subtext="BORDERLINE AMBIGUOUS",
        accent_hex="#FCEE0A",
        code="UNC-05",
    )

st.markdown(
    """
<div class="cp-notice">
    <strong style="color:#FCEE0A; font-family:'Orbitron',sans-serif; letter-spacing:0.06em;">⚡ ACADEMIC DISCLAIMER // FORENSIC NOTICE:</strong>
    This system uses classical computer vision (HOG, Hu Moments, Geometric Descriptors)
    and pairwise classification for educational purposes. Visual similarity between offline stroke patterns indicates visual resemblance,
    not certified legal or forensic authenticity.
</div>
""",
    unsafe_allow_html=True,
)

# Main Two Column Layout
left_col, right_col = st.columns([3, 2])

with left_col:
    render_section_header("SYS.ARCH // 01", "System Architecture & Workflow")
    st.markdown(
        """
        <div class="cp-panel">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.76rem; color:#00F0FF; letter-spacing:0.1em; margin-bottom:10px;">
                PIPELINE SPECIFICATION // 9-STAGE NORMALIZATION + 8,119-D FEATURE FUSION
            </div>
            <p style="margin-top:0;">
                This application implements an offline <strong>One-Shot Signature Recognition and Verification</strong> system:
            </p>
            <ol style="margin-bottom:0; padding-left:1.2rem;">
                <li style="margin-bottom:8px;">
                    <strong style="color:#00F0FF;">One-Shot Constraint:</strong> Each registered user provides <strong>exactly one</strong> signature template image during enrollment.
                </li>
                <li style="margin-bottom:8px;">
                    <strong style="color:#00F0FF;">Multi-Stage Preprocessing:</strong> Grayscale conversion, Gaussian denoising, Otsu binarization, bounding box cropping, minor skew correction, and aspect-ratio preserving padding to <strong>128 × 128</strong>.
                </li>
                <li style="margin-bottom:8px;">
                    <strong style="color:#00F0FF;">Hybrid Feature Vector (8,119 dimensions):</strong>
                    <ul style="margin-top:4px;">
                        <li><strong style="color:#00FF9D;">HOG (8,100 dims):</strong> Captures local stroke direction, gradient orientation, and edge junctions.</li>
                        <li><strong style="color:#FCEE0A;">Hu Invariant Moments (7 dims):</strong> Translation-, scale-, and rotation-invariant global shape descriptors with logarithmic scaling.</li>
                        <li><strong style="color:#FF0055;">Geometric Features (12 dims):</strong> Aspect ratio, ink occupancy, contour area ratio, stroke density, centroid, and spatial stroke balance.</li>
                    </ul>
                </li>
                <li>
                    <strong style="color:#00F0FF;">Two Recognition Modes:</strong>
                    <ul style="margin-top:4px;">
                        <li><strong style="color:#00FF9D;">1:N Identification:</strong> Compares an unknown probe against all enrolled templates; applies dynamic confidence margin filtering to report the owner or <code>Unknown</code>.</li>
                        <li><strong style="color:#FCEE0A;">1:1 Verification:</strong> Compares a probe against a claimed user's single template; reports <code>Match</code>, <code>Not Match</code>, or <code>Uncertain</code>.</li>
                    </ul>
                </li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header("NAV.DECK // 02", "Quick Navigation")
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
    with c_btn1:
        if st.button("Register User", width='stretch'):
            st.switch_page("pages/1_Register.py")
    with c_btn2:
        if st.button("1:N Identify", width='stretch'):
            st.switch_page("pages/2_Identify.py")
    with c_btn3:
        if st.button("1:1 Verify", width='stretch'):
            st.switch_page("pages/3_Verify.py")
    with c_btn4:
        if st.button("Evaluation", width='stretch'):
            st.switch_page("pages/4_Evaluation.py")

with right_col:
    render_section_header("TELEMETRY // 03", "Recent Verification Activity")
    recent_logs = stats["recent_logs"]
    if recent_logs:
        log_df = pd.DataFrame(recent_logs)
        display_df = pd.DataFrame(
            {
                "Time": log_df["created_at"],
                "Op": log_df["operation_type"],
                "Claimed": log_df["claimed_name"].fillna("-"),
                "Predicted": log_df["predicted_name"].fillna("-"),
                "Score": log_df["similarity_score"].apply(lambda s: f"{s:.3f}"),
                "Decision": log_df["decision"],
            }
        )
        st.dataframe(display_df, width='stretch', hide_index=True)
    else:
        st.info(
            "No verification attempts logged yet. Try verifying or identifying a signature in the pages menu!"
        )

    # Evaluation Summary Preview
    summary_chart = EXPERIMENT_RESULTS_DIR / "experiment_comparison_chart.png"
    if summary_chart.exists():
        render_section_header("BENCHMARK // 04", "Benchmark Preview (E1 - E6)")
        st.image(
            str(summary_chart),
            caption="One-Shot Performance Across Experiments",
            width='stretch',
        )
