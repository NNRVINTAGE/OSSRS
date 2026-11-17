"""
Cyberpunk Biometric Forensics HUD Theme & Reusable UI Components for OSSRS.
Strictly front-end presentation layer; does not alter backend or ML logic.
"""

import streamlit as st
from typing import Optional


def apply_cyberpunk_theme(page_code: str = "SYS.CORE // 00") -> None:
    """Injects global Cyberpunk HUD CSS and renders the tactical sidebar telemetry."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700;800;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

        :root {
            --cp-bg-deep: #05070D;
            --cp-bg-panel: #0A1020;
            --cp-bg-panel-alt: #0E162B;
            --cp-cyan: #00F0FF;
            --cp-cyan-dim: rgba(0, 240, 255, 0.16);
            --cp-magenta: #FF0055;
            --cp-magenta-dim: rgba(255, 0, 85, 0.16);
            --cp-yellow: #FCEE0A;
            --cp-yellow-dim: rgba(252, 238, 10, 0.16);
            --cp-green: #00FF9D;
            --cp-green-dim: rgba(0, 255, 157, 0.16);
            --cp-purple: #B829FF;
            --cp-text-main: #E2F1FF;
            --cp-text-muted: #7F9ABD;
            --cp-border: rgba(0, 240, 255, 0.32);
            --cp-font-display: 'Orbitron', 'Segoe UI', sans-serif;
            --cp-font-body: 'Rajdhani', 'Segoe UI', sans-serif;
            --cp-font-mono: 'Share Tech Mono', 'Consolas', monospace;
        }

        /* Main App Container & Cyber Grid Background */
        .stApp {
            background-color: var(--cp-bg-deep) !important;
            background-image:
                radial-gradient(circle at 15% 10%, rgba(0, 240, 255, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(255, 0, 85, 0.07) 0%, transparent 40%),
                linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px) !important;
            background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px !important;
            color: var(--cp-text-main) !important;
            font-family: var(--cp-font-body) !important;
        }

        /* Subtle CRT Scanline Overlay */
        .stApp::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 0, 0, 0.12),
                rgba(0, 0, 0, 0.12) 1px,
                transparent 1px,
                transparent 3px
            );
            pointer-events: none;
            z-index: 9999;
            opacity: 0.45;
        }

        /* Top Header Bar Transparency */
        header[data-testid="stHeader"] {
            background: rgba(5, 7, 13, 0.85) !important;
            backdrop-filter: blur(8px);
            border-bottom: 1px solid rgba(0, 240, 255, 0.18);
        }

        /* Typography Overrides */
        h1, h2, h3, .stSubheader {
            font-family: var(--cp-font-display) !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase;
            color: var(--cp-text-main) !important;
        }

        p, li, label, .stMarkdown {
            font-family: var(--cp-font-body) !important;
            font-size: 1.04rem;
            line-height: 1.55;
            color: var(--cp-text-main);
        }

        code, pre {
            font-family: var(--cp-font-mono) !important;
            color: var(--cp-cyan) !important;
            background-color: rgba(0, 240, 255, 0.08) !important;
            border: 1px solid rgba(0, 240, 255, 0.25) !important;
            border-radius: 2px !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070B14 0%, #0A1122 100%) !important;
            border-right: 1px solid rgba(0, 240, 255, 0.35) !important;
            box-shadow: 4px 0 24px rgba(0, 240, 255, 0.08) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            font-family: var(--cp-font-display) !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            border-left: 3px solid transparent;
            border-radius: 0 4px 4px 0 !important;
            margin: 4px 8px !important;
            padding: 8px 12px !important;
            transition: all 0.2s ease !important;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
            background: rgba(0, 240, 255, 0.12) !important;
            border-left: 3px solid var(--cp-cyan) !important;
            box-shadow: inset 0 0 12px rgba(0, 240, 255, 0.15);
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: linear-gradient(90deg, rgba(0, 240, 255, 0.22) 0%, rgba(0, 240, 255, 0.03) 100%) !important;
            border-left: 3px solid var(--cp-yellow) !important;
            color: var(--cp-cyan) !important;
            text-shadow: 0 0 8px rgba(0, 240, 255, 0.5);
        }

        /* Cyberpunk Buttons */
        .stButton > button {
            font-family: var(--cp-font-display) !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.09em !important;
            text-transform: uppercase !important;
            color: var(--cp-cyan) !important;
            background: linear-gradient(135deg, rgba(10, 18, 36, 0.95) 0%, rgba(0, 240, 255, 0.12) 100%) !important;
            border: 1px solid var(--cp-cyan) !important;
            border-radius: 0px !important;
            clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px));
            padding: 0.6rem 1.1rem !important;
            box-shadow: 0 0 12px rgba(0, 240, 255, 0.15), inset 0 0 8px rgba(0, 240, 255, 0.08) !important;
            transition: all 0.2s ease-in-out !important;
        }

        .stButton > button:hover {
            color: #05070D !important;
            background: var(--cp-cyan) !important;
            border-color: var(--cp-cyan) !important;
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.65), 0 0 4px #FFFFFF !important;
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"] {
            color: #05070D !important;
            background: linear-gradient(90deg, var(--cp-cyan) 0%, #00FF9D 100%) !important;
            border: 1px solid #FFFFFF !important;
            box-shadow: 0 0 18px rgba(0, 240, 255, 0.45) !important;
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(90deg, var(--cp-yellow) 0%, var(--cp-cyan) 100%) !important;
            box-shadow: 0 0 26px rgba(252, 238, 10, 0.75) !important;
        }

        /* Inputs, Selectboxes, NumberInputs, TextInputs */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background-color: rgba(10, 16, 32, 0.92) !important;
            border: 1px solid rgba(0, 240, 255, 0.38) !important;
            border-radius: 2px !important;
            color: var(--cp-text-main) !important;
            font-family: var(--cp-font-mono) !important;
        }

        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {
            border-color: var(--cp-yellow) !important;
            box-shadow: 0 0 12px rgba(252, 238, 10, 0.35) !important;
        }

        /* File Uploader Dropzone */
        [data-testid="stFileUploader"] section {
            background: linear-gradient(135deg, rgba(10, 16, 32, 0.9) 0%, rgba(0, 240, 255, 0.05) 100%) !important;
            border: 1px dashed var(--cp-cyan) !important;
            border-radius: 4px !important;
            padding: 1.2rem !important;
            box-shadow: inset 0 0 18px rgba(0, 240, 255, 0.06);
            transition: all 0.25s ease;
        }

        [data-testid="stFileUploader"] section:hover {
            border-color: var(--cp-yellow) !important;
            background: rgba(252, 238, 10, 0.05) !important;
            box-shadow: inset 0 0 22px rgba(252, 238, 10, 0.12), 0 0 12px rgba(0, 240, 255, 0.2);
        }

        /* Expander Styling */
        details[data-testid="stExpander"] {
            background: rgba(10, 16, 32, 0.85) !important;
            border: 1px solid rgba(0, 240, 255, 0.3) !important;
            border-left: 3px solid var(--cp-yellow) !important;
            border-radius: 2px !important;
        }

        details[data-testid="stExpander"] summary {
            font-family: var(--cp-font-display) !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.06em !important;
            color: var(--cp-yellow) !important;
        }

        /* Native st.metric Override */
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(10, 16, 32, 0.95) 0%, rgba(14, 24, 46, 0.85) 100%);
            border: 1px solid rgba(0, 240, 255, 0.32);
            border-top: 2px solid var(--cp-cyan);
            padding: 14px 16px;
            clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.45);
        }

        [data-testid="stMetricLabel"] {
            font-family: var(--cp-font-mono) !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--cp-text-muted) !important;
            font-size: 0.78rem !important;
        }

        [data-testid="stMetricValue"] {
            font-family: var(--cp-font-display) !important;
            color: var(--cp-cyan) !important;
            text-shadow: 0 0 10px rgba(0, 240, 255, 0.35);
            font-size: 1.55rem !important;
        }

        /* DataFrame Container */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(0, 240, 255, 0.35) !important;
            box-shadow: 0 0 18px rgba(0, 240, 255, 0.08);
        }

        /* Image Frame Styling */
        [data-testid="stImage"] img {
            border: 1px solid rgba(0, 240, 255, 0.4);
            background: #070B14;
            padding: 4px;
            box-shadow: 0 0 14px rgba(0, 240, 255, 0.14);
        }

        [data-testid="stImageCaption"] {
            font-family: var(--cp-font-mono) !important;
            color: var(--cp-cyan) !important;
            letter-spacing: 0.05em;
            font-size: 0.78rem !important;
        }

        /* Dividers */
        hr {
            border-color: rgba(0, 240, 255, 0.25) !important;
            box-shadow: 0 0 8px rgba(0, 240, 255, 0.2);
        }

        /* Custom Cyberpunk HUD Banner */
        .cp-hero-banner {
            position: relative;
            background: linear-gradient(135deg, rgba(10, 16, 32, 0.96) 0%, rgba(16, 24, 48, 0.9) 60%, rgba(0, 240, 255, 0.09) 100%);
            border: 1px solid rgba(0, 240, 255, 0.45);
            border-left: 4px solid var(--cp-cyan);
            padding: 20px 24px;
            margin-bottom: 22px;
            clip-path: polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
            box-shadow: 0 0 28px rgba(0, 240, 255, 0.12), inset 0 0 24px rgba(0, 240, 255, 0.05);
        }

        .cp-hero-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 8px;
            font-family: var(--cp-font-mono);
            font-size: 0.76rem;
            letter-spacing: 0.12em;
            color: var(--cp-cyan);
        }

        .cp-badge {
            background: rgba(0, 240, 255, 0.14);
            border: 1px solid var(--cp-cyan);
            color: var(--cp-cyan);
            padding: 2px 10px;
            font-family: var(--cp-font-mono);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .cp-badge-yellow {
            background: rgba(252, 238, 10, 0.15);
            border: 1px solid var(--cp-yellow);
            color: var(--cp-yellow);
        }

        .cp-hero-title {
            font-family: var(--cp-font-display);
            font-size: 1.85rem;
            font-weight: 800;
            color: #FFFFFF;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 4px 0 6px 0;
            text-shadow: 0 0 14px rgba(0, 240, 255, 0.45);
        }

        .cp-hero-sub {
            font-family: var(--cp-font-body);
            font-size: 1.02rem;
            color: var(--cp-text-muted);
            margin: 0;
            letter-spacing: 0.02em;
        }

        /* HUD Metric Card */
        .cp-metric-card {
            position: relative;
            background: linear-gradient(160deg, rgba(10, 16, 32, 0.96) 0%, rgba(14, 22, 42, 0.9) 100%);
            border: 1px solid rgba(0, 240, 255, 0.32);
            border-top: 3px solid var(--cp-cyan);
            padding: 16px 14px;
            text-align: left;
            clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            margin-bottom: 12px;
        }

        .cp-metric-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: var(--cp-font-mono);
            font-size: 0.7rem;
            color: var(--cp-text-muted);
            letter-spacing: 0.1em;
            margin-bottom: 6px;
        }

        .cp-metric-val {
            font-family: var(--cp-font-display);
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 4px;
        }

        .cp-metric-label {
            font-family: var(--cp-font-body);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #B8CBE0;
        }

        /* HUD Panel Box */
        .cp-panel {
            background: linear-gradient(145deg, rgba(10, 16, 32, 0.92) 0%, rgba(8, 12, 24, 0.95) 100%);
            border: 1px solid rgba(0, 240, 255, 0.3);
            border-left: 3px solid var(--cp-cyan);
            padding: 18px 20px;
            margin: 12px 0;
            clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
        }

        .cp-notice {
            background: linear-gradient(90deg, rgba(252, 238, 10, 0.12) 0%, rgba(10, 16, 32, 0.92) 100%);
            border: 1px solid rgba(252, 238, 10, 0.45);
            border-left: 4px solid var(--cp-yellow);
            padding: 13px 18px;
            margin: 14px 0 20px 0;
            font-family: var(--cp-font-body);
            font-size: 0.96rem;
            color: #F8FAFC;
            clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
        }

        .cp-section-hdr {
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid rgba(0, 240, 255, 0.3);
            padding-bottom: 6px;
            margin: 18px 0 14px 0;
        }

        .cp-section-tag {
            font-family: var(--cp-font-mono);
            font-size: 0.75rem;
            color: #05070D;
            background: var(--cp-cyan);
            padding: 2px 7px;
            font-weight: 700;
            letter-spacing: 0.08em;
        }

        .cp-section-title {
            font-family: var(--cp-font-display);
            font-size: 1.08rem;
            font-weight: 700;
            color: var(--cp-text-main);
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin: 0;
        }

        /* Biometric Verdict Banner */
        .cp-verdict {
            padding: 18px 20px;
            margin: 12px 0 16px 0;
            clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
        }

        .cp-verdict-match {
            background: linear-gradient(135deg, rgba(0, 255, 157, 0.16) 0%, rgba(10, 18, 32, 0.95) 100%);
            border: 1px solid var(--cp-green);
            border-left: 5px solid var(--cp-green);
            box-shadow: 0 0 24px rgba(0, 255, 157, 0.18);
        }

        .cp-verdict-uncertain {
            background: linear-gradient(135deg, rgba(252, 238, 10, 0.16) 0%, rgba(10, 18, 32, 0.95) 100%);
            border: 1px solid var(--cp-yellow);
            border-left: 5px solid var(--cp-yellow);
            box-shadow: 0 0 24px rgba(252, 238, 10, 0.18);
        }

        .cp-verdict-reject {
            background: linear-gradient(135deg, rgba(255, 0, 85, 0.18) 0%, rgba(10, 18, 32, 0.95) 100%);
            border: 1px solid var(--cp-magenta);
            border-left: 5px solid var(--cp-magenta);
            box-shadow: 0 0 24px rgba(255, 0, 85, 0.2);
        }

        /* User Vault Card */
        .cp-user-card {
            background: linear-gradient(160deg, rgba(10, 16, 32, 0.95) 0%, rgba(14, 24, 46, 0.9) 100%);
            border: 1px solid rgba(0, 240, 255, 0.35);
            border-top: 2px solid var(--cp-cyan);
            padding: 12px;
            margin-bottom: 8px;
            clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar HUD Telemetry Block
    with st.sidebar:
        st.markdown(
            f"""
            <div style="border:1px solid rgba(0,240,255,0.4); background:rgba(5,9,18,0.9); padding:12px 14px; margin-bottom:14px; clip-path:polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);">
                <div style="display:flex; justify-content:space-between; align-items:center; font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#00FF9D; letter-spacing:0.1em;">
                    <span>● BIOMETRIC CORE</span>
                    <span>ONLINE</span>
                </div>
                <div style="font-family:'Orbitron',sans-serif; font-size:1.05rem; font-weight:800; color:#FFFFFF; letter-spacing:0.08em; margin:6px 0 2px 0; text-shadow:0 0 10px rgba(0,240,255,0.5);">
                    OSSRS // HUD v2.6
                </div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#7F9ABD; letter-spacing:0.06em;">
                    NODE: {page_code}<br/>
                    DIM: 8,119-D [HOG+HU+GEO]<br/>
                    NORM: 128×128 CENTROID
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hud_header(
    title: str,
    subtitle: str,
    sys_code: str = "SYS.NODE // 00",
    status_tag: str = "ONE-SHOT PROTOCOL ACTIVE",
) -> None:
    """Renders the Cyberpunk top page command banner."""
    st.markdown(
        f"""
        <div class="cp-hero-banner">
            <div class="cp-hero-top">
                <span>[{sys_code}] // OFFLINE BIOMETRIC FORENSICS TERMINAL</span>
                <span class="cp-badge cp-badge-yellow">{status_tag}</span>
            </div>
            <div class="cp-hero-title">{title}</div>
            <p class="cp-hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(tag: str, title: str) -> None:
    """Renders a tactical Cyberpunk section header bar."""
    st.markdown(
        f"""
        <div class="cp-section-hdr">
            <span class="cp-section-tag">{tag}</span>
            <h3 class="cp-section-title">{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    accent_hex: str = "#00F0FF",
    code: str = "TEL-01",
) -> None:
    """Renders a chamfered Cyberpunk metric card."""
    st.markdown(
        f"""
        <div class="cp-metric-card" style="border-top-color: {accent_hex};">
            <div class="cp-metric-header">
                <span>//{code}</span>
                <span style="color:{accent_hex};">● LIVE</span>
            </div>
            <div class="cp-metric-val" style="color: {accent_hex}; text-shadow: 0 0 12px {accent_hex}55;">
                {value}
            </div>
            <div class="cp-metric-label">{label}</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#7F9ABD; margin-top:4px;">
                {subtext}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_hud(
    decision: str,
    headline: str,
    score: float,
    match_thresh: float,
    uncert_lower: float,
    margin: Optional[float] = None,
    margin_thresh: Optional[float] = None,
    explanation: str = "",
    disclaimer: str = "",
) -> None:
    """Renders a Cyberpunk biometric decision terminal with a visual confidence gauge."""
    if decision == "Match":
        cls_name = "cp-verdict-match"
        accent = "#00FF9D"
        status_code = "STATUS // VERIFIED_MATCH"
        icon = "⚡"
    elif decision == "Uncertain":
        cls_name = "cp-verdict-uncertain"
        accent = "#FCEE0A"
        status_code = "STATUS // AMBIGUOUS_SIGNAL"
        icon = "⚠️"
    else:
        cls_name = "cp-verdict-reject"
        accent = "#FF0055"
        status_code = f"STATUS // {decision.upper().replace(' ', '_')}"
        icon = "⛔"

    pct_score = max(0.0, min(100.0, score * 100.0))
    pct_match = max(0.0, min(100.0, match_thresh * 100.0))
    pct_uncert = max(0.0, min(100.0, uncert_lower * 100.0))

    margin_html = ""
    if margin is not None and margin_thresh is not None:
        margin_ok = margin >= margin_thresh
        m_color = "#00FF9D" if margin_ok else "#FCEE0A"
        margin_html = f"""
        <div style="margin-top:8px; font-family:'Share Tech Mono',monospace; font-size:0.82rem; color:#E2F1FF;">
            TOP-1 VS TOP-2 SEPARATION MARGIN: <strong style="color:{m_color};">{margin:.3f}</strong>
            <span style="color:#7F9ABD;"> (REQUIRED MINIMUM: {margin_thresh:.3f})</span>
        </div>
        """

    disclaimer_html = ""
    if disclaimer:
        disclaimer_html = f"""
        <div style="margin-top:10px; padding-top:8px; border-top:1px dashed rgba(0,240,255,0.25); font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#7F9ABD;">
            🛡️ {disclaimer}
        </div>
        """

    st.markdown(
        f"""
        <div class="cp-verdict {cls_name}">
            <div style="display:flex; justify-content:space-between; align-items:center; font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:{accent}; letter-spacing:0.1em; margin-bottom:4px;">
                <span>{status_code}</span>
                <span>SCORE: {score:.4f}</span>
            </div>
            <div style="font-family:'Orbitron',sans-serif; font-size:1.35rem; font-weight:800; color:{accent}; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:10px; text-shadow:0 0 12px {accent}66;">
                {icon} {headline}
            </div>
            <div style="background:rgba(5,7,13,0.85); border:1px solid rgba(0,240,255,0.28); padding:10px 12px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-family:'Share Tech Mono',monospace; font-size:0.74rem; color:#B8CBE0; margin-bottom:5px;">
                    <span>BIOMETRIC SIMILARITY TELEMETRY: <strong style="color:{accent};">{score:.3f}</strong></span>
                    <span>UNCERTAIN: {uncert_lower:.2f} | MATCH THRESHOLD: {match_thresh:.2f}</span>
                </div>
                <div style="position:relative; height:12px; background:#05070D; border:1px solid rgba(255,255,255,0.18); overflow:hidden;">
                    <div style="position:absolute; left:0; top:0; bottom:0; width:{pct_score:.1f}%; background:linear-gradient(90deg, #00F0FF, {accent}); box-shadow:0 0 10px {accent};"></div>
                    <div style="position:absolute; left:{pct_uncert:.1f}%; top:0; bottom:0; width:2px; background:#FCEE0A;" title="Uncertain Lower Bound"></div>
                    <div style="position:absolute; left:{pct_match:.1f}%; top:0; bottom:0; width:2px; background:#00FF9D;" title="Match Threshold"></div>
                </div>
                {margin_html}
            </div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:1.02rem; color:#E2F1FF; line-height:1.45;">
                <strong>Diagnostic Readout:</strong> {explanation}
            </div>
            {disclaimer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
