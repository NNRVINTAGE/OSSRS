import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from config.settings import (
    DEFAULT_COSINE_MATCH_THRESHOLD,
    DEFAULT_COSINE_UNCERTAIN_LOWER,
    DEFAULT_SVM_MATCH_THRESHOLD,
    DEFAULT_SVM_UNCERTAIN_LOWER,
    MODELS_DIR,
    DECISION_MATCH,
    DECISION_NOT_MATCH,
    DECISION_UNCERTAIN,
)
from src.preprocessing import preprocess_signature_image, load_image
from src.feature_extraction import extract_features
from src.database import get_all_registered_templates, log_verification_event
from src.verification import verify_signature
from src.model_training import load_model
from src.pdf_handler import get_pdf_page_count, render_pdf_page_to_image, crop_roi
from src.ui_theme import (
    apply_cyberpunk_theme,
    render_hud_header,
    render_section_header,
    render_decision_hud,
)

st.set_page_config(
    page_title="1:1 Signature Verification", page_icon="🛡️", layout="wide"
)

apply_cyberpunk_theme("VERIFY_1V1 // 03")

render_hud_header(
    title="🛡️ 1:1 Signature Verification",
    subtitle="Compare a submitted signature against a specific claimed user's enrolled template. The system verifies stroke topology and geometry, outputting Match, Not Match, or Uncertain.",
    sys_code="MOD.VERIFY // 03",
    status_tag="1:1 AUTHENTICITY PROTOCOL",
)

# Fetch registered templates
registered = get_all_registered_templates()

if not registered:
    st.warning(
        "⚠️ No users are registered yet. Please enroll a user on the **Register** page first."
    )
    if st.button("Go to Register Page"):
        st.switch_page("pages/1_Register.py")
    st.stop()

col_claim, col_probe = st.columns([1, 1])

with col_claim:
    render_section_header(
        "CLAIM.ID // 01", "1. Claimed Identity & Enrolled Template"
    )

    user_options = {f"{u['name']} (ID: {u['user_id']})": u for u in registered}
    selected_label = st.selectbox("Select Claimed User", list(user_options.keys()))
    claimed_user = user_options[selected_label]

    st.markdown(
        f"""
        <div class="cp-user-card">
            <div style="display:flex; justify-content:space-between; font-family:'Share Tech Mono',monospace; font-size:0.74rem; color:#00F0FF;">
                <span>TARGET SUBJECT // ID #{claimed_user['user_id']:02d}</span>
                <span style="color:#00FF9D;">● TEMPLATE LOCKED</span>
            </div>
            <div style="font-family:'Orbitron',sans-serif; font-size:1.1rem; font-weight:700; color:#FFFFFF; margin:4px 0;">
                {claimed_user['name']}
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#7F9ABD;">
                ENROLLED TIMESTAMP: {claimed_user['created_at']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if Path(claimed_user["image_path"]).exists():
        st.image(
            claimed_user["image_path"],
            caption=f"Primary Template for {claimed_user['name']}",
            width=220,
        )
    else:
        st.error("Template image file not found on disk.")

    st.divider()
    model_choice = st.selectbox(
        "Verification Matching Model",
        [
            "Cosine Similarity (HOG + Hu + Geometry)",
            "Learned SVM (Pairwise Classifier)",
        ],
        index=0,
    )

    is_svm = "SVM" in model_choice
    def_thresh = 0.40 if is_svm else DEFAULT_COSINE_MATCH_THRESHOLD
    def_uncert = 0.40 if is_svm else DEFAULT_COSINE_UNCERTAIN_LOWER

    with st.expander("⚙️ Decision Thresholds"):
        match_thresh = st.slider(
            "Match Threshold",
            0.10,
            0.95,
            def_thresh,
            0.01,
            key=f"v_thresh_{'svm' if is_svm else 'cosine'}_40",
        )
        uncert_lower = st.slider(
            "Uncertain Lower Bound",
            0.10,
            0.85,
            def_uncert,
            0.01,
            key=f"v_uncert_{'svm' if is_svm else 'cosine'}_40",
        )
        st.caption(
            "💡 **PCD Threshold Guide**: Nilai di atas Match Threshold = *Match*. "
            "Antara Uncertain dan Match = *Uncertain*. Di bawah Uncertain = *Not Match*."
        )

with col_probe:
    render_section_header(
        "PROBE.INPUT // 02", "2. Upload Submitted Probe Signature"
    )

    input_file = st.file_uploader(
        "Upload Signature for Verification",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload the questioned signature to verify",
    )

    probe_img = None
    if input_file is not None:
        file_bytes = input_file.read()
        file_ext = Path(input_file.name).suffix.lower()

        if file_ext == ".pdf":
            st.info("📄 PDF Document detected. Rendered with PyMuPDF.")
            num_pages = get_pdf_page_count(file_bytes)
            page_idx = st.number_input(
                "Select PDF Page",
                min_value=0,
                max_value=max(0, num_pages - 1),
                value=0,
                step=1,
                key="v_p",
            )
            page_img = render_pdf_page_to_image(
                file_bytes, page_number=page_idx, dpi=150
            )

            c1, c2 = st.columns(2)
            with c1:
                x_pct = st.slider("X Position (%)", 0, 90, 10, 1, key="v_x")
                w_pct = st.slider("Width (%)", 10, 100, 80, 1, key="v_w")
            with c2:
                y_pct = st.slider("Y Position (%)", 0, 90, 60, 1, key="v_y")
                h_pct = st.slider("Height (%)", 10, 100, 35, 1, key="v_h")
            probe_img = crop_roi(page_img, x_pct, y_pct, w_pct, h_pct)
            st.image(probe_img, caption="Cropped ROI", width='stretch')
        else:
            probe_img = load_image(file_bytes)

# Verification Decision Section
st.divider()
if probe_img is not None:
    render_section_header(
        "VERDICT.HUD // 03", "3. Verification Decision & Analysis"
    )

    processed_probe, debug = preprocess_signature_image(probe_img, return_debug=True)
    probe_features = extract_features(processed_probe)
    tmpl_features = claimed_user["features"]

    if tmpl_features is None:
        st.error("Error: Could not load feature vector for claimed user.")
        st.stop()

    # Model selection
    svm_model = None
    method = "cosine"
    if "SVM" in model_choice:
        svm_path = MODELS_DIR / "linear_svm_pipeline.joblib"
        if svm_path.exists():
            try:
                svm_model = load_model("linear_svm_pipeline.joblib")
                method = "svm"
            except Exception as e:
                st.warning(
                    f"Could not load SVM: {e}. Falling back to Cosine Similarity."
                )
        else:
            st.warning("Trained SVM model not found. Using Cosine Similarity.")

    # Run 1:1 Verification
    verif_res = verify_signature(
        probe_features=probe_features,
        template_features=tmpl_features,
        claimed_user_name=claimed_user["name"],
        method=method,
        svm_model=svm_model,
        match_threshold=match_thresh,
        uncertain_lower=uncert_lower,
    )

    dec = verif_res["decision"]
    score = verif_res["similarity_score"]

    # Display Result
    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        render_decision_hud(
            decision=dec,
            headline=f"Decision: {dec}",
            score=score,
            match_thresh=match_thresh,
            uncert_lower=uncert_lower,
            explanation=verif_res["explanation"],
            disclaimer=verif_res["disclaimer"],
        )

        # Log to Database
        log_id = log_verification_event(
            claimed_user_id=claimed_user["user_id"],
            predicted_user_id=claimed_user["user_id"]
            if dec == DECISION_MATCH
            else None,
            similarity_score=score,
            decision=dec,
            operation_type="verification",
        )
        st.caption(f"Logged to verification database with Event ID #{log_id}")

    with r_col2:
        st.markdown(
            """
            <div style="font-family:'Orbitron',sans-serif; font-size:0.9rem; color:#00F0FF; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:10px;">
                🔬 Visual Comparison (128x128 Preprocessed)
            </div>
            """,
            unsafe_allow_html=True,
        )
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.image(processed_probe, caption="Submitted Probe", width=180)
        with cmp2:
            if Path(claimed_user["image_path"]).exists():
                st.image(
                    claimed_user["image_path"],
                    caption=f"Registered Template ({claimed_user['name']})",
                    width=180,
                )
else:
    st.info("Upload a probe signature document above to run 1:1 verification.")
