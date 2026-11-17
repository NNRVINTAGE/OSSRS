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
    DECISION_UNCERTAIN
)
from src.preprocessing import preprocess_signature_image, load_image
from src.feature_extraction import extract_features
from src.database import get_all_registered_templates, log_verification_event
from src.verification import verify_signature
from src.model_training import load_model
from src.pdf_handler import get_pdf_page_count, render_pdf_page_to_image, crop_roi

st.set_page_config(page_title="1:1 Signature Verification", page_icon="🛡️", layout="wide")

st.title("🛡️ 1:1 Signature Verification")
st.markdown("""
Compare a submitted signature against a **specific claimed user's enrolled template**.
The system verifies stroke topology and geometry, outputting **Match**, **Not Match**, or **Uncertain**.
""")

# Fetch registered templates
registered = get_all_registered_templates()

if not registered:
    st.warning("⚠️ No users are registered yet. Please enroll a user on the **Register** page first.")
    if st.button("Go to Register Page"):
        st.switch_page("pages/1_Register.py")
    st.stop()

col_claim, col_probe = st.columns([1, 1])

with col_claim:
    st.subheader("1. Claimed Identity & Enrolled Template")
    
    user_options = {f"{u['name']} (ID: {u['user_id']})": u for u in registered}
    selected_label = st.selectbox("Select Claimed User", list(user_options.keys()))
    claimed_user = user_options[selected_label]
    
    st.write(f"**Selected Identity:** {claimed_user['name']}")
    st.write(f"Enrolled on: {claimed_user['created_at']}")
    
    if Path(claimed_user["image_path"]).exists():
        st.image(claimed_user["image_path"], caption=f"Primary Template for {claimed_user['name']}", width=220)
    else:
        st.error("Template image file not found on disk.")
        
    st.divider()
    model_choice = st.selectbox(
        "Verification Matching Model",
        ["Cosine Similarity (HOG + Hu + Geometry)", "Learned SVM (Pairwise Classifier)"],
        index=0
    )
    
    is_svm = "SVM" in model_choice
    def_thresh = 0.40 if is_svm else DEFAULT_COSINE_MATCH_THRESHOLD
    def_uncert = 0.40 if is_svm else DEFAULT_COSINE_UNCERTAIN_LOWER
    
    with st.expander("⚙️ Decision Thresholds"):
        match_thresh = st.slider("Match Threshold", 0.10, 0.95, def_thresh, 0.01, key=f"v_thresh_{'svm' if is_svm else 'cosine'}_40")
        uncert_lower = st.slider("Uncertain Lower Bound", 0.10, 0.85, def_uncert, 0.01, key=f"v_uncert_{'svm' if is_svm else 'cosine'}_40")
        st.caption(
            "💡 **PCD Threshold Guide**: Nilai di atas Match Threshold = *Match*. "
            "Antara Uncertain dan Match = *Uncertain*. Di bawah Uncertain = *Not Match*."
        )

with col_probe:
    st.subheader("2. Upload Submitted Probe Signature")
    
    input_file = st.file_uploader(
        "Upload Signature for Verification",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload the questioned signature to verify"
    )
    
    probe_img = None
    if input_file is not None:
        file_bytes = input_file.read()
        file_ext = Path(input_file.name).suffix.lower()
        
        if file_ext == ".pdf":
            st.info("📄 PDF Document detected. Rendered with PyMuPDF.")
            num_pages = get_pdf_page_count(file_bytes)
            page_idx = st.number_input("Select PDF Page", min_value=0, max_value=max(0, num_pages - 1), value=0, step=1, key="v_p")
            page_img = render_pdf_page_to_image(file_bytes, page_number=page_idx, dpi=150)
            
            c1, c2 = st.columns(2)
            with c1:
                x_pct = st.slider("X Position (%)", 0, 90, 10, 1, key="v_x")
                w_pct = st.slider("Width (%)", 10, 100, 80, 1, key="v_w")
            with c2:
                y_pct = st.slider("Y Position (%)", 0, 90, 60, 1, key="v_y")
                h_pct = st.slider("Height (%)", 10, 100, 35, 1, key="v_h")
            probe_img = crop_roi(page_img, x_pct, y_pct, w_pct, h_pct)
            st.image(probe_img, caption="Cropped ROI", use_container_width=True)
        else:
            probe_img = load_image(file_bytes)

# Verification Decision Section
st.divider()
if probe_img is not None:
    st.subheader("3. Verification Decision & Analysis")
    
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
                st.warning(f"Could not load SVM: {e}. Falling back to Cosine Similarity.")
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
        uncertain_lower=uncert_lower
    )
    
    dec = verif_res["decision"]
    score = verif_res["similarity_score"]
    
    # Display Result
    r_col1, r_col2 = st.columns([1, 1])
    
    with r_col1:
        if dec == DECISION_MATCH:
            st.success(f"## Decision: {DECISION_MATCH} ✅")
        elif dec == DECISION_UNCERTAIN:
            st.warning(f"## Decision: {DECISION_UNCERTAIN} ⚠️")
        else:
            st.error(f"## Decision: {DECISION_NOT_MATCH} ❌")
            
        st.write(f"**Similarity Score:** `{score:.4f}`")
        st.write(f"**Match Threshold:** `{match_thresh:.3f}` | **Uncertain Boundary:** `{uncert_lower:.3f}`")
        st.info(verif_res["explanation"])
        st.caption(f"🛡️ {verif_res['disclaimer']}")
        
        # Log to Database
        log_id = log_verification_event(
            claimed_user_id=claimed_user["user_id"],
            predicted_user_id=claimed_user["user_id"] if dec == DECISION_MATCH else None,
            similarity_score=score,
            decision=dec,
            operation_type="verification"
        )
        st.caption(f"Logged to verification database with Event ID #{log_id}")

    with r_col2:
        st.markdown("**Visual Comparison (128x128 Preprocessed):**")
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.image(processed_probe, caption="Submitted Probe", width=180)
        with cmp2:
            if Path(claimed_user["image_path"]).exists():
                st.image(claimed_user["image_path"], caption=f"Registered Template ({claimed_user['name']})", width=180)
else:
    st.info("Upload a probe signature document above to run 1:1 verification.")
