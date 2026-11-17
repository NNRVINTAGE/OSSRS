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
    DEFAULT_TOP_MARGIN_THRESHOLD,
    MODELS_DIR,
    DECISION_MATCH,
    DECISION_UNCERTAIN,
    DECISION_UNKNOWN
)
from src.preprocessing import preprocess_signature_image, load_image
from src.feature_extraction import extract_features
from src.database import get_all_registered_templates, log_verification_event
from src.identification import identify_signature
from src.model_training import load_model
from src.pdf_handler import get_pdf_page_count, render_pdf_page_to_image, crop_roi

st.set_page_config(page_title="1:N Signature Identification", page_icon="🔎", layout="wide")

st.title("🔎 1:N Signature Identification")
st.markdown("""
Submit an unknown signature. The system will extract features and compare against **all registered templates**,
returning the ranked candidates and identifying the writer, or reporting **Unknown** / **Uncertain** if confidence is insufficient.
""")

# Load registered templates
registered = get_all_registered_templates()

if not registered:
    st.warning("⚠️ No users are currently registered in the system. Please visit the **Register** page to enroll users first.")
    if st.button("Go to Register Page"):
        st.switch_page("pages/1_Register.py")
    st.stop()

col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("1. Upload Query Signature")
    
    # Model Selector
    model_choice = st.selectbox(
        "Select Identification Model",
        ["Cosine Similarity (HOG + Hu + Geometry)", "Learned SVM (Pairwise Classifier)"],
        index=0
    )
    
    is_svm = "SVM" in model_choice
    def_thresh = 0.40 if is_svm else DEFAULT_COSINE_MATCH_THRESHOLD
    def_uncert = 0.40 if is_svm else DEFAULT_COSINE_UNCERTAIN_LOWER

    # Sensitivity controls
    with st.expander("⚙️ Advanced Threshold Settings"):
        match_thresh = st.slider("Match Threshold", 0.10, 0.95, def_thresh, 0.01, key=f"id_thresh_{'svm' if is_svm else 'cosine'}_40")
        uncert_lower = st.slider("Uncertain Lower Bound", 0.10, 0.85, def_uncert, 0.01, key=f"id_uncert_{'svm' if is_svm else 'cosine'}_40")
        margin_thresh = st.slider("Top-1 vs Top-2 Confidence Margin", 0.01, 0.20, DEFAULT_TOP_MARGIN_THRESHOLD, 0.01)
        st.caption(
            "💡 **PCD Threshold Guide**: Nilai di atas Match Threshold = *Match*. "
            "Margin membedakan antara kandidat peringkat 1 dan 2 agar identifikasi tidak ambigu."
        )
        
    input_file = st.file_uploader(
        "Upload Signature Document",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload an unknown signature image or PDF"
    )
    
    query_image = None
    if input_file is not None:
        file_bytes = input_file.read()
        file_ext = Path(input_file.name).suffix.lower()
        
        if file_ext == ".pdf":
            st.info("📄 PDF Document detected. Rendered with PyMuPDF.")
            num_pages = get_pdf_page_count(file_bytes)
            page_idx = st.number_input("Select PDF Page", min_value=0, max_value=max(0, num_pages - 1), value=0, step=1)
            page_img = render_pdf_page_to_image(file_bytes, page_number=page_idx, dpi=150)
            
            c1, c2 = st.columns(2)
            with c1:
                x_pct = st.slider("X Position (%)", 0, 90, 10, 1, key="id_x")
                w_pct = st.slider("Width (%)", 10, 100, 80, 1, key="id_w")
            with c2:
                y_pct = st.slider("Y Position (%)", 0, 90, 60, 1, key="id_y")
                h_pct = st.slider("Height (%)", 10, 100, 35, 1, key="id_h")
            query_image = crop_roi(page_img, x_pct, y_pct, w_pct, h_pct)
            st.image(query_image, caption="Cropped ROI", use_container_width=True)
        else:
            query_image = load_image(file_bytes)

with col_result:
    st.subheader("2. Identification Output & Ranking")
    
    if query_image is not None:
        processed_query, debug = preprocess_signature_image(query_image, return_debug=True)
        query_features = extract_features(processed_query)
        
        # Display query inspection
        st.image(processed_query, caption="Normalized Query (128x128)", width=160)
        
        # Determine model
        svm_model = None
        method = "cosine"
        if "SVM" in model_choice:
            svm_path = MODELS_DIR / "linear_svm_pipeline.joblib"
            if svm_path.exists():
                try:
                    svm_model = load_model("linear_svm_pipeline.joblib")
                    method = "svm"
                except Exception as e:
                    st.warning(f"Could not load SVM model: {e}. Falling back to Cosine Similarity.")
            else:
                st.warning("Trained SVM model not found. Using Cosine Similarity.")
                
        # Run 1:N Identification
        result = identify_signature(
            probe_features=query_features,
            registered_templates=registered,
            method=method,
            svm_model=svm_model,
            match_threshold=match_thresh,
            uncertain_lower=uncert_lower,
            top_margin_threshold=margin_thresh
        )
        
        # Render Decision Banner
        dec = result["decision"]
        if dec == DECISION_MATCH:
            st.success(f"### Result: MATCH — {result['predicted_name']}")
        elif dec == DECISION_UNCERTAIN:
            st.warning(f"### Result: UNCERTAIN (Top candidate: {result['predicted_name']})")
        else:
            st.error("### Result: UNKNOWN (No Registered Match)")
            
        st.write(f"**Explanation:** {result['explanation']}")
        st.write(f"- Top Similarity Score: **{result['top_score']:.3f}** (Threshold: {match_thresh:.3f})")
        st.write(f"- Top-1 vs Top-2 Margin: **{result['margin']:.3f}** (Required: {margin_thresh:.3f})")
        
        # Log to Database
        log_id = log_verification_event(
            claimed_user_id=None,
            predicted_user_id=result["predicted_user_id"],
            similarity_score=result["top_score"],
            decision=result["decision"],
            operation_type="identification"
        )
        st.caption(f"Logged to verification database with Event ID #{log_id}")
        
        # Ranked Candidates Table
        st.divider()
        st.subheader("📊 Ranked Candidate Enrolled Templates")
        ranked = result["ranked_candidates"]
        if ranked:
            ranked_df = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "User Name": c["name"],
                    "Similarity Score": f"{c['score']:.4f}",
                    "Status": "Top Match" if i == 0 and dec == DECISION_MATCH else ("Contender" if c['score'] >= uncert_lower else "Rejected")
                }
                for i, c in enumerate(ranked)
            ])
            st.dataframe(ranked_df, use_container_width=True, hide_index=True)
            
            # Show visual comparison with top registered template
            top_cand = ranked[0]
            top_tmpl = [t for t in registered if t["user_id"] == top_cand["user_id"]][0]
            
            cmp_c1, cmp_c2 = st.columns(2)
            with cmp_c1:
                st.image(processed_query, caption="Query Signature", width=200)
            with cmp_c2:
                if Path(top_tmpl["image_path"]).exists():
                    st.image(top_tmpl["image_path"], caption=f"Top Template ({top_cand['name']})", width=200)
    else:
        st.info("Upload a query document on the left to begin 1:N identification.")
