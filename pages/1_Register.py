import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from config.settings import DECISION_MATCH
from src.preprocessing import preprocess_signature_image, load_image
from src.feature_extraction import extract_features, get_feature_dimensions
from src.database import (
    register_user_signature,
    get_all_registered_templates,
    delete_user,
)
from src.pdf_handler import get_pdf_page_count, render_pdf_page_to_image, crop_roi
from src.ui_theme import (
    apply_cyberpunk_theme,
    render_hud_header,
    render_section_header,
)

st.set_page_config(
    page_title="Register Signature | One-Shot Recognition",
    page_icon="✍️",
    layout="wide",
)

apply_cyberpunk_theme("ENROLL_NODE // 01")

render_hud_header(
    title="➕ Register Signature Template (One-Shot Enrollment)",
    subtitle="Enroll a user into the offline signature recognition system. Under the One-Shot Registration constraint, exactly one genuine signature template is registered per user.",
    sys_code="MOD.ENROLL // 01",
    status_tag="1-TEMPLATE BIOMETRIC LOCK",
)

col_left, col_right = st.columns([1, 1])

with col_left:
    render_section_header("INPUT.DOC // 01", "1. User Information & Document Input")
    user_name = st.text_input(
        "Full Name / User Identity", placeholder="e.g. John Doe, Prof. Smith"
    )
    overwrite_existing = st.checkbox(
        "Overwrite existing template if user already registered?", value=False
    )

    input_file = st.file_uploader(
        "Upload Signature Document",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload a scanned signature image or PDF document",
    )

    extracted_image = None

    if input_file is not None:
        file_bytes = input_file.read()
        file_ext = Path(input_file.name).suffix.lower()

        if file_ext == ".pdf":
            st.info(
                "📄 PDF Document detected. PyMuPDF rendered the document. Please select page and crop signature region."
            )
            num_pages = get_pdf_page_count(file_bytes)
            page_idx = st.number_input(
                "Select PDF Page",
                min_value=0,
                max_value=max(0, num_pages - 1),
                value=0,
                step=1,
            )

            page_img = render_pdf_page_to_image(
                file_bytes, page_number=page_idx, dpi=150
            )

            st.write("Adjust Region of Interest (ROI) sliders (% coordinates):")
            c1, c2 = st.columns(2)
            with c1:
                x_pct = st.slider("X Position (%)", 0, 90, 10, 1)
                w_pct = st.slider("Width (%)", 10, 100, 80, 1)
            with c2:
                y_pct = st.slider("Y Position (%)", 0, 90, 60, 1)
                h_pct = st.slider("Height (%)", 10, 100, 35, 1)

            cropped_roi = crop_roi(page_img, x_pct, y_pct, w_pct, h_pct)
            extracted_image = cropped_roi

            # Show preview of PDF page with bounding box
            st.image(
                cropped_roi,
                caption="Selected Signature ROI",
                width='stretch',
            )
        else:
            extracted_image = load_image(file_bytes)

with col_right:
    render_section_header("PIPELINE // 02", "2. Preprocessing & Feature Extraction")

    if extracted_image is not None and user_name.strip():
        # Run preprocessing pipeline with debug visualizer
        processed_img, debug = preprocess_signature_image(
            extracted_image, return_debug=True
        )

        # Display side-by-side inspection
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        with sub_c1:
            st.image(
                debug["original"],
                caption="Original Input",
                width='stretch',
            )
        with sub_c2:
            st.image(
                debug["binary"],
                caption="Otsu Binary (Ink=255)",
                width='stretch',
            )
        with sub_c3:
            st.image(
                debug["processed"],
                caption="128x128 Normalized Template",
                width='stretch',
            )

        # Extract features
        features = extract_features(processed_img)
        dims = get_feature_dimensions()

        st.markdown(
            f"""
            <div class="cp-panel" style="border-left-color:#00FF9D;">
                <div style="display:flex; justify-content:space-between; align-items:center; font-family:'Share Tech Mono',monospace; font-size:0.78rem; color:#00FF9D; margin-bottom:6px;">
                    <span>⚡ FEATURE VECTOR EXTRACTED</span>
                    <span>DIM: {len(features):,}</span>
                </div>
                <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; margin-top:8px;">
                    <div style="background:rgba(0,240,255,0.08); border:1px solid rgba(0,240,255,0.3); padding:8px;">
                        <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#7F9ABD;">HOG LOCAL STROKES</div>
                        <div style="font-family:'Orbitron',sans-serif; font-size:1.1rem; font-weight:700; color:#00F0FF;">{dims['hog']:,}</div>
                    </div>
                    <div style="background:rgba(252,238,10,0.08); border:1px solid rgba(252,238,10,0.3); padding:8px;">
                        <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#7F9ABD;">HU MOMENTS</div>
                        <div style="font-family:'Orbitron',sans-serif; font-size:1.1rem; font-weight:700; color:#FCEE0A;">{dims['hu']}</div>
                    </div>
                    <div style="background:rgba(255,0,85,0.08); border:1px solid rgba(255,0,85,0.3); padding:8px;">
                        <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#7F9ABD;">GEOMETRIC METRICS</div>
                        <div style="font-family:'Orbitron',sans-serif; font-size:1.1rem; font-weight:700; color:#FF0055;">{dims['geometry']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "💾 Complete Registration", type="primary", width='stretch'
        ):
            res = register_user_signature(
                name=user_name.strip(),
                processed_image_bytes_or_arr=processed_img,
                feature_vector=features,
                overwrite=overwrite_existing,
            )

            if res["success"]:
                st.balloons()
                st.success(
                    f"User '{user_name}' registered successfully with 1 template!"
                )
                st.info(f"Image saved to: `{res['image_path']}`")
            else:
                st.error(res["error"])
    elif extracted_image is not None:
        st.warning("⚠️ Please provide a user name above to complete registration.")
    else:
        st.info("Upload a document on the left to preview preprocessing stages.")

# Section: Currently Registered Users
st.divider()
render_section_header("BIO.VAULT // 03", "👥 Currently Registered Users")
registered = get_all_registered_templates()

if registered:
    st.markdown(
        f"""
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.86rem; color:#00F0FF; margin-bottom:14px;">
            ACTIVE BIOMETRIC VAULT // TOTAL ENROLLED TEMPLATES: <strong>{len(registered)}</strong> (RECOMMENDED: 3–10 USERS)
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(min(len(registered), 4))

    for idx, reg in enumerate(registered):
        col = cols[idx % len(cols)]
        with col:
            st.markdown(
                f"""
                <div class="cp-user-card">
                    <div style="display:flex; justify-content:space-between; font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#00F0FF;">
                        <span>ID #{reg['user_id']:02d}</span>
                        <span style="color:#00FF9D;">● ENROLLED</span>
                    </div>
                    <div style="font-family:'Orbitron',sans-serif; font-size:0.98rem; font-weight:700; color:#FFFFFF; margin:4px 0;">
                        {reg['name']}
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#7F9ABD;">
                        TIMESTAMP: {reg['created_at']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if Path(reg["image_path"]).exists():
                st.image(
                    reg["image_path"],
                    caption=f"ID: {reg['user_id']} // {reg['name']}",
                    width=160,
                )
            if st.button(f"🗑️ Delete", key=f"del_{reg['user_id']}", width='stretch'):
                delete_user(reg["user_id"])
                st.rerun()
else:
    st.info(
        "No registered users found in the database. Use the form above to register templates."
    )
