import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from config.settings import DECISION_MATCH
from src.preprocessing import preprocess_signature_image, load_image
from src.feature_extraction import extract_features, get_feature_dimensions
from src.database import register_user_signature, get_all_registered_templates, delete_user
from src.pdf_handler import get_pdf_page_count, render_pdf_page_to_image, crop_roi

st.set_page_config(page_title="Register Signature | One-Shot Recognition", page_icon="✍️", layout="wide")

st.title("➕ Register Signature Template (One-Shot Enrollment)")
st.markdown("""
Enroll a user into the offline signature recognition system.
Under the **One-Shot Registration** constraint, exactly **one genuine signature template** is registered per user.
""")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. User Information & Document Input")
    user_name = st.text_input("Full Name / User Identity", placeholder="e.g. John Doe, Prof. Smith")
    overwrite_existing = st.checkbox("Overwrite existing template if user already registered?", value=False)
    
    input_file = st.file_uploader(
        "Upload Signature Document",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload a scanned signature image or PDF document"
    )
    
    extracted_image = None
    
    if input_file is not None:
        file_bytes = input_file.read()
        file_ext = Path(input_file.name).suffix.lower()
        
        if file_ext == ".pdf":
            st.info("📄 PDF Document detected. PyMuPDF rendered the document. Please select page and crop signature region.")
            num_pages = get_pdf_page_count(file_bytes)
            page_idx = st.number_input("Select PDF Page", min_value=0, max_value=max(0, num_pages - 1), value=0, step=1)
            
            page_img = render_pdf_page_to_image(file_bytes, page_number=page_idx, dpi=150)
            
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
            st.image(cropped_roi, caption="Selected Signature ROI", use_container_width=True)
        else:
            extracted_image = load_image(file_bytes)

with col_right:
    st.subheader("2. Preprocessing & Feature Extraction")
    
    if extracted_image is not None and user_name.strip():
        # Run preprocessing pipeline with debug visualizer
        processed_img, debug = preprocess_signature_image(extracted_image, return_debug=True)
        
        # Display side-by-side inspection
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        with sub_c1:
            st.image(debug["original"], caption="Original Input", use_container_width=True)
        with sub_c2:
            st.image(debug["binary"], caption="Otsu Binary (Ink=255)", use_container_width=True)
        with sub_c3:
            st.image(debug["processed"], caption="128x128 Normalized Template", use_container_width=True)
            
        # Extract features
        features = extract_features(processed_img)
        dims = get_feature_dimensions()
        
        st.success(f"Feature vector extracted successfully! Total dimensions: **{len(features):,}**")
        st.markdown(f"""
        - **HOG local strokes**: {dims['hog']:,} descriptors
        - **Hu invariant moments**: {dims['hu']} shape descriptors
        - **Geometric features**: {dims['geometry']} structural metrics
        """)
        
        if st.button("💾 Complete Registration", type="primary", use_container_width=True):
            res = register_user_signature(
                name=user_name.strip(),
                processed_image_bytes_or_arr=processed_img,
                feature_vector=features,
                overwrite=overwrite_existing
            )
            
            if res["success"]:
                st.balloons()
                st.success(f"User '{user_name}' registered successfully with 1 template!")
                st.info(f"Image saved to: `{res['image_path']}`")
            else:
                st.error(res["error"])
    elif extracted_image is not None:
        st.warning("⚠️ Please provide a user name above to complete registration.")
    else:
        st.info("Upload a document on the left to preview preprocessing stages.")

# Section: Currently Registered Users
st.divider()
st.subheader("👥 Currently Registered Users")
registered = get_all_registered_templates()

if registered:
    st.write(f"Total enrolled templates: **{len(registered)}** (recommended: 3–10 users)")
    cols = st.columns(min(len(registered), 4))
    
    for idx, reg in enumerate(registered):
        col = cols[idx % len(cols)]
        with col:
            st.markdown(f"**{reg['name']}**")
            if Path(reg["image_path"]).exists():
                st.image(reg["image_path"], caption=f"ID: {reg['user_id']}", width=160)
            st.caption(f"Enrolled: {reg['created_at']}")
            if st.button(f"🗑️ Delete", key=f"del_{reg['user_id']}"):
                delete_user(reg["user_id"])
                st.rerun()
else:
    st.info("No registered users found in the database. Use the form above to register templates.")
