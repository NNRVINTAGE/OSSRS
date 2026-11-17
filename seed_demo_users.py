"""
Seeds the local SQLite database with 5 genuine user templates from the CEDAR dataset
for immediate demonstration of 1:N Identification and 1:1 Verification.
"""
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.database import init_db, register_user_signature, get_all_registered_templates
from src.preprocessing import preprocess_signature_image
from src.feature_extraction import extract_features

def seed(force: bool = False):
    init_db()
    existing = get_all_registered_templates()
    if len(existing) >= 3 and not force:
        print(f"Database already contains {len(existing)} enrolled users. Use force=True to re-seed.")
        return

    # Sample 5 distinct writers from the dataset
    demo_writers = [
        ("Alice Vance", "Dataset/full_org/original_1_1.png"),
        ("Bob Chen", "Dataset/full_org/original_2_1.png"),
        ("Catherine Ross", "Dataset/full_org/original_3_1.png"),
        ("David Miller", "Dataset/full_org/original_4_1.png"),
        ("Emma Watson", "Dataset/full_org/original_5_1.png"),
    ]

    print("Enrolling 5 demonstration users with 1 template each...")
    for name, img_rel_path in demo_writers:
        img_path = BASE_DIR / img_rel_path
        if not img_path.exists():
            print(f"Warning: {img_path} not found.")
            continue
            
        processed = preprocess_signature_image(img_path)
        feat = extract_features(processed)
        res = register_user_signature(
            name=name,
            processed_image_bytes_or_arr=processed,
            feature_vector=feat,
            overwrite=True
        )
        print(f"  Enrolled '{name}' (User ID: {res['user_id']})")

    print("Seed completed successfully!")

if __name__ == "__main__":
    seed()
