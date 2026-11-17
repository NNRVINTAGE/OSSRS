import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
from src.database import get_all_registered_templates
from src.verification import verify_signature

model = joblib.load("data/models/linear_svm_pipeline.joblib")
users = get_all_registered_templates()
print("Registered users:", [u["name"] for u in users])

if len(users) >= 2:
    u1 = users[0]
    u2 = users[1]
    res_diff = verify_signature(u1["features"], u2["features"], claimed_user_name=u2["name"], method="svm", svm_model=model)
    print(f"\n[TEST 1] SVM Verification: {u1['name']} vs {u2['name']} (Different Persons)")
    print("  Decision:", res_diff["decision"])
    print("  Similarity Score:", f"{res_diff['similarity_score']:.4f}")

    res_same = verify_signature(u1["features"], u1["features"], claimed_user_name=u1["name"], method="svm", svm_model=model)
    print(f"\n[TEST 2] SVM Verification: {u1['name']} vs {u1['name']} (Same Person)")
    print("  Decision:", res_same["decision"])
    print("  Similarity Score:", f"{res_same['similarity_score']:.4f}")
