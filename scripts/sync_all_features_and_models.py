import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import joblib
from src.database import get_db_connection, get_all_registered_templates
from src.preprocessing import preprocess_signature_image
from src.feature_extraction import extract_features
from src.similarity import compute_pair_features
from src.dataset_loader import get_cedar_metadata, get_writer_splits, create_pairwise_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from src.model_training import save_model

print("1. Updating Registered Users in SQLite Database & Template Files...")
templates = get_all_registered_templates()
for t in templates:
    img_p = t["image_path"]
    feat_p = t.get("feature_path")
    if img_p and Path(img_p).exists() and feat_p:
        proc = preprocess_signature_image(img_p)
        new_feat = extract_features(proc)
        np.save(feat_p, new_feat)
        print(f"  Updated template features for {t['name']} (shape: {new_feat.shape}, norm: {np.linalg.norm(new_feat):.4f})")

print("\n2. Updating cedar_feature_cache.npy combined features...")
cache_p = BASE_DIR / "data" / "processed" / "cedar_feature_cache.npy"
if cache_p.exists():
    cache = np.load(cache_p, allow_pickle=True).item()
    w_h = np.sqrt(0.75)
    w_u = np.sqrt(0.15)
    w_g = np.sqrt(0.10)
    for k in cache:
        h = cache[k]["hog"]
        u = cache[k]["hu"]
        g = cache[k]["geom"]
        cache[k]["combined"] = np.concatenate([w_h * h, w_u * u, w_g * g], axis=0).astype(np.float32)
    np.save(cache_p, cache)
    print(f"  Successfully updated {len(cache)} cached items in {cache_p}")

print("\n3. Retraining Calibrated Linear SVM on updated weighted features...")
df = get_cedar_metadata()
train_wids, val_wids, test_wids = get_writer_splits(df, train_count=35, val_count=10, test_count=10, random_seed=42)
combined_cache = {p: cache[p]["combined"] for p in cache}

X_train, y_train = create_pairwise_dataset(df, combined_cache, train_wids, pairs_per_writer=35, random_seed=42)
print(f"  Train pairs shape: {X_train.shape}")

pipe = Pipeline([
    ("scaler", StandardScaler(copy=False)),
    ("svm", CalibratedClassifierCV(LinearSVC(dual=False, C=1.0, max_iter=2500, random_state=42), cv=3))
])
pipe.fit(X_train, y_train)

save_model(pipe, "linear_svm_pipeline.joblib")
save_model(pipe, "best_model.joblib")
print("  Models successfully trained and saved to data/models/!")

print("\n=== SANITY TEST: ABI vs ALICE VANCE vs ALICE GENUINE ===")
from src.verification import verify_signature

alice_feat = cache[df[(df['writer_id'] == 1) & (df['is_forgery'] == 0)]['filepath'].iloc[0]]["combined"]
alice_gen2_feat = cache[df[(df['writer_id'] == 1) & (df['is_forgery'] == 0)]['filepath'].iloc[1]]["combined"]

# Get Abi's updated features
t_abi = [t for t in get_all_registered_templates() if t['name'] == 'abi'][0]
abi_feat = t_abi["features"]

# 1. Alice vs Alice (Same Person)
res_same_cos = verify_signature(alice_gen2_feat, alice_feat, claimed_user_name="Alice Vance", method="cosine", match_threshold=0.68)
res_same_svm = verify_signature(alice_gen2_feat, alice_feat, claimed_user_name="Alice Vance", method="svm", svm_model=pipe, match_threshold=0.50)
print(f"Same Person (Alice vs Alice):")
print(f"  Cosine: {res_same_cos['decision']} (score: {res_same_cos['similarity_score']:.4f}) [Expected: Match]")
print(f"  SVM:    {res_same_svm['decision']} (score: {res_same_svm['similarity_score']:.4f}) [Expected: Match]")

# 2. Abi vs Alice (Different Person)
res_diff_cos = verify_signature(abi_feat, alice_feat, claimed_user_name="Alice Vance", method="cosine", match_threshold=0.68)
res_diff_svm = verify_signature(abi_feat, alice_feat, claimed_user_name="Alice Vance", method="svm", svm_model=pipe, match_threshold=0.50)
print(f"\nDifferent Person (Abi vs Alice):")
print(f"  Cosine: {res_diff_cos['decision']} (score: {res_diff_cos['similarity_score']:.4f}) [Expected: Not Match]")
print(f"  SVM:    {res_diff_svm['decision']} (score: {res_diff_svm['similarity_score']:.4f}) [Expected: Not Match]")
