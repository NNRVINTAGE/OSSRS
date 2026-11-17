import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DATASET_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.dataset_loader import get_cedar_metadata, get_writer_splits, create_pairwise_dataset
from src.preprocessing import preprocess_signature_image
from src.feature_extraction import extract_features
from src.similarity import cosine_similarity, compute_pair_features
from src.model_training import train_svm_pipeline, evaluate_pair_classifier, save_model


def main():
    print("=" * 60)
    print("RE-EXTRACTING CEDAR FEATURES WITH CURRENT PIPELINE & RETRAINING SVM")
    print("=" * 60)

    df = get_cedar_metadata()
    print(f"Total CEDAR images: {len(df)}")
    
    train_wids, val_wids, test_wids = get_writer_splits(df, train_count=35, val_count=10, test_count=10, random_seed=42)
    print(f"Train writers: {len(train_wids)}, Val writers: {len(val_wids)}, Test writers: {len(test_wids)}")

    # 1. Extract features for all images using the CURRENT preprocessing & extraction
    cache_path = PROCESSED_DATA_DIR / "cedar_feature_cache.npy"
    print(f"\nExtracting normalized features for all 2,640 images...")
    t0 = time.time()
    
    cache = {}
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        p = row["filepath"]
        try:
            proc = preprocess_signature_image(p)
            combined_f = extract_features(proc, include_hog=True, include_hu=True, include_geometry=True)
            hog_f = extract_features(proc, include_hog=True, include_hu=False, include_geometry=False)
            hu_f = extract_features(proc, include_hog=False, include_hu=True, include_geometry=False)
            geom_f = extract_features(proc, include_hog=False, include_hu=False, include_geometry=True)
            hog_hu = np.concatenate([hog_f, hu_f], axis=0).astype(np.float32)
            
            cache[p] = {
                "hog": hog_f,
                "hu": hu_f,
                "geom": geom_f,
                "hog_hu": hog_hu,
                "combined": combined_f
            }
        except Exception as e:
            print(f"Error on {p}: {e}")

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] extracted ({(i+1)/total*100:.1f}%) in {time.time()-t0:.1f}s")

    print(f"All features extracted in {time.time()-t0:.2f}s. Saving updated cache to {cache_path}...")
    np.save(cache_path, cache)

    # 2. Build pairwise dataset on Train Writers
    print("\nBuilding pairwise dataset from 35 train writers...")
    combined_cache = {p: cache[p]["combined"] for p in cache}
    X_train_pairs, y_train_pairs = create_pairwise_dataset(
        df, combined_cache, train_wids, pairs_per_writer=40, random_seed=42, augment_positive=False
    )
    print(f"Train pairs shape: {X_train_pairs.shape}, Labels balance: {np.bincount(y_train_pairs)}")

    # 3. Train Linear SVM Pipeline
    print("\nTraining Linear SVM Pipeline (StandardScaler + SVC(linear, probability=True))...")
    lin_pipeline, lin_time = train_svm_pipeline(
        X_train_pairs, y_train_pairs, kernel="linear", C=1.0, max_iter=3000, random_state=42
    )
    print(f"Linear SVM trained in {lin_time:.2f}s")

    # 4. Train RBF SVM Pipeline
    print("\nTraining RBF SVM Pipeline (StandardScaler + SVC(rbf, probability=True))...")
    rbf_pipeline, rbf_time = train_svm_pipeline(
        X_train_pairs, y_train_pairs, kernel="rbf", C=1.0, max_iter=3000, random_state=42
    )
    print(f"RBF SVM trained in {rbf_time:.2f}s")

    # 5. Evaluate on Validation Writers
    print("\nEvaluating on Validation Writers...")
    X_val_pairs, y_val_pairs = create_pairwise_dataset(
        df, combined_cache, val_wids, pairs_per_writer=30, random_seed=42, augment_positive=False
    )
    
    val_lin_eval = evaluate_pair_classifier(lin_pipeline, X_val_pairs, y_val_pairs, threshold=0.5)
    print(f"Linear SVM Validation: Acc={val_lin_eval['accuracy']:.4f}, EER={val_lin_eval['eer']:.4f}, EER_Thresh={val_lin_eval['eer_threshold']:.4f}, AUC={val_lin_eval['roc_auc']:.4f}")

    val_rbf_eval = evaluate_pair_classifier(rbf_pipeline, X_val_pairs, y_val_pairs, threshold=0.5)
    print(f"RBF SVM Validation: Acc={val_rbf_eval['accuracy']:.4f}, EER={val_rbf_eval['eer']:.4f}, EER_Thresh={val_rbf_eval['eer_threshold']:.4f}, AUC={val_rbf_eval['roc_auc']:.4f}")

    # 6. Evaluate on Test Writers (Unseen writers)
    print("\nEvaluating on Unseen Test Writers...")
    X_test_pairs, y_test_pairs = create_pairwise_dataset(
        df, combined_cache, test_wids, pairs_per_writer=30, random_seed=42, augment_positive=False
    )
    test_lin_eval = evaluate_pair_classifier(lin_pipeline, X_test_pairs, y_test_pairs, threshold=0.5)
    print(f"Linear SVM Test: Acc={test_lin_eval['accuracy']:.4f}, EER={test_lin_eval['eer']:.4f}, AUC={test_lin_eval['roc_auc']:.4f}")

    test_rbf_eval = evaluate_pair_classifier(rbf_pipeline, X_test_pairs, y_test_pairs, threshold=0.5)
    print(f"RBF SVM Test: Acc={test_rbf_eval['accuracy']:.4f}, EER={test_rbf_eval['eer']:.4f}, AUC={test_rbf_eval['roc_auc']:.4f}")

    # 7. Save models
    save_model(lin_pipeline, "linear_svm_pipeline.joblib")
    save_model(rbf_pipeline, "rbf_svm_pipeline.joblib")
    best_pipeline = lin_pipeline if val_lin_eval['accuracy'] >= val_rbf_eval['accuracy'] else rbf_pipeline
    save_model(best_pipeline, "best_model.joblib")
    print(f"\nModels successfully saved to {MODELS_DIR}!")

    # 8. Test specifically on cross-writer and user signatures
    print("\n--- SANITY CHECK TESTS ---")
    p1 = df[(df["writer_id"] == 1) & (df["is_forgery"] == 0)]["filepath"].iloc[0]
    p1_same = df[(df["writer_id"] == 1) & (df["is_forgery"] == 0)]["filepath"].iloc[1]
    p2 = df[(df["writer_id"] == 2) & (df["is_forgery"] == 0)]["filepath"].iloc[0]
    p1_forg = df[(df["writer_id"] == 1) & (df["is_forgery"] == 1)]["filepath"].iloc[0]

    f1 = combined_cache[p1]
    f1_same = combined_cache[p1_same]
    f2 = combined_cache[p2]
    f1_forg = combined_cache[p1_forg]

    for name, model in [("Linear SVM", lin_pipeline), ("RBF SVM", rbf_pipeline)]:
        pair_same = compute_pair_features(f1, f1_same)
        pair_diff = compute_pair_features(f1, f2)
        pair_forg = compute_pair_features(f1, f1_forg)

        prob_same = model.predict_proba([pair_same])[0][1]
        prob_diff = model.predict_proba([pair_diff])[0][1]
        prob_forg = model.predict_proba([pair_forg])[0][1]

        print(f"\n{name} Sanity Check:")
        print(f"  Same Writer (w1 vs w1): prob = {prob_same:.4f} (Expected > 0.50)")
        print(f"  Different Writer (w1 vs w2): prob = {prob_diff:.4f} (Expected < 0.30)")
        print(f"  Skilled Forgery (w1 vs forg1): prob = {prob_forg:.4f} (Expected < 0.50)")


if __name__ == "__main__":
    main()
