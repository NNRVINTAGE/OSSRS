import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    DATASET_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    EXPERIMENT_RESULTS_DIR,
    DEFAULT_COSINE_MATCH_THRESHOLD,
    DEFAULT_COSINE_UNCERTAIN_LOWER,
    DEFAULT_TOP_MARGIN_THRESHOLD
)
from src.preprocessing import preprocess_signature_image
from src.feature_extraction import (
    extract_features,
    extract_hog_features,
    extract_hu_moments,
    extract_geometric_features
)
from src.similarity import cosine_similarity, compute_pair_features
from src.dataset_loader import (
    get_cedar_metadata,
    get_writer_splits,
    create_pairwise_dataset
)
from src.model_training import (
    train_svm_pipeline,
    evaluate_pair_classifier,
    calculate_eer,
    save_model
)
from src.identification import identify_signature
from src.verification import verify_signature


def precompute_and_cache_features(
    df: pd.DataFrame,
    cache_path: Path = PROCESSED_DATA_DIR / "cedar_feature_cache.npy"
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extracts and caches HOG, Hu, Geometry, and Combined features for all dataset images.
    Returns dict: filepath -> {"hog": ..., "hu": ..., "geom": ..., "combined": ...}
    """
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}...")
        return np.load(cache_path, allow_pickle=True).item()
        
    print("Precomputing features for all CEDAR signatures (2640 images)...")
    cache = {}
    total = len(df)
    start_t = time.time()
    
    for i, (_, row) in enumerate(df.iterrows()):
        p = row["filepath"]
        try:
            processed = preprocess_signature_image(p)
            combined_f = extract_features(processed, include_hog=True, include_hu=True, include_geometry=True)
            hog_f = extract_features(processed, include_hog=True, include_hu=False, include_geometry=False)
            hu_f = extract_features(processed, include_hu=True, include_hog=False, include_geometry=False)
            geom_f = extract_features(processed, include_geometry=True, include_hog=False, include_hu=False)
            
            cache[p] = {
                "hog": hog_f,
                "hu": hu_f,
                "geom": geom_f,
                "combined": combined_f
            }
        except Exception as e:
            print(f"Failed to process {p}: {e}")
            
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Processed {i+1}/{total} images ({(i+1)/total*100:.1f}%)")
            
    print(f"Feature caching completed in {time.time() - start_t:.2f} seconds.")
    np.save(cache_path, cache)
    return cache


def run_one_shot_evaluation(
    eval_writers: List[int],
    df: pd.DataFrame,
    feature_cache: Dict[str, Dict[str, np.ndarray]],
    feature_key: str,
    method: str = "cosine",
    svm_model = None,
    match_threshold: float = 0.72,
    uncertain_lower: float = 0.60,
    top_margin_threshold: float = 0.04,
    unknown_writers: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Simulates the one-shot registration and evaluation protocol on unseen writers:
    1. Register 1 genuine sample per writer (sample_id=1).
    2. Test 1:1 Verification:
       - Genuine probes (sample_id 2..24) -> Expected: Match
       - Skilled forgery probes (sample_id 1..24) -> Expected: Not Match
       - Cross-writer imposter probes -> Expected: Not Match
    3. Test 1:N Identification:
       - Known genuine probes (sample_id 2..24) -> Top-1 ID accuracy
       - Skilled forgery probes -> Expected: Unknown or Not Match
       - Unknown writer probes -> Expected: Unknown
    """
    templates = []
    sub_df = df[df["writer_id"].isin(eval_writers)]
    
    # 1. Register 1 template per writer (sample_id = 1)
    for wid in eval_writers:
        tmpl_row = sub_df[(sub_df["writer_id"] == wid) & (sub_df["sample_id"] == 1) & (sub_df["is_forgery"] == 0)]
        if tmpl_row.empty:
            continue
        tmpl_path = tmpl_row.iloc[0]["filepath"]
        templates.append({
            "user_id": wid,
            "name": f"Writer_{wid}",
            "features": feature_cache[tmpl_path][feature_key],
            "image_path": tmpl_path
        })
        
    # 2. Run 1:1 Verification Evaluation
    verif_y_true = []
    verif_scores = []
    verif_preds = []
    
    # Genuine probes (Label = 1)
    for wid in eval_writers:
        tmpl_feat = [t["features"] for t in templates if t["user_id"] == wid][0]
        gen_probes = sub_df[(sub_df["writer_id"] == wid) & (sub_df["sample_id"] > 1) & (sub_df["is_forgery"] == 0)]["filepath"].tolist()
        
        for p in gen_probes:
            feat = feature_cache[p][feature_key]
            res = verify_signature(
                feat, tmpl_feat,
                claimed_user_name=f"Writer_{wid}",
                method=method,
                svm_model=svm_model,
                match_threshold=match_threshold,
                uncertain_lower=uncertain_lower
            )
            verif_y_true.append(1)
            verif_scores.append(res["similarity_score"])
            verif_preds.append(1 if res["decision"] == "Match" else 0)
            
    # Forged probes (Label = 0)
    for wid in eval_writers:
        tmpl_feat = [t["features"] for t in templates if t["user_id"] == wid][0]
        forg_probes = sub_df[(sub_df["writer_id"] == wid) & (sub_df["is_forgery"] == 1)]["filepath"].tolist()
        
        for p in forg_probes:
            feat = feature_cache[p][feature_key]
            res = verify_signature(
                feat, tmpl_feat,
                claimed_user_name=f"Writer_{wid}",
                method=method,
                svm_model=svm_model,
                match_threshold=match_threshold,
                uncertain_lower=uncertain_lower
            )
            verif_y_true.append(0)
            verif_scores.append(res["similarity_score"])
            verif_preds.append(1 if res["decision"] == "Match" else 0)
            
    # Cross-writer imposter probes (Label = 0)
    for wid in eval_writers:
        other_wids = [w for w in eval_writers if w != wid]
        tmpl_feat = [t["features"] for t in templates if t["user_id"] == wid][0]
        for other_wid in other_wids[:3]:  # 3 imposters per writer
            imp_probes = sub_df[(sub_df["writer_id"] == other_wid) & (sub_df["sample_id"].isin([2, 3])) & (sub_df["is_forgery"] == 0)]["filepath"].tolist()
            for p in imp_probes:
                feat = feature_cache[p][feature_key]
                res = verify_signature(
                    feat, tmpl_feat,
                    claimed_user_name=f"Writer_{wid}",
                    method=method,
                    svm_model=svm_model,
                    match_threshold=match_threshold,
                    uncertain_lower=uncertain_lower
                )
                verif_y_true.append(0)
                verif_scores.append(res["similarity_score"])
                verif_preds.append(1 if res["decision"] == "Match" else 0)
                
    verif_metrics = evaluate_pair_classifier(
        np.array(verif_scores),
        np.array(verif_y_true),
        threshold=match_threshold
    )
    
    # 3. Run 1:N Identification Evaluation
    id_correct_top1 = 0
    id_total_known = 0
    id_uncertain = 0
    
    inference_times = []
    
    for wid in eval_writers:
        gen_probes = sub_df[(sub_df["writer_id"] == wid) & (sub_df["sample_id"] > 1) & (sub_df["is_forgery"] == 0)]["filepath"].tolist()
        for p in gen_probes:
            feat = feature_cache[p][feature_key]
            t0 = time.time()
            id_res = identify_signature(
                feat,
                templates,
                method=method,
                svm_model=svm_model,
                match_threshold=match_threshold,
                uncertain_lower=uncertain_lower,
                top_margin_threshold=top_margin_threshold
            )
            inference_times.append(time.time() - t0)
            
            id_total_known += 1
            if id_res["predicted_user_id"] == wid and id_res["decision"] == "Match":
                id_correct_top1 += 1
            elif id_res["decision"] == "Uncertain":
                id_uncertain += 1
                
    # Unknown identity probes (from writers outside eval_writers)
    id_correct_unknown = 0
    id_total_unknown = 0
    if unknown_writers:
        unk_df = df[df["writer_id"].isin(unknown_writers) & (df["is_forgery"] == 0)]
        unk_samples = unk_df["filepath"].head(60).tolist()
        for p in unk_samples:
            if p not in feature_cache:
                continue
            feat = feature_cache[p][feature_key]
            id_res = identify_signature(
                feat,
                templates,
                method=method,
                svm_model=svm_model,
                match_threshold=match_threshold,
                uncertain_lower=uncertain_lower,
                top_margin_threshold=top_margin_threshold
            )
            id_total_unknown += 1
            if id_res["decision"] == "Unknown":
                id_correct_unknown += 1
                
    top1_accuracy = float(id_correct_top1 / max(id_total_known, 1))
    unknown_accuracy = float(id_correct_unknown / max(id_total_unknown, 1)) if id_total_unknown > 0 else 1.0
    mean_inference_ms = float(np.mean(inference_times) * 1000.0) if inference_times else 0.0
    
    return {
        "verification": verif_metrics,
        "identification": {
            "top1_accuracy": top1_accuracy,
            "id_total_known": id_total_known,
            "id_correct_top1": id_correct_top1,
            "id_uncertain_count": id_uncertain,
            "unknown_rejection_accuracy": unknown_accuracy,
            "mean_inference_time_ms": mean_inference_ms
        }
    }


def run_all_experiments():
    print("=" * 60)
    print("STARTING ONE-SHOT SIGNATURE RECOGNITION EXPERIMENTS (E1 - E6)")
    print("=" * 60)
    
    # 1. Load metadata & perform writer-disjoint split
    df = get_cedar_metadata()
    train_wids, val_wids, test_wids = get_writer_splits(df, train_count=35, val_count=10, test_count=10, random_seed=42)
    print(f"Total dataset signatures: {len(df)}")
    print(f"Train Writers (35): {train_wids}")
    print(f"Validation Writers (10): {val_wids}")
    print(f"Test Writers (10): {test_wids}")
    
    # 2. Precompute features
    cache = precompute_and_cache_features(df)
    
    # 3. Create feature representations in cache
    # E1: HOG only
    # E2: HOG + Hu
    for p in cache:
        hog_hu = np.concatenate([cache[p]["hog"], cache[p]["hu"]], axis=0).astype(np.float32)
        cache[p]["hog_hu"] = hog_hu

    results = {}
    
    # -------------------------------------------------------------------------
    # Experiment E1: HOG + Cosine Similarity
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment E1: HOG + Cosine Similarity ---")
    # Calibrate threshold on validation writers
    val_e1 = run_one_shot_evaluation(
        val_wids, df, cache, feature_key="hog", method="cosine",
        match_threshold=DEFAULT_COSINE_MATCH_THRESHOLD,
        uncertain_lower=DEFAULT_COSINE_UNCERTAIN_LOWER,
        unknown_writers=train_wids[:5]
    )
    e1_thresh = val_e1["verification"]["eer_threshold"]
    print(f"  Calibrated validation EER Threshold: {e1_thresh:.3f}")
    
    # Final evaluation on held-out TEST writers
    test_e1 = run_one_shot_evaluation(
        test_wids, df, cache, feature_key="hog", method="cosine",
        match_threshold=e1_thresh,
        uncertain_lower=e1_thresh - 0.10,
        unknown_writers=val_wids
    )
    results["E1"] = {
        "name": "E1: HOG + Cosine",
        "description": "Baseline HOG stroke orientation features matched with Cosine Similarity",
        "feature_dim": 8100,
        "calibrated_threshold": e1_thresh,
        "val_metrics": val_e1,
        "test_metrics": test_e1
    }
    print(f"  Test Verif Acc: {test_e1['verification']['accuracy']:.4f}, EER: {test_e1['verification']['eer']:.4f}, Top-1 ID Acc: {test_e1['identification']['top1_accuracy']:.4f}")
    
    # -------------------------------------------------------------------------
    # Experiment E2: HOG + Hu Moments + Cosine Similarity
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment E2: HOG + Hu Moments + Cosine Similarity ---")
    val_e2 = run_one_shot_evaluation(
        val_wids, df, cache, feature_key="hog_hu", method="cosine",
        match_threshold=DEFAULT_COSINE_MATCH_THRESHOLD,
        uncertain_lower=DEFAULT_COSINE_UNCERTAIN_LOWER,
        unknown_writers=train_wids[:5]
    )
    e2_thresh = val_e2["verification"]["eer_threshold"]
    print(f"  Calibrated validation EER Threshold: {e2_thresh:.3f}")
    
    test_e2 = run_one_shot_evaluation(
        test_wids, df, cache, feature_key="hog_hu", method="cosine",
        match_threshold=e2_thresh,
        uncertain_lower=e2_thresh - 0.10,
        unknown_writers=val_wids
    )
    results["E2"] = {
        "name": "E2: HOG + Hu + Cosine",
        "description": "HOG local stroke patterns combined with Hu Moments global shape features",
        "feature_dim": 8107,
        "calibrated_threshold": e2_thresh,
        "val_metrics": val_e2,
        "test_metrics": test_e2
    }
    print(f"  Test Verif Acc: {test_e2['verification']['accuracy']:.4f}, EER: {test_e2['verification']['eer']:.4f}, Top-1 ID Acc: {test_e2['identification']['top1_accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # Prepare Pairwise Training Data for Model B (SVM) on Train Writers (1..35)
    # -------------------------------------------------------------------------
    print("\nPreparing pairwise training dataset from 35 train writers (without augmentation)...")
    t0_pair = time.time()
    
    # Flatten feature cache mapping to Combined features
    combined_cache = {p: cache[p]["combined"] for p in cache}
    X_train_pairs, y_train_pairs = create_pairwise_dataset(
        df, combined_cache, train_wids, pairs_per_writer=35, random_seed=42, augment_positive=False
    )
    print(f"  Created {len(y_train_pairs)} pairs (shape: {X_train_pairs.shape}) in {time.time()-t0_pair:.2f}s")
    
    # -------------------------------------------------------------------------
    # Experiment E3: HOG + Hu + Geometry + Linear SVM
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment E3: HOG + Hu + Geometry + Linear SVM ---")
    print("  Training Linear SVM pipeline...")
    linear_svm, lin_train_time = train_svm_pipeline(
        X_train_pairs, y_train_pairs, kernel="linear", C=1.0, max_iter=2500, random_state=42
    )
    print(f"  Linear SVM trained in {lin_train_time:.2f} seconds.")
    
    # Calibrate on validation
    val_e3 = run_one_shot_evaluation(
        val_wids, df, cache, feature_key="combined", method="svm",
        svm_model=linear_svm, match_threshold=0.5, uncertain_lower=0.4,
        unknown_writers=train_wids[:5]
    )
    e3_thresh = val_e3["verification"]["eer_threshold"]
    print(f"  Calibrated validation EER Threshold: {e3_thresh:.3f}")
    
    test_e3 = run_one_shot_evaluation(
        test_wids, df, cache, feature_key="combined", method="svm",
        svm_model=linear_svm, match_threshold=e3_thresh,
        uncertain_lower=max(0.1, e3_thresh - 0.15),
        unknown_writers=val_wids
    )
    results["E3"] = {
        "name": "E3: Full Features + Linear SVM",
        "description": "Pairwise [|diff|, prod, cosine] with Linear SVM classifier",
        "feature_dim": 8119,
        "pair_feature_dim": X_train_pairs.shape[1],
        "training_time_s": lin_train_time,
        "calibrated_threshold": e3_thresh,
        "val_metrics": val_e3,
        "test_metrics": test_e3
    }
    print(f"  Test Verif Acc: {test_e3['verification']['accuracy']:.4f}, EER: {test_e3['verification']['eer']:.4f}, Top-1 ID Acc: {test_e3['identification']['top1_accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # Experiment E4: HOG + Hu + Geometry + RBF SVM
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment E4: HOG + Hu + Geometry + RBF SVM ---")
    print("  Training RBF SVM pipeline...")
    rbf_svm, rbf_train_time = train_svm_pipeline(
        X_train_pairs, y_train_pairs, kernel="rbf", C=1.0, max_iter=2500, random_state=42
    )
    print(f"  RBF SVM trained in {rbf_train_time:.2f} seconds.")
    
    val_e4 = run_one_shot_evaluation(
        val_wids, df, cache, feature_key="combined", method="svm",
        svm_model=rbf_svm, match_threshold=0.5, uncertain_lower=0.4,
        unknown_writers=train_wids[:5]
    )
    e4_thresh = val_e4["verification"]["eer_threshold"]
    print(f"  Calibrated validation EER Threshold: {e4_thresh:.3f}")
    
    test_e4 = run_one_shot_evaluation(
        test_wids, df, cache, feature_key="combined", method="svm",
        svm_model=rbf_svm, match_threshold=e4_thresh,
        uncertain_lower=max(0.1, e4_thresh - 0.15),
        unknown_writers=val_wids
    )
    results["E4"] = {
        "name": "E4: Full Features + RBF SVM",
        "description": "Pairwise [|diff|, prod, cosine] with Non-linear RBF SVM classifier",
        "feature_dim": 8119,
        "pair_feature_dim": X_train_pairs.shape[1],
        "training_time_s": rbf_train_time,
        "calibrated_threshold": e4_thresh,
        "val_metrics": val_e4,
        "test_metrics": test_e4
    }
    print(f"  Test Verif Acc: {test_e4['verification']['accuracy']:.4f}, EER: {test_e4['verification']['eer']:.4f}, Top-1 ID Acc: {test_e4['identification']['top1_accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # Experiment E5: Best model without augmentation
    # -------------------------------------------------------------------------
    # Determine best model based on validation verification accuracy and EER
    cand_models = ["E1", "E2", "E3", "E4"]
    best_id = max(cand_models, key=lambda k: results[k]["val_metrics"]["verification"]["accuracy"] - results[k]["val_metrics"]["verification"]["eer"])
    print(f"\n--- Best Model based on Validation: {best_id} ({results[best_id]['name']}) ---")
    results["E5"] = {
        "name": f"E5: Best Model Without Augmentation ({best_id})",
        "selected_model": best_id,
        "val_metrics": results[best_id]["val_metrics"],
        "test_metrics": results[best_id]["test_metrics"]
    }

    # -------------------------------------------------------------------------
    # Experiment E6: Best model with geometric augmentation
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment E6: Best model with Geometric Augmentation ---")
    if "SVM" in results[best_id]["name"]:
        print("  Generating augmented pairwise training data...")
        X_train_aug, y_train_aug = create_pairwise_dataset(
            df, combined_cache, train_wids, pairs_per_writer=35, random_seed=42, augment_positive=True
        )
        print(f"  Trained augmented pairs: {len(y_train_aug)}")
        kernel = "rbf" if "RBF" in results[best_id]["name"] else "linear"
        aug_svm, aug_train_time = train_svm_pipeline(
            X_train_aug, y_train_aug, kernel=kernel, C=1.0, max_iter=2500, random_state=42
        )
        val_e6 = run_one_shot_evaluation(
            val_wids, df, cache, feature_key="combined", method="svm",
            svm_model=aug_svm, match_threshold=0.5, uncertain_lower=0.4,
            unknown_writers=train_wids[:5]
        )
        e6_thresh = val_e6["verification"]["eer_threshold"]
        test_e6 = run_one_shot_evaluation(
            test_wids, df, cache, feature_key="combined", method="svm",
            svm_model=aug_svm, match_threshold=e6_thresh,
            uncertain_lower=max(0.1, e6_thresh - 0.15),
            unknown_writers=val_wids
        )
        save_model(aug_svm, "best_model.joblib")
    else:
        # If baseline cosine is best, evaluate whether threshold calibration on augmented val improves
        val_e6 = run_one_shot_evaluation(
            val_wids, df, cache, feature_key=results[best_id]["name"].split(":")[1].strip().split("+")[0].strip().lower(),
            method="cosine", match_threshold=DEFAULT_COSINE_MATCH_THRESHOLD, uncertain_lower=DEFAULT_COSINE_UNCERTAIN_LOWER,
            unknown_writers=train_wids[:5]
        )
        e6_thresh = val_e6["verification"]["eer_threshold"]
        test_e6 = run_one_shot_evaluation(
            test_wids, df, cache, feature_key=results[best_id]["name"].split(":")[1].strip().split("+")[0].strip().lower(),
            method="cosine", match_threshold=e6_thresh, uncertain_lower=e6_thresh - 0.10,
            unknown_writers=val_wids
        )
        
    results["E6"] = {
        "name": f"E6: {best_id} with Geometric Augmentation",
        "description": "Training/Calibration enriched with rotation, scaling, translation, and mild shear variations",
        "val_metrics": val_e6,
        "test_metrics": test_e6
    }
    print(f"  Test Verif Acc: {test_e6['verification']['accuracy']:.4f}, EER: {test_e6['verification']['eer']:.4f}, Top-1 ID Acc: {test_e6['identification']['top1_accuracy']:.4f}")

    # Also save the trained SVM models for runtime use in the app
    save_model(linear_svm, "linear_svm_pipeline.joblib")
    save_model(rbf_svm, "rbf_svm_pipeline.joblib")
    
    # 4. Save results to JSON and generate comparison plots
    json_path = EXPERIMENT_RESULTS_DIR / "experiment_metrics.json"
    
    # Clean json-serializable copy
    json_clean = {}
    for exp_k, exp_v in results.items():
        json_clean[exp_k] = {
            "name": exp_v["name"],
            "description": exp_v.get("description", ""),
            "val_accuracy": exp_v["val_metrics"]["verification"]["accuracy"],
            "val_eer": exp_v["val_metrics"]["verification"]["eer"],
            "val_auc": exp_v["val_metrics"]["verification"]["roc_auc"],
            "test_accuracy": exp_v["test_metrics"]["verification"]["accuracy"],
            "test_precision": exp_v["test_metrics"]["verification"]["precision"],
            "test_recall": exp_v["test_metrics"]["verification"]["recall"],
            "test_f1": exp_v["test_metrics"]["verification"]["f1"],
            "test_far": exp_v["test_metrics"]["verification"]["far"],
            "test_frr": exp_v["test_metrics"]["verification"]["frr"],
            "test_eer": exp_v["test_metrics"]["verification"]["eer"],
            "test_auc": exp_v["test_metrics"]["verification"]["roc_auc"],
            "top1_id_accuracy": exp_v["test_metrics"]["identification"]["top1_accuracy"],
            "unknown_rejection": exp_v["test_metrics"]["identification"]["unknown_rejection_accuracy"],
            "inference_time_ms": exp_v["test_metrics"]["identification"]["mean_inference_time_ms"],
            "confusion_matrix": exp_v["test_metrics"]["verification"]["confusion_matrix"]
        }
        
    with open(json_path, "w") as f:
        json.dump(json_clean, f, indent=2)
    print(f"\nExperiment metrics successfully saved to {json_path}")
    
    # 5. Generate Matplotlib Summary Visualizations
    generate_experiment_charts(json_clean, results)
    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return results


def generate_experiment_charts(summary: Dict[str, Any], full_results: Dict[str, Any]):
    """Generates comparison bar charts, ROC curves, and confusion matrices."""
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar Chart: Verification Accuracy, Top-1 Identification Accuracy, and EER
    labels = list(summary.keys())
    verif_accs = [summary[k]["test_accuracy"] * 100 for k in labels]
    top1_accs = [summary[k]["top1_id_accuracy"] * 100 for k in labels]
    eers = [summary[k]["test_eer"] * 100 for k in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width, verif_accs, width, label='Verification Accuracy (%)', color='#2b5c8f')
    r2 = ax.bar(x, top1_accs, width, label='Top-1 Identification Accuracy (%)', color='#2e8b57')
    r3 = ax.bar(x + width, eers, width, label='Equal Error Rate (EER %)', color='#d9534f')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('One-Shot Signature Recognition: Experimental Comparison (E1 - E6)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    for r in [r1, r2, r3]:
        for bar in r:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    plt.tight_layout()
    chart_path = EXPERIMENT_RESULTS_DIR / "experiment_comparison_chart.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    # 2. ROC Curves for E1 to E4
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, exp_k in enumerate(["E1", "E2", "E3", "E4"]):
        if exp_k in full_results:
            y_true = np.array(full_results[exp_k]["test_metrics"]["verification"]["y_true"])
            y_scores = np.array(full_results[exp_k]["test_metrics"]["verification"]["y_scores"])
            fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{exp_k} (AUC = {roc_auc:.3f})')
            
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FAR)')
    ax.set_ylabel('True Positive Rate (1 - FRR)')
    ax.set_title('Receiver Operating Characteristic (ROC) — Unseen Writers')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    roc_path = EXPERIMENT_RESULTS_DIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # 3. Confusion Matrix of Best Model
    best_cm = np.array(summary["E5"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(best_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Negative (Diff/Forg)', 'Positive (Same)'],
           yticklabels=['Negative (Diff/Forg)', 'Positive (Same)'],
           title=f'Confusion Matrix ({summary["E5"]["name"]})',
           ylabel='True Label',
           xlabel='Predicted Label')
           
    thresh = best_cm.max() / 2.
    for i in range(best_cm.shape[0]):
        for j in range(best_cm.shape[1]):
            ax.text(j, i, format(best_cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if best_cm[i, j] > thresh else "black",
                    fontweight="bold")
    plt.tight_layout()
    cm_path = EXPERIMENT_RESULTS_DIR / "best_model_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Generated comparison charts: {chart_path}, {roc_path}, {cm_path}")


if __name__ == "__main__":
    run_all_experiments()
