import time
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix
)

from config.settings import MODELS_DIR


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the corresponding optimal threshold.
    EER is the point where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Find index where |fpr - fnr| is minimized
    idx_eer = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)
    eer_threshold = float(thresholds[idx_eer]) if idx_eer < len(thresholds) else 0.5
    return eer, eer_threshold


def evaluate_pair_classifier(
    model_or_scores,
    X_val_or_y_true: np.ndarray,
    y_val: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates binary classification performance (same-writer=1 vs diff/forgery=0).
    Can take either a trained Pipeline model (with X, y) or direct scores (y_true, scores).
    """
    if hasattr(model_or_scores, "predict_proba"):
        model = model_or_scores
        X_val = X_val_or_y_true
        y_true = y_val
        y_probs = model.predict_proba(X_val)[:, 1]
    else:
        y_true = X_val_or_y_true
        y_probs = model_or_scores

    y_pred = (y_probs >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # FAR: False Acceptance Rate = FP / (FP + TN)
    far = float(fp / max(fp + tn, 1))
    # FRR: False Rejection Rate = FN / (FN + TP)
    frr = float(fn / max(fn + tp, 1))
    
    eer, eer_thresh = calculate_eer(y_true, y_probs)
    
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_probs, pos_label=1)
    roc_auc = float(auc(fpr_curve, tpr_curve))
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "far": far,
        "frr": frr,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist() if isinstance(y_true, np.ndarray) else list(y_true),
        "y_scores": y_probs.tolist() if isinstance(y_probs, np.ndarray) else list(y_probs)
    }


def train_svm_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "linear",
    C: float = 1.0,
    max_iter: int = 2500,
    random_state: int = 42
) -> Tuple[Pipeline, float]:
    """
    Trains an SVM pipeline with StandardScaler and probability estimation.
    Returns (trained_pipeline, train_duration_seconds).
    """
    start_time = time.time()
    
    pipeline = Pipeline([
        ("scaler", StandardScaler(copy=False)),
        ("svm", SVC(
            kernel=kernel,
            C=C,
            probability=True,
            max_iter=max_iter,
            random_state=random_state
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    duration = time.time() - start_time
    return pipeline, duration


def save_model(pipeline: Pipeline, filename: str) -> Path:
    """Serializes model pipeline to MODELS_DIR."""
    save_path = MODELS_DIR / filename
    joblib.dump(pipeline, save_path)
    return save_path


def load_model(filename: str) -> Pipeline:
    """Loads model pipeline from MODELS_DIR."""
    load_path = MODELS_DIR / filename
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")
    return joblib.load(load_path)
