import numpy as np
from typing import Dict, Any, Optional

from config.settings import (
    DEFAULT_COSINE_MATCH_THRESHOLD,
    DEFAULT_COSINE_UNCERTAIN_LOWER,
    DECISION_MATCH,
    DECISION_NOT_MATCH,
    DECISION_UNCERTAIN
)
from src.similarity import cosine_similarity, compute_pair_features


def verify_signature(
    probe_features: np.ndarray,
    template_features: np.ndarray,
    claimed_user_name: str = "Claimed User",
    method: str = "cosine",
    svm_model: Optional[Any] = None,
    match_threshold: float = DEFAULT_COSINE_MATCH_THRESHOLD,
    uncertain_lower: float = DEFAULT_COSINE_UNCERTAIN_LOWER
) -> Dict[str, Any]:
    """
    Performs 1:1 Signature Verification of a submitted signature against
    the single registered template of the claimed user.
    
    Returns:
        {
            "decision": "Match" | "Not Match" | "Uncertain",
            "similarity_score": float,
            "threshold": float,
            "uncertain_lower": float,
            "explanation": str,
            "disclaimer": str
        }
    """
    cos_sim = float(cosine_similarity(probe_features, template_features))
    
    if method == "svm" and svm_model is not None:
        pair_feat = compute_pair_features(probe_features, template_features)
        prob = float(svm_model.predict_proba([pair_feat])[0][1])
        # Safeguard: If signatures are identical or near-identical (cosine >= 0.90),
        # pairwise difference is near-zero (out-of-distribution for SVM trained only on distinct pairs).
        # We ensure score reflects physical identity.
        score = max(prob, cos_sim) if cos_sim >= 0.90 else prob
    else:  # default cosine
        score = cos_sim
        
    disclaimer = (
        "Academic Prototype Notice: High visual similarity indicates stroke resemblance "
        "and geometric alignment, not forensic or certified legal authenticity."
    )
    
    if score >= match_threshold:
        decision = DECISION_MATCH
        explanation = (
            f"Signature verified as '{claimed_user_name}'. "
            f"Similarity score ({score:.3f}) exceeds the validation match threshold ({match_threshold:.3f})."
        )
    elif score >= uncertain_lower:
        decision = DECISION_UNCERTAIN
        explanation = (
            f"Similarity score ({score:.3f}) falls in the indeterminate borderline interval "
            f"[{uncertain_lower:.3f}, {match_threshold:.3f}). The signature shows partial resemblance "
            "but insufficient confidence to confirm identity without manual inspection."
        )
    else:
        decision = DECISION_NOT_MATCH
        explanation = (
            f"Signature rejected. Similarity score ({score:.3f}) is well below the threshold "
            f"({match_threshold:.3f}). Significant discrepancy detected in stroke geometry or orientation."
        )
        
    return {
        "decision": decision,
        "similarity_score": score,
        "threshold": match_threshold,
        "uncertain_lower": uncertain_lower,
        "explanation": explanation,
        "disclaimer": disclaimer
    }
