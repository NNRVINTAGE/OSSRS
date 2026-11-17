import numpy as np
from typing import List, Dict, Any, Optional

from config.settings import (
    DEFAULT_COSINE_MATCH_THRESHOLD,
    DEFAULT_COSINE_UNCERTAIN_LOWER,
    DEFAULT_TOP_MARGIN_THRESHOLD,
    DECISION_MATCH,
    DECISION_UNCERTAIN,
    DECISION_UNKNOWN
)
from src.similarity import cosine_similarity, compute_pair_features


def identify_signature(
    probe_features: np.ndarray,
    registered_templates: List[Dict[str, Any]],
    method: str = "cosine",
    svm_model: Optional[Any] = None,
    match_threshold: float = DEFAULT_COSINE_MATCH_THRESHOLD,
    uncertain_lower: float = DEFAULT_COSINE_UNCERTAIN_LOWER,
    top_margin_threshold: float = DEFAULT_TOP_MARGIN_THRESHOLD
) -> Dict[str, Any]:
    """
    Performs 1:N Signature Identification of an unknown probe signature
    against all registered user templates.
    
    Returns:
        {
            "decision": "Match" | "Uncertain" | "Unknown",
            "predicted_user_id": int or None,
            "predicted_name": str or None,
            "top_score": float,
            "margin": float,
            "ranked_candidates": List[Dict],
            "explanation": str
        }
    """
    if not registered_templates:
        return {
            "decision": DECISION_UNKNOWN,
            "predicted_user_id": None,
            "predicted_name": None,
            "top_score": 0.0,
            "margin": 0.0,
            "ranked_candidates": [],
            "explanation": "No users are currently registered in the database."
        }
        
    candidates = []
    
    for t in registered_templates:
        tmpl_feat = t.get("features")
        if tmpl_feat is None:
            continue
            
        cos_sim = float(cosine_similarity(probe_features, tmpl_feat))
        if method == "svm" and svm_model is not None:
            pair_feat = compute_pair_features(probe_features, tmpl_feat)
            prob = float(svm_model.predict_proba([pair_feat])[0][1])
            # Safeguard for identical / near-identical vectors
            score = max(prob, cos_sim) if cos_sim >= 0.90 else prob
        else:  # default cosine
            score = cos_sim
            
        candidates.append({
            "user_id": t["user_id"],
            "name": t["name"],
            "score": score,
            "image_path": t.get("image_path")
        })
        
    if not candidates:
        return {
            "decision": DECISION_UNKNOWN,
            "predicted_user_id": None,
            "predicted_name": None,
            "top_score": 0.0,
            "margin": 0.0,
            "ranked_candidates": [],
            "explanation": "Could not extract features from registered templates."
        }
        
    # Sort descending by similarity score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    top = candidates[0]
    top_score = top["score"]
    
    if len(candidates) > 1:
        second = candidates[1]
        margin = top_score - second["score"]
    else:
        margin = top_score
        
    # Decision Logic
    if top_score < match_threshold:
        if top_score < uncertain_lower:
            decision = DECISION_UNKNOWN
            explanation = (
                f"Similarity score ({top_score:.3f}) is below the acceptable threshold ({match_threshold:.3f}). "
                "The signature does not sufficiently resemble any registered template and is classified as Unknown."
            )
            predicted_id = None
            predicted_name = None
        else:
            decision = DECISION_UNCERTAIN
            explanation = (
                f"Top similarity ({top_score:.3f}) falls in the ambiguous borderline zone "
                f"[{uncertain_lower:.3f}, {match_threshold:.3f}). Confidence is insufficient to confirm identity."
            )
            predicted_id = top["user_id"]
            predicted_name = top["name"]
    else:
        # Score is above match_threshold, check separation margin against 2nd candidate
        if len(candidates) > 1 and margin < top_margin_threshold:
            decision = DECISION_UNCERTAIN
            explanation = (
                f"Top candidate '{top['name']}' ({top_score:.3f}) is closely contested by '{candidates[1]['name']}' "
                f"({candidates[1]['score']:.3f}). The separation margin ({margin:.3f}) is below required confidence ({top_margin_threshold:.3f})."
            )
            predicted_id = top["user_id"]
            predicted_name = top["name"]
        else:
            decision = DECISION_MATCH
            explanation = (
                f"Signature identified as '{top['name']}' with similarity score of {top_score:.3f} "
                f"and clear confidence margin of {margin:.3f} over second candidate."
            )
            predicted_id = top["user_id"]
            predicted_name = top["name"]
            
    return {
        "decision": decision,
        "predicted_user_id": predicted_id,
        "predicted_name": predicted_name,
        "top_score": top_score,
        "margin": margin,
        "ranked_candidates": candidates,
        "explanation": explanation
    }
