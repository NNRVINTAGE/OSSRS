import numpy as np
from typing import Union, Tuple


def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-10) -> float:
    """
    Computes standard cosine similarity between two 1D vectors u and v:
    cosine_similarity = dot(u, v) / (||u|| * ||v|| + eps)
    Result is in range [-1.0, 1.0], typically [0.0, 1.0] for non-negative descriptors.
    """
    u_flat = u.flatten().astype(np.float32)
    v_flat = v.flatten().astype(np.float32)
    
    norm_u = np.linalg.norm(u_flat)
    norm_v = np.linalg.norm(v_flat)
    
    if norm_u < eps or norm_v < eps:
        return 0.0
        
    dot_val = np.dot(u_flat, v_flat)
    similarity = dot_val / (norm_u * norm_v + eps)
    return float(np.clip(similarity, -1.0, 1.0))


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Computes Euclidean distance between two vectors."""
    u_flat = u.flatten().astype(np.float32)
    v_flat = v.flatten().astype(np.float32)
    return float(np.linalg.norm(u_flat - v_flat))


def compute_pair_features(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
    """
    Constructs the pairwise feature vector for Model B (SVM classifier):
    Pair Features = [abs(F_A - F_B), F_A * F_B, cosine_similarity]
    
    Captures:
    1. Absolute feature difference (dissimilarity across each descriptor dimension)
    2. Elementwise product (correlation across each descriptor dimension)
    3. Global cosine similarity scalar
    """
    a = feat_a.flatten().astype(np.float32)
    b = feat_b.flatten().astype(np.float32)
    
    abs_diff = np.abs(a - b)
    prod = a * b
    cos_sim = np.array([cosine_similarity(a, b)], dtype=np.float32)
    
    pair_vec = np.concatenate([abs_diff, prod, cos_sim], axis=0).astype(np.float32)
    return pair_vec
