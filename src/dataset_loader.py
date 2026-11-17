import re
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

from config.settings import DATASET_DIR, PROCESSED_DATA_DIR
from src.preprocessing import preprocess_signature_image
from src.feature_extraction import extract_features
from src.similarity import compute_pair_features


def get_cedar_metadata(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """
    Scans the CEDAR dataset directory and returns a DataFrame of all signature files.
    """
    records = []
    
    org_dir = dataset_dir / "full_org"
    forg_dir = dataset_dir / "full_forg"
    
    if not org_dir.exists() or not forg_dir.exists():
        raise FileNotFoundError(f"CEDAR dataset folders not found in {dataset_dir}")
        
    # Genuine signatures
    for p in org_dir.glob("*.png"):
        m = re.match(r"original_(\d+)_(\d+)\.png", p.name)
        if m:
            records.append({
                "filepath": str(p),
                "writer_id": int(m.group(1)),
                "sample_id": int(m.group(2)),
                "is_forgery": 0,
                "label": "genuine"
            })
            
    # Forged signatures
    for p in forg_dir.glob("*.png"):
        m = re.match(r"forgeries_(\d+)_(\d+)\.png", p.name)
        if m:
            records.append({
                "filepath": str(p),
                "writer_id": int(m.group(1)),
                "sample_id": int(m.group(2)),
                "is_forgery": 1,
                "label": "forgery"
            })
            
    df = pd.DataFrame(records)
    return df


def get_writer_splits(
    df: pd.DataFrame,
    train_count: int = 35,
    val_count: int = 10,
    test_count: int = 10,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Performs writer-disjoint splitting of writer IDs into train, val, and test groups.
    Ensures zero writer leakage between training and testing.
    """
    unique_writers = sorted(df["writer_id"].unique())
    total_writers = len(unique_writers)
    
    if train_count + val_count + test_count > total_writers:
        raise ValueError(f"Requested splits ({train_count}+{val_count}+{test_count}) exceed total writers ({total_writers})")
        
    rng = random.Random(random_seed)
    shuffled_writers = list(unique_writers)
    rng.shuffle(shuffled_writers)
    
    train_writers = sorted(shuffled_writers[:train_count])
    val_writers = sorted(shuffled_writers[train_count : train_count + val_count])
    test_writers = sorted(shuffled_writers[train_count + val_count : train_count + val_count + test_count])
    
    return train_writers, val_writers, test_writers


def build_feature_cache(
    df: pd.DataFrame,
    include_hog: bool = True,
    include_hu: bool = True,
    include_geometry: bool = True,
    cache_path: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """
    Extracts features for all images in the metadata and optionally caches to disk.
    Returns a dictionary mapping filepath -> feature_vector.
    """
    if cache_path is not None and cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True).item()
        return loaded
        
    cache = {}
    total = len(df)
    for idx, row in df.iterrows():
        p = row["filepath"]
        try:
            processed = preprocess_signature_image(p)
            feat = extract_features(
                processed,
                include_hog=include_hog,
                include_hu=include_hu,
                include_geometry=include_geometry
            )
            cache[p] = feat
        except Exception as e:
            print(f"Error processing {p}: {e}")
            
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, cache)
        
    return cache


def create_pairwise_dataset(
    df: pd.DataFrame,
    feature_cache: Dict[str, np.ndarray],
    writer_ids: List[int],
    pairs_per_writer: int = 40,
    random_seed: int = 42,
    augment_positive: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds pairwise training samples (X_pairs, y_pairs) from the designated writers.
    Label 1 = Same Writer (Genuine vs Genuine from same person)
    Label 0 = Different Writer or Forgery (Skilled forgery or Different writer)
    """
    rng = random.Random(random_seed)
    np_rng = np.random.default_rng(random_seed)
    
    X_pairs = []
    y_pairs = []
    
    writer_set = set(writer_ids)
    sub_df = df[df["writer_id"].isin(writer_set)]
    
    # Pre-group filepaths by writer and authenticity
    writer_genuines: Dict[int, List[str]] = {}
    writer_forgeries: Dict[int, List[str]] = {}
    
    for wid in writer_ids:
        gen = sub_df[(sub_df["writer_id"] == wid) & (sub_df["is_forgery"] == 0)]["filepath"].tolist()
        forg = sub_df[(sub_df["writer_id"] == wid) & (sub_df["is_forgery"] == 1)]["filepath"].tolist()
        writer_genuines[wid] = [p for p in gen if p in feature_cache]
        writer_forgeries[wid] = [p for p in forg if p in feature_cache]
        
    from src.augmentation import augment_signature
    
    for wid in writer_ids:
        gen_list = writer_genuines[wid]
        forg_list = writer_forgeries[wid]
        
        if len(gen_list) < 2:
            continue
            
        # 1. Positive pairs: same writer genuine vs genuine
        num_pos = pairs_per_writer
        for _ in range(num_pos):
            p1, p2 = rng.sample(gen_list, 2)
            f1 = feature_cache[p1]
            f2 = feature_cache[p2]
            
            pair_feat = compute_pair_features(f1, f2)
            X_pairs.append(pair_feat)
            y_pairs.append(1)
            
            # If augment_positive is True, create an augmented pair
            if augment_positive:
                try:
                    # Preprocess and augment
                    img1 = preprocess_signature_image(p1)
                    aug_img1 = augment_signature(img1)
                    f1_aug = extract_features(aug_img1)
                    pair_feat_aug = compute_pair_features(f1_aug, f2)
                    X_pairs.append(pair_feat_aug)
                    y_pairs.append(1)
                except Exception:
                    pass
                
        # 2. Negative pairs - Type A: Genuine vs Skilled Forgery (same claimed identity)
        num_neg_forg = pairs_per_writer // 2
        for _ in range(num_neg_forg):
            if forg_list:
                p_gen = rng.choice(gen_list)
                p_forg = rng.choice(forg_list)
                f_gen = feature_cache[p_gen]
                f_forg = feature_cache[p_forg]
                
                pair_feat = compute_pair_features(f_gen, f_forg)
                X_pairs.append(pair_feat)
                y_pairs.append(0)
                
        # 3. Negative pairs - Type B: Genuine vs Genuine from different writer
        other_wids = [w for w in writer_ids if w != wid]
        num_neg_diff = pairs_per_writer - num_neg_forg
        for _ in range(num_neg_diff):
            diff_wid = rng.choice(other_wids)
            p_gen1 = rng.choice(gen_list)
            p_gen2 = rng.choice(writer_genuines[diff_wid])
            
            f1 = feature_cache[p_gen1]
            f2 = feature_cache[p_gen2]
            
            pair_feat = compute_pair_features(f1, f2)
            X_pairs.append(pair_feat)
            y_pairs.append(0)
            
    X_arr = np.array(X_pairs, dtype=np.float32)
    y_arr = np.array(y_pairs, dtype=np.int32)
    
    # Shuffle pairs
    perm = np_rng.permutation(len(y_arr))
    return X_arr[perm], y_arr[perm]
