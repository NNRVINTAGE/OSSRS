import pytest
import numpy as np
import cv2

from src.feature_extraction import (
    extract_hog_features,
    extract_hu_moments,
    extract_geometric_features,
    extract_features,
    get_feature_dimensions
)


def test_feature_dimensions():
    """Verify exact feature vector lengths."""
    dims = get_feature_dimensions()
    assert dims["hog"] == 8100
    assert dims["hu"] == 7
    assert dims["geometry"] == 12
    assert dims["total"] == 8119


def test_feature_extraction_pipeline():
    """Test extracting features from a 128x128 synthetic signature."""
    dummy_img = np.zeros((128, 128), dtype=np.uint8)
    # Draw stroke
    cv2.line(dummy_img, (20, 40), (100, 90), 255, 3)
    cv2.circle(dummy_img, (64, 64), 15, 255, 2)

    # 1. HOG
    hog_feat = extract_hog_features(dummy_img)
    assert hog_feat.shape == (8100,)
    assert not np.isnan(hog_feat).any()

    # 2. Hu moments
    hu_feat = extract_hu_moments(dummy_img)
    assert hu_feat.shape == (7,)
    assert not np.isnan(hu_feat).any()

    # 3. Geometric features
    geom_feat = extract_geometric_features(dummy_img)
    assert geom_feat.shape == (12,)
    assert not np.isnan(geom_feat).any()

    # 4. Combined
    combined = extract_features(dummy_img)
    assert combined.shape == (8119,)
    assert not np.isnan(combined).any()
