import pytest
import numpy as np

from src.similarity import cosine_similarity, compute_pair_features


def test_cosine_similarity():
    """Verify mathematical properties of cosine similarity."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    c = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    d = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Identical vectors -> similarity = 1.0
    assert pytest.approx(cosine_similarity(a, b), 1e-4) == 1.0

    # Opposite vectors -> similarity = -1.0
    assert pytest.approx(cosine_similarity(a, c), 1e-4) == -1.0

    # Zero vector safeguard -> similarity = 0.0
    assert cosine_similarity(a, d) == 0.0


def test_pairwise_feature_construction():
    """Test pairwise feature combination vector."""
    dim = 8119
    f1 = np.ones(dim, dtype=np.float32)
    f2 = np.ones(dim, dtype=np.float32) * 0.5

    pair_feat = compute_pair_features(f1, f2)
    # Expected length: 2 * 8119 + 1 = 16239
    assert pair_feat.shape == (2 * dim + 1,)
    assert not np.isnan(pair_feat).any()
