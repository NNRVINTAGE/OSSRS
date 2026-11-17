import pytest
import numpy as np
import cv2
from pathlib import Path

from src.preprocessing import (
    preprocess_signature_image,
    to_grayscale,
    binarize_signature,
    get_signature_bounding_box,
    resize_with_aspect_ratio
)


def test_synthetic_image_preprocessing():
    """Test preprocessing on a synthetic signature image."""
    # Create synthetic white image with a black signature stroke
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.line(img, (50, 100), (350, 120), (0, 0, 0), 4)
    cv2.circle(img, (200, 100), 30, (0, 0, 0), 3)

    processed, debug = preprocess_signature_image(img, return_debug=True)

    # Assert shape is 128x128
    assert processed.shape == (128, 128)
    # Assert values are uint8
    assert processed.dtype == np.uint8
    # Assert debug stages are present
    assert "original" in debug
    assert "binary" in debug
    assert "cropped" in debug
    assert "processed" in debug
    # Foreground ink pixels must exist
    assert np.sum(processed > 0) > 50


def test_real_dataset_image_preprocessing():
    """Test preprocessing on an actual image from the CEDAR dataset."""
    sample_file = Path("Dataset/full_org/original_1_1.png")
    if sample_file.exists():
        processed = preprocess_signature_image(sample_file)
        assert processed.shape == (128, 128)
        assert np.sum(processed > 0) > 100
