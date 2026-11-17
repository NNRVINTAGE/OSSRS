import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
TEMPLATES_DIR = DATA_DIR / "templates"
DATABASE_PATH = DATA_DIR / "signature_recognition.db"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENT_RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Ensure directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, TEMPLATES_DIR, EXPERIMENTS_DIR, EXPERIMENT_RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Image Preprocessing Configuration
IMAGE_SIZE = (128, 128)  # Height x Width (128x128)
BLUR_KERNEL_SIZE = (3, 3)
THRESHOLD_METHOD = "otsu"  # 'otsu' or 'adaptive'
MIN_BOUNDING_BOX_AREA = 100  # Minimum foreground pixels to detect signature

# Feature Extraction Configuration
# HOG
HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

# Decision & Thresholding Defaults
# Calibrated via validation experiments and centroid-aligned feature representations
DEFAULT_COSINE_MATCH_THRESHOLD = 0.80
DEFAULT_COSINE_UNCERTAIN_LOWER = 0.76
DEFAULT_TOP_MARGIN_THRESHOLD = 0.025

DEFAULT_SVM_MATCH_THRESHOLD = 0.40
DEFAULT_SVM_UNCERTAIN_LOWER = 0.40

# Verification Decision Labels
DECISION_MATCH = "Match"
DECISION_NOT_MATCH = "Not Match"
DECISION_UNCERTAIN = "Uncertain"
DECISION_UNKNOWN = "Unknown"
