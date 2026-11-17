import cv2
import numpy as np
import random
from typing import Optional, List


def augment_signature(
    image: np.ndarray,
    rotation_range: float = 5.0,        # +/- 5 degrees
    scale_range: tuple = (0.92, 1.08),  # 0.92 to 1.08 scale
    translation_range: int = 4,         # +/- 4 pixels
    shear_range: float = 0.05,          # Mild shear
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Applies controlled geometric transformations to a preprocessed 128x128 signature image.
    Preserves signature stroke topology while creating realistic geometric variations.
    Strictly intended for training pairwise models (Experiment E6 vs E5).
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
        
    h, w = image.shape[:2]
    
    # 1. Random rotation & scale
    angle = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    center = (w / 2.0, h / 2.0)
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 2. Random translation
    tx = np.random.uniform(-translation_range, translation_range)
    ty = np.random.uniform(-translation_range, translation_range)
    rot_mat[0, 2] += tx
    rot_mat[1, 2] += ty
    
    # 3. Random mild shear
    sh_x = np.random.uniform(-shear_range, shear_range)
    sh_y = np.random.uniform(-shear_range, shear_range)
    shear_mat = np.array([
        [1.0, sh_x, 0.0],
        [sh_y, 1.0, 0.0]
    ], dtype=np.float32)
    
    # Apply rotation, scale, translation
    transformed = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Apply shear
    augmented = cv2.warpAffine(
        transformed,
        shear_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Binarize back cleanly if original was binary
    _, augmented = cv2.threshold(augmented, 40, 255, cv2.THRESH_BINARY)
    return augmented


def generate_augmented_variants(
    image: np.ndarray,
    num_variants: int = 2
) -> List[np.ndarray]:
    """Generates multiple augmented variants of a single image."""
    variants = []
    for _ in range(num_variants):
        variants.append(augment_signature(image))
    return variants
