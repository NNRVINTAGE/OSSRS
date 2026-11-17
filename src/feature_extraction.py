import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from skimage.feature import hog

from config.settings import HOG_CONFIG, IMAGE_SIZE


def extract_hog_features(
    image: np.ndarray,
    orientations: int = HOG_CONFIG["orientations"],
    pixels_per_cell: Tuple[int, int] = HOG_CONFIG["pixels_per_cell"],
    cells_per_block: Tuple[int, int] = HOG_CONFIG["cells_per_block"],
    block_norm: str = HOG_CONFIG["block_norm"]
) -> np.ndarray:
    """
    Extracts Histogram of Oriented Gradients (HOG) features from the processed 128x128 image.
    Captures edge directionalities, stroke junctions, and local stroke distribution.
    Output dimension: 8,100 floats for 128x128, 8x8 cells, 2x2 blocks, 9 orientations.
    """
    # Normalize image to float [0.0, 1.0]
    if image.dtype != np.float32 and image.dtype != np.float64:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.copy()
        
    features = hog(
        img_float,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
        feature_vector=True
    )
    return features.astype(np.float32)


def extract_hu_moments(image: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Extracts the 7 Hu Invariant Moments from the signature shape.
    Hu moments provide scale, translation, and rotation-invariant shape representation.
    Applies logarithmic transform -sign(h) * log10(|h| + eps) to compress dynamic range
    and safeguard against numerical instability.
    Output dimension: 7 floats.
    """
    # Image must be uint8 or float
    if image.dtype != np.uint8:
        img_u8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        img_u8 = image

    moments = cv2.moments(img_u8)
    hu_raw = cv2.HuMoments(moments).flatten()
    
    # Logarithmic transformation with absolute magnitude preservation
    # (prevents cross-moment sign flips on discrete binary strokes from destabilizing dot products)
    hu_log = -np.sign(hu_raw) * np.log10(np.abs(hu_raw) + eps)
    hu_stable = np.abs(hu_log)
    
    # Clean any NaNs or Infs if image was completely blank
    hu_clean = np.nan_to_num(hu_stable, nan=0.0, posinf=20.0, neginf=0.0)
    return hu_clean.astype(np.float32)


def extract_geometric_features(image: np.ndarray) -> np.ndarray:
    """
    Extracts 12 global and local geometric features from the 128x128 signature:
    1. Aspect ratio: bounding-box width / max(bounding-box height, 1)
    2. Ink occupancy ratio: fg_pixels / (bbox_w * bbox_h)
    3. Normalized bbox width: bbox_w / 128.0
    4. Normalized bbox height: bbox_h / 128.0
    5. Contour area ratio: sum(contour_areas) / (bbox_w * bbox_h)
    6. Overall stroke density: fg_pixels / (128 * 128)
    7. Centroid X (horizontal center of mass normalized): cx / 128.0
    8. Centroid Y (vertical center of mass normalized): cy / 128.0
    9. Horizontal spread (normalized x-variance): var_x / (128^2)
    10. Vertical spread (normalized y-variance): var_y / (128^2)
    11. Left-to-Right stroke balance: left_half_pixels / max(right_half_pixels, 1)
    12. Top-to-Bottom stroke balance: top_half_pixels / max(bottom_half_pixels, 1)
    
    Output dimension: 12 floats.
    """
    if image.dtype != np.uint8:
        img_u8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        img_u8 = image
        
    h_img, w_img = img_u8.shape[:2]
    total_pixels = float(h_img * w_img)
    
    # Threshold check to locate foreground pixels
    fg_mask = img_u8 > 30
    fg_count = np.sum(fg_mask)
    
    if fg_count < 10:
        # Empty or virtually blank signature
        return np.zeros(12, dtype=np.float32)
        
    coords = cv2.findNonZero(img_u8)
    x, y, w, h = cv2.boundingRect(coords)
    
    # 1. Aspect ratio
    aspect_ratio = float(w) / max(float(h), 1.0)
    
    # 2. Ink occupancy ratio
    bbox_area = max(float(w * h), 1.0)
    ink_occupancy = float(fg_count) / bbox_area
    
    # 3 & 4. Bounding box dimensions normalized
    norm_w = float(w) / float(w_img)
    norm_h = float(h) / float(h_img)
    
    # 5. Contour area ratio
    contours, _ = cv2.findContours(img_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area_sum = sum(cv2.contourArea(c) for c in contours) if contours else 0.0
    contour_area_ratio = float(contour_area_sum) / bbox_area
    
    # 6. Overall stroke density
    stroke_density = float(fg_count) / total_pixels
    
    # 7 & 8. Centroid
    moments = cv2.moments(img_u8)
    if moments["m00"] > 0:
        cx = (moments["m10"] / moments["m00"]) / float(w_img)
        cy = (moments["m01"] / moments["m00"]) / float(h_img)
    else:
        cx = 0.5
        cy = 0.5
        
    # 9 & 10. Horizontal and vertical spread (variances)
    y_indices, x_indices = np.where(fg_mask)
    var_x = float(np.var(x_indices)) / (float(w_img) ** 2) if len(x_indices) > 0 else 0.0
    var_y = float(np.var(y_indices)) / (float(h_img) ** 2) if len(y_indices) > 0 else 0.0
    
    # 11 & 12. Quadrant / Half ratios
    half_w = w_img // 2
    half_h = h_img // 2
    left_fg = float(np.sum(fg_mask[:, :half_w]))
    right_fg = float(np.sum(fg_mask[:, half_w:]))
    top_fg = float(np.sum(fg_mask[:half_h, :]))
    bottom_fg = float(np.sum(fg_mask[half_h:, :]))
    
    lr_balance = left_fg / max(right_fg, 1.0)
    tb_balance = top_fg / max(bottom_fg, 1.0)
    # Clip balance ratios to avoid extreme outliers
    lr_balance = np.clip(lr_balance, 0.0, 10.0)
    tb_balance = np.clip(tb_balance, 0.0, 10.0)
    
    geom_features = np.array([
        aspect_ratio,
        ink_occupancy,
        norm_w,
        norm_h,
        contour_area_ratio,
        stroke_density,
        cx,
        cy,
        var_x,
        var_y,
        lr_balance,
        tb_balance
    ], dtype=np.float32)
    
    return np.nan_to_num(geom_features, nan=0.0)


def extract_features(
    image: np.ndarray,
    include_hog: bool = True,
    include_hu: bool = True,
    include_geometry: bool = True
) -> np.ndarray:
    """
    Extracts and concatenates the selected feature representations:
    Combined Features = [Normalized HOG, Normalized Hu Moments, Normalized Geometric Features].
    
    Subvectors are individually L2-normalized before combination so that local stroke patterns (HOG),
    global shape invariants (Hu), and spatial structure (Geometry) have balanced influence
    without high-magnitude descriptors dominating or destabilizing similarity comparisons.
    
    Guarantees consistent dimensionality (8,119 floats total) for registration, training, and inference.
    """
    components = []
    
    if include_hog:
        f_hog = extract_hog_features(image)
        norm_h = np.linalg.norm(f_hog)
        components.append(f_hog / (norm_h + 1e-10))
        
    if include_hu:
        f_hu = extract_hu_moments(image)
        norm_u = np.linalg.norm(f_hu)
        components.append(f_hu / (norm_u + 1e-10))
        
    if include_geometry:
        f_geom = extract_geometric_features(image)
        norm_g = np.linalg.norm(f_geom)
        components.append(f_geom / (norm_g + 1e-10))
        
    if not components:
        raise ValueError("At least one feature extraction method must be enabled.")
        
    combined = np.concatenate(components, axis=0).astype(np.float32)
    return combined


def get_feature_dimensions(
    include_hog: bool = True,
    include_hu: bool = True,
    include_geometry: bool = True
) -> Dict[str, int]:
    """Helper returning feature sub-vector lengths."""
    dims = {}
    total = 0
    if include_hog:
        dims["hog"] = 8100
        total += 8100
    if include_hu:
        dims["hu"] = 7
        total += 7
    if include_geometry:
        dims["geometry"] = 12
        total += 12
    dims["total"] = total
    return dims
