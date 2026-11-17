import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional

from config.settings import IMAGE_SIZE, BLUR_KERNEL_SIZE, THRESHOLD_METHOD, MIN_BOUNDING_BOX_AREA


def load_image(image_input: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """
    Loads an image from a file path, raw bytes, or returns existing numpy array.
    Ensures image is in BGR or Grayscale numpy format.
    """
    if isinstance(image_input, np.ndarray):
        return image_input.copy()
    
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        # Use cv2.imdecode to safely support paths with non-ascii characters if any
        with open(path_str, "rb") as f:
            bytes_data = f.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image from path: {path_str}")
        return img
        
    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from provided byte stream.")
        return img

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts image to grayscale if it is in color."""
    if len(image.shape) == 2:
        return image.copy()
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # Handle RGBA - blend with white background
            alpha = image[:, :, 3] / 255.0
            bgr = image[:, :, :3]
            white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
            blended = (bgr * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
            return cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unexpected image shape: {image.shape}")


def reduce_noise(gray_image: np.ndarray, kernel_size: Tuple[int, int] = BLUR_KERNEL_SIZE) -> np.ndarray:
    """Applies gentle Gaussian blur to reduce high-frequency scan/camera noise."""
    return cv2.GaussianBlur(gray_image, kernel_size, 0)


def flatten_illumination(gray_image: np.ndarray) -> np.ndarray:
    """
    Normalizes uneven lighting and eliminates camera shadows (e.g. from smartphone cameras).
    Uses morphological closing with a large structural element to estimate the background illumination,
    then divides the grayscale image by this background estimation.
    """
    h, w = gray_image.shape
    k_size = max(15, (min(h, w) // 10) | 1)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    bg = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel_bg)
    bg[bg == 0] = 1  # Safeguard against division by zero
    normalized = cv2.divide(gray_image, bg, scale=255)
    return normalized


def trim_black_borders(gray_image: np.ndarray, thresh: float = 40.0) -> np.ndarray:
    """
    Trims dark camera viewport or letterbox borders on outer margins if present (in 2D: rows & cols),
    preventing edge borders from interfering with illumination normalization and thresholding.
    """
    r_means = np.mean(gray_image, axis=1)
    c_means = np.mean(gray_image, axis=0)
    v_rows = np.where(r_means > thresh)[0]
    v_cols = np.where(c_means > thresh)[0]
    if len(v_rows) > 0 and len(v_cols) > 0:
        r0 = v_rows[0] if v_rows[0] > 3 else 0
        r1 = v_rows[-1] + 1 if (gray_image.shape[0] - v_rows[-1]) > 3 else gray_image.shape[0]
        c0 = v_cols[0] if v_cols[0] > 3 else 0
        c1 = v_cols[-1] + 1 if (gray_image.shape[1] - v_cols[-1]) > 3 else gray_image.shape[1]
        return gray_image[r0:r1, c0:c1]
    return gray_image


def remove_ruled_lines(binary_image: np.ndarray) -> np.ndarray:
    """
    Detects and removes long horizontal lines from ruled notebook / folio paper
    or printed document divider lines, using morphological opening with a horizontal line kernel.
    Preserves localized signature strokes and underlines.
    """
    h, w = binary_image.shape
    line_len = max(25, int(w * 0.15))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, h_kernel)
    
    line_pixels = np.count_nonzero(lines)
    
    # If horizontal lines are detected
    if 0 < line_pixels:
        # Dilate line mask slightly vertically (3px) to erase full thickness of printed ruled lines
        v_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        lines_dilated = cv2.dilate(lines, v_dilate_kernel, iterations=1)
        
        # Subtract lines
        no_lines = cv2.subtract(binary_image, lines_dilated)
        
        # Heal and reconnect signature strokes that crossed the ruled lines (morphological bridge / closing)
        recon_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        reconnected = cv2.morphologyEx(no_lines, cv2.MORPH_CLOSE, recon_kernel)
        
        if np.count_nonzero(reconnected) >= 40:
            return reconnected
            
    return binary_image


def filter_small_noise(binary_image: np.ndarray) -> np.ndarray:
    """
    Removes small isolated salt-and-pepper noise specks and paper grain
    using connected components area filtering with a safe fixed threshold (8px).
    Prevents large document images from erroneously scaling noise thresholds to erase genuine strokes.
    """
    h, w = binary_image.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    cleaned = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect = comp_w / max(1, comp_h)
        
        # Discard flat isolated horizontal line fragments (aspect > 8.0 with small height <= 3)
        if aspect > 8.0 and comp_h <= 3:
            continue
            
        # Safe fixed small noise threshold: delete dust (1-7px), preserve all genuine strokes
        if area >= 8:
            cleaned[labels == i] = 255
            
    if np.count_nonzero(cleaned) < 40:
        return binary_image
    return cleaned


def binarize_signature(gray_image: np.ndarray, method: str = THRESHOLD_METHOD) -> np.ndarray:
    """
    Binarizes the grayscale image so that the signature ink is foreground (255)
    and the background paper is black (0).
    Uses Otsu's thresholding or Adaptive Gaussian thresholding.
    """
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            8
        )
    else:  # otsu
        # Otsu thresholding with inversion (dark ink -> 255 foreground, white paper -> 0 background)
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def get_signature_bounding_box(binary_image: np.ndarray, margin: int = 4) -> Tuple[int, int, int, int]:
    """
    Finds the bounding box [x, y, w, h] surrounding all significant foreground ink pixels.
    """
    coords = cv2.findNonZero(binary_image)
    if coords is None or len(coords) < MIN_BOUNDING_BOX_AREA:
        # Fallback to entire image if foreground is too faint
        h, w = binary_image.shape
        return 0, 0, w, h
    
    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = binary_image.shape
    
    # Add margin
    x_min = max(0, x - margin)
    y_min = max(0, y - margin)
    x_max = min(img_w, x + w + margin)
    y_max = min(img_h, y + h + margin)
    
    return x_min, y_min, (x_max - x_min), (y_max - y_min)


def correct_skew(binary_crop: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Corrects minor tilt/skew of the signature bounding box using cv2.minAreaRect.
    Only corrects if angle is within [-max_angle, max_angle] to prevent unwanted flips.
    """
    coords = cv2.findNonZero(binary_crop)
    if coords is None or len(coords) < MIN_BOUNDING_BOX_AREA:
        return binary_crop
    
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # OpenCV minAreaRect returns angle in [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Safeguard: only correct small natural skews
    if abs(angle) > max_angle or abs(angle) < 0.5:
        return binary_crop
        
    (h, w) = binary_crop.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary_crop,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return rotated


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    pad_value: int = 0
) -> np.ndarray:
    """
    Resizes image to target_size (H, W) preserving original aspect ratio by centering
    and padding with pad_value (0 for black background).
    """
    target_h, target_w = target_size
    h, w = image.shape[:2]
    
    if h == 0 or w == 0:
        return np.zeros(target_size, dtype=np.uint8)
        
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to target_size
    padded = np.full((target_h, target_w), pad_value, dtype=image.dtype)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return padded


def align_by_centroid(image: np.ndarray, target_size: Tuple[int, int] = IMAGE_SIZE, max_shift: int = 15) -> np.ndarray:
    """
    Translates the signature so that its Center of Mass (Centroid) is aligned
    with the center of the target canvas. This provides strong translation and
    positional invariance, based on Digital Image Processing (PCD) geometric moments.
    """
    M = cv2.moments(image)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        target_cx = target_size[1] / 2.0
        target_cy = target_size[0] / 2.0
        
        dx = float(target_cx - cx)
        dy = float(target_cy - cy)
        
        # Limit shift so strokes do not clip outside canvas borders
        dx = float(np.clip(dx, -max_shift, max_shift))
        dy = float(np.clip(dy, -max_shift, max_shift))
        
        T = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(
            image,
            T,
            target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return aligned
    return image


def preprocess_signature_image(
    image_input: Union[str, Path, bytes, np.ndarray],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    return_debug: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Full standard signature preprocessing pipeline:
    1. Load image
    2. Convert to grayscale
    3. Noise reduction
    4. Otsu / Adaptive thresholding (foreground ink = 255, background = 0)
    5. Detect signature bounding box
    6. Crop bounding box
    7. Minor skew correction
    8. Aspect-ratio preserving resize & padding to target_size (128x128)
    9. Intensity normalization
    
    Returns:
        processed: Normalized binary/grayscale signature (128x128, uint8 0 and 255)
        debug_stages (optional): Dict of intermediate stages for UI visualization
    """
    # 1. Load image
    orig = load_image(image_input)
    
    # 2. Grayscale & margin border cleanup
    gray = to_grayscale(orig)
    trimmed_gray = trim_black_borders(gray)
    
    # 3. Illumination flattening (eliminates smartphone shadows and non-uniform lighting)
    flat_gray = flatten_illumination(trimmed_gray)
    
    # 4. Noise reduction
    denoised = reduce_noise(flat_gray)
    
    # 5. Binarization (Otsu / Adaptive)
    binary = binarize_signature(denoised, method=THRESHOLD_METHOD)
    
    # 6. Notebook line removal & noise filtering
    binary_no_lines = remove_ruled_lines(binary)
    binary_clean = filter_small_noise(binary_no_lines)
    
    # 7. Foreground detection & bounding box
    x, y, w, h = get_signature_bounding_box(binary_clean)
    
    # 8. Crop
    cropped = binary_clean[y : y + h, x : x + w]
    
    # 9. Minor skew correction
    deskewed = correct_skew(cropped)
    
    # 10. Aspect-ratio preserving resize to target_size
    resized = resize_with_aspect_ratio(deskewed, target_size=target_size, pad_value=0)
    
    # 11. Centroid alignment (invariance to positional translation via Center of Mass)
    processed = align_by_centroid(resized, target_size=target_size)
    
    if return_debug:
        debug_stages = {
            "original": orig,
            "grayscale": gray,
            "normalized": flat_gray,
            "binary": binary_clean,
            "cropped": cropped,
            "deskewed": deskewed,
            "processed": processed
        }
        return processed, debug_stages
        
    return processed
