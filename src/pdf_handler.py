import pymupdf
import numpy as np
import cv2
from typing import Union, Tuple, Optional
from pathlib import Path


def get_pdf_page_count(pdf_input: Union[str, Path, bytes]) -> int:
    """Returns the total number of pages in the PDF document."""
    if isinstance(pdf_input, (str, Path)):
        doc = pymupdf.open(str(pdf_input))
    else:
        doc = pymupdf.open(stream=pdf_input, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


def render_pdf_page_to_image(
    pdf_input: Union[str, Path, bytes],
    page_number: int = 0,
    dpi: int = 150
) -> np.ndarray:
    """
    Renders a specific PDF page to an RGB numpy array at specified DPI.
    Uses PyMuPDF pixmap rendering.
    """
    if isinstance(pdf_input, (str, Path)):
        doc = pymupdf.open(str(pdf_input))
    else:
        doc = pymupdf.open(stream=pdf_input, filetype="pdf")
        
    if page_number < 0 or page_number >= len(doc):
        doc.close()
        raise IndexError(f"Page index {page_number} out of bounds (0..{len(doc)-1})")
        
    page = doc.load_page(page_number)
    # Scale matrix for target DPI (standard PDF DPI is 72)
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert pixmap buffer to numpy array (H, W, 3) in RGB
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
    doc.close()
    return img


def crop_roi(
    image: np.ndarray,
    x_percent: float,
    y_percent: float,
    w_percent: float,
    h_percent: float
) -> np.ndarray:
    """
    Crops a sub-region (ROI) from an image using percentage coordinates [0.0 - 100.0].
    """
    img_h, img_w = image.shape[:2]
    
    x = int(np.clip(x_percent / 100.0 * img_w, 0, img_w - 1))
    y = int(np.clip(y_percent / 100.0 * img_h, 0, img_h - 1))
    w = int(np.clip(w_percent / 100.0 * img_w, 10, img_w - x))
    h = int(np.clip(h_percent / 100.0 * img_h, 10, img_h - y))
    
    return image[y : y + h, x : x + w]
