"""Utility functions for SAM segmentation pipeline."""

from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps


def mask_to_polygon(
    mask: np.ndarray,
    tolerance: float = 2.0,
    threshold: float = 0.5
) -> List[List[Tuple[int, int]]]:
    """
    Convert binary mask to polygon coordinates.

    Args:
        mask: Binary mask array (H, W) or (1, H, W)
        tolerance: Tolerance for polygon approximation (lower = more precise)
        threshold: Threshold for binarizing mask

    Returns:
        List of polygons, where each polygon is a list of (x, y) coordinates
    """
    if mask.ndim > 2:
        mask = mask.squeeze()

    mask_uint8 = (mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, tolerance, True)
        if len(approx) >= 3:
            polygon = approx.squeeze().tolist()
            if isinstance(polygon, list) and polygon and isinstance(polygon[0], list):
                polygons.append(polygon)
    return polygons


def get_supported_image_extensions() -> List[str]:
    """
    Return list of supported image file extensions.

    Returns:
        List of extensions (e.g., ['.jpg', '.jpeg', '.png', ...])
    """
    return [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]


def load_image_with_exif(
    image_path: Union[str, Path],
    enable_exif: bool = True
) -> Image.Image:
    """
    Load image with optional EXIF orientation correction.

    Args:
        image_path: Path to image file
        enable_exif: If True, apply EXIF orientation correction

    Returns:
        PIL Image object
    """
    image = Image.open(image_path)
    if enable_exif:
        image = ImageOps.exif_transpose(image)
    return image


def collect_images(
    directory: Path,
    extensions: List[str] = None
) -> List[Path]:
    """
    Find all supported image files in a directory.

    Args:
        directory: Directory to search
        extensions: List of file extensions to include (default: all supported)

    Returns:
        Sorted list of image file paths
    """
    if extensions is None:
        extensions = get_supported_image_extensions()

    files = set()
    for ext in extensions:
        files.update(directory.glob(f"*{ext}"))
        files.update(directory.glob(f"*{ext.upper()}"))

    # Sort by name for stable ordering
    return sorted(files, key=lambda p: p.name.lower())
