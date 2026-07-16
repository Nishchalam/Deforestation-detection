"""
utils.py

Helper functions for image loading, resizing, and coordinate operations.
"""

from PIL import Image
from typing import Tuple

def load_and_resize_image(image_path: str, target_size: Tuple[int, int] = None) -> Image.Image:
    """Loads an image and optionally resizes it."""
    img = Image.open(image_path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
    return img

def bbox_to_coords(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Converts bounding box (x, y, w, h) to (x0, y0, x1, y1) format."""
    x, y, w, h = bbox
    return x, y, x + w, y + h
