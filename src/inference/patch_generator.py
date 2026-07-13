"""
patch_generator.py

Slices a large Sentinel-2 satellite image into smaller, 
model-compatible patches.
"""

import numpy as np
from PIL import Image
from typing import Generator, Tuple, Dict, Any


class PatchGenerator:
    """
    Splits a large image into smaller patches of fixed size.
    """
    def __init__(self, patch_size: int = 64, stride: int = 64):
        """
        Parameters
        ----------
        patch_size : int
            Size of the square patch (e.g., 64 or 224).
        stride : int
            Stride between patches. If stride == patch_size, 
            patches are non-overlapping.
        """
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, image_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields patches and their bounding box coordinates.
        
        Yields
        ------
        dict
            Contains 'image' (PIL.Image) and 'bbox' (x, y, w, h).
        """
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                
                box = (x, y, x + self.patch_size, y + self.patch_size)
                patch = image.crop(box)
                
                yield {
                    "image": patch,
                    "bbox": (x, y, self.patch_size, self.patch_size)
                }

    def get_grid_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Returns the number of patches along width and height.
        """
        image = Image.open(image_path)
        width, height = image.size
        
        nx = (width - self.patch_size) // self.stride + 1
        ny = (height - self.patch_size) // self.stride + 1
        
        return nx, ny
