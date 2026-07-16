"""
patch_generator.py

Slices a large Sentinel-2 satellite image into smaller patches (overlapping or non-overlapping)
and reconstructs the full image/map from prediction grids.
"""

import numpy as np
from PIL import Image
from typing import Generator, Tuple, Dict, Any, List

class PatchGenerator:
    """
    Extracts patches from a large image using a sliding window and stitches them back.
    """
    def __init__(self, patch_size: int = 64, stride: int = 64):
        """
        Parameters
        ----------
        patch_size : int
            Size of the square patch (e.g., 64 or 224).
        stride : int
            Stride between patches. If stride < patch_size, patches will overlap.
        """
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, image_path_or_img: Any) -> Generator[Dict[str, Any], None, None]:
        """
        Extracts patches and yields each patch PIL image along with bounding box coordinates.
        
        Parameters
        ----------
        image_path_or_img : str or PIL.Image.Image
            The large image source.
            
        Yields
        ------
        dict
            Contains 'image' (PIL.Image.Image), 'bbox' (x, y, w, h), and 'index' (x_idx, y_idx).
        """
        if isinstance(image_path_or_img, (str, Path)):
            image = Image.open(image_path_or_img).convert("RGB")
        else:
            image = image_path_or_img.convert("RGB")
            
        width, height = image.size

        y_idx = 0
        for y in range(0, height - self.patch_size + 1, self.stride):
            x_idx = 0
            for x in range(0, width - self.patch_size + 1, self.stride):
                box = (x, y, x + self.patch_size, y + self.patch_size)
                patch = image.crop(box)
                
                yield {
                    "image": patch,
                    "bbox": (x, y, self.patch_size, self.patch_size),
                    "index": (x_idx, y_idx)
                }
                x_idx += 1
            y_idx += 1

    def get_grid_dimensions(self, image_path_or_img: Any) -> Tuple[int, int]:
        """
        Computes the grid layout (nx, ny) of patches.
        """
        if isinstance(image_path_or_img, (str, Path)):
            image = Image.open(image_path_or_img)
        else:
            image = image_path_or_img
            
        width, height = image.size
        
        nx = (width - self.patch_size) // self.stride + 1
        ny = (height - self.patch_size) // self.stride + 1
        
        return nx, ny

    def reconstruct_image(
        self, 
        patches: List[np.ndarray], 
        bboxes: List[Tuple[int, int, int, int]], 
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Stitches patches back into a single image. Handles overlapping regions by averaging.
        
        Parameters
        ----------
        patches : list of np.ndarray
            List of patch arrays of shape (H_patch, W_patch, C).
        bboxes : list of Tuple[int, int, int, int]
            List of (x, y, w, h) bounding boxes.
        original_size : Tuple[int, int]
            Size of the original image as (width, height).
            
        Returns
        -------
        np.ndarray
            Reconstructed image of shape (height, width, C).
        """
        width, height = original_size
        first_patch = patches[0]
        channels = first_patch.shape[2] if len(first_patch.shape) > 2 else 1
        
        recon_shape = (height, width, channels) if channels > 1 else (height, width)
        recon_img = np.zeros(recon_shape, dtype=np.float32)
        count_img = np.zeros((height, width), dtype=np.float32)
        
        for patch, bbox in zip(patches, bboxes):
            x, y, w, h = bbox
            
            # Ensure patch doesn't have trailing singleton channel dim if channels == 1
            if len(patch.shape) == 3 and patch.shape[2] == 1:
                patch_data = patch.squeeze(axis=2)
            else:
                patch_data = patch
                
            # Stitch patch
            if channels > 1:
                recon_img[y:y+h, x:x+w, :] += patch_data
            else:
                recon_img[y:y+h, x:x+w] += patch_data
                
            count_img[y:y+h, x:x+w] += 1.0
            
        # Normalize overlapping regions
        count_mask = count_img > 0
        if channels > 1:
            for c in range(channels):
                recon_img[:, :, c][count_mask] /= count_img[count_mask]
        else:
            recon_img[count_mask] /= count_img[count_mask]
            
        return np.clip(recon_img, 0, 255).astype(np.uint8)
from pathlib import Path
