"""
landcover_mapper.py

Maps an entire Sentinel-2 image to land cover classes using 
the PatchGenerator and LandCoverPredictor.
"""

from typing import Dict, Any, List
import numpy as np
from PIL import Image

from src.inference.predictor import LandCoverPredictor
from src.inference.patch_generator import PatchGenerator


class LandCoverMapper:
    """
    Coordinates the patching and prediction to generate a 
    full land-cover map.
    """
    def __init__(self, predictor: LandCoverPredictor, patch_size: int = 64):
        """
        Parameters
        ----------
        predictor : LandCoverPredictor
            Initialized predictor model.
        patch_size : int
            Size of patches to extract.
        """
        self.predictor = predictor
        self.patch_generator = PatchGenerator(
            patch_size=patch_size, 
            stride=patch_size  # Non-overlapping patches
        )

    def generate_map(self, image_path: str, batch_size: int = 32) -> Dict[str, Any]:
        """
        Generates a land-cover map for the given image.
        
        Parameters
        ----------
        image_path : str
            Path to the large Sentinel-2 image.
        batch_size : int
            Batch size for inference.
            
        Returns
        -------
        dict
            Contains:
            - 'classes': list of predicted classes for each patch
            - 'bboxes': list of bounding boxes for each patch
            - 'dimensions': (nx, ny) grid dimensions
        """
        patches = []
        bboxes = []
        predictions = []

        for patch_data in self.patch_generator.extract_patches(image_path):
            patches.append(patch_data["image"])
            bboxes.append(patch_data["bbox"])

            if len(patches) == batch_size:
                preds = self.predictor.predict(patches)
                predictions.extend(preds)
                patches = []

        # Process remaining patches
        if len(patches) > 0:
            preds = self.predictor.predict(patches)
            predictions.extend(preds)

        nx, ny = self.patch_generator.get_grid_dimensions(image_path)

        return {
            "classes": predictions,
            "bboxes": bboxes,
            "dimensions": (nx, ny)
        }
