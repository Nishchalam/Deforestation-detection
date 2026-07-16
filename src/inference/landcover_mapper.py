"""
landcover_mapper.py

Generates full land-cover classification maps and confidence scores from Sentinel-2 imagery
using the PatchGenerator and LandCoverPredictor.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

from src.inference.predictor import LandCoverPredictor
from src.inference.patch_generator import PatchGenerator

class LandCoverMapper:
    """
    Coordinates patching, predictions, color mapping, and reconstruction of land cover maps.
    """
    # Color definition for 10 EuroSAT classes
    COLOR_MAP = {
        "AnnualCrop": [240, 150, 150],            # Light Red/Pink
        "Forest": [0, 128, 0],                     # Dark Green
        "HerbaceousVegetation": [150, 240, 150],   # Light Green
        "Highway": [128, 128, 128],                # Gray
        "Industrial": [255, 0, 0],                 # Red
        "Pasture": [255, 255, 0],                  # Yellow
        "PermanentCrop": [200, 100, 50],           # Brown
        "Residential": [255, 165, 0],              # Orange
        "River": [0, 0, 255],                      # Blue
        "SeaLake": [0, 255, 255]                   # Cyan
    }

    def __init__(self, predictor: LandCoverPredictor, patch_size: int = 64, stride: int = 64):
        """
        Parameters
        ----------
        predictor : LandCoverPredictor
            Initialized model predictor.
        patch_size : int
            Patch width/height.
        stride : int
            Stride size.
        """
        self.predictor = predictor
        self.patch_size = patch_size
        self.stride = stride
        self.patch_generator = PatchGenerator(patch_size=patch_size, stride=stride)

    def generate_map(self, image_path: str, batch_size: int = 32) -> Dict[str, Any]:
        """
        Runs patch classification across the image and reconstructs the full maps.
        
        Returns
        -------
        Dict[str, Any]
            - 'classes': list of predicted classes
            - 'confidences': list of max confidence scores
            - 'bboxes': list of bounding boxes
            - 'prediction_map': PIL.Image.Image (colored segmentation map)
            - 'confidence_map': PIL.Image.Image (grayscale confidence heatmap)
            - 'original_image': PIL.Image.Image (original source image)
        """
        original_img = Image.open(image_path).convert("RGB")
        width, height = original_img.size
        
        patches = []
        bboxes = []
        
        # Extract patches
        for patch_data in self.patch_generator.extract_patches(original_img):
            patches.append(patch_data["image"])
            bboxes.append(patch_data["bbox"])
            
        # Predict in batches
        predictions = []
        confidences = []
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            preds, confs, _ = self.predictor.predict_detailed(batch_patches)
            predictions.extend(preds)
            confidences.extend(confs)
            
        # Reconstruct color map patches
        color_patches = []
        conf_patches = []
        
        for pred, conf in zip(predictions, confidences):
            color = self.COLOR_MAP.get(pred, [0, 0, 0])
            # Create color patch
            cp = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            cp[:, :, :] = color
            color_patches.append(cp)
            
            # Create confidence patch (scale 0-1 to 0-255 grayscale)
            cfp = np.zeros((self.patch_size, self.patch_size, 1), dtype=np.uint8)
            cfp[:, :, 0] = int(conf * 255)
            conf_patches.append(cfp)
            
        # Stitch back
        recon_color = self.patch_generator.reconstruct_image(color_patches, bboxes, (width, height))
        recon_conf = self.patch_generator.reconstruct_image(conf_patches, bboxes, (width, height))
        
        # Convert to PIL
        prediction_map = Image.fromarray(recon_color)
        confidence_map = Image.fromarray(recon_conf.squeeze(), mode="L")
        
        return {
            "classes": predictions,
            "confidences": confidences,
            "bboxes": bboxes,
            "prediction_map": prediction_map,
            "confidence_map": confidence_map,
            "original_image": original_img
        }
