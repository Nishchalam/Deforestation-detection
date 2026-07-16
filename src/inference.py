import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Generator, Tuple, Dict, Any, List, Union, Optional

from src.preprocessing import test_transform

class PatchGenerator:
    """
    Extracts patches from a large image using a sliding window and stitches them back.
    """
    def __init__(self, patch_size: int = 64, stride: int = 64):
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, image_path_or_img: Any) -> Generator[Dict[str, Any], None, None]:
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
        width, height = original_size
        first_patch = patches[0]
        channels = first_patch.shape[2] if len(first_patch.shape) > 2 else 1
        
        recon_shape = (height, width, channels) if channels > 1 else (height, width)
        recon_img = np.zeros(recon_shape, dtype=np.float32)
        count_img = np.zeros((height, width), dtype=np.float32)
        
        for patch, bbox in zip(patches, bboxes):
            x, y, w, h = bbox
            
            if len(patch.shape) == 3 and patch.shape[2] == 1:
                patch_data = patch.squeeze(axis=2)
            else:
                patch_data = patch
                
            if channels > 1:
                recon_img[y:y+h, x:x+w, :] += patch_data
            else:
                recon_img[y:y+h, x:x+w] += patch_data
                
            count_img[y:y+h, x:x+w] += 1.0
            
        count_mask = count_img > 0
        if channels > 1:
            for c in range(channels):
                recon_img[:, :, c][count_mask] /= count_img[count_mask]
        else:
            recon_img[count_mask] /= count_img[count_mask]
            
        return np.clip(recon_img, 0, 255).astype(np.uint8)


class LandCoverPredictor:
    """
    Predictor class for satellite patch classifications.
    """
    def __init__(
        self, 
        model: nn.Module, 
        checkpoint_path: Optional[str] = None, 
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
            
        self.model.eval()
        self.transform = test_transform
        
        self.classes = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
            "Industrial", "Pasture", "PermanentCrop", "Residential",
            "River", "SeaLake"
        ]

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

    @torch.no_grad()
    def predict(self, images: Union[Image.Image, List[Image.Image]]) -> List[str]:
        classes, _, _ = self.predict_detailed(images)
        return classes

    @torch.no_grad()
    def predict_detailed(
        self, 
        images: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[List[str], List[float], List[List[float]]]:
        if not isinstance(images, list):
            images = [images]
            
        tensor_batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        outputs = self.model(tensor_batch)
        probabilities = torch.softmax(outputs, dim=1)
        
        confidences, predictions = probabilities.max(dim=1)
        
        pred_classes = [self.classes[p.item()] for p in predictions]
        conf_scores = [float(c.item()) for c in confidences]
        prob_dist = [p.cpu().tolist() for p in probabilities]
        
        return pred_classes, conf_scores, prob_dist


class LandCoverMapper:
    """
    Coordinates patching, predictions, color mapping, and reconstruction of land cover maps.
    """
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
        self.predictor = predictor
        self.patch_size = patch_size
        self.stride = stride
        self.patch_generator = PatchGenerator(patch_size=patch_size, stride=stride)

    def generate_map(self, image_path_or_img: Any, batch_size: int = 32) -> Dict[str, Any]:
        if isinstance(image_path_or_img, (str, Path)):
            original_img = Image.open(image_path_or_img).convert("RGB")
        else:
            original_img = image_path_or_img.convert("RGB")
            
        width, height = original_img.size
        
        patches = []
        bboxes = []
        
        for patch_data in self.patch_generator.extract_patches(original_img):
            patches.append(patch_data["image"])
            bboxes.append(patch_data["bbox"])
            
        predictions = []
        confidences = []
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            preds, confs, _ = self.predictor.predict_detailed(batch_patches)
            predictions.extend(preds)
            confidences.extend(confs)
            
        color_patches = []
        conf_patches = []
        
        for pred, conf in zip(predictions, confidences):
            color = self.COLOR_MAP.get(pred, [0, 0, 0])
            cp = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            cp[:, :, :] = color
            color_patches.append(cp)
            
            cfp = np.zeros((self.patch_size, self.patch_size, 1), dtype=np.uint8)
            cfp[:, :, 0] = int(conf * 255)
            conf_patches.append(cfp)
            
        recon_color = self.patch_generator.reconstruct_image(color_patches, bboxes, (width, height))
        recon_conf = self.patch_generator.reconstruct_image(conf_patches, bboxes, (width, height))
        
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
