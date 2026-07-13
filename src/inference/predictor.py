"""
predictor.py

Handles loading a trained classification model and running inference
on image patches.
"""

import torch
import torch.nn as nn
from typing import List, Union
from PIL import Image

from src.data.transforms import test_transform


class LandCoverPredictor:
    """
    Predicts land cover class for given image patches.
    """
    def __init__(
        self, 
        model: nn.Module, 
        checkpoint_path: str, 
        device: torch.device = None
    ):
        """
        Initialize the predictor.
        
        Parameters
        ----------
        model : nn.Module
            The model architecture.
        checkpoint_path : str
            Path to the saved weights.
        device : torch.device
            Device to run inference on.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.model.to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        self.transform = test_transform
        
        # Hardcoded classes for EuroSAT (can be externalized later)
        self.classes = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
            "Industrial", "Pasture", "PermanentCrop", "Residential",
            "River", "SeaLake"
        ]

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load weights from checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    @torch.no_grad()
    def predict(self, images: Union[Image.Image, List[Image.Image]]) -> List[str]:
        """
        Predict land cover class for a single image or a list of images.
        """
        if not isinstance(images, list):
            images = [images]

        # Apply transforms
        tensor_batch = torch.stack([self.transform(img) for img in images]).to(self.device)

        # Forward pass
        outputs = self.model(tensor_batch)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = probabilities.argmax(dim=1)

        return [self.classes[p.item()] for p in predictions]
