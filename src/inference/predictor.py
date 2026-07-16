"""
predictor.py

Handles loading a trained model checkpoint and running inference on patches,
extracting predictions, softmax probabilities, and confidence scores.
"""

import torch
import torch.nn as nn
from typing import List, Union, Dict, Any, Tuple
from PIL import Image

from src.data.transforms import test_transform

class LandCoverPredictor:
    """
    Predictor class for satellite patch classifications.
    """
    def __init__(
        self, 
        model: nn.Module, 
        checkpoint_path: str, 
        device: torch.device = None
    ):
        """
        Parameters
        ----------
        model : nn.Module
            Instantiated PyTorch neural network model.
        checkpoint_path : str
            Path to model checkpoints.
        device : torch.device, optional
            Device to run inference on.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        
        self.transform = test_transform
        
        self.classes = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
            "Industrial", "Pasture", "PermanentCrop", "Residential",
            "River", "SeaLake"
        ]

    def _load_checkpoint(self, checkpoint_path: str):
        """Loads weight parameters into model."""
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
        """
        Predicts classes for the input image or batch.
        """
        classes, _, _ = self.predict_detailed(images)
        return classes

    @torch.no_grad()
    def predict_detailed(
        self, 
        images: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[List[str], List[float], List[List[float]]]:
        """
        Runs detailed inference.
        
        Returns
        -------
        Tuple[List[str], List[float], List[List[float]]]
            - Predicted class labels
            - Maximum confidence scores (softmax probabilities)
            - Full class probability distributions
        """
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
