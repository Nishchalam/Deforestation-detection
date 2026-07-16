"""
lime.py

Implements a self-contained LIME (Local Interpretable Model-agnostic Explanations) simulator
using grid-based perturbations, target probability sampling, and linear least-squares.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional

class LIMESimulator:
    """
    Performs grid-based perturb-and-predict attribution modeling from first principles.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def generate_lime_attribution(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        grid_dim: int = 4,
        num_perturbations: int = 100,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Fits a local linear surrogate model on random perturbation samples to measure block importances.
        """
        self.model.eval()
        device = input_tensor.device
        
        # Forward pass to get baseline target class
        with torch.no_grad():
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(torch.argmax(output, dim=1).item())
                
        num_superpixels = grid_dim * grid_dim
        
        # Generate random binary perturbations
        perturbations = np.random.randint(0, 2, size=(num_perturbations, num_superpixels))
        
        # Keep original image as first perturbation
        perturbations[0, :] = 1
        
        _, _, height, width = input_tensor.shape
        block_h = height // grid_dim
        block_w = width // grid_dim
        
        target_probs = []
        
        # Run forward pass for each perturbed input
        for p in perturbations:
            perturbed_tensor = input_tensor.clone()
            
            # Apply mask
            for idx in range(num_superpixels):
                if p[idx] == 0:
                    r = idx // grid_dim
                    c = idx % grid_dim
                    perturbed_tensor[0, :, r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w] = fill_value
                    
            with torch.no_grad():
                out = self.model(perturbed_tensor)
                prob = torch.softmax(out, dim=1)[0, class_idx].item()
                target_probs.append(prob)
                
        # Solve least-squares to find weights: perturbations * weights = target_probs
        # Ridge regression formulation: w = (X^T X + alpha*I)^(-1) X^T y
        X = perturbations
        y = np.array(target_probs)
        
        # Standardize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-6
        X_norm = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_norm = y - y_mean
        
        # Fit Ridge regression
        alpha = 1.0
        XTX = np.dot(X_norm.T, X_norm)
        I = np.eye(num_superpixels)
        weights = np.dot(np.linalg.inv(XTX + alpha * I), np.dot(X_norm.T, y_norm))
        
        # Generate importance map
        importance_map = np.zeros((grid_dim, grid_dim), dtype=np.float32)
        for idx in range(num_superpixels):
            r = idx // grid_dim
            c = idx % grid_dim
            importance_map[r, c] = weights[idx]
            
        # Keep positive evidence and normalize
        importance_map = np.maximum(importance_map, 0)
        if np.max(importance_map) > 0:
            importance_map = importance_map / np.max(importance_map)
            
        # Resize to input dimensions (224x224)
        importance_map = cv2.resize(importance_map, (height, width), interpolation=cv2.INTER_NEAREST)
        return importance_map
