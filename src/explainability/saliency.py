"""
saliency.py

Implements Vanilla Saliency and Input*Gradient explanation maps.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class SaliencyMap:
    """
    Generates saliency attributions indicating pixel-level sensitivity.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def generate_saliency(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Vanilla Saliency map (max absolute value of gradients across color channels).
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())
            
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Extract gradient
        gradient = input_tensor.grad.cpu().data.numpy()[0]
        # Max of absolute gradients across channels
        saliency = np.max(np.abs(gradient), axis=0)
        
        # Scale to [0, 1]
        if np.max(saliency) > 0:
            saliency = saliency / np.max(saliency)
            
        return saliency

    def generate_input_gradient(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Input * Gradient attribution map.
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())
            
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        gradient = input_tensor.grad.cpu().data.numpy()[0]
        input_np = input_tensor.cpu().data.numpy()[0]
        
        input_grad = input_np * gradient
        input_grad = np.max(np.abs(input_grad), axis=0)
        
        if np.max(input_grad) > 0:
            input_grad = input_grad / np.max(input_grad)
            
        return input_grad
