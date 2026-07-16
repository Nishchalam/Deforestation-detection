"""
gradcamplusplus.py

Implements Grad-CAM++ (Generalized Class Activation Mapping) for PyTorch models.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional

class GradCAMPlusPlus:
    """
    Computes Grad-CAM++ heatmaps to improve localization and support multiple occurrences.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """Removes the forward and backward hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates Grad-CAM++ heatmap.
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())
            
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Calculate second-order and third-order gradients
        grads_power_2 = gradients ** 2
        grads_power_3 = gradients ** 3
        
        sum_activations = np.sum(activations, axis=(1, 2))
        
        # Compute alpha coefficients
        eps = 1e-7
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None, None] * grads_power_3 + eps)
        
        # Positive gradients check
        weights = np.sum(aij * np.maximum(gradients, 0), axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        cam = cv2.resize(cam, (224, 224))
        return cam
