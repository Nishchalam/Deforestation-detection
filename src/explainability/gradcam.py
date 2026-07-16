"""
gradcam.py

Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for PyTorch models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional

class GradCAM:
    """
    Computes Grad-CAM heatmaps for any target Conv2D layer in a PyTorch model.
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
        Generates Grad-CAM heatmap for the given input and target class.
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
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to keep only features that positively influence the target class
        cam = np.maximum(cam, 0)
        
        # Scale to [0, 1]
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        # Resize to match input image size (224x224)
        cam = cv2.resize(cam, (224, 224))
        return cam

    def overlay_heatmap(self, image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlays heatmap on original image array.
        """
        # Apply jet colormap
        colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        blended = alpha * colored_heatmap + (1 - alpha) * image_np
        return np.clip(blended, 0, 255).astype(np.uint8)
