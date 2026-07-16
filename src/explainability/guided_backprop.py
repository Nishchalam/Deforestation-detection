"""
guided_backprop.py

Implements Guided Backpropagation for PyTorch models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class GuidedBackprop:
    """
    Computes Guided Backpropagation attributions by registering hooks on ReLU layers.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        
        # Register hooks on all ReLU layers to modify backward behavior
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(module.register_full_backward_hook(self._guided_relu_backward))

    def _guided_relu_backward(self, module, grad_in, grad_out):
        """
        Only backpropagates positive gradients associated with positive activations.
        """
        # grad_in[0] is the gradient w.r.t input of the ReLU block
        # grad_out[0] is the gradient w.r.t output of the ReLU block
        grad = grad_out[0]
        positive_grad = torch.clamp(grad, min=0.0)
        return (positive_grad,)

    def generate_gradients(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates guided gradients for the target class w.r.t input image.
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())
            
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Extract gradients from input image
        gradients = input_tensor.grad.cpu().data.numpy()[0]
        # Transpose to HWC shape
        gradients = np.transpose(gradients, (1, 2, 0))
        
        # Normalize to [0, 1] for visualization
        gradients = gradients - np.min(gradients)
        if np.max(gradients) > 0:
            gradients = gradients / np.max(gradients)
            
        return gradients

    def remove_hooks(self):
        """Removes the ReLU hooks."""
        for hook in self.hooks:
            hook.remove()
