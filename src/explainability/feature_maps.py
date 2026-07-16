"""
feature_maps.py

Extracts and visualizes intermediate convolutional feature maps.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

class FeatureMapExtractor:
    """
    Extracts forward activation tensors from specified layer modules.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = None
        self.hook = None

    def _hook_fn(self, module, input, output):
        self.activations = output.detach().cpu()

    def extract_features(self, input_tensor: torch.Tensor, target_layer: nn.Module) -> np.ndarray:
        """
        Runs a forward pass and captures feature maps.
        """
        self.model.eval()
        self.hook = target_layer.register_forward_hook(self._hook_fn)
        
        with torch.no_grad():
            self.model(input_tensor)
            
        self.hook.remove()
        
        # Return activations as numpy array (C, H, W)
        return self.activations[0].numpy()
