"""
utils.py

Helper functions for hook registrations, conv layer selections, and preprocessing.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Union

def find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Finds the last Conv2d layer in any PyTorch model.
    """
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    return None

def find_target_layer_by_name(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    """
    Retrieves a specific layer by module name.
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None

def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Resizes and normalizes PIL image to a PyTorch tensor batch.
    """
    img_resized = image.resize((224, 224))
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    
    # Normalize (ImageNet statistics)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_arr = (img_arr - mean) / std
    
    # Convert to HWC -> CHW -> BCHW
    img_tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor.requires_grad = True
    return img_tensor
