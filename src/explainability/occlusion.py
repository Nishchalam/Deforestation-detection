"""
occlusion.py

Implements sliding-window occlusion sensitivity analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class OcclusionSensitivity:
    """
    Measures drop in class confidence when parts of the image are masked (occluded).
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def generate_sensitivity_map(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        box_size: int = 32,
        stride: int = 16,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Generates sensitivity heatmap of shape (H, W).
        """
        self.model.eval()
        device = input_tensor.device
        
        # Forward pass to get target class if not provided
        with torch.no_grad():
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(torch.argmax(output, dim=1).item())
            baseline_prob = torch.softmax(output, dim=1)[0, class_idx].item()
            
        _, _, height, width = input_tensor.shape
        sensitivity_map = np.zeros((height, width), dtype=np.float32)
        visit_counts = np.zeros((height, width), dtype=np.float32)
        
        # Slide window
        for y in range(0, height - box_size + 1, stride):
            for x in range(0, width - box_size + 1, stride):
                # Clone and occlude
                occluded_tensor = input_tensor.clone()
                occluded_tensor[0, :, y:y+box_size, x:x+box_size] = fill_value
                
                with torch.no_grad():
                    output_occ = self.model(occluded_tensor)
                    prob_occ = torch.softmax(output_occ, dim=1)[0, class_idx].item()
                    
                # Sensitivity = baseline - probability under occlusion (higher = drop in confidence)
                drop = baseline_prob - prob_occ
                
                sensitivity_map[y:y+box_size, x:x+box_size] += drop
                visit_counts[y:y+box_size, x:x+box_size] += 1
                
        # Average drops
        count_mask = visit_counts > 0
        sensitivity_map[count_mask] /= visit_counts[count_mask]
        
        # Clamp negative values (occlusion *improved* confidence) to zero
        sensitivity_map = np.maximum(sensitivity_map, 0)
        
        # Scale to [0, 1]
        if np.max(sensitivity_map) > 0:
            sensitivity_map = sensitivity_map / np.max(sensitivity_map)
            
        return sensitivity_map
