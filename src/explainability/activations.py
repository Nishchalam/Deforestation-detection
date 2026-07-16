"""
activations.py

Analyzes channel activations, activation histograms, and dead filters.
"""

import numpy as np
from typing import Dict, Any

def analyze_layer_activations(feature_map: np.ndarray, threshold: float = 1e-6) -> Dict[str, Any]:
    """
    Computes activation profiles and dead filter ratios.
    """
    channels = feature_map.shape[0]
    
    # Calculate channel means
    channel_means = np.mean(feature_map, axis=(1, 2))
    
    # Dead filters: channels where all elements are less than the threshold
    dead_filters = []
    for c in range(channels):
        if np.max(np.abs(feature_map[c])) < threshold:
            dead_filters.append(c)
            
    dead_ratio = len(dead_filters) / channels
    
    return {
        "channel_means": channel_means.tolist(),
        "dead_filter_indices": dead_filters,
        "dead_filter_ratio": round(dead_ratio, 4),
        "total_channels": channels
    }
