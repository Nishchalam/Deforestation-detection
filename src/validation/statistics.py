"""
statistics.py

Computes forest areas, class distribution statistics, and transitions over time.
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple

def compute_area_statistics(
    classes_before: List[str],
    classes_after: List[str],
    patch_size_m: float = 64.0
) -> Dict[str, Any]:
    """
    Computes area measurements in hectares for each class before and after.
    """
    patch_area_ha = (patch_size_m * patch_size_m) / 10000.0
    
    count_before = Counter(classes_before)
    count_after = Counter(classes_after)
    
    all_classes = set(classes_before + classes_after)
    
    area_before = {}
    area_after = {}
    difference = {}
    
    for c in all_classes:
        a_before = count_before.get(c, 0) * patch_area_ha
        a_after = count_after.get(c, 0) * patch_area_ha
        area_before[c] = round(a_before, 2)
        area_after[c] = round(a_after, 2)
        difference[c] = round(a_after - a_before, 2)
        
    return {
        "area_before_ha": area_before,
        "area_after_ha": area_after,
        "difference_ha": difference
    }

def compute_confidence_statistics(confidences: List[float]) -> Dict[str, Any]:
    """
    Computes summary stats for model prediction confidences.
    """
    confs = np.array(confidences)
    if len(confs) == 0:
        return {}
        
    return {
        "mean": round(float(np.mean(confs)), 4),
        "std": round(float(np.std(confs)), 4),
        "min": round(float(np.min(confs)), 4),
        "max": round(float(np.max(confs)), 4),
        "median": round(float(np.median(confs)), 4),
        "percentile_25": round(float(np.percentile(confs, 25)), 4),
        "percentile_75": round(float(np.percentile(confs, 75)), 4)
    }
