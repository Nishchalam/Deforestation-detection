"""
utils.py

Validation utility helpers.
"""

from typing import List
import numpy as np

def calculate_overall_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Calculates overall classification accuracy from lists."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    return float(np.mean(y_t == y_p))
