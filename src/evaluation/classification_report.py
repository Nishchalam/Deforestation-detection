"""
classification_report.py

Generates classification reports.
"""

from sklearn.metrics import classification_report
import numpy as np
from typing import List, Optional

def generate_classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_names: Optional[List[str]] = None
) -> str:
    """
    Computes a text report showing the main classification metrics.
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
