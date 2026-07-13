"""
confusion_matrix.py

Generates confusion matrices.
"""

from sklearn.metrics import confusion_matrix
import numpy as np

def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)
