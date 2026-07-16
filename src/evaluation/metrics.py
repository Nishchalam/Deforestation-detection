"""
metrics.py

Functions for evaluating model performance.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    return precision_score(y_true, y_pred, average=average, zero_division=0)

def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    return recall_score(y_true, y_pred, average=average, zero_division=0)

def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns a dictionary of standard classification metrics.
    """
    accuracy = calculate_accuracy(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "top1_accuracy": accuracy,
        "precision": calculate_precision(y_true, y_pred),
        "recall": calculate_recall(y_true, y_pred),
        "f1": calculate_f1(y_true, y_pred)
    }
