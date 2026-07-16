"""
metrics.py

Calculates performance metrics (Accuracy, IoU, Precision, Recall) for change detection
when compared to a ground-truth change mask.
"""

from typing import List
import numpy as np

def calculate_change_metrics(pred_mask: List[int], gt_mask: List[int]) -> dict:
    """
    Computes scores compared to a ground truth binary change mask.
    """
    y_pred = np.array(pred_mask)
    y_true = np.array(gt_mask)
    
    if len(y_pred) != len(y_true):
        raise ValueError("Predicted and ground-truth masks must be of equal length.")
        
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    
    # Intersection over Union (IoU) of changed class
    iou = tp / max(1, tp + fp + fn)
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "iou": round(iou, 4),
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }
    }
