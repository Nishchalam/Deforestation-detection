"""
metrics.py

Implements validation metrics calculations for land-cover classification and change detection,
including Balanced Accuracy, specificity, FPR, FNR, IoU, and Dice coefficients.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def compute_binary_validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes exhaustive binary classification metrics for change masks.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    # Calculate confusion elements
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    # Standard metrics
    sensitivity = tp / max(1, tp + fn) # Recall
    specificity = tn / max(1, tn + fp)
    
    precision = tp / max(1, tp + fp)
    recall = sensitivity
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    
    iou = tp / max(1, tp + fp + fn)
    dice = f1 # Dice coefficient is identical to F1 score in binary classification
    
    fpr = fp / max(1, tn + fp)
    fnr = fn / max(1, tp + fn)
    
    balanced_acc = (sensitivity + specificity) / 2.0
    overall_acc = (tp + tn) / max(1, tp + tn + fp + fn)
    
    return {
        "overall_accuracy": round(overall_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "iou": round(iou, 4),
        "dice_coefficient": round(dice, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }
    }

def compute_multiclass_validation_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> dict:
    """
    Computes multiclass statistics for land-cover classification comparisons.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    
    return {
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
