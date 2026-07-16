import time
import torch
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluates the model on a dataloader.
    
    Returns
    -------
    metrics : dict
        A dictionary containing average loss, classification metrics, and latency stats.
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Model predictions.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Track latency
    total_images = 0
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            batch_size = labels.size(0)
            total_images += batch_size
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "top1_accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "loss": round(running_loss / max(1, total_images), 4),
        "inference_time_per_image": round(total_time / max(1, total_images), 6),
        "images_per_second": round(total_images / max(0.0001, total_time), 2),
        "total_inference_time": round(total_time, 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics, y_true, y_pred
