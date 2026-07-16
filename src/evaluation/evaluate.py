import time
import torch
import numpy as np
from typing import Dict, Any, Tuple
from src.evaluation.metrics import evaluate_all

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
    metrics = evaluate_all(y_true, y_pred)
    metrics["loss"] = running_loss / max(1, total_images)
    metrics["inference_time_per_image"] = total_time / max(1, total_images)
    metrics["images_per_second"] = total_images / max(0.0001, total_time)
    metrics["total_inference_time"] = total_time
    
    return metrics, y_true, y_pred
