"""
visualization.py

Plotting functions for training history, confusion matrices, and image predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import torch

def plot_loss(history: Dict[str, list], save_path: Optional[Path] = None):
    """Plots training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    if 'validation_loss' in history:
        plt.plot(history['validation_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_accuracy(history: Dict[str, list], save_path: Optional[Path] = None):
    """Plots training and validation accuracy."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    if 'validation_accuracy' in history:
        plt.plot(history['validation_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_lr(history: Dict[str, list], save_path: Optional[Path] = None):
    """Plots learning rate over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.get('learning_rate', []), label='Learning Rate', color='orange')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: Optional[Path] = None):
    """Plots a confusion matrix using seaborn."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def show_predictions(images: torch.Tensor, true_labels: torch.Tensor, preds: torch.Tensor, classes: List[str], num_images: int = 4):
    """Displays a grid of images with their predicted and true labels."""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(min(num_images, len(images))):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # Unnormalize if needed (assuming standard normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax = axes[i] if num_images > 1 else axes
        ax.imshow(img)
        title_color = "green" if true_labels[i] == preds[i] else "red"
        ax.set_title(f"True: {classes[true_labels[i]]}\nPred: {classes[preds[i]]}", color=title_color)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_misclassified(images: torch.Tensor, true_labels: torch.Tensor, preds: torch.Tensor, classes: List[str], num_images: int = 4):
    """Displays a grid of misclassified images."""
    misclassified_idx = torch.where(true_labels != preds)[0]
    if len(misclassified_idx) == 0:
        print("No misclassified images found in this batch.")
        return
        
    num_to_show = min(num_images, len(misclassified_idx))
    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 5))
    
    for i in range(num_to_show):
        idx = misclassified_idx[i]
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax = axes[i] if num_to_show > 1 else axes
        ax.imshow(img)
        ax.set_title(f"True: {classes[true_labels[idx]]}\nPred: {classes[preds[idx]]}", color="red")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
