import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Optional

def plot_loss_curves(history: dict, save_dir: Path):
    """Plots and saves the training and validation loss curves."""
    if "train_loss" not in history or "val_loss" not in history:
        return
        
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curves.png", dpi=150)
    plt.close()

def plot_accuracy_curves(history: dict, save_dir: Path):
    """Plots and saves the training and validation accuracy curves."""
    # Support both accuracy and train_accuracy keys
    train_acc = history.get("train_accuracy", history.get("accuracy", []))
    val_acc = history.get("val_accuracy", [])
    
    if not train_acc or not val_acc:
        return
        
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Val Accuracy", marker="o")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_curves.png", dpi=150)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, target_names: List[str], save_dir: Path):
    """Plots and saves the confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=target_names, 
        yticklabels=target_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()

def plot_misclassified_examples(
    images: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_names: List[str], 
    save_dir: Path,
    max_examples: int = 16
):
    """Plots and saves a grid of misclassified examples."""
    misclassified_indices = np.where(y_true != y_pred)[0]
    if len(misclassified_indices) == 0:
        return
        
    num_plots = min(max_examples, len(misclassified_indices))
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    
    for i in range(len(axes)):
        if i < num_plots:
            idx = misclassified_indices[i]
            img = images[idx]
            
            # Normalize image back to [0, 1] if normalized
            if img.min() < 0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img.transpose(1, 2, 0)
                img = img * std + mean
                img = np.clip(img, 0, 1)
            elif img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
                
            axes[i].imshow(img)
            true_label = target_names[y_true[idx]]
            pred_label = target_names[y_pred[idx]]
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color="red", fontsize=8)
            axes[i].axis("off")
        else:
            axes[i].axis("off")
            
    plt.tight_layout()
    plt.savefig(save_dir / "misclassified_examples.png", dpi=150)
    plt.close()

def plot_prediction_grid(
    images: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_names: List[str], 
    save_dir: Path,
    max_examples: int = 16
):
    """Plots and saves a prediction grid (highlighting correct/incorrect predictions)."""
    num_plots = min(max_examples, len(images))
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    
    for i in range(len(axes)):
        if i < num_plots:
            img = images[i]
            
            if img.min() < 0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img.transpose(1, 2, 0)
                img = img * std + mean
                img = np.clip(img, 0, 1)
            elif img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
                
            axes[i].imshow(img)
            true_label = target_names[y_true[i]]
            pred_label = target_names[y_pred[i]]
            is_correct = (y_true[i] == y_pred[i])
            color = "green" if is_correct else "red"
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=8)
            axes[i].axis("off")
        else:
            axes[i].axis("off")
            
    plt.tight_layout()
    plt.savefig(save_dir / "prediction_grid.png", dpi=150)
    plt.close()
