"""
visualization.py

Implements validation plotting routines: Confusion Matrices, Transition Matrix heatmaps,
histograms, pie charts, and before/after comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict

def plot_validation_confusion_matrix(cm: np.ndarray, target_names: List[str], save_path: Path):
    """Plots and saves the multiclass confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Validation Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_transition_matrix_heatmap(matrix: np.ndarray, classes: List[str], save_path: Path):
    """Plots the transition matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", xticklabels=classes, yticklabels=classes)
    plt.title("Class Transitions Heatmap (Year A -> Year B)")
    plt.ylabel("Class Year A")
    plt.xlabel("Class Year B")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confidence_histogram(confidences: List[float], save_path: Path):
    """Plots a histogram of prediction confidences."""
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(confidences), color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {np.mean(confidences):.2f}")
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_forest_area_pie(area_a: float, area_b: float, save_path: Path):
    """Plots a pie chart of forest area comparisons."""
    plt.figure(figsize=(6, 6))
    labels = ["Forest Year A", "Forest Year B"]
    sizes = [area_a, area_b]
    colors = ["forestgreen", "lightgreen"]
    
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, shadow=True)
    plt.title("Forest Area Canopy Comparison (Hectares)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_landcover_distribution_bars(
    area_stats: Dict[str, Dict[str, float]], 
    classes: List[str], 
    save_path: Path
):
    """Plots comparison bar charts of landcover areas before and after."""
    df_data = []
    for c in classes:
        df_data.append({
            "Class": c,
            "Year A (ha)": area_stats["area_before_ha"].get(c, 0.0),
            "Year B (ha)": area_stats["area_after_ha"].get(c, 0.0)
        })
        
    import pandas as pd
    df = pd.DataFrame(df_data)
    df.set_index("Class", inplace=True)
    
    ax = df.plot(kind="bar", figsize=(12, 6), width=0.8, color=["skyblue", "orange"])
    plt.title("Land Cover Class Distribution Comparison")
    plt.ylabel("Area (Hectares)")
    plt.xlabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
