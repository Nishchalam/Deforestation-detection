import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

def plot_validation_confusion_matrix(cm: np.ndarray, classes: List[str], path: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_transition_matrix_heatmap(matrix: np.ndarray, classes: List[str], path: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Oranges", xticklabels=classes, yticklabels=classes)
    plt.title("Land Cover Transition Matrix")
    plt.ylabel("Year A Class")
    plt.xlabel("Year B Class")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_confidence_histogram(confidences: List[float], path: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
    plt.title("Prediction Confidence Histogram")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_forest_area_comparison(forest_before: float, forest_after: float, path: Path):
    """Plots a simple bar chart comparing forest area before and after."""
    plt.figure(figsize=(6, 5))
    labels = ['Year A', 'Year B']
    areas = [forest_before, forest_after]
    colors = ['green', 'forestgreen']
    
    plt.bar(labels, areas, color=colors, edgecolor='black', width=0.5)
    plt.title("Forest Area Comparison (ha)")
    plt.ylabel("Area (hectares)")
    for i, v in enumerate(areas):
        plt.text(i, v + (max(areas)*0.01), f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def export_validation_reports(
    metrics: Dict[str, Any],
    classes: List[str],
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Metrics JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Generate validation_report.md
    md_lines = [
        "# Deforestation Detection Validation Report\n\n",
        "This automated report summarizes the performance of the change detection pipeline against ground-truth validation data.\n\n",
        "## 📊 Quantitative Metrics\n\n",
        "| Metric | Value |\n",
        "| :--- | :---: |\n"
    ]
    
    for key, val in metrics.items():
        if key != "change_matrix":
            md_lines.append(f"| {key.replace('_', ' ').title()} | {val} |\n")
            
    if "change_matrix" in metrics:
        cm = metrics["change_matrix"]
        md_lines.extend([
            "\n## 📈 Change Matrix (Patches)\n\n",
            f"* **True Positives (Deforested)**: {cm.get('true_positives_patches', 0)}\n",
            f"* **False Positives**: {cm.get('false_positives_patches', 0)}\n",
            f"* **False Negatives**: {cm.get('false_negatives_patches', 0)}\n",
            f"* **True Negatives (Stable)**: {cm.get('true_negatives_patches', 0)}\n"
        ])
        
    with open(output_dir / "validation_report.md", "w") as f:
        f.writelines(md_lines)
        
    print(f"Validation reports exported to {output_dir}")
