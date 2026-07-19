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


def generate_demo_data(data_dir: str):
    """
    Generates realistic multitemporal demo images (1024x1024) by stitching EuroSAT patches.
    Saves 'sentinel2_2018.png' (Year A), 'sentinel2_2022.png' (Year B),
    and 'hansen_validation_mask.png' to the specified data_dir.
    """
    from PIL import Image
    import random
    
    # Ensure reproducible layout
    random.seed(42)
    
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    y1_file = data_dir_path / "sentinel2_2018.png"
    y2_file = data_dir_path / "sentinel2_2022.png"
    mask_file = data_dir_path / "hansen_validation_mask.png"
    
    # If files already exist, do nothing
    if y1_file.exists() and y2_file.exists() and mask_file.exists():
        return
        
    eurosat_dir = Path("data/raw/EuroSAT")
    if not eurosat_dir.exists():
        eurosat_dir = Path("../data/raw/EuroSAT")
    if not eurosat_dir.exists():
        raise FileNotFoundError("EuroSAT raw dataset not found. Please ensure it is downloaded at data/raw/EuroSAT.")
        
    # Helper to gather images from a class folder
    def get_class_images(class_name: str) -> List[Path]:
        folder = eurosat_dir / class_name
        return list(folder.glob("*.jpg"))
        
    forest_imgs = get_class_images("Forest")
    pasture_imgs = get_class_images("Pasture")
    river_imgs = get_class_images("River")
    highway_imgs = get_class_images("Highway")
    crop_imgs = get_class_images("AnnualCrop")
    
    if not (forest_imgs and pasture_imgs and river_imgs and highway_imgs and crop_imgs):
        raise ValueError("Missing some EuroSAT classes in data/raw/EuroSAT.")
        
    # Create 1024x1024 black canvases
    img_a = Image.new("RGB", (1024, 1024))
    img_b = Image.new("RGB", (1024, 1024))
    mask = Image.new("L", (1024, 1024), 0)
    
    grid_size = 16
    patch_size = 64
    
    # Grid coordinates of deforestation
    defor_x_range = range(2, 6) # columns 2 to 5 (inclusive)
    defor_y_range = range(2, 6) # rows 2 to 5 (inclusive)
    
    for y_idx in range(grid_size):
        for x_idx in range(grid_size):
            box = (x_idx * patch_size, y_idx * patch_size, (x_idx + 1) * patch_size, (y_idx + 1) * patch_size)
            
            # Select patch class for Year A
            if x_idx == 8:
                class_img_path = random.choice(river_imgs)
            elif y_idx == 8:
                class_img_path = random.choice(highway_imgs)
            elif y_idx < 8:
                class_img_path = random.choice(forest_imgs)
            else:
                class_img_path = random.choice(crop_imgs)
                
            patch_a = Image.open(class_img_path).resize((patch_size, patch_size))
            img_a.paste(patch_a, box)
            
            # Select patch class for Year B
            if y_idx in defor_y_range and x_idx in defor_x_range:
                # Deforestation: Forest becomes Pasture
                class_img_path_b = random.choice(pasture_imgs)
                mask_patch = Image.new("L", (patch_size, patch_size), 255)
            else:
                # Stable cell: same as Year A
                class_img_path_b = class_img_path
                mask_patch = Image.new("L", (patch_size, patch_size), 0)
                
            if class_img_path_b != class_img_path:
                patch_b = Image.open(class_img_path_b).resize((patch_size, patch_size))
            else:
                patch_b = patch_a
                
            img_b.paste(patch_b, box)
            mask.paste(mask_patch, box)
            
    img_a.save(y1_file)
    img_b.save(y2_file)
    mask.save(mask_file)
    print(f"Generated demo multitemporal dataset in '{data_dir}':")
    print(f"  - Year A: {y1_file}")
    print(f"  - Year B: {y2_file}")
    print(f"  - Hansen Loss Mask: {mask_file}")

