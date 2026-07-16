"""
visualization.py

Generates and saves visual outputs for land-cover mapping:
prediction overlay, confidence heatmap, and a color-coded class legend.
"""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

def create_prediction_overlay(
    original_img: Image.Image, 
    prediction_map: Image.Image, 
    alpha: float = 0.4
) -> Image.Image:
    """
    Blends the original satellite image with the prediction map.
    """
    orig_rgba = original_img.convert("RGBA")
    pred_rgba = prediction_map.convert("RGBA")
    
    # Perform alpha blending
    blended = Image.blend(orig_rgba, pred_rgba, alpha)
    return blended.convert("RGB")

def save_class_legend(color_map: Dict[str, List[int]], save_path: Path):
    """
    Generates and saves a legend panel explaining the class colors.
    """
    plt.figure(figsize=(5, 4))
    legend_patches = []
    
    for class_name, rgb in sorted(color_map.items()):
        color_hex = [c / 255.0 for c in rgb]
        patch = mpatches.Patch(color=color_hex, label=class_name)
        legend_patches.append(patch)
        
    plt.legend(handles=legend_patches, loc="center", frameon=False, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

def save_side_by_side_grid(
    original_img: Image.Image,
    prediction_map: Image.Image,
    overlay_img: Image.Image,
    confidence_map: Image.Image,
    save_path: Path
):
    """
    Creates a single dashboard grid of all 4 outputs and saves it.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Sentinel-2")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(prediction_map)
    axes[0, 1].set_title("Land Cover Map")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(overlay_img)
    axes[1, 0].set_title("Overlay (Alpha=0.4)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(confidence_map, cmap="gray")
    axes[1, 1].set_title("Confidence Heatmap")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
