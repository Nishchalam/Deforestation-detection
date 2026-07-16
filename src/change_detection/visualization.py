"""
visualization.py

Generates publication-quality comparison dashboards displaying original frames,
classification maps, change differences, and binary overlays.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from typing import Dict, List

def generate_change_visualization_dashboard(
    original_a: Image.Image,
    original_b: Image.Image,
    map_a: Image.Image,
    map_b: Image.Image,
    binary_mask: Image.Image,
    overlay_img: Image.Image,
    color_map: Dict[str, List[int]],
    save_path: Path
):
    """
    Saves a 2x3 panel dashboard illustrating temporal change details.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Year A
    axes[0, 0].imshow(original_a)
    axes[0, 0].set_title("Original (Year A)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(map_a)
    axes[0, 1].set_title("Land Cover Map (Year A)")
    axes[0, 1].axis("off")
    
    # Legend panel
    legend_patches = []
    for class_name, rgb in sorted(color_map.items()):
        color_hex = [c / 255.0 for c in rgb]
        patch = mpatches.Patch(color=color_hex, label=class_name)
        legend_patches.append(patch)
    axes[0, 2].legend(handles=legend_patches, loc="center", frameon=False, fontsize=10)
    axes[0, 2].set_title("Class Color Legend")
    axes[0, 2].axis("off")
    
    # Row 2: Year B
    axes[1, 0].imshow(original_b)
    axes[1, 0].set_title("Original (Year B)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(map_b)
    axes[1, 1].set_title("Land Cover Map (Year B)")
    axes[1, 1].axis("off")
    
    # Deforestation overlay
    axes[1, 2].imshow(overlay_img)
    axes[1, 2].set_title("Deforestation Overlay (Forest -> Non-Forest)")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def generate_binary_mask_plot(binary_mask: Image.Image, save_path: Path):
    """
    Plots the binary deforestation mask.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Binary Deforestation Mask (1=Deforested)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
