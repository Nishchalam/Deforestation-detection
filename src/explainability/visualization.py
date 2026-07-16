"""
visualization.py

Generates comparison dashboards of different Explainable AI visual attributions.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

def generate_explainability_dashboard(
    original_img: Image.Image,
    saliency_map: np.ndarray,
    occlusion_map: np.ndarray,
    gradcam_map: np.ndarray,
    gradcam_pp_map: np.ndarray,
    lime_map: np.ndarray,
    save_path: Path
):
    """
    Saves a 2x3 dashboard panel comparing all XAI attributions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Original
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    # 2. Saliency
    axes[0, 1].imshow(saliency_map, cmap="hot")
    axes[0, 1].set_title("Vanilla Saliency Map")
    axes[0, 1].axis("off")
    
    # 3. Occlusion
    axes[0, 2].imshow(occlusion_map, cmap="jet")
    axes[0, 2].set_title("Occlusion Sensitivity Map")
    axes[0, 2].axis("off")
    
    # 4. Grad-CAM
    axes[1, 0].imshow(gradcam_map, cmap="jet")
    axes[1, 0].set_title("Grad-CAM Activation")
    axes[1, 0].axis("off")
    
    # 5. Grad-CAM++
    axes[1, 1].imshow(gradcam_pp_map, cmap="jet")
    axes[1, 1].set_title("Grad-CAM++ Activation")
    axes[1, 1].axis("off")
    
    # 6. LIME simulation
    axes[1, 2].imshow(lime_map, cmap="RdYlGn")
    axes[1, 2].set_title("LIME Superpixel Importance")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
