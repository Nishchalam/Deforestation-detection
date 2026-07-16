"""
statistics.py

Calculates area statistics, class distributions, and transition tables,
and exports reports to JSON, CSV, and txt formats.
"""

import json
import csv
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any

def calculate_forest_statistics(
    changes: List[Tuple[str, str, bool]],
    deforestation_mask: List[int],
    patch_size_m: float = 64.0
) -> Dict[str, Any]:
    """
    Computes summary metrics for forest loss and gain.
    
    Parameters
    ----------
    changes : list of tuples
        List of (class_a, class_b, is_changed) transitions.
    deforestation_mask : list of int
        Binary mask (1 = deforested, 0 = stable/other).
    patch_size_m : float
        Side length of one patch in meters (e.g. 64m for EuroSAT-style).
        
    Returns
    -------
    dict
        Forest area stats, stable/changed counts.
    """
    patch_area_ha = (patch_size_m * patch_size_m) / 10000.0 # Sq meters to Hectares
    
    classes_a = [c[0] for c in changes]
    classes_b = [c[1] for c in changes]
    
    forest_a_count = classes_a.count("Forest")
    forest_b_count = classes_b.count("Forest")
    
    forest_area_a_ha = forest_a_count * patch_area_ha
    forest_area_b_ha = forest_b_count * patch_area_ha
    
    deforested_count = sum(deforestation_mask)
    forest_loss_ha = deforested_count * patch_area_ha
    
    # Forest Gain: transition from Non-Forest to Forest
    gain_count = sum(1 for c_a, c_b, _ in changes if c_a != "Forest" and c_b == "Forest")
    forest_gain_ha = gain_count * patch_area_ha
    
    percentage_loss = (deforested_count / max(1, forest_a_count)) * 100.0
    
    changed_patches = sum(1 for _, _, is_changed in changes if is_changed)
    stable_patches = len(changes) - changed_patches
    
    dist_before = dict(Counter(classes_a))
    dist_after = dict(Counter(classes_b))
    
    return {
        "forest_patches_before": forest_a_count,
        "forest_patches_after": forest_b_count,
        "forest_area_before_ha": round(forest_area_a_ha, 2),
        "forest_area_after_ha": round(forest_area_b_ha, 2),
        "forest_loss_ha": round(forest_loss_ha, 2),
        "forest_gain_ha": round(forest_gain_ha, 2),
        "percentage_forest_loss": round(percentage_loss, 2),
        "changed_patches_count": changed_patches,
        "stable_patches_count": stable_patches,
        "class_distribution_before": dist_before,
        "class_distribution_after": dist_after
    }

def export_reports(
    stats: Dict[str, Any],
    transition_matrix: np.ndarray,
    classes: List[str],
    output_dir: Path
):
    """
    Saves statistics.json, transition_matrix.csv, and summary.txt.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save JSON
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    # 2. Save Transition Matrix CSV
    with open(output_dir / "transition_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["From/To"] + classes)
        for i, row in enumerate(transition_matrix):
            writer.writerow([classes[i]] + list(row))
            
    # 3. Save Summary txt
    summary_text = [
        "===================================================\n",
        "         DEFORESTATION DETECTION SUMMARY REPORT      \n",
        "===================================================\n\n",
        f"Forest Area Before (Year A): {stats['forest_area_before_ha']:.2f} ha ({stats['forest_patches_before']} patches)\n",
        f"Forest Area After (Year B):  {stats['forest_area_after_ha']:.2f} ha ({stats['forest_patches_after']} patches)\n",
        f"Deforestation Loss:          {stats['forest_loss_ha']:.2f} ha ({stats['forest_patches_after'] - stats['forest_patches_before'] + stats['forest_gain_ha']/0.4} patches approx)\n",
        f"Afforestation/Forest Gain:   {stats['forest_gain_ha']:.2f} ha\n",
        f"Percentage Forest Canopy Loss: {stats['percentage_forest_loss']:.2f}%\n\n",
        f"Total Stable Patches:        {stats['stable_patches_count']}\n",
        f"Total Changed Patches:       {stats['changed_patches_count']}\n\n",
        "===================================================\n"
    ]
    with open(output_dir / "summary.txt", "w") as f:
        f.writelines(summary_text)
