"""
validator.py

Orchestrates the validation pipeline: compares predictions against ground truths,
computes statistics, generates plots, and exports standard validation reports.
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.validation.metrics import compute_binary_validation_metrics, compute_multiclass_validation_metrics
from src.validation.statistics import compute_area_statistics, compute_confidence_statistics
from src.validation.visualization import (
    plot_validation_confusion_matrix, 
    plot_transition_matrix_heatmap, 
    plot_confidence_histogram,
    plot_forest_area_pie,
    plot_landcover_distribution_bars
)
from src.validation.report import export_validation_reports

class DeforestationPipelineValidator:
    """
    Validates land-cover mapping and change detection masks against ground truths.
    """
    def __init__(self, classes: List[str] = None, patch_size_m: float = 64.0):
        self.classes = classes or [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
            "Industrial", "Pasture", "PermanentCrop", "Residential",
            "River", "SeaLake"
        ]
        self.patch_size_m = patch_size_m

    def validate_classification(
        self,
        y_true: List[str],
        y_pred: List[str],
        confidences: List[float],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Validates multiclass classification predictions.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Metrics
        metrics = compute_multiclass_validation_metrics(y_true, y_pred, self.classes)
        
        # 2. Statistics
        stats = compute_area_statistics(y_true, y_pred, self.patch_size_m)
        conf_stats = compute_confidence_statistics(confidences)
        stats["confidence_summary"] = conf_stats
        
        # 3. Plots
        cm_array = np.array(metrics["confusion_matrix"])
        plot_validation_confusion_matrix(cm_array, self.classes, output_dir / "confusion_matrix.png")
        plot_confidence_histogram(confidences, output_dir / "confidence_histogram.png")
        plot_landcover_distribution_bars(stats, self.classes, output_dir / "landcover_distribution.png")
        
        forest_before = stats["area_before_ha"].get("Forest", 0.0)
        forest_after = stats["area_after_ha"].get("Forest", 0.0)
        plot_forest_area_pie(forest_before, forest_after, output_dir / "forest_area_comparison.png")
        
        # 4. Export Reports
        export_validation_reports(metrics, stats, self.classes, output_dir)
        
        return {
            "metrics": metrics,
            "statistics": stats
        }

    def validate_deforestation_mask(
        self,
        pred_mask: List[int],
        gt_mask: List[int],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Validates binary deforestation masks.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics = compute_binary_validation_metrics(gt_mask, pred_mask)
        
        # Export binary mask report
        with open(output_dir / "deforestation_metrics.json", "w") as f:
            import json
            json.dump(metrics, f, indent=4)
            
        return metrics
