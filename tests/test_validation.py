import pytest
from pathlib import Path
import numpy as np
from src.change_detection import validate_deforestation
from src.utils import (
    plot_validation_confusion_matrix,
    plot_transition_matrix_heatmap,
    plot_confidence_histogram,
    plot_forest_area_comparison,
    export_validation_reports
)

def test_validate_deforestation_metrics():
    pred_mask = [1, 0, 1, 0]
    gt_mask = [1, 0, 0, 0]
    
    metrics = validate_deforestation(pred_mask, gt_mask, patch_size_m=64.0)
    
    # tp=1, fp=1, fn=0, tn=2
    # precision = 1/2 = 0.5
    # recall = 1/1 = 1.0
    # iou = 1/2 = 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["iou"] == 0.5
    assert metrics["change_matrix"]["true_positives_patches"] == 1

def test_plotting_and_reports(tmp_path):
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    
    # Test plotting functions
    cm = np.array([[10, 2], [1, 15]])
    classes = ["Forest", "Non-Forest"]
    plot_validation_confusion_matrix(cm, classes, output_dir / "confusion_matrix.png")
    assert (output_dir / "confusion_matrix.png").exists()
    
    trans_matrix = np.array([[8, 2], [0, 10]])
    plot_transition_matrix_heatmap(trans_matrix, classes, output_dir / "transition_heatmap.png")
    assert (output_dir / "transition_heatmap.png").exists()
    
    plot_confidence_histogram([0.9, 0.8, 0.95], output_dir / "conf_hist.png")
    assert (output_dir / "conf_hist.png").exists()
    
    plot_forest_area_comparison(12.5, 9.2, output_dir / "forest_comp.png")
    assert (output_dir / "forest_comp.png").exists()
    
    # Test export report
    metrics = {
        "iou": 0.8,
        "precision": 0.9,
        "recall": 0.85,
        "change_matrix": {
            "true_positives_patches": 5,
            "false_positives_patches": 1
        }
    }
    export_validation_reports(metrics, classes, output_dir)
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "validation_report.md").exists()
