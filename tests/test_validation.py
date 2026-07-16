import pytest
from pathlib import Path
from src.validation import DeforestationPipelineValidator

def test_pipeline_validator(tmp_path):
    validator = DeforestationPipelineValidator(patch_size_m=100.0)
    
    y_true = ["Forest", "Forest", "Residential", "SeaLake"]
    y_pred = ["Forest", "AnnualCrop", "Residential", "SeaLake"]
    confidences = [0.9, 0.8, 0.95, 0.85]
    
    output_dir = tmp_path / "reports"
    
    results = validator.validate_classification(
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        output_dir=output_dir
    )
    
    # 3 correct classifications out of 4 -> 0.75 accuracy
    assert results["metrics"]["accuracy"] == 0.75
    
    # Check that reports and plots exist
    assert (output_dir / "confusion_matrix.png").exists()
    assert (output_dir / "confidence_histogram.png").exists()
    assert (output_dir / "landcover_distribution.png").exists()
    assert (output_dir / "forest_area_comparison.png").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "statistics.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "validation_report.md").exists()

def test_validate_deforestation_mask(tmp_path):
    validator = DeforestationPipelineValidator()
    
    pred_mask = [1, 0, 0]
    gt_mask = [1, 0, 0]
    output_dir = tmp_path / "reports_defor"
    
    metrics = validator.validate_deforestation_mask(
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        output_dir=output_dir
    )
    
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["iou"] == 1.0
    assert (output_dir / "deforestation_metrics.json").exists()
