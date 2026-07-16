import pytest
import numpy as np
from src.change_detection import (
    ChangeDetector, 
    DeforestationDetector, 
    validate_deforestation
)

def test_change_detector():
    detector = ChangeDetector(confidence_threshold=0.5)
    
    map_a = {
        "classes": ["Forest", "Forest", "Residential", "SeaLake"],
        "confidences": [0.8, 0.9, 0.7, 0.4]  # 0.4 is below threshold
    }
    
    map_b = {
        "classes": ["AnnualCrop", "Forest", "Residential", "Forest"],
        "confidences": [0.9, 0.9, 0.8, 0.8]
    }
    
    changes = detector.detect_patch_changes(map_a, map_b)
    
    # 4 patches total
    assert len(changes) == 4
    # First patch: Forest -> AnnualCrop (change accepted, conf > 0.5)
    assert changes[0] == ("Forest", "AnnualCrop", True)
    # Second patch: Forest -> Forest (no change)
    assert changes[1] == ("Forest", "Forest", False)
    # Third patch: Residential -> Residential (no change)
    assert changes[2] == ("Residential", "Residential", False)
    # Fourth patch: SeaLake -> Forest (change rejected, conf_a is 0.4 < 0.5 -> kept as SeaLake)
    assert changes[3] == ("SeaLake", "SeaLake", False)
    
    # Compute transition matrix
    matrix = detector.compute_transition_matrix(changes)
    assert matrix.shape == (10, 10)
    # 1 transition from Forest to AnnualCrop
    forest_idx = detector.class_to_idx["Forest"]
    crop_idx = detector.class_to_idx["AnnualCrop"]
    assert matrix[forest_idx, crop_idx] == 1

def test_deforestation_detector():
    detector = DeforestationDetector(forest_class="Forest")
    
    changes = [
        ("Forest", "AnnualCrop", True),
        ("Forest", "Forest", False),
        ("Residential", "Residential", False),
        ("SeaLake", "Forest", True)
    ]
    
    mask = detector.detect_deforestation(changes)
    
    # Deforestation mask: 1 for Forest -> Non-Forest, 0 otherwise
    assert mask == [1, 0, 0, 0]

def test_statistics_and_metrics():
    pred_mask = [1, 0, 0]
    gt_mask = [1, 0, 0]
    
    # 100m x 100m patch = 1 hectare (10,000 sq m)
    metrics = validate_deforestation(pred_mask, gt_mask, patch_size_m=100.0)
    
    assert metrics["forest_area_lost_pred_ha"] == 1.0
    assert metrics["forest_area_lost_gt_ha"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["change_matrix"]["true_positives_patches"] == 1
