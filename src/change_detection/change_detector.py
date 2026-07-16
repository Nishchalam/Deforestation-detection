"""
change_detector.py

Implements core ChangeDetector logic to compare multi-temporal classification outputs,
compute transition matrices, and support confidence-gated comparisons.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

class ChangeDetector:
    """
    Detects land-cover transitions between two temporal acquisitions.
    """
    def __init__(self, confidence_threshold: float = 0.0):
        """
        Parameters
        ----------
        confidence_threshold : float
            Minimum model confidence required to accept a classification change.
            If a change is detected but either prediction's confidence is below this,
            the change is rejected (flagged as stable/no change).
        """
        self.confidence_threshold = confidence_threshold
        self.classes = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
            "Industrial", "Pasture", "PermanentCrop", "Residential",
            "River", "SeaLake"
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def detect_patch_changes(
        self,
        map_a: Dict[str, Any],
        map_b: Dict[str, Any]
    ) -> List[Tuple[str, str, bool]]:
        """
        Compares two patch lists and identifies changes with confidence thresholding.
        
        Returns
        -------
        list of tuples
            Each tuple contains (class_a, class_b, is_changed).
        """
        classes_a = map_a["classes"]
        classes_b = map_b["classes"]
        confs_a = map_a.get("confidences", [1.0] * len(classes_a))
        confs_b = map_b.get("confidences", [1.0] * len(classes_b))
        
        if len(classes_a) != len(classes_b):
            raise ValueError("Temporal maps must have the same patch dimensions.")
            
        changes = []
        for c_a, c_b, conf_a, conf_b in zip(classes_a, classes_b, confs_a, confs_b):
            if c_a == c_b:
                changes.append((c_a, c_b, False))
            else:
                # If confidence thresholding is enabled
                if conf_a < self.confidence_threshold or conf_b < self.confidence_threshold:
                    # Reject change -> keep as original Year A class
                    changes.append((c_a, c_a, False))
                else:
                    changes.append((c_a, c_b, True))
                    
        return changes

    def compute_transition_matrix(self, changes: List[Tuple[str, str, bool]]) -> np.ndarray:
        """
        Generates a 10x10 transition matrix where row=class_a and col=class_b.
        """
        matrix = np.zeros((10, 10), dtype=np.int32)
        for class_a, class_b, _ in changes:
            idx_a = self.class_to_idx.get(class_a)
            idx_b = self.class_to_idx.get(class_b)
            if idx_a is not None and idx_b is not None:
                matrix[idx_a, idx_b] += 1
        return matrix
