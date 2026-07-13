"""
change_detector.py

Detects changes between two land-cover maps, specifically focusing 
on deforestation (Forest to Non-Forest transitions).
"""

from typing import Dict, Any, List

class DeforestationDetector:
    """
    Compares two temporal land cover maps to detect deforestation.
    """
    def __init__(self, forest_class_name: str = "Forest"):
        """
        Parameters
        ----------
        forest_class_name : str
            The exact string name used by the model for the Forest class.
        """
        self.forest_class = forest_class_name

    def detect_changes(
        self, 
        map_year1: Dict[str, Any], 
        map_year2: Dict[str, Any]
    ) -> List[bool]:
        """
        Compares two maps and returns a boolean mask indicating deforestation.
        
        Parameters
        ----------
        map_year1 : dict
            Land cover map for the older year (output of LandCoverMapper).
        map_year2 : dict
            Land cover map for the newer year (output of LandCoverMapper).
            
        Returns
        -------
        list of bool
            True if the patch transitioned from Forest to Non-Forest, 
            False otherwise.
        """
        classes1 = map_year1["classes"]
        classes2 = map_year2["classes"]

        if len(classes1) != len(classes2):
            raise ValueError("Maps must have the same number of patches.")

        deforestation_mask = []

        for c1, c2 in zip(classes1, classes2):
            # Deforestation is defined as Forest in Year 1 and NOT Forest in Year 2
            is_deforestation = (c1 == self.forest_class) and (c2 != self.forest_class)
            deforestation_mask.append(is_deforestation)

        return deforestation_mask
