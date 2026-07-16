from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image, ImageDraw

class ChangeDetector:
    """
    Detects land-cover transitions between two temporal acquisitions.
    """
    def __init__(self, confidence_threshold: float = 0.0):
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
                if conf_a < self.confidence_threshold or conf_b < self.confidence_threshold:
                    changes.append((c_a, c_a, False))
                else:
                    changes.append((c_a, c_b, True))
                    
        return changes

    def compute_transition_matrix(self, changes: List[Tuple[str, str, bool]]) -> np.ndarray:
        matrix = np.zeros((10, 10), dtype=np.int32)
        for class_a, class_b, _ in changes:
            idx_a = self.class_to_idx.get(class_a)
            idx_b = self.class_to_idx.get(class_b)
            if idx_a is not None and idx_b is not None:
                matrix[idx_a, idx_b] += 1
        return matrix


class DeforestationDetector:
    """
    Identifies deforestation transitions (Forest -> Non-Forest).
    """
    def __init__(self, forest_class: str = "Forest"):
        self.forest_class = forest_class

    def detect_deforestation(self, changes: List[Tuple[str, str, bool]]) -> List[int]:
        mask = []
        for class_a, class_b, _ in changes:
            is_deforested = (class_a == self.forest_class) and (class_b != self.forest_class)
            mask.append(1 if is_deforested else 0)
        return mask

    def generate_binary_mask_image(
        self, 
        mask: List[int], 
        bboxes: List[Tuple[int, int, int, int]], 
        grid_size: Tuple[int, int]
    ) -> Image.Image:
        width, height = grid_size
        mask_array = np.zeros((height, width), dtype=np.uint8)
        
        for is_defor, bbox in zip(mask, bboxes):
            if is_defor:
                x, y, w, h = bbox
                mask_array[y:y+h, x:x+w] = 255
                
        return Image.fromarray(mask_array, mode="L")

    def draw_deforestation_overlay(
        self,
        image_year_b: Image.Image,
        mask: List[int],
        bboxes: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int, int] = (255, 0, 0, 100)
    ) -> Image.Image:
        img_rgba = image_year_b.convert("RGBA")
        overlay = Image.new("RGBA", img_rgba.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        for is_defor, bbox in zip(mask, bboxes):
            if is_defor:
                x, y, w, h = bbox
                draw.rectangle([x, y, x+w, y+h], fill=color, outline=(255, 0, 0, 255), width=2)
                
        blended = Image.alpha_composite(img_rgba, overlay)
        return blended.convert("RGB")


def validate_deforestation(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    patch_size_m: float = 64.0
) -> Dict[str, Any]:
    """
    Compares predicted binary deforestation mask with ground truth validation mask.
    
    Parameters
    ----------
    pred_mask : np.ndarray
        Binary 1D or 2D array of predictions (1=deforested, 0=stable).
    gt_mask : np.ndarray
        Binary 1D or 2D array of ground truth (1=deforested, 0=stable).
    patch_size_m : float
        Length of patch in meters (default 64m).
        
    Returns
    -------
    Dict[str, Any]
        Validation statistics: IoU, Precision, Recall, Forest Area Lost (ha) and confusion elements.
    """
    pred_flat = np.array(pred_mask).flatten().astype(int)
    gt_flat = np.array(gt_mask).flatten().astype(int)
    
    # Assert they have the same shape
    if pred_flat.shape != gt_flat.shape:
        raise ValueError("Predicted mask and ground truth mask must have the same size.")

    # Binary confusion matrix components
    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
    tn = int(np.sum((pred_flat == 0) & (gt_flat == 0)))
    
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    iou = tp / max(1, tp + fp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    
    # Calculate areas (in hectares)
    # Area per element = patch_size_m^2. 1 hectare = 10,000 m^2.
    pixel_area_ha = (patch_size_m ** 2) / 10000.0
    forest_lost_pred_ha = np.sum(pred_flat == 1) * pixel_area_ha
    forest_lost_gt_ha = np.sum(gt_flat == 1) * pixel_area_ha
    
    change_matrix = {
        "true_positives_patches": tp,
        "false_positives_patches": fp,
        "false_negatives_patches": fn,
        "true_negatives_patches": tn
    }
    
    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "forest_area_lost_pred_ha": round(forest_lost_pred_ha, 4),
        "forest_area_lost_gt_ha": round(forest_lost_gt_ha, 4),
        "change_matrix": change_matrix
    }
