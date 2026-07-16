"""
deforestation.py

Extracts Forest to Non-Forest transition masks and overlays deforestation bounding boxes.
"""

from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw

class DeforestationDetector:
    """
    Identifies deforestation transitions (Forest -> Non-Forest).
    """
    def __init__(self, forest_class: str = "Forest"):
        self.forest_class = forest_class

    def detect_deforestation(self, changes: List[Tuple[str, str, bool]]) -> List[int]:
        """
        Creates a binary deforestation mask.
        1 = Deforested (transitioned from Forest to Non-Forest)
        0 = Stable / Other
        """
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
        """
        Creates a binary mask image where black (0) is stable and white (255) is deforested.
        """
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
        """
        Draws semi-transparent red overlays on deforested areas of the Year B image.
        """
        img_rgba = image_year_b.convert("RGBA")
        overlay = Image.new("RGBA", img_rgba.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        for is_defor, bbox in zip(mask, bboxes):
            if is_defor:
                x, y, w, h = bbox
                draw.rectangle([x, y, x+w, y+h], fill=color, outline=(255, 0, 0, 255), width=2)
                
        blended = Image.alpha_composite(img_rgba, overlay)
        return blended.convert("RGB")
        
import numpy as np
