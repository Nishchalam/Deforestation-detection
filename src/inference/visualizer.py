"""
visualizer.py

Overlays deforestation detection results onto the original Sentinel-2 imagery.
"""

from typing import List, Tuple
from PIL import Image, ImageDraw


class DeforestationVisualizer:
    """
    Visualizes deforestation by drawing bounding boxes or overlays 
    on the original image.
    """
    def __init__(self, color: str = "red", line_width: int = 3):
        """
        Parameters
        ----------
        color : str
            Color of the bounding box for deforested patches.
        line_width : int
            Thickness of the bounding box line.
        """
        self.color = color
        self.line_width = line_width

    def draw_overlay(
        self, 
        image_path: str, 
        bboxes: List[Tuple[int, int, int, int]], 
        deforestation_mask: List[bool]
    ) -> Image.Image:
        """
        Draws red boxes over patches identified as deforested.
        
        Parameters
        ----------
        image_path : str
            Path to the original Sentinel-2 image.
        bboxes : list of tuples
            List of (x, y, w, h) bounding boxes.
        deforestation_mask : list of bool
            Boolean list indicating if a patch is deforested.
            
        Returns
        -------
        PIL.Image
            The annotated image.
        """
        if len(bboxes) != len(deforestation_mask):
            raise ValueError("Bboxes and mask must have the same length.")

        image = Image.open(image_path).convert("RGBA")
        
        # Create an overlay for drawing with alpha transparency if needed later
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for bbox, is_deforested in zip(bboxes, deforestation_mask):
            if is_deforested:
                x, y, w, h = bbox
                # PIL Draw.rectangle expects [x0, y0, x1, y1]
                rect = [x, y, x + w, y + h]
                draw.rectangle(rect, outline=self.color, width=self.line_width)

        # Composite the original image with the overlay
        result = Image.alpha_composite(image, overlay)
        
        return result.convert("RGB")
