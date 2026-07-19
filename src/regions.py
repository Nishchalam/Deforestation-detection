import os
import random
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any

# Regional configurations for Rondônia, Brazil study areas
REGIONS = {
    "ji_parana": {
        "name": "Ji-Paraná",
        "bbox": [-62.0, -11.0, -61.8, -10.8],
        "image_y1": "sentinel2_ji_parana_2018.png",
        "image_y2": "sentinel2_ji_parana_2022.png", # Not used in Notebook 10
        "mask": "hansen_validation_ji_parana.png"
    },
    "porto_velho": {
        "name": "Porto Velho Frontier",
        "bbox": [-64.0, -9.0, -63.8, -8.8],
        "image_y1": "sentinel2_porto_velho_2018.png",
        "image_y2": "sentinel2_porto_velho_2022.png",
        "mask": "hansen_validation_porto_velho.png"
    },
    "ariquemes": {
        "name": "Ariquemes Corridor",
        "bbox": [-63.0, -10.5, -62.8, -10.3],
        "image_y1": "sentinel2_ariquemes_2018.png",
        "image_y2": "sentinel2_ariquemes_2022.png",
        "mask": "hansen_validation_ariquemes.png"
    }
}


def generate_region_demo_data(region_id: str, data_dir: str):
    """
    Generates realistic Sentinel-2 and validation images (1024x1024) for a specific region.
    Tiling is performed with unique land-cover and change layouts per region.
    """
    if region_id not in REGIONS:
        raise ValueError(f"Unknown region_id: {region_id}. Available: {list(REGIONS.keys())}")
        
    config = REGIONS[region_id]
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    
    y1_file = data_dir_path / config["image_y1"]
    y2_file = data_dir_path / config["image_y2"]
    mask_file = data_dir_path / config["mask"]
    
    # Check if files already exist
    if region_id == "ji_parana":
        if y1_file.exists():
            return
    else:
        if y1_file.exists() and y2_file.exists() and mask_file.exists():
            return
            
    # Find EuroSAT directory
    eurosat_dir = Path("data/raw/EuroSAT")
    if not eurosat_dir.exists():
        eurosat_dir = Path("../data/raw/EuroSAT")
    if not eurosat_dir.exists():
        raise FileNotFoundError("EuroSAT raw dataset not found. Please place it in data/raw/EuroSAT.")
        
    def get_class_images(class_name: str) -> List[Path]:
        folder = eurosat_dir / class_name
        return list(folder.glob("*.jpg"))
        
    forest_imgs = get_class_images("Forest")
    pasture_imgs = get_class_images("Pasture")
    river_imgs = get_class_images("River")
    highway_imgs = get_class_images("Highway")
    crop_imgs = get_class_images("AnnualCrop")
    
    if not (forest_imgs and pasture_imgs and river_imgs and highway_imgs and crop_imgs):
        raise ValueError("Missing essential EuroSAT class folders in data/raw/EuroSAT.")
        
    # Ensure reproducible seeding per region
    seed_map = {"ji_parana": 100, "porto_velho": 200, "ariquemes": 300}
    random.seed(seed_map.get(region_id, 42))
    
    # Image canvases
    img_a = Image.new("RGB", (1024, 1024))
    img_b = Image.new("RGB", (1024, 1024))
    mask = Image.new("L", (1024, 1024), 0)
    
    grid_size = 16
    patch_size = 64
    
    for y_idx in range(grid_size):
        for x_idx in range(grid_size):
            box = (x_idx * patch_size, y_idx * patch_size, (x_idx + 1) * patch_size, (y_idx + 1) * patch_size)
            
            # 1. Determine base class for Year A depending on Region Layout
            if region_id == "ji_parana":
                # Layout A (Ji-Paraná): vertical river at col 4, horizontal highway at row 10
                if x_idx == 4:
                    cls = "River"
                elif y_idx == 10:
                    cls = "Highway"
                elif y_idx < 10:
                    cls = "Forest"
                else:
                    cls = "AnnualCrop"
                    
            elif region_id == "porto_velho":
                # Layout B (Porto Velho): horizontal river at row 3, vertical highway at col 12
                if y_idx == 3:
                    cls = "River"
                elif x_idx == 12:
                    cls = "Highway"
                elif y_idx < 12:
                    cls = "Forest"
                else:
                    cls = "AnnualCrop"
                    
            else: # ariquemes
                # Layout C (Ariquemes): vertical river at col 8, horizontal highway at row 8
                if x_idx == 8:
                    cls = "River"
                elif y_idx == 8:
                    cls = "Highway"
                elif y_idx < 8:
                    cls = "Forest"
                else:
                    cls = "AnnualCrop"
            
            # Select patch path
            class_list = {
                "Forest": forest_imgs, "Pasture": pasture_imgs,
                "River": river_imgs, "Highway": highway_imgs,
                "AnnualCrop": crop_imgs
            }[cls]
            class_img_path = random.choice(class_list)
            
            patch_a = Image.open(class_img_path).resize((patch_size, patch_size))
            img_a.paste(patch_a, box)
            
            # 2. Deforestation change logic for Year B
            is_deforested = False
            
            if region_id == "porto_velho":
                # Deforestation B: diagonal swath of forest cleared (6 <= x + y <= 10) in Forest region
                if y_idx < 12 and x_idx != 12 and y_idx != 3:
                    if 6 <= (x_idx + y_idx) <= 10:
                        is_deforested = True
                        
            elif region_id == "ariquemes":
                # Deforestation C: block of forest cleared (2 <= x <= 5, 2 <= y <= 5)
                if y_idx < 8 and x_idx != 8 and y_idx != 8:
                    if 2 <= x_idx <= 5 and 2 <= y_idx <= 5:
                        is_deforested = True
            
            # Apply to Year B image and mask
            if is_deforested:
                class_img_path_b = random.choice(pasture_imgs)
                mask_patch = Image.new("L", (patch_size, patch_size), 255)
            else:
                class_img_path_b = class_img_path
                mask_patch = Image.new("L", (patch_size, patch_size), 0)
                
            if class_img_path_b != class_img_path:
                patch_b = Image.open(class_img_path_b).resize((patch_size, patch_size))
            else:
                patch_b = patch_a
                
            img_b.paste(patch_b, box)
            mask.paste(mask_patch, box)
            
    # Save outputs
    img_a.save(y1_file)
    print(f"[{config['name']}] Saved Year A composite: {y1_file.name}")
    
    if region_id != "ji_parana":
        img_b.save(y2_file)
        mask.save(mask_file)
        print(f"[{config['name']}] Saved Year B composite: {y2_file.name}")
        print(f"[{config['name']}] Saved Hansen mask: {mask_file.name}")
