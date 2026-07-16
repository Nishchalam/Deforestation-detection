import os
import argparse
import subprocess
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import json

from src.models import create_model
from src.inference import LandCoverPredictor, LandCoverMapper, PatchGenerator
from src.change_detection import ChangeDetector, DeforestationDetector, validate_deforestation
from src.utils import (
    plot_transition_matrix_heatmap,
    plot_confidence_histogram,
    plot_forest_area_comparison,
    export_validation_reports
)

def run_download_if_needed(year1: int, year2: int, output_dir: str, bbox: list, dimensions: int):
    """Checks if demo images exist; if not, triggers download_region.py."""
    y1_file = os.path.join(output_dir, f"sentinel2_{year1}.png")
    y2_file = os.path.join(output_dir, f"sentinel2_{year2}.png")
    mask_file = os.path.join(output_dir, "hansen_validation_mask.png")
    
    if not (os.path.exists(y1_file) and os.path.exists(y2_file) and os.path.exists(mask_file)):
        print("Demo source images or validation masks not found. Triggering GEE download...")
        cmd = [
            "python", "download_region.py",
            "--year1", str(year1),
            "--year2", str(year2),
            "--dimensions", str(dimensions),
            "--output_dir", output_dir
        ]
        if bbox:
            cmd.extend(["--bbox"] + [str(x) for x in bbox])
            
        subprocess.run(cmd, check=True)
    else:
        print("Demo source images and validation masks already present.")

def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end reproducible deforestation detection demo.")
    parser.add_argument("--model", type=str, default="resnet18", help="CNN architecture (e.g. resnet18, resnet50)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model weights (.pth)")
    parser.add_argument("--year1", type=int, default=2018, help="Year A (start year)")
    parser.add_argument("--year2", type=int, default=2022, help="Year B (end year)")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size for classification")
    parser.add_argument("--stride", type=int, default=64, help="Stride for sliding window")
    parser.add_argument("--output_dir", type=str, default="reports/demo_results", help="Directory to save figures/reports")
    parser.add_argument("--data_dir", type=str, default="data/demo", help="Directory containing downloaded images")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Confidence threshold to accept class changes")
    args = parser.parse_args()

    # 1. Download imagery if needed
    run_download_if_needed(
        year1=args.year1,
        year2=args.year2,
        output_dir=args.data_dir,
        bbox=None, # uses default bbox
        dimensions=1024
    )

    y1_file = os.path.join(args.data_dir, f"sentinel2_{args.year1}.png")
    y2_file = os.path.join(args.data_dir, f"sentinel2_{args.year2}.png")
    mask_file = os.path.join(args.data_dir, "hansen_validation_mask.png")

    # 2. Instantiate Model
    print(f"\nInitializing model '{args.model}'...")
    model = create_model(args.model, num_classes=10)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from: {args.checkpoint}")
        predictor = LandCoverPredictor(model=model, checkpoint_path=args.checkpoint)
    else:
        print("WARNING: No checkpoint file specified or found. Initializing predictor with randomly initialized weights for demonstration purposes.")
        predictor = LandCoverPredictor(model=model, checkpoint_path=None)

    # 3. Predict Land Cover Maps
    print("\nRunning sliding-window land cover mapping...")
    mapper = LandCoverMapper(predictor=predictor, patch_size=args.patch_size, stride=args.stride)
    
    print(f"Mapping Year A ({args.year1})...")
    map_a = mapper.generate_map(y1_file, batch_size=32)
    
    print(f"Mapping Year B ({args.year2})...")
    map_b = mapper.generate_map(y2_file, batch_size=32)

    # Save output directories
    os.makedirs(args.output_dir, exist_ok=True)
    map_a["prediction_map"].save(os.path.join(args.output_dir, f"landcover_map_{args.year1}.png"))
    map_b["prediction_map"].save(os.path.join(args.output_dir, f"landcover_map_{args.year2}.png"))

    # 4. Detect Changes & Deforestation
    print("\nDetecting land cover changes...")
    detector = ChangeDetector(confidence_threshold=args.confidence_threshold)
    changes = detector.detect_patch_changes(map_a, map_b)
    
    # Compute transition matrix
    trans_matrix = detector.compute_transition_matrix(changes)
    plot_transition_matrix_heatmap(
        trans_matrix, 
        detector.classes, 
        os.path.join(args.output_dir, "transition_matrix_heatmap.png")
    )

    print("Running deforestation detection (Forest -> Non-Forest transitions)...")
    defor_detector = DeforestationDetector(forest_class="Forest")
    pred_defor_mask = defor_detector.detect_deforestation(changes)
    
    # Generate binary mask and overlay images
    img_b = Image.open(y2_file)
    bin_mask_img = defor_detector.generate_binary_mask_image(
        pred_defor_mask, 
        map_b["bboxes"], 
        img_b.size
    )
    bin_mask_img.save(os.path.join(args.output_dir, "detected_deforestation_mask.png"))
    
    overlay_img = defor_detector.draw_deforestation_overlay(
        img_b, 
        pred_defor_mask, 
        map_b["bboxes"]
    )
    overlay_img.save(os.path.join(args.output_dir, "deforestation_overlay_visual.png"))

    # 5. Load and Grid-downsample Hansen Ground Truth Validation Mask
    print("\nLoading Hansen Global Forest Change ground-truth validation mask...")
    gt_mask_img = Image.open(mask_file).convert("L")
    gt_array = np.array(gt_mask_img)
    
    width, height = img_b.size
    gt_patches = []
    
    # Downsample GEE pixel mask (1024x1024) to patch grid (e.g. 16x16)
    for bbox in map_b["bboxes"]:
        x, y, w, h = bbox
        patch_pixels = gt_array[y:y+h, x:x+w]
        # Hansen mask has value 255 for forest loss, 0 otherwise
        loss_ratio = np.mean(patch_pixels == 255)
        # If > 10% of the patch area suffered loss, count it as ground-truth deforestation
        gt_patches.append(1 if loss_ratio > 0.10 else 0)

    # 6. Validate Deforestation Detection
    print("Computing deforestation validation statistics...")
    metrics = validate_deforestation(
        pred_mask=pred_defor_mask,
        gt_mask=gt_patches,
        patch_size_m=float(args.patch_size)
    )

    # 7. Generate Confidence Histogram and Forest Area Comparisons
    plot_confidence_histogram(
        map_a["confidences"] + map_b["confidences"],
        os.path.join(args.output_dir, "confidence_histogram.png")
    )
    
    # Estimate forest areas in hectares
    pixel_area_ha = (args.patch_size ** 2) / 10000.0
    forest_a_patches = sum(1 for c in map_a["classes"] if c == "Forest")
    forest_b_patches = sum(1 for c in map_b["classes"] if c == "Forest")
    forest_a_ha = forest_a_patches * pixel_area_ha
    forest_b_ha = forest_b_patches * pixel_area_ha
    
    plot_forest_area_comparison(
        forest_a_ha, 
        forest_b_ha, 
        os.path.join(args.output_dir, "forest_area_comparison.png")
    )

    # 8. Export Reports & Summary
    export_validation_reports(
        metrics=metrics,
        classes=detector.classes,
        output_dir=Path(args.output_dir)
    )
    
    print("\n" + "="*50)
    print("               DEMO SUMMARY STATISTICS")
    print("="*50)
    print(f"Study Area Bounding Box Size   : {img_b.size[0]}x{img_b.size[1]} pixels")
    print(f"Total Patches Evaluated        : {len(pred_defor_mask)}")
    print(f"Detected Deforested Patches    : {sum(pred_defor_mask)}")
    print(f"Ground-Truth Deforested Patches: {sum(gt_patches)}")
    print(f"Estimated Forest Loss (Model)  : {metrics['forest_area_lost_pred_ha']} hectares")
    print(f"Estimated Forest Loss (Hansen) : {metrics['forest_area_lost_gt_ha']} hectares")
    print("-"*50)
    print(f"Intersection over Union (IoU)  : {metrics['iou']:.4f}")
    print(f"Precision                      : {metrics['precision']:.4f}")
    print(f"Recall                         : {metrics['recall']:.4f}")
    print(f"F1-Score                       : {metrics['f1_score']:.4f}")
    print("="*50)
    print(f"All figures and reports exported successfully to: {args.output_dir}/")

if __name__ == "__main__":
    main()
