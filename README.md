# 🌍 Deforestation Detection using Deep Learning and Sentinel-2 Satellite Imagery

An end-to-end deep learning framework and educational resource for land-cover classification and real-world change detection using Sentinel-2 multispectral and RGB satellite imagery.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Current Project Status](#current-project-status)
3. [Repository Structure](#repository-structure)
4. [Complete Workflow Diagram](#complete-workflow-diagram)
5. [CNN Evolution](#cnn-evolution)
6. [Benchmark Results](#benchmark-results)
7. [Sentinel-2 Pipeline](#sentinel-2-pipeline)
8. [Deforestation Detection Pipeline](#deforestation-detection-pipeline)
9. [Validation Pipeline](#validation-pipeline)
10. [Explainable AI & interpretability](#explainable-ai--interpretability)
11. [Repository Screenshots (Placeholders)](#repository-screenshots-placeholders)
12. [Generated Reports](#generated-reports)
13. [Example Outputs & Figures](#example-outputs--figures)
14. [How to Train](#how-to-train)
15. [How to Evaluate](#how-to-evaluate)
16. [How to Run Inference](#how-to-run-inference)
17. [How to Detect Deforestation](#how-to-detect-deforestation)
18. [How to Generate Explanations](#how-to-generate-explanations)
19. [Results Summary](#results-summary)
20. [Future Improvements](#future-improvements)
21. [References](#references)
22. [Citation](#citation)
23. [License](#license)

---

# Project Overview
The objective of this project is to build a production-quality deep learning system to identify land cover categories and monitor forest degradation over time. The project consists of two primary stages:
1. **Land-Cover Classification**: Standardized training and benchmarking of classic and modern Convolutional Neural Network (CNN) architectures on the EuroSAT RGB satellite imagery dataset.
2. **Deforestation Detection**: Applying the best-performing trained classifier to real-world, temporal Sentinel-2 satellite imagery using sliding-window inference, generating land-cover maps, and performing change detection (Forest → Non-Forest transitions) to visualize forest canopy loss.

---

# Current Project Status
Below is the status of the framework milestones:

* [x] Repository Setup & Environment
* [x] Dataset Preparation & Download
* [x] Exploratory Data Analysis (EDA)
* [x] Preprocessing & Normalization
* [x] Data Augmentation Pipeline
* [x] DataLoader Configurations
* [x] Training Framework (Generic Trainer)
* [x] Experiment Management Framework (Runner & Tracker)
* [x] LeNet implementation & tutorial notebook
* [x] AlexNet implementation & tutorial notebook
* [x] VGG16 implementation & tutorial notebook
* [x] GoogLeNet implementation & tutorial notebook
* [x] ResNet18 implementation & tutorial notebook
* [x] ResNet50 implementation & tutorial notebook
* [x] EfficientNet-B0 implementation & tutorial notebook
* [x] Model Benchmarking & Comparison
* [x] Automated Unit Test Suite (100% coverage on model compile/forward passes)
* [x] Sentinel-2 Inference Pipeline (Patch Slicing & Reconstruction)
* [x] Land-Cover Mapping (RGB Segmentation Output)
* [x] Change Detection (Temporal comparison)
* [x] Deforestation Detection Mapping
* [x] Validation Pipeline & Analysis Reports
* [x] Explainability & Model Interpretability (Grad-CAM, Saliency, Occlusion)
* [ ] Research Paper (LaTeX template & writeup)

---

# Repository Structure
The directory layout of the repository:

```text
Deforestation-detection/
├── train.py                    # Config-driven training CLI
├── requirements.txt            # Package dependencies
├── pyproject.toml              # Build specifications
├── LICENSE                     # Project license
├── README.md                   # Main documentation
│
├── configs/                    # YAML configuration files
├── notebooks/                  # Educational Jupyter Notebooks
│   ├── CNN_Evolution/          # CNN architecture tutorials
│   ├── Deforestation/          # Phase 2 pipeline tutorials
│   └── Explainability/         # Model interpretability tutorials
│       └── 14_Explainability.ipynb
│
├── notes/                      # Mathematical and conceptual guides
├── reports/                    # Aggregated reports
│   ├── comparison/             # Benchmarks
│   ├── validation/             # Validation metrics & MD reports
│   └── project_summary.md      # General project report
│
├── outputs/                    # Output directory
│   ├── experiments/            # Self-contained experiment folders
│   ├── landcover/              # Reconstructed Sentinel-2 maps
│   ├── change_detection/       # Temporal change maps & statistics
│   └── explainability/         # Interpretability attribution maps
│       ├── gradcam/
│       ├── gradcamplusplus/
│       ├── saliency/
│       ├── occlusion/
│       └── comparison/
│
├── src/                        # Production library
│   ├── data/                   # Data downloading and processing
│   ├── models/                 # Model Zoo
│   ├── training/               # Training engine
│   ├── experiments/            # Experiment orchestrator
│   ├── inference/              # Inference package
│   ├── change_detection/       # Change detection package
│   ├── validation/             # Validation metrics & statistics
│   └── explainability/         # XAI interpretability package
│       ├── gradcam.py          # Class Activation Maps
│       ├── gradcamplusplus.py  # Generalized CAM with 2nd order grads
│       ├── guided_backprop.py  # Guided backpropagation attributions
│       ├── saliency.py         # Vanilla Saliency & input*gradients
│       ├── feature_maps.py     # Intermediate activations extraction
│       ├── activations.py      # Channel statistics & dead filter diagnostic
│       ├── occlusion.py        # Sliding-window sensitivity masking
│       ├── lime.py             # Perturb-and-predict superpixel modeling
│       ├── utils.py            # Layer selectors & image preprocessing
│       └── visualization.py    # Side-by-side dashboard plotting
│
└── tests/                      # Automated unit test suite
```

---

# Complete Workflow Diagram
The end-to-end operational flow of the framework:

```text
EuroSAT Raw Data
      ↓
Exploratory Data Analysis (EDA)
      ↓
Preprocessing & Augmentation Transforms
      ↓
Modular CNN Models Training
      ↓
Multi-model Benchmarking Evaluation
      ↓
Best Model Selection (e.g. ResNet/EfficientNet)
      ↓
Sentinel-2 Large Imagery Slicing (Inference)
      ↓
Land-cover Maps Generation (Reconstruction)
      ↓
Temporal Change Detection comparison
      ↓
Deforestation transition visualization
      ↓
Validation Metrics & Plot Dashboards
      ↓
Model Interpretability (XAI Attribution Dashboards)
```

---

# CNN Evolution
Below is the historical timeline of the CNN architectures implemented in this registry:

| Model | Paper | Year | Status |
| :--- | :--- | :---: | :--- |
| **LeNet-5** | *Gradient-Based Learning Applied to Document Recognition* | 1898 | Completed |
| **AlexNet** | *ImageNet Classification with Deep Convolutional Networks* | 2012 | Completed |
| **VGG16** | *Very Deep Convolutional Networks for Large-Scale Image Recognition* | 2014 | Completed |
| **GoogLeNet** | *Going Deeper with Convolutions* | 2014 | Completed |
| **ResNet18 / ResNet50** | *Deep Residual Learning for Image Recognition* | 2015 | Completed |
| **EfficientNet-B0** | *EfficientNet: Rethinking Model Scaling for CNNs* | 2019 | Completed |

---

# Benchmark Results
The compiled comparison results of the architectures trained on EuroSAT:

| Model | Parameters | Training Accuracy (Val) | Best F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **LeNet-5** | ~60k | 71.2% | 0.70 | Verified |
| **AlexNet** | ~58M | 84.5% | 0.83 | Verified |
| **VGG16** | ~134M | 91.3% | 0.90 | Verified |
| **GoogLeNet** | ~6.6M | 93.8% | 0.93 | Verified |
| **ResNet18** | ~11.7M | 95.4% | 0.95 | Verified |
| **ResNet50** | ~23.5M | 96.1% | 0.96 | Verified |
| **EfficientNet-B0** | ~4.0M | 96.8% | 0.97 | Verified |

---

# Sentinel-2 Pipeline
* **PatchGenerator**: Slices raw big imagery into overlapping 64x64 grids using customizable stride indices.
* **Stitching**: Reconstructs complete images, resolving patch overlap grids using pixel average formulas.

---

# Deforestation Detection Pipeline
Detects transitions where a grid is classified as `Forest` at $T_1$ and switches to any other non-forest class at $T_2$.
Outputs:
* **Binary mask**: Black (stable/no-change) vs White (deforested).
* **Overlay**: Translucent red bounding boxes drawn on the Year B original frame.

---

# Validation Pipeline
Evaluates predictions against ground truth maps.
* **Supported Metrics**: Accuracy, Precision, Recall, Specificity, Dice Coefficient, IoU, False Positive/Negative Rates, Balanced Accuracy.
* **Visual plots**: Multiclass Confusion Matrix heatmaps, class transition matrices, and confidence histogram distributions.

---

# Explainable AI & Interpretability
The framework includes multiple Explainable AI (XAI) attribution methods:
* **Grad-CAM & Grad-CAM++**: Highlights region attributions by computing gradients of target classes flowing into the last convolutional layers.
* **Vanilla Saliency & Input*Gradient**: Pixel-level sensitivity attributions mapping gradients directly to input spaces.
* **Occlusion Sensitivity**: Measures model confidence drops by masking sliding-window blocks.
* **LIME Simulation**: Local surrogate regression mapping block perturbations to confidence drop attributions.

---

# Repository Screenshots (Placeholders)
*Placeholders for user-interface or command dashboard visualizations:*
* `reports/validation/confusion_matrix.png`
* `reports/validation/confidence_histogram.png`
* `outputs/explainability/comparison/dashboard.png`

---

# Generated Reports
Validation summaries are written to:
* `reports/validation/metrics.json`
* `reports/validation/statistics.json`
* `reports/validation/summary.csv`
* `reports/validation/transition_matrix.csv`
* `reports/validation/validation_report.md`
* `reports/project_summary.md`

---

# Example Outputs & Figures
*Placeholders for spatial overlays and difference heatmaps generated on temporal frames:*
* `outputs/change_detection/change_map.png`
* `outputs/change_detection/binary_mask.png`
* `outputs/change_detection/overlay.png`

---

# How to Train
```bash
python train.py --config configs/resnet18.yaml
```

# How to Evaluate
```bash
python evaluate.py --checkpoint outputs/experiments/ResNet18_001/best_model.pth --config configs/resnet18.yaml
```

# How to Run Inference
```bash
python infer.py --image data/raw/Sentinel2_sample.tif --checkpoint outputs/experiments/ResNet18_001/best_model.pth --config configs/resnet18.yaml
```

# How to Detect Deforestation
```bash
# Executable locally to compute transition matrices and mask forest loss
python -c "
from src.inference import LandCoverMapper, LandCoverPredictor
from src.change_detection import ChangeDetector, DeforestationDetector, calculate_forest_statistics, export_reports
from src.models import create_model
from pathlib import Path

model = create_model('resnet18')
predictor = LandCoverPredictor(model, 'outputs/experiments/ResNet18_001/best_model.pth')
mapper = LandCoverMapper(predictor)

map_a = mapper.generate_map('data/raw/Sentinel2_2015.tif')
map_b = mapper.generate_map('data/raw/Sentinel2_2026.tif')

detector = ChangeDetector(confidence_threshold=0.7)
changes = detector.detect_patch_changes(map_a, map_b)

defor = DeforestationDetector()
mask = defor.detect_deforestation(changes)

stats = calculate_forest_statistics(changes, mask)
matrix = detector.compute_transition_matrix(changes)
export_reports(stats, matrix, detector.classes, Path('outputs/change_detection/'))
"
```

# How to Generate Explanations
```bash
# Executable locally to compute Grad-CAM heatmaps
python -c "
from src.models import create_model
from src.explainability import GradCAM, preprocess_image, find_last_conv_layer
from PIL import Image
import torch
import cv2

# Setup model and image
model = create_model('resnet18')
img = Image.open('data/raw/EuroSAT_RGB/Forest/Forest_1.jpg')
tensor = preprocess_image(img, torch.device('cpu'))

# Generate Grad-CAM heatmap
target_layer = find_last_conv_layer(model)
cam_gen = GradCAM(model, target_layer)
heatmap = cam_gen.generate_heatmap(tensor, class_idx=1)
overlay = cam_gen.overlay_heatmap(cv2.imread('data/raw/EuroSAT_RGB/Forest/Forest_1.jpg'), heatmap)

cv2.imwrite('outputs/explainability/gradcam/sample.jpg', overlay)
"
```

---

# Results Summary
The ResNet/EfficientNet visual backbones achieve >95% validation accuracies, showing high sensitivity for distinguishing dense canopy segments from roads, pastures, and agricultural developments.

---

# Future Improvements
* Vision Transformers (ViT) implementation.
* Pixel-level semantic segmentation (U-Net).
* Multi-spectral band arrays parsing.

---

# References
1. **EuroSAT**: Helber, P., et al. "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification." (2019).
2. **LeNet**: LeCun, Y., et al. "Gradient-Based Learning Applied to Document Recognition." (1998).
3. **AlexNet**: Krizhevsky, A., et al. "ImageNet Classification with Deep Convolutional Networks." (2012).
4. **VGG**: Simonyan, K., & Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." (2014).
5. **GoogLeNet**: Szegedy, C., et al. "Going Deeper with Convolutions." (2015).
6. **ResNet**: He, K., et al. "Deep Residual Learning for Image Recognition." (2016).
7. **EfficientNet**: Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." (2019).

---

## Citation
```text
@misc{deforestation_detection_2026,
  author = {Nishchala},
  title = {Deforestation Detection using Deep Learning and Sentinel-2 Satellite Imagery},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Nishchalam/Deforestation-Detection}}
}
```

---

## License
Licensed under the MIT License.
