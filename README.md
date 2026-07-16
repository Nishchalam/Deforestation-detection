# 🌍 Deforestation Detection using Deep Learning and Sentinel-2 Satellite Imagery

An end-to-end deep learning framework and educational resource for land-cover classification and real-world change detection using Sentinel-2 multispectral and RGB satellite imagery.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Current Project Status](#current-project-status)
3. [Repository Structure](#repository-structure)
4. [CNN Evolution](#cnn-evolution)
5. [Experimental Pipeline](#experimental-pipeline)
6. [Results](#results)
7. [Running the Project](#running-the-project)
8. [Future Work](#future-work)
9. [References](#references)
10. [Citation](#citation)
11. [License](#license)

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
* [ ] Model Explainability (Grad-CAM & Activation Maps)
* [ ] Final Research Report & Publication

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
│   ├── base.yaml               # Defaults baseline configuration
│   ├── lenet.yaml
│   ├── alexnet.yaml
│   ├── vgg16.yaml
│   ├── googlenet.yaml
│   ├── resnet18.yaml
│   ├── resnet50.yaml
│   └── efficientnet_b0.yaml
│
├── notebooks/                  # Educational Jupyter Notebooks
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  # Transform and augmentation pipeline
│   ├── CNN_Evolution/          # CNN architecture tutorials
│   │   ├── 03_LeNet.ipynb
│   │   ├── 04_AlexNet.ipynb
│   │   ├── 05_VGG16.ipynb
│   │   ├── 06_GoogLeNet.ipynb
│   │   ├── 07_ResNet18.ipynb
│   │   ├── 08_ResNet50.ipynb
│   │   └── 09_EfficientNet.ipynb
│   └── Deforestation/          # Phase 2 pipeline tutorials
│       ├── 11_Sentinel2_Inference.ipynb
│       └── 12_Change_Detection.ipynb
│
├── notes/                      # Mathematical and conceptual guides
│   ├── CNN_History.md
│   ├── LeNet_Notes.md
│   ├── AlexNet_Notes.md
│   ├── VGG_Notes.md
│   ├── GoogLeNet_Notes.md
│   ├── ResNet_Notes.md
│   └── EfficientNet_Notes.md
│
├── reports/                    # Aggregated reports
│   └── comparison/
│       ├── comparison.csv      # CSV comparison database
│       ├── comparison.md       # Markdown comparison table
│       └── summary.json        # High-level JSON report
│
├── outputs/                    # Output directory
│   ├── experiments/            # Self-contained experiment folders
│   ├── landcover/              # Reconstructed Sentinel-2 maps
│   └── change_detection/       # Temporal change maps & statistics
│       ├── landcover_2015.png
│       ├── landcover_2026.png
│       ├── change_map.png
│       ├── binary_mask.png
│       ├── overlay.png
│       ├── statistics.json
│       ├── transition_matrix.csv
│       └── summary.txt
│
├── src/                        # Production library
│   ├── data/                   # Data downloading and processing
│   │   ├── dataset.py
│   │   ├── verify.py
│   │   ├── transforms.py
│   │   ├── download.py
│   │   └── dataloader.py
│   ├── models/                 # Model Zoo
│   │   ├── common.py           # BaseCNN class
│   │   ├── lenet.py
│   │   ├── alexnet.py
│   │   ├── vgg.py
│   │   ├── googlenet.py
│   │   ├── resnet.py
│   │   ├── efficientnet.py
│   │   └── zoo.py              # create_model() constructor
│   ├── training/               # Training engine
│   │   ├── trainer.py          # Unified Training Loop
│   │   ├── callbacks.py        # Callbacks system
│   │   ├── losses.py           # Loss instantiator
│   │   ├── utils.py            # Checkpointing and history saver
│   │   └── logger.py           # Telemetry logger
│   ├── experiments/            # Experiment orchestrator
│   │   ├── experiment.py
│   │   ├── registry.py
│   │   ├── runner.py
│   │   ├── tracker.py
│   │   └── utils.py
│   ├── inference/              # Inference package
│   │   ├── patch_generator.py  # Sliding window patch generator
│   │   ├── predictor.py        # Multi-device model predictor
│   │   ├── landcover_mapper.py # Grid reconstruction mapper
│   │   ├── postprocessing.py   # Spatial majority filter
│   │   └── visualization.py    # Overlay and legend generators
│   └── change_detection/       # Change detection package
│       ├── change_detector.py  # Map comparison & transition matrices
│       ├── deforestation.py    # Forest transition masks
│       ├── metrics.py          # Validation scoring
│       ├── statistics.py       # Forest area summary statistics
│       ├── utils.py            # Transition grid helpers
│       └── visualization.py    # Dashboard panels & difference mapping
│
└── utils/                      # Miscellaneous utilities
    ├── visualization.py
    └── paths.py
│
└── tests/                      # Automated unit test suite
    ├── test_dataset.py
    ├── test_dataloader.py
    ├── test_trainer.py
    ├── test_runner.py
    ├── test_googlenet.py
    ├── test_resnet.py
    ├── test_efficientnet.py
    ├── test_inference.py
    └── test_change_detection.py
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

# Experimental Pipeline
The end-to-end layout of the framework operations:

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
```

---

# Results

### 1. Training Curves
*Placeholders for learning curves (Loss/Accuracy) per experiment will be loaded from `outputs/experiments/<experiment>/figures/loss_curves.png`.*

### 2. Confusion Matrix
*Placeholders for confusion matrix grids will be loaded from `outputs/experiments/<experiment>/figures/confusion_matrix.png`.*

### 3. Model Comparison
*The latest benchmarking results will compile to the `reports/comparison/comparison.md` file.*

### 4. Land-cover Maps
*The visual land-cover map predictions will be saved to `outputs/landcover/`.*

### 5. Deforestation Maps
*The final deforestation transition mask overlays will be saved to `outputs/change_detection/`.*

---

# Running the Project

### 1. Dataset Download
Ensure Kaggle API keys are stored in `~/.kaggle/`:
```bash
python src/data/download.py
```

### 2. Model Training
Train any configuration using `train.py`:
```bash
python train.py --config configs/resnet18.yaml
```

### 3. Model Benchmarking
Compare all experiment runs:
```bash
python src/evaluation/comparison.py
```
This updates comparison tables under `reports/comparison/`.

### 4. Running Deforestation Change Detection
Run multi-temporal change mapping between Year A and Year B acquisitions:
```bash
# Executable locally to compute transition matrices and mask forest loss
python -c "
from src.inference import LandCoverMapper, LandCoverPredictor
from src.change_detection import ChangeDetector, DeforestationDetector, calculate_forest_statistics, export_reports
from src.models import create_model
from pathlib import Path

# Load model
model = create_model('resnet18')
predictor = LandCoverPredictor(model, 'outputs/experiments/ResNet18_001/best_model.pth')
mapper = LandCoverMapper(predictor)

# Map Year A and Year B satellite acquisitions
map_a = mapper.generate_map('data/raw/Sentinel2_2015.tif')
map_b = mapper.generate_map('data/raw/Sentinel2_2026.tif')

# Run change detection
detector = ChangeDetector(confidence_threshold=0.7)
changes = detector.detect_patch_changes(map_a, map_b)

# Generate deforestation mask
defor = DeforestationDetector()
mask = defor.detect_deforestation(changes)

# Compute and save statistics and transition matrices
stats = calculate_forest_statistics(changes, mask)
matrix = detector.compute_transition_matrix(changes)
export_reports(stats, matrix, detector.classes, Path('outputs/change_detection/'))
"
```

---

# Future Work
* **Transfer Learning**: Pretrained ImageNet fine-tuning vs training from scratch comparison.
* **Explainability**: Grad-CAM, activation maps, and misclassified grids visualization.
* **Global Forest Watch Validation**: Aligning deforestation map transition overlays with actual GFW data.
* **Streamlit Dashboard**: Interactive UI for uploading satellite imagery and mapping changes.
* **Research Paper**: Formulating results into a publication-quality manuscript.

---

# References
1. **EuroSAT**: Helber, P., et al. "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification." (2019).
2. **LeNet**: LeCun, Y., et al. "Gradient-Based Learning Applied to Document Recognition." (1998).
3. **AlexNet**: Krizhevsky, A., et al. "ImageNet Classification with Deep Convolutional Networks." (2012).
4. **VGG**: Simonyan, K., & Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." (2014).
5. **GoogLeNet**: Szegedy, C., et al. "Going Deeper with Convolutions." (2015).
6. **ResNet**: He, K., et al. "Deep Residual Learning for Image Recognition." (2016).
7. **EfficientNet**: Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." (2019).
8. **Global Forest Watch**: Remote sensing datasets for forest coverage.

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
