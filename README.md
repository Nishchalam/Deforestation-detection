# 🌍 Deforestation Detection using Deep Learning and Sentinel-2 Satellite Imagery

An end-to-end deep learning framework and educational resource for land-cover classification and real-world change detection using Sentinel-2 multispectral and RGB satellite imagery.

---

## 📚 Documentation Hub
Explore our detailed guides and academic reports inside the [docs/](docs/) directory:
* **[PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)**: High-level companion guide for students and recruiters explaining project objectives, structures, and results.
* **[IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)**: Step-by-step build manual to reconstruct the repository chronologically from scratch.
* **[TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)**: Academic-grade system description detailing mathematical formulations and design trade-offs.
* **[RESEARCH_PAPER.md](docs/RESEARCH_PAPER.md)**: IEEE-formatted scientific manuscript draft.

---

## Current Project Status
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
* [ ] Research Paper publication

---

## Repository Structure
```text
Deforestation-detection/
├── train.py                    # Config-driven training CLI
├── requirements.txt            # Package dependencies
├── pyproject.toml              # Build specifications
├── README.md                   # Main landing page
│
├── configs/                    # YAML configuration files
├── docs/                       # Project documentation hub
│   ├── PROJECT_GUIDE.md        # Companion guide
│   ├── IMPLEMENTATION_GUIDE.md # Chronological build guide
│   ├── TECHNICAL_REPORT.md     # Mathematical & technical report
│   ├── RESEARCH_PAPER.md       # IEEE draft paper
│   ├── architecture/
│   ├── figures/
│   └── tables/
│
├── notebooks/                  # Educational Jupyter Notebooks
├── reports/                    # Aggregated reports
├── outputs/                    # Output directory
├── src/                        # Production library
└── tests/                      # Automated unit test suite
```

---

## Complete Workflow Diagram
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

## Quick Start Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Nishchalam/Deforestation-Detection.git
cd Deforestation-Detection

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Download
```bash
python src/data/download.py
```

### 3. Model Training
```bash
python train.py --config configs/resnet18.yaml
```

### 4. Running Deforestation Change Detection
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
