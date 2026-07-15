# 🌍 Deforestation Detection using Deep Learning and Sentinel-2 Satellite Imagery

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)

An end-to-end deep learning framework and educational resource for land-cover classification and real-world change detection using Sentinel-2 multispectral and RGB satellite imagery.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Objectives](#3-objectives)
4. [Repository Structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [Dataset](#6-dataset)
7. [Data Pipeline](#7-data-pipeline)
8. [EDA](#8-eda)
9. [Preprocessing](#9-preprocessing)
10. [Training Framework](#10-training-framework)
11. [CNN Architectures](#11-cnn-architectures)
12. [Experiment Management](#12-experiment-management)
13. [TensorBoard Usage](#13-tensorboard-usage)
14. [Configuration Files](#14-configuration-files)
15. [Outputs Directory](#15-outputs-directory)
16. [Training](#16-training)
17. [Evaluation](#17-evaluation)
18. [Inference](#18-inference)
19. [Phase 2 Deforestation Detection Pipeline](#19-phase-2-deforestation-detection-pipeline)
20. [Future Work](#20-future-work)
21. [References](#21-references)
22. [Citation](#22-citation)
23. [License](#23-license)
24. [Project Roadmap](#24-project-roadmap)

---

## 1. Project Overview
This repository implements a production-grade, config-driven deep learning research framework designed to classify land cover and detect deforestation from satellite data. The project operates in two independent phases:
* **Phase 1 (Land Cover Classification)**: Standardized training, comparison, and evaluation of classic and modern CNN architectures on the EuroSAT RGB benchmark dataset.
* **Phase 2 (Change Detection & Deforestation Mapping)**: Slicing raw, large-scale temporal Sentinel-2 images into patches, running patch-wise inference, mapping land-cover classifications over years, and identifying forest-to-non-forest transitions (change detection).

---

## 2. Motivation
Tropical and temperate forests act as critical carbon sinks. Rapid agricultural expansion and logging are driving global deforestation, contributing to climate change and biodiversity loss. Standard ground surveys are labor-intensive, while remote sensing coupled with deep learning offers a scalable, automated alternative for monitoring large, remote forest regions in real-time.

---

## 3. Objectives
* Build a **reproducible, modular experiment management framework** for deep learning research.
* Compare historical and modern CNN architectures to trace the evolution of visual feature extractors.
* Apply the trained classifiers to verify forest coverage changes across temporal satellite maps.
* Serve as an **educational catalog** for university-level learning of CNN architectures from first principles.

---

## 4. Repository Structure
The complete directory tree of the repository:

```text
Deforestation-detection/
├── train.py                    # Config-driven training CLI
├── evaluate.py                 # Evaluation script
├── infer.py                    # Deforestation detection inference CLI
├── requirements.txt            # Package dependencies
├── pyproject.toml              # Build tool specifications
├── LICENSE                     # Project license
├── README.md                   # Main documentation
│
├── configs/                    # YAML configuration files
│   ├── base.yaml               # Defaults configuration file
│   ├── lenet.yaml
│   ├── alexnet.yaml
│   ├── vgg16.yaml
│   ├── googlenet.yaml
│   ├── resnet18.yaml
│   ├── resnet50.yaml
│   └── efficientnet_b0.yaml
│
├── notebooks/                  # Educational Notebooks (Self-contained, built from scratch)
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  # Transform and augmentation pipeline
│   ├── CNN_Evolution/          # CNN architecture tutorials
│   │   ├── 03_LeNet.ipynb
│   │   ├── 04_AlexNet.ipynb
│   │   ├── 05_VGG16.ipynb
│   │   ├── 06_GoogLeNet.ipynb
│   │   ├── 07_ResNet.ipynb
│   │   ├── 08_EfficientNet.ipynb
│   │   └── 09_Model_Comparison.ipynb
│   ├── Deforestation/          # Phase 2 pipeline tutorials
│   │   ├── 10_Patch_Generation.ipynb
│   │   ├── 11_LandCover_Mapping.ipynb
│   │   ├── 12_Change_Detection.ipynb
│   │   └── 13_Visualization.ipynb
│   └── Sandbox/                # Directory for user experimentation
│
├── notes/                      # Mathematical and architectural conceptual guides
│   ├── CNN_History.md
│   ├── LeNet_Notes.md
│   ├── AlexNet_Notes.md
│   ├── VGG_Notes.md
│   ├── GoogLeNet_Notes.md
│   ├── ResNet_Notes.md
│   └── EfficientNet_Notes.md
│
├── outputs/                    # Output directory
│   ├── experiments/            # Self-contained experiment runs
│   ├── checkpoints/            # Model weight backups
│   ├── logs/                   # Generic system log files
│   ├── figures/                # Visualizations
│   └── predictions/            # Evaluation output matrices
│
├── src/                        # Production library
│   ├── data/                   # Data downloading, parsing, and dataloaders
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
│   ├── training/               # Training modules
│   │   ├── trainer.py          # Unified Training Engine
│   │   ├── callbacks.py        # Callbacks (EarlyStopping, Checkpoint, etc.)
│   │   ├── losses.py           # Criterion instantiators
│   │   ├── utils.py            # Checkpoint and history saving utilities
│   │   ├── logger.py           # Multi-backend Experiment Logger
│   │   └── metrics.py          # Accuracy helper
│   ├── experiments/            # Experiment management framework
│   │   ├── experiment.py       # Experiment dataclass
│   │   ├── registry.py         # Model registration system
│   │   ├── runner.py           # Experiment Runner execution logic
│   │   ├── tracker.py          # Metadata serialization (metrics.json)
│   │   └── utils.py            # Folder naming indexers
│   ├── evaluation/             # Post-training evaluation
│   │   ├── metrics.py          # F1, Recall, Precision metrics
│   │   ├── confusion_matrix.py
│   │   ├── classification_report.py
│   │   └── plots.py
│   ├── inference/              # Phase 2 inference package
│   │   ├── landcover_mapper.py
│   │   ├── patch_generator.py
│   │   ├── visualizer.py
│   │   ├── change_detector.py
│   │   └── predictor.py
│   └── utils/                  # Miscellaneous utilities
│       ├── visualization.py    # Matplotlib & Seaborn wrappers
│       └── paths.py            # Centralized Path Manager
│
└── tests/                      # Automated unit test suite
    ├── test_dataset.py
    ├── test_dataloader.py
    ├── test_trainer.py
    └── test_lenet.py
```

---

## 5. Installation
Follow these commands to configure the workspace environment:

```bash
# Clone the repository
git clone https://github.com/Nishchalam/Deforestation-Detection.git
cd Deforestation-Detection

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Dataset
This framework uses the **EuroSAT RGB** dataset.
* **Composition**: 27,000 Sentinel-2 satellite patches of size 64×64 pixels.
* **Classes**: 10 classes corresponding to different land cover categories:
  * Forest, Herbaceous Vegetation, Pasture, River, Sea/Lake, Annual Crop, Permanent Crop, Highway, Residential, Industrial.

### Download Command
The dataset download is automated via the Kaggle API. Ensure you have your `kaggle.json` credentials configured.
```bash
python src/data/download.py
```
This extracts the raw imagery into `data/raw/EuroSAT_RGB/` and sets up the folder structure.

---

## 7. Data Pipeline
The data pipeline handles formatting raw files and creating balanced partitions:
1. Iterates over raw land cover category directories.
2. Formats a structured list mapping `image_path` to labels and class names.
3. Splits data into **Train (80%)**, **Validation (10%)**, and **Test (10%)** subsets.
4. Serializes splits to `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/test.csv`.

---

## 8. EDA
<details>
<summary><b>Click to expand EDA details</b></summary>

Exploratory Data Analysis is performed in `notebooks/01_EDA.ipynb`.
* **Class Distribution Check**: Confirms that classes are uniformly distributed (~2,000 to 3,000 images per class) to avoid class imbalance.
* **Visualizing Channels**: Renders RGB channels to examine spectral qualities of vegetation, urban, and aquatic regions.
* **Statistical Profiling**: Calculates mean and standard deviation per channel across the dataset to inform normalization parameters.
</details>

---

## 9. Preprocessing
Preprocessing and augmentation steps defined in `src/data/transforms.py`:

| Split | Processing Steps | Purpose |
|---|---|---|
| **Train** | Resize (224, 224) -> Random Horizontal/Vertical Flip -> Random Rotation -> Color Jitter -> ToTensor -> Normalize | Prevents overfitting and teaches invariant orientations. |
| **Validation / Test** | Resize (224, 224) -> ToTensor -> Normalize | Resizes to model input dimensions without spatial distortions. |

*Normalization parameters match ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` to support fine-tuning from pre-trained weights.*

---

## 10. Training Framework
The training logic resides in a modular loop:
* **Trainer (`src/training/trainer.py`)**: Runs training/validation epochs. Completely decoupled from specific checkpoints and logging details.
* **Callbacks (`src/training/callbacks.py`)**: Subscribes to lifecycle hooks (`on_epoch_end`, `on_train_end`) to manage states:
  * **`EarlyStopping`**: Monitors validation loss and halts execution when it plateaus.
  * **`ModelCheckpoint`**: Saves the `best_model.pth` (metric-driven) and `last_model.pth`.
  * **`CSVLogger`**: Logs metrics per epoch directly to a CSV file.

---

## 11. CNN Architectures
The framework implements a model zoo representing the evolutionary timeline of CNNs. All models inherit from `BaseCNN` in `src/models/common.py` (which handles parameters counting, parameter freeze/unfreeze, and checkpoint saving/loading):

| Model | Classification Level | Key Properties |
|---|---|---|
| **LeNet-5** | Classical Base | Customized for 224x224 RGB inputs; uses AdaptiveAvgPool2d. |
| **AlexNet** | Classical Scaling | Uses Dropout, ReLUs, and large convolutional kernels. |
| **VGG16** | Deep Stacking | Stacks 3x3 convolutions to demonstrate depth impact. |
| **GoogLeNet** | Multi-Scale Convolutions | Features parallel Inception blocks and auxiliary classifiers. |
| **ResNet (18 & 50)** | Deep Residuals | Uses skip connections to train deep networks. |
| **EfficientNet-B0** | Compound Scaling | Compound scaling of depth, width, and resolution. |

---

## 12. Experiment Management
<details>
<summary><b>Click to expand details on the Experiment System</b></summary>

We implement a decoupled, configuration-driven experiment system inside `src/experiments/`:
* **`experiment.py`**: A model class containing the state of one run (name, config, history, start and end times).
* **`registry.py`**: A registry enabling dynamic model instantiation using `create_model(model_name)` without hardcoding constructors in `train.py`.
* **`runner.py`**: High-level execution manager. Parses YAML, sets seeds, creates dataloaders, constructs optimizer/scheduler parameters, configures callbacks, runs training, and logs results.
* **`tracker.py`**: Captures runtime metadata (git commit hash, training duration, best accuracy, optimizer, learning rate) and serializes it to `metrics.json`.
* **`utils.py`**: Generates sequential, self-contained run directories (e.g. `LeNet_001`, `LeNet_002`).
</details>

---

## 13. TensorBoard Usage
All experiments log real-time telemetry:
* **Metrics**: Validation Loss, Validation Accuracy, Training Loss, and Training Accuracy logged per epoch.
* **Graphs**: Model computation graphs visualized inside TensorBoard.
* **Hyperparameters**: Optimizers, learning rates, and batch sizes plotted alongside metrics.

To start TensorBoard monitoring:
```bash
tensorboard --logdir outputs/experiments/
```

---

## 14. Configuration Files
Hyperparameters are managed via YAML files in `configs/`.

<details>
<summary><b>View `configs/lenet.yaml` structure</b></summary>

```yaml
experiment:
  name: "lenet"
  seed: 42
  device: "cuda"

model:
  name: "lenet"
  in_channels: 3
  num_classes: 10

dataset:
  name: "EuroSAT"
  batch_size: 32
  num_workers: 4

training:
  epochs: 20
  loss: "cross_entropy"
  optimizer:
    name: "adam"
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    name: "reduce_lr_on_plateau"
    patience: 3
    factor: 0.1
    mode: "min"

callbacks:
  early_stopping:
    enabled: true
    patience: 5
    monitor: "val_loss"
    mode: "min"
  model_checkpoint:
    enabled: true
    monitor: "val_accuracy"
    mode: "max"
  tensorboard:
    enabled: true
  csv_logger:
    enabled: true
```
</details>

---

## 15. Outputs Directory
Instead of storing outputs globally, each experiment generates a self-contained, indexed directory structure:

```text
outputs/experiments/
└── LeNet_001/
    ├── config.yaml            # Config copy used for this run
    ├── training.log           # Plaintext log file
    ├── hyperparams.json       # Hyperparameters copy
    ├── metrics.json           # Tracker metadata (Git hash, duration, etc.)
    ├── metrics_history.json   # Step-by-step metrics in JSON
    ├── metrics_history.csv    # Step-by-step metrics in CSV
    ├── best_model.pth         # Best checkpoint weights
    ├── last_model.pth         # Last epoch checkpoint weights
    ├── figures/               # Matplotlib output figures
    └── tensorboard/           # TensorBoard telemetry files
```

---

## 16. Training
To start a new experiment, run `train.py` pointing to a configuration YAML file:

```bash
# Train LeNet-5
python train.py --config configs/lenet.yaml

# Train ResNet18
python train.py --config configs/resnet18.yaml
```

---

## 17. Evaluation
The evaluation script compares the trained model against the hold-out test split:

```bash
python evaluate.py --checkpoint outputs/experiments/LeNet_001/best_model.pth --config configs/lenet.yaml
```

The script evaluates the model and logs the following metrics:
* Accuracy, Macro Precision, Recall, and F1-score.
* Serializes a Confusion Matrix and Classification Report to the experiment folder.

---

## 18. Inference
For individual patch prediction, run inference on standard images:

```bash
python infer.py --image path/to/sample.png --checkpoint outputs/experiments/LeNet_001/best_model.pth --config configs/lenet.yaml
```

---

## 19. Phase 2 Deforestation Detection Pipeline
In Phase 2, the framework uses the trained land-cover classifier to detect deforestation over time:

```text
Temporal Satellite TIF (Year 1) ──> Patch Slicing (64x64) ──> Classifier Inference ──> Land Cover Map (Year 1)
                                                                                               │
                                                                                               v
Temporal Satellite TIF (Year 2) ──> Patch Slicing (64x64) ──> Classifier Inference ──> Land Cover Map (Year 2)
                                                                                               │
                                                                                               v
                                                                                   Change Detection Algorithm
                                                                                               │
                                                                                               v
                                                                                    Deforestation Map
                                                                                    (Forest -> Non-Forest)
```

### Execution Command
```bash
python infer.py --image1 data/raw/Sentinel2_2020.tif --image2 data/raw/Sentinel2_2025.tif --checkpoint outputs/experiments/LeNet_001/best_model.pth --config configs/lenet.yaml
```
This generates a deforestation change-mask highlighted in red where forest canopy transitions to non-forest (crop, residential, road, bare soil).

---

## 20. Future Work
* Integrate support for **Multispectral (13-band) Sentinel-2 data** to leverage infrared channels.
* Compare CNN backbones with **Vision Transformers (ViT)**.
* Transition from patch-wise classification to **semantic segmentation** (e.g., U-Net, DeepLabV3) for pixel-precise deforestation boundaries.
* Incorporate Global Forest Watch ground-truth datasets for validation.

---

## 21. References
1. **EuroSAT**: Helber, P., et al. "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification." (2019).
2. **LeNet**: LeCun, Y., et al. "Gradient-Based Learning Applied to Document Recognition." (1998).
3. **AlexNet**: Krizhevsky, A., et al. "ImageNet Classification with Deep Convolutional Neural Networks." (2012).
4. **VGG**: Simonyan, K., & Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." (2014).
5. **GoogLeNet**: Szegedy, C., et al. "Going Deeper with Convolutions." (2015).
6. **ResNet**: He, K., et al. "Deep Residual Learning for Image Recognition." (2016).
7. **EfficientNet**: Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." (2019).

---

## 22. Citation
If you use this repository for your research or educational materials, please cite it as:

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

## 23. License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 24. Project Roadmap
Below is the status of the project's milestones:

- [x] Phase 1 - Baseline Setup (EuroSAT preparation, Data pipeline)
- [x] Phase 1 - Base model architecture definition (LeNet-5)
- [x] Phase 1 - Reusable Trainer Engine implementation
- [x] Phase 3 - Custom registration system and Model Zoo interface
- [x] Phase 3 - Decoupled experiment runner, tracker, and config parser
- [x] Phase 3 - Multi-backend logger updates (JSON/CSV histories)
- [x] Phase 3 - Unit tests suite and dry-runs verification
- [x] Phase 3 - Overhaul README.md and document structures
- [ ] Phase 1 - CNN Evolution implementation (AlexNet, VGG16, GoogLeNet, ResNet, EfficientNet)
- [ ] Phase 2 - Deforestation Detection Pipeline (Patch extractors, land-cover mapping, temporal change-detection)
- [ ] Deployment - Streamlit App / Visualization UI
