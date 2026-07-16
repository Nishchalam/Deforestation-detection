# 📖 Project Guide: Land-Cover Classification and Deforestation Detection

This companion guide is designed for developers, students, and recruiters to explore the project goals, repository architecture, and implementation details.

---

## 1. Project Overview & Objectives
This project implements an end-to-end deep learning pipeline to map land-cover classifications and monitor deforestation over time. Using Sentinel-2 satellite imagery, we identify 10 distinct land-cover classes (e.g., Forest, Highway, Residential, Annual Crop) and trace Forest → Non-Forest transitions (deforestation) across multi-temporal frames.

### Core Objectives:
* Compare historical and modern CNN backbones on the EuroSAT benchmark dataset.
* Implement a robust, config-driven pipeline for training, evaluation, and inference.
* Perform multi-temporal sliding window classification to construct land-cover maps.
* Detect forest canopy loss, compute transition matrices, and visualize change overlays.
* Provide model interpretability through Explainable AI (XAI) attributions.

---

## 2. Directory Layout & Setup
The codebase is structured logically to separate experiment configurations, training logic, inference pipelines, and automated reporting:
* `configs/`: Manage training hyperparameters via YAML.
* `notebooks/`: Self-contained tutorial artifacts for CNN evolution, inference, and change mapping.
* `reports/`: Automatically generated benchmark comparisons and validation files.
* `src/`: Reusable python library containing dataloaders, model architectures, training executors, and interpretability modules.
* `tests/`: Extensive pytest unit tests covering 100% of pipeline modules.

For installation details, please refer to the root [README.md](../README.md).

---

## 3. Dataset & CNN Evolution
The pipeline uses **EuroSAT RGB**, a dataset of 27,000 Sentinel-2 patches of size 64x64 pixels. We compare the following CNN backbones:
1. **LeNet-5**: Historical baseline adapted to 224x224 RGB inputs.
2. **AlexNet**: Showed the importance of Dropout, ReLU, and GPU-driven stacking.
3. **VGG16**: Demonstrated that depth with small 3x3 convolutions is highly effective.
4. **GoogLeNet**: Multi-scale convolutions utilizing parallel Inception modules.
5. **ResNet18 / ResNet50**: Solved gradient degradation via residual shortcut bypasses.
6. **EfficientNet-B0**: Balances width, depth, and input resolutions using compound scaling.

---

## 4. End-to-End Pipeline Workflow
The operational workflow of the deforestation detection system is structured as follows:

```text
Temporal Satellite TIF (Year A) ──> Patch Slicing (64x64) ──> Classifier Inference ──> Land Cover Map (Year A)
                                                                                                │
                                                                                                v
Temporal Satellite TIF (Year B) ──> Patch Slicing (64x64) ──> Classifier Inference ──> Land Cover Map (Year B)
                                                                                                │
                                                                                                v
                                                                                    Change Detection Algorithm
                                                                                                │
                                                                                                v
                                                                                    Deforestation Map
                                                                                    (Forest -> Non-Forest)
                                                                                                │
                                                                                                v
                                                                                    Validation & XAI Dashboards
```

---

## 5. Performance Validation & Explainability
Our pipeline includes robust diagnostic tools:
* **Validation**: Checks classifications against ground truth to calculate accuracy, sensitivity, specificity, Dice coefficients, and IoU.
* **Explainable AI (XAI)**: We implement Grad-CAM, Grad-CAM++, Vanilla Saliency, Occlusion Sensitivity, and LIME simulators to map the input pixels influencing classification decisions.

---

## 6. Future Work
* Integrate raw 13-band multispectral Sentinel-2 TIFF images.
* Deploy a web-based Streamlit dashboard.
* Implement semantic segmentation architectures (U-Net).
