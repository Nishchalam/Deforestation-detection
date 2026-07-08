# 🌍 Deforestation Detection using Deep Learning and Sentinel-2 Satellite Imagery

## Overview

This repository presents an end-to-end deep learning pipeline for land-cover classification and deforestation detection using Sentinel-2 satellite imagery.

The project is divided into two independent modules:

1. **Land Cover Classification**
   - Train and compare multiple CNN architectures on the EuroSAT dataset.
   - Study the evolution of convolutional neural networks from LeNet to EfficientNet.

2. **Deforestation Detection**
   - Apply the best-performing classifier to Sentinel-2 imagery.
   - Generate land-cover maps.
   - Detect forest-to-non-forest transitions across multiple years.

---

## Objectives

- Perform comprehensive exploratory data analysis (EDA).
- Build a reproducible preprocessing pipeline.
- Compare classical and modern CNN architectures.
- Investigate transfer learning.
- Visualize learned features using Grad-CAM.
- Generate land-cover maps.
- Detect potential deforestation.
- Compare predictions with Global Forest Watch (future work).

---

## Dataset

### EuroSAT RGB

- 27,000 Sentinel-2 images
- 10 land-cover classes
- RGB imagery
- Image size: 64×64 pixels

### Sentinel-2

Real-world satellite imagery downloaded using the Google Earth Engine API.

---

## CNN Architectures

- LeNet-5
- AlexNet
- VGG16
- GoogLeNet (Inception-v1)
- ResNet18
- ResNet50
- EfficientNet-B0

---

## Project Workflow

Dataset Download

↓

Exploratory Data Analysis

↓

Preprocessing

↓

Data Augmentation

↓

CNN Training

↓

Model Comparison

↓

Transfer Learning

↓

Sentinel-2 Inference

↓

Land Cover Mapping

↓

Change Detection

↓

Potential Deforestation Map

---

## Repository Structure

(Add directory tree)

---

## Installation

```bash
git clone https://github.com/Nishchalam/Deforestation-Detection.git

cd Deforestation-Detection

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## Download Dataset

```bash
python src/data/download.py
```

---

## Project Status

- [x] Repository Setup
- [x] Dataset Download
- [ ] Exploratory Data Analysis
- [ ] Data Preprocessing
- [ ] LeNet
- [ ] AlexNet
- [ ] VGG16
- [ ] GoogLeNet
- [ ] ResNet18
- [ ] ResNet50
- [ ] EfficientNet
- [ ] Model Comparison
- [ ] Transfer Learning
- [ ] Sentinel-2 Inference
- [ ] Deforestation Detection
- [ ] Streamlit Deployment

---

## License

MIT