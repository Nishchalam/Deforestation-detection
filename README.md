# Deforestation Detection using Deep Learning and Sentinel-2 Satellite Imagery

## Overview

This project develops an end-to-end deep learning pipeline for automated deforestation detection from satellite imagery.

A ResNet50-based land-cover classifier is first trained on the EuroSAT dataset using transfer learning. The trained classifier is then applied to Sentinel-2 imagery from different years to generate land-cover maps. By comparing these maps, forest-to-non-forest transitions are identified as potential deforestation events.

---

## Features

- Transfer Learning using ResNet50
- EuroSAT Land Cover Classification
- Sentinel-2 Satellite Image Processing
- Patch-based Inference
- Land Cover Mapping
- Deforestation Detection
- Change Visualization
- Confusion Matrix & Classification Report

---

## Pipeline

EuroSAT Dataset

↓

Image Preprocessing

↓

Transfer Learning (ResNet50)

↓

Land Cover Classification

↓

Sentinel-2 Images

↓

Patch Extraction

↓

Inference

↓

Land Cover Maps

↓

Change Detection

↓

Potential Deforestation Map

---

## Dataset

### EuroSAT

- 27,000 Sentinel-2 RGB Images
- 10 Land Cover Classes

### Sentinel-2

- 2017 Image
- 2026 Image

---

## Model

- Backbone: ResNet50
- Transfer Learning
- Cross Entropy Loss
- Adam Optimizer

---

## Results

The model predicts land-cover classes for each image patch.

Potential deforestation is identified wherever:

Forest → Non-Forest

---

## Future Work

- Multi-spectral Sentinel-2
- Vision Transformers
- Semantic Segmentation (U-Net)
- Global Forest Watch Validation
- Temporal Deep Learning

---

## Repository Structure

(Add folder tree here)

---

## License

MIT