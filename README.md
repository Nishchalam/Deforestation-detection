# 🌍 Deforestation Detection using Deep Learning and Sentinel-2 Satellite Imagery

## Overview

This repository presents an end-to-end deep learning pipeline for land-cover classification and deforestation detection using Sentinel-2 satellite imagery.

The project is divided into two independent modules:

1. **Phase 1: Land Cover Classification**
   - Train and compare multiple CNN architectures on the EuroSAT RGB dataset.
   - Study the evolution of convolutional neural networks from classical architectures (LeNet, AlexNet, VGG) to modern ones (ResNet, EfficientNet).

2. **Phase 2: Deforestation Detection**
   - Apply the best-performing classifier to real-world Sentinel-2 imagery.
   - Generate land-cover maps.
   - Detect forest-to-non-forest transitions across multiple years to create a deforestation map.

---

## Dataset

### EuroSAT RGB

- **Content**: 27,000 Sentinel-2 satellite images.
- **Classes**: 10 land-cover classes (e.g., Forest, Annual Crop, Highway, etc.).
- **Format**: RGB imagery.
- **Original Size**: 64×64 pixels (rescaled to 224×224 for most modern architectures).

### Sentinel-2

Real-world satellite imagery downloaded for inference, allowing for patch-by-patch analysis of regions over different time periods.

---

## Architecture Comparison

The project systematically implements and compares the following architectures:

- **LeNet-5**: Adapted for 224x224 RGB inputs.
- **AlexNet**: Showcasing deeper networks and dropout.
- **VGG16**: Emphasizing small (3x3) convolutions and depth.
- **GoogLeNet (Inception-v1)**: Introducing the Inception module and auxiliary classifiers.
- **ResNet18 / ResNet50**: Introducing residual connections to solve vanishing gradients.
- **EfficientNet-B0**: Utilizing compound scaling for optimal performance and efficiency.

---

## Project Workflow

```text
Dataset Download -> EDA -> Preprocessing -> Data Augmentation
      ↓
Model Selection (LeNet, AlexNet, VGG, ResNet, etc.)
      ↓
Model Training (with TensorBoard logging & Checkpointing)
      ↓
Model Evaluation & Comparison
      ↓
Sentinel-2 Image Inference (Patch Extraction)
      ↓
Land Cover Mapping
      ↓
Change Detection (Forest -> Non-Forest)
      ↓
Deforestation Map Generation
```

---

## Repository Structure

```text
Deforestation-detection/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── predictions/
├── src/
│   ├── data/          # Dataset, DataLoaders, Transforms
│   ├── models/        # CNN Architectures
│   ├── training/      # Generic Trainer, Metrics, Losses
│   ├── evaluation/
│   ├── inference/     # Predictor, Change Detector, Mapping
│   └── utils/
├── train.py           # Main training script
├── evaluate.py        # Main evaluation script
├── infer.py           # Main inference script
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Nishchalam/Deforestation-Detection.git
cd Deforestation-Detection

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Download & Prepare Dataset
Download the EuroSAT dataset and generate the train/validation/test CSV splits.
```bash
python src/data/download.py
```

### 2. Training
Use `train.py` to train a specific model. Checkpoints and logs will be saved in `outputs/`.
```bash
# Train LeNet
python train.py --model lenet --epochs 20 --batch-size 32 --lr 0.001

# Train ResNet18
python train.py --model resnet18 --epochs 30 --batch-size 64
```

### 3. TensorBoard Monitoring
Monitor training curves and learning rates in real-time.
```bash
tensorboard --logdir outputs/logs
```
*(Insert TensorBoard Screenshot Here)*
`![TensorBoard Example](outputs/figures/tensorboard_placeholder.png)`

### 4. Evaluation
*(Script under development)*
```bash
python evaluate.py --model resnet18 --checkpoint outputs/checkpoints/resnet18/best_model.pth
```

### 5. Inference (Phase 2)
*(Pipeline under development)*
Generate a deforestation map by comparing two Sentinel-2 images.
```bash
python infer.py --image1 path/to/year1.tif --image2 path/to/year2.tif --checkpoint outputs/checkpoints/resnet18/best_model.pth
```

---

## Visualizations

### Land Cover Mapping
*(Insert Screenshot Here)*
`![Land Cover Map Example](outputs/figures/landcover_placeholder.png)`

### Deforestation Detection
*(Insert Screenshot Here)*
`![Deforestation Example](outputs/figures/deforestation_placeholder.png)`

---

## Future Work

- Compare predictions with Global Forest Watch data.
- Explore Vision Transformers (ViT) for Land Cover Classification.
- Deploy the inference pipeline as a Streamlit application.
- Explore semantic segmentation instead of patch-based classification for finer boundaries.

---

## References

1. **EuroSAT**: Helber, P., et al. "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification." (2019)
2. **LeNet**: LeCun, Y., et al. "Gradient-Based Learning Applied to Document Recognition." (1998)
3. **AlexNet**: Krizhevsky, A., et al. "ImageNet Classification with Deep Convolutional Neural Networks." (2012)
4. **VGG**: Simonyan, K., & Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." (2014)
5. **GoogLeNet**: Szegedy, C., et al. "Going Deeper with Convolutions." (2015)
6. **ResNet**: He, K., et al. "Deep Residual Learning for Image Recognition." (2016)
7. **EfficientNet**: Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." (2019)

---
## License
MIT