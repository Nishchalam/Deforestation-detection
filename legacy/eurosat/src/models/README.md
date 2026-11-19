# CNN Architectures

This directory contains all convolutional neural network architectures used in this project.

The objective is to compare the evolution of CNNs for land-cover classification on the EuroSAT dataset using a common training and preprocessing pipeline.

---

# Design Philosophy

Each model is implemented from scratch whenever possible to understand its architectural design and evolution.

All models use

- the same dataset
- the same train/validation/test split
- the same augmentation pipeline
- the same optimizer (unless otherwise stated)
- the same evaluation metrics

This ensures that differences in performance arise primarily from the network architecture rather than differences in preprocessing or training.

---

# Implemented Models

- LeNet-5
- AlexNet
- VGG16
- GoogLeNet (Inception-v1)
- ResNet18
- ResNet50
- EfficientNet-B0

---

# LeNet Adaptation

The original LeNet-5 (LeCun et al., 1998) was designed for handwritten digit recognition on the MNIST dataset.

Original assumptions:

- Input size: 32 × 32
- Single-channel grayscale images
- Average Pooling
- tanh activations
- Output classes: 10 digits

This project adapts LeNet for EuroSAT while preserving its overall architecture.

Changes introduced:

- Input changed from **32×32×1** to **224×224×3**
- RGB images instead of grayscale
- Adaptive Average Pooling before the classifier to maintain a fixed feature size
- Number of output neurons changed from 10 digits to 10 EuroSAT land-cover classes

The convolutional feature extractor follows the original LeNet design as closely as possible. Only the input interface and classifier are adapted to accommodate modern satellite imagery while keeping the preprocessing pipeline identical across all evaluated CNN architectures.

---

# Why Not Resize to 32×32?

Resizing only LeNet inputs to 32×32 would introduce an additional preprocessing branch.

Since this project aims to compare multiple CNN architectures fairly, all models receive the same 224×224 RGB images.

This isolates the effect of architectural differences and avoids confounding factors introduced by different preprocessing pipelines.