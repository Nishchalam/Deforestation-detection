# Technical Report: Deforestation Monitoring System using Deep Learning

**Author**: Nishchala  
**Date**: July 2026  

---

## Abstract
This report presents the design and verification of a deep-learning-based deforestation monitoring system. We analyze classic and modern Convolutional Neural Network (CNN) backbones, implement multi-temporal Sentinel-2 inference, and compile model explainability attributions.

---

## 1. Introduction & Objectives
Monitoring forest canopy degradation is key to combating climate change. Remote sensing platforms, like the Sentinel-2 mission, provide high-resolution multispectral imagery. Applying deep learning classifiers to these images enables automated, regional monitoring of land-use changes.

### System Objectives:
1. Compare CNN backbones on the EuroSAT RGB classification dataset.
2. Develop a sliding-window grid mapper to reconstruct land-cover maps.
3. Compute temporal transition matrices to detect Forest → Non-Forest changes.
4. Integrate Explainable AI (XAI) overlays to evaluate model decision criteria.

---

## 2. Methodology & System Design
The system architecture consists of decoupled pipelines:

```text
Input Satellite Bands ──> Preprocessing ──> CNN Classifier ──> Land Cover Grid ──> Change Detection ──> Validation & XAI
```

### A. Preprocessing
Images are resized to 224x224 and normalized using ImageNet parameters:
$$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### B. CNN Zoo
* **LeNet-5 (1998)**: Classical 2-layer convolution baseline.
* **AlexNet (2012)**: Relies on Dropout and ReLUs.
* **VGG16 (2014)**: Small 3x3 filter stacking.
* **GoogLeNet (2014)**: Multi-scale parallel Inception blocks.
* **ResNet18 / ResNet50 (2015)**: Skip connections to solve vanishing gradients.
* **EfficientNet-B0 (2019)**: Optimized scale balance across width, depth, and resolution.

---

## 3. Temporal Change & Deforestation Mapping
We compare classification grids at Time $T_1$ and Time $T_2$.
* **Transition Matrix**: A 10x10 matrix $M$ where $M_{i,j}$ counts transitions from class $i$ to class $j$.
* **Deforestation Mask**: A binary mask $D$ where:
  $$D = \begin{cases} 1 & \text{if } \text{Class}(T_1) = \text{'Forest'} \text{ and } \text{Class}(T_2) \neq \text{'Forest'} \\ 0 & \text{otherwise} \end{cases}$$

---

## 4. Performance Validation & XAI
* **Validation Metrics**: Computes Accuracy, F1-Score, specificity, and Intersection-over-Union (IoU):
  $$\text{IoU} = \frac{TP}{TP + FP + FN}$$
* **Explainability (Grad-CAM)**: Captures gradients flowing into the last convolutional layers to generate activation heatmaps:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

---

## 5. Discussion & Future Work
Modern architectures (ResNet/EfficientNet) achieve >95% accuracy on EuroSAT, showing high sensitivity for distinguishing dense canopy segments from roads and agricultural developments. Future work will integrate raw 13-band multispectral data and transition from patch-based mapping to pixel-level semantic segmentation (U-Net).

---

## References
1. Helber, P., et al. "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification." (2019).
2. He, K., et al. "Deep Residual Learning for Image Recognition." (2016).
3. Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." (2019).
