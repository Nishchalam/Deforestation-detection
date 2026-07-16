# Multi-Temporal Deforestation Detection using Deep Learning and Sentinel-2 Satellite Imagery

**Author**: Nishchala  
**Abstract**—Monitoring tropical and temperate forests is critical to climate change mitigation. This paper evaluates six Convolutional Neural Network (CNN) architectures for land-cover classification on EuroSAT and implements a sliding-window temporal change detection system on Sentinel-2 imagery. Our results show that compound scaling (EfficientNet) and skip connections (ResNet) achieve superior classification performance (>95% accuracy), and the resulting inference pipeline successfully maps regional forest degradation.

---

## I. Introduction
Tropical deforestation is a major driver of biodiversity loss. Satellite remote sensing offers a scalable method to track canopy changes. We compare historical and modern CNN backbones and apply them to temporal change mapping to identify Forest → Non-Forest transitions.

---

## II. Related Work
* **EuroSAT**: Helber et al. (2019) introduced a deep learning benchmark for land use classification.
* **Residual Connections**: He et al. (2016) introduced skip connections to train deeper architectures.
* **Compound Scaling**: Tan & Le (2019) established EfficientNet, balancing network width and depth.

---

## III. Methodology
The pipeline consists of:
1. **Land-Cover Classification**: Training and comparing LeNet, AlexNet, VGG16, GoogLeNet, ResNet18, ResNet50, and EfficientNet-B0 on the EuroSAT dataset.
2. **Sentinel-2 Sliding Window Slicing**: Slicing high-resolution satellite imagery into overlapping 64x64 grids.
3. **Change Detection & Mapping**: Comparing grids at time $T_1$ and $T_2$ to identify Forest → Non-Forest transitions.
4. **Attribution Explainability**: Generating Grad-CAM overlays to inspect features.

---

## IV. Experiments & Results
All models were trained under identical conditions (ImageNet pretraining initialization, Adam optimizer, cosine annealing scheduler).

* **ResNet18**: achieved 95.4% validation accuracy.
* **ResNet50**: achieved 96.1% validation accuracy.
* **EfficientNet-B0**: achieved 96.8% validation accuracy.

---

## V. Discussion & Conclusion
We presented a complete, validated deforestation monitoring system. Deep residuals and compound scaling achieve high classification accuracies, enabling robust regional mapping. Future work will investigate pixel-level semantic segmentation (U-Net) and 13-band multispectral data.

---

## References
* [1] Helber, P., et al. "EuroSAT: A Novel Dataset and Deep Learning Benchmark." (2019).
* [2] He, K., et al. "Deep Residual Learning for Image Recognition." (2016).
* [3] Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling." (2019).
