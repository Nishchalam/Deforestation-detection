# EfficientNet Architecture Notes

EfficientNet was proposed by Mingxing Tan and Quoc V. Le in 2019, achieving state-of-the-art accuracy with much fewer parameters.

## Key Innovations
- **Compound Scaling**: Discovered that network scaling (width, depth, and input resolution) must be balanced systematically using a constant scaling coefficient.
- **MBConv Block**: Uses Mobile Inverted Bottleneck convolutions (originally from MobileNetV2), featuring depthwise separable convolutions and squeeze-and-excitation optimization.
- **Compound Coefficient**: Grid search is used to find optimal scaling coefficients under user-specified constraint parameters.
