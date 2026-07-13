# VGG Architecture Notes

VGG16 and VGG19 were developed by the Visual Geometry Group at Oxford in 2014.

## Key Innovations
- **Small kernels (3x3)**: Stacking two 3x3 convs has the effective receptive field of one 5x5 conv, but has fewer parameters and more non-linearities.
- **Homogeneous design**: The network only consists of 3x3 convs, ReLU, and 2x2 max pooling.
- **Deep network**: VGG proved that network depth is crucial for high performance.
