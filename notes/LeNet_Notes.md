# LeNet-5 Architecture Notes

LeNet-5 was designed by Yann LeCun et al. in 1998 for handwritten character recognition.

## Architecture details
- **Input**: 32x32 grayscale images
- **Layer 1 (C1)**: Convolution (6 filters, 5x5 kernel, stride=1) -> Output: 28x28x6
- **Layer 2 (S2)**: Average Pooling (2x2 kernel, stride=2) -> Output: 14x14x6
- **Layer 3 (C3)**: Convolution (16 filters, 5x5 kernel, stride=1, sparse connection) -> Output: 10x10x16
- **Layer 4 (S4)**: Average Pooling (2x2 kernel, stride=2) -> Output: 5x5x16
- **Layer 5 (C5)**: Convolution (120 filters, 5x5 kernel, stride=1) -> Output: 1x1x120 (treated as fully connected)
- **Layer 6 (F6)**: Fully Connected (84 units)
- **Output**: 10 units (using Euclidean Radial Basis Function network)

## Adapted implementation for EuroSAT
In this project, LeNet-5 is adapted to accept 224x224 RGB inputs.
An `AdaptiveAvgPool2d((5,5))` layer is added before the classifier to accommodate the larger spatial dimensions.
