# GoogLeNet (Inception-v1) Notes

GoogLeNet was proposed by Google in 2014, winning the ImageNet challenge.

## Key Innovations
- **Inception module**: Runs parallel convolutions of different sizes (1x1, 3x3, 5x5) and 3x3 max pooling, and concatenates the outputs.
- **1x1 Convolutions**: Used for dimensionality reduction before expensive 3x3 and 5x5 convs.
- **Auxiliary Classifiers**: Injected gradients during training to solve vanishing gradients.
- **Global Average Pooling**: Replaced fully connected layers at the end, reducing parameters.
