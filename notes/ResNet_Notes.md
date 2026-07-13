# ResNet Architecture Notes

ResNet (Residual Networks) was developed by Kaiming He et al. in 2015, enabling training of extremely deep networks (152+ layers).

## Key Innovations
- **Residual (Skip) Connections**: Adds the input to the layer's output: \(F(x) + x\).
- **Identity Mapping**: Allows gradients to flow directly back through the shortcut connections without degradation, solving the vanishing gradient problem.
- **Bottleneck Blocks**: Uses 1x1, 3x3, 1x1 convolutions in deeper versions (ResNet50+) to reduce parameter footprint.
