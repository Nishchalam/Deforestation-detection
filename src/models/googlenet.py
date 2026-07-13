"""
GoogLeNet (Inception-v1)

Reference
---------
C. Szegedy et al.,
"Going Deeper with Convolutions,"
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

Original Architecture
---------------------
Input (224x224x3)

The architecture is built using "Inception Modules".
Each Inception Module applies four parallel operations:
1. 1x1 Conv
2. 1x1 Conv -> 3x3 Conv
3. 1x1 Conv -> 5x5 Conv
4. 3x3 MaxPool -> 1x1 Conv

These parallel branches are concatenated along the channel dimension.
The 1x1 convolutions act as dimensionality reduction bottlenecks to keep
computational complexity in check.

Main Branch:
→ Conv 7x7, stride=2, padding=3
→ MaxPool 3x3, stride=2, padding=1
→ Conv 1x1
→ Conv 3x3, padding=1
→ MaxPool 3x3, stride=2, padding=1

→ Inception 3a, Inception 3b
→ MaxPool 3x3, stride=2, padding=1

→ Inception 4a, 4b, 4c, 4d, 4e
→ MaxPool 3x3, stride=2, padding=1

→ Inception 5a, 5b
→ Global Average Pooling (7x7 -> 1x1)
→ Dropout(0.4)
→ Linear(1000)

Architectural Decisions
-----------------------
- Network-in-Network approach (1x1 convs) is used heavily for dimension reduction.
- Multiple kernel sizes (1x1, 3x3, 5x5) operate at the same level, allowing the network
  to extract features at varying scales simultaneously.
- Global Average Pooling replaces the dense fully connected layers found in AlexNet and VGG,
  drastically reducing the parameter count (GoogLeNet has ~6.8M parameters vs VGG16's ~138M).
- Original GoogLeNet included Auxiliary Classifiers to combat vanishing gradients, but 
  they are omitted here for simplicity, focusing purely on the main feature extraction path.

Adaptation for this Project
---------------------------
- AdaptiveAvgPool2d(1, 1) is used at the end to guarantee a 1x1 output regardless of input size.
- Output classes changed to 10 for EuroSAT.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block (Conv2d + ReLU).
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # We can omit BatchNorm to stay true to the 2015 paper, 
        # though modern implementations (like Inception-v2+) use it.
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class InceptionModule(nn.Module):
    """
    Inception Module.
    """
    def __init__(
        self,
        in_channels,
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        out_pool,
    ):
        super().__init__()

        # Branch 1: 1x1 Conv
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        # Branch 2: 1x1 Conv -> 3x3 Conv
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        # Branch 3: 1x1 Conv -> 5x5 Conv
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        # Branch 4: 3x3 MaxPool -> 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class GoogLeNet(nn.Module):
    """
    Adapted GoogLeNet (Inception-v1) for EuroSAT classification.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()

        # --------------------------------------------------
        # Feature Extractor
        # --------------------------------------------------

        self.features = nn.Sequential(
            
            # Initial Layers
            ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Inception Block 3
            InceptionModule(192, 64, 96, 128, 16, 32, 32),   # 3a
            InceptionModule(256, 128, 128, 192, 32, 96, 64), # 3b
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Inception Block 4
            InceptionModule(480, 192, 96, 208, 16, 48, 64),  # 4a
            InceptionModule(512, 160, 112, 224, 24, 64, 64), # 4b
            InceptionModule(512, 128, 128, 256, 24, 64, 64), # 4c
            InceptionModule(512, 112, 144, 288, 32, 64, 64), # 4d
            InceptionModule(528, 256, 160, 320, 32, 128, 128), # 4e
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Inception Block 5
            InceptionModule(832, 256, 160, 320, 32, 128, 128), # 5a
            InceptionModule(832, 384, 192, 384, 48, 128, 128), # 5b

            # Adaptive Pooling to (1,1) regardless of spatial dimensions
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        Initialize network weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
