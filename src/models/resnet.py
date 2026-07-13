"""
ResNet (Residual Networks)

Reference
---------
K. He, X. Zhang, S. Ren, and J. Sun,
"Deep Residual Learning for Image Recognition,"
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Architectural Decisions
-----------------------
- Deep networks suffer from vanishing/exploding gradients and degradation 
  (where adding more layers leads to higher training error).
- ResNet addresses this via Residual Connections (Skip Connections).
- Instead of learning a direct mapping H(x), the network learns a residual F(x) = H(x) - x,
  which is easier to optimize. If the identity mapping is optimal, it's easier to push
  the residual to zero than to fit an identity mapping with nonlinear layers.
- ResNet18 uses "BasicBlocks" (two 3x3 convolutions).
- ResNet50 uses "BottleneckBlocks" (1x1 -> 3x3 -> 1x1 convolutions). The 1x1 layers 
  reduce and restore dimensions (bottleneck), allowing the network to be much deeper
  without blowing up the parameter count.

Adaptation for this Project
---------------------------
- The input stem (Conv 7x7 -> MaxPool) remains standard for 224x224 inputs.
- AdaptiveAvgPool2d(1, 1) is used before the classifier.
- Output classes changed to 10 for EuroSAT.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Basic Block for ResNet18 and ResNet34.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels * self.expansion, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = self.relu(out)
        
        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck Block for ResNet50, ResNet101, and ResNet152.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 1x1 convolution (dimension reduction)
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution (dimension restoration/expansion)
        self.conv3 = nn.Conv2d(
            out_channels, 
            out_channels * self.expansion, 
            kernel_size=1, 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels * self.expansion, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Base ResNet architecture.
    """

    def __init__(self, block, num_blocks, in_channels=3, num_classes=10):
        super().__init__()
        
        self.in_planes = 64

        # --------------------------------------------------
        # Feature Extractor
        # --------------------------------------------------
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        
        x = self.fc(x)
        
        return x

    def _initialize_weights(self):
        """
        Initialize network weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def ResNet18(in_channels=3, num_classes=10):
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)


def ResNet50(in_channels=3, num_classes=10):
    """Constructs a ResNet-50 model."""
    return ResNet(BottleneckBlock, [3, 4, 6, 3], in_channels, num_classes)
