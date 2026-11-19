"""
EfficientNet-B0

Reference
---------
M. Tan and Q. V. Le,
"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,"
International Conference on Machine Learning (ICML), 2019.

Architectural Decisions
-----------------------
- Conventional CNN scaling typically modifies either depth (number of layers),
  width (number of channels), or image resolution. EfficientNet introduces a 
  "compound scaling" method that uniformly scales all three using fixed coefficients.
- The base network, EfficientNet-B0, was discovered using Neural Architecture Search
  (NAS) optimizing for accuracy and FLOPS.
- The core building block is the Mobile Inverted Bottleneck Convolution (MBConv)
  with Squeeze-and-Excitation (SE) optimization.
- MBConv: Expands channels -> Depthwise Conv -> Squeeze-and-Excitation -> Project channels.
  This is highly parameter-efficient.
- Swish activation function (x * sigmoid(x)) is used instead of standard ReLU.

Adaptation for this Project
---------------------------
- AdaptiveAvgPool2d(1, 1) is used before the classifier.
- Output classes changed to 10 for EuroSAT.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.common import BaseCNN


class Swish(nn.Module):
    """
    Swish Activation Function.
    f(x) = x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.
    Adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution block.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        expand_ratio, 
        se_ratio=0.25
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        
        expanded_channels = in_channels * expand_ratio
        
        layers = []
        
        # 1. Expansion Phase (Pointwise)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish(),
            ])
            
        # 2. Depthwise Convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(
                expanded_channels, 
                expanded_channels, 
                kernel_size, 
                stride, 
                padding, 
                groups=expanded_channels, 
                bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            Swish(),
        ])
        
        # 3. Squeeze-and-Excitation
        reduced_dim = max(1, int(in_channels * se_ratio))
        layers.append(SqueezeExcitation(expanded_channels, reduced_dim))
        
        # 4. Projection Phase (Pointwise)
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


class EfficientNetB0(BaseCNN):
    """
    EfficientNet-B0 adapted for EuroSAT classification.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        
        # MBConv Blocks configuration for B0
        # Format: (expand_ratio, channels, num_layers, stride, kernel_size)
        b0_config = [
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]
        
        blocks = []
        in_c = 32
        for expand_ratio, out_c, num_layers, stride, kernel_size in b0_config:
            for i in range(num_layers):
                s = stride if i == 0 else 1
                blocks.append(
                    MBConv(in_c, out_c, kernel_size, s, expand_ratio)
                )
                in_c = out_c
                
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_c, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        Initialize network weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # EfficientNet uses a scaled fan-out initialization
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
