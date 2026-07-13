"""
VGG16

Reference
---------
K. Simonyan and A. Zisserman,
"Very Deep Convolutional Networks for Large-Scale Image Recognition,"
International Conference on Learning Representations (ICLR), 2015.

Original Architecture
---------------------
Input (224x224x3)

Block 1:
→ Conv(64, 3x3, padding=1) + ReLU
→ Conv(64, 3x3, padding=1) + ReLU
→ MaxPool(2x2, stride=2)

Block 2:
→ Conv(128, 3x3, padding=1) + ReLU
→ Conv(128, 3x3, padding=1) + ReLU
→ MaxPool(2x2, stride=2)

Block 3:
→ Conv(256, 3x3, padding=1) + ReLU
→ Conv(256, 3x3, padding=1) + ReLU
→ Conv(256, 3x3, padding=1) + ReLU
→ MaxPool(2x2, stride=2)

Block 4:
→ Conv(512, 3x3, padding=1) + ReLU
→ Conv(512, 3x3, padding=1) + ReLU
→ Conv(512, 3x3, padding=1) + ReLU
→ MaxPool(2x2, stride=2)

Block 5:
→ Conv(512, 3x3, padding=1) + ReLU
→ Conv(512, 3x3, padding=1) + ReLU
→ Conv(512, 3x3, padding=1) + ReLU
→ MaxPool(2x2, stride=2)

Classifier:
→ Flatten (to 512 * 7 * 7)
→ FC(4096) + ReLU + Dropout(0.5)
→ FC(4096) + ReLU + Dropout(0.5)
→ FC(1000)

Architectural Decisions
-----------------------
- VGG pioneered the idea that instead of using large receptive fields (like 11x11 or 7x7), 
  one can stack multiple 3x3 convolutions to achieve the same effective receptive field 
  with fewer parameters and more non-linearities.
- Spatial dimensions are consistently halved via max pooling, while channel depth is 
  consistently doubled until it reaches 512.

Adaptation for this Project
---------------------------
- AdaptiveAvgPool2d(7, 7) is inserted before the classifier. While 224x224 inputs 
  naturally pool to 7x7, this adaptive layer ensures the classifier does not crash 
  if inputs of slightly different sizes are provided.
- Output features in the final FC layer changed to 10 for EuroSAT.
"""

import torch
import torch.nn as nn


class VGG16(nn.Module):
    """
    VGG16 adapted for EuroSAT classification.

    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels.

    num_classes : int, default=10
        Number of output classes.
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

            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Adaptive pooling for arbitrary input sizes
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, num_classes),
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

        Conv2d:
            Kaiming Normal (He initialization) due to ReLU activations.

        Linear:
            Normal distribution with mean=0, std=0.01.

        Bias:
            Zero
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
