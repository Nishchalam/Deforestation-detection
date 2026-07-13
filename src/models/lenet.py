"""
LeNet-5

Reference
---------
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner,
"Gradient-Based Learning Applied to Document Recognition,"
Proceedings of the IEEE, 1998.

Original Architecture
---------------------
Input (32×32×1)

→ Conv(6, 5×5)
→ tanh
→ AvgPool

→ Conv(16, 5×5)
→ tanh
→ AvgPool

→ FC(120)
→ tanh

→ FC(84)
→ tanh

→ FC(10)

Adaptation for this Project
---------------------------
The original LeNet was designed for 32×32 grayscale images.

For EuroSAT:

- Input changed to 224×224×3
- Adaptive Average Pooling added before the classifier
- Output changed to 10 EuroSAT classes

The convolutional architecture remains faithful to the
original paper.
"""

import torch
import torch.nn as nn
from src.models.common import BaseCNN

class LeNet(BaseCNN):
    """
    Adapted LeNet-5 for EuroSAT classification.

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

            # C1
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0,
            ),

            nn.Tanh(),

            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),

            # C3
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
            ),

            nn.Tanh(),

            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Adaptation for larger inputs
            nn.AdaptiveAvgPool2d((5, 5)),
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                16 * 5 * 5,
                120,
            ),

            nn.Tanh(),

            nn.Linear(
                120,
                84,
            ),

            nn.Tanh(),

            nn.Linear(
                84,
                num_classes,
            ),
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
            Xavier Uniform

        Linear:
            Xavier Uniform

        Bias:
            Zero
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)