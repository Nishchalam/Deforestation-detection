"""
AlexNet

Reference
---------
A. Krizhevsky, I. Sutskever, and G. E. Hinton,
"ImageNet Classification with Deep Convolutional Neural Networks,"
Advances in Neural Information Processing Systems, 2012.

Original Architecture
---------------------
Input (227x227x3)

→ Conv(96, 11x11, stride=4, padding=0)
→ ReLU
→ MaxPool(3x3, stride=2)
→ LRN (Local Response Normalization)

→ Conv(256, 5x5, padding=2)
→ ReLU
→ MaxPool(3x3, stride=2)
→ LRN

→ Conv(384, 3x3, padding=1)
→ ReLU

→ Conv(384, 3x3, padding=1)
→ ReLU

→ Conv(256, 3x3, padding=1)
→ ReLU
→ MaxPool(3x3, stride=2)

→ Flatten
→ Dropout(0.5)
→ FC(4096)
→ ReLU

→ Dropout(0.5)
→ FC(4096)
→ ReLU

→ FC(1000)

Adaptation for this Project
---------------------------
- Input size adapted to 224x224x3 (using padding=2 in the first Conv layer), which is standard practice today.
- Channel sizes are adjusted to [64, 192, 384, 256, 256] matching PyTorch's torchvision implementation, reflecting the removal of the 2-GPU grouping split present in the original paper.
- Local Response Normalization (LRN) is omitted, as modern research shows it offers little benefit (Batch Normalization is usually preferred today, but we leave it out to stay closer to the original paper's spirit).
- AdaptiveAvgPool2d(6, 6) is added before the classifier to ensure the fully connected layers always receive a consistent input size, allowing varying input resolutions while maintaining the exact classifier structure.
- Output classes changed to 10 for EuroSAT.
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Adapted AlexNet for EuroSAT classification.

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

            # Layer 1
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 2
            nn.Conv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 3
            nn.Conv2d(
                in_channels=192,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # Layer 5
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Adaptation for dynamic input sizes
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

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
            Normal distribution with mean=0, std=0.01 (as per original paper).

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
