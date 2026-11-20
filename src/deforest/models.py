"""U-Net with a ResNet-18 encoder (segmentation_models_pytorch)."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def build_unet(encoder: str = "resnet18", encoder_weights: str | None = "imagenet",
               in_channels: int = 6) -> torch.nn.Module:
    """For in_channels != 3 with pretrained weights, smp adapts the first conv by cycling the
    RGB filters (https://smp.readthedocs.io/en/latest/insights.html#input-channels)."""
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                    in_channels=in_channels, classes=1)
