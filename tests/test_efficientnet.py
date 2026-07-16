import pytest
import torch
from src.models import create_model

def test_efficientnet_initialization():
    model = create_model("efficientnet-b0", in_channels=3, num_classes=10)
    assert model is not None

def test_efficientnet_forward_pass():
    model = create_model("efficientnet-b0", in_channels=3, num_classes=10)
    x = torch.randn(2, 3, 224, 224) 
    out = model(x)
    assert out.shape == (2, 10)
