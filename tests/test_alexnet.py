import pytest
import torch
from src.models import create_model

def test_alexnet_initialization():
    model = create_model("alexnet", in_channels=3, num_classes=10)
    assert model is not None

def test_alexnet_forward_pass():
    model = create_model("alexnet", in_channels=3, num_classes=10)
    # AlexNet expects 224x224 RGB input
    x = torch.randn(2, 3, 224, 224) 
    out = model(x)
    assert out.shape == (2, 10)

def test_alexnet_parameter_count():
    model = create_model("alexnet", in_channels=3, num_classes=10)
    assert model.num_parameters() > 0
