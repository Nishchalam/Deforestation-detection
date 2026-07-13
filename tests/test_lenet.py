import pytest
import torch
from src.models import create_model

def test_lenet_initialization():
    model = create_model("lenet", in_channels=3, num_classes=10)
    assert model is not None

def test_lenet_forward_pass():
    model = create_model("lenet", in_channels=3, num_classes=10)
    # EuroSAT images are typically resized to 224x224 in our pipeline
    x = torch.randn(2, 3, 224, 224) 
    out = model(x)
    assert out.shape == (2, 10)

def test_lenet_parameter_count():
    model = create_model("lenet", in_channels=3, num_classes=10)
    # Ensure parameter counting method from BaseCNN works
    assert model.num_parameters() > 0
