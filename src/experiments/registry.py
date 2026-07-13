"""
registry.py

Implements a central model registry.
"""

from typing import Dict, Callable
import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str):
    """Decorator to register a model class or factory function."""
    def decorator(cls_or_fn):
        _MODEL_REGISTRY[name.lower().strip()] = cls_or_fn
        return cls_or_fn
    return decorator

def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Creates and returns the requested model instance.
    
    Parameters
    ----------
    model_name : str
        Name of the model (e.g. 'lenet', 'resnet18').
    **kwargs
        Hyperparameters passed to the model constructor (e.g., in_channels, num_classes).
    """
    name_clean = model_name.lower().strip()
    if name_clean not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name_clean](**kwargs)

# Pre-register existing architectures to avoid circular imports and keep loading clean
from src.models.lenet import LeNet
from src.models.alexnet import AlexNet
from src.models.vgg import VGG16
from src.models.googlenet import GoogLeNet
from src.models.resnet import ResNet18, ResNet50
from src.models.efficientnet import EfficientNetB0

_MODEL_REGISTRY["lenet"] = LeNet
_MODEL_REGISTRY["alexnet"] = AlexNet
_MODEL_REGISTRY["vgg16"] = VGG16
_MODEL_REGISTRY["googlenet"] = GoogLeNet
_MODEL_REGISTRY["resnet18"] = ResNet18
_MODEL_REGISTRY["resnet50"] = ResNet50
_MODEL_REGISTRY["efficientnet-b0"] = EfficientNetB0
_MODEL_REGISTRY["efficientnet_b0"] = EfficientNetB0
