from .lenet import LeNet
from .alexnet import AlexNet
from .vgg import VGG16
from .googlenet import GoogLeNet
from .resnet import ResNet18, ResNet50
from .efficientnet import EfficientNetB0

def create_model(model_name: str, **kwargs):
    """
    Simple factory function to create model instances directly.
    """
    name_clean = model_name.lower().replace("_", "").replace("-", "").strip()
    
    if name_clean == "lenet":
        return LeNet(**kwargs)
    elif name_clean == "alexnet":
        return AlexNet(**kwargs)
    elif name_clean == "vgg16":
        return VGG16(**kwargs)
    elif name_clean == "googlenet":
        return GoogLeNet(**kwargs)
    elif name_clean == "resnet18":
        return ResNet18(**kwargs)
    elif name_clean == "resnet50":
        return ResNet50(**kwargs)
    elif name_clean == "efficientnetb0":
        return EfficientNetB0(**kwargs)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available: lenet, alexnet, vgg16, googlenet, resnet18, resnet50, efficientnetb0"
        )

__all__ = [
    "LeNet",
    "AlexNet",
    "VGG16",
    "GoogLeNet",
    "ResNet18",
    "ResNet50",
    "EfficientNetB0",
    "create_model",
]