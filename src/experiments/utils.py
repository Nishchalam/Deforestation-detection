"""
utils.py

Helper functions for managing experiment configs and finding directories.
"""

import yaml
from pathlib import Path

def load_yaml_config(config_path: Path) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_next_experiment_dir(model_name: str, base_experiments_dir: Path) -> Path:
    """
    Computes the next available sequential directory for the given model.
    E.g. lenet -> LeNet_001, LeNet_002, etc.
    """
    base_experiments_dir = Path(base_experiments_dir)
    base_experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # Capitalize the model name part (e.g. LeNet) for standard folders
    clean_model_name = model_name
    if clean_model_name.lower() == "lenet":
        clean_model_name = "LeNet"
    elif clean_model_name.lower() == "alexnet":
        clean_model_name = "AlexNet"
    elif clean_model_name.lower() == "vgg16":
        clean_model_name = "VGG16"
    elif clean_model_name.lower() == "googlenet":
        clean_model_name = "GoogLeNet"
    elif clean_model_name.lower() == "resnet18":
        clean_model_name = "ResNet18"
    elif clean_model_name.lower() == "resnet50":
        clean_model_name = "ResNet50"
    elif clean_model_name.lower() in ["efficientnet-b0", "efficientnet_b0"]:
        clean_model_name = "EfficientNetB0"
    else:
        clean_model_name = clean_model_name.capitalize()

    index = 1
    while True:
        exp_name = f"{clean_model_name}_{index:03d}"
        exp_dir = base_experiments_dir / exp_name
        if not exp_dir.exists():
            return exp_dir
        index += 1
