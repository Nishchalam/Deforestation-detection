"""
utils.py

Common utilities for training, checkpointing, and experiment management.
"""

import os
import random
import numpy as np
import torch
import time
from datetime import datetime
from pathlib import Path
import yaml
import csv
from typing import Dict, Any, Optional

from src.utils.paths import EXPERIMENTS_DIR

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds: float) -> str:
    """
    Formats a time in seconds to a string (e.g., '1h 2m 3s').
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    elif m > 0:
        return f"{int(m)}m {int(s)}s"
    else:
        return f"{s:.2f}s"

def create_experiment_folder(experiment_name: str, config: Dict[str, Any]) -> Path:
    """
    Creates a timestamped folder for the current experiment and saves the config.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment (e.g., 'LeNet').
    config : dict
        Configuration dictionary to save in the experiment folder.
        
    Returns
    -------
    Path
        Path to the newly created experiment folder.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{experiment_name}"
    exp_dir = EXPERIMENTS_DIR / folder_name
    
    # Create required subdirectories
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "tensorboard").mkdir(exist_ok=True)
    
    # Save the config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    return exp_dir

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    epoch: int, 
    path: Path,
    scheduler: Optional[Any] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
):
    """
    Saves a model checkpoint including optimizer and scheduler states.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extra_metadata:
        checkpoint.update(extra_metadata)
        
    torch.save(checkpoint, path)

def load_checkpoint(
    path: Path, 
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Loads a model checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    return checkpoint

def save_training_history(history: Dict[str, list], path: Path):
    """
    Saves a dictionary of metrics history (lists of equal length) to a CSV file.
    """
    if not history:
        return
        
    keys = list(history.keys())
    length = len(history[keys[0]])
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        
        for i in range(length):
            row = [history[k][i] for k in keys]
            writer.writerow(row)