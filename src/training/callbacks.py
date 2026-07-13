"""
callbacks.py

Implementation of standard training callbacks (EarlyStopping, ModelCheckpoint, etc.)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

from src.training.utils import save_checkpoint, save_training_history


class Callback:
    """Base class for all callbacks."""
    def on_train_begin(self, logs: Dict[str, Any] = None): pass
    def on_train_end(self, logs: Dict[str, Any] = None): pass
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None): pass
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None): pass
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None): pass
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None): pass


class EarlyStopping(Callback):
    """
    Stops training if a monitored metric has stopped improving.
    """
    def __init__(self, monitor: str = "val_loss", min_delta: float = 0.0, patience: int = 5, mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if mode == "min" else -np.inf
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.mode == "min":
            improvement = (self.best - current) > self.min_delta
        else:
            improvement = (current - self.best) > self.min_delta

        if improvement:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True


class ModelCheckpoint(Callback):
    """
    Saves the model after every epoch if the monitored metric improves.
    """
    def __init__(
        self, 
        filepath: Path, 
        monitor: str = "val_loss", 
        mode: str = "min", 
        save_best_only: bool = True
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = np.inf if mode == "min" else -np.inf

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        current = logs.get(self.monitor)
        model = logs.get("model")
        optimizer = logs.get("optimizer")
        scheduler = logs.get("scheduler")
        
        if current is None or model is None:
            return

        save = False
        if self.mode == "min":
            if current < self.best:
                self.best = current
                save = True
        else:
            if current > self.best:
                self.best = current
                save = True

        if save or not self.save_best_only:
            # Save best model
            best_path = self.filepath / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, best_path, scheduler)
            
        # Always save last model
        last_path = self.filepath / "last_model.pth"
        save_checkpoint(model, optimizer, epoch, last_path, scheduler)


class LearningRateMonitor(Callback):
    """
    Logs the learning rate during training.
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        optimizer = logs.get("optimizer")
        if optimizer:
            lr = optimizer.param_groups[0]['lr']
            self.logger.log_metrics({"learning_rate": lr}, epoch)


class TensorBoardCallback(Callback):
    """
    Logs standard metrics to TensorBoard.
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float)) and k not in ["epoch"]}
        self.logger.log_metrics(metrics, epoch)


class CSVLogger(Callback):
    """
    Saves training history to a CSV file.
    """
    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath
        self.history = {}

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if not logs:
            return
        
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                if k not in self.history:
                    self.history[k] = []
                self.history[k].append(v)
                
        save_training_history(self.history, self.filepath)