"""
experiment.py

Encapsulates all metadata and metrics of a single experiment.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch.nn as nn

class Experiment:
    """
    Data model representing a single experiment instance.
    """
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        model: Optional[nn.Module] = None,
        checkpoint_dir: Optional[Path] = None,
        tensorboard_dir: Optional[Path] = None
    ):
        self.name = name
        self.config = config
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        
        self.history: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Starts the timer for this experiment."""
        self.start_time = time.time()

    def end(self):
        """Stops the timer for this experiment."""
        self.end_time = time.time()

    def get_duration(self) -> Optional[float]:
        """Returns the total elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
