"""
logger.py

ExperimentLogger for managing TensorBoard, JSON metadata, and standard logging.
"""

import logging
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    """
    Central logger for tracking experiments, metrics, images, and model graphs.
    """
    def __init__(self, exp_dir: Path):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.exp_dir / "tensorboard"
        self.tensorboard_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # Setup standard Python logger
        self.logger = logging.getLogger(f"ExperimentLogger_{exp_dir.name}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to prevent duplicate logging
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(self.exp_dir / "training.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
            
        self.metrics_history: Dict[str, list] = {}

    def log_info(self, message: str):
        """Logs an info message to file and console."""
        self.logger.info(message)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Logs metrics to TensorBoard and appends to history."""
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
            
        for k, v in metrics.items():
            if k not in self.metrics_history:
                self.metrics_history[k] = []
            self.metrics_history[k].append({"step": step, "value": float(v)})

    def log_hyperparams(self, hparam_dict: Dict[str, Any], metric_dict: Optional[Dict[str, float]] = None):
        """Logs hyperparameters to TensorBoard and JSON."""
        with open(self.exp_dir / "hyperparams.json", "w") as f:
            json.dump(hparam_dict, f, indent=4)
            
        if metric_dict:
            flat_hparams = {k: str(v) for k, v in hparam_dict.items()}
            self.writer.add_hparams(flat_hparams, metric_dict)

    def log_graph(self, model, input_to_model):
        """Logs model graph to TensorBoard."""
        self.writer.add_graph(model, input_to_model)

    def log_image(self, tag: str, image_tensor, step: int):
        """Logs an image tensor to TensorBoard."""
        self.writer.add_image(tag, image_tensor, step)

    def log_histogram(self, tag: str, values, step: int):
        """Logs a histogram to TensorBoard."""
        self.writer.add_histogram(tag, values, step)

    def close(self):
        """Closes the TensorBoard writer and writes metrics history to JSON/CSV."""
        self.writer.close()
        
        if self.metrics_history:
            with open(self.exp_dir / "metrics_history.json", "w") as f:
                json.dump(self.metrics_history, f, indent=4)
                
            csv_file = self.exp_dir / "metrics_history.csv"
            keys = sorted(self.metrics_history.keys())
            try:
                steps_data = {}
                for k, v_list in self.metrics_history.items():
                    for item in v_list:
                        s = item["step"]
                        if s not in steps_data:
                            steps_data[s] = {}
                        steps_data[s][k] = item["value"]
                        
                with open(csv_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["step"] + keys)
                    writer.writeheader()
                    for s in sorted(steps_data.keys()):
                        row = {"step": s}
                        row.update(steps_data[s])
                        writer.writerow(row)
            except Exception as e:
                self.log_info(f"Failed to save metrics CSV: {str(e)}")

        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
