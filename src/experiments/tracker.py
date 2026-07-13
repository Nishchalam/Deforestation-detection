"""
tracker.py

Stores and manages experiment metadata.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

def get_git_commit_hash() -> str:
    """Retrieves the current git commit hash if running in a git repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "not_a_git_repository"

class ExperimentTracker:
    """
    Maintains and serializes experiment run parameters and final statistics.
    """
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.metadata: Dict[str, Any] = {}

    def track(
        self,
        name: str,
        training_time: Optional[float],
        best_accuracy: float,
        learning_rate: float,
        optimizer: str,
        dataset: str,
        checkpoint_path: str,
        tensorboard_path: str
    ):
        """Prepares metadata structure for serialization."""
        self.metadata = {
            "experiment_name": name,
            "training_time_seconds": training_time,
            "best_accuracy": best_accuracy,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "dataset": dataset,
            "checkpoint_path": str(checkpoint_path),
            "tensorboard_path": str(tensorboard_path),
            "git_commit_hash": get_git_commit_hash()
        }

    def save(self):
        """Saves tracked metadata to metrics.json inside the experiment directory."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metadata, f, indent=4)
