import pytest
from unittest.mock import patch
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.experiments.runner import ExperimentRunner

class MockDictDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.X = torch.randn(8, 3, 224, 224)
        self.y = torch.randint(0, 10, (8,))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {"image": self.X[idx], "label": self.y[idx]}

@pytest.fixture
def mock_loaders():
    train_loader = DataLoader(MockDictDataset(), batch_size=4)
    val_loader = DataLoader(MockDictDataset(), batch_size=4)
    test_loader = DataLoader(MockDictDataset(), batch_size=4)
    return train_loader, val_loader, test_loader

def test_experiment_runner(tmp_path, mock_loaders):
    train_loader, val_loader, test_loader = mock_loaders
    
    config_content = """
experiment:
  name: "lenet"
  seed: 42
  device: "cpu"

model:
  name: "lenet"
  in_channels: 3
  num_classes: 10

dataset:
  name: "EuroSAT"
  batch_size: 4
  num_workers: 0

training:
  epochs: 1
  loss: "cross_entropy"
  optimizer:
    name: "adam"
    lr: 0.001
    weight_decay: 0.0

callbacks:
  early_stopping:
    enabled: true
  model_checkpoint:
    enabled: true
  tensorboard:
    enabled: true
  csv_logger:
    enabled: true
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    
    runner = ExperimentRunner(config_file)
    exp_dir = tmp_path / "experiments"
    
    with patch("src.experiments.runner.create_dataloaders", return_value=(train_loader, val_loader, test_loader)), \
         patch("src.experiments.runner.EXPERIMENTS_DIR", exp_dir):
         
        exp = runner.run()
        
        assert exp is not None
        assert exp_dir.exists()
        
        subdirs = list(exp_dir.iterdir())
        assert len(subdirs) == 1
        run_dir = subdirs[0]
        assert run_dir.name == "LeNet_001"
        
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "metrics_history.json").exists()
        assert (run_dir / "metrics_history.csv").exists()
