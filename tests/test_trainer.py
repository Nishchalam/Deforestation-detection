import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training import Trainer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

@pytest.fixture
def dummy_loaders():
    X = torch.randn(8, 1, 1, 10)
    y = torch.randint(0, 2, (8,))
    
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            return {"image": self.X[idx], "label": self.y[idx]}

    dataset = DictDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)
    return train_loader, val_loader

def test_trainer_initialization(dummy_loaders):
    train_loader, val_loader = dummy_loaders
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=1,
        device=torch.device("cpu")
    )
    
    assert trainer.epochs == 1
    assert trainer.device.type == "cpu"
    
def test_trainer_fit(dummy_loaders):
    train_loader, val_loader = dummy_loaders
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=2,
        device=torch.device("cpu")
    )
    
    trainer.fit()
    assert len(trainer.history["epoch"]) == 2

def test_early_stopping(dummy_loaders):
    train_loader, val_loader = dummy_loaders
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=10,
        early_stopping_patience=2,
        device=torch.device("cpu"),
        model_name="test_early_stopping"
    )
    
    # Mock validate to return a constant high loss so validation loss does not improve
    trainer.validate = lambda: (10.0, 0.5, 0.1)
    
    history = trainer.fit()
    # Patience is 2.
    # Epoch 1: val_loss=10.0 (improved from inf). best_val_loss=10.0, counter=0
    # Epoch 2: val_loss=10.0. counter=1
    # Epoch 3: val_loss=10.0. counter=2 (triggers early stopping)
    # Total epochs run should be exactly 3.
    assert len(history["epoch"]) == 3

def test_resume_training(dummy_loaders, tmp_path):
    train_loader, val_loader = dummy_loaders
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=2,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        model_name="test_resume"
    )
    
    trainer.fit()
    
    # Verify last checkpoint was saved
    last_checkpoint_path = tmp_path / "test_resume" / "last_model.pth"
    assert last_checkpoint_path.exists()
    
    # Create a new trainer to resume
    model2 = SimpleModel()
    optimizer2 = torch.optim.Adam(model2.parameters())
    trainer2 = Trainer(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer2,
        epochs=4,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        model_name="test_resume"
    )
    
    trainer2.load_checkpoint(str(last_checkpoint_path))
    assert trainer2.start_epoch == 2
    assert len(trainer2.history["epoch"]) == 2
    
    # Fit for remaining epochs
    trainer2.fit()
    assert len(trainer2.history["epoch"]) == 4
