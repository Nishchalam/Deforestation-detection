import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.training.trainer import Trainer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        # x is (B, C, H, W). Flatten to (B, C*H*W)
        x = x.view(x.size(0), -1)
        # Dummy linear layer assuming input is 10 features total for the sake of test
        return self.fc(x)

@pytest.fixture
def dummy_loaders():
    # Create fake data: B=4, C=1, H=1, W=10
    X = torch.randn(8, 1, 1, 10)
    y = torch.randint(0, 2, (8,))
    
    # Needs to match dict format expected by trainer
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
    
    # Assuming fit returns something or at least runs without error
    trainer.fit()
    assert True
