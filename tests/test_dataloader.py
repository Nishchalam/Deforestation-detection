import pytest
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import create_dataloaders

PROCESSED_DATA_DIR = Path("data/processed")

@pytest.mark.skipif(not (PROCESSED_DATA_DIR / "train.csv").exists(), reason="Dataset not prepared yet")
def test_create_dataloaders():
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=2)
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    
    # Check batch shapes
    train_batch = next(iter(train_loader))
    assert "image" in train_batch
    assert "label" in train_batch
    assert train_batch["image"].shape == (2, 3, 224, 224)
    assert train_batch["label"].shape == (2,)
