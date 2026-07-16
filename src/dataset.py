from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import train_transform, test_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class EuroSATDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(self.root_dir / row["image_path"]).convert("RGB")
        label = row["label"]
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": label,
            "class_name": row["class_name"],
            "image_path": row["image_path"]
        }

def create_dataloaders(
    data_root="data/raw/EuroSAT",
    processed_root="data/processed",
    batch_size=32,
    num_workers=0,
    pin_memory=True,
):
    data_root = PROJECT_ROOT / data_root
    processed_root = PROJECT_ROOT / processed_root
    
    assert data_root.exists(), f"Dataset directory not found: {data_root}"
    assert processed_root.exists(), f"Processed directory not found: {processed_root}"
    assert (processed_root / "train.csv").exists(), f"train.csv not found under {processed_root}"
    assert (processed_root / "validation.csv").exists(), f"validation.csv not found under {processed_root}"
    assert (processed_root / "test.csv").exists(), f"test.csv not found under {processed_root}"

    train_dataset = EuroSATDataset(
        csv_file=processed_root / "train.csv",
        root_dir=data_root,
        transform=train_transform,
    )

    val_dataset = EuroSATDataset(
        csv_file=processed_root / "validation.csv",
        root_dir=data_root,
        transform=test_transform,
    )

    test_dataset = EuroSATDataset(
        csv_file=processed_root / "test.csv",
        root_dir=data_root,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
