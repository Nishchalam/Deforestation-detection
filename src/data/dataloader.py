from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import EuroSATDataset
from .transforms import train_transform, test_transform


def create_dataloaders(
    data_root="data/raw/EuroSAT",
    processed_root="data/processed",
    batch_size=32,
    num_workers=4,
    pin_memory=True,
):
    """
    Creates train, validation and test dataloaders.

    Parameters
    ----------
    data_root : str
        Root directory containing EuroSAT images.

    processed_root : str
        Directory containing train.csv, validation.csv and test.csv.

    batch_size : int
        Batch size.

    num_workers : int
        Number of workers used by DataLoader.

    pin_memory : bool
        Pin memory for faster GPU transfer.

    Returns
    -------
    train_loader
    val_loader
    test_loader
    """

    data_root = Path(data_root)
    processed_root = Path(processed_root)

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