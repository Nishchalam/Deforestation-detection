from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class EuroSATDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.data = pd.read_csv(csv_file)

        self.root_dir = Path(root_dir)

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image = Image.open(
            self.root_dir / row["image_path"]
        ).convert("RGB")

        label = row["label"]

        if self.transform:

            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "class_name": row["class_name"],
            "image_path": row["image_path"]
        }