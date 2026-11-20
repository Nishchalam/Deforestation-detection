"""Load composites and labels; torch Dataset over (year, tile) pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .tiles import Tile


def load_composite(raw: str | Path, year: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> (6,H,W) uint16 reflectance x1e4, (H,W) n_obs, rasterio profile."""
    with rasterio.open(Path(raw) / f"s2_{year}.tif") as src:
        arr = src.read()
        return arr[:6], arr[6], src.profile


def load_labels(processed: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with rasterio.open(Path(processed) / "labels.tif") as src:
        return {d: src.read(i + 1).astype(bool) for i, d in enumerate(src.descriptions)}, src.profile


def band_stats(images: list[np.ndarray], tiles: list[Tile]) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std over the given tiles of all images (train tiles only)."""
    px = np.concatenate([img[(slice(None),) + t.window].reshape(img.shape[0], -1)
                         for img in images for t in tiles], axis=1).astype(np.float64)
    return px.mean(1).astype(np.float32), px.std(1).astype(np.float32)


def normalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((img.astype(np.float32) - mean[:, None, None]) / std[:, None, None]).astype(np.float32)


class TileDataset(Dataset):
    """Items are (year, tile). Returns dict(image (6,S,S) float, mask (1,S,S) float,
    valid (1,S,S) float). `augment` applies random flips and 90-degree rotations."""

    def __init__(self, images: dict[int, np.ndarray], masks: dict[int, np.ndarray], valid: np.ndarray,
                 tiles: list[Tile], augment: bool = False, repeats: int = 1):
        self.images, self.masks, self.valid = images, masks, valid
        self.items = [(y, t) for _ in range(repeats) for y in sorted(images) for t in tiles]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        year, t = self.items[i]
        x = self.images[year][(slice(None),) + t.window]
        y = self.masks[year][t.window][None].astype(np.float32)
        v = self.valid[t.window][None].astype(np.float32)
        if self.augment:
            k = np.random.randint(4)
            x, y, v = (np.rot90(a, k, axes=(1, 2)) for a in (x, y, v))
            if np.random.rand() < 0.5:
                x, y, v = (a[:, :, ::-1] for a in (x, y, v))
        return {"image": torch.from_numpy(np.ascontiguousarray(x)),
                "mask": torch.from_numpy(np.ascontiguousarray(y)),
                "valid": torch.from_numpy(np.ascontiguousarray(v))}
