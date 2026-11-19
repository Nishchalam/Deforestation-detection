"""Plot helpers for Sentinel-2 composites and masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


def rgb(img: np.ndarray, bands=("B4", "B3", "B2"), lo: float = 2, hi: float = 98,
        ref: np.ndarray | None = None) -> np.ndarray:
    """(C,H,W) reflectance -> (H,W,3) float in [0,1] with a percentile stretch.
    Pass `ref` to reuse another image's stretch (same colours across years)."""
    idx = [BANDS.index(b) for b in bands]
    x = img[idx].astype(np.float32)
    r = (ref if ref is not None else img)[idx].astype(np.float32)
    out = np.empty_like(x)
    for i in range(3):
        v = r[i][r[i] > 0]
        a, b = np.percentile(v, [lo, hi])
        out[i] = np.clip((x[i] - a) / (b - a), 0, 1)
    return out.transpose(1, 2, 0)


def edges(mask: np.ndarray) -> np.ndarray:
    """1-pixel inner boundary of a boolean mask."""
    return mask & ~ndimage.binary_erosion(mask)


def overlay(base: np.ndarray, mask: np.ndarray, color=(1.0, 0.0, 0.0)) -> np.ndarray:
    out = base.copy()
    out[mask] = color
    return out


def densest_window(mask: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left (row, col) of the size x size window (stride size/4) with most True pixels."""
    step = max(1, size // 4)
    best, pos = -1, (0, 0)
    for r in range(0, mask.shape[0] - size + 1, step):
        for c in range(0, mask.shape[1] - size + 1, step):
            s = int(mask[r:r + size, c:c + size].sum())
            if s > best:
                best, pos = s, (r, c)
    return pos
