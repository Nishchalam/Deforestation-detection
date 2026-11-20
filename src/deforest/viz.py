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


def error_gallery(mask: np.ndarray, cat: np.ndarray, true_loss: np.ndarray, rgb1: np.ndarray, rgb2: np.ndarray,
                  years: tuple[int, int], title: str, n: int = 10, half: int = 40):
    """The n largest error blobs of `mask`: crops of both years, blob outline yellow, other Hansen loss blue.
    Returns (figure, info list); info rows are in the crop's (row, col) of the given arrays."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from .change import components

    blobs = components(mask)[:n]
    fig, ax = plt.subplots(n // 2, 4, figsize=(12, 3.1 * (n // 2)), layout="constrained")
    info = []
    for k, (size, sl, blob) in enumerate(blobs):
        cy, cx = (sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2
        ys, xs = slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half)
        full = np.zeros(mask.shape, bool)
        full[sl] = blob
        main = pd.Series(cat[sl][blob]).value_counts().index[0]
        for j, (r, yr) in enumerate(((rgb1, years[0]), (rgb2, years[1]))):
            a = ax[k // 2, 2 * (k % 2) + j]
            o = overlay(r[ys, xs], edges(full[ys, xs]), (1, 1, 0))
            o = overlay(o, edges(true_loss[ys, xs]) & ~edges(full[ys, xs]), (0, 0.6, 1))
            a.imshow(o)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"#{k + 1} {yr}" + (f" | {size} px | {main}" if j == 0 else ""), fontsize=8)
        info.append({"rank": k + 1, "pixels": size, "row": int(cy), "col": int(cx), "main_category": main})
    fig.suptitle(f"{title}: yellow = error blob, blue = Hansen loss {years[0] + 1}-{years[1]} outline")
    return fig, info
