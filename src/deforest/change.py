"""Forest-loss maps from two forest masks, minimum mapping unit, and error maps."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

TN, TP, FP, FN = 0, 1, 2, 3


def loss_map(forest_before: np.ndarray, forest_after: np.ndarray) -> np.ndarray:
    return forest_before.astype(bool) & ~forest_after.astype(bool)


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Drop 8-connected blobs smaller than min_px pixels (minimum mapping unit)."""
    if min_px <= 1:
        return mask.astype(bool)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def error_map(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred, true = pred.astype(bool), true.astype(bool)
    out = np.full(pred.shape, TN, np.uint8)
    out[pred & true] = TP
    out[pred & ~true] = FP
    out[~pred & true] = FN
    return out


def components(mask: np.ndarray) -> list[tuple[int, tuple[slice, slice], np.ndarray]]:
    """8-connected blobs as (size, bbox slices, blob mask within bbox), largest first."""
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        blob = lab[sl] == i
        out.append((int(blob.sum()), sl, blob))
    return sorted(out, key=lambda x: -x[0])
