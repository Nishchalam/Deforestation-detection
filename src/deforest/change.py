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


def _near_boundary(mask: np.ndarray, px: int = 2) -> np.ndarray:
    m = mask.astype(bool)
    edge = (m & ~ndimage.binary_erosion(m)) | (~m & ~ndimage.binary_erosion(~m))
    return ndimage.binary_dilation(edge, iterations=px)


def categorize_fp(fp: np.ndarray, lossyear: np.ndarray, treecover: np.ndarray, threshold: float,
                  forest_y1: np.ndarray, y1: int, y2: int, prodes_loss: np.ndarray | None = None) -> np.ndarray:
    """One category per false-positive loss pixel, first matching rule wins (see notebook section 7)."""
    ly = lossyear.astype(int)
    rules = []
    if prodes_loss is not None:
        rules.append((f"PRODES {y1 + 1}-{str(y2)[2:]} deforestation", prodes_loss))
    rules += [(f"Hansen loss {y2 + 1}+", ly >= y2 - 2000 + 1),
              (f"Hansen loss <={y1}", (ly >= 1) & (ly <= y1 - 2000)),
              ("canopy2000 < threshold", treecover < threshold),
              ("forest edge (<=2 px)", _near_boundary(forest_y1)),
              ("stable forest interior", np.ones_like(fp, bool))]
    cat = np.full(fp.shape, "", object)
    for name, cond in rules:
        cat[fp & (cat == "") & cond] = name
    return cat


def categorize_fn(fn: np.ndarray, true_loss: np.ndarray, approach: str,
                  pred_forest_y1: np.ndarray | None = None, pred_forest_y2: np.ndarray | None = None,
                  small_px: int = 50) -> np.ndarray:
    """One category per false-negative loss pixel, plus ', edge' / ', interior' (<=2 px of the
    Hansen loss patch boundary)."""
    cat = np.full(fn.shape, "", object)
    if approach == "differencing":
        cat[fn & ~pred_forest_y1] = "model: non-forest in 2018"
        cat[fn & pred_forest_y1 & pred_forest_y2] = "model: still forest in 2022"
        cat[fn & pred_forest_y1 & ~pred_forest_y2] = "removed by MMU"
    else:
        lab, _ = ndimage.label(true_loss, structure=np.ones((3, 3)))
        small = np.bincount(lab.ravel())[lab] < small_px
        cat[fn & small] = "small Hansen patch (<0.5 ha)"
        cat[fn & ~small] = "large Hansen patch"
    cat[fn] = cat[fn] + np.where(_near_boundary(true_loss)[fn], ", edge", ", interior")
    return cat


def summarise_categories(cat: np.ndarray, mask: np.ndarray) -> dict:
    names, counts = np.unique(cat[mask], return_counts=True)
    return {str(n): {"pixels": int(c), "share": round(float(c / mask.sum()), 4)}
            for n, c in sorted(zip(names, counts), key=lambda x: -x[1])}
