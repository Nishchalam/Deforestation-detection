"""Hansen-derived labels. lossyear: 0 = no loss, k = loss detected in year 2000 + k.
Docs: https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/download.html"""
from __future__ import annotations

import numpy as np


def loss_code(year: int) -> int:
    return year - 2000


def forest_mask(treecover2000: np.ndarray, lossyear: np.ndarray, year: int, threshold: float) -> np.ndarray:
    """Forest in `year`: canopy >= threshold % in 2000 and no Hansen loss in 2001..year."""
    lost = (lossyear >= 1) & (lossyear <= loss_code(year))
    return (treecover2000 >= threshold) & ~lost


def loss_mask(treecover2000: np.ndarray, lossyear: np.ndarray, y1: int, y2: int, threshold: float) -> np.ndarray:
    """Hansen forest loss after y1 up to y2 inclusive, i.e. forest(y1) & ~forest(y2)."""
    return forest_mask(treecover2000, lossyear, y1, threshold) & ~forest_mask(treecover2000, lossyear, y2, threshold)


def valid_mask(datamask: np.ndarray, *n_obs: np.ndarray) -> np.ndarray:
    """Land pixels (Hansen datamask == 1) with at least one clear S2 observation in every composite."""
    ok = datamask == 1
    for n in n_obs:
        ok &= n > 0
    return ok
