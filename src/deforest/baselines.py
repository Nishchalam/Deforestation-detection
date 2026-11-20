"""Non-deep baselines: NDVI threshold and per-pixel random forest."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import metrics

B4, B8 = 2, 3  # indices in [B2, B3, B4, B8, B11, B12]


def ndvi(img: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - red) / (NIR + red), https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index"""
    nir, red = img[B8].astype(np.float32), img[B4].astype(np.float32)
    return (nir - red) / np.maximum(nir + red, 1e-6)


def best_ndvi_threshold(values: np.ndarray, labels: np.ndarray, grid=np.arange(0.0, 1.0001, 0.01)):
    """Threshold maximising forest IoU on the given (validation) pixels. -> (thr, table)."""
    table = []
    for t in grid:
        table.append((float(t), metrics.scores(metrics.confusion(values >= t, labels))["iou"]))
    return max(table, key=lambda r: r[1])[0], table


def sample_pixels(images: list[np.ndarray], masks: list[np.ndarray], windows: list, n: int, seed: int):
    """Uniformly sample n pixels (features (n,6), labels (n,)) from the given windows of all images."""
    X = np.concatenate([img[(slice(None),) + w].reshape(img.shape[0], -1) for img in images for w in windows], 1).T
    y = np.concatenate([m[w].ravel() for m in masks for w in windows])
    idx = np.random.default_rng(seed).choice(len(y), size=min(n, len(y)), replace=False)
    return X[idx].astype(np.float32), y[idx]


def fit_random_forest(X: np.ndarray, y: np.ndarray, seed: int, **kw) -> RandomForestClassifier:
    rf = RandomForestClassifier(random_state=seed, **kw)
    return rf.fit(X, y)


def predict_random_forest(rf: RandomForestClassifier, img: np.ndarray) -> np.ndarray:
    c, h, w = img.shape
    return rf.predict(img.reshape(c, -1).T.astype(np.float32)).reshape(h, w).astype(bool)
