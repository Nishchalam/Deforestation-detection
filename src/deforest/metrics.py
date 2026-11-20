"""Binary segmentation metrics from confusion counts, tile bootstrap CIs, and area."""
from __future__ import annotations

import numpy as np
from rasterio.transform import Affine


def confusion(pred: np.ndarray, true: np.ndarray, valid: np.ndarray | None = None) -> dict[str, int]:
    pred, true = pred.astype(bool), true.astype(bool)
    v = np.ones_like(true) if valid is None else valid.astype(bool)
    return {"tp": int((pred & true & v).sum()), "fp": int((pred & ~true & v).sum()),
            "fn": int((~pred & true & v).sum()), "tn": int((~pred & ~true & v).sum())}


def scores(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {"iou": div(tp, tp + fp + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "accuracy": div(tp + tn, tp + fp + fn + tn)}


def add(counts: list[dict[str, int]]) -> dict[str, int]:
    return {k: sum(c[k] for c in counts) for k in ("tp", "fp", "fn", "tn")}


def bootstrap_ci(tile_counts: list[dict[str, int]], n: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> dict[str, list[float]]:
    """Percentile CI of pooled scores, resampling whole tiles with replacement."""
    rng = np.random.default_rng(seed)
    arr = np.array([[c["tp"], c["fp"], c["fn"], c["tn"]] for c in tile_counts])
    samples = []
    for _ in range(n):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(0)
        samples.append(scores(dict(zip(("tp", "fp", "fn", "tn"), s))))
    out = {}
    for k in ("iou", "f1", "precision", "recall"):
        v = np.array([s[k] for s in samples])
        v = v[~np.isnan(v)]
        out[k] = [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))] if len(v) else [np.nan] * 2
    return out


def pixel_area_ha(transform: Affine) -> float:
    return abs(transform.a * transform.e) / 1e4


def area_ha(mask: np.ndarray, transform: Affine) -> float:
    return float(mask.sum()) * pixel_area_ha(transform)
