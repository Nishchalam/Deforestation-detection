"""Inference on arbitrary-size GeoTIFFs: sliding-window U-Net, forest masks and loss map + stats.

The network call is a plain function `infer(batch: float32 (N,C,S,S)) -> logits (N,S,S)`, so the
same code runs a PyTorch model (scripts) or an ONNX Runtime session (the API).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio

from . import change, metrics

Infer = Callable[[np.ndarray], np.ndarray]


def _taper(size: int, overlap: int) -> np.ndarray:
    """2-D weight that fades linearly over `overlap` px at each edge, so overlapping windows blend."""
    w = np.ones(size, np.float32)
    if overlap > 0:
        ramp = np.linspace(1.0 / overlap, 1.0, overlap, dtype=np.float32)
        w[:overlap], w[-overlap:] = ramp, ramp[::-1]
    return np.outer(w, w)


def sliding_logits(infer: Infer, x: np.ndarray, tile: int = 256, overlap: int = 64, batch: int = 8) -> np.ndarray:
    """(C,H,W) normalised input -> (H,W) logits. Reflect-pads to cover the image with windows of
    `tile` px at stride tile - overlap and averages the overlaps with a linear taper."""
    c, h, w = x.shape
    stride = tile - overlap
    ph = max(tile, int(np.ceil((h - tile) / stride)) * stride + tile) if h > tile else tile
    pw = max(tile, int(np.ceil((w - tile) / stride)) * stride + tile) if w > tile else tile
    mode = "reflect" if (ph - h) < h and (pw - w) < w else "edge"
    xp = np.pad(x, ((0, 0), (0, ph - h), (0, pw - w)), mode=mode)
    acc = np.zeros((ph, pw), np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    weight = _taper(tile, overlap)
    origins = [(r, q) for r in range(0, ph - tile + 1, stride) for q in range(0, pw - tile + 1, stride)]
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        logits = infer(np.stack([xp[:, r:r + tile, q:q + tile] for r, q in chunk]).astype(np.float32))
        for (r, q), l in zip(chunk, logits):
            acc[r:r + tile, q:q + tile] += l * weight
            wsum[r:r + tile, q:q + tile] += weight
    return (acc / wsum)[:h, :w]


@dataclass
class Pipeline:
    infer_forest: Infer
    infer_change: Infer
    mean: np.ndarray
    std: np.ndarray
    change_threshold: float
    change_mmu_px: int
    tile: int = 256
    overlap: int = 64

    def normalize(self, img: np.ndarray) -> np.ndarray:
        return ((img.astype(np.float32) - self.mean[:, None, None]) / self.std[:, None, None]).astype(np.float32)

    def forest(self, img: np.ndarray) -> np.ndarray:
        return sliding_logits(self.infer_forest, self.normalize(img), self.tile, self.overlap) > 0

    def loss_probability(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        x = np.concatenate([self.normalize(before), self.normalize(after)])
        return 1 / (1 + np.exp(-sliding_logits(self.infer_change, x, self.tile, self.overlap)))

    def run(self, before: np.ndarray, after: np.ndarray, transform, nodata_mask: np.ndarray | None = None) -> dict:
        """before/after: (6,H,W) reflectance x 1e4 on the same grid. Returns masks and stats."""
        if before.shape != after.shape or before.shape[0] != len(self.mean):
            raise ValueError(f"expected two ({len(self.mean)},H,W) arrays on one grid, got {before.shape}, {after.shape}")
        valid = np.ones(before.shape[1:], bool) if nodata_mask is None else ~nodata_mask
        f1, f2 = self.forest(before) & valid, self.forest(after) & valid
        loss = change.remove_small((self.loss_probability(before, after) >= self.change_threshold) & valid,
                                   self.change_mmu_px)
        px = metrics.pixel_area_ha(transform)
        stats = {"pixel_area_ha": px, "valid_ha": round(float(valid.sum()) * px, 2),
                 "forest_before_ha": round(metrics.area_ha(f1, transform), 2),
                 "forest_after_ha": round(metrics.area_ha(f2, transform), 2),
                 "loss_ha": round(metrics.area_ha(loss, transform), 2),
                 "loss_pixels": int(loss.sum()),
                 "change_threshold": self.change_threshold, "change_mmu_px": self.change_mmu_px}
        return {"forest_before": f1, "forest_after": f2, "loss": loss, "stats": stats}


def read_composite(path: str | Path) -> tuple[np.ndarray, dict]:
    """First 6 bands of a GeoTIFF + its profile. Zero in all bands = no data."""
    with rasterio.open(path) as src:
        if src.count < 6:
            raise ValueError(f"{path}: need 6 bands (B2,B3,B4,B8,B11,B12), found {src.count}")
        return src.read(list(range(1, 7))), src.profile


def write_mask(path: str | Path, mask: np.ndarray, profile: dict, description: str) -> Path:
    prof = {k: profile[k] for k in ("crs", "transform", "width", "height")}
    prof.update(driver="GTiff", count=1, dtype="uint8", compress="deflate", nodata=None)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(mask.astype(np.uint8), 1)
        dst.set_band_description(1, description)
    return Path(path)


def torch_infer(model, device) -> Infer:
    import torch

    def fn(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(batch).to(device)).float().cpu().numpy()[:, 0]
    return fn


def load_torch_pipeline(model_dir: str | Path, device=None) -> Pipeline:
    import torch

    from .experiment import load_unet

    md = Path(model_dir)
    spec = json.loads((md / "deploy.json").read_text())
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = {"encoder": spec["encoder"]}
    n = len(spec["bands"])
    fm = load_unet(md / spec["forest_model"], U, n, device)
    cm = load_unet(md / spec["change_model"], U, 2 * n, device)
    z = np.load(md / spec["norm_stats"])
    return Pipeline(torch_infer(fm, device), torch_infer(cm, device), z["mean"], z["std"],
                    spec["change_threshold"], spec["change_mmu_px"], spec["tile_px"])
