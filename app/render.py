"""Web-map overlays: UTM rasters -> north-up Web Mercator RGBA PNGs plus Leaflet bounds."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

from deforest import viz

WEB_MERCATOR = CRS.from_epsg(3857)

LAYER_COLORS = {
    "forest_before": (34, 139, 34),
    "forest_after": (34, 139, 34),
    "loss": (230, 30, 30),
    "hansen_loss": (255, 200, 0),
}


def _warp(arr: np.ndarray, transform, crs, resampling: Resampling):
    """Reproject (H,W) or (C,H,W) float32 to EPSG:3857. Returns (array, bounds [[s,w],[n,e]]).
    Pixels outside the source footprint stay 0."""
    bands = arr if arr.ndim == 3 else arr[None]
    _, h, w = bands.shape
    dst_tf, dw, dh = calculate_default_transform(crs, WEB_MERCATOR, w, h, *array_bounds(h, w, transform))
    out = np.zeros((bands.shape[0], dh, dw), np.float32)
    for i in range(bands.shape[0]):
        reproject(bands[i], out[i], src_transform=transform, src_crs=crs, dst_transform=dst_tf,
                  dst_crs=WEB_MERCATOR, resampling=resampling, src_nodata=None, dst_nodata=0)
    west, south, east, north = transform_bounds(WEB_MERCATOR, "EPSG:4326", *array_bounds(dh, dw, dst_tf))
    return (out if arr.ndim == 3 else out[0]), [[south, west], [north, east]]


def _png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def rgb_png(img6: np.ndarray, valid, transform, crs) -> tuple[bytes, list[list[float]]]:
    """(6,H,W) composite (B2,B3,B4,B8,B11,B12) -> true-colour RGBA PNG and bounds [[S,W],[N,E]].
    Percentile stretch uses valid pixels only; invalid and out-of-footprint pixels are transparent."""
    valid = np.asarray(valid, bool) & (img6 > 0).any(axis=0)
    rgb = np.nan_to_num(viz.rgb(img6 * valid[None], ("B4", "B3", "B2"), 2, 98))  # (H,W,3) in [0,1]
    stack = np.concatenate([rgb.transpose(2, 0, 1), valid[None].astype(np.float32)]).astype(np.float32)
    colour, bounds = _warp(stack[:3], transform, crs, Resampling.bilinear)
    alpha, _ = _warp(stack[3], transform, crs, Resampling.nearest)
    rgba = np.empty(alpha.shape + (4,), np.uint8)
    rgba[..., :3] = np.clip(colour.transpose(1, 2, 0) * 255 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(alpha > 0.5, 255, 0)
    return _png(rgba), bounds


def mask_png(mask, transform, crs, color: tuple[int, int, int], alpha: int = 170) -> tuple[bytes, list[list[float]]]:
    """Boolean mask -> flat-colour RGBA PNG (alpha where True, else 0) and bounds [[S,W],[N,E]]."""
    warped, bounds = _warp(np.asarray(mask, np.float32), transform, crs, Resampling.nearest)
    rgba = np.zeros(warped.shape + (4,), np.uint8)
    on = warped > 0.5
    rgba[on, :3] = color
    rgba[on, 3] = alpha
    return _png(rgba), bounds
