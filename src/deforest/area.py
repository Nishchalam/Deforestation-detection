"""Run the deforestation pipeline for any bounding box: request validation, Earth Engine fetch
(same Sentinel-2 recipe as training) and an area run with optional Hansen GFC comparison.

No fallbacks: an Earth Engine or data failure raises with the reason.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from pyproj import Geod

from . import labels, metrics

MIN_YEAR = 2016  # first full Sentinel-2 L1C + cloud-probability coverage
S2_BANDS = 6
HANSEN_BANDS = ["treecover2000", "lossyear", "datamask"]
Progress = Callable[[str, float], None]


class RequestError(ValueError):
    """The request cannot be served; the message is meant for the user."""


# --------------------------------------------------------------------------- validation
def utm_crs(bbox) -> str:
    """UTM zone (WGS84) of the bbox centre."""
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def bbox_area_km2(bbox) -> float:
    """Geodesic area of a lon/lat box on the WGS84 ellipsoid."""
    w, s, e, n = bbox
    area_m2, _ = Geod(ellps="WGS84").polygon_area_perimeter([w, e, e, w], [s, s, n, n])
    return abs(area_m2) / 1e6


def last_year(today: date | None = None) -> int:
    """Latest year with a usable Jun-Sep season: the current year once its October has started."""
    today = today or date.today()
    return today.year if today.month >= 10 else today.year - 1


def validate_request(bbox, start: int, end: int, months=(6, 9), max_km2: float = 500.0,
                     today: date | None = None) -> None:
    """Raise RequestError (user-readable message) unless bbox, years and months are servable."""
    try:
        w, s, e, n = (float(v) for v in bbox)
        ok = len(bbox) == 4 and all(map(math.isfinite, (w, s, e, n)))
    except (TypeError, ValueError):
        ok = False
    if not ok or not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise RequestError("bounding box must be [west, south, east, north] in degrees, "
                           "with west < east and south < north")
    ly = last_year(today)
    if start < MIN_YEAR:
        raise RequestError(f"start year must be {MIN_YEAR} or later (Sentinel-2 coverage), got {start}")
    if start >= end:
        raise RequestError(f"start year {start} must be before end year {end}")
    if end > ly:
        raise RequestError(f"end year {end} is not available yet: the latest year is {ly}")
    m1, m2 = months
    if not 1 <= m1 <= m2 <= 12:
        raise RequestError(f"months must satisfy 1 <= first <= last <= 12 (no wrap over the new year), got {m1}-{m2}")
    km2 = bbox_area_km2([w, s, e, n])
    if km2 > max_km2:
        raise RequestError(f"area is {km2:.0f} km2, above the {max_km2:g} km2 limit")


def s2_dates(year: int, months: tuple[int, int]) -> tuple[str, str]:
    """[start, end) date strings for months m1..m2 of `year` (end exclusive)."""
    m1, m2 = months
    end = f"{year + 1}-01-01" if m2 == 12 else f"{year}-{m2 + 1:02d}-01"
    return f"{year}-{m1:02d}-01", end


# --------------------------------------------------------------------------- fetch
@dataclass
class Inputs:
    """Co-registered rasters on one UTM grid. before/after: (6,H,W) reflectance x 1e4;
    n_obs_*: clear observations per pixel; treecover/lossyear/datamask: Hansen GFC layers."""

    before: np.ndarray
    after: np.ndarray
    n_obs_before: np.ndarray
    n_obs_after: np.ndarray
    treecover: np.ndarray
    lossyear: np.ndarray
    datamask: np.ndarray
    transform: rasterio.Affine
    crs: str
    hansen_asset: str
    hansen_last_year: int


def hansen_year(asset: str) -> int:
    """Last loss year of a `...global_forest_change_YYYY_v1_N` asset (the asset year)."""
    m = re.search(r"global_forest_change_(\d{4})_v", asset)
    if not m:
        raise RuntimeError(f"cannot read the year from Hansen asset {asset!r}")
    return int(m.group(1))


def fetch_inputs(bbox, start: int, end: int, months, cfg, out_dir: Path, progress: Progress) -> Inputs:
    """Download both seasonal composites and the Hansen layers for `bbox` from Earth Engine into
    `out_dir` (s2_<y>.tif, hansen.tif) and load them. Raises on any Earth Engine failure."""
    from . import gee

    out_dir = Path(out_dir)
    s2 = cfg.sentinel2
    gee.initialize()
    grid = gee.make_grid(list(bbox), utm_crs(bbox), cfg.aoi.scale)
    region = gee.grid_geometry(grid)
    tile_px = cfg.download.tile_px
    arrays, profile = {}, None
    for y, frac in ((start, 0.1), (end, 0.45)):
        d0, d1 = s2_dates(y, tuple(months))
        if gee.s2_collection(s2.collection, region, d0, d1).size().getInfo() == 0:
            raise RequestError(f"no Sentinel-2 scenes over this area between {d0} and {d1}")
        img = gee.s2_composite(s2.collection, region, d0, d1, list(s2.bands), cloud_mask=s2.cloud_mask,
                               scl_mask=s2.get("scl_mask"), s2cloudless=s2.get("s2cloudless"))
        path = gee.download_image(img, grid, out_dir / f"s2_{y}.tif", list(s2.bands) + ["n_obs"], tile_px, nodata=0)
        with rasterio.open(path) as src:
            arrays[y], profile = src.read(), src.profile
        progress(f"Sentinel-2 {y} downloaded", frac)
    asset = gee.hansen_latest() if cfg.hansen.asset == "auto" else cfg.hansen.asset
    path = gee.download_image(gee.hansen_image(asset), grid, out_dir / "hansen.tif", HANSEN_BANDS, tile_px)
    with rasterio.open(path) as src:
        tc, ly, dm = src.read()
    progress("Hansen layers downloaded", 0.8)
    b, a = arrays[start], arrays[end]
    return Inputs(b[:S2_BANDS], a[:S2_BANDS], b[S2_BANDS], a[S2_BANDS], tc, ly, dm,
                  profile["transform"], str(profile["crs"]), asset, hansen_year(asset))


# --------------------------------------------------------------------------- run
def _num(x: float) -> float | None:
    return None if math.isnan(x) else float(x)


def run_area(pipe, inp: Inputs, start: int, end: int, threshold: float) -> dict:
    """Run `pipe` on the inputs. `threshold` is the Hansen canopy-cover % that defines forest
    (the training label rule). Hansen loss and agreement are given only when end <= hansen_last_year.
    Returns forest_before, forest_after, loss, valid, hansen_loss (or None) and stats."""
    for name, n_obs in (("before", inp.n_obs_before), ("after", inp.n_obs_after)):
        if not (n_obs > 0).any():
            raise RequestError(f"no clear Sentinel-2 observations in the {name} season ({start if name == 'before' else end})")
    valid = labels.valid_mask(inp.datamask, inp.n_obs_before, inp.n_obs_after)
    if not valid.any():
        raise RequestError("no valid land pixels with clear Sentinel-2 observations in both seasons")
    res = pipe.run(inp.before, inp.after, inp.transform, nodata_mask=~valid)
    stats = res["stats"]
    fb = stats["forest_before_ha"]
    stats["loss_pct_of_forest"] = round(100 * stats["loss_ha"] / fb, 2) if fb else 0.0
    stats["clear_obs_median"] = {"before": float(np.median(inp.n_obs_before[valid])),
                                 "after": float(np.median(inp.n_obs_after[valid]))}
    hansen_loss, stats["hansen"] = None, None
    if end <= inp.hansen_last_year:
        hansen_loss = labels.loss_mask(inp.treecover, inp.lossyear, start, end, threshold) & valid
        sc = metrics.scores(metrics.confusion(res["loss"], hansen_loss, valid))
        stats["hansen"] = {"asset": inp.hansen_asset,
                           "loss_ha": round(metrics.area_ha(hansen_loss, inp.transform), 2),
                           "agreement": {k: _num(sc[k]) for k in ("iou", "f1", "precision", "recall")}}
    return {"forest_before": res["forest_before"], "forest_after": res["forest_after"], "loss": res["loss"],
            "valid": valid, "hansen_loss": hansen_loss, "stats": stats}
