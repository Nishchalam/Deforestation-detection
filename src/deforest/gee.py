"""Google Earth Engine access: Sentinel-2 composites, Hansen GFC, tiled GeoTIFF download.

Fails loudly: no interactive auth, no relaxed fallbacks. Docs:
- S2 L2A: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- Hansen GFC: https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2024_v1_12
- getDownloadURL limits: https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import ee
import numpy as np
import rasterio
import requests
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

HANSEN_PARENT = "projects/earthengine-public/assets/UMD/hansen"


def _env(name: str) -> str | None:
    """An env var with surrounding whitespace and one pair of matching quotes removed: values pasted
    into hosting UIs (e.g. Hugging Face Space variables) often carry them."""
    v = (os.environ.get(name) or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    return v or None


def _service_account() -> tuple[object, str | None] | None:
    """Service-account credentials from $EE_SERVICE_ACCOUNT_KEY (JSON text) or the file named by
    $GOOGLE_APPLICATION_CREDENTIALS. Returns (credentials, key's project_id), or None if neither is set."""
    key = _env("EE_SERVICE_ACCOUNT_KEY")
    if not key:
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not path:
            return None
        key = Path(path).read_text()
    info = json.loads(key)
    return ee.ServiceAccountCredentials(info["client_email"], key_data=key), info.get("project_id")


def initialize(project: str | None = None) -> str:
    """Initialise EE with a service account ($EE_SERVICE_ACCOUNT_KEY JSON text, else the key file
    $GOOGLE_APPLICATION_CREDENTIALS) if set, else the cached user credentials. Project: `project`,
    else $EE_PROJECT, else the key's project_id / the project stored with the cached credentials.
    Returns the project used. Raises if not authenticated."""
    sa = _service_account()
    credentials, key_project = sa if sa else (None, None)
    project = project or _env("EE_PROJECT") or key_project
    if not project and credentials is None:
        cred = Path(ee.oauth.get_credentials_path())
        if cred.exists():
            project = json.loads(cred.read_text()).get("project")
    if not project:
        raise RuntimeError("No GEE project: set EE_PROJECT or pass --project.")
    try:
        # credentials=None would mean "no credentials"; omit it to use the cached user credentials
        kwargs = {} if credentials is None else {"credentials": credentials}
        ee.Initialize(project=project, **kwargs)
    except Exception as e:  # noqa: BLE001 - re-raise with instructions
        hint = "Check the service-account key." if credentials is not None else "Run `earthengine authenticate`."
        # the cause is shown on the public /health page: first line only, emails redacted
        cause = re.sub(r"[\w.+-]+@[\w.-]+", "<email>", (str(e).strip().splitlines() or [""])[0])[:300]
        raise RuntimeError(f"Earth Engine init failed for project {project!r}: {type(e).__name__}: {cause} "
                           f"{hint}") from e
    return str(project)


# --------------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Grid:
    """A north-up raster grid in a projected CRS."""

    crs: str
    x0: float  # left edge (m)
    y0: float  # top edge (m)
    scale: float
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        return Affine(self.scale, 0, self.x0, 0, -self.scale, self.y0)

    def ee_transform(self, col_off: int = 0, row_off: int = 0) -> list[float]:
        return [self.scale, 0, self.x0 + col_off * self.scale, 0, -self.scale, self.y0 - row_off * self.scale]


def make_grid(bbox: list[float], crs: str, scale: float) -> Grid:
    """Project a lon/lat bbox to `crs` and snap it outward to multiples of `scale`,
    so the grid coincides with the native Sentinel-2 10 m pixel grid."""
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    x0 = math.floor(left / scale) * scale
    y0 = math.ceil(top / scale) * scale
    x1 = math.ceil(right / scale) * scale
    y1 = math.floor(bottom / scale) * scale
    return Grid(crs, x0, y0, scale, int(round((x1 - x0) / scale)), int(round((y0 - y1) / scale)))


def grid_geometry(grid: Grid) -> ee.Geometry:
    x1 = grid.x0 + grid.width * grid.scale
    y1 = grid.y0 - grid.height * grid.scale
    return ee.Geometry.Rectangle([grid.x0, y1, x1, grid.y0], proj=grid.crs, geodesic=False)


# --------------------------------------------------------------------------- Sentinel-2
def season_dates(year: int, season: list[str]) -> tuple[str, str]:
    return f"{year}-{season[0]}", f"{year}-{season[1]}"


def s2_collection(collection: str, region: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    return ee.ImageCollection(collection).filterBounds(region).filterDate(start, end)


def scene_summary(col: ee.ImageCollection) -> dict:
    """Number of images, distinct acquisition dates and MGRS tiles in a collection."""
    dates = col.aggregate_array("system:time_start").map(
        lambda t: ee.Date(t).format("YYYY-MM-dd")
    ).distinct().sort()
    tiles = col.aggregate_array("MGRS_TILE").distinct().sort()
    info = ee.Dictionary({"n_images": col.size(), "dates": dates, "mgrs_tiles": tiles}).getInfo()
    info["n_dates"] = len(info["dates"])
    return info


def _mask_scl(scl_mask: list[int]):
    def fn(img: ee.Image) -> ee.Image:
        scl = img.select("SCL")
        bad = ee.Image(0)
        for v in scl_mask:
            bad = bad.Or(scl.eq(v))
        return img.updateMask(bad.Not())

    return fn


def _join_s2cloudless(col: ee.ImageCollection, prob_collection: str, region: ee.Geometry,
                      start: str, end: str) -> ee.ImageCollection:
    probs = s2_collection(prob_collection, region, start, end)
    joined = ee.Join.saveFirst("s2cloudless").apply(
        primary=col, secondary=probs,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"))
    return ee.ImageCollection(joined)


def _mask_s2cloudless(p: dict):
    """Cloud = s2cloudless probability > threshold; shadow = dark NIR inside the cloud
    footprint projected along the solar azimuth; then buffer. Follows the GEE community
    tutorial (link in configs/ariquemes.yaml), minus its SCL water test (L1C has no SCL)."""
    def fn(img: ee.Image) -> ee.Image:
        prob = ee.Image(img.get("s2cloudless")).select("probability")
        cloud = prob.gt(p["cloud_prob_threshold"])
        if p.get("mask_cirrus_qa60", False):
            cloud = cloud.Or(img.select("QA60").bitwiseAnd(1 << 11).neq(0))
        dark = img.select("B8").lt(p["nir_dark_threshold"] * 1e4)
        azimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
        proj = (cloud.directionalDistanceTransform(azimuth, p["cloud_proj_dist_km"] * 10)
                .reproject(crs=img.select(0).projection(), scale=100)
                .select("distance").mask())
        shadow = proj.multiply(dark)
        bad = (cloud.Or(shadow).focalMin(2).focalMax(p["buffer_m"] * 2 / 20)
               .reproject(crs=img.select(0).projection(), scale=20))
        return img.updateMask(bad.Not())

    return fn


def s2_composite(collection: str, region: ee.Geometry, start: str, end: str,
                 bands: list[str], cloud_mask: str = "scl", scl_mask: list[int] | None = None,
                 s2cloudless: dict | None = None) -> ee.Image:
    """Per-pixel cloud-masked median of `bands` (uint16 reflectance x 1e4) plus `n_obs`,
    the number of clear observations per pixel. 20 m bands are resampled bilinearly
    to the 10 m output grid; 10 m bands sit on their native grid, so bilinear is exact.
    cloud_mask: 'scl' (L2A Scene Classification) or 's2cloudless' (L1C + cloud probability)."""
    col = s2_collection(collection, region, start, end)
    if cloud_mask == "scl":
        col = col.map(_mask_scl(list(scl_mask)))
    elif cloud_mask == "s2cloudless":
        col = _join_s2cloudless(col, s2cloudless["collection"], region, start, end)
        col = col.map(_mask_s2cloudless(dict(s2cloudless)))
    else:
        raise ValueError(f"unknown cloud_mask {cloud_mask!r}")
    col = col.map(lambda im: im.select(bands).resample("bilinear"))
    median = col.median().round().toUint16()
    n_obs = col.select(bands[0]).count().rename("n_obs").toUint16()
    return median.addBands(n_obs).unmask(0)


# --------------------------------------------------------------------------- Hansen
def hansen_latest() -> str:
    """Asset id of the newest Hansen GFC release in the public catalog."""
    assets = ee.data.listAssets({"parent": HANSEN_PARENT}).get("assets", [])
    ids = sorted(a["id"] for a in assets if "global_forest_change_" in a["id"])
    if not ids:
        raise RuntimeError(f"No Hansen GFC assets listed under {HANSEN_PARENT}")
    # ids look like UMD/hansen/global_forest_change_2024_v1_12; sort by (year, version)
    def key(i: str) -> tuple[int, int]:
        tail = i.split("global_forest_change_")[1]  # 2024_v1_12
        year, _, ver = tail.partition("_v1_")
        return int(year), int(ver or 0)

    return max(ids, key=key)


def hansen_image(asset: str) -> ee.Image:
    """treecover2000 (%), lossyear (0 = none, k = 2000+k), datamask (1 = land), uint8."""
    return ee.Image(asset).select(["treecover2000", "lossyear", "datamask"]).toUint8()


# --------------------------------------------------------------------------- download
def _fetch(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        r = requests.get(url, timeout=300)
        if r.status_code == 200:
            return r.content
        if attempt == retries - 1:
            raise RuntimeError(f"Download failed ({r.status_code}): {r.text[:500]}")
        time.sleep(5 * 2**attempt)
    raise AssertionError


def download_image(image: ee.Image, grid: Grid, out_path: str | Path, band_names: list[str],
                   tile_px: int = 1024, nodata: float | None = None) -> Path:
    """Download `image` on `grid` in tile_px x tile_px requests and mosaic into one GeoTIFF.
    Each tile's returned transform is checked against the expected grid."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.select(band_names)
    mosaic = None
    for row in range(0, grid.height, tile_px):
        for col in range(0, grid.width, tile_px):
            w, h = min(tile_px, grid.width - col), min(tile_px, grid.height - row)
            url = image.getDownloadURL({
                "crs": grid.crs,
                "crs_transform": grid.ee_transform(col, row),
                "dimensions": f"{w}x{h}",
                "format": "GEO_TIFF",
            })
            with rasterio.MemoryFile(io.BytesIO(_fetch(url))) as mf, mf.open() as src:
                expected = grid.transform * Affine.translation(col, row)
                if not src.transform.almost_equals(expected) or (src.width, src.height) != (w, h):
                    raise RuntimeError(f"Tile ({row},{col}) grid mismatch: {src.transform} vs {expected}")
                arr = src.read()
                if mosaic is None:
                    mosaic = np.zeros((src.count, grid.height, grid.width), dtype=arr.dtype)
                mosaic[:, row:row + h, col:col + w] = arr
            print(f"  tile r{row} c{col} {w}x{h} ok", flush=True)
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=mosaic.shape[0],
                   dtype=mosaic.dtype, crs=grid.crs, transform=grid.transform, compress="deflate",
                   tiled=True, blockxsize=256, blockysize=256, nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)
    return out_path
