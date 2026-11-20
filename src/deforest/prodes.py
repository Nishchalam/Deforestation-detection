"""PRODES (INPE) yearly deforestation polygons via the TerraBrasilis WFS, rasterised on a grid.
PRODES year Y covers roughly Aug(Y-1)-Jul(Y); it maps clear-cut of primary forest only.
https://terrabrasilis.dpi.inpe.br/en/home-page/"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests
from rasterio.features import rasterize
from rasterio.warp import transform_geom

WFS = "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/ows"
LAYER = "prodes-legal-amz:yearly_deforestation"


def fetch(bbox: list[float], out: str | Path) -> dict:
    """All yearly_deforestation polygons intersecting bbox (lon_min, lat_min, lon_max, lat_max)."""
    lon0, lat0, lon1, lat1 = bbox
    params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": LAYER,
              "outputFormat": "application/json",
              "bbox": f"{lat0},{lon0},{lat1},{lon1},urn:ogc:def:crs:EPSG::4326"}
    r = requests.get(WFS, params=params, timeout=300)
    r.raise_for_status()
    fc = r.json()
    if fc.get("numberMatched") not in (None, len(fc["features"])):
        raise RuntimeError(f"WFS returned {len(fc['features'])} of {fc['numberMatched']} features")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(fc))
    return fc


def rasterize_years(fc: dict, years: range, shape: tuple[int, int], transform, crs: str) -> np.ndarray:
    """Boolean mask of PRODES polygons with `year` in `years` on the target grid.
    Source CRS is SIRGAS 2000 (EPSG:4674), sub-metre from WGS84."""
    src_crs = fc.get("crs", {}).get("properties", {}).get("name", "EPSG:4674").split("::")[-1]
    src_crs = f"EPSG:{src_crs}" if src_crs.isdigit() else src_crs
    geoms = [transform_geom(src_crs, crs, f["geometry"]) for f in fc["features"]
             if f["properties"]["year"] in years]
    if not geoms:
        return np.zeros(shape, bool)
    return rasterize(geoms, out_shape=shape, transform=transform, fill=0, default_value=1,
                     dtype="uint8").astype(bool)
