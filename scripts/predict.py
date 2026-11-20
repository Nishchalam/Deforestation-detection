"""Forest masks for two dates, a forest-loss map and area stats for any area.

From GeoTIFFs you already have (6 bands B2,B3,B4,B8,B11,B12, reflectance x 1e4, same grid):
    python scripts/predict.py --before s2_2018.tif --after s2_2022.tif --out outputs/predict/demo

Or download the composites from Earth Engine for a bbox and two years (same recipe as training):
    python scripts/predict.py --bbox -63.0 -10.5 -62.8 -10.3 --years 2018 2022 --out outputs/predict/ariquemes

Writes forest_before.tif, forest_after.tif, loss.tif and stats.json to --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deforest.predict import load_torch_pipeline, read_composite, write_mask


def download(bbox, years, data_config, out: Path) -> tuple[Path, Path]:
    from deforest import gee
    from deforest.config import load_config

    cfg = load_config(data_config)
    s2 = cfg.sentinel2
    gee.initialize()
    grid = gee.make_grid(bbox, cfg.aoi.crs if bbox == list(cfg.aoi.bbox) else utm_crs(bbox), cfg.aoi.scale)
    region = gee.grid_geometry(grid)
    paths = []
    for y in years:
        start, end = gee.season_dates(y, s2.season)
        img = gee.s2_composite(s2.collection, region, start, end, list(s2.bands), cloud_mask=s2.cloud_mask,
                               scl_mask=s2.get("scl_mask"), s2cloudless=s2.get("s2cloudless"))
        paths.append(gee.download_image(img, grid, out / f"s2_{y}.tif", list(s2.bands) + ["n_obs"],
                                        cfg.download.tile_px, nodata=0))
    return paths[0], paths[1]


def utm_crs(bbox) -> str:
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before"); p.add_argument("--after")
    p.add_argument("--bbox", nargs=4, type=float); p.add_argument("--years", nargs=2, type=int)
    p.add_argument("--data-config", default="configs/ariquemes.yaml")
    p.add_argument("--model-dir", default="outputs/models")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.bbox:
        if not args.years:
            p.error("--bbox needs --years")
        before, after = download(args.bbox, args.years, args.data_config, out)
    elif args.before and args.after:
        before, after = args.before, args.after
    else:
        p.error("give --before/--after or --bbox/--years")

    a, prof_a = read_composite(before)
    b, prof_b = read_composite(after)
    if prof_a["transform"] != prof_b["transform"] or prof_a["crs"] != prof_b["crs"] or a.shape != b.shape:
        raise SystemExit("before/after must share CRS, transform and size")
    pipe = load_torch_pipeline(args.model_dir)
    res = pipe.run(a, b, prof_a["transform"], nodata_mask=(a == 0).all(0) | (b == 0).all(0))
    for k in ("forest_before", "forest_after", "loss"):
        write_mask(out / f"{k}.tif", res[k], prof_a, k)
    stats = {"before": str(before), "after": str(after), "crs": str(prof_a["crs"]), **res["stats"]}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
