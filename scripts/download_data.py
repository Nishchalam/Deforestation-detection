"""Download Sentinel-2 dry-season composites and Hansen GFC layers on one shared 10 m grid.

    python scripts/download_data.py --config configs/ariquemes.yaml [--counts-only]

Writes <paths.raw>/s2_<year>.tif, <paths.raw>/hansen.tif and reports/metrics/data_manifest.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import ee

from deforest import gee
from deforest.config import load_config, save_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. sentinel2.collection=...")
    p.add_argument("--project", default=None)
    p.add_argument("--counts-only", action="store_true", help="print usable scene counts and exit")
    p.add_argument("--manifest", default="reports/metrics/data_manifest.json")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)

    gee.initialize(args.project)
    grid = gee.make_grid(cfg.aoi.bbox, cfg.aoi.crs, cfg.aoi.scale)
    region = gee.grid_geometry(grid)
    print(f"Grid {grid.crs} {grid.width}x{grid.height} px at {grid.scale} m, origin ({grid.x0}, {grid.y0})")

    s2 = cfg.sentinel2
    summaries = {}
    for year in cfg.years:
        start, end = gee.season_dates(year, s2.season)
        info = gee.scene_summary(gee.s2_collection(s2.collection, region, start, end))
        summaries[year] = {"start": start, "end_exclusive": end, **info}
        print(f"{s2.collection} {start}..{end}: {info['n_images']} images, "
              f"{info['n_dates']} dates, MGRS {info['mgrs_tiles']}")
        if info["n_images"] == 0:
            raise SystemExit(f"No {s2.collection} scenes for {year}; stopping.")
    if args.counts_only:
        return

    raw = Path(cfg.paths.raw)
    outputs = {}
    for year in cfg.years:
        start, end = gee.season_dates(year, s2.season)
        img = gee.s2_composite(s2.collection, region, start, end, list(s2.bands), list(s2.scl_mask))
        out = raw / f"s2_{year}.tif"
        print(f"Downloading {out}")
        gee.download_image(img, grid, out, list(s2.bands) + ["n_obs"], cfg.download.tile_px, nodata=0)
        outputs[f"s2_{year}"] = str(out)

    asset = gee.hansen_latest() if cfg.hansen.asset == "auto" else cfg.hansen.asset
    print(f"Hansen asset: {asset}")
    out = raw / "hansen.tif"
    gee.download_image(gee.hansen_image(asset), grid, out,
                       ["treecover2000", "lossyear", "datamask"], cfg.download.tile_px)
    outputs["hansen"] = str(out)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "earthengine_api": ee.__version__,
        "grid": {"crs": grid.crs, "x0": grid.x0, "y0": grid.y0, "scale": grid.scale,
                 "width": grid.width, "height": grid.height},
        "sentinel2": {"collection": s2.collection, "bands": list(s2.bands),
                      "scl_mask": list(s2.scl_mask), "scenes": summaries},
        "hansen_asset": asset,
        "files": outputs,
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    save_config(cfg, raw / "config.resolved.yaml")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
