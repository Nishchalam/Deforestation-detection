"""Build per-pixel labels from Hansen on the S2 grid.

    python scripts/make_labels.py --config configs/ariquemes.yaml

Writes <paths.processed>/labels.tif, uint8 bands:
  forest_<y1>, forest_<y2>, loss_<y1+1>_<y2>, valid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from deforest import labels
from deforest.config import load_config, save_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    raw, out_dir = Path(cfg.paths.raw), Path(cfg.paths.processed)
    y1, y2 = cfg.years
    thr = cfg.labels.treecover_threshold

    with rasterio.open(raw / "hansen.tif") as h:
        tc, ly, dm = h.read()
        profile = h.profile
    n_obs = []
    for y in (y1, y2):
        with rasterio.open(raw / f"s2_{y}.tif") as s:
            assert s.transform == profile["transform"] and s.crs == profile["crs"]
            n_obs.append(s.read(s.count))  # last band = n_obs
    valid = labels.valid_mask(dm, *n_obs)
    bands = {
        f"forest_{y1}": labels.forest_mask(tc, ly, y1, thr),
        f"forest_{y2}": labels.forest_mask(tc, ly, y2, thr),
        f"loss_{y1 + 1}_{y2}": labels.loss_mask(tc, ly, y1, y2, thr),
        "valid": valid,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.update(count=len(bands), dtype="uint8", nodata=None)
    with rasterio.open(out_dir / "labels.tif", "w", **profile) as dst:
        for i, (name, arr) in enumerate(bands.items(), start=1):
            dst.write(arr.astype(np.uint8), i)
            dst.set_band_description(i, name)
    save_config(cfg, out_dir / "config.resolved.yaml")
    for name, arr in bands.items():
        print(f"{name:>16}: {arr.mean():.4f} of pixels")
    print(f"Wrote {out_dir / 'labels.tif'}")


if __name__ == "__main__":
    main()
