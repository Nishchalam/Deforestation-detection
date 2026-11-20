"""Forest masks for two dates, a forest-loss map and area stats for any area.

From GeoTIFFs you already have (6 bands B2,B3,B4,B8,B11,B12, reflectance x 1e4, same grid):
    python scripts/predict.py --before s2_2018.tif --after s2_2022.tif --out outputs/predict/demo

Or download the composites from Earth Engine for a bbox and two years (same recipe as training):
    python scripts/predict.py --bbox -63.0 -10.5 -62.8 -10.3 --years 2018 2022 --out outputs/predict/ariquemes

Writes forest_before.tif, forest_after.tif, loss.tif and stats.json to --out (and hansen_loss.tif
when the end year is covered by Hansen). --months M1 M2 picks the season (default 6 9).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deforest import area
from deforest.config import load_config
from deforest.predict import load_torch_pipeline, read_composite, write_mask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before"); p.add_argument("--after")
    p.add_argument("--bbox", nargs=4, type=float); p.add_argument("--years", nargs=2, type=int)
    p.add_argument("--months", nargs=2, type=int, default=[6, 9], metavar=("M1", "M2"),
                   help="first and last month of the season to composite (default 6 9 = Jun-Sep)")
    p.add_argument("--data-config", default="configs/ariquemes.yaml")
    p.add_argument("--model-dir", default="outputs/models")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe_dir = args.model_dir
    if args.bbox:
        if not args.years:
            p.error("--bbox needs --years")
        start, end = args.years
        try:
            area.validate_request(args.bbox, start, end, tuple(args.months), max_km2=float("inf"))
        except area.RequestError as e:
            raise SystemExit(f"invalid request: {e}")
        cfg = load_config(args.data_config)
        inp = area.fetch_inputs(args.bbox, start, end, args.months, cfg, out,
                                lambda msg, frac: print(f"[{frac:.0%}] {msg}", flush=True))
        pipe = load_torch_pipeline(pipe_dir)
        res = area.run_area(pipe, inp, start, end, cfg.labels.treecover_threshold)
        prof = {"crs": inp.crs, "transform": inp.transform, "width": inp.before.shape[2], "height": inp.before.shape[1]}
        if res["hansen_loss"] is not None:
            write_mask(out / "hansen_loss.tif", res["hansen_loss"], prof, "hansen_loss")
        stats = {"before": str(out / f"s2_{start}.tif"), "after": str(out / f"s2_{end}.tif"), "crs": inp.crs,
                 **res["stats"]}
    elif args.before and args.after:
        a, prof_a = read_composite(args.before)
        b, prof_b = read_composite(args.after)
        if prof_a["transform"] != prof_b["transform"] or prof_a["crs"] != prof_b["crs"] or a.shape != b.shape:
            raise SystemExit("before/after must share CRS, transform and size")
        pipe = load_torch_pipeline(pipe_dir)
        res = pipe.run(a, b, prof_a["transform"], nodata_mask=(a == 0).all(0) | (b == 0).all(0))
        prof = prof_a
        stats = {"before": str(args.before), "after": str(args.after), "crs": str(prof_a["crs"]), **res["stats"]}
    else:
        p.error("give --before/--after or --bbox/--years")
    for k in ("forest_before", "forest_after", "loss"):
        write_mask(out / f"{k}.tif", res[k], prof, k)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
