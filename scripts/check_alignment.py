"""Figure proving S2 composites and Hansen layers share one grid; prints grid/label stats.

    python scripts/check_alignment.py --config configs/ariquemes.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from deforest import labels, viz
from deforest.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="reports/figures/alignment.png")
    p.add_argument("--zoom", type=int, default=300)
    p.add_argument("--stats", default="reports/metrics/data_stats.json")
    args = p.parse_args()
    cfg = load_config(args.config)
    raw = Path(cfg.paths.raw)
    y1, y2 = cfg.years
    thr = cfg.labels.treecover_threshold

    with rasterio.open(raw / f"s2_{y1}.tif") as a, rasterio.open(raw / f"s2_{y2}.tif") as b, \
            rasterio.open(raw / "hansen.tif") as h:
        for src in (b, h):  # same grid, by construction; assert it
            assert src.crs == a.crs and src.transform == a.transform and src.shape == a.shape, src.name
        s1, s2, hz = a.read(), b.read(), h.read()
        crs, transform = a.crs, a.transform
    tc, ly, dm = hz
    valid = labels.valid_mask(dm, s1[-1], s2[-1])
    f1 = labels.forest_mask(tc, ly, y1, thr) & valid
    f2 = labels.forest_mask(tc, ly, y2, thr) & valid
    loss = labels.loss_mask(tc, ly, y1, y2, thr) & valid

    img1, img2 = viz.rgb(s1[:6]), viz.rgb(s2[:6], ref=s1[:6])
    r, c = viz.densest_window(loss, args.zoom)
    win = np.s_[r:r + args.zoom, c:c + args.zoom]

    fig, ax = plt.subplots(2, 3, figsize=(15, 11), layout="constrained")
    ax[0, 0].imshow(img1); ax[0, 0].set_title(f"S2 L1C {y1} Jun-Sep median (RGB)")
    ax[0, 1].imshow(img2); ax[0, 1].set_title(f"S2 L1C {y2} Jun-Sep median (RGB)")
    hl = np.zeros(tc.shape + (3,))
    hl[f2] = (0.1, 0.5, 0.1)
    hl[loss] = (0.9, 0.1, 0.1)
    hl[labels.loss_mask(tc, ly, 2000, y1, thr)] = (0.95, 0.65, 0.1)
    ax[0, 2].imshow(hl)
    ax[0, 2].set_title(f"Hansen: forest {y2} (green), loss {y1+1}-{y2} (red),\nloss 2001-{y1} (orange)")
    for a_ in ax[0]:
        a_.add_patch(plt.Rectangle((c, r), args.zoom, args.zoom, fill=False, ec="yellow", lw=1.5))
    e = viz.edges(loss[win])
    ax[1, 0].imshow(viz.overlay(img1[win], e)); ax[1, 0].set_title(f"Zoom {y1} + Hansen loss {y1+1}-{y2} outline")
    ax[1, 1].imshow(viz.overlay(img2[win], e)); ax[1, 1].set_title(f"Zoom {y2} + same outline")
    ax[1, 2].imshow(viz.overlay(img2[win], viz.edges(f2[win]), (1, 1, 0)))
    ax[1, 2].set_title(f"Zoom {y2} + Hansen forest {y2} outline")
    for a_ in ax.flat:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{cfg.name}: {crs} {transform.a:.0f} m grid, {s1.shape[2]}x{s1.shape[1]} px; "
                 f"zoom window rows {r}-{r+args.zoom}, cols {c}-{c+args.zoom}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)

    stats = {
        "shape": list(s1.shape[1:]),
        "valid_frac": float(valid.mean()),
        f"forest_frac_{y1}": float(f1.sum() / valid.sum()),
        f"forest_frac_{y2}": float(f2.sum() / valid.sum()),
        f"loss_{y1+1}_{y2}_px": int(loss.sum()),
        f"n_obs_{y1}_min_median": [int(s1[-1].min()), float(np.median(s1[-1]))],
        f"n_obs_{y2}_min_median": [int(s2[-1].min()), float(np.median(s2[-1]))],
    }
    print(json.dumps(stats, indent=2))
    Path(args.stats).write_text(json.dumps(stats, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
