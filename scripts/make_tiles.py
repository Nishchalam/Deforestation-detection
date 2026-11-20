"""Cut the AOI into full tiles and assign whole tiles to train/val/test (blocked split).

    python scripts/make_tiles.py --config configs/ariquemes.yaml

Writes reports/metrics/split.json (tile ids, windows, per-split label stats) and
reports/figures/split.png.
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

from deforest import tiles, viz
from deforest.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="reports/metrics/split.json")
    p.add_argument("--fig", default="reports/figures/split.png")
    args = p.parse_args()
    cfg = load_config(args.config)
    y1, y2 = cfg.years
    size = cfg.tiling.tile_px

    with rasterio.open(Path(cfg.paths.processed) / "labels.tif") as src:
        f1, f2, loss, valid = (src.read(i).astype(bool) for i in range(1, 5))
        H, W = src.height, src.width
    with rasterio.open(Path(cfg.paths.raw) / f"s2_{y2}.tif") as s:
        img = viz.rgb(s.read()[:6])

    n_r, n_c = tiles.tile_grid(H, W, size)
    count = lambda m: tiles.per_tile_counts(m & valid, size)  # noqa: E731
    fc, lc, vc = count(f1) + count(f2), count(loss), 2 * count(valid)
    split = tiles.choose_split(fc, lc, vc, tuple(cfg.tiling.test_block), tuple(cfg.tiling.val_block))

    def stats(cells):
        idx = tuple(np.array(cells).T)
        v = count(valid)[idx].sum()
        return {
            "n_tiles": len(cells),
            "valid_px": int(v),
            f"forest_frac_{y1}": round(float(count(f1)[idx].sum() / v), 4),
            f"forest_frac_{y2}": round(float(count(f2)[idx].sum() / v), 4),
            f"loss_frac_{y1 + 1}_{y2}": round(float(count(loss)[idx].sum() / v), 5),
            f"loss_px_{y1 + 1}_{y2}": int(count(loss)[idx].sum()),
        }

    out = {
        "tile_px": size,
        "grid_tiles": [n_r, n_c],
        "dropped_strips_px": {"right": W - n_c * size, "bottom": H - n_r * size},
        "rule": "test = test_block whose forest+loss fractions are closest to the tiled-area fractions; "
                "val = same rule over val_block (both orientations), not touching test; rest = train",
        "all": stats([(r, c) for r in range(n_r) for c in range(n_c)]),
    }
    for name, cells in split.items():
        out[name] = {"stats": stats(cells),
                     "tiles": [{"id": tiles.Tile(r, c, size).id, "row": r, "col": c} for r, c in cells]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    colors = {"train": (0.2, 0.4, 1.0), "val": (1.0, 0.8, 0.0), "test": (1.0, 0.1, 0.1)}
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    ax.imshow(img)
    for name, cells in split.items():
        for r, c in cells:
            ax.add_patch(plt.Rectangle((c * size, r * size), size, size, fill=True, alpha=0.25,
                                       fc=colors[name], ec="white", lw=0.5))
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[n], alpha=0.6) for n in colors]
    ax.legend(handles, [f"{n} ({len(split[n])} tiles)" for n in colors], loc="lower left")
    ax.set_title(f"{cfg.name}: {size}px tiles over S2 {y2} (no overlay = dropped strips)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(args.fig, dpi=80)

    for name in ("all", "train", "val", "test"):
        s = out[name] if name == "all" else out[name]["stats"]
        print(f"{name:>5}: {s}")
    print(f"Wrote {args.out}, {args.fig}")


if __name__ == "__main__":
    main()
