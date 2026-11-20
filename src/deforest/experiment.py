"""The experiment as library code: load data, fit/train every method, predict tiles, select
post-processing on val, and score on test. Used by scripts/train.py, scripts/evaluate.py and
notebooks/end_to_end.ipynb, so there is one implementation of each step.

Artifacts live in one model directory (default outputs/models):
  norm_stats.npz, ndvi.json, random_forest.joblib, rf_change.joblib,
  unet_imagenet/best.pt, unet_scratch/best.pt, unet_change/best.pt, deploy.json
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

from . import baselines, change, data, metrics
from .models import build_unet
from .tiles import Tile
from .training import Trainer, predict_logits, seed_everything

FOREST_METHODS = ["ndvi", "random_forest", "unet_imagenet", "unet_scratch"]
DIRECT_METHODS = ["rf_change", "unet_change"]


# --------------------------------------------------------------------------- data
@dataclass
class Bundle:
    years: tuple[int, int]
    img: dict[int, np.ndarray]          # (6,H,W) uint16 per year
    forest: dict[int, np.ndarray]       # (H,W) bool per year
    hloss: np.ndarray                   # Hansen loss y1+1..y2
    valid: np.ndarray
    treecover: np.ndarray
    lossyear: np.ndarray
    transform: object
    crs: object
    tiles: dict[str, list[Tile]]
    size: int

    @property
    def y1(self) -> int:
        return self.years[0]

    @property
    def y2(self) -> int:
        return self.years[1]


def load_bundle(cfg, split_path: str | Path = "reports/metrics/split.json") -> Bundle:
    y1, y2 = cfg.years
    img = {}
    for y in (y1, y2):
        img[y], _, profile = data.load_composite(cfg.paths.raw, y)
    lab, lprof = data.load_labels(cfg.paths.processed)
    assert lprof["transform"] == profile["transform"] and lprof["crs"] == profile["crs"]
    with rasterio.open(Path(cfg.paths.raw) / "hansen.tif") as src:
        treecover, lossyear, _ = src.read()
    sp = json.loads(Path(split_path).read_text())
    size = sp["tile_px"]
    tiles = {k: [Tile(t["row"], t["col"], size) for t in sp[k]["tiles"]] for k in ("train", "val", "test")}
    return Bundle((y1, y2), img, {y1: lab[f"forest_{y1}"], y2: lab[f"forest_{y2}"]}, lab[f"loss_{y1 + 1}_{y2}"],
                  lab["valid"], treecover, lossyear, profile["transform"], profile["crs"], tiles, size)


def windows(tiles: list[Tile]) -> list:
    return [t.window for t in tiles]


def stack_pixels(arrs: list[np.ndarray], tiles: list[Tile]) -> np.ndarray:
    return np.concatenate([a[t.window].ravel() for a in arrs for t in tiles])


class Normalizer:
    """Per-band standardisation with statistics from the train tiles of both years."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

    @classmethod
    def fit(cls, b: Bundle) -> "Normalizer":
        return cls(*data.band_stats([b.img[b.y1], b.img[b.y2]], b.tiles["train"]))

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        z = np.load(path)
        return cls(z["mean"], z["std"])

    def save(self, path: str | Path, bands: list[str]) -> None:
        np.savez(path, mean=self.mean, std=self.std, bands=np.array(bands))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return data.normalize(img, self.mean, self.std)


# --------------------------------------------------------------------------- fitting
def fit_ndvi(b: Bundle, grid: list[float]) -> tuple[float, list]:
    lo, hi, step = grid
    nd = {y: baselines.ndvi(b.img[y]) for y in b.years}
    return baselines.best_ndvi_threshold(stack_pixels([nd[b.y1], nd[b.y2]], b.tiles["val"]),
                                         stack_pixels([b.forest[b.y1], b.forest[b.y2]], b.tiles["val"]),
                                         np.arange(lo, hi + 1e-9, step))


def fit_rf_forest(b: Bundle, B: dict, seed: int):
    X, y = baselines.sample_pixels([b.img[b.y1], b.img[b.y2]], [b.forest[b.y1], b.forest[b.y2]],
                                   windows(b.tiles["train"]), B["rf_pixels"], seed)
    return baselines.fit_random_forest(X, y, seed, n_estimators=B["rf_trees"], max_depth=B["rf_max_depth"],
                                       n_jobs=B["rf_n_jobs"])


def fit_rf_change(b: Bundle, CR: dict, n_jobs: int, seed: int):
    X, y = baselines.sample_pixels([np.concatenate([b.img[b.y1], b.img[b.y2]])], [b.hloss],
                                   windows(b.tiles["train"]), CR["pixels"], seed)
    return baselines.fit_random_forest(X, y, seed, n_estimators=CR["trees"], max_depth=CR["max_depth"],
                                       class_weight=CR["class_weight"], n_jobs=n_jobs)


def _worker_init(seed: int):
    def fn(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
    return fn


def train_unet(b: Bundle, norm: Normalizer, kind: str, encoder_weights: str | None, U: dict, run_dir: str | Path,
               device: torch.device, seed: int, epochs: int, lr: float, patience: int, repeats: int,
               max_batches: int | None = None) -> tuple[torch.nn.Module, dict, list[dict]]:
    """kind='forest': 6-band input, target forest(year), samples = train tiles x both years.
    kind='change': 12-band input (y1 then y2), target Hansen loss, samples = train tiles."""
    seed_everything(seed)
    if kind == "forest":
        images = {y: norm(b.img[y]) for y in b.years}
        masks, in_ch = b.forest, U["in_channels"]
    elif kind == "change":
        images = {0: np.concatenate([norm(b.img[b.y1]), norm(b.img[b.y2])])}
        masks, in_ch = {0: b.hloss}, 2 * U["in_channels"]
    else:
        raise ValueError(kind)
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(data.TileDataset(images, masks, b.valid, b.tiles["train"], augment=True, repeats=repeats),
                    batch_size=U["batch_size"], shuffle=True, generator=g, num_workers=U["num_workers"],
                    worker_init_fn=_worker_init(seed), drop_last=True)
    va = DataLoader(data.TileDataset(images, masks, b.valid, b.tiles["val"]),
                    batch_size=U["batch_size"], num_workers=U["num_workers"])
    model = build_unet(U["encoder"], encoder_weights, in_ch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=U["weight_decay"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(model, tr, va, opt, run_dir, epochs=epochs, patience=patience, use_amp=U["amp"],
                      device=device, max_batches=max_batches)
    t0 = time.time()
    history = trainer.fit()
    model = trainer.load_best()
    info = {"kind": kind, "encoder": U["encoder"], "encoder_weights": encoder_weights, "in_channels": in_ch,
            "best_epoch": trainer.best_epoch, "best_val_iou": trainer.best, "epochs_run": len(history),
            "train_time_s": round(time.time() - t0, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None}
    return model, info, history


def load_unet(path: str | Path, U: dict, in_channels: int, device: torch.device) -> torch.nn.Module:
    model = build_unet(U["encoder"], None, in_channels)
    model.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
    return model.to(device).eval()


# --------------------------------------------------------------------------- artifacts
@dataclass
class Models:
    norm: Normalizer
    ndvi_threshold: float
    rf: object
    rf_change: object
    unets: dict  # name -> model (unet_imagenet, unet_scratch, unet_change)


def train_all(b: Bundle, ucfg, model_dir: str | Path, device: torch.device, fast: bool = False,
              only: list[str] | None = None, log=print) -> dict:
    """Fit/train the selected methods and write their artifacts. Returns training summaries."""
    U, B, CU, CR = dict(ucfg.unet), dict(ucfg.baselines), dict(ucfg.change_unet), dict(ucfg.change_rf)
    max_batches = None
    if fast:
        fd = ucfg.fast_dev_run
        U.update(epochs=fd.epochs, num_workers=0)
        B.update(rf_pixels=fd.rf_pixels, rf_trees=fd.rf_trees)
        CU.update(epochs=fd.epochs)
        CR.update(pixels=fd.rf_pixels, trees=fd.rf_trees)
        max_batches = fd.max_batches
    seed = ucfg.seed
    md = Path(model_dir)
    md.mkdir(parents=True, exist_ok=True)
    todo = only or FOREST_METHODS + DIRECT_METHODS
    norm = Normalizer.fit(b)
    norm.save(md / "norm_stats.npz", ["B2", "B3", "B4", "B8", "B11", "B12"])
    out = {}
    if "ndvi" in todo:
        thr, table = fit_ndvi(b, B["ndvi_grid"])
        (md / "ndvi.json").write_text(json.dumps({"threshold": thr, "val_iou_by_threshold": table}))
        out["ndvi"] = {"threshold": thr, "val_iou": max(r[1] for r in table)}
        log(f"NDVI threshold chosen on val: {thr:.2f} (val IoU {out['ndvi']['val_iou']:.4f})")
    if "random_forest" in todo:
        t0 = time.time()
        rf = fit_rf_forest(b, B, seed)
        joblib.dump(rf, md / "random_forest.joblib")
        out["random_forest"] = {"pixels": B["rf_pixels"], "trees": B["rf_trees"], "fit_s": round(time.time() - t0, 1),
                                "feature_importances": dict(zip(["B2", "B3", "B4", "B8", "B11", "B12"],
                                                                rf.feature_importances_.round(4).tolist()))}
        log(f"random forest: {out['random_forest']}")
    for name, weights in U["variants"].items():
        if f"unet_{name}" in todo:
            _, info, hist = train_unet(b, norm, "forest", weights, U, md / f"unet_{name}", device, seed,
                                       U["epochs"], U["lr"], U["patience"], U["repeats"], max_batches)
            out[f"unet_{name}"] = {**info, "config": U, "history": hist}
    if "rf_change" in todo:
        t0 = time.time()
        rfc = fit_rf_change(b, CR, B["rf_n_jobs"], seed)
        joblib.dump(rfc, md / "rf_change.joblib")
        out["rf_change"] = {**CR, "fit_s": round(time.time() - t0, 1)}
        log(f"change random forest: {out['rf_change']}")
    if "unet_change" in todo:
        _, info, hist = train_unet(b, norm, "change", CU["encoder_weights"], U, md / "unet_change", device, seed,
                                   CU["epochs"], CU["lr"], CU["patience"], CU["repeats"], max_batches)
        out["unet_change"] = {**info, "config": {**U, **CU}, "history": hist}
    return out


def load_models(ucfg, model_dir: str | Path, device: torch.device) -> Models:
    """Load whatever artifacts exist in model_dir (missing ones are None / absent)."""
    md, U = Path(model_dir), dict(ucfg.unet)
    unets = {}
    for n in U["variants"]:
        if (md / f"unet_{n}" / "best.pt").exists():
            unets[f"unet_{n}"] = load_unet(md / f"unet_{n}" / "best.pt", U, U["in_channels"], device)
    if (md / "unet_change" / "best.pt").exists():
        unets["unet_change"] = load_unet(md / "unet_change" / "best.pt", U, 2 * U["in_channels"], device)
    opt = lambda f, fn: fn(md / f) if (md / f).exists() else None  # noqa: E731
    return Models(Normalizer.load(md / "norm_stats.npz"),
                  opt("ndvi.json", lambda f: json.loads(f.read_text())["threshold"]),
                  opt("random_forest.joblib", joblib.load), opt("rf_change.joblib", joblib.load), unets)


def available(m: Models) -> tuple[list[str], list[str]]:
    forest = [x for x in FOREST_METHODS if (x == "ndvi" and m.ndvi_threshold is not None)
              or (x == "random_forest" and m.rf is not None) or x in m.unets]
    direct = [x for x in DIRECT_METHODS if (x == "rf_change" and m.rf_change is not None) or x in m.unets]
    return forest, direct


def write_train_metrics(out: dict, metrics_dir: str | Path, device) -> None:
    for name, info in out.items():
        info = {k: v for k, v in info.items() if k != "history"}
        path = Path(metrics_dir) / f"train_{name}.json"
        path.write_text(json.dumps({"method": name, "device": str(device), **info}, indent=2))
        print("wrote", path)


# --------------------------------------------------------------------------- prediction
def predict_forest(m: Models, method: str, b: Bundle, year: int, tiles: list[Tile], device) -> dict[str, np.ndarray]:
    if method == "ndvi":
        nd = baselines.ndvi(b.img[year])
        return {t.id: nd[t.window] >= m.ndvi_threshold for t in tiles}
    if method == "random_forest":
        return {t.id: baselines.predict_random_forest(m.rf, b.img[year][(slice(None),) + t.window]) for t in tiles}
    x = torch.from_numpy(np.stack([m.norm(b.img[year][(slice(None),) + t.window]) for t in tiles]))
    return {t.id: l > 0 for t, l in zip(tiles, predict_logits(m.unets[method], x, device))}


def predict_change_prob(m: Models, method: str, b: Bundle, tiles: list[Tile], device) -> dict[str, np.ndarray]:
    both = np.concatenate([b.img[b.y1], b.img[b.y2]])
    if method == "unet_change":
        x = torch.from_numpy(np.stack([np.concatenate([m.norm(b.img[b.y1][(slice(None),) + t.window]),
                                                       m.norm(b.img[b.y2][(slice(None),) + t.window])]) for t in tiles]))
        return {t.id: 1 / (1 + np.exp(-l)) for t, l in zip(tiles, predict_logits(m.unets[method], x, device))}
    if method == "rf_change":
        return {t.id: m.rf_change.predict_proba(both[(slice(None),) + t.window].reshape(12, -1).T.astype(np.float32))[:, 1]
                .reshape(b.size, b.size) for t in tiles}
    raise ValueError(method)


def predict_all(m: Models, b: Bundle, device) -> tuple[dict, dict]:
    """PRED[method][split][year] -> {tile_id: forest mask}; PROB[method][split] -> {tile_id: loss prob}."""
    forest, direct = available(m)
    pred = {meth: {s: {y: predict_forest(m, meth, b, y, b.tiles[s], device) for y in b.years} for s in ("val", "test")}
            for meth in forest}
    prob = {meth: {s: predict_change_prob(m, meth, b, b.tiles[s], device) for s in ("val", "test")}
            for meth in direct}
    return pred, prob


# --------------------------------------------------------------------------- evaluation
def tile_counts(pred_by_tile: dict, truth: np.ndarray, valid: np.ndarray, tiles: list[Tile]) -> list[dict]:
    return [metrics.confusion(pred_by_tile[t.id], truth[t.window], valid[t.window]) for t in tiles]


def evaluate_forest(b: Bundle, pred: dict, n_boot: int, seed: int) -> dict:
    res = {}
    for meth, p in pred.items():
        tc = [c for y in b.years for c in tile_counts(p["test"][y], b.forest[y], b.valid, b.tiles["test"])]
        pooled = metrics.add(tc)
        res[meth] = {"pooled": {**metrics.scores(pooled), "counts": pooled},
                     "ci95": metrics.bootstrap_ci(tc, n=n_boot, seed=seed),
                     "per_year": {y: metrics.scores(metrics.add(tile_counts(p["test"][y], b.forest[y], b.valid,
                                                                            b.tiles["test"]))) for y in b.years}}
    return res


def _score_change(b: Bundle, final_test: dict, n_boot: int, seed: int) -> dict:
    tc = tile_counts(final_test, b.hloss, b.valid, b.tiles["test"])
    pooled = metrics.add(tc)
    return {"pooled": {**metrics.scores(pooled), "counts": pooled},
            "ci95": metrics.bootstrap_ci(tc, n=n_boot, seed=seed)}


def evaluate_change(b: Bundle, pred: dict, prob: dict, C: dict, n_boot: int, seed: int) -> tuple[dict, dict]:
    """Differencing methods: MMU chosen on val. Direct methods: threshold + MMU chosen jointly on val.
    Returns (results, FINAL[method][split] -> {tile_id: loss mask})."""
    val_iou = lambda f: metrics.scores(metrics.add(tile_counts(f, b.hloss, b.valid, b.tiles["val"])))["iou"]  # noqa: E731
    res, final = {}, {}
    for meth, p in pred.items():
        def loss_tiles(s, k, p=p):
            return {t.id: change.remove_small(change.loss_map(p[s][b.y1][t.id], p[s][b.y2][t.id]), k) for t in b.tiles[s]}
        grid = {k: val_iou(loss_tiles("val", k)) for k in C["mmu_candidates_px"]}
        k = max(grid, key=grid.get)
        final[meth] = {s: loss_tiles(s, k) for s in ("val", "test")}
        res[meth] = {"approach": "differencing", "mmu_px": k, "val_iou": grid[k], "val_iou_by_mmu": grid,
                     **_score_change(b, final[meth]["test"], n_boot, seed),
                     "no_mmu": metrics.scores(metrics.add(tile_counts(loss_tiles("test", 0), b.hloss, b.valid,
                                                                      b.tiles["test"])))}
    for meth, pr in prob.items():
        def thr_tiles(s, th, k, pr=pr):
            return {tid: change.remove_small(v >= th, k) for tid, v in pr[s].items()}
        grid = {(th, k): val_iou(thr_tiles("val", th, k)) for th in C["prob_thresholds"] for k in C["mmu_candidates_px"]}
        th, k = max(grid, key=grid.get)
        final[meth] = {s: thr_tiles(s, th, k) for s in ("val", "test")}
        res[meth] = {"approach": "direct", "threshold": th, "mmu_px": k, "val_iou": grid[(th, k)],
                     **_score_change(b, final[meth]["test"], n_boot, seed),
                     "no_mmu": metrics.scores(metrics.add(tile_counts(thr_tiles("test", th, 0), b.hloss, b.valid,
                                                                      b.tiles["test"])))}
    return res, final


def test_block(b: Bundle) -> tuple[tuple[slice, slice], callable]:
    """Window of the (rectangular) test block and a function mosaicking {tile_id: array} into it."""
    cells = sorted((t.row, t.col) for t in b.tiles["test"])
    r0, c0 = cells[0]
    nr, nc = cells[-1][0] - r0 + 1, cells[-1][1] - c0 + 1
    assert nr * nc == len(cells), "test block is not a full rectangle"
    win = np.s_[r0 * b.size:(r0 + nr) * b.size, c0 * b.size:(c0 + nc) * b.size]

    def mosaic(by_tile: dict) -> np.ndarray:
        out = np.zeros((nr * b.size, nc * b.size), bool)
        for t in b.tiles["test"]:
            out[(t.row - r0) * b.size:(t.row - r0 + 1) * b.size,
                (t.col - c0) * b.size:(t.col - c0 + 1) * b.size] = by_tile[t.id]
        return out

    return win, mosaic


def area_summary(b: Bundle, final: dict, prodes_loss: np.ndarray | None) -> dict:
    win, mosaic = test_block(b)
    area = {"pixel_area_ha": metrics.pixel_area_ha(b.transform), "hansen_ha": metrics.area_ha(b.hloss[win], b.transform)}
    for meth, f in final.items():
        area[f"{meth}_ha"] = metrics.area_ha(mosaic(f["test"]), b.transform)
    if prodes_loss is not None:
        area["prodes_ha"] = metrics.area_ha(prodes_loss[win], b.transform)
    return {"split": "test", "years": f"{b.y1 + 1}-{b.y2}", "test_area_ha": metrics.area_ha(b.valid[win], b.transform),
            **area}


def prodes_summary(b: Bundle, final: dict, prodes_loss: np.ndarray) -> dict:
    win, mosaic = test_block(b)
    pl = prodes_loss[win]
    out = {"reference": f"PRODES yearly_deforestation {b.y1 + 1}-{b.y2} (TerraBrasilis WFS)",
           "hansen_vs_prodes": metrics.scores(metrics.confusion(b.hloss[win], pl, b.valid[win])),
           "hansen_vs_prodes_whole_aoi": metrics.scores(metrics.confusion(b.hloss, prodes_loss, b.valid))}
    for meth, f in final.items():
        out[f"{meth}_vs_prodes"] = metrics.scores(metrics.confusion(mosaic(f["test"]), pl, b.valid[win]))
    out["prodes_test_ha"] = metrics.area_ha(pl, b.transform)
    return out


def deploy_spec(change_res: dict, ucfg) -> dict:
    """What predict.py / the API run: forest masks from the ImageNet U-Net, loss from the direct
    change U-Net with the threshold and MMU chosen on val."""
    r = change_res["unet_change"]
    return {"forest_model": "unet_imagenet/best.pt", "change_model": "unet_change/best.pt",
            "bands": ["B2", "B3", "B4", "B8", "B11", "B12"], "norm_stats": "norm_stats.npz",
            "encoder": ucfg.unet.encoder, "change_threshold": r["threshold"], "change_mmu_px": r["mmu_px"],
            "forest_threshold": 0.5, "tile_px": 256, "selection": "threshold and MMU chosen on val tiles"}
