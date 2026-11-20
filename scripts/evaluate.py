"""Score every trained method on the test tiles (post-processing chosen on val tiles).

    python scripts/evaluate.py

Writes reports/metrics/{forest_mask_test,change_test,area_test,prodes_test}.json and
<model-dir>/deploy.json (the configuration predict.py and the API run).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from deforest import experiment, prodes
from deforest.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", default="configs/ariquemes.yaml")
    p.add_argument("--config", default="configs/unet.yaml")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-prodes", action="store_true", help="skip the PRODES download/check")
    args = p.parse_args()

    fast = os.environ.get("FAST_DEV_RUN", "0") == "1"
    root = Path("outputs/fast_dev") if fast else Path(".")
    model_dir = Path(args.model_dir) if args.model_dir else root / "outputs/models"
    md = root / "reports/metrics"
    md.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() and not fast
                          else ("cpu" if args.device == "auto" else args.device))
    cfg, ucfg = load_config(args.data_config), load_config(args.config)
    C = dict(ucfg.change)
    n_boot = ucfg.fast_dev_run.bootstrap if fast else C["bootstrap"]
    seed = ucfg.seed

    def save(obj, name):
        (md / name).write_text(json.dumps(obj, indent=2))
        print("wrote", md / name)

    b = experiment.load_bundle(cfg)
    models = experiment.load_models(ucfg, model_dir, device)
    pred, prob = experiment.predict_all(models, b, device)

    forest = experiment.evaluate_forest(b, pred, n_boot, seed)
    save({"task": "forest mask", "split": "test", "n_tiles": len(b.tiles["test"]), "years": list(b.years),
          "ndvi_threshold": models.ndvi_threshold, "results": forest}, "forest_mask_test.json")
    change_res, final = experiment.evaluate_change(b, pred, prob, C, n_boot, seed)
    save({"task": "forest loss", "reference": f"Hansen lossyear {b.y1 + 1}-{b.y2}", "split": "test",
          "selection": "MMU (and threshold for direct models) chosen on val tiles", "results": change_res},
         "change_test.json")

    prodes_loss = None
    if not args.no_prodes:
        fc = prodes.load_or_fetch(cfg.aoi.bbox, Path(cfg.paths.raw) / "prodes_yearly_deforestation.geojson")
        prodes_loss = prodes.rasterize_years(fc, range(b.y1 + 1, b.y2 + 1), b.valid.shape, b.transform, cfg.aoi.crs)
        save(experiment.prodes_summary(b, final, prodes_loss), "prodes_test.json")
    save(experiment.area_summary(b, final, prodes_loss), "area_test.json")

    (model_dir / "deploy.json").write_text(json.dumps(experiment.deploy_spec(change_res, ucfg), indent=2))
    print("wrote", model_dir / "deploy.json")
    for kind, res in (("forest", forest), ("loss", change_res)):
        for m, r in res.items():
            print(f"{kind:>6} {m:>15}: IoU {r['pooled']['iou']:.4f} F1 {r['pooled']['f1']:.4f}")


if __name__ == "__main__":
    main()
