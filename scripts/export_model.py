"""Export the deployed forest and change U-Nets to ONNX and check parity with PyTorch.

    python scripts/export_model.py            # -> outputs/export/, reports/metrics/export_parity.json

Parity is measured on the test tiles: max |logit difference| per model, how many pixels flip
their mask, and the loss/forest hectares of the full sliding-window pipeline on the test block
run once with PyTorch and once with ONNX Runtime.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from deforest import experiment, export, predict
from deforest.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", default="configs/ariquemes.yaml")
    p.add_argument("--model-dir", default="outputs/models")
    p.add_argument("--out", default="outputs/export")
    p.add_argument("--tol", type=float, default=1e-4, help="max allowed |logit diff|")
    args = p.parse_args()

    md, out = Path(args.model_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((md / "deploy.json").read_text())
    cpu = torch.device("cpu")
    tp = predict.load_torch_pipeline(md, cpu)
    U, n = {"encoder": spec["encoder"]}, len(spec["bands"])
    models = {"forest": experiment.load_unet(md / spec["forest_model"], U, n, cpu),
              "change": experiment.load_unet(md / spec["change_model"], U, 2 * n, cpu)}
    for k, m in models.items():
        export.to_onnx(m, n if k == "forest" else 2 * n, out / export.ONNX_FILES[k], spec["tile_px"])
        print("exported", out / export.ONNX_FILES[k])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    export.write_model_card(out, spec, tp.mean, tp.std, {
        "trained_on": "Ariquemes, Rondonia, Brazil; bbox [-63.0,-10.5,-62.8,-10.3]; 2018 vs 2022 dry-season composites",
        "labels": "Hansen GFC treecover2000 >= 30 minus lossyear (see configs/ariquemes.yaml)",
        "source_commit": commit, "torch_checkpoints": [spec["forest_model"], spec["change_model"]]})
    op, _ = export.load_onnx_pipeline(out)

    # tile-level logit parity
    cfg = load_config(args.data_config)
    b = experiment.load_bundle(cfg)
    tiles = b.tiles["test"]
    xf = np.stack([tp.normalize(b.img[y][(slice(None),) + t.window]) for y in b.years for t in tiles])
    xc = np.stack([np.concatenate([tp.normalize(b.img[b.y1][(slice(None),) + t.window]),
                                   tp.normalize(b.img[b.y2][(slice(None),) + t.window])]) for t in tiles])
    res = {}
    for k, x, inf_t, inf_o in (("forest", xf, tp.infer_forest, op.infer_forest),
                               ("change", xc, tp.infer_change, op.infer_change)):
        lt = np.concatenate([inf_t(x[i:i + 8]) for i in range(0, len(x), 8)])
        lo = np.concatenate([inf_o(x[i:i + 8]) for i in range(0, len(x), 8)])
        thr = 0.0 if k == "forest" else float(np.log(spec["change_threshold"] / (1 - spec["change_threshold"])))
        res[k] = {"tiles": int(len(x)), "max_abs_logit_diff": float(np.abs(lt - lo).max()),
                  "mean_abs_logit_diff": float(np.abs(lt - lo).mean()),
                  "mask_pixels_differing": int(((lt > thr) != (lo > thr)).sum()), "pixels": int(lt.size)}

    # full pipeline on the test block (sliding window, MMU, hectares)
    win, _ = experiment.test_block(b)
    before, after = b.img[b.y1][(slice(None),) + win], b.img[b.y2][(slice(None),) + win]
    st = tp.run(before, after, b.transform)["stats"]
    so = op.run(before, after, b.transform)["stats"]
    res["pipeline_test_block"] = {"torch": st, "onnx": so}
    res["tolerance"] = args.tol
    res["passed"] = bool(all(res[k]["max_abs_logit_diff"] <= args.tol for k in ("forest", "change")))
    res["onnx_sha256"] = {k: export.sha256(out / f) for k, f in export.ONNX_FILES.items()}
    Path("reports/metrics/export_parity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("forest", "change", "passed")}, indent=2))
    print("pipeline torch:", st, "\npipeline onnx: ", so)
    if not res["passed"]:
        raise SystemExit(f"parity check failed: tolerance {args.tol}")


if __name__ == "__main__":
    main()
