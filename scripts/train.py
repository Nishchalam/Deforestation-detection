"""Fit the baselines and train the U-Nets on the train tiles (selection on val tiles).

    python scripts/train.py                                  # everything
    python scripts/train.py --only unet_change rf_change     # a subset
    FAST_DEV_RUN=1 python scripts/train.py --device cpu      # smoke test -> outputs/fast_dev/

Writes artifacts to --model-dir and reports/metrics/train_<method>.json.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from deforest import experiment
from deforest.config import load_config, save_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", default="configs/ariquemes.yaml")
    p.add_argument("--config", default="configs/unet.yaml")
    p.add_argument("--set", nargs="*", default=[], help="overrides for --config, e.g. unet.epochs=10")
    p.add_argument("--only", nargs="*", choices=experiment.FOREST_METHODS + experiment.DIRECT_METHODS)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    fast = os.environ.get("FAST_DEV_RUN", "0") == "1"
    root = Path("outputs/fast_dev") if fast else Path(".")
    model_dir = Path(args.model_dir) if args.model_dir else root / "outputs/models"
    metrics_dir = root / "reports/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() and not fast
                          else ("cpu" if args.device == "auto" else args.device))
    cfg = load_config(args.data_config)
    ucfg = load_config(args.config, args.set)
    save_config(ucfg, model_dir / "config.resolved.yaml")

    b = experiment.load_bundle(cfg)
    out = experiment.train_all(b, ucfg, model_dir, device, fast=fast, only=args.only)
    experiment.write_train_metrics(out, metrics_dir, device)


if __name__ == "__main__":
    main()
