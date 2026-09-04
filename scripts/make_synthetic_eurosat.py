"""Generate a synthetic, EuroSAT-shaped dataset for smoke-testing the pipeline.

The real EuroSAT archive (madm.dfki.de / Zenodo / HuggingFace) is not reachable
from every environment.  This script fabricates a dataset with the same folder
layout, class names, image size and file format so that ``train.py``,
``evaluate.py`` and ``run_demo.py`` can be exercised end to end.

The images are procedurally generated colour/texture patterns -- they are NOT
satellite imagery.  Accuracy numbers obtained on this data say something about
whether the code works, and nothing at all about the published EuroSAT results.

Usage:
    python scripts/make_synthetic_eurosat.py --per-class 200
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (class name, base RGB, texture kind)
CLASSES = [
    ("AnnualCrop",           (150, 170,  70), "stripes_h"),
    ("Forest",               ( 30,  80,  35), "noise"),
    ("HerbaceousVegetation", (110, 150,  80), "blobs"),
    ("Highway",              (120, 120, 125), "stripes_v"),
    ("Industrial",           (170, 150, 140), "blocks"),
    ("Pasture",              (140, 175, 100), "flat"),
    ("PermanentCrop",        (120, 140,  60), "grid"),
    ("Residential",          (185, 140, 130), "checker"),
    ("River",                ( 60, 105, 150), "diag"),
    ("SeaLake",              ( 25,  60, 130), "gradient"),
]

SIZE = 64


def render(rng: np.random.Generator, base, kind) -> np.ndarray:
    img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    img[:] = base
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]

    if kind == "stripes_h":
        img += (np.sin(yy / 3.0) * 28)[..., None]
    elif kind == "stripes_v":
        img += (np.sin(xx / 2.0) * 34)[..., None]
    elif kind == "diag":
        img += (np.sin((xx + yy) / 5.0) * 30)[..., None]
    elif kind == "grid":
        img += ((np.sin(xx / 4.0) + np.sin(yy / 4.0)) * 18)[..., None]
    elif kind == "checker":
        img += (((xx // 8 + yy // 8) % 2) * 45 - 22)[..., None]
    elif kind == "blocks":
        blk = rng.uniform(-40, 40, size=(8, 8, 1)).repeat(8, 0).repeat(8, 1)
        img += blk
    elif kind == "blobs":
        cx, cy = rng.uniform(10, 54, 2)
        img += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 200.0) * 55)[..., None]
    elif kind == "gradient":
        img += (yy / SIZE * 50 - 25)[..., None]
    elif kind == "noise":
        img += rng.normal(0, 18, size=(SIZE, SIZE, 1))

    img += rng.normal(0, 6, size=img.shape)          # sensor noise
    img += rng.normal(0, 5, size=(1, 1, 3))          # per-scene colour shift
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = PROJECT_ROOT / "data" / "raw" / "EuroSAT"
    processed = PROJECT_ROOT / "data" / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for label, (name, base, kind) in enumerate(CLASSES):
        (raw / name).mkdir(exist_ok=True)
        for i in range(1, args.per_class + 1):
            fname = f"{name}_{i}.jpg"
            Image.fromarray(render(rng, base, kind)).save(raw / name / fname, quality=92)
            rows.append({"image_path": f"{name}/{fname}", "label": label, "class_name": name})

    df = pd.DataFrame(rows)
    # Same 80/10/10 stratified split as notebooks/02_Preprocessing.ipynb
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=df.loc[temp_idx, "label"], random_state=42
    )

    df.loc[train_idx].to_csv(processed / "train.csv", index=False)
    df.loc[val_idx].to_csv(processed / "validation.csv", index=False)
    df.loc[test_idx].to_csv(processed / "test.csv", index=False)

    print(f"Wrote {len(df)} images to {raw}")
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")


if __name__ == "__main__":
    main()
