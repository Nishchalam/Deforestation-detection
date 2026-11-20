"""Export the deployed U-Nets to ONNX and check they match PyTorch.

The export directory is what the API needs and what gets published as release assets:
    forest_unet.onnx   (N,6,256,256)  -> (N,1,256,256) forest logits
    change_unet.onnx   (N,12,256,256) -> (N,1,256,256) loss logits (before bands, then after bands)
    model_card.json    bands, normalisation, thresholds, provenance, sha256 of each .onnx
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ONNX_FILES = {"forest": "forest_unet.onnx", "change": "change_unet.onnx"}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_onnx(model, in_channels: int, path: str | Path, tile: int = 256, opset: int = 17) -> Path:
    import torch

    model = model.cpu().eval()
    dummy = torch.zeros(1, in_channels, tile, tile)
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, dynamo=False)
    import onnx
    onnx.checker.check_model(str(path))
    return Path(path)


def ort_session(path: str | Path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_infer(session):
    """ONNX Runtime session -> `infer(batch) -> (N,S,S) logits`, the callable predict.Pipeline expects."""
    name = session.get_inputs()[0].name

    def fn(batch: np.ndarray) -> np.ndarray:
        return session.run(None, {name: batch.astype(np.float32)})[0][:, 0]
    return fn


def write_model_card(out_dir: str | Path, spec: dict, mean: np.ndarray, std: np.ndarray, extra: dict) -> dict:
    out = Path(out_dir)
    card = {"bands": spec["bands"], "mean": [float(v) for v in mean], "std": [float(v) for v in std],
            "input_scale": "Sentinel-2 L1C reflectance x 1e4 (COPERNICUS/S2_HARMONIZED), median composite",
            "encoder": spec["encoder"], "tile_px": spec["tile_px"],
            "forest_threshold_logit": 0.0, "change_threshold": spec["change_threshold"],
            "change_mmu_px": spec["change_mmu_px"],
            "files": {k: {"name": f, "sha256": sha256(out / f)} for k, f in ONNX_FILES.items()}, **extra}
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    return card


def load_onnx_pipeline(model_dir: str | Path, threads: int | None = None, overlap: int = 64):
    """Pipeline from an export directory (model_card.json + the two .onnx files). No torch needed."""
    from .predict import Pipeline

    md = Path(model_dir)
    card = json.loads((md / "model_card.json").read_text())
    for k, f in card["files"].items():
        if not (md / f["name"]).exists():
            raise FileNotFoundError(f"{md / f['name']} missing; run scripts/fetch_weights.py")
    sess = {k: ort_session(md / f["name"], threads) for k, f in card["files"].items()}
    return Pipeline(onnx_infer(sess["forest"]), onnx_infer(sess["change"]),
                    np.array(card["mean"], np.float32), np.array(card["std"], np.float32),
                    card["change_threshold"], card["change_mmu_px"], card["tile_px"], overlap), card
