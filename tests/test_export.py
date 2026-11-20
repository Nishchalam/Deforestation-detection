"""ONNX export of an (untrained) U-Net reproduces the PyTorch logits and plugs into Pipeline."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from deforest import export, models, predict  # noqa: E402


def test_onnx_matches_torch(tmp_path):
    torch.manual_seed(0)
    m = models.build_unet("resnet18", None, 6).eval()
    path = export.to_onnx(m, 6, tmp_path / "m.onnx", tile=64)
    x = np.random.default_rng(0).normal(size=(3, 6, 64, 64)).astype(np.float32)
    ref = predict.torch_infer(m, torch.device("cpu"))(x)
    got = export.onnx_infer(export.ort_session(path))(x)
    assert got.shape == (3, 64, 64)
    assert np.abs(ref - got).max() < 1e-4
    # sliding window over a non-multiple size goes through the ONNX callable
    big = np.random.default_rng(1).normal(size=(6, 50, 150)).astype(np.float32)
    out = predict.sliding_logits(export.onnx_infer(export.ort_session(path)), big, tile=64, overlap=16)
    assert out.shape == (50, 150)
