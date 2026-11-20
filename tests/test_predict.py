import json

import numpy as np
import rasterio
from rasterio.transform import Affine

from deforest import predict


def test_sliding_logits_covers_odd_sizes_and_is_exact_for_pointwise_model():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 300, 517)).astype(np.float32)
    infer = lambda b: b[:, 0] * 2.0  # noqa: E731  pointwise "model": output depends only on the pixel
    out = predict.sliding_logits(infer, x, tile=128, overlap=32, batch=3)
    assert out.shape == (300, 517)
    np.testing.assert_allclose(out, x[0] * 2.0, rtol=1e-5, atol=1e-5)


def test_sliding_logits_smaller_than_tile():
    x = np.ones((6, 40, 50), np.float32)
    out = predict.sliding_logits(lambda b: b[:, 0], x, tile=64, overlap=16)
    assert out.shape == (40, 50) and np.allclose(out, 1)


def test_pipeline_run_and_geotiff_roundtrip(tmp_path):
    h, w = 70, 90
    before = np.full((6, h, w), 1000, np.uint16)
    after = before.copy()
    after[:, 10:40, 10:40] = 3000          # "cleared" square
    forest_inf = lambda b: np.where(b[:, 0] < 0, 5.0, -5.0)          # dark (below mean) = forest  # noqa: E731
    change_inf = lambda b: np.where(b[:, 6] > b[:, 0] + 0.5, 5.0, -5.0)  # brighter in year 2 = loss  # noqa: E731
    pipe = predict.Pipeline(forest_inf, change_inf, mean=np.full(6, 2000, np.float32), std=np.full(6, 1000, np.float32),
                            change_threshold=0.5, change_mmu_px=10, tile=32, overlap=8)
    t = Affine(10, 0, 500000, 0, -10, 8861420)
    res = pipe.run(before, after, t)
    assert res["loss"].sum() == 900 and res["stats"]["loss_ha"] == 9.0
    assert res["forest_before"].all() and res["stats"]["forest_after_ha"] == round((h * w - 900) * 0.01, 2)

    prof = {"crs": "EPSG:32720", "transform": t, "width": w, "height": h}
    path = predict.write_mask(tmp_path / "loss.tif", res["loss"], prof, "loss")
    with rasterio.open(path) as src:
        assert src.crs.to_string() == "EPSG:32720" and src.transform == t
        assert (src.read(1).astype(bool) == res["loss"]).all()
    json.dumps(res["stats"])  # stats are JSON-serialisable
