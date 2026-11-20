import numpy as np
import pytest
from rasterio.transform import Affine
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

from deforest import metrics


def test_scores_match_sklearn():
    rng = np.random.default_rng(0)
    true = rng.random((50, 50)) < 0.3
    pred = rng.random((50, 50)) < 0.4
    s = metrics.scores(metrics.confusion(pred, true))
    t, p = true.ravel(), pred.ravel()
    assert s["iou"] == pytest.approx(jaccard_score(t, p))
    assert s["f1"] == pytest.approx(f1_score(t, p))
    assert s["precision"] == pytest.approx(precision_score(t, p))
    assert s["recall"] == pytest.approx(recall_score(t, p))


def test_confusion_ignores_invalid_pixels():
    pred = np.array([1, 1, 0, 0], bool)
    true = np.array([1, 0, 1, 0], bool)
    valid = np.array([1, 0, 1, 1], bool)
    assert metrics.confusion(pred, true, valid) == {"tp": 1, "fp": 0, "fn": 1, "tn": 1}


def test_empty_prediction_gives_nan_precision_not_crash():
    s = metrics.scores({"tp": 0, "fp": 0, "fn": 3, "tn": 5})
    assert np.isnan(s["precision"]) and s["recall"] == 0.0


def test_add_and_bootstrap_ci_brackets_pooled_score():
    counts = [{"tp": 10 + i, "fp": 3, "fn": 2 + i % 3, "tn": 100} for i in range(9)]
    pooled = metrics.scores(metrics.add(counts))["iou"]
    lo, hi = metrics.bootstrap_ci(counts, n=500, seed=0)["iou"]
    assert lo <= pooled <= hi


def test_area_10m_pixel_is_001_ha():
    t = Affine(10, 0, 500000, 0, -10, 8861420)
    assert metrics.pixel_area_ha(t) == pytest.approx(0.01)
    assert metrics.area_ha(np.ones((100, 100), bool), t) == pytest.approx(100.0)  # 1 km^2


def test_area_uses_transform_not_patch_size():
    # the v1 bug: 64-px patches counted as 64 m x 64 m regardless of pixel size
    assert metrics.area_ha(np.ones((64, 64), bool), Affine(21.6, 0, 0, 0, -21.6, 0)) == pytest.approx(64 * 64 * 21.6**2 / 1e4)
