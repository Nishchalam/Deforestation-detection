from datetime import date

import numpy as np
import pytest
from rasterio.transform import Affine
from sklearn.metrics import f1_score

from deforest import area, labels, predict

ARIQUEMES = [-63.0, -10.5, -62.8, -10.3]
T = Affine(10, 0, 500000, 0, -10, 8861420)


def test_utm_crs():
    assert area.utm_crs(ARIQUEMES) == "EPSG:32720"
    assert area.utm_crs([12.0, 0.1, 12.2, 0.3]) == "EPSG:32633"
    assert area.utm_crs([-66.2, -10.0, -66.0, -9.8]) == "EPSG:32719"  # centred at lon -66.1


def test_area_km2():
    assert 400 < area.bbox_area_km2(ARIQUEMES) < 520


def test_last_year():
    assert area.last_year(date(2026, 9, 30)) == 2025
    assert area.last_year(date(2026, 10, 1)) == 2026


def test_validate():
    today = date(2026, 11, 20)
    area.validate_request(ARIQUEMES, 2018, 2022, today=today)  # passes
    area.validate_request(ARIQUEMES, 2018, 2026, months=(6, 9), today=today)
    bad = [
        (dict(start=2015, end=2018), "2016"),
        (dict(start=2020, end=2020), "before"),
        (dict(start=2022, end=2018), "before"),
        (dict(start=2018, end=2027), "2026"),
        (dict(start=2018, end=2022, months=(9, 6)), "month"),
        (dict(start=2018, end=2022, months=(0, 6)), "month"),
        (dict(start=2018, end=2022, months=(6, 13)), "month"),
        (dict(start=2018, end=2022, bbox=[-63.0, -10.5, -62.0, -9.5]), "500"),
        (dict(start=2018, end=2022, bbox=[-62.8, -10.3, -63.0, -10.5]), "bounding box"),
        (dict(start=2018, end=2022, bbox=[-63.0, -95.0, -62.8, -10.3]), "bounding box"),
    ]
    for kw, msg in bad:
        kw = {"bbox": ARIQUEMES, **kw}
        with pytest.raises(area.RequestError, match=msg):
            area.validate_request(today=today, **kw)


def test_validate_rejects_extreme_aspect_boxes():
    """C1: small geodesic area but a huge UTM grid (or a degenerate pole box) must be refused."""
    today = date(2026, 11, 20)
    for bbox in ([-70, -10.0005, -50, -10], [-80, 0, -60, 0.0002], [-180, 89.9, 180, 90]):
        with pytest.raises(area.RequestError, match="1 degree|1°"):
            area.validate_request(bbox, 2018, 2022, today=today)
    area.validate_request(ARIQUEMES, 2018, 2022, today=today)


def test_validate_rejects_big_projected_grid():
    """C1: a box within the 1 degree span but whose 10 m grid has a side above 10,000 px."""
    today = date(2026, 11, 20)
    with pytest.raises(area.RequestError, match="10 m grid"):
        area.validate_request([-63.0, -10.0005, -62.0, -10.0], 2018, 2022, today=today)  # ~109 km x 55 m = 11k px
    # the geodesic area passes (6 km2) but the grid does not: only the projected-grid rule catches it
    assert area.bbox_area_km2([-63.0, -10.0005, -62.0, -10.0]) < 500
    with pytest.raises(area.RequestError, match="10 m grid"):  # the same on the latitude side
        area.validate_request([-63.0, -10.0, -62.9995, -9.0], 2018, 2022, today=today)


def test_validate_rejects_unfinished_season():
    """M4: end year allowed, but the chosen months extend past today."""
    with pytest.raises(area.RequestError, match="not finished|not available"):
        area.validate_request(ARIQUEMES, 2018, 2026, months=(1, 12), today=date(2026, 11, 20))
    area.validate_request(ARIQUEMES, 2018, 2026, months=(6, 9), today=date(2026, 11, 20))
    with pytest.raises(area.RequestError, match="not finished|not available"):  # Oct-Dec 2026 on Oct 15
        area.validate_request(ARIQUEMES, 2018, 2026, months=(10, 12), today=date(2026, 10, 15))
    area.validate_request(ARIQUEMES, 2018, 2026, months=(6, 9), today=date(2026, 10, 15))


def test_s2_dates():
    assert area.s2_dates(2018, (6, 9)) == ("2018-06-01", "2018-10-01")
    assert area.s2_dates(2018, (1, 12)) == ("2018-01-01", "2019-01-01")


# ------------------------------------------------------------------ run_area
def _pipe():
    """Real Pipeline with pointwise stub nets: forest = band 0 > 0; change logit = band 1 of the
    second composite (normalisation is the identity: mean 0, std 1)."""
    return predict.Pipeline(
        infer_forest=lambda b: b[:, 0],
        infer_change=lambda b: b[:, 7],
        mean=np.zeros(6, np.float32), std=np.ones(6, np.float32),
        change_threshold=0.7, change_mmu_px=0, tile=16, overlap=4)


def _inputs(hansen_last_year=2025):
    h = w = 20
    rng = np.random.default_rng(1)
    forest_b = np.ones((h, w), bool)
    forest_a = forest_b.copy()
    forest_a[:, 10:] = False
    before = np.zeros((6, h, w), np.float32)
    after = np.zeros((6, h, w), np.float32)
    before[0] = np.where(forest_b, 1.0, -1.0)
    after[0] = np.where(forest_a, 1.0, -1.0)
    change_logit = np.where(rng.random((h, w)) < 0.3, 5.0, -5.0)
    after[1] = change_logit
    treecover = np.full((h, w), 80, np.uint8)
    treecover[:2] = 10  # below the 30 % threshold: never forest
    lossyear = np.zeros((h, w), np.uint8)
    lossyear[5:9, 3:12] = 19     # 2019, inside 2018..2022
    lossyear[12:15, 0:6] = 21    # 2021, inside
    lossyear[15:18, 8:12] = 23   # 2023, after the window
    lossyear[3:5, 15:18] = 17    # 2017, before the window
    n_obs = np.full((h, w), 7, np.uint16)
    n_obs[:, 0] = 0  # no clear observation in the first composite
    return area.Inputs(
        before=before, after=after, n_obs_before=n_obs, n_obs_after=np.full((h, w), 5, np.uint16),
        treecover=treecover, lossyear=lossyear, datamask=np.ones((h, w), np.uint8),
        transform=T, crs="EPSG:32720", hansen_asset="UMD/hansen/global_forest_change_2025_v1_13",
        hansen_last_year=hansen_last_year), change_logit


def test_run_area_hansen():
    inp, logit = _inputs()
    res = area.run_area(_pipe(), inp, 2018, 2022, 30)
    assert not res["valid"][:, 0].any() and res["valid"][:, 1:].all()  # n_obs == 0 excluded
    true = labels.loss_mask(inp.treecover, inp.lossyear, 2018, 2022, 30) & res["valid"]
    assert true.sum() > 0
    np.testing.assert_array_equal(res["hansen_loss"], true)
    h = res["stats"]["hansen"]
    assert h["asset"] == inp.hansen_asset
    assert h["loss_ha"] == round(float(true.sum()) * 0.01, 2)
    pred = (1 / (1 + np.exp(-logit)) >= 0.7) & res["valid"]
    np.testing.assert_array_equal(res["loss"], pred)
    v = res["valid"]
    assert h["agreement"]["f1"] == pytest.approx(f1_score(true[v], pred[v]))
    assert set(h["agreement"]) == {"iou", "f1", "precision", "recall"}
    s = res["stats"]
    assert s["loss_pct_of_forest"] == pytest.approx(100 * s["loss_ha"] / s["forest_before_ha"], abs=0.01)
    assert s["clear_obs_median"] == {"before": 7, "after": 5}
    assert s["loss_pixels"] == int(pred.sum()) and s["valid_ha"] == round(float(v.sum()) * 0.01, 2)


def test_loss_pct_undefined_without_forest():
    """I3: no forest before -> the share of forest lost is undefined (None), not 0."""
    inp, _ = _inputs()
    inp.before[0] = -1.0  # no forest anywhere in the first season
    inp.after[0] = -1.0
    res = area.run_area(_pipe(), inp, 2018, 2022, 30)
    assert res["stats"]["forest_before_ha"] == 0
    assert res["stats"]["loss_pct_of_forest"] is None


def test_run_area_beyond_hansen():
    inp, _ = _inputs(hansen_last_year=2025)
    res = area.run_area(_pipe(), inp, 2018, 2026, 30)
    assert res["stats"]["hansen"] is None and res["hansen_loss"] is None
    assert res["loss"].shape == (20, 20)


def test_no_clear_obs_raises():
    inp, _ = _inputs()
    inp.n_obs_after[:] = 0
    with pytest.raises(area.RequestError, match="no clear Sentinel-2 observations"):
        area.run_area(_pipe(), inp, 2018, 2022, 30)


def test_initialize_service_account(monkeypatch):
    import json

    import ee

    from deforest import gee

    key = json.dumps({"client_email": "sa@proj.iam.gserviceaccount.com", "project_id": "keyproj",
                      "private_key": "x"})
    seen = {}
    monkeypatch.setattr(ee, "ServiceAccountCredentials",
                        lambda email, key_data=None: seen.update(email=email, key=key_data) or "CRED")
    monkeypatch.setattr(ee, "Initialize", lambda credentials=None, project=None: seen.update(cred=credentials, project=project))
    monkeypatch.delenv("EE_PROJECT", raising=False)
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_KEY", key)
    assert gee.initialize() == "keyproj"
    assert seen == {"email": "sa@proj.iam.gserviceaccount.com", "key": key, "cred": "CRED", "project": "keyproj"}
    monkeypatch.setenv("EE_PROJECT", "envproj")
    assert gee.initialize() == "envproj" and seen["project"] == "envproj"
    # without a service account the cached user credentials must be used (no credentials kwarg)
    monkeypatch.delenv("EE_SERVICE_ACCOUNT_KEY")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    calls = []
    monkeypatch.setattr(ee, "Initialize", lambda **kw: calls.append(kw))
    gee.initialize()
    assert calls == [{"project": "envproj"}]


def test_initialize_strips_pasted_quotes(monkeypatch):
    """Values pasted into a hosting UI often carry quotes or a trailing newline."""
    import json

    import ee

    from deforest import gee

    key = json.dumps({"client_email": "sa@proj.iam.gserviceaccount.com", "project_id": "keyproj",
                      "private_key": "x"})
    seen = {}
    monkeypatch.setattr(ee, "ServiceAccountCredentials",
                        lambda email, key_data=None: seen.update(email=email, key=key_data) or "CRED")
    monkeypatch.setattr(ee, "Initialize", lambda credentials=None, project=None: seen.update(project=project))
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_KEY", f"  '{key}'\n")
    monkeypatch.setenv("EE_PROJECT", ' "envproj"\n')
    assert gee.initialize() == "envproj"
    assert seen == {"email": "sa@proj.iam.gserviceaccount.com", "key": key, "project": "envproj"}


def test_initialize_error_names_cause_without_emails(monkeypatch):
    import ee

    from deforest import gee

    def boom(**kw):
        raise ee.EEException("Caller sa@proj.iam.gserviceaccount.com lacks permission on project 'p'.")

    monkeypatch.delenv("EE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("EE_PROJECT", "p")
    monkeypatch.setattr(ee, "Initialize", boom)
    with pytest.raises(RuntimeError) as err:
        gee.initialize()
    msg = str(err.value)
    assert "lacks permission" in msg and "<email>" in msg and "gserviceaccount" not in msg
