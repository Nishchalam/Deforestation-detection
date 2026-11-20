import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app import jobs, main
from deforest import area
from test_area import ARIQUEMES, _inputs, _pipe

END = area.last_year()
PAYLOAD = {"bbox": ARIQUEMES, "start": 2018, "end": 2022}
ALL_LAYERS = ["before_rgb", "after_rgb", "forest_before", "forest_after", "loss", "hansen_loss"]


def _stub_fetch(hansen_last_year=2025, gate=None, started=None, error=None):
    def fetch(bbox, start, end, months, cfg, out_dir, progress):
        if started is not None:
            started.set()
        progress(f"Sentinel-2 {start} downloaded", 0.1)
        if gate is not None:
            assert gate.wait(10)
        if error:
            raise RuntimeError(error)
        inp = _inputs(hansen_last_year)[0]
        # the stub nets only read before[0] and after[:2]; give the other bands real-looking reflectance
        # so the RGB percentile stretch has data
        for arr, first in ((inp.before, 1), (inp.after, 2)):
            arr[first:] = 1000 + np.arange(arr[first:].size, dtype=np.float32).reshape(arr[first:].shape)
        return inp
    return fetch


def _client(tmp_path, fetch=None, ee=(True, "proj"), **settings):
    app = main.create_app(pipeline=_pipe(), fetch=fetch or _stub_fetch(), ee_ready=lambda: ee,
                          cache_dir=tmp_path / "cache", settings=main.Settings(**settings))
    return TestClient(app)


def _wait(client, job_id, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still {body['status']}")


def _tif(path, bands=6, size=64, fill=True, crs="EPSG:32720"):
    arr = np.zeros((bands, size, size), np.uint16)
    if fill:
        arr[:, :, : size // 2] = 1500  # right half left as no data (all-zero)
    with rasterio.open(path, "w", driver="GTiff", count=bands, dtype="uint16", width=size, height=size,
                       crs=crs, transform=from_origin(500000, 8900000, 10, 10)) as dst:
        dst.write(arr)
    return path


def _upload(client, tmp_path, before=True, after=True, **kw):
    files = {}
    if before:
        files["before"] = ("b.tif", open(_tif(tmp_path / "b.tif", **kw.get("b", {})), "rb"))
    if after:
        files["after"] = ("a.tif", open(_tif(tmp_path / "a.tif", **kw.get("a", {})), "rb"))
    try:
        return client.post("/predict", files=files)
    finally:
        for _, f in files.values():
            f.close()


# ------------------------------------------------------------------ basic routes
def test_health(tmp_path):
    with _client(tmp_path) as c:
        body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["earth_engine"] is True and body["earth_engine_detail"] == "proj"
    assert body["years"] == [2016, END] and body["max_area_km2"] == 500
    assert body["model"]["release"] == "models-v2.0" and len(body["model"]["sha256s"]) == 2
    assert body["queue"]["active"] == 0 and body["queue"]["max"] == 5


def test_index_and_static(tmp_path):
    with _client(tmp_path) as c:
        assert "Deforestation explorer" in c.get("/").text
        assert c.get("/static/index.html").status_code == 200


def test_app_does_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{root / 'src'}{os.pathsep}{root}"}
    code = ("import sys, app.main; bad = [m for m in ('torch', 'deforest.experiment', 'deforest.training',"
            " 'deforest.models', 'deforest.data') if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_missing_weights_fail_loudly(tmp_path):
    with pytest.raises(Exception, match="fetch_weights"):
        main.create_app(settings=main.Settings(model_dir=str(tmp_path / "nomodel")), cache_dir=tmp_path / "c")


# ------------------------------------------------------------------ /predict
def test_predict_two_images(tmp_path):
    with _client(tmp_path) as c:
        r = _upload(c, tmp_path)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(z.namelist()) == ["forest_after.tif", "forest_before.tif", "loss.tif", "stats.json"]
    stats = json.loads(z.read("stats.json"))
    assert "loss_ha" in stats
    assert stats["valid_ha"] == pytest.approx(32 * 64 * 0.01)  # only the filled half is valid


def test_predict_before_only(tmp_path):
    with _client(tmp_path) as c:
        r = _upload(c, tmp_path, after=False)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(z.namelist()) == ["forest_before.tif", "stats.json"]
    stats = json.loads(z.read("stats.json"))
    assert set(stats) >= {"forest_ha", "valid_ha"} and stats["valid_ha"] == pytest.approx(32 * 64 * 0.01)
    assert stats["forest_ha"] == stats["valid_ha"]  # stub forest = band 0 > 0 on every valid pixel
    with rasterio.open(io.BytesIO(z.read("forest_before.tif"))) as src:
        assert src.read(1).shape == (64, 64) and src.crs.to_epsg() == 32720


def test_predict_errors(tmp_path):
    with _client(tmp_path) as c:
        r = _upload(c, tmp_path, b={"bands": 3})
        assert r.status_code == 400 and "6 bands" in r.json()["detail"]
        r = _upload(c, tmp_path, a={"size": 32})
        assert r.status_code == 400 and "grid" in r.json()["detail"]
        r = _upload(c, tmp_path, b={"crs": "EPSG:4326"}, after=False)
        assert r.status_code == 400 and "projected" in r.json()["detail"]
        r = c.post("/predict", files={"before": ("x.tif", b"not a tiff")})
        assert r.status_code == 400
        assert c.post("/predict").status_code == 422


# ------------------------------------------------------------------ /predict-area
def test_area_request_validation(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/predict-area", json={**PAYLOAD, "bbox": [-63.0, -10.5, -62.0, -9.5]})
        assert r.status_code == 422 and "500" in r.json()["detail"]
        assert c.post("/predict-area", json={**PAYLOAD, "bbox": [1, 2, 3]}).status_code == 422
        assert c.post("/predict-area", json={**PAYLOAD, "months": [6]}).status_code == 422
        assert c.post("/predict-area", json={**PAYLOAD, "start": "x"}).status_code == 422
        r = c.post("/predict-area", json={**PAYLOAD, "end": 2015})
        assert r.status_code == 422 and "before" in r.json()["detail"]


def test_area_job_end_to_end(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/predict-area", json=PAYLOAD)
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        body = _wait(c, job_id)
        assert body["status"] == "done" and body["progress"] == 1.0 and body["message"] == "done"
        res = body["result"]
        assert list(res["layers"]) == ALL_LAYERS
        assert res["stats"]["hansen"]["asset"] and res["stats"]["loss_ha"] >= 0
        assert res["stats"]["request"] == {"bbox": ARIQUEMES, "start": 2018, "end": 2022, "months": [6, 9]}
        assert res["stats"]["crs"] == "EPSG:32720"
        (s, w), (n, e) = res["layers"]["loss"]["bounds"]
        assert s < n and w < e
        png = c.get(res["layers"]["loss"]["url"])
        assert png.status_code == 200 and png.headers["content-type"] == "image/png"
        assert png.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert c.get(f"/jobs/{job_id}/layers/nope.png").status_code == 404
        z = zipfile.ZipFile(io.BytesIO(c.get(res["download"]).content))
        assert sorted(z.namelist()) == ["forest_after.tif", "forest_before.tif", "hansen_loss.tif", "loss.tif", "stats.json"]
        stats = json.loads(z.read("stats.json"))
        assert stats["request"]["bbox"] == ARIQUEMES and stats["crs"] == "EPSG:32720" and stats["hansen_asset"]
        # the S2 composites are not kept on disk
        assert not list((tmp_path / "cache").rglob("s2_*.tif"))
        assert c.get("/jobs/unknown").status_code == 404


def test_same_payload_returns_same_job(tmp_path):
    with _client(tmp_path) as c:
        a = c.post("/predict-area", json=PAYLOAD).json()["job_id"]
        _wait(c, a)
        near = {**PAYLOAD, "bbox": [v + 1e-6 for v in ARIQUEMES]}  # same after rounding to 4 decimals
        assert c.post("/predict-area", json=near).json()["job_id"] == a
        other = {**PAYLOAD, "months": [7, 9]}
        assert c.post("/predict-area", json=other).json()["job_id"] != a


def test_cache_survives_restart(tmp_path):
    with _client(tmp_path) as c:
        _wait(c, c.post("/predict-area", json=PAYLOAD).json()["job_id"])

    def boom(*a, **k):
        raise AssertionError("must be served from the cache")
    with _client(tmp_path, fetch=boom) as c:
        job_id = c.post("/predict-area", json=PAYLOAD).json()["job_id"]
        assert _wait(c, job_id)["status"] == "done"


def test_queue_full_and_per_ip_limits(tmp_path):
    gate, started = threading.Event(), threading.Event()
    other = {**PAYLOAD, "bbox": [v + 0.01 for v in ARIQUEMES]}
    try:
        with _client(tmp_path, fetch=_stub_fetch(gate=gate, started=started), max_queue=1) as c:
            first = c.post("/predict-area", json=PAYLOAD, headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"})
            assert first.status_code == 202
            assert started.wait(10)
            body = c.get(f"/jobs/{first.json()['job_id']}").json()
            assert body["status"] == "running" and body["position"] == 0
            r = c.post("/predict-area", json=other, headers={"X-Forwarded-For": "2.2.2.2"})
            assert r.status_code == 429 and "queue" in r.json()["detail"]
            # same request while in flight: the existing job, even from the busy IP or a full queue
            same = c.post("/predict-area", json=PAYLOAD, headers={"X-Forwarded-For": "1.1.1.1"})
            assert same.status_code == 202 and same.json()["job_id"] == first.json()["job_id"]
            gate.set()
            assert _wait(c, first.json()["job_id"])["status"] == "done"
    finally:
        gate.set()


def test_per_ip_limit(tmp_path):
    gate, started = threading.Event(), threading.Event()
    other = {**PAYLOAD, "bbox": [v + 0.01 for v in ARIQUEMES]}
    try:
        with _client(tmp_path, fetch=_stub_fetch(gate=gate, started=started), max_queue=5) as c:
            h = {"X-Forwarded-For": "1.1.1.1"}
            assert c.post("/predict-area", json=PAYLOAD, headers=h).status_code == 202
            assert started.wait(10)
            r = c.post("/predict-area", json=other, headers=h)
            assert r.status_code == 429 and "already" in r.json()["detail"]
            second = c.post("/predict-area", json=other, headers={"X-Forwarded-For": "2.2.2.2"})
            assert second.status_code == 202
            assert c.get(f"/jobs/{second.json()['job_id']}").json()["position"] == 1
            gate.set()
            assert _wait(c, second.json()["job_id"])["status"] == "done"
            assert c.post("/predict-area", json={**other, "start": 2019}, headers=h).status_code == 202
    finally:
        gate.set()


def test_earth_engine_unavailable(tmp_path):
    with _client(tmp_path, ee=(False, "no credentials")) as c:
        r = c.post("/predict-area", json=PAYLOAD)
        assert r.status_code == 503 and r.json()["detail"] == "no credentials"
        assert c.get("/health").json()["earth_engine"] is False
        assert _upload(c, tmp_path).status_code == 200


def test_default_ee_ready_initialises_once(tmp_path, monkeypatch):
    from deforest import gee
    calls = []

    def init():
        calls.append(1)
        raise RuntimeError("no credentials here")
    monkeypatch.setattr(gee, "initialize", init)
    app = main.create_app(pipeline=_pipe(), fetch=_stub_fetch(), cache_dir=tmp_path / "c", settings=main.Settings())
    with TestClient(app) as c:
        for _ in range(3):
            body = c.get("/health").json()
        r = c.post("/predict-area", json=PAYLOAD)
    assert body["earth_engine"] is False and "no credentials here" in body["earth_engine_detail"]
    assert r.status_code == 503
    assert len(calls) == 1


def test_fetch_error_surfaces(tmp_path):
    with _client(tmp_path, fetch=_stub_fetch(error="EE quota")) as c:
        first = c.post("/predict-area", json=PAYLOAD).json()["job_id"]
        body = _wait(c, first)
        assert body["status"] == "error" and "EE quota" in body["error"] and "result" not in body
        # failures are not cached: the same request is retried
        again = c.post("/predict-area", json=PAYLOAD).json()["job_id"]
        assert again != first and _wait(c, again)["status"] == "error"


def test_no_hansen_layer_beyond_hansen_year(tmp_path):
    with _client(tmp_path, fetch=_stub_fetch(hansen_last_year=END - 1)) as c:
        body = _wait(c, c.post("/predict-area", json={**PAYLOAD, "end": END}).json()["job_id"])
        assert body["status"] == "done"
        assert list(body["result"]["layers"]) == ALL_LAYERS[:-1]
        assert body["result"]["stats"]["hansen"] is None
        z = zipfile.ZipFile(io.BytesIO(c.get(body["result"]["download"]).content))
        assert "hansen_loss.tif" not in z.namelist()


# ------------------------------------------------------------------ JobQueue
def _queue(tmp_path, run, **kw):
    kw = {"max_queue": 5, "cache_size": 20, **kw}
    return jobs.JobQueue(run, cache_dir=tmp_path / "q", **kw)


def test_job_timeout(tmp_path):
    def run(payload, out_dir, progress):
        progress("stage 1", 0.1)
        time.sleep(0.1)
        progress("stage 2", 0.5)
        return {}
    q = _queue(tmp_path, run, timeout_s=0.05)
    job = q.submit("k", "ip", None)
    assert job.finished.wait(10)
    assert job.status == "error" and "timed out" in job.error
    assert not (tmp_path / "q" / "k").exists()
    q.shutdown()


def test_cache_eviction_is_lru(tmp_path):
    def run(payload, out_dir, progress):
        (out_dir / "x.txt").write_text(payload)
        return {"p": payload}
    q = _queue(tmp_path, run, cache_size=2)
    for key in ("a", "b"):
        assert q.submit(key, "ip", key).finished.wait(10)
        time.sleep(0.03)
    assert q.find("a").status == "done"  # a cache hit refreshes a
    time.sleep(0.03)
    assert q.submit("c", "ip", "c").finished.wait(10)
    q.shutdown()
    assert sorted(p.name for p in (tmp_path / "q").iterdir()) == ["a", "c"]
    assert q.find("b") is None and q.find("a").result == {"p": "a"}
