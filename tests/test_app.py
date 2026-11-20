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


def _client(tmp_path, fetch=None, ee=(True, "proj"), pipeline=None, clock=time.monotonic, **settings):
    app = main.create_app(pipeline=pipeline or _pipe(), fetch=fetch or _stub_fetch(), ee_ready=lambda: ee,
                          cache_dir=tmp_path / "cache", settings=main.Settings(**settings), clock=clock)
    return TestClient(app)


def _wait(client, job_id, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still {body['status']}")


def _tif(path, bands=6, size=64, fill=True, crs="EPSG:32720", res=10):
    arr = np.zeros((bands, size, size), np.uint16)
    if fill:
        arr[:, :, : size // 2] = 1500  # right half left as no data (all-zero)
    with rasterio.open(path, "w", driver="GTiff", count=bands, dtype="uint16", width=size, height=size,
                       crs=crs, transform=from_origin(500000, 8900000, res, res)) as dst:
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
    assert body["earth_engine"] is True and body["earth_engine_detail"] == "ready"  # no GCP project id
    assert body["years"] == [2016, END] and body["max_area_km2"] == 500
    assert body["model"]["release"] == "models-v2.0" and len(body["model"]["sha256s"]) == 2
    assert body["queue"]["active"] == 0 and body["queue"]["max"] == 5


def test_health_keeps_the_failure_reason(tmp_path):
    with _client(tmp_path, ee=(False, "no credentials")) as c:
        body = c.get("/health").json()
    assert body["earth_engine"] is False and body["earth_engine_detail"] == "no credentials"


def test_index_and_static(tmp_path):
    with _client(tmp_path) as c:
        page = c.get("/").text
        assert "<title>Deforestation explorer</title>" in page
        assert "Model trained only on Ariquemes, Rondônia (Brazil), 2018 vs 2022. " \
               "Accuracy elsewhere is unmeasured — compare with the Hansen layer." in page
        assert "at most ${MAX_SPAN} degree" in page  # C1: the same span rule as the server
        assert "start+1" in page and "seasonal" in page  # M1
        assert c.get("/static/index.html").status_code == 200
        examples = c.get("/static/examples.json").json()
        assert examples[0]["note"] == "training area" and len(examples) == 5
        assert not any(ch.isdigit() for e in examples[1:] for ch in e["note"].replace("2019", "").replace("2022", ""))
        assert all(set(e) == {"name", "bbox", "start", "end", "note"} for e in examples)


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


def test_predict_requires_10m_pixels(tmp_path):
    """C2: a 1 m raster is refused from its header (it would be 100x the pixels for the same area)."""
    with _client(tmp_path) as c:
        r = _upload(c, tmp_path, b={"res": 1}, after=False)
        assert r.status_code == 400 and "10 m" in r.json()["detail"] and "got 1" in r.json()["detail"]
        r = _upload(c, tmp_path, b={"res": 30}, a={"res": 30})
        assert r.status_code == 400 and "10 m" in r.json()["detail"]
        assert _upload(c, tmp_path, b={"res": 10.4}, a={"res": 10.4}).status_code == 200  # within 5 %


def test_predict_limits_pixel_count(tmp_path):
    """C2: the size check is on width*height, not on ground area."""
    with _client(tmp_path, max_area_km2=0.1) as c:  # 0.1 km2 at 10 m = 1000 px; the upload has 4096
        r = _upload(c, tmp_path, after=False)
        assert r.status_code == 400 and "4096 px" in r.json()["detail"] and "1000 px" in r.json()["detail"]
    with _client(tmp_path, max_area_km2=0.41) as c:  # 4100 px
        assert _upload(c, tmp_path, after=False).status_code == 200


def _vrt_of(tif: Path, path: Path) -> Path:
    bands = "".join(
        f'<VRTRasterBand dataType="UInt16" band="{i}"><SimpleSource><SourceFilename relativeToVRT="0">{tif}'
        f'</SourceFilename><SourceBand>{i}</SourceBand></SimpleSource></VRTRasterBand>' for i in range(1, 7))
    path.write_text('<VRTDataset rasterXSize="64" rasterYSize="64"><SRS>EPSG:32720</SRS>'
                    '<GeoTransform>500000,10,0,8900000,0,-10</GeoTransform>' + bands + '</VRTDataset>')
    return path


def test_predict_opens_uploads_as_geotiff_only(tmp_path):
    """I1: a VRT (which can reference /vsicurl/ or local files) renamed to .tif is refused."""
    from deforest import predict
    tif = _tif(tmp_path / "real.tif")
    vrt = _vrt_of(tif, tmp_path / "evil.tif")
    with rasterio.open(vrt) as src:  # sanity: without the driver restriction GDAL happily opens it
        assert src.count == 6
    with _client(tmp_path) as c, open(vrt, "rb") as f:
        r = c.post("/predict", files={"before": ("b.tif", f)})
    assert r.status_code == 400 and "before" in r.json()["detail"]
    with pytest.raises(rasterio.errors.RasterioIOError):
        predict.read_composite(vrt)
    assert predict.read_composite(tif)[0].shape == (6, 64, 64)


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
        for bbox in ([-70, -10.0005, -50, -10], [-80, 0, -60, 0.0002], [-180, 89.9, 180, 90]):  # C1
            r = c.post("/predict-area", json={**PAYLOAD, "bbox": bbox})
            assert r.status_code == 422 and "degree" in r.json()["detail"]


def test_predict_area_body_limit(tmp_path):
    """M7: /predict-area bodies are tiny JSON; anything above 64 KB is refused before parsing."""
    with _client(tmp_path) as c:
        big = json.dumps({**PAYLOAD, "pad": "x" * 70_000})
        r = c.post("/predict-area", content=big, headers={"Content-Type": "application/json"})
        assert r.status_code == 413 and "KB" in r.json()["detail"]

        def chunks():
            yield b'{"bbox": ['
            for _ in range(70):
                yield b"0," * 500
        r = c.post("/predict-area", content=chunks(), headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert c.post("/predict-area", json=PAYLOAD).status_code == 202


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


def _shifted(i):
    """The i-th distinct, valid request (the box slides 0.01 degree east each time)."""
    return {**PAYLOAD, "bbox": [ARIQUEMES[0] + 0.01 * i, ARIQUEMES[1], ARIQUEMES[2] + 0.01 * i, ARIQUEMES[3]]}


def test_daily_limit_per_ip(tmp_path):
    now = [1000.0]
    with _client(tmp_path, max_jobs_per_ip_per_day=2, clock=lambda: now[0]) as c:
        h = {"X-Forwarded-For": "1.1.1.1"}
        for i in range(2):
            r = c.post("/predict-area", json=_shifted(i), headers=h)
            assert r.status_code == 202
            _wait(c, r.json()["job_id"])
        r = c.post("/predict-area", json=_shifted(2), headers=h)
        assert r.status_code == 429 and "daily limit" in r.json()["detail"] and "README" in r.json()["detail"]
        # another address is unaffected, and so are cache hits of the limited address
        assert c.post("/predict-area", json=_shifted(2), headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 202
        assert c.post("/predict-area", json=_shifted(0), headers=h).status_code == 202
        now[0] += 24 * 3600 + 1  # rolling 24 h window
        assert c.post("/predict-area", json=_shifted(3), headers=h).status_code == 202


def test_daily_limit_global(tmp_path):
    with _client(tmp_path, max_jobs_per_day=2, max_jobs_per_ip_per_day=50) as c:
        for i in range(2):
            r = c.post("/predict-area", json=_shifted(i), headers={"X-Forwarded-For": f"9.9.9.{i}"})
            assert r.status_code == 202
            _wait(c, r.json()["job_id"])
        r = c.post("/predict-area", json=_shifted(2), headers={"X-Forwarded-For": "9.9.9.77"})
        assert r.status_code == 429 and "daily limit" in r.json()["detail"]


def test_cache_hits_and_in_flight_hits_do_not_count_towards_daily_limits(tmp_path):
    gate, started = threading.Event(), threading.Event()
    try:
        with _client(tmp_path, fetch=_stub_fetch(gate=gate, started=started),
                     max_jobs_per_ip_per_day=1, max_jobs_per_day=1) as c:
            h = {"X-Forwarded-For": "1.1.1.1"}
            first = c.post("/predict-area", json=PAYLOAD, headers=h)
            assert first.status_code == 202 and started.wait(10)
            for _ in range(3):  # in flight: the same job, never a 429
                assert c.post("/predict-area", json=PAYLOAD, headers=h).json()["job_id"] == first.json()["job_id"]
            gate.set()
            _wait(c, first.json()["job_id"])
            for _ in range(3):  # finished (cached): same
                assert c.post("/predict-area", json=PAYLOAD, headers=h).status_code == 202
            assert c.post("/predict-area", json=_shifted(1), headers=h).status_code == 429
    finally:
        gate.set()


def test_refused_submits_do_not_count_towards_daily_limits(tmp_path):
    gate, started = threading.Event(), threading.Event()
    try:
        with _client(tmp_path, fetch=_stub_fetch(gate=gate, started=started), max_queue=1,
                     max_jobs_per_day=2, max_jobs_per_ip_per_day=2) as c:
            assert c.post("/predict-area", json=PAYLOAD, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 202
            assert started.wait(10)
            for i in range(1, 4):  # queue full: refused, and not counted
                r = c.post("/predict-area", json=_shifted(i), headers={"X-Forwarded-For": "2.2.2.2"})
                assert r.status_code == 429 and "queue" in r.json()["detail"]
            gate.set()
            _wait(c, c.post("/predict-area", json=PAYLOAD, headers={"X-Forwarded-For": "1.1.1.1"}).json()["job_id"])
            assert c.post("/predict-area", json=_shifted(1), headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 202
    finally:
        gate.set()


def test_daily_limit_settings_from_env(monkeypatch):
    st = main.Settings()
    assert (st.max_jobs_per_ip_per_day, st.max_jobs_per_day) == (10, 100)
    monkeypatch.setenv("MAX_JOBS_PER_IP_PER_DAY", "3")
    monkeypatch.setenv("MAX_JOBS_PER_DAY", "7")
    st = main.Settings.from_env()
    assert (st.max_jobs_per_ip_per_day, st.max_jobs_per_day) == (3, 7)


def test_both_seasons_share_one_colour_stretch(tmp_path, monkeypatch):
    """M2: the app renders before and after with the same reference stretch."""
    from app import render
    mine, seen, base = {}, [], _stub_fetch()
    real = render.rgb_png

    def fetch(*args):
        inp = base(*args)
        mine.update(before=inp.before, after=inp.after)
        return inp

    def spy(img, valid, transform, crs, **kw):
        if img is mine.get("before") or img is mine.get("after"):  # ignore jobs leaked by earlier tests
            seen.append(kw.get("ref"))
        return real(img, valid, transform, crs, **kw)
    monkeypatch.setattr(render, "rgb_png", spy)
    with _client(tmp_path, fetch=fetch) as c:
        assert _wait(c, c.post("/predict-area", json=PAYLOAD).json()["job_id"])["status"] == "done"
    assert len(seen) == 2 and seen[0] is not None and seen[0] is seen[1]


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


def test_unexpected_errors_are_logged_with_traceback(tmp_path, caplog):
    """L1: only user errors (RequestError) and timeouts skip the traceback; a bare ValueError is a bug."""
    import logging
    q = _queue(tmp_path, lambda payload, out, progress: (_ for _ in ()).throw(payload))
    with caplog.at_level(logging.ERROR, logger="app.jobs"):
        for key, exc in (("a", area.RequestError("no scenes")), ("b", jobs.JobTimeout("slow")),
                         ("c", ValueError("operands could not be broadcast"))):
            job = q.submit(key, key, exc)
            assert job.finished.wait(10) and job.status == "error" and job.error
    q.shutdown()
    logged = [r for r in caplog.records if r.exc_info]
    assert len(logged) == 1 and "operands could not be broadcast" in str(logged[0].exc_info[1])


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


# ------------------------------------------------------------------ /predict limits
def test_predict_upload_too_large_by_content_length(tmp_path):
    with _client(tmp_path, max_upload_mb=0.01) as c:  # ~10 KB; the 64x64x6 uint16 tif is ~49 KB
        r = _upload(c, tmp_path)
        assert r.status_code == 413 and "KB" in r.json()["detail"]  # 0.01 MB = 10 KB
    with _client(tmp_path) as c:
        assert _upload(c, tmp_path).status_code == 200


def test_predict_upload_too_large_while_streaming(tmp_path):
    # chunked body: no Content-Length, so the limit is enforced by counting bytes as they arrive
    def body():
        yield b'--x\r\nContent-Disposition: form-data; name="before"; filename="b.tif"\r\n\r\n'
        for _ in range(40):
            yield b"0" * 1000
    with _client(tmp_path, max_upload_mb=0.01) as c:
        r = c.post("/predict", content=body(), headers={"Content-Type": "multipart/form-data; boundary=x"})
        assert r.status_code == 413


def test_predict_area_limit_on_raster_size(tmp_path):
    # 64x64 px at 10 m = 0.41 km2
    with _client(tmp_path, max_area_km2=0.1) as c:
        r = _upload(c, tmp_path)
        assert r.status_code == 400 and "km2" in r.json()["detail"]
    with _client(tmp_path, max_area_km2=1) as c:
        assert _upload(c, tmp_path).status_code == 200


def test_predict_busy_returns_429(tmp_path):
    gate, started = threading.Event(), threading.Event()

    def forest(b):
        started.set()
        assert gate.wait(10)
        return b[:, 0]
    pipe = _pipe()
    pipe.infer_forest = forest
    result = {}
    try:
        with _client(tmp_path, pipeline=pipe) as c:
            t = threading.Thread(target=lambda: result.update(r=_upload(c, tmp_path, after=False)))
            t.start()
            assert started.wait(10)
            r = c.post("/predict", files={"before": ("b.tif", b"x")})
            assert r.status_code == 429 and "busy" in r.json()["detail"]
            gate.set()
            t.join(10)
            assert result["r"].status_code == 200
            assert _upload(c, tmp_path, after=False).status_code == 200  # released afterwards
    finally:
        gate.set()


# ------------------------------------------------------------------ settings, client ip, EE retry
def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("ORT_THREADS", "3")
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    monkeypatch.setenv("MAX_QUEUE", "9")
    st = main.Settings.from_env()
    assert (st.ort_threads, st.max_upload_mb, st.trusted_proxy_hops, st.max_queue) == (3, 5, 2, 9)
    monkeypatch.delenv("ORT_THREADS")
    assert main.Settings.from_env().ort_threads == 2


def test_ort_threads_passed_to_loader(tmp_path, monkeypatch):
    seen = {}

    def fake_loader(model_dir, threads=None):
        seen["threads"] = threads
        return _pipe(), None
    monkeypatch.setattr(main, "load_onnx_pipeline", fake_loader)
    main.create_app(settings=main.Settings(ort_threads=3), cache_dir=tmp_path / "c")
    assert seen["threads"] == 3


class _Req:
    def __init__(self, xff=None, host="9.9.9.9"):
        self.headers = {} if xff is None else {"x-forwarded-for": xff}
        self.client = type("C", (), {"host": host})()


def test_client_ip_trusted_hops():
    assert main.client_ip(_Req("6.6.6.6, 1.1.1.1")) == "1.1.1.1"  # leftmost is client-supplied
    assert main.client_ip(_Req("6.6.6.6, 1.1.1.1, 10.0.0.1"), hops=2) == "1.1.1.1"
    assert main.client_ip(_Req(None, host="5.5.5.5")) == "5.5.5.5"
    assert main.client_ip(_Req("1.1.1.1"), hops=2) == "9.9.9.9"  # header shorter than the hop count


def test_spoofed_forwarded_for_does_not_change_identity(tmp_path):
    gate, started = threading.Event(), threading.Event()
    other = {**PAYLOAD, "bbox": [v + 0.01 for v in ARIQUEMES]}
    try:
        with _client(tmp_path, fetch=_stub_fetch(gate=gate, started=started), max_queue=5) as c:
            r = c.post("/predict-area", json=PAYLOAD, headers={"X-Forwarded-For": "6.6.6.1, 1.1.1.1"})
            assert r.status_code == 202 and started.wait(10)
            # same real client (last entry), different spoofed leftmost entry: still the same IP
            r = c.post("/predict-area", json=other, headers={"X-Forwarded-For": "6.6.6.2, 1.1.1.1"})
            assert r.status_code == 429 and "already" in r.json()["detail"]
            # a genuinely different client is let in
            r = c.post("/predict-area", json=other, headers={"X-Forwarded-For": "6.6.6.1, 2.2.2.2"})
            assert r.status_code == 202
            gate.set()
    finally:
        gate.set()


def test_ee_failure_is_retried_after_the_window(tmp_path, monkeypatch):
    from deforest import gee
    calls, now = [], [1000.0]

    def init():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("EE down")
        return "proj"
    monkeypatch.setattr(gee, "initialize", init)
    app = main.create_app(pipeline=_pipe(), fetch=_stub_fetch(), cache_dir=tmp_path / "c",
                          settings=main.Settings(ee_retry_s=60), clock=lambda: now[0])
    with TestClient(app) as c:
        body = c.get("/health").json()
        assert body["earth_engine"] is False and "EE down" in body["earth_engine_detail"]
        now[0] += 30  # inside the window: no new attempt, the reason is kept
        body = c.get("/health").json()
        assert body["earth_engine"] is False and "EE down" in body["earth_engine_detail"] and len(calls) == 1
        assert c.post("/predict-area", json=PAYLOAD).status_code == 503
        now[0] += 31  # window over: retry succeeds
        assert c.get("/health").json()["earth_engine"] is True and len(calls) == 2
        r = c.post("/predict-area", json=PAYLOAD)
        assert r.status_code == 202 and _wait(c, r.json()["job_id"])["status"] == "done"
        c.get("/health")
        assert len(calls) == 2  # success is cached


def test_ee_import_failure_is_reported(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "deforest" and "gee" in (fromlist or ()):
            raise ImportError("no earthengine")
        return real(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert main._lazy_ee()() == (False, "no earthengine")


def test_default_job_timeout_allows_a_slow_first_download():
    assert main.Settings().job_timeout_s == 1200.0
