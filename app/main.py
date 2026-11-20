"""Public API for the deforestation explorer: /health, /predict (upload) and /predict-area (job queue).

Run with:  uvicorn --factory app.main:create_app --host 0.0.0.0 --port 7860
`create_app()` loads the ONNX weights, so it fails loudly when they are missing; there is no
module-level `app`, which keeps `import app.main` free of side effects (tests, tooling).
No torch and no training modules are imported here.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import jobs, render
from deforest import area, metrics, predict
from deforest.config import load_config
from deforest.export import load_onnx_pipeline

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
CONFIG = ROOT / "configs" / "ariquemes.yaml"
CARD = Path(__file__).resolve().parent / "model_card.json"
LAYERS = ("forest_before", "forest_after", "loss", "hansen_loss")


@dataclass
class Settings:
    max_area_km2: float = 500.0
    max_queue: int = 5
    job_timeout_s: float = 1200.0
    cache_size: int = 20
    model_dir: str = "models"
    cache_dir: str = "cache"
    model_release: str = "models-v2.0"
    max_upload_mb: float = 200.0
    ort_threads: int = 2
    trusted_proxy_hops: int = 1  # proxies in front of the app; the client is this many entries from the right of X-Forwarded-For
    ee_retry_s: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        return cls(max_area_km2=float(e("MAX_AREA_KM2", cls.max_area_km2)),
                   max_queue=int(e("MAX_QUEUE", cls.max_queue)),
                   job_timeout_s=float(e("JOB_TIMEOUT_S", cls.job_timeout_s)),
                   cache_size=int(e("CACHE_SIZE", cls.cache_size)),
                   model_dir=e("MODEL_DIR", cls.model_dir), cache_dir=e("CACHE_DIR", cls.cache_dir),
                   model_release=e("MODEL_RELEASE", cls.model_release),
                   max_upload_mb=float(e("MAX_UPLOAD_MB", cls.max_upload_mb)),
                   ort_threads=int(e("ORT_THREADS", cls.ort_threads)),
                   trusted_proxy_hops=int(e("TRUSTED_PROXY_HOPS", cls.trusted_proxy_hops)),
                   ee_retry_s=float(e("EE_RETRY_S", cls.ee_retry_s)))


class AreaRequest(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    start: int
    end: int
    months: list[int] = Field([6, 9], min_length=2, max_length=2)


def cache_key(bbox, start: int, end: int, months, sha256s: dict) -> str:
    """Identical requests against the same weights share one result."""
    blob = json.dumps([[round(float(v), 4) for v in bbox], start, end, list(months), sorted(sha256s.items())])
    return hashlib.sha1(blob.encode()).hexdigest()


def client_ip(request: Request, hops: int = 1) -> str:
    """The client behind `hops` trusted proxies: each proxy appends the address it saw to X-Forwarded-For,
    so the real client is `hops` entries from the right (anything further left is client-supplied).
    Falls back to the socket peer when the header is absent or shorter than `hops`."""
    parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if hops >= 1 and len(parts) >= hops:
        return parts[-hops]
    return request.client.host if request.client else "unknown"


def _lazy_ee(retry_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> Callable[[], tuple[bool, str]]:
    """Earth Engine readiness: initialise on first use. Success is remembered for the process; a failure
    is remembered for `retry_s` seconds (its reason stays available to /health) and then retried."""
    lock, state = threading.Lock(), {"ok": None, "detail": "", "at": 0.0}

    def ready() -> tuple[bool, str]:
        with lock:
            if state["ok"] or (state["ok"] is False and clock() - state["at"] < retry_s):
                return state["ok"], state["detail"]
            try:
                from deforest import gee
                state.update(ok=True, detail=str(gee.initialize()))
            except Exception as e:  # noqa: BLE001 - reported by /health and as the 503 detail
                state.update(ok=False, detail=str(e) or type(e).__name__, at=clock())
            return state["ok"], state["detail"]
    return ready


class UploadLimit:
    """Reject a request body above `limit` bytes with 413: by Content-Length when sent, and by counting
    the bytes as they arrive (chunked uploads). Starlette spools uploads to disk while parsing, so this
    has to run before the endpoint, not inside it."""

    def __init__(self, app, limit: int, path: str = "/predict"):
        self.app, self.limit, self.path = app, limit, path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != self.path or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        too_big = HTTPException(413, f"upload is larger than the {self.limit / 2**20:g} MB limit")
        length = dict(scope["headers"]).get(b"content-length", b"0")
        if length.isdigit() and int(length) > self.limit:
            return await JSONResponse({"detail": too_big.detail}, 413)(scope, receive, send)
        seen = 0

        async def counted():
            nonlocal seen
            msg = await receive()
            seen += len(msg.get("body", b"")) if msg["type"] == "http.request" else 0
            if seen > self.limit:
                raise too_big
            return msg
        await self.app(scope, counted, send)


def _zip(files: dict[str, bytes]) -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="deforestation.zip"'})


def _mask_tif(mask: np.ndarray, profile: dict, description: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        return predict.write_mask(Path(d) / "m.tif", mask, profile, description).read_bytes()


def create_app(pipeline=None, fetch=area.fetch_inputs, ee_ready: Callable[[], tuple[bool, str]] | None = None,
               cache_dir: Path | None = None, settings: Settings | None = None, card: dict | None = None,
               clock: Callable[[], float] = time.monotonic) -> FastAPI:
    """Build the app. `pipeline` (with `card`, default app/model_card.json) is injected by tests; otherwise
    the ONNX pipeline is loaded from `settings.model_dir`."""
    settings = settings or Settings.from_env()
    if pipeline is None:
        try:
            pipeline, card = load_onnx_pipeline(settings.model_dir, threads=settings.ort_threads)
        except (FileNotFoundError, KeyError, OSError) as e:
            raise RuntimeError(f"cannot load the model from {settings.model_dir!r}: {e}. "
                               "Run `python scripts/fetch_weights.py --dest models` first.") from e
    card = card or json.loads(CARD.read_text())
    sha256s = {f["name"]: f["sha256"] for f in card["files"].values()}
    cfg = load_config(CONFIG)
    threshold = cfg.labels.treecover_threshold
    own_ee = ee_ready is None
    ee_ready = ee_ready or _lazy_ee(settings.ee_retry_s, clock)

    def run_area_job(payload: dict, out: Path, progress: jobs.Progress) -> dict:
        bbox, start, end, months = payload["bbox"], payload["start"], payload["end"], payload["months"]
        progress(f"downloading Sentinel-2 {start}…", 0.02)
        inp = fetch(bbox, start, end, months, cfg, out / "inputs", progress)
        shutil.rmtree(out / "inputs", ignore_errors=True)  # large composites are not kept
        progress("running model", 0.85)
        res = area.run_area(pipeline, inp, start, end, threshold)
        progress("rendering layers", 0.92)
        (out / "layers").mkdir()
        layers = {}
        pngs = {"before_rgb": render.rgb_png(inp.before, res["valid"], inp.transform, inp.crs),
                "after_rgb": render.rgb_png(inp.after, res["valid"], inp.transform, inp.crs)}
        for name in LAYERS:
            if res[name] is not None:
                pngs[name] = render.mask_png(res[name], inp.transform, inp.crs, render.LAYER_COLORS[name])
        for name, (png, bounds) in pngs.items():
            (out / "layers" / f"{name}.png").write_bytes(png)
            layers[name] = {"bounds": bounds}
        h, w = res["valid"].shape
        profile = {"crs": inp.crs, "transform": inp.transform, "width": w, "height": h}
        for name in LAYERS:
            if res[name] is not None:
                predict.write_mask(out / f"{name}.tif", res[name], profile, name)
        stats = {**res["stats"], "request": {"bbox": bbox, "start": start, "end": end, "months": months},
                 "crs": inp.crs, "hansen_asset": inp.hansen_asset}
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        return {"stats": stats, "layers": layers}

    queue = jobs.JobQueue(run_area_job, settings.max_queue, Path(cache_dir or settings.cache_dir),
                          settings.cache_size, settings.job_timeout_s)

    @asynccontextmanager
    async def lifespan(_app):
        if own_ee:  # try once at startup without blocking it; failure is just remembered
            threading.Thread(target=ee_ready, daemon=True).start()
        yield
        queue.shutdown()

    app = FastAPI(title="Deforestation explorer", lifespan=lifespan)
    app.add_middleware(UploadLimit, limit=int(settings.max_upload_mb * 2**20))
    model_busy = threading.Semaphore(1)  # one /predict model run at a time: refuse rather than queue

    @app.get("/health")
    def health():
        ok, detail = ee_ready()
        return {"status": "ok", "model": {"release": settings.model_release, "sha256s": sha256s},
                "earth_engine": ok, "earth_engine_detail": detail,
                "queue": {"active": queue.active(), "max": settings.max_queue},
                "years": [area.MIN_YEAR, area.last_year()], "max_area_km2": settings.max_area_km2}

    # ------------------------------------------------------------------ upload
    @app.post("/predict")
    def predict_upload(before: UploadFile = File(...), after: UploadFile | None = File(None)):
        if not model_busy.acquire(blocking=False):
            raise HTTPException(429, "busy with another upload, try again shortly")
        try:
            return _predict_upload(before, after)
        finally:
            model_busy.release()

    def _predict_upload(before: UploadFile, after: UploadFile | None):
        with tempfile.TemporaryDirectory() as d:
            arrays = {}
            for name, up in (("before", before), ("after", after)):
                if up is None:
                    continue
                path = Path(d) / f"{name}.tif"
                with open(path, "wb") as f:
                    shutil.copyfileobj(up.file, f)
                try:
                    with rasterio.open(path) as src:  # size check from the header, before reading any pixels
                        if src.crs is None or not src.crs.is_projected:
                            raise HTTPException(400, "rasters must be in a projected CRS (metres), e.g. UTM")
                        km2 = src.width * src.height * metrics.pixel_area_ha(src.transform) / 100
                    if km2 > settings.max_area_km2:
                        raise HTTPException(400, f"{name} covers {km2:.0f} km2, above the "
                                                 f"{settings.max_area_km2:g} km2 limit")
                    arrays[name] = predict.read_composite(path)
                except (ValueError, rasterio.errors.RasterioError) as e:
                    raise HTTPException(400, f"{name}: {e}") from e
        (b, prof), second = arrays["before"], arrays.get("after")
        crs = prof["crs"]
        if second is not None:
            a, aprof = second
            if a.shape != b.shape or aprof["transform"] != prof["transform"] or aprof["crs"] != crs:
                raise HTTPException(400, "before and after must be on the same grid (size, transform and CRS)")
        tf = prof["transform"]
        valid = (b != 0).any(axis=0)
        if second is None:
            forest = pipeline.forest(b) & valid
            stats = {"forest_ha": round(metrics.area_ha(forest, tf), 2),
                     "valid_ha": round(float(valid.sum()) * metrics.pixel_area_ha(tf), 2)}
            masks = {"forest_before": forest}
        else:
            valid &= (a != 0).any(axis=0)
            res = pipeline.run(b, a, tf, nodata_mask=~valid)
            stats = res["stats"]
            masks = {k: res[k] for k in ("forest_before", "forest_after", "loss")}
        files = {f"{k}.tif": _mask_tif(m, prof, k) for k, m in masks.items()}
        files["stats.json"] = json.dumps(stats, indent=2).encode()
        return _zip(files)

    # ------------------------------------------------------------------ area jobs
    @app.post("/predict-area", status_code=202)
    def predict_area(req: AreaRequest, request: Request):
        try:
            area.validate_request(req.bbox, req.start, req.end, tuple(req.months), settings.max_area_km2)
        except area.RequestError as e:
            raise HTTPException(422, str(e)) from e
        key = cache_key(req.bbox, req.start, req.end, req.months, sha256s)
        job = queue.find(key)
        if job is None:
            ok, detail = ee_ready()
            if not ok:
                raise HTTPException(503, detail)
            payload = {"bbox": req.bbox, "start": req.start, "end": req.end, "months": req.months}
            try:
                job = queue.submit(key, client_ip(request, settings.trusted_proxy_hops), payload)
            except jobs.Busy as e:
                raise HTTPException(429, str(e)) from e
        return {"job_id": job.id}

    def _job(job_id: str) -> jobs.Job:
        job = queue.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job (it may have expired from the cache)")
        return job

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = _job(job_id)
        body = {"status": job.status, "progress": job.progress, "message": job.message,
                "position": queue.position(job.id)}
        if job.status == "error":
            body["error"] = job.error
        if job.status == "done":
            r = job.result
            body["result"] = {
                "stats": r["stats"], "download": f"/jobs/{job.id}/download.zip",
                "layers": {n: {"url": f"/jobs/{job.id}/layers/{n}.png", "bounds": v["bounds"]}
                           for n, v in r["layers"].items()}}
        return body

    @app.get("/jobs/{job_id}/layers/{name}.png")
    def job_layer(job_id: str, name: str):
        job = _job(job_id)
        if job.status != "done" or name not in job.result["layers"]:
            raise HTTPException(404, f"no layer {name!r}")
        return FileResponse(queue.path(job) / "layers" / f"{name}.png", media_type="image/png",
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/jobs/{job_id}/download.zip")
    def job_download(job_id: str):
        job = _job(job_id)
        if job.status != "done":
            raise HTTPException(404, "job is not finished")
        out = queue.path(job)
        names = [f"{n}.tif" for n in LAYERS if (out / f"{n}.tif").exists()] + ["stats.json"]
        return _zip({n: (out / n).read_bytes() for n in names})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
