# Public Deforestation Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public website + API: pick any area ≤ 500 km² and two years, and get forest masks,
a loss map, stats and an agreement score against Hansen, served from one CPU Docker image on HF Spaces.

**Architecture:**
- **`deforest.area`:** the Earth Engine fetch + inference + Hansen comparison for a bbox.
  It is shared by `scripts/predict.py` and the app.
- **`app/`:** FastAPI wraps `deforest.area` in a single-worker job queue with a disk cache,
  renders map overlays as EPSG:3857 PNGs, and serves one Leaflet page.
- **Inference:** ONNX Runtime; the weights come from release `models-v2.0`.

**Tech Stack:** FastAPI, uvicorn, onnxruntime, rasterio, earthengine-api, Pillow,
Leaflet 1.9.4 + leaflet-draw 1.0.4 (cdnjs), Docker (python:3.11-slim), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-20-public-explorer-design.md`

## Global Constraints
- **No synthetic fallbacks.** Earth Engine or data failures surface as errors with the
  reason. Tests may use hand-built arrays and stub pipelines; nothing user-facing may.
- `MAX_AREA_KM2=500`, `MAX_QUEUE=5`, one active job per client IP, job timeout 600 s,
  cache 20 results. All of these are env-overridable.
- **Years:** `2016 ≤ start < end ≤ LAST_YEAR`. `LAST_YEAR` is the current year if its
  October has started, else the previous year. **Months:** `1 ≤ m1 ≤ m2 ≤ 12`, with no
  wrap across new year. The default is Jun–Sep.
- **Hansen layer + agreement:** only when `end ≤` the last year of the Hansen asset
  (2025 for v1_13). The label rule is the same as training (`treecover_threshold` from
  `configs/ariquemes.yaml`).
- **Earth Engine credentials:** `EE_SERVICE_ACCOUNT_KEY` (JSON text) or
  `GOOGLE_APPLICATION_CREDENTIALS` (path), project from `EE_PROJECT`. Otherwise use the
  cached user credentials.
- **The app image has no torch.** The weights are fetched at container start and checked
  against the sha256 in `app/model_card.json`. The image runs as uid 1000 on
  port `${PORT:-7860}`.
- **Banner text (exact):** "Model trained only on Ariquemes, Rondônia (Brazil), 2018 vs
  2022. Accuracy elsewhere is unmeasured — compare with the Hansen layer."
- Claude creates nothing on Google Cloud or Hugging Face.

## Review Focus
1. **Area with persistent cloud** (a clear-obs count of 0 for one year): the job fails
   with "no clear Sentinel-2 observations for <year> in <months>", not an all-nodata map.
   Test in Task 1.
2. **Box straddling a UTM zone edge or the equator:** the grid uses the zone and
   hemisphere of the box centre. Area is computed on the ellipsoid, not in degrees.
   Test in Task 1.
3. **Same request submitted twice while the first is queued/running:** it returns the
   same job id (no duplicate EE downloads). Test in Task 3.
4. **End year beyond Hansen:** the result has `hansen: null`, the layer list omits
   `hansen_loss`, and the UI hides the agreement row. Test in Tasks 1 and 3.
5. **Earth Engine not configured** on the host: `/health` reports `"earth_engine": false`
   and `/predict-area` returns 503 with the reason, while `/predict` uploads still work.
   Test in Task 3.

---

### Task 1: `deforest.area`: request validation, EE fetch, area run with Hansen comparison

**Files:**
- Create: `src/deforest/area.py`
- Modify: `src/deforest/gee.py` (`initialize` gains the service-account path)
- Modify: `scripts/predict.py` (use `area.utm_crs` and `area.fetch_inputs`; drop the local
  `download`/`utm_crs`)
- Test: `tests/test_area.py`

**Interfaces:**
- Consumes:
  - `gee.make_grid`, `grid_geometry`, `season_dates`, `s2_composite`, `hansen_latest`,
    `hansen_image`, `download_image`
  - `labels.forest_mask`, `labels.loss_mask`
  - `metrics.confusion`, `scores`, `area_ha`
  - `predict.Pipeline.run`
- Produces:
  - `class RequestError(ValueError)`
  - `utm_crs(bbox) -> str`
  - `bbox_area_km2(bbox) -> float`
  - `last_year(today: date | None = None) -> int`
  - `validate_request(bbox, start, end, months=(6, 9), max_km2=500.0, today=None) -> None`
    (raises `RequestError` with a user-readable message)
  - `@dataclass Inputs(before, after, n_obs_before, n_obs_after, treecover, lossyear,
    datamask, transform, crs, hansen_asset, hansen_last_year)`
  - `fetch_inputs(bbox, start, end, months, cfg, out_dir: Path, progress: Callable[[str, float], None]) -> Inputs`
    (writes `s2_<y>.tif` and `hansen.tif` into `out_dir`)
  - `run_area(pipe, inp: Inputs, start, end, threshold: float) -> dict` with keys
    `forest_before`, `forest_after`, `loss`, `valid`, `hansen_loss` (or None), `stats`
  - `stats` contains the Pipeline stats plus:
    - `loss_pct_of_forest`
    - `clear_obs_median: {before, after}`
    - `hansen: {asset, loss_ha, agreement: {iou, f1, precision, recall}} | None`

- [ ] **Step 1: Write failing tests**
  - `test_utm_crs`: `[-63,-10.5,-62.8,-10.3]` → `EPSG:32720`; `[12.0,0.1,12.2,0.3]` →
    `EPSG:32633`; a box centred at lon −66.1 (zone 19) → `EPSG:32719`.
  - `test_area_km2`: the Ariquemes bbox is 400–520 km².
  - `test_validate`. Each of these raises `RequestError`, matched on a message substring:
    - start 2015 (`"2016"`)
    - start ≥ end
    - end > `last_year(date(2026,11,20))` (= 2026)
    - months (9,6)
    - a 1°×1° box (`"500"`)

    The Ariquemes 2018–2022 request passes.
  - `test_last_year`: `date(2026,9,30)` → 2025; `date(2026,10,1)` → 2026.
  - `test_run_area_hansen`. Inputs:
    - Stub pipeline: forest = band0 > 0, change prob from a fixed array, threshold 0.7,
      MMU 0.
    - A 20×20 grid with hand-built `treecover`/`lossyear`.

    Assert:
    - `stats.hansen.loss_ha` equals the count of hand-built loss pixels × 0.01.
    - The agreement F1 equals sklearn's `f1_score` on the same masks.
    - `n_obs == 0` pixels are excluded from `valid`.
  - `test_run_area_beyond_hansen`: with `hansen_last_year=2025` and end 2026,
    `stats.hansen is None` and `hansen_loss is None`.
  - `test_no_clear_obs_raises`: `run_area` with `n_obs_after` all 0 raises `RequestError`
    containing `"no clear Sentinel-2 observations"`.
- [ ] **Step 2: Run `pytest tests/test_area.py -q`.** Expect an ImportError.
- [ ] **Step 3: Implement `area.py`.**
  - `bbox_area_km2`: use `pyproj.Geod(ellps="WGS84").geometry_area_perimeter` on the box
    polygon.
  - `fetch_inputs`:
    - `gee.season_dates(y, [f"{m1:02d}-01", f"{m2+1:02d}-01" or "12-31"])`; for `m2 == 12`
      the end is `f"{y+1}-01-01"`.
    - Reuse `cfg.sentinel2` for collection, bands and the s2cloudless params.
    - Download Hansen bands `treecover2000`, `lossyear`, `datamask` with
      `gee.hansen_image(gee.hansen_latest())` on the same grid.
    - `hansen_last_year = 2000 + int(asset year digits) `, parsed from the
      `global_forest_change_YYYY_v…` asset name.
    - Call `progress` after each download (fractions 0.1, 0.45, 0.8).
  - `gee.initialize`: if `EE_SERVICE_ACCOUNT_KEY` is set, use
    `ee.ServiceAccountCredentials(email, key_data=...)`, with `email` taken from the JSON's
    `client_email`. Else, if `GOOGLE_APPLICATION_CREDENTIALS` is set, read that file the
    same way. Project: `EE_PROJECT`, else the key's `project_id`.
- [ ] **Step 4: Point `scripts/predict.py` at `area`.** Keep the CLI unchanged; add
  `--months M1 M2`; write `hansen_loss.tif` when available.
- [ ] **Step 5: Verify.**
  - `pytest -q`: all pass.
  - On the workstation, `python scripts/predict.py --bbox -63.0 -10.5 -62.8 -10.3 --years 2018 2022 --out outputs/predict/ariquemes`.
    Expect loss_ha within 1 ha of the earlier 1102.52, and `hansen.loss_ha` within 1% of
    1203.95 (labels: 120,395 px).
- [ ] **Step 6: Commit** "Add deforest.area: request validation, EE fetch and Hansen agreement for any bbox".

### Task 2: `app/render.py`: web-map overlays

**Files:** Create `app/__init__.py`, `app/render.py`; Test `tests/test_render.py`

**Interfaces:**
- Produces:
  - `rgb_png(img6: np.ndarray, valid, transform, crs) -> tuple[bytes, list[list[float]]]`
    (bounds `[[south, west], [north, east]]`)
  - `mask_png(mask, transform, crs, color: tuple[int, int, int], alpha=170) -> tuple[bytes, bounds]`
  - `LAYER_COLORS = {"forest_before": (34,139,34), "forest_after": (34,139,34), "loss": (230,30,30), "hansen_loss": (255,200,0)}`

- [ ] **Step 1: Failing tests.**
  - The returned PNG opens with Pillow as RGBA.
  - A mask that is all False → every alpha is 0.
  - The bounds contain the WGS84 bounds of the source (within 1e-4°) for a 100×100 UTM
    20S raster at 10 m.
  - RGB: nodata pixels have alpha 0.
- [ ] **Step 2: Implement.**
  - Reproject to EPSG:3857 with `rasterio.warp.calculate_default_transform` + `reproject`:
    nearest for masks, bilinear for RGB.
  - Get the bounds via `transform_bounds(3857 → 4326)`.
  - RGB stretch: use `viz.rgb` (B4, B3, B2) with a 2–98 percentile stretch computed on
    valid pixels.
- [ ] **Step 3: `pytest tests/test_render.py -q`** passes. **Commit** "Add map overlay rendering (UTM → Web Mercator PNG + bounds)".

### Task 3: `app/main.py`: API, job queue, cache

**Files:**
- Create: `app/main.py`, `app/jobs.py`, `requirements-app.txt`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `area.*` (Task 1), `render.*` (Task 2), `export.load_onnx_pipeline`,
  `predict.read_composite`, `write_mask`.
- Produces: `create_app(pipeline=None, fetch=area.fetch_inputs, ee_ready: Callable[[], tuple[bool, str]] = ..., cache_dir=Path, settings: Settings) -> FastAPI`
  (tests inject a stub pipeline/fetch).
  - `Settings`: dataclass, read from env.
  - `jobs.JobQueue(run: Callable, max_queue, cache_dir, cache_size)` with:
    - `submit(key, ip, payload) -> Job`
    - `get(id) -> Job | None`
    - `position(id) -> int`
  - `Job`: `id`, `status` (queued/running/done/error), `progress`, `message`, `result`,
    `error`.
- Endpoints:
  - `GET /health`: `{status, model: {release, sha256s}, earth_engine: bool, earth_engine_detail, queue}`
  - `POST /predict`: multipart `before` (required), `after` (optional).
    - Returns `application/zip` with `forest_before.tif` (+ `forest_after.tif`,
      `loss.tif`) and `stats.json`.
    - Errors: 400 on bad band count or grid mismatch.
  - `POST /predict-area`: JSON `{bbox: [w,s,e,n], start: int, end: int, months: [m1,m2]}`.
    - Returns 202 `{job_id}`.
    - Errors: 422 on `RequestError` (message as `detail`), 429 when the queue is full or
      the IP already has an active job, 503 when EE is not ready.
  - `GET /jobs/{id}`: `{status, progress, message, position, result?, error?}`.
    - `result = {stats, layers: {name: {url, bounds}}, download: url}`.
  - `GET /jobs/{id}/layers/{name}.png`, `GET /jobs/{id}/download.zip`.
  - `GET /`: `app/static/index.html`; mount `/static`.
- Cache key: `sha1(json of round(bbox, 4), start, end, months, model sha256s)`. A
  completed key returns a job already `done`. An in-flight key returns the existing job.

- [ ] **Step 1: Failing tests** (TestClient; stub pipeline as in Task 1; a stub fetch
  returning hand-built `Inputs`):
  - `/health` returns 200 with the `earth_engine` field.
  - `/predict` with two 6-band 64×64 GeoTIFFs written in `tmp_path` → a zip with 4
    members; `stats.json` has `loss_ha`.
  - With one file → a zip with `forest_before.tif` + `stats.json` only.
  - A 3-band file → 400.
  - `/predict-area` with a 1°×1° box → 422, and the detail mentions `500`.
  - A valid request → 202; polling reaches `done` within 10 s; the result has the layers
    `before_rgb`, `after_rgb`, `forest_before`, `forest_after`, `loss`, `hansen_loss`;
    the PNG URL returns `image/png`.
  - The same payload again → the same `job_id` (Review Focus 3).
  - With `MAX_QUEUE=1` and a blocking stub fetch, a second distinct request from another
    IP → 429; a second request from the same IP → 429.
  - `ee_ready` returning `(False, "no credentials")` → `/predict-area` 503 with detail
    `"no credentials"`, and `/predict` still 200 (Review Focus 5).
  - A stub fetch raising `RuntimeError("EE quota")` → the job status is `error` and
    `error` contains `"EE quota"`.
  - With an end year beyond the stub's `hansen_last_year`, the layers lack `hansen_loss`.
- [ ] **Step 2: Run them.** They fail.
- [ ] **Step 3: Implement `jobs.py` and `main.py`.**
  - `JobQueue` uses `ThreadPoolExecutor(1)`.
  - Each job writes into `cache_dir/<key>/`: the inputs, the mask tifs, PNGs,
    `stats.json` and `result.json`.
  - LRU eviction is by mtime past `cache_size`.
  - A job timeout is checked between stages; on expiry, error "timed out".
  - The client IP comes from the first `X-Forwarded-For` entry, else `request.client.host`
    (Spaces sits behind a proxy).
  - The pipeline is loaded at startup from `MODEL_DIR` (default `models/`) with
    `threads=os.cpu_count()`.
- [ ] **Step 4: `pytest -q`** passes. **Commit** "Add FastAPI app: /health, /predict, /predict-area job queue with cache".

### Task 4: `app/static/index.html`: the explorer page

**Files:** Create `app/static/index.html`, `app/static/examples.json`

**Interfaces:** Consumes the Task 3 endpoints; `examples.json` is
`[{name, bbox, start, end, note}]`.

- [ ] **Step 1: Choose example areas.** Pick 4 boxes of about 20×20 km besides Ariquemes
  (note: "training area"), with the most Hansen loss from 2019–2022 in known clearing
  frontiers: Pará (BR-163 / Novo Progresso), Amazonas (Lábrea), Acre, and one outside
  Brazil (Bolivia or Paraguay Chaco). Check with a one-off EE `reduceRegion` sum of Hansen
  loss in each, and record that number in the note.
- [ ] **Step 2: Write the page.**
  - Map and libraries:
    - Leaflet 1.9.4 + leaflet-draw 1.0.4 from cdnjs.
    - An Esri World Imagery basemap with attribution.
    - A rectangle draw tool with a live km² readout (red above 500).
  - Controls and results:
    - An example dropdown that fits the map and fills the form.
    - Start/end selects populated from `/health` (`years: [2016, last]`); month selects.
    - A Run button that POSTs to `/predict-area`, then polls `/jobs/{id}` every 3 s,
      showing the queue position and progress message.
    - The results stats table: ha with thousands separators, and an agreement row that is
      hidden when `hansen` is null.
    - Layer checkboxes add/remove `L.imageOverlay(url, bounds)`; loss (model) is on by
      default.
    - The download link.
  - Messages:
    - The permanent banner text from Global Constraints.
    - Error messages from the server are shown verbatim.
  - Layout:
    - Mobile layout: map on top, panel below, at < 800 px.
    - Light/dark colour tokens on `:root`.
- [ ] **Step 3: Verify on the workstation.**
  - Run `uvicorn app.main:app --port 7860` with `MODEL_DIR` pointing at the fetched
    weights.
  - Run the Ariquemes example through `curl` (POST + poll to done). The stats match the
    Task 1 Step 5 run.
  - Check that `curl /` returns the page and that the PNG layer URLs return 200.
  - The user views the page via an ssh tunnel (`ssh -L 7860:localhost:7860`).
- [ ] **Step 4: Commit** "Add explorer web page with example areas".

### Task 5: Docker image, CI, deployment docs

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `app/entrypoint.sh`, `.github/workflows/ci.yml`,
  `docs/deploy.md`
- Modify: `README.md` (Deployment section via the mirror)

- [ ] **Step 1: Dockerfile.**
  - `python:3.11-slim`.
  - `pip install --no-cache-dir -r requirements-app.txt`, then `pip install --no-deps .`.
  - Copy `app/`, `scripts/fetch_weights.py`, `configs/`.
  - `useradd -u 1000 user`, with `HOME=/home/user`; `MODEL_DIR=/home/user/models`.
  - `EXPOSE 7860`.
  - `entrypoint.sh`: `python scripts/fetch_weights.py --dest $MODEL_DIR && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}`.
  - `.dockerignore`: data, outputs, .venv, legacy, notebooks, reports, .git.
- [ ] **Step 2: `.github/workflows/ci.yml`.** Triggers: `pull_request`, plus `push` to
  `main` and `v2-real-data`.
  - **Job `test`:** ubuntu-latest, Python 3.11. Install CPU torch from
    `https://download.pytorch.org/whl/cpu`, then `requirements.txt` and `-e .`. Run
    `pytest -q` and `python scripts/readme_tables.py --check`.
  - **Job `docker`:** `docker build -t deforest .`; run it detached on 7860; poll
    `/health` for up to 120 s; assert `status == "ok"`; then `docker logs`.
- [ ] **Step 3: `docs/deploy.md`.**
  - Local: `docker build`/`run` and `uvicorn`.
  - curl examples for `/predict` and `/predict-area` + polling.
  - **Earth Engine service account, step by step:**
    1. In the Cloud console for the project, enable the Earth Engine API.
    2. Under IAM & Admin → Service Accounts, create one.
    3. Grant the roles "Earth Engine Resource Viewer" and "Service Usage Consumer".
    4. Under Keys, add a JSON key.
    5. Make sure the project is registered for noncommercial Earth Engine use.
  - **HF Space:** create a Space with the Docker SDK. Settings → secrets:
    `EE_SERVICE_ACCOUNT_KEY` (paste the JSON) and `EE_PROJECT`. Push the repo to the
    Space remote. The Space README needs a YAML header (`sdk: docker`, `app_port: 7860`)
    as the Space's README.
- [ ] **Step 4: Verify.** CI runs on push. Both jobs are green (check through the GitHub
  API with the stored token). Then commit "Add Dockerfile, CI workflow and deployment
  docs", followed by README updates.

### Task 6 (after the user creates the accounts): deploy to HF Spaces
- [ ] The user creates the service account + Space and adds the secrets, following
  `docs/deploy.md`. Claude pushes only with the user's go-ahead (push target: the Space
  git remote).
- [ ] Verify the live URL: `/health` reports `earth_engine: true`; the Ariquemes example
  completes; the stats equal the workstation run.
