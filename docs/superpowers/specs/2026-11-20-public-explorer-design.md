# Public deforestation explorer: design

Approved 2026-11-20.

## Intent
Anyone with the link picks an area on a map plus a start and end year, and gets the forest
masks, a forest-loss map and hectare stats for that area. It is a portfolio piece: it should
work for areas far from the training site, and it should be honest about how well it does.

## Goals
- The website and the API live in one container, hosted publicly on Hugging Face Spaces
  (Docker SDK).
- Any area of 500 km² or less, any year pair from 2016 to the last complete dry season,
  and an adjustable season (default Jun–Sep).
- Map layers: before RGB, after RGB, forest before, forest after, loss (model), loss (Hansen).
- Stats: forest before/after ha, loss ha, loss as % of forest-before, Hansen loss ha, and
  agreement with Hansen (IoU, F1, precision, recall).
- Downloadable GeoTIFFs and stats.json.
- API endpoints from the brief:
  - `GET /health`
  - `POST /predict`: one or two 6-band GeoTIFFs in, zip of mask GeoTIFFs + stats.json out
  - `POST /predict-area` and `GET /jobs/{id}`: what the website uses.

## Non-goals
Accounts, persistence beyond a small in-memory/disk cache, per-year time series, GPU
inference, and areas larger than 500 km².

## Anti-goals (a failure even if it works)
- Any fake or fallback imagery. If Earth Engine fails, the request fails with the reason.
- Implying the model is validated outside Ariquemes. A permanent banner says it is not.
  Hansen is shown whenever it covers the years.
- One visitor exhausting the shared Earth Engine quota or CPU.

## Constraints
- Earth Engine authenticates with a service account. The key JSON comes from the
  `EE_SERVICE_ACCOUNT_KEY` secret (or `GOOGLE_APPLICATION_CREDENTIALS`), with the project
  in `EE_PROJECT`. Locally, the cached user credentials work.
- The user creates the Google Cloud and Hugging Face resources. Claude writes instructions
  and creates nothing on those accounts.
- The image is CPU-only with no torch: onnxruntime, rasterio, earthengine-api, fastapi.
  Weights come from GitHub release `models-v2.0` via `scripts/fetch_weights.py` at
  container start, verified by sha256.
- Free Spaces hardware is 2 vCPU / 16 GB, and Spaces expose port 7860.
- Sentinel-2 L1C + s2cloudless starts mid-2015, so the first full dry season is 2016.
  The latest Hansen asset ends in 2025: the Hansen layer and agreement only appear when
  the end year is ≤ the Hansen last year.

## Approaches considered
1. **FastAPI + one static page with Leaflet (chosen).** One server for the API and the
   site. Leaflet handles box drawing and layer toggles natively.
2. **Gradio.** Native on Spaces, but map drawing and layer toggles are awkward, and FastAPI
   endpoints would still be needed.
3. **Streamlit + folium.** Reruns the whole script on each interaction, so it is awkward
   for queued long jobs. It also needs a separate API.

## Design
- **`src/deforest/area.py`** (shared by `scripts/predict.py` and the app):
  - `utm_crs(bbox)` and `validate_request(bbox, y1, y2, months)`, which checks the area
    cap, year range and month order.
  - `fetch_inputs(bbox, y1, y2, months, cfg, out_dir, progress)` downloads two S2
    composites (+ `n_obs`) and the Hansen layers on one UTM 10 m grid, using the existing
    `gee` functions and the training recipe from `configs/ariquemes.yaml`.
  - `run_area(pipe, inputs)` runs the model pipeline. When Hansen covers the years it adds
    the Hansen forest/loss masks (`labels.forest_mask` / `labels.loss_mask`, treecover
    threshold from config) and agreement via `metrics.confusion` / `scores`.
- **Nodata:** a pixel with 0 clear observations in either year is nodata. Stats report
  valid ha and the clear-observation median per year.
- **`app/`:**
  - `main.py`: FastAPI. Loads the ONNX pipeline at startup (`export.load_onnx_pipeline`).
    A single-worker job queue (`ThreadPoolExecutor(1)`) handles requests. Jobs live in a
    dict with status/progress/position/result/error. Results are cached on disk keyed by a
    hash of the rounded bbox, years and months, with LRU eviction past N entries.
    Endpoints: `/`, `/health`, `/predict`, `/predict-area`, `/jobs/{id}`, and
    `/jobs/{id}/layers/{name}.png`, `/jobs/{id}/download.zip`.
  - `render.py`: reprojects each mask/RGB from UTM to EPSG:3857 (rasterio.warp, nearest
    for masks, bilinear for RGB) and writes a transparent PNG plus lat/lon bounds, so
    Leaflet `imageOverlay` lines up exactly.
  - `static/index.html`: a single page using Leaflet 1.9.4 and leaflet-draw from cdnjs,
    with an Esri World Imagery basemap.
    - Controls: draw a box, example areas, years, months, a run button, queue position and
      progress, layer toggles, a stats panel and a download link.
    - A banner states the model was trained only on Ariquemes.
    - Errors from the server are shown verbatim.
- **Limits:**
  - `MAX_AREA_KM2=500`, `MAX_QUEUE=5` (beyond that: 429 "busy, try later").
  - Per-IP: one active job.
  - A job timeout of 10 min.
- **Example areas:** Ariquemes (labelled "training area"), plus 3–4 other ~20×20 km boxes
  in active clearing frontiers, chosen from Hansen loss.
- **Docker:**
  - `python:3.11-slim`, with `requirements-app.txt` and the package installed `--no-deps`.
  - A non-root user (Spaces requires uid 1000).
  - Entrypoint: `fetch_weights.py` → `uvicorn app.main:app --port ${PORT:-7860}`.
- **CI (`.github/workflows/ci.yml`)**, on every PR and push to v2-real-data:
  - pytest (CPU torch).
  - `readme_tables.py --check`.
  - docker build, start the container, `curl /health` (weights fetched from the release).
- **Docs:** `docs/deploy.md` covers the docker build/run and curl examples. It also gives
  step-by-step instructions for the Earth Engine service account (roles, registering the
  project for noncommercial use) and the HF Space (secrets, pushing the repo).

## Left to the builder
Exact example-area boxes, colours, PNG compression, cache size (default 20 jobs), and the
copy text.
