# Deployment

The explorer is one FastAPI process (`app/main.py`) that serves the web page and the API, runs the ONNX model on CPU, and fetches Sentinel-2 composites from Google Earth Engine (GEE) for area requests. It ships as a Docker image that runs anywhere, and as a free Hugging Face Space (Gradio SDK, see below).

The ONNX weights are not in the image or the repository. The container downloads them from the GitHub release `models-v2.0` at start and checks their sha256 against `app/model_card.json`, so the first start needs network access.

## Run locally

```bash
docker build -t deforest .
docker run --rm -p 7860:7860 deforest                      # upload API only
docker run --rm -p 7860:7860 -e EE_PROJECT=<gcp-project> \
    -e EE_SERVICE_ACCOUNT_KEY="$(cat ~/ee-key.json)" deforest   # with area requests
```

Keep the service-account key outside the repository (here `~/ee-key.json`). `.gitignore` also covers `/key.json`, but a key that is never inside the work tree cannot be committed or copied into an image by accident.

Open <http://localhost:7860>. Without an Earth Engine key `/health` still reports `status: "ok"` with `earth_engine: false`, `/predict` works and `/predict-area` answers 503.

Without Docker (needs the weights in `models/`):

```bash
pip install -r requirements-app.txt && pip install --no-deps .
python scripts/fetch_weights.py --dest models
uvicorn --factory app.main:create_app --host 0.0.0.0 --port 7860
```

There is no module-level `app`; `create_app()` fails at start if the weights are missing.

## Configuration

All settings are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `MAX_AREA_KM2` | 500 | largest area (and upload) accepted |
| `MAX_QUEUE` | 5 | jobs waiting or running before new ones get 429 (also: one active job per client) |
| `JOB_TIMEOUT_S` | 1200 | per-job time limit |
| `CACHE_SIZE` | 20 | finished jobs kept (identical requests are served from here) |
| `MAX_UPLOAD_MB` | 200 | `/predict` upload limit |
| `ORT_THREADS` | 2 | ONNX Runtime threads |
| `TRUSTED_PROXY_HOPS` | 1 | proxies in front of the app, for the per-client limit (see below) |
| `EE_RETRY_S` | 60 | wait before retrying a failed Earth Engine start |
| `MAX_JOBS_PER_IP_PER_DAY` | 10 | new area jobs one client address may start in a rolling 24 h (answer 429 above it) |
| `MAX_JOBS_PER_DAY` | 100 | new area jobs over all clients in a rolling 24 h; each one spends Earth Engine quota |
| `MODEL_DIR`, `CACHE_DIR` | `models`, `cache` | the image sets both under `/home/user` |
| `MODEL_RELEASE` | `models-v2.0` | GitHub release the weights come from |
| `EE_PROJECT` | | Cloud project used for Earth Engine |
| `EE_SERVICE_ACCOUNT_KEY` | | the service-account JSON key, as text |
| `GOOGLE_APPLICATION_CREDENTIALS` | | alternative: path to that key file |

## API

```bash
curl http://localhost:7860/health

# two co-registered 6-band GeoTIFFs in a projected CRS (second one optional) -> zip of masks + stats.json
curl -F before=@s2_2018.tif -F after=@s2_2022.tif http://localhost:7860/predict -o result.zip

# area request: returns a job id, then poll until status is "done" (or "error")
curl -X POST http://localhost:7860/predict-area -H 'content-type: application/json' \
     -d '{"bbox": [-63.0, -10.5, -62.8, -10.3], "start": 2018, "end": 2022, "months": [6, 9]}'
# {"job_id": "..."}
curl http://localhost:7860/jobs/<job_id>                  # status, progress, then stats + layer urls
curl -o area.zip http://localhost:7860/jobs/<job_id>/download.zip
```

`bbox` is `[west, south, east, north]` in degrees, at most 1 degree per side, and its 10 m grid must stay within 10,000 px per side and `MAX_AREA_KM2` (plus 20 %). The end of the chosen season must be in the past. Only new jobs count towards the daily limits; a repeated request is served from the cache or the running job. Uploads must be GeoTIFFs with 10 m pixels (+-5 %) in a projected CRS, with at most `MAX_AREA_KM2` / 100 m² pixels and at most 10,000 px per side. Interactive docs are at `/docs`.

## Earth Engine service account

Area requests need Earth Engine access that does not depend on a person's login. Create a service account once:

1. In the [Cloud console](https://console.cloud.google.com/) for your project, enable the **Earth Engine API** (APIs & Services, Library).
2. Under **IAM & Admin, Service Accounts**, create a service account (any name, e.g. `deforest-explorer`).
3. Grant it the roles **Earth Engine Resource Viewer** and **Service Usage Consumer** (IAM & Admin, IAM, edit the account's roles).
4. Open the service account, **Keys, Add key, Create new key, JSON**. Keep the downloaded file private, outside the repository (for example `~/ee-key.json`), and never commit it.
5. Make sure the project is [registered for noncommercial Earth Engine use](https://code.earthengine.google.com/register). Unregistered projects fail to initialise.

The key's JSON text goes into `EE_SERVICE_ACCOUNT_KEY` and the project id into `EE_PROJECT`. After the app is running, `/health` shows `earth_engine: true`, or the reason it failed in `earth_engine_detail`.

## Hugging Face Space

Docker Spaces need a paid Hugging Face plan, and on a free account Gradio Spaces only run on **ZeroGPU**, so the free deployment uses the **Gradio SDK** on **ZeroGPU**. Gradio itself is not used: the Space runs [`space_app.py`](../space_app.py), which fetches the weights and serves the same FastAPI app as the Docker image, with inference on the CPU. ZeroGPU will not start a Space without a `@spaces.GPU` function and a startup report, so `space_app.py` declares a never-called placeholder and wraps the app in the `spaces` middleware that sends the report.

1. Create a new Space at <https://huggingface.co/new-space>: SDK **Gradio**, template **Blank**, hardware **ZeroGPU** (the only free choice; CPU basic needs PRO for Gradio Spaces).
2. Under **Settings, Variables and secrets**, add `EE_SERVICE_ACCOUNT_KEY` as a **secret** (paste the whole JSON key; never as a variable, variables are public) and `EE_PROJECT` (the Cloud project id) as a variable or secret. Variables such as `MAX_AREA_KM2` can be added the same way.
3. Push the repository:

   ```bash
   scripts/push_space.sh https://huggingface.co/spaces/<user>/<space>
   ```

   The script snapshots `HEAD`, writes the Space card from [`docs/space/README.md`](space/README.md) with a Gradio header (`sdk_version`, `python_version: "3.12"`, `app_file: space_app.py`), replaces `requirements.txt` with `requirements-app.txt` (no torch), adds `packages.txt` (`libexpat1` for rasterio) and force-pushes a single commit to the Space's `main`. The Space is a deploy target, so its history is disposable. Git asks for your Hugging Face username and a write access token as the password. With a paid plan, `scripts/push_space.sh --docker <url>` deploys the Dockerfile instead (create the Space with SDK Docker).
4. The Space installs the requirements, runs `space_app.py`, downloads the weights and serves on port 7860. Build and start logs are on the Space's page.

To deploy a new version, commit on GitHub and run the script again. A new model release only needs a change of `app/model_card.json` and `MODEL_RELEASE`.

### Check the client address after the first deploy

The `/predict-area` queue allows one active request per client, and the daily limit counts new jobs per client, both identified by the address in `X-Forwarded-For`. `TRUSTED_PROXY_HOPS=1` assumes the Spaces proxy appends the client address to that header, so the client is the last entry. This has not been confirmed. After the first deploy, check it in one of two ways:

- Submit an area from one network, and while it runs submit a different area from a second network (a phone on mobile data). Both must be accepted. If the second answers "you already have a request in progress", all clients look like one address (the proxy's), so the value is wrong.
- Submit two DIFFERENT areas from one network (identical requests share one job, so they never trigger the limit). The second must be refused. If it is accepted when the request carries its own `X-Forwarded-For: 1.2.3.4` header, clients can bypass the limit, so the value is too high.

Adjust `TRUSTED_PROXY_HOPS` (a Space variable, no rebuild) and test again.

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on pull requests and on pushes to `main` and `v2-real-data`:

- `test`: CPU torch, `requirements.txt`, `pytest -q` and `scripts/readme_tables.py --check`. No data files or credentials are needed.
- `docker`: builds the image, starts it without Earth Engine credentials, waits up to 120 s for `/health` (it downloads the weights from the GitHub release) and asserts `status == "ok"`.
