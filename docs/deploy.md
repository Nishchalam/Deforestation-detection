# Deployment

The explorer is one FastAPI process (`app/main.py`) that serves the web page and the API, runs the ONNX model on CPU, and fetches Sentinel-2 composites from Google Earth Engine (GEE) for area requests. The image is built for Hugging Face Spaces (Docker SDK) but runs anywhere.

The ONNX weights are not in the image or the repository. The container downloads them from the GitHub release `models-v2.0` at start and checks their sha256 against `app/model_card.json`, so the first start needs network access.

## Run locally

```bash
docker build -t deforest .
docker run --rm -p 7860:7860 deforest                      # upload API only
docker run --rm -p 7860:7860 -e EE_PROJECT=<gcp-project> \
    -e EE_SERVICE_ACCOUNT_KEY="$(cat key.json)" deforest   # with area requests
```

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

`bbox` is `[west, south, east, north]` in degrees. Interactive docs are at `/docs`.

## Earth Engine service account

Area requests need Earth Engine access that does not depend on a person's login. Create a service account once:

1. In the [Cloud console](https://console.cloud.google.com/) for your project, enable the **Earth Engine API** (APIs & Services, Library).
2. Under **IAM & Admin, Service Accounts**, create a service account (any name, e.g. `deforest-explorer`).
3. Grant it the roles **Earth Engine Resource Viewer** and **Service Usage Consumer** (IAM & Admin, IAM, edit the account's roles).
4. Open the service account, **Keys, Add key, Create new key, JSON**. Keep the downloaded file private and never commit it.
5. Make sure the project is [registered for noncommercial Earth Engine use](https://code.earthengine.google.com/register). Unregistered projects fail to initialise.

The key's JSON text goes into `EE_SERVICE_ACCOUNT_KEY` and the project id into `EE_PROJECT`. After the app is running, `/health` shows `earth_engine: true`, or the reason it failed in `earth_engine_detail`.

## Hugging Face Space

1. Create a new Space at <https://huggingface.co/new-space> with SDK **Docker** (blank template). Free CPU hardware is enough.
2. Under **Settings, Variables and secrets**, add two secrets: `EE_SERVICE_ACCOUNT_KEY` (paste the whole JSON key) and `EE_PROJECT` (the Cloud project id). Variables such as `MAX_AREA_KM2` can be added the same way.
3. Push the repository:

   ```bash
   scripts/push_space.sh https://huggingface.co/spaces/<user>/<space>
   ```

   The script snapshots `HEAD`, replaces `README.md` with [`docs/space/README.md`](space/README.md) (the Space card: it needs the YAML header with `sdk: docker` and `app_port: 7860`, which the GitHub README must not have) and force-pushes a single commit to the Space's `main`. The Space is a deploy target, so its history is disposable. Git asks for your Hugging Face username and a write access token as the password.
4. The Space builds the Dockerfile, starts the container, downloads the weights and serves on port 7860. Build and start logs are on the Space's page.

To deploy a new version, commit on GitHub and run the script again. A new model release only needs a change of `app/model_card.json` and `MODEL_RELEASE`.

### Check the client address after the first deploy

The `/predict-area` queue allows one active request per client, identified by the address in `X-Forwarded-For`. `TRUSTED_PROXY_HOPS=1` assumes the Spaces proxy appends the client address to that header, so the client is the last entry. This has not been confirmed. After the first deploy, check it in one of two ways:

- Submit an area from one network, and while it runs submit another from a second network (a phone on mobile data). Both must be accepted. If the second answers "you already have a request in progress", all clients look like one address (the proxy's), so the value is wrong.
- Submit twice from one network. The second must be refused. If it is accepted when the request carries its own `X-Forwarded-For: 1.2.3.4` header, clients can bypass the limit, so the value is too high.

Adjust `TRUSTED_PROXY_HOPS` (a Space variable, no rebuild) and test again.

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on pull requests and on pushes to `main` and `v2-real-data`:

- `test`: CPU torch, `requirements.txt`, `pytest -q` and `scripts/readme_tables.py --check`. No data files or credentials are needed.
- `docker`: builds the image, starts it without Earth Engine credentials, waits up to 120 s for `/health` (it downloads the weights from the GitHub release) and asserts `status == "ok"`.
