#!/usr/bin/env bash
# Container start: fetch the ONNX weights from the GitHub release (sha256-verified, cached in
# $MODEL_DIR), then serve. Weights are not baked into the image, so a model release needs no rebuild.
set -euo pipefail
python scripts/fetch_weights.py --dest "${MODEL_DIR:-models}"
exec uvicorn --factory app.main:create_app --host 0.0.0.0 --port "${PORT:-7860}"
