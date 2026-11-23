"""Entry point for a Hugging Face Space on the free Gradio SDK (CPU basic).

Docker Spaces need a paid plan, so the free deployment runs this file instead of the Dockerfile:
the Gradio SDK installs requirements.txt (the app's pinned runtime deps, swapped in by
scripts/push_space.sh) and runs `python space_app.py`. Gradio itself is not used; this fetches the
ONNX weights (sha256-verified) and serves the same FastAPI app the Docker image runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

home = Path(os.environ.get("HOME", ROOT))
os.environ.setdefault("MODEL_DIR", str(home / "models"))
os.environ.setdefault("CACHE_DIR", str(home / "cache"))

if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_weights.py"), "--dest", os.environ["MODEL_DIR"]],
                   check=True)
    import uvicorn

    uvicorn.run("app.main:create_app", factory=True, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
