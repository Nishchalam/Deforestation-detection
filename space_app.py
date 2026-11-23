"""Entry point for a Hugging Face Space on the free Gradio SDK.

Docker Spaces need a paid plan, and free Gradio Spaces run on ZeroGPU hardware, so the free
deployment runs this file instead of the Dockerfile: the Gradio SDK installs requirements.txt (the
app's pinned runtime deps, swapped in by scripts/push_space.sh) and runs `python space_app.py`.
Gradio itself is not used; this fetches the ONNX weights (sha256-verified) and serves the same
FastAPI app the Docker image runs. Inference stays on the CPU.

ZeroGPU refuses to start a Space unless (a) at least one function is decorated with @spaces.GPU and
(b) the server reports startup to it. The `spaces` package does (b) from its ASGI middleware, so the
app is wrapped in it, and a never-called placeholder satisfies (a). Off ZeroGPU (Docker, local) the
`spaces` package is absent or inert and the app is served unchanged.
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


def build_app():
    try:
        import spaces  # preinstalled on Gradio-SDK Spaces
    except ImportError:
        if os.environ.get("SPACES_ZERO_GPU"):
            raise  # on ZeroGPU the startup check would fail later with a vaguer message
        spaces = None

    from app.main import create_app

    app = create_app()
    if spaces is None:
        return app

    @spaces.GPU
    def _zerogpu_placeholder() -> None:  # never called; the model runs on the CPU
        return None

    return spaces.zero.ZeroGPUMiddleware(app)  # reports startup to ZeroGPU on lifespan start


if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_weights.py"), "--dest", os.environ["MODEL_DIR"]],
                   check=True)
    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
