"""Download the ONNX weights from the GitHub release and verify their sha256.

    python scripts/fetch_weights.py [--dest models/] [--release models-v2.0]

The expected hashes come from app/model_card.json, which is committed, so a changed or
truncated download fails loudly. Uses only the standard library (runs in the API image).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "Nishchalam/Deforestation-detection"
CARD = Path(__file__).resolve().parents[1] / "app" / "model_card.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: Path, release: str, card_path: Path = CARD) -> Path:
    card = json.loads(card_path.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    for f in card["files"].values():
        out = dest / f["name"]
        if out.exists() and sha256(out) == f["sha256"]:
            print(f"ok (cached) {out}")
            continue
        url = f"https://github.com/{REPO}/releases/download/{release}/{f['name']}"
        print(f"downloading {url}")
        tmp = out.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as w:
            shutil.copyfileobj(r, w, 1 << 20)
        got = sha256(tmp)
        if got != f["sha256"]:
            tmp.unlink()
            sys.exit(f"sha256 mismatch for {f['name']}: expected {f['sha256']}, got {got}")
        tmp.rename(out)
        print(f"ok {out}")
    shutil.copy(card_path, dest / "model_card.json")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=os.environ.get("MODEL_DIR", "models"))
    p.add_argument("--release", default=os.environ.get("MODEL_RELEASE", "models-v2.0"))
    a = p.parse_args()
    fetch(Path(a.dest), a.release)


if __name__ == "__main__":
    main()
