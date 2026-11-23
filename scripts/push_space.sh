#!/usr/bin/env bash
# Publish the current HEAD to a Hugging Face Space.
#
#   scripts/push_space.sh https://huggingface.co/spaces/<user>/<space>            # Gradio SDK (free CPU)
#   scripts/push_space.sh --docker https://huggingface.co/spaces/<user>/<space>   # Docker SDK (paid plan)
#
# Gradio SDK (default): Docker Spaces need a paid plan, so the free Space uses the Gradio SDK on
# CPU basic. The snapshot's requirements.txt is replaced by requirements-app.txt (no torch) and
# the Space runs space_app.py, which serves the same FastAPI app as the Docker image.
#
# Refuses any remote that is not a Hugging Face Space URL, or that equals this repo's origin.
# Exports the tracked files of HEAD (git archive, so uncommitted changes are not included) into a
# temp dir, drops what the image does not need, swaps docs/space/README.md in as README.md (the
# Space card with its YAML header), commits it as a fresh single-commit repo and pushes it to the
# remote's `main`.
#
# This FORCE-pushes: the Space repo is a deploy target, not shared history, so each deploy
# replaces it. Do not point this at the GitHub repo. It never runs by itself, and it needs your
# Hugging Face credentials (a write token as the git password, or a configured credential helper).
set -euo pipefail

sdk=gradio
if [ "${1:-}" = "--docker" ]; then sdk=docker; shift; fi
[ $# -eq 1 ] || { echo "usage: $0 [--docker] <space-git-url>" >&2; exit 2; }
remote=$1
repo=$(git rev-parse --show-toplevel)

# Never force-push anywhere but a Hugging Face Space, and never to this repo's own origin.
case "$remote" in
  https://huggingface.co/spaces/* | git@hf.co:spaces/*) ;;
  *) echo "refusing: $remote is not a Hugging Face Space URL (https://huggingface.co/spaces/<user>/<space>)" >&2; exit 2 ;;
esac
origin=$(git -C "$repo" remote get-url origin 2>/dev/null || true)
if [ -n "$origin" ] && [ "${remote%.git}" = "${origin%.git}" ]; then
  echo "refusing: $remote is this repository's own origin" >&2; exit 2
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git -C "$repo" archive HEAD | tar -x -C "$tmp"
# not needed by the Dockerfile (.dockerignore already keeps them out of the image)
rm -rf "$tmp/notebooks" "$tmp/legacy" "$tmp/reports/figures" "$tmp/docs/superpowers" "$tmp/.github" "$tmp/.superpowers"
if [ "$sdk" = docker ]; then
  cp "$tmp/docs/space/README.md" "$tmp/README.md"
else
  # Gradio-SDK card: same text, different header; runtime deps only; system lib for rasterio
  sed -e 's/^sdk: docker$/sdk: gradio\nsdk_version: 6.32.0\npython_version: "3.12"\napp_file: space_app.py/' \
      -e '/^app_port:/d' "$tmp/docs/space/README.md" > "$tmp/README.md"
  cp "$tmp/requirements-app.txt" "$tmp/requirements.txt"
  echo libexpat1 > "$tmp/packages.txt"
  rm -f "$tmp/Dockerfile" "$tmp/.dockerignore"
fi

cd "$tmp"
git init -q
git checkout -q -b main
git add -A
git -c user.name="deploy" -c user.email="deploy@localhost" \
    commit -q -m "Deploy $(git -C "$repo" rev-parse --short HEAD)"
git push --force "$remote" main
