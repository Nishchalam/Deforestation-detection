#!/usr/bin/env bash
# Publish the current HEAD to a Hugging Face Space (Docker SDK).
#
#   scripts/push_space.sh https://huggingface.co/spaces/<user>/<space>
#
# Exports the tracked files of HEAD (git archive, so uncommitted changes are not included) into a
# temp dir, drops what the image does not need, swaps docs/space/README.md in as README.md (the
# Space card with its YAML header), commits it as a fresh single-commit repo and pushes it to the
# remote's `main`.
#
# This FORCE-pushes: the Space repo is a deploy target, not shared history, so each deploy
# replaces it. Do not point this at the GitHub repo. It never runs by itself, and it needs your
# Hugging Face credentials (a write token as the git password, or a configured credential helper).
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 <space-git-url>" >&2; exit 2; }
remote=$1
repo=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git -C "$repo" archive HEAD | tar -x -C "$tmp"
# not needed by the Dockerfile (.dockerignore already keeps them out of the image)
rm -rf "$tmp/notebooks" "$tmp/legacy" "$tmp/reports/figures" "$tmp/docs/superpowers" "$tmp/.github"
cp "$tmp/docs/space/README.md" "$tmp/README.md"

cd "$tmp"
git init -q
git checkout -q -b main
git add -A
git -c user.name="deploy" -c user.email="deploy@localhost" \
    commit -q -m "Deploy $(git -C "$repo" rev-parse --short HEAD)"
git push --force "$remote" main
