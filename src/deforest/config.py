"""YAML config loading. Configs are plain nested dicts exposed with attribute access."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with read-only attribute access to (nested) keys."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) else value


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load YAML; `overrides` are 'a.b=value' strings, value parsed as YAML."""
    cfg = yaml.safe_load(Path(path).read_text())
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key, yaml.safe_load(raw))
    return Config(cfg)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(dict(cfg), sort_keys=False))
