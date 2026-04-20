from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return PROJECT_ROOT


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_project_config(config_dir: Path | None = None) -> dict[str, Any]:
    root = project_root()
    config_dir = config_dir or (root / "configs")
    merged: dict[str, Any] = {}
    for name in (
        "pipeline.yaml",
        "data.yaml",
        "quality.yaml",
        "preprocessing.yaml",
        "modeling.yaml",
    ):
        path = config_dir / name
        if path.exists():
            merged = _deep_merge(merged, load_yaml(path))
    return merged


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Any, path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    return path


def get_logger(name: str, logging_config: dict[str, Any] | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    cfg = logging_config or {}
    level_name = str(cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    datefmt = cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    logger.setLevel(level)
    logger.propagate = False

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(stream_handler)
    return logger
