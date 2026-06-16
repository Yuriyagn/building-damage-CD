from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_train_manifest(cfg: dict[str, Any]) -> str:
    return str(cfg["dataset"]["train_manifest"])


def get_val_manifest(cfg: dict[str, Any]) -> str:
    return str(cfg["dataset"]["val_manifest"])


def train_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("train", {}))


def augmentation_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("augmentation", {}))


def input_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("input", {}))
