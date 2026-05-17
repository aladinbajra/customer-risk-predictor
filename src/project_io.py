from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_parent(path: str | Path) -> Path:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    input_path = project_path(path)
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_metric_dict(metrics: dict[str, Any]) -> dict[str, float | int | str]:
    cleaned: dict[str, float | int | str] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            cleaned[key] = round(value, 6)
        else:
            cleaned[key] = value
    return cleaned




