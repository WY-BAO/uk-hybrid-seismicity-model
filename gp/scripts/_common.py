"""Shared utilities for the baseline multinomial GP pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASELINE_ROOT / "config" / "baseline_config.json"


def load_config(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    config_path = (path or DEFAULT_CONFIG).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config, config_path


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (BASELINE_ROOT / path).resolve()


def ensure_directories() -> None:
    for relative in (
        "input",
        "output/stan",
        "output/posterior",
        "output/tables",
        "output/figures",
    ):
        (BASELINE_ROOT / relative).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def posterior_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "q025": float(np.quantile(array, 0.025)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "q95": float(np.quantile(array, 0.95)),
        "q975": float(np.quantile(array, 0.975)),
    }
