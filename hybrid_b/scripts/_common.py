"""Shared utilities for Hybrid B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


HYBRID_B_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HYBRID_B_ROOT.parent
DEFAULT_CONFIG = HYBRID_B_ROOT / "config" / "hybrid_b_config.json"


def load_config(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    config_path = (path or DEFAULT_CONFIG).resolve()
    return json.loads(config_path.read_text(encoding="utf-8")), config_path


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_baseline_config(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    path = project_path(config["paths"]["baseline_config"])
    return json.loads(path.read_text(encoding="utf-8")), path


def baseline_path(baseline_config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (baseline_config_path.parent.parent / path).resolve()


def ensure_directories() -> None:
    for relative in (
        "input",
        "output/stan",
        "output/posterior",
        "output/tables",
        "output/figures",
    ):
        (HYBRID_B_ROOT / relative).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def source_informed_softmax(
    p_source: np.ndarray, correction: np.ndarray
) -> np.ndarray:
    """Compute the softmax on positive-source cells, then reinsert exact zeros."""
    source = np.asarray(p_source, dtype=float)
    field = np.asarray(correction, dtype=float)
    if source.ndim != 1:
        raise ValueError("p_source must be one-dimensional")
    if field.shape[-1] != source.size:
        raise ValueError("The final correction dimension must match p_source")
    if not np.isfinite(source).all() or np.any(source < 0.0):
        raise ValueError("p_source must be finite and nonnegative")
    if not np.isfinite(field).all():
        raise ValueError("correction must be finite")
    if float(source.sum()) <= 0.0:
        raise ValueError("p_source must contain positive total mass")

    positive = source > 0.0
    positive_log_weight = np.log(source[positive]) + field[..., positive]
    maximum = np.max(positive_log_weight, axis=-1, keepdims=True)
    positive_weight = np.exp(positive_log_weight - maximum)
    positive_probability = positive_weight / positive_weight.sum(
        axis=-1, keepdims=True
    )
    probability = np.zeros_like(field, dtype=float)
    probability[..., positive] = positive_probability
    return probability


def validate_probability_array(probability: np.ndarray, tolerance: float) -> float:
    values = np.asarray(probability, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("Probability array has no cell dimension")
    if not np.isfinite(values).all():
        raise RuntimeError("Probability array contains NaN or infinity")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError("Probability array contains a value outside [0, 1]")
    error = float(np.max(np.abs(values.sum(axis=-1) - 1.0)))
    if error > tolerance:
        raise RuntimeError(f"Probability sums differ from one by up to {error:.17g}")
    return error


def posterior_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "q975": float(np.quantile(array, 0.975)),
    }
