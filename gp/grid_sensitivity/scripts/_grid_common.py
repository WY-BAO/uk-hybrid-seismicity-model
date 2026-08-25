"""Shared utilities for the baseline multinomial grid sensitivity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


SENSITIVITY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = SENSITIVITY_ROOT.parent
PROJECT_ROOT = BASELINE_ROOT.parent
SENSITIVITY_CONFIG = SENSITIVITY_ROOT / "config" / "grid_sensitivity_config.json"
BASELINE_CONFIG = BASELINE_ROOT / "config" / "baseline_config.json"
STAN_FILE = BASELINE_ROOT / "stan" / "baseline_gp.stan"

if str(BASELINE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT / "scripts"))

from _common import posterior_summary, resolve_project_path, sha256_file, write_json  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_json(BASELINE_CONFIG), load_json(SENSITIVITY_CONFIG)


def load_preparation_module():
    path = BASELINE_ROOT / "scripts" / "01_prepare_gp_data.py"
    spec = importlib.util.spec_from_file_location("baseline_gp_preparation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def degree_label(degree: float) -> str:
    return f"degree_{degree:.1f}".replace(".", "p")


def case_directory(degree: float) -> Path:
    return SENSITIVITY_ROOT / "runs" / degree_label(degree)


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def ensure_output_directories() -> None:
    for relative in ("runs", "output", "output/tables", "output/figures", "output/maps"):
        (SENSITIVITY_ROOT / relative).mkdir(parents=True, exist_ok=True)


def load_exact_catalogue(
    baseline_config: dict[str, Any], preparation_module
) -> tuple[pd.DataFrame, Path, Path]:
    catalogue_path = resolve_project_path(baseline_config["paths"]["l5_catalogue"])
    l5_input_path = resolve_project_path(baseline_config["paths"]["l5_stan_input"])
    catalogue = pd.read_csv(catalogue_path, low_memory=False)
    l5_input = load_json(l5_input_path)
    events = preparation_module.load_l5_catalogue(catalogue, l5_input, baseline_config)
    if len(events) != int(baseline_config["expected"]["earthquakes"]):
        raise RuntimeError(f"Expected 1013 exact L5 events, found {len(events)}")
    return events, catalogue_path, l5_input_path


def build_case_data(
    degree: float,
    baseline_config: dict[str, Any],
    preparation_module,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    case_config = json.loads(json.dumps(baseline_config))
    case_config["grid"]["degree"] = float(degree)
    assigned, cells = preparation_module.build_grid(events, case_config)
    data = preparation_module.stan_data(cells, case_config)
    if len(assigned) != 1013 or assigned["event_grid_id"].isna().any():
        raise RuntimeError(f"degree={degree}: not every earthquake has exactly one assignment")
    if int(cells["count"].sum()) != 1013:
        raise RuntimeError(f"degree={degree}: grid counts do not sum to 1013")
    if set(data) != {
        "N",
        "D",
        "x",
        "count",
        "cell_area_km2",
        "distance_scale_km",
        "jitter",
        "alpha_prior_sd",
        "rho_prior_logmean",
        "rho_prior_logsd",
    }:
        raise RuntimeError(f"degree={degree}: unexpected Stan data fields {sorted(data)}")
    return assigned, cells, data


def assert_current_multinomial_source() -> dict[str, Any]:
    source = STAN_FILE.read_text(encoding="utf-8")
    lower = source.lower()
    checks = {
        "multinomial_likelihood_present": "count ~ multinomial(spatial_probability)" in source,
        "poisson_absent": "poisson" not in lower,
        "exposure_absent": "log_exposure_area" not in lower and "exposure_year" not in lower,
        "fixed_time_absent": "33-year" not in lower and "33 year" not in lower,
        "intercept_a_absent": not bool(
            re.search(r"\breal(?:<[^>]+>)?\s+a\s*;", source)
            or re.search(r"\ba_prior_(?:mean|sd)\b", source)
        ),
        "area_aware_probability_present": (
            "softmax(log(cell_area_km2) + gp_effect)" in source
        ),
        "exp_quad_kernel_present": "gp_exp_quad_cov(x, alpha, rho)" in source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Baseline Stan-source identity checks failed: {checks}")
    return checks


def common_pairing_indices(
    baseline_config: dict[str, Any], gp_draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    l5_path = resolve_project_path(baseline_config["paths"]["l5_posterior_draws"])
    variable = str(baseline_config["l5_combination"]["variable"])
    frame = pd.read_csv(l5_path)
    if variable not in frame:
        raise KeyError(f"{variable} missing from {l5_path}")
    l5_values = frame[variable].to_numpy(float)
    if len(l5_values) != gp_draws:
        raise RuntimeError(
            "Grid sensitivity requires equal GP and L5 draw counts for common, "
            "without-replacement pairing"
        )
    base_seed = int(baseline_config["l5_combination"]["seed"])
    gp_rng = np.random.default_rng(stable_seed(base_seed, "multinomial-grid-sensitivity|gp"))
    l5_rng = np.random.default_rng(stable_seed(base_seed, "multinomial-grid-sensitivity|l5"))
    gp_index = gp_rng.permutation(gp_draws)
    l5_index = l5_rng.permutation(len(l5_values))
    return gp_index, l5_index, l5_values[l5_index], l5_path


def save_common_pairing_indices(
    gp_index: np.ndarray,
    l5_index: np.ndarray,
    l5_paired: np.ndarray,
    l5_path: Path,
    baseline_config: dict[str, Any],
) -> None:
    target = SENSITIVITY_ROOT / "output" / "common_pairing_indices.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as loaded:
            if not (
                np.array_equal(loaded["gp_draw_index"], gp_index)
                and np.array_equal(loaded["l5_draw_index"], l5_index)
                and np.array_equal(loaded["L5_total_activity"], l5_paired)
            ):
                raise RuntimeError("Existing common pairing indices do not match deterministic indices")
    else:
        np.savez_compressed(
            target,
            gp_draw_index=gp_index,
            l5_draw_index=l5_index,
            L5_total_activity=l5_paired,
        )
    write_json(
        SENSITIVITY_ROOT / "output" / "common_pairing_manifest.json",
        {
            "method": "common independent permutations; one-to-one without replacement",
            "draws": int(len(l5_paired)),
            "base_seed": int(baseline_config["l5_combination"]["seed"]),
            "l5_posterior": str(l5_path),
            "l5_posterior_sha256": sha256_file(l5_path),
            "all_gp_indices_used_once": bool(np.unique(gp_index).size == len(gp_index)),
            "all_l5_indices_used_once": bool(np.unique(l5_index).size == len(l5_index)),
        },
    )


def spherical_area_km2(
    lon_lo: float, lon_hi: float, lat_lo: float, lat_hi: float, earth_radius_km: float
) -> float:
    if lon_hi <= lon_lo or lat_hi <= lat_lo:
        return 0.0
    return float(
        earth_radius_km**2
        * math.radians(lon_hi - lon_lo)
        * (math.sin(math.radians(lat_hi)) - math.sin(math.radians(lat_lo)))
    )


def build_overlap_transfer(
    model_grid: pd.DataFrame,
    evaluation_grid: pd.DataFrame,
    earth_radius_km: float,
    tolerance: float,
) -> tuple[sparse.csr_matrix, dict[str, float]]:
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    eval_lon_lo = evaluation_grid["lon_lo"].to_numpy(float)
    eval_lon_hi = evaluation_grid["lon_hi"].to_numpy(float)
    eval_lat_lo = evaluation_grid["lat_lo"].to_numpy(float)
    eval_lat_hi = evaluation_grid["lat_hi"].to_numpy(float)
    for model_index, cell in enumerate(model_grid.itertuples(index=False)):
        candidates = np.flatnonzero(
            (eval_lon_hi > float(cell.lon_lo) + tolerance)
            & (eval_lon_lo < float(cell.lon_hi) - tolerance)
            & (eval_lat_hi > float(cell.lat_lo) + tolerance)
            & (eval_lat_lo < float(cell.lat_hi) - tolerance)
        )
        for eval_index in candidates:
            lon_lo = max(float(cell.lon_lo), float(eval_lon_lo[eval_index]))
            lon_hi = min(float(cell.lon_hi), float(eval_lon_hi[eval_index]))
            lat_lo = max(float(cell.lat_lo), float(eval_lat_lo[eval_index]))
            lat_hi = min(float(cell.lat_hi), float(eval_lat_hi[eval_index]))
            overlap = spherical_area_km2(
                lon_lo, lon_hi, lat_lo, lat_hi, earth_radius_km
            )
            if overlap > 0.0:
                rows.append(model_index)
                columns.append(int(eval_index))
                weights.append(overlap / float(cell.grid_area_km2))
    transfer = sparse.csr_matrix(
        (weights, (rows, columns)), shape=(len(model_grid), len(evaluation_grid))
    )
    row_error = float(np.max(np.abs(np.asarray(transfer.sum(axis=1)).ravel() - 1.0)))
    eval_area = evaluation_grid["grid_area_km2"].to_numpy(float)
    covered_eval_area = np.asarray(
        transfer.multiply(model_grid["grid_area_km2"].to_numpy(float)[:, None]).sum(axis=0)
    ).ravel()
    coverage_error = float(np.max(np.abs(covered_eval_area / eval_area - 1.0)))
    if row_error > 5e-10 or coverage_error > 5e-10:
        raise RuntimeError(
            f"Overlap transfer is not conservative: row={row_error}, coverage={coverage_error}"
        )
    return transfer, {
        "nonzero_overlaps": int(transfer.nnz),
        "maximum_model_cell_weight_sum_error": row_error,
        "maximum_evaluation_cell_area_coverage_error": coverage_error,
    }


def summarise_draws(values: np.ndarray) -> dict[str, float]:
    summary = posterior_summary(values)
    return {
        "mean": summary["mean"],
        "median": summary["median"],
        "sd": summary["sd"],
        "q025": summary["q025"],
        "q975": summary["q975"],
    }
