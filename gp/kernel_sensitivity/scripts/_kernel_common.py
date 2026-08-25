"""Shared validation, covariance, diagnostic, and provenance utilities."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SENSITIVITY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = SENSITIVITY_ROOT.parent
PROJECT_ROOT = BASELINE_ROOT.parent
CONFIG_FILE = SENSITIVITY_ROOT / "config" / "kernel_sensitivity_config.json"
STAN_FILE = SENSITIVITY_ROOT / "stan" / "kernel_sensitivity_multinomial.stan"
BASELINE_CONFIG_FILE = BASELINE_ROOT / "config" / "baseline_config.json"
BASELINE_STAN_FILE = BASELINE_ROOT / "stan" / "baseline_gp.stan"
BASELINE_STAN_DATA = BASELINE_ROOT / "input" / "stan_data.json"
BASELINE_GRID = BASELINE_ROOT / "input" / "grid_cells.csv"
BASELINE_EVENTS = BASELINE_ROOT / "input" / "gp_earthquakes.csv"
BASELINE_PREPARATION_MANIFEST = BASELINE_ROOT / "input" / "preparation_manifest.json"
BASELINE_RUN_MANIFEST = BASELINE_ROOT / "output" / "stan" / "run_manifest.json"
BASELINE_COMBINATION_MANIFEST = (
    BASELINE_ROOT / "output" / "posterior" / "combination_manifest.json"
)
BASELINE_GP_POSTERIOR = (
    BASELINE_ROOT / "output" / "posterior" / "gp_posterior_draws.npz"
)
BASELINE_ACTIVITY_POSTERIOR = (
    BASELINE_ROOT / "output" / "posterior" / "baseline_posterior_draws.npz"
)
BASELINE_VERIFICATION_REPORT = BASELINE_ROOT / "output" / "verification_report.md"
BASELINE_README = BASELINE_ROOT / "README.md"
LEGACY_PROVENANCE_ROOT = BASELINE_ROOT / "legacy_kernel_definitions"
WORKFLOW_FILES = [
    BASELINE_ROOT / "scripts" / "01_prepare_gp_data.py",
    BASELINE_ROOT / "scripts" / "02_run_gp.py",
    BASELINE_ROOT / "scripts" / "03_combine_with_l5.py",
    BASELINE_ROOT / "scripts" / "04_summarize_results.py",
]
LEGACY_DEFINITION_FILES = [
    LEGACY_PROVENANCE_ROOT / "kernel_sensitivity_q1.stan",
    LEGACY_PROVENANCE_ROOT / "rational_quadratic_q2_correction.stan",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=convert) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def configs() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_json(BASELINE_CONFIG_FILE), load_json(CONFIG_FILE)


def kernel_order() -> list[str]:
    return list(load_json(CONFIG_FILE)["kernel_order"])


def kernel_spec(kernel: str) -> dict[str, Any]:
    return load_json(CONFIG_FILE)["kernels"][kernel]


def case_root(kernel: str) -> Path:
    return SENSITIVITY_ROOT / "runs" / kernel


def ensure_directories() -> None:
    for relative in ("output", "tables", "figures", "maps", "runs"):
        (SENSITIVITY_ROOT / relative).mkdir(parents=True, exist_ok=True)
    for kernel in kernel_order():
        for relative in ("stan", "posterior", "tables"):
            (case_root(kernel) / relative).mkdir(parents=True, exist_ok=True)


def posterior_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.quantile(array, 0.5)),
        "sd": float(np.std(array, ddof=1)),
        "q025": float(np.quantile(array, 0.025)),
        "q975": float(np.quantile(array, 0.975)),
    }


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def resolve_baseline_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASELINE_ROOT / path).resolve()


def pairing_context(
    baseline_config: dict[str, Any], gp_draws: int
) -> dict[str, Any]:
    l5_path = resolve_baseline_path(baseline_config["paths"]["l5_posterior_draws"])
    variable = str(baseline_config["l5_combination"]["variable"])
    frame = pd.read_csv(l5_path)
    if variable not in frame:
        raise KeyError(f"{variable!r} is absent from {l5_path}")
    l5_values = frame[variable].to_numpy(float)
    if len(l5_values) != gp_draws:
        raise RuntimeError("The GP and L5 posterior files must both contain 4000 draws")
    if not np.isfinite(l5_values).all() or np.any(l5_values <= 0):
        raise RuntimeError("L5 total-rate draws must be finite and positive")
    base_seed = int(baseline_config["l5_combination"]["seed"])
    gp_seed = stable_seed(base_seed, "degree=1|gp")
    l5_seed = stable_seed(base_seed, "degree=1|l5")
    gp_index = np.random.default_rng(gp_seed).permutation(gp_draws)
    l5_index = np.random.default_rng(l5_seed).permutation(len(l5_values))
    return {
        "gp_index": gp_index,
        "l5_index": l5_index,
        "l5_paired": l5_values[l5_index],
        "l5_path": l5_path,
        "l5_variable": variable,
        "base_seed": base_seed,
        "gp_seed": gp_seed,
        "l5_seed": l5_seed,
    }


def source_audit() -> dict[str, Any]:
    baseline_source = BASELINE_STAN_FILE.read_text(encoding="utf-8")
    new_source = STAN_FILE.read_text(encoding="utf-8")
    data = load_json(BASELINE_STAN_DATA)
    new_count_likelihoods = re.findall(
        r"\bcount\s*~\s*([A-Za-z_][A-Za-z0-9_]*)", new_source
    )
    exact_data_keys = {
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
    }
    checks = {
        "baseline_multinomial": "count ~ multinomial(spatial_probability)" in baseline_source,
        "new_multinomial": "count ~ multinomial(spatial_probability)" in new_source,
        "baseline_area_softmax": "softmax(log(cell_area_km2) + gp_effect)" in baseline_source,
        "new_area_softmax": "softmax(log(cell_area_km2) + gp_effect)" in new_source,
        "baseline_centering": "gp_effect = raw_gp_effect - mean(raw_gp_effect)" in baseline_source,
        "new_centering": "gp_effect = raw_gp_effect - mean(raw_gp_effect)" in new_source,
        "baseline_jitter": "K[n, n] += jitter" in baseline_source,
        "new_jitter": "K[n, n] += jitter" in new_source,
        "baseline_rho_sampled": "rho ~ lognormal(rho_prior_logmean, rho_prior_logsd)" in baseline_source,
        "new_rho_sampled": "rho ~ lognormal(rho_prior_logmean, rho_prior_logsd)" in new_source,
        "baseline_alpha_prior": "alpha ~ normal(0, alpha_prior_sd)" in baseline_source,
        "new_alpha_prior": "alpha ~ normal(0, alpha_prior_sd)" in new_source,
        "baseline_eta_prior": "eta ~ std_normal()" in baseline_source,
        "new_eta_prior": "eta ~ std_normal()" in new_source,
        "new_has_only_multinomial_count_likelihood": new_count_likelihoods
        == ["multinomial"],
        "new_has_no_scalar_intercept_parameter": not bool(
            re.search(r"\breal(?:<[^>]+>)?\s+a\s*;", new_source)
        ),
        "baseline_data_keys_exact": set(data) == exact_data_keys,
        "new_data_adds_only_kernel_id": "int<lower=1, upper=7> kernel_id;" in new_source,
    }
    checks["all_passed"] = bool(all(checks.values()))
    if not checks["all_passed"]:
        raise RuntimeError(f"Source audit failed: {checks}")
    return checks


def validate_baseline() -> dict[str, Any]:
    ensure_directories()
    baseline_config, _ = configs()
    required = [
        CONFIG_FILE,
        STAN_FILE,
        BASELINE_CONFIG_FILE,
        BASELINE_STAN_FILE,
        BASELINE_STAN_DATA,
        BASELINE_GRID,
        BASELINE_EVENTS,
        BASELINE_PREPARATION_MANIFEST,
        BASELINE_RUN_MANIFEST,
        BASELINE_COMBINATION_MANIFEST,
        BASELINE_GP_POSTERIOR,
        BASELINE_ACTIVITY_POSTERIOR,
        BASELINE_VERIFICATION_REPORT,
        BASELINE_README,
        *WORKFLOW_FILES,
        *LEGACY_DEFINITION_FILES,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    run_manifest = load_json(BASELINE_RUN_MANIFEST)
    combination = load_json(BASELINE_COMBINATION_MANIFEST)
    preparation = load_json(BASELINE_PREPARATION_MANIFEST)
    data = load_json(BASELINE_STAN_DATA)
    grid = pd.read_csv(BASELINE_GRID)
    events = pd.read_csv(BASELINE_EVENTS)

    if run_manifest["stan_file_sha256"] != sha256_file(BASELINE_STAN_FILE):
        raise RuntimeError("The baseline baseline Stan hash is stale")
    if run_manifest["stan_data_sha256"] != sha256_file(BASELINE_STAN_DATA):
        raise RuntimeError("The baseline baseline data hash is stale")
    if len(events) != 1013 or events["l5_catalogue_row"].tolist() != list(range(1, 1014)):
        raise RuntimeError("The current event input is not the ordered 1013-event L5 catalogue")
    if len(grid) != 132 or grid["grid_id"].tolist() != list(range(1, 133)):
        raise RuntimeError("The current grid is not the baseline 132-cell grid")
    if int(grid["count"].sum()) != 1013 or sum(data["count"]) != 1013:
        raise RuntimeError("Cell counts do not sum to 1013")
    if not np.array_equal(grid["count"].to_numpy(int), np.asarray(data["count"], int)):
        raise RuntimeError("Saved grid counts differ from Stan data")
    expected_x = grid[["grid_x_km", "grid_y_km"]].to_numpy(float) / float(
        data["distance_scale_km"]
    )
    if not np.allclose(expected_x, np.asarray(data["x"], float), rtol=0, atol=1e-12):
        raise RuntimeError("Saved projected coordinates differ from Stan data")
    if not np.allclose(
        grid["grid_area_km2"].to_numpy(float),
        np.asarray(data["cell_area_km2"], float),
        rtol=0,
        atol=1e-10,
    ):
        raise RuntimeError("Saved physical cell areas differ from Stan data")
    if preparation["earthquakes"] != 1013 or preparation["grid_cells"] != 132:
        raise RuntimeError("Preparation manifest does not describe the current baseline")
    if preparation["selection"]["fixed_gp_observation_window"]:
        raise RuntimeError("A fixed GP observation window unexpectedly appears")
    if preparation["selection"]["additional_gp_year_filter"]:
        raise RuntimeError("An additional GP year filter unexpectedly appears")
    if preparation["selection"]["additional_gp_magnitude_filter"]:
        raise RuntimeError("An additional GP magnitude filter unexpectedly appears")

    priors = baseline_config["model"]["priors"]
    expected_prior_values = {
        "alpha_prior_sd": float(priors["alpha_sd"]),
        "rho_prior_logmean": float(priors["rho_logmean"]),
        "rho_prior_logsd": float(priors["rho_logsd"]),
    }
    for field, expected in expected_prior_values.items():
        if not math.isclose(float(data[field]), expected, rel_tol=0, abs_tol=1e-14):
            raise RuntimeError(f"Current Stan data {field} differs from baseline config")
    if not math.isclose(
        math.exp(float(data["rho_prior_logmean"])) * float(data["distance_scale_km"]),
        150.0,
        rel_tol=0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("The rho prior median is not 150 km")

    with np.load(BASELINE_GP_POSTERIOR, allow_pickle=False) as gp:
        gp_draws = int(len(gp["alpha"]))
        p_error = float(
            np.max(np.abs(gp["spatial_probability"].sum(axis=1) - 1.0))
        )
    pairing = pairing_context(baseline_config, gp_draws)
    with np.load(BASELINE_ACTIVITY_POSTERIOR, allow_pickle=False) as activity:
        if not np.array_equal(activity["gp_draw_index"], pairing["gp_index"]):
            raise RuntimeError("Baseline GP pairing differs from the current procedure")
        if not np.array_equal(activity["l5_draw_index"], pairing["l5_index"]):
            raise RuntimeError("Baseline L5 pairing differs from the current procedure")
        if not np.array_equal(activity["L5_total_activity"], pairing["l5_paired"]):
            raise RuntimeError("Baseline paired L5 sequence differs from source L5 draws")
        rate_error = float(
            np.max(
                np.abs(
                    activity["activity_rate_cell"].sum(axis=1)
                    - activity["L5_total_activity"]
                )
            )
        )
    if combination["l5_posterior_sha256"] != sha256_file(pairing["l5_path"]):
        raise RuntimeError("The baseline combination manifest has a stale L5 hash")

    source_checks = source_audit()
    protected_document = PROJECT_ROOT / "Draft.docx"
    result = {
        "baseline_files": {
            str(path.resolve()): sha256_file(path)
            for path in [
                *WORKFLOW_FILES,
                BASELINE_STAN_FILE,
                BASELINE_CONFIG_FILE,
                BASELINE_README,
                BASELINE_VERIFICATION_REPORT,
            ]
        },
        "source_input_hashes": {
            "stan_data": sha256_file(BASELINE_STAN_DATA),
            "grid": sha256_file(BASELINE_GRID),
            "events": sha256_file(BASELINE_EVENTS),
            "catalogue": preparation["catalogue_sha256"],
            "l5_stan_input": preparation["l5_stan_input_sha256"],
            "l5_posterior": sha256_file(pairing["l5_path"]),
            "baseline_gp_posterior": sha256_file(BASELINE_GP_POSTERIOR),
            "baseline_activity_posterior": sha256_file(BASELINE_ACTIVITY_POSTERIOR),
            "legacy_definition_files_only": {
                str(path.resolve()): sha256_file(path) for path in LEGACY_DEFINITION_FILES
            },
        },
        "events": len(events),
        "grid_cells": len(grid),
        "sum_counts": int(grid["count"].sum()),
        "zero_count_cells": int((grid["count"] == 0).sum()),
        "maximum_cell_count": int(grid["count"].max()),
        "posterior_draws": gp_draws,
        "probability_sum_error": p_error,
        "activity_rate_sum_error": rate_error,
        "rho_prior_median_km": 150.0,
        "rho_prior_logsd": float(data["rho_prior_logsd"]),
        "source_audit": source_checks,
        "pairing": {
            "method": combination["pairing"]["method"],
            "base_seed": pairing["base_seed"],
            "gp_seed": pairing["gp_seed"],
            "l5_seed": pairing["l5_seed"],
        },
        "protected_dissertation": str(protected_document),
        "protected_dissertation_sha256_before": (
            sha256_file(protected_document) if protected_document.is_file() else None
        ),
    }
    write_json(SENSITIVITY_ROOT / "output" / "baseline_compatibility.json", result)
    return result


def correlation_values(
    kernel_id: int, distance: np.ndarray | float, rho: np.ndarray | float
) -> np.ndarray:
    d = np.asarray(distance, dtype=float)
    length = np.asarray(rho, dtype=float)
    r = d / length
    if kernel_id == 1:
        return np.exp(-0.5 * np.square(r))
    if kernel_id == 2:
        z = np.sqrt(5.0) * r
        return (1.0 + z + (5.0 / 3.0) * np.square(r)) * np.exp(-z)
    if kernel_id == 3:
        z = np.sqrt(3.0) * r
        return (1.0 + z) * np.exp(-z)
    if kernel_id == 4:
        return np.exp(-r)
    if kernel_id == 5:
        return np.square(1.0 / (1.0 + 0.25 * np.square(r)))
    if kernel_id == 6:
        return 1.0 / (1.0 + np.square(r))
    if kernel_id == 7:
        return np.where(r < 1.0, np.power(1.0 - r, 4) * (1.0 + 4.0 * r), 0.0)
    raise ValueError(f"Unknown kernel id {kernel_id}")


def covariance_preflight(
    x: np.ndarray, kernel_id: int, alpha: float, rho: float, jitter: float
) -> dict[str, Any]:
    coordinates = np.asarray(x, dtype=float)
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=2
    )
    covariance = float(alpha) ** 2 * correlation_values(kernel_id, distances, rho)
    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    diagonal = covariance.copy()
    diagonal[np.diag_indices_from(diagonal)] += float(jitter)
    eigenvalue = float(np.linalg.eigvalsh(diagonal).min())
    cholesky_success = True
    try:
        np.linalg.cholesky(diagonal)
    except np.linalg.LinAlgError:
        cholesky_success = False
    return {
        "all_values_finite": bool(np.isfinite(covariance).all()),
        "maximum_symmetry_error": symmetry_error,
        "minimum_eigenvalue_after_jitter": eigenvalue,
        "cholesky_success": cholesky_success,
        "passed": bool(
            np.isfinite(covariance).all()
            and symmetry_error <= 1e-12
            and cholesky_success
        ),
    }


def diagnostic_summary(
    chain_files: list[Path],
    cmdstan_path: Path,
    summary_file: Path,
    diagnose_file: Path,
    max_treedepth: int,
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_file.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            str(cmdstan_path / "bin" / "stansummary.exe"),
            "--percentiles=2.5,50,97.5",
            "--sig_figs=8",
            f"--csv_filename={summary_file}",
            *[str(path.resolve()) for path in chain_files],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr.strip():
        (summary_file.parent / "stansummary_stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
    summary = pd.read_csv(
        summary_file, index_col=0, comment="#", float_precision="high"
    )
    summary = summary.loc[
        [name == "lp__" or not str(name).endswith("__") for name in summary.index]
    ]
    summary.index.name = "variable"
    summary.to_csv(summary_file)
    diagnosed = subprocess.run(
        [
            str(cmdstan_path / "bin" / "diagnose.exe"),
            *[str(path.resolve()) for path in chain_files],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    diagnose_file.write_text(diagnosed.stdout, encoding="utf-8")
    method = [
        pd.read_csv(
            path,
            comment="#",
            usecols=["divergent__", "treedepth__"],
            float_precision="high",
        )
        for path in chain_files
    ]
    divergences = int(sum(frame["divergent__"].sum() for frame in method))
    treedepth_hits = int(
        sum((frame["treedepth__"] >= max_treedepth).sum() for frame in method)
    )
    result = {
        "maximum_rhat": float(summary["R_hat"].dropna().max()),
        "minimum_bulk_ess": float(summary["ESS_bulk"].dropna().min()),
        "minimum_tail_ess": float(summary["ESS_tail"].dropna().min()),
        "divergences": divergences,
        "treedepth_hits": treedepth_hits,
        "key_parameter_rhat": {
            name: float(summary.loc[name, "R_hat"])
            for name in ("alpha", "rho", "rho_km")
        },
    }
    checks = {
        "rhat_pass": result["maximum_rhat"] <= float(thresholds["maximum_rhat"]),
        "bulk_ess_pass": result["minimum_bulk_ess"] >= float(thresholds["minimum_bulk_ess"]),
        "tail_ess_pass": result["minimum_tail_ess"] >= float(thresholds["minimum_tail_ess"]),
        "divergences_pass": divergences <= int(thresholds["maximum_divergences"]),
        "treedepth_pass": treedepth_hits <= int(thresholds["maximum_treedepth_hits"]),
    }
    result["thresholds"] = thresholds
    result["checks"] = checks
    result["all_passed"] = bool(all(checks.values()))
    return result, summary


def chain_elapsed_seconds(chain_file: Path) -> float:
    with chain_file.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 16384))
        tail = stream.read().decode("utf-8", errors="replace")
    matches = re.findall(r"#\s+([0-9.]+) seconds \(Total\)", tail)
    if not matches:
        raise RuntimeError(f"Could not extract runtime from {chain_file}")
    return float(matches[-1])


def equations_frame() -> pd.DataFrame:
    config = load_json(CONFIG_FILE)
    return pd.DataFrame(
        [
            {
                "kernel": kernel,
                "kernel_id": int(config["kernels"][kernel]["id"]),
                "label": config["kernels"][kernel]["label"],
                "normalised_correlation_equation": config["kernels"][kernel]["equation"],
                "covariance_equation": "K(d) = alpha^2 R(d)",
            }
            for kernel in config["kernel_order"]
        ]
    )
