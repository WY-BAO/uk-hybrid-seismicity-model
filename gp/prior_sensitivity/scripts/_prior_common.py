"""Shared utilities for the current multinomial GP prior sensitivity."""

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
BASELINE_CONFIG = BASELINE_ROOT / "config" / "baseline_config.json"
SENSITIVITY_CONFIG = SENSITIVITY_ROOT / "config" / "prior_sensitivity_config.json"
STAN_FILE = BASELINE_ROOT / "stan" / "baseline_gp.stan"
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
WORKFLOW_FILES = [
    BASELINE_ROOT / "scripts" / "01_prepare_gp_data.py",
    BASELINE_ROOT / "scripts" / "02_run_gp.py",
    BASELINE_ROOT / "scripts" / "03_combine_with_l5.py",
    BASELINE_ROOT / "scripts" / "04_summarize_results.py",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_json(BASELINE_CONFIG), load_json(SENSITIVITY_CONFIG)


def resolve_baseline_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASELINE_ROOT / path).resolve()


def ensure_directories() -> None:
    for relative in ("runs", "output/tables", "output/figures", "output/audit"):
        (SENSITIVITY_ROOT / relative).mkdir(parents=True, exist_ok=True)


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


def pairing_indices(baseline_config: dict[str, Any], gp_draws: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, dict[str, int]]:
    l5_path = resolve_baseline_path(baseline_config["paths"]["l5_posterior_draws"])
    variable = str(baseline_config["l5_combination"]["variable"])
    frame = pd.read_csv(l5_path)
    if variable not in frame.columns:
        raise KeyError(f"{variable} is absent from {l5_path}")
    l5_values = frame[variable].to_numpy(float)
    if len(l5_values) != gp_draws:
        raise RuntimeError("Current GP and L5 posterior draw counts must both be 4000")
    base_seed = int(baseline_config["l5_combination"]["seed"])
    gp_seed = stable_seed(base_seed, "degree=1|gp")
    l5_seed = stable_seed(base_seed, "degree=1|l5")
    gp_index = np.random.default_rng(gp_seed).permutation(gp_draws)
    l5_index = np.random.default_rng(l5_seed).permutation(len(l5_values))
    return (
        gp_index,
        l5_index,
        l5_values[l5_index],
        l5_path,
        {"base_seed": base_seed, "gp_seed": gp_seed, "l5_seed": l5_seed},
    )


def source_audit(baseline_config: dict[str, Any]) -> dict[str, Any]:
    source = STAN_FILE.read_text(encoding="utf-8")
    lower = source.lower()
    data = load_json(BASELINE_STAN_DATA)
    checks = {
        "stan_file": str(STAN_FILE),
        "stan_sha256": sha256_file(STAN_FILE),
        "workflow_files": {str(path): sha256_file(path) for path in WORKFLOW_FILES},
        "multinomial_present": "count ~ multinomial(spatial_probability)" in source,
        "exp_quad_kernel_present": "gp_exp_quad_cov(x, alpha, rho)" in source,
        "area_aware_softmax_present": "softmax(log(cell_area_km2) + gp_effect)" in source,
        "centred_field_present": "gp_effect = raw_gp_effect - mean(raw_gp_effect)" in source,
        "eta_standard_normal_present": "eta ~ std_normal()" in source,
        "alpha_half_normal_present": "alpha ~ normal(0, alpha_prior_sd)" in source,
        "rho_lognormal_present": "rho ~ lognormal(rho_prior_logmean, rho_prior_logsd)" in source,
        "poisson_absent": "poisson" not in lower,
        "exposure_absent": "exposure" not in lower and "log_exposure_area" not in data,
        "intercept_absent": not bool(re.search(r"\breal(?:<[^>]+>)?\s+a\s*;", source)),
        "stan_data_keys_exact": set(data) == {
            "N", "D", "x", "count", "cell_area_km2", "distance_scale_km",
            "jitter", "alpha_prior_sd", "rho_prior_logmean", "rho_prior_logsd",
        },
        "expected_1013_events": sum(data["count"]) == 1013,
        "expected_132_cells": int(data["N"]) == 132,
        "baseline_alpha_sd": math.isclose(float(data["alpha_prior_sd"]), 1.0),
        "baseline_rho_median_km": math.isclose(
            math.exp(float(data["rho_prior_logmean"]))
            * float(data["distance_scale_km"]),
            150.0,
        ),
        "baseline_rho_logsd": math.isclose(float(data["rho_prior_logsd"]), 0.8),
    }
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    checks["all_source_checks_passed"] = bool(all(boolean_checks))
    if not checks["all_source_checks_passed"]:
        raise RuntimeError(f"Baseline source audit failed: {checks}")
    return checks


def validate_baseline() -> dict[str, Any]:
    baseline_config, _ = load_configs()
    required = [
        BASELINE_CONFIG, STAN_FILE, BASELINE_STAN_DATA, BASELINE_GRID,
        BASELINE_EVENTS, BASELINE_PREPARATION_MANIFEST, BASELINE_RUN_MANIFEST,
        BASELINE_COMBINATION_MANIFEST, BASELINE_GP_POSTERIOR,
        BASELINE_ACTIVITY_POSTERIOR, *WORKFLOW_FILES,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing current baseline artifacts: {missing}")
    run_manifest = load_json(BASELINE_RUN_MANIFEST)
    combination_manifest = load_json(BASELINE_COMBINATION_MANIFEST)
    preparation = load_json(BASELINE_PREPARATION_MANIFEST)
    if run_manifest["stan_file_sha256"] != sha256_file(STAN_FILE):
        raise RuntimeError("Baseline posterior Stan hash is stale")
    if run_manifest["stan_data_sha256"] != sha256_file(BASELINE_STAN_DATA):
        raise RuntimeError("Baseline posterior data hash is stale")
    grid = pd.read_csv(BASELINE_GRID)
    events = pd.read_csv(BASELINE_EVENTS)
    if len(events) != 1013 or events["l5_catalogue_row"].tolist() != list(range(1, 1014)):
        raise RuntimeError("Baseline event file is not the exact ordered 1013-event catalogue")
    if len(grid) != 132 or int(grid["count"].sum()) != 1013:
        raise RuntimeError("Baseline grid is not the validated 132-cell grid")
    with np.load(BASELINE_GP_POSTERIOR, allow_pickle=False) as gp:
        gp_draws = int(len(gp["alpha"]))
        gp_probability_error = float(
            np.max(np.abs(gp["spatial_probability"].sum(axis=1) - 1.0))
        )
    gp_index, l5_index, l5_paired, l5_path, seeds = pairing_indices(
        baseline_config, gp_draws
    )
    with np.load(BASELINE_ACTIVITY_POSTERIOR, allow_pickle=False) as activity:
        if not np.array_equal(activity["gp_draw_index"], gp_index):
            raise RuntimeError("Baseline GP pairing does not match current procedure")
        if not np.array_equal(activity["l5_draw_index"], l5_index):
            raise RuntimeError("Baseline L5 pairing does not match current procedure")
        rate_error = float(
            np.max(
                np.abs(
                    activity["activity_rate_cell"].sum(axis=1)
                    - activity["L5_total_activity"]
                )
            )
        )
        if not np.array_equal(activity["L5_total_activity"], l5_paired):
            raise RuntimeError("Baseline paired L5 draws differ from current procedure")
    result = {
        "baseline_config": str(BASELINE_CONFIG),
        "baseline_config_sha256": sha256_file(BASELINE_CONFIG),
        "stan_data_sha256": sha256_file(BASELINE_STAN_DATA),
        "stan_sha256": sha256_file(STAN_FILE),
        "catalogue_sha256": preparation["catalogue_sha256"],
        "l5_input_sha256": preparation["l5_stan_input_sha256"],
        "l5_posterior": str(l5_path),
        "l5_posterior_sha256": sha256_file(l5_path),
        "gp_posterior_sha256": sha256_file(BASELINE_GP_POSTERIOR),
        "activity_posterior_sha256": sha256_file(BASELINE_ACTIVITY_POSTERIOR),
        "events": len(events),
        "grid_cells": len(grid),
        "sum_counts": int(grid["count"].sum()),
        "posterior_draws": gp_draws,
        "maximum_probability_sum_error": gp_probability_error,
        "maximum_activity_rate_sum_error": rate_error,
        "pairing_seeds": seeds,
        "run_manifest": run_manifest,
        "combination_manifest": combination_manifest,
    }
    write_json(SENSITIVITY_ROOT / "output" / "baseline_compatibility.json", result)
    return result


def sweep_values(config: dict[str, Any], sweep: str) -> list[float]:
    key = {
        "alpha": "alpha_prior_sd_values",
        "rho_median": "rho_prior_median_km_values",
        "rho_logsd": "rho_prior_logsd_values",
    }[sweep]
    return [float(value) for value in config[key]]


def baseline_value(config: dict[str, Any], sweep: str) -> float:
    key = {
        "alpha": "alpha_prior_sd",
        "rho_median": "rho_prior_median_km",
        "rho_logsd": "rho_prior_logsd",
    }[sweep]
    return float(config["baseline"][key])


def value_label(value: float) -> str:
    text = f"{value:.10g}".replace("-", "m").replace(".", "p")
    return text


def case_directory(sweep: str, value: float) -> Path:
    return SENSITIVITY_ROOT / "runs" / sweep / f"value_{value_label(value)}"


def make_case_data(
    baseline_data: dict[str, Any], sweep: str, value: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(json.dumps(baseline_data))
    before = {
        "alpha_prior_sd": float(data["alpha_prior_sd"]),
        "rho_prior_logmean": float(data["rho_prior_logmean"]),
        "rho_prior_logsd": float(data["rho_prior_logsd"]),
    }
    if sweep == "alpha":
        data["alpha_prior_sd"] = float(value)
        requested = {"alpha_prior_sd": float(value)}
    elif sweep == "rho_median":
        scale = float(data["distance_scale_km"])
        data["rho_prior_logmean"] = float(math.log(float(value) / scale))
        requested = {
            "rho_prior_median_km": float(value),
            "rho_prior_logmean_stan_units": float(data["rho_prior_logmean"]),
        }
    elif sweep == "rho_logsd":
        data["rho_prior_logsd"] = float(value)
        requested = {"rho_prior_logsd": float(value)}
    else:
        raise ValueError(sweep)
    after = {
        "alpha_prior_sd": float(data["alpha_prior_sd"]),
        "rho_prior_logmean": float(data["rho_prior_logmean"]),
        "rho_prior_logsd": float(data["rho_prior_logsd"]),
    }
    changed = [name for name in before if not math.isclose(before[name], after[name])]
    return data, {"before": before, "after": after, "changed_fields": changed, "requested": requested}


def chain_elapsed_seconds(chain_file: Path) -> float:
    with chain_file.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 8192))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = re.findall(r"#\s+([0-9.]+) seconds \(Total\)", tail)
    if not matches:
        raise RuntimeError(f"Could not read elapsed time from {chain_file}")
    return float(matches[-1])


def diagnostic_summary(
    chain_files: list[Path],
    cmdstan_path: Path,
    summary_file: Path,
    diagnose_file: Path,
    max_treedepth: int,
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_file.unlink(missing_ok=True)
    subprocess.run(
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
    summary = pd.read_csv(summary_file, index_col=0, comment="#", float_precision="high")
    summary = summary.loc[
        [name == "lp__" or not str(name).endswith("__") for name in summary.index]
    ]
    summary.index.name = "variable"
    summary.to_csv(summary_file)
    diagnosed = subprocess.run(
        [str(cmdstan_path / "bin" / "diagnose.exe"), *[str(path.resolve()) for path in chain_files]],
        check=True,
        capture_output=True,
        text=True,
    )
    diagnose_file.write_text(diagnosed.stdout, encoding="utf-8")
    method = [
        pd.read_csv(path, comment="#", usecols=["divergent__", "treedepth__"], float_precision="high")
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
