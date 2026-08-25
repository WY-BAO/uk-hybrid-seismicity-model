"""Fit the seven multinomial GP covariance-kernel cases independently."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import cmdstanpy
import numpy as np
import pandas as pd

from _kernel_common import (
    BASELINE_ACTIVITY_POSTERIOR,
    BASELINE_CONFIG_FILE,
    BASELINE_GP_POSTERIOR,
    BASELINE_GRID,
    BASELINE_ROOT,
    BASELINE_STAN_DATA,
    CONFIG_FILE,
    LEGACY_DEFINITION_FILES,
    SENSITIVITY_ROOT,
    STAN_FILE,
    canonical_json_sha256,
    case_root,
    chain_elapsed_seconds,
    configs,
    covariance_preflight,
    diagnostic_summary,
    ensure_directories,
    equations_frame,
    kernel_order,
    kernel_spec,
    load_json,
    pairing_context,
    posterior_summary,
    sha256_file,
    validate_baseline,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["all"],
        help="Kernel slugs, 'alternatives', or 'all'.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Reuse only cases whose manifest hashes match the current workflow.",
    )
    return parser.parse_args()


def expand_requested(values: list[str]) -> list[str]:
    order = kernel_order()
    expanded: list[str] = []
    for value in values:
        if value == "all":
            candidates = order
        elif value == "alternatives":
            candidates = order[1:]
        else:
            candidates = [value]
        for candidate in candidates:
            if candidate not in order:
                raise ValueError(f"Unknown kernel {candidate!r}")
            if candidate not in expanded:
                expanded.append(candidate)
    return [kernel for kernel in order if kernel in expanded]


def completed_case_is_current(kernel: str, baseline_data_hash: str) -> bool:
    root = case_root(kernel)
    manifest_file = root / "run_manifest.json"
    required = [
        manifest_file,
        root / "posterior" / "gp_posterior_draws.npz",
        root / "posterior" / "activity_posterior_draws.npz",
        root / "tables" / "mcmc_summary.csv",
        root / "diagnose.txt",
    ]
    if not all(path.is_file() for path in required):
        return False
    manifest = load_json(manifest_file)
    return bool(
        manifest.get("kernel") == kernel
        and manifest.get("new_stan_sha256") == sha256_file(STAN_FILE)
        and manifest.get("baseline_stan_data_sha256") == baseline_data_hash
        and manifest.get("complete") is True
    )


def preflight() -> pd.DataFrame:
    baseline_data = load_json(BASELINE_STAN_DATA)
    config = load_json(CONFIG_FILE)
    x = np.asarray(baseline_data["x"], dtype=float)
    alpha = 1.0
    rho = float(np.exp(float(baseline_data["rho_prior_logmean"])))
    rows = []
    for kernel in config["kernel_order"]:
        spec = config["kernels"][kernel]
        validation = covariance_preflight(
            x,
            int(spec["id"]),
            alpha,
            rho,
            float(baseline_data["jitter"]),
        )
        rows.append(
            {
                "kernel": kernel,
                "kernel_id": int(spec["id"]),
                "label": spec["label"],
                "equation": spec["equation"],
                "alpha": alpha,
                "rho_stan_units": rho,
                **validation,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(SENSITIVITY_ROOT / "tables" / "covariance_preflight.csv", index=False)
    equations_frame().to_csv(
        SENSITIVITY_ROOT / "tables" / "implemented_kernel_equations.csv", index=False
    )
    if not bool(frame["passed"].all()):
        raise RuntimeError(
            "Covariance preflight failed for "
            + ", ".join(frame.loc[~frame["passed"], "kernel"])
        )

    main_legacy = LEGACY_DEFINITION_FILES[0].read_text(encoding="utf-8")
    correction = LEGACY_DEFINITION_FILES[1].read_text(encoding="utf-8")
    legacy_audit = {
        "definition_files_inspected_only": {
            str(path.resolve()): sha256_file(path) for path in LEGACY_DEFINITION_FILES
        },
        "main_legacy_rq_comment_identifies_q1": "Rational quadratic with fixed q = 1" in main_legacy,
        "main_legacy_rq_q1_expression_present": "inv(1 + 0.5 * square(r))" in main_legacy,
        "isolated_q2_correction_present": "square(inv(1 + 0.25 * square(r)))" in correction,
        "new_analysis_uses_requested_q2": "square(inv(1 + 0.25 * square(r)))"
        in STAN_FILE.read_text(encoding="utf-8"),
        "legacy_numerical_inputs_or_results_loaded": False,
        "reported_difference": config["legacy_definition_audit"]["discrepancy"],
    }
    legacy_audit["all_definition_checks_passed"] = bool(
        all(
            legacy_audit[name]
            for name in (
                "main_legacy_rq_comment_identifies_q1",
                "main_legacy_rq_q1_expression_present",
                "isolated_q2_correction_present",
                "new_analysis_uses_requested_q2",
            )
        )
    )
    write_json(SENSITIVITY_ROOT / "output" / "legacy_definition_audit.json", legacy_audit)
    if not legacy_audit["all_definition_checks_passed"]:
        raise RuntimeError("Legacy kernel-definition audit failed")
    return frame


def cell_summary(
    grid: pd.DataFrame,
    gp_effect: np.ndarray,
    probability: np.ndarray,
    activity_rate: np.ndarray,
) -> pd.DataFrame:
    result = grid.copy()
    for prefix, values in (
        ("gp_effect", gp_effect),
        ("spatial_probability", probability),
        ("activity_rate_per_year", activity_rate),
    ):
        result[f"{prefix}_mean"] = np.mean(values, axis=0)
        result[f"{prefix}_median"] = np.quantile(values, 0.5, axis=0)
        result[f"{prefix}_q025"] = np.quantile(values, 0.025, axis=0)
        result[f"{prefix}_q975"] = np.quantile(values, 0.975, axis=0)
    result["activity_rate_95pct_width"] = (
        result["activity_rate_per_year_q975"]
        - result["activity_rate_per_year_q025"]
    )
    return result


def run_kernel(
    kernel: str,
    model: cmdstanpy.CmdStanModel,
    baseline_config: dict[str, Any],
    sensitivity_config: dict[str, Any],
    baseline_data: dict[str, Any],
    pairing: dict[str, Any],
    grid: pd.DataFrame,
    cmdstan_path: Path,
) -> dict[str, Any]:
    spec = sensitivity_config["kernels"][kernel]
    root = case_root(kernel)
    case_data = copy.deepcopy(baseline_data)
    case_data["kernel_id"] = int(spec["id"])
    unchanged = {name: case_data[name] for name in baseline_data}
    if unchanged != baseline_data or set(case_data) != set(baseline_data) | {"kernel_id"}:
        raise RuntimeError(f"{kernel}: a non-kernel Stan data field changed")
    data_file = root / "stan_data.json"
    write_json(data_file, case_data)
    data_difference = {
        "kernel": kernel,
        "kernel_id": int(spec["id"]),
        "only_added_field": "kernel_id",
        "all_baseline_fields_exactly_equal": True,
        "baseline_fields_canonical_sha256": canonical_json_sha256(baseline_data),
        "case_baseline_fields_canonical_sha256": canonical_json_sha256(unchanged),
        "only_model_change": "normalised covariance correlation function R(d)",
    }
    write_json(root / "data_difference.json", data_difference)

    sampling = baseline_config["sampling"]
    started = time.perf_counter()
    fit = model.sample(
        data=str(data_file),
        chains=int(sampling["chains"]),
        parallel_chains=int(sampling["parallel_chains"]),
        iter_warmup=int(sampling["warmup_per_chain"]),
        iter_sampling=int(sampling["sampling_per_chain"]),
        seed=int(sampling["seed"]),
        adapt_delta=float(sampling["adapt_delta"]),
        max_treedepth=int(sampling["max_treedepth"]),
        sig_figs=int(sampling["sig_figs"]),
        refresh=int(sampling["refresh"]),
        output_dir=str(root / "stan"),
        show_progress=False,
    )
    runtime_seconds = time.perf_counter() - started
    variables = {
        name: fit.stan_variable(name)
        for name in ("alpha", "rho", "rho_km", "gp_effect", "spatial_probability")
    }
    probability = np.asarray(variables["spatial_probability"], dtype=float)
    gp_effect = np.asarray(variables["gp_effect"], dtype=float)
    cell_area = np.asarray(case_data["cell_area_km2"], dtype=float)
    log_weight = np.log(cell_area)[None, :] + gp_effect
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    python_probability = np.exp(log_weight)
    python_probability /= python_probability.sum(axis=1, keepdims=True)
    probability_sum_error = float(np.max(np.abs(probability.sum(axis=1) - 1.0)))
    probability_crosscheck_error = float(
        np.max(np.abs(probability - python_probability))
    )
    centering_error = float(np.max(np.abs(gp_effect.mean(axis=1))))
    if probability_sum_error > float(
        baseline_config["validation"]["probability_sum_tolerance"]
    ):
        raise RuntimeError(f"{kernel}: spatial probabilities do not sum to one")
    if probability_crosscheck_error > float(
        baseline_config["validation"]["python_stan_probability_tolerance"]
    ):
        raise RuntimeError(f"{kernel}: Stan/Python area-softmax cross-check failed")
    if centering_error > 1e-10:
        raise RuntimeError(f"{kernel}: centred GP field check failed")

    gp_index = np.asarray(pairing["gp_index"], dtype=int)
    l5_index = np.asarray(pairing["l5_index"], dtype=int)
    l5_paired = np.asarray(pairing["l5_paired"], dtype=float)
    paired_probability = probability[gp_index]
    activity_rate = paired_probability * l5_paired[:, None]
    rate_sum_error = float(
        np.max(np.abs(activity_rate.sum(axis=1) - l5_paired))
    )
    if rate_sum_error >= float(baseline_config["l5_combination"]["sum_tolerance"]):
        raise RuntimeError(f"{kernel}: final activity-rate draws do not sum to L5")

    gp_file = root / "posterior" / "gp_posterior_draws.npz"
    activity_file = root / "posterior" / "activity_posterior_draws.npz"
    np.savez_compressed(gp_file, **variables)
    np.savez_compressed(
        activity_file,
        alpha=variables["alpha"][gp_index],
        rho=variables["rho"][gp_index],
        rho_km=variables["rho_km"][gp_index],
        gp_effect=gp_effect[gp_index],
        spatial_probability=paired_probability,
        activity_rate_cell=activity_rate,
        L5_total_activity=l5_paired,
        gp_draw_index=gp_index,
        l5_draw_index=l5_index,
    )
    pd.DataFrame(
        {
            "alpha": variables["alpha"],
            "rho": variables["rho"],
            "rho_km": variables["rho_km"],
        }
    ).to_csv(root / "tables" / "posterior_alpha_rho_draws.csv", index=False)
    cell_summary(grid, gp_effect, probability, activity_rate).to_csv(
        root / "tables" / "cell_posterior_summary.csv", index=False
    )

    chain_files = [Path(path).resolve() for path in fit.runset.csv_files]
    stdout_files = [Path(path).resolve() for path in fit.runset.stdout_files]
    sampling_warning_lines = []
    for stdout_file in stdout_files:
        if not stdout_file.is_file():
            continue
        for line in stdout_file.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if "Exception:" in stripped or "Warning:" in stripped:
                sampling_warning_lines.append(
                    {"file": str(stdout_file), "message": stripped}
                )
    diagnostics, _ = diagnostic_summary(
        chain_files,
        cmdstan_path,
        root / "tables" / "mcmc_summary.csv",
        root / "diagnose.txt",
        int(sampling["max_treedepth"]),
        sensitivity_config["diagnostic_thresholds"],
    )
    chain_runtimes = [chain_elapsed_seconds(path) for path in chain_files]
    manifest = {
        "complete": True,
        "kernel": kernel,
        "kernel_id": int(spec["id"]),
        "kernel_label": spec["label"],
        "normalised_correlation_equation": spec["equation"],
        "covariance_equation": "K(d) = alpha^2 R(d)",
        "only_changed_quantity": "covariance kernel",
        "rho_sampled_independently_in_this_fit": True,
        "new_stan_file": str(STAN_FILE.resolve()),
        "new_stan_sha256": sha256_file(STAN_FILE),
        "baseline_stan_data": str(BASELINE_STAN_DATA.resolve()),
        "baseline_stan_data_sha256": sha256_file(BASELINE_STAN_DATA),
        "case_stan_data": str(data_file.resolve()),
        "case_stan_data_sha256": sha256_file(data_file),
        "baseline_config": str(BASELINE_CONFIG_FILE.resolve()),
        "baseline_config_sha256": sha256_file(BASELINE_CONFIG_FILE),
        "sampling": sampling,
        "posterior_draws": int(len(variables["alpha"])),
        "chain_csv_files": [str(path) for path in chain_files],
        "chain_csv_sha256": {str(path): sha256_file(path) for path in chain_files},
        "chain_stdout_files": [str(path) for path in stdout_files],
        "sampling_warning_lines": sampling_warning_lines,
        "runtime_seconds_wall": runtime_seconds,
        "runtime_minutes_wall": runtime_seconds / 60.0,
        "chain_runtime_seconds": chain_runtimes,
        "maximum_chain_runtime_seconds": max(chain_runtimes),
        "diagnostics": diagnostics,
        "priors": {
            "alpha": "Half-Normal(0, 1)",
            "rho": "LogNormal(log(150 km), 0.8)",
            "eta": "standard Normal",
            "alpha_prior_sd": float(case_data["alpha_prior_sd"]),
            "rho_prior_logmean_stan_units": float(case_data["rho_prior_logmean"]),
            "rho_prior_logsd": float(case_data["rho_prior_logsd"]),
        },
        "conservation": {
            "maximum_spatial_probability_sum_error": probability_sum_error,
            "maximum_python_stan_probability_discrepancy": probability_crosscheck_error,
            "maximum_gp_effect_centering_error": centering_error,
            "maximum_activity_rate_sum_error": rate_sum_error,
        },
        "l5_pairing": {
            "same_for_all_kernels": True,
            "base_seed": int(pairing["base_seed"]),
            "gp_seed": int(pairing["gp_seed"]),
            "l5_seed": int(pairing["l5_seed"]),
            "gp_index_sha256": hashlib_array(gp_index),
            "l5_index_sha256": hashlib_array(l5_index),
            "l5_paired_sha256": hashlib_array(l5_paired),
        },
        "posterior_summaries": {
            "alpha": posterior_summary(variables["alpha"]),
            "rho": posterior_summary(variables["rho"]),
            "rho_km": posterior_summary(variables["rho_km"]),
        },
        "outputs": {
            "gp_posterior": str(gp_file.resolve()),
            "activity_posterior": str(activity_file.resolve()),
            "mcmc_summary": str((root / "tables" / "mcmc_summary.csv").resolve()),
            "cell_summary": str((root / "tables" / "cell_posterior_summary.csv").resolve()),
            "diagnose": str((root / "diagnose.txt").resolve()),
        },
    }
    write_json(root / "run_manifest.json", manifest)
    print(
        f"{spec['label']}: alpha median={manifest['posterior_summaries']['alpha']['median']:.6g}; "
        f"rho median={manifest['posterior_summaries']['rho_km']['median']:.6g} km; "
        f"R-hat max={diagnostics['maximum_rhat']:.6g}; "
        f"divergences={diagnostics['divergences']}; "
        f"treedepth hits={diagnostics['treedepth_hits']}; "
        f"runtime={runtime_seconds / 60.0:.2f} min",
        flush=True,
    )
    return manifest


def hashlib_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = __import__("hashlib").sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def compare_eq_to_baseline(sensitivity_config: dict[str, Any]) -> dict[str, Any]:
    with np.load(BASELINE_GP_POSTERIOR, allow_pickle=False) as baseline_gp:
        baseline = {name: baseline_gp[name] for name in baseline_gp.files}
    with np.load(
        case_root("exp_quad") / "posterior" / "gp_posterior_draws.npz",
        allow_pickle=False,
    ) as new_gp:
        new = {name: new_gp[name] for name in new_gp.files}
    with np.load(BASELINE_ACTIVITY_POSTERIOR, allow_pickle=False) as baseline_activity:
        baseline_rate = baseline_activity["activity_rate_cell"].copy()
    with np.load(
        case_root("exp_quad") / "posterior" / "activity_posterior_draws.npz",
        allow_pickle=False,
    ) as new_activity:
        new_rate = new_activity["activity_rate_cell"].copy()

    baseline_alpha = posterior_summary(baseline["alpha"])
    new_alpha = posterior_summary(new["alpha"])
    baseline_rho = posterior_summary(baseline["rho_km"])
    new_rho = posterior_summary(new["rho_km"])
    baseline_p_mean = np.mean(baseline["spatial_probability"], axis=0)
    new_p_mean = np.mean(new["spatial_probability"], axis=0)
    baseline_rate_mean = np.mean(baseline_rate, axis=0)
    new_rate_mean = np.mean(new_rate, axis=0)
    p_difference = new_p_mean - baseline_p_mean
    rate_difference = new_rate_mean - baseline_rate_mean
    metrics = {
        "absolute_alpha_median_difference": abs(new_alpha["median"] - baseline_alpha["median"]),
        "absolute_rho_km_median_difference": abs(new_rho["median"] - baseline_rho["median"]),
        "probability_mean_pearson": float(np.corrcoef(new_p_mean, baseline_p_mean)[0, 1]),
        "probability_mean_rmse": float(np.sqrt(np.mean(np.square(p_difference)))),
        "probability_mean_maximum_absolute_difference": float(np.max(np.abs(p_difference))),
        "activity_mean_pearson": float(np.corrcoef(new_rate_mean, baseline_rate_mean)[0, 1]),
        "activity_mean_rmse_per_year": float(np.sqrt(np.mean(np.square(rate_difference)))),
        "activity_mean_maximum_absolute_difference_per_year": float(np.max(np.abs(rate_difference))),
    }
    threshold = sensitivity_config["eq_reproduction_thresholds"]
    checks = {
        "alpha_median_pass": metrics["absolute_alpha_median_difference"]
        <= float(threshold["maximum_absolute_alpha_median_difference"]),
        "rho_median_pass": metrics["absolute_rho_km_median_difference"]
        <= float(threshold["maximum_absolute_rho_km_median_difference"]),
        "probability_pearson_pass": metrics["probability_mean_pearson"]
        >= float(threshold["minimum_probability_mean_pearson"]),
        "probability_rmse_pass": metrics["probability_mean_rmse"]
        <= float(threshold["maximum_probability_mean_rmse"]),
        "activity_pearson_pass": metrics["activity_mean_pearson"]
        >= float(threshold["minimum_activity_mean_pearson"]),
        "activity_rmse_pass": metrics["activity_mean_rmse_per_year"]
        <= float(threshold["maximum_activity_mean_rmse_per_year"]),
    }
    result = {
        "passed": bool(all(checks.values())),
        "comparison_basis": "new independent custom-EQ MCMC versus baseline built-in-EQ multinomial baseline",
        "baseline": {"alpha": baseline_alpha, "rho_km": baseline_rho},
        "new_eq": {"alpha": new_alpha, "rho_km": new_rho},
        "metrics": metrics,
        "thresholds": threshold,
        "checks": checks,
        "cells_compared": 132,
    }
    write_json(SENSITIVITY_ROOT / "output" / "eq_baseline_reproduction.json", result)
    pd.DataFrame(
        {
            "grid_id": np.arange(1, 133),
            "baseline_probability_mean": baseline_p_mean,
            "new_eq_probability_mean": new_p_mean,
            "probability_difference": p_difference,
            "baseline_activity_rate_mean": baseline_rate_mean,
            "new_eq_activity_rate_mean": new_rate_mean,
            "activity_rate_difference": rate_difference,
        }
    ).to_csv(SENSITIVITY_ROOT / "tables" / "eq_baseline_cell_comparison.csv", index=False)
    if not result["passed"]:
        raise RuntimeError(f"New EQ workflow failed baseline reproduction: {result}")
    return result


def require_current_eq_reproduction() -> dict[str, Any]:
    path = SENSITIVITY_ROOT / "output" / "eq_baseline_reproduction.json"
    manifest_path = case_root("exp_quad") / "run_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Run and verify exp_quad before fitting alternative kernels")
    result = load_json(path)
    manifest = load_json(manifest_path)
    if not result.get("passed"):
        raise RuntimeError("The saved EQ reproduction check did not pass")
    if manifest.get("new_stan_sha256") != sha256_file(STAN_FILE):
        raise RuntimeError("The saved EQ fit does not match the current Stan source")
    if manifest.get("baseline_stan_data_sha256") != sha256_file(BASELINE_STAN_DATA):
        raise RuntimeError("The saved EQ fit does not match the current baseline data")
    return result


def main() -> int:
    args = parse_args()
    ensure_directories()
    baseline_check = validate_baseline()
    preflight_frame = preflight()
    baseline_config, sensitivity_config = configs()
    baseline_data = load_json(BASELINE_STAN_DATA)
    grid = pd.read_csv(BASELINE_GRID)
    pairing = pairing_context(baseline_config, int(baseline_check["posterior_draws"]))

    print("Implemented covariance equations:", flush=True)
    for row in equations_frame().itertuples(index=False):
        print(f"  {row.kernel_id}. {row.label}: {row.normalised_correlation_equation}; K(d) = alpha^2 R(d)", flush=True)
    print(
        "Model gate: current 1013-event, 132-cell, area-aware multinomial GP; "
        "rho is sampled independently in every fit.",
        flush=True,
    )

    requested = expand_requested(args.kernels)
    if any(kernel != "exp_quad" for kernel in requested) and "exp_quad" not in requested:
        require_current_eq_reproduction()

    configured_cmdstan = baseline_config["sampling"].get("cmdstan_path")
    cmdstan_path = (
        Path(configured_cmdstan).expanduser()
        if configured_cmdstan
        else Path(cmdstanpy.cmdstan_path())
    )
    cmdstanpy.set_cmdstan_path(str(cmdstan_path))
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
    baseline_data_hash = sha256_file(BASELINE_STAN_DATA)
    manifests = {}
    for kernel in requested:
        if args.skip_completed and completed_case_is_current(kernel, baseline_data_hash):
            print(f"{kernel_spec(kernel)['label']}: current completed fit reused", flush=True)
            manifests[kernel] = load_json(case_root(kernel) / "run_manifest.json")
        else:
            manifests[kernel] = run_kernel(
                kernel,
                model,
                baseline_config,
                sensitivity_config,
                baseline_data,
                pairing,
                grid,
                cmdstan_path,
            )
        if kernel == "exp_quad":
            comparison = compare_eq_to_baseline(sensitivity_config)
            print(
                "EQ reproduction passed: "
                f"alpha median difference={comparison['metrics']['absolute_alpha_median_difference']:.6g}; "
                f"rho median difference={comparison['metrics']['absolute_rho_km_median_difference']:.6g} km; "
                f"activity Pearson={comparison['metrics']['activity_mean_pearson']:.8f}",
                flush=True,
            )
        elif "exp_quad" in requested:
            require_current_eq_reproduction()

    write_json(
        SENSITIVITY_ROOT / "output" / "last_run.json",
        {
            "requested_kernels": requested,
            "baseline_validation": str(
                (SENSITIVITY_ROOT / "output" / "baseline_compatibility.json").resolve()
            ),
            "covariance_preflight_all_passed": bool(preflight_frame["passed"].all()),
            "completed_manifests": {
                kernel: str((case_root(kernel) / "run_manifest.json").resolve())
                for kernel in manifests
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
