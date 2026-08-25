"""Run one-at-a-time prior sweeps for the multinomial GP."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path

import cmdstanpy
import numpy as np

from _prior_common import (
    BASELINE_ACTIVITY_POSTERIOR,
    BASELINE_CONFIG,
    BASELINE_GP_POSTERIOR,
    BASELINE_RUN_MANIFEST,
    BASELINE_STAN_DATA,
    SENSITIVITY_ROOT,
    STAN_FILE,
    baseline_value,
    case_directory,
    chain_elapsed_seconds,
    diagnostic_summary,
    ensure_directories,
    load_configs,
    load_json,
    make_case_data,
    pairing_indices,
    posterior_summary,
    sha256_file,
    source_audit,
    sweep_values,
    validate_baseline,
    write_json,
)


SWEEPS = ("alpha", "rho_median", "rho_logsd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", nargs="*", choices=SWEEPS)
    parser.add_argument("--values", nargs="*", type=float)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def baseline_mcmc(
    run_manifest: dict, thresholds: dict
) -> dict:
    diagnostics = run_manifest["diagnostics"]
    result = {
        "maximum_rhat": float(diagnostics["maximum_rhat"]),
        "minimum_bulk_ess": float(diagnostics["minimum_bulk_ess"]),
        "minimum_tail_ess": float(diagnostics["minimum_tail_ess"]),
        "divergences": int(diagnostics["total_divergences"]),
        "treedepth_hits": int(diagnostics["maximum_treedepth_warnings"]),
    }
    result["thresholds"] = thresholds
    result["checks"] = {
        "rhat_pass": result["maximum_rhat"] <= float(thresholds["maximum_rhat"]),
        "bulk_ess_pass": result["minimum_bulk_ess"] >= float(thresholds["minimum_bulk_ess"]),
        "tail_ess_pass": result["minimum_tail_ess"] >= float(thresholds["minimum_tail_ess"]),
        "divergences_pass": result["divergences"] <= int(thresholds["maximum_divergences"]),
        "treedepth_pass": result["treedepth_hits"] <= int(thresholds["maximum_treedepth_hits"]),
    }
    result["all_passed"] = bool(all(result["checks"].values()))
    return result


def save_pairing(
    gp_index: np.ndarray,
    l5_index: np.ndarray,
    l5_paired: np.ndarray,
    l5_path: Path,
    pairing_seeds: dict,
) -> None:
    target = SENSITIVITY_ROOT / "output" / "common_l5_pairing.npz"
    if target.is_file():
        with np.load(target, allow_pickle=False) as loaded:
            if not (
                np.array_equal(loaded["gp_draw_index"], gp_index)
                and np.array_equal(loaded["l5_draw_index"], l5_index)
                and np.array_equal(loaded["L5_total_activity"], l5_paired)
            ):
                raise RuntimeError("Saved common L5 pairing differs from current procedure")
    else:
        np.savez_compressed(
            target,
            gp_draw_index=gp_index,
            l5_draw_index=l5_index,
            L5_total_activity=l5_paired,
        )
    write_json(
        SENSITIVITY_ROOT / "output" / "common_l5_pairing_manifest.json",
        {
            "method": "Current 03_combine_with_l5.py independent permutations without replacement",
            "draws": int(len(l5_paired)),
            "l5_posterior": str(l5_path),
            "l5_posterior_sha256": sha256_file(l5_path),
            "pairing_seeds": pairing_seeds,
            "all_gp_draws_used_once": np.unique(gp_index).size == len(gp_index),
            "all_l5_draws_used_once": np.unique(l5_index).size == len(l5_index),
        },
    )


def run_case(
    model: cmdstanpy.CmdStanModel,
    sweep: str,
    value: float,
    baseline_config: dict,
    sensitivity_config: dict,
    baseline_data: dict,
    baseline_audit: dict,
    source_checks: dict,
    gp_index: np.ndarray,
    l5_index: np.ndarray,
    l5_paired: np.ndarray,
    l5_path: Path,
    cmdstan_path: Path,
    prepare_only: bool,
) -> None:
    case_dir = case_directory(sweep, value)
    input_dir = case_dir / "input"
    stan_dir = case_dir / "stan"
    posterior_dir = case_dir / "posterior"
    tables_dir = case_dir / "tables"
    for path in (input_dir, stan_dir, posterior_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)
    status_file = case_dir / "status.json"
    started = time.perf_counter()
    is_baseline = math.isclose(value, baseline_value(sensitivity_config, sweep))
    data, prior_change = make_case_data(baseline_data, sweep, value)
    write_json(input_dir / "stan_data.json", data)
    expected_change = [] if is_baseline else [
        {"alpha": "alpha_prior_sd", "rho_median": "rho_prior_logmean", "rho_logsd": "rho_prior_logsd"}[sweep]
    ]
    if prior_change["changed_fields"] != expected_change:
        raise RuntimeError(
            f"{sweep}={value}: changed fields {prior_change['changed_fields']} != {expected_change}"
        )
    preparation = {
        "sweep": sweep,
        "tested_value": value,
        "is_baseline": is_baseline,
        "grid_cells": int(data["N"]),
        "sum_cell_counts": int(sum(data["count"])),
        "prior_change": prior_change,
        "case_stan_data_sha256": sha256_file(input_dir / "stan_data.json"),
        "baseline_stan_data_sha256": sha256_file(BASELINE_STAN_DATA),
    }
    write_json(input_dir / "preparation_summary.json", preparation)
    if prepare_only:
        write_json(status_file, {"status": "prepared_only", **preparation})
        print(
            f"PREPARED sweep={sweep} value={value:g} baseline={is_baseline} change={expected_change}",
            flush=True,
        )
        return
    write_json(status_file, {"status": "sampling", **preparation})
    try:
        sampling = baseline_config["sampling"]
        thresholds = sensitivity_config["diagnostic_thresholds"]
        if is_baseline:
            run_manifest = load_json(BASELINE_RUN_MANIFEST)
            chain_files = [Path(path) for path in run_manifest["chain_csv_files"]]
            with np.load(BASELINE_GP_POSTERIOR, allow_pickle=False) as loaded:
                alpha = loaded["alpha"]
                rho = loaded["rho"]
                rho_km = loaded["rho_km"]
                gp_effect = loaded["gp_effect"]
                spatial_probability = loaded["spatial_probability"]
            with np.load(BASELINE_ACTIVITY_POSTERIOR, allow_pickle=False) as loaded:
                activity_rate = loaded["activity_rate_cell"]
                paired_probability = loaded["spatial_probability"]
                if not np.array_equal(loaded["gp_draw_index"], gp_index):
                    raise RuntimeError("Reused baseline GP draw pairing differs")
                if not np.array_equal(loaded["l5_draw_index"], l5_index):
                    raise RuntimeError("Reused baseline L5 draw pairing differs")
            diagnostics = baseline_mcmc(run_manifest, thresholds)
            sampling_walltime = max(chain_elapsed_seconds(path) for path in chain_files)
            posterior_source = {
                "reused_baseline_baseline": True,
                "gp_posterior": str(BASELINE_GP_POSTERIOR),
                "gp_posterior_sha256": sha256_file(BASELINE_GP_POSTERIOR),
                "activity_posterior": str(BASELINE_ACTIVITY_POSTERIOR),
                "activity_posterior_sha256": sha256_file(BASELINE_ACTIVITY_POSTERIOR),
            }
        else:
            fit = model.sample(
                data=str(input_dir / "stan_data.json"),
                chains=int(sampling["chains"]),
                parallel_chains=int(sampling["parallel_chains"]),
                iter_warmup=int(sampling["warmup_per_chain"]),
                iter_sampling=int(sampling["sampling_per_chain"]),
                seed=int(sampling["seed"]),
                adapt_delta=float(sampling["adapt_delta"]),
                max_treedepth=int(sampling["max_treedepth"]),
                sig_figs=int(sampling["sig_figs"]),
                refresh=int(sampling["refresh"]),
                output_dir=str(stan_dir),
                show_progress=True,
            )
            alpha = fit.stan_variable("alpha")
            rho = fit.stan_variable("rho")
            rho_km = fit.stan_variable("rho_km")
            gp_effect = fit.stan_variable("gp_effect")
            spatial_probability = fit.stan_variable("spatial_probability")
            chain_files = [Path(path).resolve() for path in fit.runset.csv_files]
            diagnostics, _ = diagnostic_summary(
                chain_files,
                cmdstan_path,
                tables_dir / "mcmc_summary.csv",
                stan_dir / "diagnose.txt",
                int(sampling["max_treedepth"]),
                thresholds,
            )
            sampling_walltime = max(chain_elapsed_seconds(path) for path in chain_files)
            paired_probability = spatial_probability[gp_index]
            activity_rate = paired_probability * l5_paired[:, None]
            np.savez_compressed(
                posterior_dir / "gp_posterior_draws.npz",
                alpha=alpha,
                rho=rho,
                rho_km=rho_km,
                gp_effect=gp_effect,
                spatial_probability=spatial_probability,
            )
            np.savez_compressed(
                posterior_dir / "final_activity_rate_draws.npz",
                gp_effect=gp_effect[gp_index],
                spatial_probability=paired_probability,
                activity_rate_cell=activity_rate,
                alpha=alpha[gp_index],
                rho=rho[gp_index],
                rho_km=rho_km[gp_index],
                L5_total_activity=l5_paired,
                gp_draw_index=gp_index,
                l5_draw_index=l5_index,
            )
            posterior_source = {
                "reused_baseline_baseline": False,
                "gp_posterior": str(posterior_dir / "gp_posterior_draws.npz"),
                "activity_posterior": str(posterior_dir / "final_activity_rate_draws.npz"),
            }
            del fit
        tolerance_p = float(baseline_config["validation"]["probability_sum_tolerance"])
        tolerance_rate = float(baseline_config["l5_combination"]["sum_tolerance"])
        probability_error = float(np.max(np.abs(spatial_probability.sum(axis=1) - 1.0)))
        paired_probability_error = float(
            np.max(np.abs(paired_probability.sum(axis=1) - 1.0))
        )
        rate_error = float(
            np.max(np.abs(activity_rate.sum(axis=1) - l5_paired))
        )
        centering_error = float(np.max(np.abs(gp_effect.mean(axis=1))))
        if probability_error >= tolerance_p or paired_probability_error >= tolerance_p:
            raise RuntimeError(f"Probability conservation failed for {sweep}={value}")
        if rate_error >= tolerance_rate:
            raise RuntimeError(f"Activity-rate conservation failed for {sweep}={value}")
        summary = {
            **preparation,
            "posterior_draws": int(len(alpha)),
            "alpha": posterior_summary(alpha),
            "rho_km": posterior_summary(rho_km),
            "mcmc": diagnostics,
            "runtime_seconds": sampling_walltime,
            "validation": {
                "all_1013_events_retained": int(sum(data["count"])) == 1013,
                "same_132_cell_grid": int(data["N"]) == 132,
                "every_probability_draw_sums_to_one": True,
                "maximum_probability_sum_error": probability_error,
                "maximum_paired_probability_sum_error": paired_probability_error,
                "every_activity_rate_draw_sums_to_l5": True,
                "maximum_activity_rate_sum_error": rate_error,
                "maximum_gp_effect_centering_error": centering_error,
            },
            "invariance": {
                "baseline_source_checks": source_checks,
                "baseline_audit_hash": sha256_file(SENSITIVITY_ROOT / "output" / "baseline_compatibility.json"),
                "baseline_config": str(BASELINE_CONFIG),
                "baseline_config_sha256": sha256_file(BASELINE_CONFIG),
                "stan_file": str(STAN_FILE),
                "stan_file_sha256": sha256_file(STAN_FILE),
                "baseline_stan_data_sha256": sha256_file(BASELINE_STAN_DATA),
                "catalogue_sha256": baseline_audit["catalogue_sha256"],
                "l5_input_sha256": baseline_audit["l5_input_sha256"],
                "l5_posterior": str(l5_path),
                "l5_posterior_sha256": sha256_file(l5_path),
                "sampling": sampling,
                "only_changed_fields": prior_change["changed_fields"],
            },
            "posterior_source": posterior_source,
            "chain_csv_files": [str(path) for path in chain_files],
        }
        write_json(case_dir / "run_summary.json", summary)
        write_json(
            status_file,
            {
                "status": "completed",
                "sweep": sweep,
                "tested_value": value,
                "is_baseline": is_baseline,
                "mcmc_all_passed": diagnostics["all_passed"],
                "runtime_seconds": sampling_walltime,
            },
        )
        print(
            f"COMPLETED sweep={sweep} value={value:g} baseline={is_baseline} "
            f"Rhat={diagnostics['maximum_rhat']:.5f} "
            f"bulk={diagnostics['minimum_bulk_ess']:.1f} "
            f"time={sampling_walltime / 60:.2f}min",
            flush=True,
        )
        del alpha, rho, rho_km, gp_effect, spatial_probability, activity_rate, paired_probability
        gc.collect()
    except Exception as exc:
        write_json(
            status_file,
            {
                "status": "failed",
                "sweep": sweep,
                "tested_value": value,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "pipeline_elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


def main() -> int:
    args = parse_args()
    ensure_directories()
    baseline_config, sensitivity_config = load_configs()
    source_checks = source_audit(baseline_config)
    baseline_audit = validate_baseline()
    baseline_data = load_json(BASELINE_STAN_DATA)
    sweeps = tuple(args.sweeps or SWEEPS)
    configured_cmdstan = baseline_config["sampling"].get("cmdstan_path")
    cmdstan_path = (
        Path(configured_cmdstan).expanduser()
        if configured_cmdstan
        else Path(cmdstanpy.cmdstan_path())
    )
    cmdstanpy.set_cmdstan_path(str(cmdstan_path))
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
    gp_draws = int(baseline_audit["posterior_draws"])
    gp_index, l5_index, l5_paired, l5_path, pairing_seeds = pairing_indices(
        baseline_config, gp_draws
    )
    save_pairing(gp_index, l5_index, l5_paired, l5_path, pairing_seeds)
    for sweep in sweeps:
        allowed = sweep_values(sensitivity_config, sweep)
        values = [float(value) for value in (args.values or allowed)]
        if any(value not in allowed for value in values):
            raise ValueError(f"Invalid {sweep} values; allowed={allowed}")
        for value in values:
            status_file = case_directory(sweep, value) / "status.json"
            if args.skip_completed and status_file.is_file():
                status = load_json(status_file)
                if status.get("status") == "completed":
                    print(f"SKIPPED sweep={sweep} value={value:g}: completed", flush=True)
                    continue
            run_case(
                model,
                sweep,
                value,
                baseline_config,
                sensitivity_config,
                baseline_data,
                baseline_audit,
                source_checks,
                gp_index,
                l5_index,
                l5_paired,
                l5_path,
                cmdstan_path,
                args.prepare_only,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
