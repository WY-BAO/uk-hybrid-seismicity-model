"""Fit the multinomial GP at each requested grid resolution."""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import time
import traceback
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd

from _grid_common import (
    BASELINE_CONFIG,
    SENSITIVITY_ROOT,
    STAN_FILE,
    assert_current_multinomial_source,
    build_case_data,
    case_directory,
    common_pairing_indices,
    degree_label,
    ensure_output_directories,
    load_configs,
    load_exact_catalogue,
    load_preparation_module,
    save_common_pairing_indices,
    sha256_file,
    summarise_draws,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="*", type=float)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--reuse-existing-chains",
        action="store_true",
        help="Finalize existing four-chain CSVs instead of sampling again",
    )
    return parser.parse_args()


def diagnostic_summary(
    chain_files: list[Path],
    cmdstan_path: Path,
    summary_file: Path,
    diagnose_file: Path,
    max_treedepth: int,
    thresholds: dict,
) -> tuple[dict, object]:
    chain_files = [Path(path).resolve() for path in chain_files]
    summary_file.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            str(cmdstan_path / "bin" / "stansummary.exe"),
            "--percentiles=2.5,50,97.5",
            "--sig_figs=8",
            f"--csv_filename={summary_file}",
            *[str(path) for path in chain_files],
        ],
        check=True,
        capture_output=True,
        text=True,
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
            *[str(path) for path in chain_files],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    diagnose_file.write_text(diagnosed.stdout, encoding="utf-8")
    finite_rhat = summary["R_hat"].dropna().to_numpy(float)
    finite_bulk = summary["ESS_bulk"].dropna().to_numpy(float)
    finite_tail = summary["ESS_tail"].dropna().to_numpy(float)
    method_frames = [
        pd.read_csv(
            path,
            comment="#",
            usecols=["divergent__", "treedepth__"],
            float_precision="high",
        )
        for path in chain_files
    ]
    divergences = int(sum(frame["divergent__"].sum() for frame in method_frames))
    treedepth_hits = int(
        sum((frame["treedepth__"] >= max_treedepth).sum() for frame in method_frames)
    )
    values = {
        "maximum_rhat": float(np.max(finite_rhat)),
        "minimum_bulk_ess": float(np.min(finite_bulk)),
        "minimum_tail_ess": float(np.min(finite_tail)),
        "divergences": divergences,
        "treedepth_hits": treedepth_hits,
    }
    checks = {
        "rhat_pass": values["maximum_rhat"] <= float(thresholds["maximum_rhat"]),
        "bulk_ess_pass": values["minimum_bulk_ess"] >= float(thresholds["minimum_bulk_ess"]),
        "tail_ess_pass": values["minimum_tail_ess"] >= float(thresholds["minimum_tail_ess"]),
        "divergences_pass": divergences <= int(thresholds["maximum_divergences"]),
        "treedepth_pass": treedepth_hits <= int(thresholds["maximum_treedepth_hits"]),
    }
    values["thresholds"] = thresholds
    values["checks"] = checks
    values["all_passed"] = bool(all(checks.values()))
    return values, summary


def chain_elapsed_seconds(chain_file: Path) -> float:
    """Read CmdStan's reported total elapsed time from the end of a chain CSV."""
    with chain_file.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 8192))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = re.findall(r"#\s+([0-9.]+) seconds \(Total\)", tail)
    if not matches:
        raise RuntimeError(f"Could not read CmdStan elapsed time from {chain_file}")
    return float(matches[-1])


def run_case(
    degree: float,
    model: cmdstanpy.CmdStanModel,
    baseline_config: dict,
    sensitivity_config: dict,
    preparation_module,
    events,
    catalogue_path: Path,
    l5_input_path: Path,
    source_checks: dict,
    cmdstan_path: Path,
    prepare_only: bool,
    reuse_existing_chains: bool,
) -> None:
    case_dir = case_directory(degree)
    input_dir = case_dir / "input"
    posterior_dir = case_dir / "posterior"
    stan_dir = case_dir / "stan"
    tables_dir = case_dir / "tables"
    for path in (input_dir, posterior_dir, stan_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)
    status_file = case_dir / "status.json"
    started = time.perf_counter()
    write_json(status_file, {"status": "preparing", "degree": degree})
    try:
        assigned, grid, data = build_case_data(
            degree, baseline_config, preparation_module, events
        )
        assigned.to_csv(input_dir / "assigned_earthquakes.csv", index=False)
        grid.to_csv(input_dir / "grid_cells.csv", index=False)
        write_json(input_dir / "stan_data.json", data)
        n_cells = int(len(grid))
        zero_cells = int((grid["count"] == 0).sum())
        preparation = {
            "degree": degree,
            "grid_cells": n_cells,
            "zero_count_cells": zero_cells,
            "zero_count_proportion": float(zero_cells / n_cells),
            "maximum_cell_count": int(grid["count"].max()),
            "earthquakes": int(len(assigned)),
            "sum_cell_counts": int(grid["count"].sum()),
            "every_earthquake_assigned_exactly_once": bool(
                len(assigned) == 1013 and assigned["event_grid_id"].notna().all()
            ),
            "partial_boundary_cells": bool(
                np.any((grid["lon_hi"] - grid["lon_lo"]).to_numpy(float) < degree - 1e-10)
                or np.any((grid["lat_hi"] - grid["lat_lo"]).to_numpy(float) < degree - 1e-10)
            ),
        }
        write_json(input_dir / "preparation_summary.json", preparation)
        if prepare_only:
            write_json(
                status_file,
                {
                    "status": "prepared_only",
                    "degree": degree,
                    "grid_cells": n_cells,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            print(
                f"PREPARED {degree_label(degree)} cells={n_cells} zero={zero_cells} "
                f"max_count={preparation['maximum_cell_count']}",
                flush=True,
            )
            return

        write_json(
            status_file,
            {"status": "sampling", "degree": degree, "grid_cells": n_cells},
        )
        sampling = baseline_config["sampling"]
        existing_chains = sorted(stan_dir.glob("*.csv"))
        fit = None
        if reuse_existing_chains:
            if len(existing_chains) != int(sampling["chains"]):
                raise RuntimeError(
                    f"degree={degree}: expected {sampling['chains']} existing chains, "
                    f"found {len(existing_chains)}"
                )
            posterior_file = posterior_dir / "gp_posterior_draws.npz"
            if not posterior_file.is_file():
                raise FileNotFoundError(
                    f"degree={degree}: existing posterior NPZ is required with reused chains"
                )
            with np.load(posterior_file, allow_pickle=False) as loaded:
                alpha = loaded["alpha"]
                rho = loaded["rho"]
                rho_km = loaded["rho_km"]
                gp_effect = loaded["gp_effect"]
                spatial_probability = loaded["spatial_probability"]
            chain_files = [path.resolve() for path in existing_chains]
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
        probability_sum_error = float(
            np.max(np.abs(spatial_probability.sum(axis=1) - 1.0))
        )
        centering_error = float(np.max(np.abs(gp_effect.mean(axis=1))))
        tolerance = float(sensitivity_config["conservation_tolerance"])
        if probability_sum_error >= tolerance or centering_error >= tolerance:
            raise RuntimeError(
                f"degree={degree}: GP validation failed, p={probability_sum_error}, "
                f"centering={centering_error}"
            )
        np.savez_compressed(
            posterior_dir / "gp_posterior_draws.npz",
            alpha=alpha,
            rho=rho,
            rho_km=rho_km,
            gp_effect=gp_effect,
            spatial_probability=spatial_probability,
        )
        gp_index, l5_index, l5_paired, l5_path = common_pairing_indices(
            baseline_config, len(alpha)
        )
        save_common_pairing_indices(
            gp_index, l5_index, l5_paired, l5_path, baseline_config
        )
        paired_probability = spatial_probability[gp_index]
        activity_rate = paired_probability * l5_paired[:, None]
        model_rate_error = float(
            np.max(np.abs(activity_rate.sum(axis=1) - l5_paired))
        )
        if model_rate_error >= tolerance:
            raise RuntimeError(
                f"degree={degree}: model-grid activity rates do not conserve L5"
            )
        np.savez_compressed(
            posterior_dir / "final_activity_rate_draws.npz",
            spatial_probability=paired_probability,
            activity_rate_cell=activity_rate,
            L5_total_activity=l5_paired,
            alpha=alpha[gp_index],
            rho=rho[gp_index],
            rho_km=rho_km[gp_index],
            gp_draw_index=gp_index,
            l5_draw_index=l5_index,
        )
        diagnostics, mcmc_summary = diagnostic_summary(
            chain_files,
            cmdstan_path,
            tables_dir / "mcmc_summary.csv",
            stan_dir / "diagnose.txt",
            int(sampling["max_treedepth"]),
            sensitivity_config["diagnostic_thresholds"],
        )
        pipeline_elapsed = time.perf_counter() - started
        sampling_walltime = max(chain_elapsed_seconds(path) for path in chain_files)
        summary = {
            **preparation,
            "alpha": summarise_draws(alpha),
            "rho_km": summarise_draws(rho_km),
            "posterior_draws": int(len(alpha)),
            "mcmc": diagnostics,
            "validation": {
                "all_1013_events_assigned_once": preparation[
                    "every_earthquake_assigned_exactly_once"
                ],
                "cell_counts_sum_to_1013": preparation["sum_cell_counts"] == 1013,
                "every_model_probability_draw_sums_to_one": True,
                "maximum_model_probability_sum_error": probability_sum_error,
                "maximum_gp_effect_centering_error": centering_error,
                "every_model_grid_rate_draw_sums_to_l5": True,
                "maximum_model_grid_rate_sum_error": model_rate_error,
                "baseline_source_checks": source_checks,
            },
            "invariance": {
                "only_intentional_configuration_change": "grid.degree",
                "baseline_config": str(BASELINE_CONFIG),
                "baseline_config_sha256": sha256_file(BASELINE_CONFIG),
                "stan_file": str(STAN_FILE),
                "stan_file_sha256": sha256_file(STAN_FILE),
                "catalogue": str(catalogue_path),
                "catalogue_sha256": sha256_file(catalogue_path),
                "l5_stan_input": str(l5_input_path),
                "l5_stan_input_sha256": sha256_file(l5_input_path),
                "l5_posterior": str(l5_path),
                "l5_posterior_sha256": sha256_file(l5_path),
                "kernel": baseline_config["model"]["kernel"],
                "priors": baseline_config["model"]["priors"],
                "jitter": baseline_config["model"]["jitter"],
                "coordinates": baseline_config["coordinates"],
                "domain": {
                    key: baseline_config["grid"][key]
                    for key in (
                        "longitude_min",
                        "longitude_max",
                        "latitude_min",
                        "latitude_max",
                    )
                },
                "sampling": sampling,
                "l5_combination": baseline_config["l5_combination"],
            },
            "sampling_walltime_seconds": sampling_walltime,
            "pipeline_elapsed_seconds": pipeline_elapsed,
            "chain_csv_files": [str(path) for path in chain_files],
            "reused_existing_chain_csvs": bool(reuse_existing_chains),
        }
        write_json(case_dir / "run_summary.json", summary)
        write_json(
            status_file,
            {
                "status": "completed",
                "degree": degree,
                "grid_cells": n_cells,
                "mcmc_all_passed": diagnostics["all_passed"],
                "sampling_walltime_seconds": sampling_walltime,
                "pipeline_elapsed_seconds": pipeline_elapsed,
            },
        )
        print(
            f"COMPLETED {degree_label(degree)} cells={n_cells} "
            f"Rhat={diagnostics['maximum_rhat']:.5f} "
            f"bulk={diagnostics['minimum_bulk_ess']:.1f} "
            f"sampling_time={sampling_walltime / 60:.1f}min",
            flush=True,
        )
        del alpha, rho, rho_km, gp_effect, spatial_probability, activity_rate
        if fit is not None:
            del fit
        gc.collect()
    except Exception as exc:
        write_json(
            status_file,
            {
                "status": "failed",
                "degree": degree,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


def main() -> int:
    args = parse_args()
    ensure_output_directories()
    baseline_config, sensitivity_config = load_configs()
    requested = (
        args.resolutions
        if args.resolutions
        else [float(value) for value in sensitivity_config["resolutions_degrees"]]
    )
    allowed = {float(value) for value in sensitivity_config["resolutions_degrees"]}
    if not requested or any(float(value) not in allowed for value in requested):
        raise ValueError(f"Requested resolutions must be selected from {sorted(allowed)}")
    source_checks = assert_current_multinomial_source()
    preparation_module = load_preparation_module()
    events, catalogue_path, l5_input_path = load_exact_catalogue(
        baseline_config, preparation_module
    )
    sampling = baseline_config["sampling"]
    configured_cmdstan = sampling.get("cmdstan_path")
    cmdstan_path = (
        Path(configured_cmdstan).expanduser()
        if configured_cmdstan
        else Path(cmdstanpy.cmdstan_path())
    )
    if not cmdstan_path.is_dir():
        raise FileNotFoundError(cmdstan_path)
    cmdstanpy.set_cmdstan_path(str(cmdstan_path))
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))
    for degree in requested:
        status_path = case_directory(float(degree)) / "status.json"
        if args.skip_completed and status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            required_status = "prepared_only" if args.prepare_only else "completed"
            if status.get("status") == required_status:
                print(f"SKIPPED {degree_label(float(degree))}: {required_status}", flush=True)
                continue
        run_case(
            float(degree),
            model,
            baseline_config,
            sensitivity_config,
            preparation_module,
            events,
            catalogue_path,
            l5_input_path,
            source_checks,
            cmdstan_path,
            args.prepare_only,
            args.reuse_existing_chains,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
