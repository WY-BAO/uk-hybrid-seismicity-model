"""Compile and sample Hybrid B on the source-covered modelling domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd

from _common import (
    HYBRID_B_ROOT,
    ensure_directories,
    load_baseline_config,
    load_config,
    sha256_file,
    source_informed_softmax,
    validate_probability_array,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    baseline_config, baseline_config_path = load_baseline_config(config)
    ensure_directories()

    preflight_file = HYBRID_B_ROOT / "input" / "preflight_report.json"
    stan_data_file = HYBRID_B_ROOT / "input" / "stan_data.json"
    stan_file = HYBRID_B_ROOT / "stan" / "hybrid_b_gp.stan"
    if not preflight_file.is_file():
        raise FileNotFoundError("Run scripts/01_prepare_hybrid_b_data.py first")
    preflight = json.loads(preflight_file.read_text(encoding="utf-8"))
    if preflight.get("status") != "ready_for_sampling":
        raise RuntimeError(
            "Hybrid B sampling is blocked by preflight: "
            f"{preflight.get('status')}"
        )
    if not stan_data_file.is_file():
        raise FileNotFoundError("Successful preflight did not produce stan_data.json")

    stan_data = json.loads(stan_data_file.read_text(encoding="utf-8"))
    audit = pd.read_csv(HYBRID_B_ROOT / "input" / "p_source.csv").sort_values(
        "grid_id"
    )
    p_source_full = audit["p_source"].to_numpy(float)
    count_full = audit["count"].to_numpy(int)
    modelled_index = np.flatnonzero(p_source_full > 0.0)
    excluded_index = np.flatnonzero(p_source_full == 0.0)
    p_source_model = np.asarray(stan_data["p_source"], dtype=float)
    count_model = np.asarray(stan_data["count"], dtype=int)
    if int(stan_data["N"]) != modelled_index.size:
        raise RuntimeError("Stan N is not the exact 126-cell model domain")
    if not np.allclose(
        p_source_model,
        p_source_full[modelled_index],
        rtol=0.0,
        atol=2e-16,
    ):
        raise RuntimeError("Stan p_source rows do not match the full-grid model domain")
    if not np.array_equal(count_model, count_full[modelled_index]):
        raise RuntimeError("Stan counts do not match the full-grid model domain")
    if np.any(p_source_model <= 0.0):
        raise RuntimeError("A zero-source cell entered the Stan model domain")
    modelled_count = int(count_model.sum())
    excluded_count = int(count_full[excluded_index].sum())
    expected = config["expected"]
    if modelled_count != int(expected["hybrid_b_likelihood_earthquakes"]):
        raise RuntimeError("Hybrid B likelihood count is not 1005")
    if excluded_count != int(expected["hybrid_b_excluded_earthquakes"]):
        raise RuntimeError("Hybrid B excluded-event count is not 8")

    sampling = baseline_config["sampling"]
    configured_cmdstan = sampling.get("cmdstan_path")
    cmdstan_path = (
        Path(configured_cmdstan).expanduser()
        if configured_cmdstan
        else Path(cmdstanpy.cmdstan_path())
    )
    if not cmdstan_path.is_dir():
        raise FileNotFoundError(f"Configured CmdStan installation not found: {cmdstan_path}")
    cmdstanpy.set_cmdstan_path(str(cmdstan_path))
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))
    fit = model.sample(
        data=str(stan_data_file),
        chains=int(sampling["chains"]),
        parallel_chains=int(sampling["parallel_chains"]),
        iter_warmup=int(sampling["warmup_per_chain"]),
        iter_sampling=int(sampling["sampling_per_chain"]),
        seed=int(sampling["seed"]),
        adapt_delta=float(sampling["adapt_delta"]),
        max_treedepth=int(sampling["max_treedepth"]),
        sig_figs=int(sampling["sig_figs"]),
        refresh=int(sampling["refresh"]),
        output_dir=str(HYBRID_B_ROOT / "output" / "stan"),
        show_progress=True,
    )

    alpha = fit.stan_variable("alpha")
    rho = fit.stan_variable("rho")
    rho_km = fit.stan_variable("rho_km")
    f_correction_model = fit.stan_variable("f_correction")
    p_hybrid_model = fit.stan_variable("p_hybrid")
    f_correction = np.zeros((len(alpha), p_source_full.size), dtype=float)
    p_hybrid = np.zeros_like(f_correction)
    f_correction[:, modelled_index] = f_correction_model
    p_hybrid[:, modelled_index] = p_hybrid_model
    tolerance = float(config["validation"]["probability_sum_tolerance"])
    probability_sum_error = validate_probability_array(p_hybrid, tolerance)
    maximum_excluded_probability = float(
        np.max(np.abs(p_hybrid[:, excluded_index]))
    )
    if maximum_excluded_probability != 0.0:
        raise RuntimeError("A Hybrid B domain-excluded cell has nonzero probability")
    python_probability = source_informed_softmax(p_source_full, f_correction)
    python_stan_error = float(np.max(np.abs(p_hybrid - python_probability)))
    if python_stan_error > float(
        config["validation"]["python_stan_probability_tolerance"]
    ):
        raise RuntimeError(
            f"Python/Stan Hybrid probability discrepancy={python_stan_error:.17g}"
        )
    centring_error = float(
        np.max(np.abs(f_correction[:, modelled_index].mean(axis=1)))
    )
    if centring_error > 1e-10:
        raise RuntimeError(f"Centred GP correction check failed: {centring_error:.17g}")

    maximum_excluded_correction = float(
        np.max(np.abs(f_correction[:, excluded_index]))
    )
    if maximum_excluded_correction != 0.0:
        raise RuntimeError("A domain-excluded cell has a nonzero restored correction")
    zero_correction = source_informed_softmax(
        p_source_full, np.zeros_like(p_source_full)
    )
    zero_correction_error = float(
        np.max(np.abs(zero_correction - p_source_full))
    )
    if zero_correction_error > np.finfo(float).eps * 8:
        raise RuntimeError("Zero-correction test failed")

    posterior_file = HYBRID_B_ROOT / "output" / "posterior" / "hybrid_b_gp_draws.npz"
    np.savez_compressed(
        posterior_file,
        p_source=p_source_full,
        p_hybrid=p_hybrid,
        f_correction=f_correction,
        alpha=alpha,
        rho=rho,
        rho_km=rho_km,
        modelled_cell_index=modelled_index,
        excluded_cell_index=excluded_index,
    )
    summary = fit.summary()
    summary.index.name = "variable"
    summary_file = HYBRID_B_ROOT / "output" / "tables" / "hybrid_b_mcmc_summary.csv"
    summary.to_csv(summary_file)
    diagnose_text = fit.diagnose()
    diagnose_file = HYBRID_B_ROOT / "output" / "stan" / "diagnose.txt"
    diagnose_file.write_text(diagnose_text, encoding="utf-8")

    method_variables = fit.method_variables()
    max_treedepth = int(sampling["max_treedepth"])
    diagnostics = {
        "maximum_rhat": float(summary["R_hat"].max(skipna=True)),
        "minimum_bulk_ess": float(summary["ESS_bulk"].min(skipna=True)),
        "minimum_tail_ess": float(summary["ESS_tail"].min(skipna=True)),
        "total_divergences": int(np.asarray(method_variables["divergent__"]).sum()),
        "maximum_treedepth_warnings": int(
            np.count_nonzero(np.asarray(method_variables["treedepth__"]) >= max_treedepth)
        ),
        "alpha_rhat": float(summary.loc["alpha", "R_hat"]),
        "rho_rhat": float(summary.loc["rho", "R_hat"]),
    }
    manifest = {
        "status": "sampled",
        "configuration": str(config_path),
        "baseline_configuration": str(baseline_config_path),
        "stan_model": str(stan_file),
        "stan_model_sha256": sha256_file(stan_file),
        "stan_data": str(stan_data_file),
        "stan_data_sha256": sha256_file(stan_data_file),
        "sampling_reused_exactly_from_baseline": sampling,
        "posterior_draws": int(len(alpha)),
        "model_domain": {
            "full_grid_cells": int(p_source_full.size),
            "modelled_source_covered_cells": int(modelled_index.size),
            "excluded_zero_source_cells": int(excluded_index.size),
            "gp_covariance_dimension": int(stan_data["N"]),
            "full_catalogue_events": int(count_full.sum()),
            "modelled_likelihood_events": modelled_count,
            "excluded_events": excluded_count,
        },
        "chain_csv_files": [str(Path(path).resolve()) for path in fit.runset.csv_files],
        "diagnostics": diagnostics,
        "validation": {
            "maximum_p_hybrid_sum_error": probability_sum_error,
            "maximum_python_stan_probability_difference": python_stan_error,
            "maximum_modelled_f_correction_draw_mean": centring_error,
            "zero_correction_maximum_difference": zero_correction_error,
            "maximum_excluded_cell_probability": maximum_excluded_probability,
            "maximum_excluded_cell_correction": maximum_excluded_correction,
            "all_probabilities_finite_and_in_unit_interval": True,
        },
        "outputs": {
            "posterior": str(posterior_file),
            "summary": str(summary_file),
            "diagnose": str(diagnose_file),
        },
    }
    write_json(HYBRID_B_ROOT / "output" / "stan" / "run_manifest.json", manifest)
    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
