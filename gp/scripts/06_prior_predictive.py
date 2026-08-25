"""Generate optional prior draws for the GP spatial probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import BASELINE_ROOT, load_config, posterior_summary, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = load_config(args.config)
    settings = config["prior_predictive"]
    priors = config["model"]["priors"]
    stan_data_file = BASELINE_ROOT / "input" / "stan_data.json"
    if not stan_data_file.is_file():
        raise FileNotFoundError("Run scripts/01_prepare_gp_data.py first")
    stan_data = json.loads(stan_data_file.read_text(encoding="utf-8"))

    draws = int(settings["draws"])
    rng = np.random.default_rng(int(settings["seed"]))
    alpha = np.abs(rng.normal(0.0, float(priors["alpha_sd"]), draws))
    rho = rng.lognormal(
        float(priors["rho_logmean"]), float(priors["rho_logsd"]), draws
    )
    coordinates = np.asarray(stan_data["x"], dtype=float)
    cell_area = np.asarray(stan_data["cell_area_km2"], dtype=float)
    jitter = float(stan_data["jitter"])
    squared_distance = np.sum(
        (coordinates[:, None, :] - coordinates[None, :, :]) ** 2,
        axis=2,
    )
    log_area = np.log(cell_area)
    spatial_probability = np.empty((draws, len(cell_area)), dtype=float)
    entropy = np.empty(draws, dtype=float)
    maximum_probability = np.empty(draws, dtype=float)
    effective_cells = np.empty(draws, dtype=float)
    for draw in range(draws):
        covariance = alpha[draw] ** 2 * np.exp(
            -0.5 * squared_distance / rho[draw] ** 2
        )
        covariance.flat[:: len(cell_area) + 1] += jitter
        raw_effect = np.linalg.cholesky(covariance) @ rng.normal(size=len(cell_area))
        centred_effect = raw_effect - raw_effect.mean()
        log_weight = log_area + centred_effect
        probability = np.exp(log_weight - np.max(log_weight))
        probability /= probability.sum()
        spatial_probability[draw] = probability
        entropy[draw] = -np.sum(probability * np.log(probability))
        maximum_probability[draw] = np.max(probability)
        effective_cells[draw] = 1.0 / np.sum(probability**2)

    probability_sum_error = float(
        np.max(np.abs(spatial_probability.sum(axis=1) - 1.0))
    )
    tolerance = float(config["validation"]["probability_sum_tolerance"])
    if probability_sum_error > tolerance:
        raise RuntimeError("Prior spatial probabilities do not sum to one")

    diagnostic_frame = pd.DataFrame(
        {
            "alpha": alpha,
            "rho_stan_units": rho,
            "rho_km": rho * float(config["coordinates"]["stan_unit_km"]),
            "spatial_probability_entropy": entropy,
            "maximum_cell_probability": maximum_probability,
            "effective_number_of_cells": effective_cells,
        }
    )
    diagnostic_file = (
        BASELINE_ROOT / "output" / "tables" / "prior_spatial_draw_diagnostics.csv"
    )
    diagnostic_frame.to_csv(diagnostic_file, index=False)

    cell_probability_file = (
        BASELINE_ROOT / "output" / "tables" / "prior_spatial_probability_summary.csv"
    )
    pd.DataFrame(
        {
            "grid_id": np.arange(1, len(cell_area) + 1),
            "prior_probability_mean": spatial_probability.mean(axis=0),
            "prior_probability_q025": np.quantile(
                spatial_probability, 0.025, axis=0
            ),
            "prior_probability_median": np.quantile(
                spatial_probability, 0.5, axis=0
            ),
            "prior_probability_q975": np.quantile(
                spatial_probability, 0.975, axis=0
            ),
        }
    ).to_csv(cell_probability_file, index=False)

    summary = {
        "purpose": "optional prior support for relative spatial allocation",
        "absolute_activity_rate_estimated": False,
        "observation_time_used": False,
        "draws": draws,
        "seed": int(settings["seed"]),
        "maximum_probability_sum_error": probability_sum_error,
        "summaries": {
            column: posterior_summary(diagnostic_frame[column].to_numpy(float))
            for column in diagnostic_frame.columns
        },
        "outputs": {
            "draw_diagnostics": str(diagnostic_file),
            "cell_probability_summary": str(cell_probability_file),
        },
    }
    write_json(
        BASELINE_ROOT / "output" / "tables" / "prior_predictive_summary.json",
        summary,
    )
    print(f"Prior spatial-probability draws = {draws}")
    print(f"Maximum probability sum error = {probability_sum_error:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
