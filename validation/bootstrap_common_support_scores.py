"""Year-block bootstrap for paired common-support holdout log scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
TABLES = OUTPUT / "tables"
SEED = 20260825
N_BOOTSTRAP = 20_000


def main() -> None:
    cells = pd.read_csv(TABLES / "model_cell_predictions_and_test_counts.csv").sort_values("grid_id")
    test = pd.read_csv(OUTPUT / "catalogue_test_2013_2022.csv")
    support = cells["source_support"].astype(bool).to_numpy()
    supported_grid_ids = set(cells.loc[support, "grid_id"].astype(int))
    test = test.loc[test["grid_id"].astype(int).isin(supported_grid_ids)].copy()
    test["test_year"] = np.floor(test["year_fraction"].to_numpy(float)).astype(int)

    model_columns = {
        "Source-zone": "source_rate_mean",
        "GP": "gp_rate_mean_pre2013",
        "Hybrid A equal": "hybrid_a_equal_rate_mean",
        "Hybrid A uncertainty": "hybrid_a_uncertainty_rate_mean",
        "Hybrid B": "hybrid_b_rate_mean_pre2013",
    }
    losses: dict[str, np.ndarray] = {}
    supported_ids = cells.loc[support, "grid_id"].astype(int).to_numpy()
    event_cells = test["grid_id"].astype(int).to_numpy()
    for model, column in model_columns.items():
        rate = cells.loc[support, column].to_numpy(float)
        probability = rate / rate.sum()
        lookup = dict(zip(supported_ids, probability))
        losses[model] = -np.log(np.asarray([lookup[cell] for cell in event_cells]))

    years = np.arange(2013, 2023)
    year_count = np.asarray([(test["test_year"] == year).sum() for year in years], dtype=int)
    year_loss = {
        model: np.asarray([values[test["test_year"].to_numpy() == year].sum() for year in years])
        for model, values in losses.items()
    }
    rng = np.random.default_rng(SEED)
    sampled_year_indices = rng.integers(0, len(years), size=(N_BOOTSTRAP, len(years)))
    sampled_counts = year_count[sampled_year_indices].sum(axis=1)
    bootstrap_scores = {
        model: values[sampled_year_indices].sum(axis=1) / sampled_counts
        for model, values in year_loss.items()
    }
    gp_bootstrap = bootstrap_scores["GP"]

    rows = []
    for model in model_columns:
        observed = float(np.mean(losses[model]))
        samples = bootstrap_scores[model]
        delta = samples - gp_bootstrap
        rows.append(
            {
                "model": model,
                "observed_common_support_mean_nll": observed,
                "year_block_bootstrap_nll_q025": float(np.quantile(samples, 0.025)),
                "year_block_bootstrap_nll_q975": float(np.quantile(samples, 0.975)),
                "observed_delta_nll_vs_gp": float(observed - np.mean(losses["GP"])),
                "year_block_bootstrap_delta_vs_gp_q025": float(np.quantile(delta, 0.025)),
                "year_block_bootstrap_delta_vs_gp_q975": float(np.quantile(delta, 0.975)),
                "bootstrap_fraction_better_than_gp": float(np.mean(delta < 0.0)),
                "bootstrap_replicates": N_BOOTSTRAP,
                "bootstrap_block": "calendar year",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "paired_year_block_bootstrap_log_scores.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

