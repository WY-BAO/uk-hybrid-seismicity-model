"""Retrospective 2013--2022 temporal holdout for the five spatial models.

Training uses events with year_fraction < 2013.  The ten calendar years
2013 <= year_fraction < 2023 are reserved for testing.  All outputs are kept
under Temporal_Holdout_2013 so that the thesis baseline results are untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cmdstanpy
import matplotlib
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

CATALOGUE = ROOT / "catalogue" / "output" / "filtered_catalogue_regional.csv"
GRID = ROOT / "gp" / "input" / "grid_cells.csv"
SOURCE = ROOT / "hybrid_a" / "outputs" / "source_model_132_grid_mw2.csv"
OLD_UNCERTAINTY = (
    ROOT
    / "hybrid_a"
    / "outputs"
    / "uncertainty_weighting"
    / "hybrid_A_uncertainty_weighted_132_cell_summary.csv"
)
L5_STAN = ROOT / "l5" / "stafford_l5_latent_magnitude.stan"
GP_STAN = ROOT / "gp" / "stan" / "baseline_gp.stan"
HYBRID_B_STAN = ROOT / "hybrid_b" / "stan" / "hybrid_b_gp.stan"

TRAIN_END = 2013.0
TEST_END = 2023.0
TEST_YEARS = TEST_END - TRAIN_END
SEED = 20260713
CHAINS = 4
WARMUP = 1000
SAMPLING = 1000

MW_MIN = 3.0
MW_MAX = 6.5
MW_FLOOR = 1.0
ML_DETECTION_THRESHOLD = 2.0
DM_ML = 0.1
N_MAG_QUAD = 100
FILTER_TABLE = pd.DataFrame(
    [
        {"ML": 2.0, "complete_from": 1990},
        {"ML": 2.5, "complete_from": 1979},
        {"ML": 3.0, "complete_from": 1970},
        {"ML": 3.5, "complete_from": 1850},
        {"ML": 4.0, "complete_from": 1750},
        {"ML": 4.5, "complete_from": 1700},
        {"ML": 5.0, "complete_from": 1650},
        {"ML": 5.5, "complete_from": 1650},
        {"ML": 6.5, "complete_from": 1000},
    ]
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "q025": float(np.quantile(values, 0.025)),
        "q50": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
    }


def sampler_diagnostics(fit: cmdstanpy.CmdStanMCMC) -> dict[str, float | int]:
    summary = fit.summary()
    methods = fit.method_variables()
    rhat = summary["R_hat"].replace([np.inf, -np.inf], np.nan).dropna()
    ess = summary["ESS_bulk"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "max_rhat": float(rhat.max()),
        "min_ess_bulk": float(ess.min()),
        "divergences": int(np.sum(methods["divergent__"])),
        "max_treedepth_observed": int(np.max(methods["treedepth__"])),
        "treedepth_10_hits": int(np.sum(methods["treedepth__"] >= 10)),
    }


def assign_grid_id(events: pd.DataFrame, grid: pd.DataFrame) -> np.ndarray:
    lon_index = np.floor(events["lon"].to_numpy(float) + 8.0).astype(int)
    lat_index = np.floor(events["lat"].to_numpy(float) - 49.0).astype(int)
    lon_index = np.clip(lon_index, int(grid["lon_index"].min()), int(grid["lon_index"].max()))
    lat_index = np.clip(lat_index, int(grid["lat_index"].min()), int(grid["lat_index"].max()))
    lookup = {
        (int(row.lon_index), int(row.lat_index)): int(row.grid_id)
        for row in grid.itertuples(index=False)
    }
    return np.asarray([lookup[(lo, la)] for lo, la in zip(lon_index, lat_index)], dtype=int)


def counts_by_cell(events: pd.DataFrame, n_cells: int) -> np.ndarray:
    return np.bincount(events["grid_id"].to_numpy(int) - 1, minlength=n_cells).astype(int)


def fit_l5(train: pd.DataFrame) -> tuple[np.ndarray, dict]:
    l5_dir = OUTPUT / "l5"
    run_dir = l5_dir / "stan_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "N": int(len(train)),
        "ml_reported": train["ml"].astype(float).to_list(),
        "sigma_ml": train["sigma_ml"].astype(float).to_list(),
        "sigma_round": float(DM_ML / math.sqrt(12.0)),
        "dm_ml": DM_ML,
        "mw_min": MW_MIN,
        "mw_max": MW_MAX,
        "mw_floor": MW_FLOOR,
        "ml_detection_threshold": ML_DETECTION_THRESHOLD,
        "n_exposure_rows": int(len(FILTER_TABLE)),
        "exposure_min_ml": FILTER_TABLE["ML"].astype(float).to_list(),
        "exposure_time": (TRAIN_END - FILTER_TABLE["complete_from"].astype(float)).to_list(),
        "n_quad": N_MAG_QUAD,
        "beta_prior_mean": float(math.log(10.0)),
        "beta_prior_sd": 0.5,
        "lambda_prior_mean": float(math.log(100.0)),
        "lambda_prior_sd": 2.0,
    }
    write_json(l5_dir / "l5_input_pre2013.json", data)
    model = CmdStanModel(stan_file=str(L5_STAN))
    fit = model.sample(
        data=data,
        chains=CHAINS,
        parallel_chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLING,
        seed=SEED,
        output_dir=str(run_dir),
        show_progress=False,
        refresh=200,
    )
    rate_mw3 = fit.stan_variable("lambda_mw_min").reshape(-1)
    beta = fit.stan_variable("beta").reshape(-1)
    lambda_floor = fit.stan_variable("lambda_floor").reshape(-1)
    fraction_mw2 = (
        np.exp(-beta * (2.0 - MW_FLOOR))
        - np.exp(-beta * (MW_MAX - MW_FLOOR))
    ) / (1.0 - np.exp(-beta * (MW_MAX - MW_FLOOR)))
    rate_mw2 = lambda_floor * fraction_mw2
    b_value = fit.stan_variable("b").reshape(-1)
    draws = pd.DataFrame(
        {
            "draw": np.arange(1, len(rate_mw2) + 1),
            "beta": beta,
            "b": b_value,
            "lambda_floor": lambda_floor,
            "lambda_mw2": rate_mw2,
            "lambda_mw3": rate_mw3,
        }
    )
    draws.to_csv(l5_dir / "l5_posterior_draws.csv", index=False)
    result = {
        "training_events": int(len(train)),
        "catalogue_end_year": TRAIN_END,
        "b": quantiles(b_value),
        "lambda_mw2_events_per_year": quantiles(rate_mw2),
        "lambda_mw3_events_per_year": quantiles(rate_mw3),
        "diagnostics": sampler_diagnostics(fit),
    }
    write_json(l5_dir / "l5_result_pre2013.json", result)
    return rate_mw2, result


def fit_gp(grid: pd.DataFrame, train_count: np.ndarray) -> tuple[np.ndarray, dict]:
    gp_dir = OUTPUT / "gp"
    run_dir = gp_dir / "stan_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    x = np.column_stack(
        [
            grid["grid_x_km"].to_numpy(float) / 100.0,
            grid["grid_y_km"].to_numpy(float) / 100.0,
        ]
    )
    data = {
        "N": int(len(grid)),
        "D": 2,
        "x": x.tolist(),
        "count": train_count.astype(int).tolist(),
        "cell_area_km2": grid["grid_area_km2"].astype(float).to_list(),
        "distance_scale_km": 100.0,
        "jitter": 1e-6,
        "alpha_prior_sd": 1.0,
        "rho_prior_logmean": float(math.log(1.5)),
        "rho_prior_logsd": 0.8,
    }
    write_json(gp_dir / "gp_input_pre2013.json", data)
    model = CmdStanModel(stan_file=str(GP_STAN))
    fit = model.sample(
        data=data,
        chains=CHAINS,
        parallel_chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLING,
        seed=SEED,
        output_dir=str(run_dir),
        show_progress=False,
        refresh=200,
        adapt_delta=0.9,
        max_treedepth=10,
        sig_figs=18,
    )
    p = fit.stan_variable("spatial_probability")
    parameters = pd.DataFrame(
        {
            "draw": np.arange(1, p.shape[0] + 1),
            "alpha": fit.stan_variable("alpha").reshape(-1),
            "rho_100km": fit.stan_variable("rho").reshape(-1),
            "rho_km": fit.stan_variable("rho_km").reshape(-1),
        }
    )
    parameters.to_csv(gp_dir / "gp_parameter_draws.csv", index=False)
    np.savez_compressed(gp_dir / "gp_spatial_probability_draws.npz", spatial_probability=p)
    result = {
        "training_events": int(train_count.sum()),
        "alpha": quantiles(parameters["alpha"].to_numpy()),
        "rho_km": quantiles(parameters["rho_km"].to_numpy()),
        "diagnostics": sampler_diagnostics(fit),
    }
    write_json(gp_dir / "gp_result_pre2013.json", result)
    return p, result


def fit_hybrid_b(
    grid: pd.DataFrame,
    train_count: np.ndarray,
    source_rate: np.ndarray,
) -> tuple[np.ndarray, dict, np.ndarray]:
    hb_dir = OUTPUT / "hybrid_b"
    run_dir = hb_dir / "stan_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    included = source_rate > 0.0
    p_source = source_rate[included] / source_rate[included].sum()
    x = np.column_stack(
        [
            grid.loc[included, "grid_x_km"].to_numpy(float) / 100.0,
            grid.loc[included, "grid_y_km"].to_numpy(float) / 100.0,
        ]
    )
    data = {
        "N": int(np.sum(included)),
        "D": 2,
        "x": x.tolist(),
        "count": train_count[included].astype(int).tolist(),
        "p_source": p_source.tolist(),
        "expected_modelled_count": int(train_count[included].sum()),
        "distance_scale_km": 100.0,
        "jitter": 1e-6,
        "alpha_prior_sd": 1.0,
        "rho_prior_logmean": float(math.log(1.5)),
        "rho_prior_logsd": 0.8,
    }
    write_json(hb_dir / "hybrid_b_input_pre2013.json", data)
    model = CmdStanModel(stan_file=str(HYBRID_B_STAN))
    fit = model.sample(
        data=data,
        chains=CHAINS,
        parallel_chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLING,
        seed=SEED,
        output_dir=str(run_dir),
        show_progress=False,
        refresh=200,
        adapt_delta=0.9,
        max_treedepth=10,
        sig_figs=18,
    )
    p_included = fit.stan_variable("p_hybrid")
    p_full = np.zeros((p_included.shape[0], len(grid)), dtype=float)
    p_full[:, included] = p_included
    np.savez_compressed(hb_dir / "hybrid_b_probability_draws.npz", spatial_probability=p_full)
    result = {
        "modelled_cells": int(np.sum(included)),
        "excluded_cells": int(np.sum(~included)),
        "modelled_training_events": int(train_count[included].sum()),
        "excluded_training_events": int(train_count[~included].sum()),
        "alpha": quantiles(fit.stan_variable("alpha").reshape(-1)),
        "rho_km": quantiles(fit.stan_variable("rho_km").reshape(-1)),
        "diagnostics": sampler_diagnostics(fit),
    }
    write_json(hb_dir / "hybrid_b_result_pre2013.json", result)
    return p_full, result, included


def normalized(rate: np.ndarray) -> np.ndarray:
    total = float(np.sum(rate))
    if total <= 0.0:
        raise ValueError("Rate vector has a non-positive total")
    return rate / total


def score_model(name: str, rate: np.ndarray, test_count: np.ndarray, support: np.ndarray) -> dict:
    p = normalized(rate)
    observed = test_count > 0
    impossible = int(test_count[(p <= 0.0) & observed].sum())
    if impossible:
        mean_nll = math.inf
    else:
        mean_nll = float(-np.sum(test_count[observed] * np.log(p[observed])) / test_count.sum())
    empirical_p = test_count / test_count.sum()
    probability_rmse = float(np.sqrt(np.mean((p - empirical_p) ** 2)))
    observed_rate = test_count / TEST_YEARS
    rate_rmse = float(np.sqrt(np.mean((rate - observed_rate) ** 2)))

    common_count = test_count[support]
    common_p = p[support] / p[support].sum()
    common_observed = common_count > 0
    common_nll = float(
        -np.sum(common_count[common_observed] * np.log(common_p[common_observed]))
        / common_count.sum()
    )
    common_empirical_p = common_count / common_count.sum()
    common_probability_rmse = float(np.sqrt(np.mean((common_p - common_empirical_p) ** 2)))
    return {
        "model": name,
        "predicted_total_rate_per_year": float(rate.sum()),
        "observed_test_total_rate_per_year": float(test_count.sum() / TEST_YEARS),
        "total_rate_error_per_year": float(rate.sum() - test_count.sum() / TEST_YEARS),
        "conditional_mean_nll_full_domain": mean_nll,
        "full_domain_impossible_test_events": impossible,
        "spatial_probability_rmse_full_domain": probability_rmse,
        "cell_rate_rmse_events_per_year": rate_rmse,
        "conditional_mean_nll_common_source_support": common_nll,
        "spatial_probability_rmse_common_source_support": common_probability_rmse,
        "common_support_test_events": int(common_count.sum()),
    }


def save_figures(scores: pd.DataFrame, rate_by_model: dict[str, np.ndarray]) -> None:
    names = scores["model"].tolist()
    colours = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    common_nll = scores["conditional_mean_nll_common_source_support"].to_numpy(float)
    y_positions = np.arange(len(names))
    nll_left = float(np.min(common_nll) - 0.015)
    axes[0].hlines(y_positions, nll_left, common_nll, color="#D9D9D9", linewidth=2)
    axes[0].scatter(common_nll, y_positions, color=colours, s=65, zorder=3)
    axes[0].set_yticks(y_positions, names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(nll_left, float(np.max(common_nll) + 0.02))
    axes[0].grid(axis="x", color="#EAEAEA", linewidth=0.8)
    axes[0].set_xlabel("Mean negative log score (lower is better)")
    axes[0].set_title("(a) Spatial score on common support")
    prob_rmse = scores["spatial_probability_rmse_full_domain"].to_numpy(float)
    axes[1].barh(names, prob_rmse, color=colours)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RMSE of cell probability (lower is better)")
    axes[1].set_title("(b) Full-domain spatial RMSE")
    totals = scores["predicted_total_rate_per_year"].to_numpy(float)
    axes[2].barh(names, totals, color=colours)
    axes[2].axvline(
        scores["observed_test_total_rate_per_year"].iloc[0],
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Observed 2013–2022",
    )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Events/year")
    axes[2].set_title("(c) Total activity-rate comparison")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "holdout_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    p_matrix = np.vstack([normalized(rate_by_model[name]) for name in names])
    correlation = np.corrcoef(p_matrix)
    raw_rmse = np.sqrt(np.mean((p_matrix[:, None, :] - p_matrix[None, :, :]) ** 2, axis=2))
    pd.DataFrame(correlation, index=names, columns=names).to_csv(TABLES / "pairwise_probability_correlation.csv")
    pd.DataFrame(raw_rmse, index=names, columns=names).to_csv(TABLES / "pairwise_probability_rmse.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for ax, matrix, title, fmt, cmap in [
        (axes[0], correlation, "(a) Pearson correlation of spatial probabilities", ".2f", "YlGnBu"),
        (axes[1], raw_rmse, "(b) RMSE between spatial probabilities", ".4f", "YlOrRd"),
    ]:
        image = ax.imshow(matrix, cmap=cmap, aspect="equal")
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_title(title, fontsize=10)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "pairwise_spatial_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in [OUTPUT, TABLES, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)

    catalogue = pd.read_csv(CATALOGUE)
    grid = pd.read_csv(GRID).sort_values("grid_id").reset_index(drop=True)
    source = pd.read_csv(SOURCE).sort_values("grid_id").reset_index(drop=True)
    old_uncertainty = pd.read_csv(OLD_UNCERTAINTY).sort_values("cell_id").reset_index(drop=True)
    if len(grid) != 132 or not np.array_equal(grid["grid_id"].to_numpy(), source["grid_id"].to_numpy()):
        raise RuntimeError("The grid and source-model cell ordering do not match")

    catalogue = catalogue.copy()
    catalogue["grid_id"] = assign_grid_id(catalogue, grid)
    full_count = counts_by_cell(catalogue, len(grid))
    if not np.array_equal(full_count, grid["count"].to_numpy(int)):
        mismatch = np.flatnonzero(full_count != grid["count"].to_numpy(int)) + 1
        raise RuntimeError(f"Independent event-to-cell assignment failed in cells {mismatch.tolist()}")

    train = catalogue.loc[catalogue["year_fraction"] < TRAIN_END].copy()
    test = catalogue.loc[
        (catalogue["year_fraction"] >= TRAIN_END) & (catalogue["year_fraction"] < TEST_END)
    ].copy()
    if len(train) + len(test) != len(catalogue):
        raise RuntimeError("Catalogue split does not cover all retained events")
    train.to_csv(OUTPUT / "catalogue_train_pre2013.csv", index=False)
    test.to_csv(OUTPUT / "catalogue_test_2013_2022.csv", index=False)
    train_count = counts_by_cell(train, len(grid))
    test_count = counts_by_cell(test, len(grid))

    l5_draws, l5_result = fit_l5(train)
    gp_p_draws, gp_result = fit_gp(grid, train_count)
    source_mean = source["source_activity_rate_mean"].to_numpy(float)
    source_sd = source["source_activity_rate_sd"].to_numpy(float)
    hybrid_b_p_draws, hybrid_b_result, common_support = fit_hybrid_b(grid, train_count, source_mean)

    rng = np.random.default_rng(SEED + 100)
    gp_l5 = l5_draws[rng.permutation(len(l5_draws))]
    hb_l5 = l5_draws[rng.permutation(len(l5_draws))]
    gp_rate_draws = gp_p_draws * gp_l5[:, None]
    hybrid_b_rate_draws = hybrid_b_p_draws * hb_l5[:, None]
    gp_mean = np.mean(gp_rate_draws, axis=0)
    gp_sd = np.std(gp_rate_draws, axis=0, ddof=1)
    hybrid_b_mean = np.mean(hybrid_b_rate_draws, axis=0)
    hybrid_b_sd = np.std(hybrid_b_rate_draws, axis=0, ddof=1)

    equal_mean = 0.5 * gp_mean + 0.5 * source_mean
    source_present = source_mean > 0.0
    normal_weight = np.full(len(grid), np.nan)
    denominator = gp_sd[source_present] ** 2 + source_sd[source_present] ** 2
    normal_weight[source_present] = source_sd[source_present] ** 2 / denominator
    h = old_uncertainty["fixed_zone_overlap_fraction_h_i"].to_numpy(float)
    cell_gp_weight = np.ones(len(grid))
    cell_gp_weight[source_present] = (
        0.5 * h[source_present] + (1.0 - h[source_present]) * normal_weight[source_present]
    )
    uncertainty_mean = cell_gp_weight * gp_mean + (1.0 - cell_gp_weight) * source_mean

    rate_by_model = {
        "Source-zone": source_mean,
        "GP": gp_mean,
        "Hybrid A equal": equal_mean,
        "Hybrid A uncertainty": uncertainty_mean,
        "Hybrid B": hybrid_b_mean,
    }
    scores = pd.DataFrame(
        [score_model(name, rate, test_count, common_support) for name, rate in rate_by_model.items()]
    )
    scores.to_csv(TABLES / "model_test_scores.csv", index=False)

    cells = grid[
        ["grid_id", "lon_lo", "lon_hi", "lat_lo", "lat_hi", "grid_lon", "grid_lat"]
    ].copy()
    cells["training_count_pre2013"] = train_count
    cells["test_count_2013_2022"] = test_count
    cells["source_support"] = common_support
    cells["source_rate_mean"] = source_mean
    cells["gp_rate_mean_pre2013"] = gp_mean
    cells["gp_rate_sd_pre2013"] = gp_sd
    cells["hybrid_a_equal_rate_mean"] = equal_mean
    cells["hybrid_a_uncertainty_gp_weight"] = cell_gp_weight
    cells["hybrid_a_uncertainty_rate_mean"] = uncertainty_mean
    cells["hybrid_b_rate_mean_pre2013"] = hybrid_b_mean
    cells["hybrid_b_rate_sd_pre2013"] = hybrid_b_sd
    cells.to_csv(TABLES / "model_cell_predictions_and_test_counts.csv", index=False)
    save_figures(scores, rate_by_model)

    impossible_cells = cells.loc[
        (~cells["source_support"]) & (cells["test_count_2013_2022"] > 0),
        ["grid_id", "test_count_2013_2022", "lon_lo", "lon_hi", "lat_lo", "lat_hi"],
    ]
    impossible_cells.to_csv(TABLES / "hybrid_b_out_of_support_test_events.csv", index=False)
    summary = {
        "design": {
            "training_period": "events before 2013-01-01",
            "test_period": "2013-01-01 to 2022-12-31",
            "test_exposure_years": TEST_YEARS,
            "training_events": int(len(train)),
            "test_events": int(len(test)),
            "test_occupied_cells": int(np.sum(test_count > 0)),
            "observed_test_rate_per_year": float(len(test) / TEST_YEARS),
        },
        "l5": l5_result,
        "gp": gp_result,
        "hybrid_b": hybrid_b_result,
        "hybrid_b_out_of_support_test_events": int(test_count[~common_support].sum()),
        "hybrid_b_out_of_support_test_cells": impossible_cells["grid_id"].astype(int).tolist(),
        "score_table": scores.replace({np.inf: None}).to_dict(orient="records"),
        "interpretation_notes": [
            "The conditional spatial log score is primary because the fitted GP models spatial allocation rather than recurrence time.",
            "Hybrid B assigns exactly zero probability to source-zero cells by construction; its full-domain score is therefore infinite when test events occur there.",
            "Common-support scores remove all source-zero cells and renormalize every model, so the five models are compared on the same 126-cell domain.",
            "This is retrospective rather than fully prospective validation because the source-zone model and modelling choices may contain information selected after 2012.",
        ],
    }
    write_json(OUTPUT / "holdout_summary.json", summary)

    lines = [
        "# Retrospective 2013–2022 temporal holdout",
        "",
        f"Training used {len(train)} events before 2013; testing used {len(test)} events during 2013–2022 ({len(test) / TEST_YEARS:.1f} events/year).",
        "",
        "The primary comparison is the conditional spatial mean negative log score. Lower values are better. Hybrid B's full-domain score is infinite if any test event lies in a source-zero cell, so the common-source-support score is also reported for all models.",
        "",
        scores.to_csv(index=False, float_format="%.6f").strip(),
        "",
        "## Important limitation",
        "",
        "This is a retrospective temporal holdout, not a fully independent prospective forecast. The source-zone model and some modelling choices may use information selected after 2012. The comparison is still useful as a relative stress test, but should not be presented as leakage-free validation.",
        "",
    ]
    (OUTPUT / "README_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(scores.to_string(index=False))
    print(f"\nOutputs written to {OUTPUT}")


if __name__ == "__main__":
    main()
