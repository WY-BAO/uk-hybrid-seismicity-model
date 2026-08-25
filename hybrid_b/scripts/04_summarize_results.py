"""Summarize Hybrid B draws and compare the five requested spatial models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    HYBRID_B_ROOT,
    load_config,
    posterior_summary,
    project_path,
    validate_probability_array,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def add_draw_summary(frame: pd.DataFrame, prefix: str, values: np.ndarray) -> None:
    frame[f"{prefix}_mean"] = np.mean(values, axis=0)
    frame[f"{prefix}_sd"] = np.std(values, axis=0, ddof=1)
    frame[f"{prefix}_q025"] = np.quantile(values, 0.025, axis=0)
    frame[f"{prefix}_median"] = np.quantile(values, 0.5, axis=0)
    frame[f"{prefix}_q975"] = np.quantile(values, 0.975, axis=0)


def requested_comparison(config: dict, hybrid_summary: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(project_path(config["paths"]["source_grid"]))
    baseline = pd.read_csv(project_path(config["paths"]["baseline_cell_summary"]))
    equal = pd.read_csv(project_path(config["paths"]["equal_weight_hybrid_a"]))
    uncertainty = pd.read_csv(
        project_path(config["paths"]["uncertainty_weighted_hybrid_a"])
    ).rename(columns={"cell_id": "grid_id"})
    frames = [
        source.loc[:, ["grid_id", "source_activity_rate_mean"]],
        baseline.loc[:, ["grid_id", "activity_rate_per_year_mean"]].rename(
            columns={"activity_rate_per_year_mean": "baseline_gp_activity_rate_mean"}
        ),
        equal.loc[:, ["grid_id", "hybrid_activity_rate_mean"]].rename(
            columns={"hybrid_activity_rate_mean": "equal_weight_hybrid_a_mean"}
        ),
        uncertainty.loc[
            :, ["grid_id", "uncertainty_weighted_hybrid_mean"]
        ],
        hybrid_summary.loc[
            :,
            [
                "grid_id",
                "hybrid_b_activity_rate_per_year_mean",
                "f_correction_mean",
            ],
        ],
    ]
    comparison = frames[0]
    for frame in frames[1:]:
        comparison = comparison.merge(frame, on="grid_id", validate="one_to_one")
    comparison["hybrid_b_minus_source"] = (
        comparison["hybrid_b_activity_rate_per_year_mean"]
        - comparison["source_activity_rate_mean"]
    )
    comparison["hybrid_b_minus_baseline_gp"] = (
        comparison["hybrid_b_activity_rate_per_year_mean"]
        - comparison["baseline_gp_activity_rate_mean"]
    )
    comparison["hybrid_b_minus_equal_weight_hybrid_a"] = (
        comparison["hybrid_b_activity_rate_per_year_mean"]
        - comparison["equal_weight_hybrid_a_mean"]
    )
    comparison["hybrid_b_minus_uncertainty_weighted_hybrid_a"] = (
        comparison["hybrid_b_activity_rate_per_year_mean"]
        - comparison["uncertainty_weighted_hybrid_mean"]
    )
    return comparison


def main() -> int:
    args = parse_args()
    config, _ = load_config(args.config)
    grid = pd.read_csv(project_path(config["paths"]["baseline_grid"])).sort_values(
        "grid_id"
    )
    p_source_table = pd.read_csv(HYBRID_B_ROOT / "input" / "p_source.csv").sort_values(
        "grid_id"
    )
    posterior_file = HYBRID_B_ROOT / "output" / "posterior" / "hybrid_b_posterior_draws.npz"
    if not posterior_file.is_file():
        raise FileNotFoundError("Run scripts/03_combine_with_l5.py first")
    with np.load(posterior_file, allow_pickle=False) as loaded:
        posterior = {name: loaded[name] for name in loaded.files}

    summary = grid.reset_index(drop=True).copy()
    summary["source_cell_rate_per_year"] = p_source_table[
        "source_cell_rate_per_year"
    ].to_numpy(float)
    summary["p_source"] = p_source_table["p_source"].to_numpy(float)
    summary["hybrid_b_domain_status"] = p_source_table[
        "hybrid_b_domain_status"
    ].astype(str).to_numpy()
    add_draw_summary(summary, "f_correction", posterior["f_correction"])
    add_draw_summary(summary, "p_hybrid", posterior["p_hybrid"])
    add_draw_summary(
        summary,
        "hybrid_b_activity_rate_per_year",
        posterior["activity_rate_cell"],
    )
    summary_file = HYBRID_B_ROOT / "output" / "tables" / "hybrid_b_cell_summary.csv"
    summary.to_csv(summary_file, index=False)

    scalar_rows = [
        {"quantity": name, **posterior_summary(posterior[key])}
        for name, key in (
            ("alpha", "alpha"),
            ("rho_stan_units", "rho"),
            ("rho_km", "rho_km"),
            ("L5_total_activity_per_year", "L5_total_activity"),
        )
    ]
    scalar_file = HYBRID_B_ROOT / "output" / "tables" / "posterior_scalar_summary.csv"
    pd.DataFrame(scalar_rows).to_csv(scalar_file, index=False)

    comparison = requested_comparison(config, summary)
    comparison_file = HYBRID_B_ROOT / "output" / "tables" / "five_model_comparison.csv"
    comparison.to_csv(comparison_file, index=False)
    hybrid_mean = comparison["hybrid_b_activity_rate_per_year_mean"].to_numpy(
        float
    )
    comparison_metrics = {}
    for label, column in (
        ("source_zone_grid", "source_activity_rate_mean"),
        ("baseline_multinomial_gp", "baseline_gp_activity_rate_mean"),
        ("equal_weight_hybrid_a", "equal_weight_hybrid_a_mean"),
        (
            "uncertainty_weighted_hybrid_a",
            "uncertainty_weighted_hybrid_mean",
        ),
    ):
        reference = comparison[column].to_numpy(float)
        difference = hybrid_mean - reference
        comparison_metrics[label] = {
            "pearson_correlation": float(np.corrcoef(hybrid_mean, reference)[0, 1]),
            "rmse_activity_rate_per_year": float(
                np.sqrt(np.mean(np.square(difference)))
            ),
            "mean_hybrid_b_minus_reference": float(np.mean(difference)),
            "maximum_absolute_cell_difference": float(
                np.max(np.abs(difference))
            ),
        }

    correction_columns = [
        "grid_id",
        "grid_lon",
        "grid_lat",
        "count",
        "hybrid_b_domain_status",
        "source_cell_rate_per_year",
        "f_correction_mean",
        "hybrid_b_activity_rate_per_year_mean",
    ]
    modelled_summary = summary.loc[
        summary["hybrid_b_domain_status"].eq("modelled_source_covered")
    ]
    correction_extremes = pd.concat(
        [
            modelled_summary.nlargest(10, "f_correction_mean").assign(
                correction_direction="largest_positive"
            ),
            modelled_summary.nsmallest(10, "f_correction_mean").assign(
                correction_direction="largest_negative"
            ),
        ],
        ignore_index=True,
    )
    correction_file = (
        HYBRID_B_ROOT / "output" / "tables" / "gp_correction_extremes.csv"
    )
    correction_extremes.loc[
        :, ["correction_direction", *correction_columns]
    ].to_csv(correction_file, index=False)

    probability_error = validate_probability_array(
        posterior["p_hybrid"],
        float(config["validation"]["probability_sum_tolerance"]),
    )
    l5_error = float(
        np.max(
            np.abs(
                posterior["activity_rate_cell"].sum(axis=1)
                - posterior["L5_total_activity"]
            )
        )
    )
    excluded_mask = posterior["p_source"] == 0.0
    modelled_mask = ~excluded_mask
    centring_error = float(
        np.max(
            np.abs(posterior["f_correction"][:, modelled_mask].mean(axis=1))
        )
    )
    maximum_excluded_probability = float(
        np.max(np.abs(posterior["p_hybrid"][:, excluded_mask]))
    )
    maximum_excluded_activity_rate = float(
        np.max(np.abs(posterior["activity_rate_cell"][:, excluded_mask]))
    )
    maximum_excluded_correction = float(
        np.max(np.abs(posterior["f_correction"][:, excluded_mask]))
    )
    modelled_count = int(summary.loc[~excluded_mask, "count"].sum())
    excluded_count = int(summary.loc[excluded_mask, "count"].sum())
    checks = {
        "p_source_sums_to_one": bool(
            abs(float(posterior["p_source"].sum()) - 1.0)
            <= float(config["validation"]["probability_sum_tolerance"])
        ),
        "every_p_hybrid_draw_is_valid": True,
        "every_p_hybrid_draw_sums_to_one": True,
        "model_domain_f_correction_is_centred_per_draw": centring_error < 1e-10,
        "every_activity_draw_sums_to_l5": l5_error
        < float(config["validation"]["l5_sum_tolerance"]),
        "six_domain_excluded_cells_have_exact_zero_probability": bool(
            excluded_mask.sum() == 6 and maximum_excluded_probability == 0.0
        ),
        "six_domain_excluded_cells_have_exact_zero_activity_rate": bool(
            maximum_excluded_activity_rate == 0.0
        ),
        "six_domain_excluded_cells_are_zero_valued_output_rows": bool(
            maximum_excluded_correction == 0.0
        ),
        "likelihood_uses_126_cells_and_1005_events": bool(
            (~excluded_mask).sum() == 126
            and modelled_count == 1005
            and excluded_count == 8
        ),
        "all_requested_comparison_models_present": len(comparison) == 132,
    }
    run_manifest = json.loads(
        (HYBRID_B_ROOT / "output" / "stan" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    report = {
        "all_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "diagnostics": run_manifest["diagnostics"],
        "posterior": {row["quantity"]: row for row in scalar_rows},
        "comparisons": comparison_metrics,
        "validation": {
            "maximum_p_hybrid_sum_error": probability_error,
            "maximum_l5_conservation_discrepancy": l5_error,
            "maximum_modelled_f_correction_draw_mean": centring_error,
            "maximum_excluded_cell_probability": maximum_excluded_probability,
            "maximum_excluded_cell_activity_rate": maximum_excluded_activity_rate,
            "maximum_excluded_cell_correction": maximum_excluded_correction,
        },
        "model_domain": {
            "full_grid_cells": 132,
            "modelled_source_covered_cells": 126,
            "excluded_zero_source_cells": 6,
            "gp_covariance_dimension": 126,
            "full_catalogue_events": int(summary["count"].sum()),
            "modelled_likelihood_events": modelled_count,
            "excluded_events": excluded_count,
        },
        "outputs": {
            "cell_summary": str(summary_file),
            "scalar_summary": str(scalar_file),
            "five_model_comparison": str(comparison_file),
            "gp_correction_extremes": str(correction_file),
        },
        "interpretation_guardrail": (
            "Positive mean f_correction increases a cell relative to p_source; negative "
            "decreases it. Visual appearance alone is not evidence that Hybrid B is better."
        ),
    }
    write_json(HYBRID_B_ROOT / "output" / "hybrid_b_summary.json", report)
    if not report["all_checks_passed"]:
        raise RuntimeError("At least one Hybrid B result verification failed")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
