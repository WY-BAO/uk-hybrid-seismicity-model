"""Create revised baseline summaries, comparisons, and verification report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd

from _common import (
    BASELINE_ROOT,
    load_config,
    posterior_summary,
    resolve_project_path,
    write_json,
)


PROBABILITIES = (0.025, 0.05, 0.5, 0.95, 0.975)


def prior_row(
    parameter: str,
    units: str,
    mean: float,
    sd: float,
    quantile: Callable[[float], float],
) -> dict[str, float | str]:
    values: dict[str, float | str] = {
        "parameter": parameter,
        "units": units,
        "mean": mean,
        "sd": sd,
    }
    for label, probability in zip(
        ("q025", "q05", "median", "q95", "q975"), PROBABILITIES
    ):
        values[label] = quantile(probability)
    return values


def analytic_prior_summary(config: dict) -> pd.DataFrame:
    priors = config["model"]["priors"]
    normal = NormalDist()
    alpha_sd = float(priors["alpha_sd"])
    rho_mu = float(priors["rho_logmean"])
    rho_sigma = float(priors["rho_logsd"])
    km_scale = float(config["coordinates"]["stan_unit_km"])
    rows = [
        prior_row(
            "alpha",
            "centred log-spatial-intensity amplitude",
            alpha_sd * math.sqrt(2.0 / math.pi),
            alpha_sd * math.sqrt(1.0 - 2.0 / math.pi),
            lambda p: alpha_sd * normal.inv_cdf((p + 1.0) / 2.0),
        ),
        prior_row(
            "rho",
            "Stan spatial units",
            math.exp(rho_mu + rho_sigma**2 / 2.0),
            math.sqrt(
                (math.exp(rho_sigma**2) - 1.0)
                * math.exp(2.0 * rho_mu + rho_sigma**2)
            ),
            lambda p: math.exp(rho_mu + rho_sigma * normal.inv_cdf(p)),
        ),
        prior_row(
            "rho",
            "km",
            km_scale * math.exp(rho_mu + rho_sigma**2 / 2.0),
            km_scale
            * math.sqrt(
                (math.exp(rho_sigma**2) - 1.0)
                * math.exp(2.0 * rho_mu + rho_sigma**2)
            ),
            lambda p: km_scale
            * math.exp(rho_mu + rho_sigma * normal.inv_cdf(p)),
        ),
        prior_row(
            "eta_i",
            "standard-normal latent variable",
            0.0,
            1.0,
            normal.inv_cdf,
        ),
    ]
    return pd.DataFrame(rows)


def add_draw_summary(frame: pd.DataFrame, prefix: str, values: np.ndarray) -> None:
    frame[f"{prefix}_mean"] = np.mean(values, axis=0)
    frame[f"{prefix}_sd"] = np.std(values, axis=0, ddof=1)
    frame[f"{prefix}_median"] = np.quantile(values, 0.5, axis=0)
    frame[f"{prefix}_q025"] = np.quantile(values, 0.025, axis=0)
    frame[f"{prefix}_q975"] = np.quantile(values, 0.975, axis=0)


def compare_with_legacy(
    revised: pd.DataFrame, legacy_file: Path
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    if not legacy_file.is_file():
        raise FileNotFoundError(
            f"Legacy 33-year baseline summary is required for comparison: {legacy_file}"
        )
    legacy = pd.read_csv(legacy_file)
    required = {"grid_id", "activity_rate_per_year_mean"}
    if not required.issubset(legacy.columns):
        raise KeyError(f"Legacy baseline lacks columns {sorted(required - set(legacy.columns))}")
    columns = [
        "grid_id",
        "lon_lo",
        "lon_hi",
        "lat_lo",
        "lat_hi",
        "grid_lon",
        "grid_lat",
        "activity_rate_per_year_mean",
    ]
    old = legacy.loc[:, [name for name in columns if name in legacy.columns]].copy()
    old = old.rename(
        columns={"activity_rate_per_year_mean": "legacy_activity_rate_mean"}
    )
    current = revised.loc[
        :,
        [
            "grid_id",
            "lon_lo",
            "lon_hi",
            "lat_lo",
            "lat_hi",
            "grid_lon",
            "grid_lat",
            "activity_rate_per_year_mean",
        ],
    ].copy()
    current = current.rename(
        columns={"activity_rate_per_year_mean": "revised_activity_rate_mean"}
    )
    comparison = current.merge(
        old.loc[:, ["grid_id", "legacy_activity_rate_mean"]],
        on="grid_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        raise RuntimeError("Legacy and revised baseline grids do not have identical cell IDs")
    comparison = comparison.drop(columns="_merge")
    comparison["difference_revised_minus_legacy"] = (
        comparison["revised_activity_rate_mean"]
        - comparison["legacy_activity_rate_mean"]
    )
    comparison["absolute_difference"] = comparison[
        "difference_revised_minus_legacy"
    ].abs()
    old_values = comparison["legacy_activity_rate_mean"].to_numpy(float)
    new_values = comparison["revised_activity_rate_mean"].to_numpy(float)
    differences = new_values - old_values
    summary: dict[str, float | int | str] = {
        "legacy_file": str(legacy_file),
        "cells_compared": int(len(comparison)),
        "pearson_correlation": float(np.corrcoef(old_values, new_values)[0, 1]),
        "rmse_events_per_year": float(np.sqrt(np.mean(differences**2))),
        "maximum_absolute_cell_difference_events_per_year": float(
            np.max(np.abs(differences))
        ),
    }
    top_five = comparison.nlargest(5, "absolute_difference").reset_index(drop=True)
    return summary, top_five


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = load_config(args.config)
    grid_file = BASELINE_ROOT / "input" / "grid_cells.csv"
    posterior_file = (
        BASELINE_ROOT
        / "output"
        / "posterior"
        / "baseline_posterior_draws.npz"
    )
    gp_posterior_file = (
        BASELINE_ROOT / "output" / "posterior" / "gp_posterior_draws.npz"
    )
    for path in (grid_file, posterior_file, gp_posterior_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    grid = pd.read_csv(grid_file)
    with np.load(posterior_file, allow_pickle=False) as loaded:
        posterior = {name: loaded[name] for name in loaded.files}
    with np.load(gp_posterior_file, allow_pickle=False) as loaded:
        gp_posterior = {name: loaded[name] for name in loaded.files}

    cell_summary = grid.copy()
    add_draw_summary(cell_summary, "gp_effect", gp_posterior["gp_effect"])
    add_draw_summary(
        cell_summary,
        "spatial_probability",
        gp_posterior["spatial_probability"],
    )
    add_draw_summary(
        cell_summary, "activity_rate_per_year", posterior["activity_rate_cell"]
    )
    cell_summary["activity_rate_95pct_width"] = (
        cell_summary["activity_rate_per_year_q975"]
        - cell_summary["activity_rate_per_year_q025"]
    )
    cell_summary_file = BASELINE_ROOT / "output" / "tables" / "cell_posterior_summary.csv"
    cell_summary.to_csv(cell_summary_file, index=False)

    scalar_sources = (
        ("alpha", gp_posterior["alpha"]),
        ("rho_stan_units", gp_posterior["rho"]),
        ("rho_km", gp_posterior["rho_km"]),
        ("L5_total_activity_per_year", posterior["L5_total_activity"]),
    )
    scalar_rows = [
        {"quantity": label, **posterior_summary(values)}
        for label, values in scalar_sources
    ]
    scalar_file = BASELINE_ROOT / "output" / "tables" / "posterior_scalar_summary.csv"
    pd.DataFrame(scalar_rows).to_csv(scalar_file, index=False)

    prior_frame = analytic_prior_summary(config)
    prior_file = BASELINE_ROOT / "output" / "tables" / "prior_summary.csv"
    prior_frame.to_csv(prior_file, index=False)

    run_manifest = json.loads(
        (BASELINE_ROOT / "output" / "stan" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    combination_manifest = json.loads(
        (
            BASELINE_ROOT / "output" / "posterior" / "combination_manifest.json"
        ).read_text(encoding="utf-8")
    )
    preparation_manifest = json.loads(
        (BASELINE_ROOT / "input" / "preparation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    stan_data = json.loads(
        (BASELINE_ROOT / "input" / "stan_data.json").read_text(encoding="utf-8")
    )
    stan_source = (BASELINE_ROOT / "stan" / "baseline_gp.stan").read_text(
        encoding="utf-8"
    )

    probability_sum_error = float(
        np.max(
            np.abs(gp_posterior["spatial_probability"].sum(axis=1) - 1.0)
        )
    )
    total_discrepancy = np.abs(
        posterior["activity_rate_cell"].sum(axis=1)
        - posterior["L5_total_activity"]
    )
    maximum_l5_discrepancy = float(total_discrepancy.max())
    gp_centering_error = float(
        np.max(np.abs(gp_posterior["gp_effect"].mean(axis=1)))
    )
    posterior_keys = set(posterior) | set(gp_posterior)
    obsolete_output_keys = sorted(
        name
        for name in posterior_keys
        if name == "a" or name.startswith("diagnostic_direct_gp")
    )
    stan_has_intercept = bool(
        re.search(r"\breal\s+a\s*;", stan_source)
        or re.search(r"\ba_prior_(?:mean|sd)\b", stan_source)
    )
    stan_has_exposure = "log_exposure_area" in stan_source
    stan_data_has_exposure = any(
        name in stan_data for name in ("T", "exposure_years", "log_exposure_area")
    )
    count_sum = int(grid["count"].sum())
    zero_count_cells = int((grid["count"] == 0).sum())
    maximum_cell_count = int(grid["count"].max())
    expected = config["expected"]

    checks = [
        (
            "catalogue_exactly_matches_l5",
            bool(
                preparation_manifest["selection"][
                    "exactly_matches_l5_event_order_and_numeric_inputs"
                ]
            ),
        ),
        (
            "no_fixed_gp_observation_window",
            not preparation_manifest["selection"]["fixed_gp_observation_window"],
        ),
        ("earthquake_count_matches_expected", count_sum == int(expected["earthquakes"])),
        ("grid_cell_count_matches_expected", len(grid) == int(expected["grid_cells"])),
        ("cell_counts_sum_to_catalogue", count_sum == preparation_manifest["earthquakes"]),
        (
            "multinomial_likelihood_used",
            "multinomial(" in stan_source and "poisson_log(" not in stan_source,
        ),
        (
            "every_spatial_probability_draw_sums_to_one",
            probability_sum_error
            < float(config["validation"]["probability_sum_tolerance"]),
        ),
        ("centred_gp_effect_draws", gp_centering_error < 1e-10),
        ("intercept_a_absent", not stan_has_intercept and "a" not in posterior_keys),
        (
            "exposure_inputs_absent",
            not stan_has_exposure and not stan_data_has_exposure,
        ),
        ("direct_gp_absolute_rate_outputs_absent", not obsolete_output_keys),
        (
            "every_final_draw_sums_to_l5",
            maximum_l5_discrepancy
            < float(config["l5_combination"]["sum_tolerance"]),
        ),
        (
            "equal_draw_counts_paired_without_replacement",
            bool(
                combination_manifest["pairing"]["equal_input_draw_counts"]
                and not combination_manifest["pairing"]["replacement_used"]
                and combination_manifest["validation"][
                    "all_gp_draws_used_exactly_once"
                ]
                and combination_manifest["validation"][
                    "all_l5_draws_used_exactly_once"
                ]
            ),
        ),
    ]
    check_frame = pd.DataFrame(checks, columns=["check", "passed"])
    check_frame["detail"] = ""
    details = {
        "earthquake_count_matches_expected": str(count_sum),
        "grid_cell_count_matches_expected": str(len(grid)),
        "every_spatial_probability_draw_sums_to_one": (
            f"maximum absolute error={probability_sum_error:.17g}"
        ),
        "centred_gp_effect_draws": f"maximum absolute draw mean={gp_centering_error:.17g}",
        "direct_gp_absolute_rate_outputs_absent": (
            ", ".join(obsolete_output_keys) if obsolete_output_keys else "no obsolete output keys"
        ),
        "every_final_draw_sums_to_l5": (
            f"maximum absolute discrepancy={maximum_l5_discrepancy:.17g}"
        ),
    }
    for check, detail in details.items():
        check_frame.loc[check_frame["check"].eq(check), "detail"] = detail
    checks_file = BASELINE_ROOT / "output" / "tables" / "verification_checks.csv"
    check_frame.to_csv(checks_file, index=False)

    alpha_summary = posterior_summary(gp_posterior["alpha"])
    rho_summary = posterior_summary(gp_posterior["rho"])
    rho_km_summary = posterior_summary(gp_posterior["rho_km"])
    report_file = BASELINE_ROOT / "output" / "verification_report.md"
    report_lines = [
        "# Baseline multinomial GP checks",
        "",
        f"1. Exact catalogue: `{preparation_manifest['catalogue']}`",
        f"2. Earthquakes: {preparation_manifest['earthquakes']}",
        (
            "3. Event years: "
            f"{preparation_manifest['minimum_event_year']}–"
            f"{preparation_manifest['maximum_event_year']}; ML range "
            f"{preparation_manifest['minimum_ml']:g}–{preparation_manifest['maximum_ml']:g}"
        ),
        "4. Fixed 33-year GP observation window: absent",
        f"5. Grid cells: {len(grid)}",
        f"6. Sum of cell counts: {count_sum}",
        f"7. Zero-count cells: {zero_count_cells}",
        f"8. Maximum cell count: {maximum_cell_count}",
        f"9. Maximum |sum(p)-1|: {probability_sum_error:.3e}",
        (
            "10. alpha posterior: mean "
            f"{alpha_summary['mean']:.6g}, SD {alpha_summary['sd']:.6g}, "
            f"95% CrI [{alpha_summary['q025']:.6g}, {alpha_summary['q975']:.6g}]"
        ),
        (
            "11. rho posterior: mean "
            f"{rho_summary['mean']:.6g} Stan units / {rho_km_summary['mean']:.6g} km; "
            f"95% CrI [{rho_summary['q025']:.6g}, {rho_summary['q975']:.6g}] units"
        ),
        "12. Intercept a: absent",
        "13. T/log_exposure_area: absent",
        "14. Direct GP absolute activity-rate outputs: absent",
        f"15. Maximum |sum(cell rates)-paired L5 total|: {maximum_l5_discrepancy:.3e}",
    ]
    report_lines.extend(
        [
            "",
            f"All automated checks passed: {bool(check_frame['passed'].all())}",
            "",
        ]
    )
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    baseline_summary = {
        "data": preparation_manifest,
        "gp_sampling": run_manifest,
        "bayesian_combination": combination_manifest,
        "posterior": {row["quantity"]: row for row in scalar_rows},
        "catalogue_grid_diagnostics": {
            "sum_cell_counts": count_sum,
            "zero_count_cells": zero_count_cells,
            "maximum_cell_count": maximum_cell_count,
        },
        "verification": {
            "all_checks_passed": bool(check_frame["passed"].all()),
            "maximum_spatial_probability_sum_error": probability_sum_error,
            "maximum_l5_sum_discrepancy": maximum_l5_discrepancy,
            "maximum_gp_effect_centering_error": gp_centering_error,
        },
        "outputs": {
            "main_posterior": str(posterior_file),
            "cell_summary": str(cell_summary_file),
            "scalar_summary": str(scalar_file),
            "prior_summary": str(prior_file),
            "verification_checks": str(checks_file),
            "verification_report": str(report_file),
        },
    }
    write_json(BASELINE_ROOT / "output" / "baseline_summary.json", baseline_summary)
    print(check_frame.to_string(index=False), flush=True)
    print(f"Verification report: {report_file}", flush=True)
    if not bool(check_frame["passed"].all()):
        raise RuntimeError("One or more baseline verification checks failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
