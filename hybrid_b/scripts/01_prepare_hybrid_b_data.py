"""Prepare and audit the source-informed Hybrid B input without running Stan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    HYBRID_B_ROOT,
    ensure_directories,
    load_baseline_config,
    load_config,
    project_path,
    sha256_file,
    source_informed_softmax,
    validate_probability_array,
    write_json,
)


GRID_COLUMNS = (
    "grid_id",
    "lon_lo",
    "lon_hi",
    "lat_lo",
    "lat_hi",
    "grid_lon",
    "grid_lat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def maximum_column_error(left: pd.DataFrame, right: pd.DataFrame) -> float:
    return float(
        max(
            np.max(
                np.abs(
                    left[column].to_numpy(float)
                    - right[column].to_numpy(float)
                )
            )
            for column in GRID_COLUMNS
        )
    )


def baseline_snapshot(config: dict) -> dict[str, dict[str, str]]:
    labels = {
        "stan_model": "baseline_stan_model",
        "config": "baseline_config",
        "grid": "baseline_grid",
        "stan_data": "baseline_stan_data",
        "gp_posterior": "baseline_gp_posterior",
        "combined_posterior": "baseline_combined_posterior",
    }
    return {
        label: {
            "path": str(project_path(config["paths"][key])),
            "sha256": sha256_file(project_path(config["paths"][key])),
        }
        for label, key in labels.items()
    }


def main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    baseline_config, baseline_config_path = load_baseline_config(config)
    ensure_directories()

    paths = {key: project_path(value) for key, value in config["paths"].items()}
    required = (
        paths["baseline_stan_model"],
        paths["baseline_stan_data"],
        paths["baseline_grid"],
        paths["baseline_gp_posterior"],
        paths["baseline_combined_posterior"],
        paths["source_grid"],
        paths["source_fraction_matrix"],
        paths["source_zone_rates"],
        paths["source_verification"],
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    expected = config["expected"]
    grid = pd.read_csv(paths["baseline_grid"]).sort_values("grid_id").reset_index(drop=True)
    source = pd.read_csv(paths["source_grid"]).sort_values("grid_id").reset_index(drop=True)
    fractions = (
        pd.read_csv(paths["source_fraction_matrix"])
        .sort_values("grid_id")
        .reset_index(drop=True)
    )
    source_verification = json.loads(paths["source_verification"].read_text(encoding="utf-8"))
    baseline_data = json.loads(paths["baseline_stan_data"].read_text(encoding="utf-8"))

    if len(grid) != int(expected["grid_cells"]) or grid["grid_id"].nunique() != len(grid):
        raise RuntimeError("The baseline baseline grid is not 132 unique cells")
    if int(grid["count"].sum()) != int(expected["earthquakes"]):
        raise RuntimeError("The baseline grid counts do not sum to 1013")
    if len(source) != len(grid) or not np.array_equal(
        source["grid_id"].to_numpy(int), grid["grid_id"].to_numpy(int)
    ):
        raise RuntimeError("The current Hybrid A source grid is not aligned by grid_id")
    grid_alignment_error = maximum_column_error(grid, source)
    if grid_alignment_error > 0.0:
        raise RuntimeError(f"Source/grid geometry mismatch: {grid_alignment_error}")
    if not np.array_equal(
        np.asarray(baseline_data["count"], dtype=int), grid["count"].to_numpy(int)
    ):
        raise RuntimeError("Baseline Stan counts and the baseline grid differ")

    source_hash = sha256_file(paths["source_grid"])
    expected_source_hash = str(config["source"]["expected_grid_sha256"])
    verification_source_hash = str(
        source_verification["outputs"]["source_grid_table_sha256"]
    )
    if source_hash != expected_source_hash or source_hash != verification_source_hash:
        raise RuntimeError("Current Hybrid A source-grid hash differs from its verified value")
    if source_verification["inputs"]["old_approach_D_numerical_rates_used"]:
        raise RuntimeError("The selected current source input unexpectedly uses Approach D")
    if source_verification["method"]["mean_formula"] != "mu_i=sum_z(f_iz*mu_z)":
        raise RuntimeError("The source-grid mean formula differs from the verified conversion")

    rate_column = str(config["source"]["cell_rate_column"])
    source_cell_rate = source[rate_column].to_numpy(float)
    if not np.isfinite(source_cell_rate).all() or np.any(source_cell_rate < 0.0):
        raise RuntimeError("Source cell activity rates must be finite and nonnegative")
    source_total = float(source_cell_rate.sum())
    if source_total <= 0.0:
        raise RuntimeError("Source cell activity rates have no positive total")
    p_source = source_cell_rate / source_total
    probability_tolerance = float(config["validation"]["probability_sum_tolerance"])
    p_source_sum_error = validate_probability_array(p_source, probability_tolerance)

    if len(fractions) != len(grid) or not np.array_equal(
        fractions["grid_id"].to_numpy(int), grid["grid_id"].to_numpy(int)
    ):
        raise RuntimeError("Source fraction matrix is not aligned to the baseline grid")
    rates = pd.read_csv(paths["source_zone_rates"])
    rates = rates.loc[
        np.isclose(rates["threshold_mw"].to_numpy(float), float(config["source"]["threshold_mw"]))
    ].copy()
    zone_columns = [column for column in fractions.columns if column != "grid_id"]
    rate_lookup = rates.set_index("zone")[str(config["source"]["zone_mean_column"])]
    missing_zones = sorted(set(zone_columns) - set(rate_lookup.index.astype(str)))
    if missing_zones:
        raise RuntimeError(f"Missing Mw>=2 rates for source zones: {missing_zones}")
    reconstructed_rate = fractions[zone_columns].to_numpy(float) @ rate_lookup.reindex(
        zone_columns
    ).to_numpy(float)
    reconstruction_error = float(np.max(np.abs(reconstructed_rate - source_cell_rate)))
    if reconstruction_error > float(config["validation"]["source_reconstruction_tolerance"]):
        raise RuntimeError(
            "Source cell means do not reproduce the verified Hybrid A conversion: "
            f"maximum difference={reconstruction_error:.17g}"
        )

    fraction_sum = fractions[zone_columns].sum(axis=1).to_numpy(float)
    zero_mask = p_source <= 0.0
    zero_reason = (
        "No current source-zone polygon overlaps this cell: covered area, "
        "contributing-zone count, and every A_iz/A_z fraction are zero."
    )
    p_source_table = grid.loc[:, list(GRID_COLUMNS) + ["count"]].copy()
    p_source_table["source_cell_rate_per_year"] = source_cell_rate
    p_source_table["p_source"] = p_source
    p_source_table["source_model_covered_area_km2"] = source[
        "source_model_covered_area_km2"
    ].to_numpy(float)
    p_source_table["n_contributing_zones"] = source[
        "n_contributing_zones"
    ].to_numpy(int)
    p_source_table["sum_source_zone_fractions"] = fraction_sum
    p_source_table["zero_source_reason"] = np.where(zero_mask, zero_reason, "")
    p_source_table["zero_source_has_observed_events"] = zero_mask & (
        p_source_table["count"].to_numpy(int) > 0
    )
    p_source_table["hybrid_b_domain_status"] = np.where(
        zero_mask, "excluded_zero_source_coverage", "modelled_source_covered"
    )

    p_source_file = HYBRID_B_ROOT / "input" / "p_source.csv"
    zero_file = HYBRID_B_ROOT / "input" / "zero_source_probability_cells.csv"
    p_source_table.to_csv(p_source_file, index=False)
    p_source_table.loc[zero_mask].to_csv(zero_file, index=False)

    zero_correction = source_informed_softmax(p_source, np.zeros_like(p_source))
    zero_correction_error = float(np.max(np.abs(zero_correction - p_source)))
    if zero_correction_error > np.finfo(float).eps * 8:
        raise RuntimeError("Deterministic zero-correction identity test failed")

    zero_rows = p_source_table.loc[zero_mask]
    zero_cells = [
        {
            "grid_id": int(row.grid_id),
            "grid_lon": float(row.grid_lon),
            "grid_lat": float(row.grid_lat),
            "lon_bounds": [float(row.lon_lo), float(row.lon_hi)],
            "lat_bounds": [float(row.lat_lo), float(row.lat_hi)],
            "observed_count": int(row.count),
            "p_source": float(row.p_source),
            "excluded_from_hybrid_b_likelihood": True,
            "reason": zero_reason,
        }
        for row in zero_rows.itertuples(index=False)
    ]
    positive_index = np.flatnonzero(p_source > 0.0) + 1
    modelled_count = int(grid.loc[~zero_mask, "count"].sum())
    excluded_count = int(grid.loc[zero_mask, "count"].sum())
    if int(positive_index.size) != int(expected["hybrid_b_modelled_cells"]):
        raise RuntimeError("Unexpected number of source-covered Hybrid B cells")
    if int(zero_mask.sum()) != int(expected["hybrid_b_excluded_cells"]):
        raise RuntimeError("Unexpected number of zero-coverage Hybrid B cells")
    if modelled_count != int(expected["hybrid_b_likelihood_earthquakes"]):
        raise RuntimeError("Unexpected number of Hybrid B likelihood earthquakes")
    if excluded_count != int(expected["hybrid_b_excluded_earthquakes"]):
        raise RuntimeError("Unexpected number of excluded Hybrid B earthquakes")

    # Stan receives only the 126 source-covered cells and their 1005 events.
    # The complete grid and excluded rows remain in the audit CSV and are
    # restored after fitting.
    stan_data_file = HYBRID_B_ROOT / "input" / "stan_data.json"
    if stan_data_file.exists():
        stan_data_file.unlink()
    modelled_index = positive_index - 1
    hybrid_data = {
        "N": int(modelled_index.size),
        "D": int(baseline_data["D"]),
        "x": np.asarray(baseline_data["x"], dtype=float)[modelled_index].tolist(),
        "count": grid.loc[~zero_mask, "count"].to_numpy(int).tolist(),
        "p_source": p_source[~zero_mask].tolist(),
        "expected_modelled_count": modelled_count,
        "distance_scale_km": baseline_data["distance_scale_km"],
        "jitter": baseline_data["jitter"],
        "alpha_prior_sd": baseline_data["alpha_prior_sd"],
        "rho_prior_logmean": baseline_data["rho_prior_logmean"],
        "rho_prior_logsd": baseline_data["rho_prior_logsd"],
    }
    write_json(stan_data_file, hybrid_data)

    report = {
        "status": "ready_for_sampling",
        "configuration": str(config_path),
        "baseline_configuration": str(baseline_config_path),
        "baseline_values_reused_without_change": {
            "grid": baseline_config["grid"],
            "coordinates": baseline_config["coordinates"],
            "model": baseline_config["model"],
            "sampling": baseline_config["sampling"],
            "l5_combination": baseline_config["l5_combination"],
        },
        "dimensions": {
            "grid_cells": int(len(grid)),
            "full_catalogue_count": int(grid["count"].sum()),
            "hybrid_b_modelled_cells": int(positive_index.size),
            "hybrid_b_excluded_cells": int(zero_mask.sum()),
            "hybrid_b_likelihood_count": modelled_count,
            "hybrid_b_excluded_event_count": excluded_count,
        },
        "source_input": {
            "path": str(paths["source_grid"]),
            "sha256": source_hash,
            "type": "cell annual activity rate",
            "units": "Mw >= 2 events per year per cell",
            "formula": "source_cell_rate[i]=sum_z(lambda_z*A_iz/A_z)",
            "is_rate_density": False,
            "cell_area_must_not_be_multiplied_again": True,
            "old_approach_D_numerical_rates_used": False,
            "total_rate_per_year": source_total,
            "maximum_used_vs_current_hybrid_a_difference": 0.0,
            "maximum_fraction_matrix_reconstruction_difference": reconstruction_error,
        },
        "p_source": {
            "sum": float(p_source.sum()),
            "maximum_sum_to_one_error": p_source_sum_error,
            "minimum": float(p_source.min()),
            "minimum_positive": float(p_source[p_source > 0.0].min()),
            "maximum": float(p_source.max()),
            "nonpositive_cells": int(zero_mask.sum()),
            "positive_cells_used_by_softmax_and_likelihood": int(
                positive_index.size
            ),
            "output": str(p_source_file),
        },
        "zero_source_cells": zero_cells,
        "observed_events_excluded_with_zero_source_cells": excluded_count,
        "zero_source_cells_with_positive_count": int(
            (zero_rows["count"].to_numpy(int) > 0).sum()
        ),
        "zero_correction_test": {
            "definition": (
                "softmax(log(p_source_positive)+0) on positive cells, then reinsert "
                "exact zeros; the full 132-cell result equals p_source"
            ),
            "maximum_absolute_difference": zero_correction_error,
            "passed": True,
        },
        "zero_source_treatment": {
            "softmax_scope": "strictly positive p_source cells only",
            "multinomial_scope": "strictly positive p_source cells only",
            "gp_field_cells": int(positive_index.size),
            "final_probability_cells": int(len(grid)),
            "zero_cells_reinserted_exactly": True,
            "log_zero_evaluated": False,
            "epsilon_or_background_added": False,
            "source_uncertainty_used": False,
            "zero_source_cells_are_outside_modelling_domain": True,
            "events_in_zero_source_cells_excluded_from_likelihood": excluded_count,
        },
        "stan_input_written": True,
        "stan_sampling_permitted": True,
        "blocking_reason": "",
        "baseline_snapshot": baseline_snapshot(config),
    }
    report_file = HYBRID_B_ROOT / "input" / "preflight_report.json"
    write_json(report_file, report)

    print(json.dumps({key: report[key] for key in ("status", "dimensions", "source_input", "p_source", "observed_events_excluded_with_zero_source_cells", "zero_source_cells_with_positive_count", "zero_correction_test", "zero_source_treatment", "stan_input_written")}, indent=2))
    if zero_mask.any():
        print("\nAll zero-source cells excluded from the Hybrid B modelling domain:")
        print(
            zero_rows.loc[
                :,
                [
                    "grid_id",
                    "grid_lon",
                    "grid_lat",
                    "count",
                    "source_cell_rate_per_year",
                    "p_source",
                    "source_model_covered_area_km2",
                    "n_contributing_zones",
                    "zero_source_has_observed_events",
                ],
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
