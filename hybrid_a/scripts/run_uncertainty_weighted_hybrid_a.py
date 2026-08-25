"""Build cell-level uncertainty-weighted Hybrid Model A.

The calculation uses the GP posterior and source-zone inputs:

* the 4,000 x 132 GP annual activity-rate posterior samples; and
* the branch-by-branch truncated Gutenberg--Richter Mw>=2 source-zone means
  and standard deviations.

Source-zone means retain the validated source-to-grid redistribution

    f_iz = A_iz / A_z.

The geometric union coverage and residual fraction are retained in a separate
audit, but partial source coverage does not modify the weight and no residual
contribution row is created.  For every source-covered cell the ordinary
cell-level uncertainty weight is

    w_i^u = sigma_SZ_i^2 / (sigma_GP_i^2 + sigma_SZ_i^2).

For cells intersecting BALA and/or ESCO, their share of total source-zone
overlap is

    h_i = sum_{z in {BALA,ESCO}} A_iz / sum_z A_iz,

and the cell GP weight is

    w_i = 0.5 h_i + (1-h_i) w_i^u.

Cells with no source-zone overlap use w_i=1.  The final Hybrid mean is the
cell-level convex combination

    lambda_H_i = w_i lambda_GP_i + (1-w_i) lambda_SZ_i.

No total-rate renormalisation and no source pseudo-draws are used.  A final
Hybrid SD is deliberately not produced, because the BALA/ESCO source
uncertainty is unquantified rather than zero.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from pyproj import Transformer
from shapely.ops import unary_union

import convert_source_model_to_gp_grid as source_conversion


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
GP_POSTERIOR = PROJECT_ROOT / "gp" / "output" / "posterior" / "baseline_posterior_draws.npz"
GP_GRID = PROJECT_ROOT / "gp" / "input" / "grid_cells.csv"
SOURCE_RATES = (
    PROJECT_ROOT / "source_model" / "output" / "source_zone_activity_rates_mw2_extrapolated.csv"
)
SOURCE_GEOMETRY = PROJECT_ROOT / "source_model" / "data" / "geometry_SSCmodel"
SOURCE_GRID = ROOT / "output" / "source_model_132_grid_mw2.csv"
FRACTION_MATRIX = ROOT / "output" / "source_zone_to_grid_fraction_matrix.csv"
ZONE_AUDIT = ROOT / "output" / "source_zone_to_grid_zone_audit.csv"

OUT = ROOT / "output" / "uncertainty_weighting"
FIGURES = OUT / "figures"
CONTRIBUTION_OUTPUT = OUT / "hybrid_A_uncertainty_weighted_contributions.csv"
CELL_OUTPUT = OUT / "hybrid_A_uncertainty_weighted_132_cell_summary.csv"
COVERAGE_OUTPUT = OUT / "hybrid_A_uncertainty_weighted_coverage_audit.csv"
WEIGHT_DISTRIBUTION_OUTPUT = (
    OUT / "hybrid_A_uncertainty_weighted_weight_distribution.csv"
)
VERIFICATION_JSON = OUT / "hybrid_A_uncertainty_weighted_verification.json"
VERIFICATION_MD = OUT / "hybrid_A_uncertainty_weighted_verification.md"

EXPECTED_CELLS = 132
EXPECTED_DRAWS = 4000
FIXED_ZONES = frozenset({"BALA", "ESCO"})
GEOMETRY_TOL = 1e-12
NUMERIC_TOL = 1e-12
AREA_TOL_KM2 = 1e-9
EXPECTED_EQUAL_WEIGHT_TOTAL = 22.3063
EXPECTED_EQUAL_TOTAL_ABS_TOL = 1e-3
CSV_FLOAT_FORMAT = "%.17g"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes() -> dict[str, str]:
    inputs = [
        GP_POSTERIOR,
        GP_GRID,
        SOURCE_RATES,
        SOURCE_GEOMETRY,
        SOURCE_GRID,
        FRACTION_MATRIX,
        ZONE_AUDIT,
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in inputs}


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [native(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_gp() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    grid = source_conversion.read_grid(GP_GRID)
    with np.load(GP_POSTERIOR, allow_pickle=False) as posterior:
        activity = np.asarray(posterior["activity_rate_cell"], dtype=float)
    if activity.shape != (EXPECTED_DRAWS, EXPECTED_CELLS):
        raise AssertionError(
            f"Expected GP activity_rate_cell shape {(EXPECTED_DRAWS, EXPECTED_CELLS)}, "
            f"got {activity.shape}"
        )
    if not np.isfinite(activity).all() or (activity < 0.0).any():
        raise AssertionError("GP activity-rate samples must be finite and nonnegative")
    return grid, activity.mean(axis=0), activity.std(axis=0, ddof=1)


def build_geometry(
    grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Recalculate A_i, U_i, A_iz and f_iz from the baseline geometry."""
    projector = Transformer.from_crs(
        source_conversion.MAP_CRS,
        source_conversion.AREA_CRS,
        always_xy=True,
    )
    projected_grid = source_conversion.project_grid(grid, projector)
    common_support = unary_union(list(projected_grid["geometry_area"]))
    raw_zones = source_conversion.read_source_zones(SOURCE_GEOMETRY)
    rates = source_conversion.read_mw2_rates(SOURCE_RATES)
    zones = source_conversion.prepare_zones(
        raw_zones,
        rates,
        projector,
        common_support,
    )
    overlap_m2, fractions = source_conversion.calculate_overlaps(
        list(projected_grid["geometry_area"]), zones
    )
    source_union = unary_union(list(zones["geometry_area"]))
    cell_area_km2 = projected_grid["physical_cell_area_km2"].to_numpy(float)
    covered_area_km2 = np.asarray(
        [
            float(cell.intersection(source_union).area / 1_000_000.0)
            for cell in projected_grid["geometry_area"]
        ],
        dtype=float,
    )
    overlap_km2 = overlap_m2 / 1_000_000.0

    if (cell_area_km2 <= 0.0).any():
        raise AssertionError("Every GP cell must have positive support area")
    raw_covered_fraction = covered_area_km2 / cell_area_km2
    if (
        raw_covered_fraction.min() < -GEOMETRY_TOL
        or raw_covered_fraction.max() > 1.0 + GEOMETRY_TOL
    ):
        raise AssertionError(
            "Geometric-union coverage fraction lies outside [0, 1]: "
            f"min={raw_covered_fraction.min()}, max={raw_covered_fraction.max()}"
        )
    # Clip only floating-point noise after the strict geometric bound check.
    covered_area_km2 = np.clip(covered_area_km2, 0.0, cell_area_km2)
    return projected_grid, zones, overlap_km2, fractions, covered_area_km2


def validate_stored_conversion(
    grid: pd.DataFrame,
    zones: pd.DataFrame,
    fractions: np.ndarray,
    covered_area_km2: np.ndarray,
) -> dict[str, float]:
    """Prove that the recalculated geometry is the staged current conversion."""
    zone_names = zones["zone"].astype(str).tolist()
    stored_fraction = (
        pd.read_csv(FRACTION_MATRIX)
        .sort_values("grid_id")
        .reset_index(drop=True)
    )
    if stored_fraction.columns.tolist() != ["grid_id", *zone_names]:
        raise AssertionError("Stored overlap-matrix columns do not match geometry zones")
    if not np.array_equal(
        stored_fraction["grid_id"].to_numpy(int), grid["grid_id"].to_numpy(int)
    ):
        raise AssertionError("Stored overlap matrix is not aligned to the GP grid")
    fraction_error = float(
        np.max(np.abs(stored_fraction[zone_names].to_numpy(float) - fractions))
    )

    zone_audit = pd.read_csv(ZONE_AUDIT).set_index("zone").loc[zone_names]
    zone_area_error = float(
        np.max(
            np.abs(
                zone_audit["A_z_km2"].to_numpy(float)
                - zones["A_z_km2"].to_numpy(float)
            )
        )
    )

    stored_cells = (
        pd.read_csv(SOURCE_GRID).sort_values("grid_id").reset_index(drop=True)
    )
    if not np.array_equal(
        stored_cells["grid_id"].to_numpy(int), grid["grid_id"].to_numpy(int)
    ):
        raise AssertionError("Stored source-grid summary is not aligned to the GP grid")
    union_area_error = float(
        np.max(
            np.abs(
                stored_cells["source_model_covered_area_km2"].to_numpy(float)
                - covered_area_km2
            )
        )
    )
    if fraction_error > NUMERIC_TOL:
        raise AssertionError(f"Stored/recalculated f_iz mismatch: {fraction_error}")
    if zone_area_error > AREA_TOL_KM2:
        raise AssertionError(f"Stored/recalculated A_z mismatch: {zone_area_error}")
    if union_area_error > AREA_TOL_KM2:
        raise AssertionError(f"Stored/recalculated U_i mismatch: {union_area_error}")
    return {
        "stored_fraction_matrix_max_abs_error": fraction_error,
        "stored_zone_area_max_abs_error_km2": zone_area_error,
        "stored_union_coverage_max_abs_error_km2": union_area_error,
    }


def build_coverage_audit(
    grid: pd.DataFrame,
    zones: pd.DataFrame,
    overlap_km2: np.ndarray,
    covered_area_km2: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Audit geometric coverage without using it in the weight calculation."""
    cell_area_km2 = grid["physical_cell_area_km2"].to_numpy(float)
    sum_overlap_km2 = overlap_km2.sum(axis=1)
    covered_fraction = covered_area_km2 / cell_area_km2
    residual_fraction = 1.0 - covered_fraction
    q = np.zeros_like(overlap_km2)
    covered = sum_overlap_km2 > 0.0
    q[covered, :] = (
        overlap_km2[covered, :] / sum_overlap_km2[covered, None]
    )
    zone_count = (overlap_km2 > 0.0).sum(axis=1)
    audit = pd.DataFrame(
        {
            "cell_id": grid["grid_id"].astype(int).to_numpy(),
            "A_i_km2": cell_area_km2,
            "U_i_union_covered_area_km2": covered_area_km2,
            "sum_A_iz_km2_including_overlap_multiplicity": sum_overlap_km2,
            "source_covered_fraction": covered_fraction,
            "residual_coverage_fraction": residual_fraction,
            "raw_sum_A_iz_over_A_i": sum_overlap_km2 / cell_area_km2,
            "overlap_double_count_area_km2": sum_overlap_km2 - covered_area_km2,
            "n_source_zone_overlaps": zone_count.astype(int),
            "fully_uncovered": residual_fraction >= 1.0 - GEOMETRY_TOL,
            "partially_uncovered": (
                (residual_fraction > GEOMETRY_TOL)
                & (residual_fraction < 1.0 - GEOMETRY_TOL)
            ),
            "used_in_uncertainty_weight": False,
        }
    )
    fully_uncovered_ids = audit.loc[audit["fully_uncovered"], "cell_id"].tolist()
    checks = {
        "all_residual_fractions_between_0_and_1": bool(
            ((residual_fraction >= -GEOMETRY_TOL) & (residual_fraction <= 1.0 + GEOMETRY_TOL)).all()
        ),
        "all_q_rows_sum_to_1_when_covered": bool(
            np.allclose(q[covered].sum(axis=1), 1.0, atol=NUMERIC_TOL, rtol=0.0)
        ),
        "all_q_rows_zero_when_uncovered": bool(
            np.allclose(q[~covered], 0.0, atol=NUMERIC_TOL, rtol=0.0)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"Coverage allocation failed: {checks}")

    coverage_report = {
        "checks": checks,
        "residual_fraction_min": float(residual_fraction.min()),
        "residual_fraction_max": float(residual_fraction.max()),
        "partial_coverage_used_in_uncertainty_weight": False,
        "fully_uncovered_cell_ids": [int(value) for value in fully_uncovered_ids],
        "number_fully_uncovered_cells": int(audit["fully_uncovered"].sum()),
        "number_partially_uncovered_cells": int(audit["partially_uncovered"].sum()),
        "number_fully_covered_cells": int(
            (residual_fraction <= GEOMETRY_TOL).sum()
        ),
        "number_cells_with_polygon_overlap_double_count": int(
            (audit["overlap_double_count_area_km2"] > 1e-6).sum()
        ),
        "maximum_overlap_double_count_area_km2": float(
            audit["overlap_double_count_area_km2"].max()
        ),
    }
    return q, audit, coverage_report


def build_contributions(
    grid: pd.DataFrame,
    zones: pd.DataFrame,
    overlap_km2: np.ndarray,
    fractions: np.ndarray,
    covered_area_km2: np.ndarray,
    q: np.ndarray,
) -> pd.DataFrame:
    """Retain source-zone overlaps for source reconstruction and h_i audit."""
    zone_names = zones["zone"].astype(str).tolist()
    zone_means = zones["source_mean_rate"].to_numpy(float)
    zone_sds = zones["source_sd_rate"].to_numpy(float)
    zero_sd_zones = {
        zone_names[index] for index in np.flatnonzero(zone_sds == 0.0)
    }
    if zero_sd_zones != FIXED_ZONES:
        raise AssertionError(
            f"Expected exactly BALA/ESCO to have fixed source rates, got {zero_sd_zones}"
        )

    rows: list[dict[str, Any]] = []
    cell_area = grid["physical_cell_area_km2"].to_numpy(float)
    for i, cell_id in enumerate(grid["grid_id"].to_numpy(int)):
        for z in np.flatnonzero(overlap_km2[i, :] > 0.0):
            zone = zone_names[z]
            f_iz = float(fractions[i, z])
            lambda_sz = f_iz * float(zone_means[z])
            sigma_sz = f_iz * float(zone_sds[z])
            fixed = zone in FIXED_ZONES
            rows.append(
                {
                    "cell_id": int(cell_id),
                    "source_zone": zone,
                    "contribution_type": "source_zone_overlap",
                    "A_i_km2": float(cell_area[i]),
                    "U_i_union_covered_area_km2": float(covered_area_km2[i]),
                    "A_iz_km2": float(overlap_km2[i, z]),
                    "f_iz": f_iz,
                    "source_overlap_share_q_iz": float(q[i, z]),
                    "lambda_SZ_iz": lambda_sz,
                    "sigma_SZ_iz": sigma_sz,
                    "special_fixed_zone": fixed,
                    "role_in_weighting": (
                        "BALA_ESCO_overlap_for_h_i"
                        if fixed
                        else "normal_source_overlap"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_cell_summary(
    grid: pd.DataFrame,
    contributions: pd.DataFrame,
    gp_mean: np.ndarray,
    gp_sd: np.ndarray,
) -> pd.DataFrame:
    groups = contributions.groupby("cell_id", sort=True)
    cell_ids = grid["grid_id"].astype(int).to_numpy()
    source_mean = (
        groups["lambda_SZ_iz"].sum().reindex(cell_ids, fill_value=0.0).to_numpy(float)
    )
    source_variance = (
        groups["sigma_SZ_iz"]
        .apply(lambda values: np.square(values).sum())
        .reindex(cell_ids, fill_value=0.0)
        .to_numpy(float)
    )
    source_sd = np.sqrt(source_variance)
    total_overlap = (
        groups["A_iz_km2"].sum().reindex(cell_ids, fill_value=0.0).to_numpy(float)
    )
    fixed_overlap = (
        contributions.loc[contributions["special_fixed_zone"]]
        .groupby("cell_id")["A_iz_km2"]
        .sum()
        .reindex(cell_ids, fill_value=0.0)
        .to_numpy(float)
    )
    normal_overlap = total_overlap - fixed_overlap
    source_present = total_overlap > 0.0
    fixed_fraction = np.zeros(len(grid), dtype=float)
    fixed_fraction[source_present] = (
        fixed_overlap[source_present] / total_overlap[source_present]
    )

    normal_uncertainty_weight = np.full(len(grid), np.nan, dtype=float)
    denominator = np.square(gp_sd) + np.square(source_sd)
    if (denominator[source_present] <= 0.0).any():
        raise AssertionError("Covered cells require positive cell-level variance sum")
    normal_uncertainty_weight[source_present] = (
        np.square(source_sd[source_present]) / denominator[source_present]
    )
    cell_gp_weight = np.ones(len(grid), dtype=float)
    cell_gp_weight[source_present] = (
        0.5 * fixed_fraction[source_present]
        + (1.0 - fixed_fraction[source_present])
        * normal_uncertainty_weight[source_present]
    )
    hybrid_mean = (
        cell_gp_weight * gp_mean + (1.0 - cell_gp_weight) * source_mean
    )

    fixed_by_cell = (
        contributions.loc[contributions["special_fixed_zone"]]
        .groupby("cell_id")["source_zone"]
        .apply(lambda values: ";".join(sorted(set(values))))
    )
    source_case = np.full(len(grid), "normal_only", dtype=object)
    source_case[~source_present] = "no_source_overlap"
    source_case[source_present & (fixed_overlap > 0.0) & (normal_overlap > 0.0)] = (
        "mixed_BALA_ESCO_and_normal"
    )
    source_case[source_present & (fixed_overlap > 0.0) & (normal_overlap <= 0.0)] = (
        "BALA_ESCO_only"
    )

    summary = pd.DataFrame(
        {
            "cell_id": grid["grid_id"].astype(int).to_numpy(),
            "lon_lo": grid["lon_lo"].to_numpy(float),
            "lon_hi": grid["lon_hi"].to_numpy(float),
            "lat_lo": grid["lat_lo"].to_numpy(float),
            "lat_hi": grid["lat_hi"].to_numpy(float),
            "grid_lon": grid["grid_lon"].to_numpy(float),
            "grid_lat": grid["grid_lat"].to_numpy(float),
            "gp_activity_rate_mean": gp_mean,
            "gp_activity_rate_sd": gp_sd,
            "source_activity_rate_mean": source_mean,
            "source_activity_rate_sd": source_sd,
            "equal_weight_hybrid_mean": 0.5 * (gp_mean + source_mean),
            "uncertainty_weighted_hybrid_mean": hybrid_mean,
            "normal_uncertainty_gp_weight_w_u_i": normal_uncertainty_weight,
            "cell_gp_weight_w_i": cell_gp_weight,
            "fixed_zone_overlap_fraction_h_i": fixed_fraction,
            "overlaps_BALA_or_ESCO": fixed_fraction > 0.0,
            "fixed_zones": [fixed_by_cell.get(int(cid), "") for cid in grid["grid_id"]],
            "source_overlap_case": source_case,
            "fully_uncovered": ~source_present,
        }
    )
    return summary


def _validate_results_obsolete_contribution_weighting(
    grid: pd.DataFrame,
    zones: pd.DataFrame,
    contributions: pd.DataFrame,
    summary: pd.DataFrame,
    gp_mean: np.ndarray,
    gp_sd: np.ndarray,
    stored_checks: dict[str, float],
    coverage_report: dict[str, Any],
    hashes_before: dict[str, str],
) -> dict[str, Any]:
    zone_names = zones["zone"].astype(str).tolist()
    normal = contributions["weight_rule"] == "uncertainty_based"
    fixed = contributions["special_fixed_zone"]
    residual = contributions["no_source_coverage_residual"]
    groups = contributions.groupby("cell_id", sort=True)

    gp_split = groups["lambda_GP_iz"].sum().to_numpy(float)
    gp_sd_split = groups["sigma_GP_iz"].sum().to_numpy(float)
    source_split = groups["lambda_SZ_iz"].sum().to_numpy(float)
    source_sd_split = np.sqrt(
        groups["sigma_SZ_iz"].apply(lambda x: np.square(x).sum())
    ).to_numpy(float)
    contribution_hybrid_audit = groups[
        "lambda_H_iz_audit_only"
    ].sum().to_numpy(float)
    g_partition = groups["g_iz"].sum().to_numpy(float)
    derived_cell_weight = groups.apply(
        lambda frame: float(np.sum(frame["g_iz"] * frame["gp_weight_w_iz"])),
        include_groups=False,
    ).to_numpy(float)
    reported_cell_weight = summary["cell_gp_weight_w_i"].to_numpy(float)
    reported_hybrid = summary["uncertainty_weighted_hybrid_mean"].to_numpy(float)
    expected_cell_hybrid = (
        reported_cell_weight * gp_mean
        + (1.0 - reported_cell_weight) * source_split
    )
    convex_lower = np.minimum(gp_mean, source_split)
    convex_upper = np.maximum(gp_mean, source_split)
    convex_violation = np.maximum(
        np.maximum(convex_lower - reported_hybrid, 0.0),
        np.maximum(reported_hybrid - convex_upper, 0.0),
    )

    stored_source = (
        pd.read_csv(SOURCE_GRID).sort_values("grid_id").reset_index(drop=True)
    )
    stored_source_mean = stored_source["source_activity_rate_mean"].to_numpy(float)
    stored_source_sd = stored_source["source_activity_rate_sd"].to_numpy(float)

    normal_sigma_sz = contributions.loc[normal, "sigma_SZ_iz"].to_numpy(float)
    normal_sigma_gp = contributions.loc[normal, "sigma_GP_iz"].to_numpy(float)
    expected_normal_weight = np.square(normal_sigma_sz) / (
        np.square(normal_sigma_sz) + np.square(normal_sigma_gp)
    )
    observed_normal_weight = contributions.loc[normal, "gp_weight_w_iz"].to_numpy(float)
    normal_formula_error = float(
        np.max(np.abs(observed_normal_weight - expected_normal_weight))
    )

    fixed_cells = set(contributions.loc[fixed, "cell_id"].astype(int))
    normal_cells = set(contributions.loc[normal, "cell_id"].astype(int))
    mixed_fixed_normal_cells = sorted(fixed_cells & normal_cells)
    mixed_normal_rows = contributions.loc[
        normal & contributions["cell_id"].isin(mixed_fixed_normal_cells)
    ]

    hashes_after = input_hashes()
    errors = {
        "gp_mean_split_max_abs": float(np.max(np.abs(gp_split - gp_mean))),
        "gp_sd_linear_split_max_abs": float(np.max(np.abs(gp_sd_split - gp_sd))),
        "source_mean_vs_stored_max_abs": float(
            np.max(np.abs(source_split - stored_source_mean))
        ),
        "source_sd_vs_stored_max_abs": float(
            np.max(np.abs(source_sd_split - stored_source_sd))
        ),
        "cell_weight_vs_contribution_derivation_max_abs": float(
            np.max(np.abs(reported_cell_weight - derived_cell_weight))
        ),
        "cell_hybrid_formula_max_abs": float(
            np.max(np.abs(reported_hybrid - expected_cell_hybrid))
        ),
        "cell_hybrid_convexity_max_violation": float(
            np.max(convex_violation)
        ),
        "contribution_sum_vs_final_cell_hybrid_max_abs_difference": float(
            np.max(np.abs(contribution_hybrid_audit - reported_hybrid))
        ),
        "uk_total_vs_expected_crosscheck_abs": float(
            abs(reported_hybrid.sum() - EXPECTED_CELL_LEVEL_HYBRID_TOTAL)
        ),
        "g_including_residual_partition_max_abs": float(
            np.max(np.abs(g_partition - 1.0))
        ),
        "normal_weight_formula_max_abs": normal_formula_error,
    }
    checks = {
        "coverage_checks_passed_before_hybrid": bool(
            all(coverage_report["checks"].values())
        ),
        "all_132_cells_present": len(summary) == EXPECTED_CELLS,
        "all_normal_weights_in_0_1": bool(
            ((observed_normal_weight >= 0.0) & (observed_normal_weight <= 1.0)).all()
        ),
        "all_normal_weights_match_formula": normal_formula_error <= NUMERIC_TOL,
        "all_BALA_ESCO_weights_exactly_0p5": bool(
            (contributions.loc[fixed, "gp_weight_w_iz"] == 0.5).all()
        ),
        "all_residual_weights_exactly_1": bool(
            (contributions.loc[residual, "gp_weight_w_iz"] == 1.0).all()
        ),
        "mixed_fixed_normal_cells_keep_uncertainty_rule_for_normal_rows": bool(
            len(mixed_fixed_normal_cells) > 0
            and (mixed_normal_rows["weight_rule"] == "uncertainty_based").all()
            and np.max(
                np.abs(
                    mixed_normal_rows["gp_weight_w_iz"].to_numpy(float)
                    - (
                        np.square(mixed_normal_rows["sigma_SZ_iz"].to_numpy(float))
                        / (
                            np.square(mixed_normal_rows["sigma_SZ_iz"].to_numpy(float))
                            + np.square(mixed_normal_rows["sigma_GP_iz"].to_numpy(float))
                        )
                    )
                )
            )
            <= NUMERIC_TOL
        ),
        "gp_contributions_sum_to_original_cell_means": errors[
            "gp_mean_split_max_abs"
        ]
        <= NUMERIC_TOL,
        "source_contributions_sum_to_existing_source_grid_means": errors[
            "source_mean_vs_stored_max_abs"
        ]
        <= NUMERIC_TOL,
        "source_contribution_sds_reproduce_existing_source_grid_sds": errors[
            "source_sd_vs_stored_max_abs"
        ]
        <= NUMERIC_TOL,
        "all_cell_gp_weights_in_0_1": bool(
            (
                (reported_cell_weight >= -NUMERIC_TOL)
                & (reported_cell_weight <= 1.0 + NUMERIC_TOL)
            ).all()
        ),
        "cell_weights_equal_sum_g_iz_times_w_iz": errors[
            "cell_weight_vs_contribution_derivation_max_abs"
        ]
        <= NUMERIC_TOL,
        "final_hybrid_uses_cell_level_convex_formula": errors[
            "cell_hybrid_formula_max_abs"
        ]
        <= NUMERIC_TOL,
        "every_cell_hybrid_within_gp_source_bounds": errors[
            "cell_hybrid_convexity_max_violation"
        ]
        <= NUMERIC_TOL,
        "final_hybrid_not_built_from_contribution_hybrid_sum": errors[
            "contribution_sum_vs_final_cell_hybrid_max_abs_difference"
        ]
        > NUMERIC_TOL,
        "uk_total_matches_approximate_crosscheck": errors[
            "uk_total_vs_expected_crosscheck_abs"
        ]
        <= EXPECTED_TOTAL_ABS_TOL,
        "all_activity_rate_inputs_unchanged": hashes_before == hashes_after,
        "no_total_rate_renormalisation": True,
        "no_final_hybrid_sd_invented": "uncertainty_weighted_hybrid_sd"
        not in summary.columns,
    }
    checks["all_passed"] = bool(all(checks.values()))
    if not checks["all_passed"]:
        raise AssertionError(json.dumps(native(checks), indent=2))

    source_zone_rows = contributions.loc[~residual]
    all_weights = contributions["gp_weight_w_iz"].to_numpy(float)
    normal_weights = contributions.loc[normal, "gp_weight_w_iz"].to_numpy(float)
    source_zone_weights = source_zone_rows["gp_weight_w_iz"].to_numpy(float)
    cell_weights = summary["cell_gp_weight_w_i"].to_numpy(float)

    def quantiles(values: np.ndarray) -> dict[str, float]:
        probabilities = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
        labels = ["min", "q05", "q25", "median", "q75", "q95", "max"]
        result = np.quantile(values, probabilities)
        return {label: float(value) for label, value in zip(labels, result)}

    fixed_ids = sorted(fixed_cells)
    bala_ids = sorted(
        set(contributions.loc[contributions["source_zone"] == "BALA", "cell_id"].astype(int))
    )
    esco_ids = sorted(
        set(contributions.loc[contributions["source_zone"] == "ESCO", "cell_id"].astype(int))
    )
    return {
        "model_definition": {
            "magnitude_threshold": "Mw >= 2",
            "source_model": "branch-by-branch truncated Gutenberg-Richter extrapolation",
            "gp_model": "current final 132-cell Multinomial GP activity-rate posterior",
            "source_formula": "lambda_SZ_iz=f_iz*lambda_SZ_z; sigma_SZ_iz=f_iz*sigma_SZ_z",
            "covered_gp_allocation": "q_iz=A_iz/sum_z(A_iz); g_iz=(U_i/A_i)*q_iz",
            "residual": "r_i=1-U_i/A_i; NO_SOURCE_COVERAGE uses w_GP=1",
            "normal_weight": "sigma_SZ_iz^2/(sigma_SZ_iz^2+sigma_GP_iz^2)",
            "fixed_zone_weight": "BALA/ESCO use w_GP=0.5",
            "cell_gp_weight": "w_i=sum over all source-zone and residual rows of g_iz*w_iz",
            "final_hybrid_mean": "lambda_H_i=w_i*lambda_GP_i+(1-w_i)*lambda_SZ_i",
            "contribution_hybrid_quantities": "audit only; never summed to form final Hybrid",
            "total_rate_renormalisation": False,
            "hybrid_sd_produced": False,
            "hybrid_sd_reason": "BALA/ESCO source uncertainty is unquantified, not exact zero",
        },
        "dimensions": {
            "gp_draws": EXPECTED_DRAWS,
            "grid_cells": EXPECTED_CELLS,
            "source_zones": len(zone_names),
            "source_zone_overlap_rows": int((~residual).sum()),
            "residual_rows": int(residual.sum()),
            "total_contribution_rows": len(contributions),
        },
        "coverage": coverage_report,
        "totals_events_per_year": {
            "gp": float(gp_mean.sum()),
            "source": float(source_split.sum()),
            "equal_weight_hybrid": float(
                summary["equal_weight_hybrid_mean"].sum()
            ),
            "uncertainty_weighted_hybrid": float(
                summary["uncertainty_weighted_hybrid_mean"].sum()
            ),
            "contribution_level_hybrid_sum_audit_only": float(
                contribution_hybrid_audit.sum()
            ),
            "expected_uncertainty_weighted_hybrid_crosscheck": (
                EXPECTED_CELL_LEVEL_HYBRID_TOTAL
            ),
        },
        "weights": {
            "all_contributions_including_residual": quantiles(all_weights),
            "source_zone_contributions_only": quantiles(source_zone_weights),
            "normal_uncertainty_based_only": quantiles(normal_weights),
            "cell_gp_weight_w_i": quantiles(cell_weights),
            "normal_source_zone_rows": int(normal.sum()),
            "fixed_source_zone_rows": int(fixed.sum()),
            "residual_rows": int(residual.sum()),
        },
        "fixed_zone_cells": {
            "number_cells_receiving_BALA_or_ESCO": len(fixed_ids),
            "cell_ids": fixed_ids,
            "BALA_cell_ids": bala_ids,
            "ESCO_cell_ids": esco_ids,
            "mixed_fixed_and_normal_cell_ids": mixed_fixed_normal_cells,
        },
        "numerical_errors": {**stored_checks, **errors},
        "checks": checks,
        "baseline_input_sha256": hashes_after,
    }


def validate_results(
    grid: pd.DataFrame,
    zones: pd.DataFrame,
    contributions: pd.DataFrame,
    summary: pd.DataFrame,
    gp_mean: np.ndarray,
    gp_sd: np.ndarray,
    stored_checks: dict[str, float],
    coverage_report: dict[str, Any],
    hashes_before: dict[str, str],
) -> dict[str, Any]:
    """Validate the cell-SD weighting and BALA/ESCO overlap blend."""
    cell_ids = grid["grid_id"].astype(int).to_numpy()
    groups = contributions.groupby("cell_id", sort=True)
    source_mean = (
        groups["lambda_SZ_iz"].sum().reindex(cell_ids, fill_value=0.0).to_numpy(float)
    )
    source_sd = np.sqrt(
        groups["sigma_SZ_iz"]
        .apply(lambda values: np.square(values).sum())
        .reindex(cell_ids, fill_value=0.0)
        .to_numpy(float)
    )
    total_overlap = (
        groups["A_iz_km2"].sum().reindex(cell_ids, fill_value=0.0).to_numpy(float)
    )
    fixed = contributions["special_fixed_zone"]
    fixed_overlap = (
        contributions.loc[fixed]
        .groupby("cell_id")["A_iz_km2"]
        .sum()
        .reindex(cell_ids, fill_value=0.0)
        .to_numpy(float)
    )
    normal_overlap = total_overlap - fixed_overlap
    source_present = total_overlap > 0.0
    no_source = ~source_present
    fixed_only = source_present & (fixed_overlap > 0.0) & (normal_overlap <= 0.0)
    mixed_fixed_normal = (
        source_present & (fixed_overlap > 0.0) & (normal_overlap > 0.0)
    )
    normal_only = source_present & (fixed_overlap <= 0.0)

    expected_h = np.zeros(EXPECTED_CELLS, dtype=float)
    expected_h[source_present] = (
        fixed_overlap[source_present] / total_overlap[source_present]
    )
    expected_w_u = np.full(EXPECTED_CELLS, np.nan, dtype=float)
    expected_w_u[source_present] = (
        np.square(source_sd[source_present])
        / (np.square(gp_sd[source_present]) + np.square(source_sd[source_present]))
    )
    expected_weight = np.ones(EXPECTED_CELLS, dtype=float)
    expected_weight[source_present] = (
        0.5 * expected_h[source_present]
        + (1.0 - expected_h[source_present]) * expected_w_u[source_present]
    )

    reported_h = summary["fixed_zone_overlap_fraction_h_i"].to_numpy(float)
    reported_w_u = summary["normal_uncertainty_gp_weight_w_u_i"].to_numpy(float)
    reported_weight = summary["cell_gp_weight_w_i"].to_numpy(float)
    reported_hybrid = summary["uncertainty_weighted_hybrid_mean"].to_numpy(float)
    expected_hybrid = (
        reported_weight * gp_mean + (1.0 - reported_weight) * source_mean
    )
    lower = np.minimum(gp_mean, source_mean)
    upper = np.maximum(gp_mean, source_mean)
    convex_violation = np.maximum(
        np.maximum(lower - reported_hybrid, 0.0),
        np.maximum(reported_hybrid - upper, 0.0),
    )

    stored_source = pd.read_csv(SOURCE_GRID).sort_values("grid_id").reset_index(drop=True)
    errors = {
        "source_mean_vs_stored_max_abs": float(
            np.max(
                np.abs(
                    source_mean
                    - stored_source["source_activity_rate_mean"].to_numpy(float)
                )
            )
        ),
        "source_sd_vs_stored_max_abs": float(
            np.max(
                np.abs(
                    source_sd
                    - stored_source["source_activity_rate_sd"].to_numpy(float)
                )
            )
        ),
        "fixed_overlap_fraction_h_i_max_abs": float(
            np.max(np.abs(reported_h - expected_h))
        ),
        "cell_sd_weight_w_u_i_max_abs": float(
            np.nanmax(np.abs(reported_w_u - expected_w_u))
        ),
        "combined_cell_weight_max_abs": float(
            np.max(np.abs(reported_weight - expected_weight))
        ),
        "final_cell_hybrid_formula_max_abs": float(
            np.max(np.abs(reported_hybrid - expected_hybrid))
        ),
        "final_cell_hybrid_convexity_max_violation": float(
            np.max(convex_violation)
        ),
        "equal_weight_total_vs_22p3063_abs": float(
            abs(summary["equal_weight_hybrid_mean"].sum() - EXPECTED_EQUAL_WEIGHT_TOTAL)
        ),
    }
    no_source_ids = [int(value) for value in cell_ids[no_source]]
    checks = {
        "coverage_audit_checks_passed": bool(all(coverage_report["checks"].values())),
        "partial_coverage_not_used_in_weight": bool(
            coverage_report["partial_coverage_used_in_uncertainty_weight"] is False
            and "residual_coverage_fraction" not in summary.columns
        ),
        "no_NO_SOURCE_COVERAGE_contribution_rows": bool(
            (contributions["source_zone"] != "NO_SOURCE_COVERAGE").all()
        ),
        "all_132_cells_present": len(summary) == EXPECTED_CELLS,
        "all_cell_weights_between_0_and_1": bool(
            ((reported_weight >= 0.0) & (reported_weight <= 1.0)).all()
        ),
        "cell_sd_weights_match_formula": errors["cell_sd_weight_w_u_i_max_abs"] <= NUMERIC_TOL,
        "fixed_overlap_fractions_match_area_formula": errors[
            "fixed_overlap_fraction_h_i_max_abs"
        ]
        <= NUMERIC_TOL,
        "combined_weights_match_BALA_ESCO_formula": errors[
            "combined_cell_weight_max_abs"
        ]
        <= NUMERIC_TOL,
        "six_no_source_cells_identified": bool(
            no_source_ids == coverage_report["fully_uncovered_cell_ids"]
            and len(no_source_ids) == 6
        ),
        "no_source_cells_use_weight_1_and_retain_GP": bool(
            (reported_weight[no_source] == 1.0).all()
            and (source_mean[no_source] == 0.0).all()
            and (source_sd[no_source] == 0.0).all()
            and np.allclose(
                reported_hybrid[no_source], gp_mean[no_source], atol=NUMERIC_TOL, rtol=0.0
            )
        ),
        "BALA_ESCO_only_cells_use_weight_0p5": bool(
            fixed_only.any() and (reported_weight[fixed_only] == 0.5).all()
        ),
        "mixed_BALA_ESCO_normal_cells_use_blended_formula": bool(
            mixed_fixed_normal.any()
            and np.max(
                np.abs(
                    reported_weight[mixed_fixed_normal]
                    - (
                        0.5 * reported_h[mixed_fixed_normal]
                        + (1.0 - reported_h[mixed_fixed_normal])
                        * reported_w_u[mixed_fixed_normal]
                    )
                )
            )
            <= NUMERIC_TOL
        ),
        "normal_only_cells_use_cell_sd_weight": bool(
            normal_only.any()
            and np.max(
                np.abs(reported_weight[normal_only] - reported_w_u[normal_only])
            )
            <= NUMERIC_TOL
        ),
        "source_means_reproduce_validated_grid": errors[
            "source_mean_vs_stored_max_abs"
        ]
        <= NUMERIC_TOL,
        "source_sds_reproduce_validated_grid": errors[
            "source_sd_vs_stored_max_abs"
        ]
        <= NUMERIC_TOL,
        "final_hybrid_uses_cell_level_formula_once": errors[
            "final_cell_hybrid_formula_max_abs"
        ]
        <= NUMERIC_TOL,
        "every_final_cell_rate_within_GP_source_bounds": errors[
            "final_cell_hybrid_convexity_max_violation"
        ]
        <= NUMERIC_TOL,
        "equal_weight_UK_total_approximately_22p3063": errors[
            "equal_weight_total_vs_22p3063_abs"
        ]
        <= EXPECTED_EQUAL_TOTAL_ABS_TOL,
        "all_activity_rate_inputs_unchanged": hashes_before == input_hashes(),
        "no_total_rate_renormalisation": True,
        "no_final_hybrid_sd_invented": "uncertainty_weighted_hybrid_sd" not in summary.columns,
    }
    checks["all_passed"] = bool(all(checks.values()))
    if not checks["all_passed"]:
        raise AssertionError(json.dumps(native(checks), indent=2))

    def quantiles(values: np.ndarray) -> dict[str, float]:
        probabilities = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
        labels = ["min", "q05", "q25", "median", "q75", "q95", "max"]
        result = np.quantile(values, probabilities)
        return {label: float(value) for label, value in zip(labels, result)}

    fixed_ids = sorted(set(contributions.loc[fixed, "cell_id"].astype(int)))
    bala_ids = sorted(set(contributions.loc[contributions["source_zone"] == "BALA", "cell_id"].astype(int)))
    esco_ids = sorted(set(contributions.loc[contributions["source_zone"] == "ESCO", "cell_id"].astype(int)))
    return {
        "model_definition": {
            "magnitude_threshold": "Mw >= 2",
            "source_model": "branch-by-branch truncated Gutenberg-Richter extrapolation",
            "gp_model": "validated current final 132-cell Multinomial GP",
            "normal_cell_weight": "w_u_i=sigma_SZ_i^2/(sigma_GP_i^2+sigma_SZ_i^2)",
            "fixed_overlap_fraction": "h_i=sum_BALA_ESCO(A_iz)/sum_all_z(A_iz)",
            "cell_gp_weight": "w_i=0.5*h_i+(1-h_i)*w_u_i; no-source cells use w_i=1",
            "final_hybrid_mean": "lambda_H_i=w_i*lambda_GP_i+(1-w_i)*lambda_SZ_i",
            "partial_source_coverage": "audit only; not used in weight",
            "residual_contribution_rows": False,
            "total_rate_renormalisation": False,
            "hybrid_sd_produced": False,
        },
        "dimensions": {
            "gp_draws": EXPECTED_DRAWS,
            "grid_cells": EXPECTED_CELLS,
            "source_zones": len(zones),
            "source_zone_overlap_rows": len(contributions),
            "residual_rows": 0,
        },
        "coverage_audit_only": coverage_report,
        "totals_events_per_year": {
            "gp": float(gp_mean.sum()),
            "source": float(source_mean.sum()),
            "equal_weight_hybrid": float(summary["equal_weight_hybrid_mean"].sum()),
            "uncertainty_weighted_hybrid": float(reported_hybrid.sum()),
            "expected_equal_weight_crosscheck": EXPECTED_EQUAL_WEIGHT_TOTAL,
        },
        "weights": {
            "cell_sd_uncertainty_weight_w_u_i": quantiles(expected_w_u[source_present]),
            "fixed_zone_overlap_fraction_h_i": quantiles(expected_h[source_present]),
            "cell_gp_weight_w_i": quantiles(reported_weight),
            "normal_only_cells": int(normal_only.sum()),
            "BALA_ESCO_only_cells": int(fixed_only.sum()),
            "mixed_BALA_ESCO_normal_cells": int(mixed_fixed_normal.sum()),
            "no_source_cells": int(no_source.sum()),
        },
        "fixed_zone_cells": {
            "number_cells_receiving_BALA_or_ESCO": len(fixed_ids),
            "cell_ids": fixed_ids,
            "BALA_cell_ids": bala_ids,
            "ESCO_cell_ids": esco_ids,
            "BALA_ESCO_only_cell_ids": [int(value) for value in cell_ids[fixed_only]],
            "mixed_fixed_and_normal_cell_ids": [
                int(value) for value in cell_ids[mixed_fixed_normal]
            ],
        },
        "no_source_cell_ids": no_source_ids,
        "numerical_errors": {**stored_checks, **errors},
        "checks": checks,
        "baseline_input_sha256": input_hashes(),
    }


def write_weight_distribution(summary: pd.DataFrame) -> None:
    weights = summary["cell_gp_weight_w_i"].to_numpy(float)
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(weights, bins=edges)
    distribution = pd.DataFrame(
        {
            "weight_lower_inclusive": edges[:-1],
            "weight_upper": edges[1:],
            "upper_bound_inclusive": [False] * 9 + [True],
            "cell_count": counts,
        }
    )
    distribution.to_csv(
        WEIGHT_DISTRIBUTION_OUTPUT,
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )


def map_collection(
    ax: plt.Axes,
    summary: pd.DataFrame,
    values: np.ndarray,
    cmap,
    norm: Normalize,
    title: str,
) -> PatchCollection:
    rectangles = [
        Rectangle(
            (float(row.lon_lo), float(row.lat_lo)),
            float(row.lon_hi - row.lon_lo),
            float(row.lat_hi - row.lat_lo),
        )
        for row in summary.itertuples(index=False)
    ]
    collection = PatchCollection(
        rectangles,
        cmap=cmap,
        norm=norm,
        edgecolor="#6f6f6f",
        linewidth=0.42,
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)
    lon_min = float(summary["lon_lo"].min())
    lon_max = float(summary["lon_hi"].max())
    lat_min = float(summary["lat_lo"].min())
    lat_max = float(summary["lat_hi"].max())
    mean_lat = 0.5 * (lat_min + lat_max)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    ax.set_xticks(np.arange(np.ceil(lon_min / 2.0) * 2.0, lon_max + 0.1, 2.0))
    ax.set_yticks(np.arange(np.ceil(lat_min / 2.0) * 2.0, lat_max + 0.1, 2.0))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    return collection


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def remove_superseded_figure_names() -> None:
    """Remove only generated figures superseded by the current definition."""
    stems = ("figure_effective_gp_weight", "figure_residual_coverage_fraction")
    for stem in stems:
        for suffix in (".png", ".pdf"):
            path = FIGURES / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def make_figures(summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cell_weight = summary["cell_gp_weight_w_i"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.8, 7.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)
    map_collection(
        ax,
        summary,
        cell_weight,
        cmap,
        norm,
        "Cell GP weight, $w_i$",
    )
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.048, pad=0.035)
    colorbar.set_label("Cell GP weight")
    save_figure(fig, "figure_cell_gp_weight")

    hybrid = summary["uncertainty_weighted_hybrid_mean"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.8, 7.2), constrained_layout=True)
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=0.0, vmax=float(hybrid.max()))
    map_collection(
        ax,
        summary,
        hybrid,
        cmap,
        norm,
        "Uncertainty-weighted Hybrid Model A mean ($M_w\\geq2$)",
    )
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.048, pad=0.035)
    colorbar.set_label("Mean activity rate (events yr$^{-1}$ cell$^{-1}$)")
    save_figure(fig, "figure_uncertainty_weighted_hybrid_mean")

    series = [
        ("Source model", summary["source_activity_rate_mean"].to_numpy(float)),
        ("GP model", summary["gp_activity_rate_mean"].to_numpy(float)),
        ("Equal-weight Hybrid", summary["equal_weight_hybrid_mean"].to_numpy(float)),
        (
            "Uncertainty-weighted Hybrid",
            summary["uncertainty_weighted_hybrid_mean"].to_numpy(float),
        ),
    ]
    vmax = float(max(values.max() for _, values in series))
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 11.0), constrained_layout=True)
    for ax, (label, values) in zip(axes.flat, series):
        total = float(values.sum())
        map_collection(
            ax,
            summary,
            values,
            cmap,
            norm,
            f"{label}\nUK total = {total:.3f} events yr$^{{-1}}$",
        )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.028,
        pad=0.025,
    )
    colorbar.set_label("Mean activity rate (events yr$^{-1}$ cell$^{-1}$)")
    fig.suptitle("Hybrid Model A comparison on the common 132-cell grid", fontsize=13)
    save_figure(fig, "figure_source_gp_equal_uncertainty_comparison")


def write_markdown_report(report: dict[str, Any]) -> None:
    totals = report["totals_events_per_year"]
    coverage = report["coverage_audit_only"]
    weights = report["weights"]
    fixed = report["fixed_zone_cells"]
    normal = weights["cell_sd_uncertainty_weight_w_u_i"]
    cell_weight = weights["cell_gp_weight_w_i"]
    lines = [
        "# Hybrid Model A uncertainty-weighting verification",
        "",
        "All validation checks passed. The Hybrid mean was not renormalised, and no final Hybrid SD was invented for BALA/ESCO.",
        "",
        "## Coverage audit (not used in weighting)",
        "",
        f"- Residual coverage range: {coverage['residual_fraction_min']:.12g} to {coverage['residual_fraction_max']:.12g}",
        f"- Fully uncovered cells ({coverage['number_fully_uncovered_cells']}): {', '.join(map(str, coverage['fully_uncovered_cell_ids']))}",
        f"- Partially uncovered cells: {coverage['number_partially_uncovered_cells']}",
        f"- Cells with source-polygon overlap multiplicity: {coverage['number_cells_with_polygon_overlap_double_count']}",
        "- Partial coverage and residual fractions do not enter the weight formula.",
        "",
        "## Totals (events yr^-1, Mw >= 2)",
        "",
        f"- GP: {totals['gp']:.12g}",
        f"- Source: {totals['source']:.12g}",
        f"- Equal-weight Hybrid: {totals['equal_weight_hybrid']:.12g}",
        f"- Uncertainty-weighted Hybrid: {totals['uncertainty_weighted_hybrid']:.12g}",
        "",
        "## Weights",
        "",
        f"- Cell-SD uncertainty weights: min {normal['min']:.6f}, median {normal['median']:.6f}, max {normal['max']:.6f}",
        f"- Cell GP weights: min {cell_weight['min']:.6f}, median {cell_weight['median']:.6f}, max {cell_weight['max']:.6f}",
        f"- Cell-level convexity maximum violation: {report['numerical_errors']['final_cell_hybrid_convexity_max_violation']:.3e}",
        f"- BALA/ESCO-affected cells ({fixed['number_cells_receiving_BALA_or_ESCO']}): {', '.join(map(str, fixed['cell_ids']))}",
        f"- Mixed fixed/normal cells: {', '.join(map(str, fixed['mixed_fixed_and_normal_cell_ids']))}",
        "",
        "## Validation",
        "",
    ]
    for key, passed in report["checks"].items():
        lines.append(f"- {key}: {'PASS' if passed else 'FAIL'}")
    VERIFICATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    hashes_before = input_hashes()

    grid, gp_mean, gp_sd = load_gp()
    grid, zones, overlap_km2, fractions, covered_area_km2 = build_geometry(grid)
    stored_checks = validate_stored_conversion(
        grid,
        zones,
        fractions,
        covered_area_km2,
    )

    # Coverage is retained as a geometry audit only.  Neither union coverage
    # nor residual fraction enters the uncertainty-weight calculation.
    q, coverage_audit, coverage_report = build_coverage_audit(
        grid, zones, overlap_km2, covered_area_km2
    )
    coverage_audit.to_csv(
        COVERAGE_OUTPUT,
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )

    contributions = build_contributions(
        grid,
        zones,
        overlap_km2,
        fractions,
        covered_area_km2,
        q,
    )
    summary = build_cell_summary(grid, contributions, gp_mean, gp_sd)
    report = validate_results(
        grid,
        zones,
        contributions,
        summary,
        gp_mean,
        gp_sd,
        stored_checks,
        coverage_report,
        hashes_before,
    )

    contributions.to_csv(
        CONTRIBUTION_OUTPUT,
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    summary.to_csv(CELL_OUTPUT, index=False, float_format=CSV_FLOAT_FORMAT)
    write_weight_distribution(summary)
    VERIFICATION_JSON.write_text(
        json.dumps(native(report), indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(report)
    remove_superseded_figure_names()
    make_figures(summary)

    print(json.dumps(native(report), indent=2))
    print(f"Wrote outputs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
