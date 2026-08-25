"""Convert the 22-zone Mw>=2 source model to the 132-cell GP grid.

The conversion uses physical overlap areas in EPSG:3035 and propagates
independent cross-zone standard deviations. It performs no Hybrid weighting.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import segmentize
from shapely.geometry import Polygon, box
from shapely.ops import transform, unary_union
from shapely.validation import make_valid


SCRIPT = Path(__file__).resolve()
HYBRID_ROOT = SCRIPT.parents[1]
PROJECT_ROOT = HYBRID_ROOT.parent

SOURCE_RATES = (
    PROJECT_ROOT / "source_model" / "output" / "source_zone_activity_rates_mw2_extrapolated.csv"
)
SOURCE_GEOMETRY = PROJECT_ROOT / "source_model" / "data" / "geometry_SSCmodel"
GRID_CELLS = PROJECT_ROOT / "gp" / "input" / "grid_cells.csv"
GP_CELL_SUMMARY = (
    PROJECT_ROOT / "gp" / "output" / "tables" / "cell_posterior_summary.csv"
)

OUTPUT_DIR = HYBRID_ROOT / "output"
CELL_OUTPUT = OUTPUT_DIR / "source_model_132_grid_mw2.csv"
FRACTION_OUTPUT = OUTPUT_DIR / "source_zone_to_grid_fraction_matrix.csv"
ZONE_AUDIT_OUTPUT = OUTPUT_DIR / "source_zone_to_grid_zone_audit.csv"
VERIFICATION_OUTPUT = OUTPUT_DIR / "source_to_grid_verification.json"

MAP_CRS = "EPSG:4326"
AREA_CRS = "EPSG:3035"
MAX_SEGMENT_DEGREES = 0.05
EXPECTED_ZONE_COUNT = 22
EXPECTED_CELL_COUNT = 132
EXPECTED_SOURCE_MEAN_TOTAL = 29.1457934998
SOURCE_TOTAL_CHECK_TOLERANCE = 1e-9
CONSERVATION_ABS_TOLERANCE = 1e-10
AREA_CLOSURE_REL_TOLERANCE = 1e-10
FORMULA_ABS_TOLERANCE = 1e-13


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_source_zones(path: Path) -> list[dict[str, object]]:
    """Read the BGS ASCII geometry exactly as in the previous implementation."""

    header = re.compile(r"^\s*([^,]+),\s*(\d+)\s*$")
    lines = path.read_text(encoding="utf-8").splitlines()
    zones: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        index += 1
        if not current:
            continue
        match = header.match(current)
        if match is None:
            raise ValueError(f"Invalid source-zone header: {current!r}")
        zone = match.group(1).strip()
        n_vertices = int(match.group(2))
        vertices: list[tuple[float, float]] = []
        for _ in range(n_vertices):
            fields = lines[index].split()
            index += 1
            if len(fields) < 2:
                raise ValueError(f"Invalid vertex for source zone {zone}")
            latitude, longitude = map(float, fields[:2])
            vertices.append((longitude, latitude))
        geometry = make_valid(Polygon(vertices))
        if geometry.is_empty:
            raise ValueError(f"Empty geometry for source zone {zone}")
        zones.append({"zone": zone, "geometry": geometry})

    names = [str(item["zone"]) for item in zones]
    if len(zones) != EXPECTED_ZONE_COUNT or len(set(names)) != EXPECTED_ZONE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_ZONE_COUNT} unique source zones, found {len(zones)}"
        )
    return zones


def read_grid(path: Path) -> pd.DataFrame:
    """Read and preserve the baseline 132-cell grid and ordering."""

    grid = pd.read_csv(path).sort_values("grid_id").reset_index(drop=True)
    required = {
        "grid_id",
        "lon_lo",
        "lon_hi",
        "lat_lo",
        "lat_hi",
        "grid_lon",
        "grid_lat",
    }
    missing = required - set(grid.columns)
    if missing:
        raise KeyError(f"Grid is missing required columns: {sorted(missing)}")
    if len(grid) != EXPECTED_CELL_COUNT or grid["grid_id"].nunique() != EXPECTED_CELL_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_CELL_COUNT} unique GP cells, found {len(grid)}"
        )
    grid["geometry"] = [
        box(row.lon_lo, row.lat_lo, row.lon_hi, row.lat_hi)
        for row in grid.itertuples(index=False)
    ]
    return grid


def project_geometry(geometry: object, projector: Transformer) -> object:
    """Densify in EPSG:4326, then transform to EPSG:3035 as previously."""

    projected = make_valid(
        transform(
            projector.transform,
            segmentize(geometry, max_segment_length=MAX_SEGMENT_DEGREES),
        )
    )
    if projected.is_empty or projected.area <= 0:
        raise AssertionError("Projected geometry has no positive physical area")
    return projected


def project_grid(grid: pd.DataFrame, projector: Transformer) -> pd.DataFrame:
    result = grid.copy()
    result["geometry_area"] = [
        project_geometry(geometry, projector) for geometry in result["geometry"]
    ]
    result["physical_cell_area_km2"] = [
        float(geometry.area / 1_000_000.0) for geometry in result["geometry_area"]
    ]
    return result


def read_mw2_rates(path: Path) -> pd.DataFrame:
    """Select only the 22 physical-zone rows at threshold Mw=2.0."""

    source = pd.read_csv(path)
    required = {"zone", "threshold_mw", "mean_rate_per_year", "sd_rate_per_year"}
    missing = required - set(source.columns)
    if missing:
        raise KeyError(f"Source table is missing columns: {sorted(missing)}")
    rates = source.loc[np.isclose(source["threshold_mw"], 2.0)].copy()
    if len(rates) != EXPECTED_ZONE_COUNT or rates["zone"].nunique() != EXPECTED_ZONE_COUNT:
        raise AssertionError("Mw=2 source table does not contain 22 physical zones")
    for column in ("mean_rate_per_year", "sd_rate_per_year"):
        rates[column] = pd.to_numeric(rates[column], errors="raise")
        if not np.isfinite(rates[column]).all() or (rates[column] < 0).any():
            raise AssertionError(f"Invalid values in source column {column}")
    total = float(rates["mean_rate_per_year"].sum())
    if abs(total - EXPECTED_SOURCE_MEAN_TOTAL) > SOURCE_TOTAL_CHECK_TOLERANCE:
        raise AssertionError(
            f"Unexpected source-zone Mw>=2 mean total: {total:.15g}"
        )
    return rates


def prepare_zones(
    raw_zones: list[dict[str, object]],
    rates: pd.DataFrame,
    projector: Transformer,
    common_support: object,
) -> pd.DataFrame:
    """Clip every projected source zone to the union of the 132 GP cells."""

    rate_lookup = rates.set_index("zone")
    geometry_names = {str(item["zone"]) for item in raw_zones}
    if set(rate_lookup.index.astype(str)) != geometry_names:
        raise AssertionError("Source rate and geometry zone names do not match")

    records: list[dict[str, object]] = []
    for item in raw_zones:
        zone = str(item["zone"])
        full_geometry_area = project_geometry(item["geometry"], projector)
        geometry_area = make_valid(full_geometry_area.intersection(common_support))
        area_m2 = float(geometry_area.area)
        if geometry_area.is_empty or area_m2 <= 0:
            raise AssertionError(f"Zone {zone} has no area on common GP support")
        records.append(
            {
                "zone": zone,
                "geometry_area": geometry_area,
                "full_source_zone_area_km2": float(
                    full_geometry_area.area / 1_000_000.0
                ),
                "A_z_km2": area_m2 / 1_000_000.0,
                "source_mean_rate": float(
                    rate_lookup.loc[zone, "mean_rate_per_year"]
                ),
                "source_sd_rate": float(rate_lookup.loc[zone, "sd_rate_per_year"]),
            }
        )
    return pd.DataFrame(records)


def calculate_overlaps(
    projected_cells: list[object], zones: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return A_iz in m2 and f_iz=A_iz/A_z using clipped common-support A_z."""

    overlap_m2 = np.asarray(
        [
            [float(cell.intersection(zone).area) for zone in zones["geometry_area"]]
            for cell in projected_cells
        ],
        dtype=float,
    )
    zone_area_m2 = zones["A_z_km2"].to_numpy(float) * 1_000_000.0
    fractions = overlap_m2 / zone_area_m2[None, :]
    if not np.isfinite(fractions).all() or (fractions < 0).any():
        raise AssertionError("Invalid source-zone overlap fractions")
    return overlap_m2, fractions


def source_moments(
    fractions: np.ndarray, zone_means: np.ndarray, zone_sds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply linear means and independent cross-zone variance propagation."""

    means = fractions @ zone_means
    variances = np.square(fractions) @ np.square(zone_sds)
    return means, np.sqrt(variances)


def maximum_relative_difference(observed: np.ndarray, reference: np.ndarray) -> float:
    difference = np.abs(observed - reference)
    nonzero = np.abs(reference) > 1e-15
    if not nonzero.any():
        return 0.0
    return float(np.max(difference[nonzero] / np.abs(reference[nonzero])))


def half_degree_grid(grid: pd.DataFrame, projector: Transformer) -> pd.DataFrame:
    """Split each baseline 1-degree cell into four matching 0.5-degree cells."""

    records: list[dict[str, object]] = []
    for parent_index, row in enumerate(grid.itertuples(index=False)):
        lon_mid = (float(row.lon_lo) + float(row.lon_hi)) / 2.0
        lat_mid = (float(row.lat_lo) + float(row.lat_hi)) / 2.0
        bounds = (
            (float(row.lon_lo), lon_mid, float(row.lat_lo), lat_mid),
            (lon_mid, float(row.lon_hi), float(row.lat_lo), lat_mid),
            (float(row.lon_lo), lon_mid, lat_mid, float(row.lat_hi)),
            (lon_mid, float(row.lon_hi), lat_mid, float(row.lat_hi)),
        )
        for child_index, (lon_lo, lon_hi, lat_lo, lat_hi) in enumerate(bounds):
            geometry = box(lon_lo, lat_lo, lon_hi, lat_hi)
            records.append(
                {
                    "parent_index": parent_index,
                    "parent_grid_id": int(row.grid_id),
                    "child_index": child_index,
                    "geometry_area": project_geometry(geometry, projector),
                }
            )
    result = pd.DataFrame(records)
    if len(result) != EXPECTED_CELL_COUNT * 4:
        raise AssertionError("0.5-degree verification grid must contain 528 cells")
    return result


def verify_half_degree_aggregation(
    grid: pd.DataFrame,
    zones: pd.DataFrame,
    direct_fractions: np.ndarray,
    zone_means: np.ndarray,
    zone_sds: np.ndarray,
    projector: Transformer,
) -> dict[str, float | int | str]:
    """Aggregate child fractions first, retaining same-zone covariance for SD."""

    children = half_degree_grid(grid, projector)
    _, child_fractions = calculate_overlaps(
        list(children["geometry_area"]), zones
    )
    aggregated_fractions = np.zeros_like(direct_fractions)
    np.add.at(
        aggregated_fractions,
        children["parent_index"].to_numpy(int),
        child_fractions,
    )
    direct_mean, direct_sd = source_moments(
        direct_fractions, zone_means, zone_sds
    )
    aggregated_mean, aggregated_sd = source_moments(
        aggregated_fractions, zone_means, zone_sds
    )
    mean_difference = np.abs(aggregated_mean - direct_mean)
    sd_difference = np.abs(aggregated_sd - direct_sd)
    fraction_difference = np.abs(aggregated_fractions - direct_fractions)
    if float(mean_difference.max()) > 1e-11 or float(sd_difference.max()) > 1e-11:
        raise AssertionError("0.5-degree aggregation does not reproduce 1-degree results")
    return {
        "child_grid_cells": int(len(children)),
        "aggregation_method_for_sd": (
            "sum source-zone fractions over four children first; then compute "
            "sqrt(sum_z((f_parent,z*sigma_z)^2))"
        ),
        "maximum_fraction_absolute_difference": float(fraction_difference.max()),
        "mean_maximum_absolute_difference": float(mean_difference.max()),
        "mean_maximum_relative_difference": maximum_relative_difference(
            aggregated_mean, direct_mean
        ),
        "sd_maximum_absolute_difference": float(sd_difference.max()),
        "sd_maximum_relative_difference": maximum_relative_difference(
            aggregated_sd, direct_sd
        ),
    }


def align_with_gp_summary(grid: pd.DataFrame) -> dict[str, object]:
    """Verify one-to-one row alignment with the GP grid summary."""

    if not GP_CELL_SUMMARY.is_file():
        raise FileNotFoundError(GP_CELL_SUMMARY)
    gp = pd.read_csv(GP_CELL_SUMMARY).sort_values("grid_id").reset_index(drop=True)
    columns = [
        "grid_id",
        "lon_lo",
        "lon_hi",
        "lat_lo",
        "lat_hi",
        "grid_lon",
        "grid_lat",
    ]
    if len(gp) != len(grid):
        raise AssertionError("GP cell summary row count differs from Hybrid grid")
    numeric_error = 0.0
    for column in columns:
        left = grid[column].to_numpy(float)
        right = gp[column].to_numpy(float)
        numeric_error = max(numeric_error, float(np.max(np.abs(left - right))))
    if numeric_error > 1e-12:
        raise AssertionError("Hybrid source-grid rows do not align with GP summary")
    return {
        "gp_cell_summary": str(GP_CELL_SUMMARY),
        "rows": int(len(gp)),
        "one_to_one_grid_id_and_geometry_alignment": True,
        "maximum_numeric_alignment_error": numeric_error,
    }


def main() -> int:
    required = (
        SOURCE_RATES,
        SOURCE_GEOMETRY,
        GRID_CELLS,
        GP_CELL_SUMMARY,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    rates = read_mw2_rates(SOURCE_RATES)
    raw_zones = read_source_zones(SOURCE_GEOMETRY)
    grid = read_grid(GRID_CELLS)
    alignment = align_with_gp_summary(grid)
    projector = Transformer.from_crs(MAP_CRS, AREA_CRS, always_xy=True)
    grid = project_grid(grid, projector)
    common_support = unary_union(list(grid["geometry_area"]))
    zones = prepare_zones(raw_zones, rates, projector, common_support)

    overlap_m2, fractions = calculate_overlaps(
        list(grid["geometry_area"]), zones
    )
    zone_names = zones["zone"].astype(str).tolist()
    zone_means = zones["source_mean_rate"].to_numpy(float)
    zone_sds = zones["source_sd_rate"].to_numpy(float)
    cell_means, cell_sds = source_moments(fractions, zone_means, zone_sds)

    zone_area_m2 = zones["A_z_km2"].to_numpy(float) * 1_000_000.0
    area_closure_m2 = overlap_m2.sum(axis=0) - zone_area_m2
    maximum_area_closure_m2 = float(np.max(np.abs(area_closure_m2)))
    maximum_area_closure_relative = float(
        np.max(np.abs(area_closure_m2) / zone_area_m2)
    )
    if maximum_area_closure_relative > AREA_CLOSURE_REL_TOLERANCE:
        raise AssertionError("Source-zone area closure failed")

    source_total = float(zone_means.sum())
    cell_total = float(cell_means.sum())
    conservation_error = cell_total - source_total
    if abs(conservation_error) > CONSERVATION_ABS_TOLERANCE:
        raise AssertionError("Source mean activity rate is not conserved")

    projected_cells = list(grid["geometry_area"])
    source_region = unary_union(list(zones["geometry_area"]))
    covered_area_km2 = np.asarray(
        [float(cell.intersection(source_region).area / 1_000_000.0) for cell in projected_cells]
    )
    positive = np.asarray(
        [
            overlap_m2[index]
            > max(float(projected_cells[index].area) * 1e-12, 1e-3)
            for index in range(len(projected_cells))
        ],
        dtype=bool,
    )
    contributor_count = positive.sum(axis=1)
    dominant_indices = np.argmax(overlap_m2, axis=1)
    dominant_zone = [
        zone_names[index] if contributor_count[row] > 0 else ""
        for row, index in enumerate(dominant_indices)
    ]

    # For genuinely single-zone cells, verify mean and SD scale by the same f_iz.
    strict_positive = fractions > 1e-14
    strict_single_indices = np.where(strict_positive.sum(axis=1) == 1)[0]
    single_mean_errors: list[float] = []
    single_sd_errors: list[float] = []
    single_mean_ratio_errors: list[float] = []
    single_sd_ratio_errors: list[float] = []
    zero_sd_single_cells = 0
    for row in strict_single_indices:
        zone_index = int(np.flatnonzero(strict_positive[row])[0])
        fraction = float(fractions[row, zone_index])
        expected_mean = fraction * zone_means[zone_index]
        expected_sd = fraction * zone_sds[zone_index]
        single_mean_errors.append(abs(cell_means[row] - expected_mean))
        single_sd_errors.append(abs(cell_sds[row] - expected_sd))
        single_mean_ratio_errors.append(
            abs(cell_means[row] / zone_means[zone_index] - fraction)
        )
        if zone_sds[zone_index] > 0:
            single_sd_ratio_errors.append(
                abs(cell_sds[row] / zone_sds[zone_index] - fraction)
            )
        else:
            zero_sd_single_cells += 1
    maximum_single_error = max(
        single_mean_errors + single_sd_errors + single_mean_ratio_errors + single_sd_ratio_errors,
        default=0.0,
    )
    if maximum_single_error > 1e-12:
        raise AssertionError("Single-zone mean/SD fraction identity failed")

    half_degree = verify_half_degree_aggregation(
        grid, zones, fractions, zone_means, zone_sds, projector
    )

    output_columns = [
        "grid_id",
        "lon_lo",
        "lon_hi",
        "lat_lo",
        "lat_hi",
        "grid_lon",
        "grid_lat",
    ]
    cells = grid.loc[:, output_columns].copy()
    cells["source_activity_rate_mean"] = cell_means
    cells["source_activity_rate_sd"] = cell_sds
    cells["source_model_covered_area_km2"] = covered_area_km2
    cells["n_contributing_zones"] = contributor_count.astype(int)
    cells["dominant_zone"] = dominant_zone
    cells["mixed_source_zone_boundary_cell"] = contributor_count > 1

    fraction_table = pd.DataFrame(fractions, columns=zone_names)
    fraction_table.insert(0, "grid_id", grid["grid_id"].astype(int).to_numpy())

    zones = zones.copy()
    zones["sum_A_iz_km2"] = overlap_m2.sum(axis=0) / 1_000_000.0
    zones["area_closure_error_km2"] = area_closure_m2 / 1_000_000.0
    zones["fraction_sum"] = fractions.sum(axis=0)
    zones["redistributed_mean_rate"] = zone_means * zones["fraction_sum"].to_numpy(float)
    zone_audit_columns = [
        "zone",
        "source_mean_rate",
        "source_sd_rate",
        "A_z_km2",
        "full_source_zone_area_km2",
        "sum_A_iz_km2",
        "area_closure_error_km2",
        "fraction_sum",
        "redistributed_mean_rate",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_csv(CELL_OUTPUT, index=False)
    fraction_table.to_csv(FRACTION_OUTPUT, index=False)
    zones.loc[:, zone_audit_columns].to_csv(ZONE_AUDIT_OUTPUT, index=False)

    first_ten_columns = [
        "grid_id",
        "grid_lon",
        "grid_lat",
        "source_activity_rate_mean",
        "source_activity_rate_sd",
        "dominant_zone",
        "n_contributing_zones",
    ]
    first_ten = cells.loc[:, first_ten_columns].head(10)
    top_ten_sd = cells.nlargest(10, "source_activity_rate_sd").loc[
        :, first_ten_columns
    ]
    verification = {
        "method": {
            "preserved_operations": [
                "BGS latitude/longitude reversal before Shapely construction",
                "EPSG:4326 geometry",
                "0.05-degree densification before area projection",
                "EPSG:3035 physical-area calculation",
                "union of the same 132 projected GP cells as common support",
                "A_z equals zone z intersected with that common support",
                "A_iz equals projected grid-cell/zone intersection area",
            ],
            "mean_formula": "mu_i=sum_z(f_iz*mu_z)",
            "sd_formula": "sigma_i=sqrt(sum_z((f_iz*sigma_z)^2))",
            "cross_zone_covariance": "assumed zero because none is available",
            "normal_distribution_generated": False,
        },
        "inputs": {
            "source_rates": str(SOURCE_RATES),
            "source_rates_sha256": sha256_file(SOURCE_RATES),
            "source_geometry": str(SOURCE_GEOMETRY),
            "source_geometry_sha256": sha256_file(SOURCE_GEOMETRY),
            "gp_grid": str(GRID_CELLS),
            "gp_grid_sha256": sha256_file(GRID_CELLS),
            "source_threshold_mw": 2.0,
            "source_mean_column": "mean_rate_per_year",
            "source_sd_column": "sd_rate_per_year",
            "old_approach_D_numerical_rates_used": False,
            "old_section_5_source_grid_numerical_values_used": False,
        },
        "alignment": alignment,
        "verification": {
            "number_of_source_zones": int(len(zones)),
            "number_of_gp_cells": int(len(cells)),
            "source_zone_mean_total_before_redistribution": source_total,
            "redistributed_cell_mean_total": cell_total,
            "conservation_absolute_error": abs(conservation_error),
            "conservation_relative_error": abs(conservation_error) / source_total,
            "minimum_source_cell_mean": float(cell_means.min()),
            "maximum_source_cell_mean": float(cell_means.max()),
            "minimum_source_cell_sd": float(cell_sds.min()),
            "maximum_source_cell_sd": float(cell_sds.max()),
            "cells_receiving_multiple_source_zones": int(
                np.count_nonzero(contributor_count > 1)
            ),
            "cells_receiving_no_source_zone": int(
                np.count_nonzero(contributor_count == 0)
            ),
            "maximum_source_zone_area_closure_error_m2": maximum_area_closure_m2,
            "maximum_source_zone_area_closure_error_km2": (
                maximum_area_closure_m2 / 1_000_000.0
            ),
            "maximum_source_zone_area_closure_relative_error": (
                maximum_area_closure_relative
            ),
            "single_zone_cells_checked": int(len(strict_single_indices)),
            "single_zone_zero_sd_cells_checked_by_absolute_identity": (
                zero_sd_single_cells
            ),
            "single_zone_maximum_mean_absolute_identity_error": max(
                single_mean_errors, default=0.0
            ),
            "single_zone_maximum_sd_absolute_identity_error": max(
                single_sd_errors, default=0.0
            ),
            "single_zone_maximum_mean_fraction_ratio_error": max(
                single_mean_ratio_errors, default=0.0
            ),
            "single_zone_maximum_sd_fraction_ratio_error": max(
                single_sd_ratio_errors, default=0.0
            ),
        },
        "half_degree_consistency": half_degree,
        "first_10_grid_cells": first_ten.to_dict(orient="records"),
        "top_10_cells_by_source_sd": top_ten_sd.to_dict(orient="records"),
        "outputs": {
            "source_grid_table": str(CELL_OUTPUT),
            "source_grid_table_sha256": sha256_file(CELL_OUTPUT),
            "fraction_matrix": str(FRACTION_OUTPUT),
            "fraction_matrix_sha256": sha256_file(FRACTION_OUTPUT),
            "zone_audit": str(ZONE_AUDIT_OUTPUT),
            "zone_audit_sha256": sha256_file(ZONE_AUDIT_OUTPUT),
        },
    }
    write_json(VERIFICATION_OUTPUT, verification)

    print(json.dumps(json_ready(verification["verification"]), indent=2))
    print(json.dumps(json_ready(half_degree), indent=2))
    print("\nFirst 10 grid cells:")
    print(first_ten.to_string(index=False))
    print("\nTop 10 cells by source SD:")
    print(top_ten_sd.to_string(index=False))
    print(f"\nWrote {CELL_OUTPUT}")
    print(f"Wrote {FRACTION_OUTPUT}")
    print(f"Wrote {ZONE_AUDIT_OUTPUT}")
    print(f"Wrote {VERIFICATION_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
