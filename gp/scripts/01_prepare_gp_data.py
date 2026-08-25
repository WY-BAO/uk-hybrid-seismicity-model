"""Prepare the L5-complete catalogue, grid counts, and multinomial GP input."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import (
    BASELINE_ROOT,
    ensure_directories,
    load_config,
    resolve_project_path,
    sha256_file,
    write_json,
)


def fixed_domain_edges(lower: float, upper: float, step: float) -> np.ndarray:
    """Use full-width cells and clip only a possible final boundary cell."""
    edges = lower + np.arange(math.floor((upper - lower) / step) + 1) * step
    edges = edges[edges < upper - 1e-10]
    return np.append(edges, upper)


def lon_lat_to_xy_km(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    earth_radius_km: float,
    projection_lon: float,
    projection_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude and latitude to local Cartesian coordinates."""
    lat0_rad = math.radians(projection_lat)
    x_km = earth_radius_km * math.cos(lat0_rad) * np.radians(lon - projection_lon)
    y_km = earth_radius_km * np.radians(lat - projection_lat)
    return x_km, y_km


def spherical_cell_area_km2(
    lon_lo: float,
    lon_hi: float,
    lat_lo: float,
    lat_hi: float,
    earth_radius_km: float,
) -> float:
    return float(
        earth_radius_km**2
        * abs(math.radians(lon_hi - lon_lo))
        * abs(math.sin(math.radians(lat_hi)) - math.sin(math.radians(lat_lo)))
    )


def load_l5_catalogue(
    catalogue: pd.DataFrame,
    l5_input: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate and return the exact ordered event set used to build L5."""
    grid = config["grid"]
    required = ("year", "year_fraction", "lat", "lon", "ml", "sigma_ml")
    missing = [name for name in required if name not in catalogue.columns]
    if missing:
        raise KeyError(f"L5 catalogue is missing required columns: {missing}")
    events = catalogue.copy()
    events.insert(0, "l5_catalogue_row", np.arange(1, len(events) + 1, dtype=int))
    for name in required:
        events[name] = pd.to_numeric(events[name], errors="coerce")
    if not np.isfinite(events[list(required)].to_numpy(float)).all():
        raise RuntimeError("L5 catalogue contains non-finite required values")
    if not events["lat"].between(
        float(grid["latitude_min"]), float(grid["latitude_max"]), inclusive="both"
    ).all() or not events["lon"].between(
        float(grid["longitude_min"]), float(grid["longitude_max"]), inclusive="both"
    ).all():
        raise RuntimeError("An L5 catalogue event lies outside the configured GP grid")

    tolerance = float(config["validation"]["catalogue_numeric_tolerance"])
    l5_n = int(l5_input["N"])
    l5_ml = np.asarray(l5_input["ml_reported"], dtype=float)
    l5_sigma = np.asarray(l5_input["sigma_ml"], dtype=float)
    if l5_n != len(events) or len(l5_ml) != len(events) or len(l5_sigma) != len(events):
        raise RuntimeError("L5 catalogue length does not match the saved L5 Stan input")
    if not np.allclose(
        events["ml"].to_numpy(float), l5_ml, rtol=0.0, atol=tolerance
    ) or not np.allclose(
        events["sigma_ml"].to_numpy(float), l5_sigma, rtol=0.0, atol=tolerance
    ):
        raise RuntimeError(
            "L5 catalogue row order does not match ml_reported/sigma_ml in its Stan input"
        )
    return events


def build_grid(events: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_config = config["grid"]
    coordinate_config = config["coordinates"]
    degree = float(grid_config["degree"])
    lon_edges = fixed_domain_edges(
        float(grid_config["longitude_min"]),
        float(grid_config["longitude_max"]),
        degree,
    )
    lat_edges = fixed_domain_edges(
        float(grid_config["latitude_min"]),
        float(grid_config["latitude_max"]),
        degree,
    )
    rows: list[dict[str, float | int]] = []
    grid_id = 1
    for lat_index, (lat_lo, lat_hi) in enumerate(zip(lat_edges[:-1], lat_edges[1:])):
        for lon_index, (lon_lo, lon_hi) in enumerate(zip(lon_edges[:-1], lon_edges[1:])):
            rows.append(
                {
                    "grid_id": grid_id,
                    "lon_index": lon_index,
                    "lat_index": lat_index,
                    "lon_lo": float(lon_lo),
                    "lon_hi": float(lon_hi),
                    "lat_lo": float(lat_lo),
                    "lat_hi": float(lat_hi),
                    "grid_lon": float((lon_lo + lon_hi) / 2.0),
                    "grid_lat": float((lat_lo + lat_hi) / 2.0),
                    "grid_area_km2": spherical_cell_area_km2(
                        lon_lo,
                        lon_hi,
                        lat_lo,
                        lat_hi,
                        float(coordinate_config["earth_radius_km"]),
                    ),
                }
            )
            grid_id += 1
    cells = pd.DataFrame(rows)
    cells["grid_x_km"], cells["grid_y_km"] = lon_lat_to_xy_km(
        cells["grid_lon"].to_numpy(float),
        cells["grid_lat"].to_numpy(float),
        earth_radius_km=float(coordinate_config["earth_radius_km"]),
        projection_lon=float(coordinate_config["projection_longitude"]),
        projection_lat=float(coordinate_config["projection_latitude"]),
    )

    n_lon = len(lon_edges) - 1
    n_lat = len(lat_edges) - 1
    lon_index = np.searchsorted(lon_edges, events["lon"].to_numpy(float), side="right") - 1
    lat_index = np.searchsorted(lat_edges, events["lat"].to_numpy(float), side="right") - 1
    lon_index = np.clip(lon_index, 0, n_lon - 1)
    lat_index = np.clip(lat_index, 0, n_lat - 1)
    assigned = events.copy()
    assigned["event_grid_id"] = lat_index * n_lon + lon_index + 1
    counts = np.bincount(
        assigned["event_grid_id"].to_numpy(int), minlength=len(cells) + 1
    )[1:]
    cells["count"] = counts.astype(int)
    return assigned, cells


def stan_data(cells: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    coordinates = config["coordinates"]
    model = config["model"]
    priors = model["priors"]
    scale = float(coordinates["stan_unit_km"])
    return {
        "N": int(len(cells)),
        "D": 2,
        "x": [
            [float(row.grid_x_km / scale), float(row.grid_y_km / scale)]
            for row in cells.itertuples(index=False)
        ],
        "count": cells["count"].astype(int).tolist(),
        "cell_area_km2": cells["grid_area_km2"].astype(float).tolist(),
        "distance_scale_km": scale,
        "jitter": float(model["jitter"]),
        "alpha_prior_sd": float(priors["alpha_sd"]),
        "rho_prior_logmean": float(priors["rho_logmean"]),
        "rho_prior_logsd": float(priors["rho_logsd"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    ensure_directories()
    catalogue_path = resolve_project_path(config["paths"]["l5_catalogue"])
    l5_input_path = resolve_project_path(config["paths"]["l5_stan_input"])
    if not catalogue_path.is_file():
        raise FileNotFoundError(catalogue_path)
    if not l5_input_path.is_file():
        raise FileNotFoundError(l5_input_path)
    catalogue = pd.read_csv(catalogue_path, low_memory=False)
    l5_input = json.loads(l5_input_path.read_text(encoding="utf-8"))
    events = load_l5_catalogue(catalogue, l5_input, config)
    events, cells = build_grid(events, config)

    expected = config["expected"]
    if len(events) != int(expected["earthquakes"]):
        raise RuntimeError(
            f"Expected {expected['earthquakes']} GP earthquakes, found {len(events)}"
        )
    if len(cells) != int(expected["grid_cells"]):
        raise RuntimeError(f"Expected {expected['grid_cells']} cells, found {len(cells)}")
    if int(cells["count"].sum()) != len(events):
        raise RuntimeError("Grid counts do not sum to the selected earthquake count")
    if np.any(cells["grid_area_km2"].to_numpy(float) <= 0.0):
        raise RuntimeError("Every grid cell must have strictly positive physical area")

    event_file = BASELINE_ROOT / "input" / "gp_earthquakes.csv"
    grid_file = BASELINE_ROOT / "input" / "grid_cells.csv"
    stan_file = BASELINE_ROOT / "input" / "stan_data.json"
    events.to_csv(event_file, index=False)
    cells.to_csv(grid_file, index=False)
    write_json(stan_file, stan_data(cells, config))

    manifest = {
        "configuration": str(config_path),
        "catalogue": str(catalogue_path),
        "catalogue_sha256": sha256_file(catalogue_path),
        "l5_stan_input": str(l5_input_path),
        "l5_stan_input_sha256": sha256_file(l5_input_path),
        "selection": {
            "source": "retained L5 catalogue",
            "additional_gp_magnitude_filter": False,
            "additional_gp_year_filter": False,
            "fixed_gp_observation_window": False,
            "exactly_matches_l5_event_order_and_numeric_inputs": True,
        },
        "earthquakes": int(len(events)),
        "minimum_event_year": int(events["year"].min()),
        "maximum_event_year": int(events["year"].max()),
        "minimum_ml": float(events["ml"].min()),
        "maximum_ml": float(events["ml"].max()),
        "grid_cells": int(len(cells)),
        "occupied_cells": int((cells["count"] > 0).sum()),
        "zero_count_cells": int((cells["count"] == 0).sum()),
        "maximum_cell_count": int(cells["count"].max()),
        "sum_cell_counts": int(cells["count"].sum()),
        "coordinate_transformation": {
            "method": "local equirectangular",
            "projection_longitude": config["coordinates"]["projection_longitude"],
            "projection_latitude": config["coordinates"]["projection_latitude"],
            "stan_unit_km": config["coordinates"]["stan_unit_km"],
        },
        "outputs": {
            "earthquakes": str(event_file),
            "grid": str(grid_file),
            "stan_data": str(stan_file),
        },
    }
    write_json(BASELINE_ROOT / "input" / "preparation_manifest.json", manifest)

    print(f"L5 catalogue = {catalogue_path}")
    print(f"Number of GP earthquakes = {len(events)}")
    print(f"Event years = {int(events['year'].min())}-{int(events['year'].max())}")
    print(f"Magnitude range = ML {events['ml'].min():g}-{events['ml'].max():g}")
    print(f"Grid cells = {len(cells)}")
    print("Fixed GP observation window = none")
    print(
        "Coordinate scale = 1 Stan distance unit = "
        f"{config['coordinates']['stan_unit_km']:g} km"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
