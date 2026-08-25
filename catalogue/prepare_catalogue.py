"""Prepare the natural-earthquake catalogue and apply the regional filter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "catalogue" / "data" / "Data.csv"
DEFAULT_OUTPUT = ROOT / "catalogue" / "output"

CATALOGUE_END_YEAR = 2023.0
ML_THRESHOLD = 2.0
LATITUDE_RANGE = (49.0, 61.0)
LONGITUDE_RANGE = (-8.0, 3.0)

NATIONAL_MODERN = {2.0: 1990, 2.5: 1979}
REGIONAL_COMPLETENESS = {
    "UK": {
        3.0: 1970,
        3.5: 1850,
        4.0: 1750,
        4.5: 1700,
        5.0: 1650,
        5.5: 1650,
        6.5: 1000,
    },
    "SE": {
        3.0: 1970,
        3.5: 1850,
        4.0: 1750,
        4.5: 1700,
        5.0: 1650,
        5.5: 1300,
        6.5: 1000,
    },
    "Dogger": {
        3.0: 1970,
        3.5: 1970,
        4.0: 1850,
        4.5: 1750,
        5.0: 1650,
        5.5: 1650,
        6.5: 1000,
    },
    "Viking": {
        3.0: 1970,
        3.5: 1970,
        4.0: 1970,
        4.5: 1900,
        5.0: 1900,
        5.5: 1900,
        6.5: 1700,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-events", type=int, default=1013)
    return parser.parse_args()


def default_sigma_ml(year: np.ndarray) -> np.ndarray:
    return np.where(
        year < 1900,
        0.50,
        np.where(year < 1970, 0.40, np.where(year < 1990, 0.25, 0.15)),
    )


def catalogue_region(latitude: float, longitude: float) -> str:
    if longitude >= 0.5 and latitude >= 56.0:
        return "Viking"
    if longitude >= 0.5 and 53.0 <= latitude < 56.0:
        return "Dogger"
    if latitude <= 52.5 and -0.5 <= longitude <= 2.5:
        return "SE"
    return "UK"


def complete_from_year(magnitude: float, region: str) -> int:
    if magnitude < 3.0:
        return NATIONAL_MODERN[2.5] if magnitude >= 2.5 else NATIONAL_MODERN[2.0]
    table = REGIONAL_COMPLETENESS[region]
    applicable_threshold = max(
        threshold for threshold in table if magnitude >= threshold
    )
    return table[applicable_threshold]


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.DataFrame(
        {
            "year": pd.to_numeric(raw["Year"], errors="coerce"),
            "month": pd.to_numeric(raw["Month"], errors="coerce").fillna(6.0),
            "day": pd.to_numeric(raw["Day"], errors="coerce").fillna(15.0),
            "lat": pd.to_numeric(raw["Latitude"], errors="coerce"),
            "lon": pd.to_numeric(raw["Longitude"], errors="coerce"),
            "ml": pd.to_numeric(raw["ML"], errors="coerce"),
            "sigma_ml_reported": pd.to_numeric(raw["Error (ML)"], errors="coerce"),
            "magnitude_type": raw["Type M"].fillna("").astype(str).str.strip().str.upper(),
            "event_type": raw["Type of event"].fillna("").astype(str).str.strip().str.upper(),
            "location": raw["Location"].fillna("").astype(str),
        }
    )
    events["year_fraction"] = (
        events["year"]
        + (events["month"] - 1.0) / 12.0
        + (events["day"] - 1.0) / 365.25
    )

    missing_sigma = (
        events["sigma_ml_reported"].isna()
        | (events["sigma_ml_reported"] >= 999.0)
        | (events["sigma_ml_reported"] <= 0.0)
    )
    events["sigma_ml"] = events["sigma_ml_reported"]
    events.loc[missing_sigma, "sigma_ml"] = default_sigma_ml(
        events.loc[missing_sigma, "year"].to_numpy(float)
    )
    events["sigma_ml_source"] = np.where(
        missing_sigma, "default_by_year", "reported"
    )

    lat_min, lat_max = LATITUDE_RANGE
    lon_min, lon_max = LONGITUDE_RANGE
    natural = events.loc[
        (events["event_type"] == "")
        & (events["magnitude_type"] == "ML")
        & events[["year", "ml", "lat", "lon"]].notna().all(axis=1)
        & events["lat"].between(lat_min, lat_max)
        & events["lon"].between(lon_min, lon_max)
    ].copy()

    candidates = natural.loc[natural["ml"] >= ML_THRESHOLD].copy()
    candidates["complete_from_year"] = [
        complete_from_year(magnitude, "UK") for magnitude in candidates["ml"]
    ]
    candidates["region"] = [
        catalogue_region(lat, lon)
        for lat, lon in zip(candidates["lat"], candidates["lon"])
    ]
    candidates["complete_from_regional"] = [
        complete_from_year(magnitude, region)
        for magnitude, region in zip(candidates["ml"], candidates["region"])
    ]
    filtered = candidates.loc[
        (candidates["year_fraction"] >= candidates["complete_from_regional"])
        & (candidates["year_fraction"] < CATALOGUE_END_YEAR)
    ].copy()
    return natural, filtered


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input)
    natural, filtered = prepare(raw)
    if args.expected_events >= 0 and len(filtered) != args.expected_events:
        raise RuntimeError(
            f"Expected {args.expected_events} retained events, found {len(filtered)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    natural.to_csv(args.output_dir / "natural_catalogue.csv", index=False)
    filtered.to_csv(args.output_dir / "filtered_catalogue_regional.csv", index=False)
    summary = {
        "raw_rows": int(len(raw)),
        "natural_ml_events": int(len(natural)),
        "filtered_events": int(len(filtered)),
        "region_counts": {
            str(key): int(value)
            for key, value in filtered["region"].value_counts().sort_index().items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
