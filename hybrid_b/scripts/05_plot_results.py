"""Plot the source baseline, Hybrid B activity rate, and GP correction maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from _common import HYBRID_B_ROOT, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def panel(ax, grid: pd.DataFrame, values: np.ndarray, title: str, label: str, cmap: str, norm=None):
    patches = [
        Rectangle((row.lon_lo, row.lat_lo), row.lon_hi - row.lon_lo, row.lat_hi - row.lat_lo)
        for row in grid.itertuples(index=False)
    ]
    collection = PatchCollection(
        patches, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.4
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)
    ax.set_xlim(grid["lon_lo"].min(), grid["lon_hi"].max())
    ax.set_ylim(grid["lat_lo"].min(), grid["lat_hi"].max())
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(grid["grid_lat"].mean()))))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    plt.colorbar(collection, ax=ax, shrink=0.78, label=label)


def main() -> int:
    args = parse_args()
    load_config(args.config)
    table = pd.read_csv(
        HYBRID_B_ROOT / "output" / "tables" / "hybrid_b_cell_summary.csv"
    ).sort_values("grid_id")
    correction = table["f_correction_mean"].to_numpy(float)
    correction_limit = float(max(abs(correction.min()), abs(correction.max()), 1e-12))
    specifications = (
        (
            table["p_source"].to_numpy(float),
            "Source baseline spatial probability",
            "p_source",
            "viridis",
            None,
            "map_source_baseline_probability.png",
        ),
        (
            table["hybrid_b_activity_rate_per_year_mean"].to_numpy(float),
            "Hybrid B posterior mean activity rate",
            "Mw >= 2 events yr$^{-1}$ cell$^{-1}$",
            "magma",
            None,
            "map_hybrid_b_mean_activity_rate.png",
        ),
        (
            correction,
            "Posterior mean GP correction to source baseline",
            "mean f_correction",
            "coolwarm",
            TwoSlopeNorm(vmin=-correction_limit, vcenter=0.0, vmax=correction_limit),
            "map_hybrid_b_gp_correction.png",
        ),
    )
    figure_dir = HYBRID_B_ROOT / "output" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for values, title, label, cmap, norm, filename in specifications:
        fig, ax = plt.subplots(figsize=(7.5, 8.0), constrained_layout=True)
        panel(ax, table, values, title, label, cmap, norm)
        output = figure_dir / filename
        fig.savefig(output, dpi=300)
        plt.close(fig)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
