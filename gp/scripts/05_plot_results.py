"""Generate the baseline input, posterior, and uncertainty figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from _common import BASELINE_ROOT, load_config


def grid_figure(
    grid: pd.DataFrame,
    values: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    output: Path,
    events: pd.DataFrame | None = None,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 8.0), constrained_layout=True)
    patches = [
        Rectangle(
            (row.lon_lo, row.lat_lo),
            row.lon_hi - row.lon_lo,
            row.lat_hi - row.lat_lo,
        )
        for row in grid.itertuples(index=False)
    ]
    collection = PatchCollection(
        patches, cmap=cmap, edgecolor="white", linewidth=0.45
    )
    collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)
    colorbar = fig.colorbar(collection, ax=ax, shrink=0.82)
    colorbar.set_label(colorbar_label)
    if events is not None:
        ax.scatter(
            events["lon"],
            events["lat"],
            s=5,
            c="black",
            alpha=0.45,
            linewidths=0,
            label=f"Retained earthquakes (n={len(events)})",
        )
        ax.legend(loc="lower left", frameon=True, fontsize=8)
    ax.set_xlim(grid["lon_lo"].min(), grid["lon_hi"].max())
    ax.set_ylim(grid["lat_lo"].min(), grid["lat_hi"].max())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(title)
    fig.savefig(output, dpi=250)
    plt.close(fig)


def baseline_activity_and_uncertainty_figure(
    grid: pd.DataFrame,
    *,
    png_output: Path,
    pdf_output: Path,
) -> None:
    """Plot baseline GP mean activity rate and posterior SD in Hybrid style."""

    mean_values = grid["activity_rate_per_year_mean"].to_numpy(float)
    sd_values = grid["activity_rate_per_year_sd"].to_numpy(float)
    if len(grid) != 132 or grid["grid_id"].nunique() != 132:
        raise AssertionError("Expected 132 unique baseline GP grid cells")
    if (
        not np.isfinite(mean_values).all()
        or not np.isfinite(sd_values).all()
        or (mean_values < 0.0).any()
        or (sd_values < 0.0).any()
    ):
        raise AssertionError("Baseline GP means and SDs must be finite/nonnegative")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.4),
        constrained_layout=True,
    )
    specifications = [
        (
            axes[0],
            mean_values,
            "viridis",
            "Baseline GP mean activity rate",
            "Mean activity rate\n(events yr$^{-1}$ cell$^{-1}$)",
            "(a)",
        ),
        (
            axes[1],
            sd_values,
            "magma",
            "Baseline GP uncertainty: cell SD",
            "Activity-rate SD\n(events yr$^{-1}$ cell$^{-1}$)",
            "(b)",
        ),
    ]

    lon_min = float(grid["lon_lo"].min())
    lon_max = float(grid["lon_hi"].max())
    lat_min = float(grid["lat_lo"].min())
    lat_max = float(grid["lat_hi"].max())
    mean_latitude = 0.5 * (lat_min + lat_max)

    for ax, values, cmap_name, title, colorbar_label, panel_label in specifications:
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=0.0, vmax=float(np.max(values)))
        patches = [
            Rectangle(
                (float(row.lon_lo), float(row.lat_lo)),
                float(row.lon_hi - row.lon_lo),
                float(row.lat_hi - row.lat_lo),
            )
            for row in grid.itertuples(index=False)
        ]
        collection = PatchCollection(
            patches,
            cmap=cmap,
            norm=norm,
            edgecolor="#858585",
            linewidth=0.40,
        )
        collection.set_array(values)
        ax.add_collection(collection)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_latitude)))
        ax.set_xticks(
            np.arange(np.ceil(lon_min / 2.0) * 2.0, lon_max + 0.1, 2.0)
        )
        ax.set_yticks(
            np.arange(np.ceil(lat_min / 2.0) * 2.0, lat_max + 0.1, 2.0)
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.text(
            -0.18,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontweight="semibold",
            ha="left",
            va="bottom",
        )
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=0.050,
            pad=0.038,
        )
        colorbar.set_label(colorbar_label, fontsize=8.6)

    fig.savefig(png_output, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_config(args.config)
    grid = pd.read_csv(BASELINE_ROOT / "output" / "tables" / "cell_posterior_summary.csv")
    events = pd.read_csv(BASELINE_ROOT / "input" / "gp_earthquakes.csv")
    figure_dir = BASELINE_ROOT / "output" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        (
            "figure_1_input_counts_and_earthquakes.png",
            grid["count"].to_numpy(float),
            "Input earthquakes and 1-degree grid counts",
            "Earthquake count",
            events,
            "YlOrRd",
        ),
        (
            "figure_2_posterior_mean_gp_effect.png",
            grid["gp_effect_mean"].to_numpy(float),
            "Posterior mean GP spatial effect",
            "Posterior mean f",
            None,
            "coolwarm",
        ),
        (
            "figure_3_posterior_mean_spatial_probability.png",
            grid["spatial_probability_mean"].to_numpy(float),
            "Posterior mean spatial probability",
            "Posterior mean probability",
            None,
            "viridis",
        ),
        (
            "figure_4_posterior_mean_activity_rate.png",
            grid["activity_rate_per_year_mean"].to_numpy(float),
            "Final posterior mean annual activity rate",
            "Mw >= 2 events per year",
            None,
            "magma",
        ),
        (
            "figure_5_activity_rate_95pct_width.png",
            grid["activity_rate_95pct_width"].to_numpy(float),
            "Width of the 95% credible interval for annual activity rate",
            "97.5th minus 2.5th percentile",
            None,
            "cividis",
        ),
    ]
    for filename, values, title, label, event_data, cmap in specifications:
        output = figure_dir / filename
        grid_figure(
            grid,
            values,
            title=title,
            colorbar_label=label,
            output=output,
            events=event_data,
            cmap=cmap,
        )
        print(output, flush=True)

    combined_png = figure_dir / "figure_4_baseline_gp_activity_and_uncertainty.png"
    combined_pdf = figure_dir / "figure_4_baseline_gp_activity_and_uncertainty.pdf"
    baseline_activity_and_uncertainty_figure(
        grid,
        png_output=combined_png,
        pdf_output=combined_pdf,
    )
    print(combined_png, flush=True)
    print(combined_pdf, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
