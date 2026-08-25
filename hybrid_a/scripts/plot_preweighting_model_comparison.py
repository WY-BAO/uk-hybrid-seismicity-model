"""Compare source-model and GP cell-mean activity rates before weighting.

This diagnostic reads the two cell summary tables,
aligns their 132 one-degree cells, and compares activity-rate magnitude.  It
does not read or create Hybrid weights and does not use posterior SDs or
credible-interval widths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


SCRIPT = Path(__file__).resolve()
HYBRID_ROOT = SCRIPT.parents[1]
PROJECT_ROOT = HYBRID_ROOT.parent

GP_SUMMARY = (
    PROJECT_ROOT / "gp" / "output" / "tables" / "cell_posterior_summary.csv"
)
SOURCE_GRID = HYBRID_ROOT / "output" / "source_model_132_grid_mw2.csv"
OUT = HYBRID_ROOT / "output" / "pre_weighting_comparison"

COMPARISON_CSV = OUT / "source_gp_cell_mean_comparison.csv"
TOP10_CSV = OUT / "source_gp_top10_absolute_differences.csv"
VERIFICATION_JSON = OUT / "source_gp_cell_mean_comparison_verification.json"
FIGURE_PNG = OUT / "figure_section5_2_source_gp_cell_mean_comparison.png"
FIGURE_PDF = OUT / "figure_section5_2_source_gp_cell_mean_comparison.pdf"

N_CELLS = 132
N_ANNOTATE = 10
ALIGNMENT_ATOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_align() -> tuple[pd.DataFrame, float]:
    gp_columns = [
        "grid_id",
        "lon_lo",
        "lon_hi",
        "lat_lo",
        "lat_hi",
        "grid_lon",
        "grid_lat",
        "activity_rate_per_year_mean",
    ]
    source_columns = [
        "grid_id",
        "lon_lo",
        "lon_hi",
        "lat_lo",
        "lat_hi",
        "grid_lon",
        "grid_lat",
        "source_activity_rate_mean",
    ]
    gp = pd.read_csv(GP_SUMMARY, usecols=gp_columns).sort_values("grid_id")
    source = pd.read_csv(SOURCE_GRID, usecols=source_columns).sort_values("grid_id")
    gp = gp.reset_index(drop=True)
    source = source.reset_index(drop=True)

    if len(gp) != N_CELLS or gp["grid_id"].nunique() != N_CELLS:
        raise AssertionError("Current GP summary must contain 132 unique grid cells")
    if len(source) != N_CELLS or source["grid_id"].nunique() != N_CELLS:
        raise AssertionError("Current source grid must contain 132 unique grid cells")
    if not np.array_equal(
        gp["grid_id"].to_numpy(int), source["grid_id"].to_numpy(int)
    ):
        raise AssertionError("GP and source tables do not align by grid_id")

    coordinate_columns = [
        "lon_lo",
        "lon_hi",
        "lat_lo",
        "lat_hi",
        "grid_lon",
        "grid_lat",
    ]
    coordinate_error = float(
        np.max(
            np.abs(
                gp[coordinate_columns].to_numpy(float)
                - source[coordinate_columns].to_numpy(float)
            )
        )
    )
    if coordinate_error > ALIGNMENT_ATOL:
        raise AssertionError(
            "GP and source grid coordinates differ: "
            f"maximum absolute mismatch={coordinate_error}"
        )

    comparison = source[coordinate_columns + ["grid_id"]].copy()
    comparison = comparison[
        [
            "grid_id",
            "lon_lo",
            "lon_hi",
            "lat_lo",
            "lat_hi",
            "grid_lon",
            "grid_lat",
        ]
    ]
    comparison["source_activity_rate_mean"] = source[
        "source_activity_rate_mean"
    ].to_numpy(float)
    comparison["gp_activity_rate_mean"] = gp[
        "activity_rate_per_year_mean"
    ].to_numpy(float)
    comparison["difference_gp_minus_source"] = (
        comparison["gp_activity_rate_mean"]
        - comparison["source_activity_rate_mean"]
    )
    comparison["absolute_difference"] = np.abs(
        comparison["difference_gp_minus_source"]
    )
    comparison["absolute_difference_rank"] = (
        comparison["absolute_difference"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    numeric = comparison[
        [
            "source_activity_rate_mean",
            "gp_activity_rate_mean",
            "difference_gp_minus_source",
            "absolute_difference",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise AssertionError("Comparison values must all be finite")
    if (
        (comparison["source_activity_rate_mean"] < 0.0).any()
        or (comparison["gp_activity_rate_mean"] < 0.0).any()
    ):
        raise AssertionError("Both model activity-rate means must be nonnegative")
    return comparison, coordinate_error


def calculate_metrics(comparison: pd.DataFrame) -> dict[str, float]:
    source = comparison["source_activity_rate_mean"].to_numpy(float)
    gp = comparison["gp_activity_rate_mean"].to_numpy(float)
    difference = gp - source
    return {
        "pearson_correlation": float(np.corrcoef(source, gp)[0, 1]),
        "rmse_events_per_year_per_cell": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
        "mean_signed_difference_gp_minus_source": float(np.mean(difference)),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
    }


def plot_scatter(ax: plt.Axes, comparison: pd.DataFrame) -> None:
    source = comparison["source_activity_rate_mean"].to_numpy(float)
    gp = comparison["gp_activity_rate_mean"].to_numpy(float)

    limit = float(max(np.max(source), np.max(gp)) * 1.06)
    ax.plot(
        [0.0, limit],
        [0.0, limit],
        linestyle="--",
        color="#4f4f4f",
        linewidth=1.15,
        zorder=1,
    )
    ax.scatter(
        source,
        gp,
        s=27,
        facecolor="#347b98",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.78,
        zorder=2,
    )

    label_offsets = {
        22: (8, 5),
        7: (8, -10),
        110: (-25, 8),
        8: (8, -12),
        132: (8, -12),
        121: (10, 10),
        51: (-24, 11),
        79: (-24, -10),
        80: (8, 5),
        50: (-22, 5),
    }
    top10 = comparison.nsmallest(N_ANNOTATE, "absolute_difference_rank")
    for row in top10.itertuples(index=False):
        ax.annotate(
            str(int(row.grid_id)),
            xy=(row.source_activity_rate_mean, row.gp_activity_rate_mean),
            xytext=label_offsets[int(row.grid_id)],
            textcoords="offset points",
            fontsize=7.4,
            color="#303030",
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.45,
                "shrinkA": 1.5,
                "shrinkB": 2.5,
            },
            zorder=3,
        )

    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Source-model mean activity rate\n(events yr$^{-1}$ cell$^{-1}$)")
    ax.set_ylabel("GP posterior mean activity rate\n(events yr$^{-1}$ cell$^{-1}$)")
    ax.set_title("Cell-by-cell activity-rate comparison")
    ax.grid(True, color="#d1d1d1", linewidth=0.55, alpha=0.72)
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="#4f4f4f",
            linewidth=1.15,
            label="1:1 line",
        )
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    ax.text(
        -0.14,
        1.04,
        "(a)",
        transform=ax.transAxes,
        fontweight="semibold",
        ha="left",
        va="bottom",
    )


def plot_difference_map(
    fig: plt.Figure,
    ax: plt.Axes,
    comparison: pd.DataFrame,
) -> None:
    difference = comparison["difference_gp_minus_source"].to_numpy(float)
    absolute_limit = float(np.max(np.abs(difference)))
    norm = TwoSlopeNorm(vmin=-absolute_limit, vcenter=0.0, vmax=absolute_limit)
    cmap = plt.get_cmap("RdBu_r")

    rectangles = [
        Rectangle(
            (float(row.lon_lo), float(row.lat_lo)),
            float(row.lon_hi - row.lon_lo),
            float(row.lat_hi - row.lat_lo),
        )
        for row in comparison.itertuples(index=False)
    ]
    collection = PatchCollection(
        rectangles,
        cmap=cmap,
        norm=norm,
        edgecolor="#858585",
        linewidth=0.40,
    )
    collection.set_array(difference)
    ax.add_collection(collection)

    for row, value in zip(comparison.itertuples(index=False), difference):
        red, green, blue, _ = cmap(norm(float(value)))
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        text_colour = "white" if luminance < 0.46 else "#222222"
        ax.text(
            float(row.grid_lon),
            float(row.grid_lat),
            str(int(row.grid_id)),
            ha="center",
            va="center",
            fontsize=5.0,
            color=text_colour,
            zorder=3,
        )

    lon_min = float(comparison["lon_lo"].min())
    lon_max = float(comparison["lon_hi"].max())
    lat_min = float(comparison["lat_lo"].min())
    lat_max = float(comparison["lat_hi"].max())
    mean_latitude = 0.5 * (lat_min + lat_max)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_latitude)))
    ax.set_xticks(np.arange(np.ceil(lon_min / 2.0) * 2.0, lon_max + 0.1, 2.0))
    ax.set_yticks(np.arange(np.ceil(lat_min / 2.0) * 2.0, lat_max + 0.1, 2.0))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial difference: GP minus source")
    ax.text(
        -0.18,
        1.04,
        "(b)",
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
    colorbar.set_label(
        "$\\Delta_i=\\bar{\\lambda}^{GP}_i-\\bar{\\lambda}^{source}_i$\n"
        "(events yr$^{-1}$ cell$^{-1}$)",
        fontsize=8.6,
    )


def make_figure(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 6.25),
        gridspec_kw={"width_ratios": [1.08, 0.92]},
        constrained_layout=True,
    )
    plot_scatter(axes[0], comparison)
    plot_difference_map(fig, axes[1], comparison)
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


def make_verification(
    comparison: pd.DataFrame,
    coordinate_error: float,
    metrics: dict[str, float],
) -> dict:
    source = comparison["source_activity_rate_mean"].to_numpy(float)
    gp = comparison["gp_activity_rate_mean"].to_numpy(float)
    difference = comparison["difference_gp_minus_source"].to_numpy(float)
    absolute = comparison["absolute_difference"].to_numpy(float)
    expected_difference = gp - source

    checks = {
        "gp_summary_has_132_unique_grid_ids": bool(
            len(comparison) == N_CELLS
            and comparison["grid_id"].nunique() == N_CELLS
        ),
        "gp_source_grid_ids_aligned": bool(
            np.array_equal(
                comparison["grid_id"].to_numpy(int),
                np.arange(1, N_CELLS + 1),
            )
        ),
        "gp_source_coordinates_aligned": coordinate_error <= ALIGNMENT_ATOL,
        "difference_is_exactly_gp_minus_source": bool(
            np.max(np.abs(difference - expected_difference)) <= 1e-15
        ),
        "absolute_difference_is_exact": bool(
            np.max(np.abs(absolute - np.abs(expected_difference))) <= 1e-15
        ),
        "activity_rate_means_finite_nonnegative": bool(
            np.isfinite(source).all()
            and np.isfinite(gp).all()
            and (source >= 0.0).all()
            and (gp >= 0.0).all()
        ),
        "metrics_are_finite": bool(np.isfinite(list(metrics.values())).all()),
        "no_weighting_performed": True,
        "no_sd_or_credible_interval_used": True,
    }
    top10_columns = [
        "absolute_difference_rank",
        "grid_id",
        "grid_lon",
        "grid_lat",
        "source_activity_rate_mean",
        "gp_activity_rate_mean",
        "difference_gp_minus_source",
        "absolute_difference",
    ]
    top10 = comparison.nsmallest(N_ANNOTATE, "absolute_difference_rank")
    return {
        "analysis": "Source-versus-GP cell-mean comparison before weighting",
        "formula": "difference_gp_minus_source = GP_mean - source_mean",
        "inputs": {
            "gp_cell_summary": str(GP_SUMMARY.relative_to(PROJECT_ROOT)),
            "gp_cell_summary_sha256": sha256(GP_SUMMARY),
            "gp_mean_column": "activity_rate_per_year_mean",
            "source_grid": str(SOURCE_GRID.relative_to(PROJECT_ROOT)),
            "source_grid_sha256": sha256(SOURCE_GRID),
            "source_mean_column": "source_activity_rate_mean",
        },
        "dimensions": {"grid_cells": int(len(comparison))},
        "metrics": metrics,
        "model_mean_totals_events_per_year": {
            "source": float(np.sum(source)),
            "gp": float(np.sum(gp)),
            "gp_minus_source": float(np.sum(difference)),
        },
        "maximum_coordinate_alignment_error_degrees": coordinate_error,
        "top_10_cells_by_absolute_difference": top10[top10_columns].to_dict(
            orient="records"
        ),
        "checks": {**checks, "all_passed": bool(all(checks.values()))},
        "outputs": {
            "figure_png": str(FIGURE_PNG.relative_to(PROJECT_ROOT)),
            "figure_pdf": str(FIGURE_PDF.relative_to(PROJECT_ROOT)),
            "comparison_csv": str(COMPARISON_CSV.relative_to(PROJECT_ROOT)),
            "top10_csv": str(TOP10_CSV.relative_to(PROJECT_ROOT)),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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

    comparison, coordinate_error = load_and_align()
    metrics = calculate_metrics(comparison)
    verification = make_verification(comparison, coordinate_error, metrics)
    if not verification["checks"]["all_passed"]:
        raise AssertionError(json.dumps(verification["checks"], indent=2))

    comparison.to_csv(COMPARISON_CSV, index=False, float_format="%.17g")
    comparison.nsmallest(N_ANNOTATE, "absolute_difference_rank").to_csv(
        TOP10_CSV,
        index=False,
        float_format="%.17g",
    )
    VERIFICATION_JSON.write_text(
        json.dumps(verification, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figure(comparison)

    print(json.dumps(verification, indent=2))
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
