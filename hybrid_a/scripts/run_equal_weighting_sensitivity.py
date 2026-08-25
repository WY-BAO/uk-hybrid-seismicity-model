"""Build the equal-weight Hybrid reference and its weight-sensitivity figures.

The calculation uses the GP posterior and the source model on the common grid:

* the 4,000 x 132 multinomial-GP annual activity-rate posterior; and
* the branch-wise truncated-GR Mw>=2 source model converted to the same 132
  one-degree cells.

For GP weight w, the posterior-mean sensitivity curve is

    mean(lambda_H_i(w)) = w * mean(lambda_GP_i)
                          + (1 - w) * mean(lambda_source_i).

The equal-weight reference uses w=0.5.  Its cell SD is propagated by
quadrature under independence of the GP and source-model uncertainties:

    sd(lambda_H_i(w)) = sqrt((w * sd_GP_i)^2
                             + ((1 - w) * sd_source_i)^2).

No source-model pseudo-draws are generated and no total-rate renormalisation
is applied.
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
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
GP_POSTERIOR = PROJECT_ROOT / "gp" / "output" / "posterior" / "baseline_posterior_draws.npz"
GP_GRID = PROJECT_ROOT / "gp" / "input" / "grid_cells.csv"
SOURCE_GRID = ROOT / "output" / "source_model_132_grid_mw2.csv"
OUT = ROOT / "output" / "equal_weighting"
FIGURES = OUT / "figures"

EQUAL_GP_WEIGHT = 0.5
WEIGHT_VALUES = np.linspace(0.0, 1.0, 101)
N_HIGHLIGHT = 10
ALIGNMENT_ATOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    grid = pd.read_csv(GP_GRID).sort_values("grid_id").reset_index(drop=True)
    source = (
        pd.read_csv(SOURCE_GRID).sort_values("grid_id").reset_index(drop=True)
    )

    with np.load(GP_POSTERIOR, allow_pickle=False) as posterior:
        activity = np.asarray(posterior["activity_rate_cell"], dtype=float)

    if activity.shape != (4000, 132):
        raise AssertionError(
            f"Expected current GP activity_rate_cell shape (4000, 132), got "
            f"{activity.shape}"
        )
    if len(grid) != 132 or grid["grid_id"].nunique() != 132:
        raise AssertionError("GP grid must contain 132 unique cells")
    if len(source) != 132 or source["grid_id"].nunique() != 132:
        raise AssertionError("Source grid must contain 132 unique cells")
    if not np.array_equal(
        grid["grid_id"].to_numpy(int), source["grid_id"].to_numpy(int)
    ):
        raise AssertionError("GP and source grids are not aligned by grid_id")

    coordinate_columns = [
        "lon_lo",
        "lon_hi",
        "lat_lo",
        "lat_hi",
        "grid_lon",
        "grid_lat",
    ]
    coordinate_error = np.max(
        np.abs(
            grid[coordinate_columns].to_numpy(float)
            - source[coordinate_columns].to_numpy(float)
        )
    )
    if coordinate_error > ALIGNMENT_ATOL:
        raise AssertionError(
            f"GP/source cell-coordinate mismatch: max error={coordinate_error}"
        )
    if not np.isfinite(activity).all() or (activity < 0.0).any():
        raise AssertionError("GP activity-rate draws must be finite and nonnegative")

    source_mean = source["source_activity_rate_mean"].to_numpy(float)
    source_sd = source["source_activity_rate_sd"].to_numpy(float)
    if (
        not np.isfinite(source_mean).all()
        or not np.isfinite(source_sd).all()
        or (source_mean < 0.0).any()
        or (source_sd < 0.0).any()
    ):
        raise AssertionError("Source-model means and SDs must be finite/nonnegative")

    frame = grid.copy()
    frame["source_activity_rate_mean"] = source_mean
    frame["source_activity_rate_sd"] = source_sd
    frame["source_model_covered_area_km2"] = source[
        "source_model_covered_area_km2"
    ].to_numpy(float)
    frame["source_model_present"] = (
        frame["source_model_covered_area_km2"] > 1e-6
    )
    frame["gp_activity_rate_mean"] = np.mean(activity, axis=0)
    frame["gp_activity_rate_sd"] = np.std(activity, axis=0, ddof=1)
    return frame, activity


def build_equal_weight_summary(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    source_mean = result["source_activity_rate_mean"].to_numpy(float)
    gp_mean = result["gp_activity_rate_mean"].to_numpy(float)
    source_sd = result["source_activity_rate_sd"].to_numpy(float)
    gp_sd = result["gp_activity_rate_sd"].to_numpy(float)

    result["source_weight"] = 1.0 - EQUAL_GP_WEIGHT
    result["gp_weight"] = EQUAL_GP_WEIGHT
    result["hybrid_activity_rate_mean"] = (
        EQUAL_GP_WEIGHT * gp_mean + (1.0 - EQUAL_GP_WEIGHT) * source_mean
    )
    result["hybrid_activity_rate_sd_independence"] = np.sqrt(
        (EQUAL_GP_WEIGHT * gp_sd) ** 2
        + ((1.0 - EQUAL_GP_WEIGHT) * source_sd) ** 2
    )
    result["gp_minus_source_activity_rate"] = gp_mean - source_mean
    result["absolute_weight_sensitivity"] = np.abs(gp_mean - source_mean)
    result["sensitivity_rank"] = (
        result["absolute_weight_sensitivity"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return result


def sensitivity_curves(summary: pd.DataFrame) -> np.ndarray:
    source_mean = summary["source_activity_rate_mean"].to_numpy(float)
    gp_mean = summary["gp_activity_rate_mean"].to_numpy(float)
    return (
        WEIGHT_VALUES[:, None] * gp_mean[None, :]
        + (1.0 - WEIGHT_VALUES[:, None]) * source_mean[None, :]
    )


def style_curve_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    curves: np.ndarray,
    add_panel_label: bool,
) -> None:
    for cell_index in range(curves.shape[1]):
        ax.plot(
            WEIGHT_VALUES,
            curves[:, cell_index],
            color="#b8b8b8",
            linewidth=0.65,
            alpha=0.30,
            zorder=1,
        )

    highlighted = summary.nsmallest(N_HIGHLIGHT, "sensitivity_rank")
    colors = plt.get_cmap("tab10")(np.arange(N_HIGHLIGHT))
    for colour, (_, row) in zip(colors, highlighted.iterrows()):
        cell_index = int(row.name)
        ax.plot(
            WEIGHT_VALUES,
            curves[:, cell_index],
            color=colour,
            linewidth=1.7,
            label=f"Cell {int(row['grid_id'])}",
            zorder=3,
        )

    ax.axvline(
        EQUAL_GP_WEIGHT,
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        zorder=2,
    )
    y_max = float(np.max(curves))
    ax.text(
        EQUAL_GP_WEIGHT,
        y_max * 1.018,
        "$w=0.5$",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max * 1.10)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("GP weight, $w$")
    ax.set_ylabel(
        "Posterior-mean hybrid activity rate\n"
        "(events yr$^{-1}$ cell$^{-1}$)"
    )
    ax.set_title("Sensitivity of grid-cell activity rates to model weighting")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.75)
    ax.legend(
        title="Largest source–GP differences",
        loc="upper center",
        bbox_to_anchor=(0.66, 0.99),
        frameon=False,
        fontsize=7.3,
        title_fontsize=7.5,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.6,
    )
    if add_panel_label:
        ax.text(
            -0.14,
            1.04,
            "(a)",
            transform=ax.transAxes,
            fontweight="semibold",
            ha="left",
            va="bottom",
        )


def style_map_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    norm: Normalize,
    cmap,
    add_panel_label: bool,
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
        edgecolor="#787878",
        linewidth=0.38,
    )
    collection.set_array(summary["absolute_weight_sensitivity"].to_numpy(float))
    ax.add_collection(collection)

    lon_min = float(summary["lon_lo"].min())
    lon_max = float(summary["lon_hi"].max())
    lat_min = float(summary["lat_lo"].min())
    lat_max = float(summary["lat_hi"].max())
    mean_latitude = 0.5 * (lat_min + lat_max)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_latitude)))
    ax.set_xticks(np.arange(np.ceil(lon_min / 2.0) * 2.0, lon_max + 0.1, 2.0))
    ax.set_yticks(np.arange(np.ceil(lat_min / 2.0) * 2.0, lat_max + 0.1, 2.0))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sensitivity to model weighting")
    if add_panel_label:
        ax.text(
            -0.18,
            1.04,
            "(b)",
            transform=ax.transAxes,
            fontweight="semibold",
            ha="left",
            va="bottom",
        )
    return collection


def add_sensitivity_colorbar(fig: plt.Figure, ax: plt.Axes, cmap, norm) -> None:
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.048,
        pad=0.035,
    )
    colorbar.set_label(
        "$|\\bar{\\lambda}^{GP}_i-\\bar{\\lambda}^{source}_i|$\n"
        "(events yr$^{-1}$ cell$^{-1}$)",
        fontsize=8.5,
    )


def style_equal_weight_quantity_map(
    ax: plt.Axes,
    summary: pd.DataFrame,
    values: np.ndarray,
    cmap,
    norm: Normalize,
    title: str,
    panel_label: str,
) -> None:
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
        edgecolor="#858585",
        linewidth=0.40,
    )
    collection.set_array(values)
    ax.add_collection(collection)

    lon_min = float(summary["lon_lo"].min())
    lon_max = float(summary["lon_hi"].max())
    lat_min = float(summary["lat_lo"].min())
    lat_max = float(summary["lat_hi"].max())
    mean_latitude = 0.5 * (lat_min + lat_max)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_latitude)))
    ax.set_xticks(np.arange(np.ceil(lon_min / 2.0) * 2.0, lon_max + 0.1, 2.0))
    ax.set_yticks(np.arange(np.ceil(lat_min / 2.0) * 2.0, lat_max + 0.1, 2.0))
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


def add_quantity_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    cmap,
    norm: Normalize,
    label: str,
) -> None:
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.050,
        pad=0.038,
    )
    colorbar.set_label(label, fontsize=8.6)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(summary: pd.DataFrame, curves: np.ndarray) -> None:
    cmap = plt.get_cmap("viridis")
    sensitivity = summary["absolute_weight_sensitivity"].to_numpy(float)
    norm = Normalize(vmin=0.0, vmax=float(np.max(sensitivity)))

    fig, ax = plt.subplots(figsize=(8.0, 6.1), constrained_layout=True)
    style_curve_axis(ax, summary, curves, add_panel_label=False)
    save_figure(fig, "figure_equal_weighting_cell_sensitivity")

    fig, ax = plt.subplots(figsize=(6.5, 7.2), constrained_layout=True)
    style_map_axis(ax, summary, norm, cmap, add_panel_label=False)
    add_sensitivity_colorbar(fig, ax, cmap, norm)
    save_figure(fig, "figure_equal_weighting_spatial_sensitivity")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.3),
        gridspec_kw={"width_ratios": [1.22, 1.0]},
        constrained_layout=True,
    )
    style_curve_axis(axes[0], summary, curves, add_panel_label=True)
    style_map_axis(axes[1], summary, norm, cmap, add_panel_label=True)
    add_sensitivity_colorbar(fig, axes[1], cmap, norm)
    save_figure(fig, "figure_equal_weighting_sensitivity_combined")

    hybrid_mean = summary["hybrid_activity_rate_mean"].to_numpy(float)
    hybrid_sd = summary["hybrid_activity_rate_sd_independence"].to_numpy(float)
    mean_cmap = plt.get_cmap("viridis")
    sd_cmap = plt.get_cmap("magma")
    mean_norm = Normalize(vmin=0.0, vmax=float(np.max(hybrid_mean)))
    sd_norm = Normalize(vmin=0.0, vmax=float(np.max(hybrid_sd)))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.4),
        constrained_layout=True,
    )
    style_equal_weight_quantity_map(
        axes[0],
        summary,
        hybrid_mean,
        mean_cmap,
        mean_norm,
        "Hybrid mean activity rate ($w=0.5$)",
        "(a)",
    )
    add_quantity_colorbar(
        fig,
        axes[0],
        mean_cmap,
        mean_norm,
        "Mean activity rate\n(events yr$^{-1}$ cell$^{-1}$)",
    )
    style_equal_weight_quantity_map(
        axes[1],
        summary,
        hybrid_sd,
        sd_cmap,
        sd_norm,
        "Propagated uncertainty: cell SD ($w=0.5$)",
        "(b)",
    )
    add_quantity_colorbar(
        fig,
        axes[1],
        sd_cmap,
        sd_norm,
        "Activity-rate SD\n(events yr$^{-1}$ cell$^{-1}$)",
    )
    save_figure(fig, "figure_equal_weighting_activity_and_uncertainty")


def make_verification(
    summary: pd.DataFrame,
    activity: np.ndarray,
    curves: np.ndarray,
) -> dict:
    source_mean = summary["source_activity_rate_mean"].to_numpy(float)
    gp_mean = summary["gp_activity_rate_mean"].to_numpy(float)
    equal_mean = summary["hybrid_activity_rate_mean"].to_numpy(float)
    sensitivity = summary["absolute_weight_sensitivity"].to_numpy(float)
    equal_index = int(np.argmin(np.abs(WEIGHT_VALUES - EQUAL_GP_WEIGHT)))

    endpoint_source_error = float(np.max(np.abs(curves[0] - source_mean)))
    endpoint_gp_error = float(np.max(np.abs(curves[-1] - gp_mean)))
    midpoint_error = float(np.max(np.abs(curves[equal_index] - equal_mean)))
    slope_error = float(
        np.max(np.abs(np.abs(curves[-1] - curves[0]) - sensitivity))
    )
    all_checks = {
        "gp_activity_shape_4000_by_132": activity.shape == (4000, 132),
        "grid_has_132_cells": len(summary) == 132,
        "source_endpoint_w0_exact": endpoint_source_error <= 1e-14,
        "gp_endpoint_w1_exact": endpoint_gp_error <= 1e-14,
        "equal_weight_midpoint_exact": midpoint_error <= 1e-14,
        "spatial_sensitivity_equals_endpoint_difference": slope_error <= 1e-14,
        "equal_hybrid_means_nonnegative": bool((equal_mean >= 0.0).all()),
        "equal_hybrid_sds_finite_nonnegative": bool(
            np.isfinite(summary["hybrid_activity_rate_sd_independence"]).all()
            and (summary["hybrid_activity_rate_sd_independence"] >= 0.0).all()
        ),
    }
    return {
        "model_definition": {
            "gp_weight_symbol": "w",
            "hybrid_mean_formula": "w*GP_mean + (1-w)*source_mean",
            "equal_gp_weight": EQUAL_GP_WEIGHT,
            "equal_source_weight": 1.0 - EQUAL_GP_WEIGHT,
            "hybrid_sd_formula": "sqrt((w*GP_sd)^2 + ((1-w)*source_sd)^2)",
            "uncertainty_assumption": "independent GP and source-model uncertainties",
            "total_rate_renormalisation": False,
            "source_pseudo_draws_generated": False,
        },
        "baseline_inputs": {
            "gp_posterior": str(GP_POSTERIOR.relative_to(ROOT)),
            "gp_posterior_sha256": sha256(GP_POSTERIOR),
            "source_grid": str(SOURCE_GRID.relative_to(ROOT)),
            "source_grid_sha256": sha256(SOURCE_GRID),
            "gp_grid": str(GP_GRID.relative_to(ROOT)),
            "gp_grid_sha256": sha256(GP_GRID),
        },
        "dimensions": {
            "gp_draws": int(activity.shape[0]),
            "grid_cells": int(activity.shape[1]),
            "tested_gp_weight_values": int(len(WEIGHT_VALUES)),
            "source_absent_cells": int((~summary["source_model_present"]).sum()),
        },
        "totals_events_per_year": {
            "gp_posterior_mean": float(np.sum(gp_mean)),
            "source_model_mean": float(np.sum(source_mean)),
            "equal_weight_hybrid_mean": float(np.sum(equal_mean)),
            "expected_equal_weight_hybrid_mean": float(
                EQUAL_GP_WEIGHT * np.sum(gp_mean)
                + (1.0 - EQUAL_GP_WEIGHT) * np.sum(source_mean)
            ),
        },
        "sensitivity": {
            "maximum_absolute_change_per_unit_weight": float(np.max(sensitivity)),
            "mean_absolute_change_per_unit_weight": float(np.mean(sensitivity)),
            "top_10_grid_ids": [
                int(value)
                for value in summary.nsmallest(N_HIGHLIGHT, "sensitivity_rank")[
                    "grid_id"
                ]
            ],
        },
        "numerical_errors": {
            "source_endpoint_max_abs": endpoint_source_error,
            "gp_endpoint_max_abs": endpoint_gp_error,
            "equal_midpoint_max_abs": midpoint_error,
            "sensitivity_slope_max_abs": slope_error,
        },
        "checks": {**all_checks, "all_passed": bool(all(all_checks.values()))},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
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

    frame, activity = load_inputs()
    summary = build_equal_weight_summary(frame)
    curves = sensitivity_curves(summary)
    verification = make_verification(summary, activity, curves)
    if not verification["checks"]["all_passed"]:
        raise AssertionError(json.dumps(verification["checks"], indent=2))

    summary.to_csv(
        OUT / "equal_weighting_w0p5_cell_summary.csv",
        index=False,
        float_format="%.12g",
    )
    top10_columns = [
        "sensitivity_rank",
        "grid_id",
        "grid_lon",
        "grid_lat",
        "source_activity_rate_mean",
        "gp_activity_rate_mean",
        "gp_minus_source_activity_rate",
        "absolute_weight_sensitivity",
        "hybrid_activity_rate_mean",
        "source_model_present",
    ]
    summary.nsmallest(N_HIGHLIGHT, "sensitivity_rank")[top10_columns].to_csv(
        OUT / "equal_weighting_top10_sensitivity_cells.csv",
        index=False,
        float_format="%.12g",
    )
    (OUT / "equal_weighting_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figures(summary, curves)

    print(json.dumps(verification, indent=2))
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
