"""Regrid fitted multinomial GP draws, compare with 1 degree, and plot results."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from _grid_common import (
    SENSITIVITY_ROOT,
    build_overlap_transfer,
    case_directory,
    ensure_output_directories,
    load_configs,
    load_preparation_module,
    write_json,
)


def build_evaluation_grid(baseline_config: dict, sensitivity_config: dict) -> pd.DataFrame:
    preparation = load_preparation_module()
    config = json.loads(json.dumps(baseline_config))
    config["grid"]["degree"] = float(sensitivity_config["evaluation_grid_degree"])
    empty = pd.DataFrame({"lon": np.array([], dtype=float), "lat": np.array([], dtype=float)})
    _, grid = preparation.build_grid(empty, config)
    if len(grid) != 13200:
        raise RuntimeError(f"Expected 13200 common 0.1-degree cells, found {len(grid)}")
    return grid.drop(columns="count")


def add_summary_columns(frame: pd.DataFrame, prefix: str, draws: np.ndarray) -> None:
    quantiles = np.quantile(draws, [0.025, 0.5, 0.975], axis=0)
    frame[f"{prefix}_mean"] = np.mean(draws, axis=0)
    frame[f"{prefix}_median"] = quantiles[1]
    frame[f"{prefix}_q025"] = quantiles[0]
    frame[f"{prefix}_q975"] = quantiles[2]
    frame[f"{prefix}_95pct_width"] = quantiles[2] - quantiles[0]


def run_summary_row(summary: dict, regrid: dict) -> dict:
    return {
        "degree": summary["degree"],
        "grid_cells": summary["grid_cells"],
        "zero_count_cells": summary["zero_count_cells"],
        "zero_count_proportion": summary["zero_count_proportion"],
        "maximum_cell_count": summary["maximum_cell_count"],
        **{f"alpha_{key}": value for key, value in summary["alpha"].items()},
        **{f"rho_km_{key}": value for key, value in summary["rho_km"].items()},
        "maximum_rhat": summary["mcmc"]["maximum_rhat"],
        "minimum_bulk_ess": summary["mcmc"]["minimum_bulk_ess"],
        "minimum_tail_ess": summary["mcmc"]["minimum_tail_ess"],
        "divergences": summary["mcmc"]["divergences"],
        "treedepth_hits": summary["mcmc"]["treedepth_hits"],
        "mcmc_all_passed": summary["mcmc"]["all_passed"],
        "model_probability_max_sum_error": summary["validation"][
            "maximum_model_probability_sum_error"
        ],
        "common_probability_max_sum_error": regrid[
            "maximum_common_probability_sum_error"
        ],
        "model_rate_max_sum_error": summary["validation"][
            "maximum_model_grid_rate_sum_error"
        ],
        "common_rate_max_sum_error": regrid["maximum_common_rate_sum_error"],
        "all_validation_checks_passed": bool(
            summary["validation"]["all_1013_events_assigned_once"]
            and summary["validation"]["cell_counts_sum_to_1013"]
            and regrid["every_common_probability_draw_sums_to_one"]
            and regrid["every_common_rate_draw_sums_to_l5"]
        ),
        "runtime_minutes": summary["sampling_walltime_seconds"] / 60.0,
    }


def style_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.set_xticks(np.arange(0.5, 2.01, 0.1))
    ax.tick_params(axis="x", rotation=45)


def save_line_figures(run_table: pd.DataFrame, comparison: pd.DataFrame, figure_dir: Path) -> None:
    degrees = run_table["degree"].to_numpy(float)
    fig, ax1 = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax2 = ax1.twinx()
    first = ax1.plot(degrees, run_table["grid_cells"], "o-", label="Grid cells")
    second = ax2.plot(
        degrees,
        run_table["zero_count_proportion"],
        "s--",
        color="tab:orange",
        label="Zero-count proportion",
    )
    ax1.set_xlabel("Grid resolution (degrees)")
    ax1.set_ylabel("Number of grid cells")
    ax2.set_ylabel("Proportion of zero-count cells")
    ax1.set_xticks(degrees)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.25, linewidth=0.7)
    ax1.legend(first + second, [item.get_label() for item in first + second], loc="upper right")
    ax1.set_title("Figure A. Grid size and zero-count cells")
    fig.savefig(figure_dir / "figure_A_grid_cells_zero_proportion.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    median = run_table["rho_km_median"].to_numpy(float)
    lower = median - run_table["rho_km_q025"].to_numpy(float)
    upper = run_table["rho_km_q975"].to_numpy(float) - median
    ax.errorbar(degrees, median, yerr=[lower, upper], fmt="o-", capsize=3)
    style_axis(ax, "Grid resolution (degrees)", "Posterior rho (km)")
    ax.set_title("Figure B. Posterior length scale by grid resolution")
    fig.savefig(figure_dir / "figure_B_rho_by_resolution.png", dpi=240)
    plt.close(fig)

    for filename, column, ylabel, title in (
        (
            "figure_C_rmse_vs_reference.png",
            "rmse_vs_1deg_events_per_year_per_eval_cell",
            "RMSE (events/year per 0.1-degree cell)",
            "Figure C. Difference from the 1-degree reference",
        ),
        (
            "figure_D_pearson_vs_reference.png",
            "pearson_vs_1deg",
            "Pearson correlation",
            "Figure D. Correlation with the 1-degree reference",
        ),
        (
            "figure_E_mean_credible_interval_width.png",
            "mean_95pct_credible_interval_width",
            "Mean 95% CrI width (events/year per 0.1-degree cell)",
            "Figure E. Common-grid posterior uncertainty",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
        ax.plot(comparison["degree"], comparison[column], "o-")
        style_axis(ax, "Grid resolution (degrees)", ylabel)
        ax.set_title(title)
        fig.savefig(figure_dir / filename, dpi=240)
        plt.close(fig)


def common_map_array(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["lat_index", "lon_index"])
    n_lat = int(ordered["lat_index"].max()) + 1
    n_lon = int(ordered["lon_index"].max()) + 1
    values = ordered["activity_rate_per_year_mean"].to_numpy(float).reshape(n_lat, n_lon)
    lon_edges = np.sort(np.unique(np.r_[ordered["lon_lo"], ordered["lon_hi"]]))
    lat_edges = np.sort(np.unique(np.r_[ordered["lat_lo"], ordered["lat_hi"]]))
    return values, lon_edges, lat_edges


def save_common_maps(common_frames: dict[float, pd.DataFrame], map_dir: Path) -> None:
    selected = [0.5, 1.0, 2.0]
    arrays = {degree: common_map_array(common_frames[degree]) for degree in selected}
    all_values = np.concatenate([arrays[degree][0].ravel() for degree in selected])
    vmin = float(np.min(all_values[all_values > 0]))
    vmax = float(np.max(all_values))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for degree in selected:
        values, lon_edges, lat_edges = arrays[degree]
        fig, ax = plt.subplots(figsize=(6.8, 7.2), constrained_layout=True)
        image = ax.pcolormesh(lon_edges, lat_edges, values, shading="flat", norm=norm, cmap="viridis")
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.set_aspect(1.0 / np.cos(np.deg2rad(55.0)))
        ax.set_title(f"{degree:.1f}-degree GP on common 0.1-degree support")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Posterior mean annual activity rate per 0.1-degree cell")
        fig.savefig(
            map_dir / f"common_map_degree_{str(degree).replace('.', 'p')}.png", dpi=240
        )
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2), constrained_layout=True, sharex=True, sharey=True)
    image = None
    for ax, degree in zip(axes, selected):
        values, lon_edges, lat_edges = arrays[degree]
        image = ax.pcolormesh(lon_edges, lat_edges, values, shading="flat", norm=norm, cmap="viridis")
        ax.set_xlabel("Longitude (degrees)")
        ax.set_title(f"{degree:.1f} degree")
        ax.set_aspect(1.0 / np.cos(np.deg2rad(55.0)))
    axes[0].set_ylabel("Latitude (degrees)")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86)
    colorbar.set_label("Posterior mean annual activity rate per 0.1-degree cell")
    fig.suptitle("Common-support posterior mean activity-rate maps")
    fig.savefig(map_dir / "common_maps_0p5_1p0_2p0_same_scale.png", dpi=240)
    plt.close(fig)
    write_json(
        map_dir / "map_colour_scale.json",
        {"scale": "logarithmic", "vmin": vmin, "vmax": vmax, "common_support": "0.1 degree"},
    )


def main() -> int:
    ensure_output_directories()
    baseline_config, sensitivity_config = load_configs()
    resolutions = [float(value) for value in sensitivity_config["resolutions_degrees"]]
    evaluation_grid = build_evaluation_grid(baseline_config, sensitivity_config)
    evaluation_grid.to_csv(SENSITIVITY_ROOT / "output" / "evaluation_grid_0p1.csv", index=False)
    with np.load(
        SENSITIVITY_ROOT / "output" / "common_pairing_indices.npz", allow_pickle=False
    ) as paired:
        gp_index = paired["gp_draw_index"]
        l5_paired = paired["L5_total_activity"]
    tolerance = float(sensitivity_config["conservation_tolerance"])
    overlap_tolerance = float(sensitivity_config["overlap_tolerance"])
    earth_radius = float(baseline_config["coordinates"]["earth_radius_km"])
    run_rows: list[dict] = []
    common_frames: dict[float, pd.DataFrame] = {}
    regrid_manifests: dict[float, dict] = {}
    for degree in resolutions:
        case_dir = case_directory(degree)
        run_summary = json.loads((case_dir / "run_summary.json").read_text(encoding="utf-8"))
        model_grid = pd.read_csv(case_dir / "input" / "grid_cells.csv")
        with np.load(case_dir / "posterior" / "gp_posterior_draws.npz", allow_pickle=False) as loaded:
            model_probability = loaded["spatial_probability"][gp_index]
        transfer, overlap_validation = build_overlap_transfer(
            model_grid, evaluation_grid, earth_radius, overlap_tolerance
        )
        common_probability = np.asarray((transfer.T @ model_probability.T).T)
        model_probability_error = float(
            np.max(np.abs(model_probability.sum(axis=1) - 1.0))
        )
        common_probability_error = float(
            np.max(np.abs(common_probability.sum(axis=1) - 1.0))
        )
        common_rate = common_probability * l5_paired[:, None]
        common_rate_error = float(
            np.max(np.abs(common_rate.sum(axis=1) - l5_paired))
        )
        if max(model_probability_error, common_probability_error, common_rate_error) >= tolerance:
            raise RuntimeError(
                f"degree={degree}: regridding conservation failed: "
                f"{model_probability_error}, {common_probability_error}, {common_rate_error}"
            )
        common_summary = evaluation_grid.copy()
        add_summary_columns(common_summary, "spatial_probability", common_probability)
        add_summary_columns(common_summary, "activity_rate_per_year", common_rate)
        common_file = case_dir / "tables" / "common_grid_posterior_summary.csv"
        common_summary.to_csv(common_file, index=False)
        regrid_manifest = {
            "degree": degree,
            "model_grid_cells": int(len(model_grid)),
            "evaluation_grid_degree": sensitivity_config["evaluation_grid_degree"],
            "evaluation_grid_cells": int(len(evaluation_grid)),
            "density_transfer_rule": "p_i / A_i multiplied by spherical overlap area",
            "overlap_matrix": overlap_validation,
            "maximum_model_probability_sum_error_after_pairing": model_probability_error,
            "every_common_probability_draw_sums_to_one": True,
            "maximum_common_probability_sum_error": common_probability_error,
            "every_common_rate_draw_sums_to_l5": True,
            "maximum_common_rate_sum_error": common_rate_error,
            "paired_draws": int(len(l5_paired)),
            "common_draws_retained": False,
            "common_draw_processing": (
                "all draw-level common probabilities and rates were materialised for "
                "validation and quantiles; the saved artifact is the cell posterior summary"
            ),
            "output": str(common_file),
        }
        write_json(case_dir / "regrid_manifest.json", regrid_manifest)
        regrid_manifests[degree] = regrid_manifest
        run_rows.append(run_summary_row(run_summary, regrid_manifest))
        common_frames[degree] = common_summary
        print(
            f"REGRIDDED degree={degree:.1f} common_p_error={common_probability_error:.3e} "
            f"common_rate_error={common_rate_error:.3e}",
            flush=True,
        )
        del model_probability, common_probability, common_rate, transfer
        gc.collect()
    run_table = pd.DataFrame(run_rows).sort_values("degree").reset_index(drop=True)
    run_table.to_csv(
        SENSITIVITY_ROOT / "output" / "tables" / "grid_resolution_run_summary.csv",
        index=False,
    )
    reference_degree = float(sensitivity_config["reference_grid_degree"])
    reference = common_frames[reference_degree]["activity_rate_per_year_mean"].to_numpy(float)
    comparison_rows = []
    for degree in resolutions:
        frame = common_frames[degree]
        values = frame["activity_rate_per_year_mean"].to_numpy(float)
        difference = values - reference
        width = frame["activity_rate_per_year_95pct_width"].to_numpy(float)
        reference_width = common_frames[reference_degree][
            "activity_rate_per_year_95pct_width"
        ].to_numpy(float)
        comparison_rows.append(
            {
                "degree": degree,
                "reference_degree": reference_degree,
                "pearson_vs_1deg": float(np.corrcoef(values, reference)[0, 1]),
                "rmse_vs_1deg_events_per_year_per_eval_cell": float(
                    np.sqrt(np.mean(difference**2))
                ),
                "maximum_absolute_difference_events_per_year_per_eval_cell": float(
                    np.max(np.abs(difference))
                ),
                "mean_95pct_credible_interval_width": float(np.mean(width)),
                "reference_mean_95pct_credible_interval_width": float(
                    np.mean(reference_width)
                ),
                "difference_in_mean_95pct_width": float(
                    np.mean(width) - np.mean(reference_width)
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("degree").reset_index(drop=True)
    comparison.to_csv(
        SENSITIVITY_ROOT / "output" / "tables" / "common_grid_comparison_vs_1deg.csv",
        index=False,
    )
    figure_dir = SENSITIVITY_ROOT / "output" / "figures"
    map_dir = SENSITIVITY_ROOT / "output" / "maps"
    save_line_figures(run_table, comparison, figure_dir)
    save_common_maps(common_frames, map_dir)
    write_json(
        SENSITIVITY_ROOT / "output" / "regridding_summary.json",
        {
            "evaluation_grid_degree": sensitivity_config["evaluation_grid_degree"],
            "evaluation_grid_cells": int(len(evaluation_grid)),
            "reference_degree": reference_degree,
            "resolutions": resolutions,
            "all_regridding_checks_passed": bool(
                all(
                    item["every_common_probability_draw_sums_to_one"]
                    and item["every_common_rate_draw_sums_to_l5"]
                    for item in regrid_manifests.values()
                )
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
