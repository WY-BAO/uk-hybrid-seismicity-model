"""Summarise all fitted kernels and create the requested comparison figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from _kernel_common import (
    BASELINE_GRID,
    BASELINE_STAN_DATA,
    CONFIG_FILE,
    SENSITIVITY_ROOT,
    STAN_FILE,
    case_root,
    configs,
    correlation_values,
    ensure_directories,
    kernel_order,
    load_json,
    posterior_summary,
    sha256_file,
    validate_baseline,
    write_json,
)


def plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_png_pdf(figure: plt.Figure, basename: str) -> None:
    figure.savefig(SENSITIVITY_ROOT / "figures" / f"{basename}.png")
    figure.savefig(SENSITIVITY_ROOT / "figures" / f"{basename}.pdf")
    plt.close(figure)


def load_case(kernel: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    root = case_root(kernel)
    manifest_file = root / "run_manifest.json"
    gp_file = root / "posterior" / "gp_posterior_draws.npz"
    activity_file = root / "posterior" / "activity_posterior_draws.npz"
    for path in (manifest_file, gp_file, activity_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed {kernel} artifact: {path}")
    manifest = load_json(manifest_file)
    if not manifest.get("complete"):
        raise RuntimeError(f"{kernel} manifest is not complete")
    if manifest.get("new_stan_sha256") != sha256_file(STAN_FILE):
        raise RuntimeError(f"{kernel} fit is stale relative to current Stan source")
    if manifest.get("baseline_stan_data_sha256") != sha256_file(BASELINE_STAN_DATA):
        raise RuntimeError(f"{kernel} fit is stale relative to current baseline data")
    with np.load(gp_file, allow_pickle=False) as loaded:
        gp = {name: loaded[name] for name in loaded.files}
    with np.load(activity_file, allow_pickle=False) as loaded:
        activity = {name: loaded[name] for name in loaded.files}
    return manifest, gp, activity


def parameter_row(kernel: str, manifest: dict[str, Any], gp: dict[str, np.ndarray]) -> dict[str, Any]:
    alpha = posterior_summary(gp["alpha"])
    rho = posterior_summary(gp["rho"])
    rho_km = posterior_summary(gp["rho_km"])
    diagnostic = manifest["diagnostics"]
    mcmc = pd.read_csv(case_root(kernel) / "tables" / "mcmc_summary.csv", index_col=0)
    max_rhat_variable = str(mcmc["R_hat"].idxmax())
    return {
        "kernel": kernel,
        "kernel_label": manifest["kernel_label"],
        "kernel_id": int(manifest["kernel_id"]),
        **{f"alpha_{name}": value for name, value in alpha.items()},
        **{f"rho_{name}": value for name, value in rho.items()},
        **{f"rho_km_{name}": value for name, value in rho_km.items()},
        "max_rhat": float(diagnostic["maximum_rhat"]),
        "max_rhat_variable": max_rhat_variable,
        "min_ess_bulk": float(diagnostic["minimum_bulk_ess"]),
        "min_ess_tail": float(diagnostic["minimum_tail_ess"]),
        "divergences": int(diagnostic["divergences"]),
        "treedepth_hits": int(diagnostic["treedepth_hits"]),
        "diagnostics_passed": bool(diagnostic["all_passed"]),
        "sampling_warning_count": int(len(manifest.get("sampling_warning_lines", []))),
        "runtime_seconds": float(manifest["runtime_seconds_wall"]),
        "runtime_minutes": float(manifest["runtime_minutes_wall"]),
    }


def correlation_curves(
    config: dict[str, Any], cases: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    curve = config["correlation_curve"]
    distances = np.arange(
        int(curve["minimum_distance_km"]),
        int(curve["maximum_distance_km"]) + int(curve["step_km"]),
        int(curve["step_km"]),
        dtype=float,
    )
    rows = []
    for kernel in config["kernel_order"]:
        manifest = cases[kernel]["manifest"]
        rho_km = np.asarray(cases[kernel]["gp"]["rho_km"], dtype=float)
        values = correlation_values(
            int(manifest["kernel_id"]),
            distances[None, :],
            rho_km[:, None],
        )
        mean = np.mean(values, axis=0)
        median = np.quantile(values, 0.5, axis=0)
        q025 = np.quantile(values, 0.025, axis=0)
        q975 = np.quantile(values, 0.975, axis=0)
        rows.extend(
            {
                "kernel": kernel,
                "kernel_label": manifest["kernel_label"],
                "distance_km": int(distance),
                "correlation_mean": float(mean[index]),
                "correlation_median": float(median[index]),
                "correlation_q025": float(q025[index]),
                "correlation_q975": float(q975[index]),
            }
            for index, distance in enumerate(distances)
        )
    return pd.DataFrame(rows)


def spatial_comparison(cases: dict[str, dict[str, Any]]) -> pd.DataFrame:
    eq = np.mean(cases["exp_quad"]["activity"]["activity_rate_cell"], axis=0)
    rows = []
    for kernel in kernel_order():
        values = np.mean(cases[kernel]["activity"]["activity_rate_cell"], axis=0)
        difference = values - eq
        if kernel == "exp_quad":
            pearson = 1.0
            rmse = 0.0
            mad = 0.0
            maximum = 0.0
        else:
            pearson = float(np.corrcoef(values, eq)[0, 1])
            rmse = float(np.sqrt(np.mean(np.square(difference))))
            mad = float(np.mean(np.abs(difference)))
            maximum = float(np.max(np.abs(difference)))
        rows.append(
            {
                "kernel": kernel,
                "kernel_label": cases[kernel]["manifest"]["kernel_label"],
                "pearson_vs_EQ": pearson,
                "rmse_vs_EQ": rmse,
                "mean_absolute_difference": mad,
                "maximum_absolute_difference": maximum,
            }
        )
    return pd.DataFrame(rows)


def conservation_table(cases: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    pairing_hashes = set()
    for kernel in kernel_order():
        manifest = cases[kernel]["manifest"]
        gp = cases[kernel]["gp"]
        activity = cases[kernel]["activity"]
        p_error = float(
            np.max(np.abs(gp["spatial_probability"].sum(axis=1) - 1.0))
        )
        paired_p_error = float(
            np.max(np.abs(activity["spatial_probability"].sum(axis=1) - 1.0))
        )
        rate_error = float(
            np.max(
                np.abs(
                    activity["activity_rate_cell"].sum(axis=1)
                    - activity["L5_total_activity"]
                )
            )
        )
        pairing_hashes.add(
            (
                manifest["l5_pairing"]["gp_index_sha256"],
                manifest["l5_pairing"]["l5_index_sha256"],
                manifest["l5_pairing"]["l5_paired_sha256"],
            )
        )
        rows.append(
            {
                "kernel": kernel,
                "kernel_label": manifest["kernel_label"],
                "maximum_raw_probability_sum_error": p_error,
                "maximum_paired_probability_sum_error": paired_p_error,
                "maximum_activity_rate_sum_error": rate_error,
                "maximum_gp_effect_centering_error": float(
                    np.max(np.abs(gp["gp_effect"].mean(axis=1)))
                ),
                "probability_passed": p_error <= 1e-12 and paired_p_error <= 1e-12,
                "activity_rate_passed": rate_error < 1e-10,
            }
        )
    if len(pairing_hashes) != 1:
        raise RuntimeError("The seven fits did not use the identical L5 pairing")
    frame = pd.DataFrame(rows)
    if not bool(frame[["probability_passed", "activity_rate_passed"]].all().all()):
        raise RuntimeError("One or more posterior conservation checks failed")
    return frame


def plot_correlation(curves: pd.DataFrame, config: dict[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(config["kernel_order"])))
    for color, kernel in zip(colors, config["kernel_order"]):
        case = curves[curves["kernel"] == kernel]
        axis.plot(
            case["distance_km"],
            case["correlation_median"],
            linewidth=1.8,
            color=color,
            label=config["kernels"][kernel]["short_label"],
        )
    axis.set_xlim(0, 500)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Distance between grid-cell centres (km)")
    axis.set_ylabel("Normalised covariance K(d)/K(0)")
    axis.grid(True, alpha=0.22, linewidth=0.6)
    axis.legend(ncol=2, frameon=False, loc="upper right")
    axis.set_title("Fitted normalised spatial correlation by covariance kernel")
    save_png_pdf(figure, "kernel_normalised_correlation_vs_distance")


def plot_spatial_comparison(comparison: pd.DataFrame) -> None:
    alternatives = comparison[comparison["kernel"] != "exp_quad"].copy()
    alternatives = alternatives.iloc[::-1].reset_index(drop=True)
    labels = alternatives["kernel_label"].tolist()
    y = np.arange(len(alternatives))
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    axes[0].barh(y, alternatives["pearson_vs_EQ"], color="#4472a3")
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Pearson correlation with EQ")
    axes[0].set_title("(a) Posterior-mean spatial correlation")
    lower = max(0.0, float(alternatives["pearson_vs_EQ"].min()) - 0.03)
    axes[0].set_xlim(lower, 1.005)
    axes[0].axvline(1.0, color="0.25", linewidth=0.9)
    for index, value in enumerate(alternatives["pearson_vs_EQ"]):
        axes[0].text(value, index, f" {value:.4f}", va="center", fontsize=8.5)

    axes[1].barh(y, alternatives["rmse_vs_EQ"], color="#bd6b38")
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("RMSE (events/year per cell)")
    axes[1].set_title("(b) Posterior-mean spatial RMSE")
    upper = float(alternatives["rmse_vs_EQ"].max())
    axes[1].set_xlim(0, upper * 1.18 if upper > 0 else 1)
    for index, value in enumerate(alternatives["rmse_vs_EQ"]):
        axes[1].text(value, index, f" {value:.4g}", va="center", fontsize=8.5)
    save_png_pdf(figure, "kernel_spatial_comparison_ab")


def plot_parameters(parameters: pd.DataFrame) -> None:
    frame = parameters.iloc[::-1].reset_index(drop=True)
    labels = frame["kernel_label"].tolist()
    y = np.arange(len(frame))
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.1), constrained_layout=True)
    for axis, prefix, xlabel, title, color in (
        (axes[0], "alpha", "alpha", "(a) Posterior alpha", "#4472a3"),
        (axes[1], "rho_km", "rho (km)", "(b) Posterior rho", "#bd6b38"),
    ):
        median = frame[f"{prefix}_median"].to_numpy(float)
        q025 = frame[f"{prefix}_q025"].to_numpy(float)
        q975 = frame[f"{prefix}_q975"].to_numpy(float)
        axis.errorbar(
            median,
            y,
            xerr=np.vstack([median - q025, q975 - median]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            markersize=5,
        )
        axis.set_yticks(y, labels)
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.22, linewidth=0.6)
    save_png_pdf(figure, "kernel_posterior_alpha_rho_ab")


def map_outputs(
    grid: pd.DataFrame, cases: dict[str, dict[str, Any]], config: dict[str, Any]
) -> pd.DataFrame:
    means = {
        kernel: np.mean(cases[kernel]["activity"]["activity_rate_cell"], axis=0)
        for kernel in config["kernel_order"]
    }
    all_values = np.concatenate(list(means.values()))
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    patches = [
        Rectangle(
            (float(row.lon_lo), float(row.lat_lo)),
            float(row.lon_hi - row.lon_lo),
            float(row.lat_hi - row.lat_lo),
        )
        for row in grid.itertuples(index=False)
    ]
    for kernel in config["kernel_order"]:
        figure, axis = plt.subplots(figsize=(6.2, 7.2), constrained_layout=True)
        collection = PatchCollection(
            patches,
            cmap="viridis",
            edgecolor=(0.2, 0.2, 0.2, 0.28),
            linewidth=0.28,
        )
        collection.set_array(means[kernel])
        collection.set_clim(vmin, vmax)
        axis.add_collection(collection)
        axis.set_xlim(float(grid["lon_lo"].min()), float(grid["lon_hi"].max()))
        axis.set_ylim(float(grid["lat_lo"].min()), float(grid["lat_hi"].max()))
        axis.set_aspect(1.0 / np.cos(np.deg2rad(55.0)))
        axis.set_xlabel("Longitude (degrees)")
        axis.set_ylabel("Latitude (degrees)")
        axis.set_title(
            f"Posterior mean annual activity rate — {config['kernels'][kernel]['label']}"
        )
        colorbar = figure.colorbar(collection, ax=axis, shrink=0.78)
        colorbar.set_label("Annual activity rate (events/year per cell)")
        figure.savefig(SENSITIVITY_ROOT / "maps" / f"posterior_mean_activity_{kernel}.png")
        plt.close(figure)
    pd.DataFrame(
        {
            "kernel": list(config["kernel_order"]),
            "kernel_label": [
                config["kernels"][kernel]["label"] for kernel in config["kernel_order"]
            ],
            "posterior_mean_activity_min": [float(np.min(means[k])) for k in config["kernel_order"]],
            "posterior_mean_activity_max": [float(np.max(means[k])) for k in config["kernel_order"]],
            "shared_colour_scale_min": vmin,
            "shared_colour_scale_max": vmax,
        }
    ).to_csv(SENSITIVITY_ROOT / "tables" / "kernel_map_colour_scale.csv", index=False)
    wide = pd.DataFrame({"grid_id": grid["grid_id"].to_numpy(int)})
    for kernel in config["kernel_order"]:
        wide[f"{kernel}_activity_rate_mean"] = means[kernel]
    wide.to_csv(
        SENSITIVITY_ROOT / "tables" / "kernel_cell_posterior_mean_activity_rates.csv",
        index=False,
    )
    return wide


def main() -> int:
    ensure_directories()
    baseline_check = validate_baseline()
    _, config = configs()
    cases: dict[str, dict[str, Any]] = {}
    for kernel in config["kernel_order"]:
        manifest, gp, activity = load_case(kernel)
        cases[kernel] = {"manifest": manifest, "gp": gp, "activity": activity}

    parameters = pd.DataFrame(
        [
            parameter_row(kernel, cases[kernel]["manifest"], cases[kernel]["gp"])
            for kernel in config["kernel_order"]
        ]
    )
    parameters.to_csv(
        SENSITIVITY_ROOT / "tables" / "kernel_parameter_summary.csv", index=False
    )
    curves = correlation_curves(config, cases)
    curves.to_csv(
        SENSITIVITY_ROOT / "tables" / "kernel_normalised_correlation_curves.csv",
        index=False,
    )
    comparison = spatial_comparison(cases)
    comparison.to_csv(
        SENSITIVITY_ROOT / "tables" / "kernel_spatial_comparison_vs_EQ.csv",
        index=False,
    )
    conservation = conservation_table(cases)
    conservation.to_csv(
        SENSITIVITY_ROOT / "tables" / "kernel_conservation_checks.csv", index=False
    )
    grid = pd.read_csv(BASELINE_GRID)
    map_outputs(grid, cases, config)

    plot_style()
    plot_correlation(curves, config)
    plot_spatial_comparison(comparison)
    plot_parameters(parameters)

    summary = {
        "complete": True,
        "kernels": list(config["kernel_order"]),
        "events": baseline_check["events"],
        "grid_cells": baseline_check["grid_cells"],
        "all_diagnostics_passed": bool(parameters["diagnostics_passed"].all()),
        "all_conservation_checks_passed": bool(
            conservation[["probability_passed", "activity_rate_passed"]].all().all()
        ),
        "same_l5_pairing_for_all_kernels": True,
        "outputs": {
            "parameter_summary": str(
                (SENSITIVITY_ROOT / "tables" / "kernel_parameter_summary.csv").resolve()
            ),
            "correlation_curves": str(
                (SENSITIVITY_ROOT / "tables" / "kernel_normalised_correlation_curves.csv").resolve()
            ),
            "spatial_comparison": str(
                (SENSITIVITY_ROOT / "tables" / "kernel_spatial_comparison_vs_EQ.csv").resolve()
            ),
            "figures": str((SENSITIVITY_ROOT / "figures").resolve()),
            "maps": str((SENSITIVITY_ROOT / "maps").resolve()),
        },
    }
    write_json(SENSITIVITY_ROOT / "output" / "analysis_summary.json", summary)
    print(parameters.to_string(index=False), flush=True)
    print(comparison.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
