"""Summarise, compare, and plot the three current multinomial GP prior sweeps."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _prior_common import (
    BASELINE_ACTIVITY_POSTERIOR,
    SENSITIVITY_ROOT,
    baseline_value,
    case_directory,
    ensure_directories,
    load_configs,
    load_json,
    sweep_values,
    write_json,
)


OUTPUT_NAMES = {
    "alpha": "alpha_prior_sensitivity.csv",
    "rho_median": "rho_prior_median_sensitivity.csv",
    "rho_logsd": "rho_prior_logsd_sensitivity.csv",
}


def load_activity(case_summary: dict) -> np.ndarray:
    if case_summary["posterior_source"]["reused_baseline_baseline"]:
        path = BASELINE_ACTIVITY_POSTERIOR
    else:
        path = Path(case_summary["posterior_source"]["activity_posterior"])
    with np.load(path, allow_pickle=False) as loaded:
        return loaded["activity_rate_cell"]


def row_from_summary(
    sweep: str,
    value: float,
    summary: dict,
    reference_mean: np.ndarray,
) -> dict:
    activity = load_activity(summary)
    activity_mean = activity.mean(axis=0)
    difference = activity_mean - reference_mean
    prior_after = summary["prior_change"]["after"]
    row = {
        "sweep": sweep,
        "tested_value": value,
        "is_baseline": bool(summary["is_baseline"]),
        "alpha_prior_sd": float(prior_after["alpha_prior_sd"]),
        "rho_prior_logmean_stan_units": float(prior_after["rho_prior_logmean"]),
        "rho_prior_logsd": float(prior_after["rho_prior_logsd"]),
        "rho_prior_median_km": float(
            np.exp(prior_after["rho_prior_logmean"]) * 100.0
        ),
        "alpha_mean": summary["alpha"]["mean"],
        "alpha_median": summary["alpha"]["median"],
        "alpha_sd": summary["alpha"]["sd"],
        "alpha_q025": summary["alpha"]["q025"],
        "alpha_q975": summary["alpha"]["q975"],
        "rho_km_mean": summary["rho_km"]["mean"],
        "rho_km_median": summary["rho_km"]["median"],
        "rho_km_sd": summary["rho_km"]["sd"],
        "rho_km_q025": summary["rho_km"]["q025"],
        "rho_km_q975": summary["rho_km"]["q975"],
        "pearson_vs_baseline": float(np.corrcoef(activity_mean, reference_mean)[0, 1]),
        "rmse_vs_baseline_events_per_year_per_cell": float(
            np.sqrt(np.mean(difference**2))
        ),
        "maximum_rhat": summary["mcmc"]["maximum_rhat"],
        "minimum_bulk_ess": summary["mcmc"]["minimum_bulk_ess"],
        "minimum_tail_ess": summary["mcmc"]["minimum_tail_ess"],
        "divergences": summary["mcmc"]["divergences"],
        "treedepth_hits": summary["mcmc"]["treedepth_hits"],
        "mcmc_all_passed": summary["mcmc"]["all_passed"],
        "runtime_minutes": summary["runtime_seconds"] / 60.0,
        "maximum_probability_sum_error": summary["validation"]["maximum_probability_sum_error"],
        "maximum_activity_rate_sum_error": summary["validation"]["maximum_activity_rate_sum_error"],
        "baseline_baseline_reused": summary["posterior_source"]["reused_baseline_baseline"],
        "only_changed_fields": "|".join(summary["prior_change"]["changed_fields"]),
    }
    del activity
    return row


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def error_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    median: str,
    q025: str,
    q975: str,
    ylabel: str,
    panel: str,
    baseline_x: float,
) -> None:
    values = frame[x].to_numpy(float)
    centre = frame[median].to_numpy(float)
    lower = frame[q025].to_numpy(float)
    upper = frame[q975].to_numpy(float)
    ax.errorbar(
        values,
        centre,
        yerr=np.vstack([centre - lower, upper - centre]),
        fmt="o-",
        linewidth=1.6,
        markersize=4.5,
        capsize=3,
        label="Posterior median and 95% CrI",
    )
    ax.axvline(baseline_x, color="0.25", linestyle="--", linewidth=1.2, label="Baseline prior")
    failed = frame.loc[~frame["mcmc_all_passed"]]
    if not failed.empty:
        ax.scatter(
            failed[x], failed[median], marker="x", s=55, linewidths=1.8,
            color="#b2182b", label="MCMC diagnostic flag", zorder=4,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(panel, loc="left", fontweight="bold")
    style_axis(ax)


def spatial_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    ylabel: str,
    panel: str,
    baseline_x: float,
) -> None:
    ax.plot(frame[x], frame[y], "o-", linewidth=1.6, markersize=4.5)
    ax.axvline(baseline_x, color="0.25", linestyle="--", linewidth=1.2, label="Baseline prior")
    failed = frame.loc[~frame["mcmc_all_passed"]]
    if not failed.empty:
        ax.scatter(
            failed[x], failed[y], marker="x", s=55, linewidths=1.8,
            color="#b2182b", label="MCMC diagnostic flag", zorder=4,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(panel, loc="left", fontweight="bold")
    style_axis(ax)


def plot_sweep(
    sweep: str,
    frame: pd.DataFrame,
    baseline_x: float,
    figures_dir: Path,
) -> list[str]:
    settings = {
        "alpha": {
            "x": "alpha_prior_sd",
            "xlabel": r"Half-Normal prior scale $\sigma_\alpha$",
            "posterior": [
                ("alpha_median", "alpha_q025", "alpha_q975", r"Posterior $\alpha$", r"(a) Amplitude response"),
                ("rho_km_median", "rho_km_q025", "rho_km_q975", r"Posterior $\rho$ (km)", r"(b) Length-scale response"),
            ],
            "stem": "alpha_prior",
            "title": "Amplitude-prior sensitivity",
        },
        "rho_median": {
            "x": "rho_prior_median_km",
            "xlabel": r"Physical prior median of $\rho$ (km)",
            "posterior": [
                ("rho_km_median", "rho_km_q025", "rho_km_q975", r"Posterior $\rho$ (km)", r"(a) Length-scale response"),
                ("alpha_median", "alpha_q025", "alpha_q975", r"Posterior $\alpha$", r"(b) Amplitude response"),
            ],
            "stem": "rho_prior_median",
            "title": "Length-scale prior-median sensitivity",
        },
        "rho_logsd": {
            "x": "rho_prior_logsd",
            "xlabel": r"LogNormal prior SD on the log scale",
            "posterior": [
                ("rho_km_median", "rho_km_q025", "rho_km_q975", r"Posterior $\rho$ (km)", r"(a) Length-scale response"),
                ("alpha_median", "alpha_q025", "alpha_q975", r"Posterior $\alpha$", r"(b) Amplitude response"),
            ],
            "stem": "rho_prior_logsd",
            "title": "Length-scale prior log-SD sensitivity",
        },
    }[sweep]
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.titlesize": 13,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, specification in zip(axes, settings["posterior"]):
        median, q025, q975, ylabel, panel = specification
        error_panel(
            ax, frame, settings["x"], median, q025, q975, ylabel, panel, baseline_x
        )
        ax.set_xlabel(settings["xlabel"])
    handles, labels = axes[0].get_legend_handles_labels()
    if len(labels) < 3:
        other_handles, other_labels = axes[1].get_legend_handles_labels()
        for handle, label in zip(other_handles, other_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle(settings["title"], y=0.985)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=len(labels), frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.76, wspace=0.22)
    posterior_file = figures_dir / f"{settings['stem']}_posterior_response.png"
    fig.savefig(posterior_file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    spatial_panel(
        axes[0], frame, settings["x"], "pearson_vs_baseline",
        "Pearson correlation with baseline", "(a) Spatial correlation", baseline_x,
    )
    spatial_panel(
        axes[1], frame, settings["x"],
        "rmse_vs_baseline_events_per_year_per_cell",
        "RMSE (events/year per 1-degree cell)", "(b) Spatial RMSE", baseline_x,
    )
    for ax in axes:
        ax.set_xlabel(settings["xlabel"])
    handles, labels = axes[0].get_legend_handles_labels()
    if len(labels) < 2:
        other_handles, other_labels = axes[1].get_legend_handles_labels()
        for handle, label in zip(other_handles, other_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle(f"{settings['title']}: spatial activity-rate response", y=0.985)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=len(labels), frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.76, wspace=0.22)
    spatial_file = figures_dir / f"{settings['stem']}_spatial_response.png"
    fig.savefig(spatial_file, dpi=300)
    plt.close(fig)
    return [str(posterior_file), str(spatial_file)]


def main() -> int:
    ensure_directories()
    _, sensitivity_config = load_configs()
    tables_dir = SENSITIVITY_ROOT / "output" / "tables"
    figures_dir = SENSITIVITY_ROOT / "output" / "figures"
    with np.load(BASELINE_ACTIVITY_POSTERIOR, allow_pickle=False) as loaded:
        reference_mean = loaded["activity_rate_cell"].mean(axis=0)
    output_manifest: dict[str, dict] = {}
    for sweep in ("alpha", "rho_median", "rho_logsd"):
        rows = []
        for value in sweep_values(sensitivity_config, sweep):
            summary_file = case_directory(sweep, value) / "run_summary.json"
            if not summary_file.is_file():
                raise FileNotFoundError(summary_file)
            summary = load_json(summary_file)
            rows.append(row_from_summary(sweep, value, summary, reference_mean))
            gc.collect()
        frame = pd.DataFrame(rows).sort_values("tested_value").reset_index(drop=True)
        table_file = tables_dir / OUTPUT_NAMES[sweep]
        frame.to_csv(table_file, index=False)
        figures = plot_sweep(
            sweep, frame, baseline_value(sensitivity_config, sweep), figures_dir
        )
        output_manifest[sweep] = {
            "rows": int(len(frame)),
            "table": str(table_file),
            "figures": figures,
            "mcmc_failed_values": frame.loc[
                ~frame["mcmc_all_passed"], "tested_value"
            ].tolist(),
        }
        print(
            f"SUMMARISED sweep={sweep} rows={len(frame)} "
            f"failures={output_manifest[sweep]['mcmc_failed_values']}",
            flush=True,
        )
    write_json(SENSITIVITY_ROOT / "output" / "summary_manifest.json", output_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
