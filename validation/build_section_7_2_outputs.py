"""Build the two separate Section 7.2 comparisons.

1. Full-data pairwise spatial correlation between the five final models.
2. Predictive accuracy on the 2013--2022 temporal holdout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SECTION_OUTPUT = HERE / "output" / "section_7_2"
TABLES = SECTION_OUTPUT / "tables"
FIGURES = SECTION_OUTPUT / "figures"

FULL_MODELS = (
    ROOT
    / "hybrid_b"
    / "output"
    / "tables"
    / "five_model_comparison.csv"
)
HOLDOUT_CELLS = (
    ROOT
    / "validation"
    / "output"
    / "tables"
    / "model_cell_predictions_and_test_counts.csv"
)

MODEL_COLUMNS_FULL = {
    "Source-zone": "source_activity_rate_mean",
    "GP": "baseline_gp_activity_rate_mean",
    "Hybrid A equal": "equal_weight_hybrid_a_mean",
    "Hybrid A uncertainty": "uncertainty_weighted_hybrid_mean",
    "Hybrid B": "hybrid_b_activity_rate_per_year_mean",
}

MODEL_COLUMNS_TEST = {
    "Source-zone": "source_rate_mean",
    "GP": "gp_rate_mean_pre2013",
    "Hybrid A equal": "hybrid_a_equal_rate_mean",
    "Hybrid A uncertainty": "hybrid_a_uncertainty_rate_mean",
    "Hybrid B": "hybrid_b_rate_mean_pre2013",
}

MODEL_COLOURS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
        }
    )


def pairwise_correlation() -> pd.DataFrame:
    frame = pd.read_csv(FULL_MODELS).sort_values("grid_id")
    if len(frame) != 132:
        raise RuntimeError(f"Expected 132 final-model cells, found {len(frame)}")
    names = list(MODEL_COLUMNS_FULL)
    spatial_proportions = []
    for column in MODEL_COLUMNS_FULL.values():
        rate = frame[column].to_numpy(float)
        spatial_proportions.append(rate / rate.sum())
    matrix = np.corrcoef(np.vstack(spatial_proportions))
    result = pd.DataFrame(matrix, index=names, columns=names)
    result.to_csv(TABLES / "pairwise_spatial_correlation_full_models.csv")

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Pairwise spatial correlation of the five final models", pad=12)
    for row in range(len(names)):
        for column in range(len(names)):
            value = matrix[row, column]
            text_colour = "white" if value >= 0.83 else "black"
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_colour,
                fontsize=9,
            )
    colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colourbar.set_label("Pearson correlation coefficient")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_pairwise_spatial_correlation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_pairwise_spatial_correlation.pdf", bbox_inches="tight")
    plt.close(fig)
    return result


def predictive_accuracy() -> pd.DataFrame:
    frame = pd.read_csv(HOLDOUT_CELLS).sort_values("grid_id")
    if len(frame) != 132:
        raise RuntimeError(f"Expected 132 holdout cells, found {len(frame)}")
    observed_rate = frame["test_count_2013_2022"].to_numpy(float) / 10.0
    observed_total = float(observed_rate.sum())
    rows = []
    for model, column in MODEL_COLUMNS_TEST.items():
        predicted = frame[column].to_numpy(float)
        rows.append(
            {
                "model": model,
                "cell_rate_rmse_events_per_year": float(
                    np.sqrt(np.mean((predicted - observed_rate) ** 2))
                ),
                "cell_rate_mae_events_per_year": float(
                    np.mean(np.abs(predicted - observed_rate))
                ),
                "correlation_with_observed_cell_rates": float(
                    np.corrcoef(predicted, observed_rate)[0, 1]
                ),
                "predicted_total_rate_per_year": float(predicted.sum()),
                "observed_total_rate_per_year": observed_total,
                "absolute_total_rate_error_per_year": float(
                    abs(predicted.sum() - observed_total)
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "temporal_holdout_predictive_accuracy.csv", index=False)

    names = result["model"].tolist()
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))

    rmse = result["cell_rate_rmse_events_per_year"].to_numpy(float)
    axes[0].barh(y, rmse, color=MODEL_COLOURS)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE (events/year/cell; lower is better)")
    axes[0].set_title("(a) Cell-rate prediction error")
    axes[0].grid(axis="x", color="#E6E6E6", linewidth=0.7)
    axes[0].set_axisbelow(True)
    for row, value in enumerate(rmse):
        axes[0].text(value + 0.003, row, f"{value:.3f}", va="center", fontsize=8)
    axes[0].set_xlim(0.0, max(rmse) * 1.18)

    agreement = result["correlation_with_observed_cell_rates"].to_numpy(float)
    agreement_left = 0.45
    axes[1].hlines(y, agreement_left, agreement, color="#D9D9D9", linewidth=2)
    axes[1].scatter(agreement, y, color=MODEL_COLOURS, s=55, zorder=3)
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(agreement_left, 0.67)
    axes[1].set_xlabel("Correlation with observed cell rates (higher is better)")
    axes[1].set_title("(b) Spatial agreement with test data")
    axes[1].grid(axis="x", color="#E6E6E6", linewidth=0.7)
    for row, value in enumerate(agreement):
        axes[1].text(value + 0.006, row, f"{value:.3f}", va="center", fontsize=8)

    totals = result["predicted_total_rate_per_year"].to_numpy(float)
    axes[2].barh(y, totals, color=MODEL_COLOURS)
    axes[2].axvline(
        observed_total,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"Observed: {observed_total:.1f}/year",
    )
    axes[2].set_yticks(y, names)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("National activity rate (events/year)")
    axes[2].set_title("(c) Predicted and observed totals")
    axes[2].legend(frameon=False, loc="lower right", fontsize=8)
    axes[2].grid(axis="x", color="#E6E6E6", linewidth=0.7)
    axes[2].set_axisbelow(True)
    for row, value in enumerate(totals):
        axes[2].text(value + 0.35, row, f"{value:.2f}", va="center", fontsize=8)
    axes[2].set_xlim(0.0, max(max(totals), observed_total) * 1.14)

    fig.tight_layout(w_pad=2.4)
    fig.savefig(FIGURES / "figure_temporal_holdout_predictive_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_temporal_holdout_predictive_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    return result


def write_section_draft(correlation: pd.DataFrame, accuracy: pd.DataFrame) -> None:
    c = correlation
    a = accuracy.set_index("model")
    lines = [
        "### 7.2 Comparison of the final spatial models",
        "",
        "Two complementary comparisons were conducted. First, pairwise correlations were calculated between the 132-cell spatial distributions of the five final models. This is a descriptive comparison of the models fitted to the complete catalogue and is not part of the temporal test. Second, predictive accuracy was evaluated using a retrospective temporal holdout in which events before 2013 were used for fitting and events during 2013–2022 were retained for testing.",
        "",
        "#### 7.2.1 Pairwise spatial correlation",
        "",
        f"The equal-weight Hybrid A remained most similar to the source-zone model (r = {c.loc['Hybrid A equal', 'Source-zone']:.3f}), whereas the uncertainty-weighted Hybrid A was more strongly correlated with the GP (r = {c.loc['Hybrid A uncertainty', 'GP']:.3f}). Hybrid B was also strongly correlated with the GP (r = {c.loc['Hybrid B', 'GP']:.3f}), indicating that the fitted GP correction substantially influenced its final spatial pattern. These correlations describe similarity between the final models only; they do not measure predictive accuracy.",
        "",
        "#### 7.2.2 Predictive accuracy for the 2013–2022 test period",
        "",
        "For the temporal test, the observed annual rate in each cell was calculated as the number of test events divided by the ten-year test duration. Predictive accuracy was assessed by comparing these observed cell rates with the rates predicted from the pre-2013 training catalogue. Cell-rate RMSE measures the magnitude of the prediction error, while correlation with the observed cell rates measures agreement in the spatial pattern.",
        "",
        f"The uncertainty-weighted Hybrid A produced the lowest cell-rate RMSE ({a.loc['Hybrid A uncertainty', 'cell_rate_rmse_events_per_year']:.4f} events/year/cell) and the highest correlation with the observed test rates (r = {a.loc['Hybrid A uncertainty', 'correlation_with_observed_cell_rates']:.3f}). However, its RMSE was almost identical to that of the equal-weight Hybrid A ({a.loc['Hybrid A equal', 'cell_rate_rmse_events_per_year']:.4f}), so the difference should not be interpreted as a substantial improvement. The equal-weight Hybrid A predicted a national total of {a.loc['Hybrid A equal', 'predicted_total_rate_per_year']:.2f} events/year, which was closest to the observed test rate of {a.loc['Hybrid A equal', 'observed_total_rate_per_year']:.1f} events/year.",
        "",
        f"Hybrid B produced an RMSE of {a.loc['Hybrid B', 'cell_rate_rmse_events_per_year']:.4f} and a correlation of {a.loc['Hybrid B', 'correlation_with_observed_cell_rates']:.3f}. It therefore did not improve clearly on the baseline GP. In addition, two test events occurred in cells assigned zero activity by the source-zone model and Hybrid B. The source-zone model gave the largest RMSE ({a.loc['Source-zone', 'cell_rate_rmse_events_per_year']:.4f}) and the lowest correlation with the observed cell rates (r = {a.loc['Source-zone', 'correlation_with_observed_cell_rates']:.3f}).",
        "",
        "Overall, no hybrid formulation was uniformly superior. The uncertainty-weighted Hybrid A provided the best nominal cell-level spatial accuracy, while the equal-weight Hybrid A gave the most accurate national total. Hybrid B remained close to the GP and retained the structural limitation associated with zero-source cells. The temporal analysis should be interpreted as a retrospective test rather than a fully prospective validation because the wider modelling framework, including the grid and prior choices, was developed using the complete study context.",
        "",
    ]
    (SECTION_OUTPUT / "section_7_2_draft.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    correlation = pairwise_correlation()
    accuracy = predictive_accuracy()
    write_section_draft(correlation, accuracy)
    print("Pairwise spatial correlation:\n")
    print(correlation.round(4).to_string())
    print("\nTemporal holdout predictive accuracy:\n")
    print(accuracy.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
