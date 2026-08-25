"""Extend the existing BGS activity-rate figure from Mw 3 down to Mw 2.

The released BGS model retains m_min = 3.0.  Values for 2.0 <= Mw < 3.0
are branch-wise extrapolations of the same truncated Gutenberg-Richter
relationship. Existing values for Mw >= 3 are copied unchanged into the
extended table and figure.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
SOURCE_DIR = SCRIPT.parent
OUTPUT_DIR = SOURCE_DIR / "output"
BASELINE_SCRIPT = SOURCE_DIR / "plot_truncated_gr_activity_rates.py"
BASELINE_ZONE_CSV = OUTPUT_DIR / "source_zone_truncated_gr_activity_rates.csv"
BASELINE_COMPONENT_CSV = OUTPUT_DIR / "source_component_truncated_gr_activity_rates.csv"

MW_REFERENCE = 3.0
MW_EXTRAPOLATION_MIN = 2.0
MW_MAX = 7.1
BIPARTITE_SPLIT = 4.5
EXTRAPOLATED_THRESHOLDS = np.round(np.arange(2.0, 3.0, 0.1), 1)


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("bgs_baseline_activity_rates", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline implementation: {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = load_baseline_module()
ZONE_ORDER = BASELINE.ZONE_ORDER
COMPONENT_SPECS = BASELINE.COMPONENT_SPECS
PHYSICAL_COMPONENTS = BASELINE.PHYSICAL_COMPONENTS
FMD_FILE = BASELINE.FMD_FILE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sorted_zone_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_zone_order"] = result["zone"].map({zone: i for i, zone in enumerate(ZONE_ORDER)})
    return (
        result.sort_values(["_zone_order", "threshold_mw"])
        .drop(columns="_zone_order")
        .reset_index(drop=True)
    )


def sorted_component_table(frame: pd.DataFrame) -> pd.DataFrame:
    component_order = list(COMPONENT_SPECS)
    result = frame.copy()
    result["_component_order"] = result["component"].map(
        {component: i for i, component in enumerate(component_order)}
    )
    return (
        result.sort_values(["_component_order", "threshold_mw"])
        .drop(columns="_component_order")
        .reset_index(drop=True)
    )


def truncated_branch_rate(
    lambda3: np.ndarray,
    b_value: np.ndarray,
    threshold: float,
    component_support_min: float,
    component_max: float,
) -> np.ndarray:
    """Return branch rates, extrapolating only normal/lower components below Mw 3."""
    if component_support_min == BIPARTITE_SPLIT:
        # MMCW2/MENA2 are upper components.  Below Mw 4.5 their exceedance
        # contribution remains exactly the rate evaluated at Mw 4.5.
        effective_threshold = max(float(threshold), BIPARTITE_SPLIT)
    else:
        # Normal zones and MMCW1/MENA1 use the requested extrapolation below
        # the original BGS reference magnitude; MW_REFERENCE remains 3.0.
        effective_threshold = float(threshold)

    if effective_threshold >= component_max:
        return np.zeros_like(lambda3)

    beta = b_value * math.log(10.0)
    numerator = (
        np.exp(-beta * (effective_threshold - MW_REFERENCE))
        - np.exp(-beta * (component_max - MW_REFERENCE))
    )
    denominator = 1.0 - np.exp(-beta * (component_max - MW_REFERENCE))
    return lambda3 * numerator / denominator


def component_extensions(blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for component, (source_block, support_min, component_max) in COMPONENT_SPECS.items():
        frame = blocks[source_block]
        expected = 1 if component in {"BALA", "ESCO"} else 25
        if len(frame) != expected:
            raise ValueError(f"{component}: expected {expected} branches, found {len(frame)}")

        raw_weights = frame["weight"].to_numpy(float)
        weight_sum = float(raw_weights.sum())
        if not 0.98 <= weight_sum <= 1.02:
            raise ValueError(f"{component}: branch weights sum to {weight_sum}")
        weights = raw_weights / weight_sum
        lambda3 = np.power(10.0, frame["log10_lambda3"].to_numpy(float))
        b_values = frame["b"].to_numpy(float)

        for threshold in EXTRAPOLATED_THRESHOLDS:
            rates = truncated_branch_rate(
                lambda3,
                b_values,
                float(threshold),
                float(support_min),
                float(component_max),
            )
            mean = float(np.sum(weights * rates))
            variance = float(np.sum(weights * np.square(rates - mean)))
            rows.append(
                {
                    "component": component,
                    "source_block": source_block,
                    "threshold_mw": float(threshold),
                    "mean_rate_per_year": mean,
                    "sd_rate_per_year": math.sqrt(max(variance, 0.0)),
                    "support_min_mw": float(support_min),
                    "component_max_mw": float(component_max),
                    "branch_count": len(frame),
                    "weight_sum_raw": weight_sum,
                }
            )
    return pd.DataFrame(rows)


def physical_zone_extensions(components: pd.DataFrame) -> pd.DataFrame:
    lookup = components.set_index(["component", "threshold_mw"])
    rows: list[dict[str, float | str | bool]] = []
    for zone in ZONE_ORDER:
        names = PHYSICAL_COMPONENTS[zone]
        for threshold in EXTRAPOLATED_THRESHOLDS:
            records = [lookup.loc[(name, float(threshold))] for name in names]
            means = [float(record["mean_rate_per_year"]) for record in records]
            sds = [float(record["sd_rate_per_year"]) for record in records]
            rows.append(
                {
                    "zone": zone,
                    "threshold_mw": float(threshold),
                    "mean_rate_per_year": float(sum(means)),
                    "sd_rate_per_year": math.sqrt(sum(sd * sd for sd in sds)),
                    "lower_component_mean_rate_per_year": means[0],
                    "upper_component_mean_rate_per_year": means[1] if len(means) == 2 else 0.0,
                    "bipartite": len(names) == 2,
                    "components": "+".join(names),
                }
            )
    return pd.DataFrame(rows)


def component_join_values(blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate the extrapolation formula at Mw 3 solely for continuity QA."""
    rows = []
    for component, (source_block, support_min, component_max) in COMPONENT_SPECS.items():
        frame = blocks[source_block]
        weights = frame["weight"].to_numpy(float)
        weights = weights / weights.sum()
        rates = truncated_branch_rate(
            np.power(10.0, frame["log10_lambda3"].to_numpy(float)),
            frame["b"].to_numpy(float),
            MW_REFERENCE,
            float(support_min),
            float(component_max),
        )
        rows.append(
            {
                "component": component,
                "mean_rate_per_year": float(np.sum(weights * rates)),
            }
        )
    return pd.DataFrame(rows).set_index("component")


def validate(
    baseline_zones: pd.DataFrame,
    baseline_components: pd.DataFrame,
    combined_zones: pd.DataFrame,
    combined_components: pd.DataFrame,
    blocks: dict[str, pd.DataFrame],
) -> dict[str, object]:
    copied_baseline_zones = sorted_zone_table(
        combined_zones[combined_zones["threshold_mw"] >= MW_REFERENCE]
    )
    copied_baseline_components = sorted_component_table(
        combined_components[combined_components["threshold_mw"] >= MW_REFERENCE]
    )
    pd.testing.assert_frame_equal(
        copied_baseline_zones,
        sorted_zone_table(baseline_zones),
        check_exact=True,
        check_dtype=True,
    )
    pd.testing.assert_frame_equal(
        copied_baseline_components,
        sorted_component_table(baseline_components),
        check_exact=True,
        check_dtype=True,
    )

    thresholds = np.sort(combined_zones["threshold_mw"].unique())
    expected_thresholds = np.round(np.arange(2.0, 7.0 + 0.05, 0.1), 1)
    if not np.array_equal(thresholds, expected_thresholds):
        raise AssertionError("Combined thresholds must be Mw 2.0--7.0 in 0.1 increments")
    if len(combined_zones) != 22 * len(expected_thresholds):
        raise AssertionError("Unexpected combined source-zone curve dimensions")

    for zone, frame in combined_zones.groupby("zone"):
        rates = frame.sort_values("threshold_mw")["mean_rate_per_year"].to_numpy(float)
        if np.any(np.diff(rates) > 1e-12) or np.any(rates <= 0.0):
            raise AssertionError(f"{zone}: invalid combined exceedance curve")

    join_from_formula = component_join_values(blocks)
    baseline_at_three = baseline_components[
        np.isclose(baseline_components["threshold_mw"], MW_REFERENCE)
    ].set_index("component")
    component_join_differences = (
        join_from_formula["mean_rate_per_year"]
        - baseline_at_three["mean_rate_per_year"]
    ).abs()
    max_component_join_difference = float(component_join_differences.max())
    if max_component_join_difference > 1e-14:
        raise AssertionError(
            f"Extrapolated components do not join the baseline at Mw 3: "
            f"{max_component_join_difference}"
        )

    extension = combined_components[combined_components["threshold_mw"] < MW_REFERENCE]
    for upper in ("MMCW2", "MENA2"):
        upper_extension = extension[extension["component"] == upper]["mean_rate_per_year"]
        upper_at_split = baseline_components[
            (baseline_components["component"] == upper)
            & np.isclose(baseline_components["threshold_mw"], BIPARTITE_SPLIT)
        ]["mean_rate_per_year"].iloc[0]
        if not np.allclose(upper_extension, upper_at_split, rtol=0.0, atol=1e-14):
            raise AssertionError(f"{upper}: upper component changed below Mw 4.5")

    return {
        "physical_zones": 22,
        "threshold_count": len(expected_thresholds),
        "threshold_min_mw": float(expected_thresholds.min()),
        "threshold_max_mw": float(expected_thresholds.max()),
        "threshold_step_mw": 0.1,
        "original_bgs_mmin_mw": MW_REFERENCE,
        "truncation_mmax_mw": MW_MAX,
        "extrapolated_thresholds": [2.0, 2.9],
        "baseline_values_mw_ge_3_copied_exactly": True,
        "max_abs_component_join_difference_at_mw3": max_component_join_difference,
        "all_zone_curves_monotone_decreasing": True,
        "upper_components_constant_below_mw4_5": True,
        "bipartite_zones": ["MMCW", "MENA"],
    }


def plot_curves(
    combined: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 7.8,
        }
    )
    fig, ax_all = plt.subplots(figsize=(10.8, 6.5))

    base_colours = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Dark2").colors)
    zone_colours = {zone: base_colours[index] for index, zone in enumerate(ZONE_ORDER)}
    zone_colours["MMCW"] = "#9c2f45"
    zone_colours["MENA"] = "#6b4c9a"

    ax_all.axvspan(2.0, 3.0, color="#b9d9ec", alpha=0.30, zorder=0)
    for zone in ZONE_ORDER:
        original = baseline[baseline["zone"] == zone].sort_values("threshold_mw")
        extrapolated = combined[
            (combined["zone"] == zone) & (combined["threshold_mw"] < MW_REFERENCE)
        ].sort_values("threshold_mw")
        join_point = original[np.isclose(original["threshold_mw"], MW_REFERENCE)]
        extrapolated_with_join = pd.concat([extrapolated, join_point], ignore_index=True)
        special = zone in {"MMCW", "MENA"}
        style = {
            "color": zone_colours[zone],
            "linewidth": 2.5 if special else 1.25,
            "alpha": 1.0 if special else 0.84,
            "zorder": 4 if special else 2,
        }
        # Draw the original dataset as its own untouched segment.
        ax_all.plot(
            original["threshold_mw"],
            original["mean_rate_per_year"],
            label=zone,
            **style,
        )
        # Add only the extrapolated segment, including the original Mw=3 point
        # so that the two paths meet exactly.
        ax_all.plot(
            extrapolated_with_join["threshold_mw"],
            extrapolated_with_join["mean_rate_per_year"],
            **style,
        )

    ax_all.set_yscale("log")
    ax_all.set_xlabel(r"Magnitude threshold, $m$ in $M_w \geq m$")
    ax_all.set_ylabel(r"Annual exceedance rate, $\lambda(M_w \geq m)$ (yr$^{-1}$)")
    ax_all.set_title("All 22 physical source zones", loc="left")
    ax_all.grid(which="major", color="#d7d7d7", linewidth=0.6)
    ax_all.grid(which="minor", color="#ececec", linewidth=0.4)
    ax_all.axvline(MW_REFERENCE, color="#555555", linestyle=":", linewidth=1.1, zorder=1)
    ax_all.axvline(BIPARTITE_SPLIT, color="#666666", linestyle="--", linewidth=0.9, zorder=1)
    ax_all.text(
        2.5,
        0.975,
        "Extrapolated range",
        transform=ax_all.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#4f6673",
        fontsize=9,
        zorder=5,
    )
    ax_all.legend(
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handlelength=2.2,
    )
    ax_all.set_xlim(2.0, 7.0)
    ax_all.set_xticks(np.arange(2.0, 7.01, 0.5))
    fig.suptitle(
        "Truncated Gutenberg–Richter activity-rate curves for the BGS source model",
        x=0.47,
        y=0.975,
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.012,
        "Mw 2.0–3.0 is extrapolated with the truncated relationship while retaining "
        "the original BGS reference mmin = 3.0 and Mmax = 7.1.",
        fontsize=8.4,
        color="#444444",
    )
    fig.subplots_adjust(left=0.095, right=0.78, top=0.90, bottom=0.12)

    png_path = OUTPUT_DIR / "source_zone_activity_rates_mw2_extrapolated.png"
    svg_path = OUTPUT_DIR / "source_zone_activity_rates_mw2_extrapolated.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    required = [FMD_FILE, BASELINE_ZONE_CSV, BASELINE_COMPONENT_CSV, BASELINE_SCRIPT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required input(s) missing: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_zones = pd.read_csv(BASELINE_ZONE_CSV)
    baseline_components = pd.read_csv(BASELINE_COMPONENT_CSV)
    blocks = BASELINE.parse_fmd(FMD_FILE)

    extension_components = component_extensions(blocks)
    extension_zones = physical_zone_extensions(extension_components)
    combined_components = sorted_component_table(
        pd.concat([extension_components, baseline_components], ignore_index=True)
    )
    combined_zones = sorted_zone_table(
        pd.concat([extension_zones, baseline_zones], ignore_index=True)
    )
    validation = validate(
        baseline_zones,
        baseline_components,
        combined_zones,
        combined_components,
        blocks,
    )

    component_path = OUTPUT_DIR / "source_component_activity_rates_mw2_extrapolated.csv"
    curve_path = OUTPUT_DIR / "source_zone_activity_rates_mw2_extrapolated.csv"
    manifest_path = OUTPUT_DIR / "calculation_manifest.json"
    combined_components.to_csv(component_path, index=False)
    combined_zones.to_csv(curve_path, index=False)
    png_path, svg_path = plot_curves(combined_zones, baseline_zones)

    manifest = {
        "purpose": "Extrapolate the source-zone activity rates to Mw 2",
        "input_fmd": str(FMD_FILE),
        "input_fmd_sha256": sha256_file(FMD_FILE),
        "baseline_zone_csv": str(BASELINE_ZONE_CSV),
        "baseline_zone_csv_sha256": sha256_file(BASELINE_ZONE_CSV),
        "baseline_component_csv": str(BASELINE_COMPONENT_CSV),
        "baseline_component_csv_sha256": sha256_file(BASELINE_COMPONENT_CSV),
        "normal_and_lower_formula": (
            "lambda(>=m)=lambda3*[exp(-beta*(m-3))-exp(-beta*(component_max-3))]"
            "/[1-exp(-beta*(component_max-3))]"
        ),
        "parameter_conversion": "lambda3=10^x; beta=b*ln(10)",
        "original_bgs_mmin_mw": MW_REFERENCE,
        "normal_mmax_mw": MW_MAX,
        "lower_component_mmax_mw": BIPARTITE_SPLIT,
        "upper_component_rule": (
            "MMCW2/MENA2 are evaluated at Mw 4.5 and held constant for all lower thresholds"
        ),
        "bipartite_aggregation": (
            "MMCW=MMCW1+MMCW2 and MENA=MENA1+MENA2; branch weights are applied "
            "separately within each component"
        ),
        "unbounded_approximation_used": False,
        "validation": validation,
        "outputs": {
            "physical_zone_curves_csv": str(curve_path),
            "component_curves_csv": str(component_path),
            "figure_png": str(png_path),
            "figure_svg": str(svg_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {curve_path}")
    print(f"Wrote {component_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {manifest_path}")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
