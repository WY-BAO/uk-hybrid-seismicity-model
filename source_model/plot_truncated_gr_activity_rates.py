"""Plot BGS source-zone activity rates over Mw thresholds 3.0--7.0.

The calculation propagates the released BGS frequency-magnitude-distribution
(FMD) branches before taking weighted means.  MMCW and MENA retain their
published bipartite definitions: the physical-zone curve is the sum of its
lower (3.0 <= Mw < 4.5) and upper (4.5 <= Mw <= 7.1) components.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
SOURCE_DIR = SCRIPT.parent
OUTPUT_DIR = SOURCE_DIR / "output"
FMD_FILE = SOURCE_DIR / "data" / "FMD_SSCmodel.txt"

MW_REFERENCE = 3.0
MW_MAX = 7.1
BIPARTITE_SPLIT = 4.5
THRESHOLDS = np.round(np.arange(3.0, 7.0 + 0.05, 0.1), 1)

ZONE_ORDER = [
    "CORN",
    "RHEN",
    "WCHA",
    "DOVE",
    "SLPT",
    "EANG",
    "MMCE",
    "PENN",
    "MMCW",
    "MENA",
    "EISB",
    "CUMF",
    "BALA",
    "SC1M",
    "SC34",
    "SC78",
    "SC9",
    "ESCO",
    "IREL",
    "VIKI",
    "NORM",
    "PASC",
]

COMPONENT_SPECS = {
    **{zone: (zone, 3.0, MW_MAX) for zone in ZONE_ORDER if zone not in {"MMCW", "MENA"}},
    "MMCW1": ("MMC1", 3.0, BIPARTITE_SPLIT),
    "MMCW2": ("MMCW2", BIPARTITE_SPLIT, MW_MAX),
    "MENA1": ("MENA1", 3.0, BIPARTITE_SPLIT),
    "MENA2": ("MENA2", BIPARTITE_SPLIT, MW_MAX),
}

PHYSICAL_COMPONENTS = {
    **{zone: (zone,) for zone in ZONE_ORDER if zone not in {"MMCW", "MENA"}},
    "MMCW": ("MMCW1", "MMCW2"),
    "MENA": ("MENA1", "MENA2"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_fmd(path: Path) -> dict[str, pd.DataFrame]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: dict[str, pd.DataFrame] = {}
    index = 0
    while index < len(lines):
        name = lines[index].strip()
        index += 1
        if not name:
            continue
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            raise ValueError(f"Missing branch count after {name}")
        count = int(lines[index].strip())
        index += 1
        rows: list[list[float]] = []
        while len(rows) < count and index < len(lines):
            text = lines[index].strip()
            index += 1
            if not text:
                continue
            values = [float(token.replace(",", ".")) for token in text.split()[:3]]
            if len(values) != 3:
                raise ValueError(f"Malformed FMD row for {name}: {text}")
            rows.append(values)
        if len(rows) != count:
            raise ValueError(f"Expected {count} branches for {name}, found {len(rows)}")
        blocks[name] = pd.DataFrame(rows, columns=["log10_lambda3", "b", "weight"])
    return blocks


def branch_rate(
    lambda3_reference: np.ndarray,
    b_value: np.ndarray,
    threshold: float,
    support_min: float,
    component_max: float,
) -> np.ndarray:
    """Return branch rates above ``threshold`` within a truncated component."""
    effective_threshold = max(float(threshold), support_min)
    if effective_threshold >= component_max:
        return np.zeros_like(lambda3_reference)
    beta = b_value * math.log(10.0)
    numerator = (
        np.exp(-beta * (effective_threshold - MW_REFERENCE))
        - np.exp(-beta * (component_max - MW_REFERENCE))
    )
    denominator = 1.0 - np.exp(-beta * (component_max - MW_REFERENCE))
    return lambda3_reference * numerator / denominator


def component_curves(blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for component, (source_block, support_min, component_max) in COMPONENT_SPECS.items():
        if source_block not in blocks:
            raise KeyError(f"Released FMD is missing component {source_block}")
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

        for threshold in THRESHOLDS:
            rates = branch_rate(lambda3, b_values, threshold, support_min, component_max)
            mean = float(np.sum(weights * rates))
            variance = float(np.sum(weights * np.square(rates - mean)))
            rows.append(
                {
                    "component": component,
                    "source_block": source_block,
                    "threshold_mw": float(threshold),
                    "mean_rate_per_year": mean,
                    "sd_rate_per_year": math.sqrt(max(variance, 0.0)),
                    "support_min_mw": support_min,
                    "component_max_mw": component_max,
                    "branch_count": len(frame),
                    "weight_sum_raw": weight_sum,
                }
            )
    return pd.DataFrame(rows)


def physical_zone_curves(components: pd.DataFrame) -> pd.DataFrame:
    lookup = components.set_index(["component", "threshold_mw"])
    rows: list[dict[str, float | str | bool]] = []
    for zone in ZONE_ORDER:
        names = PHYSICAL_COMPONENTS[zone]
        for threshold in THRESHOLDS:
            records = [lookup.loc[(name, float(threshold))] for name in names]
            means = [float(record["mean_rate_per_year"]) for record in records]
            sds = [float(record["sd_rate_per_year"]) for record in records]
            rows.append(
                {
                    "zone": zone,
                    "threshold_mw": float(threshold),
                    "mean_rate_per_year": float(sum(means)),
                    # MMCW/MENA have no released cross-component covariance. This
                    # is a display-only independent-component SD, matching the
                    # existing source-model audit treatment.
                    "sd_rate_per_year": math.sqrt(sum(sd * sd for sd in sds)),
                    "lower_component_mean_rate_per_year": means[0],
                    "upper_component_mean_rate_per_year": means[1] if len(means) == 2 else 0.0,
                    "bipartite": len(names) == 2,
                    "components": "+".join(names),
                }
            )
    result = pd.DataFrame(rows)
    result["zone"] = pd.Categorical(result["zone"], categories=ZONE_ORDER, ordered=True)
    return result.sort_values(["zone", "threshold_mw"]).reset_index(drop=True)


def validate(curves: pd.DataFrame) -> dict[str, object]:
    if len(curves) != len(ZONE_ORDER) * len(THRESHOLDS):
        raise AssertionError("Unexpected physical-zone curve dimensions")
    if curves["zone"].nunique() != 22:
        raise AssertionError("Expected all 22 physical source zones")
    if not np.allclose(sorted(curves["threshold_mw"].unique()), THRESHOLDS):
        raise AssertionError("Magnitude thresholds are not the requested 0.1 increments")
    for zone, frame in curves.groupby("zone", observed=True):
        rates = frame.sort_values("threshold_mw")["mean_rate_per_year"].to_numpy(float)
        if np.any(np.diff(rates) > 1e-12):
            raise AssertionError(f"{zone}: exceedance rate is not monotone decreasing")
        if np.any(rates <= 0.0):
            raise AssertionError(f"{zone}: non-positive plotted rate")

    split = curves[np.isclose(curves["threshold_mw"], BIPARTITE_SPLIT)].set_index("zone")
    for zone in ("MMCW", "MENA"):
        if not math.isclose(
            float(split.loc[zone, "lower_component_mean_rate_per_year"]),
            0.0,
            abs_tol=1e-14,
        ):
            raise AssertionError(f"{zone}: lower component should terminate at Mw 4.5")

    return {
        "zones": 22,
        "threshold_count": len(THRESHOLDS),
        "threshold_min_mw": float(THRESHOLDS.min()),
        "threshold_max_mw": float(THRESHOLDS.max()),
        "threshold_step_mw": 0.1,
        "mmax_mw": MW_MAX,
        "monotone_decreasing_all_zones": True,
        "bipartite_zones_combined": ["MMCW", "MENA"],
        "bipartite_split_mw": BIPARTITE_SPLIT,
    }


def plot_curves(curves: pd.DataFrame) -> tuple[Path, Path]:
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

    for zone in ZONE_ORDER:
        frame = curves[curves["zone"] == zone]
        special = zone in {"MMCW", "MENA"}
        ax_all.plot(
            frame["threshold_mw"],
            frame["mean_rate_per_year"],
            color=zone_colours[zone],
            linewidth=2.5 if special else 1.25,
            alpha=1.0 if special else 0.84,
            label=zone,
            zorder=4 if special else 2,
        )

    ax_all.set_yscale("log")
    ax_all.set_xlabel(r"Magnitude threshold, $m$ in $M_w \geq m$")
    ax_all.set_ylabel(r"Annual exceedance rate, $\lambda(M_w \geq m)$ (yr$^{-1}$)")
    ax_all.set_title("All 22 physical source zones", loc="left")
    ax_all.grid(which="major", color="#d7d7d7", linewidth=0.6)
    ax_all.grid(which="minor", color="#ececec", linewidth=0.4)
    ax_all.axvline(BIPARTITE_SPLIT, color="#666666", linestyle="--", linewidth=0.9, zorder=1)
    ax_all.legend(
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handlelength=2.2,
    )
    ax_all.set_xlim(3.0, 7.0)
    ax_all.set_xticks(np.arange(3.0, 7.01, 0.5))
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
        "Weighted means of the released BGS recurrence branches; Mmax = 7.1. "
        "MMCW and MENA totals sum their lower- and upper-magnitude components.",
        fontsize=8.4,
        color="#444444",
    )
    fig.subplots_adjust(left=0.095, right=0.78, top=0.90, bottom=0.12)

    png_path = OUTPUT_DIR / "source_zone_truncated_gr_activity_rates.png"
    svg_path = OUTPUT_DIR / "source_zone_truncated_gr_activity_rates.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    if not FMD_FILE.is_file():
        raise FileNotFoundError(FMD_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = parse_fmd(FMD_FILE)
    components = component_curves(blocks)
    curves = physical_zone_curves(components)
    validation = validate(curves)

    component_path = OUTPUT_DIR / "source_component_truncated_gr_activity_rates.csv"
    curve_path = OUTPUT_DIR / "source_zone_truncated_gr_activity_rates.csv"
    manifest_path = OUTPUT_DIR / "calculation_manifest.json"
    components.to_csv(component_path, index=False)
    curves.to_csv(curve_path, index=False)
    png_path, svg_path = plot_curves(curves)

    manifest = {
        "input_fmd": str(FMD_FILE),
        "input_fmd_sha256": sha256_file(FMD_FILE),
        "formula": (
            "lambda(>=m)=lambda3*[exp(-beta*(max(m,support_min)-3))"
            "-exp(-beta*(component_max-3))]/[1-exp(-beta*(component_max-3))]"
        ),
        "aggregation": (
            "Weighted means are calculated over released component branches. "
            "MMCW=MMCW1+MMCW2 and MENA=MENA1+MENA2 at every threshold."
        ),
        "uncertainty_note": (
            "For MMCW/MENA only, displayed component SDs are combined in quadrature; "
            "the released file provides no cross-component covariance."
        ),
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


if __name__ == "__main__":
    main()
