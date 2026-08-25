"""Scale GP spatial-probability draws by independent L5 total-rate draws."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    BASELINE_ROOT,
    ensure_directories,
    load_config,
    resolve_project_path,
    sha256_file,
    write_json,
)


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    ensure_directories()
    gp_file = BASELINE_ROOT / "output" / "posterior" / "gp_posterior_draws.npz"
    if not gp_file.is_file():
        raise FileNotFoundError("Run scripts/02_run_gp.py first")
    l5_file = resolve_project_path(config["paths"]["l5_posterior_draws"])
    variable = str(config["l5_combination"]["variable"])
    l5_frame = pd.read_csv(l5_file)
    if variable not in l5_frame.columns:
        raise KeyError(f"L5 posterior variable {variable!r} is absent from {l5_file}")
    l5_values = l5_frame[variable].to_numpy(float)
    if not np.isfinite(l5_values).all() or np.any(l5_values <= 0):
        raise RuntimeError("L5 total-rate draws must all be finite and positive")

    with np.load(gp_file) as loaded:
        gp = {name: loaded[name] for name in loaded.files}
    gp_draws = int(len(gp["rho"]))
    l5_draws = int(len(l5_values))
    if gp_draws < 1 or l5_draws < 1:
        raise RuntimeError("Both posterior sources must contain at least one draw")

    # Pair independent posteriors reproducibly without positional coupling.
    # When the empirical posterior sizes match, independent permutations use
    # every draw from both models exactly once and avoid needless resampling
    # noise.  The audited with-replacement rule remains an explicit fallback
    # only for unequal draw counts; inputs are never silently truncated.
    base_seed = int(config["l5_combination"]["seed"])
    gp_seed = stable_seed(base_seed, "degree=1|gp")
    l5_seed = stable_seed(base_seed, "degree=1|l5")
    gp_rng = np.random.default_rng(gp_seed)
    l5_rng = np.random.default_rng(l5_seed)
    equal_draw_counts = gp_draws == l5_draws
    if equal_draw_counts:
        gp_index = gp_rng.permutation(gp_draws)
        pairing_method = (
            "independent random permutations of all GP and all L5 posterior "
            "draws; one-to-one pairing without replacement"
        )
    else:
        gp_index = gp_rng.integers(0, gp_draws, size=l5_draws)
        pairing_method = (
            "sample GP posterior rows with replacement to the L5 draw count; "
            "independently permute and retain all L5 posterior draws"
        )
    l5_index = l5_rng.permutation(l5_draws)
    gp_unique_draws_used = int(np.unique(gp_index).size)
    l5_unique_draws_used = int(np.unique(l5_index).size)
    if equal_draw_counts and gp_unique_draws_used != gp_draws:
        raise RuntimeError("Equal-count GP pairing did not use every draw exactly once")
    if l5_unique_draws_used != l5_draws:
        raise RuntimeError("L5 pairing did not retain every draw exactly once")
    l5_paired = l5_values[l5_index]
    spatial_probability = gp["spatial_probability"][gp_index]
    probability_sum_error = float(
        np.max(np.abs(spatial_probability.sum(axis=1) - 1.0))
    )
    probability_tolerance = float(
        config["validation"]["probability_sum_tolerance"]
    )
    if probability_sum_error > probability_tolerance:
        raise RuntimeError(
            "Input GP spatial probabilities do not sum to one: "
            f"maximum error={probability_sum_error:.17g}"
        )
    activity_rate_cell = spatial_probability * l5_paired[:, None]
    discrepancy = np.abs(activity_rate_cell.sum(axis=1) - l5_paired)
    maximum_discrepancy = float(discrepancy.max())
    tolerance = float(config["l5_combination"]["sum_tolerance"])
    if maximum_discrepancy >= tolerance:
        raise RuntimeError(
            "Draw-level L5 scaling failed: maximum discrepancy "
            f"{maximum_discrepancy:.17g} is not below {tolerance:.17g}"
        )

    output_file = (
        BASELINE_ROOT
        / "output"
        / "posterior"
        / "baseline_posterior_draws.npz"
    )
    np.savez_compressed(
        output_file,
        gp_effect=gp["gp_effect"][gp_index],
        spatial_probability=spatial_probability,
        activity_rate_cell=activity_rate_cell,
        rho=gp["rho"][gp_index],
        rho_km=gp["rho_km"][gp_index],
        alpha=gp["alpha"][gp_index],
        L5_total_activity=l5_paired,
        gp_draw_index=gp_index,
        l5_draw_index=l5_index,
    )
    manifest = {
        "configuration": str(config_path),
        "gp_posterior_file": str(gp_file),
        "gp_posterior_sha256": sha256_file(gp_file),
        "l5_posterior_file": str(l5_file),
        "l5_posterior_sha256": sha256_file(l5_file),
        "l5_variable": variable,
        "gp_posterior_draws": gp_draws,
        "l5_posterior_draws": l5_draws,
        "retained_combined_draws": l5_draws,
        "pairing": {
            "method": pairing_method,
            "equal_input_draw_counts": equal_draw_counts,
            "replacement_used": not equal_draw_counts,
            "base_seed": base_seed,
            "gp_permutation_or_resampling_seed": gp_seed,
            "l5_permutation_seed": l5_seed,
            "silent_truncation": False,
            "gp_unique_draws_used": gp_unique_draws_used,
            "l5_unique_draws_used": l5_unique_draws_used,
        },
        "validation": {
            "every_spatial_probability_draw_sums_to_one": True,
            "maximum_spatial_probability_sum_error": probability_sum_error,
            "all_gp_draws_used_exactly_once": bool(
                equal_draw_counts and gp_unique_draws_used == gp_draws
            ),
            "all_l5_draws_used_exactly_once": bool(
                l5_unique_draws_used == l5_draws
            ),
            "all_final_draws_sum_to_paired_l5_total": True,
            "maximum_absolute_discrepancy": maximum_discrepancy,
            "strict_tolerance": tolerance,
        },
        "output": str(output_file),
    }
    write_json(
        BASELINE_ROOT / "output" / "posterior" / "combination_manifest.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
