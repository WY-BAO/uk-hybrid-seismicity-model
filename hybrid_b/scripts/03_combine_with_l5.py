"""Scale Hybrid B spatial-probability draws by the unchanged L5 posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    HYBRID_B_ROOT,
    baseline_path,
    load_baseline_config,
    load_config,
    sha256_file,
    validate_probability_array,
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
    baseline_config, baseline_config_path = load_baseline_config(config)
    gp_file = HYBRID_B_ROOT / "output" / "posterior" / "hybrid_b_gp_draws.npz"
    if not gp_file.is_file():
        raise FileNotFoundError("Run scripts/02_run_hybrid_b.py first")
    l5_file = baseline_path(
        baseline_config_path, baseline_config["paths"]["l5_posterior_draws"]
    )
    variable = str(baseline_config["l5_combination"]["variable"])
    l5_frame = pd.read_csv(l5_file)
    if variable not in l5_frame:
        raise KeyError(f"L5 posterior variable {variable!r} is absent")
    l5_values = l5_frame[variable].to_numpy(float)
    if not np.isfinite(l5_values).all() or np.any(l5_values <= 0.0):
        raise RuntimeError("L5 total-rate draws must all be finite and positive")

    with np.load(gp_file, allow_pickle=False) as loaded:
        gp = {name: loaded[name] for name in loaded.files}
    gp_draws = int(len(gp["rho"]))
    l5_draws = int(len(l5_values))
    base_seed = int(baseline_config["l5_combination"]["seed"])
    gp_seed = stable_seed(base_seed, "degree=1|gp")
    l5_seed = stable_seed(base_seed, "degree=1|l5")
    gp_rng = np.random.default_rng(gp_seed)
    l5_rng = np.random.default_rng(l5_seed)
    equal_draw_counts = gp_draws == l5_draws
    if equal_draw_counts:
        gp_index = gp_rng.permutation(gp_draws)
        pairing_method = "baseline independent permutations without replacement"
    else:
        gp_index = gp_rng.integers(0, gp_draws, size=l5_draws)
        pairing_method = "baseline unequal-count fallback: GP resampling with replacement"
    l5_index = l5_rng.permutation(l5_draws)
    l5_paired = l5_values[l5_index]
    p_hybrid = gp["p_hybrid"][gp_index]
    probability_error = validate_probability_array(
        p_hybrid, float(config["validation"]["probability_sum_tolerance"])
    )
    activity_rate_cell = p_hybrid * l5_paired[:, None]
    excluded_index = np.asarray(gp["excluded_cell_index"], dtype=int)
    maximum_excluded_probability = float(
        np.max(np.abs(p_hybrid[:, excluded_index]))
    )
    maximum_excluded_activity_rate = float(
        np.max(np.abs(activity_rate_cell[:, excluded_index]))
    )
    if maximum_excluded_probability != 0.0 or maximum_excluded_activity_rate != 0.0:
        raise RuntimeError("Domain-excluded cells must retain exact zero probability/rate")
    discrepancy = np.abs(activity_rate_cell.sum(axis=1) - l5_paired)
    maximum_discrepancy = float(discrepancy.max())
    tolerance = float(baseline_config["l5_combination"]["sum_tolerance"])
    if maximum_discrepancy >= tolerance:
        raise RuntimeError(
            f"L5 conservation failed: maximum discrepancy={maximum_discrepancy:.17g}"
        )

    output_file = HYBRID_B_ROOT / "output" / "posterior" / "hybrid_b_posterior_draws.npz"
    np.savez_compressed(
        output_file,
        p_source=gp["p_source"],
        p_hybrid=p_hybrid,
        f_correction=gp["f_correction"][gp_index],
        activity_rate_cell=activity_rate_cell,
        alpha=gp["alpha"][gp_index],
        rho=gp["rho"][gp_index],
        rho_km=gp["rho_km"][gp_index],
        L5_total_activity=l5_paired,
        gp_draw_index=gp_index,
        l5_draw_index=l5_index,
        modelled_cell_index=gp["modelled_cell_index"],
        excluded_cell_index=gp["excluded_cell_index"],
    )
    manifest = {
        "configuration": str(config_path),
        "gp_posterior": str(gp_file),
        "gp_posterior_sha256": sha256_file(gp_file),
        "l5_posterior": str(l5_file),
        "l5_posterior_sha256": sha256_file(l5_file),
        "l5_variable": variable,
        "pairing": {
            "method": pairing_method,
            "equal_input_draw_counts": equal_draw_counts,
            "replacement_used": not equal_draw_counts,
            "base_seed": base_seed,
            "gp_seed": gp_seed,
            "l5_seed": l5_seed,
            "gp_unique_draws_used": int(np.unique(gp_index).size),
            "l5_unique_draws_used": int(np.unique(l5_index).size),
        },
        "validation": {
            "maximum_p_hybrid_sum_error": probability_error,
            "maximum_l5_conservation_discrepancy": maximum_discrepancy,
            "maximum_excluded_cell_probability": maximum_excluded_probability,
            "maximum_excluded_cell_activity_rate": maximum_excluded_activity_rate,
            "strict_tolerance": tolerance,
        },
        "output": str(output_file),
    }
    write_json(HYBRID_B_ROOT / "output" / "posterior" / "combination_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
