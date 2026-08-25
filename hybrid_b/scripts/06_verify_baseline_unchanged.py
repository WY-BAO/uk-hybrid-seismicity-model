"""Load and verify that the existing baseline files and outputs remain unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _common import HYBRID_B_ROOT, sha256_file, write_json


def main() -> int:
    preflight_file = HYBRID_B_ROOT / "input" / "preflight_report.json"
    if not preflight_file.is_file():
        raise FileNotFoundError("Run scripts/01_prepare_hybrid_b_data.py first")
    preflight = json.loads(preflight_file.read_text(encoding="utf-8"))
    snapshot = preflight["baseline_snapshot"]
    hashes = {}
    for label, record in snapshot.items():
        observed = sha256_file(Path(record["path"]))
        hashes[label] = {
            "path": record["path"],
            "expected_sha256": record["sha256"],
            "observed_sha256": observed,
            "unchanged": observed == record["sha256"],
        }

    gp_path = snapshot["gp_posterior"]["path"]
    combined_path = snapshot["combined_posterior"]["path"]
    with np.load(gp_path, allow_pickle=False) as loaded:
        spatial_probability = loaded["spatial_probability"]
        gp_effect = loaded["gp_effect"]
    with np.load(combined_path, allow_pickle=False) as loaded:
        activity = loaded["activity_rate_cell"]
        l5_total = loaded["L5_total_activity"]

    numerical = {
        "gp_probability_shape": list(spatial_probability.shape),
        "gp_effect_shape": list(gp_effect.shape),
        "maximum_probability_sum_error": float(
            np.max(np.abs(spatial_probability.sum(axis=1) - 1.0))
        ),
        "maximum_gp_effect_centring_error": float(
            np.max(np.abs(gp_effect.mean(axis=1)))
        ),
        "combined_activity_shape": list(activity.shape),
        "maximum_l5_conservation_discrepancy": float(
            np.max(np.abs(activity.sum(axis=1) - l5_total))
        ),
    }
    checks = {
        "all_snapshotted_baseline_hashes_unchanged": all(
            record["unchanged"] for record in hashes.values()
        ),
        "baseline_gp_shapes_are_4000_by_132": spatial_probability.shape
        == (4000, 132)
        and gp_effect.shape == (4000, 132),
        "baseline_probabilities_still_sum_to_one": numerical[
            "maximum_probability_sum_error"
        ]
        <= 1e-12,
        "baseline_gp_effect_still_centred": numerical[
            "maximum_gp_effect_centring_error"
        ]
        <= 1e-10,
        "baseline_l5_conservation_still_holds": numerical[
            "maximum_l5_conservation_discrepancy"
        ]
        <= 1e-10,
    }
    report = {
        "all_checks_passed": bool(all(checks.values())),
        "method": "Loaded existing baseline result; no baseline script was executed or written.",
        "hashes": hashes,
        "numerical": numerical,
        "checks": checks,
    }
    output = HYBRID_B_ROOT / "output" / "baseline_unchanged_verification.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise RuntimeError("Existing baseline files/results changed or failed validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
