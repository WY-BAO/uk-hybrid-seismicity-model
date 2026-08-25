"""Derive the regional L5 rate above Mw 2 from the existing posterior draws.

The L5 reference magnitude ``mw_min`` is used only in Stan's generated
quantities block.  It does not enter the likelihood or priors.  Consequently,
the posterior does not need to be sampled again when the reporting threshold
changes from Mw >= 3 to Mw >= 2.  This script runs Stan's standalone generated
quantities step on the original four-chain posterior and saves a fully
traceable Mw-2 result without overwriting the Mw-3 baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cmdstanpy import cmdstan_path


L5_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = L5_DIR / "output" / "mw2"
BASE_INPUT = L5_DIR / "output" / "l5_input_regional.json"
BASE_RESULT = L5_DIR / "output" / "l5_result_regional.json"
BASE_RUN_DIR = L5_DIR / "output" / "stan_run_regional"
STAN_FILE = L5_DIR / "stafford_l5_latent_magnitude.stan"
STAN_EXE = STAN_FILE.with_suffix(".exe")
CMDSTAN_PATH = Path(cmdstan_path())
TARGET_MW_MIN = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
    }


def lambda_above(
    beta: np.ndarray,
    lambda_floor: np.ndarray,
    mw_min: float,
    mw_floor: float,
    mw_max: float,
) -> np.ndarray:
    fraction = (
        np.exp(-beta * (mw_min - mw_floor))
        - np.exp(-beta * (mw_max - mw_floor))
    ) / (1.0 - np.exp(-beta * (mw_max - mw_floor)))
    return lambda_floor * fraction


def read_cmdstan_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", float_precision="high")


def native_stansummary(csv_files: list[Path], output_csv: Path) -> pd.DataFrame:
    executable = CMDSTAN_PATH / "bin" / "stansummary.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"Missing CmdStan stansummary executable: {executable}")
    command = [
        str(executable),
        "--sig_figs=10",
        f"--csv_filename={output_csv}",
        *[str(path) for path in csv_files],
    ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("CmdStan stansummary failed: " + completed.stdout.strip())
    return pd.read_csv(output_csv, index_col=0, comment="#", float_precision="high")


def run_generated_quantities(
    chain_files: list[Path], mw2_input_file: Path, output_dir: Path
) -> list[Path]:
    if not STAN_EXE.is_file():
        raise FileNotFoundError(f"Missing compiled Stan executable: {STAN_EXE}")
    output_files = []
    for chain, fitted_params in enumerate(chain_files, start=1):
        output_file = output_dir / f"mw2_generated_quantities_chain_{chain}.csv"
        if output_file.exists():
            output_files.append(output_file)
            continue
        command = [
            str(STAN_EXE),
            "generate_quantities",
            f"fitted_params={fitted_params}",
            "data",
            f"file={mw2_input_file}",
            "random",
            f"seed={20260813 + chain}",
            "output",
            f"file={output_file}",
            "refresh=0",
            "sig_figs=12",
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Generated quantities failed for chain {chain}: "
                + completed.stdout.strip()
            )
        if not output_file.is_file():
            raise RuntimeError(
                f"Generated quantities did not create the expected chain {chain} output"
            )
        output_files.append(output_file)
    return output_files


def markdown_summary(result: dict[str, Any]) -> str:
    b = result["posterior"]["b"]
    lam2 = result["posterior"]["lambda_mw_ge_2_per_year"]
    lam3 = result["posterior"]["lambda_mw_ge_3_per_year"]
    ratio = result["posterior"]["lambda2_to_lambda3_ratio"]
    diagnostics = result["diagnostics"]
    return f"""# Regional L5 result at Mw >= 2

The existing regional-composite L5 posterior was re-expressed at a reference
magnitude of Mw = 2 using Stan's standalone generated-quantities step. No new
MCMC sampling was required because `mw_min` occurs only in generated quantities
and therefore does not affect the likelihood, priors, or posterior parameters.

| Quantity | Posterior mean | 95% credible interval |
|---|---:|---:|
| b value | {b['mean']:.4f} | [{b['q025']:.4f}, {b['q975']:.4f}] |
| lambda(Mw >= 2), yr^-1 | {lam2['mean']:.3f} | [{lam2['q025']:.3f}, {lam2['q975']:.3f}] |
| lambda(Mw >= 3), yr^-1 | {lam3['mean']:.3f} | [{lam3['q025']:.3f}, {lam3['q975']:.3f}] |
| lambda2 / lambda3 | {ratio['mean']:.3f} | [{ratio['q025']:.3f}, {ratio['q975']:.3f}] |

## Diagnostics and validation

- Posterior draws: {result['posterior_draws']} from {result['chains']} chains.
- Divergent transitions in the original fit: {diagnostics['divergences']}.
- Maximum-treedepth hits in the original fit: {diagnostics['treedepth_hits']}.
- Derived lambda(Mw >= 2) R-hat: {diagnostics['lambda_mw_ge_2_rhat']:.5f}.
- Derived lambda(Mw >= 2) bulk ESS: {diagnostics['lambda_mw_ge_2_ess_bulk']:.1f}.
- Derived lambda(Mw >= 2) tail ESS: {diagnostics['lambda_mw_ge_2_ess_tail']:.1f}.
- Maximum absolute difference between the Stan-generated and independently
  recomputed Mw-2 draws: {result['validation']['max_abs_difference_stan_vs_manual_mw2']:.3e}.
- Maximum absolute difference between the saved and independently recomputed
  Mw-3 draws: {result['validation']['max_abs_difference_saved_vs_manual_mw3']:.3e}.

The magnitude catalogue, selection correction, completeness/exposure model,
magnitude uncertainties, b-value prior, rate prior, and posterior parameter
draws are identical to the regional Mw-3 baseline. Only the reported reference
magnitude changes.
"""


def main() -> int:
    required = [BASE_INPUT, BASE_RESULT, STAN_FILE, STAN_EXE, CMDSTAN_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(missing))

    chain_files = sorted(BASE_RUN_DIR.glob("stafford_l5_latent_magnitude-*.csv"))
    if len(chain_files) != 4:
        raise RuntimeError(f"Expected four regional chain CSVs, found {len(chain_files)}")

    base_input = json.loads(BASE_INPUT.read_text(encoding="utf-8"))
    base_result = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    mw_floor = float(base_input["mw_floor"])
    mw_max = float(base_input["mw_max"])
    original_mw_min = float(base_input["mw_min"])
    if not (mw_floor <= TARGET_MW_MIN <= mw_max):
        raise ValueError("Target Mw minimum lies outside the model integration range")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gq_dir = OUTPUT_DIR / "generated_quantities"
    gq_dir.mkdir(parents=True, exist_ok=True)

    mw2_input = dict(base_input)
    mw2_input["mw_min"] = TARGET_MW_MIN
    mw2_input_file = OUTPUT_DIR / "l5_input_regional_mw2.json"
    mw2_input_file.write_text(json.dumps(mw2_input, indent=2) + "\n", encoding="utf-8")

    gq_files = run_generated_quantities(chain_files, mw2_input_file, gq_dir)
    if len(gq_files) != 4:
        raise RuntimeError(f"Expected four generated-quantities CSVs, found {len(gq_files)}")

    original_frames = [read_cmdstan_csv(path) for path in chain_files]
    gq_frames = [read_cmdstan_csv(path) for path in gq_files]
    for chain, (original, gq) in enumerate(zip(original_frames, gq_frames), start=1):
        required_original = {"beta", "lambda_floor", "b", "lambda_mw_min", "divergent__", "treedepth__"}
        required_gq = {"b", "lambda_mw_min"}
        if missing_columns := sorted(required_original - set(original.columns)):
            raise RuntimeError(f"Original chain {chain} is missing {missing_columns}")
        if missing_columns := sorted(required_gq - set(gq.columns)):
            raise RuntimeError(f"Generated chain {chain} is missing {missing_columns}")
        if len(original) != len(gq):
            raise RuntimeError(f"Draw count mismatch in chain {chain}")

    beta = np.concatenate([frame["beta"].to_numpy(float) for frame in original_frames])
    lambda_floor = np.concatenate(
        [frame["lambda_floor"].to_numpy(float) for frame in original_frames]
    )
    b = np.concatenate([frame["b"].to_numpy(float) for frame in original_frames])
    lambda3_saved = np.concatenate(
        [frame["lambda_mw_min"].to_numpy(float) for frame in original_frames]
    )
    lambda2_stan = np.concatenate(
        [frame["lambda_mw_min"].to_numpy(float) for frame in gq_frames]
    )
    lambda2_manual = lambda_above(beta, lambda_floor, TARGET_MW_MIN, mw_floor, mw_max)
    lambda3_manual = lambda_above(beta, lambda_floor, original_mw_min, mw_floor, mw_max)

    max_diff_mw2 = float(np.max(np.abs(lambda2_stan - lambda2_manual)))
    max_diff_mw3 = float(np.max(np.abs(lambda3_saved - lambda3_manual)))
    # The new GQ files use 12 significant figures.  The original 2026-08-03
    # chain CSVs used CmdStan's 8-significant-figure default, so their saved
    # Mw-3 values legitimately have a looser round-trip tolerance.
    if max_diff_mw2 > 1e-8 or max_diff_mw3 > 5e-6:
        raise RuntimeError(
            f"Generated-quantity validation failed: Mw2={max_diff_mw2}, Mw3={max_diff_mw3}"
        )

    draw_rows = []
    offset = 0
    for chain, frame in enumerate(original_frames, start=1):
        n_draws = len(frame)
        segment = slice(offset, offset + n_draws)
        draw_rows.append(
            pd.DataFrame(
                {
                    "chain": chain,
                    "draw": np.arange(1, n_draws + 1),
                    "beta": beta[segment],
                    "b": b[segment],
                    "lambda_floor": lambda_floor[segment],
                    "lambda_mw_ge_2_per_year": lambda2_stan[segment],
                    "lambda_mw_ge_3_per_year": lambda3_saved[segment],
                }
            )
        )
        offset += n_draws
    draws = pd.concat(draw_rows, ignore_index=True)
    draws_file = OUTPUT_DIR / "l5_posterior_draws_mw2.csv"
    draws.to_csv(draws_file, index=False, float_format="%.12g")

    stansummary_file = OUTPUT_DIR / "stansummary_mw2.csv"
    stansummary = native_stansummary(gq_files, stansummary_file)
    lambda2_row = stansummary.loc["lambda_mw_min"]
    divergences = int(sum(frame["divergent__"].sum() for frame in original_frames))
    treedepth_hits = int(
        sum((frame["treedepth__"] >= 10).sum() for frame in original_frames)
    )
    ratio = lambda2_stan / lambda3_saved

    result = {
        "result_type": "exact_generated_quantity_reparameterization",
        "filter": base_result.get("filter"),
        "events": int(base_input["N"]),
        "chains": len(chain_files),
        "posterior_draws": int(len(draws)),
        "original_reference_magnitude": original_mw_min,
        "target_reference_magnitude": TARGET_MW_MIN,
        "mw_floor": mw_floor,
        "mw_max": mw_max,
        "posterior": {
            "b": quantile_summary(b),
            "lambda_mw_ge_2_per_year": quantile_summary(lambda2_stan),
            "lambda_mw_ge_3_per_year": quantile_summary(lambda3_saved),
            "lambda2_to_lambda3_ratio": quantile_summary(ratio),
        },
        "diagnostics": {
            "divergences": divergences,
            "treedepth_hits": treedepth_hits,
            "lambda_mw_ge_2_rhat": float(lambda2_row["R_hat"]),
            "lambda_mw_ge_2_ess_bulk": float(lambda2_row["ESS_bulk"]),
            "lambda_mw_ge_2_ess_tail": float(lambda2_row["ESS_tail"]),
            "original_fit_max_rhat": float(base_result["max_rhat"]),
            "original_fit_min_ess_bulk": float(base_result["min_ess_bulk"]),
        },
        "validation": {
            "mw_min_is_generated_quantities_only": True,
            "no_new_mcmc_sampling": True,
            "same_posterior_parameter_draws": True,
            "max_abs_difference_stan_vs_manual_mw2": max_diff_mw2,
            "max_abs_difference_saved_vs_manual_mw3": max_diff_mw3,
        },
        "provenance": {
            "stan_model": str(STAN_FILE.resolve()),
            "stan_model_sha256": sha256_file(STAN_FILE),
            "base_input": str(BASE_INPUT.resolve()),
            "base_input_sha256": sha256_file(BASE_INPUT),
            "base_result": str(BASE_RESULT.resolve()),
            "base_chain_csvs": [str(path.resolve()) for path in chain_files],
            "generated_quantities_csvs": [str(path.resolve()) for path in gq_files],
            "mw2_input": str(mw2_input_file.resolve()),
            "posterior_draws_csv": str(draws_file.resolve()),
        },
    }
    result_file = OUTPUT_DIR / "l5_result_regional_mw2.json"
    result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_file = OUTPUT_DIR / "RESULTS_MW2.md"
    summary_file.write_text(markdown_summary(result), encoding="utf-8")

    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved result: {result_file}", flush=True)
    print(f"Saved draws: {draws_file}", flush=True)
    print(f"Saved summary: {summary_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
