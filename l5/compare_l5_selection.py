"""Compare the L5 model with and without the catalogue-selection term.

The script does not run catalogue preparation. Both fits consume the same
retained catalogue and the same fixed Stan-data JSON. The
no-selection Stan source is required to be byte-for-byte equivalent to the
baseline source after one exact source-line substitution:

    p_select_grid[q] = Phi(...);  ->  p_select_grid[q] = 1.0;

Any failed control check stops the experiment before sampling or reporting a
comparison as valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cmdstanpy
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, cmdstan_version, set_cmdstan_path


ROOT = Path(__file__).resolve().parents[1]
L5_DIR = Path(__file__).resolve().parent

BASELINE_PIPELINE = L5_DIR / "run_l5_regional.py"
CATALOGUE_FILE = ROOT / "catalogue" / "output" / "filtered_catalogue_regional.csv"
STAN_INPUT_FILE = L5_DIR / "output" / "l5_input_regional.json"
FULL_STAN_FILE = L5_DIR / "stafford_l5_latent_magnitude.stan"
NO_SELECTION_STAN_FILE = L5_DIR / "stafford_l5_no_selection.stan"
DEFAULT_OUTPUT_DIR = L5_DIR / "output" / "selection_ablation"
CMDSTAN_PATH = Path(cmdstanpy.cmdstan_path())

# Exactly the baseline settings in run_l5_regional.py.
MCMC_SETTINGS = {
    "chains": 4,
    "parallel_chains": 4,
    "iter_warmup": 1000,
    "iter_sampling": 1000,
    "seed": 20260713,
    "refresh": 200,
    "show_progress": False,
}

FULL_SELECTION_LINE = (
    "    p_select_grid[q] = "
    "Phi((ml_q - ml_detection_threshold) / sigma_select);"
)
NO_SELECTION_LINE = "    p_select_grid[q] = 1.0;"

REQUIRED_DATA_KEYS = {
    "N",
    "ml_reported",
    "sigma_ml",
    "sigma_round",
    "dm_ml",
    "mw_min",
    "mw_max",
    "mw_floor",
    "ml_detection_threshold",
    "n_exposure_rows",
    "exposure_min_ml",
    "exposure_time",
    "n_quad",
    "beta_prior_mean",
    "beta_prior_sd",
    "lambda_prior_mean",
    "lambda_prior_sd",
}


class ControlledAblationError(RuntimeError):
    """Raised when a control required for a valid ablation is not satisfied."""


def log(message: str = "") -> None:
    print(message, flush=True)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlledAblationError(message)


def load_fixed_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    required_paths = [
        BASELINE_PIPELINE,
        CATALOGUE_FILE,
        STAN_INPUT_FILE,
        FULL_STAN_FILE,
        NO_SELECTION_STAN_FILE,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    require(not missing, "Required baseline/ablation files are missing: " + ", ".join(missing))

    catalogue = pd.read_csv(CATALOGUE_FILE)
    stan_data = json.loads(STAN_INPUT_FILE.read_text(encoding="utf-8"))
    missing_keys = sorted(REQUIRED_DATA_KEYS - set(stan_data))
    require(not missing_keys, "Frozen Stan input is missing keys: " + ", ".join(missing_keys))
    return catalogue, stan_data


def validate_inputs_and_model_sources(
    catalogue: pd.DataFrame, stan_data: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Validate every ablation control before either model is sampled."""

    require("ml" in catalogue.columns, "Retained catalogue has no 'ml' column.")
    require(
        "sigma_ml" in catalogue.columns,
        "Retained catalogue has no 'sigma_ml' column.",
    )

    n_input = int(stan_data["N"])
    require(
        len(catalogue) == n_input,
        f"Catalogue has {len(catalogue)} rows but fixed Stan input has N={n_input}.",
    )
    require(
        len(stan_data["ml_reported"]) == n_input,
        "Length of ml_reported does not equal N in fixed Stan input.",
    )
    require(
        len(stan_data["sigma_ml"]) == n_input,
        "Length of sigma_ml does not equal N in fixed Stan input.",
    )

    catalogue_ml = catalogue["ml"].to_numpy(dtype=float)
    input_ml = np.asarray(stan_data["ml_reported"], dtype=float)
    catalogue_sigma = catalogue["sigma_ml"].to_numpy(dtype=float)
    input_sigma = np.asarray(stan_data["sigma_ml"], dtype=float)

    require(
        np.array_equal(catalogue_ml, input_ml),
        "Ordered retained-catalogue ML values do not exactly match the fixed Stan input.",
    )
    require(
        np.array_equal(catalogue_sigma, input_sigma),
        "Ordered retained-catalogue magnitude uncertainties do not exactly match "
        "the fixed Stan input.",
    )
    require(np.all(np.isfinite(input_sigma)), "Magnitude uncertainties contain non-finite values.")
    require(np.all(input_sigma >= 0.0), "Magnitude uncertainties contain negative values.")

    require(
        int(stan_data["n_exposure_rows"]) == len(stan_data["exposure_min_ml"])
        == len(stan_data["exposure_time"]),
        "Completeness/exposure arrays have inconsistent lengths.",
    )
    require(int(stan_data["n_quad"]) > 0, "n_quad must be positive.")
    require(
        float(stan_data["mw_floor"]) < float(stan_data["mw_max"]),
        "mw_floor must be below mw_max.",
    )
    require(
        float(stan_data["mw_floor"])
        <= float(stan_data["mw_min"])
        <= float(stan_data["mw_max"]),
        "mw_min must lie within the quadrature limits.",
    )

    full_source = FULL_STAN_FILE.read_text(encoding="utf-8")
    no_selection_source = NO_SELECTION_STAN_FILE.read_text(encoding="utf-8")
    require(
        full_source.count(FULL_SELECTION_LINE) == 1,
        "Baseline Stan source does not contain exactly one expected P_sel assignment.",
    )
    require(
        no_selection_source.count(NO_SELECTION_LINE) == 1,
        "No-selection Stan source does not contain exactly one P_sel=1 assignment.",
    )
    expected_no_selection_source = full_source.replace(
        FULL_SELECTION_LINE, NO_SELECTION_LINE, 1
    )
    require(
        no_selection_source == expected_no_selection_source,
        "Stan sources differ by more than the single controlled P_sel assignment.",
    )

    event_payload = {
        "N": n_input,
        "ml_reported": stan_data["ml_reported"],
        "sigma_ml": stan_data["sigma_ml"],
    }
    uncertainty_payload = {
        "sigma_ml": stan_data["sigma_ml"],
        "sigma_round": stan_data["sigma_round"],
        "dm_ml": stan_data["dm_ml"],
    }
    completeness_payload = {
        "n_exposure_rows": stan_data["n_exposure_rows"],
        "exposure_min_ml": stan_data["exposure_min_ml"],
        "exposure_time": stan_data["exposure_time"],
    }
    prior_payload = {
        "beta_prior_mean": stan_data["beta_prior_mean"],
        "beta_prior_sd": stan_data["beta_prior_sd"],
        "lambda_prior_mean": stan_data["lambda_prior_mean"],
        "lambda_prior_sd": stan_data["lambda_prior_sd"],
    }
    quadrature_payload = {
        "n_quad": stan_data["n_quad"],
        "mw_floor": stan_data["mw_floor"],
        "mw_min": stan_data["mw_min"],
        "mw_max": stan_data["mw_max"],
    }

    # Both CmdStan calls below receive the same data file and the same
    # shared MCMC_SETTINGS dictionary. Source equality above proves every Stan
    # operation other than the P_sel assignment is unchanged.
    validation = {
        "same_earthquake_rows": True,
        "same_N": True,
        "same_magnitude_uncertainties": True,
        "same_completeness_model": True,
        "same_magnitude_conversion": True,
        "same_priors": True,
        "same_quadrature_settings": True,
        "same_mw_limits": True,
        "same_mcmc_settings": True,
        "individual_event_likelihood_unchanged": True,
        "only_selection_probability_removed": True,
        "catalogue_filtering_not_rerun": True,
    }

    catalogue_rows_csv = catalogue.to_csv(index=False, lineterminator="\n")
    provenance = {
        "baseline_pipeline": str(BASELINE_PIPELINE.resolve()),
        "baseline_pipeline_sha256": sha256_file(BASELINE_PIPELINE),
        "input_catalogue": str(CATALOGUE_FILE.resolve()),
        "input_catalogue_sha256": sha256_file(CATALOGUE_FILE),
        "ordered_catalogue_rows_sha256": sha256_bytes(catalogue_rows_csv.encode("utf-8")),
        "stan_input": str(STAN_INPUT_FILE.resolve()),
        "stan_input_sha256": sha256_file(STAN_INPUT_FILE),
        "canonical_stan_input_sha256": sha256_json(stan_data),
        "event_payload_sha256": sha256_json(event_payload),
        "magnitude_uncertainty_payload_sha256": sha256_json(uncertainty_payload),
        "completeness_payload_sha256": sha256_json(completeness_payload),
        "prior_payload_sha256": sha256_json(prior_payload),
        "quadrature_payload_sha256": sha256_json(quadrature_payload),
        "mcmc_settings_sha256": sha256_json(MCMC_SETTINGS),
        "full_stan_model": str(FULL_STAN_FILE.resolve()),
        "full_stan_model_sha256": sha256_file(FULL_STAN_FILE),
        "no_selection_stan_model": str(NO_SELECTION_STAN_FILE.resolve()),
        "no_selection_stan_model_sha256": sha256_file(NO_SELECTION_STAN_FILE),
        "controlled_source_replacement": {
            "from": FULL_SELECTION_LINE.strip(),
            "to": NO_SELECTION_LINE.strip(),
        },
    }
    return validation, provenance


def print_preflight(stan_data: dict[str, Any], provenance: dict[str, Any]) -> None:
    log("=" * 78)
    log("CONTROLLED L5 SELECTION ABLATION — FROZEN BASELINE INPUT")
    log("=" * 78)
    log(f"Baseline pipeline:                       {provenance['baseline_pipeline']}")
    log(f"Input catalogue:                         {provenance['input_catalogue']}")
    log(f"Stan input:                              {provenance['stan_input']}")
    log(f"Full L5 Stan model:                      {provenance['full_stan_model']}")
    log(f"No-selection copied Stan model:          {provenance['no_selection_stan_model']}")
    log(f"Retained event count N:                  {int(stan_data['N'])}")
    log(f"Catalogue SHA-256:                       {provenance['input_catalogue_sha256']}")
    log(f"Event payload SHA-256:                   {provenance['event_payload_sha256']}")
    log(
        "Mw limits (floor, min, max):             "
        f"({stan_data['mw_floor']}, {stan_data['mw_min']}, {stan_data['mw_max']})"
    )
    log(f"Quadrature points:                       {int(stan_data['n_quad'])}")
    log(
        "MCMC (chains, warmup, sampling, seed):   "
        f"({MCMC_SETTINGS['chains']}, {MCMC_SETTINGS['iter_warmup']}, "
        f"{MCMC_SETTINGS['iter_sampling']}, {MCMC_SETTINGS['seed']})"
    )
    log("Preflight control checks:                PASS")
    log("=" * 78)


def finite_extreme(summary: pd.DataFrame, column: str, operation: str) -> float:
    values = pd.to_numeric(summary[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    require(len(values) > 0, f"No finite values found in CmdStan summary column {column!r}.")
    if operation == "max":
        return float(np.max(values))
    if operation == "min":
        return float(np.min(values))
    raise ValueError(operation)


def native_stansummary(csv_files: list[str]) -> pd.DataFrame:
    """Run CmdStan's stansummary without CmdStanPy's Windows polling loop."""

    executable = CMDSTAN_PATH / "bin" / "stansummary.exe"
    require(executable.is_file(), f"CmdStan stansummary executable is missing: {executable}")
    require(len(csv_files) > 0, "No CmdStan chain CSVs were supplied to stansummary.")
    summary_csv = Path(csv_files[0]).resolve().parent / "native_stansummary.csv"
    summary_csv.unlink(missing_ok=True)
    command = [
        str(executable),
        "--sig_figs=8",
        f"--csv_filename={summary_csv}",
        *csv_files,
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
    require(
        completed.returncode == 0,
        "CmdStan stansummary failed: " + completed.stdout.strip(),
    )
    require(summary_csv.is_file(), "CmdStan stansummary did not create its CSV output.")
    raw_summary = pd.read_csv(
        summary_csv,
        index_col=0,
        comment="#",
        float_precision="high",
    )
    # Match CmdStanMCMC.summary(): retain lp__ and model quantities, but not
    # sampler-internal columns such as accept_stat__ or treedepth__.
    keep = [name == "lp__" or not str(name).endswith("__") for name in raw_summary.index]
    return raw_summary.loc[keep]


def summarise_fit(fit: cmdstanpy.CmdStanMCMC, elapsed_seconds: float) -> dict[str, Any]:
    # CmdStanPy 1.3.0's NumPy draw assembler can stall in this Windows/Python
    # environment. Read the completed CmdStan CSV columns directly; this is a
    # reporting-only workaround and does not change sampling or model inputs.
    chain_frames = [
        pd.read_csv(path, comment="#", float_precision="high")
        for path in fit.runset.csv_files
    ]
    required_columns = {"b", "lambda_mw_min", "divergent__"}
    for path, frame in zip(fit.runset.csv_files, chain_frames):
        missing_columns = sorted(required_columns - set(frame.columns))
        require(
            not missing_columns,
            f"CmdStan CSV {path} is missing columns: {', '.join(missing_columns)}",
        )
    b = np.concatenate([frame["b"].to_numpy(dtype=float) for frame in chain_frames])
    lam = np.concatenate(
        [frame["lambda_mw_min"].to_numpy(dtype=float) for frame in chain_frames]
    )
    summary = native_stansummary(list(fit.runset.csv_files))
    divergences = int(sum(frame["divergent__"].sum() for frame in chain_frames))

    return {
        "posterior": {
            "b": {
                "mean": float(np.mean(b)),
                "q025": float(np.quantile(b, 0.025)),
                "q975": float(np.quantile(b, 0.975)),
            },
            "lambda_mw_ge_3_per_year": {
                "mean": float(np.mean(lam)),
                "q025": float(np.quantile(lam, 0.025)),
                "q975": float(np.quantile(lam, 0.975)),
            },
        },
        "diagnostics": {
            "max_rhat": finite_extreme(summary, "R_hat", "max"),
            "min_ess_bulk": finite_extreme(summary, "ESS_bulk", "min"),
            "divergences": divergences,
            "summary_scope": "all quantities returned by CmdStanMCMC.summary()",
        },
        "sampling_elapsed_seconds": float(elapsed_seconds),
        "draws": int(len(b)),
        "chain_csv_files": [str(Path(path).resolve()) for path in fit.runset.csv_files],
    }


def run_case(
    label: str,
    stan_file: Path,
    run_dir: Path,
    n_events: int,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"\n[{label}] Stan model: {stan_file.resolve()}")
    log(f"[{label}] Shared Stan input: {STAN_INPUT_FILE.resolve()}")
    log(f"[{label}] N = {n_events}")
    log(f"[{label}] Compiling/loading executable ...")
    model = CmdStanModel(stan_file=str(stan_file))
    log(f"[{label}] Executable: {Path(model.exe_file).resolve()}")
    log(f"[{label}] Sampling ...")
    started = time.perf_counter()
    fit = model.sample(
        data=str(STAN_INPUT_FILE),
        chains=MCMC_SETTINGS["chains"],
        parallel_chains=MCMC_SETTINGS["parallel_chains"],
        iter_warmup=MCMC_SETTINGS["iter_warmup"],
        iter_sampling=MCMC_SETTINGS["iter_sampling"],
        seed=MCMC_SETTINGS["seed"],
        output_dir=str(run_dir),
        refresh=MCMC_SETTINGS["refresh"],
        show_progress=MCMC_SETTINGS["show_progress"],
    )
    elapsed = time.perf_counter() - started
    result = summarise_fit(fit, elapsed)
    result["N"] = n_events
    result["stan_model"] = str(stan_file.resolve())
    result["stan_executable"] = str(Path(model.exe_file).resolve())
    log(
        f"[{label}] Done in {elapsed:.1f} s: "
        f"mean b={result['posterior']['b']['mean']:.6f}, "
        "mean lambda(Mw>=3)="
        f"{result['posterior']['lambda_mw_ge_3_per_year']['mean']:.6f} yr^-1"
    )
    return result


def comparison_rows(
    full: dict[str, Any], no_selection: dict[str, Any]
) -> list[dict[str, Any]]:
    full_b = full["posterior"]["b"]
    no_b = no_selection["posterior"]["b"]
    full_lam = full["posterior"]["lambda_mw_ge_3_per_year"]
    no_lam = no_selection["posterior"]["lambda_mw_ge_3_per_year"]
    pairs = [
        ("Number of retained events N", full["N"], no_selection["N"]),
        ("Mean b", full_b["mean"], no_b["mean"]),
        ("2.5% b", full_b["q025"], no_b["q025"]),
        ("97.5% b", full_b["q975"], no_b["q975"]),
        (
            "Mean lambda(Mw >= 3) [yr^-1]",
            full_lam["mean"],
            no_lam["mean"],
        ),
        (
            "2.5% lambda(Mw >= 3) [yr^-1]",
            full_lam["q025"],
            no_lam["q025"],
        ),
        (
            "97.5% lambda(Mw >= 3) [yr^-1]",
            full_lam["q975"],
            no_lam["q975"],
        ),
        (
            "Maximum R-hat",
            full["diagnostics"]["max_rhat"],
            no_selection["diagnostics"]["max_rhat"],
        ),
        (
            "Minimum bulk ESS",
            full["diagnostics"]["min_ess_bulk"],
            no_selection["diagnostics"]["min_ess_bulk"],
        ),
    ]
    return [
        {
            "Quantity": quantity,
            "Full L5 with selection": full_value,
            "Without selection": no_selection_value,
            "Difference": no_selection_value - full_value,
        }
        for quantity, full_value, no_selection_value in pairs
    ]


def percent_changes(
    full: dict[str, Any], no_selection: dict[str, Any]
) -> dict[str, float]:
    full_b = float(full["posterior"]["b"]["mean"])
    no_b = float(no_selection["posterior"]["b"]["mean"])
    full_lam = float(full["posterior"]["lambda_mw_ge_3_per_year"]["mean"])
    no_lam = float(no_selection["posterior"]["lambda_mw_ge_3_per_year"]["mean"])
    require(full_b != 0.0, "Cannot calculate percentage change: full-model mean b is zero.")
    require(
        full_lam != 0.0,
        "Cannot calculate percentage change: full-model mean lambda is zero.",
    )
    return {
        "posterior_mean_b_percent": 100.0 * (no_b - full_b) / full_b,
        "posterior_mean_lambda_mw_ge_3_percent": 100.0
        * (no_lam - full_lam)
        / full_lam,
    }


def format_value(value: Any, quantity: str) -> str:
    if quantity == "Number of retained events N":
        return str(int(round(float(value))))
    if quantity == "Minimum bulk ESS":
        return f"{float(value):.2f}"
    return f"{float(value):.8g}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Quantity",
        "Full L5 with selection",
        "Without selection",
        "Difference",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        quantity = str(row["Quantity"])
        values = [
            format_value(row["Full L5 with selection"], quantity),
            format_value(row["Without selection"], quantity),
            format_value(row["Difference"], quantity),
        ]
        lines.append(f"| {quantity} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def diagnostic_note(full: dict[str, Any], no_selection: dict[str, Any]) -> str:
    max_rhat = max(
        full["diagnostics"]["max_rhat"],
        no_selection["diagnostics"]["max_rhat"],
    )
    min_ess = min(
        full["diagnostics"]["min_ess_bulk"],
        no_selection["diagnostics"]["min_ess_bulk"],
    )
    divergences = (
        full["diagnostics"]["divergences"]
        + no_selection["diagnostics"]["divergences"]
    )
    if max_rhat <= 1.01 and min_ess >= 400 and divergences == 0:
        return (
            "Both fits met the reporting checks used here (maximum R-hat <= 1.01, "
            "minimum bulk ESS >= 400, and zero divergent transitions)."
        )
    return (
        "At least one sampling diagnostic warrants caution: "
        f"overall maximum R-hat={max_rhat:.5g}, overall minimum bulk ESS={min_ess:.2f}, "
        f"total divergences={divergences}."
    )


def write_outputs(
    output_dir: Path,
    stan_data: dict[str, Any],
    validation: dict[str, bool],
    provenance: dict[str, Any],
    full: dict[str, Any],
    no_selection: dict[str, Any],
) -> dict[str, Path]:
    require(all(validation.values()), "One or more final ablation control checks failed.")
    require(full["N"] == no_selection["N"] == int(stan_data["N"]), "Final N mismatch.")

    rows = comparison_rows(full, no_selection)
    changes = percent_changes(full, no_selection)
    created_at = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "selection_ablation_comparison.csv"
    md_path = output_dir / "selection_ablation_comparison.md"
    full_json_path = output_dir / "selection_ablation_full_result.json"
    no_selection_json_path = output_dir / "selection_ablation_no_selection_result.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.10g")

    shared_metadata = {
        "experiment": "controlled_ablation_of_selection_probability_in_expected_count",
        "created_at_utc": created_at,
        "valid_controlled_ablation": True,
        "difference_definition": "Without selection - Full L5 with selection",
        "baseline_N": int(stan_data["N"]),
        "mcmc_settings": MCMC_SETTINGS,
        "stan_data_controls": {
            key: stan_data[key]
            for key in [
                "sigma_round",
                "dm_ml",
                "mw_floor",
                "mw_min",
                "mw_max",
                "ml_detection_threshold",
                "n_exposure_rows",
                "exposure_min_ml",
                "exposure_time",
                "n_quad",
                "beta_prior_mean",
                "beta_prior_sd",
                "lambda_prior_mean",
                "lambda_prior_sd",
            ]
        },
        "validation": validation,
        "provenance": provenance,
        "percentage_changes_without_selection_minus_full": changes,
        "comparison": rows,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cmdstanpy": cmdstanpy.__version__,
            "cmdstan": str(cmdstan_version()),
        },
    }

    full_payload = {
        **shared_metadata,
        "case": "Full L5 with selection",
        "expected_count_integrand": "T(Mw) * f_GR(Mw | beta) * P_sel(Mw)",
        "result": full,
    }
    no_selection_payload = {
        **shared_metadata,
        "case": "Without selection",
        "expected_count_integrand": "T(Mw) * f_GR(Mw | beta)",
        "P_sel_at_all_quadrature_points": 1.0,
        "result": no_selection,
    }
    full_json_path.write_text(
        json.dumps(full_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    no_selection_json_path.write_text(
        json.dumps(no_selection_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    full_b = full["posterior"]["b"]["mean"]
    no_b = no_selection["posterior"]["b"]["mean"]
    full_lam = full["posterior"]["lambda_mw_ge_3_per_year"]["mean"]
    no_lam = no_selection["posterior"]["lambda_mw_ge_3_per_year"]["mean"]
    checks_md = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}"
        for name, passed in validation.items()
    )
    md_text = f"""# Controlled L5 selection ablation

Both cases use the same regional-completeness catalogue and fixed Stan input. Catalogue preparation is not repeated during this comparison.

- Input catalogue: `{provenance['input_catalogue']}`
- Stan input: `{provenance['stan_input']}`
- Full Stan model: `{provenance['full_stan_model']}`
- No-selection Stan model: `{provenance['no_selection_stan_model']}`
- Retained event count: **N = {int(stan_data['N'])}**
- Catalogue SHA-256: `{provenance['input_catalogue_sha256']}`
- Event-payload SHA-256: `{provenance['event_payload_sha256']}`
- MCMC: {MCMC_SETTINGS['chains']} chains, {MCMC_SETTINGS['iter_warmup']} warmup + {MCMC_SETTINGS['iter_sampling']} sampling iterations per chain, seed {MCMC_SETTINGS['seed']}

## Comparison

Difference is defined as **Without selection - Full L5 with selection**.

{markdown_table(rows)}

## Percentage change in posterior means

- Mean b: **{changes['posterior_mean_b_percent']:+.4f}%**
- Mean lambda(Mw >= 3): **{changes['posterior_mean_lambda_mw_ge_3_percent']:+.4f}%**

## Validation

{checks_md}

All required controls passed. Both cases used exactly the same ordered earthquake rows and N; the same event-specific magnitude uncertainties, ML-Mw conversion, completeness function and exposure data, priors, Mw limits, quadrature settings, and MCMC settings were retained. The only source change was `P_sel(Mw) = 1` at every quadrature point in the copied Stan model, so only the selection correction in the expected-count integral was removed. The individual-event magnitude-uncertainty likelihood was unchanged.

## Short interpretation

Removing the selection correction changed posterior mean b from {full_b:.6f} to {no_b:.6f} ({changes['posterior_mean_b_percent']:+.4f}%) and posterior mean lambda(Mw >= 3) from {full_lam:.6f} to {no_lam:.6f} yr^-1 ({changes['posterior_mean_lambda_mw_ge_3_percent']:+.4f}%). These values report the controlled posterior shift from the specified ablation; no scientific-model adjustment was made in response to the result. {diagnostic_note(full, no_selection)}
"""
    md_path.write_text(md_text, encoding="utf-8")
    return {
        "csv": csv_path,
        "markdown": md_path,
        "full_json": full_json_path,
        "no_selection_json": no_selection_json_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and print the fixed baseline controls without sampling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        catalogue, stan_data = load_fixed_inputs()
        validation, provenance = validate_inputs_and_model_sources(catalogue, stan_data)
        print_preflight(stan_data, provenance)
        if args.preflight_only:
            log("Preflight-only mode: no model was compiled or sampled.")
            return 0

        require(CMDSTAN_PATH.is_dir(), f"Configured CmdStan path does not exist: {CMDSTAN_PATH}")
        set_cmdstan_path(str(CMDSTAN_PATH))
        output_dir = args.output_dir.resolve()
        n_events = int(stan_data["N"])

        full = run_case(
            "Full L5 with selection",
            FULL_STAN_FILE,
            output_dir / "stan_run_full",
            n_events,
        )
        no_selection = run_case(
            "Without selection",
            NO_SELECTION_STAN_FILE,
            output_dir / "stan_run_no_selection",
            n_events,
        )
        paths = write_outputs(
            output_dir,
            stan_data,
            validation,
            provenance,
            full,
            no_selection,
        )

        log("\nFINAL VALIDATION: PASS")
        log("- both cases used exactly the same ordered earthquake rows")
        log(f"- both cases used the same N = {n_events}")
        log("- the same magnitude uncertainties were supplied")
        log("- the same completeness model was used")
        log("- the same priors, Mw limits, quadrature, and MCMC settings were used")
        log("- the only intended model change was P_sel(Mw)=1 in the expected-count integral")
        log("\nSaved:")
        for path in paths.values():
            log(f"- {path.resolve()}")
        return 0
    except ControlledAblationError as exc:
        log(f"\nCONTROLLED ABLATION INVALID — STOPPED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
