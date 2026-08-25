"""Compile and sample the baseline multinomial spatial GP."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd

from _common import BASELINE_ROOT, ensure_directories, load_config, sha256_file, write_json


def indexed_columns(columns: list[str], prefix: str) -> list[str]:
    selected = [name for name in columns if name.startswith(f"{prefix}.")]
    return sorted(selected, key=lambda value: int(value.rsplit(".", 1)[1]))


def read_completed_chains(
    chain_files: list[Path],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read only required columns with pandas, avoiding a slow NumPy fallback."""
    header = pd.read_csv(chain_files[0], comment="#", nrows=0).columns.tolist()
    vector_prefixes = ("gp_effect", "spatial_probability")
    scalar_names = {
        "alpha",
        "rho",
        "rho_km",
        "divergent__",
        "treedepth__",
    }
    keep = [
        name
        for name in header
        if name in scalar_names
        or any(name.startswith(f"{prefix}.") for prefix in vector_prefixes)
    ]
    frames = [
        pd.read_csv(path, comment="#", usecols=keep, float_precision="high")
        for path in chain_files
    ]
    draws = pd.concat(frames, ignore_index=True)
    variables = {
        "alpha": draws["alpha"].to_numpy(float),
        "rho": draws["rho"].to_numpy(float),
        "rho_km": draws["rho_km"].to_numpy(float),
        "gp_effect": draws[
            indexed_columns(draws.columns.tolist(), "gp_effect")
        ].to_numpy(float),
        "spatial_probability": draws[
            indexed_columns(draws.columns.tolist(), "spatial_probability")
        ].to_numpy(float),
    }
    method_variables = {
        "divergent__": draws["divergent__"].to_numpy(float),
        "treedepth__": draws["treedepth__"].to_numpy(float),
    }
    return variables, method_variables


def cmdstan_diagnostics(
    cmdstan_path: Path, chain_files: list[Path]
) -> tuple[pd.DataFrame, str]:
    """Run CmdStan's native summary and diagnose utilities on saved chains."""
    summary_file = BASELINE_ROOT / "output" / "tables" / "gp_mcmc_summary.csv"
    summary_file.unlink(missing_ok=True)
    stansummary = cmdstan_path / "bin" / "stansummary.exe"
    diagnose = cmdstan_path / "bin" / "diagnose.exe"
    subprocess.run(
        [
            str(stansummary),
            "--percentiles=5,50,95",
            "--sig_figs=6",
            f"--csv_filename={summary_file}",
            *[str(path) for path in chain_files],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = pd.read_csv(
        summary_file,
        index_col=0,
        comment="#",
        float_precision="high",
    )
    summary = summary.loc[
        [name == "lp__" or not str(name).endswith("__") for name in summary.index]
    ]
    summary.index.name = "variable"
    summary.to_csv(summary_file)
    completed = subprocess.run(
        [str(diagnose), *[str(path) for path in chain_files]],
        check=True,
        capture_output=True,
        text=True,
    )
    return summary, completed.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--reuse-chain-csv",
        action="store_true",
        help="Post-process the existing four chain CSVs instead of sampling again",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    ensure_directories()
    stan_file = BASELINE_ROOT / "stan" / "baseline_gp.stan"
    stan_data_file = BASELINE_ROOT / "input" / "stan_data.json"
    if not stan_data_file.is_file():
        raise FileNotFoundError("Run scripts/01_prepare_gp_data.py first")

    configured_cmdstan = config["sampling"].get("cmdstan_path")
    cmdstan_path = (
        Path(configured_cmdstan).expanduser()
        if configured_cmdstan
        else Path(cmdstanpy.cmdstan_path())
    )
    if not cmdstan_path.is_dir():
        raise FileNotFoundError(f"Configured CmdStan installation not found: {cmdstan_path}")
    cmdstanpy.set_cmdstan_path(str(cmdstan_path))
    sampling = config["sampling"]
    if args.reuse_chain_csv:
        previous_manifest_file = (
            BASELINE_ROOT / "output" / "stan" / "run_manifest.json"
        )
        if not previous_manifest_file.is_file():
            raise FileNotFoundError(
                "No current run manifest is available for --reuse-chain-csv"
            )
        previous_manifest = json.loads(
            previous_manifest_file.read_text(encoding="utf-8")
        )
        current_stan_hash = sha256_file(stan_file)
        current_data_hash = sha256_file(stan_data_file)
        if previous_manifest.get("stan_file_sha256") != current_stan_hash or (
            previous_manifest.get("stan_data_sha256") != current_data_hash
        ):
            raise RuntimeError(
                "Existing chains do not match the current multinomial Stan model/input"
            )
        chain_files = [
            Path(path) for path in previous_manifest.get("chain_csv_files", [])
        ]
        if len(chain_files) != int(sampling["chains"]):
            raise RuntimeError(
                f"Expected {sampling['chains']} existing chain CSVs, found "
                f"{len(chain_files)}"
            )
        variables, method_variables = read_completed_chains(chain_files)
        summary, diagnostic_text = cmdstan_diagnostics(cmdstan_path, chain_files)
    else:
        model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))
        fit = model.sample(
            data=str(stan_data_file),
            chains=int(sampling["chains"]),
            parallel_chains=int(sampling["parallel_chains"]),
            iter_warmup=int(sampling["warmup_per_chain"]),
            iter_sampling=int(sampling["sampling_per_chain"]),
            seed=int(sampling["seed"]),
            adapt_delta=float(sampling["adapt_delta"]),
            max_treedepth=int(sampling["max_treedepth"]),
            sig_figs=int(sampling["sig_figs"]),
            refresh=int(sampling["refresh"]),
            output_dir=str(BASELINE_ROOT / "output" / "stan"),
            show_progress=True,
        )
        variables = {
            name: fit.stan_variable(name)
            for name in (
                "alpha",
                "rho",
                "rho_km",
                "gp_effect",
                "spatial_probability",
            )
        }
        method_variables = fit.method_variables()
        summary = fit.summary()
        diagnostic_text = fit.diagnose()
        chain_files = [Path(path).resolve() for path in fit.runset.csv_files]

    alpha = variables["alpha"]
    rho = variables["rho"]
    rho_km = variables["rho_km"]
    gp_effect = variables["gp_effect"]
    spatial_probability = variables["spatial_probability"]
    stan_data = json.loads(stan_data_file.read_text(encoding="utf-8"))
    cell_area = np.asarray(stan_data["cell_area_km2"], dtype=float)

    # Independently reproduce Stan's area-aware softmax in Python.
    log_weight = np.log(cell_area)[None, :] + gp_effect
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    python_probability = np.exp(log_weight)
    python_probability /= python_probability.sum(axis=1, keepdims=True)
    probability_crosscheck = float(
        np.max(np.abs(spatial_probability - python_probability))
    )
    probability_sum_error = float(
        np.max(np.abs(spatial_probability.sum(axis=1) - 1.0))
    )
    probability_sum_tolerance = float(
        config["validation"]["probability_sum_tolerance"]
    )
    crosscheck_tolerance = float(
        config["validation"]["python_stan_probability_tolerance"]
    )
    if (
        probability_sum_error > probability_sum_tolerance
        or probability_crosscheck > crosscheck_tolerance
    ):
        raise RuntimeError(
            "Spatial-probability validation failed: "
            f"sum error={probability_sum_error}, "
            f"Python/Stan discrepancy={probability_crosscheck}"
        )

    posterior_file = BASELINE_ROOT / "output" / "posterior" / "gp_posterior_draws.npz"
    np.savez_compressed(
        posterior_file,
        alpha=alpha,
        rho=rho,
        rho_km=rho_km,
        gp_effect=gp_effect,
        spatial_probability=spatial_probability,
    )

    summary.index.name = "variable"
    summary_file = BASELINE_ROOT / "output" / "tables" / "gp_mcmc_summary.csv"
    summary.to_csv(summary_file)
    diagnostic_file = BASELINE_ROOT / "output" / "stan" / "diagnose.txt"
    diagnostic_file.write_text(diagnostic_text, encoding="utf-8")

    # Match the audited grid-sensitivity convention: extrema are taken over
    # the complete Stan summary (including lp__ and transformed quantities).
    max_rhat = float(summary["R_hat"].max(skipna=True))
    min_ess_bulk = float(summary["ESS_bulk"].min(skipna=True))
    min_ess_tail = float(summary["ESS_tail"].min(skipna=True))
    divergences = int(np.asarray(method_variables["divergent__"]).sum())
    treedepth_hits = int(
        np.count_nonzero(
            np.asarray(method_variables["treedepth__"])
            >= int(sampling["max_treedepth"])
        )
    )
    key_variables = ["alpha", "rho", "rho_km"]
    key_rhat = {name: float(summary.loc[name, "R_hat"]) for name in key_variables}

    chain_files = [Path(path).resolve() for path in chain_files]
    manifest = {
        "configuration": str(config_path),
        "stan_file": str(stan_file),
        "stan_file_sha256": sha256_file(stan_file),
        "stan_data": str(stan_data_file),
        "stan_data_sha256": sha256_file(stan_data_file),
        "likelihood": "multinomial conditional on the observed catalogue total",
        "estimates_absolute_gp_rate": False,
        "cmdstan_path": str(cmdstan_path.resolve()),
        "cmdstanpy_version": cmdstanpy.__version__,
        "sampling": sampling,
        "posterior_draws": int(len(alpha)),
        "reused_existing_chain_csvs": bool(args.reuse_chain_csv),
        "chain_csv_files": [str(path) for path in chain_files],
        "outputs": {
            "posterior": str(posterior_file),
            "mcmc_summary": str(summary_file),
            "diagnose": str(diagnostic_file),
        },
        "diagnostics": {
            "maximum_rhat": max_rhat,
            "minimum_bulk_ess": min_ess_bulk,
            "minimum_tail_ess": min_ess_tail,
            "total_divergences": divergences,
            "maximum_treedepth_warnings": treedepth_hits,
            "key_parameter_rhat": key_rhat,
            "all_key_parameter_rhat_below_1_01": all(
                value < 1.01 for value in key_rhat.values()
            ),
        },
        "probability_validation": {
            "every_draw_sums_to_one": True,
            "maximum_sum_to_one_error": probability_sum_error,
            "maximum_python_stan_discrepancy": probability_crosscheck,
            "sum_tolerance": probability_sum_tolerance,
            "python_stan_crosscheck_tolerance": crosscheck_tolerance,
        },
    }
    write_json(BASELINE_ROOT / "output" / "stan" / "run_manifest.json", manifest)
    print(json.dumps(manifest["diagnostics"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
