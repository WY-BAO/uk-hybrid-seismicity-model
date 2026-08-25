"""Fit the L5 model to the regional-completeness catalogue."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


ROOT = Path(__file__).resolve().parents[1]
L5_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOGUE = ROOT / "catalogue" / "output" / "filtered_catalogue_regional.csv"
DEFAULT_OUTPUT = L5_ROOT / "output"
STAN_FILE = L5_ROOT / "stafford_l5_latent_magnitude.stan"

MW_MIN = 3.0
MW_MAX = 6.5
MW_FLOOR = 1.0
ML_DETECTION_THRESHOLD = 2.0
CATALOGUE_END_YEAR = 2023.0
DM_ML = 0.1
N_MAG_QUAD = 100

FILTER_TABLE = pd.DataFrame(
    [
        {"ML": 2.0, "complete_from": 1990},
        {"ML": 2.5, "complete_from": 1979},
        {"ML": 3.0, "complete_from": 1970},
        {"ML": 3.5, "complete_from": 1850},
        {"ML": 4.0, "complete_from": 1750},
        {"ML": 4.5, "complete_from": 1700},
        {"ML": 5.0, "complete_from": 1650},
        {"ML": 5.5, "complete_from": 1650},
        {"ML": 6.5, "complete_from": 1000},
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--sampling", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def build_stan_input(catalogue: pd.DataFrame) -> dict:
    return {
        "N": int(len(catalogue)),
        "ml_reported": catalogue["ml"].astype(float).to_list(),
        "sigma_ml": catalogue["sigma_ml"].astype(float).to_list(),
        "sigma_round": float(DM_ML / math.sqrt(12.0)),
        "dm_ml": DM_ML,
        "mw_min": MW_MIN,
        "mw_max": MW_MAX,
        "mw_floor": MW_FLOOR,
        "ml_detection_threshold": ML_DETECTION_THRESHOLD,
        "n_exposure_rows": int(len(FILTER_TABLE)),
        "exposure_min_ml": FILTER_TABLE["ML"].astype(float).to_list(),
        "exposure_time": (
            CATALOGUE_END_YEAR - FILTER_TABLE["complete_from"].astype(float)
        ).to_list(),
        "n_quad": N_MAG_QUAD,
        "beta_prior_mean": float(math.log(10.0)),
        "beta_prior_sd": 0.5,
        "lambda_prior_mean": float(math.log(100.0)),
        "lambda_prior_sd": 2.0,
    }


def main() -> None:
    args = parse_args()
    catalogue = pd.read_csv(args.catalogue)
    if len(catalogue) != 1013:
        raise RuntimeError(f"Expected 1013 retained events, found {len(catalogue)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / "stan_run_regional"
    run_dir.mkdir(parents=True, exist_ok=True)
    stan_input_path = args.output_dir / "l5_input_regional.json"
    result_path = args.output_dir / "l5_result_regional.json"

    stan_input = build_stan_input(catalogue)
    stan_input_path.write_text(json.dumps(stan_input, indent=2) + "\n", encoding="utf-8")

    cmdstanpy.set_cmdstan_path(cmdstanpy.cmdstan_path())
    model = CmdStanModel(stan_file=str(STAN_FILE))
    fit = model.sample(
        data=str(stan_input_path),
        chains=args.chains,
        parallel_chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.sampling,
        seed=args.seed,
        output_dir=str(run_dir),
        show_progress=False,
        refresh=200,
    )

    b_value = fit.stan_variable("b")
    activity_rate = fit.stan_variable("lambda_mw_min")
    fit_summary = fit.summary()
    result = {
        "events": int(stan_input["N"]),
        "b_mean": float(np.mean(b_value)),
        "b_q025": float(np.quantile(b_value, 0.025)),
        "b_q975": float(np.quantile(b_value, 0.975)),
        "lambda_mw_min_mean": float(np.mean(activity_rate)),
        "lambda_mw_min_q025": float(np.quantile(activity_rate, 0.025)),
        "lambda_mw_min_q975": float(np.quantile(activity_rate, 0.975)),
        "max_rhat": float(fit_summary["R_hat"].max()),
        "min_ess_bulk": float(fit_summary["ESS_bulk"].min()),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
