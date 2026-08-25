# Hybrid Gaussian process model for UK seismicity rates

Code used for the dissertation model. The workflow estimates a UK-wide
activity rate with the L5 Bayesian model, distributes that rate with a spatial
Gaussian process, and combines the GP result with the BGS source-zone model.

## Repository structure

- `catalogue/`: catalogue cleaning and regional completeness filter.
- `l5/`: L5 rate model, L4 comparison, and Mw 2 generated quantities.
- `source_model/`: source-zone rate curves and the Mw 3 to Mw 2 extrapolation.
- `gp/`: baseline multinomial GP and grid, prior, and kernel sensitivities.
- `hybrid_a/`: source-to-grid conversion and weighted combinations.
- `hybrid_b/`: source-informed multinomial GP.
- `validation/`: retrospective 2013–2022 temporal holdout and paired
  year-block bootstrap.

Generated inputs, posterior samples, figures, and tables are not stored in the
repository. The original study used Python 3.12 and CmdStan 2.39.0.

## Inputs

Place the following files before running the workflow:

```text
catalogue/data/Data.csv
source_model/data/FMD_SSCmodel.txt
source_model/data/geometry_SSCmodel
```

`Data.csv` is expected to use the BGS catalogue column names referenced in
`catalogue/prepare_catalogue.py`. The two source-model files are the BGS
frequency-magnitude and geometry inputs.

## Environment

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install CmdStan through `cmdstanpy` or set `sampling.cmdstan_path` in
`gp/config/baseline_config.json`.

## Main workflow

```bash
python catalogue/prepare_catalogue.py

python l5/run_l5_regional.py
python l5/derive_l5_mw2.py

python source_model/plot_truncated_gr_activity_rates.py
python source_model/plot_mw2_extrapolated_activity_rates.py

python gp/scripts/01_prepare_gp_data.py
python gp/scripts/02_run_gp.py
python gp/scripts/03_combine_with_l5.py
python gp/scripts/04_summarize_results.py
python gp/scripts/05_plot_results.py

python hybrid_a/scripts/convert_source_model_to_gp_grid.py
python hybrid_a/scripts/plot_preweighting_model_comparison.py
python hybrid_a/scripts/run_equal_weighting_sensitivity.py
python hybrid_a/scripts/run_uncertainty_weighted_hybrid_a.py

python hybrid_b/scripts/01_prepare_hybrid_b_data.py
python hybrid_b/scripts/02_run_hybrid_b.py
python hybrid_b/scripts/03_combine_with_l5.py
python hybrid_b/scripts/04_summarize_results.py
python hybrid_b/scripts/05_plot_results.py
```

The retrospective temporal holdout is run after the main workflow has created
the catalogue, grid, source-to-grid, and uncertainty-weighting outputs:

```bash
python validation/run_temporal_holdout.py
python validation/bootstrap_common_support_scores.py
```

It fits L5, the baseline GP, and the Hybrid B correction using only events
before 2013, then scores all five spatial formulations on the 2013–2022 test
catalogue. Results are reported both on the full grid and on the common
source-positive support; exact zero-probability test events are not smoothed.

The L4 comparison is run separately:

```bash
python l5/compare_l5_selection.py
python l5/derive_l4_mw2.py
```

Sensitivity analyses are contained in `gp/grid_sensitivity`,
`gp/prior_sensitivity`, and `gp/kernel_sensitivity`. Each directory has a run
script followed by a summarisation script.

## Model scope

The spatial GP uses a multinomial likelihood conditional on the retained event
count. It estimates relative spatial probabilities, not an additional temporal
occurrence rate; annual rates are obtained by scaling with the L5 posterior.
Declustering is not applied. The 1-degree grid is a baseline resolution rather
than a validated optimum.

Hybrid A's uncertainty-based weights are illustrative because source-model
branch spread and GP posterior standard deviation are not equivalent measures
of uncertainty. Hybrid B is fitted only within the source-covered domain, so
eight events outside the source zones are excluded from its spatial likelihood.
The L5 national activity-rate posterior is unchanged by that exclusion.
