# ICLR 2027 Reproducibility Code

Public-release scripts corresponding to the supplied experiments.

## Files

- `n50_objective.py` — N=50 portfolio objective experiment.
- `n2_region.py` — N=2 trading-region recovery figure against the finite-horizon DP/QVI reference.
- `reservoir_release.py` — stochastic two-reservoir transfer experiment.
- `reservoir_table.py` — publication table generated from the reservoir experiment.
- `cara_highdim_regime.py` — high-dimensional independent-CARA neural regime experiment.

The spacecraft script is intentionally excluded.

## Cleaning policy

The public version removes implementation-era naming, classifier/auxiliary visualization code
that is not used in the final N=2 region result, ornamental/debug-only code, and redundant comments.
Training objectives, numerical dynamics, optimizer settings, Monte Carlo budgets, and reported
experiment hyperparameters are otherwise preserved.

For the N=2 script, the parameter values are taken from the latest supplied version:
`init_pi_sum_high=1.6`, `r=0.02`, `rho=0.0`, `n_train_steps=1000`,
`inner_mc_paths=256`, region tolerance `5e-2`, and seed `80`.

## CARA prerequisite

`cara_highdim_regime.py` is the supplied global-network experiment (CELL 2). It expects the
frozen independent-CARA oracle dataset and `oracle_manifest.json` produced by CELL 1 in
`cara_cara_uncorr_oracle/`. The oracle-generation CELL 1 was not included in the supplied
CARA body file, so it is not reconstructed here.

## Validation

All Python files in this archive pass Python syntax compilation.
Full paper-scale training was not rerun as part of this cleanup.
