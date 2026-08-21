# AQUISTIL

Air-quality data imputation and evaluation pipeline, including regional pooled
evaluation, feature selection, missingness regimes, and classical/ensemble
imputation models.

The active model, region, feature-selection, missingness-regime, and
missingness-level settings are defined in `code/config_spatial.py`.

## AQUISTIL-R

`AQUISTIL_R` is the active generic model. It creates regime-matched artificial
gaps from observed values, collects out-of-fold predictions from AQUISTIL,
MICE_AQUISTIL, MICE-BR, AQUISTIL_A, and LightGBM, and learns a non-negative
robust blend. It preserves observed targets exactly and adds uncertainty,
confidence, and expert-disagreement columns for imputed rows.

Run one offline regime from the project root with:

```bash
../.venv/bin/python code/main.py --regime short_gap --skip-api-refresh
```

Deep-learning models such as `BRITS` require TensorFlow in the same venv used
by the Slurm job:

```bash
../.venv/bin/python -m pip install -r requirements-deeplearning.txt
```

The `AQUISTIL_R_*` settings in `code/config_spatial.py` control experts,
validation folds, regularization, safeguards, and bounds. The default three
validation folds target full evaluation quality. Set
`AQUISTIL_R_VALIDATION_FOLDS = 1` for a faster smoke run, or `0` to use the
regime priors without fitting the stack.
