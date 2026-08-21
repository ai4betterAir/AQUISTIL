"""MICE-AQUISTIL: LightGBM chained predictors with an AQUISTIL target model.

The auxiliary-variable stage uses chained LightGBM regressions while excluding
the target from every auxiliary model. The completed predictor frame is then
passed to AQUISTIL, which supplies leakage-safe observed history, temporal,
spatial, gap and event-aware target modelling.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from Model.AQUISTIL import impute_mice as aquistil_impute


MODEL_NAME = "MICE_AQUISTIL"


def _numeric_predictors(data, input_columns):
    """Return a numeric predictor frame without changing the source data."""
    columns = list(dict.fromkeys(c for c in input_columns if c in data.columns))
    return data[columns].apply(pd.to_numeric, errors="coerce")


def _initialise_predictors(frame, original_missing):
    """Median initialisation used only as a starting point for chained updates."""
    current = frame.copy()
    for column in current.columns:
        observed = ~original_missing[column]
        median = current.loc[observed, column].median()
        current.loc[original_missing[column], column] = (
            float(median) if np.isfinite(median) else 0.0
        )
    return current


def _calendar_frame(data):
    out = pd.DataFrame(index=data.index)
    if "DateTime" not in data.columns:
        return out
    dt = pd.to_datetime(data["DateTime"], errors="coerce")
    out["mice_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out["mice_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    out["mice_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    out["mice_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    out["mice_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    out["mice_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    return out.fillna(0.0)


def _mice_complete_predictors(
    data,
    input_columns,
    max_iter=5,
    tol=1e-3,
    random_state=42,
):
    """Complete auxiliary predictors with leakage-safe LightGBM equations."""
    numeric = _numeric_predictors(data, input_columns)
    if numeric.empty:
        return numeric

    original_missing = numeric.isna()
    current = _initialise_predictors(numeric, original_missing)
    calendar = _calendar_frame(data)
    columns_to_impute = [c for c in current if original_missing[c].any()]

    for iteration in range(1, int(max_iter) + 1):
        changes = []
        for position, column in enumerate(columns_to_impute):
            observed = ~original_missing[column]
            missing = original_missing[column]
            if observed.sum() < 30 or not missing.any():
                continue

            other_columns = [c for c in current.columns if c != column]
            X = pd.concat([current[other_columns], calendar], axis=1)
            if X.shape[1] == 0:
                continue

            previous = current.loc[missing, column].to_numpy(dtype=float).copy()
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=250,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=random_state + iteration * 101 + position,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X.loc[observed], numeric.loc[observed, column])
            updated = model.predict(X.loc[missing])
            current.loc[missing, column] = updated
            current.loc[observed, column] = numeric.loc[observed, column]
            changes.append(float(np.mean(np.abs(updated - previous))))

        mean_change = float(np.mean(changes)) if changes else 0.0
        logging.info(
            "[%s] predictor-chain iteration=%d mean_change=%.6f variables=%d",
            MODEL_NAME,
            iteration,
            mean_change,
            len(changes),
        )
        if mean_change < float(tol):
            break

    return current


def impute_mice(
    data,
    target_column,
    input_columns,
    max_iter=5,
    tol=1e-3,
    random_state=42,
    custom_strategies=None,
    **kwargs,
):
    """Complete predictors by chained LightGBM, then impute target by AQUISTIL."""
    original = data.copy()
    original_target = pd.to_numeric(original[target_column], errors="coerce")
    target_missing = original_target.isna()
    if not target_missing.any():
        return original

    completed = _mice_complete_predictors(
        original,
        [column for column in input_columns if column != target_column],
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    working = original.copy()
    for column in completed.columns:
        # Predictor completion is part of the internal model input. The final
        # returned frame restores originally observed predictor values below.
        working[column] = completed[column]
    working[target_column] = original_target

    result = aquistil_impute(
        working,
        target_column,
        input_columns,
        random_state=random_state,
        custom_strategies=custom_strategies,
        **kwargs,
    )
    result.loc[~target_missing, target_column] = original_target.loc[~target_missing]
    for column in completed.columns:
        observed = original[column].notna()
        result.loc[observed, column] = original.loc[observed, column]

    logging.info(
        "[%s] completed predictor chain and imputed %d target rows",
        MODEL_NAME,
        int(target_missing.sum()),
    )
    return result
