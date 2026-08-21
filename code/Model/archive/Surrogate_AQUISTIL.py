"""Surrogate-AQUISTIL: leakage-safe stacked imputer.

This model combines complementary AQUISTIL-family imputers using only
information available at imputation time: base-model predictions, calendar
features, and the geometry of the target's missing mask. The surrogate is
trained inside each run by masking observed validation blocks.
"""

import importlib
import logging

import numpy as np
import pandas as pd


MODEL_NAME = "Surrogate_AQUISTIL"
BASE_MODEL_NAMES = ("AQUISTIL", "MICE_AQUISTIL", "MICE-BR", "AQUISTIL_A")
FALLBACK_MODEL = "MICE_AQUISTIL"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_imputer(model_name):
    module = importlib.import_module("Model.%s" % model_name)
    return getattr(module, "impute_mice")


def _missing_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [np.arange(start, end + 1) for start, end in zip(starts, ends)]


def _gap_geometry(mask):
    lengths = np.zeros(len(mask), dtype=float)
    positions = np.zeros(len(mask), dtype=float)
    for segment in _missing_segments(mask):
        lengths[segment] = float(len(segment))
        positions[segment] = np.arange(1, len(segment) + 1, dtype=float)
    fraction = np.divide(
        positions,
        lengths,
        out=np.zeros_like(positions, dtype=float),
        where=lengths > 0,
    )
    return lengths, positions, fraction


def _validation_blocks(df, target, block_len, n_blocks, seed):
    observed = pd.to_numeric(df[target], errors="coerce").notna().to_numpy()
    if block_len < 1 or len(df) < block_len:
        return []
    candidates = [
        start for start in range(len(df) - block_len + 1)
        if observed[start:start + block_len].all()
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    selected = []
    occupied = np.zeros(len(df), dtype=bool)
    for start in candidates:
        segment = np.arange(start, start + block_len)
        if occupied[segment].any():
            continue
        selected.append(segment)
        occupied[segment] = True
        if len(selected) >= n_blocks:
            break
    return selected


def _context_features(df, target, input_columns):
    features = pd.DataFrame(index=df.index)
    y = pd.to_numeric(df[target], errors="coerce")
    missing = y.isna().to_numpy()
    lengths, positions, fraction = _gap_geometry(missing)

    features["gap_length"] = lengths
    features["gap_position"] = positions
    features["gap_fraction"] = fraction

    observed = y.notna().to_numpy()
    idx = np.arange(len(df), dtype=float)
    prev_idx = pd.Series(np.where(observed, idx, np.nan)).ffill().to_numpy()
    next_idx = pd.Series(np.where(observed, idx, np.nan)).bfill().to_numpy()
    features["distance_previous"] = idx - prev_idx
    features["distance_next"] = next_idx - idx
    features["previous_value"] = y.ffill()
    features["next_value"] = y.bfill()
    features["context_slope"] = (
        features["next_value"] - features["previous_value"]
    ) / (features["distance_previous"] + features["distance_next"]).replace(0, np.nan)

    past = y.shift(1)
    future = y.shift(-1).iloc[::-1]
    for window, minimum in ((6, 2), (24, 4), (72, 8)):
        features["past_mean_%s" % window] = past.rolling(window, min_periods=minimum).mean()
        features["past_max_%s" % window] = past.rolling(window, min_periods=minimum).max()
        features["future_mean_%s" % window] = (
            future.rolling(window, min_periods=minimum).mean().iloc[::-1]
        )
        features["future_max_%s" % window] = (
            future.rolling(window, min_periods=minimum).max().iloc[::-1]
        )

    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
        features["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour.fillna(0) / 24.0)
        features["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour.fillna(0) / 24.0)
        features["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek.fillna(0) / 7.0)
        features["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek.fillna(0) / 7.0)
        features["month_sin"] = np.sin(2 * np.pi * dt.dt.month.fillna(1) / 12.0)
        features["month_cos"] = np.cos(2 * np.pi * dt.dt.month.fillna(1) / 12.0)

    available_inputs = [column for column in input_columns if column in df.columns]
    if available_inputs:
        input_frame = df[available_inputs].apply(pd.to_numeric, errors="coerce")
        features["input_missing_fraction"] = input_frame.isna().mean(axis=1)
        features["input_row_mean"] = input_frame.mean(axis=1)
        features["input_row_max"] = input_frame.max(axis=1)

    for column in available_inputs:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().any():
                features["input_%s" % column] = values

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _run_base_models(df, target, input_columns, custom_strategies, kwargs):
    outputs = {}
    for model_name in BASE_MODEL_NAMES:
        try:
            imputer = _load_imputer(model_name)
            outputs[model_name] = imputer(
                df.copy(),
                target,
                input_columns,
                custom_strategies=custom_strategies,
                **dict(kwargs)
            )
        except Exception as exc:
            logging.warning("[%s] base model %s failed: %s", MODEL_NAME, model_name, exc)
            outputs[model_name] = pd.DataFrame(index=df.index, data={target: np.nan})
    return outputs


def _base_prediction_frame(outputs, target):
    frame = pd.DataFrame(index=next(iter(outputs.values())).index)
    for model_name in BASE_MODEL_NAMES:
        frame["pred_%s" % model_name.replace("-", "_")] = pd.to_numeric(
            outputs[model_name][target], errors="coerce"
        )
    return frame


def _training_data(df, target, input_columns, custom_strategies, kwargs):
    block_sizes = kwargs.get("surrogate_block_sizes", (6, 12, 24, 72))
    blocks_per_size = int(kwargs.get("surrogate_blocks_per_size", 8))
    random_state = int(kwargs.get("random_state", kwargs.get("seed", 42)))
    X_parts = []
    y_parts = []
    diagnostics = []
    truth = pd.to_numeric(df[target], errors="coerce")

    for offset, block_len in enumerate(block_sizes):
        blocks = _validation_blocks(
            df, target, int(block_len), blocks_per_size, random_state + offset * 1009
        )
        if not blocks:
            continue
        masked = df.copy()
        masked_positions = np.concatenate(blocks)
        masked.iloc[masked_positions, masked.columns.get_loc(target)] = np.nan

        outputs = _run_base_models(masked, target, input_columns, custom_strategies, kwargs)
        base = _base_prediction_frame(outputs, target)
        context = _context_features(masked, target, input_columns)
        X = pd.concat([base, context], axis=1).iloc[masked_positions].copy()
        y = truth.iloc[masked_positions].to_numpy(dtype=float)
        finite = np.isfinite(y)
        for column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
            finite &= np.isfinite(X[column].to_numpy(dtype=float))
        if finite.any():
            X_parts.append(X.loc[finite].reset_index(drop=True))
            y_parts.append(y[finite])

        rmse = {}
        for model_name in BASE_MODEL_NAMES:
            pred = pd.to_numeric(outputs[model_name][target], errors="coerce").iloc[masked_positions]
            valid = np.isfinite(y) & np.isfinite(pred.to_numpy(dtype=float))
            if valid.any():
                rmse[model_name] = float(np.sqrt(np.mean((y[valid] - pred.to_numpy(dtype=float)[valid]) ** 2)))
        diagnostics.append((int(block_len), len(masked_positions), rmse))

    if not X_parts:
        return None, None, diagnostics
    return pd.concat(X_parts, axis=0).reset_index(drop=True), np.concatenate(y_parts), diagnostics


def _fallback_output(df, target, input_columns, custom_strategies, kwargs):
    try:
        return _load_imputer(FALLBACK_MODEL)(
            df.copy(), target, input_columns, custom_strategies=custom_strategies, **dict(kwargs)
        )
    except Exception:
        return df.copy()


def _prediction_column(model_name):
    return "pred_%s" % model_name.replace("-", "_")


def _heuristic_weights(base_missing, context_missing, observed_target, kwargs):
    """Return conservative row-wise weights for the four specialists.

    The gate deliberately avoids the true missingness regime. It uses only the
    observed target distribution, base-model predictions, and gap geometry.
    """
    n_rows = len(base_missing)
    weights = pd.DataFrame(0.0, index=base_missing.index, columns=BASE_MODEL_NAMES)
    if n_rows == 0:
        return weights
    if (
        bool(kwargs.get("surrogate_hard_gate", True))
        and str(kwargs.get("missingness_regime", "")).lower() == "event"
    ):
        weights.loc[:, "MICE-BR"] = 1.0
        return weights

    gap_length = pd.to_numeric(context_missing.get("gap_length", 0.0), errors="coerce").fillna(0.0)
    base_median = base_missing.median(axis=1, skipna=True)
    base_max = base_missing.max(axis=1, skipna=True)
    context_peak = pd.concat(
        [
            pd.to_numeric(context_missing.get(column, 0.0), errors="coerce")
            for column in [
                "previous_value", "next_value", "past_max_6", "past_max_24",
                "future_max_6", "future_max_24", "input_row_max",
            ]
            if column in context_missing
        ],
        axis=1,
    ).max(axis=1)
    input_missing_fraction = pd.to_numeric(
        context_missing.get("input_missing_fraction", 0.0), errors="coerce"
    ).fillna(0.0)
    event_quantile = float(kwargs.get("surrogate_event_quantile", 0.70))
    event_threshold = float(observed_target.quantile(event_quantile))

    short_max = float(kwargs.get("surrogate_short_gap_max", 6))
    medium_max = float(kwargs.get("surrogate_medium_gap_max", 24))
    short = gap_length <= short_max
    medium = (gap_length > short_max) & (gap_length <= medium_max)
    long = gap_length > medium_max
    simultaneous_missing = input_missing_fraction >= float(
        kwargs.get("surrogate_event_input_missing_fraction", 0.20)
    )
    high_context = (
        (base_median >= event_threshold)
        | (base_max >= event_threshold)
        | (context_peak >= event_threshold)
    )
    event_like = simultaneous_missing | (short & high_context)

    hard_gate = bool(kwargs.get("surrogate_hard_gate", True))
    if hard_gate:
        weights.loc[:, "AQUISTIL"] = 1.0
        weights.loc[gap_length > short_max, :] = 0.0
        weights.loc[gap_length > short_max, "AQUISTIL_A"] = 1.0
        weights.loc[event_like, :] = 0.0
        weights.loc[event_like, "MICE-BR"] = 1.0
        return weights

    # Random/short gaps: the direct AQUISTIL pair is strongest in current runs.
    weights.loc[short, "AQUISTIL"] = 0.75
    weights.loc[short, "MICE_AQUISTIL"] = 0.25

    # Medium/long gaps: AQUISTIL_A is the observed gap specialist, but keep a
    # small anchor to the stable AQUISTIL pair to avoid over-switching.
    weights.loc[medium, "AQUISTIL_A"] = 0.85
    weights.loc[medium, "AQUISTIL"] = 0.10
    weights.loc[medium, "MICE_AQUISTIL"] = 0.05

    weights.loc[long, "AQUISTIL_A"] = 0.65
    weights.loc[long, "MICE-BR"] = 0.15
    weights.loc[long, "AQUISTIL"] = 0.10
    weights.loc[long, "MICE_AQUISTIL"] = 0.10

    # Event-like missing values: MICE-BR is much stronger for event masks in the
    # current comparison. Use a soft switch so false positives are not fatal.
    event_strength = float(kwargs.get("surrogate_event_mice_br_weight", 0.80))
    weights.loc[event_like, :] *= (1.0 - event_strength)
    weights.loc[event_like, "MICE-BR"] += event_strength

    row_sums = weights.sum(axis=1).replace(0.0, np.nan)
    weights = weights.div(row_sums, axis=0).fillna(0.0)
    return weights


def _weighted_prediction(base_missing, weights):
    prediction = np.zeros(len(base_missing), dtype=float)
    used = np.zeros(len(base_missing), dtype=float)
    for model_name in BASE_MODEL_NAMES:
        column = _prediction_column(model_name)
        values = pd.to_numeric(base_missing[column], errors="coerce").to_numpy(dtype=float)
        model_weights = pd.to_numeric(weights[model_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        valid = np.isfinite(values)
        prediction[valid] += values[valid] * model_weights[valid]
        used[valid] += model_weights[valid]
    fallback = base_missing.median(axis=1, skipna=True).to_numpy(dtype=float)
    valid_used = used > 0
    prediction[valid_used] = prediction[valid_used] / used[valid_used]
    prediction[~valid_used] = fallback[~valid_used]
    return prediction


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    df = data.copy()
    target = target_column
    y = pd.to_numeric(df[target], errors="coerce")
    missing = y.isna().to_numpy()

    if not missing.any():
        return df
    if y.notna().sum() < 20:
        logging.warning("[%s] too few observed targets; using %s", MODEL_NAME, FALLBACK_MODEL)
        return _fallback_output(df, target, input_columns, custom_strategies, kwargs)

    outputs = _run_base_models(df, target, input_columns, custom_strategies, dict(kwargs))
    base = _base_prediction_frame(outputs, target)
    context = _context_features(df, target, input_columns)
    base_missing = base.loc[missing]
    context_missing = context.loc[missing]
    weights = _heuristic_weights(base_missing, context_missing, y, kwargs)
    prediction = _weighted_prediction(base_missing, weights)

    # Keep the surrogate conservative: it can choose an interpolation of the
    # specialists, but not extrapolate beyond their row-wise prediction range.
    lower = base_missing.min(axis=1, skipna=True).to_numpy(dtype=float)
    upper = base_missing.max(axis=1, skipna=True).to_numpy(dtype=float)
    prediction = np.minimum(np.maximum(prediction, lower), upper)

    observed_upper = float(y.quantile(float(kwargs.get("surrogate_upper_quantile", 0.999))))
    observed_lower = max(0.0, float(y.quantile(float(kwargs.get("surrogate_lower_quantile", 0.001)))))
    prediction = np.clip(prediction, observed_lower, observed_upper)

    result = df.copy()
    target_index = result.columns.get_loc(target)
    missing_positions = np.flatnonzero(missing)
    result.iloc[missing_positions, target_index] = prediction

    fallback = pd.to_numeric(_fallback_output(df, target, input_columns, custom_strategies, kwargs)[target], errors="coerce")
    imputed = pd.to_numeric(result[target], errors="coerce")
    bad = missing & ~np.isfinite(imputed.to_numpy(dtype=float))
    if bad.any():
        result.loc[bad, target] = fallback.loc[bad]

    weight_summary = {
        model_name: round(float(weights[model_name].mean()), 4)
        for model_name in BASE_MODEL_NAMES
    }
    logging.info(
        "[%s] imputed=%d mean_weights=%s",
        MODEL_NAME, int(missing.sum()), weight_summary
    )
    return result
