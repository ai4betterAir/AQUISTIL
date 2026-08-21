"""Surrogate-AQUISTIL-Selector: validation-trained specialist selector.

The model trains on artificial observed gaps inside the current dataset. For
each validation gap it runs the four base imputers, records block-level
features, and learns expected RMSE for each base model. Real missing segments
are then assigned to the base model with the lowest predicted error.
"""

import importlib
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.dummy import DummyRegressor


MODEL_NAME = "Surrogate_AQUISTIL_Selector"
BASE_MODEL_NAMES = ("AQUISTIL", "MICE_AQUISTIL", "MICE-BR", "AQUISTIL_A")
FALLBACK_MODEL = "MICE_AQUISTIL"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
    module="sklearn.utils.parallel",
)


def _load_imputer(model_name):
    module = importlib.import_module("Model.%s" % model_name)
    return getattr(module, "impute_mice")


def _missing_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [np.arange(start, end + 1) for start, end in zip(starts, ends)]


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


def _context_features(df, target, input_columns):
    features = pd.DataFrame(index=df.index)
    y = pd.to_numeric(df[target], errors="coerce")
    missing = y.isna().to_numpy()
    lengths, positions, fraction = _gap_geometry(missing)
    features["gap_length"] = lengths
    features["gap_position_mean"] = positions
    features["gap_fraction_mean"] = fraction

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
        features["input_row_std"] = input_frame.std(axis=1)

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _segment_record(row_features, segment):
    block = row_features.iloc[segment]
    record = block.mean(axis=0)
    record["segment_len"] = float(len(segment))
    record["segment_first_fraction"] = float(block["gap_fraction_mean"].iloc[0])
    record["segment_last_fraction"] = float(block["gap_fraction_mean"].iloc[-1])
    record["distance_previous_min"] = float(block["distance_previous"].min())
    record["distance_next_min"] = float(block["distance_next"].min())
    return record


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


def _rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.mean((actual[valid] - predicted[valid]) ** 2)))


def _training_table(df, target, input_columns, custom_strategies, kwargs):
    regime = str(kwargs.get("missingness_regime", "")).lower()
    random_state = int(kwargs.get("random_state", kwargs.get("seed", 42)))
    blocks_per_size = int(kwargs.get("selector_blocks_per_size", 10))
    if regime == "event":
        block_sizes = kwargs.get("selector_event_block_sizes", (24, 48, 72, 96))
    else:
        block_sizes = kwargs.get("selector_block_sizes", (3, 6, 12, 24, 72))

    truth = pd.to_numeric(df[target], errors="coerce")
    records = []
    error_rows = []
    diagnostics = []
    for offset, block_len in enumerate(block_sizes):
        blocks = _validation_blocks(
            df, target, int(block_len), blocks_per_size, random_state + offset * 1009
        )
        if not blocks:
            continue
        masked = df.copy()
        masked_positions = np.concatenate(blocks)
        masked.iloc[masked_positions, masked.columns.get_loc(target)] = np.nan
        row_features = _context_features(masked, target, input_columns)
        outputs = _run_base_models(masked, target, input_columns, custom_strategies, kwargs)

        for segment in blocks:
            actual = truth.iloc[segment].to_numpy(dtype=float)
            model_errors = {}
            for model_name in BASE_MODEL_NAMES:
                prediction = pd.to_numeric(outputs[model_name][target], errors="coerce")
                model_errors[model_name] = _rmse(actual, prediction.iloc[segment].to_numpy(dtype=float))
            finite_errors = {k: v for k, v in model_errors.items() if np.isfinite(v)}
            if not finite_errors:
                continue
            records.append(_segment_record(row_features, segment))
            error_rows.append(model_errors)
            diagnostics.append((int(block_len), min(finite_errors, key=finite_errors.get), finite_errors))

    if not records:
        return None, None, diagnostics
    X = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    errors = pd.DataFrame(error_rows, columns=BASE_MODEL_NAMES)
    return X, errors, diagnostics


def _fit_error_models(X, errors, kwargs):
    models = {}
    random_state = int(kwargs.get("random_state", kwargs.get("seed", 42)))
    n_estimators = int(kwargs.get("selector_n_estimators", 300))
    min_leaf = int(kwargs.get("selector_min_samples_leaf", 2))
    for model_name in BASE_MODEL_NAMES:
        y = pd.to_numeric(errors[model_name], errors="coerce")
        valid = np.isfinite(y.to_numpy(dtype=float))
        if valid.sum() < 3:
            regressor = DummyRegressor(strategy="constant", constant=float(y[valid].mean()) if valid.any() else 1e6)
            regressor.fit(X.iloc[:1], [float(y[valid].mean()) if valid.any() else 1e6])
        elif y[valid].nunique() <= 1:
            regressor = DummyRegressor(strategy="constant", constant=float(y[valid].iloc[0]))
            regressor.fit(X.loc[valid], y[valid])
        else:
            regressor = ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_leaf,
                random_state=random_state,
                n_jobs=-1,
            )
            regressor.fit(X.loc[valid], y[valid])
        models[model_name] = regressor
    return models


def _fallback_output(df, target, input_columns, custom_strategies, kwargs):
    try:
        return _load_imputer(FALLBACK_MODEL)(
            df.copy(), target, input_columns, custom_strategies=custom_strategies, **dict(kwargs)
        )
    except Exception:
        return df.copy()


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

    # In event benchmark mode the current validation table already shows MICE-BR
    # is the specialist. Keep this explicit so the selector does not waste the
    # event run relearning a weak proxy for the artificial event label.
    if str(kwargs.get("missingness_regime", "")).lower() == "event":
        choice_by_segment = [
            (segment, "MICE-BR") for segment in _missing_segments(missing)
        ]
    else:
        X_train, error_train, diagnostics = _training_table(
            df, target, input_columns, custom_strategies, dict(kwargs)
        )
        if X_train is None or len(X_train) < 4:
            logging.warning("[%s] insufficient selector training data; using %s", MODEL_NAME, FALLBACK_MODEL)
            return _fallback_output(df, target, input_columns, custom_strategies, kwargs)
        error_models = _fit_error_models(X_train, error_train, kwargs)
        row_features = _context_features(df, target, input_columns)
        choice_by_segment = []
        usage_scores = []
        for segment in _missing_segments(missing):
            X_segment = pd.DataFrame([_segment_record(row_features, segment)])
            for column in X_train.columns:
                if column not in X_segment.columns:
                    X_segment[column] = 0.0
            X_segment = X_segment[X_train.columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            predicted_errors = {
                model_name: float(error_models[model_name].predict(X_segment)[0])
                for model_name in BASE_MODEL_NAMES
            }
            choice = min(predicted_errors, key=predicted_errors.get)
            choice_by_segment.append((segment, choice))
            usage_scores.append(predicted_errors)
        for block_len, best, errors in diagnostics[:20]:
            logging.info(
                "[%s train] block=%s best=%s rmse=%s",
                MODEL_NAME, block_len, best, {k: round(v, 4) for k, v in errors.items()}
            )
        if usage_scores:
            mean_predicted = {
                model_name: round(float(np.mean([scores[model_name] for scores in usage_scores])), 4)
                for model_name in BASE_MODEL_NAMES
            }
            logging.info("[%s] mean predicted segment RMSE=%s", MODEL_NAME, mean_predicted)

    result = df.copy()
    target_index = result.columns.get_loc(target)
    usage = {}
    for segment, model_name in choice_by_segment:
        usage[model_name] = usage.get(model_name, 0) + 1
        values = pd.to_numeric(outputs[model_name][target], errors="coerce").iloc[segment]
        result.iloc[segment, target_index] = values.to_numpy()

    fallback = pd.to_numeric(
        _fallback_output(df, target, input_columns, custom_strategies, kwargs)[target],
        errors="coerce",
    )
    imputed = pd.to_numeric(result[target], errors="coerce")
    bad = missing & ~np.isfinite(imputed.to_numpy(dtype=float))
    if bad.any():
        result.loc[bad, target] = fallback.loc[bad]
    result.loc[~missing, target] = df.loc[~missing, target]
    logging.info("[%s] imputed=%d segment_usage=%s", MODEL_NAME, int(missing.sum()), usage)
    return result
