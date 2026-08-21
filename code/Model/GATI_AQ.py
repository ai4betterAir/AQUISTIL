"""Leakage-safe gated specialist imputer for air-quality time series.

The gate is trained on individually scored artificial gaps. Candidate models
are run once per batch of equal-length gaps, while every gap receives its own
winner label. Gate features are rebuilt after masking and summarized per gap,
matching the way the gate is used for genuine missing segments.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from Model.AQUISTIL import impute_mice as aquistil_impute
from Model.AQUISTIL_A import impute_mice as aquistil_a_impute
from Model.MICE import impute_mice as mice_impute
from Model.MICE_AQUISTIL import impute_mice as mice_aquistil_impute


MODEL_NAME = "GATI_AQ"
FALLBACK_ID = 3
IMPUTERS = {
    0: ("MICE", mice_impute),
    1: ("AQUISTIL_A", aquistil_a_impute),
    2: ("AQUISTIL", aquistil_impute),
    3: ("MICE_AQUISTIL", mice_aquistil_impute),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _missing_segments(mask):
    """Return positional arrays for every contiguous True segment."""
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [np.arange(start, end + 1) for start, end in zip(starts, ends)]


def _complete_gap_geometry(mask):
    """Return complete gap length and one-based position for missing rows."""
    mask = np.asarray(mask, dtype=bool)
    lengths = np.zeros(mask.size, dtype=int)
    positions = np.zeros(mask.size, dtype=int)
    for segment in _missing_segments(mask):
        lengths[segment] = len(segment)
        positions[segment] = np.arange(1, len(segment) + 1)
    return lengths, positions


def _validation_blocks(df, target, block_len, n_blocks, seed):
    """Select non-overlapping, fully observed contiguous validation blocks."""
    observed = pd.to_numeric(df[target], errors="coerce").notna().to_numpy()
    if block_len < 1 or observed.size < block_len:
        return []
    candidates = [
        start for start in range(observed.size - block_len + 1)
        if observed[start : start + block_len].all()
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    selected = []
    occupied = np.zeros(observed.size, dtype=bool)
    for start in candidates:
        segment = np.arange(start, start + block_len)
        if occupied[segment].any():
            continue
        selected.append(segment)
        occupied[segment] = True
        if len(selected) >= n_blocks:
            break
    return sorted(selected, key=lambda segment: int(segment[0]))


def build_gate_features(df, target):
    """Build features available after the target's missing mask is known.

    Target context excludes the current row. Both past and future observed
    context may be used because this pipeline performs offline imputation.
    """
    if "DateTime" not in df.columns:
        raise ValueError("GATI_AQ requires a 'DateTime' column")

    y = pd.to_numeric(df[target], errors="coerce")
    missing = y.isna().to_numpy()
    lengths, positions = _complete_gap_geometry(missing)
    dt = pd.to_datetime(df["DateTime"], errors="coerce")
    features = pd.DataFrame(index=df.index)
    features["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    features["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    features["gap_length"] = lengths
    features["gap_position_fraction"] = np.divide(
        positions, lengths, out=np.zeros_like(lengths, dtype=float), where=lengths > 0
    )

    # All rolling windows are shifted so the row's true target can never enter
    # its own gate features. Reversing supplies the equivalent future context.
    past = y.shift(1)
    future = y.shift(-1).iloc[::-1]
    for window, minimum in ((6, 2), (24, 4), (72, 8)):
        features[f"past_mean_{window}"] = past.rolling(window, min_periods=minimum).mean()
        features[f"past_std_{window}"] = past.rolling(window, min_periods=minimum).std()
        features[f"future_mean_{window}"] = (
            future.rolling(window, min_periods=minimum).mean().iloc[::-1]
        )
        features[f"future_std_{window}"] = (
            future.rolling(window, min_periods=minimum).std().iloc[::-1]
        )

    observed = y.notna().to_numpy()
    index = np.arange(len(df))
    previous = pd.Series(np.where(observed, index, np.nan)).ffill().to_numpy()
    following = pd.Series(np.where(observed, index, np.nan)).bfill().to_numpy()
    features["distance_previous"] = index - previous
    features["distance_next"] = following - index
    features["previous_value"] = y.ffill()
    features["next_value"] = y.bfill()
    features["context_slope"] = (
        features["next_value"] - features["previous_value"]
    ) / (features["distance_previous"] + features["distance_next"]).replace(0, np.nan)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _segment_record(features, segment):
    """Summarize one gap into one gate-training or prediction record."""
    block = features.iloc[segment]
    record = block.mean(axis=0)
    record["gap_start_hour_sin"] = block["hour_sin"].iloc[0]
    record["gap_start_hour_cos"] = block["hour_cos"].iloc[0]
    return record


def _run_candidate(fn, frame, target, input_columns, custom_strategies, kwargs):
    return fn(
        frame.copy(), target, input_columns,
        custom_strategies=custom_strategies, **kwargs
    )


def _training_cases(
    df, target, input_columns, block_sizes, blocks_per_size,
    custom_strategies, kwargs, random_state,
):
    """Create independently scored, leakage-safe block-level gate cases."""
    records, labels, errors = [], [], []
    for size_index, block_len in enumerate(block_sizes):
        blocks = _validation_blocks(
            df, target, int(block_len), int(blocks_per_size),
            random_state + size_index * 1009,
        )
        if not blocks:
            continue
        combined = np.concatenate(blocks)
        masked = df.copy()
        truth = pd.to_numeric(df[target], errors="coerce")
        masked.iloc[combined, masked.columns.get_loc(target)] = np.nan
        gate_features = build_gate_features(masked, target)

        predictions = {}
        for model_id, (name, fn) in IMPUTERS.items():
            try:
                output = _run_candidate(
                    fn, masked, target, input_columns, custom_strategies, kwargs
                )
                predictions[model_id] = pd.to_numeric(output[target], errors="coerce")
            except Exception as exc:
                logging.warning("[%s] validation candidate %s failed: %s", MODEL_NAME, name, exc)
                predictions[model_id] = pd.Series(np.nan, index=df.index)

        for segment in blocks:
            model_errors = {}
            actual = truth.iloc[segment].to_numpy(dtype=float)
            for model_id, prediction in predictions.items():
                predicted = prediction.iloc[segment].to_numpy(dtype=float)
                valid = np.isfinite(actual) & np.isfinite(predicted)
                model_errors[model_id] = (
                    float(np.sqrt(np.mean((actual[valid] - predicted[valid]) ** 2)))
                    if valid.any() else np.inf
                )
            winner = min(model_errors, key=model_errors.get)
            if not np.isfinite(model_errors[winner]):
                continue
            records.append(_segment_record(gate_features, segment))
            labels.append(winner)
            errors.append(model_errors)
    return records, np.asarray(labels, dtype=int), errors


def impute_mice(
    data, target_column, input_columns, custom_strategies=None, **kwargs
):
    """Impute each genuine gap with the gate-selected specialist."""
    df = data.copy()
    target = target_column
    if "DateTime" not in df.columns:
        raise ValueError("GATI_AQ requires a 'DateTime' column")
    missing = pd.to_numeric(df[target], errors="coerce").isna().to_numpy()
    if not missing.any() or not (~missing).any():
        return df if not missing.any() else _run_candidate(
            mice_aquistil_impute, df, target, input_columns, custom_strategies, kwargs
        )

    random_state = int(kwargs.get("random_state", 42))
    block_sizes = kwargs.pop("gati_block_sizes", (6, 12, 24, 72))
    blocks_per_size = int(kwargs.pop("gati_blocks_per_size", 8))
    confidence_threshold = float(kwargs.pop("gati_confidence_threshold", 0.60))
    records, labels, validation_errors = _training_cases(
        df, target, input_columns, block_sizes, blocks_per_size,
        custom_strategies, kwargs, random_state,
    )
    if not records:
        logging.warning("[%s] no valid gate cases; using MICE_AQUISTIL", MODEL_NAME)
        return _run_candidate(
            mice_aquistil_impute, df, target, input_columns, custom_strategies, kwargs
        )

    X_train = pd.DataFrame(records).fillna(0.0)
    mean_error = {
        model_id: float(np.mean([case[model_id] for case in validation_errors]))
        for model_id in IMPUTERS
    }
    global_best = min(mean_error, key=mean_error.get)
    classes = np.unique(labels)
    gate = None
    if classes.size > 1:
        gate = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.04, num_leaves=15,
            min_child_samples=5, reg_lambda=1.0, random_state=random_state,
            n_jobs=-1, verbosity=-1,
        )
        gate.fit(X_train, labels)

    features = build_gate_features(df, target)
    segments = _missing_segments(missing)
    X_predict = pd.DataFrame(
        [_segment_record(features, segment) for segment in segments]
    ).reindex(columns=X_train.columns, fill_value=0.0)
    if gate is None:
        choices = np.full(len(segments), global_best, dtype=int)
        confidence = np.ones(len(segments), dtype=float)
    else:
        probabilities = gate.predict_proba(X_predict)
        positions = np.argmax(probabilities, axis=1)
        choices = gate.classes_[positions].astype(int)
        confidence = probabilities[np.arange(len(segments)), positions]
        choices[confidence < confidence_threshold] = FALLBACK_ID

    required = set(int(choice) for choice in choices)
    outputs = {}
    for model_id in required:
        outputs[model_id] = _run_candidate(
            IMPUTERS[model_id][1], df, target, input_columns,
            custom_strategies, kwargs,
        )
    result = df.copy()
    fallback_output = None
    usage = {}
    for segment, model_id in zip(segments, choices):
        name = IMPUTERS[int(model_id)][0]
        usage[name] = usage.get(name, 0) + 1
        values = pd.to_numeric(outputs[int(model_id)][target], errors="coerce").iloc[segment]
        if values.isna().any():
            if fallback_output is None:
                fallback_output = _run_candidate(
                    mice_aquistil_impute, df, target, input_columns,
                    custom_strategies, kwargs,
                )
            values = values.fillna(pd.to_numeric(fallback_output[target], errors="coerce").iloc[segment])
        result.iloc[segment, result.columns.get_loc(target)] = values.to_numpy()

    # Preserve all originally observed target values exactly.
    result.loc[~missing, target] = df.loc[~missing, target]
    logging.info(
        "[%s] cases=%d classes=%s global_best=%s usage=%s mean_confidence=%.3f",
        MODEL_NAME, len(labels), classes.tolist(), IMPUTERS[global_best][0], usage,
        float(np.mean(confidence)),
    )
    return result
