# code/Model/AQUISTIL.py
"""AQUISTIL vNext: LightGBM-first spatial-temporal imputation.

Design goals:
- keep the public `impute_mice(...)` interface
- preserve original observations and original missing mask
- avoid self-fed target-history leakage on originally missing rows
- use one strong LightGBM backbone
- keep AQUISTIL identity through spatial-temporal, gap-aware, and event-aware features
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from spatial import prepare_spatial_temporal_data

MODEL_NAME = "AQUISTIL"
GAP_BANDS = ((1, 6), (7, 24), (25, 72), (73, 10**9))
GAP_NAMES = ("short", "medium", "long", "extreme")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class EventModels:
    classifier: lgb.LGBMClassifier | None
    regressor: lgb.LGBMRegressor | None
    threshold: float
    probability_threshold: float
    max_blend: float
    cap: float


def _gap_lengths(mask: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros(mask.size, dtype=int)
    groups = np.zeros(mask.size, dtype=int) if groups is None else np.asarray(groups)
    for group in pd.unique(groups):
        positions = np.flatnonzero(groups == group)
        local = mask[positions]
        starts = np.flatnonzero(local & ~np.r_[False, local[:-1]])
        ends = np.flatnonzero(local & ~np.r_[local[1:], False])
        for start, end in zip(starts, ends):
            result[positions[start : end + 1]] = end - start + 1
    return result


def _gap_band(lengths: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths)
    out = np.zeros(lengths.size, dtype=int)
    out[(lengths >= 7) & (lengths <= 24)] = 1
    out[(lengths >= 25) & (lengths <= 72)] = 2
    out[lengths >= 73] = 3
    return out


def _site_token(column: str) -> str:
    text = re.sub(r"^spatial_", "", str(column), flags=re.I)
    text = re.sub(r"^idw_spatial_", "", text, flags=re.I)
    return re.split(r"[_:]", text)[-1]


def _distance_km(a: dict, b: dict) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
    return 12742.0 * asin(
        sqrt(
            sin((lat2 - lat1) / 2) ** 2
            + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
        )
    )


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _causal_group_ffill(frame: pd.DataFrame, groups: np.ndarray | None = None) -> pd.DataFrame:
    out = frame.copy()
    if groups is None:
        return out.ffill()
    if len(groups) != len(out):
        raise ValueError(
            f"Group labels length ({len(groups)}) does not match feature rows ({len(out)})"
        )
    group_series = pd.Series(np.asarray(groups), index=out.index)
    return out.groupby(group_series, sort=False).ffill()


def _clean_features(
    frame: pd.DataFrame,
    train_mask: pd.Series | None = None,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    out = frame.apply(_safe_numeric)
    out = _causal_group_ffill(out, groups=groups)
    reference = out.loc[train_mask] if train_mask is not None else out
    medians = reference.median(numeric_only=True).fillna(0.0)
    return out.fillna(medians).fillna(0.0)


def _adaptive_spatial_aggregate(
    df: pd.DataFrame,
    target: str,
    feature_columns: Iterable[str],
    observed_mask: np.ndarray,
    site_name: str,
    config,
) -> tuple[pd.Series, list[tuple[str, float, float, float]]]:
    spatial_columns = [
        c
        for c in feature_columns
        if c in df.columns
        and (str(c).lower().startswith("spatial_") or str(c).lower().startswith("idw_spatial_"))
    ]
    if not spatial_columns:
        return pd.Series(np.nan, index=df.index, name="adaptive_spatial"), []

    coords = getattr(config, "SITE_COORDINATES", {}) or {}
    coord_map = {re.sub(r"[^A-Za-z0-9]", "", str(k)).upper(): v for k, v in coords.items()}
    target_coord = coord_map.get(re.sub(r"[^A-Za-z0-9]", "", str(site_name)).upper())

    y = _safe_numeric(df[target])
    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    inventory: list[tuple[str, float, float, float]] = []

    for col in spatial_columns:
        values = _safe_numeric(df[col])
        valid_train = observed_mask & values.notna().to_numpy()
        corr = abs(float(y.loc[valid_train].corr(values.loc[valid_train]))) if valid_train.sum() >= 12 else 0.0
        if not np.isfinite(corr):
            corr = 0.0

        distance = 1.0
        neighbor_coord = coord_map.get(re.sub(r"[^A-Za-z0-9]", "", _site_token(col)).upper())
        if target_coord and neighbor_coord:
            distance = max(_distance_km(target_coord, neighbor_coord), 1.0)

        weight = max(corr, 0.05) / distance
        valid = values.notna()
        numerator.loc[valid] += weight * values.loc[valid]
        denominator.loc[valid] += weight
        inventory.append((col, corr, distance, weight))

    adaptive = numerator.div(denominator.where(denominator > 0)).rename("adaptive_spatial")
    return adaptive, inventory


def _observed_history_features(
    series: pd.Series,
    observed_mask: np.ndarray,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    y = _safe_numeric(series).copy()
    y.loc[~observed_mask] = np.nan

    group_series = pd.Series(
        np.zeros(len(y), dtype=int) if groups is None else np.asarray(groups),
        index=y.index,
    )

    history = pd.DataFrame(index=y.index)

    for lag in (1, 3, 6, 12, 24, 48, 72):
        lagged = y.groupby(group_series, sort=False).shift(lag)
        history[f"lag_{lag}"] = lagged
        history[f"lag_{lag}_available"] = lagged.notna().astype(float)

    past = y.groupby(group_series, sort=False).shift(1)

    for window, minimum in ((6, 2), (12, 3), (24, 6), (72, 12)):
        grouped = past.groupby(group_series, sort=False)
        history[f"roll_mean_{window}"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=minimum).mean()
        )
        history[f"roll_std_{window}"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=minimum).std()
        )
        history[f"roll_max_{window}"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=minimum).max()
        )
        history[f"roll_min_{window}"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=minimum).min()
        )

    history["diff_1"] = history["lag_1"] - history["lag_3"]
    history["diff_3"] = history["lag_3"] - history["lag_6"]
    history["diff_6"] = history["lag_6"] - history["lag_12"]
    history["accel_1"] = history["diff_1"] - history["diff_1"].shift(1)

    for window in (24, 72):
        mean_col = f"roll_mean_{window}"
        std_col = f"roll_std_{window}"
        history[f"zscore_{window}"] = (
            history["lag_1"] - history[mean_col]
        ) / (history[std_col].replace(0.0, np.nan))

    q80 = y.loc[observed_mask].quantile(0.80) if observed_mask.sum() else np.nan
    q90 = y.loc[observed_mask].quantile(0.90) if observed_mask.sum() else np.nan
    elevated_80 = (history["lag_1"] >= q80).astype(float) if np.isfinite(q80) else pd.Series(0.0, index=y.index)
    elevated_90 = (history["lag_1"] >= q90).astype(float) if np.isfinite(q90) else pd.Series(0.0, index=y.index)

    grouped80 = elevated_80.groupby(group_series, sort=False)
    grouped90 = elevated_90.groupby(group_series, sort=False)
    history["hours_above_q80_last_24"] = grouped80.transform(
        lambda values: values.rolling(24, min_periods=1).sum()
    )
    history["hours_above_q90_last_24"] = grouped90.transform(
        lambda values: values.rolling(24, min_periods=1).sum()
    )

    return history


def _calendar_features(dt: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=dt.index)
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["dayofyear"] = dt.dt.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 366)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 366)
    return out


def _site_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "Site" in df.columns:
        codes, _ = pd.factorize(df["Site"].astype(str), sort=True)
        out["site_code"] = codes.astype(float)
    if "Region" in df.columns:
        codes, _ = pd.factorize(df["Region"].astype(str), sort=True)
        out["region_code"] = codes.astype(float)
    return out


def _gap_features(
    original_missing: np.ndarray,
    groups: np.ndarray | None = None,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    gap_length = _gap_lengths(original_missing, groups=groups)
    gap_band = _gap_band(gap_length)
    if index is not None and len(index) != len(original_missing):
        raise ValueError(
            f"Gap feature index length ({len(index)}) does not match mask length ({len(original_missing)})"
        )
    out = pd.DataFrame(
        {
            "gap_length": gap_length.astype(float),
            "gap_band": gap_band.astype(float),
            "is_missing_original": original_missing.astype(float),
        },
        index=index,
    )
    for band_index, band_name in enumerate(GAP_NAMES):
        out[f"gap_is_{band_name}"] = (gap_band == band_index).astype(float)
    return out


def _event_score_features(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)
    z24 = _safe_numeric(features.get("zscore_24", pd.Series(np.nan, index=features.index)))
    z72 = _safe_numeric(features.get("zscore_72", pd.Series(np.nan, index=features.index)))
    diff1 = _safe_numeric(features.get("diff_1", pd.Series(np.nan, index=features.index)))
    diff6 = _safe_numeric(features.get("diff_6", pd.Series(np.nan, index=features.index)))
    persistence = _safe_numeric(features.get("hours_above_q80_last_24", pd.Series(np.nan, index=features.index)))
    lag1 = _safe_numeric(features.get("lag_1", pd.Series(np.nan, index=features.index)))
    spatial = _safe_numeric(features.get("adaptive_spatial", pd.Series(np.nan, index=features.index)))

    out["event_score"] = (
        0.40 * z24.fillna(0.0)
        + 0.20 * z72.fillna(0.0)
        + 0.15 * diff1.fillna(0.0)
        + 0.10 * diff6.fillna(0.0)
        + 0.10 * persistence.fillna(0.0) / 24.0
        + 0.05 * (lag1 - spatial).fillna(0.0)
    )
    out["target_vs_spatial"] = (lag1 - spatial)
    out["spatial_rise_1"] = spatial - spatial.shift(1)
    return out


def _build_features(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    original_target: pd.Series,
    original_missing: np.ndarray,
    groups: np.ndarray | None = None,
    use_history_features: bool = True,
    use_spatial_features: bool = True,
    use_event_features: bool = True,
    use_calendar_features: bool = True,
    use_site_features: bool = True,
    use_gap_features: bool = True,
) -> pd.DataFrame:
    base = pd.DataFrame(index=df.index)

    raw_inputs = [c for c in feature_columns if c in df.columns and c != target]
    if not use_spatial_features:
        raw_inputs = [
            c for c in raw_inputs
            if not str(c).lower().startswith(("spatial_", "idw_spatial_"))
            and c != "adaptive_spatial"
        ]
    if raw_inputs:
        base = pd.concat([base, df[raw_inputs].copy()], axis=1)

    if use_calendar_features:
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
        base = pd.concat([base, _calendar_features(dt)], axis=1)
    if use_site_features:
        base = pd.concat([base, _site_features(df)], axis=1)
    if use_gap_features:
        base = pd.concat(
            [base, _gap_features(original_missing, groups=groups, index=df.index)],
            axis=1,
        )

    if use_history_features:
        observed_mask = ~original_missing
        history = _observed_history_features(original_target, observed_mask, groups=groups)
        base = pd.concat([base, history], axis=1)

    if use_spatial_features and "adaptive_spatial" in df.columns:
        base["adaptive_spatial"] = _safe_numeric(df["adaptive_spatial"])

    if use_event_features:
        base = pd.concat([base, _event_score_features(base)], axis=1)

    return base


def _warm_start(
    df: pd.DataFrame,
    target: str,
    original_target: pd.Series,
    original_missing: np.ndarray,
) -> pd.Series:
    current = original_target.copy()
    observed = ~original_missing

    global_median = float(original_target.loc[observed].median()) if observed.sum() else 0.0

    if "adaptive_spatial" in df.columns:
        current.loc[original_missing] = _safe_numeric(df.loc[original_missing, "adaptive_spatial"])

    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
        hour = dt.dt.hour
        hour_medians = original_target.loc[observed].groupby(hour.loc[observed]).median()
        need = current.loc[original_missing].isna()
        fill_values = hour.loc[original_missing].map(hour_medians)
        current.loc[original_missing] = current.loc[original_missing].where(~need, fill_values)

    current = current.fillna(global_median)
    return current


def _lgbm_regressor(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _lgbm_quantile(seed: int, alpha: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _event_classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _make_event_window_labels(
    y: pd.Series,
    groups: np.ndarray | None = None,
    percentile: float = 0.90,
    min_duration: int = 3,
    expand: int = 3,
) -> pd.Series:
    values = _safe_numeric(y)
    labels = pd.Series(0, index=values.index, dtype=int)
    group_series = pd.Series(
        np.zeros(len(values), dtype=int) if groups is None else np.asarray(groups),
        index=values.index,
    )

    for group in pd.unique(group_series):
        idx = group_series[group_series == group].index
        local = values.loc[idx]
        observed = local.dropna()
        if len(observed) < 30:
            continue

        threshold = float(observed.quantile(percentile))
        candidate = (local >= threshold).fillna(False).to_numpy()

        if candidate.any():
            starts = np.flatnonzero(candidate & ~np.r_[False, candidate[:-1]])
            ends = np.flatnonzero(candidate & ~np.r_[candidate[1:], False])

            for start, end in zip(starts, ends):
                if end - start + 1 < min_duration:
                    continue
                left = max(start - expand, 0)
                right = min(end + expand, len(candidate) - 1)
                labels.loc[idx[left : right + 1]] = 1

    return labels


def _fit_event_models(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
    event_percentile: float,
    event_probability_threshold: float,
    event_max_blend: float,
    event_cap_quantile: float,
) -> EventModels:
    train_mask = np.asarray(train_mask, dtype=bool)
    labels = _make_event_window_labels(y, groups=groups, percentile=event_percentile)
    train_labels = labels.loc[train_mask]
    counts = train_labels.value_counts()

    if len(counts) < 2 or int(counts.min()) < 20:
        logging.info("[%s] event refinement disabled: insufficient event labels %s", MODEL_NAME, counts.to_dict())
        return EventModels(None, None, np.nan, event_probability_threshold, event_max_blend, np.nan)

    classifier = _event_classifier(seed)
    classifier.fit(X.loc[train_mask], train_labels)

    event_rows = train_mask & (labels.to_numpy() == 1)
    if event_rows.sum() < 30:
        logging.info("[%s] event regressor disabled: insufficient event rows", MODEL_NAME)
        return EventModels(classifier, None, 1.0, event_probability_threshold, event_max_blend, np.nan)

    regressor = _lgbm_regressor(seed + 101)
    regressor.fit(X.loc[event_rows], y.loc[event_rows])

    train_event_values = _safe_numeric(y.loc[event_rows]).dropna()
    cap = float(train_event_values.quantile(event_cap_quantile)) if not train_event_values.empty else np.nan

    return EventModels(
        classifier=classifier,
        regressor=regressor,
        threshold=1.0,
        probability_threshold=event_probability_threshold,
        max_blend=event_max_blend,
        cap=cap,
    )


def _apply_event_refinement(
    base_prediction: np.ndarray,
    X_missing: pd.DataFrame,
    event_models: EventModels,
) -> tuple[np.ndarray, np.ndarray]:
    if event_models.classifier is None or event_models.regressor is None:
        return base_prediction, np.zeros(len(base_prediction), dtype=float)

    probability = event_models.classifier.predict_proba(X_missing)[:, 1]
    confidence = np.clip(
        (probability - event_models.probability_threshold)
        / max(1.0 - event_models.probability_threshold, 1e-12),
        0.0,
        1.0,
    )

    smooth_probability = (
        pd.Series(probability, index=X_missing.index)
        .rolling(window=3, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )
    smooth_confidence = np.clip(
        (smooth_probability - event_models.probability_threshold)
        / max(1.0 - event_models.probability_threshold, 1e-12),
        0.0,
        1.0,
    )

    blend = event_models.max_blend * smooth_confidence
    event_prediction = event_models.regressor.predict(X_missing)
    delta = event_prediction - base_prediction

    if np.isfinite(event_models.cap):
        delta = np.clip(delta, -event_models.cap, event_models.cap)

    refined = base_prediction + blend * delta
    return refined, smooth_probability


def impute_mice(
    data: pd.DataFrame,
    target_column: str,
    input_columns: list[str],
    max_iter: int = 8,
    tol: float = 1e-4,
    random_state: int = 42,
    custom_strategies=None,
    **kwargs,
) -> pd.DataFrame:
    import config_spatial as config

    original = data.copy()
    if "DateTime" not in original.columns:
        raise ValueError("DateTime column is required")

    original["DateTime"] = pd.to_datetime(original["DateTime"], errors="coerce")
    target = target_column
    original_y = _safe_numeric(original[target])
    original_missing = original_y.isna().to_numpy().copy()
    observed = ~original_missing

    if not original_missing.any() or observed.sum() < 50:
        return original

    groups = (
        original["Site"].astype(str).to_numpy()
        if "Site" in original.columns
        else np.zeros(len(original), dtype=int)
    )

    site_name = kwargs.get("site_name") or kwargs.get("site") or getattr(config, "TARGET_SITE", "")
    use_event_refinement = bool(
        kwargs.get("event_refinement", getattr(config, "AQUISTIL_EVENT_REFINEMENT_ENABLED", True))
    )
    use_history_features = bool(kwargs.get("history_features", True))
    use_spatial_features = bool(kwargs.get("spatial_features", True))
    use_calendar_features = bool(kwargs.get("calendar_features", True))
    use_site_features = bool(kwargs.get("site_features", True))
    use_gap_features = bool(kwargs.get("gap_features", True))
    use_uncertainty_models = bool(kwargs.get("uncertainty_models", True))
    event_percentile = float(
        np.clip(
            kwargs.get("event_percentile", getattr(config, "AQUISTIL_EVENT_PERCENTILE", 0.90)),
            0.75,
            0.99,
        )
    )
    event_probability_threshold = float(
        np.clip(
            kwargs.get(
                "event_probability_threshold",
                getattr(config, "AQUISTIL_EVENT_PROBABILITY_THRESHOLD", 0.55),
            ),
            0.0,
            0.99,
        )
    )
    event_max_blend = float(
        np.clip(
            kwargs.get("event_max_blend", getattr(config, "AQUISTIL_EVENT_MAX_BLEND", 0.70)),
            0.0,
            1.0,
        )
    )
    event_cap_quantile = float(
        np.clip(
            kwargs.get("event_cap_quantile", getattr(config, "AQUISTIL_EVENT_CAP_QUANTILE", 0.98)),
            0.80,
            1.0,
        )
    )
    spatial_config = dict(
        input_directory=config.INPUT_DIRECTORY,
        target_site=site_name,
        use_spatial=(
            use_spatial_features and getattr(config, "USE_SPATIAL_FEATURES", True)
        ),
        use_temporal=getattr(config, "USE_TEMPORAL_FEATURES", True),
        use_lagged=False,
        use_rolling=False,
    )

    try:
        df, feature_columns = prepare_spatial_temporal_data(original, target, input_columns, spatial_config)
        if len(df) != len(original) or not df.index.equals(original.index):
            alignment_issue = (
                "row count changed"
                if len(df) != len(original)
                else "index labels or row order changed"
            )
            logging.warning(
                "[%s] feature preparation alignment failed: %s "
                "(prepared_rows=%d, input_rows=%d); using local input frame",
                MODEL_NAME,
                alignment_issue,
                len(df),
                len(original),
            )
            df, feature_columns = original.copy(), list(input_columns)
    except Exception as exc:
        logging.warning("[%s] feature preparation fallback: %s", MODEL_NAME, exc)
        df, feature_columns = original.copy(), list(input_columns)

    df = df.loc[:, ~df.columns.duplicated()].copy()
    feature_columns = list(dict.fromkeys(feature_columns))
    df[target] = original_y.to_numpy()

    if use_spatial_features:
        adaptive_spatial, inventory = _adaptive_spatial_aggregate(
            df=df,
            target=target,
            feature_columns=feature_columns,
            observed_mask=observed,
            site_name=site_name,
            config=config,
        )
    else:
        adaptive_spatial = pd.Series(np.nan, index=df.index, name="adaptive_spatial")
        inventory = []
    if inventory:
        df["adaptive_spatial"] = adaptive_spatial
        if "adaptive_spatial" not in feature_columns:
            feature_columns = list(feature_columns) + ["adaptive_spatial"]

    logging.info("[%s] adaptive spatial contributors=%d", MODEL_NAME, len(inventory))

    current = _warm_start(df, target, original_y, original_missing)
    df[target] = current.to_numpy()

    raw_X = _build_features(
        df=df,
        target=target,
        feature_columns=list(feature_columns),
        original_target=original_y,
        original_missing=original_missing,
        groups=groups,
        use_history_features=use_history_features,
        use_spatial_features=use_spatial_features,
        use_event_features=use_event_refinement,
        use_calendar_features=use_calendar_features,
        use_site_features=use_site_features,
        use_gap_features=use_gap_features,
    )
    X = _clean_features(raw_X, train_mask=pd.Series(observed, index=df.index), groups=groups)
    y = original_y

    calendar_names = {
        "hour", "dayofweek", "month", "dayofyear", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    }
    raw_input_names = {
        column for column in feature_columns if column in df.columns and column != target
    }
    logging.info("[%s] final model feature matrix: %d features", MODEL_NAME, X.shape[1])
    logging.info(
        "[%s] ablation switches history=%s spatial=%s event=%s calendar=%s "
        "site=%s gap=%s uncertainty=%s",
        MODEL_NAME,
        use_history_features,
        use_spatial_features,
        use_event_refinement,
        use_calendar_features,
        use_site_features,
        use_gap_features,
        use_uncertainty_models,
    )
    logging.info("[%s]   - Selected/raw input features: %d", MODEL_NAME, len(raw_input_names & set(X.columns)))
    logging.info(
        "[%s]   - Spatial features: %d",
        MODEL_NAME,
        sum(
            str(column).lower().startswith(("spatial_", "idw_spatial_"))
            or column == "adaptive_spatial"
            for column in X.columns
        ),
    )
    logging.info("[%s]   - Calendar features: %d", MODEL_NAME, len(calendar_names & set(X.columns)))
    logging.info("[%s]   - Lag features and availability flags: %d", MODEL_NAME, sum(str(c).startswith("lag_") for c in X.columns))
    logging.info("[%s]   - Rolling features: %d", MODEL_NAME, sum(str(c).startswith("roll_") for c in X.columns))
    logging.info(
        "[%s]   - Derived history/event/gap features: %d",
        MODEL_NAME,
        sum(
            str(c).startswith(("diff_", "accel_", "zscore_", "hours_above_", "event_", "gap_"))
            or c in {"is_missing_original", "target_vs_spatial", "spatial_rise_1"}
            for c in X.columns
        ),
    )

    backbone = _lgbm_regressor(random_state)
    backbone.fit(X.loc[observed], y.loc[observed])

    missing_index = X.index[original_missing]
    X_missing = X.loc[missing_index]
    base_prediction = backbone.predict(X_missing)

    event_probability = np.full(len(X_missing), np.nan, dtype=float)
    final_prediction = base_prediction.copy()

    if use_event_refinement:
        event_models = _fit_event_models(
            X=X,
            y=y,
            train_mask=observed,
            groups=groups,
            seed=random_state + 1000,
            event_percentile=event_percentile,
            event_probability_threshold=event_probability_threshold,
            event_max_blend=event_max_blend,
            event_cap_quantile=event_cap_quantile,
        )
        final_prediction, event_probability = _apply_event_refinement(
            base_prediction=base_prediction,
            X_missing=X_missing,
            event_models=event_models,
        )
        logging.info(
            "[%s] event refinement active_rows=%d/%d mean_probability=%.4f",
            MODEL_NAME,
            int(np.sum(np.nan_to_num(event_probability) >= event_probability_threshold)),
            len(event_probability),
            float(np.nanmean(event_probability)) if len(event_probability) else float("nan"),
        )

    lower = np.full(len(X_missing), np.nan, dtype=float)
    upper = np.full(len(X_missing), np.nan, dtype=float)
    uncertainty = np.full(len(X_missing), np.nan, dtype=float)
    if use_uncertainty_models:
        q10 = _lgbm_quantile(random_state + 1, 0.10)
        q90 = _lgbm_quantile(random_state + 2, 0.90)
        q10.fit(X.loc[observed], y.loc[observed])
        q90.fit(X.loc[observed], y.loc[observed])

        lower = q10.predict(X_missing)
        upper = q90.predict(X_missing)
        uncertainty = np.maximum(upper - lower, 0.0)

    result = original.copy()
    result.loc[original_missing, target] = final_prediction
    result.loc[observed, target] = original_y.loc[observed].to_numpy()
    result[f"{target}_Uncertainty90"] = np.nan
    result.loc[original_missing, f"{target}_Uncertainty90"] = uncertainty
    result[f"{target}_EventProbability"] = np.nan
    result.loc[original_missing, f"{target}_EventProbability"] = event_probability

    return result
