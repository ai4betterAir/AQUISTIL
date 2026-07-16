"""AQUISTIL4: mask-aware, leakage-safe spatial-temporal imputation.

The implementation deliberately keeps the public ``impute_mice`` interface
used by main.py.  Original observations are immutable and the original missing
mask is captured once, before any initialization or iterative update.
"""

import logging
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from spatial import prepare_spatial_temporal_data

MODEL_NAME = "AQUISTIL4"
GAP_BANDS = ((1, 6), (7, 24), (25, 72), (73, 10**9))
GAP_NAMES = ("short", "medium", "long", "extreme")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _gap_lengths(mask, groups=None):
    """Return total contiguous-gap length at each masked position."""
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


def _gap_band(lengths):
    lengths = np.asarray(lengths)
    out = np.zeros(lengths.size, dtype=int)
    out[(lengths >= 7) & (lengths <= 24)] = 1
    out[(lengths >= 25) & (lengths <= 72)] = 2
    out[lengths >= 73] = 3
    return out


def _causal_clean(frame, train_mask=None, groups=None):
    """Numeric cleanup with no backward fill or future-value propagation."""
    out = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if groups is None:
        out = out.ffill()
    else:
        out = out.groupby(pd.Series(np.asarray(groups), index=out.index), sort=False).ffill()
    reference = out.loc[train_mask] if train_mask is not None else out
    medians = reference.median(numeric_only=True).fillna(0.0)
    return out.fillna(medians).fillna(0.0)


def _site_token(column):
    text = re.sub(r"^spatial_", "", str(column), flags=re.I)
    return re.split(r"[_:]", text)[-1]


def _distance_km(a, b):
    from math import asin, cos, radians, sin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
    return 12742.0 * asin(sqrt(sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2))


def _adaptive_spatial_aggregate(df, target, feature_columns, observed_mask, site_name, config):
    """Correlation-distance weighted aggregate of available spatial columns."""
    # Stage 3 commonly selects an already aggregated IDW feature rather than
    # raw ``spatial_*`` neighbour columns.  Treat both forms as spatial input;
    # otherwise the adaptive spatial branch is silently disabled even though
    # IDW_Spatial_PM25 is present in the selected feature list.
    spatial = [
        c for c in feature_columns
        if c in df.columns
        and (
            str(c).lower().startswith("spatial_")
            or str(c).lower().startswith("idw_spatial_")
        )
    ]
    if not spatial:
        return pd.Series(np.nan, index=df.index, name="adaptive_spatial"), []

    coords = getattr(config, "SITE_COORDINATES", {}) or {}
    coord_map = {re.sub(r"[^A-Za-z0-9]", "", str(k)).upper(): v for k, v in coords.items()}
    target_coord = coord_map.get(re.sub(r"[^A-Za-z0-9]", "", str(site_name)).upper())
    y = pd.to_numeric(df[target], errors="coerce")
    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    inventory = []
    for col in spatial:
        values = pd.to_numeric(df[col], errors="coerce")
        valid_train = observed_mask & values.notna()
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
    return numerator.div(denominator.where(denominator > 0)).rename("adaptive_spatial"), inventory


def _make_artificial_masks(observed_mask, rng, repetitions=3, groups=None):
    """Create mask-aware validation gaps for all four deployment gap bands."""
    observed_mask = np.asarray(observed_mask, dtype=bool)
    masks = []
    groups = np.zeros(len(observed_mask), dtype=int) if groups is None else np.asarray(groups)
    lengths = (3, 12, 48, 96)
    for band, gap_len in enumerate(lengths):
        candidates = [
            i for i in range(1, len(observed_mask) - gap_len)
            if groups[i] == groups[i + gap_len - 1]
            and observed_mask[i : i + gap_len].all()
        ]
        if not candidates:
            continue
        for start in rng.choice(candidates, size=min(repetitions, len(candidates)), replace=False):
            mask = np.zeros(len(observed_mask), dtype=bool)
            mask[start : start + gap_len] = True
            masks.append((band, mask))
    return masks


def _base_features(df, target, feature_columns, strict, missing_geometry, groups=None):
    X = df[[c for c in feature_columns if c in df.columns and c != target]].copy()
    dt = pd.to_datetime(df["DateTime"], errors="coerce")
    X["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    y = pd.to_numeric(df[target], errors="coerce")
    group_series = pd.Series(np.zeros(len(df), dtype=int) if groups is None else groups, index=df.index)
    for lag in (1, 6, 24, 72):
        X[f"lag_{lag}"] = y.groupby(group_series, sort=False).shift(lag)
    past = y.groupby(group_series, sort=False).shift(1)
    for window, minimum in ((6, 2), (24, 6), (72, 12)):
        grouped = past.groupby(group_series, sort=False)
        X[f"roll_mean_{window}"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=minimum).mean()
        )
        X[f"roll_std_{window}"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=minimum).std()
        )
    X["gap_length"] = missing_geometry
    return X


def _model(seed, objective="regression", alpha=None):
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=63,
                  min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
                  reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    if objective == "quantile":
        return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
    return lgb.LGBMRegressor(objective="regression", **params)


def _fit_experts(X, y, train_mask, artificial_masks, seed, groups=None):
    """Fit four experts; artificial gaps supply band-specific validation errors."""
    experts, errors, residuals = [], [], []
    global_rows = np.asarray(train_mask, dtype=bool)
    for band in range(4):
        relevant = [mask for b, mask in artificial_masks if b == band]
        validation = np.logical_or.reduce(relevant) if relevant else np.zeros(len(y), dtype=bool)
        fit_rows = global_rows & ~validation
        if fit_rows.sum() < 50:
            fit_rows, validation = global_rows, np.zeros(len(y), dtype=bool)
        model = _model(seed + band)
        model.fit(X.loc[fit_rows], y.loc[fit_rows])
        if validation.any():
            # Artificially hidden targets must not re-enter through target-lag
            # or target-rolling columns. Replace those values and carry only
            # information available before the artificial gap.
            X_validation = X.copy()
            history_cols = [c for c in X.columns if str(c).startswith(("lag_", "roll_"))]
            if history_cols:
                X_validation.loc[validation, history_cols] = np.nan
                if groups is None:
                    X_validation[history_cols] = X_validation[history_cols].ffill().fillna(0.0)
                else:
                    group_series = pd.Series(np.asarray(groups), index=X_validation.index)
                    X_validation[history_cols] = (
                        X_validation[history_cols]
                        .groupby(group_series, sort=False)
                        .ffill()
                        .fillna(0.0)
                    )
            pred = model.predict(X_validation.loc[validation])
            resid = y.loc[validation].to_numpy() - pred
            error = float(np.sqrt(np.mean(resid ** 2)))
            bias = float(np.mean(resid))
        else:
            fitted = model.predict(X.loc[fit_rows])
            resid = y.loc[fit_rows].to_numpy() - fitted
            error = float(np.sqrt(np.mean(resid ** 2)))
            bias = float(np.mean(resid))
        experts.append(model)
        errors.append(max(error, 1e-6))
        residuals.append(bias)
    return experts, np.asarray(errors), np.asarray(residuals)


def impute_mice(data, target_column, input_columns, max_iter=8, tol=1e-4,
                random_state=42, custom_strategies=None, **kwargs):
    import config_spatial as config

    original = data.copy()
    if "DateTime" not in original.columns:
        raise ValueError("DateTime column is required")
    original["DateTime"] = pd.to_datetime(original["DateTime"], errors="coerce")
    target = target_column
    original_y = pd.to_numeric(original[target], errors="coerce")
    original_missing = original_y.isna().to_numpy().copy()  # fixed for the full run
    observed = ~original_missing
    groups = (original["Site"].astype(str).to_numpy()
              if "Site" in original.columns else np.zeros(len(original), dtype=int))
    if not original_missing.any() or observed.sum() < 50:
        return original

    strict = bool(kwargs.get("strict_feature_list", getattr(config, "STRICT_PROGRESSIVE_FEATURE_LIST", False)))
    uncertainty_damping = bool(kwargs.get("uncertainty_damping", False))
    site_name = kwargs.get("site_name") or kwargs.get("site") or getattr(config, "TARGET_SITE", "")
    spatial_config = dict(input_directory=config.INPUT_DIRECTORY, target_site=site_name,
                          use_spatial=(False if strict else config.USE_SPATIAL_FEATURES),
                          use_temporal=(False if strict else config.USE_TEMPORAL_FEATURES),
                          use_lagged=False, use_rolling=False)
    try:
        df, feature_columns = prepare_spatial_temporal_data(original, target, input_columns, spatial_config)
    except Exception as exc:
        logging.warning("[%s] feature preparation fallback: %s", MODEL_NAME, exc)
        df, feature_columns = original.copy(), list(input_columns)

    # Restore authoritative target/mask after any preprocessing.
    df[target] = original_y.to_numpy()
    spatial_agg, inventory = _adaptive_spatial_aggregate(df, target, feature_columns, observed, site_name, config)
    if inventory and not strict:
        df["adaptive_spatial"] = spatial_agg
        feature_columns = list(feature_columns) + ["adaptive_spatial"]
    logging.info("[%s] adaptive spatial contributors=%d", MODEL_NAME, len(inventory))

    rng = np.random.default_rng(random_state)
    artificial_masks = _make_artificial_masks(observed, rng, groups=groups)
    original_gap_lengths = _gap_lengths(original_missing, groups=groups)
    gap_bands = _gap_band(original_gap_lengths)
    init = float(original_y.loc[observed].median())
    current = original_y.copy()
    current.loc[original_missing] = init
    previous = current.loc[original_missing].to_numpy().copy()
    uncertainty = np.zeros(original_missing.sum(), dtype=float)

    for iteration in range(1, max_iter + 1):
        df[target] = current.to_numpy()
        # Geometry always comes from the fixed original mask, never the filled target.
        raw_X = _base_features(df, target, feature_columns, strict, original_gap_lengths, groups=groups)
        X = _causal_clean(raw_X, pd.Series(observed, index=df.index), groups=groups)
        y = original_y

        temporal_cols = [
            c for c in X.columns
            if not str(c).lower().startswith(("spatial_", "idw_spatial_"))
            and c != "adaptive_spatial"
        ]
        spatial_cols = [c for c in X.columns if c not in temporal_cols]
        X_temporal = X[temporal_cols] if temporal_cols else X
        X_spatial = X[temporal_cols + spatial_cols] if spatial_cols else X_temporal

        temporal_experts, temporal_err, temporal_bias = _fit_experts(
            X_temporal, y, observed, artificial_masks, random_state + iteration * 20,
            groups=groups,
        )
        spatial_experts, spatial_err, spatial_bias = _fit_experts(
            X_spatial, y, observed, artificial_masks, random_state + iteration * 20 + 7,
            groups=groups,
        )
        pred_rows = np.flatnonzero(original_missing)
        proposed = np.empty(len(pred_rows), dtype=float)
        for band in range(4):
            select = gap_bands[pred_rows] == band
            if not select.any():
                continue
            rows = pred_rows[select]
            temporal_pred = temporal_experts[band].predict(X_temporal.iloc[rows]) + temporal_bias[band]
            spatial_pred = spatial_experts[band].predict(X_spatial.iloc[rows]) + spatial_bias[band]
            wt = 1.0 / temporal_err[band] ** 2
            ws = 1.0 / spatial_err[band] ** 2 if spatial_cols else 0.0
            proposed[select] = (wt * temporal_pred + ws * spatial_pred) / max(wt + ws, 1e-12)

        # Quantile interval width controls the per-row damping rate.
        q10, q90 = _model(random_state + iteration, "quantile", 0.10), _model(random_state + iteration + 1, "quantile", 0.90)
        q10.fit(X_spatial.loc[observed], y.loc[observed])
        q90.fit(X_spatial.loc[observed], y.loc[observed])
        uncertainty = np.maximum(q90.predict(X_spatial.loc[original_missing]) - q10.predict(X_spatial.loc[original_missing]), 0.0)
        scale = max(float(np.nanmedian(uncertainty)), 1e-6)
        if uncertainty_damping:
            update_rate = np.clip(0.75 / (1.0 + uncertainty / scale), 0.15, 0.65)
        else:
            # Prediction intervals quantify uncertainty; by default they do
            # not pull predictions back towards the median initialization.
            update_rate = 1.0
        updated = previous + update_rate * (proposed - previous)
        current.loc[original_missing] = updated
        current.loc[observed] = original_y.loc[observed]  # immutable observations
        change = float(np.mean(np.abs(updated - previous)))
        logging.info("[%s] iteration=%d mean_change=%.6f median_PI90=%.4f", MODEL_NAME, iteration, change, float(np.median(uncertainty)))
        previous = updated.copy()
        if change < tol:
            break

    result = original.copy()
    result.loc[original_missing, target] = current.loc[original_missing].to_numpy()
    result.loc[observed, target] = original_y.loc[observed].to_numpy()
    result[f"{target}_Uncertainty90"] = np.nan
    result.loc[original_missing, f"{target}_Uncertainty90"] = uncertainty
    return result
