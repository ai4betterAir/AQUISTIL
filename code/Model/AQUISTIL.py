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
GAP_BANDS = ((1, 23), (24, 71), (72, 240), (241, 10**9))
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
    out = np.full(lengths.size, -1, dtype=int)
    out[(lengths >= 1) & (lengths <= 23)] = 0
    out[(lengths >= 24) & (lengths <= 71)] = 1
    out[(lengths >= 72) & (lengths <= 240)] = 2
    out[lengths >= 241] = 3
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
    forward_fill: bool = True,
    median_fill: bool = True,
) -> pd.DataFrame:
    out = frame.apply(_safe_numeric)
    if forward_fill:
        out = _causal_group_ffill(out, groups=groups)
    if median_fill:
        reference = out.loc[train_mask] if train_mask is not None else out
        medians = reference.median(numeric_only=True).fillna(0.0)
        out = out.fillna(medians).fillna(0.0)
    return out


def _adaptive_spatial_aggregate(
    df: pd.DataFrame,
    target: str,
    feature_columns: Iterable[str],
    observed_mask: np.ndarray,
    site_name: str,
    config,
) -> tuple[pd.Series, pd.Series, list[tuple[str, float, float, float]]]:
    spatial_columns = [
        c
        for c in feature_columns
        if c in df.columns
        and (str(c).lower().startswith("spatial_") or str(c).lower().startswith("idw_spatial_"))
    ]
    if not spatial_columns and {"DateTime", "Site"}.issubset(df.columns):
        times = pd.to_datetime(df["DateTime"], errors="coerce")
        sites = df["Site"].astype(str)
        values = _safe_numeric(df[target])
        pooled = pd.DataFrame(
            {"DateTime": times, "Site": sites, "Value": values}
        ).pivot_table(
            index="DateTime", columns="Site", values="Value", aggfunc="first"
        )
        adaptive = pd.Series(np.nan, index=df.index, name="adaptive_spatial")
        support = pd.Series(0, index=df.index, name="adaptive_spatial_support", dtype=int)
        inventory: list[tuple[str, float, float, float]] = []
        coords = getattr(config, "SITE_COORDINATES", {}) or {}
        coord_map = {
            re.sub(r"[^A-Za-z0-9]", "", str(key)).upper(): value
            for key, value in coords.items()
        }

        for local_site in pd.unique(sites):
            if local_site not in pooled.columns:
                continue
            local_rows = sites.eq(local_site)
            local_times = times.loc[local_rows]
            own = _safe_numeric(pooled[local_site])
            numerator = np.zeros(int(local_rows.sum()), dtype=float)
            denominator = np.zeros(int(local_rows.sum()), dtype=float)
            local_support = np.zeros(int(local_rows.sum()), dtype=int)
            local_coord = coord_map.get(
                re.sub(r"[^A-Za-z0-9]", "", local_site).upper()
            )

            for neighbor in pooled.columns:
                if neighbor == local_site:
                    continue
                neighbor_values = _safe_numeric(pooled[neighbor])
                valid_train = own.notna() & neighbor_values.notna()
                corr = (
                    abs(float(own.loc[valid_train].corr(neighbor_values.loc[valid_train])))
                    if int(valid_train.sum()) >= 12
                    else 0.0
                )
                if not np.isfinite(corr):
                    corr = 0.0
                distance = 1.0
                neighbor_coord = coord_map.get(
                    re.sub(r"[^A-Za-z0-9]", "", str(neighbor)).upper()
                )
                if local_coord and neighbor_coord:
                    distance = max(_distance_km(local_coord, neighbor_coord), 1.0)
                weight = max(corr, 0.05) / distance
                aligned = neighbor_values.reindex(local_times).to_numpy(dtype=float)
                valid = np.isfinite(aligned)
                numerator[valid] += weight * aligned[valid]
                denominator[valid] += weight
                local_support[valid] += 1
                inventory.append(
                    (f"pooled:{local_site}<-{neighbor}", corr, distance, weight)
                )

            local_adaptive = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )
            adaptive.loc[local_rows] = local_adaptive
            support.loc[local_rows] = local_support
        return adaptive, support, inventory

    if not spatial_columns:
        return (
            pd.Series(np.nan, index=df.index, name="adaptive_spatial"),
            pd.Series(0, index=df.index, name="adaptive_spatial_support", dtype=int),
            [],
        )

    coords = getattr(config, "SITE_COORDINATES", {}) or {}
    coord_map = {re.sub(r"[^A-Za-z0-9]", "", str(k)).upper(): v for k, v in coords.items()}
    target_coord = coord_map.get(re.sub(r"[^A-Za-z0-9]", "", str(site_name)).upper())

    y = _safe_numeric(df[target])
    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    support = pd.Series(0, index=df.index, dtype=int)
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
        support.loc[valid] += 1
        inventory.append((col, corr, distance, weight))

    adaptive = numerator.div(denominator.where(denominator > 0)).rename("adaptive_spatial")
    return adaptive, support.rename("adaptive_spatial_support"), inventory


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


def _linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[finite]
    y = values[finite]
    return float(np.polyfit(x, y, 1)[0])


def _missing_topology_features(
    original_target: pd.Series,
    original_missing: np.ndarray,
    groups: np.ndarray | None = None,
    include_boundary_values: bool = False,
    allow_future_context: bool = True,
) -> pd.DataFrame:
    """Build per-site missing-run topology and optional explicit boundaries."""
    missing = np.asarray(original_missing, dtype=bool)
    y = _safe_numeric(original_target)
    group_values = (
        np.zeros(len(y), dtype=int) if groups is None else np.asarray(groups)
    )
    out = pd.DataFrame(
        {
            "gap_total_length": np.zeros(len(y), dtype=float),
            "gap_position": np.zeros(len(y), dtype=float),
            "gap_fraction_completed": np.zeros(len(y), dtype=float),
            "hours_since_previous_target_observation": np.zeros(len(y), dtype=float),
            "hours_until_next_target_observation": np.zeros(len(y), dtype=float),
            "distance_to_nearest_observed_boundary": np.zeros(len(y), dtype=float),
        },
        index=y.index,
    )
    boundary_columns = ["previous_observed_target"]
    boundary_columns += [f"previous_{window}h_mean" for window in (3, 6, 12, 24)]
    boundary_columns += ["pre_gap_slope"]
    if allow_future_context:
        boundary_columns += ["next_observed_target"]
        boundary_columns += [f"next_{window}h_mean" for window in (3, 6, 12, 24)]
        boundary_columns += ["post_gap_slope"]
    if include_boundary_values:
        for column in boundary_columns:
            out[column] = np.nan

    for group in pd.unique(group_values):
        positions = np.flatnonzero(group_values == group)
        local_missing = missing[positions]
        local_y = y.iloc[positions].to_numpy(dtype=float)
        if include_boundary_values:
            for local_position in np.flatnonzero(~local_missing):
                row_index = out.index[positions[local_position]]
                if local_position > 0 and np.isfinite(local_y[local_position - 1]):
                    out.loc[row_index, "previous_observed_target"] = local_y[
                        local_position - 1
                    ]
                    for window in (3, 6, 12, 24):
                        values = local_y[
                            max(0, local_position - window) : local_position
                        ]
                        out.loc[row_index, f"previous_{window}h_mean"] = (
                            np.nanmean(values) if np.isfinite(values).any() else np.nan
                        )
                    out.loc[row_index, "pre_gap_slope"] = _linear_slope(
                        local_y[max(0, local_position - 6) : local_position]
                    )
                if (
                    allow_future_context
                    and local_position + 1 < len(local_y)
                    and np.isfinite(local_y[local_position + 1])
                ):
                    out.loc[row_index, "next_observed_target"] = local_y[
                        local_position + 1
                    ]
                    for window in (3, 6, 12, 24):
                        values = local_y[
                            local_position + 1 : min(
                                len(local_y), local_position + 1 + window
                            )
                        ]
                        out.loc[row_index, f"next_{window}h_mean"] = (
                            np.nanmean(values) if np.isfinite(values).any() else np.nan
                        )
                    out.loc[row_index, "post_gap_slope"] = _linear_slope(
                        local_y[
                            local_position + 1 : min(len(local_y), local_position + 7)
                        ]
                    )
        starts = np.flatnonzero(local_missing & ~np.r_[False, local_missing[:-1]])
        ends = np.flatnonzero(local_missing & ~np.r_[local_missing[1:], False])

        for start, end in zip(starts, ends):
            run_positions = positions[start : end + 1]
            length = end - start + 1
            sequence = np.arange(1, length + 1, dtype=float)
            until = np.arange(length, 0, -1, dtype=float)
            has_previous = start > 0 and np.isfinite(local_y[start - 1])
            has_next = end + 1 < len(local_y) and np.isfinite(local_y[end + 1])

            out.loc[out.index[run_positions], "gap_total_length"] = length
            out.loc[out.index[run_positions], "gap_position"] = sequence
            out.loc[out.index[run_positions], "gap_fraction_completed"] = sequence / length
            out.loc[
                out.index[run_positions], "hours_since_previous_target_observation"
            ] = sequence if has_previous else np.nan
            if allow_future_context:
                out.loc[
                    out.index[run_positions], "hours_until_next_target_observation"
                ] = until if has_next else np.nan
                nearest = np.minimum(sequence, until)
                if not has_previous:
                    nearest = until
                if not has_next:
                    nearest = sequence
                if not has_previous and not has_next:
                    nearest[:] = np.nan
            else:
                out.loc[
                    out.index[run_positions], "hours_until_next_target_observation"
                ] = np.nan
                nearest = sequence if has_previous else np.full(length, np.nan)
            out.loc[
                out.index[run_positions], "distance_to_nearest_observed_boundary"
            ] = nearest

            if not include_boundary_values:
                continue
            if has_previous:
                out.loc[
                    out.index[run_positions], "previous_observed_target"
                ] = local_y[start - 1]
                for window in (3, 6, 12, 24):
                    values = local_y[max(0, start - window) : start]
                    out.loc[
                        out.index[run_positions], f"previous_{window}h_mean"
                    ] = np.nanmean(values) if np.isfinite(values).any() else np.nan
                pre_values = local_y[max(0, start - 6) : start]
                out.loc[out.index[run_positions], "pre_gap_slope"] = _linear_slope(
                    pre_values
                )
            if allow_future_context and has_next:
                out.loc[out.index[run_positions], "next_observed_target"] = local_y[end + 1]
                for window in (3, 6, 12, 24):
                    values = local_y[end + 1 : min(len(local_y), end + 1 + window)]
                    out.loc[
                        out.index[run_positions], f"next_{window}h_mean"
                    ] = np.nanmean(values) if np.isfinite(values).any() else np.nan
                post_values = local_y[end + 1 : min(len(local_y), end + 7)]
                out.loc[out.index[run_positions], "post_gap_slope"] = _linear_slope(
                    post_values
                )

    return out


def _gap_expert_threshold(config, target: str, override=None) -> int:
    configured = (
        override
        if override is not None
        else getattr(config, "AQUISTIL_GAP_EXPERT_MIN_RUN_LENGTH", 2)
    )
    if isinstance(configured, dict):
        configured = configured.get(_target_key(target), configured.get("default", 2))
    threshold = int(configured)
    if threshold < 2:
        raise ValueError("AQUISTIL gap-expert threshold must be at least 2 hours")
    return threshold


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


def _lgbm_regressor(seed: int, n_jobs: int = -1) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=60,
        max_depth=3,
        learning_rate=0.08,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )


def _lgbm_quantile(seed: int, alpha: float, n_jobs: int = -1) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=60,
        max_depth=3,
        learning_rate=0.08,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )


def _event_classifier(seed: int, n_jobs: int = -1) -> lgb.LGBMClassifier:
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
        n_jobs=n_jobs,
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
    n_jobs: int = -1,
) -> EventModels:
    train_mask = np.asarray(train_mask, dtype=bool)
    labels = _make_event_window_labels(y, groups=groups, percentile=event_percentile)
    train_labels = labels.loc[train_mask]
    counts = train_labels.value_counts()

    if len(counts) < 2 or int(counts.min()) < 20:
        logging.info("[%s] event refinement disabled: insufficient event labels %s", MODEL_NAME, counts.to_dict())
        return EventModels(None, None, np.nan, event_probability_threshold, event_max_blend, np.nan)

    classifier = _event_classifier(seed, n_jobs=n_jobs)
    classifier.fit(X.loc[train_mask], train_labels)

    event_rows = train_mask & (labels.to_numpy() == 1)
    if event_rows.sum() < 30:
        logging.info("[%s] event regressor disabled: insufficient event rows", MODEL_NAME)
        return EventModels(classifier, None, 1.0, event_probability_threshold, event_max_blend, np.nan)

    regressor = _lgbm_regressor(seed + 101, n_jobs=n_jobs)
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


def _target_key(target: str) -> str:
    token = str(target).strip().upper().replace(" ", "")
    if token in {"PM25", "PM2.5"}:
        return "PM2.5"
    return token


def _event_refinement_enabled(requested: bool, regime: str, allowed_regimes) -> bool:
    allowed = {str(value).strip().lower() for value in allowed_regimes}
    return bool(requested) and str(regime or "").strip().lower() in allowed


def _hourly_reference(
    original: pd.DataFrame,
    target: str,
    original_y: pd.Series,
    observed: np.ndarray,
    missing_index: pd.Index,
) -> pd.Series:
    global_median = float(original_y.loc[observed].median()) if observed.sum() else 0.0
    reference = pd.Series(global_median, index=missing_index, dtype=float)
    if "DateTime" not in original.columns:
        return reference

    dt = pd.to_datetime(original["DateTime"], errors="coerce")
    hour_medians = original_y.loc[observed].groupby(dt.loc[observed].dt.hour).median()
    hours = dt.loc[missing_index].dt.hour
    hourly = hours.map(hour_medians)
    return reference.where(hourly.isna(), hourly)


def _adaptive_reference(
    reference_features: pd.DataFrame,
    missing_index: pd.Index,
    min_support: int = 1,
) -> pd.Series:
    reference = pd.Series(np.nan, index=missing_index, dtype=float)
    if "adaptive_spatial" not in reference_features.columns:
        return reference

    reference = _safe_numeric(reference_features["adaptive_spatial"]).reindex(missing_index)
    if "adaptive_spatial_support" in reference_features.columns:
        support = _safe_numeric(reference_features["adaptive_spatial_support"]).reindex(
            missing_index
        )
        reference = reference.where(support >= max(int(min_support), 1))
    return reference


def _observed_prediction_bounds(
    y: pd.Series,
    observed: np.ndarray,
    config,
) -> tuple[float, float]:
    values = _safe_numeric(y.loc[observed]).dropna()
    if values.empty:
        return -np.inf, np.inf

    lower_q = float(getattr(config, "AQUISTIL_ADAPTIVE_LOWER_QUANTILE", 0.01))
    upper_q = float(getattr(config, "AQUISTIL_ADAPTIVE_UPPER_QUANTILE", 0.995))
    iqr_factor = float(getattr(config, "AQUISTIL_ADAPTIVE_IQR_FACTOR", 1.5))
    lower_q = float(np.clip(lower_q, 0.0, 0.50))
    upper_q = float(np.clip(upper_q, 0.50, 1.0))

    q_low = float(values.quantile(lower_q))
    q_high = float(values.quantile(upper_q))
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    iqr = max(q75 - q25, 0.0)
    lower = q_low - iqr_factor * iqr
    upper = q_high + iqr_factor * iqr
    if bool(getattr(config, "AQUISTIL_ADAPTIVE_NONNEGATIVE", True)):
        lower = max(lower, 0.0)
    if not np.isfinite(lower):
        lower = -np.inf
    if not np.isfinite(upper) or upper <= lower:
        upper = np.inf
    return lower, upper


def _adaptive_gap_blend_weights(
    target: str,
    regime: str,
    X_missing: pd.DataFrame,
    uncertainty: np.ndarray,
    config,
) -> np.ndarray:
    target_rules = getattr(config, "AQUISTIL_ADAPTIVE_BLEND_RULES", {}) or {}
    rules = target_rules.get(_target_key(target), {})
    weights = np.zeros(len(X_missing), dtype=float)
    regime_key = str(regime or "").strip().lower()
    if regime_key in rules:
        weights = np.maximum(weights, float(rules[regime_key]))

    gap_length = (
        _safe_numeric(X_missing["gap_length"]).to_numpy()
        if "gap_length" in X_missing.columns
        else np.zeros(len(X_missing), dtype=float)
    )
    if "gap_medium" in rules:
        weights = np.where(
            gap_length >= 24,
            np.maximum(weights, float(rules["gap_medium"])),
            weights,
        )
    if "gap_long" in rules:
        weights = np.where(gap_length >= 72, np.maximum(weights, float(rules["gap_long"])), weights)
    if "gap_extreme" in rules:
        weights = np.where(gap_length >= 241, np.maximum(weights, float(rules["gap_extreme"])), weights)

    extra_rules = getattr(config, "AQUISTIL_ADAPTIVE_UNCERTAINTY_EXTRA_BLEND", {}) or {}
    extra = float(extra_rules.get(_target_key(target), 0.0))
    if extra > 0 and len(uncertainty):
        finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
        if finite_uncertainty.size:
            q = float(getattr(config, "AQUISTIL_ADAPTIVE_UNCERTAINTY_QUANTILE", 0.75))
            threshold = float(np.quantile(finite_uncertainty, np.clip(q, 0.0, 1.0)))
            weights = np.where(uncertainty >= threshold, weights + extra, weights)

    return np.clip(weights, 0.0, 1.0)


def _apply_adaptive_gap_guardrails(
    final_prediction: np.ndarray,
    X_missing: pd.DataFrame,
    reference_features: pd.DataFrame,
    original: pd.DataFrame,
    target: str,
    original_y: pd.Series,
    observed: np.ndarray,
    missing_index: pd.Index,
    uncertainty: np.ndarray,
    regime: str,
    config,
    enabled: bool | None = None,
    diagnostic_name: str = MODEL_NAME,
) -> tuple[np.ndarray, np.ndarray]:
    if enabled is None:
        enabled = bool(getattr(config, "AQUISTIL_ADAPTIVE_GAP_GUARDRAILS_ENABLED", True))
    candidate_weights = _adaptive_gap_blend_weights(
        target, regime, X_missing, uncertainty, config
    )
    reference = _adaptive_reference(
        reference_features=reference_features,
        missing_index=missing_index,
        min_support=getattr(config, "AQUISTIL_ADAPTIVE_MIN_SPATIAL_CONTRIBUTORS", 1),
    ).to_numpy(dtype=float)
    valid_reference = np.isfinite(reference)
    weights = np.where(
        valid_reference & bool(enabled), candidate_weights, 0.0
    )
    adjusted = np.asarray(final_prediction, dtype=float).copy()
    active = weights > 0
    adjusted[active] = (
        (1.0 - weights[active]) * adjusted[active]
        + weights[active] * reference[active]
    )

    logging.info(
        "[%s] adaptive QA enabled=%s valid_spatial_references=%d "
        "active_blend_rows=%d/%d mean_blend=%.6f max_blend=%.6f",
        diagnostic_name,
        bool(enabled),
        int(valid_reference.sum()),
        int(active.sum()),
        len(weights),
        float(weights[active].mean()) if active.any() else 0.0,
        float(weights[active].max()) if active.any() else 0.0,
    )
    logging.info(
        "[%s] adaptive gap guardrail target=%s regime=%s active_rows=%d/%d mean_blend=%.3f",
        diagnostic_name,
        target,
        regime,
        int(active.sum()),
        len(weights),
        float(weights[active].mean()) if active.any() else 0.0,
    )
    return adjusted, weights


def _finalize_predictions(
    prediction: np.ndarray,
    original: pd.DataFrame,
    target: str,
    original_y: pd.Series,
    observed: np.ndarray,
    missing_index: pd.Index,
    config,
) -> np.ndarray:
    """Apply target-wide robust bounds and an emergency non-finite fallback."""
    finalized = np.asarray(prediction, dtype=float).copy()
    clip_targets = getattr(config, "AQUISTIL_FINAL_CLIP_TARGETS", {}) or {}
    if bool(clip_targets.get(_target_key(target), False)):
        lower, upper = _observed_prediction_bounds(original_y, observed, config)
        finalized = np.clip(finalized, lower, upper)

    invalid = ~np.isfinite(finalized)
    if invalid.any():
        emergency = _hourly_reference(
            original, target, original_y, observed, missing_index
        ).to_numpy(dtype=float)
        finalized[invalid] = emergency[invalid]
        logging.warning(
            "[%s] replaced %d non-finite predictions with hourly/global medians",
            MODEL_NAME,
            int(invalid.sum()),
        )
    return finalized


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
    missingness_regime = str(kwargs.get("missingness_regime", "") or "").strip().lower()
    event_refinement_requested = bool(
        kwargs.get("event_refinement", getattr(config, "AQUISTIL_EVENT_REFINEMENT_ENABLED", True))
    )
    allowed_event_regimes = getattr(
        config, "AQUISTIL_EVENT_REFINEMENT_REGIMES", ("event", "")
    )
    use_event_refinement = _event_refinement_enabled(
        event_refinement_requested, missingness_regime, allowed_event_regimes
    )
    use_event_features = bool(kwargs.get("event_features", True))
    use_history_features = bool(kwargs.get("history_features", True))
    use_spatial_features = bool(kwargs.get("spatial_features", True))
    use_calendar_features = bool(kwargs.get("calendar_features", True))
    use_site_features = bool(kwargs.get("site_features", True))
    use_gap_features = bool(kwargs.get("gap_features", True))
    use_uncertainty_models = bool(kwargs.get("uncertainty_models", True))
    forward_fill_features = bool(kwargs.get("forward_fill_features", True))
    median_fill_features = bool(kwargs.get("median_fill_features", True))
    use_adaptive_gap_guardrails = bool(
        kwargs.get(
            "adaptive_gap_guardrails",
            getattr(config, "AQUISTIL_ADAPTIVE_GAP_GUARDRAILS_ENABLED", True),
        )
    )
    use_regime_aware = bool(
        kwargs.get(
            "regime_aware",
            getattr(config, "AQUISTIL_REGIME_AWARE_ENABLED", True),
        )
    ) and use_history_features
    diagnostic_name = str(kwargs.get("model_name", MODEL_NAME))
    log_feature_list = bool(kwargs.get("log_feature_list", False))
    gap_expert_threshold = _gap_expert_threshold(
        config,
        target,
        override=kwargs.get("gap_expert_min_run_length"),
    )
    gap_boundary_features = bool(
        kwargs.get(
            "gap_boundary_features",
            getattr(config, "AQUISTIL_GAP_BOUNDARY_FEATURES_ENABLED", False),
        )
    )
    gap_allow_future_context = bool(
        kwargs.get(
            "gap_allow_future_context",
            getattr(config, "AQUISTIL_GAP_ALLOW_FUTURE_CONTEXT", True),
        )
    )
    if bool(getattr(config, "AQUISTIL_GAP_MASKED_TRAINING_ENABLED", False)):
        logging.warning(
            "[%s] masked-gap training is staged but not enabled in this simple "
            "regime-aware verification implementation",
            MODEL_NAME,
        )
    try:
        model_n_jobs = int(kwargs.get("n_jobs", getattr(config, "MODEL_N_JOBS", -1)))
    except (TypeError, ValueError):
        model_n_jobs = -1
    if model_n_jobs == 0:
        model_n_jobs = -1
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
    if _target_key(target) == "OZONE" and missingness_regime in {"short_gap", "medium_gap"}:
        event_max_blend = min(
            event_max_blend,
            float(getattr(config, "AQUISTIL_OZONE_SHORT_MEDIUM_EVENT_MAX_BLEND", 0.20)),
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
        adaptive_spatial, adaptive_support, inventory = _adaptive_spatial_aggregate(
            df=df,
            target=target,
            feature_columns=feature_columns,
            observed_mask=observed,
            site_name=site_name,
            config=config,
        )
    else:
        adaptive_spatial = pd.Series(np.nan, index=df.index, name="adaptive_spatial")
        adaptive_support = pd.Series(0, index=df.index, name="adaptive_spatial_support")
        inventory = []
    if inventory:
        df["adaptive_spatial"] = adaptive_spatial
        df["adaptive_spatial_support"] = adaptive_support
        if "adaptive_spatial" not in feature_columns:
            feature_columns = list(feature_columns) + ["adaptive_spatial"]
        if "adaptive_spatial_support" not in feature_columns:
            feature_columns = list(feature_columns) + ["adaptive_spatial_support"]

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
        use_event_features=use_event_features,
        use_calendar_features=use_calendar_features,
        use_site_features=use_site_features,
        use_gap_features=use_gap_features,
    )
    X = _clean_features(
        raw_X,
        train_mask=pd.Series(observed, index=df.index),
        groups=groups,
        forward_fill=forward_fill_features,
        median_fill=median_fill_features,
    )
    topology_features = _missing_topology_features(
        original_target=original_y,
        original_missing=original_missing,
        groups=groups,
        include_boundary_values=gap_boundary_features,
        allow_future_context=gap_allow_future_context,
    )
    raw_gap_X = _build_features(
        df=df,
        target=target,
        feature_columns=list(feature_columns),
        original_target=original_y,
        original_missing=original_missing,
        groups=groups,
        use_history_features=False,
        use_spatial_features=use_spatial_features,
        use_event_features=use_event_features,
        use_calendar_features=use_calendar_features,
        use_site_features=use_site_features,
        use_gap_features=use_gap_features,
    )
    raw_gap_X = pd.concat([raw_gap_X, topology_features], axis=1)
    gap_X = _clean_features(
        raw_gap_X,
        train_mask=pd.Series(observed, index=df.index),
        groups=groups,
        forward_fill=forward_fill_features,
        median_fill=median_fill_features,
    )
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
        "[%s] switches history=%s spatial=%s event_features=%s event_refinement=%s "
        "calendar=%s site=%s gap=%s uncertainty=%s forward_fill=%s median_fill=%s "
        "adaptive_guardrails=%s regime_aware=%s gap_threshold=%d boundary_features=%s future_context=%s",
        MODEL_NAME,
        use_history_features,
        use_spatial_features,
        use_event_features,
        use_event_refinement,
        use_calendar_features,
        use_site_features,
        use_gap_features,
        use_uncertainty_models,
        forward_fill_features,
        median_fill_features,
        use_adaptive_gap_guardrails,
        use_regime_aware,
        gap_expert_threshold,
        gap_boundary_features,
        gap_allow_future_context,
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
    if log_feature_list:
        logging.info(
            "[%s] final feature list (%d): %s",
            diagnostic_name,
            len(X.columns),
            list(X.columns),
        )

    backbone = _lgbm_regressor(random_state, n_jobs=model_n_jobs)
    backbone.fit(X.loc[observed], y.loc[observed])

    missing_index = X.index[original_missing]
    X_missing = X.loc[missing_index]
    base_prediction = backbone.predict(X_missing)
    missing_run_lengths = topology_features.loc[
        missing_index, "gap_total_length"
    ].to_numpy(dtype=float)
    gap_route = use_regime_aware & (missing_run_lengths >= gap_expert_threshold)
    gap_prediction = np.full(len(X_missing), np.nan, dtype=float)
    if use_regime_aware and np.any(gap_route):
        gap_backbone = _lgbm_regressor(random_state, n_jobs=model_n_jobs)
        gap_backbone.fit(gap_X.loc[observed], y.loc[observed])
        gap_prediction = gap_backbone.predict(gap_X.loc[missing_index])
        base_prediction[gap_route] = gap_prediction[gap_route]
    logging.info(
        "[%s] topology router threshold=%dh history_rows=%d gap_rows=%d",
        MODEL_NAME,
        gap_expert_threshold,
        int((~gap_route).sum()),
        int(gap_route.sum()),
    )

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
            n_jobs=model_n_jobs,
        )
        final_prediction, event_probability = _apply_event_refinement(
            base_prediction=base_prediction,
            X_missing=X_missing,
            event_models=event_models,
        )
        event_probability = np.asarray(event_probability, dtype=float).copy()
        if np.any(gap_route):
            final_prediction[gap_route] = gap_prediction[gap_route]
            event_probability[gap_route] = np.nan
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
        q10 = _lgbm_quantile(random_state + 1, 0.10, n_jobs=model_n_jobs)
        q90 = _lgbm_quantile(random_state + 2, 0.90, n_jobs=model_n_jobs)
        q10.fit(X.loc[observed], y.loc[observed])
        q90.fit(X.loc[observed], y.loc[observed])

        lower = q10.predict(X_missing)
        upper = q90.predict(X_missing)
        if use_regime_aware and np.any(gap_route):
            gap_q10 = _lgbm_quantile(random_state + 1, 0.10, n_jobs=model_n_jobs)
            gap_q90 = _lgbm_quantile(random_state + 2, 0.90, n_jobs=model_n_jobs)
            gap_q10.fit(gap_X.loc[observed], y.loc[observed])
            gap_q90.fit(gap_X.loc[observed], y.loc[observed])
            gap_lower = gap_q10.predict(gap_X.loc[missing_index])
            gap_upper = gap_q90.predict(gap_X.loc[missing_index])
            lower[gap_route] = gap_lower[gap_route]
            upper[gap_route] = gap_upper[gap_route]
        uncertainty = np.maximum(upper - lower, 0.0)

    final_prediction, adaptive_blend = _apply_adaptive_gap_guardrails(
        final_prediction=final_prediction,
        X_missing=X_missing,
        reference_features=raw_X.loc[missing_index],
        original=original,
        target=target,
        original_y=original_y,
        observed=observed,
        missing_index=missing_index,
        uncertainty=uncertainty,
        regime=missingness_regime,
        config=config,
        enabled=use_adaptive_gap_guardrails,
        diagnostic_name=diagnostic_name,
    )
    final_prediction = _finalize_predictions(
        prediction=final_prediction,
        original=original,
        target=target,
        original_y=original_y,
        observed=observed,
        missing_index=missing_index,
        config=config,
    )

    result = original.copy()
    result.loc[original_missing, target] = final_prediction
    result.loc[observed, target] = original_y.loc[observed].to_numpy()
    result[f"{target}_Uncertainty90"] = np.nan
    result.loc[original_missing, f"{target}_Uncertainty90"] = uncertainty
    result[f"{target}_EventProbability"] = np.nan
    result.loc[original_missing, f"{target}_EventProbability"] = event_probability
    result[f"{target}_AdaptiveBlend"] = np.nan
    result.loc[original_missing, f"{target}_AdaptiveBlend"] = adaptive_blend
    result[f"{target}_Expert"] = "observed"
    selected_expert = np.where(gap_route, "gap", "history")
    result.loc[original_missing, f"{target}_Expert"] = selected_expert
    result[f"{target}_GapExpertThreshold"] = gap_expert_threshold
    for topology_column in topology_features.columns:
        result[f"{target}_{topology_column}"] = topology_features[
            topology_column
        ].to_numpy()
    gap_length = _gap_lengths(original_missing, groups=groups)
    result[f"{target}_GapLength"] = 0
    result.loc[original_missing, f"{target}_GapLength"] = gap_length[original_missing]
    bounds = {
        "short_gap": (1, 23),
        "medium_gap": (24, 71),
        "long_gap": (72, 240),
    }.get(missingness_regime)
    result[f"{target}_GapInRegime"] = np.nan
    if bounds:
        in_regime = (gap_length >= bounds[0]) & (gap_length <= bounds[1])
        result.loc[original_missing, f"{target}_GapInRegime"] = in_regime[
            original_missing
        ].astype(float)

    return result
