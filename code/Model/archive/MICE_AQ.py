"""Air-quality-specialized MICE baseline.

MICE_AQ keeps Bayesian-ridge chained imputation but models the target on a
log1p scale, adds leakage-safe observed target context and calendar features,
and averages posterior imputations. Original observations and predictors are
never overwritten in the returned frame.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


MODEL_NAME = "MICE_AQ"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _calendar_features(data):
    features = pd.DataFrame(index=data.index)
    if "DateTime" not in data.columns:
        return features
    dt = pd.to_datetime(data["DateTime"], errors="coerce")
    features["aq_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    features["aq_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    features["aq_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    features["aq_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    features["aq_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    features["aq_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    return features.fillna(0.0)


def _target_context(target):
    """Features derived strictly from other observed target timestamps."""
    y = pd.to_numeric(target, errors="coerce").clip(lower=0.0)
    log_y = np.log1p(y)
    context = pd.DataFrame(index=target.index)

    # Shift before filling/rolling: an observed row can never use its own label.
    past = log_y.shift(1)
    future_reversed = log_y.shift(-1).iloc[::-1]
    context["aq_previous_log"] = past.ffill()
    context["aq_next_log"] = log_y.shift(-1).bfill()
    for window, minimum in ((6, 2), (24, 4), (72, 8)):
        context["aq_past_mean_%d" % window] = past.rolling(
            window, min_periods=minimum
        ).mean()
        context["aq_past_std_%d" % window] = past.rolling(
            window, min_periods=minimum
        ).std()
        context["aq_future_mean_%d" % window] = future_reversed.rolling(
            window, min_periods=minimum
        ).mean().iloc[::-1]
        context["aq_future_std_%d" % window] = future_reversed.rolling(
            window, min_periods=minimum
        ).std().iloc[::-1]
        context["aq_past_median_%d" % window] = past.rolling(
            window, min_periods=minimum
        ).median()
        context["aq_future_median_%d" % window] = future_reversed.rolling(
            window, min_periods=minimum
        ).median().iloc[::-1]

    observed = log_y.notna().to_numpy()
    positions = np.arange(len(log_y))
    previous_index = pd.Series(
        np.where(observed, positions, np.nan), index=target.index
    ).shift(1).ffill()
    next_index = pd.Series(
        np.where(observed, positions, np.nan), index=target.index
    ).shift(-1).bfill()
    context["aq_distance_previous"] = positions - previous_index
    context["aq_distance_next"] = next_index - positions
    span = context["aq_distance_previous"] + context["aq_distance_next"]
    context["aq_context_slope"] = (
        context["aq_next_log"] - context["aq_previous_log"]
    ) / span.replace(0, np.nan)
    fraction = context["aq_distance_previous"] / span.replace(0, np.nan)
    context["aq_linear_bridge_log"] = context["aq_previous_log"] + fraction * (
        context["aq_next_log"] - context["aq_previous_log"]
    )
    return context.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _standardize_predictors(frame):
    """Scale predictors while leaving the log-target column unchanged."""
    result = frame.copy()
    for column in result.columns[1:]:
        values = result[column]
        mean = values.mean(skipna=True)
        scale = values.std(skipna=True)
        if pd.isna(mean):
            continue
        if pd.isna(scale) or scale <= np.finfo(float).eps:
            result[column] = values - mean
        else:
            result[column] = (values - mean) / scale
    return result


def _missing_run_lengths(missing):
    """Return the length of the consecutive missing run for every row."""
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    lengths = missing.groupby(groups).transform("sum")
    return lengths.where(missing, 0).astype(int)


def _model_frame(data, target_column, input_columns):
    predictors = []
    for column in input_columns:
        if column == target_column or column not in data.columns or column in predictors:
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.notna().any():
            predictors.append(column)
    frame = data[predictors].apply(pd.to_numeric, errors="coerce")
    frame = pd.concat(
        [frame, _calendar_features(data), _target_context(data[target_column])],
        axis=1,
    )
    # Avoid ambiguous or duplicate feature names supplied by upstream stages.
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame.insert(
        0,
        "__mice_aq_target_log__",
        np.log1p(pd.to_numeric(data[target_column], errors="coerce").clip(lower=0.0)),
    )
    return frame


def impute_mice(
    data,
    target_column,
    input_columns,
    max_iter=15,
    random_state=42,
    tol=1e-3,
    custom_strategies=None,
    **kwargs
):
    """Impute target values with bounded, log-scale MICE.

    Posterior sampling is opt-in because Gaussian draws on the log scale can
    become implausibly large after ``expm1`` and dominate RMSE.  Predictions
    are constrained to the observed target range (or a caller-supplied upper
    bound), and multiple draws are combined with a median.
    """
    original = data.copy()
    target = pd.to_numeric(original[target_column], errors="coerce")
    missing = target.isna()
    if not missing.any() or not target.notna().any():
        return original

    posterior_draws = max(1, int(kwargs.get("mice_aq_draws", 1)))
    sample_posterior = bool(kwargs.get("mice_aq_sample_posterior", False))
    if posterior_draws > 1 and not sample_posterior:
        logging.warning(
            "[%s] mice_aq_draws=%d has no effect without posterior sampling; "
            "using one deterministic draw",
            MODEL_NAME, posterior_draws,
        )
        posterior_draws = 1

    observed_target = target.loc[~missing].clip(lower=0.0)
    upper_bound = kwargs.get("mice_aq_upper_bound")
    if upper_bound is None:
        upper_quantile = float(kwargs.get("mice_aq_upper_quantile", 0.999))
        if not 0.0 < upper_quantile <= 1.0:
            raise ValueError("mice_aq_upper_quantile must be in (0, 1]")
        upper_bound = float(observed_target.quantile(upper_quantile))
    else:
        upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound < 0.0:
        raise ValueError("mice_aq_upper_bound must be finite and non-negative")

    lower_log = 0.0
    upper_log = np.log1p(upper_bound)
    raw_context = _target_context(original[target_column])
    frame = _standardize_predictors(
        _model_frame(original, target_column, input_columns)
    )
    missing_positions = np.flatnonzero(missing.to_numpy())
    predictions = []
    for draw in range(posterior_draws):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=int(random_state) + draw * 1009,
            sample_posterior=sample_posterior,
            initial_strategy="median",
            skip_complete=True,
        )
        transformed = imputer.fit_transform(frame)
        target_log = np.clip(
            transformed[missing_positions, 0], lower_log, upper_log
        )
        predictions.append(np.expm1(target_log))

    result = original.copy()
    imputed = np.median(np.vstack(predictions), axis=0)

    # For longer internal gaps, cautiously blend toward interpolation between
    # the nearest observed neighbours.  Short/random gaps remain pure MICE so
    # the model retains its strong performance in those regimes.
    bridge_start = max(1, int(kwargs.get("mice_aq_bridge_start", 12)))
    bridge_max_weight = float(kwargs.get("mice_aq_bridge_max_weight", 0.5))
    if not 0.0 <= bridge_max_weight <= 1.0:
        raise ValueError("mice_aq_bridge_max_weight must be in [0, 1]")
    run_lengths = _missing_run_lengths(missing).loc[missing].to_numpy()
    bridge_log = raw_context.loc[missing, "aq_linear_bridge_log"].to_numpy()
    previous_distance = raw_context.loc[missing, "aq_distance_previous"].to_numpy()
    next_distance = raw_context.loc[missing, "aq_distance_next"].to_numpy()
    has_two_sided_context = (previous_distance > 0) & (next_distance > 0)
    bridge_weight = np.where(
        has_two_sided_context & (run_lengths >= bridge_start),
        bridge_max_weight * np.minimum(
            1.0, (run_lengths - bridge_start + 1) / float(bridge_start)
        ),
        0.0,
    )
    bridge_prediction = np.clip(np.expm1(bridge_log), 0.0, upper_bound)
    imputed = (1.0 - bridge_weight) * imputed + bridge_weight * bridge_prediction
    if not np.isfinite(imputed).all():
        raise ValueError("MICE_AQ produced non-finite target predictions")
    result.loc[missing, target_column] = imputed
    result.loc[~missing, target_column] = original.loc[~missing, target_column]
    logging.info(
        "[%s] imputed=%d predictors=%d posterior_draws=%d "
        "sample_posterior=%s prediction_min=%.6g prediction_median=%.6g "
        "prediction_max=%.6g upper_bound=%.6g bridge_rows=%d",
        MODEL_NAME, int(missing.sum()), frame.shape[1] - 1, posterior_draws,
        sample_posterior, float(np.min(imputed)), float(np.median(imputed)),
        float(np.max(imputed)), upper_bound, int(np.count_nonzero(bridge_weight)),
    )
    return result
