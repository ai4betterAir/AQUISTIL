"""AQUISTIL-R: robust, regime-aware stacked air-quality imputation.

The model keeps the repository's public ``impute_mice`` contract while
combining the strongest complementary imputers in the current benchmark:

* AQUISTIL and MICE_AQUISTIL for random and short missingness;
* AQUISTIL_A for medium and long gaps;
* MICE-BR for event-dependent missingness;
* LightGBM as an additional nonlinear diversity expert.

Stacking data is created inside each invocation. Observed target values are
artificially hidden with masks matching the active missingness regime, every
base expert is fitted without those values, and non-negative convex weights
are learned from the resulting out-of-fold predictions. Sparse validation
evidence is shrunk toward conservative priors. At prediction time unavailable
experts are removed row by row, large expert disagreement is pulled toward a
weighted median, and calibrated uncertainty plus confidence are returned.

No value from a genuinely missing target row is used to train the stack or to
construct its validation labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import re
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MODEL_NAME = "AQUISTIL_R"
DEFAULT_BASE_MODELS = (
    "AQUISTIL",
    "MICE_AQUISTIL",
    "MICE-BR",
    "AQUISTIL_A",
    "LightGBM",
)
SUPPORTED_REGIMES = (
    "random",
    "short_gap",
    "medium_gap",
    "long_gap",
    "event",
)

# Priors are deliberately broad. They encode the roles seen in the comparison
# table but retain weight on every expert so local validation can override them.
REGIME_PRIORS: Mapping[str, Mapping[str, float]] = {
    "random": {
        "AQUISTIL": 0.38,
        "MICE_AQUISTIL": 0.32,
        "MICE-BR": 0.08,
        "AQUISTIL_A": 0.12,
        "LightGBM": 0.10,
    },
    "short_gap": {
        "AQUISTIL": 0.42,
        "MICE_AQUISTIL": 0.28,
        "MICE-BR": 0.06,
        "AQUISTIL_A": 0.14,
        "LightGBM": 0.10,
    },
    "medium_gap": {
        "AQUISTIL": 0.18,
        "MICE_AQUISTIL": 0.16,
        "MICE-BR": 0.08,
        "AQUISTIL_A": 0.46,
        "LightGBM": 0.12,
    },
    "long_gap": {
        "AQUISTIL": 0.13,
        "MICE_AQUISTIL": 0.14,
        "MICE-BR": 0.15,
        "AQUISTIL_A": 0.48,
        "LightGBM": 0.10,
    },
    "event": {
        "AQUISTIL": 0.16,
        "MICE_AQUISTIL": 0.18,
        "MICE-BR": 0.42,
        "AQUISTIL_A": 0.12,
        "LightGBM": 0.12,
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class StackFit:
    """One fitted convex stack and its validation calibration."""

    weights: np.ndarray
    validation_rows: int
    rmse: float
    mae: float
    residual_q90: float


def _option(kwargs: Mapping, keyword: str, config_name: str, default):
    """Resolve a keyword first, then config_spatial, then a local default."""
    if keyword in kwargs:
        return kwargs[keyword]
    try:
        config = importlib.import_module("config_spatial")
        return getattr(config, config_name, default)
    except Exception:
        return default


def _normalise_regime(value) -> str | None:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "mcar": "random",
        "random_missingness": "random",
        "short": "short_gap",
        "shortgap": "short_gap",
        "medium": "medium_gap",
        "mediumgap": "medium_gap",
        "long": "long_gap",
        "longgap": "long_gap",
        "mnar": "event",
        "event_dependent": "event",
    }
    token = aliases.get(token, token)
    return token if token in SUPPORTED_REGIMES else None


def _group_codes(frame: pd.DataFrame) -> np.ndarray:
    if "Site" not in frame.columns:
        return np.zeros(len(frame), dtype=int)
    codes, _ = pd.factorize(frame["Site"].astype(str), sort=False)
    return codes.astype(int)


def _segments(mask: Sequence[bool], groups: Sequence | None = None) -> list[np.ndarray]:
    """Return contiguous positional segments without crossing group boundaries."""
    mask_array = np.asarray(mask, dtype=bool)
    group_array = (
        np.zeros(mask_array.size, dtype=int)
        if groups is None
        else np.asarray(groups)
    )
    if group_array.size != mask_array.size:
        raise ValueError("groups and mask must have the same length")

    result: list[np.ndarray] = []
    start = None
    for position, is_selected in enumerate(mask_array):
        continues = (
            is_selected
            and start is not None
            and position > 0
            and mask_array[position - 1]
            and group_array[position] == group_array[position - 1]
        )
        if is_selected and start is None:
            start = position
        elif is_selected and not continues:
            result.append(np.arange(start, position, dtype=int))
            start = position
        elif not is_selected and start is not None:
            result.append(np.arange(start, position, dtype=int))
            start = None
    if start is not None:
        result.append(np.arange(start, mask_array.size, dtype=int))
    return result


def _gap_lengths(mask: Sequence[bool], groups: Sequence | None = None) -> np.ndarray:
    lengths = np.zeros(len(mask), dtype=int)
    for segment in _segments(mask, groups):
        lengths[segment] = len(segment)
    return lengths


def _gap_band(length: int | float) -> str:
    value = int(length)
    if value <= 1:
        return "point"
    if value <= 23:
        return "short"
    if value <= 71:
        return "medium"
    return "long"


def _infer_regime(missing: Sequence[bool], groups: Sequence | None = None) -> str:
    segments = _segments(missing, groups)
    if not segments:
        return "random"
    lengths = np.asarray([len(segment) for segment in segments], dtype=float)
    singleton_fraction = float(np.mean(lengths == 1))
    if singleton_fraction >= 0.70:
        return "random"
    median_length = float(np.median(lengths))
    if median_length <= 23:
        return "short_gap"
    if median_length <= 71:
        return "medium_gap"
    return "long_gap"


def _project_simplex(values: Sequence[float]) -> np.ndarray:
    """Euclidean projection onto non-negative weights summing to one."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("simplex projection requires a non-empty vector")
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    support = np.flatnonzero(ordered - cumulative / np.arange(1, vector.size + 1) > 0)
    if not support.size:
        return np.full(vector.size, 1.0 / vector.size)
    rho = int(support[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    total = float(projected.sum())
    return projected / total if total > 0 else np.full(vector.size, 1.0 / vector.size)


def _prior_weights(models: Sequence[str], regime: str) -> np.ndarray:
    source = REGIME_PRIORS.get(regime, REGIME_PRIORS["random"])
    weights = np.asarray([float(source.get(model, 1.0)) for model in models], dtype=float)
    weights = np.maximum(weights, 0.0)
    return weights / weights.sum() if weights.sum() else np.full(len(models), 1.0 / len(models))


def _robust_convex_weights(
    predictions: np.ndarray,
    truth: np.ndarray,
    prior: np.ndarray,
    sample_weight: np.ndarray | None = None,
    regularization: float = 0.03,
    huber_delta: float = 1.5,
    max_steps: int = 1200,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Fit a simplex-constrained blend with a clipped robust gradient.

    The data term combines Huber influence with a lightly winsorised squared
    component. This keeps sensitivity to RMSE while preventing a few extreme
    validation rows from collapsing the stack onto a single expert.
    """
    matrix = np.asarray(predictions, dtype=float)
    target = np.asarray(truth, dtype=float)
    prior = _project_simplex(prior)
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("predictions must be a 2D matrix aligned with truth")
    if matrix.shape[1] != prior.size:
        raise ValueError("prior length must equal the number of experts")
    if target.size < 2:
        return prior.copy()

    finite_target = target[np.isfinite(target)]
    centre = float(np.median(finite_target)) if finite_target.size else 0.0
    q25, q75 = np.quantile(finite_target, [0.25, 0.75]) if finite_target.size else (0.0, 1.0)
    robust_scale = float(q75 - q25)
    if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
        robust_scale = float(np.std(finite_target)) if finite_target.size else 1.0
    if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
        robust_scale = 1.0

    scaled_matrix = (matrix - centre) / robust_scale
    scaled_target = (target - centre) / robust_scale
    weights_per_row = (
        np.ones(target.size, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    weights_per_row = np.nan_to_num(weights_per_row, nan=0.0, posinf=0.0, neginf=0.0)
    weights_per_row = np.maximum(weights_per_row, 0.0)
    if weights_per_row.sum() <= 0:
        weights_per_row[:] = 1.0
    weights_per_row /= weights_per_row.sum()

    spectral = float(np.linalg.norm(scaled_matrix, ord=2) ** 2)
    step = 0.9 / max(spectral / max(target.size, 1) + 2.0 * regularization, 1e-6)
    current = prior.copy()

    for _ in range(int(max_steps)):
        residual = scaled_matrix @ current - scaled_target
        huber_score = np.clip(residual, -huber_delta, huber_delta)
        rmse_score = np.clip(residual, -3.0 * huber_delta, 3.0 * huber_delta)
        influence = 0.80 * huber_score + 0.20 * rmse_score
        gradient = (
            scaled_matrix.T @ (weights_per_row * influence)
            + 2.0 * regularization * (current - prior)
        )
        updated = _project_simplex(current - step * gradient)
        if float(np.max(np.abs(updated - current))) <= tolerance:
            current = updated
            break
        current = updated
    return current


def _load_imputer(model_name: str) -> Callable:
    module = importlib.import_module("Model.%s" % model_name)
    imputer = getattr(module, "impute_mice", None)
    if not callable(imputer):
        raise AttributeError("Model.%s has no callable impute_mice" % model_name)
    return imputer


def _coerce_prediction(output, target: str, index: pd.Index) -> pd.Series:
    if isinstance(output, pd.DataFrame):
        if target not in output.columns:
            return pd.Series(np.nan, index=index, dtype=float)
        values = pd.to_numeric(output[target], errors="coerce").to_numpy()
    elif isinstance(output, pd.Series):
        values = pd.to_numeric(output, errors="coerce").to_numpy()
    else:
        values = np.asarray(output)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
    if values.ndim != 1 or len(values) != len(index):
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.Series(values, index=index, dtype=float).replace([np.inf, -np.inf], np.nan)


def _base_kwargs(kwargs: Mapping, regime: str, seed: int, max_iter: int) -> dict:
    clean = {
        key: value
        for key, value in kwargs.items()
        if not str(key).lower().startswith("aquistil_r_")
    }
    clean["missingness_regime"] = regime
    clean["random_state"] = int(seed)
    clean["seed"] = int(seed)
    clean.setdefault("max_iter", int(max_iter))
    return clean


def _run_base_models(
    frame: pd.DataFrame,
    target: str,
    input_columns: Sequence[str],
    models: Sequence[str],
    custom_strategies,
    kwargs: Mapping,
    regime: str,
    seed: int,
    max_iter: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    predictions = pd.DataFrame(index=frame.index, columns=list(models), dtype=float)
    failures: dict[str, str] = {}
    clean_kwargs = _base_kwargs(kwargs, regime, seed, max_iter)

    for offset, model_name in enumerate(models):
        try:
            imputer = _load_imputer(model_name)
            model_kwargs = dict(clean_kwargs)
            model_kwargs["random_state"] = int(seed + 1009 * offset)
            model_kwargs["seed"] = int(seed + 1009 * offset)
            output = imputer(
                frame.copy(),
                target,
                list(input_columns),
                custom_strategies=custom_strategies,
                **model_kwargs,
            )
            predictions[model_name] = _coerce_prediction(output, target, frame.index)
            if predictions[model_name].notna().sum() == 0:
                failures[model_name] = "returned no finite target predictions"
        except Exception as exc:
            failures[model_name] = "%s: %s" % (type(exc).__name__, exc)
            logging.warning("[%s] expert %s failed: %s", MODEL_NAME, model_name, exc)
    return predictions, failures


def _validation_target_count(
    observed_count: int,
    validation_fraction: float,
    min_points: int,
    max_points: int,
    min_training_points: int,
) -> int:
    available = max(0, observed_count - min_training_points)
    if available <= 0:
        return 0
    requested = int(round(observed_count * validation_fraction))
    requested = max(requested, min(min_points, available))
    return min(requested, max_points, available)


def _random_validation_mask(
    observed: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros(observed.size, dtype=bool)
    positions = np.flatnonzero(observed)
    if positions.size and target_count:
        selected = rng.choice(positions, size=min(target_count, positions.size), replace=False)
        mask[selected] = True
    return mask


def _event_validation_mask(
    target: pd.Series,
    observed: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
    quantile: float,
) -> np.ndarray:
    mask = np.zeros(observed.size, dtype=bool)
    positions = np.flatnonzero(observed)
    if positions.size == 0 or target_count <= 0:
        return mask

    values = pd.to_numeric(target.iloc[positions], errors="coerce").to_numpy(dtype=float)
    threshold = float(np.nanquantile(values, quantile))
    event_positions = positions[values >= threshold]
    event_count = min(target_count, event_positions.size)
    selected = (
        list(rng.choice(event_positions, size=event_count, replace=False))
        if event_count
        else []
    )
    remaining_count = target_count - len(selected)
    if remaining_count > 0:
        remainder = np.setdiff1d(positions, np.asarray(selected, dtype=int), assume_unique=False)
        remainder_values = pd.to_numeric(target.iloc[remainder], errors="coerce")
        ranks = remainder_values.rank(method="average", pct=True).fillna(0.5).to_numpy()
        probabilities = np.exp(4.0 * ranks)
        probabilities /= probabilities.sum()
        selected.extend(
            rng.choice(
                remainder,
                size=min(remaining_count, remainder.size),
                replace=False,
                p=probabilities,
            ).tolist()
        )
    mask[np.asarray(selected, dtype=int)] = True
    return mask


def _gap_validation_mask(
    observed: np.ndarray,
    groups: np.ndarray,
    target_count: int,
    min_gap: int,
    max_gap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = np.zeros(observed.size, dtype=bool)
    available = observed.copy()

    while int(selected.sum()) < target_count:
        remaining = target_count - int(selected.sum())
        if remaining < min_gap:
            break
        runs = _segments(available, groups)
        eligible = [segment for segment in runs if len(segment) >= min_gap]
        if not eligible:
            break
        run = eligible[int(rng.integers(0, len(eligible)))]
        upper = min(max_gap, len(run), remaining)
        if upper < min_gap:
            break
        length = int(rng.integers(min_gap, upper + 1))
        start_offset = int(rng.integers(0, len(run) - length + 1))
        positions = run[start_offset : start_offset + length]
        selected[positions] = True
        available[positions] = False

    # Small or naturally sparse series may not contain the requested regime's
    # minimum block. Use the longest safe observed run rather than silently
    # changing the validation problem to random missingness or masking too much.
    if not selected.any():
        runs = _segments(observed, groups)
        if runs:
            longest = max(runs, key=len)
            length = min(len(longest), target_count, max_gap)
            if length > 0:
                start_offset = int(rng.integers(0, len(longest) - length + 1))
                selected[longest[start_offset : start_offset + length]] = True
    return selected


def _build_validation_mask(
    frame: pd.DataFrame,
    target: str,
    regime: str,
    seed: int,
    validation_fraction: float,
    min_points: int,
    max_points: int,
    min_training_points: int,
    event_quantile: float,
) -> np.ndarray:
    values = pd.to_numeric(frame[target], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    observed = values.notna().to_numpy()
    target_count = _validation_target_count(
        int(observed.sum()),
        validation_fraction,
        min_points,
        max_points,
        min_training_points,
    )
    if target_count <= 0:
        return np.zeros(len(frame), dtype=bool)

    rng = np.random.default_rng(int(seed))
    groups = _group_codes(frame)
    if regime == "event":
        return _event_validation_mask(values, observed, target_count, rng, event_quantile)
    if regime == "random":
        return _random_validation_mask(observed, target_count, rng)

    ranges = {
        "short_gap": (1, 23),
        "medium_gap": (24, 71),
        "long_gap": (72, 240),
    }
    minimum, maximum = ranges[regime]
    return _gap_validation_mask(
        observed,
        groups,
        target_count,
        minimum,
        maximum,
        rng,
    )


def _oof_predictions(
    frame: pd.DataFrame,
    target: str,
    input_columns: Sequence[str],
    models: Sequence[str],
    regime: str,
    folds: int,
    validation_fraction: float,
    min_validation_points: int,
    max_validation_points: int,
    min_training_points: int,
    event_quantile: float,
    custom_strategies,
    kwargs: Mapping,
    random_state: int,
    max_iter: int,
) -> pd.DataFrame:
    truth = pd.to_numeric(frame[target], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    groups = _group_codes(frame)
    parts: list[pd.DataFrame] = []
    seen_masks: set[tuple[int, ...]] = set()

    for fold in range(int(folds)):
        fold_seed = int(random_state + 7919 * (fold + 1))
        validation_mask = _build_validation_mask(
            frame,
            target,
            regime,
            fold_seed,
            validation_fraction,
            min_validation_points,
            max_validation_points,
            min_training_points,
            event_quantile,
        )
        positions = np.flatnonzero(validation_mask)
        signature = tuple(positions.tolist())
        if not positions.size or signature in seen_masks:
            continue
        seen_masks.add(signature)

        masked = frame.copy()
        masked.iloc[positions, masked.columns.get_loc(target)] = np.nan
        predictions, failures = _run_base_models(
            masked,
            target,
            input_columns,
            models,
            custom_strategies,
            kwargs,
            regime,
            fold_seed,
            max_iter,
        )
        lengths = _gap_lengths(validation_mask, groups)
        part = predictions.iloc[positions].reset_index(drop=True)
        part["_truth"] = truth.iloc[positions].to_numpy(dtype=float)
        part["_regime"] = regime
        part["_band"] = [_gap_band(lengths[position]) for position in positions]
        part["_fold"] = fold
        finite_prediction = part[list(models)].notna().any(axis=1)
        finite_truth = np.isfinite(part["_truth"].to_numpy(dtype=float))
        part = part.loc[finite_prediction & finite_truth]
        if not part.empty:
            parts.append(part)
        logging.info(
            "[%s] OOF fold=%d regime=%s masked=%d usable=%d failures=%s",
            MODEL_NAME,
            fold + 1,
            regime,
            len(positions),
            len(part),
            sorted(failures),
        )

    if not parts:
        return pd.DataFrame(columns=list(models) + ["_truth", "_regime", "_band", "_fold"])
    return pd.concat(parts, ignore_index=True)


def _fit_stack(
    table: pd.DataFrame,
    models: Sequence[str],
    regime: str,
    min_expert_coverage: float,
    prior_strength: float,
    regularization: float,
    huber_delta: float,
) -> StackFit:
    prior = _prior_weights(models, regime)
    if table.empty:
        return StackFit(prior, 0, np.nan, np.nan, np.nan)

    prediction_frame = table[list(models)].apply(pd.to_numeric, errors="coerce")
    truth = pd.to_numeric(table["_truth"], errors="coerce").to_numpy(dtype=float)
    coverage = prediction_frame.notna().mean(axis=0).to_numpy(dtype=float)
    active = coverage >= float(min_expert_coverage)
    if not active.any():
        active[int(np.argmax(coverage))] = True

    active_indices = np.flatnonzero(active)
    active_frame = prediction_frame.iloc[:, active_indices].copy()
    row_centre = active_frame.median(axis=1, skipna=True)
    for column in active_frame.columns:
        active_frame[column] = active_frame[column].fillna(row_centre)
    valid = np.isfinite(truth) & active_frame.notna().all(axis=1).to_numpy()
    if int(valid.sum()) < 2:
        return StackFit(prior, int(valid.sum()), np.nan, np.nan, np.nan)

    active_prior = prior[active_indices]
    active_prior = _project_simplex(active_prior)
    learned = _robust_convex_weights(
        active_frame.loc[valid].to_numpy(dtype=float),
        truth[valid],
        active_prior,
        regularization=regularization,
        huber_delta=huber_delta,
    )
    evidence = float(valid.sum()) / max(float(valid.sum()) + float(prior_strength), 1.0)
    learned = _project_simplex(evidence * learned + (1.0 - evidence) * active_prior)

    full_weights = np.zeros(len(models), dtype=float)
    full_weights[active_indices] = learned
    prediction = active_frame.loc[valid].to_numpy(dtype=float) @ learned
    residual = truth[valid] - prediction
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    residual_q90 = float(np.quantile(np.abs(residual), 0.90))
    return StackFit(full_weights, int(valid.sum()), rmse, mae, residual_q90)


def _fit_routes(
    table: pd.DataFrame,
    models: Sequence[str],
    regime: str,
    min_route_rows: int,
    min_expert_coverage: float,
    prior_strength: float,
    regularization: float,
    huber_delta: float,
) -> dict[tuple[str, str], StackFit]:
    fits: dict[tuple[str, str], StackFit] = {}
    overall = _fit_stack(
        table,
        models,
        regime,
        min_expert_coverage,
        prior_strength,
        regularization,
        huber_delta,
    )
    fits[(regime, "*")] = overall

    if not table.empty:
        for band, subset in table.groupby("_band", sort=False):
            if len(subset) < int(min_route_rows):
                continue
            fits[(regime, str(band))] = _fit_stack(
                subset,
                models,
                regime,
                min_expert_coverage,
                prior_strength,
                regularization,
                huber_delta,
            )
    return fits


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(ordered_weights.sum())
    position = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(position, len(ordered_values) - 1)])


def _temporal_fallback(
    target: pd.Series,
    groups: np.ndarray,
) -> np.ndarray:
    numeric = pd.to_numeric(target, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    group_series = pd.Series(groups, index=numeric.index)
    filled = numeric.groupby(group_series, sort=False).transform(
        lambda values: values.interpolate(method="linear", limit_direction="both")
    )
    median = float(numeric.median()) if numeric.notna().any() else 0.0
    return filled.fillna(median).to_numpy(dtype=float)


def _plausible_bounds(
    target: pd.Series,
    groups: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
    iqr_factor: float,
    explicit_lower,
    explicit_upper,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(target, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    lower = np.full(len(numeric), np.nan, dtype=float)
    upper = np.full(len(numeric), np.nan, dtype=float)
    global_values = numeric.dropna().to_numpy(dtype=float)

    def limits(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, np.inf
        q_low, q25, q75, q_high = np.quantile(
            finite, [lower_quantile, 0.25, 0.75, upper_quantile]
        )
        spread = max(float(q75 - q25), float(np.std(finite)) * 0.25, 1e-6)
        low = float(q_low - iqr_factor * spread)
        high = float(q_high + iqr_factor * spread)
        if float(np.nanmin(finite)) >= 0.0:
            low = max(0.0, low)
        return low, max(high, low)

    global_low, global_high = limits(global_values)
    for group in pd.unique(groups):
        positions = np.flatnonzero(groups == group)
        local = numeric.iloc[positions].dropna().to_numpy(dtype=float)
        local_low, local_high = limits(local) if local.size >= 20 else (global_low, global_high)
        lower[positions] = local_low
        upper[positions] = local_high

    if explicit_lower is not None:
        lower[:] = float(explicit_lower)
    if explicit_upper is not None:
        upper[:] = float(explicit_upper)
    return lower, upper


def _safe_models(value: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        models = [token.strip() for token in value.split(",") if token.strip()]
    else:
        models = [str(token).strip() for token in value if str(token).strip()]
    models = list(dict.fromkeys(models))
    models = [model for model in models if model not in {MODEL_NAME, "AQUISTIL-R"}]
    if not models:
        raise ValueError("AQUISTIL-R requires at least one base model")
    return tuple(models)


def impute_mice(
    data: pd.DataFrame,
    target_column: str,
    input_columns: Sequence[str],
    max_iter: int = 8,
    tol: float = 1e-4,
    random_state: int = 42,
    custom_strategies=None,
    **kwargs,
) -> pd.DataFrame:
    """Impute ``target_column`` with a robust regime-aware stacked ensemble.

    Important keyword overrides use the ``aquistil_r_`` prefix. The matching
    ``AQUISTIL_R_*`` constants in ``config_spatial.py`` provide global defaults.
    """
    del tol  # The stack has its own convergence tolerance.
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if target_column not in data.columns:
        raise KeyError("target column %r is not present" % target_column)

    original = data.copy()
    target = pd.to_numeric(original[target_column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    missing = target.isna().to_numpy()
    observed_count = int((~missing).sum())
    if not missing.any():
        return original

    min_training_points = int(
        _option(kwargs, "aquistil_r_min_training_points", "AQUISTIL_R_MIN_TRAINING_POINTS", 50)
    )
    models = _safe_models(
        _option(kwargs, "aquistil_r_base_models", "AQUISTIL_R_BASE_MODELS", DEFAULT_BASE_MODELS)
    )
    groups = _group_codes(original)
    requested_regime = _normalise_regime(kwargs.get("missingness_regime"))
    regime = requested_regime or _infer_regime(missing, groups)
    seed = int(kwargs.get("seed", random_state))

    if observed_count < min_training_points:
        logging.warning(
            "[%s] only %d observed targets; skipping stack training and using available experts",
            MODEL_NAME,
            observed_count,
        )

    folds = int(_option(kwargs, "aquistil_r_validation_folds", "AQUISTIL_R_VALIDATION_FOLDS", 3))
    base_fraction = float(
        _option(kwargs, "aquistil_r_validation_fraction", "AQUISTIL_R_VALIDATION_FRACTION", 0.12)
    )
    actual_fraction = float(missing.mean())
    validation_fraction = float(np.clip(max(base_fraction, min(actual_fraction, 0.25)), 0.02, 0.35))
    min_validation_points = int(
        _option(kwargs, "aquistil_r_min_validation_points", "AQUISTIL_R_MIN_VALIDATION_POINTS", 48)
    )
    max_validation_points = int(
        _option(kwargs, "aquistil_r_max_validation_points", "AQUISTIL_R_MAX_VALIDATION_POINTS", 500)
    )
    event_quantile = float(
        np.clip(
            _option(kwargs, "aquistil_r_event_quantile", "AQUISTIL_R_EVENT_QUANTILE", 0.90),
            0.50,
            0.99,
        )
    )
    min_route_rows = int(
        _option(kwargs, "aquistil_r_min_route_rows", "AQUISTIL_R_MIN_ROUTE_ROWS", 24)
    )
    min_expert_coverage = float(
        np.clip(
            _option(kwargs, "aquistil_r_min_expert_coverage", "AQUISTIL_R_MIN_EXPERT_COVERAGE", 0.80),
            0.0,
            1.0,
        )
    )
    prior_strength = float(
        max(_option(kwargs, "aquistil_r_prior_strength", "AQUISTIL_R_PRIOR_STRENGTH", 40.0), 0.0)
    )
    regularization = float(
        max(_option(kwargs, "aquistil_r_regularization", "AQUISTIL_R_REGULARIZATION", 0.03), 0.0)
    )
    huber_delta = float(
        max(_option(kwargs, "aquistil_r_huber_delta", "AQUISTIL_R_HUBER_DELTA", 1.5), 0.1)
    )

    if folds > 0 and observed_count >= min_training_points:
        oof = _oof_predictions(
            original,
            target_column,
            input_columns,
            models,
            regime,
            folds,
            validation_fraction,
            min_validation_points,
            max_validation_points,
            min_training_points,
            event_quantile,
            custom_strategies,
            kwargs,
            seed,
            max_iter,
        )
    else:
        oof = pd.DataFrame()

    route_fits = _fit_routes(
        oof,
        models,
        regime,
        min_route_rows,
        min_expert_coverage,
        prior_strength,
        regularization,
        huber_delta,
    )
    for route, fit in route_fits.items():
        logging.info(
            "[%s] route=%s rows=%d weights=%s rmse=%.4f mae=%.4f",
            MODEL_NAME,
            route,
            fit.validation_rows,
            {model: round(float(weight), 4) for model, weight in zip(models, fit.weights)},
            fit.rmse,
            fit.mae,
        )

    predictions, final_failures = _run_base_models(
        original,
        target_column,
        input_columns,
        models,
        custom_strategies,
        kwargs,
        regime,
        seed + 104729,
        max_iter,
    )
    prediction_matrix = predictions.to_numpy(dtype=float)
    fallback = _temporal_fallback(target, groups)
    actual_gap_lengths = _gap_lengths(missing, groups)
    robust_scale = float(target.quantile(0.75) - target.quantile(0.25))
    if not np.isfinite(robust_scale) or robust_scale <= 1e-9:
        robust_scale = float(target.std())
    if not np.isfinite(robust_scale) or robust_scale <= 1e-9:
        robust_scale = 1.0

    disagreement_blend = float(
        np.clip(
            _option(
                kwargs,
                "aquistil_r_disagreement_blend",
                "AQUISTIL_R_DISAGREEMENT_BLEND",
                0.35,
            ),
            0.0,
            1.0,
        )
    )
    event_disagreement_blend = float(
        np.clip(
            _option(
                kwargs,
                "aquistil_r_event_disagreement_blend",
                "AQUISTIL_R_EVENT_DISAGREEMENT_BLEND",
                0.15,
            ),
            0.0,
            1.0,
        )
    )
    disagreement_start = float(
        max(
            _option(
                kwargs,
                "aquistil_r_disagreement_start",
                "AQUISTIL_R_DISAGREEMENT_START",
                0.35,
            ),
            0.0,
        )
    )
    disagreement_full = float(
        max(
            _option(
                kwargs,
                "aquistil_r_disagreement_full",
                "AQUISTIL_R_DISAGREEMENT_FULL",
                1.25,
            ),
            disagreement_start + 1e-6,
        )
    )

    result_values = target.to_numpy(dtype=float).copy()
    uncertainty = np.full(len(original), np.nan, dtype=float)
    confidence = np.full(len(original), np.nan, dtype=float)
    disagreement = np.full(len(original), np.nan, dtype=float)
    used_weights = np.zeros((len(original), len(models)), dtype=float)
    prior = _prior_weights(models, regime)

    for position in np.flatnonzero(missing):
        band = _gap_band(actual_gap_lengths[position])
        fit = route_fits.get((regime, band), route_fits.get((regime, "*")))
        base_weights = fit.weights.copy() if fit is not None else prior.copy()
        values = prediction_matrix[position]
        available = np.isfinite(values)
        row_weights = np.where(available, base_weights, 0.0)
        if row_weights.sum() <= 0 and available.any():
            row_weights = np.where(available, prior, 0.0)
        if row_weights.sum() <= 0 and available.any():
            row_weights = available.astype(float)

        if row_weights.sum() <= 0:
            result_values[position] = fallback[position]
            uncertainty[position] = robust_scale
            confidence[position] = 0.0
            continue

        row_weights /= row_weights.sum()
        used_weights[position] = row_weights
        available_values = values[available]
        available_weights = row_weights[available]
        available_weights /= available_weights.sum()
        weighted_mean = float(np.dot(available_values, available_weights))
        weighted_median = _weighted_median(available_values, available_weights)
        row_disagreement = float(
            np.sqrt(np.dot(available_weights, (available_values - weighted_mean) ** 2))
        )
        disagreement[position] = row_disagreement
        ratio = row_disagreement / robust_scale
        pull = float(
            np.clip(
                (ratio - disagreement_start) / (disagreement_full - disagreement_start),
                0.0,
                1.0,
            )
        )
        max_pull = event_disagreement_blend if regime == "event" else disagreement_blend
        result_values[position] = (1.0 - max_pull * pull) * weighted_mean + max_pull * pull * weighted_median

        residual_floor = fit.residual_q90 if fit is not None else np.nan
        spread90 = 1.645 * row_disagreement
        uncertainty[position] = (
            max(spread90, residual_floor)
            if np.isfinite(residual_floor)
            else max(spread90, 0.25 * robust_scale)
        )
        evidence = (
            fit.validation_rows / (fit.validation_rows + prior_strength)
            if fit is not None and fit.validation_rows > 0
            else 0.0
        )
        availability = float(available.sum()) / len(models)
        confidence[position] = availability * (0.5 + 0.5 * evidence) / (
            1.0 + uncertainty[position] / robust_scale
        )

    lower_quantile = float(
        np.clip(
            _option(kwargs, "aquistil_r_lower_quantile", "AQUISTIL_R_LOWER_QUANTILE", 0.001),
            0.0,
            0.25,
        )
    )
    upper_quantile = float(
        np.clip(
            _option(kwargs, "aquistil_r_upper_quantile", "AQUISTIL_R_UPPER_QUANTILE", 0.999),
            0.75,
            1.0,
        )
    )
    iqr_factor = float(
        max(_option(kwargs, "aquistil_r_bound_iqr_factor", "AQUISTIL_R_BOUND_IQR_FACTOR", 3.0), 0.0)
    )
    lower, upper = _plausible_bounds(
        target,
        groups,
        lower_quantile,
        upper_quantile,
        iqr_factor,
        _option(kwargs, "aquistil_r_lower_bound", "AQUISTIL_R_LOWER_BOUND", None),
        _option(kwargs, "aquistil_r_upper_bound", "AQUISTIL_R_UPPER_BOUND", None),
    )
    result_values[missing] = np.clip(result_values[missing], lower[missing], upper[missing])

    result = original.copy()
    result.loc[missing, target_column] = result_values[missing]
    result.loc[~missing, target_column] = original.loc[~missing, target_column]
    add_diagnostics = bool(
        _option(kwargs, "aquistil_r_add_diagnostics", "AQUISTIL_R_ADD_DIAGNOSTICS", True)
    )
    if add_diagnostics:
        result[f"{target_column}_AQUISTIL_R_Uncertainty90"] = uncertainty
        result[f"{target_column}_AQUISTIL_R_Confidence"] = confidence
        result[f"{target_column}_AQUISTIL_R_Disagreement"] = disagreement

    if bool(kwargs.get("aquistil_r_save_components", False)):
        for model_index, model_name in enumerate(models):
            token = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
            prediction_column = f"{target_column}_AQUISTIL_R_Pred_{token}"
            weight_column = f"{target_column}_AQUISTIL_R_Weight_{token}"
            result[prediction_column] = np.nan
            result[weight_column] = np.nan
            result.loc[missing, prediction_column] = prediction_matrix[missing, model_index]
            result.loc[missing, weight_column] = used_weights[missing, model_index]

    remaining = pd.to_numeric(result.loc[missing, target_column], errors="coerce").isna().sum()
    if remaining:
        bad_positions = np.flatnonzero(missing)[
            pd.to_numeric(result.loc[missing, target_column], errors="coerce").isna().to_numpy()
        ]
        result.iloc[bad_positions, result.columns.get_loc(target_column)] = fallback[bad_positions]

    logging.info(
        "[%s] regime=%s imputed=%d OOF_rows=%d failed_experts=%s mean_uncertainty=%.4f",
        MODEL_NAME,
        regime,
        int(missing.sum()),
        len(oof),
        sorted(final_failures),
        float(np.nanmean(uncertainty[missing])),
    )
    return result


__all__ = [
    "MODEL_NAME",
    "DEFAULT_BASE_MODELS",
    "StackFit",
    "impute_mice",
]
