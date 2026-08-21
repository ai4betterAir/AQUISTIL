import numpy as np
import pandas as pd
import logging

GAP_REGIME_BOUNDS = {
    "short_gap": (1, 23),
    "medium_gap": (24, 71),
    "long_gap": (72, 240),
}

# ============================================================
# Helper: find contiguous blocks
# ============================================================
def _find_valid_blocks(valid_idx, block_len):
    blocks = []
    start = 0
    while start < len(valid_idx):
        end = start
        while end + 1 < len(valid_idx) and valid_idx[end + 1] == valid_idx[end] + 1:
            end += 1
        if end - start + 1 >= block_len:
            blocks.append(valid_idx[start:end + 1])
        start = end + 1
    return blocks


def _contiguous_run_lengths(mask):
    values = np.asarray(mask, dtype=bool)
    if not values.any():
        return np.array([], dtype=int)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False])
    return ends - starts + 1


def mask_gap_lengths(mask):
    """Return each row's enclosing masked-run length, or zero when unmasked."""
    values = np.asarray(mask, dtype=bool)
    result = np.zeros(len(values), dtype=int)
    if not values.any():
        return result
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False])
    for start, end in zip(starts, ends):
        result[start : end + 1] = end - start + 1
    return result


def has_valid_gap_candidate(series, regime):
    """Return True when a target series can host an anchored pure synthetic gap."""
    if regime not in GAP_REGIME_BOUNDS:
        return pd.to_numeric(series, errors="coerce").notna().sum() > 0
    min_gap, _ = GAP_REGIME_BOUNDS[regime]
    valid_idx = np.where(pd.to_numeric(series, errors="coerce").notna())[0]
    return any(len(block) >= min_gap + 2 for block in _find_valid_blocks(valid_idx, min_gap + 2))


def gap_mask_diagnostics(mask, regime, requested_fraction=None, observed_count=None):
    """Summarize whether a simulated gap mask obeys its declared regime."""
    values = np.asarray(mask, dtype=bool)
    lengths = _contiguous_run_lengths(values)
    lower, upper = GAP_REGIME_BOUNDS.get(str(regime), (None, None))
    if lower is None:
        in_regime = np.ones(len(lengths), dtype=bool)
    else:
        in_regime = (lengths >= lower) & (lengths <= upper)
    in_regime_points = int(lengths[in_regime].sum()) if len(lengths) else 0
    masked_count = int(values.sum())
    denominator = int(observed_count) if observed_count is not None else len(values)

    return {
        "Requested_Missingness": (
            float(requested_fraction) if requested_fraction is not None else np.nan
        ),
        "Achieved_Missingness": masked_count / denominator if denominator else 0.0,
        "Number_of_Gaps": int(len(lengths)),
        "Min_Gap": int(lengths.min()) if len(lengths) else 0,
        "Median_Gap": float(np.median(lengths)) if len(lengths) else 0.0,
        "Mean_Gap": float(lengths.mean()) if len(lengths) else 0.0,
        "Max_Gap": int(lengths.max()) if len(lengths) else 0,
        "Isolated_Masked_Points": int(lengths[lengths == 1].sum()),
        "Out_of_Regime_Points": masked_count - in_regime_points,
        "Gap_Purity": in_regime_points / masked_count if masked_count else np.nan,
    }


def effective_gap_mask_diagnostics(
    artificial_mask,
    original_missing_mask,
    regime,
    requested_fraction=None,
    observed_count=None,
):
    """Summarize artificial gaps and their effective lengths after natural gaps."""
    artificial = np.asarray(artificial_mask, dtype=bool)
    original = np.asarray(original_missing_mask, dtype=bool)
    if len(artificial) != len(original):
        raise ValueError("artificial_mask and original_missing_mask must have equal length")

    effective = original | artificial
    artificial_lengths = mask_gap_lengths(artificial)
    effective_lengths = mask_gap_lengths(effective)
    selected_artificial = artificial_lengths[artificial]
    selected_effective = effective_lengths[artificial]
    lower, upper = GAP_REGIME_BOUNDS.get(str(regime), (None, None))
    masked_count = int(artificial.sum())
    denominator = int(observed_count) if observed_count is not None else int((~original).sum())

    if lower is None:
        artificial_in_regime = np.ones(len(selected_artificial), dtype=bool)
        effective_in_regime = np.ones(len(selected_effective), dtype=bool)
    else:
        artificial_in_regime = (
            (selected_artificial >= lower) & (selected_artificial <= upper)
        )
        effective_in_regime = (
            (selected_effective >= lower) & (selected_effective <= upper)
        )

    def _stat(values, fn, default=0):
        return fn(values) if len(values) else default

    diagnostics = {
        "Requested_Missingness": (
            float(requested_fraction) if requested_fraction is not None else np.nan
        ),
        "Achieved_Missingness": masked_count / denominator if denominator else 0.0,
        "N_Observed_Before_Masking": int(denominator),
        "N_Artificially_Masked": masked_count,
        "Number_of_Artificial_Gaps": int(len(_contiguous_run_lengths(artificial))),
        "Min_Artificial_Gap": int(_stat(selected_artificial, np.min)),
        "Median_Artificial_Gap": float(_stat(selected_artificial, np.median, 0.0)),
        "Mean_Artificial_Gap": float(_stat(selected_artificial, np.mean, 0.0)),
        "Max_Artificial_Gap": int(_stat(selected_artificial, np.max)),
        "Min_Effective_Gap": int(_stat(selected_effective, np.min)),
        "Median_Effective_Gap": float(_stat(selected_effective, np.median, 0.0)),
        "Mean_Effective_Gap": float(_stat(selected_effective, np.mean, 0.0)),
        "Max_Effective_Gap": int(_stat(selected_effective, np.max)),
        "Isolated_Masked_Points": int((selected_artificial == 1).sum()),
        "Out_of_Regime_Artificial_Points": int((~artificial_in_regime).sum()),
        "Out_of_Regime_Effective_Points": int((~effective_in_regime).sum()),
        "Artificial_Gap_Purity": (
            float(artificial_in_regime.mean()) if masked_count else np.nan
        ),
        "Effective_Gap_Purity": (
            float(effective_in_regime.mean()) if masked_count else np.nan
        ),
    }
    diagnostics.update(
        {
            # Backwards-compatible names used by existing outputs.
            "Number_of_Gaps": diagnostics["Number_of_Artificial_Gaps"],
            "Min_Gap": diagnostics["Min_Artificial_Gap"],
            "Median_Gap": diagnostics["Median_Artificial_Gap"],
            "Mean_Gap": diagnostics["Mean_Artificial_Gap"],
            "Max_Gap": diagnostics["Max_Artificial_Gap"],
            "Out_of_Regime_Points": diagnostics["Out_of_Regime_Effective_Points"],
            "Gap_Purity": diagnostics["Effective_Gap_Purity"],
        }
    )
    return diagnostics


def assert_effective_gap_purity(artificial_mask, original_missing_mask, regime, atol=1e-12):
    """Fail fast when a pure synthetic gap mask merges with natural missingness."""
    if regime not in GAP_REGIME_BOUNDS:
        return
    diagnostics = effective_gap_mask_diagnostics(artificial_mask, original_missing_mask, regime)
    purity = diagnostics["Effective_Gap_Purity"]
    if diagnostics["N_Artificially_Masked"] and not np.isclose(purity, 1.0, atol=atol):
        raise AssertionError(
            "%s effective gap purity is %.6f; diagnostics=%s"
            % (regime, purity, diagnostics)
        )


# ============================================================
# 1) RANDOM missingness (MCAR)
# ============================================================
def apply_random_missingness(df, target, frac, seed=42):
    rng = np.random.default_rng(seed)

    observed_idx = df[df[target].notna()].index
    n = int(round(len(observed_idx) * frac))

    chosen = rng.choice(observed_idx, size=n, replace=False)

    mask = pd.Series(False, index=df.index)
    mask.loc[chosen] = True

    df.loc[chosen, target] = np.nan
    return df, mask


# ============================================================
# 2–4) GAP-BASED missingness
# ============================================================
def apply_gap_missingness(
    df,
    target,
    min_gap,
    max_gap,
    frac,
    seed=42
):
    """Mask non-overlapping, regime-pure gaps without random supplementation."""
    rng = np.random.default_rng(seed)
    y = df[target]
    original_missing = pd.to_numeric(y, errors="coerce").isna().to_numpy()

    valid_idx = np.where(y.notna())[0]
    total_to_mask = int(round(len(valid_idx) * frac))

    if total_to_mask <= 0 or not len(valid_idx):
        return df, pd.Series(False, index=df.index)

    # Available observed segments include one anchor observation on both sides
    # of every candidate gap. Synthetic gaps therefore cannot touch natural
    # missingness at either edge of an observed block.
    available = [
        np.asarray(block, dtype=int)
        for block in _find_valid_blocks(valid_idx, min_gap + 2)
    ]
    masked = []
    while available:
        viable = [i for i, block in enumerate(available) if len(block) >= min_gap + 2]
        if not viable:
            break

        remaining = total_to_mask - len(masked)
        if remaining <= 0:
            break

        capacities = np.asarray(
            [max(min(len(available[i]) - 2, max_gap) - min_gap + 1, 1) for i in viable],
            dtype=float,
        )
        block_pos = viable[int(rng.choice(len(viable), p=capacities / capacities.sum()))]
        block = available.pop(block_pos)
        largest = min(max_gap, len(block) - 2)

        if remaining < min_gap:
            # Add a whole minimum-size gap only when that is closer to the
            # requested count than stopping below target.
            if abs(remaining - min_gap) >= abs(remaining):
                break
            gap_len = min_gap
        elif remaining <= largest:
            gap_len = remaining
        else:
            gap_len = int(rng.integers(min_gap, largest + 1))

        start = int(rng.integers(1, len(block) - gap_len))
        masked.extend(block[start : start + gap_len].tolist())

        left = block[: max(start - 1, 0)]
        right = block[min(start + gap_len + 1, len(block)) :]
        if len(left) >= min_gap + 2:
            available.append(left)
        if len(right) >= min_gap + 2:
            available.append(right)

    mask = pd.Series(False, index=df.index)
    if len(masked) > 0:
        mask.iloc[masked] = True
        df.iloc[masked, df.columns.get_loc(target)] = np.nan

    regime = next(
        (name for name, bounds in GAP_REGIME_BOUNDS.items() if bounds == (min_gap, max_gap)),
        "gap",
    )
    diagnostics = effective_gap_mask_diagnostics(
        mask,
        original_missing,
        regime,
        requested_fraction=frac,
        observed_count=len(valid_idx),
    )
    assert_effective_gap_purity(mask, original_missing, regime)
    logging.info(
        "%s requested=%.2f%% achieved=%.2f%% gaps=%d artificial_range=%d-%d effective_range=%d-%d effective_purity=%.3f",
        regime,
        100 * frac,
        100 * diagnostics["Achieved_Missingness"],
        diagnostics["Number_of_Artificial_Gaps"],
        diagnostics["Min_Artificial_Gap"],
        diagnostics["Max_Artificial_Gap"],
        diagnostics["Min_Effective_Gap"],
        diagnostics["Max_Effective_Gap"],
        diagnostics["Effective_Gap_Purity"],
    )

    return df, mask


# ============================================================
# 5) EVENT-DEPENDENT missingness (MNAR)
# ============================================================
def apply_event_dependent_missingness(
    df,
    target,
    frac,
    quantile=0.90,
    seed=42
):
    """Mask daily-maximum pollution events only.

    For the event regime, one candidate event is defined per day: the observed
    timestamp where ``target`` reaches that day's maximum. ``frac`` is applied
    to this daily-event pool, so 10% masks roughly 10% of daily maxima, 50%
    masks roughly half of daily maxima, and non-event rows are not used to fill
    the quota.
    """
    rng = np.random.default_rng(seed)

    # Keep an independent immutable copy for selection diagnostics.  A Series
    # obtained directly from ``df[target]`` may share storage with ``df`` and
    # turn into NaN when the selected rows are masked below.
    y = pd.to_numeric(df[target], errors="coerce").copy(deep=True)

    observed = y.dropna()
    if len(observed) == 0:
        logging.error("No observed values to mask for event regime")
        return df, pd.Series(False, index=df.index)
    if "DateTime" not in df.columns:
        raise ValueError("Event missingness requires a DateTime column to select daily maxima")

    timestamps = pd.to_datetime(df["DateTime"], errors="coerce")
    event_frame = pd.DataFrame({"Date": timestamps.dt.date, "Value": y}, index=df.index)
    event_frame = event_frame.loc[event_frame["Value"].notna() & event_frame["Date"].notna()]
    if event_frame.empty:
        logging.error("No dated observed values to mask for event regime")
        return df, pd.Series(False, index=df.index)

    # idxmax returns one event row per day. If several hours tie for the daily
    # maximum, the first timestamp in the current sort order is used.
    candidates = event_frame.groupby("Date", sort=False)["Value"].idxmax().to_numpy()

    total_to_mask = min(int(len(candidates) * frac), len(candidates))
    if frac > 0 and total_to_mask == 0 and len(candidates):
        total_to_mask = 1

    chosen = list(rng.choice(candidates, size=total_to_mask, replace=False))

    mask = pd.Series(False, index=df.index)
    mask.loc[chosen] = True
    df.loc[chosen, target] = np.nan

    logging.info(
        "Event regime masked %d/%d daily maxima (%.2f%% of event days); "
        "masked mean=%.2f, observed mean=%.2f, daily max pool mean=%.2f",
        len(chosen), len(candidates), 100 * len(chosen) / max(len(candidates), 1),
        y.loc[chosen].mean(), observed.mean(), y.loc[candidates].mean(),
    )

    return df, mask


# ============================================================
# MASTER DISPATCH FUNCTION
# ============================================================
def apply_missingness(
    df,
    target,
    regime,
    frac,
    seed=42
):
    """
    regime ∈ {
        'random',
        'short_gap',
        'medium_gap',
        'long_gap',
        'event'
    }
    """

    df = df.copy()

    # Supported regime names
    SUPPORTED_REGIMES = ['random', 'short_gap', 'medium_gap', 'long_gap', 'event']

    # If a list/tuple of regimes is provided, return a dict of results per regime
    if isinstance(regime, (list, tuple)):
        results = {}
        for r in regime:
            if r not in SUPPORTED_REGIMES:
                raise ValueError(f"Unknown missingness regime: {r}")
            results[r] = apply_missingness(df.copy(), target, r, frac, seed)
        return results

    # Special keyword to run all regimes and return a mapping
    if regime in (None, 'all', 'ALL'):
        results = {}
        for r in SUPPORTED_REGIMES:
            results[r] = apply_missingness(df.copy(), target, r, frac, seed)
        return results

    # Single regime dispatch (backwards-compatible)
    if regime == "random":
        return apply_random_missingness(df, target, frac, seed)

    elif regime in GAP_REGIME_BOUNDS:
        min_gap, max_gap = GAP_REGIME_BOUNDS[regime]
        return apply_gap_missingness(df, target, min_gap, max_gap, frac, seed)

    elif regime == "event":
        return apply_event_dependent_missingness(df, target, frac, seed=seed)

    else:
        raise ValueError(f"Unknown missingness regime: {regime}")
