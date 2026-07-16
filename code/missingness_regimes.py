import numpy as np
import pandas as pd
import logging

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


# ============================================================
# 1) RANDOM missingness (MCAR)
# ============================================================
def apply_random_missingness(df, target, frac, seed=42):
    rng = np.random.default_rng(seed)

    observed_idx = df[df[target].notna()].index
    n = int(len(observed_idx) * frac)

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
    """
    min_gap, max_gap in HOURS
    """
    rng = np.random.default_rng(seed)
    y = df[target]

    valid_idx = np.where(y.notna())[0]
    total_to_mask = int(len(valid_idx) * frac)

    blocks = _find_valid_blocks(valid_idx, min_gap)

    # If no sufficiently long contiguous blocks exist, try relaxing the
    # minimum block length incrementally (down to 1) before falling back
    # to random masking. This avoids excessive random fallbacks on sparse
    # series while still respecting the intent of 'gap' regimes.
    min_gap_used = min_gap
    if len(blocks) == 0 and min_gap > 1:
        for trial_gap in range(min_gap - 1, 0, -1):
            blocks = _find_valid_blocks(valid_idx, trial_gap)
            if len(blocks) > 0:
                logging.warning(
                    f"No contiguous blocks >= {min_gap} found; using smaller block size {trial_gap} for gap-based masking"
                )
                min_gap_used = trial_gap
                break

    rng.shuffle(blocks)

    masked = []

    # If no sufficiently long contiguous blocks exist, fall back to random
    # masking among available observed indices so the regime still produces
    # a result (useful for sparse datasets where long gaps cannot be formed).
    if len(blocks) == 0:
        logging.warning(
            f"No contiguous blocks >= {min_gap} (or smaller) found; falling back to random masking for frac={frac}"
        )
        chosen = rng.choice(valid_idx, size=min(total_to_mask, len(valid_idx)), replace=False)
        masked = list(chosen)
    else:
        for block in blocks:
            gap_len = rng.integers(min_gap_used, max_gap + 1)
            if len(block) < gap_len:
                continue

            start = rng.integers(0, len(block) - gap_len + 1)
            chosen = block[start:start + gap_len]
            masked.extend(chosen)

            if len(masked) >= total_to_mask:
                break

        # If we didn't reach the required number of masked samples, fill the
        # remainder with random observed indices not already masked.
        if len(masked) < total_to_mask:
            remaining = list(set(valid_idx) - set(masked))
            need = total_to_mask - len(masked)
            if len(remaining) > 0 and need > 0:
                extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
                masked.extend(list(extra))

    # Preserve selection order, remove overlap, and prevent the final block
    # from overshooting the configured missingness count.
    masked = list(dict.fromkeys(masked))[:total_to_mask]

    mask = pd.Series(False, index=df.index)
    if len(masked) > 0:
        mask.iloc[masked] = True
        df.iloc[masked, df.columns.get_loc(target)] = np.nan

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
    """Mask exactly ``frac`` of observed rows, prioritising pollution events.

    Values at or above ``quantile`` form the primary event pool.  If the
    requested fraction is larger than that pool, all event rows are retained
    and the balance is sampled from the remaining observations with
    percentile-rank weights.  Thus a 20% level means 20% of observations, not
    20% of the top decile (the old behaviour was approximately 2%).
    """
    rng = np.random.default_rng(seed)

    # Keep an independent immutable copy for selection diagnostics.  A Series
    # obtained directly from ``df[target]`` may share storage with ``df`` and
    # turn into NaN when the selected rows are masked below.
    y = pd.to_numeric(df[target], errors="coerce").copy(deep=True)
    
    # Debug: check data distribution
    logging.info(
        f"Event regime stats: min={y.min():.2f}, max={y.max():.2f}, "
        f"median={y.median():.2f}, 90th={y.quantile(0.9):.2f}"
    )
    
    thresh = y.quantile(quantile)
    
    # ✅ Handle case where threshold is too high
    if pd.isna(thresh) or thresh == y.max():
        logging.warning(f"Threshold too high ({thresh}), lowering to 80th percentile")
        thresh = y.quantile(0.80)

    observed = y.dropna()
    candidates = observed[observed >= thresh].index
    
    logging.info(f"Event regime:  {len(candidates)} values >= {thresh:.2f}")

    total_to_mask = min(int(len(observed) * frac), len(observed))
    if frac > 0 and total_to_mask == 0 and len(observed):
        total_to_mask = 1

    # If no high-value candidates exist, fall back to selecting random
    # observed indices so the regime still yields masked values.
    if len(candidates) == 0:
        logging.warning("No high values found for event masking; falling back to random masking")
        if len(observed) == 0:
            logging.error("No observed values to mask for event fallback")
            return df, pd.Series(False, index=df.index)
        chosen = rng.choice(observed.index, size=total_to_mask, replace=False)
    else:
        event_count = min(total_to_mask, len(candidates))
        chosen_event = list(rng.choice(candidates, size=event_count, replace=False))
        need = total_to_mask - event_count
        chosen = chosen_event
        if need > 0:
            remainder = observed.drop(index=chosen_event)
            # Strongly favour larger concentrations while retaining stochastic
            # MNAR sampling and enough candidates for large fractions (30/50%).
            percentile = remainder.rank(method="average", pct=True).to_numpy()
            weights = np.exp(4.0 * percentile)
            weights = weights / weights.sum()
            chosen.extend(
                rng.choice(remainder.index, size=min(need, len(remainder)), replace=False, p=weights)
            )

    mask = pd.Series(False, index=df.index)
    mask.loc[chosen] = True
    df.loc[chosen, target] = np.nan

    logging.info(
        "Event regime masked %d/%d observed values (%.2f%%); masked mean=%.2f, "
        "observed mean=%.2f, threshold=%.2f",
        len(chosen), len(observed), 100 * len(chosen) / max(len(observed), 1),
        y.loc[chosen].mean(), observed.mean(), thresh,
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

    elif regime == "short_gap":
        return apply_gap_missingness(df, target, 1, 23, frac, seed)

    elif regime == "medium_gap":
        return apply_gap_missingness(df, target, 24, 71, frac, seed)

    elif regime == "long_gap":
        return apply_gap_missingness(df, target, 72, 240, frac, seed)

    elif regime == "event":
        return apply_event_dependent_missingness(df, target, frac, seed=seed)

    else:
        raise ValueError(f"Unknown missingness regime: {regime}")
