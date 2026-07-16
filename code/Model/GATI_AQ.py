import numpy as np
import pandas as pd
import logging

# Gate model
import lightgbm as lgb

# Candidate imputers (already in your pipeline)
from Model.MissForest import impute_mice as mf_impute
from Model.XGBoost import impute_mice as xgb_impute
from Model.LightGBM import impute_mice as lgb_impute

MODEL_NAME = "GATI_AQ"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------
# Candidate imputers (already in your pipeline)
# ---------------------------------------------------------------------
IMPUTERS = {
    0: ("MissForest", mf_impute),
    1: ("XGBoost", xgb_impute),
    2: ("LightGBM", lgb_impute),
}


# ---------------------------------------------------------------------
# Temporal block masking (NO leakage)
# ---------------------------------------------------------------------
def temporal_block_mask(df, target, block_len, seed=42, n_blocks=20):
    """
    Create a boolean mask of positions to hide in contiguous temporal blocks.
    Only masks positions where target is originally observed (not NaN).
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(df), dtype=bool)

    valid = np.where(df[target].notna().values)[0]
    if len(valid) < block_len:
        return mask

    max_start = max(1, len(df) - block_len - 1)
    starts = rng.integers(0, max_start, size=n_blocks)

    for s in starts:
        block = np.arange(s, s + block_len)
        # Only mask originally non-missing target points
        block = block[df[target].iloc[block].notna().values]
        mask[block] = True

    return mask


# ---------------------------------------------------------------------
# Gate features (cheap, available at runtime)
# ---------------------------------------------------------------------
def build_gate_features(df, target):
    """
    Build gate features that are:
      - cheap (no heavy models)
      - available at runtime
      - informative about missingness regimes (gap length, season/time)
    """
    X = pd.DataFrame(index=df.index)

    # Time features
    dt = pd.to_datetime(df["DateTime"])
    X["hour"] = dt.dt.hour.astype(int)
    X["month"] = dt.dt.month.astype(int)
    X["dow"] = dt.dt.dayofweek.astype(int)

    # Target-based rolling stats (use only past context; ffill/bfill is ok for gating, no label leakage)
    y = pd.to_numeric(df[target], errors="coerce")

    X["roll_mean_24"] = y.rolling(24, min_periods=6).mean()
    X["roll_std_24"] = y.rolling(24, min_periods=6).std()
    X["roll_mean_72"] = y.rolling(72, min_periods=12).mean()
    X["roll_std_72"] = y.rolling(72, min_periods=12).std()

    # Gap-aware features (CRITICAL)
    is_missing = y.isna().astype(int)

    # consecutive missing run-length (counts within each missing segment)
    # For non-missing points this will be 0
    run_id = (is_missing == 0).cumsum()
    X["gap_length"] = is_missing.groupby(run_id).cumcount()
    X["is_missing"] = is_missing

    # Simple regime flags
    X["is_night"] = X["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    X["is_winter"] = X["month"].isin([6, 7, 8]).astype(int)

    # Fill
    X = X.ffill().bfill().fillna(0.0)

    # Ensure numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X


# ---------------------------------------------------------------------
# MAIN ENTRY POINT (called by main.py)
# ---------------------------------------------------------------------
def impute_mice(
    data,
    target_column,
    input_columns,
    custom_strategies=None,
    **kwargs
):
    """
    GATI-AQ: Gated Adaptive Temporal-spatial Imputation for Air Quality

    Key fixes vs your previous version:
      ✅ Gate supervision is BLOCK-LEVEL winner (stable regimes, not point noise)
      ✅ Gate has GAP-AWARE features (missingness geometry)
      ✅ Optional confidence fallback to LightGBM (safer deployment)
    """
    logging.info("🚀 GATI-AQ started")

    df = data.copy()
    target = target_column

    import config_spatial as config
    if getattr(config, "STRICT_PROGRESSIVE_FEATURE_LIST", False):
        logging.info(
            "Strict Stage 3 feature contract active; bypassing GATI gate/context "
            "features and using the exact-list LightGBM path."
        )
        return lgb_impute(df, target, input_columns, **kwargs)

    # Safety checks
    if "DateTime" not in df.columns:
        raise ValueError("GATI_AQ requires a 'DateTime' column for gate features.")

    # If target completely missing -> fallback directly (nothing to learn)
    if df[target].notna().sum() == 0:
        logging.warning("⚠️ Target is completely missing. Fallback → LightGBM imputer.")
        return lgb_impute(df, target, input_columns)

    # ---------------------------------------------------------
    # 1) BENCHMARK UNDER REALISTIC MASKING (BLOCK-LEVEL WINNER)
    # ---------------------------------------------------------
    X_gate_parts = []
    y_gate_parts = []

    gate_features_full = build_gate_features(df, target)

    # Multiple block sizes to learn regimes
    block_sizes = [6, 12, 24, 72]
    seed_base = 42

    for b_i, block in enumerate(block_sizes):
        mask = temporal_block_mask(df, target, block_len=block, seed=seed_base + b_i, n_blocks=20)
        if mask.sum() == 0:
            continue

        df_masked = df.copy()
        y_true = pd.to_numeric(df.loc[mask, target], errors="coerce").values
        df_masked.loc[mask, target] = np.nan

        preds = {}
        for mid, (_, fn) in IMPUTERS.items():
            try:
                out = fn(df_masked.copy(), target, input_columns)
                preds[mid] = pd.to_numeric(out.loc[mask, target], errors="coerce").values
            except Exception as e:
                logging.warning(f"⚠️ Candidate imputer failed during benchmarking: {IMPUTERS[mid][0]}: {e}")
                preds[mid] = np.full_like(y_true, np.nan, dtype=float)

        # BLOCK-LEVEL winner: pick model with lowest block RMSE (stable target)
        rmses = []
        for mid in sorted(preds.keys()):
            err = y_true - preds[mid]
            rmse = np.sqrt(np.nanmean(err ** 2))
            rmses.append(rmse)

        if np.all(np.isnan(rmses)):
            continue

        winner_mid = int(np.nanargmin(rmses))

        # Supervise all rows in the block with the same winner label
        X_case = gate_features_full.loc[mask]
        y_case = np.full(mask.sum(), winner_mid, dtype=int)

        X_gate_parts.append(X_case)
        y_gate_parts.append(y_case)

        logging.info(f"🧪 Block={block}h winner={IMPUTERS[winner_mid][0]} RMSEs={dict(zip([IMPUTERS[k][0] for k in sorted(preds.keys())], rmses))}")

    if len(X_gate_parts) == 0:
        logging.warning("⚠️ No valid benchmarking blocks. Fallback → LightGBM")
        return lgb_impute(df, target, input_columns)

    X_gate = pd.concat(X_gate_parts, axis=0)
    y_gate = np.concatenate(y_gate_parts, axis=0)

    # ---------------------------------------------------------
    # 2) TRAIN SURROGATE GATE
    # ---------------------------------------------------------
    gate = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    gate.fit(X_gate, y_gate)
    logging.info("✅ GATI surrogate trained")

    # ---------------------------------------------------------
    # 3) GATED IMPUTATION (PRODUCTION MODE)
    # ---------------------------------------------------------
    missing_mask = df[target].isna().values
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        logging.info("✅ No missing values found in target. Returning original df.")
        return df

    X_pred = build_gate_features(df, target).loc[missing_mask]

    # Predict probabilities for confidence-aware fallback
    proba = gate.predict_proba(X_pred)
    chosen_models = np.argmax(proba, axis=1).astype(int)
    confidence = np.max(proba, axis=1)

    # Confidence safeguard: if gate is uncertain, fallback to LightGBM (id=2)
    # You can tune this threshold; 0.55–0.65 is typical.
    CONF_THRESH = 0.60
    fallback_id = 2  # LightGBM
    chosen_models_safe = chosen_models.copy()
    chosen_models_safe[confidence < CONF_THRESH] = fallback_id

    # Run only required models
    model_outputs = {}
    for mid in np.unique(chosen_models_safe):
        _, fn = IMPUTERS[int(mid)]
        model_outputs[int(mid)] = fn(df.copy(), target, input_columns)

    # Assign imputed values (robust indexing)
    miss_idx = np.where(missing_mask)[0]
    final_vals = np.empty(n_missing, dtype=float)

    for k, row_idx in enumerate(miss_idx):
        mid = int(chosen_models_safe[k])
        out_df = model_outputs[mid]
        final_vals[k] = pd.to_numeric(out_df.iloc[row_idx][target], errors="coerce")

    # If any NaNs still exist (rare), final fallback to LightGBM output
    if np.isnan(final_vals).any():
        logging.warning("⚠️ Some gated outputs are NaN. Filling remaining with LightGBM.")
        if fallback_id not in model_outputs:
            model_outputs[fallback_id] = lgb_impute(df.copy(), target, input_columns)
        lgb_vals = pd.to_numeric(model_outputs[fallback_id].loc[missing_mask, target], errors="coerce").values
        final_vals = np.where(np.isnan(final_vals), lgb_vals, final_vals)

    df.loc[missing_mask, target] = final_vals

    # Logging usage summary
    uniq, counts = np.unique(chosen_models_safe, return_counts=True)
    usage = {IMPUTERS[int(u)][0]: int(c) for u, c in zip(uniq, counts)}
    logging.info(f"🎯 GATI usage (after confidence fallback): {usage}")
    logging.info(f"🧠 Mean gate confidence: {float(np.mean(confidence)):.3f}, below-thresh: {int((confidence < CONF_THRESH).sum())}/{n_missing}")
    logging.info("🏁 GATI-AQ completed")

    return df
