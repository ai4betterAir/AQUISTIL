import os
import numpy as np
import pandas as pd
import logging

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Base models (your existing modules)
from Model.LightGBM import impute_mice as lgb_impute
from Model.XGBoost import impute_mice as xgb_impute
from Model.MissForest import impute_mice as mf_impute

MODEL_NAME = "SuperLearner"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------
# Masking: contiguous temporal blocks (NO leakage; masked points have ground truth)
# ---------------------------------------------------------------------
def temporal_block_mask(df, target, block_len, seed=42, n_blocks=20):
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(df), dtype=bool)

    valid = np.where(pd.to_numeric(df[target], errors="coerce").notna().values)[0]
    if len(valid) < block_len:
        return mask

    max_start = max(1, len(df) - block_len - 1)
    starts = rng.integers(0, max_start, size=n_blocks)

    for s in starts:
        block = np.arange(s, s + block_len)
        # only mask originally observed values
        block = block[pd.to_numeric(df[target].iloc[block], errors="coerce").notna().values]
        mask[block] = True

    return mask


# ---------------------------------------------------------------------
# Regime/context features (cheap)
# ---------------------------------------------------------------------
def build_context_features(df, target):
    """
    Context features available at runtime.
    Uses only past-safe target history for lags/rolling.
    """
    X = pd.DataFrame(index=df.index)

    dt = pd.to_datetime(df["DateTime"])
    X["hour"] = dt.dt.hour.astype(int)
    X["dow"] = dt.dt.dayofweek.astype(int)
    X["month"] = dt.dt.month.astype(int)
    X["is_night"] = X["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    X["is_winter"] = X["month"].isin([6, 7, 8]).astype(int)

    y = pd.to_numeric(df[target], errors="coerce")

    # past-safe lags
    X["lag_1"] = y.shift(1)
    X["lag_6"] = y.shift(6)
    X["lag_24"] = y.shift(24)

    # past-safe rolling stats
    X["roll_mean_24"] = y.shift(1).rolling(24, min_periods=6).mean()
    X["roll_std_24"] = y.shift(1).rolling(24, min_periods=6).std()

    # missingness geometry (for real missing, this is meaningful)
    is_missing = y.isna().astype(int)
    run_id = (is_missing == 0).cumsum()
    X["gap_length"] = is_missing.groupby(run_id).cumcount()
    X["gap_is_long"] = (X["gap_length"] >= 24).astype(int)

    return X.ffill().bfill().fillna(0.0)


# ---------------------------------------------------------------------
# Main: Residual Super Learner
# ---------------------------------------------------------------------
def impute_mice(
    data: pd.DataFrame,
    target_column: str,
    input_columns: list,
    custom_strategies=None,
    **kwargs
):
    """
    Residual Super Learner (Option A):
      - Run base imputers (LGB, XGB, MissForest) internally
      - Train a Ridge meta-model on *masked block* positions:
          target residual = y_true - y_lgb_pred
          features include base predictions + context regime features
      - Predict correction for real missing rows:
          y_hat = y_lgb + correction
    """

    df = data.copy()
    target = target_column

    import config_spatial as config
    if getattr(config, "STRICT_PROGRESSIVE_FEATURE_LIST", False):
        logging.info(
            "Strict Stage 3 feature contract active; bypassing SuperLearner "
            "context/meta features and using the exact-list LightGBM path."
        )
        return lgb_impute(df, target, input_columns, **kwargs)

    if "DateTime" not in df.columns:
        raise ValueError("SuperLearner_AQ requires a 'DateTime' column")

    y = pd.to_numeric(df[target], errors="coerce")
    if y.notna().sum() == 0:
        logging.warning("Target fully missing; cannot train. Falling back to LightGBM.")
        return lgb_impute(df, target, input_columns)

    # Optional artifact output directory
    out_dir = kwargs.get("out_dir", None)
    site_name = kwargs.get("site_name", "")
    # Canonicalize site_name to first token (e.g., CHULLORA_AQMS_Processed -> CHULLORA)
    if site_name:
        try:
            site_name = str(site_name).split('_')[0]
        except Exception:
            site_name = str(site_name)
    prefix = f"{MODEL_NAME}_{site_name}_{target}".strip("_").replace(" ", "_")
    if out_dir:
        pass

    # -----------------------------------------------------------------
    # 1) Create training data for the meta-model via block masking
    #    (out-of-sample targets)
    # -----------------------------------------------------------------
    blocks = kwargs.get("block_sizes", [6, 12, 24, 72])
    n_blocks = int(kwargs.get("n_blocks", 20))
    seed_base = int(kwargs.get("seed", 42))

    X_meta_parts = []
    y_meta_parts = []

    # We train correction for LGB baseline
    # Correction target: resid = y_true - y_lgb_pred
    # Meta features: [yhat_lgb, yhat_xgb, yhat_mf] + context + block_len + pos_in_block
    for bi, blen in enumerate(blocks):
        mask = temporal_block_mask(df, target, block_len=blen, seed=seed_base + bi, n_blocks=n_blocks)
        if mask.sum() == 0:
            continue

        df_masked = df.copy()
        y_true = pd.to_numeric(df.loc[mask, target], errors="coerce").values
        df_masked.loc[mask, target] = np.nan

        # Run base models ONCE for this masked dataset
        lgb_out = lgb_impute(df_masked.copy(), target, input_columns)
        xgb_out = xgb_impute(df_masked.copy(), target, input_columns)
        mf_out = mf_impute(df_masked.copy(), target, input_columns)

        yhat_lgb = pd.to_numeric(lgb_out.loc[mask, target], errors="coerce").values
        yhat_xgb = pd.to_numeric(xgb_out.loc[mask, target], errors="coerce").values
        yhat_mf  = pd.to_numeric(mf_out.loc[mask, target], errors="coerce").values

        # Build context features from the MASKED frame (so gap geometry matches training regime)
        ctx = build_context_features(df_masked, target).loc[mask].reset_index(drop=True)

        # Add block metadata features (very informative)
        # position within block: approximate using gap_length computed in ctx
        # (deep in gap vs edge)
        ctx["block_len"] = float(blen)
        ctx["pos_in_block"] = np.minimum(ctx["gap_length"].values.astype(float), float(blen))

        # Assemble meta features
        X_meta = pd.concat(
            [
                pd.DataFrame({
                    "pred_lgb": yhat_lgb,
                    "pred_xgb": yhat_xgb,
                    "pred_mf":  yhat_mf,
                }),
                ctx.reset_index(drop=True)
            ],
            axis=1
        )

        # Target residual (only where both are finite)
        resid = y_true - yhat_lgb

        X_meta_parts.append(X_meta)
        y_meta_parts.append(resid)

        rmse_lgb = float(np.sqrt(np.nanmean((y_true - yhat_lgb) ** 2)))
        rmse_xgb = float(np.sqrt(np.nanmean((y_true - yhat_xgb) ** 2)))
        rmse_mf  = float(np.sqrt(np.nanmean((y_true - yhat_mf) ** 2)))
        logging.info(f"[SuperLearner train] block={blen}h  RMSE(LGB/XGB/MF)=({rmse_lgb:.3f}/{rmse_xgb:.3f}/{rmse_mf:.3f})  n_masked={mask.sum()}")

    if len(X_meta_parts) == 0:
        logging.warning("No valid masked blocks for training. Falling back to LightGBM.")
        return lgb_impute(df, target, input_columns)

    X_meta_train = pd.concat(X_meta_parts, axis=0)
    y_meta_train = np.concatenate([np.asarray(v) for v in y_meta_parts], axis=0)

    # Remove non-finite rows (rare)
    m = np.isfinite(y_meta_train)
    for c in X_meta_train.columns:
        X_meta_train[c] = pd.to_numeric(X_meta_train[c], errors="coerce")
    X_meta_train = X_meta_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_meta_train = X_meta_train.loc[m].reset_index(drop=True)
    y_meta_train = y_meta_train[m]

    # -----------------------------------------------------------------
    # 2) Train a *regularised* correction model (Ridge)
    #    This avoids variance explosion and model switching noise.
    # -----------------------------------------------------------------
    ridge_alpha = float(kwargs.get("ridge_alpha", 10.0))  # larger = more conservative corrections
    meta_model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=ridge_alpha, random_state=42))
    ])
    meta_model.fit(X_meta_train, y_meta_train)
    logging.info(f"✅ SuperLearner meta-model trained (Ridge alpha={ridge_alpha}) on n={len(X_meta_train)} samples")

    # Optionally save coefficients for debugging/paper
    if out_dir:
        try:
            coef = meta_model.named_steps["ridge"].coef_
            feat = X_meta_train.columns.tolist()
            coef_df = pd.DataFrame({"feature": feat, "coef": coef}).sort_values("coef", ascending=False)
            coef_path = os.path.join(out_dir, f"{prefix}_ridge_coefficients.csv")
            coef_df.to_csv(coef_path, index=False)
            logging.info(f"✅ Saved ridge coefficients: {coef_path}")
        except Exception as e:
            logging.warning(f"Could not save ridge coefficients: {e}")

    # -----------------------------------------------------------------
    # 3) Apply to real missing values
    # -----------------------------------------------------------------
    missing_mask = y.isna().values
    if missing_mask.sum() == 0:
        return df

    # Base predictions on the real dataset
    lgb_full = lgb_impute(df.copy(), target, input_columns)
    xgb_full = xgb_impute(df.copy(), target, input_columns)
    mf_full  = mf_impute(df.copy(), target, input_columns)

    yhat_lgb_m = pd.to_numeric(lgb_full.loc[missing_mask, target], errors="coerce").values
    yhat_xgb_m = pd.to_numeric(xgb_full.loc[missing_mask, target], errors="coerce").values
    yhat_mf_m  = pd.to_numeric(mf_full.loc[missing_mask, target], errors="coerce").values

    ctx_m = build_context_features(df, target).loc[missing_mask].reset_index(drop=True)

    # For real missing values we don’t know block_len; we approximate using gap_length
    # (use gap_length as proxy and cap it)
    ctx_m["block_len"] = np.clip(ctx_m["gap_length"].values.astype(float), 0.0, 168.0)  # cap at 7 days
    ctx_m["pos_in_block"] = ctx_m["gap_length"].values.astype(float)

    X_meta_pred = pd.concat(
        [
            pd.DataFrame({
                "pred_lgb": yhat_lgb_m,
                "pred_xgb": yhat_xgb_m,
                "pred_mf":  yhat_mf_m,
            }),
            ctx_m
        ],
        axis=1
    )

    # align/clean
    for c in X_meta_train.columns:
        if c not in X_meta_pred.columns:
            X_meta_pred[c] = 0.0
    X_meta_pred = X_meta_pred[X_meta_train.columns].copy()
    X_meta_pred = X_meta_pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    correction = meta_model.predict(X_meta_pred)

    # Final prediction = LGB + correction (anchored)
    y_final = yhat_lgb_m + correction

    # Optional safety clamp: prevent insane corrections (rare)
    clamp = float(kwargs.get("correction_clamp", 5.0))  # in PM units; tune
    y_final = np.where(np.abs(correction) > clamp, yhat_lgb_m, y_final)

    df.loc[missing_mask, target] = y_final

    # Optional extra columns for analysis
    if kwargs.get("save_components", False):
        df.loc[missing_mask, f"{target}__pred_lgb"] = yhat_lgb_m
        df.loc[missing_mask, f"{target}__pred_xgb"] = yhat_xgb_m
        df.loc[missing_mask, f"{target}__pred_mf"] = yhat_mf_m
        df.loc[missing_mask, f"{target}__corr"] = correction

    logging.info(
        f"🎯 SuperLearner completed: imputed {missing_mask.sum()} values "
        f"(mean |corr|={float(np.mean(np.abs(correction))):.3f})"
    )
    return df
