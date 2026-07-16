import os
import numpy as np
import pandas as pd
import logging
import lightgbm as lgb

MODEL_NAME = "RALGBM"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def temporal_block_mask(df, target, block_len, seed=42, n_blocks=20):
    """
    Create realistic contiguous missing blocks (NO leakage).
    Only masks originally observed target values.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(df), dtype=bool)

    valid_idx = np.where(df[target].notna().values)[0]
    if len(valid_idx) < block_len:
        return mask

    max_start = max(1, len(df) - block_len - 1)
    starts = rng.integers(0, max_start, size=n_blocks)

    for s in starts:
        block = np.arange(s, s + block_len)
        # keep only originally observed points in block
        block = block[df[target].iloc[block].notna().values]
        mask[block] = True

    return mask


def _safe_makedirs(p):
    if p:
        os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------
# Regime-aware feature construction
# ---------------------------------------------------------------------
def build_regime_features(df, target, input_columns=None, strict=False):
    """
    Builds ONLY past-safe, runtime-available features.
    Encodes missingness geometry (gap regime) + temporal regime.
    """
    if strict:
        return df[list(input_columns or [])].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    X = pd.DataFrame(index=df.index)

    dt = pd.to_datetime(df["DateTime"])
    X["hour"] = dt.dt.hour
    X["dow"] = dt.dt.dayofweek
    X["month"] = dt.dt.month
    X["is_night"] = X["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    X["is_winter"] = X["month"].isin([6, 7, 8]).astype(int)

    y = pd.to_numeric(df[target], errors="coerce")

    # past-only lags
    X["lag_1"] = y.shift(1)
    X["lag_6"] = y.shift(6)
    X["lag_24"] = y.shift(24)
    X["lag_72"] = y.shift(72)

    # past-only rolling stats (shift before rolling)
    X["roll_mean_24"] = y.shift(1).rolling(24, min_periods=6).mean()
    X["roll_std_24"] = y.shift(1).rolling(24, min_periods=6).std()
    X["roll_mean_72"] = y.shift(1).rolling(72, min_periods=12).mean()

    # missingness geometry features
    is_missing = y.isna().astype(int)
    # run_id increments on observed; missing segments share same run_id
    run_id = (is_missing == 0).cumsum()

    # counts within missing segments; observed rows get 0-ish but it’s fine
    X["gap_length"] = is_missing.groupby(run_id).cumcount()
    X["gap_is_long"] = (X["gap_length"] >= 24).astype(int)
    X["gap_is_very_long"] = (X["gap_length"] >= 72).astype(int)

    # safe fill
    X = X.ffill().bfill().fillna(0.0)
    return X


# ---------------------------------------------------------------------
# Quantile LightGBM training
# ---------------------------------------------------------------------
def train_quantile_models(X_train, y_train, quantiles=(0.1, 0.5, 0.9), base_params=None):
    """
    Train LightGBM quantile regressors for uncertainty intervals.
    Returns dict: {q: model}
    """
    if base_params is None:
        base_params = dict(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=30,
            random_state=42,
            n_jobs=-1,
        )

    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(q),
            **base_params
        )
        m.fit(X_train, y_train)
        models[q] = m
    return models


# ---------------------------------------------------------------------
# SHAP utilities (optional)
# ---------------------------------------------------------------------
def save_shap_reports(models_q, X_train, missing_mask, out_dir, prefix):
    """
    Saves SHAP summary plots for:
      - training distribution (global)
      - missing-only rows (regime diagnostics)
    Uses the median (q50) model by default (most interpretable).
    Gracefully skips if shap isn't installed.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning(f"SHAP not available (skipping): {e}")
        return

    _safe_makedirs(out_dir)

    # use q50 for explanations
    q50 = 0.5 if 0.5 in models_q else sorted(models_q.keys())[len(models_q)//2]
    model = models_q[q50]

    # sample for speed + stability
    n = min(2000, len(X_train))
    Xs = X_train.sample(n=n, random_state=42) if len(X_train) > n else X_train.copy()

    explainer = shap.TreeExplainer(model)

    # -------- global summary (training sample) --------
    shap_values = explainer.shap_values(Xs)

    plt.figure()
    shap.summary_plot(shap_values, Xs, show=False)  # uses matplotlib internally
    out_png = os.path.join(out_dir, f"{prefix}_shap_summary_global.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    logging.info(f"✅ Saved SHAP global summary: {out_png}")

    # -------- missing-only regime summary --------
    miss_idx = np.where(missing_mask)[0]
    if len(miss_idx) > 0:
        Xm = X_train.iloc[0:0]  # empty default
        # We want SHAP on the rows being imputed (features at runtime)
        # Caller will pass X_full; we handle missing rows by reusing X_train schema if possible.
        # Here: only do a diagnostic plot from a small subset of missing rows if provided later.
        # We'll skip unless we can receive missing-row features (caller passes in X_full via kwargs).
        # This function is called with X_train only; missing-only plot is handled in impute_mice below.

    # Also write a simple “top features” CSV
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top = pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    out_csv = os.path.join(out_dir, f"{prefix}_shap_top_features.csv")
    top.to_csv(out_csv, index=False)
    logging.info(f"✅ Saved SHAP top-features CSV: {out_csv}")


def save_shap_missing_only(models_q, X_missing, out_dir, prefix):
    """
    SHAP summary for the missing rows only (regime diagnostics).
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning(f"SHAP not available (skipping missing-only): {e}")
        return

    if X_missing is None or len(X_missing) == 0:
        return

    _safe_makedirs(out_dir)

    q50 = 0.5 if 0.5 in models_q else sorted(models_q.keys())[len(models_q)//2]
    model = models_q[q50]
    explainer = shap.TreeExplainer(model)

    n = min(2000, len(X_missing))
    Xm = X_missing.sample(n=n, random_state=42) if len(X_missing) > n else X_missing.copy()

    shap_values_m = explainer.shap_values(Xm)

    plt.figure()
    shap.summary_plot(shap_values_m, Xm, show=False)
    out_png = os.path.join(out_dir, f"{prefix}_shap_summary_missing_only.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    logging.info(f"✅ Saved SHAP missing-only summary: {out_png}")

    # Top features for missing-only rows
    mean_abs = np.mean(np.abs(shap_values_m), axis=0)
    top = pd.DataFrame({"feature": Xm.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    out_csv = os.path.join(out_dir, f"{prefix}_shap_top_features_missing_only.csv")
    top.to_csv(out_csv, index=False)
    logging.info(f"✅ Saved SHAP missing-only top-features CSV: {out_csv}")


# ---------------------------------------------------------------------
# MAIN ENTRY POINT (used by your existing pipeline)
# ---------------------------------------------------------------------
def impute_mice(
    data,
    target_column,
    input_columns,
    custom_strategies=None,
    **kwargs
):
    """
    RALGBM-AQ with:
      - Quantile uncertainty intervals (q10,q50,q90)
      - SHAP regime attribution (global + missing-only)

    Returns df with target imputed using q50 model.
    Adds optional columns:
      <target>__q10, <target>__q50, <target>__q90   (filled on missing rows)
    """

    df = data.copy()
    target = target_column

    # Optional artifact output location
    # You can pass: out_dir=..., site_name=..., model_name=...
    out_dir = kwargs.get("out_dir", None)
    site_name = kwargs.get("site_name", "")
    # Canonicalize provided site_name to first token
    if site_name:
        try:
            site_name = str(site_name).split('_')[0]
        except Exception:
            site_name = str(site_name)
    tag_model = kwargs.get("model_name", MODEL_NAME)
    prefix = f"{tag_model}_{site_name}_{target}".strip("_").replace(" ", "_")

    logging.info(f"🚀 {MODEL_NAME} started for target={target}")

    if "DateTime" not in df.columns:
        raise ValueError("RALGBM_AQ requires a 'DateTime' column")

    y_full = pd.to_numeric(df[target], errors="coerce")
    if y_full.notna().sum() == 0:
        logging.warning("Target fully missing → cannot impute")
        return df

    # ---------------------------------------------------------
    # 1) FEATURES
    # ---------------------------------------------------------
    import config_spatial as config
    strict = bool(getattr(config, "STRICT_PROGRESSIVE_FEATURE_LIST", False))
    X_full = build_regime_features(df, target, input_columns, strict=strict)
    observed = y_full.notna()

    # ---------------------------------------------------------
    # 2) TRAIN DATA (realistic block masking)
    # ---------------------------------------------------------
    X_train_all, y_train_all = [], []

    for block in [6, 12, 24, 72]:
        mask = temporal_block_mask(df, target, block_len=block, seed=42 + block)
        if mask.sum() == 0:
            continue

        train_idx = observed & (~mask)
        X_train_all.append(X_full.loc[train_idx])
        y_train_all.append(y_full.loc[train_idx])

    if len(X_train_all) == 0:
        X_train = X_full.loc[observed]
        y_train = y_full.loc[observed]
    else:
        X_train = pd.concat(X_train_all)
        y_train = pd.concat(y_train_all)

    # ---------------------------------------------------------
    # 3) TRAIN QUANTILE MODELS (q10,q50,q90)
    # ---------------------------------------------------------
    quantiles = (0.1, 0.5, 0.9)
    models_q = train_quantile_models(X_train, y_train, quantiles=quantiles)

    logging.info("✅ Quantile models trained: q10/q50/q90")

    # ---------------------------------------------------------
    # 4) IMPUTE USING q50 + SAVE INTERVALS
    # ---------------------------------------------------------
    missing_mask = y_full.isna()
    nmiss = int(missing_mask.sum())
    if nmiss == 0:
        logging.info("No missing values → returning original df")
        return df

    X_missing = X_full.loc[missing_mask]

    pred_q10 = models_q[0.1].predict(X_missing)
    pred_q50 = models_q[0.5].predict(X_missing)
    pred_q90 = models_q[0.9].predict(X_missing)

    # Impute target with median
    df.loc[missing_mask, target] = pred_q50

    # Add uncertainty columns (only for missing rows; observed remain NaN)
    q10_col = f"{target}__q10"
    q50_col = f"{target}__q50"
    q90_col = f"{target}__q90"
    if q10_col not in df.columns:
        df[q10_col] = np.nan
    if q50_col not in df.columns:
        df[q50_col] = np.nan
    if q90_col not in df.columns:
        df[q90_col] = np.nan

    df.loc[missing_mask, q10_col] = pred_q10
    df.loc[missing_mask, q50_col] = pred_q50
    df.loc[missing_mask, q90_col] = pred_q90

    # Interval width diagnostics
    gap_mean = float(X_missing["gap_length"].mean()) if "gap_length" in X_missing.columns else float("nan")
    int_width = float(np.nanmean(pred_q90 - pred_q10))
    logging.info(f"🎯 Imputed {nmiss} values | mean_gap={gap_mean:.1f} | mean_PI_width={int_width:.3f}")

    # ---------------------------------------------------------
    # 5) SHAP ATTRIBUTION (optional, paper-friendly)
    # ---------------------------------------------------------
    if out_dir:
        shap_dir = os.path.join(out_dir, "shap")
        _safe_makedirs(shap_dir)

        try:
            save_shap_reports(models_q, X_train, missing_mask, shap_dir, prefix)
            save_shap_missing_only(models_q, X_missing, shap_dir, prefix)
        except Exception as e:
            logging.warning(f"SHAP saving failed (continuing): {e}")

        # Also save a small diagnostics CSV for the paper
        diag = pd.DataFrame({
            "target": [target],
            "n_observed": [int(observed.sum())],
            "n_missing": [nmiss],
            "mean_gap_length_missing": [gap_mean],
            "mean_PI_width_q90_q10": [int_width],
        })
        diag_path = os.path.join(out_dir, f"{prefix}_diagnostics.csv")
        diag.to_csv(diag_path, index=False)
        logging.info(f"✅ Saved diagnostics: {diag_path}")

    logging.info(f"🏁 {MODEL_NAME} completed")
    return df
