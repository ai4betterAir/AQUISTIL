import numpy as np
import pandas as pd
import logging
import lightgbm as lgb

MODEL_NAME = "LGBMPlus"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------
# Feature construction (PAST-SAFE only)
# ---------------------------------------------------------------------
def build_features(df, target, input_columns=None, strict=False):
    if strict:
        return df[list(input_columns or [])].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    X = pd.DataFrame(index=df.index)

    if "DateTime" not in df.columns:
        raise ValueError("DateTime column required")

    dt = pd.to_datetime(df["DateTime"])
    X["hour"] = dt.dt.hour
    X["dow"] = dt.dt.dayofweek
    X["month"] = dt.dt.month
    X["is_night"] = X["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    X["is_winter"] = X["month"].isin([6, 7, 8]).astype(int)

    y = pd.to_numeric(df[target], errors="coerce")

    # lags
    X["lag_1"] = y.shift(1)
    X["lag_6"] = y.shift(6)
    X["lag_24"] = y.shift(24)

    # rolling (shifted)
    X["roll_mean_24"] = y.shift(1).rolling(24, min_periods=6).mean()
    X["roll_std_24"] = y.shift(1).rolling(24, min_periods=6).std()

    # gap geometry
    is_missing = y.isna().astype(int)
    run_id = (is_missing == 0).cumsum()
    X["gap_length"] = is_missing.groupby(run_id).cumcount()

    return X.ffill().bfill().fillna(0.0)


# ---------------------------------------------------------------------
# Train a LightGBM model
# ---------------------------------------------------------------------
def train_lgbm(X, y, objective="regression", alpha=None):
    params = dict(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=30,
        random_state=42,
        n_jobs=-1,
    )

    if objective == "quantile":
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            **params
        )
    else:
        model = lgb.LGBMRegressor(
            objective=objective,
            **params
        )

    model.fit(X, y)
    return model


# ---------------------------------------------------------------------
# Main entry point (pipeline-compatible)
# ---------------------------------------------------------------------
def impute_mice(
    data,
    target_column,
    input_columns,
    custom_strategies=None,
    **kwargs
):
    """
    Multi-Objective LightGBM Hybrid for Air Quality Imputation.

    Returns:
        df with target imputed
        optional uncertainty columns if requested
    """

    df = data.copy()
    target = target_column

    y = pd.to_numeric(df[target], errors="coerce")
    if y.notna().sum() == 0:
        logging.warning("Target fully missing — cannot impute")
        return df

    # -------------------------------------------------
    # 1. Features
    # -------------------------------------------------
    import config_spatial as config
    strict = bool(getattr(config, "STRICT_PROGRESSIVE_FEATURE_LIST", False))
    X = build_features(df, target, input_columns, strict=strict)

    observed = y.notna()
    missing = y.isna()

    X_train = X.loc[observed]
    y_train = y.loc[observed]

    # -------------------------------------------------
    # 2. Train LightGBM variants
    # -------------------------------------------------
    logging.info("Training LightGBM (MSE)")
    m_mse = train_lgbm(X_train, y_train, objective="regression")

    logging.info("Training LightGBM (MAE)")
    m_mae = train_lgbm(X_train, y_train, objective="regression_l1")

    logging.info("Training LightGBM (Quantile 0.5)")
    m_q50 = train_lgbm(X_train, y_train,
                       objective="quantile", alpha=0.5)

    # -------------------------------------------------
    # 3. Predict missing values
    # -------------------------------------------------
    if missing.sum() == 0:
        return df

    X_miss = X.loc[missing]

    p_mse = m_mse.predict(X_miss)
    p_mae = m_mae.predict(X_miss)
    p_q50 = m_q50.predict(X_miss)

    # fixed soft blend (no leakage, no tuning)
    y_final = (
        0.60 * p_mse +
        0.25 * p_mae +
        0.15 * p_q50
    )

    df.loc[missing, target] = y_final

    # -------------------------------------------------
    # 4. Optional uncertainty outputs
    # -------------------------------------------------
    if kwargs.get("save_components", False):
        df.loc[missing, f"{target}__mse"] = p_mse
        df.loc[missing, f"{target}__mae"] = p_mae
        df.loc[missing, f"{target}__q50"] = p_q50

    mean_gap = (
        float(X_miss["gap_length"].mean())
        if "gap_length" in X_miss.columns
        else float("nan")
    )
    logging.info(
        f"✅ {MODEL_NAME} completed | "
        f"imputed={missing.sum()} | "
        f"mean_gap={mean_gap:.1f}"
    )

    return df
