import numpy as np
import pandas as pd
import logging
import lightgbm as lgb

MODEL_NAME = "LGBM_AQ_Plus_Adaptive"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------
# Adaptive feature construction
# ---------------------------------------------------------------------
def build_features(df, target, station_col=None, input_columns=None, strict=False):
    if strict:
        return df[list(input_columns or [])].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    X = pd.DataFrame(index=df.index)

    if "DateTime" not in df.columns:
        raise ValueError("DateTime column required")

    dt = pd.to_datetime(df["DateTime"])
    X["hour"] = dt.dt.hour
    X["dow"] = dt.dt.dayofweek
    X["month"] = dt.dt.month
    X["is_night"] = X["hour"].between(0, 5).astype(int)
    X["is_winter"] = X["month"].isin([6, 7, 8]).astype(int)

    y = pd.to_numeric(df[target], errors="coerce")

    # ------------------------------
    # Lag features (safe)
    # ------------------------------
    X["lag_1"] = y.shift(1)
    X["lag_6"] = y.shift(6)
    X["lag_24"] = y.shift(24)

    X["roll_mean_24"] = y.shift(1).rolling(24, min_periods=6).mean()
    X["roll_std_24"] = y.shift(1).rolling(24, min_periods=6).std()

    # ------------------------------
    # Gap geometry
    # ------------------------------
    is_missing = y.isna().astype(int)
    run_id = (is_missing == 0).cumsum()
    X["gap_length"] = is_missing.groupby(run_id).cumcount()
    X["gap_is_long"] = (X["gap_length"] >= 24).astype(int)
    X["gap_is_very_long"] = (X["gap_length"] >= 72).astype(int)

    # ------------------------------
    # OPTIONAL: Spatial features
    # ------------------------------
    if station_col and station_col in df.columns:
        logging.info("🧭 Multi-station mode detected — adding spatial features")

        # Safe aggregation: same time, other stations
        grp = df.groupby("DateTime")[target]

        X["neighbor_mean"] = grp.transform("mean")
        X["neighbor_median"] = grp.transform("median")
        X["neighbor_max"] = grp.transform("max")

        # Avoid self-leakage dominance
        X["neighbor_mean"] = X["neighbor_mean"].where(
            df[target].isna(), np.nan
        )

    return X.ffill().bfill().fillna(0.0)


# ---------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------
def train_lgbm(X, y):
    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=30,
        random_state=42,
        n_jobs=-1,
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
    Adaptive LightGBM imputer:
      - Single-site or multi-site automatically
      - Long-gap robust
      - No gating, no superlearner instability
    """

    df = data.copy()
    target = target_column

    station_col = kwargs.get("station_col", None)

    y = pd.to_numeric(df[target], errors="coerce")
    if y.notna().sum() == 0:
        logging.warning("Target fully missing — cannot impute")
        return df

    # ------------------------------
    # Feature build
    # ------------------------------
    import config_spatial as config
    strict = bool(getattr(config, "STRICT_PROGRESSIVE_FEATURE_LIST", False))
    X = build_features(df, target, station_col=station_col, input_columns=input_columns, strict=strict)

    observed = y.notna()
    missing = y.isna()

    X_train = X.loc[observed]
    y_train = y.loc[observed]

    # ------------------------------
    # Train model
    # ------------------------------
    logging.info("Training adaptive LightGBM imputer")
    model = train_lgbm(X_train, y_train)

    # ------------------------------
    # Predict missing
    # ------------------------------
    if missing.sum() == 0:
        return df

    X_miss = X.loc[missing]
    y_pred = model.predict(X_miss)

    df.loc[missing, target] = y_pred

    # ``gap_length`` is an optional engineered feature.  In strict Stage 3
    # mode the selected feature contract intentionally excludes it, so it
    # must not be required merely for a completion log message.
    mean_gap = (
        "{:.1f}".format(X_miss["gap_length"].mean())
        if "gap_length" in X_miss.columns
        else "not-selected"
    )

    logging.info(
        f"✅ {MODEL_NAME} completed | "
        f"imputed={missing.sum()} | "
        f"mean_gap={mean_gap}"
    )

    return df
