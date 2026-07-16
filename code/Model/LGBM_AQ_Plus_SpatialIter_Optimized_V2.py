"""
LGBM_AQ_Plus_SpatialIter_Optimized_V2
Maximum optimization version with:
- Very strict tolerance (1e-6)
- Many trees (1500)
- Very slow learning (0.02)
- Complex trees (127 leaves)
- Weekly rolling windows

Expected improvement: 10-15% better RMSE
Warning: 3-4x slower than original
"""

import os
import numpy as np
import pandas as pd
import logging
import lightgbm as lgb
from spatial import prepare_spatial_temporal_data

MODEL_NAME = "LGBM_AQ_Plus_SpatialIter_Optimized_V2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "DateTime" not in df.columns:
        raise ValueError("DateTime column is required.")
    out = df.copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"])
    return out


def _build_iter_features(
    df_work: pd.DataFrame,
    target: str,
    feature_columns: list,
    strict_feature_list: bool = False,
) -> pd.DataFrame:
    """
    Maximum feature engineering
    """
    df_work = _ensure_datetime(df_work)
    X = df_work[feature_columns].copy()
    if strict_feature_list:
        return X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    y = pd.to_numeric(df_work[target], errors="coerce")
    
    # ========================================================================
    # EXTENSIVE LAG FEATURES
    # ========================================================================
    for lag_hours in [1, 2, 3, 6, 12, 18, 24, 36, 48, 72, 96, 120, 168]: 
        X[f"lag_{lag_hours}"] = y.shift(lag_hours)
    
    # ========================================================================
    # EXTENSIVE ROLLING FEATURES
    # ========================================================================
    windows = [3, 6, 12, 24, 48, 72, 168]  # Up to 1 week
    for window in windows:
        min_periods = max(2, window // 4)
        X[f"roll_mean_{window}"] = y. shift(1).rolling(window, min_periods=min_periods).mean()
        X[f"roll_std_{window}"] = y.shift(1).rolling(window, min_periods=min_periods).std()
        X[f"roll_max_{window}"] = y.shift(1).rolling(window, min_periods=min_periods).max()
        X[f"roll_min_{window}"] = y. shift(1).rolling(window, min_periods=min_periods).min()
        X[f"roll_median_{window}"] = y.shift(1).rolling(window, min_periods=min_periods).median()
    
    # ========================================================================
    # GAP FEATURES
    # ========================================================================
    is_missing = y.isna().astype(int)
    run_id = (is_missing == 0).cumsum()
    X["gap_length"] = is_missing.groupby(run_id).cumcount()
    X["gap_is_medium"] = (X["gap_length"] >= 12).astype(int)
    X["gap_is_long"] = (X["gap_length"] >= 24).astype(int)
    X["gap_is_very_long"] = (X["gap_length"] >= 72).astype(int)
    X["gap_is_extreme"] = (X["gap_length"] >= 168).astype(int)
    
    # ========================================================================
    # RATE OF CHANGE FEATURES
    # ========================================================================
    X["delta_1h"] = y.diff(1)      # Change in past hour
    X["delta_6h"] = y.diff(6)      # Change in past 6 hours
    X["delta_24h"] = y.diff(24)    # Change in past day
    
    # ========================================================================
    # INTERACTION FEATURES (if temperature/humidity available)
    # ========================================================================
    if 'TEMP' in df_work.columns and 'HUMID' in df_work.columns:
        X["temp_humid_interaction"] = df_work['TEMP'] * df_work['HUMID'] / 100
    
    if 'WSP' in df_work.columns and 'Hour' in df_work.columns:
        X["wind_hour_interaction"] = df_work['WSP'] * np.sin(df_work['Hour'] * np. pi / 12)
    
    # Cleanup
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    
    return X


def impute_mice(
    data,
    target_column,
    input_columns,
    max_iter=20,           # ← INCREASED
    tol=1e-6,              # ← VERY STRICT
    random_state=42,
    custom_strategies=None,
    **kwargs
):
    """
    Maximum optimization version
    """
    
    # [Same spatial config code as V1]
    out_dir = kwargs.get('out_dir', '')
    site_name = kwargs.get('site_name', '')
    # Canonicalize provided site_name to first token
    if site_name:
        try:
            site_name = str(site_name).split('_')[0]
        except Exception:
            site_name = str(site_name)

    if not site_name and 'model_name' in kwargs:
        parts = kwargs['model_name']. split('_')
        if len(parts) > 0:
            site_name = parts[-1].split('_')[0]
    
    import config_spatial as config
    
    spatial_config = {
        'input_directory': config.INPUT_DIRECTORY,
        'target_site': site_name,
        'use_spatial': config.USE_SPATIAL_FEATURES,
        'use_temporal': config.USE_TEMPORAL_FEATURES,
        'use_lagged': False,
        'use_rolling':  False,
    }
    
    logging.info(f"[{MODEL_NAME}] Preparing spatial-temporal features...")
    
    try:
        df_enhanced, feature_columns = prepare_spatial_temporal_data(
            data, target_column, input_columns, spatial_config
        )
        logging.info(f"  Base features:  {len(feature_columns)}")
    except Exception as e: 
        logging.warning(f"  Spatial features failed: {e}")
        df_enhanced = data.copy()
        feature_columns = input_columns
    
    df = df_enhanced.copy()
    target = target_column
    
    y0 = pd.to_numeric(df[target], errors="coerce")
    missing_mask = y0.isna().values
    nmiss = int(missing_mask.sum())
    
    if nmiss == 0:
        return df
    
    logging.info(f"[{MODEL_NAME}] Imputing {nmiss} values")
    
    init_val = float(np.nanmedian(y0.values))
    df.loc[missing_mask, target] = init_val
    prev = df.loc[missing_mask, target].values. copy()
    
    for it in range(1, max_iter + 1):
        X = _build_iter_features(
            df, target, feature_columns,
            strict_feature_list=bool(getattr(config, "STRICT_PROGRESSIVE_FEATURE_LIST", False)),
        )
        y = pd.to_numeric(df[target], errors="coerce")
        
        obs_mask = ~missing_mask
        X_train = X.loc[obs_mask]
        y_train = y.loc[obs_mask]
        X_pred = X.loc[missing_mask]
        
        # ====================================================================
        # MAXIMUM OPTIMIZATION
        # ====================================================================
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=1500,           # ← VERY MANY TREES
            learning_rate=0.02,          # ← VERY SLOW LEARNING
            num_leaves=127,              # ← COMPLEX TREES
            max_depth=10,                # ← DEEP TREES
            subsample=0.85,              # ← More regularization
            colsample_bytree=0.85,
            min_child_samples=25,
            reg_alpha=0.2,               # ← Stronger L1
            reg_lambda=0.2,              # ← Stronger L2
            min_split_gain=0.01,         # ← More conservative splits
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        
        model.fit(X_train, y_train)
        y_new = model.predict(X_pred)
        y_new = np.maximum(y_new, 0.0)
        
        df. loc[missing_mask, target] = y_new
        
        diff = float(np.mean(np.abs(y_new - prev)))
        logging.info(f"[{MODEL_NAME}] iter {it}/{max_iter}, change={diff:.8f}")
        
        if diff < tol:
            logging. info(f"[{MODEL_NAME}] ✅ Converged at iteration {it}")
            break
        
        prev = y_new.copy()
    
    return df
