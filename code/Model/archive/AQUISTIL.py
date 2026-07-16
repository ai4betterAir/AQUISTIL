import os
import numpy as np
import pandas as pd
import logging
import lightgbm as lgb
from spatial import prepare_spatial_temporal_data

MODEL_NAME = "AQUISTIL"

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _ensure_datetime(df:  pd.DataFrame) -> pd.DataFrame:
    if "DateTime" not in df.columns:
        raise ValueError("DateTime column is required.")
    out = df.copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"])
    return out


def _build_iter_features(
    df_work: pd.DataFrame,
    target:  str,
    feature_columns: list,
    strict_feature_list: bool = False,
) -> pd.DataFrame:
    """
    Build feature matrix using prepared features (including spatial/temporal)
    """
    df_work = _ensure_datetime(df_work)
    
    # Use all prepared features
    X = df_work[feature_columns].copy()

    # Stage 3 has already evaluated and selected the complete feature set.  In
    # strict mode, do not silently expand that set with model-specific target
    # history or missingness-geometry features.
    if strict_feature_list:
        return X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    
    # Add iterative lag/rolling from current filled target
    y = pd.to_numeric(df_work[target], errors="coerce")
    
    # Past-only lags
    X["lag_1"] = y.shift(1)
    X["lag_6"] = y.shift(6)
    X["lag_24"] = y.shift(24)
    X["lag_72"] = y.shift(72)
    
    # Past-only rolling
    X["roll_mean_24"] = y.shift(1).rolling(24, min_periods=6).mean()
    X["roll_std_24"] = y.shift(1).rolling(24, min_periods=6).std()
    X["roll_mean_72"] = y.shift(1).rolling(72, min_periods=12).mean()
    
    # Missingness geometry
    is_missing = y.isna().astype(int)
    run_id = (is_missing == 0).cumsum()
    X["gap_length"] = is_missing. groupby(run_id).cumcount()
    X["gap_is_long"] = (X["gap_length"] >= 24).astype(int)
    X["gap_is_very_long"] = (X["gap_length"] >= 72).astype(int)
    
    # Final cleanup
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return X


def impute_mice(
    data,
    target_column,
    input_columns,
    max_iter=10,
    tol=1e-4,
    random_state=42,
    custom_strategies=None,
    **kwargs
):
    """
    Iterative LightGBM imputer with spatial-temporal features
    """
    
    # ✅ EXTRACT SPATIAL CONFIG FROM KWARGS
    out_dir = kwargs.get('out_dir', '')

    # Load spatial config early so we can fallback to any configured default
    import config_spatial as config
    strict_feature_list = bool(
        kwargs.get(
            "strict_feature_list",
            getattr(config, "STRICT_PROGRESSIVE_FEATURE_LIST", False),
        )
    )

    # Accept several possible kwarg names for explicit site selection
    site_name = (
        kwargs.get('site_name') or
        kwargs.get('site') or
        kwargs.get('target_site') or
        kwargs.get('station') or
        ''
    )

    # If not provided, try to parse a site token from `model_name` (common delimiters)
    if not site_name and 'model_name' in kwargs and isinstance(kwargs['model_name'], str):
        model_name = kwargs['model_name']
        sanitized = model_name.replace('-', '_').replace('.', '_')
        parts = [p for p in sanitized.split('_') if p]
        if parts:
            candidate = parts[-1]
            candidate = ''.join(ch for ch in candidate if ch.isalnum())
            site_name = candidate

    # Final fallback: use configured default target site if available
    if not site_name and hasattr(config, 'TARGET_SITE'):
        site_name = getattr(config, 'TARGET_SITE', '')

    # Build spatial config
    spatial_config = {
        'input_directory': config.INPUT_DIRECTORY,
        'target_site': site_name,
        'use_spatial': config.USE_SPATIAL_FEATURES,
        'use_temporal': config.USE_TEMPORAL_FEATURES,
        'use_lagged':  False,  # We build lags internally
        'use_rolling': False,  # We build rolling features internally
    }
    
    logging.info(f"[{MODEL_NAME}] Preparing spatial-temporal features...")
    logging.info(f"  Site: {site_name}")
    logging.info(f"  Spatial features: {spatial_config['use_spatial']}")
    logging.info(f"  Temporal features: {spatial_config['use_temporal']}")
    
    # ✅ PREPARE SPATIAL-TEMPORAL DATA
    try:
        df_enhanced, feature_columns = prepare_spatial_temporal_data(
            data, 
            target_column, 
            input_columns, 
            spatial_config
        )
        logging.info(f"  Total features prepared: {len(feature_columns)}")
    except Exception as e: 
        logging.warning(f"  Failed to load spatial features: {e}")
        logging.warning(f"  Falling back to local features only")
        df_enhanced = data.copy()
        feature_columns = input_columns
    
    df = df_enhanced.copy()
    target = target_column
    
    y0 = pd.to_numeric(df[target], errors="coerce")
    missing_mask = y0.isna().values
    nmiss = int(missing_mask.sum())
    
    if nmiss == 0:
        return df
    
    # Initialize missing values
    init_val = float(np.nanmedian(y0.values))
    df.loc[missing_mask, target] = init_val
    prev = df.loc[missing_mask, target]. values. copy()
    
    rng = np.random.default_rng(random_state)
    
    for it in range(1, max_iter + 1):
        # Rebuild features
        X = _build_iter_features(
            df, target, feature_columns, strict_feature_list=strict_feature_list
        )
        y = pd.to_numeric(df[target], errors="coerce")
        
        # Train only on originally observed rows
        obs_mask = ~missing_mask
        X_train = X. loc[obs_mask]
        y_train = y.loc[obs_mask]
        
        # Predict on originally missing rows only
        X_pred = X.loc[missing_mask]
        
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=30,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        
        model.fit(X_train, y_train)
        y_new = model.predict(X_pred)
        
        # Optional clamp
        clamp = float(kwargs.get("clamp", 0.0))
        if clamp > 0:
            y_new = np.clip(y_new, a_min=-clamp, a_max=clamp)
        
        df.loc[missing_mask, target] = y_new
        
        diff = float(np.mean(np.abs(y_new - prev)))
        logging.info(f"[{MODEL_NAME}] iter {it}/{max_iter} mean_change={diff:.6f}")
        
        if diff < tol:
            logging.info(f"[{MODEL_NAME}] converged at iter {it}")
            break
        
        prev = y_new. copy()
    
    return df
