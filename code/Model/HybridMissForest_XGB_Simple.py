"""
SIMPLE Hybrid:  MissForest + XGBoost
KISS Principle: Keep It Simple, Stupid

No fancy tricks, just: 
1. MissForest baseline
2. XGBoost refinement  
3. Simple weighted average (if XGB improves)

Author: Dr.  Masrur
Last Updated: 2026-01-05
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.error("XGBoost required:  pip install xgboost")

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "HybridMissForest_XGB_Simple"


# ============================================================================
# SIMPLE MISSFOREST - NO TRICKS
# ============================================================================

def simple_missforest(data, max_iter=3, n_estimators=100, random_state=42):
    """
    Dead simple MissForest - exactly as in the original paper
    """
    if isinstance(data, pd.DataFrame):
        columns = data.columns
        index = data.index
        X = data.values
    else:
        columns, index, X = None, None, data
    
    mask = np.isnan(X)
    X_filled = X. copy()
    
    # Step 1: Initialize with median
    for j in range(X.shape[1]):
        if np.any(mask[:, j]):
            X_filled[mask[:, j], j] = np.nanmedian(X[:, j])
    
    # Step 2: Iterate
    for iteration in range(max_iter):
        X_old = X_filled.copy()
        
        for j in range(X.shape[1]):
            if not np.any(mask[:, j]):
                continue
            
            obs_idx = ~mask[:, j]
            miss_idx = mask[:, j]
            feature_cols = [i for i in range(X.shape[1]) if i != j]
            
            if len(feature_cols) == 0 or obs_idx.sum() < 10:
                continue
            
            X_train = X_filled[obs_idx][: , feature_cols]
            y_train = X_filled[obs_idx, j]
            X_test = X_filled[miss_idx][:, feature_cols]
            
            # Simple Random Forest - default parameters work best
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=None,  # Let it grow naturally
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1
            )
            
            rf. fit(X_train, y_train)
            X_filled[miss_idx, j] = rf.predict(X_test)
        
        # Check convergence
        change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
        logging.info(f"  MissForest iter {iteration + 1}/{max_iter}:  change={change:.6f}")
        
        if change < 1e-4:
            logging.info(f"  MissForest converged at iteration {iteration + 1}")
            break
    
    if columns is not None:
        X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
    
    return X_filled


# ============================================================================
# SIMPLE XGBOOST - NO TRICKS
# ============================================================================

def simple_xgboost(data, max_iter=3, n_estimators=100, random_state=42):
    """
    Dead simple XGBoost imputation
    """
    if not HAS_XGBOOST: 
        raise ImportError("XGBoost required")
    
    if isinstance(data, pd.DataFrame):
        columns = data.columns
        index = data.index
        X = data.values
    else:
        columns, index, X = None, None, data
    
    mask = np.isnan(X)
    X_filled = X. copy()
    
    # Step 1: Initialize with median
    for j in range(X.shape[1]):
        if np.any(mask[:, j]):
            X_filled[mask[:, j], j] = np.nanmedian(X[:, j])
    
    # Step 2: Iterate
    for iteration in range(max_iter):
        X_old = X_filled. copy()
        
        for j in range(X.shape[1]):
            if not np.any(mask[:, j]):
                continue
            
            obs_idx = ~mask[:, j]
            miss_idx = mask[:, j]
            feature_cols = [i for i in range(X.shape[1]) if i != j]
            
            if len(feature_cols) == 0 or obs_idx.sum() < 10:
                continue
            
            X_train = X_filled[obs_idx][: , feature_cols]
            y_train = X_filled[obs_idx, j]
            X_test = X_filled[miss_idx][:, feature_cols]
            
            # Simple XGBoost - default parameters
            xgb_model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.3,  # Default
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )
            
            xgb_model.fit(X_train, y_train, verbose=False)
            X_filled[miss_idx, j] = xgb_model.predict(X_test)
        
        change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
        logging.info(f"  XGBoost iter {iteration + 1}/{max_iter}: change={change:.6f}")
        
        if change < 1e-4:
            logging.info(f"  XGBoost converged at iteration {iteration + 1}")
            break
    
    if columns is not None:
        X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
    
    return X_filled


# ============================================================================
# SIMPLE HYBRID - JUST PICK THE BEST ONE
# ============================================================================

def simple_hybrid(data, original_mask, random_state=42):
    """
    Ultra-simple hybrid: 
    1. Run MissForest
    2. Run XGBoost  
    3. Evaluate both on observed data
    4. Return whichever is better
    
    No fancy ensemble, no meta-learning, just pick the winner
    """
    logging.info("="*80)
    logging.info("SIMPLE HYBRID: MissForest vs XGBoost (Pick Best)")
    logging.info("="*80)
    
    columns = data.columns
    index = data.index
    observed_mask = ~original_mask. values
    
    # Run MissForest
    logging. info("\n[Model 1/2] MissForest...")
    mf_result = simple_missforest(data, max_iter=3, n_estimators=100, random_state=random_state)
    
    # Evaluate MissForest on observed data
    if observed_mask.sum() > 0:
        mf_pred = mf_result. values[observed_mask]
        true_vals = data.values[observed_mask]
        mf_r2 = r2_score(true_vals, mf_pred)
        mf_rmse = np.sqrt(mean_squared_error(true_vals, mf_pred))
    else:
        mf_r2, mf_rmse = 0, 0
    
    logging.info(f"✅ MissForest:  R²={mf_r2:.4f}, RMSE={mf_rmse:.4f}")
    
    # Run XGBoost
    logging.info("\n[Model 2/2] XGBoost...")
    xgb_result = simple_xgboost(data, max_iter=3, n_estimators=100, random_state=random_state)
    
    # Evaluate XGBoost on observed data
    if observed_mask.sum() > 0:
        xgb_pred = xgb_result.values[observed_mask]
        xgb_r2 = r2_score(true_vals, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(true_vals, xgb_pred))
    else:
        xgb_r2, xgb_rmse = 0, 0
    
    logging.info(f"✅ XGBoost: R²={xgb_r2:.4f}, RMSE={xgb_rmse:.4f}")
    
    # Pick the better one
    logging.info(f"\n{'='*80}")
    logging.info("MODEL SELECTION:")
    logging.info(f"{'='*80}")
    
    if xgb_r2 > mf_r2:
        logging.info(f"  MissForest: R²={mf_r2:.4f}")
        logging.info(f"  XGBoost:     R²={xgb_r2:.4f} ← SELECTED")
        logging.info(f"\n  XGBoost wins by {(xgb_r2 - mf_r2):.4f} R² points")
        result = xgb_result
    else:
        logging.info(f"  MissForest:  R²={mf_r2:.4f} ← SELECTED")
        logging.info(f"  XGBoost:    R²={xgb_r2:.4f}")
        logging.info(f"\n  MissForest wins by {(mf_r2 - xgb_r2):.4f} R² points")
        result = mf_result
    
    logging.info(f"{'='*80}\n")
    
    return result


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def impute_mice(data, target_column, input_columns, max_iter=10, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None):
    """
    Main imputation function
    """
    logging.info(f"Starting {MODEL_NAME} imputation...")
    logging.info("⚠️  NOTE:  Spatial features are DISABLED to prevent data leakage")
    
    np.random.seed(random_state)
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    
    if target_column in input_columns:
        input_columns = [col for col in input_columns if col != target_column]
    
    # Use ONLY local features (NO SPATIAL)
    data_to_use = data.copy()
    
    # Apply custom strategies
    if custom_strategies:
        for col, strategy in custom_strategies.items():
            if col in data_to_use.columns:
                if strategy == "mean":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].mean())
                elif strategy == "median":
                    data_to_use[col] = data_to_use[col]. fillna(data_to_use[col].median())
                elif isinstance(strategy, (int, float)):
                    data_to_use[col] = data_to_use[col].fillna(strategy)
    
    # Prepare for imputation
    columns_for_imputation = input_columns + [target_column]
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    data_for_imputation = data_to_use[columns_for_imputation]. copy()
    
    logging.info(f"Using {len(columns_for_imputation)} LOCAL features:  {', '.join(columns_for_imputation)}")
    
    original_mask = data_for_imputation.isna()
    
    # Run simple hybrid
    imputed_df = simple_hybrid(data_for_imputation, original_mask, random_state)
    
    # Create final output
    final_df = data. copy()
    for col in columns_for_imputation:
        if col in data. columns:
            final_df[col] = imputed_df[col]
    
    logging.info(f"{MODEL_NAME} imputation completed successfully.\n")
    return final_df


if __name__ == "__main__": 
    logging.info("Testing Simple Hybrid...")
    
    np.random.seed(42)
    n = 2000
    
    # Realistic data
    t = np.linspace(0, 10, n)
    f1 = 50 + 10 * np.sin(t) + np.random.randn(n) * 3
    f2 = f1 * 0.75 + 5 * np.cos(t) + np.random.randn(n) * 4
    f3 = f1 * 0.4 + f2 * 0.3 + np.random.randn(n) * 3
    
    data = pd.DataFrame({
        'feature_0': f1,
        'feature_1': f2,
        'feature_2': f3,
        'feature_3': np.random.randn(n) * 5 + 25
    })
    
    mask = np.random.random((n, 1)) < 0.3
    original = data.loc[mask[: , 0], 'feature_0']. copy()
    data.loc[mask[:, 0], 'feature_0'] = np.nan
    
    print(f"Missing:  {mask.sum()}")
    
    imputed = impute_mice(data, 'feature_0', ['feature_1', 'feature_2', 'feature_3'])
    
    imputed_vals = imputed.loc[mask[:, 0], 'feature_0'].values
    rmse = np.sqrt(np.mean((imputed_vals - original) ** 2))
    r2 = r2_score(original, imputed_vals)
    
    print(f"\nRMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("✅ Test completed!")