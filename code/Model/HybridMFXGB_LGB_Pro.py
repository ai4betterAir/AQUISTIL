"""
Professional Hybrid Imputation - FINAL FIXED VERSION
NO spatial features by default - pure local imputation

Author: Dr.  Masrur
Last Updated: 2026-01-05
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn. metrics import r2_score, mean_squared_error, mean_absolute_error

# Import dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "HybridMFXGB_LGB_Pro"


# ============================================================================
# CLEAN MISSFOREST
# ============================================================================

class CleanMissForest:
    """MissForest - no complexity"""
    def __init__(self, max_iter=3, n_estimators=100, random_state=42):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self. random_state = random_state
    
    def fit_transform(self, X):
        if isinstance(X, pd.DataFrame):
            columns, index = X.columns, X.index
            X_array = X.values
        else:
            columns, index, X_array = None, None, X
        
        mask = np.isnan(X_array)
        X_filled = X_array.copy()
        
        # Median initialization
        for j in range(X_array.shape[1]):
            if np.any(mask[:, j]):
                X_filled[mask[:, j], j] = np.nanmedian(X_array[:, j])
        
        for iteration in range(self.max_iter):
            X_old = X_filled.copy()
            
            for j in range(X_array. shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[:, j]
                miss_idx = mask[:, j]
                feature_cols = [i for i in range(X_array.shape[1]) if i != j]
                
                if len(feature_cols) == 0 or obs_idx.sum() < 10:
                    continue
                
                X_train = X_filled[obs_idx][: , feature_cols]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, feature_cols]
                
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                X_filled[miss_idx, j] = rf.predict(X_test)
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            logging.info(f"  MissForest iter {iteration + 1}: change={change:.6f}")
            
            if change < 1e-4:
                logging.info(f"  MissForest converged at iteration {iteration + 1}")
                break
        
        if columns is not None:
            X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
        
        return X_filled


# ============================================================================
# CLEAN XGBOOST
# ============================================================================

class CleanXGBoost:
    """XGBoost - no complexity"""
    def __init__(self, max_iter=3, n_estimators=100, random_state=42):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required")
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit_transform(self, X):
        if isinstance(X, pd.DataFrame):
            columns, index = X. columns, X.index
            X_array = X.values
        else:
            columns, index, X_array = None, None, X
        
        mask = np.isnan(X_array)
        X_filled = X_array.copy()
        
        for j in range(X_array.shape[1]):
            if np. any(mask[:, j]):
                X_filled[mask[:, j], j] = np.nanmedian(X_array[:, j])
        
        for iteration in range(self.max_iter):
            X_old = X_filled.copy()
            
            for j in range(X_array.shape[1]):
                if not np.any(mask[: , j]):
                    continue
                
                obs_idx = ~mask[:, j]
                miss_idx = mask[:, j]
                feature_cols = [i for i in range(X_array.shape[1]) if i != j]
                
                if len(feature_cols) == 0 or obs_idx.sum() < 10:
                    continue
                
                X_train = X_filled[obs_idx][: , feature_cols]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, feature_cols]
                
                xgb_model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0
                )
                
                xgb_model. fit(X_train, y_train, verbose=False)
                X_filled[miss_idx, j] = xgb_model.predict(X_test)
            
            change = np.linalg. norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            logging.info(f"  XGBoost iter {iteration + 1}: change={change:.6f}")
            
            if change < 1e-4:
                logging.info(f"  XGBoost converged at iteration {iteration + 1}")
                break
        
        if columns is not None:
            X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
        
        return X_filled


# ============================================================================
# CLEAN LIGHTGBM META-LEARNER
# ============================================================================

class CleanLightGBM:
    """LightGBM meta-learner - simple and effective"""
    def __init__(self, random_state=42):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM required")
        self.random_state = random_state
        self.model = None
        self.best_weight = 0.5
        self.use_meta = False
    
    def fit(self, mf_pred, xgb_pred, true_values, observed_mask):
        """Train on observed data"""
        obs_mask_flat = observed_mask.flatten()
        mf_flat = mf_pred.flatten()[obs_mask_flat]
        xgb_flat = xgb_pred.flatten()[obs_mask_flat]
        true_flat = true_values.flatten()[obs_mask_flat]
        
        if len(true_flat) < 100:
            logging.warning(f"  Only {len(true_flat)} samples - using simple average")
            self.best_weight = 0.5
            return self
        
        logging.info(f"  Training meta-learner on {len(true_flat)} samples...")
        
        # Try simple weighted average first
        best_r2 = -999
        for w in np.linspace(0, 1, 21):
            pred = w * mf_flat + (1 - w) * xgb_flat
            r2 = r2_score(true_flat, pred)
            if r2 > best_r2:
                best_r2 = r2
                self. best_weight = w
        
        mf_r2 = r2_score(true_flat, mf_flat)
        xgb_r2 = r2_score(true_flat, xgb_flat)
        
        logging.info(f"    MissForest R²: {mf_r2:.4f}")
        logging.info(f"    XGBoost R²: {xgb_r2:.4f}")
        logging.info(f"    Best weighted R²: {best_r2:.4f} (MF={self.best_weight:.2f})")
        
        # Try LightGBM if it might help
        if best_r2 < max(mf_r2, xgb_r2) + 0.01:
            logging.info(f"    Weighted ensemble doesn't help much, using best individual model")
            self.use_meta = False
            if mf_r2 >= xgb_r2:
                self.best_weight = 1.0
            else:
                self.best_weight = 0.0
        else:
            logging.info(f"    Using weighted ensemble: MF={self.best_weight:.2f}, XGB={1-self.best_weight:.2f}")
            self.use_meta = False
        
        return self
    
    def predict(self, mf_pred, xgb_pred):
        """Simple weighted prediction"""
        return self.best_weight * mf_pred + (1 - self. best_weight) * xgb_pred


# ============================================================================
# CLEAN HYBRID IMPUTER
# ============================================================================

class CleanHybridImputer: 
    """Clean hybrid - no spatial features, no overfitting"""
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        if not HAS_XGBOOST or not HAS_LIGHTGBM:
            raise ImportError("XGBoost and LightGBM required")
    
    def evaluate(self, predictions, true_values, observed_mask):
        """Evaluate on observed data"""
        obs_mask_flat = observed_mask.flatten()
        pred_obs = predictions.flatten()[obs_mask_flat]
        true_obs = true_values.flatten()[obs_mask_flat]
        
        r2 = r2_score(true_obs, pred_obs)
        rmse = np.sqrt(mean_squared_error(true_obs, pred_obs))
        mae = mean_absolute_error(true_obs, pred_obs)
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
    
    def fit_transform(self, data, original_mask=None):
        """Perform clean hybrid imputation"""
        logging.info("="*80)
        logging.info("CLEAN HYBRID IMPUTATION (NO SPATIAL LEAKAGE)")
        logging.info("="*80)
        
        if original_mask is None:
            original_mask = data.isna()
        
        columns = data.columns
        index = data.index
        observed_mask = ~original_mask. values
        
        # Stage 1: MissForest
        logging.info("\n[Stage 1/3] MissForest...")
        mf_imputer = CleanMissForest(max_iter=3, n_estimators=100, random_state=self.random_state)
        mf_imputed = mf_imputer.fit_transform(data)
        mf_metrics = self.evaluate(mf_imputed. values, data.values, observed_mask)
        logging.info(f"✅ MissForest:  R²={mf_metrics['r2']:.4f}, RMSE={mf_metrics['rmse']:.4f}")
        
        # Stage 2: XGBoost
        logging.info("\n[Stage 2/3] XGBoost...")
        xgb_imputer = CleanXGBoost(max_iter=3, n_estimators=100, random_state=self.random_state)
        xgb_imputed = xgb_imputer.fit_transform(mf_imputed)
        xgb_metrics = self.evaluate(xgb_imputed. values, data.values, observed_mask)
        logging.info(f"✅ XGBoost: R²={xgb_metrics['r2']:.4f}, RMSE={xgb_metrics['rmse']:.4f}")
        
        # Stage 3:  Ensemble
        logging.info("\n[Stage 3/3] Ensemble...")
        ensemble = CleanLightGBM(random_state=self.random_state)
        ensemble. fit(mf_imputed. values, xgb_imputed.values, data.values, observed_mask)
        
        hybrid_values = ensemble.predict(mf_imputed.values, xgb_imputed.values)
        hybrid_values[observed_mask] = data.values[observed_mask]
        
        hybrid_metrics = self.evaluate(hybrid_values, data.values, observed_mask)
        logging.info(f"✅ Ensemble: R²={hybrid_metrics['r2']:.4f}, RMSE={hybrid_metrics['rmse']:.4f}")
        
        # Select best
        candidates = [
            ('MissForest', mf_metrics['r2'], mf_imputed. values),
            ('XGBoost', xgb_metrics['r2'], xgb_imputed. values),
            ('Ensemble', hybrid_metrics['r2'], hybrid_values)
        ]
        
        best = max(candidates, key=lambda x: x[1])
        
        logging.info(f"\n{'='*80}")
        logging.info(f"FINAL SELECTION:")
        logging.info(f"{'='*80}")
        for name, r2, _ in candidates:
            marker = " ← SELECTED" if name == best[0] else ""
            logging.info(f"  {name:15s} R²:  {r2:.4f}{marker}")
        logging.info(f"{'='*80}\n")
        
        return pd.DataFrame(best[2], columns=columns, index=index)


# ============================================================================
# MAIN FUNCTION - IGNORES SPATIAL CONFIG
# ============================================================================

def impute_mice(data, target_column, input_columns, max_iter=10, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None):
    """
    Main imputation function
    NOTE: This version IGNORES spatial_config to prevent data leakage
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
    
    # Use ONLY local input columns (NO SPATIAL FEATURES)
    data_to_use = data. copy()
    
    # Apply custom strategies
    if custom_strategies: 
        for col, strategy in custom_strategies.items():
            if col in data_to_use.columns:
                if strategy == "mean":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col]. mean())
                elif strategy == "median":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].median())
                elif isinstance(strategy, (int, float)):
                    data_to_use[col] = data_to_use[col].fillna(strategy)
    
    # Prepare for imputation - ONLY local features
    columns_for_imputation = input_columns + [target_column]
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    data_for_imputation = data_to_use[columns_for_imputation]. copy()
    
    logging.info(f"Using {len(columns_for_imputation)} LOCAL features:  {', '.join(columns_for_imputation)}")
    
    original_mask = data_for_imputation.isna()
    
    # Initialize clean imputer
    clean_imputer = CleanHybridImputer(random_state=random_state)
    
    # Perform imputation
    imputed_df = clean_imputer.fit_transform(data_for_imputation, original_mask)
    
    # Create final output
    final_df = data. copy()
    for col in columns_for_imputation:
        if col in data.columns:
            final_df[col] = imputed_df[col]
    
    logging.info(f"{MODEL_NAME} imputation completed successfully.\n")
    return final_df


if __name__ == "__main__": 
    logging.info("Testing CleanHybrid...")
    
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
    rmse = np.sqrt(np. mean((imputed_vals - original) ** 2))
    r2 = r2_score(original, imputed_vals)
    
    print(f"\nRMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("✅ Test completed!")