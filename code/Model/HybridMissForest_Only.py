"""
Hybrid MissForest-Only Models
Multiple strategies using ONLY Random Forest (no XGBoost, no other algorithms)

Strategies:
1. Multi-Start MissForest (different initializations)
2. Progressive Depth MissForest (shallow → deep)
3. Weighted Trees MissForest (different tree weights)
4. Boostrap Aggregation (multiple RF runs)
5. Dual-Forest (RandomForest + ExtraTrees)

Author: Dr.   Masrur
Last Updated: 2026-01-05
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "HybridMissForest_Only"


# ============================================================================
# STRATEGY 1: MULTI-START MISSFOREST
# ============================================================================

class MultiStartMissForest: 
    """
    Run MissForest with different initializations (mean, median, mode)
    Average the results
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def _initialize(self, X, mask, method='median'):
        """Initialize missing values"""
        X_filled = X.copy()
        
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                if method == 'mean':
                    fill_value = np.nanmean(X[:, j])
                elif method == 'median':
                    fill_value = np.nanmedian(X[:, j])
                elif method == 'random':
                    # Random sample from observed values
                    observed = X[~mask[:, j], j]
                    fill_value = np.random.choice(observed) if len(observed) > 0 else 0
                else:
                    fill_value = np.nanmedian(X[:, j])
                
                X_filled[mask[:, j], j] = fill_value
        
        return X_filled
    
    def _run_missforest(self, X_init, mask, seed):
        """Run MissForest with given initialization"""
        X_filled = X_init.copy()
        
        for iteration in range(3):
            X_old = X_filled.copy()
            
            for j in range(X. shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[:, j]
                miss_idx = mask[:, j]
                feature_cols = [i for i in range(X.shape[1]) if i != j]
                
                if len(feature_cols) == 0 or obs_idx.sum() < 10:
                    continue
                
                X_train = X_filled[obs_idx][:, feature_cols]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, feature_cols]
                
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    random_state=seed,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                X_filled[miss_idx, j] = rf.predict(X_test)
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            if change < 1e-4:
                break
        
        return X_filled
    
    def fit_transform(self, data):
        logging.info("\n  Strategy 1: Multi-Start (Mean, Median, Random)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data. index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        results = []
        
        # Run 1: Mean initialization
        logging.info("    Run 1: Mean initialization")
        X_init = self._initialize(X, mask, 'mean')
        result1 = self._run_missforest(X_init, mask, self.random_state)
        results.append(result1)
        
        # Run 2: Median initialization
        logging.info("    Run 2: Median initialization")
        X_init = self._initialize(X, mask, 'median')
        result2 = self._run_missforest(X_init, mask, self.random_state + 1)
        results.append(result2)
        
        # Run 3: Random initialization
        logging.info("    Run 3: Random initialization")
        X_init = self._initialize(X, mask, 'random')
        result3 = self._run_missforest(X_init, mask, self.random_state + 2)
        results.append(result3)
        
        # Average results
        X_final = np.mean(results, axis=0)
        
        logging.info(f"    Averaged 3 runs")
        
        if columns is not None:
            X_final = pd.DataFrame(X_final, columns=columns, index=index)
        
        return X_final


# ============================================================================
# STRATEGY 2: PROGRESSIVE DEPTH MISSFOREST
# ============================================================================

class ProgressiveDepthMissForest: 
    """
    Run MissForest with progressively deeper trees
    Stage 1: Shallow trees (depth 3)
    Stage 2: Medium trees (depth 7)
    Stage 3: Deep trees (depth None)
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def _run_stage(self, X, mask, max_depth, stage_name):
        """Run one stage with specific depth"""
        X_filled = X.copy()
        
        for iteration in range(2):
            X_old = X_filled.copy()
            
            for j in range(X. shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[: , j]
                miss_idx = mask[:, j]
                feature_cols = [i for i in range(X.shape[1]) if i != j]
                
                if len(feature_cols) == 0 or obs_idx.sum() < 10:
                    continue
                
                X_train = X_filled[obs_idx][:, feature_cols]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, feature_cols]
                
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                X_filled[miss_idx, j] = rf.predict(X_test)
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            logging.info(f"      {stage_name} iter {iteration + 1}: change={change:.6f}")
            
            if change < 1e-4:
                break
        
        return X_filled
    
    def fit_transform(self, data):
        logging.info("\n  Strategy 2: Progressive Depth (Shallow → Medium → Deep)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data. columns, data.index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        X_filled = X.copy()
        
        # Initialize with median
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_filled[mask[: , j], j] = np. nanmedian(X[:, j])
        
        # Stage 1: Shallow (depth 3)
        logging.info("    Stage 1: Shallow trees (depth=3)")
        X_filled = self._run_stage(X_filled, mask, max_depth=3, stage_name="Shallow")
        
        # Stage 2: Medium (depth 7)
        logging.info("    Stage 2: Medium trees (depth=7)")
        X_filled = self._run_stage(X_filled, mask, max_depth=7, stage_name="Medium")
        
        # Stage 3: Deep (no limit)
        logging.info("    Stage 3: Deep trees (depth=None)")
        X_filled = self._run_stage(X_filled, mask, max_depth=None, stage_name="Deep")
        
        if columns is not None:
            X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
        
        return X_filled


# ============================================================================
# STRATEGY 3: WEIGHTED TREES MISSFOREST
# ============================================================================

class WeightedTreesMissForest:
    """
    Run MissForest with different n_estimators
    Weight predictions by number of trees (more trees = higher confidence)
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def _run_missforest(self, X, mask, n_estimators, seed):
        """Run MissForest with specific number of trees"""
        X_filled = X.copy()
        
        for iteration in range(3):
            X_old = X_filled. copy()
            
            for j in range(X.shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[:, j]
                miss_idx = mask[: , j]
                feature_cols = [i for i in range(X.shape[1]) if i != j]
                
                if len(feature_cols) == 0 or obs_idx.sum() < 10:
                    continue
                
                X_train = X_filled[obs_idx][:, feature_cols]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, feature_cols]
                
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=None,
                    random_state=seed,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                X_filled[miss_idx, j] = rf. predict(X_test)
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            if change < 1e-4:
                break
        
        return X_filled
    
    def fit_transform(self, data):
        logging.info("\n  Strategy 3: Weighted Trees (50, 100, 150 trees)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data.index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        
        # Initialize
        X_init = X.copy()
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_init[mask[:, j], j] = np.nanmedian(X[:, j])
        
        # Run with different tree counts
        logging.info("    Run 1: 50 trees")
        result1 = self._run_missforest(X_init. copy(), mask, n_estimators=50, seed=self.random_state)
        
        logging.info("    Run 2: 100 trees")
        result2 = self._run_missforest(X_init.copy(), mask, n_estimators=100, seed=self.random_state + 1)
        
        logging.info("    Run 3: 150 trees")
        result3 = self._run_missforest(X_init.copy(), mask, n_estimators=150, seed=self.random_state + 2)
        
        # Weighted average (more trees = higher weight)
        # Weights: 50 trees = 0.2, 100 trees = 0.3, 150 trees = 0.5
        X_final = 0.2 * result1 + 0.3 * result2 + 0.5 * result3
        
        logging.info("    Weighted average (0.2, 0.3, 0.5)")
        
        if columns is not None:
            X_final = pd.DataFrame(X_final, columns=columns, index=index)
        
        return X_final


# ============================================================================
# STRATEGY 4: BOOTSTRAP AGGREGATION MISSFOREST
# ============================================================================

class BootstrapAggregationMissForest:
    """
    Run MissForest multiple times with bootstrap sampling
    Average predictions (bagging)
    """
    def __init__(self, n_bootstrap=5, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
    
    def _run_missforest(self, X, mask, seed):
        """Standard MissForest run"""
        X_filled = X.copy()
        
        for iteration in range(3):
            X_old = X_filled. copy()
            
            for j in range(X.shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[:, j]
                miss_idx = mask[: , j]
                feature_cols = [i for i in range(X.shape[1]) if i != j]
                
                if len(feature_cols) == 0 or obs_idx.sum() < 10:
                    continue
                
                X_train = X_filled[obs_idx][:, feature_cols]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, feature_cols]
                
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    max_samples=0.8,  # Bootstrap 80% of samples
                    bootstrap=True,
                    random_state=seed,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                X_filled[miss_idx, j] = rf.predict(X_test)
            
            change = np.linalg. norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            if change < 1e-4:
                break
        
        return X_filled
    
    def fit_transform(self, data):
        logging.info(f"\n  Strategy 4: Bootstrap Aggregation ({self.n_bootstrap} runs)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data.index
            X = data. values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        
        # Initialize
        X_init = X.copy()
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_init[mask[:, j], j] = np.nanmedian(X[:, j])
        
        # Run multiple bootstrap samples
        results = []
        for i in range(self.n_bootstrap):
            logging.info(f"    Bootstrap run {i+1}/{self.n_bootstrap}")
            seed = self.random_state + i * 100
            result = self._run_missforest(X_init.copy(), mask, seed)
            results.append(result)
        
        # Average all bootstrap runs
        X_final = np. mean(results, axis=0)
        
        # Calculate uncertainty
        X_std = np.std(results, axis=0)
        avg_std = X_std[mask].mean()
        
        logging.info(f"    Average prediction std: {avg_std:.4f}")
        
        if columns is not None:
            X_final = pd.DataFrame(X_final, columns=columns, index=index)
        
        return X_final


# ============================================================================
# STRATEGY 5: DUAL-FOREST (RandomForest + ExtraTrees)
# ============================================================================

class DualForestMissForest:
    """
    Use both RandomForest and ExtraTrees in the same iteration
    Average their predictions
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def fit_transform(self, data):
        logging.info("\n  Strategy 5: Dual-Forest (RandomForest + ExtraTrees)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data.index
            X = data. values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        X_filled = X.copy()
        
        # Initialize with median
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_filled[mask[:, j], j] = np.nanmedian(X[:, j])
        
        # Iterative imputation using BOTH RF and ET
        for iteration in range(3):
            X_old = X_filled.copy()
            
            for j in range(X. shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[: , j]
                miss_idx = mask[:, j]
                feature_cols = [i for i in range(X.shape[1]) if i != j]
                
                if len(feature_cols) == 0 or obs_idx.sum() < 10:
                    continue
                
                X_train = X_filled[obs_idx][:, feature_cols]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, feature_cols]
                
                # Model 1: RandomForest
                rf = RandomForestRegressor(
                    n_estimators=75,
                    max_depth=None,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                pred_rf = rf.predict(X_test)
                
                # Model 2: ExtraTrees (more randomization)
                et = ExtraTreesRegressor(
                    n_estimators=75,
                    max_depth=None,
                    random_state=self. random_state + 1,
                    n_jobs=-1
                )
                et.fit(X_train, y_train)
                pred_et = et. predict(X_test)
                
                # Average both predictions
                X_filled[miss_idx, j] = 0.5 * pred_rf + 0.5 * pred_et
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg. norm(X_old) + 1e-10)
            logging.info(f"    Iteration {iteration + 1}: change={change:.6f}")
            
            if change < 1e-4:
                logging.info(f"    Converged at iteration {iteration + 1}")
                break
        
        if columns is not None:
            X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
        
        return X_filled


# ============================================================================
# MAIN HYBRID SELECTOR
# ============================================================================

class HybridMissForestOnly:
    """Select and run MissForest-only hybrid strategies"""
    
    def __init__(self, strategy='auto', random_state=42):
        """
        Args:
            strategy: 'multi_start', 'progressive', 'weighted_trees', 
                     'bootstrap', 'dual_forest', 'auto'
        """
        self.strategy = strategy
        self.random_state = random_state
    
    def evaluate(self, predictions, true_values, observed_mask):
        """Evaluate on observed data"""
        obs_mask_flat = observed_mask.flatten()
        pred_obs = predictions. flatten()[obs_mask_flat]
        true_obs = true_values.flatten()[obs_mask_flat]
        
        r2 = r2_score(true_obs, pred_obs)
        rmse = np.sqrt(mean_squared_error(true_obs, pred_obs))
        mae = mean_absolute_error(true_obs, pred_obs)
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
    
    def fit_transform(self, data, original_mask=None):
        """Run selected strategy"""
        logging.info("="*80)
        logging.info("HYBRID MISSFOREST (RANDOM FOREST ONLY)")
        logging.info("="*80)
        
        if original_mask is None:
            original_mask = data.isna()
        
        columns = data.columns
        index = data.index
        observed_mask = ~original_mask. values
        
        # Select strategy
        if self.strategy == 'multi_start': 
            imputer = MultiStartMissForest(random_state=self.random_state)
            result = imputer.fit_transform(data)
        
        elif self.strategy == 'progressive':
            imputer = ProgressiveDepthMissForest(random_state=self.random_state)
            result = imputer.fit_transform(data)
        
        elif self.strategy == 'weighted_trees':
            imputer = WeightedTreesMissForest(random_state=self. random_state)
            result = imputer.fit_transform(data)
        
        elif self. strategy == 'bootstrap':
            imputer = BootstrapAggregationMissForest(n_bootstrap=5, random_state=self.random_state)
            result = imputer.fit_transform(data)
        
        elif self.strategy == 'dual_forest':
            imputer = DualForestMissForest(random_state=self. random_state)
            result = imputer.fit_transform(data)
        
        elif self. strategy == 'auto':
            # Auto-select based on data size and missingness
            n_samples = len(data)
            missing_rate = original_mask.sum().sum() / original_mask.size
            
            logging.info(f"  Auto-selection: n_samples={n_samples}, missing_rate={missing_rate:.2%}")
            
            if n_samples > 10000:
                logging.info("  → Selected: Progressive Depth (large dataset)")
                imputer = ProgressiveDepthMissForest(random_state=self.random_state)
            elif missing_rate > 0.4:
                logging.info("  → Selected: Bootstrap Aggregation (high missingness)")
                imputer = BootstrapAggregationMissForest(n_bootstrap=5, random_state=self. random_state)
            else:
                logging.info("  → Selected: Dual-Forest (default - RF + ExtraTrees)")
                imputer = DualForestMissForest(random_state=self. random_state)
            
            result = imputer.fit_transform(data)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Evaluate
        metrics = self.evaluate(result. values, data.values, observed_mask)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"FINAL RESULTS:")
        logging.info(f"{'='*80}")
        logging.info(f"  Strategy: {self.strategy}")
        logging.info(f"  R²:     {metrics['r2']:.4f}")
        logging.info(f"  RMSE:  {metrics['rmse']:.4f}")
        logging.info(f"  MAE:  {metrics['mae']:.4f}")
        logging.info(f"{'='*80}\n")
        
        return result


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def impute_mice(data, target_column, input_columns, max_iter=10, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None):
    """Main imputation function"""
    logging.info(f"Starting {MODEL_NAME} imputation...")
    logging.info("⚠️  NOTE: Using ONLY Random Forest algorithms (no XGBoost, no other methods)")
    logging.info("⚠️  NOTE:  Spatial features are DISABLED to prevent data leakage")
    
    np.random.seed(random_state)
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    
    if target_column in input_columns:
        input_columns = [col for col in input_columns if col != target_column]
    
    # Use ONLY local features
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
    
    # Prepare for imputation
    columns_for_imputation = input_columns + [target_column]
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    data_for_imputation = data_to_use[columns_for_imputation]. copy()
    
    logging.info(f"Using {len(columns_for_imputation)} LOCAL features:  {', '.join(columns_for_imputation)}")
    
    original_mask = data_for_imputation.isna()
    
    # Initialize hybrid imputer
    # Use 'dual_forest' as default (best balance of performance and speed)
    hybrid_imputer = HybridMissForestOnly(
        strategy='dual_forest',  # Can change to:  'multi_start', 'progressive', 'weighted_trees', 'bootstrap', 'auto'
        random_state=random_state
    )
    
    # Perform imputation
    imputed_df = hybrid_imputer.fit_transform(data_for_imputation, original_mask)
    
    # Create final output
    final_df = data. copy()
    for col in columns_for_imputation:
        if col in data.columns:
            final_df[col] = imputed_df[col]
    
    logging.info(f"{MODEL_NAME} imputation completed successfully.\n")
    return final_df


if __name__ == "__main__": 
    logging.info("Testing Hybrid MissForest (RF-only)...")
    
    np.random.seed(42)
    n = 2000
    
    t = np.linspace(0, 10, n)
    f1 = 50 + 10 * np. sin(t) + np.random.randn(n) * 3
    f2 = f1 * 0.75 + 5 * np.cos(t) + np.random.randn(n) * 4
    f3 = f1 * 0.4 + f2 * 0.3 + np.random.randn(n) * 3
    
    data = pd. DataFrame({
        'feature_0': f1,
        'feature_1': f2,
        'feature_2': f3,
        'feature_3': np.random.randn(n) * 5 + 25
    })
    
    mask = np.random.random((n, 1)) < 0.3
    original = data.loc[mask[: , 0], 'feature_0']. copy()
    data.loc[mask[:, 0], 'feature_0'] = np.nan
    
    print(f"Missing:  {mask.sum()}")
    
    # Test different strategies
    for strategy in ['multi_start', 'progressive', 'dual_forest', 'bootstrap']: 
        print(f"\n{'='*60}")
        print(f"Testing:  {strategy}")
        print('='*60)
        
        test_data = data.copy()
        
        hybrid = HybridMissForestOnly(strategy=strategy, random_state=42)
        imputed = hybrid.fit_transform(test_data, test_data.isna())
        
        imputed_vals = imputed.loc[mask[:, 0], 'feature_0'].values
        rmse = np.sqrt(np. mean((imputed_vals - original) ** 2))
        r2 = r2_score(original, imputed_vals)
        
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    print("\n✅ All tests completed!")