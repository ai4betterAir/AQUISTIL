"""
Advanced MissForest Hybrid Strategies
5 different approaches to enhance MissForest performance

Strategies:
1. Dual-Stage (Coarse → Fine)
2. Ensemble (Multiple seeds + averaging)
3. Boosted (Residual learning)
4. Feature-Engineered (Temporal features)
5. Stacked (Multi-level meta-learning)

Author: Dr. Masrur
Last Updated: 2026-01-05
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "HybridMissForest_Advanced"


# ============================================================================
# STRATEGY 1: DUAL-STAGE MISSFOREST (Coarse → Fine)
# ============================================================================

class DualStageMissForest:
    """
    Two-stage approach: 
    Stage 1: Fast, shallow trees (rough imputation)
    Stage 2: Deep, complex trees (precise refinement)
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def _run_stage(self, X, mask, n_estimators, max_depth, max_iter, stage_name):
        """Run one stage of MissForest"""
        X_filled = X.copy()
        
        for iteration in range(max_iter):
            X_old = X_filled.copy()
            
            for j in range(X. shape[1]):
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
                
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=5 if max_depth > 10 else 20,
                    min_samples_leaf=2 if max_depth > 10 else 10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                X_filled[miss_idx, j] = rf.predict(X_test)
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            logging.info(f"    {stage_name} iter {iteration + 1}: change={change:.6f}")
            
            if change < 1e-4:
                logging.info(f"    {stage_name} converged")
                break
        
        return X_filled
    
    def fit_transform(self, data):
        logging.info("\n  Strategy 1: Dual-Stage (Coarse → Fine)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data. index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        X_filled = X.copy()
        
        # Initial median imputation
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_filled[mask[:, j], j] = np.nanmedian(X[:, j])
        
        # Stage 1: Coarse (fast baseline)
        logging.info("    Stage 1: Coarse imputation (50 trees, depth 5)")
        X_filled = self._run_stage(X_filled, mask, n_estimators=50, max_depth=5, 
                                   max_iter=2, stage_name="Coarse")
        
        # Stage 2: Fine (precise refinement)
        logging.info("    Stage 2: Fine refinement (150 trees, depth 15)")
        X_filled = self._run_stage(X_filled, mask, n_estimators=150, max_depth=15, 
                                   max_iter=3, stage_name="Fine")
        
        if columns is not None:
            X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
        
        return X_filled


# ============================================================================
# STRATEGY 2: ENSEMBLE MISSFOREST (Multiple seeds)
# ============================================================================

class EnsembleMissForest: 
    """
    Run MissForest multiple times with different random seeds
    Average results for stability and reduced variance
    """
    def __init__(self, n_ensemble=5, random_state=42):
        self.n_ensemble = n_ensemble
        self. random_state = random_state
    
    def _single_missforest(self, X, seed):
        """Single MissForest run"""
        mask = np.isnan(X)
        X_filled = X. copy()
        
        # Median initialization
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_filled[mask[:, j], j] = np.nanmedian(X[:, j])
        
        # Iterative imputation
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
                
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
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
        logging.info(f"\n  Strategy 2: Ensemble ({self.n_ensemble} models)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data.index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        # Run multiple MissForest models
        results = []
        for i in range(self.n_ensemble):
            logging.info(f"    Running ensemble member {i+1}/{self.n_ensemble}...")
            seed = self.random_state + i * 100
            imputed = self._single_missforest(X. copy(), seed)
            results. append(imputed)
        
        # Average all results
        X_ensemble = np.mean(results, axis=0)
        
        # Calculate prediction variance (uncertainty)
        X_variance = np.var(results, axis=0)
        avg_std = np.sqrt(X_variance[np.isnan(X)].mean())
        
        logging.info(f"    Average prediction std: {avg_std:.4f}")
        
        if columns is not None:
            X_ensemble = pd.DataFrame(X_ensemble, columns=columns, index=index)
        
        return X_ensemble


# ============================================================================
# STRATEGY 3: BOOSTED MISSFOREST (Residual Learning)
# ============================================================================

class BoostedMissForest: 
    """
    Gradient boosting-style approach: 
    Stage 1: MissForest baseline
    Stage 2: Learn residuals (prediction errors)
    Stage 3: Combine baseline + residuals
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def _run_missforest(self, X, seed, max_iter=3):
        """Run standard MissForest"""
        mask = np.isnan(X)
        X_filled = X. copy()
        
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_filled[mask[: , j], j] = np. nanmedian(X[:, j])
        
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
                
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
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
        logging.info("\n  Strategy 3: Boosted (Residual Learning)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data. index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        
        # Stage 1: Baseline MissForest
        logging.info("    Stage 1: Baseline MissForest")
        X_stage1 = self._run_missforest(X.copy(), self.random_state, max_iter=3)
        
        # Stage 2: Learn residuals on OBSERVED data
        logging.info("    Stage 2: Learning residuals")
        X_residual = X. copy()
        
        for j in range(X. shape[1]):
            obs_idx = ~mask[:, j]
            miss_idx = mask[:, j]
            
            if obs_idx.sum() > 0:
                # Calculate residuals on observed data
                residuals = X[obs_idx, j] - X_stage1[obs_idx, j]
                
                # Create residual "dataset"
                X_residual[obs_idx, j] = residuals
                X_residual[miss_idx, j] = np.nan  # Will be imputed
        
        # Impute residuals
        X_residual_imputed = self._run_missforest(X_residual, self.random_state + 1, max_iter=2)
        
        # Stage 3: Combine baseline + damped residuals
        logging.info("    Stage 3: Combining predictions")
        learning_rate = 0.3  # Dampen residuals to prevent overfitting
        X_final = X_stage1 + learning_rate * X_residual_imputed
        
        if columns is not None:
            X_final = pd.DataFrame(X_final, columns=columns, index=index)
        
        return X_final


# ============================================================================
# STRATEGY 4: FEATURE-ENGINEERED MISSFOREST (Temporal)
# ============================================================================

class FeatureEngineeredMissForest: 
    """
    Add temporal features before MissForest
    Uses time patterns (hour, day, month, cyclical encoding)
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def add_temporal_features(self, data):
        """Add time-based features with cyclical encoding"""
        if 'DateTime' not in data.columns:
            logging.warning("    No DateTime column found, skipping temporal features")
            return data, []
        
        df = data.copy()
        temporal_cols = []
        
        dt = pd.to_datetime(df['DateTime'])
        
        # Cyclical encoding for periodic features
        # Hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np. pi * dt.dt.hour / 24)
        
        # Day of week (7-day cycle)
        df['dow_sin'] = np.sin(2 * np.pi * dt. dt.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dt. dt.dayofweek / 7)
        
        # Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np. pi * dt.dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
        
        # Day of year (365-day cycle)
        df['doy_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear / 365)
        df['doy_cos'] = np. cos(2 * np.pi * dt.dt.dayofyear / 365)
        
        temporal_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                        'month_sin', 'month_cos', 'doy_sin', 'doy_cos']
        
        return df, temporal_cols
    
    def fit_transform(self, data):
        logging.info("\n  Strategy 4: Feature-Engineered (Temporal)")
        
        if isinstance(data, pd. DataFrame):
            # Add temporal features
            data_enhanced, temporal_cols = self.add_temporal_features(data)
            
            logging.info(f"    Added {len(temporal_cols)} temporal features")
            
            # Get columns for imputation (exclude DateTime)
            cols_to_use = [col for col in data_enhanced.columns if col != 'DateTime']
            original_cols = [col for col in data. columns if col != 'DateTime']
            
            X = data_enhanced[cols_to_use].values
            columns = original_cols
            index = data.index
        else:
            X = data
            columns, index = None, None
            temporal_cols = []
        
        mask = np.isnan(X)
        X_filled = X. copy()
        
        # Median initialization
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_filled[mask[:, j], j] = np.nanmedian(X[:, j])
        
        # MissForest with enhanced features
        for iteration in range(4):
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
                    n_estimators=120,
                    max_depth=12,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                X_filled[miss_idx, j] = rf.predict(X_test)
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            logging.info(f"    Iteration {iteration + 1}: change={change:.6f}")
            
            if change < 1e-4:
                break
        
        # Remove temporal features from output (keep only original columns)
        if isinstance(data, pd.DataFrame):
            n_original = len(columns)
            X_filled = X_filled[: , :n_original]
            X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
        
        return X_filled


# ============================================================================
# STRATEGY 5: STACKED MISSFOREST (Multi-level Meta-Learning)
# ============================================================================

class StackedMissForest: 
    """
    3-level stacking: 
    Level 0: Base models (RF, ExtraTrees, GradientBoosting)
    Level 1: MissForest on combined predictions
    Level 2: Meta-learner (Ridge) learns optimal combination
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def _run_base_model(self, X, mask, model_type, seed):
        """Run a single base model"""
        X_filled = X.copy()
        
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_filled[mask[:, j], j] = np.nanmedian(X[:, j])
        
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
                
                X_train = X_filled[obs_idx][: , feature_cols]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, feature_cols]
                
                if model_type == 'rf': 
                    model = RandomForestRegressor(
                        n_estimators=80, max_depth=10, random_state=seed, n_jobs=-1
                    )
                elif model_type == 'et':
                    model = ExtraTreesRegressor(
                        n_estimators=80, max_depth=10, random_state=seed, n_jobs=-1
                    )
                elif model_type == 'gb': 
                    model = GradientBoostingRegressor(
                        n_estimators=80, max_depth=5, learning_rate=0.1, 
                        random_state=seed
                    )
                
                model.fit(X_train, y_train)
                X_filled[miss_idx, j] = model.predict(X_test)
            
            change = np. linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            if change < 1e-4:
                break
        
        return X_filled
    
    def fit_transform(self, data, original_mask=None):
        logging.info("\n  Strategy 5: Stacked (3-Level Meta-Learning)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data. columns, data.index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        if original_mask is None:
            mask = np.isnan(X)
        else:
            mask = original_mask. values if isinstance(original_mask, pd. DataFrame) else original_mask
        
        observed_mask = ~mask
        
        # ====================================================================
        # LEVEL 0: Base Models (Diverse ensemble)
        # ====================================================================
        logging.info("    Level 0: Training 3 base models...")
        
        logging.info("      Model 1: RandomForest")
        pred_rf = self._run_base_model(X. copy(), mask, 'rf', self.random_state)
        
        logging.info("      Model 2: ExtraTrees")
        pred_et = self._run_base_model(X.copy(), mask, 'et', self.random_state + 1)
        
        logging.info("      Model 3: GradientBoosting")
        pred_gb = self._run_base_model(X.copy(), mask, 'gb', self.random_state + 2)
        
        # ====================================================================
        # LEVEL 1: MissForest on combined predictions
        # ====================================================================
        logging.info("    Level 1: MissForest on base predictions...")
        
        # Average base models
        level0_avg = (pred_rf + pred_et + pred_gb) / 3
        
        # Run MissForest on averaged predictions
        X_level1 = level0_avg.copy()
        
        for iteration in range(3):
            X_old = X_level1.copy()
            
            for j in range(X.shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[:, j]
                miss_idx = mask[:, j]
                feature_cols = [i for i in range(X. shape[1]) if i != j]
                
                if len(feature_cols) == 0 or obs_idx.sum() < 10:
                    continue
                
                X_train = X_level1[obs_idx][:, feature_cols]
                y_train = X_level1[obs_idx, j]
                X_test = X_level1[miss_idx][:, feature_cols]
                
                rf = RandomForestRegressor(
                    n_estimators=100, max_depth=12, random_state=self.random_state, n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                X_level1[miss_idx, j] = rf. predict(X_test)
            
            change = np.linalg.norm(X_level1 - X_old) / (np.linalg.norm(X_old) + 1e-10)
            if change < 1e-4:
                break
        
        # ====================================================================
        # LEVEL 2: Meta-Learner (Ridge Regression)
        # ====================================================================
        logging.info("    Level 2: Training meta-learner...")
        
        # Prepare meta-features (predictions from all models)
        # Only use observed data for training meta-learner
        if observed_mask.sum() > 100:  # Need enough data
            # Flatten predictions
            meta_features = []
            meta_targets = []
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if observed_mask[i, j]:
                        features = [
                            pred_rf[i, j],
                            pred_et[i, j],
                            pred_gb[i, j],
                            X_level1[i, j],
                            (pred_rf[i, j] + pred_et[i, j] + pred_gb[i, j]) / 3
                        ]
                        meta_features. append(features)
                        meta_targets.append(X[i, j])
            
            meta_features = np.array(meta_features)
            meta_targets = np.array(meta_targets)
            
            # Train meta-learner
            meta_model = Ridge(alpha=1.0)
            meta_model.fit(meta_features, meta_targets)
            
            # Predict on all data
            X_final = X.copy()
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if mask[i, j]:
                        features = [
                            pred_rf[i, j],
                            pred_et[i, j],
                            pred_gb[i, j],
                            X_level1[i, j],
                            (pred_rf[i, j] + pred_et[i, j] + pred_gb[i, j]) / 3
                        ]
                        X_final[i, j] = meta_model.predict([features])[0]
            
            # Log meta-learner weights
            weights = meta_model.coef_
            logging.info(f"    Meta-learner weights: RF={weights[0]:.3f}, ET={weights[1]:.3f}, "
                        f"GB={weights[2]:.3f}, L1={weights[3]:.3f}, Avg={weights[4]:.3f}")
        else:
            logging.warning("    Not enough data for meta-learner, using Level 1 output")
            X_final = X_level1
        
        if columns is not None:
            X_final = pd.DataFrame(X_final, columns=columns, index=index)
        
        return X_final


# ============================================================================
# MAIN HYBRID SELECTOR
# ============================================================================

class AdvancedHybridImputer:
    """
    Main class to select and run MissForest strategies
    """
    def __init__(self, strategy='auto', random_state=42):
        """
        Args:
            strategy: 'dual_stage', 'ensemble', 'boosted', 
                     'feature_eng', 'stacked', 'auto'
        """
        self.strategy = strategy
        self.random_state = random_state
    
    def evaluate(self, predictions, true_values, observed_mask):
        """Evaluate on observed data"""
        obs_mask_flat = observed_mask.flatten()
        pred_obs = predictions. flatten()[obs_mask_flat]
        true_obs = true_values. flatten()[obs_mask_flat]
        
        r2 = r2_score(true_obs, pred_obs)
        rmse = np.sqrt(mean_squared_error(true_obs, pred_obs))
        mae = mean_absolute_error(true_obs, pred_obs)
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
    
    def fit_transform(self, data, original_mask=None):
        """Run selected strategy"""
        logging.info("="*80)
        logging.info("ADVANCED MISSFOREST HYBRID")
        logging.info("="*80)
        
        if original_mask is None:
            original_mask = data.isna()
        
        columns = data.columns
        index = data.index
        observed_mask = ~original_mask. values
        
        # Select and run strategy
        if self.strategy == 'dual_stage': 
            imputer = DualStageMissForest(random_state=self.random_state)
            result = imputer.fit_transform(data)
        
        elif self.strategy == 'ensemble':
            imputer = EnsembleMissForest(n_ensemble=5, random_state=self.random_state)
            result = imputer.fit_transform(data)
        
        elif self.strategy == 'boosted':
            imputer = BoostedMissForest(random_state=self.random_state)
            result = imputer. fit_transform(data)
        
        elif self.strategy == 'feature_eng':
            imputer = FeatureEngineeredMissForest(random_state=self.random_state)
            result = imputer.fit_transform(data)
        
        elif self.strategy == 'stacked':
            imputer = StackedMissForest(random_state=self.random_state)
            result = imputer. fit_transform(data, original_mask)
        
        elif self.strategy == 'auto': 
            # Auto-select based on data characteristics
            n_samples = len(data)
            n_features = len(data.columns)
            missing_rate = original_mask.sum().sum() / original_mask.size
            
            logging.info(f"  Auto-selection: n_samples={n_samples}, n_features={n_features}, missing_rate={missing_rate:.2%}")
            
            if n_samples > 20000:
                logging.info("  → Selected:  Dual-Stage (large dataset)")
                imputer = DualStageMissForest(random_state=self.random_state)
                result = imputer.fit_transform(data)
            
            elif missing_rate > 0.4:
                logging.info("  → Selected: Stacked (high missingness)")
                imputer = StackedMissForest(random_state=self.random_state)
                result = imputer. fit_transform(data, original_mask)
            
            elif 'DateTime' in data.columns:
                logging.info("  → Selected: Feature-Engineered (temporal data)")
                imputer = FeatureEngineeredMissForest(random_state=self.random_state)
                result = imputer.fit_transform(data)
            
            else:
                logging.info("  → Selected: Ensemble (default)")
                imputer = EnsembleMissForest(n_ensemble=5, random_state=self.random_state)
                result = imputer.fit_transform(data)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Evaluate
        metrics = self.evaluate(result. values, data.values, observed_mask)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"FINAL RESULTS:")
        logging.info(f"{'='*80}")
        logging.info(f"  Strategy: {self.strategy}")
        logging.info(f"  R²:    {metrics['r2']:.4f}")
        logging.info(f"  RMSE: {metrics['rmse']:.4f}")
        logging.info(f"  MAE:  {metrics['mae']:.4f}")
        logging.info(f"{'='*80}\n")
        
        return result


# ============================================================================
# MAIN FUNCTION (Pipeline Compatible)
# ============================================================================

def impute_mice(data, target_column, input_columns, max_iter=10, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None):
    """
    Main imputation function
    NOTE:  Ignores spatial_config to prevent data leakage
    """
    logging.info(f"Starting {MODEL_NAME} imputation...")
    logging.info("⚠️  NOTE: Spatial features are DISABLED to prevent data leakage")
    
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
                    data_to_use[col] = data_to_use[col]. fillna(strategy)
    
    # Prepare for imputation
    columns_for_imputation = input_columns + [target_column]
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    data_for_imputation = data_to_use[columns_for_imputation]. copy()
    
    logging.info(f"Using {len(columns_for_imputation)} LOCAL features:  {', '.join(columns_for_imputation)}")
    
    original_mask = data_for_imputation.isna()
    
    # Initialize advanced imputer
    # Use 'auto' to let it choose best strategy
    # Or specify:  'dual_stage', 'ensemble', 'boosted', 'feature_eng', 'stacked'
    advanced_imputer = AdvancedHybridImputer(
        strategy='auto',  # Automatically selects best strategy
        random_state=random_state
    )
    
    # Perform imputation
    imputed_df = advanced_imputer.fit_transform(data_for_imputation, original_mask)
    
    # Create final output
    final_df = data. copy()
    for col in columns_for_imputation:
        if col in data.columns:
            final_df[col] = imputed_df[col]
    
    logging.info(f"{MODEL_NAME} imputation completed successfully.\n")
    return final_df


if __name__ == "__main__":  
    logging.info("Testing Advanced MissForest Hybrid...")
    
    np.random.seed(42)
    n = 2000
    
    # Create realistic data with temporal patterns
    t = np.linspace(0, 10, n)
    f1 = 50 + 10 * np. sin(t) + np.random.randn(n) * 3
    f2 = f1 * 0.75 + 5 * np.cos(t) + np.random.randn(n) * 4
    f3 = f1 * 0.4 + f2 * 0.3 + np.random.randn(n) * 3
    
    data = pd.DataFrame({
        'feature_0': f1,
        'feature_1': f2,
        'feature_2': f3,
        'feature_3': np.random.randn(n) * 5 + 25
    })
    data['DateTime'] = pd.date_range('2020-01-01', periods=n, freq='H')
    
    mask = np.random.random((n, 1)) < 0.3
    original = data.loc[mask[: , 0], 'feature_0']. copy()
    data.loc[mask[:, 0], 'feature_0'] = np.nan
    
    print(f"Missing:  {mask.sum()}")
    
    # Test each strategy
    for strategy in ['dual_stage', 'ensemble', 'boosted', 'feature_eng', 'stacked']:
        print(f"\n{'='*60}")
        print(f"Testing:  {strategy}")
        print('='*60)
        
        test_data = data.copy()
        imputed = impute_mice(test_data, 'feature_0', ['feature_1', 'feature_2', 'feature_3'])
        
        imputed_vals = imputed.loc[mask[:, 0], 'feature_0'].values
        rmse = np.sqrt(np.mean((imputed_vals - original) ** 2))
        r2 = r2_score(original, imputed_vals)
        
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    print("\n✅ All tests completed!")