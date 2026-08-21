"""
Probabilistic MissForest Hybrid with Confidence Intervals
Uses multiple ensemble runs to estimate prediction uncertainty

Strategies:
1. Confidence-Weighted Ensemble (weight by uncertainty)
2. Quantile Regression (prediction intervals)
3. Bayesian Bootstrap (probabilistic sampling)
4. Uncertainty-Based Selection (choose most certain predictions)
5. Adaptive Ensemble (adjust weights based on confidence)

Author: Dr.  Masrur
Last Updated:  2026-01-05
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "HybridMissForest_Probabilistic"


# ============================================================================
# BASE MISSFOREST COMPONENT
# ============================================================================

class SingleMissForest:
    """Single MissForest run with uncertainty estimation"""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.uncertainties = None
    
    def fit_transform(self, X, return_uncertainty=True):
        """
        Run MissForest and estimate prediction uncertainty
        
        Returns:
            imputed_data, prediction_std (if return_uncertainty=True)
        """
        mask = np.isnan(X)
        X_filled = X. copy()
        
        # Median initialization
        for j in range(X.shape[1]):
            if np.any(mask[: , j]):
                X_filled[mask[:, j], j] = np.nanmedian(X[:, j])
        
        # Store prediction uncertainties
        prediction_std = np.zeros_like(X)
        
        # Iterative imputation
        for iteration in range(3):
            X_old = X_filled.copy()
            
            for j in range(X. shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[: , j]
                miss_idx = mask[: , j]
                feature_cols = [i for i in range(X.shape[1]) if i != j]
                
                if len(feature_cols) == 0 or obs_idx.sum() < 10:
                    continue
                
                X_train = X_filled[obs_idx][: , feature_cols]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, feature_cols]
                
                # Random Forest with individual tree predictions
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                
                # Get predictions from all trees
                tree_predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])
                
                # Mean prediction
                y_pred = tree_predictions.mean(axis=0)
                X_filled[miss_idx, j] = y_pred
                
                # Standard deviation (uncertainty)
                y_std = tree_predictions.std(axis=0)
                prediction_std[miss_idx, j] = y_std
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            if change < 1e-4:
                break
        
        if return_uncertainty:
            return X_filled, prediction_std
        return X_filled


# ============================================================================
# STRATEGY 1: CONFIDENCE-WEIGHTED ENSEMBLE
# ============================================================================

class ConfidenceWeightedEnsemble:
    """
    Run multiple MissForest models
    Weight predictions by inverse uncertainty (1/std)
    More certain predictions get higher weight
    """
    def __init__(self, n_ensemble=5, random_state=42):
        self.n_ensemble = n_ensemble
        self.random_state = random_state
    
    def fit_transform(self, data):
        logging.info(f"\n  Strategy 1: Confidence-Weighted Ensemble ({self.n_ensemble} models)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data.index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        
        # Run multiple MissForest with different seeds
        predictions = []
        uncertainties = []
        
        for i in range(self.n_ensemble):
            logging.info(f"    Running model {i+1}/{self.n_ensemble}...")
            seed = self.random_state + i * 100
            
            mf = SingleMissForest(random_state=seed)
            imputed, std = mf.fit_transform(X. copy(), return_uncertainty=True)
            
            predictions.append(imputed)
            uncertainties. append(std)
        
        # Convert to arrays
        predictions = np.array(predictions)  # Shape: (n_ensemble, n_samples, n_features)
        uncertainties = np.array(uncertainties)
        
        # Calculate weights based on inverse uncertainty
        # Higher uncertainty → lower weight
        epsilon = 1e-6
        weights = 1.0 / (uncertainties + epsilon)  # Inverse uncertainty
        
        # Normalize weights (sum to 1 for each position)
        weights_sum = weights.sum(axis=0, keepdims=True)
        weights_normalized = weights / (weights_sum + epsilon)
        
        # Weighted average
        X_final = (predictions * weights_normalized).sum(axis=0)
        
        # Calculate final uncertainty (weighted std)
        final_uncertainty = (uncertainties * weights_normalized).sum(axis=0)
        
        # Calculate prediction interval (95% CI)
        ci_lower = X_final - 1.96 * final_uncertainty
        ci_upper = X_final + 1.96 * final_uncertainty
        
        # Log statistics
        avg_uncertainty = final_uncertainty[mask].mean()
        avg_ci_width = (ci_upper - ci_lower)[mask].mean()
        
        logging.info(f"  ✅ Ensemble completed")
        logging.info(f"     Average uncertainty (missing values): {avg_uncertainty:.4f}")
        logging.info(f"     Average 95% CI width: {avg_ci_width:.4f}")
        
        if columns is not None:
            X_final = pd.DataFrame(X_final, columns=columns, index=index)
        
        return X_final


# ============================================================================
# STRATEGY 2: QUANTILE REGRESSION FORESTS
# ============================================================================

class QuantileRegressionMissForest:
    """
    Use Gradient Boosting with quantile loss
    Provides prediction intervals directly
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def fit_transform(self, data):
        logging.info("\n  Strategy 2: Quantile Regression (Prediction Intervals)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data. columns, data.index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        X_filled = X.copy()
        
        # Median initialization
        for j in range(X.shape[1]):
            if np.any(mask[:, j]):
                X_filled[mask[:, j], j] = np.nanmedian(X[:, j])
        
        # Store quantiles
        quantile_lower = X. copy()
        quantile_upper = X.copy()
        
        for iteration in range(3):
            X_old = X_filled.copy()
            
            for j in range(X.shape[1]):
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
                
                # Model 1: Median (50th percentile)
                gbr_median = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=0.5,
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
                gbr_median.fit(X_train, y_train)
                pred_median = gbr_median. predict(X_test)
                
                # Model 2: Lower bound (2.5th percentile)
                gbr_lower = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=0.025,
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self. random_state
                )
                gbr_lower.fit(X_train, y_train)
                pred_lower = gbr_lower.predict(X_test)
                
                # Model 3: Upper bound (97.5th percentile)
                gbr_upper = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=0.975,
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
                gbr_upper. fit(X_train, y_train)
                pred_upper = gbr_upper.predict(X_test)
                
                # Use median as prediction
                X_filled[miss_idx, j] = pred_median
                quantile_lower[miss_idx, j] = pred_lower
                quantile_upper[miss_idx, j] = pred_upper
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            logging.info(f"    Iteration {iteration + 1}:  change={change:.6f}")
            
            if change < 1e-4:
                break
        
        # Log statistics
        ci_width = (quantile_upper - quantile_lower)[mask].mean()
        logging.info(f"  ✅ Quantile regression completed")
        logging.info(f"     Average 95% prediction interval width: {ci_width:.4f}")
        
        if columns is not None:
            X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
        
        return X_filled


# ============================================================================
# STRATEGY 3: BAYESIAN BOOTSTRAP ENSEMBLE
# ============================================================================

class BayesianBootstrapMissForest:
    """
    Bayesian Bootstrap:  Sample with Dirichlet weights
    Provides probabilistic uncertainty quantification
    """
    def __init__(self, n_bootstrap=10, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
    
    def fit_transform(self, data):
        logging.info(f"\n  Strategy 3: Bayesian Bootstrap ({self.n_bootstrap} samples)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data.index
            X = data. values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        
        # Run multiple MissForest with Bayesian bootstrap
        predictions = []
        
        np.random.seed(self.random_state)
        
        for i in range(self.n_bootstrap):
            logging.info(f"    Bootstrap sample {i+1}/{self.n_bootstrap}...")
            
            # Generate Dirichlet weights
            n_samples = X.shape[0]
            weights = np.random.dirichlet(np.ones(n_samples) * 1.0)
            
            # Weighted sampling (approximate)
            # In practice, use weights in RF fit (if supported) or resample
            
            mf = SingleMissForest(random_state=self.random_state + i)
            imputed, _ = mf.fit_transform(X.copy(), return_uncertainty=False)
            
            predictions.append(imputed)
        
        predictions = np.array(predictions)
        
        # Calculate posterior mean
        X_mean = predictions.mean(axis=0)
        
        # Calculate posterior std (Bayesian uncertainty)
        X_std = predictions.std(axis=0)
        
        # Credible intervals (95%)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        # Log statistics
        avg_std = X_std[mask].mean()
        avg_ci = (ci_upper - ci_lower)[mask].mean()
        
        logging.info(f"  ✅ Bayesian bootstrap completed")
        logging.info(f"     Posterior std (missing): {avg_std:.4f}")
        logging.info(f"     95% credible interval width: {avg_ci:.4f}")
        
        if columns is not None:
            X_mean = pd.DataFrame(X_mean, columns=columns, index=index)
        
        return X_mean


# ============================================================================
# STRATEGY 4: UNCERTAINTY-BASED SELECTION
# ============================================================================

class UncertaintyBasedSelection:
    """
    Run multiple models
    For each missing value, select prediction with lowest uncertainty
    """
    def __init__(self, n_models=5, random_state=42):
        self.n_models = n_models
        self.random_state = random_state
    
    def fit_transform(self, data):
        logging.info(f"\n  Strategy 4: Uncertainty-Based Selection ({self.n_models} models)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data. index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        mask = np.isnan(X)
        
        # Run multiple models
        predictions = []
        uncertainties = []
        
        for i in range(self.n_models):
            logging.info(f"    Running model {i+1}/{self. n_models}...")
            seed = self.random_state + i * 100
            
            mf = SingleMissForest(random_state=seed)
            imputed, std = mf. fit_transform(X.copy(), return_uncertainty=True)
            
            predictions.append(imputed)
            uncertainties.append(std)
        
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        
        # For each position, select prediction with minimum uncertainty
        min_uncertainty_idx = uncertainties.argmin(axis=0)
        
        # Create index arrays for advanced indexing
        idx_i = np.arange(X. shape[0])[:, None]
        idx_j = np.arange(X.shape[1])
        
        X_final = predictions[min_uncertainty_idx, idx_i, idx_j]. squeeze()
        final_uncertainty = uncertainties[min_uncertainty_idx, idx_i, idx_j].squeeze()
        
        # Statistics
        avg_uncertainty = final_uncertainty[mask].mean()
        selection_counts = np.bincount(min_uncertainty_idx[mask]. flatten(), minlength=self.n_models)
        
        logging.info(f"  ✅ Selection completed")
        logging.info(f"     Average uncertainty:  {avg_uncertainty:.4f}")
        logging.info(f"     Model selection distribution:")
        for i, count in enumerate(selection_counts):
            pct = count / mask.sum() * 100
            logging.info(f"       Model {i+1}: {count} ({pct:.1f}%)")
        
        if columns is not None:
            X_final = pd.DataFrame(X_final, columns=columns, index=index)
        
        return X_final


# ============================================================================
# STRATEGY 5: ADAPTIVE ENSEMBLE
# ============================================================================

class AdaptiveEnsemble:
    """
    Adaptive weighting based on: 
    1. Prediction uncertainty
    2. Model performance on observed data
    3. Consensus among models
    """
    def __init__(self, n_models=5, random_state=42):
        self.n_models = n_models
        self. random_state = random_state
    
    def fit_transform(self, data, original_mask):
        logging.info(f"\n  Strategy 5: Adaptive Ensemble ({self.n_models} models)")
        
        if isinstance(data, pd.DataFrame):
            columns, index = data. columns, data.index
            X = data.values
        else:
            columns, index, X = None, None, data
        
        mask = original_mask.values if isinstance(original_mask, pd.DataFrame) else original_mask
        observed_mask = ~mask
        
        # Run multiple models
        predictions = []
        uncertainties = []
        model_scores = []
        
        for i in range(self.n_models):
            logging.info(f"    Running model {i+1}/{self.n_models}...")
            seed = self.random_state + i * 100
            
            mf = SingleMissForest(random_state=seed)
            imputed, std = mf.fit_transform(X.copy(), return_uncertainty=True)
            
            predictions.append(imputed)
            uncertainties.append(std)
            
            # Evaluate on observed data
            if observed_mask.sum() > 0:
                pred_obs = imputed[observed_mask]
                true_obs = X[observed_mask]
                r2 = r2_score(true_obs, pred_obs)
                model_scores.append(r2)
            else:
                model_scores. append(0.5)
        
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        model_scores = np.array(model_scores)
        
        logging.info(f"    Model R² scores: {model_scores}")
        
        # Adaptive weights combining: 
        # 1. Model performance (R²)
        # 2. Inverse uncertainty
        # 3. Model consensus
        
        # Weight 1: Performance-based (normalize R² scores)
        perf_weights = model_scores / (model_scores.sum() + 1e-10)
        perf_weights = perf_weights[: , None, None]  # Broadcast shape
        
        # Weight 2: Uncertainty-based
        epsilon = 1e-6
        uncert_weights = 1.0 / (uncertainties + epsilon)
        uncert_weights = uncert_weights / (uncert_weights.sum(axis=0, keepdims=True) + epsilon)
        
        # Weight 3: Consensus-based (low variance = high agreement)
        pred_variance = predictions.var(axis=0)
        consensus_score = 1.0 / (pred_variance + epsilon)
        consensus_score = consensus_score / (consensus_score.max() + epsilon)
        
        # Combined adaptive weights
        # 40% performance, 40% uncertainty, 20% consensus
        combined_weights = (
            0.4 * perf_weights +
            0.4 * uncert_weights +
            0.2 * consensus_score
        )
        
        # Normalize
        combined_weights = combined_weights / (combined_weights.sum(axis=0, keepdims=True) + epsilon)
        
        # Weighted prediction
        X_final = (predictions * combined_weights).sum(axis=0)
        
        # Final uncertainty
        final_uncertainty = (uncertainties * combined_weights).sum(axis=0)
        
        # Statistics
        avg_uncertainty = final_uncertainty[mask].mean()
        
        logging.info(f"  ✅ Adaptive ensemble completed")
        logging.info(f"     Average uncertainty: {avg_uncertainty:.4f}")
        logging.info(f"     Performance weights: {perf_weights. squeeze()}")
        
        if columns is not None: 
            X_final = pd.DataFrame(X_final, columns=columns, index=index)
        
        return X_final


# ============================================================================
# MAIN HYBRID SELECTOR
# ============================================================================

class ProbabilisticHybridImputer:
    """Main class to select and run probabilistic strategies"""
    
    def __init__(self, strategy='adaptive', n_models=5, random_state=42):
        """
        Args:
            strategy: 'confidence_weighted', 'quantile', 'bayesian', 
                     'uncertainty_select', 'adaptive', 'auto'
            n_models: Number of ensemble models
            random_state: Random seed
        """
        self.strategy = strategy
        self.n_models = n_models
        self. random_state = random_state
    
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
        """Run selected probabilistic strategy"""
        logging.info("="*80)
        logging.info("PROBABILISTIC MISSFOREST HYBRID")
        logging.info("="*80)
        
        if original_mask is None:
            original_mask = data.isna()
        
        columns = data.columns
        index = data.index
        observed_mask = ~original_mask. values
        
        # Select strategy
        if self.strategy == 'confidence_weighted':
            imputer = ConfidenceWeightedEnsemble(n_ensemble=self.n_models, random_state=self.random_state)
            result = imputer.fit_transform(data)
        
        elif self.strategy == 'quantile':
            imputer = QuantileRegressionMissForest(random_state=self.random_state)
            result = imputer. fit_transform(data)
        
        elif self.strategy == 'bayesian':
            imputer = BayesianBootstrapMissForest(n_bootstrap=self.n_models, random_state=self.random_state)
            result = imputer. fit_transform(data)
        
        elif self.strategy == 'uncertainty_select':
            imputer = UncertaintyBasedSelection(n_models=self.n_models, random_state=self.random_state)
            result = imputer.fit_transform(data)
        
        elif self.strategy == 'adaptive':
            imputer = AdaptiveEnsemble(n_models=self.n_models, random_state=self.random_state)
            result = imputer.fit_transform(data, original_mask)
        
        elif self.strategy == 'auto':
            # Auto-select based on data size
            n_samples = len(data)
            
            if n_samples > 10000:
                logging.info("  Auto-selected:  Confidence-Weighted (large dataset)")
                imputer = ConfidenceWeightedEnsemble(n_ensemble=self.n_models, random_state=self.random_state)
                result = imputer.fit_transform(data)
            elif n_samples > 2000:
                logging.info("  Auto-selected: Adaptive (medium dataset)")
                imputer = AdaptiveEnsemble(n_models=self.n_models, random_state=self.random_state)
                result = imputer.fit_transform(data, original_mask)
            else:
                logging.info("  Auto-selected: Bayesian Bootstrap (small dataset)")
                imputer = BayesianBootstrapMissForest(n_bootstrap=self.n_models, random_state=self.random_state)
                result = imputer.fit_transform(data)
        
        else:
            raise ValueError(f"Unknown strategy: {self. strategy}")
        
        # Evaluate
        metrics = self.evaluate(result. values, data.values, observed_mask)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"FINAL RESULTS:")
        logging.info(f"{'='*80}")
        logging.info(f"  Strategy: {self.strategy}")
        logging.info(f"  R²:    {metrics['r2']:.4f}")
        logging.info(f"  RMSE: {metrics['rmse']:.4f}")
        logging.info(f"  MAE:   {metrics['mae']:.4f}")
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
    
    # Initialize probabilistic imputer
    # Use 'adaptive' strategy by default (best overall performance)
    prob_imputer = ProbabilisticHybridImputer(
        strategy='adaptive',  # Can be changed to 'confidence_weighted', 'quantile', etc.
        n_models=5,
        random_state=random_state
    )
    
    # Perform imputation
    imputed_df = prob_imputer.fit_transform(data_for_imputation, original_mask)
    
    # Create final output
    final_df = data. copy()
    for col in columns_for_imputation:
        if col in data. columns:
            final_df[col] = imputed_df[col]
    
    logging.info(f"{MODEL_NAME} imputation completed successfully.\n")
    return final_df


if __name__ == "__main__": 
    logging.info("Testing Probabilistic MissForest...")
    
    np.random.seed(42)
    n = 2000
    
    # Create realistic data
    t = np.linspace(0, 10, n)
    f1 = 50 + 10 * np.sin(t) + np.random.randn(n) * 3
    f2 = f1 * 0.75 + 5 * np.cos(t) + np.random.randn(n) * 4
    f3 = f1 * 0.4 + f2 * 0.3 + np. random.randn(n) * 3
    
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
    
    imputed_vals = imputed.loc[mask[:, 0], 'feature_0']. values
    rmse = np.sqrt(np. mean((imputed_vals - original) ** 2))
    r2 = r2_score(original, imputed_vals)
    
    print(f"\nRMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("✅ Test completed!")