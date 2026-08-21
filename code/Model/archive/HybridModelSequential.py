"""
Sequential Hybrid Imputation Model:  MissForest → BRITS
Stage 1: MissForest fills all missing values (robust baseline)
Stage 2: BRITS refines ONLY the originally missing values using temporal patterns

Author: Dr. Masrur
Last Updated: 2026-01-05
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from spatial import prepare_spatial_temporal_data

# Prefer TensorFlow for temporal refinement; PyTorch removed
HAS_TORCH = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
    logging.info("TensorFlow available — will use TensorFlow temporal refiner")
except Exception:
    HAS_TF = False

from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "SequentialHybrid"


# BRITS (PyTorch) implementation removed — TensorFlow GRU refiner will be used instead when available


def calculate_time_intervals(data):
    """Calculate time intervals for BRITS"""
    if 'DateTime' in data.columns:
        time_col = pd.to_datetime(data['DateTime'])
        delta = time_col.diff().dt.total_seconds() / 3600
        delta. fillna(1.0, inplace=True)
        return delta. values
    else:
        return np.ones(len(data))


# ============================================================================
# MISSFOREST COMPONENTS (Enhanced with Confidence Scoring)
# ============================================================================

class MissForestImputer: 
    """MissForest imputer with confidence scoring for sequential processing"""
    def __init__(self, max_iter=5, n_estimators=100, random_state=42):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_importances_ = None
        self.oob_scores_ = None
    
    def fit_transform(self, X, return_confidence=False):
        """
        Impute missing values using Random Forest
        
        Args:
            X: numpy array or DataFrame with NaN for missing values
            return_confidence:  Whether to return confidence scores
        
        Returns:
            Imputed matrix (and optionally confidence scores)
        """
        if isinstance(X, pd.DataFrame):
            columns = X.columns
            index = X.index
            X = X.values
        else:
            columns = None
            index = None
        
        mask = np.isnan(X)
        
        # Initial imputation with mean
        X_filled = X.copy()
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X_filled[mask[: , j], j] = col_means[j]
        
        # Store confidence scores (based on OOB error)
        confidence_scores = np.ones_like(X)
        
        # Iterate
        for iteration in range(self.max_iter):
            X_old = X_filled.copy()
            
            # For each column with missing values
            for j in range(X.shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[:, j]
                miss_idx = mask[:, j]
                
                features = [i for i in range(X.shape[1]) if i != j]
                
                if len(features) == 0:
                    continue
                
                X_train = X_filled[obs_idx][: , features]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, features]
                
                # Train Random Forest with OOB scoring
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10,
                    oob_score=True  # Enable out-of-bag scoring
                )
                rf.fit(X_train, y_train)
                
                # Predict
                y_pred = rf.predict(X_test)
                X_filled[miss_idx, j] = y_pred
                
                # Calculate confidence based on OOB score and prediction variance
                oob_score = rf.oob_score_ if hasattr(rf, 'oob_score_') else 0.5
                
                # Get prediction variance from individual trees
                predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])
                pred_std = np.std(predictions, axis=0)
                
                # Confidence:  higher OOB score and lower variance = higher confidence
                # Normalize to 0-1 range
                confidence = oob_score * (1 - np.tanh(pred_std / (np.std(y_train) + 1e-6)))
                confidence_scores[miss_idx, j] = confidence
            
            # Check convergence
            change = np.sum((X_filled - X_old) ** 2) / np.sum(X_old ** 2)
            
            if change < 1e-4:
                logging.info(f"  MissForest converged at iteration {iteration + 1}")
                break
        
        if columns is not None:
            X_filled_df = pd.DataFrame(X_filled, columns=columns, index=index)
            if return_confidence:
                confidence_df = pd.DataFrame(confidence_scores, columns=columns, index=index)
                return X_filled_df, confidence_df
            return X_filled_df
        
        if return_confidence:
            return X_filled, confidence_scores
        return X_filled


# ============================================================================
# SEQUENTIAL HYBRID MODEL
# ============================================================================

class SequentialHybridImputer:
    """
    Sequential Hybrid Imputation:  MissForest → BRITS
    
    Stage 1: MissForest
    - Fills ALL missing values
    - Provides robust baseline using ensemble of decision trees
    - Generates confidence scores for each imputed value
    
    Stage 2: BRITS Refinement
    - Uses MissForest output as initial values
    - Trains on observed + MissForest-imputed data
    - Refines ONLY originally missing values using temporal patterns
    - Low-confidence MissForest predictions get more refinement
    
    Integration Strategy:
    - Observed values: Keep original (never change)
    - Missing values: MissForest → BRITS refinement
    - BRITS learns temporal patterns from the complete (MF-filled) sequence
    """
    def __init__(self, 
                 missforest_max_iter=5,
                 brits_max_iter=50,
                 n_estimators=100,
                 hidden_size=64,
                 refinement_strength=0.8,  # How much BRITS refinement to apply
                 random_state=42):
        """
        Args:
            missforest_max_iter: Max iterations for MissForest
            brits_max_iter: Max iterations for BRITS training
            n_estimators: Number of trees in Random Forest
            hidden_size: Hidden size for BRITS
            refinement_strength: How much to trust BRITS refinement (0-1)
                                0 = keep MissForest only
                                1 = full BRITS refinement
            random_state: Random seed
        """
        self.missforest_max_iter = missforest_max_iter
        self.brits_max_iter = brits_max_iter
        self.n_estimators = n_estimators
        self.hidden_size = hidden_size
        self.refinement_strength = refinement_strength
        self.random_state = random_state
    
    def fit_transform(self, data, original_mask=None, use_brits=True):
        """
        Perform sequential hybrid imputation
        
        Args: 
            data: DataFrame with missing values
            original_mask: Boolean mask of originally missing values (optional)
            use_brits:  Whether to use BRITS refinement (uses TensorFlow GRU refiner when available)
        
        Returns:
            Imputed DataFrame
        """
        logging.info("="*80)
        logging.info("SEQUENTIAL HYBRID IMPUTATION:  MissForest → BRITS")
        logging.info("="*80)
        
        # Store original data and mask
        original_data = data.copy()
        if original_mask is None:
            original_mask = data.isna()
        
        columns = data.columns
        index = data.index
        
        # ====================================================================
        # STAGE 1: MissForest Baseline Imputation
        # ====================================================================
        logging.info("\n[Stage 1/2] MissForest:  Generating robust baseline...")
        logging.info("  Purpose: Fill all missing values with ensemble predictions")
        
        missforest = MissForestImputer(
            max_iter=self. missforest_max_iter,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        
        # Get MissForest predictions with confidence scores
        missforest_imputed, confidence_scores = missforest. fit_transform(
            data, 
            return_confidence=True
        )
        
        if isinstance(missforest_imputed, pd.DataFrame):
            missforest_values = missforest_imputed. values
            confidence_values = confidence_scores.values
        else:
            missforest_values = missforest_imputed
            confidence_values = confidence_scores
        
        # Calculate average confidence for originally missing values
        missing_confidence = confidence_values[original_mask. values].mean()
        logging.info(f"✅ MissForest completed")
        logging.info(f"  Average confidence on missing values: {missing_confidence:.3f}")
        
        # ====================================================================
        # STAGE 2: Temporal Refinement using TensorFlow (GRU)
        # ====================================================================
        data_filled = missforest_values.copy()
        if use_brits and HAS_TF:
            logging.info("\n[Stage 2/2] Temporal refinement of missing values (TensorFlow)")
            logging.info("  Purpose: Refine imputed values using temporal patterns")
            logging.info(f"  Refinement strength:  {self.refinement_strength:.2f}")

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_filled)

            # Create mask: 1 = observed, 0 = originally missing
            mask = (~original_mask.values).astype(float)

            # Prepare sequence inputs for TF model
            seq_X = data_scaled.reshape(1, data_scaled.shape[0], data_scaled.shape[1])
            seq_mask = mask.reshape(1, mask.shape[0], mask.shape[1])

            input_shape = (seq_X.shape[1], seq_X.shape[2])
            tf.keras.backend.clear_session()

            inputs = keras.Input(shape=input_shape, name='X')
            mask_input = keras.Input(shape=input_shape, name='M')
            x = layers.Concatenate(axis=-1)([inputs, mask_input])
            x = layers.GRU(self.hidden_size, return_sequences=True, activation='tanh')(x)
            x = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)
            tf_model = keras.Model([inputs, mask_input], x)

            tf_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

            # Keras' MSE reduces the feature axis before applying sample
            # weights, so weights must be (batch, time), not
            # (batch, time, features). Use the observed-feature proportion at
            # each timestamp as the temporal sample weight.
            sample_weight = seq_mask.mean(axis=-1)

            es = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, min_delta=1e-6)
            try:
                tf_model.fit([seq_X, seq_mask], seq_X, epochs=min(100, self.brits_max_iter), batch_size=1, verbose=0,
                             sample_weight=sample_weight, callbacks=[es])
            except Exception as e:
                logging.warning(f"TensorFlow refinement failed: {e}. Falling back to MissForest-only.")
                brits_values = data_filled
            else:
                pred = tf_model.predict([seq_X, seq_mask], verbose=0)[0]
                brits_values = scaler.inverse_transform(pred)

            logging.info("✅ TensorFlow temporal refinement completed")
        else:
            if use_brits and not HAS_TF:
                logging.warning("No TensorFlow backend available. Using MissForest-only (no temporal refinement).")
            brits_values = data_filled

        # ====================================================================
        # STAGE 3: Intelligent Integration
        # ====================================================================
        # If temporal refiner produced values (or we fell back to MF-filled), blend accordingly
        if use_brits:
            logging.info("\n[Stage 3/3] Integrating predictions...")

            final_values = original_data.values.copy()
            for i in range(final_values.shape[0]):
                for j in range(final_values.shape[1]):
                    if original_mask.iloc[i, j]:  # Originally missing
                        mf_value = missforest_values[i, j]
                        brits_value = brits_values[i, j]
                        confidence = confidence_values[i, j]

                        brits_weight = self.refinement_strength * (1 - confidence)
                        mf_weight = 1 - brits_weight

                        final_values[i, j] = mf_weight * mf_value + brits_weight * brits_value

            # Calculate how much refinement was applied
            missing_mask_array = original_mask.values
            mf_missing = missforest_values[missing_mask_array]
            final_missing = final_values[missing_mask_array]
            refinement_magnitude = np.mean(np.abs(final_missing - mf_missing)) if final_missing.size > 0 else 0.0

            logging.info(f"✅ Integration completed")
            logging.info(f"  Average refinement magnitude: {refinement_magnitude:.4f}")
            logging.info(f"  Refinement applied to {missing_mask_array.sum()} missing values")
        else:
            # Temporal refinement disabled — use MissForest only
            logging.info("\nTemporal refinement disabled. Using MissForest-only.")
            final_values = missforest_values
        
        # Return as DataFrame
        return pd.DataFrame(final_values, columns=columns, index=index)


# ============================================================================
# MAIN IMPUTATION FUNCTION (Compatible with pipeline)
# ============================================================================

def impute_mice(data, target_column, input_columns, max_iter=50, random_state=42, 
                tol=0.01, custom_strategies=None, spatial_config=None):
    """
    Sequential hybrid imputation function compatible with main pipeline
    
    Args: 
        data: Input DataFrame
        target_column: Target column to impute
        input_columns:  List of input columns
        max_iter: Maximum iterations for BRITS
        random_state: Random seed
        tol: Tolerance (not used)
        custom_strategies: Custom imputation strategies
        spatial_config:  Spatial-temporal configuration
    
    Returns: 
        Imputed DataFrame
    """
    logging.info(f"Starting {MODEL_NAME} imputation...")
    
    np.random.seed(random_state)
    if HAS_TF:
        try:
            tf.random.set_seed(random_state)
        except Exception:
            pass
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data.")
    
    # Ensure target column is not in input columns
    if target_column in input_columns:
        input_columns = [col for col in input_columns if col != target_column]
    
    # Prepare data with spatial-temporal features if configured
    if spatial_config and (spatial_config.get('use_spatial', False) or spatial_config.get('use_temporal', False)):
        logging.info("Preparing data with spatial-temporal features...")
        data_enhanced, enhanced_input_columns = prepare_spatial_temporal_data(
            data, target_column, input_columns, spatial_config
        )
        input_columns_to_use = enhanced_input_columns
        data_to_use = data_enhanced
    else:
        input_columns_to_use = input_columns
        data_to_use = data. copy()
    
    # Apply custom strategies if provided
    if custom_strategies:
        for col, strategy in custom_strategies.items():
            if col in data_to_use.columns:
                if strategy == "mean":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col]. mean())
                elif strategy == "median":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].median())
                elif isinstance(strategy, (int, float)):
                    data_to_use[col] = data_to_use[col].fillna(strategy)
    
    # Prepare data for imputation
    columns_for_imputation = input_columns_to_use + [target_column]
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    data_for_imputation = data_to_use[columns_for_imputation]. copy()
    
    # Store original missing mask
    original_mask = data_for_imputation.isna()
    
    # Initialize sequential hybrid imputer
    sequential_imputer = SequentialHybridImputer(
        missforest_max_iter=5,
        brits_max_iter=max_iter,
        n_estimators=100,
        hidden_size=64,
        refinement_strength=0.8,  # 80% BRITS refinement on low-confidence predictions
        random_state=random_state
    )
    
    # Perform sequential hybrid imputation
    imputed_df = sequential_imputer.fit_transform(
        data_for_imputation,
        original_mask=original_mask,
        use_brits=HAS_TF
    )
    
    # Create final output
    final_df = data. copy()
    for col in columns_for_imputation:
        if col in data.columns:
            final_df[col] = imputed_df[col]
    
    logging.info(f"{MODEL_NAME} imputation completed successfully.")
    return final_df


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__": 
    # Test the sequential hybrid model
    logging.info("Testing Sequential Hybrid Model...")
    
    # Generate sample time series data with temporal patterns
    np.random.seed(42)
    n_samples = 2000
    n_features = 5
    
    # Create temporal patterns
    t = np.linspace(0, 4*np.pi, n_samples)
    data = pd.DataFrame({
        'feature_0': 10 + 5 * np.sin(t) + np.random.randn(n_samples) * 0.5,
        'feature_1':  20 + 3 * np.cos(t) + np.random.randn(n_samples) * 0.5,
        'feature_2':  15 + 2 * np. sin(2*t) + np.random.randn(n_samples) * 0.5,
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.randn(n_samples),
    })
    data['DateTime'] = pd.date_range('2020-01-01', periods=n_samples, freq='H')
    
    # Introduce missing values (30%)
    mask = np.random.random(data[['feature_0', 'feature_1', 'feature_2']].shape) < 0.3
    data. loc[mask[: , 0], 'feature_0'] = np.nan
    data.loc[mask[:, 1], 'feature_1'] = np.nan
    data. loc[mask[:, 2], 'feature_2'] = np.nan
    
    print(f"Data shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum()}")
    
    # Test imputation
    imputed = impute_mice(
        data,
        target_column='feature_0',
        input_columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'],
        max_iter=30,
        random_state=42
    )
    
    print(f"\nImputed data missing values:\n{imputed.isnull().sum()}")
    print("\n✅ Sequential Hybrid model test completed!")
