"""
Hybrid BRITS + XGBoost Imputation Model
Combines temporal RNN (BRITS) with gradient boosting (XGBoost)

BRITS: Captures temporal dependencies (bidirectional RNN)
XGBoost: Captures non-linear cross-sectional patterns

Author: Dr.  Masrur
Last Updated: 2026-01-05
"""

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Try importing TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
    
    # Suppress warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    logging.info(f"✅ TensorFlow {tf.__version__} available")
except ImportError:
    HAS_TF = False
    logging.error("TensorFlow required:  pip install tensorflow")

# Try importing XGBoost
try: 
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.error("XGBoost required: pip install xgboost")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "HybridBRITS_XGB"


# ============================================================================
# BRITS COMPONENTS (TensorFlow)
# ============================================================================

if HAS_TF:
    class TemporalDecayLayer(layers.Layer):
        """Temporal decay mechanism"""
        def __init__(self, output_size, **kwargs):
            super(TemporalDecayLayer, self).__init__(**kwargs)
            self.output_size = output_size
        
        def build(self, input_shape):
            input_size = input_shape[-1]
            self.W = self.add_weight(
                name='decay_W',
                shape=(input_size, self.output_size),
                initializer='glorot_uniform',
                trainable=True
            )
            self.b = self.add_weight(
                name='decay_b',
                shape=(self. output_size,),
                initializer='zeros',
                trainable=True
            )
            super(TemporalDecayLayer, self).build(input_shape)
        
        def call(self, delta):
            linear = tf.matmul(delta, self.W) + self.b
            gamma = tf.exp(-tf.nn.relu(linear))
            return gamma
    

    class RITSCell(layers.Layer):
        """RITS Cell for BRITS"""
        def __init__(self, input_size, hidden_size, **kwargs):
            super(RITSCell, self).__init__(**kwargs)
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.state_size = hidden_size

        def build(self, input_shape):
            # input_shape is (batch, ) or (batch, features) for a single timestep; we rely on self.input_size
            input_size = self.input_size

            self.temp_decay_h = TemporalDecayLayer(self.hidden_size, name='decay_h')
            self.temp_decay_x = TemporalDecayLayer(input_size, name='decay_x')
            self.gru_cell = layers.GRUCell(self.hidden_size)
            self.regression = layers.Dense(input_size, name='regression')

            super(RITSCell, self).build(input_shape)

        def call(self, inputs, states):
            # `inputs` is a single tensor for the current timestep with last-dim = 3 * input_size
            h_prev = states[0]

            # Split concatenated inputs into x, mask, delta
            x = inputs[..., :self.input_size]
            mask = inputs[..., self.input_size:2 * self.input_size]
            delta = inputs[..., 2 * self.input_size: 3 * self.input_size]

            gamma_h = self.temp_decay_h(delta)
            h_decayed = gamma_h * h_prev

            x_imputed = self.regression(h_decayed)
            x_combined = mask * x + (1 - mask) * x_imputed

            gamma_x = self.temp_decay_x(delta)
            x_decayed = gamma_x * x_combined

            gru_input = tf.concat([x_decayed, mask], axis=-1)
            h_new, _ = self.gru_cell(gru_input, [h_decayed])

            return x_imputed, [h_new]
    

    class BRITSModel(keras.Model):
        """Bidirectional RITS Model"""
        def __init__(self, input_size, hidden_size=64, **kwargs):
            super(BRITSModel, self).__init__(**kwargs)
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            # RNNs expect a cell instance configured with input_size and hidden_size
            self.rits_forward = layers.RNN(
                RITSCell(self.input_size, self.hidden_size),
                return_sequences=True,
                name='rits_forward'
            )

            self.rits_backward = layers.RNN(
                RITSCell(self.input_size, self.hidden_size),
                return_sequences=True,
                go_backwards=True,
                name='rits_backward'
            )
            
            self.combine = layers.Dense(input_size, name='combine')
        
        def call(self, inputs, training=None):
            # Accept either a list/tuple of (x, mask, delta) or a single concatenated tensor
            if isinstance(inputs, (list, tuple)):
                x, mask, delta = inputs
                seq = tf.concat([x, mask, delta], axis=-1)
            else:
                seq = inputs

            forward_out = self.rits_forward(seq, training=training)
            backward_out = self.rits_backward(seq, training=training)
            backward_out = tf.reverse(backward_out, axis=[1])

            combined = tf.concat([forward_out, backward_out], axis=-1)
            final_imputation = self.combine(combined)

            return final_imputation, forward_out, backward_out


# ============================================================================
# BRITS IMPUTER
# ============================================================================

class BRITSImputer: 
    """BRITS-based imputation with TensorFlow"""
    def __init__(self, max_iter=50, hidden_size=64, random_state=42):
        if not HAS_TF: 
            raise ImportError("TensorFlow required")
        
        self.max_iter = max_iter
        self.hidden_size = hidden_size
        self.random_state = random_state
    
    def calculate_time_intervals(self, data):
        """Calculate time intervals for temporal decay"""
        if 'DateTime' in data.columns:
            time_col = pd.to_datetime(data['DateTime'])
            delta = time_col. diff().dt.total_seconds() / 3600
            delta. fillna(1.0, inplace=True)
            return delta. values
        else:
            return np.ones(len(data))
    
    def fit_transform(self, data):
        """Impute using BRITS"""
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data. index
            data_values = data.values
        else:
            columns, index = None, None
            data_values = data
        
        mask = (~np.isnan(data_values)).astype(float)
        
        # Initial imputation with median
        data_filled = data_values.copy()
        for j in range(data_values.shape[1]):
            col_mask = np.isnan(data_values[: , j])
            if np.any(col_mask):
                data_filled[col_mask, j] = np.nanmedian(data_values[:, j])
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_filled)
        
        # Calculate time intervals
        if isinstance(data, pd.DataFrame):
            delta_values = self.calculate_time_intervals(data)
        else:
            delta_values = np.ones(len(data_values))
        
        delta = np.tile(delta_values. reshape(-1, 1), (1, data_scaled.shape[1]))
        
        # Prepare tensors
        X = data_scaled. astype(np.float32).reshape(1, -1, data_scaled.shape[1])
        M = mask.astype(np.float32).reshape(1, -1, mask.shape[1])
        D = delta.astype(np. float32).reshape(1, -1, delta.shape[1])
        
        # Build BRITS model
        input_size = data_scaled.shape[1]
        brits_model = BRITSModel(input_size=input_size, hidden_size=self.hidden_size)
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Training function
        @tf.function
        def train_step(x, mask, delta):
            with tf. GradientTape() as tape:
                imputed, forward_out, backward_out = brits_model([x, mask, delta], training=True)
                
                # Loss on observed values
                obs_loss = tf.reduce_mean(tf.square((imputed - x) * mask))
                
                # Consistency loss
                consistency_loss = tf.reduce_mean(tf.square((forward_out - backward_out) * mask))
                
                # Smoothness loss
                diff = imputed[: , 1:, :] - imputed[:, :-1, :]
                smoothness_loss = tf.reduce_mean(tf.square(diff))
                
                total_loss = obs_loss + 0.1 * consistency_loss + 0.05 * smoothness_loss
            
            gradients = tape.gradient(total_loss, brits_model.trainable_variables)
            gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
            optimizer.apply_gradients(zip(gradients, brits_model. trainable_variables))
            
            return total_loss, obs_loss
        
        # Training loop
        logging.info("  Training BRITS...")
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            total_loss, obs_loss = train_step(
                tf.constant(X),
                tf.constant(M),
                tf.constant(D)
            )
            
            if (epoch + 1) % 10 == 0:
                logging.info(f"    Epoch {epoch + 1}/{self.max_iter}, Loss: {obs_loss. numpy():.6f}")
            
            # Early stopping
            if obs_loss. numpy() < best_loss - 0.0001:
                best_loss = obs_loss.numpy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"    Early stopping at epoch {epoch + 1}")
                    break
        
        # Get predictions
        brits_output, _, _ = brits_model([tf.constant(X), tf.constant(M), tf.constant(D)], training=False)
        brits_output = brits_output.numpy()[0]
        
        # Inverse transform
        brits_values = scaler.inverse_transform(brits_output)
        
        if columns is not None:
            brits_values = pd.DataFrame(brits_values, columns=columns, index=index)
        
        return brits_values


# ============================================================================
# XGBOOST IMPUTER
# ============================================================================

class XGBoostImputer:
    """XGBoost iterative imputation"""
    def __init__(self, max_iter=3, n_estimators=100, random_state=42):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required")
        
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit_transform(self, data):
        """Impute using XGBoost"""
        if isinstance(data, pd.DataFrame):
            columns, index = data.columns, data.index
            X_array = data.values
        else:
            columns, index, X_array = None, None, data
        
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
                
                xgb_model.fit(X_train, y_train, verbose=False)
                X_filled[miss_idx, j] = xgb_model.predict(X_test)
            
            change = np.linalg.norm(X_filled - X_old) / (np.linalg.norm(X_old) + 1e-10)
            logging.info(f"  XGBoost iter {iteration + 1}: change={change:.6f}")
            
            if change < 1e-4:
                logging.info(f"  XGBoost converged at iteration {iteration + 1}")
                break
        
        if columns is not None:
            X_filled = pd.DataFrame(X_filled, columns=columns, index=index)
        
        return X_filled


# ============================================================================
# HYBRID BRITS + XGBOOST IMPUTER
# ============================================================================

class HybridBRITSXGBImputer:
    """
    Hybrid imputation combining BRITS and XGBoost
    
    Strategy:
    1. BRITS:  Captures temporal patterns (sequential dependencies)
    2. XGBoost: Refines with non-linear cross-sectional patterns
    3. Ensemble: Weighted combination based on validation performance
    """
    def __init__(self, brits_epochs=50, xgb_iters=3, random_state=42):
        self.brits_epochs = brits_epochs
        self.xgb_iters = xgb_iters
        self.random_state = random_state
        
        if not HAS_TF or not HAS_XGBOOST:
            raise ImportError("Both TensorFlow and XGBoost required")
    
    def evaluate(self, predictions, true_values, observed_mask):
        """Evaluate on observed data"""
        obs_mask_flat = observed_mask.flatten()
        pred_obs = predictions.flatten()[obs_mask_flat]
        true_obs = true_values. flatten()[obs_mask_flat]
        
        r2 = r2_score(true_obs, pred_obs)
        rmse = np.sqrt(mean_squared_error(true_obs, pred_obs))
        mae = mean_absolute_error(true_obs, pred_obs)
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
    
    def fit_transform(self, data, original_mask=None):
        """Perform hybrid BRITS + XGBoost imputation"""
        logging.info("="*80)
        logging.info("HYBRID BRITS + XGBOOST IMPUTATION")
        logging.info("="*80)
        
        if original_mask is None:
            original_mask = data.isna()
        
        columns = data.columns
        index = data.index
        observed_mask = ~original_mask. values
        
        # ====================================================================
        # STAGE 1: BRITS (Temporal Patterns)
        # ====================================================================
        logging.info("\n[Stage 1/3] BRITS:  Temporal pattern learning...")
        
        brits_imputer = BRITSImputer(
            max_iter=self.brits_epochs,
            hidden_size=64,
            random_state=self. random_state
        )
        
        brits_imputed = brits_imputer.fit_transform(data)
        brits_metrics = self.evaluate(brits_imputed. values, data.values, observed_mask)
        logging.info(f"✅ BRITS:  R²={brits_metrics['r2']:.4f}, RMSE={brits_metrics['rmse']:.4f}")
        
        # ====================================================================
        # STAGE 2: XGBoost (Cross-sectional Refinement)
        # ====================================================================
        logging.info("\n[Stage 2/3] XGBoost: Cross-sectional refinement...")
        
        xgb_imputer = XGBoostImputer(
            max_iter=self.xgb_iters,
            n_estimators=100,
            random_state=self.random_state
        )
        
        # XGBoost uses BRITS output as starting point
        xgb_imputed = xgb_imputer.fit_transform(brits_imputed)
        xgb_metrics = self.evaluate(xgb_imputed.values, data.values, observed_mask)
        logging.info(f"✅ XGBoost: R²={xgb_metrics['r2']:.4f}, RMSE={xgb_metrics['rmse']:.4f}")
        
        # ====================================================================
        # STAGE 3: Ensemble
        # ====================================================================
        logging.info("\n[Stage 3/3] Ensemble: Finding optimal combination...")
        
        obs_mask_flat = observed_mask.flatten()
        brits_flat = brits_imputed.values.flatten()[obs_mask_flat]
        xgb_flat = xgb_imputed.values.flatten()[obs_mask_flat]
        true_flat = data.values.flatten()[obs_mask_flat]
        
        # Find optimal weight
        best_r2 = -999
        best_weight = 0.5
        
        for w in np.linspace(0, 1, 21):
            pred = w * brits_flat + (1 - w) * xgb_flat
            r2 = r2_score(true_flat, pred)
            if r2 > best_r2:
                best_r2 = r2
                best_weight = w
        
        logging.info(f"  Optimal weights:  BRITS={best_weight:.2f}, XGBoost={1-best_weight:.2f}")
        
        # Create ensemble
        ensemble_values = best_weight * brits_imputed.values + (1 - best_weight) * xgb_imputed.values
        ensemble_values[observed_mask] = data.values[observed_mask]
        
        ensemble_metrics = self.evaluate(ensemble_values, data.values, observed_mask)
        logging.info(f"✅ Ensemble: R²={ensemble_metrics['r2']:.4f}, RMSE={ensemble_metrics['rmse']:.4f}")
        
        # ====================================================================
        # FINAL SELECTION
        # ====================================================================
        candidates = [
            ('BRITS', brits_metrics['r2'], brits_imputed. values),
            ('XGBoost', xgb_metrics['r2'], xgb_imputed.values),
            ('Ensemble', ensemble_metrics['r2'], ensemble_values)
        ]
        
        best = max(candidates, key=lambda x:  x[1])
        
        logging.info(f"\n{'='*80}")
        logging.info(f"FINAL MODEL SELECTION:")
        logging.info(f"{'='*80}")
        for name, r2, _ in candidates:
            marker = " ← SELECTED" if name == best[0] else ""
            logging.info(f"  {name:15s} R²:  {r2:.4f}{marker}")
        logging.info(f"{'='*80}\n")
        
        return pd.DataFrame(best[2], columns=columns, index=index)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def impute_mice(data, target_column, input_columns, max_iter=50, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None):
    """
    Main imputation function
    NOTE:  Ignores spatial_config to prevent data leakage
    """
    logging.info(f"Starting {MODEL_NAME} imputation...")
    logging.info("⚠️  NOTE:  Spatial features are DISABLED to prevent data leakage")
    
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
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
    
    # Initialize hybrid imputer
    hybrid_imputer = HybridBRITSXGBImputer(
        brits_epochs=max_iter,
        xgb_iters=3,
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
    logging.info("Testing HybridBRITS_XGB...")
    
    np.random.seed(42)
    n = 2000
    
    # Create temporal patterns
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
    
    imputed = impute_mice(data, 'feature_0', ['feature_1', 'feature_2', 'feature_3'], max_iter=30)
    
    imputed_vals = imputed.loc[mask[:, 0], 'feature_0'].values
    rmse = np.sqrt(np. mean((imputed_vals - original) ** 2))
    r2 = r2_score(original, imputed_vals)
    
    print(f"\nRMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("✅ Test completed!")