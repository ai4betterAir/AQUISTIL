"""
TF_BRITS.py

Simplified TensorFlow-based BRITS-like imputer compatible with the pipeline.

- Trains a bidirectional GRU to predict the target (sequence-to-sequence).
- Loss is applied only on observed target timestamps (sample_weight = observed_mask).
- Falls back to sklearn IterativeImputer when TensorFlow is not available or on errors.
- Returns a DataFrame aligned with the input (same index, target column present).

Author: Assistant (adapted to DrMasrur pipeline)
"""
import logging
import numpy as np
import pandas as pd
import warnings

MODEL_NAME = "TF-BRITS"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _sklearn_iterative_impute(df, columns, max_iter=10, random_state=42, tol=0.01):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state, tol=tol)
        arr = imputer.fit_transform(df[columns])
    return pd.DataFrame(arr, columns=columns, index=df.index)


def _prepare_sequence_inputs(df, feature_cols, target_col):
    """
    Build numpy arrays for model:
      X: shape (1, seq_len, n_features)
      y: shape (1, seq_len, 1)
      mask_y: shape (1, seq_len) where 1=observed, 0=missing
    We keep batch dim = 1 because we model the whole site time series as one sequence.
    """
    seq_df = df[feature_cols + [target_col]].copy()
    seq_df = seq_df.apply(pd.to_numeric, errors="coerce")

    X_arr = seq_df[feature_cols].values.astype(np.float32)
    y_arr = seq_df[target_col].values.astype(np.float32)
    mask_y = (~np.isnan(y_arr)).astype(np.float32)

    # replace nans in X with 0 (network sees mask implicitly via missingness in y)
    X_arr = np.nan_to_num(X_arr, nan=0.0).astype(np.float32)

    # replace nan targets with 0 (we won't use them in loss)
    y_filled = np.nan_to_num(y_arr, nan=0.0).astype(np.float32)

    # add batch dim
    X = X_arr[np.newaxis, :, :]        # (1, seq_len, n_features)
    y = y_filled[np.newaxis, :, np.newaxis]  # (1, seq_len, 1)
    mask = mask_y[np.newaxis, :]      # (1, seq_len)

    return X, y, mask


def impute_mice(
    data,
    target_column,
    input_columns,
    max_iter=10,
    random_state=42,
    tol=0.01,
    custom_strategies=None,
    spatial_config=None,
    epochs=50,
    batch_size=1,
    hidden_dim=64,
    learning_rate=1e-3,
    **kwargs
):
    """
    TF BRITS-like imputer (pipeline-compatible).
    Trains a bidirectional GRU to predict the target sequence from features.
    If TF not available or training fails, falls back to IterativeImputer.
    """
    try:
        df = data.copy()
        if target_column not in df.columns:
            logging.warning(f"[{MODEL_NAME}] target '{target_column}' not in dataframe — returning original")
            return df

        # Build effective feature columns (only those actually present)
        feature_cols = [c for c in input_columns if c in df.columns and c != target_column]
        if spatial_config and (spatial_config.get('use_spatial', False) or spatial_config.get('use_temporal', False)):
            # try to enhance data with spatial-temporal features
            try:
                from spatial import prepare_spatial_temporal_data
                df_enh, feat_list = prepare_spatial_temporal_data(df, target_column, feature_cols, spatial_config)
                # use features recommended by prepare_spatial_temporal_data (which can include spatial_*)
                feature_cols = [c for c in feat_list if c in df_enh.columns]
                # Set df to enhanced for training/prediction
                df_for_model = df_enh.copy()
            except Exception as e:
                logging.warning(f"[{MODEL_NAME}] prepare_spatial_temporal_data failed: {e}; falling back to original df")
                df_for_model = df.copy()
        else:
            df_for_model = df.copy()

        # Apply custom strategies (simple deterministic fills)
        if custom_strategies:
            for col, strat in custom_strategies.items():
                if col in df_for_model.columns:
                    if strat == "mean":
                        df_for_model[col] = df_for_model[col].fillna(df_for_model[col].mean())
                    elif strat == "median":
                        df_for_model[col] = df_for_model[col].fillna(df_for_model[col].median())
                    elif isinstance(strat, (int, float)):
                        df_for_model[col] = df_for_model[col].fillna(strat)

        # If we ended up with no features (rare), fall back to simple mean fill for target
        if len(feature_cols) == 0:
            logging.warning(f"[{MODEL_NAME}] No feature columns available; filling target with mean")
            out = df.copy()
            out[target_column] = out[target_column].fillna(out[target_column].mean())
            return out

        # Prepare arrays
        X, y, mask = _prepare_sequence_inputs(df_for_model, feature_cols, target_column)

        # If there are no observed target values -> cannot train; do mean fallback
        if mask.sum() == 0:
            logging.warning(f"[{MODEL_NAME}] target fully missing -> mean fill fallback")
            out = df.copy()
            out[target_column] = out[target_column].fillna(out[target_column].mean())
            return out

        # Try to import TF
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, losses
            tf.random.set_seed(int(random_state))
            use_tf = True
        except Exception:
            logging.warning(f"[{MODEL_NAME}] TensorFlow not available; using sklearn fallback")
            use_tf = False

        if not use_tf:
            cols = feature_cols + [target_column]
            imputed_df = _sklearn_iterative_impute(df_for_model, cols, max_iter=max_iter, random_state=random_state, tol=tol)
            final = df.copy()
            # assign only target back
            if target_column in imputed_df.columns:
                final[target_column] = imputed_df[target_column].values
            return final

        # Build model: sequence-to-sequence predicting target only
        seq_len = X.shape[1]
        n_features = X.shape[2]

        # Build a small seq2seq with a Bidirectional GRU and time-distributed output
        from tensorflow.keras import layers, models, optimizers
        inputs = layers.Input(shape=(seq_len, n_features), name="inputs")
        x = layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True), merge_mode="concat")(inputs)
        x = layers.TimeDistributed(layers.Dense(hidden_dim, activation="relu"))(x)
        out = layers.TimeDistributed(layers.Dense(1, activation="linear"), name="target_out")(x)
        model = models.Model(inputs=inputs, outputs=out)

        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        # Ensure sample_weight has shape (batch, seq_len) and dtype float32
        sample_weight = np.asarray(mask, dtype=np.float32)  # (1, seq_len)
        if sample_weight.ndim == 2 and sample_weight.shape[0] == 1 and batch_size != 1:
            # In case pipeline accidentally passes a batch_size >1, broadcast to batch
            sample_weight = np.repeat(sample_weight, batch_size, axis=0)

        # Fit model. Use batch_size=1 (sequence per site).
        try:
            model.fit(X, y, sample_weight=sample_weight, epochs=epochs, batch_size=1, verbose=0)
        except Exception as e:
            logging.warning(f"[{MODEL_NAME}] TF training failed: {e}; falling back to sklearn imputer")
            cols = feature_cols + [target_column]
            imputed_df = _sklearn_iterative_impute(df_for_model, cols, max_iter=max_iter, random_state=random_state, tol=tol)
            final = df.copy()
            if target_column in imputed_df.columns:
                final[target_column] = imputed_df[target_column].values
            return final

        # Inference
        pred = model.predict(X, batch_size=1)
        # pred shape: (1, seq_len, 1) -> squeeze
        pred = pred[0, :, 0].astype(np.float32)

        # Build final DataFrame: keep observed target values, replace missing positions with predictions
        final = df.copy()
        orig_target = pd.to_numeric(df_for_model[target_column], errors='coerce').values
        imputed_vals = orig_target.copy()
        missing_positions = np.isnan(orig_target)
        imputed_vals[missing_positions] = pred[missing_positions]

        # assign back to final
        final[target_column] = imputed_vals

        return final

    except Exception as exc:
        logging.exception(f"[{MODEL_NAME}] Unexpected failure; using sklearn fallback: {exc}")
        try:
            # best-effort fallback
            cols = [c for c in (input_columns if input_columns else []) if c in data.columns] + [target_column]
            imputed_df = _sklearn_iterative_impute(data, cols, max_iter=max_iter, random_state=random_state, tol=tol)
            final = data.copy()
            if target_column in imputed_df.columns:
                final[target_column] = imputed_df[target_column].values
            return final
        except Exception:
            out = data.copy()
            out[target_column] = out[target_column].fillna(out[target_column].mean())
            return out