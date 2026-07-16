import logging
import numpy as np
import pandas as pd
import warnings

MODEL_NAME = "TFAutoencoder"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _prepare_sample_weight_for_keras(sample_weight, y):
    """
    Ensure `sample_weight` is compatible with Keras loss reduction.

    - If `sample_weight` is 1D (n_samples,), broadcast to (n_samples, n_outputs).
    - If `sample_weight` is 2D with wrong second dimension, try to broadcast or tile to match.
    - If unable to reconcile shapes, return None (Keras will treat as no sample_weight).
    """
    if sample_weight is None:
        return None

    sw = np.asarray(sample_weight)
    if sw.ndim == 0:
        return float(sw)

    y_arr = np.asarray(y)
    # y can be (n_samples,) or (n_samples, n_outputs)
    if y_arr.ndim == 1:
        if sw.ndim == 1 and sw.shape[0] == y_arr.shape[0]:
            return sw
        return None

    n_samples, n_out = y_arr.shape

    if sw.ndim == 1:
        if sw.shape[0] == n_samples:
            return np.repeat(sw[:, np.newaxis], n_out, axis=1)
        if sw.shape[0] == n_out:
            return np.repeat(sw[np.newaxis, :], n_samples, axis=0)
        return None

    if sw.ndim == 2:
        if sw.shape[0] == n_samples and sw.shape[1] == n_out:
            return sw
        if sw.shape[0] == n_samples and sw.shape[1] == 1:
            return np.repeat(sw, n_out, axis=1)
        if sw.shape[1] == n_out and sw.shape[0] == 1:
            return np.repeat(sw, n_samples, axis=0)
        return None

    return None


def _sklearn_iterative_impute(df, columns, max_iter=10, random_state=42, tol=0.01):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state, tol=tol)
        arr = imputer.fit_transform(df[columns])
    return pd.DataFrame(arr, columns=columns, index=df.index)


def impute_mice(
    data,
    target_column,
    input_columns,
    max_iter=50,
    random_state=42,
    tol=0.01,
    custom_strategies=None,
    spatial_config=None,
    epochs=50,
    batch_size=128,
    hidden_dim=64,
    learning_rate=1e-3,
    **kwargs
):
    """
    TensorFlow denoising autoencoder imputer compatible with main.py.

    Behavior:
      - Tries to use TensorFlow if installed.
      - Masks missing entries and trains to reconstruct observed values only.
      - On any failure or if tf not available, falls back to sklearn IterativeImputer.
      - Returns a DataFrame aligned with the input `data`.
    """
    try:
        df = data.copy()
        if target_column not in df.columns:
            logging.warning(f"[{MODEL_NAME}] target {target_column} not in data — returning original")
            return df

        # Build input columns list (exclude target if present)
        features = [c for c in input_columns if c in df.columns and c != target_column]
        cols_for_impute = features + [target_column]
        if len(cols_for_impute) == 0:
            logging.warning(f"[{MODEL_NAME}] No columns available for imputation; returning original")
            return df

        # Apply custom strategies (simple)
        if custom_strategies:
            for col, strat in custom_strategies.items():
                if col in df.columns:
                    if strat == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    elif strat == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif isinstance(strat, (int, float)):
                        df[col] = df[col].fillna(strat)

        # Prepare numeric DataFrame for imputation (coerce)
        numeric = df[cols_for_impute].apply(pd.to_numeric, errors="coerce")

        # If no observed target values -> mean fill and return
        if numeric[target_column].notna().sum() == 0:
            logging.warning(f"[{MODEL_NAME}] target fully missing -> mean fill fallback")
            out = df.copy()
            out[target_column] = out[target_column].fillna(out[target_column].mean())
            return out

        # Try to import TensorFlow
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, losses, callbacks
            tf.random.set_seed(int(random_state))
            use_tf = True
        except Exception:
            logging.warning(f"[{MODEL_NAME}] TensorFlow not available; using sklearn fallback")
            use_tf = False

        if not use_tf:
            return _sklearn_iterative_impute(df, numeric.columns.tolist(), max_iter=max_iter, random_state=random_state, tol=tol).combine_first(df)

        # Build arrays
        X = numeric.values.astype(np.float32)
        mask = ~np.isnan(X)  # True where observed
        X_filled = np.nan_to_num(X, nan=0.0).astype(np.float32)

        n_samples, n_features = X_filled.shape

        # Prepare dataset: we train to predict X from X with mask + mask channel
        # Input to network: [X_filled, mask_float]
        mask_float = mask.astype(np.float32)
        inp = np.concatenate([X_filled, mask_float], axis=1)

        input_dim = inp.shape[1]

        # Model: simple dense autoencoder that takes inp and outputs reconstructed X (only original n_features)
        tf.keras.backend.clear_session()
        inp_layer = layers.Input(shape=(input_dim,), name="inp")
        h = layers.Dense(hidden_dim, activation="relu")(inp_layer)
        h = layers.Dense(hidden_dim, activation="relu")(h)
        out_layer = layers.Dense(n_features, activation="linear", name="recon")(h)
        model = models.Model(inputs=inp_layer, outputs=out_layer)

        optimizer = optimizers.Adam(learning_rate=learning_rate)
        # custom loss that uses mask encoded into y_true: we will pass y_train = [X_filled | mask]
        def masked_mse(y_true, y_pred):
            # y_true: concatenation of [X_filled, mask]
            n = tf.shape(y_pred)[1]
            y_true_vals = y_true[:, :n]
            mask_vals = y_true[:, n:]
            sq = tf.square(y_true_vals - y_pred) * mask_vals
            # sum over features then normalize by number of observed entries to avoid scale issues
            denom = tf.reduce_sum(mask_vals, axis=1)
            # avoid division by zero: where denom==0 set denom=1 to prevent NaN
            denom_safe = tf.where(denom > 0, denom, tf.ones_like(denom))
            per_sample = tf.reduce_sum(sq, axis=1) / denom_safe
            return tf.reduce_mean(per_sample)

        model.compile(optimizer=optimizer, loss=masked_mse)

        # Prepare y_train by concatenating X_filled and mask_float so loss can access mask
        y_train = np.concatenate([X_filled, mask_float], axis=1).astype(np.float32)

        # callbacks
        es = callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True, verbose=0)
        # fit using y_train (contains mask)
        model.fit(inp, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

        # inference
        pred = model.predict(inp, batch_size=batch_size)
        pred = pred.astype(np.float32)

        # Merge predictions into final DataFrame: only replace positions that were NaN originally (or you may fill all)
        final = df.copy()
        arr_final = X.copy()  # keep original numeric with nans
        # For entries missing in numeric, replace with preds
        nan_mask = np.isnan(arr_final)
        arr_final[nan_mask] = pred[nan_mask]
        imputed_df = pd.DataFrame(arr_final, columns=numeric.columns, index=numeric.index)

        # Assign back to final DataFrame for columns that exist
        for c in imputed_df.columns:
            final[c] = imputed_df[c].values

        return final

    except Exception as exc:
        logging.exception(f"[{MODEL_NAME}] failed, using sklearn fallback: {exc}")
        try:
            return _sklearn_iterative_impute(data, cols_for_impute, max_iter=max_iter, random_state=random_state, tol=tol).combine_first(data)
        except Exception:
            # Last resort
            out = data.copy()
            out[target_column] = out[target_column].fillna(out[target_column].mean())
            return out