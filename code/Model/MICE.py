"""
MICE.py
Safe wrapper for MICE imputation compatible with main.py

This wrapper delegates to the unified impute_with_method where available,
but also adds robust spatial-temporal preparation and defensive return behavior
so outputs are produced for every regime and missingness level.
"""
import logging
import os
import importlib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "MICE"

# Try to reuse the central impute_with_method if available
def _load_impute_with_method():
    try:
        # prefer local impute module
        from impute import impute_with_method
        return impute_with_method
    except Exception:
        pass
    try:
        from Model.impute import impute_with_method
        return impute_with_method
    except Exception:
        pass
    # fallback: attempt to import by path relative to this file
    try:
        import importlib.util
        impute_path = os.path.join(os.path.dirname(__file__), "impute.py")
        spec = importlib.util.spec_from_file_location("impute_local", impute_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "impute_with_method")
    except Exception:
        return None

_impute_with_method = _load_impute_with_method()

def _prepare_spatial_config_from_kwargs(kwargs):
    try:
        import config_spatial as cfg
        # Determine canonical site token: prefer kwargs site_name, else derive from model_name
        site_token = kwargs.get('site_name', '') or kwargs.get('model_name', '').rsplit('_', 1)[-1]
        if site_token:
            try:
                site_token = str(site_token).split('_')[0]
            except Exception:
                site_token = str(site_token)
        spatial_config = {
            'input_directory': getattr(cfg, "INPUT_DIRECTORY", None),
            'target_site': site_token,
            'use_spatial': getattr(cfg, "USE_SPATIAL_FEATURES", False),
            'use_temporal': getattr(cfg, "USE_TEMPORAL_FEATURES", True),
            'use_lagged': getattr(cfg, "USE_LAGGED_FEATURES", False),
            'use_rolling': getattr(cfg, "USE_ROLLING_FEATURES", False),
        }
        return spatial_config
    except Exception:
        return None

def impute_mice(data, target_column, input_columns, max_iter=10, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None, **kwargs):
    """
    Robust MICE wrapper.

    Returns a pd.DataFrame always (falls back to original input on error).
    """
    df_orig = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    try:
        df = df_orig.copy()
        # If caller didn't pass spatial_config, try to construct from kwargs
        if spatial_config is None:
            spatial_config = _prepare_spatial_config_from_kwargs(kwargs)

        # Delegate to central imputer if available (this will itself handle spatial_config)
        if _impute_with_method is not None:
            try:
                res = _impute_with_method(df, target_column, input_columns,
                                          method='mice',
                                          custom_strategies=custom_strategies,
                                          spatial_config=spatial_config,
                                          max_iter=max_iter,
                                          random_state=random_state,
                                          tol=tol,
                                          **kwargs)
                # Ensure a DataFrame is returned
                if isinstance(res, pd.DataFrame):
                    # align index/length with original
                    if len(res) == len(df):
                        res.index = df.index
                    # ensure target column exists
                    if target_column not in res.columns:
                        res[target_column] = df[target_column]
                    return res
                else:
                    logging.warning(f"{MODEL_NAME}: impute_with_method returned non-DataFrame ({type(res)}). Falling back to local IterativeImputer.")
            except Exception as e:
                logging.warning(f"{MODEL_NAME}: central imputer failed: {e}. Falling back to local MICE implementation.")

        # Local fallback: use sklearn IterativeImputer defensively
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        cols = [c for c in ([target_column] + input_columns) if c in df.columns]
        if not cols:
            logging.warning(f"{MODEL_NAME}: No columns available for imputation. Returning original DataFrame.")
            return df_orig

        estimator = kwargs.get('estimator', BayesianRidge())
        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=random_state, tol=tol)

        # Coerce to numeric where possible (defensive)
        for c in cols:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                pass

        arr = imputer.fit_transform(df[cols])
        imputed_df = pd.DataFrame(arr, columns=cols, index=df.index)
        final = df.copy()
        for c in cols:
            final[c] = imputed_df[c]
        return final

    except Exception as exc:
        logging.exception(f"{MODEL_NAME} failed: {exc}. Returning original data.")
        return df_orig