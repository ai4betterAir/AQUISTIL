"""
MICE_LGBM.py
Compatibility wrapper that uses LightGBM as the internal estimator for IterativeImputer.
Provides robust returns and spatial-temporal preparation like LightGBM.py.
"""
import logging
import os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "MICE-LGBM"

def impute_mice(data, target_column, input_columns, max_iter=10, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None, **kwargs):
    df_orig = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    try:
        df = df_orig.copy()

        # Build spatial_config from kwargs/config if not supplied
        if spatial_config is None:
            try:
                import config_spatial as cfg
                # canonicalize site token
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
                }
            except Exception:
                spatial_config = None

        # Prefer to reuse central impute_with_method if available
        try:
            from impute import impute_with_method
            return impute_with_method(df, target_column, input_columns,
                                      method='mice',
                                      custom_strategies=custom_strategies,
                                      spatial_config=spatial_config,
                                      max_iter=max_iter,
                                      random_state=random_state,
                                      tol=tol,
                                      **kwargs)
        except Exception:
            logging.debug(f"{MODEL_NAME}: central imputer not available, falling back to local IterativeImputer with LGB estimator")

        # Fallback local implementation
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        try:
            import lightgbm as lgb
            estimator = lgb.LGBMRegressor(n_estimators=100, random_state=random_state)
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=50, random_state=random_state)

        cols = [c for c in ([target_column] + input_columns) if c in df.columns]
        if not cols:
            return df_orig

        for c in cols:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                pass

        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=random_state, tol=tol)
        arr = imputer.fit_transform(df[cols])
        imputed_df = pd.DataFrame(arr, columns=cols, index=df.index)
        final = df.copy()
        for c in cols:
            final[c] = imputed_df[c]
        return final

    except Exception as e:
        logging.exception(f"{MODEL_NAME} failed: {e}. Returning original DataFrame.")
        return df_orig