import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
MODEL_NAME = "MICE-RF"

def impute_mice(data, target_column, input_columns, max_iter=10, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None, **kwargs):
    df_orig = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    try:
        # Prefer central impute_with_method if available
        try:
            from impute import impute_with_method
            return impute_with_method(df_orig, target_column, input_columns,
                                       method='missforest',
                                       custom_strategies=custom_strategies,
                                       spatial_config=spatial_config,
                                       max_iter=max_iter,
                                       random_state=random_state,
                                       **kwargs)
        except Exception:
            pass

        # Fallback to IterativeImputer using RandomForest estimator
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor

        df = df_orig.copy()
        cols = [c for c in ([target_column] + input_columns) if c in df.columns]
        if not cols:
            return df_orig

        for c in cols:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                pass

        estimator = RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=random_state, tol=tol)
        arr = imputer.fit_transform(df[cols])
        imputed_df = pd.DataFrame(arr, columns=cols, index=df.index)
        for c in cols:
            df[c] = imputed_df[c]
        return df

    except Exception as e:
        logging.exception(f"{MODEL_NAME} failed: {e}. Returning original data.")
        return df_orig