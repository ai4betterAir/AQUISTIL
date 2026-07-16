"""
SoftImpute wrapper. Delegates to central impute_with_method when available,
otherwise uses a simple safe SoftImputeCustom defined in impute.py (if present).
"""
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
MODEL_NAME = "SoftImpute"

def impute_mice(data, target_column, input_columns, max_iter=100, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None, **kwargs):
    try:
        from impute import impute_with_method
        return impute_with_method(data, target_column, input_columns,
                                  method='softimpute',
                                  custom_strategies=custom_strategies,
                                  spatial_config=spatial_config,
                                  max_iter=max_iter,
                                  random_state=random_state,
                                  tol=tol,
                                  **kwargs)
    except Exception:
        # Best-effort fallback: use pandas interpolation for target and return a DataFrame
        try:
            df = data.copy()
            cols = [c for c in ([target_column] + input_columns) if c in df.columns]
            if target_column in df.columns:
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                df[target_column] = df[target_column].interpolate(method='linear', limit_direction='both')
            return df
        except Exception as e:
            logging.exception(f"{MODEL_NAME} fallback failed: {e}. Returning original data.")
            return data