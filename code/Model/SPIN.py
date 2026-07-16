import logging
import os
import importlib.util
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "SPIN"


def _load_impute():
    """
    Try multiple import locations for the shared impute entrypoint (impute_with_method).
    """
    try:
        from impute import impute_with_method  # top-level
        return impute_with_method
    except Exception:
        pass

    try:
        from Model.impute import impute_with_method  # package-style
        return impute_with_method
    except Exception:
        pass

    # Last-resort: load by path relative to this file
    impute_path = os.path.join(os.path.dirname(__file__), 'impute.py')
    if os.path.exists(impute_path):
        spec = importlib.util.spec_from_file_location('impute_local', impute_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "impute_with_method", None)

    return None


impute_with_method = _load_impute()


def _mean_fallback(data, target_column, input_columns, **kwargs):
    """
    Safe fallback: try to call Mean.impute_mice if available, otherwise
    use sklearn SimpleImputer (mean) for the target column.
    Always returns a DataFrame aligned with input `data`.
    """
    try:
        # try to import the Mean model wrapper
        try:
            from Model.Mean import impute_mice as mean_impute  # package-style
        except Exception:
            try:
                from Mean import impute_mice as mean_impute  # top-level
            except Exception:
                mean_impute = None

        if mean_impute is not None:
            logging.warning(f"[{MODEL_NAME}] Using Mean fallback imputer (Mean.impute_mice).")
            return mean_impute(data.copy(), target_column, input_columns, **kwargs)

        # Last-resort: use sklearn SimpleImputer on the target column
        import pandas as pd
        from sklearn.impute import SimpleImputer

        df = data.copy()
        if target_column not in df.columns:
            logging.warning(f"[{MODEL_NAME}] target '{target_column}' not in dataframe; returning original")
            return df

        imputer = SimpleImputer(strategy='mean')
        vals = df[[target_column]].values
        # Fit on available values and transform (SimpleImputer handles all-NaN gracefully by leaving as NaN)
        try:
            imputed = imputer.fit_transform(vals)
            df[target_column] = imputed.ravel()
        except Exception as e:
            logging.warning(f"[{MODEL_NAME}] SimpleImputer failed: {e}; leaving target untouched")
        return df

    except Exception as e:
        logging.exception(f"[{MODEL_NAME}] mean fallback completely failed: {e}")
        # As a final safe return, ensure a DataFrame is returned
        try:
            return data.copy()
        except Exception:
            import pandas as pd
            return pd.DataFrame()


def impute_mice(data, target_column, input_columns, max_iter=100, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None, **kwargs) -> Any:
    """
    Robust SPIN wrapper compatible with the pipeline.

    Behavior:
      - Attempts to call impute_with_method(..., method='spin') if available.
      - If that call raises or returns None/invalid result, falls back to mean imputer.
      - Always returns a DataFrame aligned to the input.
    """
    df = data.copy()
    try:
        # Validate inputs early
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        if target_column not in df.columns:
            logging.warning(f"[{MODEL_NAME}] target '{target_column}' not in dataframe — returning original")
            return df

        # Use provided impute_with_method if available
        if impute_with_method is None:
            logging.warning(f"[{MODEL_NAME}] impute_with_method not found — using mean fallback")
            return _mean_fallback(df, target_column, input_columns, **kwargs)

        # Call the generic impute_with_method with method='spin'
        try:
            result = impute_with_method(
                df.copy(),
                target_column,
                input_columns,
                method='spin',
                max_iter=max_iter,
                random_state=random_state,
                custom_strategies=custom_strategies,
                spatial_config=spatial_config,
                **kwargs
            )
        except TypeError:
            # Old impute_with_method variants expect different signature; try without 'method' kw
            result = impute_with_method(
                df.copy(),
                target_column,
                input_columns,
                max_iter=max_iter,
                random_state=random_state,
                custom_strategies=custom_strategies,
                spatial_config=spatial_config,
                method='spin'  # try to pass anyway
            )
        except Exception as e:
            logging.exception(f"[{MODEL_NAME}] primary imputer raised exception: {e}")
            return _mean_fallback(df, target_column, input_columns, **kwargs)

        # Validate result
        if result is None:
            logging.warning(f"[{MODEL_NAME}] impute_with_method returned None — using mean fallback")
            return _mean_fallback(df, target_column, input_columns, **kwargs)

        # Ensure result is a DataFrame and has the target column
        try:
            import pandas as pd
            if not isinstance(result, pd.DataFrame):
                logging.warning(f"[{MODEL_NAME}] imputer returned non-DataFrame (type={type(result)}). Converting...")
                # try to construct a DataFrame where possible
                result = pd.DataFrame(result, index=df.index)
            if target_column not in result.columns:
                logging.warning(f"[{MODEL_NAME}] result missing target column '{target_column}' — using mean fallback")
                return _mean_fallback(df, target_column, input_columns, **kwargs)

            # Align index and columns back to original where possible
            final = df.copy()
            # Only replace columns that exist in both
            for c in result.columns:
                if c in final.columns:
                    final[c] = result[c].values
            return final

        except Exception as e:
            logging.exception(f"[{MODEL_NAME}] Failed to validate/align imputer result: {e}")
            return _mean_fallback(df, target_column, input_columns, **kwargs)

    except Exception as exc:
        logging.exception(f"[{MODEL_NAME}] impute_mice wrapper failed: {exc}")
        return _mean_fallback(df, target_column, input_columns, **kwargs)