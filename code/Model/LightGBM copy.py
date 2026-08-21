import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging
from spatial import prepare_spatial_temporal_data
from impute_utils import safe_assign_imputed

MODEL_NAME = "LightGBM"

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def impute_mice(data, target_column, input_columns, max_iter=1, random_state=42, 
                custom_strategies=None, **kwargs):
    """
    LightGBM-based MICE imputation with spatial-temporal features
    """
    
    logging.info("Starting LIGHTGBM imputation...")
    
    # ✅ EXTRACT SPATIAL CONFIG
    out_dir = kwargs.get('out_dir', '')
    site_name = kwargs.get('site_name', '')
    # Canonicalize provided site_name to first token
    if site_name:
        try:
            site_name = str(site_name).split('_')[0]
        except Exception:
            site_name = str(site_name)

    if not site_name and 'model_name' in kwargs:
        parts = kwargs['model_name'].split('_')
        if len(parts) > 0:
            site_name = parts[-1].split('_')[0]
    
    # Build spatial config
    import config_spatial as config
    
    spatial_config = {
        'input_directory': config.INPUT_DIRECTORY,
        'target_site': site_name,
        'use_spatial':   config.USE_SPATIAL_FEATURES,
        'use_temporal': config.USE_TEMPORAL_FEATURES,
        'use_lagged':  config.USE_LAGGED_FEATURES,
        'use_rolling': config.USE_ROLLING_FEATURES,
    }
    
    logging.info(f"  Site: {site_name}")
    logging.info(f"  Spatial features: {spatial_config['use_spatial']}")
    logging.info(f"  Temporal features: {spatial_config['use_temporal']}")
    
    # ✅ PREPARE SPATIAL-TEMPORAL DATA
    try:
        df_enhanced, feature_columns = prepare_spatial_temporal_data(
            data, 
            target_column, 
            input_columns, 
            spatial_config
        )
        logging.info(f"  Total features prepared: {len(feature_columns)}")
    except Exception as e:
        logging.warning(f"  Failed to load spatial features: {e}")
        logging.warning(f"  Falling back to local features only")
        df_enhanced = data. copy()
        feature_columns = input_columns
    
    # ✅ CRITICAL FIX: Reset index to ensure consistent indexing
    df = df_enhanced.copy().reset_index(drop=True)
    
    # Apply custom strategies if provided
    if custom_strategies:
        for column, strategy in custom_strategies.items():
            if column in df. columns:
                if isinstance(strategy, (int, float)):
                    df[column]. fillna(strategy, inplace=True)
                elif strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
    
    # Get all columns for imputation (features + target)
    columns_to_impute = feature_columns + [target_column]
    columns_to_impute = [col for col in columns_to_impute if col in df.columns]
    # Remove duplicates while preserving order
    columns_to_impute = list(dict.fromkeys(columns_to_impute))
    
    # Extract data for imputation
    data_to_impute = df[columns_to_impute]. copy()
    
    # ✅ ENSURE ALL DATA IS NUMERIC (defensive: drop any problematic columns)
    cols_for_imputation = list(data_to_impute.columns)
    try:
        for col in cols_for_imputation:
            # Defensive access: ensure column exists and is convertible to a Series-like object
            if col not in data_to_impute.columns:
                continue
            try:
                col_data = data_to_impute[col]
            except Exception as e:
                logging.warning(f"Could not access column '{col}': {e}; dropping from imputation set")
                if col in data_to_impute.columns:
                    data_to_impute.drop(columns=[col], inplace=True)
                try:
                    columns_to_impute.remove(col)
                except ValueError:
                    pass
                continue

            # If the object is not list-like/Series-like, try to coerce to a Series first
            if not isinstance(col_data, (pd.Series, np.ndarray, list, tuple)):
                try:
                    col_series = pd.Series(col_data)
                except Exception as e:
                    logging.warning(f"Column '{col}' is not list-like and could not be converted to Series: {e}; dropping")
                    if col in data_to_impute.columns:
                        data_to_impute.drop(columns=[col], inplace=True)
                    try:
                        columns_to_impute.remove(col)
                    except ValueError:
                        pass
                    continue
            else:
                col_series = pd.Series(col_data)

            # Now safely attempt numeric coercion
            try:
                coerced = pd.to_numeric(col_series, errors='coerce')
                data_to_impute[col] = coerced.values
            except Exception as e_inner:
                # Catch any exception from pd.to_numeric (TypeError or others), log details and drop column
                logging.warning(f"Failed to coerce column '{col}' to numeric: {e_inner}")
                try:
                    sample = None
                    try:
                        sample = getattr(col_series, 'head', lambda n: col_series[:n])(5).tolist()
                    except Exception:
                        sample = str(type(col_series))
                    logging.debug(f"  - column '{col}' type: {type(col_series)}, sample: {sample}")
                except Exception:
                    pass
                if col in data_to_impute.columns:
                    data_to_impute.drop(columns=[col], inplace=True)
                try:
                    columns_to_impute.remove(col)
                except ValueError:
                    pass
                continue
    except TypeError as te:
        logging.error(f"TypeError during per-column coercion loop: {te}")
        # Inspect columns to find offending types/values
        for col in cols_for_imputation:
            try:
                val = data_to_impute[col]
                sample = None
                try:
                    sample = getattr(val, 'head', lambda n: val[:n])(3).tolist()
                except Exception:
                    sample = str(type(val))
                logging.error(f"  - Column '{col}' type: {type(val)}, sample: {sample}")
            except Exception as e_inspect:
                logging.error(f"  - Failed to inspect column '{col}': {e_inspect}")

        # Drop any columns that cannot be accessed as Series-like and continue
        bad_cols = []
        for col in cols_for_imputation:
            try:
                val = data_to_impute[col]
                if not isinstance(val, (pd.Series, np.ndarray, list, tuple)):
                    bad_cols.append(col)
            except Exception:
                bad_cols.append(col)

        for col in bad_cols:
            logging.warning(f"Dropping problematic column '{col}' after TypeError during coercion")
            try:
                data_to_impute.drop(columns=[col], inplace=True)
            except Exception:
                pass
            try:
                columns_to_impute.remove(col)
            except Exception:
                pass

    # Recompute columns_to_impute to match data_to_impute
    columns_to_impute = [c for c in columns_to_impute if c in data_to_impute.columns]
    if len(columns_to_impute) == 0:
        logging.error("No valid numeric columns left for imputation after coercion; aborting imputation")
        raise ValueError("No valid numeric columns for imputation")
    
    # Create LightGBM estimator
    try:
        model_n_jobs = int(kwargs.get("n_jobs", getattr(config, "MODEL_N_JOBS", -1)))
    except (TypeError, ValueError):
        model_n_jobs = -1
    if model_n_jobs == 0:
        model_n_jobs = -1

    lgb_estimator = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=model_n_jobs,
        verbose=-1
    )
    
    # Create iterative imputer
    imputer = IterativeImputer(
        estimator=lgb_estimator,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0
    )
    
    # Perform imputation
    try:
        # Diagnostics: (commented out to reduce log verbosity)
        # logging.info(f"Imputer input shape: {data_to_impute.shape}")
        # logging.info(f"Imputer input columns: {list(data_to_impute.columns)}")
        # logging.info(f"Imputer input dtypes:\n{data_to_impute.dtypes}")

        # Suppress sklearn UserWarning about feature names during iterative fitting
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='X does not have valid feature names',
                category=UserWarning
            )
            imputed_array = imputer.fit_transform(data_to_impute)

        # Validate shape and assign safely
        try:
            if imputed_array.ndim != 2 or imputed_array.shape[1] != len(columns_to_impute):
                logging.error(f"Imputer returned array with shape {imputed_array.shape} but expected (*, {len(columns_to_impute)})")
                # Use safe assignment helper to attempt a best-effort assignment
                df = safe_assign_imputed(df, columns_to_impute, imputed_array)
            else:
                df = safe_assign_imputed(df, columns_to_impute, imputed_array)
        except Exception as e:
            logging.error(f"Failed to assign imputed values safely: {e}")
            raise
        
        # Log convergence info
        if hasattr(imputer, 'n_iter_'):
            logging.info(
                f"LightGBM iterative imputation passes: {imputer.n_iter_}/{max_iter}"
            )
        
        logging.info("LIGHTGBM imputation completed successfully.")
        
    except Exception as e:
        logging.error(f"Error during LightGBM imputation: {e}")
        raise
    
    # ✅ RESTORE ORIGINAL INDEX if DataFrame had DateTime as index
    if 'DateTime' in df_enhanced.columns and df_enhanced. index.name == 'DateTime': 
        df['DateTime'] = df_enhanced['DateTime']. values
        df. set_index('DateTime', inplace=True)
    
    return df
