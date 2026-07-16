"""
Sort data and apply imputation with simulated missingness
Enhanced with input feature logging and tracking
Robustness improvements: always save per-missingness metrics (even if NaN),
handle impute_function that returns None or missing target column,
and produce a manifest JSON per model/regime to confirm saved runs.

Author: Dr.  Masrur (modified)
Last Updated: 2026-01-28 (assistant)
"""

import logging
import os
import pandas as pd
import numpy as np
from evaluation_metrics import evaluate_metrics, evaluate_metrics_by_gap, METRIC_FUNCTIONS, save_target_metrics_csv
from impute_plot import (
    save_error_distribution,
    save_residual_plot,
    save_qq_plot,
    save_correlation_heatmap,
    save_statistical_summary,
    save_scatterplot,
    save_cdf_plot,
    save_histogram,
)
from missingness_regimes import apply_missingness
import json
from typing import Optional
try:
    import config_spatial as config
    CENTRAL_OUTPUT_ROOT = getattr(config, 'OUTPUT_DIRECTORY')
except Exception:
    CENTRAL_OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Imputation_Result_Spatial_Temporal_V8'))


def _append_df_to_csv(df, fp):
    """Append DataFrame `df` to CSV at `fp`, writing header if file missing."""
    try:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        write_header = not os.path.exists(fp)
        df.to_csv(fp, mode='a', header=write_header, index=False)
        return True
    except Exception as e:
        logging.warning("Failed to append to central CSV %s: %s", fp, e)
        return False

# Insert near top (after other imports)
def _safe_write_csv(df, path, context_descr="csv"):
    """
    Try to write DataFrame to CSV and return (success:bool, exc:Exception|None).
    Ensures parent directory exists and logs failures with stack trace.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info("Saved %s to: %s", context_descr, path)
        return True, None
    except Exception as e:
        logging.exception("Failed to save %s to %s: %s", context_descr, path, e)
        return False, e


def append_metrics_to_master(save_df):
    fp = os.path.join(CENTRAL_OUTPUT_ROOT, 'metrics_master.csv')
    return _append_df_to_csv(save_df, fp)


def append_imputed_to_master(minimal_df, metadata: dict):
    """Append long-format imputed rows to central `imputed_master.csv`.

    `minimal_df` is expected to have columns: DateTime, Actual_{target}, Imputed_{target}
    `metadata` should contain: Site, Model, Regime, Missingness_Level, Target, Source_File
    """
    rows = []
    # infer target name from actual column
    actual_col = next((c for c in minimal_df.columns if str(c).startswith('Actual_')), None)
    imputed_col = next((c for c in minimal_df.columns if str(c).startswith('Imputed_')), None)
    if actual_col is None or imputed_col is None:
        logging.warning("Cannot append imputed rows: minimal_df missing Actual_/Imputed_ columns")
        return False

    # Build rows including metadata for each imputed timestamp
    for idx, r in minimal_df.iterrows():
        rows.append({
            'DateTime': r['DateTime'],
            'Site': metadata.get('Site'),
            'Model': metadata.get('Model'),
            'Missingness_Regime': metadata.get('Regime') or metadata.get('Missingness_Regime'),
            'Missingness_Level': metadata.get('Missingness_Level'),
            'Target': metadata.get('Target'),
            'Actual': r[actual_col],
            'Imputed': r[imputed_col],
            'Comments': r['Comments'] if 'Comments' in minimal_df.columns else None,
        })

    big = pd.DataFrame(rows)

    # By default do NOT create or save a centralized per-sample Imputed_Results
    # folder. This avoids extra disk usage and prevents duplicate copies.
    # To enable central per-sample saving set `SAVE_CENTRAL_IMPUTED = True` in config_spatial.py
    try:
        save_central = getattr(config, 'SAVE_CENTRAL_IMPUTED', False)
    except Exception:
        save_central = False

    if not save_central:
        # Intentionally skip writing central per-sample imputed master
        return False

    # Ensure Imputed_Results directory exists under central root
    imputed_results_dir = os.path.join(CENTRAL_OUTPUT_ROOT, 'Imputed_Results')
    try:
        os.makedirs(imputed_results_dir, exist_ok=True)
    except Exception as e:
        logging.warning("Failed to create Imputed_Results dir %s: %s", imputed_results_dir, e)

    # Append to a single per-site_target master so one CSV contains all models/regimes/levels
    site_val = metadata.get('Site', 'unknown')
    target_val = metadata.get('Target', 'unknown')
    safe_site = str(site_val).replace(' ', '_')
    safe_target = str(target_val).replace(' ', '_')
    fp = os.path.join(imputed_results_dir, f"{safe_site}_{safe_target}_imputed.csv")
    ok = _append_df_to_csv(big, fp)

    return ok

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _safe_impute_call(impute_function, *args, **kwargs) -> Optional[pd.DataFrame]:
    """
    Call impute_function and defensively handle exceptions or non-DataFrame returns.
    Returns a DataFrame (or None on hard failure).

    Optional kwargs:
      - expected_columns: list of expected column names
      - expected_index: expected index (pd.Index)
    """
    expected_columns = kwargs.pop("expected_columns", None)
    expected_index = kwargs.pop("expected_index", None)

    try:
        # Locally suppress IterativeImputer ConvergenceWarning which is non-fatal
        try:
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.impute")
                res = impute_function(*args, **kwargs)
        except Exception:
            # If sklearn not available or import fails, call without warning filter
            res = impute_function(*args, **kwargs)
    except Exception as e:
        logging.exception("Exception when calling impute_function: %s", e)
        return None

    if res is None:
        logging.warning("Imputer returned None")
        return None

    # DataFrame -> return copy
    if isinstance(res, pd.DataFrame):
        df = res.copy()
        # Ensure expected columns exist
        if expected_columns is not None:
            for c in expected_columns:
                if c not in df.columns:
                    try:
                        # fill from original input if possible
                        orig = args[0] if len(args) >= 1 and isinstance(args[0], pd.DataFrame) else None
                        if orig is not None and c in orig.columns and len(orig) == len(df):
                            df[c] = orig[c].values
                        else:
                            df[c] = np.nan
                    except Exception:
                        df[c] = np.nan
            try:
                df = df.reindex(columns=expected_columns)
            except Exception:
                pass
        if expected_index is not None and len(df) == len(expected_index):
            df.index = expected_index
        return df

    # Series -> to DataFrame
    if isinstance(res, pd.Series):
        df = res.to_frame().T if res.ndim == 1 else res.to_frame()
        if expected_index is not None and len(df) == len(expected_index):
            df.index = expected_index
        if expected_columns is not None:
            for c in expected_columns:
                if c not in df.columns:
                    df[c] = np.nan
            try:
                df = df.reindex(columns=expected_columns)
            except Exception:
                pass
        return df

    # Numpy arrays / list-like
    try:
        arr = np.asarray(res)
        if arr.ndim == 2:
            nrows, ncols = arr.shape
            idx = expected_index if expected_index is not None else (args[0].index if len(args) >= 1 and isinstance(args[0], pd.DataFrame) else None)
            if expected_columns is not None and ncols == len(expected_columns):
                return pd.DataFrame(arr, columns=expected_columns, index=idx)
            if expected_columns is not None and ncols <= len(expected_columns):
                # map returned columns to first ncols of expected_columns and fill rest
                df = pd.DataFrame(arr, columns=expected_columns[:ncols], index=idx)
                for c in expected_columns[ncols:]:
                    try:
                        orig = args[0] if len(args) >= 1 and isinstance(args[0], pd.DataFrame) else None
                        if orig is not None and c in orig.columns and len(orig) == nrows:
                            df[c] = orig[c].values
                        else:
                            df[c] = np.nan
                    except Exception:
                        df[c] = np.nan
                df = df.reindex(columns=expected_columns)
                logging.warning("Imputer returned %d columns but expected %d; padded remaining columns with original/NaN.", ncols, len(expected_columns))
                return df
            # fallback: return DataFrame with integer column names
            return pd.DataFrame(arr, index=idx)
        if arr.ndim == 1:
            # single column -> map to target if available
            idx = expected_index if expected_index is not None else (args[0].index if len(args) >= 1 and isinstance(args[0], pd.DataFrame) else None)
            col_name = expected_columns[0] if expected_columns and len(expected_columns) == 1 else None
            if col_name:
                df = pd.DataFrame(arr, columns=[col_name], index=idx)
                for c in expected_columns:
                    if c not in df.columns:
                        df[c] = np.nan
                return df.reindex(columns=expected_columns)
            df = pd.DataFrame(arr, index=idx)
            return df
    except Exception as e:
        logging.warning("Failed to coerce imputer result to ndarray DataFrame: %s", e)

    # dict-like fallback
    try:
        df = pd.DataFrame(res)
        if expected_columns is not None:
            for c in expected_columns:
                if c not in df.columns:
                    df[c] = np.nan
            df = df.reindex(columns=expected_columns)
        if expected_index is not None and len(df) == len(expected_index):
            df.index = expected_index
        return df
    except Exception:
        pass

    logging.warning("Could not convert imputer output to DataFrame; returning None.")
    return None


def sort_and_impute_by_hour(
    input_file, 
    plot_save_path, 
    imputed_data_path, 
    target_column_data_path, 
    metrics_save_path, 
    target_column, 
    input_columns, 
    missingness_levels, 
    handle_negatives='exclude', 
    impute_function=None, 
    model_name=None, 
    custom_strategies=None, 
    sort_by_hour=False,
    missingness_regime='random'  # ✅ NEW PARAMETER
):
    """
    Optionally sort data by hour, apply imputation with simulated missingness, 
    and reorganize the data back to its original order.

    Robustness notes:
    - Guarantees a per-missingness metrics CSV is written (pattern: *_all_metrics.csv)
      so aggregation / research_plots will include the model even if metrics are NaN.
    - Writes a manifest JSON file per model/regime that lists saved files for verification.
    """
    # Load the data
    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        logging.error(f"Error: {input_file} not found.")
        return

    # Ensure the DateTime column is present
    if 'DateTime' not in data.columns:
        logging.error("DateTime column is missing in the input data.")
        return

    # Convert DateTime column to datetime type
    data['DateTime'] = pd.to_datetime(data['DateTime'])

    # Coerce input feature columns and target to numeric dtypes to avoid object dtype
    try:
        cols_to_coerce = [c for c in input_columns if c in data.columns and c != target_column]
        if cols_to_coerce:
            prev_na = data[cols_to_coerce].isna().sum()
            data[cols_to_coerce] = data[cols_to_coerce].apply(pd.to_numeric, errors='coerce')
            new_na = data[cols_to_coerce].isna().sum() - prev_na
            bad = new_na[new_na > 0]
            if not bad.empty:
                logging.warning("Coerced non-numeric values to NaN in input columns: %s", bad[bad>0].to_dict())

        prev_na_t = data[target_column].isna().sum()
        data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
        new_na_t = data[target_column].isna().sum() - prev_na_t
        if new_na_t > 0:
            logging.warning("Coerced non-numeric values to NaN in target column '%s': %d", target_column, new_na_t)
    except Exception:
        logging.debug("Failed to coerce numeric columns in input file %s", input_file, exc_info=True)

    # Check if target column exists
    if target_column not in data.columns:
        logging.error(f"Target column '{target_column}' not found in data.")
        return

    # Extract site basename (full file stem) and canonical model name
    base_name = os.path.basename(input_file).split('.')[0]
    # Use the full basename for file naming so downstream verification (which
    # expects the original CSV basename) can locate outputs reliably.
    site_name = base_name
    # Short site token for compact labels (first token before underscore)
    site_short = site_name.split('_')[0] if isinstance(site_name, str) and '_' in site_name else site_name
    # model_name may be passed as '<MODEL>_<SITE>'; canonical_model removes the final '_<SITE>' part
    canonical_model = model_name.rsplit('_', 1)[0] if model_name else "Unknown"
    regime = missingness_regime  # ✅ USE PARAMETER INSTEAD OF CONFIG

    # Create directory for input feature information
    input_features_path = os.path.join(os.path.dirname(imputed_data_path), "input_features")
    os.makedirs(input_features_path, exist_ok=True)
    os.makedirs(plot_save_path, exist_ok=True)
    os.makedirs(imputed_data_path, exist_ok=True)
    os.makedirs(target_column_data_path, exist_ok=True)
    os.makedirs(metrics_save_path, exist_ok=True)

    # Store all metrics for all missingness levels
    all_metrics = []
    all_feature_info = []

    # Manifest entries to confirm saved items
    manifest_entries = []

    # Iterate through each missingness level
    for missingness in missingness_levels:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing:  {site_name} | {target_column} | Regime: {regime} | Level: {int(missingness * 100)}%")
        logging.info(f"{'='*60}")

        # Create a copy of the data for this missingness level
        data_copy = data.copy()

        # Track original missing values (never score on these)
        original_missing_mask = data_copy[target_column].isna()

        # Calculate current missingness
        current_missingness = original_missing_mask.mean()
        logging.info(f"Original missingness: {current_missingness:.2%}")

        # ==========================================================
        # APPLY STRUCTURED MISSINGNESS (NO LEAKAGE)
        # ==========================================================

        logging.info(f"Applying {regime} missingness at {int(missingness*100)}% level...")
        
        data_copy, simulated_mask = apply_missingness(
            data_copy,
            target_column,
            regime=regime,  # ✅ USING PARAMETER
            frac=missingness,
            seed=42
        )

        # Ensure we NEVER overwrite original missing values
        simulated_mask = simulated_mask & (~original_missing_mask)

        # ✅ CRITICAL DEBUG: Report what was actually masked
        logging.info(
            f"✅ Missingness applied: regime={regime} | "
            f"simulated={int(simulated_mask.sum())} ({simulated_mask.mean():.2%}) | "
            f"original={int(original_missing_mask. sum())} ({original_missing_mask.mean():.2%})"
        )
        
        # ✅ Additional validation for event regime
        if regime == 'event':
            if simulated_mask.sum() == 0:
                logging.warning("Event regime produced no masked values; falling back to random masking")
                # Fallback: apply random missingness to ensure we have simulated values
                data_copy, simulated_mask = apply_missingness(
                    data_copy,
                    target_column,
                    regime='random',
                    frac=missingness,
                    seed=42
                )
                # Ensure we NEVER overwrite original missing values
                simulated_mask = simulated_mask & (~original_missing_mask)

                if simulated_mask.sum() == 0:
                    logging.error("EVENT REGIME & fallback both failed: No values masked!")
                    logging.error(
                        f"   Target stats: min={data[target_column].min():.2f}, "
                        f"max={data[target_column].max():.2f}, "
                        f"90th percentile={data[target_column].quantile(0.9):.2f}"
                    )
                    # Still proceed: we'll save outputs with NaNs so model isn't missing from aggregator
                else:
                    masked_values = data.loc[simulated_mask, target_column]
                    logging.info(
                        f"   Fallback masking successful: masked values range "
                        f"[{masked_values.min():.2f}, {masked_values.max():.2f}]"
                    )
            else:
                masked_values = data.loc[simulated_mask, target_column]
                logging.info(
                    f"   Event masking successful: masked values range "
                    f"[{masked_values.min():.2f}, {masked_values.max():.2f}]"
                )

        # Optionally sort data by hour
        if sort_by_hour:
            logging.info("Sorting data by hour for imputation.")
            data_copy['Hour'] = data_copy['DateTime']. dt.hour
            sorted_data = data_copy.sort_values(by='Hour')
            
            # Apply imputation for each hour
            imputed_data_list = []
            for hour in sorted_data['Hour'].unique():
                hour_data = sorted_data[sorted_data['Hour'] == hour]
                # Defensive check: ensure there is at least one non-empty input feature column
                available_feature_rows = 0
                try:
                    feat_df = hour_data[input_columns] if input_columns else hour_data.drop(columns=[target_column], errors='ignore')
                    # count rows that are not all-NaN across the feature set
                    available_feature_rows = feat_df.dropna(how='all').shape[0]
                except Exception:
                    available_feature_rows = 0

                if available_feature_rows == 0:
                    logging.warning("Hour %s has no valid feature rows for imputation (all features missing or non-numeric); skipping imputer for this hour", hour)
                    hour_data_imputed = hour_data.copy()
                    hour_data_imputed[target_column] = hour_data[target_column]
                    imputed_hour_data = hour_data_imputed
                else:
                    imputed_hour_data = _safe_impute_call(
                    impute_function,
                    hour_data,
                    target_column,
                    input_columns,
                    custom_strategies=custom_strategies,
                    out_dir=os.path.dirname(imputed_data_path),
                    site_name=site_name,
                    model_name=canonical_model,
                    missingness_regime=regime,
                    expected_columns=hour_data.columns.tolist(),
                    expected_index=hour_data.index
                )
                # If imputer failed for this hour, fall back to original hour_data
                if imputed_hour_data is None or target_column not in imputed_hour_data.columns:
                    logging.warning("Imputer failed or missing target for hour %s, falling back to original values for that hour", hour)
                    hour_data_imputed = hour_data.copy()
                    hour_data_imputed[target_column] = hour_data[target_column]
                    imputed_hour_data = hour_data_imputed
                imputed_data_list.append(imputed_hour_data)

            # Reorganize the data back to its original order
            try:
                imputed_data = pd.concat(imputed_data_list).sort_index()
            except Exception as e:
                logging.error("Failed to concat per-hour imputed data: %s", e)
                imputed_data = data_copy.copy()
                imputed_data[target_column] = data_copy[target_column]
        else:
            logging.info("Performing imputation on entire dataset.")
            # Apply imputation on the entire dataset without sorting
            # Defensive check: ensure input feature matrix has some usable rows
            try:
                feat_df_full = data_copy[input_columns] if input_columns else data_copy.drop(columns=[target_column], errors='ignore')
                usable_rows = feat_df_full.dropna(how='all').shape[0]
            except Exception:
                usable_rows = 0

            if usable_rows == 0:
                logging.warning("No usable input feature rows for full-dataset imputation for %s; skipping imputer and falling back to original values.", site_name)
                imputed_data = data_copy.copy()
                imputed_data[target_column] = data_copy[target_column]
                imputed_ok = False
            else:
                imputed_data = _safe_impute_call(
                    impute_function,
                    data_copy,
                    target_column,
                    input_columns,
                    custom_strategies=custom_strategies,
                    out_dir=os.path.dirname(imputed_data_path),
                    site_name=site_name,
                    model_name=canonical_model,
                    missingness_regime=regime,
                    expected_columns=data_copy.columns.tolist(),
                    expected_index=data_copy.index
                )

            # If imputer failed or did not return the target, fall back to original (but continue)
            if imputed_data is None:
                logging.warning("Imputer returned no output for full-dataset run; falling back to original column values")
                imputed_data = data_copy.copy()
                imputed_data[target_column] = data_copy[target_column]
                imputed_ok = False
            else:
                if target_column not in imputed_data.columns:
                    logging.warning("Imputer output missing target column '%s'; falling back to original target values", target_column)
                    # Try to preserve returned columns but ensure target exists
                    imputed_data = imputed_data.copy()
                    imputed_data[target_column] = data_copy[target_column]
                    imputed_ok = False
                else:
                    imputed_ok = True

        # --- ADDITION A: after building `imputed_data` (both code paths) ---
        # Ensure imputed_data has same index as original `data`. This avoids misalignment
        # when imputers return DataFrames with different indices or orders.
        try:
            if isinstance(imputed_data, pd.DataFrame):
                if not imputed_data.index.equals(data.index):
                    logging.info("Reindexing imputed_data to match original data index")
                    try:
                        imputed_data = imputed_data.reindex(data.index)
                    except Exception:
                        imputed_data = imputed_data.reset_index(drop=True).reindex(index=data.index)
        except Exception as e:
            logging.warning("Failed to robustly reindex imputed_data: %s. Proceeding, but alignment may be wrong.", e)

        # ========================================================================
        # SAVE INPUT FEATURE INFORMATION (existing code - keep as is)
        # ========================================================================
        
        actual_columns_used = [col for col in imputed_data.columns if col in input_columns or col.startswith('spatial_') or col in ['Hour', 'Day', 'Month', 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 'DayOfWeek_sin', 'DayOfWeek_cos']]
        actual_columns_used = [col for col in actual_columns_used if col != target_column]
        
        base_features = [col for col in actual_columns_used if col in input_columns]
        spatial_features = [col for col in actual_columns_used if col.startswith('spatial_')]
        temporal_features = [col for col in actual_columns_used if col in ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos']]
        lagged_features = [col for col in actual_columns_used if 'lag_' in col]
        rolling_features = [col for col in actual_columns_used if 'rolling_' in col]
        
        feature_info = {
            'Site': site_short,
            'Model': canonical_model,
            'Target_Column': target_column,
            'Missingness_Regime': regime,  # ✅ ADD REGIME TO TRACKING
            'Missingness_Level': f"{int(missingness * 100)}%",
            'Total_Features': len(actual_columns_used),
            'Base_Features_Count': len(base_features),
            'Spatial_Features_Count': len(spatial_features),
            'Temporal_Features_Count': len(temporal_features),
            'Lagged_Features_Count': len(lagged_features),
            'Rolling_Features_Count': len(rolling_features),
            'Total_Rows': len(imputed_data),
            'Original_Missing_Count': int(original_missing_mask.sum()),
            'Simulated_Missing_Count': int(simulated_mask.sum()),
            'Total_Missing_Count': int((original_missing_mask | simulated_mask).sum()),
        }
        
        all_feature_info.append(feature_info)
        
        # (Keep rest of feature saving code as is...)
        
        # ========================================================================
        # SAVE IMPUTED DATA AND TARGET COLUMN DATA
        # ========================================================================
        
        final_imputed_data = data.copy()
        # Ensure that imputed_data has the same index length; if not, fall back
        if len(imputed_data) != len(data):
            logging.warning("Imputed data length (%d) != original data length (%d); using original values for target", len(imputed_data), len(data))
            final_imputed_data[target_column] = data[target_column]
        else:
            final_imputed_data[target_column] = imputed_data[target_column]

        final_imputed_csv_filename = f"{imputed_data_path}/{site_name}_{target_column}_{canonical_model}_{regime}_imputed_{int(missingness * 100)}.csv"
        try:
            # Save a minimal imputed CSV: DateTime, Actual_{target}, Imputed_{target}
            actual_col = f"Actual_{target_column}"
            imputed_col = f"Imputed_{target_column}"
            try:
                imputed_series = final_imputed_data[target_column] if (isinstance(final_imputed_data, pd.DataFrame) and target_column in final_imputed_data.columns) else data[target_column]
            except Exception:
                imputed_series = data[target_column]

            # Build Comments column: 'Imputed' for rows where we populated imputed values
            try:
                imputed_mask = (simulated_mask | original_missing_mask)
            except Exception:
                imputed_mask = pd.Series([False] * len(data))

            comments = ['Imputed' if imputed_mask.iat[i] else 'Original' for i in range(len(data))]

            minimal_df = pd.DataFrame({
                'DateTime': data['DateTime'].values,
                actual_col: data[target_column].values,
                imputed_col: imputed_series.values,
                'Comments': comments
            })

            # Do NOT save a per-site imputed CSV; append minimal rows to central master.
            # per-site_target file now includes model name so each model keeps its own master
            per_site_target_fp = os.path.join(
                CENTRAL_OUTPUT_ROOT,
                'Imputed_Results',
                f"{site_short}_{target_column}_{canonical_model}_imputed.csv"
            )
            metadata = {
                'Site': site_short,
                'Model': canonical_model,
                'Regime': regime,
                'Missingness_Level': int(missingness * 100),
                'Target': target_column,
            }
            append_ok = append_imputed_to_master(minimal_df, metadata)
            if append_ok:
                logging.info(f"Appended {len(minimal_df)} rows to central imputed master for {site_short} {target_column}")
            else:
                logging.warning("Failed to append imputed rows to central master for %s %s", site_short, target_column)

            try:
                os.makedirs(os.path.dirname(final_imputed_csv_filename), exist_ok=True)
                final_imputed_data.to_csv(final_imputed_csv_filename, index=False)
                logging.info(f"Saved imputed data to: {final_imputed_csv_filename}")
            except Exception as e:
                logging.error(f"Failed to save imputed data to {final_imputed_csv_filename}: {e}", exc_info=True)
                # continue — still save metrics/plots if available, but mark this run as incomplete

        except Exception as e:
            logging.error(f"Failed to save imputed data to {final_imputed_csv_filename}: {e}")

        target_column_data = data[['DateTime', target_column]].copy()
        # Initialize imputed column with NaN then fill positions (simulate only simulated+original)
        target_column_data[f"{target_column}_imputed"] = np.nan
        # If imputed_data has correct length, populate imputed col at simulated or original positions
        if len(imputed_data) == len(data):
            # Only populate imputed column at simulated_mask and original_missing_mask to reflect imputation
            target_column_data.loc[simulated_mask | original_missing_mask, f"{target_column}_imputed"] = imputed_data.loc[simulated_mask | original_missing_mask, target_column]
        else:
            # fall back: copy imputed column if available and same length else leave NaNs
            try:
                target_column_data[f"{target_column}_imputed"] = imputed_data[target_column]
            except Exception:
                pass

        target_column_data["Missing_Type"] = "None"
        target_column_data.loc[original_missing_mask, "Missing_Type"] = "Original"
        target_column_data.loc[simulated_mask, "Missing_Type"] = "Simulated"

        target_column_csv_filename = f"{target_column_data_path}/{site_name}_{target_column}_{canonical_model}_{regime}_target_column_{int(missingness * 100)}.csv"
        try:
            target_column_data.to_csv(target_column_csv_filename, index=False)
            logging.info(f"Saved target column data to: {target_column_csv_filename}")
        except Exception as e:
            logging.error(f"Failed to save target column data to {target_column_csv_filename}: {e}")

        # Extract true and imputed values for SIMULATED missing values only
        metrics_row = None
        if simulated_mask.sum() > 0:
            # Coerce extracted true/imputed arrays to numeric floats to avoid type errors
            true_values_simulated = pd.to_numeric(data.loc[simulated_mask, target_column], errors='coerce').to_numpy(dtype=float)
            if (len(imputed_data) == len(data)):
                imputed_values_simulated = pd.to_numeric(imputed_data.loc[simulated_mask, target_column], errors='coerce').to_numpy(dtype=float)
            else:
                imputed_values_simulated = np.array([])

            # Filter out negative values if handle_negatives is set to 'exclude'
            if handle_negatives == 'exclude': 
                valid_mask = (true_values_simulated >= 0) & (imputed_values_simulated >= 0)
                true_values_clean = true_values_simulated[valid_mask]
                imputed_values_clean = imputed_values_simulated[valid_mask]
            else:
                true_values_clean = true_values_simulated
                imputed_values_clean = imputed_values_simulated

            # Evaluate metrics for simulated missing values (with optional negative exclusion)
            if len(true_values_clean) > 0:
                metrics = evaluate_metrics(true_values_clean, imputed_values_clean, handle_negative=handle_negatives)

                # Add missingness level to the metrics dictionary
                metrics["Missingness"] = missingness

                # Add some standard metadata fields so rows are consistent
                metrics.setdefault("Missingness_Regime", regime)
                metrics.setdefault("Total_Features_Used", len(actual_columns_used))
                metrics.setdefault("Base_Features", len(base_features))
                metrics.setdefault("Spatial_Features", len(spatial_features))
                metrics.setdefault("Temporal_Features", len(temporal_features))
                # duplicate long/short keys if necessary
                if "Root Mean Squared Error (RMSE)" not in metrics and "RMSE" in metrics:
                    metrics["Root Mean Squared Error (RMSE)"] = metrics["RMSE"]
                if "RMSE" not in metrics and "Root Mean Squared Error (RMSE)" in metrics:
                    metrics["RMSE"] = metrics["Root Mean Squared Error (RMSE)"]

                # Build a canonical row dict and persist via helper for per-target aggregation
                try:
                    row_dict = {
                        **metrics,
                        "Missingness_Regime": regime,
                        "Missingness": missingness,
                        "Total_Features_Used": len(actual_columns_used),
                        "Base_Features": len(base_features),
                        "Spatial_Features": len(spatial_features),
                        "Temporal_Features": len(temporal_features),
                        "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                        "Model": canonical_model,
                        "StudySite": site_short,
                    }
                    try:
                        save_target_metrics_csv(
                            rows=row_dict,
                            output_dir=CENTRAL_OUTPUT_ROOT,
                            target_variable=target_column
                        )
                        logging.info("Saved target metrics via save_target_metrics_csv: %s %s %s%%", canonical_model, site_short, int(missingness*100))
                    except Exception as e:
                        logging.warning("save_target_metrics_csv failed: %s", e)
                except Exception as e:
                    logging.warning("Failed to build row_dict for save_target_metrics_csv: %s", e)

                all_metrics.append(metrics)
                metrics_row = metrics

                logging.info(f"\nMetrics for {int(missingness * 100)}% {regime}:")
                logging.info(f"  RMSE: {metrics.get('Root Mean Squared Error (RMSE)', np.nan):.4f}")
                logging.info(f"  R:  {metrics.get('Correlation Coefficient (R)', np.nan):.4f}")
                logging.info(f"  NSE: {metrics.get('Nash-Sutcliffe Efficiency (NSE)', np.nan):.4f}")

                # Stratified metrics
                try:
                    gap_metrics = evaluate_metrics_by_gap(
                        y_true=data[target_column],
                        y_pred=imputed_data[target_column],
                        simulated_mask=simulated_mask,
                        metric_functions=METRIC_FUNCTIONS
                    )
                    gap_metrics.to_csv(
                        os.path.join(
                            metrics_save_path,
                            f"{canonical_model}_{target_column}_{regime}_{int(missingness * 100)}_gapwise_metrics.csv"
                        ),
                        index=False
                    )
                except Exception as e:
                    logging.error(f"Error computing gapwise metrics: {e}")

                # Generate plots (best-effort)
                try:
                    # Save plots into central plots_by_type/ to avoid per-model/regime subfolders
                    central_plots_root = os.path.join(CENTRAL_OUTPUT_ROOT, 'plots_by_type')
                    os.makedirs(central_plots_root, exist_ok=True)
                    model_name_with_regime = f"{canonical_model}_{regime}"
                    save_scatterplot(
                        true_values_clean,
                        imputed_values_clean,
                        central_plots_root,
                        site_name,
                        target_column,
                        model_name_with_regime,
                        missingness,
                        rmse=metrics.get("Root Mean Squared Error (RMSE)"),
                        r=metrics.get("Correlation Coefficient (R)")
                    )
                    save_error_distribution(true_values_clean, imputed_values_clean, central_plots_root, site_name, target_column, model_name_with_regime, missingness)
                    save_residual_plot(true_values_clean, imputed_values_clean, central_plots_root, site_name, target_column, model_name_with_regime, missingness)
                    save_qq_plot(imputed_values_clean, central_plots_root, site_name, target_column, model_name_with_regime, missingness)
                    save_correlation_heatmap(imputed_data, central_plots_root, site_name, target_column, model_name_with_regime, missingness)
                    save_statistical_summary(data_copy, imputed_data, central_plots_root, site_name, target_column, model_name_with_regime, missingness)
                    save_cdf_plot(true_values_clean, imputed_values_clean, central_plots_root, site_name, target_column, model_name_with_regime, missingness)
                    save_histogram(data_copy, imputed_values_clean, central_plots_root, site_name, target_column, model_name_with_regime, missingness)
                except Exception as e: 
                    logging.error(f"Error generating plots: {e}")

            else:
                logging.warning("No valid simulated values after negative filtering; metrics will be NaN for this missingness level.")
        else:
            logging.warning("No simulated masked values produced for this missingness level; creating placeholder metrics row (NaNs).")

        # ---------------------------
        # Always write a per-missingness metrics CSV (so aggregator sees the model)
        # ---------------------------
        metrics_filename = os.path.join(
            metrics_save_path,
            f"{site_name}_{target_column}_{canonical_model}_{regime}_{int(missingness * 100)}_all_metrics.csv"
        )

        if metrics_row is None:
            # create placeholder metrics row with standardized keys expected by evaluate_metrics
            placeholder = {
                "Nash-Sutcliffe Efficiency (NSE)": np.nan,
                "Index of Agreement (WI)": np.nan,
                "Mean Bias Error (MBE)": np.nan,
                "Absolute Percent Bias (APB)": np.nan,
                "Kling-Gupta Efficiency (KGE)": np.nan,
                "Legate's and McCabe's Index (LM)": np.nan,
                "Root Mean Squared Error (RMSE)": np.nan,
                "Relative Root Mean Squared Error (RRMSE)": np.nan,
                "Relative Mean Absolute Error (RMAE)": np.nan,
                "Mean Absolute Error (MAE)": np.nan,
                "Mean Absolute Percentage Error (MAPE)": np.nan,
                "Correlation Coefficient (R)": np.nan,
                "Coefficient of Determination (R²)": np.nan,
                "Anomaly Correlation Coefficient (ACC)": np.nan,
                "Mean Normalized Root Mean Squared Error (NRMSE)": np.nan,
                "Missingness": missingness,
                "Missingness_Regime": regime,
            }
            save_df = pd.DataFrame([placeholder])
        else:
            # ensure Missingness & Missingness_Regime exist
            metrics_row.setdefault("Missingness", missingness)
            metrics_row.setdefault("Missingness_Regime", regime)
            save_df = pd.DataFrame([metrics_row])

        # Add some meta columns for aggregator/user convenience
        save_df["Model"] = canonical_model
        save_df["StudySite"] = site_short

        try:
            save_df.to_csv(metrics_filename, index=False)
            logging.info("Saved metrics CSV: %s", metrics_filename)
        except Exception as e:
            logging.error("Failed to save metrics CSV %s: %s", metrics_filename, e)

        # Also append metrics row to the central metrics master
        try:
            append_metrics_to_master(save_df)
        except Exception as e:
            logging.warning("Appending to central metrics master failed: %s", e)

        manifest_entries.append({
            "site": site_short,
            "target": target_column,
            "model": canonical_model,
            "regime": regime,
            "missingness_pct": int(missingness * 100),
            "imputed_file": final_imputed_csv_filename,
            "target_column_file": target_column_csv_filename,
            "metrics_file": metrics_filename,
            "simulated_masked_count": int(simulated_mask.sum()),
            "original_missing_count": int(original_missing_mask.sum())
        })

    # Save aggregated results (all missingness levels combined) as before
    if all_metrics:
        try:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_csv_filename_combined = f"{metrics_save_path}/{site_name}_{target_column}_{canonical_model}_{regime}_all_metrics.csv"
            metrics_df.to_csv(metrics_csv_filename_combined, index=False)
            logging.info(f"Saved combined all-level metrics to: {metrics_csv_filename_combined}")
        except Exception as e:
            logging.error("Failed saving combined metrics: %s", e)
    else:
        logging.warning("No metrics computed across missingness levels (all_metrics empty). Combined CSV will not contain informative rows; per-level placeholders were still saved.")

    # Save manifest JSON summarizing saved files for this model/regime
    manifest_fp = os.path.join(metrics_save_path, f"saved_runs_manifest_{canonical_model}_{regime}.json")
    try:
        with open(manifest_fp, "w") as fh:
            json.dump(manifest_entries, fh, indent=2)
        logging.info("Saved run manifest: %s", manifest_fp)
        # Also log a concise summary
        for e in manifest_entries:
            logging.info("SAVED -> model=%s site=%s regime=%s missing=%d%% imputed=%s metrics=%s",
                         e["model"], e["site"], e["regime"], e["missingness_pct"],
                         os.path.basename(e["imputed_file"]), os.path.basename(e["metrics_file"]))
    except Exception as e:
        logging.error("Failed to write manifest JSON: %s", e)