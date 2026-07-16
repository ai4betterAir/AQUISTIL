"""
Sort data and apply imputation with simulated missingness
Enhanced with input feature logging and tracking
Robustness improvements: always save per-missingness metrics (even if NaN),
handle impute_function that returns None or missing target column,
and produce a manifest JSON per model/regime to confirm saved runs.

Author: Dr.  Masrur (modified)
Last Updated: 2026-03-03 (outputs centralized via main.py)
"""

import logging
import os
import pandas as pd
import numpy as np
from evaluation_metrics import (
    evaluate_metrics,
    evaluate_metrics_by_gap,
    METRIC_FUNCTIONS,
    save_target_metrics_csv,
)
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

# ---------------------------------------------------------------------------
# Helper: we no longer decide global roots here. main.py passes directories in.
# ---------------------------------------------------------------------------

def _append_df_to_csv(df, fp):
    """Append DataFrame `df` to CSV at `fp`, writing header if file missing."""
    try:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        write_header = not os.path.exists(fp)
        df.to_csv(fp, mode="a", header=write_header, index=False)
        return True
    except Exception as e:
        logging.warning("Failed to append to CSV %s: %s", fp, e)
        return False


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


def append_imputed_to_master(minimal_df, metadata: dict, central_imputed_root: Optional[str] = None):
    """
    Append minimal per-timestamp imputed rows to a per-target master CSV.

    If `central_imputed_root` is provided (recommended), files are written to:
        <central_imputed_root>/<TARGET>_<StudySite>_imputed.csv

    Otherwise, this becomes a no-op (returns False) to avoid guessing paths.
    """
    if not central_imputed_root:
        # main.py did not provide a central root; skip global master.
        return False

    import pandas as pd
    try:
        site = metadata.get("Site") or metadata.get("StudySite") or "unknown"
        model = metadata.get("Model") or "unknown"
        regime = metadata.get("Regime") or metadata.get("Missingness_Regime") or "unknown"

        # Missingness percent
        if "Missingness_Level" in metadata and metadata["Missingness_Level"] is not None:
            try:
                missingness_pct = int(metadata["Missingness_Level"])
            except Exception:
                try:
                    missingness_pct = int(float(metadata["Missingness_Level"]))
                except Exception:
                    missingness_pct = None
        else:
            m = metadata.get("Missingness")
            if m is None:
                missingness_pct = None
            else:
                try:
                    missingness_pct = int(float(m) * 100)
                except Exception:
                    missingness_pct = None

        # Detect actual & imputed columns
        actual_col = next((c for c in minimal_df.columns if str(c).startswith("Actual_")), None)
        imputed_col = next((c for c in minimal_df.columns if str(c).startswith("Imputed_")), None)

        target_from_meta = metadata.get("Target")
        if (actual_col is None or imputed_col is None) and target_from_meta:
            ac = f"Actual_{target_from_meta}"
            ic = f"Imputed_{target_from_meta}"
            if ac in minimal_df.columns:
                actual_col = ac
            if ic in minimal_df.columns:
                imputed_col = ic

        if actual_col is None or imputed_col is None:
            numeric_cols = [
                c for c in minimal_df.columns
                if pd.api.types.is_numeric_dtype(minimal_df[c])
            ]
            if len(numeric_cols) >= 2:
                if actual_col is None:
                    actual_col = numeric_cols[0]
                if imputed_col is None:
                    imputed_col = numeric_cols[1]
            else:
                if actual_col is None:
                    minimal_df["Actual_unknown"] = pd.NA
                    actual_col = "Actual_unknown"
                if imputed_col is None:
                    minimal_df["Imputed_unknown"] = pd.NA
                    imputed_col = "Imputed_unknown"

        # DateTime column
        dt_col = "DateTime" if "DateTime" in minimal_df.columns else None
        if dt_col is None:
            for c in minimal_df.columns:
                if "date" in str(c).lower():
                    dt_col = c
                    break
        if dt_col is None:
            minimal_df = minimal_df.copy()
            minimal_df["DateTime"] = pd.RangeIndex(start=0, stop=len(minimal_df))
            dt_col = "DateTime"

        # Comments column
        comments_col = "Comments" if "Comments" in minimal_df.columns else None
        if comments_col is None:
            minimal_df = minimal_df.copy()
            minimal_df["Comments"] = [
                "Imputed"
                if (pd.notna(v) and pd.isna(a)) or (pd.notna(v) and a != v)
                else "Original"
                for a, v in zip(
                    minimal_df.get(actual_col, pd.Series([pd.NA] * len(minimal_df))),
                    minimal_df.get(imputed_col, pd.Series([pd.NA] * len(minimal_df))),
                )
            ]
            comments_col = "Comments"

        out_df = pd.DataFrame(
            {
                "DateTime": minimal_df[dt_col].astype(str),
                "Missingness": (
                    missingness_pct if missingness_pct is not None else pd.NA
                ),
                "Missingness_Regime": regime,
                "Model": model,
                "StudySite": site,
                "Actual": minimal_df[actual_col]
                .astype(object)
                .where(minimal_df[actual_col].notna(), pd.NA),
                "Imputed": minimal_df[imputed_col]
                .astype(object)
                .where(minimal_df[imputed_col].notna(), pd.NA),
                "Comments": minimal_df[comments_col].astype(str).fillna(""),
            }
        )

        imputed_results_dir = central_imputed_root
        os.makedirs(imputed_results_dir, exist_ok=True)

        target_token = target_from_meta or (
            actual_col.replace("Actual_", "")
            if actual_col and actual_col.startswith("Actual_")
            else "unknown"
        )
        safe_target = str(target_token).replace(" ", "_")
        safe_site = str(site).replace(" ", "_")
        out_fp = os.path.join(
            imputed_results_dir, f"{safe_target}_{safe_site}_imputed.csv"
        )

        write_header = not os.path.exists(out_fp)
        out_df.to_csv(out_fp, index=False, mode="a", header=write_header)
        return True
    except Exception as e:
        logging.exception("append_imputed_to_master failed: %s", e)
        return False


def append_metrics_to_master(df, central_metrics_root: Optional[str] = None):
    """
    Append metrics DataFrame or dict to a centralized master CSV.

    If `central_metrics_root` is provided, file is:
        <central_metrics_root>/master_metrics.csv

    Otherwise, this becomes a no-op and returns False.
    """
    if not central_metrics_root:
        return False

    try:
        os.makedirs(central_metrics_root, exist_ok=True)
        master_fp = os.path.join(central_metrics_root, "master_metrics.csv")

        if isinstance(df, pd.DataFrame):
            write_header = not os.path.exists(master_fp)
            df.to_csv(master_fp, mode="a", header=write_header, index=False)
        else:
            temp = pd.DataFrame([df])
            write_header = not os.path.exists(master_fp)
            temp.to_csv(master_fp, mode="a", header=write_header, index=False)
        return True
    except Exception as e:
        logging.exception("append_metrics_to_master failed: %s", e)
        return False


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        try:
            import warnings
            from sklearn.exceptions import ConvergenceWarning

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=ConvergenceWarning,
                    module="sklearn.impute",
                )
                res = impute_function(*args, **kwargs)
        except Exception:
            res = impute_function(*args, **kwargs)
    except Exception as e:
        logging.exception("Exception when calling impute_function: %s", e)
        return None

    if res is None:
        logging.warning("Imputer returned None")
        return None

    if isinstance(res, pd.DataFrame):
        df = res.copy()
        if expected_columns is not None:
            for c in expected_columns:
                if c not in df.columns:
                    try:
                        orig = (
                            args[0]
                            if len(args) >= 1 and isinstance(args[0], pd.DataFrame)
                            else None
                        )
                        if (
                            orig is not None
                            and c in orig.columns
                            and len(orig) == len(df)
                        ):
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

    try:
        arr = np.asarray(res)
        if arr.ndim == 2:
            nrows, ncols = arr.shape
            idx = (
                expected_index
                if expected_index is not None
                else (
                    args[0].index
                    if len(args) >= 1 and isinstance(args[0], pd.DataFrame)
                    else None
                )
            )
            if expected_columns is not None and ncols == len(expected_columns):
                return pd.DataFrame(arr, columns=expected_columns, index=idx)
            if expected_columns is not None and ncols <= len(expected_columns):
                df = pd.DataFrame(arr, columns=expected_columns[:ncols], index=idx)
                for c in expected_columns[ncols:]:
                    try:
                        orig = (
                            args[0]
                            if len(args) >= 1
                            and isinstance(args[0], pd.DataFrame)
                            else None
                        )
                        if (
                            orig is not None
                            and c in orig.columns
                            and len(orig) == nrows
                        ):
                            df[c] = orig[c].values
                        else:
                            df[c] = np.nan
                    except Exception:
                        df[c] = np.nan
                df = df.reindex(columns=expected_columns)
                logging.warning(
                    "Imputer returned %d columns but expected %d; padded remaining columns with original/NaN.",
                    ncols,
                    len(expected_columns),
                )
                return df
            return pd.DataFrame(arr, index=idx)
        if arr.ndim == 1:
            idx = (
                expected_index
                if expected_index is not None
                else (
                    args[0].index
                    if len(args) >= 1 and isinstance(args[0], pd.DataFrame)
                    else None
                )
            )
            col_name = (
                expected_columns[0]
                if expected_columns and len(expected_columns) == 1
                else None
            )
            if col_name:
                df = pd.DataFrame(arr, columns=[col_name], index=idx)
                for c in expected_columns:
                    if c not in df.columns:
                        df[c] = np.nan
                return df.reindex(columns=expected_columns)
            return pd.DataFrame(arr, index=idx)
    except Exception as e:
        logging.warning("Failed to coerce imputer result to ndarray DataFrame: %s", e)

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
    input_file: str,
    imputed_data_path: str,
    target_column_data_path: str,
    metrics_save_path: str,
    target_column: str,
    input_columns,
    missingness_levels,
    handle_negatives: str = "exclude",
    impute_function=None,
    model_name: Optional[str] = None,
    custom_strategies=None,
    sort_by_hour: bool = False,
    missingness_regime: str = "random",
    # NEW: central roots passed from main.py (optional)
    central_imputed_root: Optional[str] = None,
    central_metrics_root: Optional[str] = None,
    central_plots_root: Optional[str] = None,
    results_root: Optional[str] = None,
):
    """
    Optionally sort data by hour, apply imputation with simulated missingness,
    and reorganize the data back to its original order.

    All concrete output locations are provided by main.py:
      - imputed_data_path: per-model per-regime imputed CSVs
      - target_column_data_path: per-model per-regime target-column CSVs
      - metrics_save_path: per-model per-regime metrics CSVs
      - central_imputed_root: central per-target master imputed (optional)
      - central_metrics_root: central master_metrics.csv (optional)
      - central_plots_root: central plots_by_type (optional)
    """
    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        logging.error(f"Error: {input_file} not found.")
        return

    if "DateTime" not in data.columns:
        logging.error("DateTime column is missing in the input data.")
        return

    data["DateTime"] = pd.to_datetime(data["DateTime"])

    try:
        cols_to_coerce = [
            c for c in input_columns if c in data.columns and c != target_column
        ]
        if cols_to_coerce:
            prev_na = data[cols_to_coerce].isna().sum()
            data[cols_to_coerce] = data[cols_to_coerce].apply(
                pd.to_numeric, errors="coerce"
            )
            new_na = data[cols_to_coerce].isna().sum() - prev_na
            bad = new_na[new_na > 0]
            if not bad.empty:
                logging.warning(
                    "Coerced non-numeric values to NaN in input columns: %s",
                    bad[bad > 0].to_dict(),
                )

        prev_na_t = data[target_column].isna().sum()
        data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
        new_na_t = data[target_column].isna().sum() - prev_na_t
        if new_na_t > 0:
            logging.warning(
                "Coerced non-numeric values to NaN in target column '%s': %d",
                target_column,
                new_na_t,
            )
    except Exception:
        logging.debug(
            "Failed to coerce numeric columns in input file %s",
            input_file,
            exc_info=True,
        )

    if target_column not in data.columns:
        logging.error(f"Target column '{target_column}' not found in data.")
        return

    base_name = os.path.basename(input_file).split(".")[0]
    site_name = base_name
    site_short = (
        site_name.split("_")[0]
        if isinstance(site_name, str) and "_" in site_name
        else site_name
    )
    canonical_model = model_name.rsplit("_", 1)[0] if model_name else "Unknown"
    regime = missingness_regime

    os.makedirs(imputed_data_path, exist_ok=True)
    os.makedirs(target_column_data_path, exist_ok=True)
    os.makedirs(metrics_save_path, exist_ok=True)

    all_metrics = []
    all_feature_info = []
    manifest_entries = []

    # local central plots root fallback
    if not central_plots_root:
        central_plots_root = os.path.join(metrics_save_path, "..", "plots_by_type")

    for missingness in missingness_levels:
        logging.info("\n" + "=" * 60)
        logging.info(
            f"Processing:  {site_name} | {target_column} | Regime: {regime} | "
            f"Level: {int(missingness * 100)}%"
        )
        logging.info("=" * 60)

        data_copy = data.copy()
        original_missing_mask = data_copy[target_column].isna()
        current_missingness = original_missing_mask.mean()
        logging.info(f"Original missingness: {current_missingness:.2%}")

        logging.info(
            f"Applying {regime} missingness at {int(missingness * 100)}% level..."
        )
        data_copy, simulated_mask = apply_missingness(
            data_copy, target_column, regime=regime, frac=missingness, seed=42
        )
        simulated_mask = simulated_mask & (~original_missing_mask)

        logging.info(
            f"✅ Missingness applied: regime={regime} | "
            f"simulated={int(simulated_mask.sum())} ({simulated_mask.mean():.2%}) | "
            f"original={int(original_missing_mask.sum())} ({original_missing_mask.mean():.2%})"
        )

        if regime == "event":
            if simulated_mask.sum() == 0:
                logging.warning(
                    "Event regime produced no masked values; falling back to random masking"
                )
                data_copy, simulated_mask = apply_missingness(
                    data_copy,
                    target_column,
                    regime="random",
                    frac=missingness,
                    seed=42,
                )
                simulated_mask = simulated_mask & (~original_missing_mask)
                if simulated_mask.sum() == 0:
                    logging.error(
                        "EVENT REGIME & fallback both failed: No values masked!"
                    )
                    logging.error(
                        f"   Target stats: min={data[target_column].min():.2f}, "
                        f"max={data[target_column].max():.2f}, "
                        f"90th percentile={data[target_column].quantile(0.9):.2f}"
                    )
                else:
                    masked_values = data.loc[simulated_mask, target_column]
                    logging.info(
                        "   Fallback masking successful: masked values range "
                        f"[{masked_values.min():.2f}, {masked_values.max():.2f}]"
                    )
            else:
                masked_values = data.loc[simulated_mask, target_column]
                logging.info(
                    "   Event masking successful: masked values range "
                    f"[{masked_values.min():.2f}, {masked_values.max():.2f}]"
                )

        if sort_by_hour:
            logging.info("Sorting data by hour for imputation.")
            data_copy["Hour"] = data_copy["DateTime"].dt.hour
            sorted_data = data_copy.sort_values(by="Hour")
            imputed_data_list = []
            for hour in sorted_data["Hour"].unique():
                hour_data = sorted_data[sorted_data["Hour"] == hour]
                try:
                    feat_df = (
                        hour_data[input_columns]
                        if input_columns
                        else hour_data.drop(columns=[target_column], errors="ignore")
                    )
                    available_feature_rows = feat_df.dropna(how="all").shape[0]
                except Exception:
                    available_feature_rows = 0

                if available_feature_rows == 0:
                    logging.warning(
                        "Hour %s has no valid feature rows for imputation "
                        "(all features missing or non-numeric); "
                        "skipping imputer for this hour",
                        hour,
                    )
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
                        expected_index=hour_data.index,
                    )
                if imputed_hour_data is None or target_column not in imputed_hour_data.columns:
                    logging.warning(
                        "Imputer failed or missing target for hour %s, falling back to original "
                        "values for that hour",
                        hour,
                    )
                    hour_data_imputed = hour_data.copy()
                    hour_data_imputed[target_column] = hour_data[target_column]
                    imputed_hour_data = hour_data_imputed
                imputed_data_list.append(imputed_hour_data)

            try:
                imputed_data = pd.concat(imputed_data_list).sort_index()
            except Exception as e:
                logging.error("Failed to concat per-hour imputed data: %s", e)
                imputed_data = data_copy.copy()
                imputed_data[target_column] = data_copy[target_column]
        else:
            logging.info("Performing imputation on entire dataset.")
            try:
                feat_df_full = (
                    data_copy[input_columns]
                    if input_columns
                    else data_copy.drop(columns=[target_column], errors="ignore")
                )
                usable_rows = feat_df_full.dropna(how="all").shape[0]
            except Exception:
                usable_rows = 0

            if usable_rows == 0:
                logging.warning(
                    "No usable input feature rows for full-dataset imputation for %s; "
                    "skipping imputer and falling back to original values.",
                    site_name,
                )
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
                    expected_index=data_copy.index,
                )

            if imputed_data is None:
                logging.warning(
                    "Imputer returned no output for full-dataset run; "
                    "falling back to original column values"
                )
                imputed_data = data_copy.copy()
                imputed_data[target_column] = data_copy[target_column]
                imputed_ok = False
            else:
                if target_column not in imputed_data.columns:
                    logging.warning(
                        "Imputer output missing target column '%s'; "
                        "falling back to original target values",
                        target_column,
                    )
                    imputed_data = imputed_data.copy()
                    imputed_data[target_column] = data_copy[target_column]
                    imputed_ok = False
                else:
                    imputed_ok = True

        try:
            if isinstance(imputed_data, pd.DataFrame) and not imputed_data.index.equals(data.index):
                logging.info("Reindexing imputed_data to match original data index")
                try:
                    imputed_data = imputed_data.reindex(data.index)
                except Exception:
                    imputed_data = imputed_data.reset_index(drop=True).reindex(
                        index=data.index
                    )
        except Exception as e:
            logging.warning(
                "Failed to robustly reindex imputed_data: %s. "
                "Proceeding, but alignment may be wrong.",
                e,
            )

        actual_columns_used = [
            col
            for col in imputed_data.columns
            if col in input_columns
            or col.startswith("spatial_")
            or col
            in [
                "Hour",
                "Day",
                "Month",
                "DayOfWeek",
                "DayOfYear",
                "WeekOfYear",
                "Hour_sin",
                "Hour_cos",
                "Month_sin",
                "Month_cos",
                "DayOfWeek_sin",
                "DayOfWeek_cos",
            ]
        ]
        actual_columns_used = [col for col in actual_columns_used if col != target_column]

        base_features = [col for col in actual_columns_used if col in input_columns]
        spatial_features = [col for col in actual_columns_used if col.startswith("spatial_")]
        temporal_features = [
            col
            for col in actual_columns_used
            if col
            in [
                "Hour",
                "Day",
                "Month",
                "DayOfWeek",
                "DayOfYear",
                "WeekOfYear",
                "Hour_sin",
                "Hour_cos",
                "Month_sin",
                "Month_cos",
                "DayOfWeek_sin",
                "DayOfWeek_cos",
            ]
        ]
        lagged_features = [col for col in actual_columns_used if "lag_" in col]
        rolling_features = [col for col in actual_columns_used if "rolling_" in col]

        feature_info = {
            "Site": site_short,
            "Model": canonical_model,
            "Target_Column": target_column,
            "Missingness_Regime": regime,
            "Missingness_Level": f"{int(missingness * 100)}%",
            "Total_Features": len(actual_columns_used),
            "Base_Features_Count": len(base_features),
            "Spatial_Features_Count": len(spatial_features),
            "Temporal_Features_Count": len(temporal_features),
            "Lagged_Features_Count": len(lagged_features),
            "Rolling_Features_Count": len(rolling_features),
            "Total_Rows": len(imputed_data),
            "Original_Missing_Count": int(original_missing_mask.sum()),
            "Simulated_Missing_Count": int(simulated_mask.sum()),
            "Total_Missing_Count": int(
                (original_missing_mask | simulated_mask).sum()
            ),
        }
        all_feature_info.append(feature_info)

        final_imputed_data = data.copy()
        if len(imputed_data) != len(data):
            logging.warning(
                "Imputed data length (%d) != original data length (%d); "
                "using original values for target",
                len(imputed_data),
                len(data),
            )
            final_imputed_data[target_column] = data[target_column]
        else:
            final_imputed_data[target_column] = imputed_data[target_column]

        final_imputed_csv_filename = os.path.join(
            imputed_data_path,
            f"{site_name}_{target_column}_{canonical_model}_{regime}_imputed_{int(missingness * 100)}.csv",
        )
        try:
            actual_col = f"Actual_{target_column}"
            imputed_col = f"Imputed_{target_column}"
            try:
                imputed_series = (
                    final_imputed_data[target_column]
                    if isinstance(final_imputed_data, pd.DataFrame)
                    and target_column in final_imputed_data.columns
                    else data[target_column]
                )
            except Exception:
                imputed_series = data[target_column]

            try:
                imputed_mask = simulated_mask | original_missing_mask
            except Exception:
                imputed_mask = pd.Series([False] * len(data))

            comments = [
                "Imputed" if imputed_mask.iat[i] else "Original"
                for i in range(len(data))
            ]

            minimal_df = pd.DataFrame(
                {
                    "DateTime": data["DateTime"].values,
                    actual_col: data[target_column].values,
                    imputed_col: imputed_series.values,
                    "Comments": comments,
                }
            )

            metadata = {
                "Site": site_short,
                "Model": canonical_model,
                "Regime": regime,
                "Missingness_Level": int(missingness * 100),
                "Target": target_column,
            }
            append_ok = append_imputed_to_master(
                minimal_df, metadata, central_imputed_root=central_imputed_root
            )
            if append_ok:
                logging.info(
                    "Appended %d rows to central imputed master for %s %s",
                    len(minimal_df),
                    site_short,
                    target_column,
                )
            else:
                logging.warning(
                    "Failed to append imputed rows to central master for %s %s "
                    "(or central_imputed_root not provided)",
                    site_short,
                    target_column,
                )

            os.makedirs(os.path.dirname(final_imputed_csv_filename), exist_ok=True)
            final_imputed_data.to_csv(final_imputed_csv_filename, index=False)
            logging.info("Saved imputed data to: %s", final_imputed_csv_filename)
        except Exception as e:
            logging.error(
                "Failed to save imputed data to %s: %s", final_imputed_csv_filename, e
            )

        target_column_data = data[["DateTime", target_column]].copy()
        target_column_data[f"{target_column}_imputed"] = np.nan
        if len(imputed_data) == len(data):
            target_column_data.loc[
                simulated_mask | original_missing_mask,
                f"{target_column}_imputed",
            ] = imputed_data.loc[
                simulated_mask | original_missing_mask, target_column
            ]
        else:
            try:
                target_column_data[f"{target_column}_imputed"] = imputed_data[
                    target_column
                ]
            except Exception:
                pass

        target_column_data["Missing_Type"] = "None"
        target_column_data.loc[original_missing_mask, "Missing_Type"] = "Original"
        target_column_data.loc[simulated_mask, "Missing_Type"] = "Simulated"

        target_column_csv_filename = os.path.join(
            target_column_data_path,
            f"{site_name}_{target_column}_{canonical_model}_{regime}_target_column_{int(missingness * 100)}.csv",
        )
        try:
            os.makedirs(os.path.dirname(target_column_csv_filename), exist_ok=True)
            target_column_data.to_csv(target_column_csv_filename, index=False)
            logging.info("Saved target column data to: %s", target_column_csv_filename)
        except Exception as e:
            logging.exception(
                "Failed to save target column data to %s: %s",
                target_column_csv_filename,
                e,
            )

        metrics_row = None
        if simulated_mask.sum() > 0:
            true_values_simulated = pd.to_numeric(
                data.loc[simulated_mask, target_column], errors="coerce"
            ).to_numpy(dtype=float)
            if len(imputed_data) == len(data):
                imputed_values_simulated = pd.to_numeric(
                    imputed_data.loc[simulated_mask, target_column],
                    errors="coerce",
                ).to_numpy(dtype=float)
            else:
                imputed_values_simulated = np.array([])

            if handle_negatives == "exclude":
                valid_mask = (true_values_simulated >= 0) & (imputed_values_simulated >= 0)
                true_values_clean = true_values_simulated[valid_mask]
                imputed_values_clean = imputed_values_simulated[valid_mask]
            else:
                true_values_clean = true_values_simulated
                imputed_values_clean = imputed_values_simulated

            if len(true_values_clean) > 0:
                metrics = evaluate_metrics(
                    true_values_clean,
                    imputed_values_clean,
                    handle_negative=handle_negatives,
                )
                metrics["Missingness"] = missingness
                metrics.setdefault("Missingness_Regime", regime)
                metrics.setdefault("Total_Features_Used", len(actual_columns_used))
                metrics.setdefault("Base_Features", len(base_features))
                metrics.setdefault("Spatial_Features", len(spatial_features))
                metrics.setdefault("Temporal_Features", len(temporal_features))
                if "Root Mean Squared Error (RMSE)" not in metrics and "RMSE" in metrics:
                    metrics["Root Mean Squared Error (RMSE)"] = metrics["RMSE"]
                if "RMSE" not in metrics and "Root Mean Squared Error (RMSE)" in metrics:
                    metrics["RMSE"] = metrics["Root Mean Squared Error (RMSE)"]

                try:
                    row_dict = {
                        **metrics,
                        "Missingness_Regime": regime,
                        "Missingness": missingness,
                        "Total_Features_Used": len(actual_columns_used),
                        "Base_Features": len(base_features),
                        "Spatial_Features": len(spatial_features),
                        "Temporal_Features": len(temporal_features),
                        "RMSE": metrics.get(
                            "Root Mean Squared Error (RMSE)", np.nan
                        ),
                        "Model": canonical_model,
                        "StudySite": site_short,
                    }
                    # For target-level CSVs, use results_root if provided, else metrics_save_path
                    target_metrics_root = results_root or metrics_save_path
                    save_target_metrics_csv(
                        rows=row_dict,
                        output_dir=target_metrics_root,
                        target_variable=target_column,
                    )
                    logging.info(
                        "Saved target metrics via save_target_metrics_csv: "
                        "%s %s %s%%",
                        canonical_model,
                        site_short,
                        int(missingness * 100),
                    )
                except Exception as e:
                    logging.warning(
                        "Failed to build row_dict for save_target_metrics_csv: %s", e
                    )

                all_metrics.append(metrics)
                metrics_row = metrics

                logging.info(
                    "\nMetrics for %d%% %s:", int(missingness * 100), regime
                )
                logging.info(
                    "  RMSE: %.4f",
                    metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                )
                logging.info(
                    "  R:  %.4f",
                    metrics.get("Correlation Coefficient (R)", np.nan),
                )
                logging.info(
                    "  NSE: %.4f",
                    metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                )

                try:
                    gap_metrics = evaluate_metrics_by_gap(
                        y_true=data[target_column],
                        y_pred=imputed_data[target_column],
                        simulated_mask=simulated_mask,
                        metric_functions=METRIC_FUNCTIONS,
                    )
                    gap_metrics.to_csv(
                        os.path.join(
                            metrics_save_path,
                            f"{canonical_model}_{target_column}_{regime}_{int(missingness * 100)}_gapwise_metrics.csv",
                        ),
                        index=False,
                    )
                except Exception as e:
                    logging.error("Error computing gapwise metrics: %s", e)

                try:
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
                        r=metrics.get("Correlation Coefficient (R)"),
                    )
                    save_error_distribution(
                        true_values_clean,
                        imputed_values_clean,
                        central_plots_root,
                        site_name,
                        target_column,
                        model_name_with_regime,
                        missingness,
                    )
                    save_residual_plot(
                        true_values_clean,
                        imputed_values_clean,
                        central_plots_root,
                        site_name,
                        target_column,
                        model_name_with_regime,
                        missingness,
                    )
                    save_qq_plot(
                        imputed_values_clean,
                        central_plots_root,
                        site_name,
                        target_column,
                        model_name_with_regime,
                        missingness,
                    )
                    save_correlation_heatmap(
                        imputed_data,
                        central_plots_root,
                        site_name,
                        target_column,
                        model_name_with_regime,
                        missingness,
                    )
                    save_statistical_summary(
                        data_copy,
                        imputed_data,
                        central_plots_root,
                        site_name,
                        target_column,
                        model_name_with_regime,
                        missingness,
                    )
                    save_cdf_plot(
                        true_values_clean,
                        imputed_values_clean,
                        central_plots_root,
                        site_name,
                        target_column,
                        model_name_with_regime,
                        missingness,
                    )
                    save_histogram(
                        data_copy,
                        imputed_values_clean,
                        central_plots_root,
                        site_name,
                        target_column,
                        model_name_with_regime,
                        missingness,
                    )
                except Exception as e:
                    logging.error("Error generating plots: %s", e)
            else:
                logging.warning(
                    "No valid simulated values after negative filtering; "
                    "metrics will be NaN for this missingness level."
                )
        else:
            logging.warning(
                "No simulated masked values produced for this missingness level; "
                "creating placeholder metrics row (NaNs)."
            )

        metrics_filename = os.path.join(
            metrics_save_path,
            f"{site_name}_{target_column}_{canonical_model}_{regime}_{int(missingness * 100)}_all_metrics.csv",
        )

        if metrics_row is None:
            placeholder = {
                "Missingness": missingness,
                "Missingness_Regime": regime,
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
            }
            save_df = pd.DataFrame([placeholder])
        else:
            metrics_row.setdefault("Missingness", missingness)
            metrics_row.setdefault("Missingness_Regime", regime)
            save_df = pd.DataFrame([metrics_row])

        save_df["Model"] = canonical_model
        save_df["StudySite"] = site_short

        try:
            save_df.to_csv(metrics_filename, index=False)
            logging.info("Saved metrics CSV: %s", metrics_filename)
        except Exception as e:
            logging.error("Failed to save metrics CSV %s: %s", metrics_filename, e)

        try:
            append_metrics_to_master(save_df, central_metrics_root=central_metrics_root)
        except Exception as e:
            logging.warning("Appending to central metrics master failed: %s", e)

        manifest_entries.append(
            {
                "site": site_short,
                "target": target_column,
                "model": canonical_model,
                "regime": regime,
                "missingness_pct": int(missingness * 100),
                "imputed_file": final_imputed_csv_filename,
                "target_column_file": target_column_csv_filename,
                "metrics_file": metrics_filename,
                "simulated_masked_count": int(simulated_mask.sum()),
                "original_missing_count": int(original_missing_mask.sum()),
            }
        )

    # Combined metrics CSV for this site/target/model/regime
    if all_metrics:
        try:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_csv_filename_combined = os.path.join(
                metrics_save_path,
                f"{site_name}_{target_column}_{canonical_model}_{regime}_all_metrics.csv",
            )
            metrics_df.to_csv(metrics_csv_filename_combined, index=False)
            logging.info(
                "Saved combined all-level metrics to: %s", metrics_csv_filename_combined
            )
        except Exception as e:
            logging.error("Failed saving combined metrics: %s", e)
    else:
        logging.warning(
            "No metrics computed across missingness levels (all_metrics empty). "
            "Combined CSV will not contain informative rows; per-level placeholders were still saved."
        )

    manifest_fp = os.path.join(
        metrics_save_path, f"saved_runs_manifest_{canonical_model}_{regime}.json"
    )
    try:
        with open(manifest_fp, "w") as fh:
            json.dump(manifest_entries, fh, indent=2)
        logging.info("Saved run manifest: %s", manifest_fp)
        for e in manifest_entries:
            logging.info(
                "SAVED -> model=%s site=%s regime=%s missing=%d%% imputed=%s metrics=%s",
                e["model"],
                e["site"],
                e["regime"],
                e["missingness_pct"],
                os.path.basename(e["imputed_file"]),
                os.path.basename(e["metrics_file"]),
            )
    except Exception as e:
        logging.error("Failed to write manifest JSON: %s", e)