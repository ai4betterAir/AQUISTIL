"""
BaseLine.py

Baseline imputer for AQMS data.

This module now exposes:
 - impute_mice(data, target_column, input_columns, **kwargs)
   which returns a DataFrame with the target column imputed using a
   simple daily-hourly-proportion baseline method. This makes BaseLine
   compatible with the pipeline's expectations (sortdata.py / main.py).

 - run_baseline(...) helpers retained for standalone baseline generation.

Behavior of impute_mice:
 - Requires a DateTime column (named 'DateTime' or 'datetime') or a DatetimeIndex.
 - Builds an hourly-aligned series for the target.
 - Computes hourly-average proportions of daily totals across days.
 - For each timestamp, baseline = avg_hour_prop[hour] * daily_total_of_that_day.
 - Fills missing target values with baseline; falls back to hourly mean or global mean
   when necessary.
 - Returns a DataFrame aligned to the input with the target column filled.
"""
import os
import glob
import logging
import shutil
from typing import List, Sequence, Optional

import pandas as pd
import numpy as np

MODEL_NAME = "BaseLine"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# (run_baseline/process_site_file kept below for standalone use; see end of file)


def _ensure_datetime_index(df: pd.DataFrame, datetime_col_candidates=("DateTime", "datetime")) -> pd.DataFrame:
    """
    Ensure the DataFrame has a DatetimeIndex. If a DateTime column exists, convert it.
    Returns a copy with a DatetimeIndex.
    """
    df = df.copy()
    for col in datetime_col_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col])
            df = df.set_index(col).sort_index()
            return df
    # If index is already datetime-like, ensure it's a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    # Try to infer
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        raise ValueError("No DateTime column found and index could not be converted to DatetimeIndex.")


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    """
    Simple baseline imputer exposed with the same signature the pipeline expects.

    Args:
        data (pd.DataFrame): Input data containing DateTime and columns.
        target_column (str): Name of the target column to impute (e.g., 'PM2.5').
        input_columns (list): Local input columns (unused by baseline but accepted).
        custom_strategies: compatibility param (unused).
        **kwargs: ignored.

    Returns:
        pd.DataFrame: Copy of input data with target_column imputed (filled).
    """
    try:
        # Regional pooled inputs repeat every timestamp once per monitoring
        # site. Run this temporal baseline per site and preserve the caller's
        # row index/order instead of treating those rows as duplicate samples.
        if "Site" in data.columns and data["Site"].nunique(dropna=False) > 1:
            result = data.copy()
            target_position = result.columns.get_loc(target_column)
            for _, positions in data.groupby("Site", sort=False, dropna=False).indices.items():
                positions = np.asarray(positions)
                site_result = impute_mice(
                    data.iloc[positions].copy(), target_column, input_columns,
                    custom_strategies=custom_strategies, **kwargs
                )
                if len(site_result) != len(positions):
                    raise ValueError("per-site baseline changed the input row count")
                result.iloc[positions, target_position] = pd.to_numeric(
                    site_result[target_column], errors="coerce"
                ).to_numpy()
            return result

        df = data.copy()
        # Normalize DateTime -> DatetimeIndex
        try:
            df = _ensure_datetime_index(df)
        except ValueError as e:
            logging.warning(f"[{MODEL_NAME}] impute_mice: {e} — returning original DataFrame unchanged.")
            return data.copy()

        if target_column not in df.columns:
            logging.warning(f"[{MODEL_NAME}] impute_mice: target column '{target_column}' not present — returning original.")
            return data.copy()

        # Regional inputs contain one row per site and timestamp, so duplicate
        # timestamps are expected. Aggregate duplicate timestamps within each
        # site before creating the hourly axis; otherwise pandas cannot reindex.
        if not df.index.is_unique:
            aggregations = {
                column: ("mean" if pd.api.types.is_numeric_dtype(df[column]) else "first")
                for column in df.columns
            }
            df = df.groupby(level=0, sort=True).agg(aggregations)

        # Align to full hourly index (prevent misalignment with pipeline expectations)
        start = df.index.min()
        end = df.index.max()
        # Pandas 3.0 removed the deprecated uppercase hourly alias ("H").
        full_index = pd.date_range(start=start, end=end, freq="h")
        df = df.reindex(full_index)

        y = pd.to_numeric(df[target_column], errors="coerce")

        # DAILY TOTAL per day (use calendar day)
        day_index = y.index.normalize()
        daily_total = y.groupby(day_index).transform("sum")

        # Hour of day
        hour = y.index.hour

        # Hourly proportion within each day: value / daily_total
        with np.errstate(invalid="ignore", divide="ignore"):
            hourly_prop = y / daily_total

        # replace inf/nan when daily_total == 0 -> set proportion to NaN
        hourly_prop = hourly_prop.replace([np.inf, -np.inf], np.nan)

        # Average proportion per hour across days (skip NaNs)
        avg_prop_by_hour = hourly_prop.groupby(hour).mean()

        # If avg_prop_by_hour is all NaN (lack of data), fall back to hourly mean of y
        if avg_prop_by_hour.isna().all():
            logging.warning(f"[{MODEL_NAME}] hourly proportion aggregation produced all NaN. Falling back to hourly mean.")
            avg_prop_by_hour = y.groupby(hour).mean()
            # Normalize hourly means to sum to 1 across 24 hours if possible
            if not avg_prop_by_hour.isna().all():
                s = avg_prop_by_hour.sum(skipna=True)
                if s and s != 0:
                    avg_prop_by_hour = avg_prop_by_hour / s
        else:
            # For safety, if avg_prop_by_hour sums to zero or NaN, fallback
            total_prop = avg_prop_by_hour.sum(skipna=True)
            if not total_prop or np.isclose(total_prop, 0.0):
                logging.warning(f"[{MODEL_NAME}] avg hourly proportions sum to {total_prop}. Normalizing or falling back.")
                hm = y.groupby(hour).mean()
                if not hm.isna().all() and hm.sum(skipna=True) != 0:
                    avg_prop_by_hour = hm / hm.sum(skipna=True)
                else:
                    # final fallback: uniform proportions
                    avg_prop_by_hour = pd.Series(np.repeat(1.0/24.0, 24), index=range(24))

        # Build baseline series: baseline = avg_prop_by_hour[hour] * daily_total
        baseline = pd.Series(index=y.index, dtype=float)
        for h in range(24):
            mask_h = (hour == h)
            prop = avg_prop_by_hour.get(h, np.nan)
            baseline.loc[mask_h] = prop * daily_total.loc[mask_h]

        # Replace NaNs in baseline with hourly mean OR global mean
        # hourly mean
        hourly_mean = y.groupby(hour).mean()
        idx_na = baseline.isna()
        if idx_na.any():
            for ts in baseline.index[idx_na]:
                h = ts.hour
                val = hourly_mean.get(h, np.nan)
                baseline.at[ts] = val

        # Any remaining NaNs -> global mean of observed y
        if baseline.isna().any():
            global_mean = y.mean(skipna=True)
            baseline.fillna(global_mean, inplace=True)

        # Build imputed series: use original observed values when present, else baseline
        imputed = y.copy()
        missing_mask = imputed.isna()
        imputed.loc[missing_mask] = baseline.loc[missing_mask]

        # Final safety: if still NaN (unlikely), fill with global_mean or 0
        if imputed.isna().any():
            gm = imputed.mean(skipna=True)
            if pd.isna(gm):
                gm = 0.0
            imputed.fillna(gm, inplace=True)

        # Assign back to DataFrame (preserve original columns order)
        out_df = df.copy()
        out_df[target_column] = imputed

        # Return reindexed to original input ordering if input had explicit index not datetime
        # But pipeline expects same index as input; we return with DatetimeIndex (this matches pipeline behavior)
        return out_df.reset_index().rename(columns={'index': 'DateTime'}) if not isinstance(data.index, pd.DatetimeIndex) and 'DateTime' in data.columns else out_df

    except Exception as exc:
        logging.exception(f"[{MODEL_NAME}] impute_mice failed: {exc}")
        # On failure, return original input to avoid pipeline crash
        return data.copy()


# -------------------------------
# The rest of the file contains the standalone baseline generation functions
# (process_site_file / run_baseline) used when BaseLine is executed standalone.
# They are left as-is (omitted here for brevity) but must exist in the module.
# -------------------------------

# Minimal safe implementations of process_site_file/run_baseline for standalone use
def extract_site_names(column_names: Sequence[str]) -> List[str]:
    site_names = []
    for name in column_names:
        if "_" in name:
            # Use the first token before underscore as canonical site token
            parts = name.split("_")
            site_name = parts[0]
            if site_name not in site_names:
                site_names.append(site_name)
    return site_names

def process_site_file(file_path: str, site_name: str, output_directory: str) -> Optional[str]:
    try:
        site_df = pd.read_csv(file_path)
        if "DateTime" in site_df.columns:
            site_df = site_df.rename(columns={"DateTime": "datetime"})
        if "datetime" not in site_df.columns:
            return None
        site_df["datetime"] = pd.to_datetime(site_df["datetime"], errors="coerce")
        site_df = site_df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
        full_index = pd.date_range(start=site_df.index.min(), end=site_df.index.max(), freq="h")
        site_df = site_df.reindex(full_index)
        site_df = site_df.interpolate(method="time", limit_direction="both").reset_index().rename(columns={"index":"datetime"})
        # trivial baseline: copy numeric columns -> Baseline_<var>_<site>
        numeric_cols = [c for c in site_df.columns if c != "datetime" and np.issubdtype(site_df[c].dtype, np.number)]
        for c in numeric_cols:
            site_df[f"Baseline_{c}_{site_name}"] = site_df[c].fillna(method="ffill").fillna(method="bfill")
        # os.makedirs(output_directory, exist_ok=True)  # Directory creation disabled
        out_fp = os.path.join(output_directory, f"baseline_all_{site_name}.csv")
        site_df.to_csv(out_fp, index=False)
        return out_fp
    except Exception:
        return None

def run_baseline(input_directory: str, output_directory: str, file_pattern: str = "*.csv") -> List[str]:
    saved_files = []
    pattern = os.path.join(input_directory, file_pattern)
    files = sorted(glob.glob(pattern))
    for filepath in files:
        try:
            df = pd.read_csv(filepath, nrows=5)
            cols = df.columns.tolist()
            site_names = extract_site_names(cols) or [os.path.basename(filepath).split('_')[0]]
            for site in site_names:
                saved = process_site_file(filepath, site, output_directory)
                if saved:
                    saved_files.append(saved)
        except Exception:
            continue
    return saved_files


# Standalone execution convenience
if __name__ == "__main__":
    input_dir = os.environ.get("BASELINE_INPUT_DIR", "/tmp")
    output_dir = os.environ.get("BASELINE_OUTPUT_DIR", "/tmp/baseline_out")
    saved = run_baseline(input_dir, output_dir)
    if saved:
        logging.info(f"[{MODEL_NAME}] Saved baseline files:\n" + "\n".join(saved))
    else:
        logging.info(f"[{MODEL_NAME}] No baseline files generated.")
