"""
Comprehensive Feature Selection & Analysis Framework

Tests different feature configurations to identify optimal inputs for PM2.5 imputation: 
1. Individual feature importance (which variables matter?)
2. Lag configuration analysis (which time lags are useful?)
3. Rolling window optimization (which windows capture trends?)
4. Feature combination analysis (synergistic effects?)
5. Domain-specific feature groups (meteorology vs pollutants?)

Outputs:
- Feature importance rankings with visualizations
- Optimal lag/rolling configurations
- Performance comparison across configurations
- Recommendations for production deployment

Author: Dr.  Masrur
Date: 2026-01-22
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import calendar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import json

import config_spatial as config

impute_mice = None
evaluate_metrics = None
apply_missingness = None


def load_legacy_dependencies():
    """Load LGBM-based legacy analysis dependencies only when legacy mode is requested."""
    global impute_mice, evaluate_metrics, apply_missingness
    if impute_mice is not None and evaluate_metrics is not None and apply_missingness is not None:
        return

    from Model.LGBM_AQ_Plus_SpatialIter_Optimized_V2 import impute_mice as legacy_impute_mice
    from evaluation_metrics import evaluate_metrics as legacy_evaluate_metrics
    from missingness_regimes import apply_missingness as legacy_apply_missingness

    impute_mice = legacy_impute_mice
    evaluate_metrics = legacy_evaluate_metrics
    apply_missingness = legacy_apply_missingness

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# One-place base directory configuration (keep consistent with feature_selection_iXAI_basics.py)
# ---------------------------------------------------------------------------
DEFAULT_BASE_DIR = Path("/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model")
BASE_DIR = Path(os.environ.get("IMPUTATION_MODEL_BASE_DIR", str(DEFAULT_BASE_DIR))).resolve()
if not BASE_DIR.exists():
    BASE_DIR = SCRIPT_DIR.parent

OUTPUTS_DIR = BASE_DIR / "Outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = OUTPUTS_DIR
LOG_FILE = OUTPUT_DIR / "feature_selection_analysis.log"
REGIONAL_ASSESSMENT_DIR = OUTPUT_DIR / "Reginal_Assessment"
REGIONAL_ASSESSMENT_DIR.mkdir(parents=True, exist_ok=True)
BASICS_DIR = OUTPUT_DIR / "Basics"
BASICS_DIR.mkdir(parents=True, exist_ok=True)
REGIONAL_INPUT_DIR = Path(
    os.environ.get(
        "FEATURE_SELECTION_REGION_INPUT_DIR",
        "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/cnn_lstm_forecast/API_Input/Inputs",
    )
)
REGIONAL_INPUT_PATTERN = "Allobs_processed_DPE_station_api_*_ALL.csv"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Plotting style

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300  # ✅ No space
plt.rcParams['savefig.dpi'] = 300


# ============================================================================
# FEATURE GROUPS (Domain Knowledge)
# ============================================================================

FEATURE_GROUPS = {
    'meteorology_basic': ['TEMP', 'HUMID', 'WSP', 'RAIN'],
    'meteorology_wind': ['WSP', 'WDR'],
    'meteorology_full': ['TEMP', 'HUMID', 'WSP', 'RAIN', 'WDR'],
    'pollutants_pm':  ['PM10'],
    'pollutants_full': ['PM10', 'CO', 'NO', 'NO2', 'NOX', 'OZONE', 'SO2'],
    'minimal':  ['TEMP', 'HUMID', 'PM10'],
    'wind_only': ['WSP', 'WDR'],
    'temp_humid':  ['TEMP', 'HUMID'],
}

LAG_CONFIGURATIONS = {
    'none': [],
    'short':  [1, 6],
    'medium': [1, 6, 24],
    'long': [1, 6, 24, 72],
    'extensive': [1, 3, 6, 12, 24, 48, 72],
    'hourly': [1, 2, 3, 4, 5, 6],
    'daily': [24, 48, 72, 96, 120, 144, 168],
}

ROLLING_CONFIGURATIONS = {
    'none': [],
    'short': [(6, 'mean'), (6, 'std')],
    'medium': [(24, 'mean'), (24, 'std')],
    'long': [(24, 'mean'), (24, 'std'), (72, 'mean')],
    'extensive': [(6, 'mean'), (24, 'mean'), (24, 'std'), (72, 'mean'), (168, 'mean')],
    'multi_stat': [(24, 'mean'), (24, 'std'), (24, 'max'), (24, 'min')],
}

SIMPLE_XGBRF_FEATURES = [
    "PM10",
    "TEMP",
    "HUMID",
    "WSP",
    "RAIN",
    "WDR",
    "WGU",
    "OZONE",
    "CO",
    "NO",
    "NO2",
    "NOX",
    "SO2",
    "NEPH",
    "SOLAR",
]

DEFAULT_SIMPLE_XGBRF_TARGETS = ["PM2.5", "PM10", "OZONE", "NO", "NO2"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_site_token(value):
    """Normalize site names so WYONG, Wyong, and WYONG-like suffixes compare consistently."""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def canonical_site_name(value):
    """Return a stable site token for output filenames and logs."""
    text = str(value).strip().upper()
    text = text.replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def candidate_variable_names(target_column="PM2.5"):
    variables = set()
    variables.update(getattr(config, "INPUT_COLUMNS", []) or [])
    variables.update(getattr(config, "TARGET_COLUMNS", []) or [])
    variables.add(target_column)
    variables.update(["CO", "NO", "NO2", "NOX", "OZONE", "SO2", "PM10", "PM2.5"])
    variables.update(["TEMP", "HUMID", "WSP", "WDR", "RAIN", "WGU", "NEPH", "SOLAR"])
    for group_vars in FEATURE_GROUPS.values():
        variables.update(group_vars)
    return sorted({str(var).strip() for var in variables if str(var).strip()}, key=len, reverse=True)


def ensure_output_subdir(base_dir, subdir_name):
    """Create and return a subdirectory under base_dir for organizing outputs by type."""
    path = Path(base_dir) / str(subdir_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_region_target_best_predictors_json(
    target_column,
    selected_regions,
    single_feature_results_df,
    feature_combo_df=None,
    permutation_df=None,
    output_dir=REGIONAL_ASSESSMENT_DIR,
    top_n_single=5,
):
    """
    Save a JSON summary:
      { "<region>": { "<target>": { "best_predictors": [...], ... } } }

    Priority for best_predictors:
      1) Best-performing feature combination (lowest RMSE) if available
      2) Top-N single-feature predictors by RMSE
    """
    summary_dir = ensure_output_subdir(output_dir, "Summaries")
    payload = {}

    # Normalize selected regions to match tokens from the CSV names
    region_set = set(selected_regions or [])

    # Overall best single feature for this target (compiled scope)
    try:
        compiled = single_feature_results_df[single_feature_results_df["scope"] == "compiled"].copy()
        if not compiled.empty:
            best_row = compiled.sort_values("rmse").iloc[0]
            payload.setdefault("__OVERALL__", {})[str(target_column)] = {
                "best_single_feature": str(best_row["feature"]),
                "rmse": float(best_row["rmse"]),
                "mae": float(best_row["mae"]),
                "r": float(best_row["r"]),
                "nse": float(best_row["nse"]),
                "n_train": int(best_row.get("n_train", 0)),
                "n_test": int(best_row.get("n_test", 0)),
                "n_sites": int(best_row.get("n_sites", 0)) if str(best_row.get("n_sites", "")).isdigit() else best_row.get("n_sites"),
            }
    except Exception:
        logging.exception("Failed computing overall best single feature for JSON summary")

    for region_name in sorted(single_feature_results_df["region"].unique()):
        if region_name == "ALL_SELECTED":
            continue
        if region_set and region_name not in region_set:
            continue

        region_entry = payload.setdefault(region_name, {})
        target_entry = region_entry.setdefault(str(target_column), {})

        best_predictors = []
        best_source = "single_feature_top_n"

        # 1) Best combination if available
        if feature_combo_df is not None and not feature_combo_df.empty:
            region_combos = feature_combo_df[feature_combo_df["region"] == region_name].copy()
            if not region_combos.empty:
                best_row = region_combos.sort_values("rmse").iloc[0]
                combo_feats = str(best_row.get("ranked_features_used", "")).split(",")
                combo_feats = [f.strip() for f in combo_feats if f.strip()]
                if combo_feats:
                    best_predictors = combo_feats
                    best_source = "best_combo_rmse"
                    target_entry["best_combo"] = {
                        "n_features": int(best_row.get("n_features", len(combo_feats))),
                        "rmse": float(best_row.get("rmse")),
                        "mae": float(best_row.get("mae")),
                        "r": float(best_row.get("r")),
                        "nse": float(best_row.get("nse")),
                        "features": combo_feats,
                    }

        # 2) Fallback: top-N single features
        if not best_predictors:
            region_singles = single_feature_results_df[
                (single_feature_results_df["scope"] == "per_region")
                & (single_feature_results_df["region"] == region_name)
            ].sort_values("rmse")
            top = region_singles.head(int(top_n_single))
            best_predictors = top["feature"].tolist()
            target_entry["top_single_features"] = [
                {
                    "feature": row["feature"],
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "r": float(row["r"]),
                    "nse": float(row["nse"]),
                }
                for _, row in top.iterrows()
            ]

        target_entry["best_predictors"] = best_predictors
        target_entry["best_predictors_source"] = best_source

        # Optional: attach permutation importance top list (if provided)
        if permutation_df is not None and not permutation_df.empty:
            perm_region = permutation_df[permutation_df["region"] == region_name].copy()
            if not perm_region.empty and "delta_rmse_mean" in perm_region.columns:
                perm_top = perm_region.sort_values("delta_rmse_mean", ascending=False).head(int(top_n_single))
                target_entry["permutation_importance_top"] = [
                    {
                        "feature": row["feature"],
                        "delta_rmse_mean": float(row["delta_rmse_mean"]),
                        "delta_rmse_std": float(row.get("delta_rmse_std", np.nan)),
                        "base_rmse": float(row.get("base_rmse", np.nan)),
                    }
                    for _, row in perm_top.iterrows()
                ]

    out_path = summary_dir / f"BestPredictors_{safe_target_name(target_column)}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Best predictors JSON saved: {out_path}")
    return out_path


def regional_input_files():
    if not REGIONAL_INPUT_DIR.exists():
        logging.warning(f"Regional input directory does not exist: {REGIONAL_INPUT_DIR}")
        return []
    return sorted(REGIONAL_INPUT_DIR.glob(REGIONAL_INPUT_PATTERN))


def normalize_region_token(value):
    """Normalize region names so config names and CSV filename tokens can be matched."""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def region_name_from_input_file(filepath):
    name = Path(filepath).stem
    prefix = "Allobs_processed_DPE_station_api_"
    suffix = "_ALL"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def selected_regional_input_files(selected_regions=None):
    files = regional_input_files()
    if not selected_regions:
        return files

    wanted = {normalize_region_token(region) for region in selected_regions}
    selected = [
        filepath for filepath in files
        if normalize_region_token(region_name_from_input_file(filepath).replace("_", " ")) in wanted
        or normalize_region_token(region_name_from_input_file(filepath)) in wanted
    ]
    matched = {
        normalize_region_token(region_name_from_input_file(filepath).replace("_", " "))
        for filepath in selected
    } | {
        normalize_region_token(region_name_from_input_file(filepath))
        for filepath in selected
    }
    missing = [
        region for region in selected_regions
        if normalize_region_token(region) not in matched
    ]
    if missing:
        logging.warning(
            "No regional processed CSV matched selected region(s): "
            + ", ".join(str(region) for region in missing)
        )
    return selected


def split_regional_column(column_name, target_column="PM2.5"):
    for variable in candidate_variable_names(target_column):
        prefix = f"{variable}_"
        if str(column_name).upper().startswith(prefix.upper()):
            return variable, str(column_name)[len(prefix):]
    return None, None


def get_available_sites_from_regional_inputs(target_column="PM2.5"):
    sites = {}
    for filepath in regional_input_files():
        try:
            columns = pd.read_csv(filepath, nrows=0).columns
        except Exception as exc:
            logging.warning(f"Could not read columns from {filepath}: {exc}")
            continue

        for column in columns:
            if str(column).lower() in {"datetime", "date", "time", "timestamp"}:
                continue
            variable, site = split_regional_column(column, target_column)
            if variable is None or site is None:
                continue
            sites.setdefault(normalize_site_token(site), canonical_site_name(site))
    return sorted(sites.values())


def load_site_data_from_regional_inputs(site_name, target_column="PM2.5"):
    site_key = normalize_site_token(site_name)
    variables = candidate_variable_names(target_column)

    for filepath in regional_input_files():
        try:
            header = pd.read_csv(filepath, nrows=0)
        except Exception as exc:
            logging.warning(f"Could not inspect regional input file {filepath}: {exc}")
            continue

        matched_columns = []
        column_to_variable = {}
        for column in header.columns:
            if str(column).lower() in {"datetime", "date", "time", "timestamp"}:
                continue
            variable, site = split_regional_column(column, target_column)
            if variable is None or site is None:
                continue
            if normalize_site_token(site) != site_key:
                continue
            if variable in variables:
                matched_columns.append(column)
                column_to_variable[column] = variable

        if not matched_columns:
            continue

        datetime_column = next(
            (col for col in header.columns if str(col).lower() in {"datetime", "date", "timestamp"}),
            None,
        )
        use_columns = ([datetime_column] if datetime_column else []) + matched_columns
        df_raw = pd.read_csv(filepath, usecols=use_columns)

        df_site = pd.DataFrame()
        if datetime_column is not None:
            df_site["DateTime"] = pd.to_datetime(df_raw[datetime_column], errors="coerce")
        else:
            df_site["DateTime"] = pd.RangeIndex(start=0, stop=len(df_raw), step=1)

        column_mapping = {}
        for source_column, variable in column_to_variable.items():
            values = pd.to_numeric(df_raw[source_column], errors="coerce")
            column_mapping[source_column] = variable
            if variable in df_site.columns:
                df_site[variable] = df_site[variable].combine_first(values)
            else:
                df_site[variable] = values

        df_site = df_site.dropna(subset=["DateTime"])
        if target_column not in df_site.columns:
            if target_column == "PM2.5" and "PM10" in df_site.columns:
                logging.warning(
                    f"Target PM2.5 not found for {site_name} in {filepath}; using PM10 fallback for compatibility"
                )
                df_site[target_column] = df_site["PM10"].copy()
            else:
                logging.info(
                    f"Regional file {filepath} has site {site_name} but not target {target_column}; skipping"
                )
                continue

        logging.info(
            f"Loaded regional site data for {canonical_site_name(site_name)} from {filepath}: "
            f"{len(df_site)} rows, {len(df_site.columns)} columns"
        )
        return df_site, str(filepath), column_mapping

    raise FileNotFoundError(
        f"No regional wide input data found for site {site_name} in {REGIONAL_INPUT_DIR}"
    )

def get_site_columns(site_name, df):
    """Get available columns for a specific site"""
    base_columns = []
    site_suffix = site_name.upper().replace('-', '_').replace(' ', '_')
    
    # Check each potential column
    for col in df.columns:
        # Skip date/time columns
        if col in ['Date', 'Time', 'DateTime', 'date', 'time']:
            continue
            
        # Extract base column name (remove site suffix)
        if col.endswith(f"_{site_suffix}"):
            base_col = col[:-len(f"_{site_suffix}")]
            base_columns.append(base_col)
        elif '_' not in col and col not in base_columns:
            # Base column without site suffix
            base_columns.append(col)
    
    return base_columns


def normalize_column_names(df, site_name):
    """Normalize column names to remove site suffixes"""
    site_suffix = site_name.upper().replace('-', '_').replace(' ', '_')
    df_normalized = df.copy()
    
    # Create mapping from original to normalized names
    column_mapping = {}
    
    for col in df.columns:
        if col.endswith(f"_{site_suffix}"):
            base_col = col[:-len(f"_{site_suffix}")]
            column_mapping[col] = base_col
        else:
            column_mapping[col] = col
    
    # Rename columns
    df_normalized = df_normalized.rename(columns=column_mapping)
    
    return df_normalized, column_mapping


def load_site_data(site_name, target_column='PM2.5'):
    """Load data for a specific site"""
    try:
        return load_site_data_from_regional_inputs(site_name, target_column=target_column)
    except FileNotFoundError as exc:
        logging.info(f"{exc}; falling back to legacy per-site input directory")

    for filename in os.listdir(config.INPUT_DIRECTORY):
        if filename. lower().startswith(site_name.lower()) and filename.endswith('.csv'):
            filepath = os.path. join(config.INPUT_DIRECTORY, filename)
            df = pd.read_csv(filepath)
            
            # Handle different date/time column formats
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
            elif 'Date' in df.columns and 'Time' in df.columns:
                # Handle 24:00:00 time format by replacing with 00:00:00
                time_cleaned = df['Time'].str.replace('24:00:00', '00:00:00')
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + time_cleaned, errors='coerce')
            elif 'date' in df.columns and 'time' in df.columns:
                time_cleaned = df['time'].str.replace('24:00:00', '00:00:00')
                df['DateTime'] = pd.to_datetime(df['date'] + ' ' + time_cleaned, errors='coerce')
            else:
                raise ValueError(f"No date/time columns found in {filepath}")
            
            # Remove any rows with invalid dates
            df = df.dropna(subset=['DateTime'])
            
            # Normalize column names
            df, column_mapping = normalize_column_names(df, site_name)
            
            # Check for target column
            target_found = False
            
            # First try exact match
            if target_column in df.columns:
                target_found = True
            # For PM2.5, try PM10 as fallback
            elif target_column == 'PM2.5' and 'PM10' in df.columns:
                df[target_column] = df['PM10'].copy()
                target_found = True
            # Try other common variants
            elif target_column == 'PM2.5' and 'PM2_5' in df.columns:
                df[target_column] = df['PM2_5'].copy()
                target_found = True
            
            if target_found:
                return df, filepath, column_mapping
    
    raise FileNotFoundError(f"No data file found for site: {site_name}")


def build_region_feature_table(filepath, target_column="PM2.5", candidate_features=None):
    """Create one long feature table from one regional wide nowcasting CSV."""
    candidate_features = [
        feature for feature in (candidate_features or SIMPLE_XGBRF_FEATURES)
        if feature.upper() != target_column.upper()
    ]
    region_name = region_name_from_input_file(filepath)
    header = pd.read_csv(filepath, nrows=0)
    datetime_column = next(
        (col for col in header.columns if str(col).lower() in {"datetime", "date", "timestamp"}),
        None,
    )
    if datetime_column is None:
        raise ValueError(f"No datetime column found in {filepath}")

    site_columns = {}
    for column in header.columns:
        if column == datetime_column:
            continue
        variable, site = split_regional_column(column, target_column)
        if variable is None or site is None:
            continue
        site_key = normalize_site_token(site)
        site_columns.setdefault(site_key, {"site": canonical_site_name(site), "columns": {}})
        site_columns[site_key]["columns"][variable] = column

    use_columns = {datetime_column}
    for site_info in site_columns.values():
        for variable in [target_column] + list(candidate_features):
            source_column = site_info["columns"].get(variable)
            if source_column is not None:
                use_columns.add(source_column)

    region_df = pd.read_csv(filepath, usecols=sorted(use_columns))
    region_df[datetime_column] = pd.to_datetime(region_df[datetime_column], errors="coerce")

    records = []
    for site_info in site_columns.values():
        target_source = site_info["columns"].get(target_column)
        if target_source is None:
            continue

        site_name = site_info["site"]
        for feature in candidate_features:
            feature_source = site_info["columns"].get(feature)
            if feature_source is None:
                continue

            block = pd.DataFrame({
                "DateTime": region_df[datetime_column],
                "region": region_name,
                "site": site_name,
                "feature": feature,
                "target": pd.to_numeric(region_df[target_source], errors="coerce"),
                "feature_value": pd.to_numeric(region_df[feature_source], errors="coerce"),
            })
            block = block.dropna(subset=["DateTime", "target", "feature_value"])
            if not block.empty:
                records.append(block)

    if not records:
        return pd.DataFrame(columns=["DateTime", "region", "site", "feature", "target", "feature_value"])

    return pd.concat(records, ignore_index=True)


def build_region_site_wide_table(filepath, target_column="PM2.5", candidate_features=None):
    """
    Build a per-row (DateTime, site) wide table of variables for one regional input file.

    Output columns: DateTime, region, site, <variables...>
    """
    candidate_features = [
        feature for feature in (candidate_features or SIMPLE_XGBRF_FEATURES)
        if feature.upper() != target_column.upper()
    ]
    region_name = region_name_from_input_file(filepath)
    header = pd.read_csv(filepath, nrows=0)
    datetime_column = next(
        (col for col in header.columns if str(col).lower() in {"datetime", "date", "timestamp"}),
        None,
    )
    if datetime_column is None:
        raise ValueError(f"No datetime column found in {filepath}")

    site_columns = {}
    for column in header.columns:
        if column == datetime_column:
            continue
        variable, site = split_regional_column(column, target_column)
        if variable is None or site is None:
            continue
        site_key = normalize_site_token(site)
        site_columns.setdefault(site_key, {"site": canonical_site_name(site), "columns": {}})
        site_columns[site_key]["columns"][variable] = column

    use_columns = {datetime_column}
    variables = [target_column] + list(candidate_features)
    for site_info in site_columns.values():
        for variable in variables:
            source_column = site_info["columns"].get(variable)
            if source_column is not None:
                use_columns.add(source_column)

    region_df = pd.read_csv(filepath, usecols=sorted(use_columns))
    region_df[datetime_column] = pd.to_datetime(region_df[datetime_column], errors="coerce")
    region_df = region_df.dropna(subset=[datetime_column])

    records = []
    for site_info in site_columns.values():
        site_name = site_info["site"]
        if target_column not in site_info["columns"]:
            continue
        block = pd.DataFrame(
            {
                "DateTime": region_df[datetime_column],
                "region": region_name,
                "site": site_name,
            }
        )
        for variable in variables:
            source_column = site_info["columns"].get(variable)
            if source_column is None:
                block[variable] = np.nan
            else:
                block[variable] = pd.to_numeric(region_df[source_column], errors="coerce")

        # Keep rows with at least two non-null values (so correlations can be computed)
        value_cols = [col for col in variables if col in block.columns]
        block = block.dropna(subset=value_cols, how="all")
        records.append(block)

    if not records:
        cols = ["DateTime", "region", "site"] + variables
        return pd.DataFrame(columns=cols)

    return pd.concat(records, ignore_index=True)


def plot_region_correlograms_for_inputs(
    regional_input_files_list,
    target_column="PM2.5",
    candidate_features=None,
    output_dir=BASICS_DIR,
    corr_method="pearson",
    min_periods=30,
):
    """
    For each regional input file, create correlation plots to help decide which variables to include.

    Produces:
    - Correlogram heatmap (lower triangle) for variables
    - Target-correlation bar chart (signed)
    - Correlation matrix CSV
    """
    candidate_features = [
        feature for feature in (candidate_features or SIMPLE_XGBRF_FEATURES)
        if feature.upper() != target_column.upper()
    ]
    variables = [target_column] + list(candidate_features)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dirs = {
        "Missingness": ensure_output_subdir(output_dir, "Missingness"),
        "MissingnessTime": ensure_output_subdir(output_dir, "MissingnessTime"),
        "MissingnessCorr": ensure_output_subdir(output_dir, "MissingnessCorr"),
        "Distributions": ensure_output_subdir(output_dir, "Distributions"),
        "TargetVsFeatures": ensure_output_subdir(output_dir, "TargetVsFeatures"),
        "LagCorr": ensure_output_subdir(output_dir, "LagCorr"),
        "ACF": ensure_output_subdir(output_dir, "ACF"),
        "Diurnal": ensure_output_subdir(output_dir, "Diurnal"),
        "Weekday": ensure_output_subdir(output_dir, "Weekday"),
        "Correlogram": ensure_output_subdir(output_dir, "Correlogram"),
        "TargetCorr": ensure_output_subdir(output_dir, "TargetCorr"),
        "RedundancyAbsCorr": ensure_output_subdir(output_dir, "RedundancyAbsCorr"),
        "SiteTargetCorr": ensure_output_subdir(output_dir, "SiteTargetCorr"),
        "WindCondition": ensure_output_subdir(output_dir, "WindCondition"),
    }

    for filepath in regional_input_files_list:
        region_name = region_name_from_input_file(filepath)
        region_token = safe_target_name(region_name)
        try:
            wide_df = build_region_site_wide_table(
                filepath,
                target_column=target_column,
                candidate_features=candidate_features,
            )
        except Exception:
            logging.exception(f"Failed building wide table for correlogram: {filepath}")
            continue

        if wide_df.empty:
            logging.warning(f"No wide data available for correlogram in {filepath}")
            continue

        value_df = wide_df[variables].copy()
        # Drop columns with too few non-missing observations
        keep_cols = []
        for col in value_df.columns:
            if int(value_df[col].notna().sum()) >= int(min_periods):
                keep_cols.append(col)
        value_df = value_df[keep_cols]

        if value_df.shape[1] < 2:
            logging.warning(f"Not enough variables for correlogram in region {region_name} after filtering.")
            continue

        corr = value_df.corr(method=corr_method, min_periods=min_periods)
        corr_csv = output_dir / f"Correlogram_{region_token}_{safe_target_name(target_column)}_{corr_method}.csv"
        corr.to_csv(corr_csv)

        # Correlogram (lower triangle)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig_w = max(8, 0.55 * corr.shape[1] + 4)
        fig_h = max(7, 0.55 * corr.shape[0] + 3)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": f"{corr_method.title()} correlation"},
            ax=ax,
        )
        ax.set_title(f"{region_name}: variable correlogram ({corr_method}) | Target: {target_column}")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        for label in ax.get_yticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment("right")
        fig.subplots_adjust(left=0.25, right=0.96, top=0.90, bottom=0.25)
        corr_plot = output_dir / f"Correlogram_{region_token}_{safe_target_name(target_column)}_{corr_method}.png"
        plt.savefig(corr_plot, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Region correlogram saved: {corr_plot}")

        # Target correlation bars
        if target_column in corr.columns and corr.shape[0] > 1:
            target_corr = corr[target_column].drop(labels=[target_column], errors="ignore").dropna()
            if not target_corr.empty:
                target_corr = target_corr.sort_values()
                fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(target_corr) + 2)))
                colors = ["#d62728" if val < 0 else "#1f77b4" for val in target_corr.values]
                ax.barh(target_corr.index.tolist(), target_corr.values.tolist(), color=colors, edgecolor="black")
                ax.axvline(0, color="black", linewidth=1)
                ax.set_xlabel(f"{corr_method.title()} correlation with {target_column}")
                ax.set_title(f"{region_name}: correlation with target ({target_column})")
                ax.tick_params(axis="y", rotation=0)
                for label in ax.get_yticklabels():
                    label.set_rotation(0)
                    label.set_horizontalalignment("right")
                fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.12)
                target_plot = output_dir / f"TargetCorr_{region_token}_{safe_target_name(target_column)}_{corr_method}.png"
                plt.savefig(target_plot, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Region target-correlation plot saved: {target_plot}")


def plot_region_basics_for_inputs(
    regional_input_files_list,
    target_column="PM2.5",
    candidate_features=None,
    output_dir=BASICS_DIR,
    corr_method="pearson",
    min_periods=30,
    max_lag_hours=72,
    time_freq="7D",
):
    """
    Create a suite of "Basics" plots per region to understand relationships and trends.

    Each plot type is saved as a separate figure in output_dir.
    """
    candidate_features = [
        feature for feature in (candidate_features or SIMPLE_XGBRF_FEATURES)
        if feature.upper() != target_column.upper()
    ]
    variables = [target_column] + list(candidate_features)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dirs = {
        "Missingness": ensure_output_subdir(output_dir, "Missingness"),
        "MissingnessTime": ensure_output_subdir(output_dir, "MissingnessTime"),
        "MissingnessCorr": ensure_output_subdir(output_dir, "MissingnessCorr"),
        "Distributions": ensure_output_subdir(output_dir, "Distributions"),
        "TargetVsFeatures": ensure_output_subdir(output_dir, "TargetVsFeatures"),
        "LagCorr": ensure_output_subdir(output_dir, "LagCorr"),
        "ACF": ensure_output_subdir(output_dir, "ACF"),
        "Diurnal": ensure_output_subdir(output_dir, "Diurnal"),
        "Weekday": ensure_output_subdir(output_dir, "Weekday"),
        "Correlogram": ensure_output_subdir(output_dir, "Correlogram"),
        "TargetCorr": ensure_output_subdir(output_dir, "TargetCorr"),
        "RedundancyAbsCorr": ensure_output_subdir(output_dir, "RedundancyAbsCorr"),
        "SiteTargetCorr": ensure_output_subdir(output_dir, "SiteTargetCorr"),
        "WindCondition": ensure_output_subdir(output_dir, "WindCondition"),
    }

    for filepath in regional_input_files_list:
        region_name = region_name_from_input_file(filepath)
        region_token = safe_target_name(region_name)

        try:
            wide_df = build_region_site_wide_table(
                filepath,
                target_column=target_column,
                candidate_features=candidate_features,
            )
        except Exception:
            logging.exception(f"Failed building wide table for basics: {filepath}")
            continue

        if wide_df.empty:
            logging.warning(f"No wide data available for basics in {filepath}")
            continue

        wide_df = wide_df.dropna(subset=["DateTime"])
        value_df = wide_df[variables].copy()

        # Filter columns with enough observations
        keep_cols = []
        for col in value_df.columns:
            if int(value_df[col].notna().sum()) >= int(min_periods):
                keep_cols.append(col)
        value_df = value_df[keep_cols]

        if target_column not in value_df.columns:
            logging.warning(f"Target {target_column} not found for basics plots in region {region_name}")
            continue

        # Compute correlation matrix once (used by multiple plots below)
        try:
            corr = value_df.corr(method=corr_method, min_periods=min_periods)
        except Exception:
            corr = pd.DataFrame()

        # ------------------------------------------------------------------
        # 1) Missingness summary (bar)
        # ------------------------------------------------------------------
        miss_frac = value_df.isna().mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(miss_frac) + 2)))
        ax.barh(miss_frac.index.tolist(), (miss_frac.values * 100.0).tolist(), color="#6c757d", edgecolor="black")
        ax.set_xlabel("Missing (%)")
        ax.set_title(f"{region_name}: missingness by variable")
        ax.tick_params(axis="y", rotation=0)
        for label in ax.get_yticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment("right")
        fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.12)
        plot_path = plot_dirs["Missingness"] / f"Basics_Missingness_{region_token}_{safe_target_name(target_column)}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Basics plot saved: {plot_path}")

        # ------------------------------------------------------------------
        # 2) Missingness over time (heatmap)
        # ------------------------------------------------------------------
        try:
            tmp = wide_df[["DateTime"]].copy()
            tmp["bucket"] = pd.to_datetime(tmp["DateTime"], errors="coerce")
            tmp = tmp.dropna(subset=["bucket"])
            miss_time = {}
            for col in value_df.columns:
                miss_series = value_df[col].isna().to_numpy(dtype=float)
                miss_time[col] = pd.Series(miss_series, index=tmp["bucket"])
            miss_time_df = pd.DataFrame(miss_time)
            miss_time_df = miss_time_df.resample(time_freq).mean()
            if not miss_time_df.empty:
                fig_w = max(10, 0.18 * miss_time_df.shape[0] + 6)
                fig_h = max(6, 0.40 * miss_time_df.shape[1] + 3)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                sns.heatmap(
                    miss_time_df.T * 100.0,
                    cmap="Reds",
                    vmin=0,
                    vmax=100,
                    cbar_kws={"label": "Missing (%)"},
                    ax=ax,
                )
                ax.set_title(f"{region_name}: missingness over time ({time_freq})")
                ax.set_xlabel("Time bucket")
                ax.set_ylabel("Variable")
                ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="y", rotation=0)
                for label in ax.get_yticklabels():
                    label.set_rotation(0)
                    label.set_horizontalalignment("right")
                fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.20)
                plot_path = plot_dirs["MissingnessTime"] / f"Basics_MissingnessTime_{region_token}_{safe_target_name(target_column)}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed missingness-over-time plot for region {region_name}")

        # ------------------------------------------------------------------
        # 2b) Missingness correlation (do variables go missing together?)
        # ------------------------------------------------------------------
        try:
            miss_ind = value_df.isna().astype(float)
            if miss_ind.shape[1] >= 2:
                miss_corr = miss_ind.corr(method="pearson", min_periods=min_periods)
                fig_w = max(8, 0.55 * miss_corr.shape[1] + 4)
                fig_h = max(7, 0.55 * miss_corr.shape[0] + 3)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                sns.heatmap(
                    miss_corr,
                    cmap="coolwarm",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Correlation of missingness indicators"},
                    ax=ax,
                )
                ax.set_title(f"{region_name}: missingness correlation")
                ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="y", rotation=0)
                for label in ax.get_yticklabels():
                    label.set_rotation(0)
                    label.set_horizontalalignment("right")
                fig.subplots_adjust(left=0.25, right=0.96, top=0.90, bottom=0.25)
                plot_path = plot_dirs["MissingnessCorr"] / f"Basics_MissingnessCorr_{region_token}_{safe_target_name(target_column)}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed missingness-correlation plot for region {region_name}")

        # ------------------------------------------------------------------
        # 3) Distributions (boxplot)
        # ------------------------------------------------------------------
        try:
            melt = value_df.melt(var_name="variable", value_name="value").dropna()
            if not melt.empty:
                fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(value_df.columns) + 2)))
                sns.boxplot(data=melt, y="variable", x="value", ax=ax, color="#4c78a8", fliersize=1)
                ax.set_title(f"{region_name}: variable distributions (boxplot)")
                ax.set_xlabel("Value")
                ax.set_ylabel("Variable")
                ax.tick_params(axis="y", rotation=0)
                for label in ax.get_yticklabels():
                    label.set_rotation(0)
                    label.set_horizontalalignment("right")
                fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.10)
                plot_path = plot_dirs["Distributions"] / f"Basics_Distributions_{region_token}_{safe_target_name(target_column)}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed distributions plot for region {region_name}")

        # ------------------------------------------------------------------
        # 4) Target vs feature (binned mean trend) - all features in one grid
        # ------------------------------------------------------------------
        try:
            feature_cols = [c for c in value_df.columns if c != target_column]
            if feature_cols:
                n_cols = 4
                n_rows = int(np.ceil(len(feature_cols) / float(n_cols)))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.4 * n_rows))
                axes = np.atleast_1d(axes).ravel()

                y = value_df[target_column]
                for idx, feature in enumerate(feature_cols):
                    ax = axes[idx]
                    x = value_df[feature]
                    df_pair = pd.DataFrame({"x": x, "y": y}).dropna()
                    if len(df_pair) < min_periods:
                        ax.set_axis_off()
                        continue
                    # Quantile bins + mean target per bin
                    try:
                        df_pair["bin"] = pd.qcut(df_pair["x"], q=20, duplicates="drop")
                        grouped = (
                            df_pair.groupby("bin", observed=False)
                            .agg(x_mid=("x", "median"), y_mean=("y", "mean"))
                            .dropna()
                        )
                        ax.plot(grouped["x_mid"].values, grouped["y_mean"].values, linewidth=1.5)
                    except Exception:
                        # Fallback to simple scatter
                        ax.scatter(df_pair["x"].values, df_pair["y"].values, s=5, alpha=0.15)
                    ax.set_title(feature)
                    ax.set_xlabel(feature)
                    ax.set_ylabel(target_column)

                for j in range(len(feature_cols), len(axes)):
                    axes[j].set_axis_off()

                fig.suptitle(f"{region_name}: {target_column} vs features (binned mean trends)", y=0.995)
                fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06, hspace=0.40, wspace=0.25)
                plot_path = plot_dirs["TargetVsFeatures"] / f"Basics_TargetVsFeatures_{region_token}_{safe_target_name(target_column)}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed target-vs-features plot for region {region_name}")

        # ------------------------------------------------------------------
        # 5) Lag relationships (cross-correlation vs lag heatmap)
        # ------------------------------------------------------------------
        try:
            feature_cols = [c for c in value_df.columns if c != target_column]
            if feature_cols:
                lags = list(range(0, int(max_lag_hours) + 1))
                rows = []
                for feature in feature_cols:
                    series_x = value_df[feature]
                    series_y = value_df[target_column]
                    corr_vals = []
                    for lag in lags:
                        corr_vals.append(series_y.corr(series_x.shift(lag)))
                    rows.append(pd.Series(corr_vals, index=lags, name=feature))
                lag_corr = pd.DataFrame(rows)
                if not lag_corr.empty:
                    fig_w = max(10, 0.12 * len(lags) + 6)
                    fig_h = max(6, 0.35 * len(feature_cols) + 3)
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    sns.heatmap(
                        lag_corr,
                        cmap="coolwarm",
                        center=0,
                        vmin=-1,
                        vmax=1,
                        cbar_kws={"label": f"{corr_method.title()} corr({target_column}, feature shifted by lag)"},
                        ax=ax,
                    )
                    ax.set_title(f"{region_name}: lag cross-correlation (0–{max_lag_hours}h)")
                    ax.set_xlabel("Lag (hours)")
                    ax.set_ylabel("Feature")
                    ax.tick_params(axis="x", rotation=0)
                    ax.tick_params(axis="y", rotation=0)
                    for label in ax.get_yticklabels():
                        label.set_rotation(0)
                        label.set_horizontalalignment("right")
                    fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.10)
                    plot_path = plot_dirs["LagCorr"] / f"Basics_LagCorr_{region_token}_{safe_target_name(target_column)}.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed lag-correlation plot for region {region_name}")

        # ------------------------------------------------------------------
        # 6) Target autocorrelation + diurnal profile
        # ------------------------------------------------------------------
        try:
            ts = wide_df[["DateTime", target_column]].dropna()
            ts = ts.sort_values("DateTime")
            ts = ts.set_index("DateTime")[target_column]
            if len(ts) >= min_periods:
                max_lag = min(int(max_lag_hours), 168)
                acf_vals = [ts.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
                fig, ax = plt.subplots(figsize=(12, 4.5))
                ax.bar(range(1, max_lag + 1), acf_vals, color="#4c78a8", edgecolor="black")
                ax.axhline(0, color="black", linewidth=1)
                ax.set_xlabel("Lag (hours)")
                ax.set_ylabel("Autocorrelation")
                ax.set_title(f"{region_name}: {target_column} autocorrelation (ACF)")
                fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15)
                plot_path = plot_dirs["ACF"] / f"Basics_ACF_{region_token}_{safe_target_name(target_column)}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Basics plot saved: {plot_path}")

            # Diurnal profile
            dt = wide_df[["DateTime", target_column]].dropna()
            if not dt.empty:
                dt = dt.assign(hour=pd.to_datetime(dt["DateTime"]).dt.hour)
                prof = dt.groupby("hour")[target_column].mean()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(prof.index.values, prof.values, linewidth=2)
                ax.set_xlabel("Hour of day")
                ax.set_ylabel(target_column)
                ax.set_title(f"{region_name}: diurnal profile of {target_column}")
                ax.set_xticks(list(range(0, 24, 2)))
                fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)
                plot_path = plot_dirs["Diurnal"] / f"Basics_Diurnal_{region_token}_{safe_target_name(target_column)}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Basics plot saved: {plot_path}")

            # Diurnal profile by month with per-site lines + region mean (12 subplots)
            if "site" in wide_df.columns:
                dtm = wide_df[["DateTime", "site", target_column]].dropna()
                if not dtm.empty:
                    dtm = dtm.assign(
                        month=pd.to_datetime(dtm["DateTime"]).dt.month,
                        hour=pd.to_datetime(dtm["DateTime"]).dt.hour,
                    )
                    # Mean per site, per month, per hour
                    site_month_hour = (
                        dtm.groupby(["month", "site", "hour"])[target_column]
                        .mean()
                        .reset_index()
                    )
                    region_month_hour = (
                        dtm.groupby(["month", "hour"])[target_column]
                        .mean()
                        .reset_index()
                    )

                    if not site_month_hour.empty:
                        months_present = sorted(site_month_hour["month"].unique().tolist())
                        if months_present:
                            fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True)
                            axes = axes.ravel()

                            # Determine common y-limits across all months/sites for consistent scaling
                            y_values = site_month_hour[target_column].to_numpy(dtype=float)
                            if y_values.size:
                                y_min = float(np.nanmin(y_values))
                                y_max = float(np.nanmax(y_values))
                            else:
                                y_min, y_max = 0.0, 1.0
                            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                                y_min, y_max = 0.0, 1.0

                            # Use deterministic colour mapping per site
                            sites = sorted(site_month_hour["site"].unique().tolist())
                            palette = sns.color_palette("tab20", n_colors=max(1, min(20, len(sites))))
                            site_to_color = {site: palette[i % len(palette)] for i, site in enumerate(sites)}

                            for m in range(1, 13):
                                ax = axes[m - 1]
                                month_site = site_month_hour[site_month_hour["month"] == m]
                                month_region = region_month_hour[region_month_hour["month"] == m]

                                if month_site.empty:
                                    ax.set_axis_off()
                                    continue

                                for site_name in sites:
                                    site_block = month_site[month_site["site"] == site_name]
                                    if site_block.empty:
                                        continue
                                    ax.plot(
                                        site_block["hour"].values,
                                        site_block[target_column].values,
                                        color=site_to_color.get(site_name),
                                        linewidth=1.0,
                                        alpha=0.55,
                                    )

                                # Region mean line (thicker)
                                if not month_region.empty:
                                    ax.plot(
                                        month_region["hour"].values,
                                        month_region[target_column].values,
                                        color="black",
                                        linewidth=2.2,
                                        alpha=0.9,
                                    )

                                ax.set_title(calendar.month_abbr[m].upper(), fontsize=11)
                                ax.set_xlim(0, 23)
                                ax.set_ylim(y_min, y_max)
                                ax.set_xticks(list(range(0, 24, 4)))

                            # Common labels
                            fig.suptitle(
                                f"{region_name}: diurnal pattern by month (per-site + region mean)\nTarget: {target_column}",
                                y=0.995,
                                fontsize=14,
                            )
                            fig.text(0.5, 0.04, "Hour of day", ha="center")
                            fig.text(0.04, 0.5, target_column, va="center", rotation="vertical")

                            # Legend: keep minimal to avoid clutter
                            # Always show region mean label, optionally site labels if few sites
                            handles = [matplotlib.lines.Line2D([0], [0], color="black", linewidth=2.2, label="Region mean")]
                            if len(sites) <= 8:
                                for site_name in sites:
                                    handles.append(
                                        matplotlib.lines.Line2D([0], [0], color=site_to_color[site_name], linewidth=1.5, label=str(site_name))
                                    )
                            fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.01))

                            fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, hspace=0.35, wspace=0.18)
                            plot_path = plot_dirs["Diurnal"] / f"Basics_DiurnalByMonth_BySite_{region_token}_{safe_target_name(target_column)}.png"
                            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                            plt.close()
                            logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed ACF/diurnal plots for region {region_name}")

        # ------------------------------------------------------------------
        # 6b) Weekday profile of target
        # ------------------------------------------------------------------
        try:
            dt = wide_df[["DateTime", target_column]].dropna()
            if not dt.empty:
                dt = dt.assign(dow=pd.to_datetime(dt["DateTime"]).dt.dayofweek)
                prof = dt.groupby("dow")[target_column].mean()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(prof.index.values, prof.values, linewidth=2)
                ax.set_xlabel("Day of week (0=Mon)")
                ax.set_ylabel(target_column)
                ax.set_title(f"{region_name}: weekday profile of {target_column}")
                ax.set_xticks(list(range(0, 7)))
                fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)
                plot_path = plot_dirs["Weekday"] / f"Basics_Weekday_{region_token}_{safe_target_name(target_column)}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Basics plot saved: {plot_path}")

                # Weekday vs weekend diurnal by month (12 subplots, common axes)
                dtt_cols = ["DateTime", target_column] + (["site"] if "site" in wide_df.columns else [])
                dtt = wide_df[dtt_cols].dropna()
                if not dtt.empty:
                    dtt = dtt.assign(
                        month=pd.to_datetime(dtt["DateTime"]).dt.month,
                        hour=pd.to_datetime(dtt["DateTime"]).dt.hour,
                        is_weekend=(pd.to_datetime(dtt["DateTime"]).dt.dayofweek >= 5),
                    )

                    region_month_hour_week = (
                        dtt.groupby(["month", "is_weekend", "hour"])[target_column]
                        .mean()
                        .reset_index()
                    )
                    site_month_hour_week = pd.DataFrame()
                    if "site" in dtt.columns:
                        site_month_hour_week = (
                            dtt.groupby(["month", "is_weekend", "site", "hour"])[target_column]
                            .mean()
                            .reset_index()
                        )

                    if not region_month_hour_week.empty:
                        y_vals = []
                        y_vals.append(region_month_hour_week[target_column].to_numpy(dtype=float))
                        if not site_month_hour_week.empty:
                            y_vals.append(site_month_hour_week[target_column].to_numpy(dtype=float))
                        y_vals = np.concatenate([arr for arr in y_vals if arr.size]) if y_vals else np.asarray([])

                        y_min = float(np.nanmin(y_vals)) if y_vals.size else 0.0
                        y_max = float(np.nanmax(y_vals)) if y_vals.size else 1.0
                        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                            y_min, y_max = 0.0, 1.0

                        sites = sorted(site_month_hour_week["site"].unique().tolist()) if not site_month_hour_week.empty else []
                        n_colors = max(3, min(20, len(sites))) if sites else 3
                        weekday_palette = sns.color_palette("Blues", n_colors=n_colors + 2)[2:]
                        weekend_palette = sns.color_palette("Reds", n_colors=n_colors + 2)[2:]
                        site_to_idx = {site: i % n_colors for i, site in enumerate(sites)}

                        fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True)
                        axes = axes.ravel()

                        for m in range(1, 13):
                            axm = axes[m - 1]
                            m_region = region_month_hour_week[region_month_hour_week["month"] == m]
                            if m_region.empty:
                                axm.set_axis_off()
                                continue

                            # Per-site lines (same blue family for weekdays, red family for weekends)
                            if sites:
                                m_site = site_month_hour_week[site_month_hour_week["month"] == m]
                                if not m_site.empty:
                                    for site_name in sites:
                                        s_block = m_site[m_site["site"] == site_name]
                                        if s_block.empty:
                                            continue
                                        wk = s_block[s_block["is_weekend"] == False]
                                        we = s_block[s_block["is_weekend"] == True]
                                        idx = site_to_idx.get(site_name, 0)
                                        if not wk.empty:
                                            axm.plot(
                                                wk["hour"].values,
                                                wk[target_column].values,
                                                color=weekday_palette[idx],
                                                linewidth=1.1,
                                                alpha=0.35,
                                            )
                                        if not we.empty:
                                            axm.plot(
                                                we["hour"].values,
                                                we[target_column].values,
                                                color=weekend_palette[idx],
                                                linewidth=1.1,
                                                alpha=0.35,
                                            )

                            # Region mean lines (thicker)
                            wk_mean = m_region[m_region["is_weekend"] == False]
                            we_mean = m_region[m_region["is_weekend"] == True]
                            if not wk_mean.empty:
                                axm.plot(
                                    wk_mean["hour"].values,
                                    wk_mean[target_column].values,
                                    color="#08306B",
                                    linewidth=2.4,
                                    alpha=0.95,
                                )
                            if not we_mean.empty:
                                axm.plot(
                                    we_mean["hour"].values,
                                    we_mean[target_column].values,
                                    color="#67000D",
                                    linewidth=2.4,
                                    alpha=0.95,
                                )

                            axm.set_title(calendar.month_abbr[m].upper(), fontsize=11)
                            axm.set_xlim(0, 23)
                            axm.set_ylim(y_min, y_max)
                            axm.set_xticks(list(range(0, 24, 4)))

                        fig.suptitle(
                            f"{region_name}: diurnal by month (weekday vs weekend)\nTarget: {target_column}",
                            y=0.995,
                            fontsize=14,
                        )
                        fig.text(0.5, 0.04, "Hour of day", ha="center")
                        fig.text(0.04, 0.5, target_column, va="center", rotation="vertical")
                        handles = [
                            matplotlib.lines.Line2D([0], [0], color="#08306B", linewidth=2.4, label="Region mean (Weekday)"),
                            matplotlib.lines.Line2D([0], [0], color="#67000D", linewidth=2.4, label="Region mean (Weekend)"),
                            matplotlib.lines.Line2D([0], [0], color=weekday_palette[-1], linewidth=1.5, alpha=0.35, label="Sites (Weekday)"),
                            matplotlib.lines.Line2D([0], [0], color=weekend_palette[-1], linewidth=1.5, alpha=0.35, label="Sites (Weekend)"),
                        ]
                        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
                        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, hspace=0.35, wspace=0.18)
                        plot_path = plot_dirs["Weekday"] / f"Basics_WeekdayWeekend_DiurnalByMonth_{region_token}_{safe_target_name(target_column)}.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close()
                        logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed weekday plot for region {region_name}")

        # ------------------------------------------------------------------
        # 7) Correlogram (signed) + target-correlation bars + redundancy view
        # ------------------------------------------------------------------
        try:
            if corr.shape[0] >= 2:
                corr_csv = plot_dirs["Correlogram"] / f"Basics_Correlogram_{region_token}_{safe_target_name(target_column)}_{corr_method}.csv"
                corr.to_csv(corr_csv)

                # Signed correlogram (lower triangle)
                mask = np.triu(np.ones_like(corr, dtype=bool))
                fig_w = max(8, 0.55 * corr.shape[1] + 4)
                fig_h = max(7, 0.55 * corr.shape[0] + 3)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                sns.heatmap(
                    corr,
                    mask=mask,
                    cmap="coolwarm",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": f"{corr_method.title()} correlation"},
                    ax=ax,
                )
                ax.set_title(f"{region_name}: correlogram ({corr_method}) | Target: {target_column}")
                ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="y", rotation=0)
                for label in ax.get_yticklabels():
                    label.set_rotation(0)
                    label.set_horizontalalignment("right")
                fig.subplots_adjust(left=0.25, right=0.96, top=0.90, bottom=0.25)
                plot_path = plot_dirs["Correlogram"] / f"Basics_Correlogram_{region_token}_{safe_target_name(target_column)}_{corr_method}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Basics plot saved: {plot_path}")

                # Target-correlation bars
                if target_column in corr.columns:
                    target_corr = corr[target_column].drop(labels=[target_column], errors="ignore").dropna()
                    if not target_corr.empty:
                        target_corr = target_corr.sort_values()
                        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(target_corr) + 2)))
                        colors = ["#d62728" if val < 0 else "#1f77b4" for val in target_corr.values]
                        ax.barh(target_corr.index.tolist(), target_corr.values.tolist(), color=colors, edgecolor="black")
                        ax.axvline(0, color="black", linewidth=1)
                        ax.set_xlabel(f"{corr_method.title()} correlation with {target_column}")
                        ax.set_title(f"{region_name}: correlation with target ({target_column})")
                        ax.tick_params(axis="y", rotation=0)
                        for label in ax.get_yticklabels():
                            label.set_rotation(0)
                            label.set_horizontalalignment("right")
                        fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.12)
                        plot_path = plot_dirs["TargetCorr"] / f"Basics_TargetCorr_{region_token}_{safe_target_name(target_column)}_{corr_method}.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close()
                        logging.info(f"Basics plot saved: {plot_path}")

                # Redundancy: absolute correlation heatmap (ordered)
                abs_corr = corr.abs()
                order = abs_corr.sum(axis=1).sort_values(ascending=False).index.tolist()
                abs_corr = abs_corr.loc[order, order]
                fig_w = max(8, 0.55 * abs_corr.shape[1] + 4)
                fig_h = max(7, 0.55 * abs_corr.shape[0] + 3)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                sns.heatmap(
                    abs_corr,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": f"|{corr_method.title()} correlation|"},
                    ax=ax,
                )
                ax.set_title(f"{region_name}: redundancy view (absolute correlation)")
                ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="y", rotation=0)
                for label in ax.get_yticklabels():
                    label.set_rotation(0)
                    label.set_horizontalalignment("right")
                fig.subplots_adjust(left=0.25, right=0.96, top=0.90, bottom=0.25)
                plot_path = plot_dirs["RedundancyAbsCorr"] / f"Basics_RedundancyAbsCorr_{region_token}_{safe_target_name(target_column)}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed redundancy plot for region {region_name}")

        # ------------------------------------------------------------------
        # 8) Stability across sites: target correlation per site (heatmap)
        # ------------------------------------------------------------------
        try:
            if "site" in wide_df.columns:
                feature_cols = [c for c in value_df.columns if c != target_column]
                rows = []
                for site_name, site_block in wide_df.groupby("site"):
                    block_vals = site_block[variables].copy()
                    if target_column not in block_vals.columns:
                        continue
                    site_corr = {}
                    for feature in feature_cols:
                        df_pair = block_vals[[target_column, feature]].dropna()
                        site_corr[feature] = df_pair[target_column].corr(df_pair[feature]) if len(df_pair) >= min_periods else np.nan
                    rows.append(pd.Series(site_corr, name=str(site_name)))
                site_corr_df = pd.DataFrame(rows)
                if not site_corr_df.empty:
                    fig_w = max(10, 0.75 * site_corr_df.shape[1] + 5)
                    fig_h = max(6, 0.45 * site_corr_df.shape[0] + 3)
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    sns.heatmap(
                        site_corr_df,
                        cmap="coolwarm",
                        center=0,
                        vmin=-1,
                        vmax=1,
                        cbar_kws={"label": f"{corr_method.title()} corr(feature, {target_column})"},
                        ax=ax,
                    )
                    ax.set_title(f"{region_name}: per-site correlation with target")
                    ax.set_xlabel("Feature")
                    ax.set_ylabel("Site")
                    ax.tick_params(axis="x", rotation=45)
                    ax.tick_params(axis="y", rotation=0)
                    for label in ax.get_yticklabels():
                        label.set_rotation(0)
                        label.set_horizontalalignment("right")
                    fig.subplots_adjust(left=0.25, right=0.96, top=0.90, bottom=0.25)
                    plot_path = plot_dirs["SiteTargetCorr"] / f"Basics_SiteTargetCorr_{region_token}_{safe_target_name(target_column)}.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed per-site correlation plot for region {region_name}")

        # ------------------------------------------------------------------
        # 9) Wind-conditioned summary (if WDR/WSP exist)
        # ------------------------------------------------------------------
        try:
            if "WDR" in value_df.columns and "WSP" in value_df.columns:
                dfw = wide_df[["WDR", "WSP", target_column]].dropna()
                if len(dfw) >= min_periods:
                    # Direction bins (16 sectors)
                    dfw = dfw.copy()
                    dfw["WDR"] = pd.to_numeric(dfw["WDR"], errors="coerce")
                    dfw["WSP"] = pd.to_numeric(dfw["WSP"], errors="coerce")
                    dfw = dfw.dropna(subset=["WDR", "WSP", target_column])
                    if not dfw.empty:
                        dir_bins = np.linspace(0, 360, 17)
                        dir_labels = [f"{int(dir_bins[i])}-{int(dir_bins[i+1])}" for i in range(len(dir_bins) - 1)]
                        dfw["dir_bin"] = pd.cut(dfw["WDR"] % 360.0, bins=dir_bins, labels=dir_labels, include_lowest=True)
                        wsp_bins = pd.qcut(dfw["WSP"], q=6, duplicates="drop")
                        table = dfw.pivot_table(
                            index="dir_bin",
                            columns=wsp_bins,
                            values=target_column,
                            aggfunc="mean",
                            observed=False,
                        )
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.heatmap(
                            table,
                            cmap="viridis_r",
                            cbar_kws={"label": f"Mean {target_column}"},
                            ax=ax,
                        )
                        ax.set_title(f"{region_name}: {target_column} conditioned on wind direction/speed")
                        ax.set_xlabel("Wind speed bin")
                        ax.set_ylabel("Wind direction (deg bins)")
                        ax.tick_params(axis="x", rotation=45)
                        ax.tick_params(axis="y", rotation=0)
                        for label in ax.get_yticklabels():
                            label.set_rotation(0)
                            label.set_horizontalalignment("right")
                        fig.subplots_adjust(left=0.25, right=0.96, top=0.90, bottom=0.22)
                        plot_path = plot_dirs["WindCondition"] / f"Basics_WindCondition_{region_token}_{safe_target_name(target_column)}.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close()
                        logging.info(f"Basics plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed wind-conditioned plot for region {region_name}")


def make_holdout_mask_by_site(
    df,
    target_column,
    missing_frac=0.20,
    seed=42,
    min_rows_per_site=30,
    regime="random",
):
    """
    Create a boolean holdout mask for target evaluation, sampling within each site.
    Only rows with non-missing target are eligible.

    regimes:
      - random: random rows per site (default)
      - last_block: last contiguous block per site (time-ordered)
      - random_block: random contiguous block per site (time-ordered)
    """
    rng = np.random.RandomState(seed)
    eligible = df[target_column].notna()
    holdout_mask = pd.Series(False, index=df.index)

    if "site" not in df.columns:
        eligible_idx = df.index[eligible]
        if len(eligible_idx) < 2:
            return holdout_mask
        test_size = max(1, int(round(len(eligible_idx) * missing_frac)))
        test_size = min(test_size, max(1, len(eligible_idx) - 1))
        chosen = rng.choice(eligible_idx.to_numpy(), size=test_size, replace=False)
        holdout_mask.loc[chosen] = True
        return holdout_mask

    for _, site_df in df.loc[eligible].groupby("site"):
        if len(site_df) < min_rows_per_site:
            continue
        test_size = max(1, int(round(len(site_df) * missing_frac)))
        if test_size >= len(site_df):
            test_size = max(1, len(site_df) // 5)

        if regime == "random":
            chosen = rng.choice(site_df.index.to_numpy(), size=test_size, replace=False)
            holdout_mask.loc[chosen] = True
            continue

        # Block-based sampling needs a time order
        if "DateTime" in df.columns:
            site_df = site_df.sort_values("DateTime")

        if regime == "last_block":
            chosen = site_df.index.to_numpy()[-test_size:]
            holdout_mask.loc[chosen] = True
            continue

        if regime == "random_block":
            max_start = max(0, len(site_df) - test_size)
            start = int(rng.randint(0, max_start + 1)) if max_start > 0 else 0
            chosen = site_df.index.to_numpy()[start : start + test_size]
            holdout_mask.loc[chosen] = True
            continue

        raise ValueError(f"Unknown holdout regime: {regime}")
    return holdout_mask


def score_xgbrf_feature_set(
    wide_df,
    target_column,
    feature_list,
    holdout_mask,
    seed=42,
):
    """
    Train an XGBRF model using multiple predictors and evaluate on holdout_mask rows.
    """
    if wide_df.empty:
        return None

    feature_list = [feat for feat in feature_list if feat in wide_df.columns]
    if not feature_list:
        return None

    mask = holdout_mask.reindex(wide_df.index).fillna(False).astype(bool)
    train_mask = (~mask) & wide_df[target_column].notna()
    test_mask = mask & wide_df[target_column].notna()

    if int(train_mask.sum()) < 50 or int(test_mask.sum()) < 10:
        return None

    X_train = wide_df.loc[train_mask, feature_list]
    y_train = wide_df.loc[train_mask, target_column].astype(float)
    X_test = wide_df.loc[test_mask, feature_list]
    y_test = wide_df.loc[test_mask, target_column].astype(float)

    model = make_xgbrf_model(seed=seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = regression_metrics(y_test.to_numpy(dtype=float), predictions)

    return {
        "features": ",".join(feature_list),
        "n_features": int(len(feature_list)),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r": metrics["r"],
        "nse": metrics["nse"],
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_sites": int(wide_df["site"].nunique()) if "site" in wide_df.columns else np.nan,
    }


def run_region_feature_combinations(
    regional_input_files_list,
    single_feature_results_df,
    target_column="PM2.5",
    candidate_features=None,
    missing_frac=0.20,
    seed=42,
    max_features=10,
    output_dir=REGIONAL_ASSESSMENT_DIR,
):
    """
    For each region, build incremental feature combinations:
      best(1), best+2nd(2), best+2nd+3rd(3), ...
    and evaluate which combination gives best RMSE for prediction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    type_dir = ensure_output_subdir(output_dir, "FeatureCombinations")
    candidate_features = [
        feature for feature in (candidate_features or SIMPLE_XGBRF_FEATURES)
        if feature.upper() != target_column.upper()
    ]

    rows = []
    for filepath in regional_input_files_list:
        region_name = region_name_from_input_file(filepath)
        region_token = safe_target_name(region_name)

        region_rank = single_feature_results_df[
            (single_feature_results_df["scope"] == "per_region")
            & (single_feature_results_df["region"] == region_name)
        ].sort_values("rmse")

        if region_rank.empty:
            logging.warning(f"No single-feature ranking available for region {region_name}; skipping combinations.")
            continue

        ranked_features = [f for f in region_rank["feature"].tolist() if f in candidate_features]
        if not ranked_features:
            logging.warning(f"No ranked features available after filtering for region {region_name}; skipping.")
            continue

        ranked_features = ranked_features[: int(max_features)]

        try:
            wide_df = build_region_site_wide_table(
                filepath,
                target_column=target_column,
                candidate_features=candidate_features,
            )
        except Exception:
            logging.exception(f"Failed building wide table for combinations: {filepath}")
            continue

        if wide_df.empty:
            logging.warning(f"No wide data for combinations in region {region_name}")
            continue

        holdout_mask = make_holdout_mask_by_site(
            wide_df,
            target_column=target_column,
            missing_frac=missing_frac,
            seed=seed,
        )

        region_combo_rows = []
        for k in range(1, len(ranked_features) + 1):
            feats = ranked_features[:k]
            result = score_xgbrf_feature_set(
                wide_df,
                target_column=target_column,
                feature_list=feats,
                holdout_mask=holdout_mask,
                seed=seed,
            )
            if result is None:
                continue
            result.update(
                {
                    "region": region_name,
                    "target": target_column,
                    "ranked_features_used": ",".join(feats),
                }
            )
            region_combo_rows.append(result)
            rows.append(result)

        if region_combo_rows:
            region_combo_df = pd.DataFrame(region_combo_rows).sort_values("n_features")
            csv_path = type_dir / f"RegionFeatureCombos_{region_token}_{safe_target_name(target_column)}.csv"
            region_combo_df.to_csv(csv_path, index=False)
            logging.info(f"Region feature-combo table saved: {csv_path}")

            # Plot RMSE vs number of features
            try:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(region_combo_df["n_features"].values, region_combo_df["rmse"].values, linewidth=2)
                ax.set_xlabel("Number of top-ranked features included")
                ax.set_ylabel("RMSE")
                ax.set_title(f"{region_name}: incremental feature combination RMSE\nTarget: {target_column}")
                ax.grid(True, alpha=0.3)
                fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14)
                plot_path = type_dir / f"RegionFeatureCombos_{region_token}_{safe_target_name(target_column)}_rmse.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Region feature-combo plot saved: {plot_path}")
            except Exception:
                logging.exception(f"Failed plotting combo RMSE for region {region_name}")

    if not rows:
        return pd.DataFrame()

    combined_df = pd.DataFrame(rows)
    combined_path = type_dir / f"RegionFeatureCombos_ALL_{safe_target_name(target_column)}.csv"
    combined_df.to_csv(combined_path, index=False)
    logging.info(f"Combined region feature-combo table saved: {combined_path}")
    return combined_df


def run_region_feature_ablation(
    regional_input_files_list,
    target_column="PM2.5",
    candidate_features=None,
    missing_frac=0.20,
    seed=42,
    min_periods=30,
    output_dir=REGIONAL_ASSESSMENT_DIR,
):
    """
    Per region: train with all variables together (baseline), then remove one feature at a time.
    Saves delta RMSE (rmse_without - baseline) to rank feature impact.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    type_dir = ensure_output_subdir(output_dir, "Ablation")

    candidate_features = [
        feature for feature in (candidate_features or SIMPLE_XGBRF_FEATURES)
        if feature.upper() != target_column.upper()
    ]

    all_rows = []
    for filepath in regional_input_files_list:
        region_name = region_name_from_input_file(filepath)
        region_token = safe_target_name(region_name)

        try:
            wide_df = build_region_site_wide_table(
                filepath,
                target_column=target_column,
                candidate_features=candidate_features,
            )
        except Exception:
            logging.exception(f"Failed building wide table for ablation: {filepath}")
            continue

        if wide_df.empty:
            logging.warning(f"No wide data for ablation in region {region_name}")
            continue

        # Keep features that exist and have enough observations
        feature_pool = []
        for feat in candidate_features:
            if feat in wide_df.columns and int(wide_df[feat].notna().sum()) >= int(min_periods):
                feature_pool.append(feat)

        if len(feature_pool) < 2:
            logging.warning(f"Not enough usable features for ablation in region {region_name}")
            continue

        holdout_mask = make_holdout_mask_by_site(
            wide_df,
            target_column=target_column,
            missing_frac=missing_frac,
            seed=seed,
        )

        baseline = score_xgbrf_feature_set(
            wide_df,
            target_column=target_column,
            feature_list=feature_pool,
            holdout_mask=holdout_mask,
            seed=seed,
        )
        if baseline is None:
            logging.warning(f"Baseline ablation model failed for region {region_name}")
            continue

        baseline_rmse = baseline["rmse"]
        region_rows = []

        # Baseline row
        region_rows.append(
            {
                "region": region_name,
                "target": target_column,
                "feature_removed": "__BASELINE_ALL__",
                "n_features_used": int(len(feature_pool)),
                "rmse": baseline["rmse"],
                "mae": baseline["mae"],
                "r": baseline["r"],
                "nse": baseline["nse"],
                "delta_rmse": 0.0,
                "n_train": baseline["n_train"],
                "n_test": baseline["n_test"],
                "n_sites": baseline.get("n_sites", np.nan),
                "features_used": ",".join(feature_pool),
            }
        )

        for removed in feature_pool:
            feats = [f for f in feature_pool if f != removed]
            result = score_xgbrf_feature_set(
                wide_df,
                target_column=target_column,
                feature_list=feats,
                holdout_mask=holdout_mask,
                seed=seed,
            )
            if result is None:
                continue

            region_rows.append(
                {
                    "region": region_name,
                    "target": target_column,
                    "feature_removed": removed,
                    "n_features_used": int(len(feats)),
                    "rmse": result["rmse"],
                    "mae": result["mae"],
                    "r": result["r"],
                    "nse": result["nse"],
                    "delta_rmse": float(result["rmse"] - baseline_rmse),
                    "n_train": result["n_train"],
                    "n_test": result["n_test"],
                    "n_sites": result.get("n_sites", np.nan),
                    "features_used": ",".join(feats),
                }
            )

        if len(region_rows) <= 1:
            logging.warning(f"No ablation variants computed for region {region_name}")
            continue

        region_df = pd.DataFrame(region_rows)
        # Save table
        csv_path = type_dir / f"Ablation_{region_token}_{safe_target_name(target_column)}.csv"
        region_df.to_csv(csv_path, index=False)
        logging.info(f"Ablation table saved: {csv_path}")

        # Plot delta RMSE for removed features (exclude baseline)
        try:
            plot_df = region_df[region_df["feature_removed"] != "__BASELINE_ALL__"].copy()
            plot_df = plot_df.sort_values("delta_rmse", ascending=False)
            fig_h = max(4.5, 0.35 * len(plot_df) + 1.5)
            fig, ax = plt.subplots(figsize=(11, fig_h))
            colors = ["#d62728" if val > 0 else "#1f77b4" for val in plot_df["delta_rmse"].values]
            ax.barh(plot_df["feature_removed"].tolist(), plot_df["delta_rmse"].tolist(), color=colors, edgecolor="black")
            ax.axvline(0, color="black", linewidth=1)
            ax.set_xlabel("ΔRMSE = RMSE(without feature) − RMSE(all features)")
            ax.set_ylabel("Removed feature")
            ax.set_title(f"{region_name}: leave-one-out ablation impact\nTarget: {target_column} | Baseline RMSE={baseline_rmse:.3f}")
            ax.tick_params(axis="y", rotation=0)
            for label in ax.get_yticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment("right")
            fig.subplots_adjust(left=0.30, right=0.98, top=0.88, bottom=0.10)
            plot_path = type_dir / f"Ablation_{region_token}_{safe_target_name(target_column)}_delta_rmse.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"Ablation plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed ablation plot for region {region_name}")

        all_rows.extend(region_rows)

    if not all_rows:
        return pd.DataFrame()

    combined_df = pd.DataFrame(all_rows)
    combined_path = type_dir / f"Ablation_ALL_{safe_target_name(target_column)}.csv"
    combined_df.to_csv(combined_path, index=False)
    logging.info(f"Combined ablation table saved: {combined_path}")
    return combined_df


def run_region_permutation_importance(
    regional_input_files_list,
    target_column="PM2.5",
    candidate_features=None,
    missing_frac=0.20,
    seed=42,
    min_periods=30,
    n_repeats=5,
    holdout_regime="random",
    output_dir=REGIONAL_ASSESSMENT_DIR,
):
    """
    Per region: train one model with all features, then compute permutation importance on the holdout set.
    Importance = RMSE(permuted feature) - RMSE(baseline).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    type_dir = ensure_output_subdir(output_dir, "PermutationImportance")

    candidate_features = [
        feature for feature in (candidate_features or SIMPLE_XGBRF_FEATURES)
        if feature.upper() != target_column.upper()
    ]

    all_rows = []
    for filepath in regional_input_files_list:
        region_name = region_name_from_input_file(filepath)
        region_token = safe_target_name(region_name)

        try:
            wide_df = build_region_site_wide_table(
                filepath,
                target_column=target_column,
                candidate_features=candidate_features,
            )
        except Exception:
            logging.exception(f"Failed building wide table for permutation importance: {filepath}")
            continue

        if wide_df.empty:
            continue

        feature_pool = []
        for feat in candidate_features:
            if feat in wide_df.columns and int(wide_df[feat].notna().sum()) >= int(min_periods):
                feature_pool.append(feat)
        if len(feature_pool) < 2:
            continue

        holdout_mask = make_holdout_mask_by_site(
            wide_df,
            target_column=target_column,
            missing_frac=missing_frac,
            seed=seed,
            regime=holdout_regime,
        )
        mask = holdout_mask.reindex(wide_df.index).fillna(False).astype(bool)
        train_mask = (~mask) & wide_df[target_column].notna()
        test_mask = mask & wide_df[target_column].notna()
        if int(train_mask.sum()) < 50 or int(test_mask.sum()) < 10:
            continue

        X_train = wide_df.loc[train_mask, feature_pool]
        y_train = wide_df.loc[train_mask, target_column].astype(float)
        X_test = wide_df.loc[test_mask, feature_pool].copy()
        y_test = wide_df.loc[test_mask, target_column].astype(float)

        model = make_xgbrf_model(seed=seed)
        model.fit(X_train, y_train)
        base_pred = model.predict(X_test)
        base_metrics = regression_metrics(y_test.to_numpy(dtype=float), base_pred)
        base_rmse = base_metrics["rmse"]

        rng = np.random.RandomState(seed)
        region_rows = []
        for feat in feature_pool:
            deltas = []
            for rep in range(int(n_repeats)):
                X_perm = X_test.copy()
                values = X_perm[feat].to_numpy()
                permuted = values.copy()
                rng.shuffle(permuted)
                X_perm[feat] = permuted
                pred = model.predict(X_perm)
                rmse_perm = regression_metrics(y_test.to_numpy(dtype=float), pred)["rmse"]
                deltas.append(float(rmse_perm - base_rmse))

            row = {
                "region": region_name,
                "target": target_column,
                "holdout_regime": holdout_regime,
                "feature": feat,
                "base_rmse": base_rmse,
                "delta_rmse_mean": float(np.mean(deltas)) if deltas else np.nan,
                "delta_rmse_std": float(np.std(deltas)) if deltas else np.nan,
                "n_repeats": int(n_repeats),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_sites": int(wide_df["site"].nunique()) if "site" in wide_df.columns else np.nan,
            }
            region_rows.append(row)
            all_rows.append(row)

        if not region_rows:
            continue

        region_df = pd.DataFrame(region_rows).sort_values("delta_rmse_mean", ascending=False)
        csv_path = type_dir / f"PermImp_{region_token}_{safe_target_name(target_column)}_{holdout_regime}.csv"
        region_df.to_csv(csv_path, index=False)
        logging.info(f"Permutation importance table saved: {csv_path}")

        try:
            fig_h = max(4.5, 0.35 * len(region_df) + 1.5)
            fig, ax = plt.subplots(figsize=(11, fig_h))
            ax.barh(
                region_df["feature"].tolist()[::-1],
                region_df["delta_rmse_mean"].tolist()[::-1],
                xerr=region_df["delta_rmse_std"].tolist()[::-1],
                color="#4c78a8",
                edgecolor="black",
            )
            ax.axvline(0, color="black", linewidth=1)
            ax.set_xlabel("Permutation importance (ΔRMSE)")
            ax.set_ylabel("Feature")
            ax.set_title(f"{region_name}: permutation importance\nTarget: {target_column} | Baseline RMSE={base_rmse:.3f} | Regime={holdout_regime}")
            ax.tick_params(axis="y", rotation=0)
            for label in ax.get_yticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment("right")
            fig.subplots_adjust(left=0.30, right=0.98, top=0.88, bottom=0.10)
            plot_path = type_dir / f"PermImp_{region_token}_{safe_target_name(target_column)}_{holdout_regime}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"Permutation importance plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed plotting permutation importance for region {region_name}")

    if not all_rows:
        return pd.DataFrame()

    combined_df = pd.DataFrame(all_rows)
    combined_path = type_dir / f"PermImp_ALL_{safe_target_name(target_column)}_{holdout_regime}.csv"
    combined_df.to_csv(combined_path, index=False)
    logging.info(f"Combined permutation importance table saved: {combined_path}")
    return combined_df


def compute_xgbrf_shap_contribs(model, X):
    """
    Compute SHAP (TreeSHAP) contributions for an XGBoost model.

    Prefer `shap.TreeExplainer` when available, but fall back to XGBoost's native
    `pred_contribs=True` when SHAP is missing or incompatible with XGBRFRegressor.
    Returns a numpy array shaped (n_samples, n_features).
    """
    # 1) Try SHAP library (may fail for some shap/xgboost versions with XGBRFRegressor)
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        # Newer SHAP can return Explanation
        try:
            values_arr = np.asarray(values.values)  # Explanation
        except Exception:
            values_arr = np.asarray(values)
        if values_arr.ndim == 3:
            # Handle multi-output shape (n_samples, n_outputs, n_features)
            values_arr = values_arr[:, 0, :]
        return values_arr
    except Exception as exc:
        logging.info(f"SHAP TreeExplainer unavailable/incompatible ({exc}); using XGBoost pred_contribs fallback.")

    # 2) Fallback: XGBoost native TreeSHAP contributions
    try:
        import xgboost as xgb  # type: ignore

        booster = model.get_booster()
        dmat = xgb.DMatrix(X, feature_names=list(X.columns))
        contrib = booster.predict(dmat, pred_contribs=True)
        contrib = np.asarray(contrib)
        # last column is the bias term
        if contrib.ndim == 2 and contrib.shape[1] == (X.shape[1] + 1):
            contrib = contrib[:, :-1]
        return contrib
    except Exception as exc:
        raise RuntimeError(
            "Failed computing SHAP contributions. Install `shap` or ensure `xgboost` supports pred_contribs."
        ) from exc


def run_region_shap_importance_xgbrf(
    regional_input_files_list,
    target_column="PM2.5",
    candidate_features=None,
    missing_frac=0.20,
    seed=42,
    min_periods=30,
    holdout_regime="random",
    max_samples=5000,
    output_dir=REGIONAL_ASSESSMENT_DIR,
    top_n_plot=20,
):
    """
    Per region: train one multi-feature XGBRF model, then compute mean(|SHAP|) feature importance on holdout set.
    Saves CSV + bar plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    type_dir = ensure_output_subdir(output_dir, "SHAPImportance")

    candidate_features = [
        feature for feature in (candidate_features or SIMPLE_XGBRF_FEATURES)
        if feature.upper() != target_column.upper()
    ]

    rows = []
    for filepath in regional_input_files_list:
        region_name = region_name_from_input_file(filepath)
        region_token = safe_target_name(region_name)

        try:
            wide_df = build_region_site_wide_table(
                filepath,
                target_column=target_column,
                candidate_features=candidate_features,
            )
        except Exception:
            logging.exception(f"Failed building wide table for SHAP importance: {filepath}")
            continue

        if wide_df.empty:
            continue

        feature_pool = []
        for feat in candidate_features:
            if feat in wide_df.columns and int(wide_df[feat].notna().sum()) >= int(min_periods):
                feature_pool.append(feat)
        if len(feature_pool) < 2:
            continue

        holdout_mask = make_holdout_mask_by_site(
            wide_df,
            target_column=target_column,
            missing_frac=missing_frac,
            seed=seed,
            regime=holdout_regime,
        )
        mask = holdout_mask.reindex(wide_df.index).fillna(False).astype(bool)
        train_mask = (~mask) & wide_df[target_column].notna()
        test_mask = mask & wide_df[target_column].notna()
        if int(train_mask.sum()) < 50 or int(test_mask.sum()) < 10:
            continue

        X_train = wide_df.loc[train_mask, feature_pool]
        y_train = wide_df.loc[train_mask, target_column].astype(float)
        X_test = wide_df.loc[test_mask, feature_pool].copy()
        y_test = wide_df.loc[test_mask, target_column].astype(float)

        model = make_xgbrf_model(seed=seed)
        model.fit(X_train, y_train)

        # Sample to limit compute
        if len(X_test) > int(max_samples):
            X_sample = X_test.sample(n=int(max_samples), random_state=seed)
        else:
            X_sample = X_test

        try:
            shap_vals = compute_xgbrf_shap_contribs(model, X_sample)
        except Exception:
            logging.exception(f"Failed computing SHAP values for region {region_name}")
            continue

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        imp_df = pd.DataFrame({"feature": feature_pool, "mean_abs_shap": mean_abs})
        imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
        imp_df["region"] = region_name
        imp_df["target"] = target_column
        imp_df["holdout_regime"] = holdout_regime
        imp_df["n_train"] = int(train_mask.sum())
        imp_df["n_test"] = int(test_mask.sum())
        imp_df["n_sites"] = int(wide_df["site"].nunique()) if "site" in wide_df.columns else np.nan
        rows.append(imp_df)

        csv_path = type_dir / f"SHAPImportance_{region_token}_{safe_target_name(target_column)}_{holdout_regime}.csv"
        imp_df.to_csv(csv_path, index=False)
        logging.info(f"SHAP importance saved: {csv_path}")

        try:
            plot_df = imp_df.head(int(top_n_plot)).iloc[::-1]
            fig_h = max(4.5, 0.35 * len(plot_df) + 1.5)
            fig, ax = plt.subplots(figsize=(11, fig_h))
            ax.barh(plot_df["feature"].tolist(), plot_df["mean_abs_shap"].tolist(), color="#4c78a8", edgecolor="black")
            ax.set_xlabel("mean(|SHAP|) on holdout")
            ax.set_ylabel("Feature")
            ax.set_title(f"{region_name}: SHAP feature importance (XGBRF)\nTarget: {target_column} | Regime={holdout_regime}")
            ax.tick_params(axis="y", rotation=0)
            for label in ax.get_yticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment("right")
            fig.subplots_adjust(left=0.30, right=0.98, top=0.88, bottom=0.10)
            plot_path = type_dir / f"SHAPImportance_{region_token}_{safe_target_name(target_column)}_{holdout_regime}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"SHAP importance plot saved: {plot_path}")
        except Exception:
            logging.exception(f"Failed plotting SHAP importance for region {region_name}")

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    combined_path = type_dir / f"SHAPImportance_ALL_{safe_target_name(target_column)}_{holdout_regime}.csv"
    combined.to_csv(combined_path, index=False)
    logging.info(f"Combined SHAP importance saved: {combined_path}")
    return combined


def run_missingness_regime_sensitivity(
    regional_input_files_list,
    target_column="PM2.5",
    candidate_features=None,
    missing_frac=0.20,
    seed=42,
    min_periods=30,
    output_dir=REGIONAL_ASSESSMENT_DIR,
):
    """
    Evaluate baseline-all-features performance under different missingness/holdout regimes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    type_dir = ensure_output_subdir(output_dir, "MissingnessRegimeSensitivity")

    regimes = ["random", "random_block", "last_block"]
    candidate_features = [
        feature for feature in (candidate_features or SIMPLE_XGBRF_FEATURES)
        if feature.upper() != target_column.upper()
    ]

    rows = []
    for filepath in regional_input_files_list:
        region_name = region_name_from_input_file(filepath)
        region_token = safe_target_name(region_name)
        try:
            wide_df = build_region_site_wide_table(
                filepath,
                target_column=target_column,
                candidate_features=candidate_features,
            )
        except Exception:
            logging.exception(f"Failed building wide table for regime sensitivity: {filepath}")
            continue
        if wide_df.empty:
            continue

        feature_pool = []
        for feat in candidate_features:
            if feat in wide_df.columns and int(wide_df[feat].notna().sum()) >= int(min_periods):
                feature_pool.append(feat)
        if len(feature_pool) < 2:
            continue

        for regime in regimes:
            holdout_mask = make_holdout_mask_by_site(
                wide_df,
                target_column=target_column,
                missing_frac=missing_frac,
                seed=seed,
                regime=regime,
            )
            result = score_xgbrf_feature_set(
                wide_df,
                target_column=target_column,
                feature_list=feature_pool,
                holdout_mask=holdout_mask,
                seed=seed,
            )
            if result is None:
                continue
            result.update({"region": region_name, "target": target_column, "regime": regime})
            rows.append(result)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    csv_path = type_dir / f"RegimeSensitivity_{safe_target_name(target_column)}.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Missingness regime sensitivity table saved: {csv_path}")

    # Plot RMSE by regime for each region
    try:
        plot_df = df.pivot_table(index="region", columns="regime", values="rmse", aggfunc="mean")
        if not plot_df.empty:
            fig_w = max(10, 0.8 * len(plot_df.columns) + 6)
            fig_h = max(6, 0.35 * len(plot_df.index) + 3)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            sns.heatmap(
                plot_df,
                annot=True,
                fmt=".3f",
                cmap="viridis_r",
                ax=ax,
                cbar_kws={"label": "RMSE"},
            )
            ax.set_title(f"RMSE sensitivity to holdout regime | Target: {target_column}")
            ax.set_xlabel("Regime")
            ax.set_ylabel("Region")
            ax.tick_params(axis="x", rotation=0)
            ax.tick_params(axis="y", rotation=0)
            for label in ax.get_yticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment("right")
            fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.12)
            plot_path = type_dir / f"RegimeSensitivity_{safe_target_name(target_column)}_rmse.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"Missingness regime sensitivity plot saved: {plot_path}")
    except Exception:
        logging.exception("Failed plotting regime sensitivity heatmap")

    return df


def score_xgbrf_feature(data, feature_name, missing_frac=0.20, seed=42):
    """Score one feature as an XGBRF imputation predictor for simulated missing targets."""
    feature_df = data[data["feature"] == feature_name].dropna(subset=["target", "feature_value"]).copy()
    if len(feature_df) < 30:
        return None

    rng = np.random.RandomState(seed)
    test_size = max(1, int(round(len(feature_df) * missing_frac)))
    if test_size >= len(feature_df):
        test_size = max(1, len(feature_df) // 5)

    test_positions = rng.choice(feature_df.index.to_numpy(), size=test_size, replace=False)
    test_mask = feature_df.index.isin(test_positions)
    train_df = feature_df.loc[~test_mask]
    test_df = feature_df.loc[test_mask]

    if len(train_df) < 20 or len(test_df) < 1:
        return None

    model = make_xgbrf_model(seed=seed)
    model.fit(train_df[["feature_value"]], train_df["target"])
    predictions = model.predict(test_df[["feature_value"]])
    true_values = test_df["target"].to_numpy(dtype=float)
    metrics = regression_metrics(true_values, predictions)

    return {
        "feature": feature_name,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r": metrics["r"],
        "nse": metrics["nse"],
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_sites": int(feature_df["site"].nunique()),
    }


def safe_target_name(target_column):
    return str(target_column).replace(".", "").replace(" ", "_")


def make_xgbrf_model(seed=42):
    try:
        from xgboost import XGBRFRegressor
    except Exception as exc:
        raise ImportError(
            "xgboost is required for --mode simple_xgbrf. "
            "Run this in the environment where xgboost is installed."
        ) from exc

    return XGBRFRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=1.0,
        subsample=0.8,
        colsample_bynode=0.8,
        random_state=seed,
        n_jobs=-1,
        objective="reg:squarederror",
    )


def regression_metrics(true_values, predictions):
    true_values = np.asarray(true_values, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    errors = predictions - true_values
    denom = float(np.sum((true_values - np.mean(true_values)) ** 2)) if len(true_values) else 0.0
    metrics = {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "r": np.nan,
        "nse": np.nan,
    }
    if len(true_values) > 1 and np.std(true_values) > 0 and np.std(predictions) > 0:
        metrics["r"] = float(np.corrcoef(true_values, predictions)[0, 1])
    if denom > 0:
        metrics["nse"] = float(1.0 - (np.sum(errors ** 2) / denom))
    return metrics


def compare_region_vs_site_xgbrf(all_data, target_column="PM2.5", missing_frac=0.20, seed=42):
    """
    Compare one region-level model against separate site-level models.

    For each region and variable, the same site-level holdout rows are used for:
    1) one model trained with all stations/sites from that region
    2) one model trained separately for each station/site
    """
    rows = []
    rng = np.random.RandomState(seed)

    for region_name, region_data in all_data.groupby("region"):
        for feature in sorted(region_data["feature"].unique()):
            feature_df = region_data[region_data["feature"] == feature].dropna(
                subset=["target", "feature_value", "site"]
            ).copy()
            if len(feature_df) < 30:
                continue

            test_indices = []
            for _, site_df in feature_df.groupby("site"):
                if len(site_df) < 30:
                    continue
                test_size = max(1, int(round(len(site_df) * missing_frac)))
                if test_size >= len(site_df):
                    test_size = max(1, len(site_df) // 5)
                test_indices.extend(rng.choice(site_df.index.to_numpy(), size=test_size, replace=False))

            if not test_indices:
                continue

            test_mask = feature_df.index.isin(test_indices)
            region_train = feature_df.loc[~test_mask]
            region_test = feature_df.loc[test_mask]
            if len(region_train) < 20 or region_test.empty:
                continue

            region_model = make_xgbrf_model(seed=seed)
            region_model.fit(region_train[["feature_value"]], region_train["target"])

            for site_name, site_test in region_test.groupby("site"):
                site_train = feature_df.loc[(feature_df["site"] == site_name) & (~test_mask)]
                if site_test.empty:
                    continue

                region_predictions = region_model.predict(site_test[["feature_value"]])
                region_metrics = regression_metrics(site_test["target"].to_numpy(dtype=float), region_predictions)

                site_metrics = {"rmse": np.nan, "mae": np.nan, "r": np.nan, "nse": np.nan}
                if len(site_train) >= 20:
                    site_model = make_xgbrf_model(seed=seed)
                    site_model.fit(site_train[["feature_value"]], site_train["target"])
                    site_predictions = site_model.predict(site_test[["feature_value"]])
                    site_metrics = regression_metrics(
                        site_test["target"].to_numpy(dtype=float),
                        site_predictions,
                    )

                rows.append({
                    "target": target_column,
                    "feature": feature,
                    "region": region_name,
                    "site": site_name,
                    "region_model_rmse": region_metrics["rmse"],
                    "separate_site_rmse": site_metrics["rmse"],
                    "delta_rmse_site_minus_region": (
                        site_metrics["rmse"] - region_metrics["rmse"]
                        if not np.isnan(site_metrics["rmse"]) else np.nan
                    ),
                    "region_model_mae": region_metrics["mae"],
                    "separate_site_mae": site_metrics["mae"],
                    "region_model_r": region_metrics["r"],
                    "separate_site_r": site_metrics["r"],
                    "region_model_nse": region_metrics["nse"],
                    "separate_site_nse": site_metrics["nse"],
                    "n_train_region": int(len(region_train)),
                    "n_train_site": int(len(site_train)),
                    "n_test_site": int(len(site_test)),
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["region", "site", "feature"])


def plot_region_vs_site_heatmaps(comparison_df, target_column="PM2.5", output_dir=OUTPUT_DIR):
    """Plot one heatmap figure per region comparing region model vs separate site models."""
    if comparison_df.empty:
        return

    metric_specs = [
        ("rmse", "RMSE", "viridis_r", False),
        ("r", "Correlation (r)", "coolwarm", True),
        ("nse", "Nash–Sutcliffe (NSE)", "coolwarm", True),
    ]

    # Plot each region separately (avoid producing combined "SMALL_REGIONS_LT4SITES" figures).
    region_names = sorted(comparison_df["region"].unique())
    output_dir = Path(output_dir)
    type_dir = ensure_output_subdir(output_dir, "RegionModel_vs_SeparateSites")

    for region_name in region_names:
        region_df = comparison_df[comparison_df["region"] == region_name]
        feature_order = (
            region_df.groupby("feature")["region_model_rmse"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )

        fig_width = max(14, 0.85 * region_df["site"].nunique() * 2.2)
        fig_height = max(6, 0.5 * region_df["feature"].nunique() + 3)
        region_token = safe_target_name(region_name)

        for metric_key, metric_label, cmap, center_zero in metric_specs:
            # Build both tables first so we can share one colour bar and one y axis.
            region_model_df = region_df.pivot_table(
                index="feature",
                columns="site",
                values=f"region_model_{metric_key}",
                aggfunc="mean",
            ).reindex(feature_order)
            separate_site_df = region_df.pivot_table(
                index="feature",
                columns="site",
                values=f"separate_site_{metric_key}",
                aggfunc="mean",
            ).reindex(feature_order)

            values_region = region_model_df.to_numpy(dtype=float)
            values_site = separate_site_df.to_numpy(dtype=float)

            common_kwargs = {}
            if center_zero:
                max_abs = 0.0
                if not np.isnan(values_region).all():
                    max_abs = max(max_abs, float(np.nanmax(np.abs(values_region))))
                if not np.isnan(values_site).all():
                    max_abs = max(max_abs, float(np.nanmax(np.abs(values_site))))
                if max_abs > 0:
                    common_kwargs.update({"center": 0, "vmin": -max_abs, "vmax": max_abs})
            else:
                combined = np.concatenate([values_region.ravel(), values_site.ravel()])
                combined = combined[~np.isnan(combined)]
                if combined.size:
                    common_kwargs.update({"vmin": float(np.min(combined)), "vmax": float(np.max(combined))})

            fig, (ax_left, ax_right, cax) = plt.subplots(
                1,
                3,
                figsize=(fig_width, fig_height),
                sharey=True,
                gridspec_kw={"width_ratios": [1, 1, 0.05], "wspace": 0.05},
            )

            left_hm = sns.heatmap(
                region_model_df,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                ax=ax_left,
                cbar=False,
                **common_kwargs,
            )
            right_hm = sns.heatmap(
                separate_site_df,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                ax=ax_right,
                cbar_ax=cax,
                **common_kwargs,
            )

            ax_left.set_title(f"One model using all sites in this region\nRegion: {region_name} | Target: {target_column} | {metric_label}")
            ax_right.set_title(f"Separate model for each site\nRegion: {region_name} | Target: {target_column} | {metric_label}")
            ax_left.set_xlabel("Study site / station")
            ax_right.set_xlabel("Study site / station")
            ax_left.set_ylabel("Variable")
            ax_right.set_ylabel("")
            ax_left.tick_params(axis="x", rotation=45)
            ax_right.tick_params(axis="x", rotation=45)
            ax_left.tick_params(axis="y", rotation=0)
            ax_right.tick_params(axis="y", left=False, labelleft=False)

            # Ensure y tick labels are horizontal and readable
            for label in ax_left.get_yticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment("right")

            # Use constrained_layout to avoid tight_layout warnings with colourbar axes.
            fig.subplots_adjust(left=0.20, right=0.96, top=0.90, bottom=0.16, wspace=0.05)
            plot_path = type_dir / (
                f"Heatmap_{region_token}_{safe_target_name(target_column)}_RegionModel_vs_SeparateSites_{metric_key}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"Region model vs separate-site heatmap saved: {plot_path}")


def run_simple_xgbrf_feature_selection(
    target_column="PM2.5",
    selected_regions=None,
    missing_frac=0.20,
    station_heatmaps=True,
    output_dir=REGIONAL_ASSESSMENT_DIR,
    run_shap=True,
    shap_max_samples=5000,
):
    """Compare XGBRF feature RMSE for compiled selected-region data and per-region data."""
    selected_regions = selected_regions if selected_regions is not None else getattr(config, "TARGET_REGIONS", [])
    files = selected_regional_input_files(selected_regions)
    if not files:
        raise FileNotFoundError(
            f"No regional input CSVs found for selected regions {selected_regions} in {REGIONAL_INPUT_DIR}"
        )

    logging.info(f"Simple XGBRF feature selection using {len(files)} regional CSV file(s)")
    region_tables = []
    for filepath in files:
        table = build_region_feature_table(
            filepath,
            target_column=target_column,
            candidate_features=SIMPLE_XGBRF_FEATURES,
        )
        if table.empty:
            logging.warning(f"No usable feature rows found in {filepath}")
            continue
        logging.info(
            f"Loaded {len(table)} feature rows from {filepath.name} "
            f"({table['site'].nunique()} sites, {table['feature'].nunique()} features)"
        )
        region_tables.append(table)

    if not region_tables:
        raise ValueError("No usable regional feature data found for simple XGBRF feature selection")

    # "Basics" diagnostics for understanding relationships & trends (saved under Basics/)
    try:
        plot_region_basics_for_inputs(
            files,
            target_column=target_column,
            candidate_features=SIMPLE_XGBRF_FEATURES,
            output_dir=BASICS_DIR,
            corr_method="pearson",
            min_periods=30,
            max_lag_hours=72,
            time_freq="7D",
        )
    except Exception:
        logging.exception("Basics plotting failed; continuing with feature selection runs")

    all_data = pd.concat(region_tables, ignore_index=True)
    comparison_df = compare_region_vs_site_xgbrf(
        all_data,
        target_column=target_column,
        missing_frac=missing_frac,
    )
    if not comparison_df.empty:
        comparison_csv_path = Path(output_dir) / (
            f"RegionModel_vs_SeparateSites_{safe_target_name(target_column)}_rmse.csv"
        )
        comparison_df.to_csv(comparison_csv_path, index=False)
        plot_region_vs_site_heatmaps(comparison_df, target_column, output_dir=output_dir)
        logging.info(f"Region model vs separate-site XGBRF comparison saved: {comparison_csv_path}")

    features = sorted(all_data["feature"].unique())
    rows = []

    for feature in features:
        result = score_xgbrf_feature(all_data, feature, missing_frac=missing_frac, seed=42)
        if result:
            result.update({"scope": "compiled", "region": "ALL_SELECTED"})
            rows.append(result)

    for region_name, region_data in all_data.groupby("region"):
        for feature in sorted(region_data["feature"].unique()):
            result = score_xgbrf_feature(region_data, feature, missing_frac=missing_frac, seed=42)
            if result:
                result.update({"scope": "per_region", "region": region_name})
                rows.append(result)

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        raise ValueError("XGBRF feature selection produced no valid RMSE results")

    csv_path = Path(output_dir) / f"simple_xgbrf_{safe_target_name(target_column)}_compiled_vs_region_rmse.csv"
    results_df = results_df[
        ["scope", "region", "feature", "rmse", "mae", "r", "nse", "n_train", "n_test", "n_sites"]
    ].sort_values(["scope", "region", "rmse"])
    results_df.to_csv(csv_path, index=False)

    # Incremental "best + 2nd + ..." feature combinations per region (saved in Reginal_Assessment/)
    combo_df = None
    try:
        combo_df = run_region_feature_combinations(
            files,
            single_feature_results_df=results_df,
            target_column=target_column,
            candidate_features=SIMPLE_XGBRF_FEATURES,
            missing_frac=missing_frac,
            seed=42,
            max_features=10,
            output_dir=output_dir,
        )
    except Exception:
        logging.exception("Failed running per-region feature combination analysis")

    # Leave-one-out ablation: all features vs removing each feature (saved in Reginal_Assessment/Ablation/)
    try:
        run_region_feature_ablation(
            files,
            target_column=target_column,
            candidate_features=SIMPLE_XGBRF_FEATURES,
            missing_frac=missing_frac,
            seed=42,
            min_periods=30,
            output_dir=output_dir,
        )
    except Exception:
        logging.exception("Failed running per-region ablation analysis")

    # Missingness-regime sensitivity (random vs block) for baseline-all-features model
    try:
        run_missingness_regime_sensitivity(
            files,
            target_column=target_column,
            candidate_features=SIMPLE_XGBRF_FEATURES,
            missing_frac=missing_frac,
            seed=42,
            min_periods=30,
            output_dir=output_dir,
        )
    except Exception:
        logging.exception("Failed running missingness regime sensitivity")

    # Permutation importance on holdout set (baseline-all-features model)
    perm_df_random = None
    try:
        for regime in ("random", "random_block", "last_block"):
            perm_df = run_region_permutation_importance(
                files,
                target_column=target_column,
                candidate_features=SIMPLE_XGBRF_FEATURES,
                missing_frac=missing_frac,
                seed=42,
                min_periods=30,
                n_repeats=5,
                holdout_regime=regime,
                output_dir=output_dir,
            )
            if regime == "random":
                perm_df_random = perm_df
    except Exception:
        logging.exception("Failed running permutation importance analysis")

    # SHAP (TreeSHAP) feature importance for multi-feature XGBRF (replaces single-feature-only importance)
    if run_shap:
        try:
            for regime in ("random", "random_block", "last_block"):
                run_region_shap_importance_xgbrf(
                    files,
                    target_column=target_column,
                    candidate_features=SIMPLE_XGBRF_FEATURES,
                    missing_frac=missing_frac,
                    seed=42,
                    min_periods=30,
                    holdout_regime=regime,
                    max_samples=shap_max_samples,
                    output_dir=output_dir,
                    top_n_plot=20,
                )
        except Exception:
            logging.exception("Failed running SHAP feature importance analysis")

    # Save JSON mapping region -> target -> best predictors
    try:
        write_region_target_best_predictors_json(
            target_column=target_column,
            selected_regions=selected_regions,
            single_feature_results_df=results_df,
            feature_combo_df=combo_df,
            permutation_df=perm_df_random,
            output_dir=output_dir,
            top_n_single=5,
        )
    except Exception:
        logging.exception("Failed writing best-predictors JSON summary")

    # Do not create the aggregated "Heatmap_ALLRegion_*" figures (too broad / not desired).
    if station_heatmaps:
        station_results_df = run_station_variable_xgbrf_heatmaps(
            all_data,
            target_column=target_column,
            missing_frac=missing_frac,
            output_dir=output_dir,
        )
        if not station_results_df.empty:
            station_csv_path = Path(output_dir) / f"StationHeatmap_ALLRegion_{safe_target_name(target_column)}_rmse.csv"
            station_results_df.to_csv(station_csv_path, index=False)
            logging.info(f"Station-level XGBRF RMSE table saved: {station_csv_path}")
    logging.info(f"Simple XGBRF results saved: {csv_path}")
    return results_df


def plot_simple_xgbrf_heatmap(results_df, target_column="PM2.5"):
    """Save heatmap-only metric plots for compiled and per-region XGBRF scores."""
    metric_specs = [
        ("rmse", "RMSE", "viridis_r", False, True),
        ("r", "Correlation (r)", "coolwarm", True, False),
        ("nse", "Nash–Sutcliffe (NSE)", "coolwarm", True, False),
    ]

    for metric_key, metric_label, cmap, center_zero, sort_ascending in metric_specs:
        compiled_df = results_df[results_df["scope"] == "compiled"].sort_values(
            metric_key,
            ascending=sort_ascending,
        )
        region_df = results_df[results_df["scope"] == "per_region"]

        compiled_heatmap = None
        if not compiled_df.empty:
            compiled_heatmap = compiled_df.pivot_table(
                index="feature", columns="region", values=metric_key, aggfunc="mean"
            ).reindex(compiled_df["feature"].tolist())
        per_region_heatmap = None
        if not region_df.empty:
            per_region_heatmap = region_df.pivot_table(
                index="feature", columns="region", values=metric_key, aggfunc="mean"
            )
            feature_order = compiled_df["feature"].tolist() if not compiled_df.empty else sorted(per_region_heatmap.index)
            per_region_heatmap = per_region_heatmap.reindex([feat for feat in feature_order if feat in per_region_heatmap.index])

        # Common colour scale across the two panels
        common_kwargs = {}
        matrices = []
        if compiled_heatmap is not None:
            matrices.append(compiled_heatmap.to_numpy(dtype=float))
        if per_region_heatmap is not None:
            matrices.append(per_region_heatmap.to_numpy(dtype=float))

        if matrices:
            combined = np.concatenate([m.ravel() for m in matrices])
            combined = combined[~np.isnan(combined)]
            if combined.size:
                if center_zero:
                    max_abs = float(np.max(np.abs(combined)))
                    if max_abs > 0:
                        common_kwargs.update({"center": 0, "vmin": -max_abs, "vmax": max_abs})
                else:
                    common_kwargs.update({"vmin": float(np.min(combined)), "vmax": float(np.max(combined))})

        fig, (ax_left, ax_right, cax) = plt.subplots(
            1,
            3,
            figsize=(18, 7),
            sharey=True,
            gridspec_kw={"width_ratios": [1, 1, 0.05], "wspace": 0.05},
        )

        has_left = compiled_heatmap is not None and not compiled_heatmap.empty
        has_right = per_region_heatmap is not None and not per_region_heatmap.empty

        if not has_left and not has_right:
            plt.close(fig)
            continue

        if has_left:
            left_cbar_ax = None if has_right else cax
            left_cbar = not has_right
            sns.heatmap(
                compiled_heatmap,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                ax=ax_left,
                cbar=left_cbar,
                cbar_ax=left_cbar_ax,
                **common_kwargs,
            )
            ax_left.set_title(f"Compiled selected regions: {target_column} {metric_label}")
            ax_left.set_xlabel("")
            ax_left.set_ylabel("Feature")
            ax_left.tick_params(axis="y", rotation=0)
            for label in ax_left.get_yticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment("right")
        else:
            ax_left.set_axis_off()

        if has_right:
            sns.heatmap(
                per_region_heatmap,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                ax=ax_right,
                cbar=True,
                cbar_ax=cax,
                **common_kwargs,
            )
            ax_right.set_title(f"Each region separately: {target_column} {metric_label}")
            ax_right.set_xlabel("Region")
            ax_right.set_ylabel("" if has_left else "Feature")
            if has_left:
                ax_right.tick_params(axis="y", left=False, labelleft=False)
            else:
                ax_right.tick_params(axis="y", rotation=0)
                for label in ax_right.get_yticklabels():
                    label.set_rotation(0)
                    label.set_horizontalalignment("right")
        else:
            ax_right.set_axis_off()
            if not has_left:
                cax.set_axis_off()

        # Use constrained_layout to avoid tight_layout warnings with colourbar axes.
        fig.subplots_adjust(left=0.20, right=0.96, top=0.90, bottom=0.16, wspace=0.05)
        plot_path = OUTPUT_DIR / f"Heatmap_ALLRegion_{safe_target_name(target_column)}_{metric_key}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Simple XGBRF heatmap saved: {plot_path}")


def run_station_variable_xgbrf_heatmaps(all_data, target_column="PM2.5", missing_frac=0.20, output_dir=OUTPUT_DIR):
    """Score and plot RMSE heatmaps for variables by study site under each region."""
    rows = []
    for (region_name, site_name), site_data in all_data.groupby(["region", "site"]):
        for feature in sorted(site_data["feature"].unique()):
            result = score_xgbrf_feature(site_data, feature, missing_frac=missing_frac, seed=42)
            if result:
                result.update({
                    "scope": "station",
                    "region": region_name,
                    "site": site_name,
                })
                rows.append(result)

    station_results_df = pd.DataFrame(rows)
    if station_results_df.empty:
        logging.warning(f"No station-level XGBRF RMSE results produced for target: {target_column}")
        return station_results_df

    station_results_df = station_results_df[
        ["scope", "region", "site", "feature", "rmse", "mae", "r", "nse", "n_train", "n_test", "n_sites"]
    ].sort_values(["region", "site", "rmse"])
    plot_station_variable_heatmaps(station_results_df, target_column, output_dir=output_dir)
    return station_results_df


def plot_station_variable_heatmaps(station_results_df, target_column="PM2.5", output_dir=OUTPUT_DIR):
    """Save one clear variable-by-station metric heatmap per region."""
    regions = sorted(station_results_df["region"].unique())
    if not regions:
        return

    output_dir = Path(output_dir)
    type_dir = ensure_output_subdir(output_dir, "StationVariables")

    metric_specs = [
        ("rmse", "RMSE", "viridis_r", False),
        ("r", "Correlation (r)", "coolwarm", True),
        ("nse", "Nash–Sutcliffe (NSE)", "coolwarm", True),
    ]

    for region_name in regions:
        region_df = station_results_df[station_results_df["region"] == region_name]
        feature_order = (
            region_df.groupby("feature")["rmse"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )
        region_token = safe_target_name(region_name)

        for metric_key, metric_label, cmap, center_zero in metric_specs:
            heatmap_df = region_df.pivot_table(
                index="feature",
                columns="site",
                values=metric_key,
                aggfunc="mean",
            ).reindex(feature_order)
            fig_width = max(10, 0.8 * len(heatmap_df.columns) + 4)
            fig_height = max(6, 0.45 * len(heatmap_df.index) + 3)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            heatmap_kwargs = {
                "annot": True,
                "fmt": ".2f",
                "cmap": cmap,
                "ax": ax,
                "cbar": True,
            }
            if center_zero:
                values = heatmap_df.to_numpy(dtype=float)
                max_abs = np.nanmax(np.abs(values)) if not np.isnan(values).all() else 0
                heatmap_kwargs.update(
                    {
                        "center": 0,
                        "vmin": -max_abs if max_abs > 0 else None,
                        "vmax": max_abs if max_abs > 0 else None,
                    }
                )

            sns.heatmap(heatmap_df, **heatmap_kwargs)
            ax.set_title(f"{region_name}: station-level {target_column} {metric_label} by variable")
            ax.set_xlabel("Study site / station")
            ax.set_ylabel("Variable")
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)
            for label in ax.get_yticklabels():
                label.set_rotation(0)
                label.set_horizontalalignment("right")

            plt.tight_layout()
            plot_path = type_dir / (
                f"Heatmap_{region_token}_{safe_target_name(target_column)}_StationVariables_{metric_key}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"Station variable heatmap saved: {plot_path}")


def run_simple_xgbrf_multi_target_heatmaps(
    target_columns,
    selected_regions=None,
    missing_frac=0.20,
    station_heatmaps=True,
    output_dir=REGIONAL_ASSESSMENT_DIR,
    run_shap=True,
    shap_max_samples=5000,
):
    """Run XGBRF heatmap-only feature selection for each requested target."""
    all_results = []
    for target_column in target_columns:
        try:
            logging.info("=" * 80)
            logging.info(f"Running XGBRF heatmap feature selection for target: {target_column}")
            result = run_simple_xgbrf_feature_selection(
                target_column=target_column,
                selected_regions=selected_regions,
                missing_frac=missing_frac,
                station_heatmaps=station_heatmaps,
                output_dir=output_dir,
                run_shap=run_shap,
                shap_max_samples=shap_max_samples,
            )
            all_results.append(result.assign(target_column=target_column))
        except Exception:
            logging.exception(f"Failed XGBRF heatmap feature selection for target: {target_column}")

    if not all_results:
        raise ValueError("No XGBRF target heatmaps were produced")

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = Path(output_dir) / "simple_xgbrf_multi_target_compiled_vs_region_rmse.csv"
    combined.to_csv(combined_path, index=False)
    logging.info(f"Combined multi-target XGBRF RMSE table saved: {combined_path}")

    # Overall best single feature per target (compiled scope) summary JSON
    try:
        summary_dir = ensure_output_subdir(output_dir, "Summaries")
        overall = {}
        for target_column in target_columns:
            subset = combined[(combined["target_column"] == target_column) & (combined["scope"] == "compiled")].copy()
            if subset.empty:
                continue
            best_row = subset.sort_values("rmse").iloc[0]
            overall[str(target_column)] = {
                "best_single_feature": str(best_row["feature"]),
                "rmse": float(best_row["rmse"]),
                "mae": float(best_row["mae"]),
                "r": float(best_row["r"]),
                "nse": float(best_row["nse"]),
                "n_train": int(best_row.get("n_train", 0)),
                "n_test": int(best_row.get("n_test", 0)),
                "n_sites": int(best_row.get("n_sites", 0)) if str(best_row.get("n_sites", "")).isdigit() else best_row.get("n_sites"),
            }
        out_path = Path(summary_dir) / "OverallBestSingleFeature_ByTarget.json"
        with open(out_path, "w") as f:
            json.dump(overall, f, indent=2)
        logging.info(f"Overall best-feature JSON saved: {out_path}")
    except Exception:
        logging.exception("Failed writing overall best-feature JSON summary")

    return combined


def run_shap_only_multi_target(
    target_columns,
    selected_regions=None,
    missing_frac=0.20,
    output_dir=REGIONAL_ASSESSMENT_DIR,
    shap_max_samples=5000,
    regimes=None,
):
    """
    SHAP-only workflow:
    - For each target, compute SHAP importance for region-level multi-feature XGBRF models.
    - Produces only SHAP-related CSV/PNG outputs under Reginal_Assessment/SHAPImportance/.
    """
    selected_regions = selected_regions if selected_regions is not None else getattr(config, "TARGET_REGIONS", [])
    files = selected_regional_input_files(selected_regions)
    if not files:
        raise FileNotFoundError(
            f"No regional input CSVs found for selected regions {selected_regions} in {REGIONAL_INPUT_DIR}"
        )

    regimes = regimes or ["random"]
    for target_column in target_columns:
        logging.info("=" * 80)
        logging.info(f"Running SHAP-only XGBRF importance for target: {target_column}")
        for regime in regimes:
            run_region_shap_importance_xgbrf(
                files,
                target_column=target_column,
                candidate_features=SIMPLE_XGBRF_FEATURES,
                missing_frac=missing_frac,
                seed=42,
                min_periods=30,
                holdout_regime=regime,
                max_samples=shap_max_samples,
                output_dir=output_dir,
                top_n_plot=20,
            )


def apply_test_missingness(data, target_column, regime='random', frac=0.2, seed=42):
    """Apply missingness for testing"""
    original_missing = data[target_column].isna()
    data_with_missing, simulated_mask = apply_missingness(
        data, target_column, regime=regime, frac=frac, seed=seed
    )
    simulated_mask = simulated_mask & (~original_missing)
    return data_with_missing, simulated_mask


def evaluate_imputation(data_original, data_imputed, simulated_mask, target_column):
    """Evaluate imputation performance"""
    # Normalize simulated_mask into a boolean Series aligned with data_original.index
    try:
        if isinstance(simulated_mask, pd.Series):
            mask = simulated_mask.reindex(data_original.index).fillna(False).astype(bool)
        else:
            arr = np.asarray(simulated_mask)
            if arr.ndim > 1:
                arr = arr.squeeze()
            if arr.shape[0] != len(data_original):
                # fallback: try to build a Series (may raise)
                mask = pd.Series(simulated_mask).reindex(data_original.index).fillna(False).astype(bool)
            else:
                mask = pd.Series(arr, index=data_original.index).astype(bool)
    except Exception:
        logging.exception("Simulated mask could not be aligned to data index")
        return None

    try:
        true_values = data_original.loc[mask, target_column].values
    except Exception:
        logging.exception("Failed selecting true values with the aligned mask")
        return None

    # Extract imputed values robustly depending on the returned type
    imputed_values = None
    try:
        if isinstance(data_imputed, pd.DataFrame):
            if target_column in data_imputed.columns:
                try:
                    imputed_values = data_imputed.loc[mask, target_column].values
                except AssertionError:
                    positions = np.where(mask)[0]
                    imputed_values = data_imputed.iloc[positions][target_column].values
            else:
                logging.error(f"Target column '{target_column}' not found in imputed DataFrame columns: {list(data_imputed.columns)}")
                return None
        elif isinstance(data_imputed, pd.Series):
            try:
                imputed_values = data_imputed.loc[mask].values
            except AssertionError:
                positions = np.where(mask)[0]
                imputed_values = data_imputed.iloc[positions].values
        elif isinstance(data_imputed, np.ndarray):
            positions = np.where(mask)[0]
            if data_imputed.ndim == 1:
                imputed_values = data_imputed[positions]
            elif data_imputed.ndim == 2:
                # If 2D array, try to match column by name if possible, else assume last column
                try:
                    col_idx = -1
                    imputed_values = data_imputed[positions, col_idx]
                except Exception:
                    logging.exception("Failed to index numpy imputed array; shape info logged below")
                    logging.info(f"data_imputed.shape={data_imputed.shape}")
                    return None
        else:
            # Unknown type: attempt to coerce to DataFrame
            try:
                df_imp = pd.DataFrame(data_imputed)
                if target_column in df_imp.columns:
                    imputed_values = df_imp.loc[mask, target_column].values
                else:
                    positions = np.where(mask)[0]
                    imputed_values = df_imp.iloc[positions].values.squeeze()
            except Exception:
                logging.exception(f"Unrecognized imputed output type: {type(data_imputed)}")
                return None
    except Exception:
        logging.exception("Failed extracting imputed values from data_imputed")
        return None
    except Exception:
        logging.exception("Failed selecting true/imputed values with the aligned mask")
        return None

    # Remove invalid
    valid_mask = np.isfinite(true_values) & np.isfinite(imputed_values)
    if config.HANDLE_NEGATIVES == 'exclude':
        valid_mask = valid_mask & (true_values >= 0) & (imputed_values >= 0)
    
    if valid_mask.sum() < 10:
        return None
    
    true_clean = true_values[valid_mask]
    imputed_clean = imputed_values[valid_mask]
    
    metrics = evaluate_metrics(true_clean, imputed_clean, handle_negative=config.HANDLE_NEGATIVES)
    return metrics


# ============================================================================
# ANALYSIS 1: INDIVIDUAL FEATURE IMPORTANCE
# ============================================================================

def analyze_individual_features(site_name, data, target_column='PM2.5'):
    """
    Test each input feature individually to determine importance
    """
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS 1: INDIVIDUAL FEATURE IMPORTANCE")
    logging.info("="*80)
    
    available_features = [col for col in config.INPUT_COLUMNS if col in data.columns and col != target_column]
    
    results = []
    
    # Baseline: No features (temporal only)
    logging.info("\nTesting:  BASELINE (Temporal only)")
    try:
        data_missing, sim_mask = apply_test_missingness(data, target_column)
        
        data_imputed = impute_mice(
            data_missing,
            target_column,
            [],  # No input features
            site_name=site_name,
            model_name=f"LightGBM_{site_name}"
        )
        
        metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                'Feature': 'BASELINE (Temporal only)',
                'RMSE': metrics. get('Root Mean Squared Error (RMSE)', np.nan),
                'R':  metrics.get('Correlation Coefficient (R)', np.nan),
                'NSE': metrics.get('Nash-Sutcliffe Efficiency (NSE)', np.nan),
                'MAE': metrics.get('Mean Absolute Error (MAE)', np.nan),
            })
            logging.info(f"  RMSE: {results[-1]['RMSE']:.3f}, R: {results[-1]['R']:.3f}")
    except Exception as e:
        logging.exception("  Failed during baseline imputation")
    
    # Test each feature individually
    for feature in available_features:
        logging.info(f"\nTesting: {feature} (alone)")
        
        try:
            data_missing, sim_mask = apply_test_missingness(data, target_column)
            
            data_imputed = impute_mice(
                data_missing,
                target_column,
                [feature],
                site_name=site_name,
                model_name=f"LightGBM_{site_name}"
            )
            
            metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
            if metrics: 
                results.append({
                    'Feature': feature,
                    'RMSE': metrics. get('Root Mean Squared Error (RMSE)', np.nan),
                    'R': metrics. get('Correlation Coefficient (R)', np.nan),
                    'NSE': metrics.get('Nash-Sutcliffe Efficiency (NSE)', np.nan),
                    'MAE': metrics.get('Mean Absolute Error (MAE)', np.nan),
                })
                logging.info(f"  RMSE: {results[-1]['RMSE']:.3f}, R: {results[-1]['R']:.3f}")
        except Exception as e: 
            logging.exception(f"  Failed testing feature: {feature}")
    
    # All features combined
    logging.info(f"\nTesting: ALL FEATURES")
    try:
        data_missing, sim_mask = apply_test_missingness(data, target_column)
        
        data_imputed = impute_mice(
            data_missing,
            target_column,
            available_features,
            site_name=site_name,
            model_name=f"LightGBM_{site_name}"
        )
        
        metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                'Feature': 'ALL FEATURES',
                'RMSE': metrics.get('Root Mean Squared Error (RMSE)', np.nan),
                'R':  metrics.get('Correlation Coefficient (R)', np.nan),
                'NSE': metrics.get('Nash-Sutcliffe Efficiency (NSE)', np.nan),
                'MAE': metrics.get('Mean Absolute Error (MAE)', np.nan),
            })
            logging.info(f"  RMSE: {results[-1]['RMSE']:.3f}, R: {results[-1]['R']:.3f}")
    except Exception as e: 
        logging.exception("  Failed testing ALL FEATURES")
    
    # Save results
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        logging.warning("No successful imputation results to analyze")
        return results_df
    
    results_df = results_df.sort_values('RMSE')
    results_df.to_csv(OUTPUT_DIR / f"{site_name}_individual_features.csv", index=False)
    
    # Plot
    plot_individual_features(results_df, site_name)
    
    return results_df


def plot_individual_features(results_df, site_name):
    """Visualize individual feature importance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    ax = axes[0]
    results_sorted = results_df.sort_values('RMSE')
    colors = ['red' if x == 'BASELINE (Temporal only)' else 
              'darkgreen' if x == 'ALL FEATURES' else 
              'steelblue' for x in results_sorted['Feature']]
    
    bars = ax.barh(results_sorted['Feature'], results_sorted['RMSE'], color=colors, edgecolor='black')
    ax.set_xlabel('RMSE (μg/m³)', fontweight='bold')
    ax.set_title('Individual Feature Performance\n(Lower is Better)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, results_sorted['RMSE'])):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
               ha='left', va='center', fontweight='bold', fontsize=8)
    
    # R coefficient comparison
    ax = axes[1]
    results_sorted = results_df.sort_values('R', ascending=False)
    colors = ['red' if x == 'BASELINE (Temporal only)' else 
              'darkgreen' if x == 'ALL FEATURES' else 
              'steelblue' for x in results_sorted['Feature']]
    
    bars = ax.barh(results_sorted['Feature'], results_sorted['R'], color=colors, edgecolor='black')
    ax.set_xlabel('Correlation Coefficient (R)', fontweight='bold')
    ax.set_title('Individual Feature Correlation\n(Higher is Better)', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, results_sorted['R'])):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
               ha='left', va='center', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{site_name}_individual_features.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"✅ Plot saved:  {OUTPUT_DIR / f'{site_name}_individual_features.png'}")


# ============================================================================
# ANALYSIS 2: FEATURE GROUP COMBINATIONS
# ============================================================================

def analyze_feature_groups(site_name, data, target_column='PM2.5'):
    """
    Test domain-specific feature groups
    """
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS 2: FEATURE GROUP COMBINATIONS")
    logging.info("="*80)
    
    results = []
    
    for group_name, features in FEATURE_GROUPS.items():
        # Check which features are available
        available = [f for f in features if f in data.columns and f != target_column]
        
        if not available:
            logging.warning(f"Skipping {group_name}:  No features available")
            continue
        
        logging.info(f"\nTesting:  {group_name} ({len(available)} features)")
        logging.info(f"  Features: {available}")
        
        try:
            data_missing, sim_mask = apply_test_missingness(data, target_column)
            
            data_imputed = impute_mice(
                data_missing,
                target_column,
                available,
                site_name=site_name,
                model_name=f"LightGBM_{site_name}"
            )
            
            metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
            if metrics:
                results. append({
                    'Group':  group_name,
                    'Features': ', '.join(available),
                    'N_Features': len(available),
                    'RMSE': metrics.get('Root Mean Squared Error (RMSE)', np.nan),
                    'R': metrics.get('Correlation Coefficient (R)', np.nan),
                    'NSE': metrics.get('Nash-Sutcliffe Efficiency (NSE)', np.nan),
                    'MAE': metrics. get('Mean Absolute Error (MAE)', np.nan),
                })
                logging.info(f"  RMSE: {results[-1]['RMSE']:.3f}, R: {results[-1]['R']:.3f}")
        except Exception as e: 
            logging.exception(f"  Failed testing group: {group_name}")
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logging.warning(f"No successful feature group results for site={site_name}")
        cols = ['Group', 'Features', 'N_Features', 'RMSE', 'R', 'NSE', 'MAE']
        results_df = pd.DataFrame(columns=cols)
        results_df.to_csv(OUTPUT_DIR / f"{site_name}_feature_groups.csv", index=False)
        return results_df

    results_df = results_df.sort_values('RMSE')
    results_df.to_csv(OUTPUT_DIR / f"{site_name}_feature_groups.csv", index=False)
    
    plot_feature_groups(results_df, site_name)
    
    return results_df


def plot_feature_groups(results_df, site_name):
    """Visualize feature group performance"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    results_sorted = results_df.sort_values('RMSE')
    
    # Create color map based on number of features
    norm = plt.Normalize(results_sorted['N_Features'].min(), results_sorted['N_Features'].max())
    colors = plt.cm.RdYlGn_r(norm(results_sorted['N_Features']))
    
    bars = ax.barh(results_sorted['Group'], results_sorted['RMSE'], color=colors, edgecolor='black')
    ax.set_xlabel('RMSE (μg/m³)', fontweight='bold', fontsize=12)
    ax.set_title(f'Feature Group Performance:  {site_name}\n(Lower RMSE is Better)', 
                fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels and feature count
    for i, (bar, row) in enumerate(zip(bars, results_sorted. itertuples())):
        ax.text(row.RMSE, bar.get_y() + bar.get_height()/2, 
               f'  {row. RMSE:.2f} ({row.N_Features} features)',
               ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Number of Features', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{site_name}_feature_groups.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"✅ Plot saved: {OUTPUT_DIR / f'{site_name}_feature_groups.png'}")


# ============================================================================
# ANALYSIS 3: LAG CONFIGURATION ANALYSIS
# ============================================================================

def analyze_lag_configurations(site_name, data, target_column='PM2.5'):
    """
    Test different lag configurations
    """
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS 3: LAG CONFIGURATION ANALYSIS")
    logging.info("="*80)
    
    # Use best features from previous analysis
    available_features = [col for col in config.INPUT_COLUMNS if col in data.columns and col != target_column]
    
    results = []
    
    for lag_name, lag_list in LAG_CONFIGURATIONS.items():
        logging.info(f"\nTesting lag configuration: {lag_name}")
        logging.info(f"  Lags: {lag_list if lag_list else 'None'}")
        
        try: 
            data_missing, sim_mask = apply_test_missingness(data, target_column)
            
            # Temporarily modify model to use specific lags
            # (This requires modifying the model or passing lag config)
            data_imputed = impute_mice(
                data_missing,
                target_column,
                available_features,
                site_name=site_name,
                model_name=f"LightGBM_{site_name}",
                lag_config=lag_list  # Pass lag configuration
            )
            
            metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
            if metrics:
                results.append({
                    'Lag_Config': lag_name,
                    'Lags': str(lag_list),
                    'N_Lags': len(lag_list),
                    'RMSE': metrics.get('Root Mean Squared Error (RMSE)', np.nan),
                    'R': metrics.get('Correlation Coefficient (R)', np.nan),
                    'NSE': metrics.get('Nash-Sutcliffe Efficiency (NSE)', np.nan),
                })
                logging.info(f"  RMSE: {results[-1]['RMSE']:.3f}, R: {results[-1]['R']:.3f}")
        except Exception as e:
            logging.exception(f"  Failed testing lag config: {lag_name}")
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logging.warning(f"No successful lag configuration results for site={site_name}")
        cols = ['Lag_Config', 'Lags', 'N_Lags', 'RMSE', 'R', 'NSE']
        results_df = pd.DataFrame(columns=cols)
        results_df.to_csv(OUTPUT_DIR / f"{site_name}_lag_configurations.csv", index=False)
        return results_df

    results_df = results_df.sort_values('RMSE')
    results_df.to_csv(OUTPUT_DIR / f"{site_name}_lag_configurations.csv", index=False)
    
    plot_lag_configurations(results_df, site_name)
    
    return results_df


def plot_lag_configurations(results_df, site_name):
    """Visualize lag configuration performance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE vs Number of Lags
    ax = axes[0]
    ax.scatter(results_df['N_Lags'], results_df['RMSE'], s=100, alpha=0.6, edgecolors='black')
    
    for idx, row in results_df.iterrows():
        ax.annotate(row['Lag_Config'], (row['N_Lags'], row['RMSE']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Number of Lag Features', fontweight='bold')
    ax.set_ylabel('RMSE (μg/m³)', fontweight='bold')
    ax.set_title('Lag Configuration Impact on RMSE', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Bar chart of configurations
    ax = axes[1]
    results_sorted = results_df.sort_values('RMSE')
    colors = plt.cm.RdYlGn_r(np. linspace(0.2, 0.8, len(results_sorted)))
    
    bars = ax.barh(results_sorted['Lag_Config'], results_sorted['RMSE'], 
                   color=colors, edgecolor='black')
    ax.set_xlabel('RMSE (μg/m³)', fontweight='bold')
    ax.set_title('Lag Configuration Ranking', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, results_sorted['RMSE']):
        ax.text(val, bar. get_y() + bar.get_height()/2, f' {val:.2f}',
               ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{site_name}_lag_configurations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"✅ Plot saved: {OUTPUT_DIR / f'{site_name}_lag_configurations.png'}")


# ============================================================================
# ANALYSIS 4: ROLLING WINDOW ANALYSIS
# ============================================================================

def analyze_rolling_configurations(site_name, data, target_column='PM2.5'):
    """
    Test different rolling window configurations
    """
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS 4: ROLLING WINDOW CONFIGURATION ANALYSIS")
    logging.info("="*80)
    
    available_features = [col for col in config.INPUT_COLUMNS if col in data.columns and col != target_column]
    
    results = []
    
    for roll_name, roll_config in ROLLING_CONFIGURATIONS.items():
        logging.info(f"\nTesting rolling configuration: {roll_name}")
        logging.info(f"  Windows: {roll_config if roll_config else 'None'}")
        
        try: 
            data_missing, sim_mask = apply_test_missingness(data, target_column)
            
            data_imputed = impute_mice(
                data_missing,
                target_column,
                available_features,
                site_name=site_name,
                model_name=f"LightGBM_{site_name}",
                rolling_config=roll_config
            )
            
            metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
            if metrics: 
                results.append({
                    'Rolling_Config': roll_name,
                    'Windows':  str(roll_config),
                    'N_Windows': len(roll_config),
                    'RMSE': metrics.get('Root Mean Squared Error (RMSE)', np.nan),
                    'R': metrics.get('Correlation Coefficient (R)', np.nan),
                    'NSE': metrics.get('Nash-Sutcliffe Efficiency (NSE)', np.nan),
                })
                logging.info(f"  RMSE: {results[-1]['RMSE']:.3f}, R: {results[-1]['R']:.3f}")
        except Exception as e: 
            logging.exception(f"  Failed testing rolling config: {roll_name}")
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logging.warning(f"No successful rolling configuration results for site={site_name}")
        cols = ['Rolling_Config', 'Windows', 'N_Windows', 'RMSE', 'R', 'NSE']
        results_df = pd.DataFrame(columns=cols)
        results_df.to_csv(OUTPUT_DIR / f"{site_name}_rolling_configurations.csv", index=False)
        return results_df

    results_df = results_df.sort_values('RMSE')
    results_df.to_csv(OUTPUT_DIR / f"{site_name}_rolling_configurations.csv", index=False)
    
    plot_rolling_configurations(results_df, site_name)
    
    return results_df


def plot_rolling_configurations(results_df, site_name):
    """Visualize rolling window performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    results_sorted = results_df.sort_values('RMSE')
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))
    
    bars = ax.barh(results_sorted['Rolling_Config'], results_sorted['RMSE'], 
                   color=colors, edgecolor='black')
    ax.set_xlabel('RMSE (μg/m³)', fontweight='bold')
    ax.set_title(f'Rolling Window Configuration Performance: {site_name}', 
                fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val, n_win in zip(bars, results_sorted['RMSE'], results_sorted['N_Windows']):
     ax.text(val, bar.get_y() + bar.get_height()/2,
         f'  {val:.2f} ({n_win} windows)',
         ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{site_name}_rolling_configurations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"✅ Plot saved: {OUTPUT_DIR / f'{site_name}_rolling_configurations.png'}")


# ============================================================================
# ANALYSIS 5: COMPREHENSIVE SUMMARY & RECOMMENDATIONS
# ============================================================================

def generate_recommendations(site_name, individual_df, groups_df, lags_df, rolling_df):
    """
    Generate comprehensive recommendations based on all analyses
    """
    logging.info("\n" + "="*80)
    logging.info("GENERATING RECOMMENDATIONS")
    logging.info("="*80)
    
    report = []
    report.append("="*80)
    report.append(f"FEATURE SELECTION ANALYSIS SUMMARY:  {site_name}")
    report.append("="*80)
    report.append("")

    # If any analysis produced no results, write an informative incomplete-analysis note and exit.
    if individual_df is None or groups_df is None or lags_df is None or rolling_df is None:
        logging.warning("One or more analysis dataframes are None; cannot generate full recommendations.")
        with open(OUTPUT_DIR / f"{site_name}_recommendations.txt", "w") as f:
            f.write("INCOMPLETE ANALYSIS: One or more analyses did not produce results.\n")
        return
    if individual_df.empty or groups_df.empty or lags_df.empty or rolling_df.empty:
        logging.warning("One or more analysis dataframes are empty; cannot generate full recommendations.")
        with open(OUTPUT_DIR / f"{site_name}_recommendations.txt", "w") as f:
            f.write("INCOMPLETE ANALYSIS: Insufficient successful runs to generate recommendations.\n")
        logging.info(f"✅ Incomplete recommendations saved: {OUTPUT_DIR / f'{site_name}_recommendations.txt'}")
        return
    
    # 1. Best individual features
    report.append("1.  INDIVIDUAL FEATURE IMPORTANCE")
    report.append("-" * 80)
    
    baseline_rmse = individual_df[individual_df['Feature'] == 'BASELINE (Temporal only)']['RMSE'].values[0]
    all_features_rmse = individual_df[individual_df['Feature'] == 'ALL FEATURES']['RMSE']. values[0]
    
    report.append(f"\nBaseline (Temporal only): RMSE = {baseline_rmse:.3f} μg/m³")
    report.append(f"All Features: RMSE = {all_features_rmse:.3f} μg/m³")
    report.append(f"Improvement: {(baseline_rmse - all_features_rmse)/baseline_rmse*100:.1f}%")
    
    report.append("\nTop 3 Individual Features:")
    top3 = individual_df[~individual_df['Feature'].isin(['BASELINE (Temporal only)', 'ALL FEATURES'])].head(3)
    for idx, row in top3.iterrows():
        improvement = (baseline_rmse - row['RMSE'])/baseline_rmse*100
        report.append(f"  {idx+1}. {row['Feature']}:  RMSE = {row['RMSE']:.3f} ({improvement:+.1f}% vs baseline)")
    
    # 2. Best feature group
    report.append("\n2. BEST FEATURE GROUP")
    report.append("-" * 80)
    best_group = groups_df. iloc[0]
    report.append(f"\nOptimal Group: {best_group['Group']}")
    report.append(f"  Features: {best_group['Features']}")
    report.append(f"  RMSE: {best_group['RMSE']:.3f} μg/m³")
    report.append(f"  R: {best_group['R']:.3f}")
    report.append(f"  Number of features: {best_group['N_Features']}")
    
    # 3. Best lag configuration
    report.append("\n3. OPTIMAL LAG CONFIGURATION")
    report.append("-" * 80)
    best_lag = lags_df.iloc[0]
    report.append(f"\nBest Configuration: {best_lag['Lag_Config']}")
    report.append(f"  Lags: {best_lag['Lags']}")
    report.append(f"  RMSE:  {best_lag['RMSE']:.3f} μg/m³")
    
    # 4. Best rolling configuration
    report.append("\n4. OPTIMAL ROLLING WINDOW CONFIGURATION")
    report.append("-" * 80)
    best_roll = rolling_df.iloc[0]
    report.append(f"\nBest Configuration:  {best_roll['Rolling_Config']}")
    report.append(f"  Windows: {best_roll['Windows']}")
    report.append(f"  RMSE: {best_roll['RMSE']:.3f} μg/m³")
    
    # 5. Final recommendations
    report.append("\n5. PRODUCTION RECOMMENDATIONS")
    report.append("="*80)
    
    report.append("\n🎯 RECOMMENDED CONFIGURATION:")
    report.append(f"  Input Features: {best_group['Features']}")
    report.append(f"  Lag Configuration: {best_lag['Lag_Config']}")
    report.append(f"  Rolling Windows: {best_roll['Rolling_Config']}")
    report.append(f"  Expected RMSE: ~{min(best_group['RMSE'], best_lag['RMSE'], best_roll['RMSE']):.2f} μg/m³")
    
    report.append("\n📊 KEY INSIGHTS:")
    
    # Check if more features = better
    if all_features_rmse < baseline_rmse * 0.9:
        report.append("  ✅ Input features significantly improve performance (>10%)")
    else:
        report.append("  ⚠️  Input features provide marginal improvement (<10%)")
        report.append("     Consider using simpler configuration for faster deployment")
    
    # Check complexity vs performance
    if best_group['N_Features'] <= 4 and best_group['RMSE'] < all_features_rmse * 1.05:
        report.append(f"  ✅ Optimal feature group ({best_group['N_Features']} features) performs nearly as well as all features")
        report.append("     Recommend using simplified group for efficiency")
    
    report.append("\n" + "="*80)
    
    # Save report
    report_text = "\n".join(report)
    print(report_text)
    
    with open(OUTPUT_DIR / f"{site_name}_recommendations.txt", "w") as f:
        f.write(report_text)
    
    logging.info(f"✅ Recommendations saved: {OUTPUT_DIR / f'{site_name}_recommendations.txt'}")
    
    # Save JSON config for easy import
    recommended_config = {
        'site':  site_name,
        'input_features': best_group['Features']. split(', '),
        'lag_configuration': best_lag['Lag_Config'],
        'rolling_configuration': best_roll['Rolling_Config'],
        'expected_rmse': float(min(best_group['RMSE'], best_lag['RMSE'], best_roll['RMSE'])),
        'analysis_date': datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / f"{site_name}_optimal_config.json", "w") as f:
        json.dump(recommended_config, f, indent=2)
    
    logging.info(f"✅ Config saved: {OUTPUT_DIR / f'{site_name}_optimal_config.json'}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description="Feature selection analysis")
    parser.add_argument(
        "--mode",
        choices=["legacy", "simple_xgbrf"],
        default="simple_xgbrf",
        help="simple_xgbrf compares compiled vs per-region XGBRF RMSE; legacy runs the old LGBM analysis.",
    )
    default_target = (getattr(config, "TARGET_COLUMNS", None) or ["PM2.5"])[0]
    parser.add_argument(
        "--target",
        default="",
        help="Single target variable to impute/evaluate. Overrides --targets.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_SIMPLE_XGBRF_TARGETS),
        help="Comma-separated target variables for simple_xgbrf heatmaps.",
    )
    parser.add_argument(
        "--regions",
        default="",
        help="Comma-separated region names. Default uses config.TARGET_REGIONS.",
    )
    parser.add_argument(
        "--missing-frac",
        type=float,
        default=0.20,
        help="Fraction of observed target values to hide for RMSE testing.",
    )
    parser.add_argument(
        "--skip-station-heatmaps",
        action="store_true",
        help="Skip variable-by-station heatmaps under each region.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP-based XGBRF feature importance.",
    )
    parser.add_argument(
        "--only-shap",
        action="store_true",
        help="Run only SHAP importance plots (skips all other analyses).",
    )
    parser.add_argument(
        "--shap-all-regimes",
        action="store_true",
        help="When used with --only-shap, run SHAP under random/random_block/last_block regimes (default: random only).",
    )
    parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=5000,
        help="Max holdout samples per region used for SHAP computation.",
    )
    args = parser.parse_args()

    if args.mode == "simple_xgbrf":
        selected_regions = [region.strip() for region in args.regions.split(",") if region.strip()]
        if not selected_regions:
            selected_regions = getattr(config, "TARGET_REGIONS", [])
        target_columns = [target.strip() for target in args.targets.split(",") if target.strip()]
        if args.target:
            target_columns = [args.target.strip()]
        if not target_columns:
            target_columns = [default_target]
        logging.info(
            f"Running simple XGBRF-only heatmaps for targets={', '.join(target_columns)}; "
            "legacy LGBM_AQ_Plus_SpatialIter analysis is not used."
        )
        if args.only_shap:
            regimes = ["random", "random_block", "last_block"] if args.shap_all_regimes else ["random"]
            run_shap_only_multi_target(
                target_columns=target_columns,
                selected_regions=selected_regions,
                missing_frac=args.missing_frac,
                output_dir=REGIONAL_ASSESSMENT_DIR,
                shap_max_samples=args.shap_max_samples,
                regimes=regimes,
            )
            return
        run_simple_xgbrf_multi_target_heatmaps(
            target_columns=target_columns,
            selected_regions=selected_regions,
            missing_frac=args.missing_frac,
            station_heatmaps=not args.skip_station_heatmaps,
            run_shap=not args.skip_shap,
            shap_max_samples=args.shap_max_samples,
        )
        return

    load_legacy_dependencies()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║         COMPREHENSIVE FEATURE SELECTION ANALYSIS                     ║
    ║                                                                      ║
    ║  Analyzes:                                                             ║
    ║    1. Individual feature importance                                  ║
    ║    2. Feature group combinations                                     ║
    ║    3. Lag configuration optimization                                 ║
    ║    4. Rolling window optimization                                    ║
    ║    5. Comprehensive recommendations                                  ║
    ║                                                                      ║
    ║  Model:  LGBM_AQ_Plus_SpatialIter                                    ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get site to analyze
    if config.TARGET_SITES:
        site_name = config.TARGET_SITES[0]
    else:
        sites = get_available_sites_from_regional_inputs()
        site_name = sites[0] if sites else "ARMIDALE"
    
    logging.info(f"Analyzing site: {site_name}")
    
    # Load data
    try:
        data, filepath, column_mapping = load_site_data(site_name)
        logging.info(f"Loaded data:  {len(data)} rows, {len(data.columns)} columns")
        logging.info(f"Available columns: {list(data.columns)}")
    except Exception as e: 
        logging.error(f"Failed to load data: {e}")
        return
    
    # Run analyses
    try:
        # Analysis 1: Individual features
        individual_results = analyze_individual_features(site_name, data)
        
        # Analysis 2: Feature groups
        group_results = analyze_feature_groups(site_name, data)
        
        # Analysis 3: Lag configurations
        lag_results = analyze_lag_configurations(site_name, data)
        
        # Analysis 4: Rolling windows
        rolling_results = analyze_rolling_configurations(site_name, data)
        
        # Generate recommendations
        generate_recommendations(site_name, individual_results, group_results, 
                                lag_results, rolling_results)
        
        logging.info("\n" + "="*80)
        logging.info("✅ FEATURE SELECTION ANALYSIS COMPLETE")
        logging.info("="*80)
        logging.info(f"\n📁 Results saved to: {OUTPUT_DIR}/")
        logging.info(f"\n📊 Files generated:")
        logging.info(f"   - {site_name}_individual_features.csv & .png")
        logging.info(f"   - {site_name}_feature_groups.csv & .png")
        logging.info(f"   - {site_name}_lag_configurations.csv & .png")
        logging.info(f"   - {site_name}_rolling_configurations.csv & .png")
        logging.info(f"   - {site_name}_recommendations.txt")
        logging.info(f"   - {site_name}_optimal_config.json")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
