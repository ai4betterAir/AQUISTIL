"""
Region-wise and Site-wise Feature Selection Analysis Framework

Features:
1. Automatic region -> site mapping from station Excel file
2. Region-level pooled/global feature selection
3. Site-level individual feature selection
4. Combined plot per region-target:
   - left: global regional feature ranking
   - right: site-wise feature heatmap
5. Flexible region selection:
   - TARGET_REGIONS = [] -> all regions
   - TARGET_REGIONS = ["Sydney North-west"] -> selected regions only
6. Flexible target selection:
   - TARGET_COLUMN = "PM2.5" -> only PM2.5
   - TARGET_COLUMN = "" or None -> run all available targets separately

Author: Dr. Masrur / revised by ChatGPT
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# IMPORT PROJECT MODULES
# =============================================================================
from Model.LGBM_AQ_Plus_SpatialIter_Optimized_V2 import impute_mice
from evaluation_metrics import evaluate_metrics
from missingness_regimes import apply_missingness
import config_spatial as config


# =============================================================================
# USER OPTIONS
# =============================================================================

# Empty list = all regions
# Examples:
# TARGET_REGIONS = []
# TARGET_REGIONS = ["Sydney North-west"]
# TARGET_REGIONS = ["Sydney North-west", "Upper Hunter"]
TARGET_REGIONS = ["Upper Hunter"]

# Single target:
# TARGET_COLUMN = "PM2.5"
# All available targets separately:
# TARGET_COLUMN = ""
TARGET_COLUMN = "PM2.5"

# Candidate targets to auto-detect when TARGET_COLUMN is empty
POSSIBLE_TARGET_COLUMNS = [
    "PM2.5",
    "PM10",
    "OZONE",
    "CO",
    "NO",
    "NO2",
    "NOX",
    "SO2",
]

STATION_INFO_FILE = (
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQ_DATA/"
    "AquisNET_Data/Air Quality API Excel Power Query.xlsx"
)

OUTPUT_DIR = Path("feature_selection_results_regionwise")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_TOP_N_FEATURES = 10
MIN_VALID_POINTS = 10


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "feature_selection_analysis_regionwise.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def normalize_name(text):
    """Normalize strings for robust matching."""
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def sanitize_filename(text):
    """Make strings safe for output filenames."""
    return str(text).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def normalize_target_label(target):
    """Safe text for filenames/titles."""
    return str(target).replace(".", "p").replace("/", "_").replace("\\", "_").replace(" ", "_")


def get_datetime_column(df):
    """Create or standardize DateTime column."""
    df = df.copy()

    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        return df

    if "date" in df.columns and "time" in df.columns:
        time_cleaned = df["time"].astype(str).str.replace("24:00:00", "00:00:00", regex=False)
        df["DateTime"] = pd.to_datetime(
            df["date"].astype(str) + " " + time_cleaned,
            errors="coerce"
        )
        return df

    if "Date" in df.columns and "Time" in df.columns:
        time_cleaned = df["Time"].astype(str).str.replace("24:00:00", "00:00:00", regex=False)
        df["DateTime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + time_cleaned,
            errors="coerce"
        )
        return df

    raise ValueError("No supported date/time columns found.")


def normalize_column_names(df, site_name):
    """
    Remove site suffix from columns when present.
    Example: TEMP_ARMIDALE -> TEMP
    """
    site_suffix = site_name.upper().replace("-", "_").replace(" ", "_")
    column_mapping = {}

    for col in df.columns:
        if col.endswith(f"_{site_suffix}"):
            base_col = col[:-len(f"_{site_suffix}")]
            column_mapping[col] = base_col
        else:
            column_mapping[col] = col

    df_normalized = df.rename(columns=column_mapping)
    return df_normalized, column_mapping


def ensure_target_column(df, target_column):
    """
    Ensure requested target exists.
    Handles PM2.5 <-> PM2_5.
    """
    df = df.copy()

    if target_column in df.columns:
        return df

    if target_column == "PM2.5" and "PM2_5" in df.columns:
        df["PM2.5"] = df["PM2_5"]
        return df

    raise ValueError(
        f"Target column '{target_column}' not found. "
        f"Available columns: {list(df.columns)}"
    )


def get_available_input_features(df, target_column):
    """Return available model input features from config.INPUT_COLUMNS."""
    return [col for col in config.INPUT_COLUMNS if col in df.columns and col != target_column]


def read_station_metadata(station_info_file, target_regions=None):
    """
    Read station metadata Excel and build region/site mapping.
    Required columns:
        - Column1.Site_Id
        - Column1.SiteName
        - Column1.Longitude
        - Column1.Latitude
        - Column1.Region
    """
    df = pd.read_excel(station_info_file)

    required_cols = [
        "Column1.Site_Id",
        "Column1.SiteName",
        "Column1.Longitude",
        "Column1.Latitude",
        "Column1.Region",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in station metadata file: {missing}")

    df = df[required_cols].copy()
    df = df.dropna(subset=["Column1.SiteName", "Column1.Region"])

    df["Column1.SiteName"] = df["Column1.SiteName"].astype(str).str.strip()
    df["Column1.Region"] = df["Column1.Region"].astype(str).str.strip()

    if target_regions:
        target_regions_clean = [str(r).strip() for r in target_regions]
        df = df[df["Column1.Region"].isin(target_regions_clean)].copy()

    df = df.drop_duplicates(subset=["Column1.SiteName", "Column1.Region"])

    region_sites = (
        df.groupby("Column1.Region")["Column1.SiteName"]
        .apply(lambda x: sorted(x.unique().tolist()))
        .to_dict()
    )

    site_region = dict(zip(df["Column1.SiteName"], df["Column1.Region"]))

    return df, region_sites, site_region


def load_site_data(site_name, target_column=None):
    """
    Load CSV for a specific site from config.INPUT_DIRECTORY.
    Uses flexible filename matching.

    If target_column is provided and non-empty, validate it exists.
    If target_column is empty/None, only load the data.
    """
    site_key = normalize_name(site_name)
    candidate_files = []

    for filename in os.listdir(config.INPUT_DIRECTORY):
        if not filename.lower().endswith(".csv"):
            continue

        file_stem = os.path.splitext(filename)[0]
        file_key = normalize_name(file_stem)

        if file_key == site_key or file_key.startswith(site_key) or site_key in file_key:
            candidate_files.append(filename)

    if not candidate_files:
        raise FileNotFoundError(f"No data file found for site: {site_name}")

    candidate_files = sorted(candidate_files, key=len)
    filepath = os.path.join(config.INPUT_DIRECTORY, candidate_files[0])

    df = pd.read_csv(filepath)
    df = get_datetime_column(df)
    df = df.dropna(subset=["DateTime"]).copy()
    df, column_mapping = normalize_column_names(df, site_name)

    if target_column is not None and str(target_column).strip() != "":
        df = ensure_target_column(df, target_column)

    return df, filepath, column_mapping


def apply_test_missingness(data, target_column, regime="random", frac=0.2, seed=42):
    """Apply simulated missingness for evaluation."""
    original_missing = data[target_column].isna()

    data_with_missing, simulated_mask = apply_missingness(
        data,
        target_column,
        regime=regime,
        frac=frac,
        seed=seed,
    )
    simulated_mask = simulated_mask & (~original_missing)
    return data_with_missing, simulated_mask


def evaluate_imputation(data_original, data_imputed, simulated_mask, target_column):
    """
    Evaluate imputation robustly for DataFrame/Series/ndarray outputs.
    """
    try:
        if isinstance(simulated_mask, pd.Series):
            mask = simulated_mask.reindex(data_original.index).fillna(False).astype(bool)
        else:
            arr = np.asarray(simulated_mask)
            if arr.ndim > 1:
                arr = arr.squeeze()
            if arr.shape[0] != len(data_original):
                mask = pd.Series(simulated_mask).reindex(data_original.index).fillna(False).astype(bool)
            else:
                mask = pd.Series(arr, index=data_original.index).astype(bool)
    except Exception:
        logging.exception("Failed to align simulated mask with original data index.")
        return None

    try:
        true_values = data_original.loc[mask, target_column].values
    except Exception:
        logging.exception("Failed to select true values using mask.")
        return None

    try:
        if isinstance(data_imputed, pd.DataFrame):
            if target_column not in data_imputed.columns:
                logging.error(
                    f"Target column '{target_column}' not found in imputed DataFrame. "
                    f"Columns: {list(data_imputed.columns)}"
                )
                return None
            try:
                imputed_values = data_imputed.loc[mask, target_column].values
            except AssertionError:
                positions = np.where(mask)[0]
                imputed_values = data_imputed.iloc[positions][target_column].values

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
                imputed_values = data_imputed[positions, -1]
            else:
                logging.error(f"Unsupported ndarray output shape: {data_imputed.shape}")
                return None

        else:
            df_imp = pd.DataFrame(data_imputed)
            if target_column in df_imp.columns:
                imputed_values = df_imp.loc[mask, target_column].values
            else:
                positions = np.where(mask)[0]
                imputed_values = df_imp.iloc[positions].values.squeeze()

    except Exception:
        logging.exception("Failed to extract imputed values.")
        return None

    valid_mask = np.isfinite(true_values) & np.isfinite(imputed_values)

    if getattr(config, "HANDLE_NEGATIVES", None) == "exclude":
        valid_mask = valid_mask & (true_values >= 0) & (imputed_values >= 0)

    if valid_mask.sum() < MIN_VALID_POINTS:
        logging.warning(f"Too few valid points for evaluation: {valid_mask.sum()}")
        return None

    true_clean = true_values[valid_mask]
    imputed_clean = imputed_values[valid_mask]

    metrics = evaluate_metrics(
        true_clean,
        imputed_clean,
        handle_negative=getattr(config, "HANDLE_NEGATIVES", None)
    )
    return metrics


def run_imputation(
    data,
    target_column,
    input_features,
    site_name,
    model_name,
    lag_config=None,
    rolling_config=None,
):
    """
    Wrapper around impute_mice.
    Falls back safely if lag_config / rolling_config are unsupported.
    """
    kwargs = {
        "site_name": site_name,
        "model_name": model_name,
    }

    if lag_config is not None:
        kwargs["lag_config"] = lag_config
    if rolling_config is not None:
        kwargs["rolling_config"] = rolling_config

    try:
        return impute_mice(
            data,
            target_column,
            input_features,
            **kwargs,
        )
    except TypeError:
        logging.warning(
            "impute_mice does not accept lag_config / rolling_config in current version. "
            "Retrying without optional args."
        )
        return impute_mice(
            data,
            target_column,
            input_features,
            site_name=site_name,
            model_name=model_name,
        )


def get_target_columns_to_run(site_dfs, requested_target):
    """
    Determine which targets to run.

    Rules:
    - If requested_target is non-empty -> use only that target
    - If requested_target is empty/None -> use all available targets found in loaded sites
    """
    if requested_target is not None and str(requested_target).strip() != "":
        return [str(requested_target).strip()]

    available_targets = set()

    for _, df in site_dfs.items():
        for target in POSSIBLE_TARGET_COLUMNS:
            if target in df.columns:
                available_targets.add(target)
            elif target == "PM2.5" and "PM2_5" in df.columns:
                available_targets.add("PM2.5")

    return sorted(list(available_targets))


def get_common_features_for_target(site_dfs, target_column):
    """
    Get common input features across all sites for a specific target.
    Only sites that contain the target are used in the intersection.
    """
    common_features = None
    valid_site_count = 0

    for _, df in site_dfs.items():
        try:
            df_tmp = ensure_target_column(df, target_column)
        except Exception:
            continue

        available_features = get_available_input_features(df_tmp, target_column)

        if common_features is None:
            common_features = set(available_features)
        else:
            common_features &= set(available_features)

        valid_site_count += 1

    if valid_site_count == 0:
        return []

    return sorted(list(common_features)) if common_features else []


def filter_site_dfs_for_target(site_dfs, target_column):
    """
    Keep only sites that contain the requested target.
    Also harmonize PM2.5 / PM2_5 when needed.
    """
    filtered = {}

    for site_name, df in site_dfs.items():
        try:
            df_tmp = ensure_target_column(df, target_column)
            filtered[site_name] = df_tmp
        except Exception:
            logging.warning(f"Skipping site '{site_name}' for target '{target_column}' because target is unavailable.")

    return filtered


# =============================================================================
# SITE-LEVEL ANALYSIS
# =============================================================================
def analyze_individual_features(site_name, data, target_column="PM2.5", save_outputs=True, output_prefix=None):
    """
    Test each feature individually for a single site.
    """
    logging.info("\n" + "=" * 80)
    logging.info(f"ANALYSIS 1: INDIVIDUAL FEATURE IMPORTANCE | SITE = {site_name} | TARGET = {target_column}")
    logging.info("=" * 80)

    data = ensure_target_column(data, target_column)
    available_features = get_available_input_features(data, target_column)
    results = []

    # Baseline
    logging.info("Testing baseline (temporal only)")
    try:
        data_missing, sim_mask = apply_test_missingness(data, target_column)
        data_imputed = run_imputation(
            data_missing,
            target_column,
            [],
            site_name=site_name,
            model_name=f"LightGBM_{sanitize_filename(site_name)}_{normalize_target_label(target_column)}"
        )
        metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                "Feature": "BASELINE (Temporal only)",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
            })
    except Exception:
        logging.exception(f"Baseline failed for site {site_name}, target {target_column}")

    # Individual features
    for feature in available_features:
        logging.info(f"Testing site={site_name}, target={target_column}, feature={feature}")
        try:
            data_missing, sim_mask = apply_test_missingness(data, target_column)
            data_imputed = run_imputation(
                data_missing,
                target_column,
                [feature],
                site_name=site_name,
                model_name=f"LightGBM_{sanitize_filename(site_name)}_{normalize_target_label(target_column)}"
            )
            metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
            if metrics:
                results.append({
                    "Feature": feature,
                    "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                    "R": metrics.get("Correlation Coefficient (R)", np.nan),
                    "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                    "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
                })
        except Exception:
            logging.exception(f"Feature '{feature}' failed for site {site_name}, target {target_column}")

    # All features
    logging.info(f"Testing site={site_name}, target={target_column}, ALL FEATURES")
    try:
        data_missing, sim_mask = apply_test_missingness(data, target_column)
        data_imputed = run_imputation(
            data_missing,
            target_column,
            available_features,
            site_name=site_name,
            model_name=f"LightGBM_{sanitize_filename(site_name)}_{normalize_target_label(target_column)}"
        )
        metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                "Feature": "ALL FEATURES",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
            })
    except Exception:
        logging.exception(f"ALL FEATURES failed for site {site_name}, target {target_column}")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        logging.warning(f"No successful individual feature results for site {site_name}, target {target_column}")
        return results_df

    results_df = results_df.sort_values("RMSE").reset_index(drop=True)

    if save_outputs:
        prefix = output_prefix or f"{sanitize_filename(site_name)}_{normalize_target_label(target_column)}"
        results_df.to_csv(OUTPUT_DIR / f"{prefix}_individual_features.csv", index=False)
        plot_individual_features(results_df, site_name, target_column, prefix)

    return results_df


def plot_individual_features(results_df, site_name, target_column, output_prefix):
    """Plot site-level individual feature performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RMSE
    ax = axes[0]
    df_rmse = results_df.sort_values("RMSE")
    colors = [
        "red" if x == "BASELINE (Temporal only)" else
        "darkgreen" if x == "ALL FEATURES" else
        "steelblue"
        for x in df_rmse["Feature"]
    ]
    bars = ax.barh(df_rmse["Feature"], df_rmse["RMSE"], color=colors, edgecolor="black")
    ax.set_xlabel("RMSE (μg/m³)", fontweight="bold")
    ax.set_title(f"{site_name} - {target_column}\nFeature RMSE", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, df_rmse["RMSE"]):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                va="center", ha="left", fontsize=8, fontweight="bold")

    # R
    ax = axes[1]
    df_r = results_df.sort_values("R", ascending=False)
    colors = [
        "red" if x == "BASELINE (Temporal only)" else
        "darkgreen" if x == "ALL FEATURES" else
        "steelblue"
        for x in df_r["Feature"]
    ]
    bars = ax.barh(df_r["Feature"], df_r["R"], color=colors, edgecolor="black")
    ax.set_xlabel("Correlation Coefficient (R)", fontweight="bold")
    ax.set_title(f"{site_name} - {target_column}\nFeature Correlation", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, df_r["R"]):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", ha="left", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_prefix}_individual_features.png", dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# REGION-LEVEL HELPERS
# =============================================================================
def load_region_data(site_list, target_column=None):
    """
    Load all valid site data for a region.
    If target_column is provided, validate it.
    If target_column is empty/None, just load the data.
    """
    site_dfs = {}
    metadata_rows = []

    for site_name in site_list:
        try:
            df, filepath, column_mapping = load_site_data(site_name, target_column=target_column)
            df = df.copy()
            df["Site"] = site_name

            site_dfs[site_name] = df

            metadata_rows.append({
                "Site": site_name,
                "Filepath": filepath,
                "N_Rows": len(df),
                "N_Columns": len(df.columns),
                "Columns": ", ".join(df.columns),
            })

            logging.info(f"Loaded site '{site_name}' | rows={len(df)}")

        except Exception:
            logging.exception(f"Failed to load site data for '{site_name}'")

    metadata_df = pd.DataFrame(metadata_rows)
    return site_dfs, metadata_df


def analyze_region_global_features(region_name, site_dfs, common_features, target_column="PM2.5", output_prefix=None):
    """
    Pooled region-wide global feature selection using common features.
    """
    logging.info("\n" + "=" * 80)
    logging.info(f"REGION GLOBAL FEATURE ANALYSIS | REGION = {region_name} | TARGET = {target_column}")
    logging.info("=" * 80)

    pooled_frames = []
    for site_name, df in site_dfs.items():
        try:
            df_tmp = ensure_target_column(df, target_column)
        except Exception:
            continue

        keep_cols = ["DateTime", target_column, "Site"] + common_features
        keep_cols = [c for c in keep_cols if c in df_tmp.columns]
        pooled_frames.append(df_tmp[keep_cols].copy())

    if not pooled_frames:
        logging.warning(f"No pooled data available for region {region_name}, target {target_column}")
        return pd.DataFrame()

    pooled_df = pd.concat(pooled_frames, axis=0, ignore_index=True)
    pooled_df = pooled_df.dropna(subset=["DateTime"]).copy()

    logging.info(
        f"Pooled regional data for '{region_name}', target '{target_column}': "
        f"rows={len(pooled_df)}, common_features={len(common_features)}"
    )

    results = []

    # Baseline
    try:
        data_missing, sim_mask = apply_test_missingness(pooled_df, target_column)
        data_imputed = run_imputation(
            data_missing,
            target_column,
            [],
            site_name=region_name,
            model_name=f"LightGBM_REGION_{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
        )
        metrics = evaluate_imputation(pooled_df, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                "Region": region_name,
                "Feature": "BASELINE (Temporal only)",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
            })
    except Exception:
        logging.exception(f"Regional baseline failed for {region_name}, target {target_column}")

    # Individual common features
    for feature in common_features:
        try:
            data_missing, sim_mask = apply_test_missingness(pooled_df, target_column)
            data_imputed = run_imputation(
                data_missing,
                target_column,
                [feature],
                site_name=region_name,
                model_name=f"LightGBM_REGION_{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
            )
            metrics = evaluate_imputation(pooled_df, data_imputed, sim_mask, target_column)
            if metrics:
                results.append({
                    "Region": region_name,
                    "Feature": feature,
                    "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                    "R": metrics.get("Correlation Coefficient (R)", np.nan),
                    "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                    "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
                })
        except Exception:
            logging.exception(f"Regional feature '{feature}' failed for {region_name}, target {target_column}")

    # All common features
    try:
        data_missing, sim_mask = apply_test_missingness(pooled_df, target_column)
        data_imputed = run_imputation(
            data_missing,
            target_column,
            common_features,
            site_name=region_name,
            model_name=f"LightGBM_REGION_{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
        )
        metrics = evaluate_imputation(pooled_df, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                "Region": region_name,
                "Feature": "ALL FEATURES",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
            })
    except Exception:
        logging.exception(f"Regional ALL FEATURES failed for {region_name}, target {target_column}")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logging.warning(f"No global feature results for region {region_name}, target {target_column}")
        return results_df

    results_df = results_df.sort_values("RMSE").reset_index(drop=True)

    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    results_df.to_csv(OUTPUT_DIR / f"{prefix}_global_individual_features.csv", index=False)

    return results_df


def analyze_region_sites(region_name, site_dfs, target_column="PM2.5", output_prefix=None):
    """
    Run individual site analyses for all sites in the region and combine outputs.
    """
    logging.info("\n" + "=" * 80)
    logging.info(f"REGION SITE-WISE FEATURE ANALYSIS | REGION = {region_name} | TARGET = {target_column}")
    logging.info("=" * 80)

    all_site_results = []
    site_summary_rows = []

    for site_name, df in site_dfs.items():
        try:
            df_tmp = ensure_target_column(df, target_column)

            site_result = analyze_individual_features(
                site_name=site_name,
                data=df_tmp,
                target_column=target_column,
                save_outputs=True,
                output_prefix=f"{sanitize_filename(region_name)}_{sanitize_filename(site_name)}_{normalize_target_label(target_column)}"
            )

            if site_result.empty:
                continue

            site_result = site_result.copy()
            site_result["Site"] = site_name
            site_result["Region"] = region_name
            site_result["Target"] = target_column
            all_site_results.append(site_result)

            non_special = site_result[
                ~site_result["Feature"].isin(["BASELINE (Temporal only)", "ALL FEATURES"])
            ].copy()

            baseline_rows = site_result[site_result["Feature"] == "BASELINE (Temporal only)"]
            baseline_rmse = baseline_rows["RMSE"].iloc[0] if not baseline_rows.empty else np.nan

            if not non_special.empty:
                best_row = non_special.sort_values("RMSE").iloc[0]
                improvement = (
                    (baseline_rmse - best_row["RMSE"]) / baseline_rmse * 100.0
                    if pd.notna(baseline_rmse) and baseline_rmse != 0
                    else np.nan
                )
                site_summary_rows.append({
                    "Region": region_name,
                    "Target": target_column,
                    "Site": site_name,
                    "Best_Feature": best_row["Feature"],
                    "Best_RMSE": best_row["RMSE"],
                    "Best_R": best_row["R"],
                    "Baseline_RMSE": baseline_rmse,
                    "Improvement_vs_Baseline_percent": improvement,
                })

        except Exception:
            logging.exception(f"Site-wise feature analysis failed for site {site_name}, target {target_column}")

    if not all_site_results:
        return pd.DataFrame(), pd.DataFrame()

    combined_df = pd.concat(all_site_results, ignore_index=True)
    summary_df = pd.DataFrame(site_summary_rows)

    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    combined_df.to_csv(OUTPUT_DIR / f"{prefix}_all_sites_individual_features.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / f"{prefix}_site_best_feature_summary.csv", index=False)

    return combined_df, summary_df


# =============================================================================
# REGION PLOTTING
# =============================================================================
def plot_region_global_vs_sites(region_name, target_column, global_df, site_df, top_n=10, output_prefix=None):
    """
    Two-panel plot:
      Left  - global regional feature ranking
      Right - site-feature RMSE heatmap for top regional features
    """
    if global_df.empty:
        logging.warning(f"Skipping combined region plot: global_df empty for {region_name}, target {target_column}")
        return

    global_plot_df = global_df[
        ~global_df["Feature"].isin(["BASELINE (Temporal only)", "ALL FEATURES"])
    ].copy()

    if global_plot_df.empty:
        logging.warning(f"Skipping combined region plot: no normal features for {region_name}, target {target_column}")
        return

    global_plot_df = global_plot_df.sort_values("RMSE").head(top_n)
    top_features = global_plot_df["Feature"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1, 1.25]})

    # Left panel: Global pooled ranking
    ax = axes[0]
    bars = ax.barh(global_plot_df["Feature"], global_plot_df["RMSE"], edgecolor="black")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (μg/m³)", fontweight="bold")
    ax.set_title(f"{region_name} | {target_column}\nGlobal Regional Feature Selection", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, global_plot_df["RMSE"]):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                va="center", ha="left", fontsize=8, fontweight="bold")

    # Right panel: Site-wise heatmap
    ax = axes[1]
    if site_df.empty:
        ax.text(0.5, 0.5, "No site-wise results available", ha="center", va="center")
        ax.set_axis_off()
    else:
        heat_df = site_df[site_df["Feature"].isin(top_features)].pivot_table(
            index="Site",
            columns="Feature",
            values="RMSE",
            aggfunc="mean"
        )

        if not heat_df.empty:
            available_top_features = [f for f in top_features if f in heat_df.columns]
            heat_df = heat_df[available_top_features]
            sns.heatmap(
                heat_df,
                annot=True,
                fmt=".2f",
                cmap="viridis_r",
                cbar_kws={"label": "RMSE (μg/m³)"},
                ax=ax
            )
            ax.set_title(f"{region_name} | {target_column}\nSite-wise Feature Performance", fontweight="bold")
            ax.set_xlabel("Feature", fontweight="bold")
            ax.set_ylabel("Site", fontweight="bold")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(0.5, 0.5, "No matching site-feature results", ha="center", va="center")
            ax.set_axis_off()

    plt.tight_layout()
    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    plt.savefig(OUTPUT_DIR / f"{prefix}_global_vs_sites.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_region_site_best_features(region_name, target_column, site_summary_df, output_prefix=None):
    """
    Plot best feature per site in region.
    """
    if site_summary_df.empty:
        return

    df = site_summary_df.sort_values("Best_RMSE").copy()

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(df))))

    bars = ax.barh(df["Site"], df["Best_RMSE"], edgecolor="black")
    ax.invert_yaxis()
    ax.set_xlabel("Best RMSE (μg/m³)", fontweight="bold")
    ax.set_title(f"{region_name} | {target_column}\nBest Feature per Site", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, (_, row) in zip(bars, df.iterrows()):
        label = f"{row['Best_Feature']} | {row['Best_RMSE']:.2f}"
        ax.text(row["Best_RMSE"], bar.get_y() + bar.get_height() / 2, f" {label}",
                va="center", ha="left", fontsize=8, fontweight="bold")

    plt.tight_layout()
    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    plt.savefig(OUTPUT_DIR / f"{prefix}_best_feature_per_site.png", dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# REGION REPORTING
# =============================================================================
def generate_region_summary(region_name, target_column, global_df, site_summary_df, common_features, output_prefix=None):
    """
    Save a simple human-readable summary for each region-target.
    """
    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    out_txt = OUTPUT_DIR / f"{prefix}_summary.txt"

    lines = []
    lines.append("=" * 90)
    lines.append(f"REGION FEATURE SELECTION SUMMARY: {region_name} | TARGET: {target_column}")
    lines.append("=" * 90)
    lines.append("")

    lines.append(f"Number of common features across region: {len(common_features)}")
    lines.append(f"Common features: {', '.join(common_features) if common_features else 'None'}")
    lines.append("")

    if global_df.empty:
        lines.append("No global regional results available.")
    else:
        baseline = global_df[global_df["Feature"] == "BASELINE (Temporal only)"]
        all_feat = global_df[global_df["Feature"] == "ALL FEATURES"]
        normal = global_df[
            ~global_df["Feature"].isin(["BASELINE (Temporal only)", "ALL FEATURES"])
        ].copy()

        if not baseline.empty:
            lines.append(f"Baseline RMSE: {baseline['RMSE'].iloc[0]:.3f}")
        if not all_feat.empty:
            lines.append(f"All-features RMSE: {all_feat['RMSE'].iloc[0]:.3f}")

        if not normal.empty:
            best_global = normal.sort_values("RMSE").iloc[0]
            lines.append("")
            lines.append("Best global feature:")
            lines.append(f"  Feature: {best_global['Feature']}")
            lines.append(f"  RMSE: {best_global['RMSE']:.3f}")
            lines.append(f"  R: {best_global['R']:.3f}")

            lines.append("")
            lines.append("Top global features:")
            for _, row in normal.sort_values("RMSE").head(10).iterrows():
                lines.append(f"  - {row['Feature']}: RMSE={row['RMSE']:.3f}, R={row['R']:.3f}")

    lines.append("")
    lines.append("-" * 90)

    if site_summary_df.empty:
        lines.append("No site-wise summary available.")
    else:
        lines.append("Best feature by site:")
        for _, row in site_summary_df.sort_values("Best_RMSE").iterrows():
            lines.append(
                f"  - {row['Site']}: {row['Best_Feature']} | "
                f"Best_RMSE={row['Best_RMSE']:.3f} | "
                f"Improvement_vs_Baseline={row['Improvement_vs_Baseline_percent']:.2f}%"
            )

    lines.append("")
    lines.append("=" * 90)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# REGION WORKFLOW
# =============================================================================
def process_single_region(region_name, site_list, target_column=""):
    """
    Full workflow for one region.
    If target_column is empty, run all available targets separately.
    """
    logging.info("\n" + "#" * 100)
    logging.info(f"PROCESSING REGION: {region_name}")
    logging.info("#" * 100)

    # Load once without forcing a target
    site_dfs, metadata_df = load_region_data(site_list, target_column=None)

    if metadata_df is not None and not metadata_df.empty:
        metadata_df.to_csv(
            OUTPUT_DIR / f"{sanitize_filename(region_name)}_site_loading_metadata.csv",
            index=False
        )

    if not site_dfs:
        logging.warning(f"No valid site data loaded for region {region_name}")
        return []

    targets_to_run = get_target_columns_to_run(site_dfs, target_column)

    if not targets_to_run:
        logging.warning(f"No valid targets found for region {region_name}")
        return []

    logging.info(f"Targets to run for region '{region_name}': {targets_to_run}")

    processed_targets = []

    for current_target in targets_to_run:
        logging.info("-" * 80)
        logging.info(f"Running target '{current_target}' for region '{region_name}'")
        logging.info("-" * 80)

        # Keep only sites that actually have the target
        site_dfs_target = filter_site_dfs_for_target(site_dfs, current_target)

        if not site_dfs_target:
            logging.warning(f"No sites with target '{current_target}' in region '{region_name}'")
            continue

        common_features = get_common_features_for_target(site_dfs_target, current_target)

        if not common_features:
            logging.warning(
                f"No common features found across sites in region {region_name} for target {current_target}"
            )
            continue

        prefix = f"{sanitize_filename(region_name)}_{normalize_target_label(current_target)}"

        global_df = analyze_region_global_features(
            region_name=region_name,
            site_dfs=site_dfs_target,
            common_features=common_features,
            target_column=current_target,
            output_prefix=prefix
        )

        site_df, site_summary_df = analyze_region_sites(
            region_name=region_name,
            site_dfs=site_dfs_target,
            target_column=current_target,
            output_prefix=prefix
        )

        plot_region_global_vs_sites(
            region_name=region_name,
            target_column=current_target,
            global_df=global_df,
            site_df=site_df,
            top_n=PLOT_TOP_N_FEATURES,
            output_prefix=prefix
        )

        plot_region_site_best_features(
            region_name=region_name,
            target_column=current_target,
            site_summary_df=site_summary_df,
            output_prefix=prefix
        )

        generate_region_summary(
            region_name=region_name,
            target_column=current_target,
            global_df=global_df,
            site_summary_df=site_summary_df,
            common_features=common_features,
            output_prefix=prefix
        )

        processed_targets.append(current_target)

    logging.info(f"Completed region: {region_name}")
    return processed_targets


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REGION-WISE FEATURE SELECTION ANALYSIS                   ║
║                                                                              ║
║  This workflow performs:                                                     ║
║    1. Automatic region -> site extraction from station Excel                 ║
║    2. Region-level pooled global feature selection                           ║
║    3. Site-level individual feature selection                                ║
║    4. Combined plots: global regional + all sites                            ║
║                                                                              ║
║  Region selection:                                                           ║
║    TARGET_REGIONS = []                    -> all regions                     ║
║    TARGET_REGIONS = ["Sydney North-west"] -> selected region(s)              ║
║                                                                              ║
║  Target selection:                                                           ║
║    TARGET_COLUMN = "PM2.5"              -> single target                     ║
║    TARGET_COLUMN = ""                    -> all available targets separately  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
    )

    try:
        station_df, region_sites, site_region = read_station_metadata(
            STATION_INFO_FILE,
            target_regions=TARGET_REGIONS
        )
    except Exception as e:
        logging.exception(f"Failed to read station metadata file: {e}")
        return

    if station_df.empty or not region_sites:
        logging.error("No stations/regions found after filtering.")
        return

    station_df.to_csv(OUTPUT_DIR / "station_metadata_filtered.csv", index=False)

    logging.info(f"Total selected regions: {len(region_sites)}")
    for region_name, site_list in region_sites.items():
        logging.info(f"Region: {region_name} | Sites: {len(site_list)}")

    summary_rows = []

    for region_name, site_list in region_sites.items():
        try:
            processed_targets = process_single_region(
                region_name=region_name,
                site_list=site_list,
                target_column=TARGET_COLUMN
            )

            summary_rows.append({
                "Region": region_name,
                "N_Sites": len(site_list),
                "Requested_Target": TARGET_COLUMN if str(TARGET_COLUMN).strip() != "" else "ALL",
                "Processed_Targets": ", ".join(processed_targets) if processed_targets else "",
                "Status": "Success" if processed_targets else "No valid targets/features",
            })

        except Exception:
            logging.exception(f"Region processing failed for {region_name}")
            summary_rows.append({
                "Region": region_name,
                "N_Sites": len(site_list),
                "Requested_Target": TARGET_COLUMN if str(TARGET_COLUMN).strip() != "" else "ALL",
                "Processed_Targets": "",
                "Status": "Failed",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "region_processing_summary.csv", index=False)

    logging.info("\n" + "=" * 100)
    logging.info("ALL REGION PROCESSING COMPLETE")
    logging.info(f"Results saved in: {OUTPUT_DIR.resolve()}")
    logging.info("=" * 100)


if __name__ == "__main__":
    main()