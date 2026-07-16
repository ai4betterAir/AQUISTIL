"""
Region-wise and Site-wise Feature Selection using Random Forest + SHAP

Purpose
-------
This script performs feature selection for air-quality target imputation using a
Random Forest based masked-value reconstruction workflow, with SHAP-based
interpretability at both global (region pooled) and local (site-level) scales.

Main idea
---------
For each target and feature set:
1. Hide a fraction of known target values
2. Train a RandomForestRegressor on rows where target is still observed
3. Predict the hidden target values
4. Compare predictions with the original true values
5. Rank features by imputation skill (RMSE, R, NSE, MAE)
6. Fit RF with all selected/common features and compute SHAP:
   - global SHAP for pooled regional model
   - local SHAP for each site model

Outputs
-------
For each region-target pair:
- global feature selection CSV
- all site feature selection CSV
- site best-feature summary CSV
- global vs local performance figure
- global SHAP CSV
- site SHAP CSV
- global vs local SHAP figure
- summary text

Author: Revised for RF + SHAP feature selection
"""

import json
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from evaluation_metrics import evaluate_metrics
from missingness_regimes import apply_missingness
import config_spatial as config


# =============================================================================
# USER OPTIONS
# =============================================================================

# Examples:
TARGET_REGIONS = []
# TARGET_REGIONS = ["Upper Hunter"]
# TARGET_REGIONS = ["Sydney South-west"]

# Examples:
# TARGET_COLUMN = "PM2.5"
# TARGET_COLUMN = ""
TARGET_COLUMN = "PM2.5"

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
OVERVIEW_JSON_FILE = (
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQ_DATA/"
    "AquisNET_Data/Data_availability_Check/region_and_site_variable_overview.json"
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "feature_selection_results_regionwise_rf_shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_TOP_N_FEATURES = 10
PEARSON_COMBO_MAX_FEATURES = None  # None means use all Pearson-ranked features.
MIN_VALID_POINTS = 10
USE_TIME_FEATURES = "no"  # "yes" or "no"
TIME_FEATURE_COLUMNS = [
    "hour", "hour_sin", "hour_cos", "month_sin",
    "dayofweek", "dayofweek_sin", "dayofweek_cos",
    "dayofyear", "dayofyear_sin", "dayofyear_cos",
    "month", "month_cos", "week_of_year", "is_weekend", "season",
]
SHAP_MAX_SAMPLES = 1000
ENABLE_COMPREHENSIVE_FEATURE_ANALYSIS = True

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

TEMPORAL_CONFIGS = {
    "cyclical_only": ["hour_sin", "hour_cos", "month_sin", "month_cos"],
    "basic_temporal": ["hour", "dayofweek", "month"],
    "extended_temporal": [
        "hour", "dayofweek", "month", "dayofyear",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
    ],
    "seasonal_focus": ["month", "dayofyear", "month_sin", "month_cos"],
    "daily_patterns": ["hour", "dayofweek", "hour_sin", "hour_cos"],
}

LAG_CONFIGS = {
    "short": [1, 3, 6],
    "medium": [1, 6, 12, 24],
    "long": [1, 6, 12, 24, 48],
    "comprehensive": [1, 3, 6, 12, 24, 48, 72],
}

SPATIAL_CONFIGS = {
    "nearby_3": {"max_sites": 3, "max_distance": 25},
    "nearby_5": {"max_sites": 5, "max_distance": 50},
    "regional_all": {"max_sites": 10, "max_distance": 100},
    "extended": {"max_sites": 15, "max_distance": 150},
}

COMBINED_CONFIGS = {
    "minimal": {
        "temporal": "basic_temporal",
        "spatial": "nearby_3",
        "lags": "short",
    },
    "balanced": {
        "temporal": "extended_temporal",
        "spatial": "nearby_5",
        "lags": "medium",
    },
    "comprehensive": {
        "temporal": "extended_temporal",
        "spatial": "regional_all",
        "lags": "comprehensive",
    },
}

CONFIG_RESULT_COLUMNS = [
    "Configuration",
    "Feature_Type",
    "Features",
    "N_Features",
    "RMSE",
    "R",
    "WI",
    "NSE",
    "MAE",
    "Status",
    "Error",
]

GUIDED_COMBINATION_COLUMNS = [
    "Guidance_Method",
    "Region",
    "Target",
    "RowType",
    "RowName",
    "ComboRank",
    "ComboLabel",
    "Features",
    "N_Features",
    "RMSE",
    "R",
    "WI",
    "NSE",
    "MAE",
    "Source_File",
]

RUN_PEARSON_GUIDED_COMBINATIONS = []
RUN_SHAP_GUIDED_COMBINATIONS = []


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "feature_selection_analysis_regionwise_rf_shap.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


# =============================================================================
# HELPERS
# =============================================================================
def normalize_name(text):
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def sanitize_filename(text):
    return str(text).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def normalize_target_label(target):
    return str(target).replace(".", "p").replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_dataframe(df, filename, columns=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    if columns is not None:
        out_df = out_df.reindex(columns=columns)
    out_df.to_csv(OUTPUT_DIR / filename, index=False)


def write_output_manifest(filename="output_manifest.csv"):
    rows = []
    for path in sorted(OUTPUT_DIR.glob("*")):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            rows.append({
                "File": path.name,
                "Path": str(path.resolve()),
                "Size_Bytes": stat.st_size,
                "Modified_Time": pd.to_datetime(stat.st_mtime, unit="s"),
            })
        except Exception:
            logging.debug(f"Failed to add output manifest entry for {path}", exc_info=True)

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(OUTPUT_DIR / filename, index=False)
    return manifest_df


def add_guided_combination_results(store, combo_df, guidance_method, region_name, target_column, source_file=None):
    if combo_df is None or combo_df.empty:
        return

    df = combo_df.copy()
    df["Guidance_Method"] = guidance_method
    if "Region" not in df.columns:
        df["Region"] = region_name
    if "Target" not in df.columns:
        df["Target"] = target_column
    if "Source_File" not in df.columns:
        df["Source_File"] = source_file or ""

    store.append(df.reindex(columns=GUIDED_COMBINATION_COLUMNS))


def save_guided_combination_outputs():
    if RUN_PEARSON_GUIDED_COMBINATIONS:
        pearson_df = pd.concat(RUN_PEARSON_GUIDED_COMBINATIONS, ignore_index=True)
    else:
        pearson_df = pd.DataFrame(columns=GUIDED_COMBINATION_COLUMNS)

    if RUN_SHAP_GUIDED_COMBINATIONS:
        shap_df = pd.concat(RUN_SHAP_GUIDED_COMBINATIONS, ignore_index=True)
    else:
        shap_df = pd.DataFrame(columns=GUIDED_COMBINATION_COLUMNS)

    save_dataframe(
        pearson_df,
        "all_pearson_guided_combinations.csv",
        columns=GUIDED_COMBINATION_COLUMNS,
    )
    save_dataframe(
        shap_df,
        "all_shap_guided_combinations.csv",
        columns=GUIDED_COMBINATION_COLUMNS,
    )

    return pearson_df, shap_df


def use_time_features():
    return str(USE_TIME_FEATURES).strip().lower() in {"yes", "y", "true", "1"}


def get_datetime_column(df):
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
    site_key = normalize_name(site_name)
    column_mapping = {}

    for col in df.columns:
        normalized_col = str(col)

        # Most processed files use "<variable>_<site-name>" but the site part
        # may contain spaces, hyphens, or underscores depending on export path.
        if "_" in normalized_col:
            base_col, suffix = normalized_col.rsplit("_", 1)
            if normalize_name(suffix) == site_key:
                column_mapping[col] = base_col
                continue

        column_mapping[col] = col

    df_normalized = df.rename(columns=column_mapping)
    return df_normalized, column_mapping


def ensure_target_column(df, target_column):
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


def add_time_features(df):
    df = df.copy()
    if "DateTime" not in df.columns:
        return df

    dt = pd.to_datetime(df["DateTime"], errors="coerce")
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    try:
        week_values = dt.dt.isocalendar().week
    except AttributeError:
        week_values = dt.dt.weekofyear
    df["week_of_year"] = pd.to_numeric(week_values, errors="coerce").astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)
    df["is_weekend"] = np.where(df["dayofweek"].notna(), (df["dayofweek"] >= 5).astype(int), np.nan)
    df["season"] = ((df["month"] % 12 + 3) // 3).map({1: 0, 2: 1, 3: 2, 4: 3})

    return df


def get_temporal_features_by_config(df, config_name):
    df = add_time_features(df)
    feature_list = TEMPORAL_CONFIGS.get(config_name, TEMPORAL_CONFIGS["extended_temporal"])
    temporal_features = [feature for feature in feature_list if feature in df.columns]
    return df, temporal_features


def get_available_input_features(df, target_column, allowed_variables=None):
    if allowed_variables is None:
        candidate_variables = list(config.INPUT_COLUMNS)
    else:
        candidate_variables = [str(col).strip() for col in allowed_variables if str(col).strip()]

    available = [col for col in candidate_variables if col in df.columns and col != target_column]

    if use_time_features():
        for c in TIME_FEATURE_COLUMNS:
            if c in df.columns and c not in available and c != target_column:
                available.append(c)

    return available


def normalize_site_key(site_name):
    return "".join(ch for ch in str(site_name).upper() if ch.isalnum())


def remove_suffix(text, suffix):
    return text[:-len(suffix)] if suffix and text.endswith(suffix) else text


def get_site_coordinates():
    return getattr(config, "SITE_COORDINATES", {}) or {}


def resolve_coordinate_key(site_name, coordinates):
    if not coordinates:
        return None

    raw = str(site_name).strip()
    raw_upper = raw.upper()
    normalized_lookup = {
        normalize_site_key(site): site
        for site in coordinates
    }

    candidates = [
        raw,
        raw_upper,
        raw.split("_")[0],
        raw.split("-")[0],
        remove_suffix(raw_upper, "_AQMS"),
        remove_suffix(raw_upper, "_PROCESSED"),
        remove_suffix(raw_upper, "_STATION"),
        remove_suffix(raw_upper, "_SITE"),
    ]

    for candidate in candidates:
        if candidate in coordinates:
            return candidate
        normalized = normalize_site_key(candidate)
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]

    raw_key = normalize_site_key(raw)
    for normalized, original in normalized_lookup.items():
        if raw_key and (raw_key in normalized or normalized in raw_key):
            return original

    return None


def haversine_distance_km(coord_a, coord_b):
    try:
        lat1 = np.radians(float(coord_a["lat"]))
        lon1 = np.radians(float(coord_a["lon"]))
        lat2 = np.radians(float(coord_b["lat"]))
        lon2 = np.radians(float(coord_b["lon"]))
    except Exception:
        return np.nan

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(6371.0 * c)


def calculate_site_distances(target_site, all_sites, coordinates):
    target_key = resolve_coordinate_key(target_site, coordinates)
    if target_key is None:
        return {}

    distances = {}
    for site in all_sites:
        site_key = resolve_coordinate_key(site, coordinates)
        if site_key is None or site_key == target_key:
            continue

        distance = haversine_distance_km(coordinates[target_key], coordinates[site_key])
        if np.isfinite(distance):
            distances[site] = distance

    return distances


def create_spatial_features(target_site_df, site_dfs, target_site_name, spatial_config, target_column):
    coordinates = get_site_coordinates()
    if not coordinates or "DateTime" not in target_site_df.columns:
        return target_site_df, []

    other_sites = [site for site in site_dfs.keys() if site != target_site_name]
    distances = calculate_site_distances(target_site_name, other_sites, coordinates)
    if not distances:
        return target_site_df, []

    max_distance = float(spatial_config["max_distance"])
    max_sites = int(spatial_config["max_sites"])
    nearby_sites = {
        site: distance
        for site, distance in distances.items()
        if distance <= max_distance
    }
    nearby_sites = dict(sorted(nearby_sites.items(), key=lambda item: item[1])[:max_sites])

    if not nearby_sites:
        return target_site_df, []

    enhanced_df = target_site_df.copy()
    enhanced_df["DateTime"] = pd.to_datetime(enhanced_df["DateTime"], errors="coerce")
    target_datetimes = enhanced_df["DateTime"]
    spatial_feature_names = []
    local_available_vars = get_available_input_features(target_site_df, target_column)

    logging.info(
        f"Using {len(nearby_sites)} nearby sites for spatial features | "
        f"target_site={target_site_name} | sites={list(nearby_sites.keys())}"
    )

    for site_name, distance in nearby_sites.items():
        if site_name not in site_dfs:
            continue

        site_data = site_dfs[site_name].copy()
        if "DateTime" not in site_data.columns:
            continue

        site_data["DateTime"] = pd.to_datetime(site_data["DateTime"], errors="coerce")
        site_data = (
            site_data
            .dropna(subset=["DateTime"])
            .sort_values("DateTime")
            .drop_duplicates(subset=["DateTime"], keep="last")
            .set_index("DateTime")
        )

        weight = 1.0 / (1.0 + distance / 10.0)
        safe_site_name = sanitize_filename(site_name)
        for var in local_available_vars:
            if var not in site_data.columns:
                continue

            spatial_col_name = f"spatial_{var}_{safe_site_name}_dist{distance:.1f}"
            try:
                aligned_values = (
                    pd.to_numeric(site_data[var], errors="coerce")
                    .reindex(target_datetimes)
                    * weight
                )
                enhanced_df[spatial_col_name] = aligned_values.to_numpy()
                spatial_feature_names.append(spatial_col_name)
            except Exception:
                logging.debug(f"Failed to create spatial feature {spatial_col_name}", exc_info=True)

    logging.info(f"Created {len(spatial_feature_names)} spatial features for {target_site_name}")
    return enhanced_df, spatial_feature_names


def create_lag_features(df, target_column, lag_hours, include_target=False):
    df = df.copy()
    if "DateTime" not in df.columns:
        return df, []

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.sort_values("DateTime").reset_index(drop=True)
    lag_feature_names = []
    available_vars = get_available_input_features(df, target_column)

    if include_target and target_column in df.columns and target_column not in available_vars:
        available_vars.append(target_column)

    for var in available_vars:
        if var not in df.columns:
            continue

        df[var] = pd.to_numeric(df[var], errors="coerce")
        for lag in lag_hours:
            lag_col_name = f"lag_{var}_{int(lag)}h"
            try:
                df[lag_col_name] = df[var].shift(int(lag))
                lag_feature_names.append(lag_col_name)
            except Exception:
                logging.debug(f"Failed to create lag feature {lag_col_name}", exc_info=True)

    logging.info(f"Created {len(lag_feature_names)} lag features for lags: {lag_hours}")
    return df, lag_feature_names


def split_variable_string(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def clean_optional_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "nan"}:
        return None

    return text


def read_station_metadata(station_info_file, target_regions=None):
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


def read_overview_metadata(overview_json_file, target_regions=None):
    with open(overview_json_file, "r", encoding="utf-8") as json_file:
        overview = json.load(json_file)

    site_df = pd.DataFrame(overview.get("Sites", []))
    region_df = pd.DataFrame(overview.get("Regions", []))

    if site_df.empty or region_df.empty:
        raise ValueError("Overview JSON is missing 'Regions' or 'Sites' content.")

    required_site_cols = ["Region", "SiteName", "AvailableVariables"]
    missing_site_cols = [col for col in required_site_cols if col not in site_df.columns]
    if missing_site_cols:
        raise ValueError(f"Missing required site fields in overview JSON: {missing_site_cols}")

    required_region_cols = ["Region", "Sites", "CommonVariables"]
    missing_region_cols = [col for col in required_region_cols if col not in region_df.columns]
    if missing_region_cols:
        raise ValueError(f"Missing required region fields in overview JSON: {missing_region_cols}")

    site_df["Region"] = site_df["Region"].apply(clean_optional_text)
    site_df["SiteName"] = site_df["SiteName"].apply(clean_optional_text)
    site_df["AvailableVariables"] = site_df["AvailableVariables"].apply(clean_optional_text)
    region_df["Region"] = region_df["Region"].apply(clean_optional_text)
    region_df["Sites"] = region_df["Sites"].apply(clean_optional_text)
    region_df["CommonVariables"] = region_df["CommonVariables"].apply(clean_optional_text)

    site_df = site_df.dropna(subset=["Region", "SiteName", "AvailableVariables"]).copy()
    region_df = region_df.dropna(subset=["Region", "CommonVariables"]).copy()

    if target_regions:
        target_regions_clean = [str(r).strip() for r in target_regions]
        site_df = site_df[site_df["Region"].isin(target_regions_clean)].copy()
        region_df = region_df[region_df["Region"].isin(target_regions_clean)].copy()

    site_df = site_df.drop_duplicates(subset=["Region", "SiteName"]).reset_index(drop=True)
    region_df = region_df.drop_duplicates(subset=["Region"]).reset_index(drop=True)

    region_sites = (
        site_df.groupby("Region")["SiteName"]
        .apply(lambda x: sorted(x.unique().tolist()))
        .to_dict()
    )
    site_region = dict(zip(site_df["SiteName"], site_df["Region"]))
    site_available_variables = {
        row["SiteName"]: split_variable_string(row["AvailableVariables"])
        for _, row in site_df.iterrows()
    }
    region_common_variables = {
        row["Region"]: split_variable_string(row["CommonVariables"])
        for _, row in region_df.iterrows()
    }

    return site_df, region_df, region_sites, site_region, site_available_variables, region_common_variables


def load_site_data(site_name, target_column=None):
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
    df = add_time_features(df)

    if target_column is not None and str(target_column).strip() != "":
        df = ensure_target_column(df, target_column)

    return df, filepath, column_mapping


def apply_test_missingness(data, target_column, regime="random", frac=0.2, seed=42):
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
        imputed_values = data_imputed.loc[mask, target_column].values
    except Exception:
        logging.exception("Failed to extract true/imputed values for evaluation.")
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


def get_target_columns_to_run(site_dfs, requested_target):
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


def get_common_features_for_target(site_dfs, target_column, region_common_variables=None):
    common_features = None
    valid_site_count = 0

    for _, df in site_dfs.items():
        try:
            df_tmp = ensure_target_column(df, target_column)
        except Exception:
            continue

        available_features = get_available_input_features(
            df_tmp,
            target_column,
            allowed_variables=region_common_variables,
        )

        if common_features is None:
            common_features = set(available_features)
        else:
            common_features &= set(available_features)

        valid_site_count += 1

    if valid_site_count == 0:
        return []

    return sorted(list(common_features)) if common_features else []


def get_targets_from_site_metadata(site_names, site_available_variables):
    available_targets = set()
    for site_name in site_names:
        for variable in site_available_variables.get(site_name, []):
            if variable in POSSIBLE_TARGET_COLUMNS:
                available_targets.add(variable)
    return sorted(available_targets)


def filter_site_dfs_for_target(site_dfs, target_column, site_available_variables=None):
    filtered = {}

    for site_name, df in site_dfs.items():
        if (
            site_available_variables is not None
            and target_column not in set(site_available_variables.get(site_name, []))
        ):
            logging.warning(
                f"Skipping site '{site_name}' for target '{target_column}' based on overview JSON availability."
            )
            continue

        try:
            df_tmp = ensure_target_column(df, target_column)
            filtered[site_name] = df_tmp
        except Exception:
            logging.warning(
                f"Skipping site '{site_name}' for target '{target_column}' because target is unavailable."
            )

    return filtered


# =============================================================================
# RF IMPUTATION ENGINE
# =============================================================================
def rf_impute_target(data_missing, target_column, input_features, rf_params=None, return_model=False):
    """
    Train RF on observed target rows and predict missing target rows.
    """
    rf_params = rf_params or RF_PARAMS
    df = data_missing.copy()

    input_features = [f for f in input_features if f in df.columns and f != target_column]

    if len(input_features) == 0 and use_time_features():
        fallback_features = [c for c in TIME_FEATURE_COLUMNS if c in df.columns]
        input_features = fallback_features.copy()

    train_mask = df[target_column].notna()
    pred_mask = df[target_column].isna()

    if pred_mask.sum() == 0:
        if return_model:
            return df, None
        return df

    if len(input_features) == 0:
        median_val = df.loc[train_mask, target_column].median()
        df.loc[pred_mask, target_column] = median_val
        if return_model:
            return df, None
        return df

    X_train = df.loc[train_mask, input_features].copy()
    y_train = df.loc[train_mask, target_column].copy()
    X_pred = df.loc[pred_mask, input_features].copy()

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_pred_imp = imp.transform(X_pred)

    if len(y_train) < 20:
        median_val = y_train.median()
        df.loc[pred_mask, target_column] = median_val
        if return_model:
            return df, None
        return df

    model = RandomForestRegressor(**rf_params)
    model.fit(X_train_imp, y_train.values)

    y_pred = model.predict(X_pred_imp)
    df.loc[pred_mask, target_column] = y_pred

    if return_model:
        model_info = {
            "model": model,
            "imputer": imp,
            "features": input_features,
            "X_train": pd.DataFrame(X_train_imp, columns=input_features, index=X_train.index),
            "y_train": y_train.copy(),
        }
        return df, model_info

    return df


def run_rf_imputation(data, target_column, input_features):
    return rf_impute_target(
        data_missing=data,
        target_column=target_column,
        input_features=input_features,
        rf_params=RF_PARAMS,
        return_model=False,
    )


# =============================================================================
# SHAP HELPERS
# =============================================================================
def compute_rf_shap_importance(model_info, max_samples=1000):
    if model_info is None:
        return pd.DataFrame(columns=["Feature", "MeanAbsSHAP"])

    model = model_info["model"]
    X_train = model_info["X_train"].copy()
    features = model_info["features"]

    if X_train.empty:
        return pd.DataFrame(columns=["Feature", "MeanAbsSHAP"])

    if len(X_train) > max_samples:
        X_used = X_train.sample(max_samples, random_state=42)
    else:
        X_used = X_train.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_used)

    if isinstance(shap_values, list):
        shap_array = np.array(shap_values[0])
    else:
        shap_array = np.array(shap_values)

    mean_abs_shap = np.abs(shap_array).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": features,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

    return shap_df


def analyze_region_global_shap(region_name, site_dfs, common_features, target_column="PM2.5", output_prefix=None):
    logging.info(
        f"Starting global SHAP | region={region_name} | target={target_column} | "
        f"sites={len(site_dfs)} | features={len(common_features)}"
    )
    pooled_frames = []

    for _, df in site_dfs.items():
        try:
            df_tmp = ensure_target_column(df, target_column)
        except Exception:
            continue

        keep_cols = ["DateTime", target_column, "Site"] + common_features
        keep_cols = [c for c in keep_cols if c in df_tmp.columns]
        pooled_frames.append(df_tmp[keep_cols].copy())

    if not pooled_frames:
        return pd.DataFrame()

    pooled_df = pd.concat(pooled_frames, axis=0, ignore_index=True)
    pooled_df = pooled_df.dropna(subset=["DateTime"]).copy()

    data_missing, _ = apply_test_missingness(pooled_df, target_column)

    _, model_info = rf_impute_target(
        data_missing=data_missing,
        target_column=target_column,
        input_features=common_features,
        rf_params=RF_PARAMS,
        return_model=True,
    )

    shap_df = compute_rf_shap_importance(model_info, max_samples=SHAP_MAX_SAMPLES)

    if shap_df.empty:
        logging.warning(f"Global SHAP returned no results | region={region_name} | target={target_column}")
        return shap_df

    shap_df["Region"] = region_name
    shap_df["Target"] = target_column

    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    shap_df.to_csv(OUTPUT_DIR / f"{prefix}_global_shap_values.csv", index=False)
    logging.info(f"Completed global SHAP | region={region_name} | target={target_column}")

    return shap_df


def analyze_region_sites_shap(
    region_name,
    site_dfs,
    target_column="PM2.5",
    output_prefix=None,
    site_available_variables=None,
):
    site_shap_results = []
    total_sites = len(site_dfs)

    for site_idx, (site_name, df) in enumerate(site_dfs.items(), start=1):
        try:
            logging.info(
                f"Starting site SHAP | region={region_name} | target={target_column} | "
                f"site={site_name} ({site_idx}/{total_sites})"
            )
            df_tmp = ensure_target_column(df, target_column)
            available_features = get_available_input_features(
                df_tmp,
                target_column,
                allowed_variables=(site_available_variables or {}).get(site_name),
            )

            if not available_features:
                continue

            data_missing, _ = apply_test_missingness(df_tmp, target_column)

            _, model_info = rf_impute_target(
                data_missing=data_missing,
                target_column=target_column,
                input_features=available_features,
                rf_params=RF_PARAMS,
                return_model=True,
            )

            shap_df = compute_rf_shap_importance(model_info, max_samples=SHAP_MAX_SAMPLES)

            if shap_df.empty:
                continue

            shap_df["Site"] = site_name
            shap_df["Region"] = region_name
            shap_df["Target"] = target_column
            site_shap_results.append(shap_df)
            logging.info(
                f"Completed site SHAP | region={region_name} | target={target_column} | "
                f"site={site_name} ({site_idx}/{total_sites})"
            )

        except Exception:
            logging.exception(f"SHAP analysis failed for site {site_name}, target {target_column}")

    if not site_shap_results:
        return pd.DataFrame()

    combined_shap_df = pd.concat(site_shap_results, ignore_index=True)

    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    combined_shap_df.to_csv(OUTPUT_DIR / f"{prefix}_site_shap_values.csv", index=False)

    return combined_shap_df


# =============================================================================
# SITE-LEVEL FEATURE SELECTION
# =============================================================================
def analyze_individual_features(
    site_name,
    data,
    target_column="PM2.5",
    save_outputs=True,
    output_prefix=None,
    allowed_variables=None,
):
    logging.info("\n" + "=" * 80)
    logging.info(
        f"ANALYSIS 1: INDIVIDUAL FEATURE IMPORTANCE | SITE = {site_name} | TARGET = {target_column}"
    )
    logging.info("=" * 80)

    data = ensure_target_column(data, target_column)
    available_features = get_available_input_features(
        data,
        target_column,
        allowed_variables=allowed_variables,
    )
    total_features_to_test = len(available_features)
    results = []

    # Baseline
    logging.info("Testing baseline (time-only / minimal fallback)")
    try:
        data_missing, sim_mask = apply_test_missingness(data, target_column)
        data_imputed = run_rf_imputation(data_missing, target_column, [])
        metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                "Feature": "BASELINE (Temporal only)",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "WI": metrics.get("Index of Agreement (WI)", np.nan),
                "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
            })
    except Exception:
        logging.exception(f"Baseline failed for site {site_name}, target {target_column}")

    # Individual features
    for feature_idx, feature in enumerate(available_features, start=1):
        logging.info(
            f"Processing site feature | site={site_name} | target={target_column} | "
            f"feature={feature} ({feature_idx}/{total_features_to_test})"
        )
        try:
            data_missing, sim_mask = apply_test_missingness(data, target_column)
            data_imputed = run_rf_imputation(data_missing, target_column, [feature])
            metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
            if metrics:
                results.append({
                    "Feature": feature,
                    "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                    "R": metrics.get("Correlation Coefficient (R)", np.nan),
                    "WI": metrics.get("Index of Agreement (WI)", np.nan),
                    "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                    "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
                })
                logging.info(
                    f"Completed site feature | site={site_name} | target={target_column} | "
                    f"feature={feature} ({feature_idx}/{total_features_to_test})"
                )
        except Exception:
            logging.exception(f"Feature '{feature}' failed for site {site_name}, target {target_column}")

    # All features
    logging.info(f"Testing site={site_name}, target={target_column}, ALL FEATURES")
    try:
        data_missing, sim_mask = apply_test_missingness(data, target_column)
        data_imputed = run_rf_imputation(data_missing, target_column, available_features)
        metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                "Feature": "ALL FEATURES",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "WI": metrics.get("Index of Agreement (WI)", np.nan),
                "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
            })
    except Exception:
        logging.exception(f"ALL FEATURES failed for site {site_name}, target {target_column}")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        logging.warning(
            f"No successful individual feature results for site {site_name}, target {target_column}"
        )
        return results_df

    results_df = results_df.sort_values("RMSE").reset_index(drop=True)

    if save_outputs:
        prefix = output_prefix or f"{sanitize_filename(site_name)}_{normalize_target_label(target_column)}"
        results_df.to_csv(OUTPUT_DIR / f"{prefix}_individual_features.csv", index=False)
        plot_individual_features(results_df, site_name, target_column, prefix)

    return results_df


def plot_individual_features(results_df, site_name, target_column, output_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
# REGION-LEVEL FEATURE SELECTION
# =============================================================================
def load_region_data(site_list, target_column=None):
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
    logging.info("\n" + "=" * 80)
    logging.info(
        f"REGION GLOBAL FEATURE ANALYSIS | REGION = {region_name} | TARGET = {target_column}"
    )
    logging.info("=" * 80)

    pooled_frames = []
    for _, df in site_dfs.items():
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
        data_imputed = run_rf_imputation(data_missing, target_column, [])
        metrics = evaluate_imputation(pooled_df, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                "Region": region_name,
                "Feature": "BASELINE (Temporal only)",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "WI": metrics.get("Index of Agreement (WI)", np.nan),
                "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
            })
    except Exception:
        logging.exception(f"Regional baseline failed for {region_name}, target {target_column}")

    # Individual common features
    total_common_features = len(common_features)
    for feature_idx, feature in enumerate(common_features, start=1):
        logging.info(
            f"Processing regional feature | region={region_name} | target={target_column} | "
            f"feature={feature} ({feature_idx}/{total_common_features})"
        )
        try:
            data_missing, sim_mask = apply_test_missingness(pooled_df, target_column)
            data_imputed = run_rf_imputation(data_missing, target_column, [feature])
            metrics = evaluate_imputation(pooled_df, data_imputed, sim_mask, target_column)
            if metrics:
                results.append({
                    "Region": region_name,
                    "Feature": feature,
                    "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                    "R": metrics.get("Correlation Coefficient (R)", np.nan),
                    "WI": metrics.get("Index of Agreement (WI)", np.nan),
                    "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                    "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
                })
                logging.info(
                    f"Completed regional feature | region={region_name} | target={target_column} | "
                    f"feature={feature} ({feature_idx}/{total_common_features})"
                )
        except Exception:
            logging.exception(f"Regional feature '{feature}' failed for {region_name}, target {target_column}")

    # All features
    try:
        data_missing, sim_mask = apply_test_missingness(pooled_df, target_column)
        data_imputed = run_rf_imputation(data_missing, target_column, common_features)
        metrics = evaluate_imputation(pooled_df, data_imputed, sim_mask, target_column)
        if metrics:
            results.append({
                "Region": region_name,
                "Feature": "ALL FEATURES",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "WI": metrics.get("Index of Agreement (WI)", np.nan),
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


def analyze_region_sites(
    region_name,
    site_dfs,
    target_column="PM2.5",
    output_prefix=None,
    site_available_variables=None,
):
    logging.info("\n" + "=" * 80)
    logging.info(
        f"REGION SITE-WISE FEATURE ANALYSIS | REGION = {region_name} | TARGET = {target_column}"
    )
    logging.info("=" * 80)

    all_site_results = []
    site_summary_rows = []
    total_sites = len(site_dfs)

    for site_idx, (site_name, df) in enumerate(site_dfs.items(), start=1):
        try:
            logging.info(
                f"Starting site-wise analysis | region={region_name} | target={target_column} | "
                f"site={site_name} ({site_idx}/{total_sites})"
            )
            df_tmp = ensure_target_column(df, target_column)

            site_result = analyze_individual_features(
                site_name=site_name,
                data=df_tmp,
                target_column=target_column,
                save_outputs=True,
                allowed_variables=(site_available_variables or {}).get(site_name),
                output_prefix=(
                    f"{sanitize_filename(region_name)}_"
                    f"{sanitize_filename(site_name)}_"
                    f"{normalize_target_label(target_column)}"
                )
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
            logging.info(
                f"Completed site-wise analysis | region={region_name} | target={target_column} | "
                f"site={site_name} ({site_idx}/{total_sites})"
            )

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
# PLOTTING
# =============================================================================
def plot_region_global_vs_sites_metric(
    region_name,
    target_column,
    global_df,
    site_df,
    metric_col,
    metric_label,
    top_n=10,
    output_prefix=None,
    higher_is_better=False,
    fmt=".2f",
    cmap="viridis",
    legacy_filename=False,
    selected_features=None,
):
    if global_df.empty:
        logging.warning(
            f"Skipping combined region plot: global_df empty for {region_name}, target {target_column}, metric {metric_col}"
        )
        return

    global_plot_df = global_df[
        ~global_df["Feature"].isin(["BASELINE (Temporal only)", "ALL FEATURES"])
    ].copy()

    if global_plot_df.empty or metric_col not in global_plot_df.columns:
        logging.warning(
            f"Skipping combined region plot: no normal features for {region_name}, target {target_column}, metric {metric_col}"
        )
        return

    if selected_features is None:
        global_plot_df = global_plot_df.sort_values(metric_col, ascending=not higher_is_better).head(top_n)
        top_features = global_plot_df["Feature"].tolist()
    else:
        top_features = [
            feature
            for feature in selected_features
            if feature in set(global_plot_df["Feature"])
        ]

    fig_width = max(13, 0.65 * max(len(top_features), 1))
    fig_height = max(8, 0.45 * (len(site_df["Site"].unique()) + 1) if not site_df.empty else 8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if site_df.empty:
        ax.text(0.5, 0.5, "No site-wise results available", ha="center", va="center")
        ax.set_axis_off()
    else:
        site_heat_df = site_df[site_df["Feature"].isin(top_features)].pivot_table(
            index="Site",
            columns="Feature",
            values=metric_col,
            aggfunc="mean"
        )
        region_heat_df = global_df[global_df["Feature"].isin(top_features)].pivot_table(
            index="Region",
            columns="Feature",
            values=metric_col,
            aggfunc="mean"
        )
        region_heat_df.index = [f"Region: {idx}" for idx in region_heat_df.index]

        heat_df = pd.concat([region_heat_df, site_heat_df], axis=0)

        if not heat_df.empty:
            available_top_features = [feature for feature in top_features if feature in heat_df.columns]
            heat_df = heat_df[available_top_features]
            sns.heatmap(
                heat_df,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                cbar_kws={"label": metric_label},
                ax=ax
            )
            ax.set_title(
                f"{region_name} | {target_column}\nFeature Performance by Region and Site ({metric_col})",
                fontweight="bold",
            )
            ax.set_xlabel("Feature", fontweight="bold")
            ax.set_ylabel("Site / Region", fontweight="bold")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(0.5, 0.5, "No matching site-feature results", ha="center", va="center")
            ax.set_axis_off()

    plt.tight_layout()
    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    if legacy_filename:
        out_name = f"{prefix}_global_vs_sites.png"
    else:
        out_name = f"{prefix}_global_vs_sites_{metric_col.lower()}.png"
    plt.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_region_global_vs_sites(region_name, target_column, global_df, site_df, top_n=10, output_prefix=None):
    if global_df.empty:
        logging.warning(f"Skipping combined region plots: global_df empty for {region_name}, target {target_column}")
        return

    base_feature_df = global_df[
        ~global_df["Feature"].isin(["BASELINE (Temporal only)", "ALL FEATURES"])
    ].copy()

    if base_feature_df.empty:
        logging.warning(f"Skipping combined region plots: no normal features for {region_name}, target {target_column}")
        return

    selected_features = (
        base_feature_df.sort_values("RMSE")
        ["Feature"]
        .tolist()
    )
    logging.info(
        f"Using all evaluated shared feature columns for metric plots | region={region_name} | "
        f"target={target_column} | n_features={len(selected_features)} | features={selected_features}"
    )

    plot_region_global_vs_sites_metric(
        region_name=region_name,
        target_column=target_column,
        global_df=global_df,
        site_df=site_df,
        metric_col="RMSE",
        metric_label="RMSE (μg/m³)",
        top_n=top_n,
        output_prefix=output_prefix,
        higher_is_better=False,
        fmt=".2f",
        cmap="viridis_r",
        legacy_filename=True,
        selected_features=selected_features,
    )
    plot_region_global_vs_sites_metric(
        region_name=region_name,
        target_column=target_column,
        global_df=global_df,
        site_df=site_df,
        metric_col="R",
        metric_label="Correlation Coefficient (R)",
        top_n=top_n,
        output_prefix=output_prefix,
        higher_is_better=True,
        fmt=".2f",
        cmap="viridis",
        selected_features=selected_features,
    )
    plot_region_global_vs_sites_metric(
        region_name=region_name,
        target_column=target_column,
        global_df=global_df,
        site_df=site_df,
        metric_col="WI",
        metric_label="Index of Agreement (WI)",
        top_n=top_n,
        output_prefix=output_prefix,
        higher_is_better=True,
        fmt=".2f",
        cmap="magma",
        selected_features=selected_features,
    )


def calculate_feature_pearson(df, target_column, feature):
    if target_column not in df.columns or feature not in df.columns:
        return np.nan

    pair_df = df[[target_column, feature]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair_df) < MIN_VALID_POINTS:
        return np.nan

    if pair_df[target_column].std() == 0 or pair_df[feature].std() == 0:
        return np.nan

    return pair_df[target_column].corr(pair_df[feature], method="pearson")


def get_common_features_for_heatmaps(site_dfs, target_column, common_features):
    feature_order = []

    for feature in common_features:
        if (
            feature != target_column
            and feature not in feature_order
            and all(feature in df.columns for df in site_dfs.values())
        ):
            feature_order.append(feature)

    return feature_order


def analyze_target_feature_pearson(
    region_name,
    site_dfs,
    common_features,
    target_column="PM2.5",
    output_prefix=None,
    site_available_variables=None,
):
    logging.info(
        f"Starting Pearson target-feature analysis | region={region_name} | target={target_column}"
    )

    feature_order = get_common_features_for_heatmaps(
        site_dfs,
        target_column,
        common_features,
    )
    logging.info(
        f"Pearson heatmap restricted to region-common features | region={region_name} | "
        f"target={target_column} | n_features={len(feature_order)} | features={feature_order}"
    )

    records = []

    pooled_frames = []
    for site_name, df in site_dfs.items():
        try:
            df_tmp = ensure_target_column(df, target_column)
        except Exception:
            continue

        keep_cols = ["DateTime", target_column, "Site"] + feature_order
        keep_cols = [col for col in keep_cols if col in df_tmp.columns]
        pooled_frames.append(df_tmp[keep_cols].copy())

        for feature in feature_order:
            records.append({
                "Region": region_name,
                "Site": site_name,
                "Feature": feature,
                "Pearson_R": calculate_feature_pearson(df_tmp, target_column, feature),
            })

    if pooled_frames:
        pooled_df = pd.concat(pooled_frames, axis=0, ignore_index=True)
        for feature in feature_order:
            records.append({
                "Region": region_name,
                "Site": f"Region: {region_name}",
                "Feature": feature,
                "Pearson_R": calculate_feature_pearson(pooled_df, target_column, feature),
            })

    pearson_df = pd.DataFrame(records)
    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"

    if pearson_df.empty:
        logging.warning(
            f"No Pearson target-feature results | region={region_name} | target={target_column}"
        )
        return pearson_df

    pearson_df.to_csv(OUTPUT_DIR / f"{prefix}_target_feature_pearson.csv", index=False)

    heat_df = pearson_df.pivot_table(
        index="Site",
        columns="Feature",
        values="Pearson_R",
        aggfunc="mean",
    )

    row_order = [f"Region: {region_name}"] + [site for site in site_dfs.keys() if site in heat_df.index]
    row_order = [row for row in row_order if row in heat_df.index]
    col_order = [feature for feature in feature_order if feature in heat_df.columns]
    heat_df = heat_df.reindex(index=row_order, columns=col_order)

    fig_width = max(13, 0.65 * max(len(col_order), 1))
    fig_height = max(8, 0.45 * max(len(row_order), 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        cbar_kws={"label": "Pearson correlation (r)"},
        ax=ax,
    )
    ax.set_title(
        f"{region_name} | {target_column}\nPearson Correlation: Target vs Input Features",
        fontweight="bold",
    )
    ax.set_xlabel("Feature", fontweight="bold")
    ax.set_ylabel("Site / Region", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_target_feature_pearson.png", dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(
        f"Completed Pearson target-feature analysis | region={region_name} | "
        f"target={target_column} | features={feature_order}"
    )

    return pearson_df


def get_ranked_features_from_pearson(pearson_df, row_name):
    row_df = pearson_df[
        (pearson_df["Site"] == row_name)
        & pearson_df["Pearson_R"].notna()
    ].copy()

    if row_df.empty:
        return []

    row_df["AbsPearson_R"] = row_df["Pearson_R"].abs()
    return row_df.sort_values("AbsPearson_R", ascending=False)["Feature"].tolist()


def get_combo_limit(ranked_features, max_combo_size=None):
    if max_combo_size is None:
        return len(ranked_features)
    return min(int(max_combo_size), len(ranked_features))


def make_combo_label(features):
    return " + ".join(features)


def evaluate_rf_feature_combination(data, target_column, features):
    features = [feature for feature in features if feature in data.columns and feature != target_column]
    if not features:
        return None

    data_missing, sim_mask = apply_test_missingness(data, target_column)
    data_imputed = run_rf_imputation(data_missing, target_column, features)
    metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)

    if not metrics:
        return None

    return {
        "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
        "R": metrics.get("Correlation Coefficient (R)", np.nan),
        "WI": metrics.get("Index of Agreement (WI)", np.nan),
        "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
        "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
    }


def plot_pearson_guided_combo_heatmap(
    combo_df,
    region_name,
    target_column,
    metric_col,
    metric_label,
    output_prefix,
    fmt=".2f",
    cmap="viridis",
):
    if combo_df.empty or metric_col not in combo_df.columns:
        logging.warning(
            f"Skipping Pearson-guided combo heatmap | region={region_name} | "
            f"target={target_column} | metric={metric_col}"
        )
        return

    heat_df = combo_df.pivot_table(
        index="ComboLabel",
        columns="RowName",
        values=metric_col,
        aggfunc="mean",
    )

    column_order = [f"Region: {region_name}"] + [
        row for row in combo_df["RowName"].dropna().unique()
        if row != f"Region: {region_name}"
    ]
    column_order = [col for col in column_order if col in heat_df.columns]

    combo_order = (
        combo_df[["ComboRank", "ComboLabel"]]
        .drop_duplicates()
        .sort_values(["ComboRank", "ComboLabel"])["ComboLabel"]
        .tolist()
    )
    combo_order = [combo for combo in combo_order if combo in heat_df.index]

    heat_df = heat_df.reindex(index=combo_order, columns=column_order)

    fig_width = max(10, 1.6 * max(len(column_order), 1))
    fig_height = max(8, 0.55 * max(len(combo_order), 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"label": metric_label},
        ax=ax,
    )
    ax.set_title(
        f"{region_name} | {target_column}\nRF Performance for Pearson-Guided Feature Combinations ({metric_col})",
        fontweight="bold",
    )
    ax.set_xlabel("Site / Region", fontweight="bold")
    ax.set_ylabel("Pearson-ranked cumulative feature combination", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{output_prefix}_pearson_guided_combinations_{metric_col.lower()}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def analyze_pearson_guided_feature_combinations(
    region_name,
    site_dfs,
    common_features,
    pearson_df,
    target_column="PM2.5",
    output_prefix=None,
    site_available_variables=None,
    max_combo_size=PEARSON_COMBO_MAX_FEATURES,
):
    logging.info(
        f"Starting Pearson-guided RF combination analysis | region={region_name} | "
        f"target={target_column} | max_combo_size={max_combo_size}"
    )

    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    records = []

    pooled_frames = []
    all_features = get_common_features_for_heatmaps(
        site_dfs,
        target_column,
        common_features,
    )
    logging.info(
        f"Pearson-guided combo heatmaps restricted to region-common features | region={region_name} | "
        f"target={target_column} | n_features={len(all_features)} | features={all_features}"
    )

    for site_name, df in site_dfs.items():
        try:
            df_tmp = ensure_target_column(df, target_column)
        except Exception:
            continue

        keep_cols = ["DateTime", target_column, "Site"] + all_features
        keep_cols = [col for col in keep_cols if col in df_tmp.columns]
        pooled_frames.append(df_tmp[keep_cols].copy())

        ranked_features = get_ranked_features_from_pearson(pearson_df, site_name)
        if not ranked_features:
            logging.warning(
                f"No Pearson-ranked features for site combo analysis | site={site_name} | target={target_column}"
            )
            continue

        combo_limit = get_combo_limit(ranked_features, max_combo_size=max_combo_size)
        for combo_size in range(1, combo_limit + 1):
            combo_features = ranked_features[:combo_size]
            combo_label = make_combo_label(combo_features)
            logging.info(
                f"Processing Pearson-guided site combo | site={site_name} | "
                f"target={target_column} | combo_size={combo_size} | features={combo_features}"
            )
            metrics = evaluate_rf_feature_combination(df_tmp, target_column, combo_features)
            if metrics is None:
                continue

            records.append({
                "Guidance_Method": "Pearson",
                "Region": region_name,
                "Target": target_column,
                "RowType": "Site",
                "RowName": site_name,
                "ComboRank": combo_size,
                "ComboLabel": combo_label,
                "Features": ", ".join(combo_features),
                "N_Features": len(combo_features),
                **metrics,
            })

    if pooled_frames:
        pooled_df = pd.concat(pooled_frames, axis=0, ignore_index=True)
        region_row_name = f"Region: {region_name}"
        ranked_features = get_ranked_features_from_pearson(pearson_df, region_row_name)

        combo_limit = get_combo_limit(ranked_features, max_combo_size=max_combo_size)
        for combo_size in range(1, combo_limit + 1):
            combo_features = ranked_features[:combo_size]
            combo_label = make_combo_label(combo_features)
            logging.info(
                f"Processing Pearson-guided region combo | region={region_name} | "
                f"target={target_column} | combo_size={combo_size} | features={combo_features}"
            )
            metrics = evaluate_rf_feature_combination(pooled_df, target_column, combo_features)
            if metrics is None:
                continue

            records.append({
                "Guidance_Method": "Pearson",
                "Region": region_name,
                "Target": target_column,
                "RowType": "Region",
                "RowName": region_row_name,
                "ComboRank": combo_size,
                "ComboLabel": combo_label,
                "Features": ", ".join(combo_features),
                "N_Features": len(combo_features),
                **metrics,
            })

    combo_df = pd.DataFrame(records)
    if combo_df.empty:
        logging.warning(
            f"No Pearson-guided RF combination results | region={region_name} | target={target_column}"
        )
        return combo_df

    combo_df.to_csv(OUTPUT_DIR / f"{prefix}_pearson_guided_combinations.csv", index=False)

    best_combo_df = (
        combo_df.sort_values(["RowName", "RMSE"])
        .groupby(["RowType", "RowName"], as_index=False)
        .first()
        .sort_values(["RowType", "RowName"])
        .reset_index(drop=True)
    )
    best_combo_df.to_csv(
        OUTPUT_DIR / f"{prefix}_pearson_guided_best_combinations.csv",
        index=False,
    )

    plot_pearson_guided_combo_heatmap(
        combo_df,
        region_name,
        target_column,
        metric_col="RMSE",
        metric_label="RMSE (μg/m³)",
        output_prefix=prefix,
        fmt=".2f",
        cmap="viridis_r",
    )
    plot_pearson_guided_combo_heatmap(
        combo_df,
        region_name,
        target_column,
        metric_col="R",
        metric_label="Correlation Coefficient (R)",
        output_prefix=prefix,
        fmt=".2f",
        cmap="viridis",
    )
    plot_pearson_guided_combo_heatmap(
        combo_df,
        region_name,
        target_column,
        metric_col="WI",
        metric_label="Index of Agreement (WI)",
        output_prefix=prefix,
        fmt=".2f",
        cmap="magma",
    )

    logging.info(
        f"Completed Pearson-guided RF combination analysis | region={region_name} | target={target_column}"
    )
    return combo_df


def get_ranked_features_from_shap(shap_df, allowed_features, site_name=None):
    if shap_df is None or shap_df.empty:
        return []

    ranked_df = shap_df.copy()
    if site_name is not None:
        if "Site" not in ranked_df.columns:
            return []
        ranked_df = ranked_df[ranked_df["Site"] == site_name].copy()

    ranked_df = ranked_df[
        ranked_df["Feature"].isin(allowed_features)
        & ranked_df["MeanAbsSHAP"].notna()
    ].copy()

    if ranked_df.empty:
        return []

    ranked_df = (
        ranked_df.groupby("Feature", as_index=False)["MeanAbsSHAP"]
        .mean()
        .sort_values("MeanAbsSHAP", ascending=False)
    )
    return ranked_df["Feature"].tolist()


def plot_shap_guided_combo_heatmap(
    combo_df,
    region_name,
    target_column,
    metric_col,
    metric_label,
    output_prefix,
    fmt=".2f",
    cmap="viridis",
):
    if combo_df.empty or metric_col not in combo_df.columns:
        logging.warning(
            f"Skipping SHAP-guided combo heatmap | region={region_name} | "
            f"target={target_column} | metric={metric_col}"
        )
        return

    heat_df = combo_df.pivot_table(
        index="ComboLabel",
        columns="RowName",
        values=metric_col,
        aggfunc="mean",
    )

    column_order = [f"Region: {region_name}"] + [
        row for row in combo_df["RowName"].dropna().unique()
        if row != f"Region: {region_name}"
    ]
    column_order = [col for col in column_order if col in heat_df.columns]

    combo_order = (
        combo_df[["ComboRank", "ComboLabel"]]
        .drop_duplicates()
        .sort_values(["ComboRank", "ComboLabel"])["ComboLabel"]
        .tolist()
    )
    combo_order = [combo for combo in combo_order if combo in heat_df.index]

    heat_df = heat_df.reindex(index=combo_order, columns=column_order)

    fig_width = max(10, 1.6 * max(len(column_order), 1))
    fig_height = max(8, 0.55 * max(len(combo_order), 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"label": metric_label},
        ax=ax,
    )
    ax.set_title(
        f"{region_name} | {target_column}\nRF Performance for SHAP-Guided Feature Combinations ({metric_col})",
        fontweight="bold",
    )
    ax.set_xlabel("Site / Region", fontweight="bold")
    ax.set_ylabel("SHAP-ranked cumulative feature combination", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{output_prefix}_shap_guided_combinations_{metric_col.lower()}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def analyze_shap_guided_feature_combinations(
    region_name,
    site_dfs,
    common_features,
    global_shap_df,
    site_shap_df,
    target_column="PM2.5",
    output_prefix=None,
    max_combo_size=PEARSON_COMBO_MAX_FEATURES,
):
    logging.info(
        f"Starting SHAP-guided RF combination analysis | region={region_name} | "
        f"target={target_column} | max_combo_size={max_combo_size}"
    )

    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    records = []

    allowed_features = get_common_features_for_heatmaps(
        site_dfs,
        target_column,
        common_features,
    )
    logging.info(
        f"SHAP-guided combo heatmaps restricted to region-common features | region={region_name} | "
        f"target={target_column} | n_features={len(allowed_features)} | features={allowed_features}"
    )

    pooled_frames = []
    for site_name, df in site_dfs.items():
        try:
            df_tmp = ensure_target_column(df, target_column)
        except Exception:
            continue

        keep_cols = ["DateTime", target_column, "Site"] + allowed_features
        keep_cols = [col for col in keep_cols if col in df_tmp.columns]
        pooled_frames.append(df_tmp[keep_cols].copy())

        ranked_features = get_ranked_features_from_shap(
            site_shap_df,
            allowed_features,
            site_name=site_name,
        )
        if not ranked_features:
            logging.warning(
                f"No SHAP-ranked features for site combo analysis | site={site_name} | target={target_column}"
            )
            continue

        combo_limit = get_combo_limit(ranked_features, max_combo_size=max_combo_size)
        for combo_size in range(1, combo_limit + 1):
            combo_features = ranked_features[:combo_size]
            combo_label = make_combo_label(combo_features)
            logging.info(
                f"Processing SHAP-guided site combo | site={site_name} | "
                f"target={target_column} | combo_size={combo_size} | features={combo_features}"
            )
            metrics = evaluate_rf_feature_combination(df_tmp, target_column, combo_features)
            if metrics is None:
                continue

            records.append({
                "Guidance_Method": "SHAP",
                "Region": region_name,
                "Target": target_column,
                "RowType": "Site",
                "RowName": site_name,
                "ComboRank": combo_size,
                "ComboLabel": combo_label,
                "Features": ", ".join(combo_features),
                "N_Features": len(combo_features),
                **metrics,
            })

    if pooled_frames:
        pooled_df = pd.concat(pooled_frames, axis=0, ignore_index=True)
        region_row_name = f"Region: {region_name}"
        ranked_features = get_ranked_features_from_shap(
            global_shap_df,
            allowed_features,
            site_name=None,
        )

        combo_limit = get_combo_limit(ranked_features, max_combo_size=max_combo_size)
        for combo_size in range(1, combo_limit + 1):
            combo_features = ranked_features[:combo_size]
            combo_label = make_combo_label(combo_features)
            logging.info(
                f"Processing SHAP-guided region combo | region={region_name} | "
                f"target={target_column} | combo_size={combo_size} | features={combo_features}"
            )
            metrics = evaluate_rf_feature_combination(pooled_df, target_column, combo_features)
            if metrics is None:
                continue

            records.append({
                "Guidance_Method": "SHAP",
                "Region": region_name,
                "Target": target_column,
                "RowType": "Region",
                "RowName": region_row_name,
                "ComboRank": combo_size,
                "ComboLabel": combo_label,
                "Features": ", ".join(combo_features),
                "N_Features": len(combo_features),
                **metrics,
            })

    combo_df = pd.DataFrame(records)
    if combo_df.empty:
        logging.warning(
            f"No SHAP-guided RF combination results | region={region_name} | target={target_column}"
        )
        return combo_df

    combo_df.to_csv(OUTPUT_DIR / f"{prefix}_shap_guided_combinations.csv", index=False)

    best_combo_df = (
        combo_df.sort_values(["RowName", "RMSE"])
        .groupby(["RowType", "RowName"], as_index=False)
        .first()
        .sort_values(["RowType", "RowName"])
        .reset_index(drop=True)
    )
    best_combo_df.to_csv(
        OUTPUT_DIR / f"{prefix}_shap_guided_best_combinations.csv",
        index=False,
    )

    plot_shap_guided_combo_heatmap(
        combo_df,
        region_name,
        target_column,
        metric_col="RMSE",
        metric_label="RMSE (μg/m³)",
        output_prefix=prefix,
        fmt=".2f",
        cmap="viridis_r",
    )
    plot_shap_guided_combo_heatmap(
        combo_df,
        region_name,
        target_column,
        metric_col="R",
        metric_label="Correlation Coefficient (R)",
        output_prefix=prefix,
        fmt=".2f",
        cmap="viridis",
    )
    plot_shap_guided_combo_heatmap(
        combo_df,
        region_name,
        target_column,
        metric_col="WI",
        metric_label="Index of Agreement (WI)",
        output_prefix=prefix,
        fmt=".2f",
        cmap="magma",
    )

    logging.info(
        f"Completed SHAP-guided RF combination analysis | region={region_name} | target={target_column}"
    )
    return combo_df


def plot_global_vs_local_shap(region_name, target_column, global_shap_df, site_shap_df, top_n=10, output_prefix=None):
    if global_shap_df.empty:
        logging.warning(f"No global SHAP results for {region_name}, {target_column}")
        return

    gdf = global_shap_df.sort_values("MeanAbsSHAP", ascending=False).head(top_n).copy()
    top_features = gdf["Feature"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    bars = ax.barh(gdf["Feature"], gdf["MeanAbsSHAP"], edgecolor="black")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontweight="bold")
    ax.set_title(f"{region_name} | {target_column}\nGlobal SHAP Importance", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, gdf["MeanAbsSHAP"]):
        ax.text(val, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                va="center", ha="left", fontsize=8, fontweight="bold")

    ax = axes[1]
    if site_shap_df.empty:
        ax.text(0.5, 0.5, "No site-wise SHAP results available", ha="center", va="center")
        ax.set_axis_off()
    else:
        heat_df = site_shap_df[site_shap_df["Feature"].isin(top_features)].pivot_table(
            index="Site",
            columns="Feature",
            values="MeanAbsSHAP",
            aggfunc="mean"
        )

        if not heat_df.empty:
            available_cols = [f for f in top_features if f in heat_df.columns]
            heat_df = heat_df[available_cols]

            sns.heatmap(
                heat_df,
                annot=True,
                fmt=".3f",
                cmap="magma",
                cbar_kws={"label": "Mean |SHAP|"},
                ax=ax
            )
            ax.set_title(f"{region_name} | {target_column}\nLocal SHAP by Site", fontweight="bold")
            ax.set_xlabel("Feature", fontweight="bold")
            ax.set_ylabel("Site", fontweight="bold")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(0.5, 0.5, "No matching site-wise SHAP results", ha="center", va="center")
            ax.set_axis_off()

    plt.tight_layout()
    prefix = output_prefix or f"{sanitize_filename(region_name)}_{normalize_target_label(target_column)}"
    plt.savefig(OUTPUT_DIR / f"{prefix}_global_vs_local_shap.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_region_site_best_features(region_name, target_column, site_summary_df, output_prefix=None):
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
# REPORTING
# =============================================================================
def generate_region_summary(
    region_name,
    target_column,
    global_df,
    site_summary_df,
    common_features,
    global_shap_df=None,
    output_prefix=None,
):
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
        lines.append("No global regional performance results available.")
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
            lines.append("Best global performance feature:")
            lines.append(f"  Feature: {best_global['Feature']}")
            lines.append(f"  RMSE: {best_global['RMSE']:.3f}")
            lines.append(f"  R: {best_global['R']:.3f}")

    lines.append("")
    lines.append("-" * 90)

    if global_shap_df is not None and not global_shap_df.empty:
        lines.append("Top global SHAP features:")
        for _, row in global_shap_df.sort_values("MeanAbsSHAP", ascending=False).head(10).iterrows():
            lines.append(f"  - {row['Feature']}: Mean|SHAP|={row['MeanAbsSHAP']:.4f}")
    else:
        lines.append("No global SHAP results available.")

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
# COMPREHENSIVE TEMPORAL / SPATIAL / LAG FEATURE ANALYSIS
# =============================================================================
def metric_row_from_evaluation(metrics):
    return {
        "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
        "R": metrics.get("Correlation Coefficient (R)", np.nan),
        "WI": metrics.get("Index of Agreement (WI)", np.nan),
        "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
        "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
    }


def evaluate_feature_configuration(data, target_column, features):
    features = [feature for feature in features if feature in data.columns and feature != target_column]
    data_missing, sim_mask = apply_test_missingness(data, target_column)
    data_imputed = rf_impute_target(data_missing, target_column, features, RF_PARAMS)
    metrics = evaluate_imputation(data, data_imputed, sim_mask, target_column)
    if not metrics:
        return None
    return metric_row_from_evaluation(metrics)


def analyze_temporal_configurations(site_name, data, target_column, output_prefix=None):
    logging.info("\n" + "=" * 80)
    logging.info(
        f"TEMPORAL CONFIGURATION ANALYSIS | SITE = {site_name} | TARGET = {target_column}"
    )
    logging.info("=" * 80)

    results = []

    for config_name in TEMPORAL_CONFIGS:
        logging.info(f"Testing temporal config: {config_name}")
        try:
            data_with_temporal, temporal_features = get_temporal_features_by_config(
                data.copy(),
                config_name,
            )

            if not temporal_features:
                logging.warning(f"No temporal features available for config {config_name}")
                results.append({
                    "Configuration": f"temporal_{config_name}",
                    "Feature_Type": "Temporal",
                    "Features": "",
                    "N_Features": 0,
                    "Status": "Skipped: no temporal features available",
                })
                continue

            metrics = evaluate_feature_configuration(
                data_with_temporal,
                target_column,
                temporal_features,
            )
            if metrics is None:
                results.append({
                    "Configuration": f"temporal_{config_name}",
                    "Feature_Type": "Temporal",
                    "Features": ", ".join(temporal_features),
                    "N_Features": len(temporal_features),
                    "Status": "Skipped: evaluation failed or too few valid points",
                })
                continue

            results.append({
                "Configuration": f"temporal_{config_name}",
                "Feature_Type": "Temporal",
                "Features": ", ".join(temporal_features),
                "N_Features": len(temporal_features),
                "Status": "Success",
                **metrics,
            })
            logging.info(
                f"Temporal config complete | site={site_name} | config={config_name} | "
                f"RMSE={metrics['RMSE']:.3f} | R={metrics['R']:.3f}"
            )
        except Exception as e:
            logging.exception(f"Temporal config {config_name} failed for {site_name}")
            results.append({
                "Configuration": f"temporal_{config_name}",
                "Feature_Type": "Temporal",
                "Features": ", ".join(TEMPORAL_CONFIGS.get(config_name, [])),
                "N_Features": 0,
                "Status": "Failed",
                "Error": str(e),
            })

    results_df = pd.DataFrame(results)
    if output_prefix:
        save_dataframe(
            results_df,
            f"{output_prefix}_temporal_configs.csv",
            columns=CONFIG_RESULT_COLUMNS,
        )
    return results_df


def analyze_spatial_configurations(site_name, site_dfs, target_column, output_prefix=None):
    logging.info("\n" + "=" * 80)
    logging.info(
        f"SPATIAL CONFIGURATION ANALYSIS | SITE = {site_name} | TARGET = {target_column}"
    )
    logging.info("=" * 80)

    if site_name not in site_dfs:
        logging.warning(f"Site {site_name} not found in site_dfs")
        return pd.DataFrame()

    base_data = site_dfs[site_name].copy()
    results = []

    for config_name, spatial_config in SPATIAL_CONFIGS.items():
        logging.info(
            f"Testing spatial config: {config_name} "
            f"(max_sites={spatial_config['max_sites']}, "
            f"max_distance={spatial_config['max_distance']} km)"
        )
        try:
            data_with_spatial, spatial_features = create_spatial_features(
                base_data,
                site_dfs,
                site_name,
                spatial_config,
                target_column,
            )

            if not spatial_features:
                logging.warning(f"No spatial features created for config {config_name}")
                results.append({
                    "Configuration": f"spatial_{config_name}",
                    "Feature_Type": "Spatial",
                    "Features": "",
                    "N_Features": 0,
                    "Max_Sites": spatial_config["max_sites"],
                    "Max_Distance": spatial_config["max_distance"],
                    "Status": "Skipped: no spatial features created",
                })
                continue

            metrics = evaluate_feature_configuration(
                data_with_spatial,
                target_column,
                spatial_features,
            )
            if metrics is None:
                results.append({
                    "Configuration": f"spatial_{config_name}",
                    "Feature_Type": "Spatial",
                    "Features": f"{len(spatial_features)} spatial features",
                    "N_Features": len(spatial_features),
                    "Max_Sites": spatial_config["max_sites"],
                    "Max_Distance": spatial_config["max_distance"],
                    "Status": "Skipped: evaluation failed or too few valid points",
                })
                continue

            results.append({
                "Configuration": f"spatial_{config_name}",
                "Feature_Type": "Spatial",
                "Features": f"{len(spatial_features)} spatial features",
                "N_Features": len(spatial_features),
                "Max_Sites": spatial_config["max_sites"],
                "Max_Distance": spatial_config["max_distance"],
                "Status": "Success",
                **metrics,
            })
            logging.info(
                f"Spatial config complete | site={site_name} | config={config_name} | "
                f"RMSE={metrics['RMSE']:.3f} | R={metrics['R']:.3f}"
            )
        except Exception as e:
            logging.exception(f"Spatial config {config_name} failed for {site_name}")
            results.append({
                "Configuration": f"spatial_{config_name}",
                "Feature_Type": "Spatial",
                "Features": "",
                "N_Features": 0,
                "Max_Sites": spatial_config["max_sites"],
                "Max_Distance": spatial_config["max_distance"],
                "Status": "Failed",
                "Error": str(e),
            })

    results_df = pd.DataFrame(results)
    if output_prefix:
        save_dataframe(
            results_df,
            f"{output_prefix}_spatial_configs.csv",
            columns=CONFIG_RESULT_COLUMNS + ["Max_Sites", "Max_Distance"],
        )
    return results_df


def analyze_lag_configurations(site_name, data, target_column, output_prefix=None):
    logging.info("\n" + "=" * 80)
    logging.info(f"LAG CONFIGURATION ANALYSIS | SITE = {site_name} | TARGET = {target_column}")
    logging.info("=" * 80)

    results = []

    for config_name, lag_hours in LAG_CONFIGS.items():
        logging.info(f"Testing lag config: {config_name} (lags: {lag_hours})")
        try:
            data_with_lags, lag_features = create_lag_features(
                data.copy(),
                target_column,
                lag_hours,
                include_target=False,
            )

            if not lag_features:
                logging.warning(f"No lag features created for config {config_name}")
                results.append({
                    "Configuration": f"lag_{config_name}",
                    "Feature_Type": "Lag",
                    "Features": f"Lags: {lag_hours}",
                    "N_Features": 0,
                    "Lag_Hours": str(lag_hours),
                    "Status": "Skipped: no lag features created",
                })
                continue

            metrics = evaluate_feature_configuration(data_with_lags, target_column, lag_features)
            if metrics is None:
                results.append({
                    "Configuration": f"lag_{config_name}",
                    "Feature_Type": "Lag",
                    "Features": f"Lags: {lag_hours}",
                    "N_Features": len(lag_features),
                    "Lag_Hours": str(lag_hours),
                    "Status": "Skipped: evaluation failed or too few valid points",
                })
                continue

            results.append({
                "Configuration": f"lag_{config_name}",
                "Feature_Type": "Lag",
                "Features": f"Lags: {lag_hours}",
                "N_Features": len(lag_features),
                "Lag_Hours": str(lag_hours),
                "Status": "Success",
                **metrics,
            })
            logging.info(
                f"Lag config complete | site={site_name} | config={config_name} | "
                f"RMSE={metrics['RMSE']:.3f} | R={metrics['R']:.3f}"
            )
        except Exception as e:
            logging.exception(f"Lag config {config_name} failed for {site_name}")
            results.append({
                "Configuration": f"lag_{config_name}",
                "Feature_Type": "Lag",
                "Features": f"Lags: {lag_hours}",
                "N_Features": 0,
                "Lag_Hours": str(lag_hours),
                "Status": "Failed",
                "Error": str(e),
            })

    results_df = pd.DataFrame(results)
    if output_prefix:
        save_dataframe(
            results_df,
            f"{output_prefix}_lag_configs.csv",
            columns=CONFIG_RESULT_COLUMNS + ["Lag_Hours"],
        )
    return results_df


def analyze_combined_configurations(site_name, site_dfs, target_column, output_prefix=None):
    logging.info("\n" + "=" * 80)
    logging.info(
        f"COMBINED CONFIGURATION ANALYSIS | SITE = {site_name} | TARGET = {target_column}"
    )
    logging.info("=" * 80)

    if site_name not in site_dfs:
        return pd.DataFrame()

    base_data = site_dfs[site_name].copy()
    results = []

    for combo_name, combo_config in COMBINED_CONFIGS.items():
        logging.info(f"Testing combined config: {combo_name}")
        try:
            enhanced_data = base_data.copy()
            all_features = []
            feature_breakdown = {}

            temporal_config = combo_config["temporal"]
            enhanced_data, temporal_features = get_temporal_features_by_config(
                enhanced_data,
                temporal_config,
            )
            all_features.extend(temporal_features)
            feature_breakdown["temporal"] = len(temporal_features)

            spatial_config_name = combo_config["spatial"]
            enhanced_data, spatial_features = create_spatial_features(
                enhanced_data,
                site_dfs,
                site_name,
                SPATIAL_CONFIGS[spatial_config_name],
                target_column,
            )
            all_features.extend(spatial_features)
            feature_breakdown["spatial"] = len(spatial_features)

            lag_config_name = combo_config["lags"]
            enhanced_data, lag_features = create_lag_features(
                enhanced_data,
                target_column,
                LAG_CONFIGS[lag_config_name],
                include_target=False,
            )
            all_features.extend(lag_features)
            feature_breakdown["lag"] = len(lag_features)

            all_features = list(dict.fromkeys(all_features))
            logging.info(
                f"Combined config features | site={site_name} | config={combo_name} | "
                f"total={len(all_features)} | temporal={feature_breakdown['temporal']} | "
                f"spatial={feature_breakdown['spatial']} | lag={feature_breakdown['lag']}"
            )

            if not all_features:
                logging.warning(f"No features created for combined config {combo_name}")
                results.append({
                    "Configuration": f"combined_{combo_name}",
                    "Feature_Type": "Combined",
                    "Temporal_Config": temporal_config,
                    "Spatial_Config": spatial_config_name,
                    "Lag_Config": lag_config_name,
                    "N_Features": 0,
                    "N_Features_Total": 0,
                    "N_Temporal": feature_breakdown["temporal"],
                    "N_Spatial": feature_breakdown["spatial"],
                    "N_Lag": feature_breakdown["lag"],
                    "Status": "Skipped: no features created",
                })
                continue

            metrics = evaluate_feature_configuration(enhanced_data, target_column, all_features)
            if metrics is None:
                results.append({
                    "Configuration": f"combined_{combo_name}",
                    "Feature_Type": "Combined",
                    "Temporal_Config": temporal_config,
                    "Spatial_Config": spatial_config_name,
                    "Lag_Config": lag_config_name,
                    "N_Features": len(all_features),
                    "N_Features_Total": len(all_features),
                    "N_Temporal": feature_breakdown["temporal"],
                    "N_Spatial": feature_breakdown["spatial"],
                    "N_Lag": feature_breakdown["lag"],
                    "Status": "Skipped: evaluation failed or too few valid points",
                })
                continue

            results.append({
                "Configuration": f"combined_{combo_name}",
                "Feature_Type": "Combined",
                "Temporal_Config": temporal_config,
                "Spatial_Config": spatial_config_name,
                "Lag_Config": lag_config_name,
                "N_Features": len(all_features),
                "N_Features_Total": len(all_features),
                "N_Temporal": feature_breakdown["temporal"],
                "N_Spatial": feature_breakdown["spatial"],
                "N_Lag": feature_breakdown["lag"],
                "Status": "Success",
                **metrics,
            })
            logging.info(
                f"Combined config complete | site={site_name} | config={combo_name} | "
                f"RMSE={metrics['RMSE']:.3f} | R={metrics['R']:.3f}"
            )
        except Exception as e:
            logging.exception(f"Combined config {combo_name} failed for {site_name}")
            results.append({
                "Configuration": f"combined_{combo_name}",
                "Feature_Type": "Combined",
                "Temporal_Config": combo_config.get("temporal"),
                "Spatial_Config": combo_config.get("spatial"),
                "Lag_Config": combo_config.get("lags"),
                "N_Features": 0,
                "N_Features_Total": 0,
                "N_Temporal": np.nan,
                "N_Spatial": np.nan,
                "N_Lag": np.nan,
                "Status": "Failed",
                "Error": str(e),
            })

    results_df = pd.DataFrame(results)
    if output_prefix:
        save_dataframe(
            results_df,
            f"{output_prefix}_combined_configs.csv",
            columns=CONFIG_RESULT_COLUMNS + [
                "Temporal_Config",
                "Spatial_Config",
                "Lag_Config",
                "N_Features_Total",
                "N_Temporal",
                "N_Spatial",
                "N_Lag",
            ],
        )
    return results_df


def plot_comprehensive_analysis(results_df, site_name, target_column, output_prefix):
    if results_df.empty:
        return

    plot_df = results_df.copy()
    plot_df["N_Features"] = pd.to_numeric(plot_df["N_Features"], errors="coerce")
    plot_df["RMSE"] = pd.to_numeric(plot_df["RMSE"], errors="coerce")
    plot_df["R"] = pd.to_numeric(plot_df["R"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    feature_types = plot_df["Feature_Type"].dropna().unique()
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(feature_types), 1)))

    ax = axes[0, 0]
    for idx, feature_type in enumerate(feature_types):
        subset = plot_df[plot_df["Feature_Type"] == feature_type]
        ax.scatter(
            subset["N_Features"],
            subset["RMSE"],
            label=feature_type,
            color=colors[idx],
            s=80,
            alpha=0.7,
        )
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs Number of Features by Type")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for idx, feature_type in enumerate(feature_types):
        subset = plot_df[plot_df["Feature_Type"] == feature_type]
        ax.scatter(
            subset["N_Features"],
            subset["R"],
            label=feature_type,
            color=colors[idx],
            s=80,
            alpha=0.7,
        )
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Correlation Coefficient (R)")
    ax.set_title("Correlation vs Number of Features by Type")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    top10 = plot_df.sort_values("RMSE").head(10)
    bars = ax.barh(range(len(top10)), top10["RMSE"])
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["Configuration"], fontsize=8)
    ax.set_xlabel("RMSE")
    ax.set_title("Top 10 Configurations by RMSE")
    ax.invert_yaxis()
    for bar, val in zip(bars, top10["RMSE"]):
        if pd.notna(val):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                    va="center", ha="left", fontsize=8)

    ax = axes[1, 1]
    scatter = ax.scatter(
        plot_df["N_Features"],
        plot_df["RMSE"],
        c=plot_df["R"],
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("RMSE")
    ax.set_title("Performance vs Complexity (Color = R)")

    if plot_df["RMSE"].notna().any():
        best_idx = plot_df["RMSE"].idxmin()
        best_row = plot_df.loc[best_idx]
        ax.annotate(
            f"Best: {best_row['Configuration']}",
            xy=(best_row["N_Features"], best_row["RMSE"]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    plt.colorbar(scatter, ax=ax, label="Correlation (R)")
    plt.suptitle(f"{site_name} | {target_column} | Comprehensive Feature Analysis")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_prefix}_comprehensive_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_comprehensive_summary(results_df, site_name, target_column, output_prefix):
    if results_df.empty:
        return

    summary_file = OUTPUT_DIR / f"{output_prefix}_comprehensive_summary.txt"
    sorted_df = results_df.sort_values("RMSE").reset_index(drop=True)
    best_config = sorted_df.iloc[0]

    lines = []
    lines.append("=" * 100)
    lines.append("COMPREHENSIVE FEATURE ANALYSIS SUMMARY")
    lines.append(f"Site: {site_name} | Target: {target_column}")
    lines.append("=" * 100)
    lines.append("")
    lines.append("BEST OVERALL CONFIGURATION:")
    lines.append(f"   Configuration: {best_config['Configuration']}")
    lines.append(f"   Feature Type: {best_config['Feature_Type']}")
    lines.append(f"   Features: {best_config['N_Features']}")
    lines.append(f"   RMSE: {best_config['RMSE']:.3f}")
    lines.append(f"   R: {best_config['R']:.3f}")
    lines.append(f"   NSE: {best_config['NSE']:.3f}")
    lines.append("")
    lines.append("BEST BY FEATURE TYPE:")
    for feature_type in sorted_df["Feature_Type"].dropna().unique():
        subset = sorted_df[sorted_df["Feature_Type"] == feature_type]
        best = subset.iloc[0]
        lines.append(f"   {feature_type}: {best['Configuration']} (RMSE: {best['RMSE']:.3f})")

    baseline = sorted_df[sorted_df["Feature_Type"] == "Baseline"]
    if not baseline.empty:
        baseline_rmse = baseline["RMSE"].iloc[0]
        best_rmse = best_config["RMSE"]
        improvement = (
            (baseline_rmse - best_rmse) / baseline_rmse * 100.0
            if pd.notna(baseline_rmse) and baseline_rmse != 0
            else np.nan
        )
        lines.append("")
        lines.append("PERFORMANCE INSIGHTS:")
        lines.append(f"   Baseline RMSE: {baseline_rmse:.3f}")
        lines.append(f"   Best RMSE: {best_rmse:.3f}")
        lines.append(f"   Improvement: {improvement:.1f}%")

    lines.append("")
    lines.append("COMPLEXITY ANALYSIS:")
    simple_configs = sorted_df[sorted_df["N_Features"] <= 10]
    if not simple_configs.empty:
        best_simple = simple_configs.iloc[0]
        lines.append(
            f"   Best simple config (<=10 features): {best_simple['Configuration']} | "
            f"RMSE={best_simple['RMSE']:.3f} | features={best_simple['N_Features']}"
        )

    complex_configs = sorted_df[sorted_df["N_Features"] > 20]
    if not complex_configs.empty:
        best_complex = complex_configs.iloc[0]
        lines.append(
            f"   Best complex config (>20 features): {best_complex['Configuration']} | "
            f"RMSE={best_complex['RMSE']:.3f} | features={best_complex['N_Features']}"
        )

    lines.append("")
    lines.append("TOP 5 RECOMMENDATIONS:")
    for idx, (_, row) in enumerate(sorted_df.head(5).iterrows(), start=1):
        lines.append(
            f"   {idx}. {row['Configuration']}: RMSE={row['RMSE']:.3f}, "
            f"R={row['R']:.3f}, features={row['N_Features']}"
        )
    lines.append("=" * 100)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_comprehensive_feature_analysis(site_name, site_dfs, target_column, output_prefix=None):
    logging.info("\n" + "#" * 100)
    logging.info(f"COMPREHENSIVE FEATURE ANALYSIS: {site_name} | {target_column}")
    logging.info("#" * 100)

    if site_name not in site_dfs:
        logging.error(f"Site {site_name} not found in available sites")
        return pd.DataFrame()

    base_data = ensure_target_column(site_dfs[site_name].copy(), target_column)
    prefix = output_prefix or f"{sanitize_filename(site_name)}_{normalize_target_label(target_column)}"
    all_results = []

    try:
        base_features = get_available_input_features(base_data, target_column)
        metrics = evaluate_feature_configuration(base_data, target_column, base_features)
        if metrics is not None:
            all_results.append({
                "Configuration": "baseline",
                "Feature_Type": "Baseline",
                "Features": ", ".join(base_features),
                "N_Features": len(base_features),
                "Status": "Success",
                **metrics,
            })
        else:
            all_results.append({
                "Configuration": "baseline",
                "Feature_Type": "Baseline",
                "Features": ", ".join(base_features),
                "N_Features": len(base_features),
                "Status": "Skipped: evaluation failed or too few valid points",
            })
    except Exception as e:
        logging.exception(f"Baseline comprehensive analysis failed for {site_name}")
        all_results.append({
            "Configuration": "baseline",
            "Feature_Type": "Baseline",
            "Features": "",
            "N_Features": 0,
            "Status": "Failed",
            "Error": str(e),
        })

    temporal_results = analyze_temporal_configurations(site_name, base_data, target_column, prefix)
    if not temporal_results.empty:
        all_results.extend(temporal_results.to_dict("records"))

    spatial_results = analyze_spatial_configurations(site_name, site_dfs, target_column, prefix)
    if not spatial_results.empty:
        all_results.extend(spatial_results.to_dict("records"))

    lag_results = analyze_lag_configurations(site_name, base_data, target_column, prefix)
    if not lag_results.empty:
        all_results.extend(lag_results.to_dict("records"))

    combined_results = analyze_combined_configurations(site_name, site_dfs, target_column, prefix)
    if not combined_results.empty:
        all_results.extend(combined_results.to_dict("records"))

    if not all_results:
        logging.warning(f"No successful comprehensive configurations for {site_name}")
        empty_df = pd.DataFrame([{
            "Configuration": "none",
            "Feature_Type": "Comprehensive",
            "Features": "",
            "N_Features": 0,
            "Status": "Failed: no configurations were evaluated",
        }]).reindex(columns=CONFIG_RESULT_COLUMNS)
        save_dataframe(
            empty_df,
            f"{prefix}_comprehensive_analysis.csv",
            columns=CONFIG_RESULT_COLUMNS,
        )
        generate_comprehensive_summary(empty_df, site_name, target_column, prefix)
        return pd.DataFrame()

    comprehensive_df = pd.DataFrame(all_results)
    comprehensive_df["N_Features"] = pd.to_numeric(comprehensive_df["N_Features"], errors="coerce")
    comprehensive_df["RMSE"] = pd.to_numeric(comprehensive_df["RMSE"], errors="coerce")
    comprehensive_df = comprehensive_df.sort_values("RMSE").reset_index(drop=True)
    save_dataframe(comprehensive_df, f"{prefix}_comprehensive_analysis.csv")

    plot_comprehensive_analysis(comprehensive_df, site_name, target_column, prefix)
    generate_comprehensive_summary(comprehensive_df, site_name, target_column, prefix)
    logging.info(f"Comprehensive feature analysis complete for {site_name}")
    return comprehensive_df


# =============================================================================
# REGION WORKFLOW
# =============================================================================
def process_single_region(
    region_name,
    site_list,
    target_column="",
    site_available_variables=None,
    region_common_variables=None,
):
    logging.info("\n" + "#" * 100)
    logging.info(f"PROCESSING REGION: {region_name}")
    logging.info("#" * 100)

    site_dfs, metadata_df = load_region_data(site_list, target_column=None)

    if metadata_df is not None and not metadata_df.empty:
        metadata_df.to_csv(
            OUTPUT_DIR / f"{sanitize_filename(region_name)}_site_loading_metadata.csv",
            index=False
        )

    if not site_dfs:
        logging.warning(f"No valid site data loaded for region {region_name}")
        return []

    if target_column is not None and str(target_column).strip() != "":
        targets_to_run = [str(target_column).strip()]
    else:
        targets_to_run = get_targets_from_site_metadata(site_list, site_available_variables or {})

    if not targets_to_run:
        logging.warning(f"No valid targets found for region {region_name}")
        return []

    logging.info(f"Targets to run for region '{region_name}': {targets_to_run}")

    processed_targets = []

    for current_target in targets_to_run:
        logging.info("-" * 80)
        logging.info(f"Running target '{current_target}' for region '{region_name}'")
        logging.info("-" * 80)

        site_dfs_target = filter_site_dfs_for_target(
            site_dfs,
            current_target,
            site_available_variables=site_available_variables,
        )

        if not site_dfs_target:
            logging.warning(f"No sites with target '{current_target}' in region '{region_name}'")
            continue

        if ENABLE_COMPREHENSIVE_FEATURE_ANALYSIS:
            for site_name in site_dfs_target.keys():
                site_prefix = (
                    f"{sanitize_filename(region_name)}_"
                    f"{sanitize_filename(site_name)}_"
                    f"{normalize_target_label(current_target)}"
                )
                try:
                    run_comprehensive_feature_analysis(
                        site_name=site_name,
                        site_dfs=site_dfs_target,
                        target_column=current_target,
                        output_prefix=site_prefix,
                    )
                except Exception:
                    logging.exception(
                        f"Comprehensive feature analysis failed | "
                        f"region={region_name} | site={site_name} | target={current_target}"
                    )

        common_features = get_common_features_for_target(
            site_dfs_target,
            current_target,
            region_common_variables=region_common_variables,
        )

        if not common_features:
            logging.warning(
                f"No common features found across sites in region {region_name} for target {current_target}"
            )
            continue

        logging.info(
            f"Applied regional common input features | region={region_name} | "
            f"target={current_target} | n_features={len(common_features)} | "
            f"features={common_features}"
        )
        for site_name, df_tmp in site_dfs_target.items():
            site_features = get_available_input_features(
                df_tmp,
                current_target,
                allowed_variables=(site_available_variables or {}).get(site_name),
            )
            logging.info(
                f"Applied site input features | region={region_name} | site={site_name} | "
                f"target={current_target} | n_features={len(site_features)} | "
                f"features={site_features}"
            )

        prefix = f"{sanitize_filename(region_name)}_{normalize_target_label(current_target)}"

        # Performance-based feature selection
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
            site_available_variables=site_available_variables,
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

        pearson_df = analyze_target_feature_pearson(
            region_name=region_name,
            site_dfs=site_dfs_target,
            common_features=common_features,
            target_column=current_target,
            output_prefix=prefix,
            site_available_variables=site_available_variables,
        )

        pearson_combo_df = analyze_pearson_guided_feature_combinations(
            region_name=region_name,
            site_dfs=site_dfs_target,
            common_features=common_features,
            pearson_df=pearson_df,
            target_column=current_target,
            output_prefix=prefix,
            site_available_variables=site_available_variables,
            max_combo_size=PEARSON_COMBO_MAX_FEATURES,
        )
        add_guided_combination_results(
            RUN_PEARSON_GUIDED_COMBINATIONS,
            pearson_combo_df,
            guidance_method="Pearson",
            region_name=region_name,
            target_column=current_target,
            source_file=f"{prefix}_pearson_guided_combinations.csv",
        )

        plot_region_site_best_features(
            region_name=region_name,
            target_column=current_target,
            site_summary_df=site_summary_df,
            output_prefix=prefix
        )

        # SHAP analysis
        global_shap_df = analyze_region_global_shap(
            region_name=region_name,
            site_dfs=site_dfs_target,
            common_features=common_features,
            target_column=current_target,
            output_prefix=prefix
        )

        site_shap_df = analyze_region_sites_shap(
            region_name=region_name,
            site_dfs=site_dfs_target,
            target_column=current_target,
            site_available_variables=site_available_variables,
            output_prefix=prefix
        )

        shap_combo_df = analyze_shap_guided_feature_combinations(
            region_name=region_name,
            site_dfs=site_dfs_target,
            common_features=common_features,
            global_shap_df=global_shap_df,
            site_shap_df=site_shap_df,
            target_column=current_target,
            output_prefix=prefix,
            max_combo_size=PEARSON_COMBO_MAX_FEATURES,
        )
        add_guided_combination_results(
            RUN_SHAP_GUIDED_COMBINATIONS,
            shap_combo_df,
            guidance_method="SHAP",
            region_name=region_name,
            target_column=current_target,
            source_file=f"{prefix}_shap_guided_combinations.csv",
        )

        plot_global_vs_local_shap(
            region_name=region_name,
            target_column=current_target,
            global_shap_df=global_shap_df,
            site_shap_df=site_shap_df,
            top_n=PLOT_TOP_N_FEATURES,
            output_prefix=prefix
        )

        generate_region_summary(
            region_name=region_name,
            target_column=current_target,
            global_df=global_df,
            site_summary_df=site_summary_df,
            common_features=common_features,
            global_shap_df=global_shap_df,
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
║      REGION-WISE FEATURE SELECTION: RF + SHAP + TEMPORAL/SPATIAL/LAG        ║
║                                                                              ║
║  Region selection:                                                           ║
║    TARGET_REGIONS = []                    -> all regions                     ║
║    TARGET_REGIONS = ["Upper Hunter"]      -> selected region(s)              ║
║                                                                              ║
║  Target selection:                                                           ║
║    TARGET_COLUMN = "PM2.5"                -> single target                   ║
║    TARGET_COLUMN = ""                     -> all available targets           ║
║                                                                              ║
║  Enhanced site analysis: temporal, spatial, lag, and combined configs         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
    )

    try:
        site_df, region_df, region_sites, site_region, site_available_variables, region_common_variables = read_overview_metadata(
            OVERVIEW_JSON_FILE,
            target_regions=TARGET_REGIONS
        )
    except Exception as e:
        logging.exception(f"Failed to read overview JSON file: {e}")
        return

    if site_df.empty or not region_sites:
        logging.error("No stations/regions found after filtering.")
        return

    site_df.to_csv(OUTPUT_DIR / "station_metadata_filtered.csv", index=False)
    region_df.to_csv(OUTPUT_DIR / "region_metadata_filtered.csv", index=False)

    logging.info(f"Total selected regions: {len(region_sites)}")
    for region_name, site_list in region_sites.items():
        logging.info(f"Region: {region_name} | Sites: {len(site_list)}")

    summary_rows = []

    for region_name, site_list in region_sites.items():
        try:
            processed_targets = process_single_region(
                region_name=region_name,
                site_list=site_list,
                target_column=TARGET_COLUMN,
                site_available_variables=site_available_variables,
                region_common_variables=region_common_variables.get(region_name, []),
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
    save_dataframe(summary_df, "region_processing_summary.csv")
    save_dataframe(summary_df, "enhanced_region_processing_summary.csv")
    save_guided_combination_outputs()
    write_output_manifest()

    logging.info("\n" + "=" * 100)
    logging.info("ALL REGION PROCESSING COMPLETE")
    logging.info(f"Results saved in: {OUTPUT_DIR.resolve()}")
    logging.info("=" * 100)


if __name__ == "__main__":
    main()
