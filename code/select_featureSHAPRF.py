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
import glob
import os
import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from evaluation_metrics import evaluate_metrics
from missingness_regimes import apply_missingness
import config_spatial as config


# =============================================================================
# USER OPTIONS
# =============================================================================

# Examples:
# TARGET_REGIONS = []
# TARGET_REGIONS = ["Upper Hunter"]
TARGET_REGIONS = ["Sydney South-west"]

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
# Legacy overview metadata JSON (no longer required when USE_WIDE_REGION_INPUTS=True)
OVERVIEW_JSON_FILE = None

# Best predictors JSON (region token -> target -> predictors)
BEST_PREDICTORS_JSON_FILE = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/BestPredictors_ByRegionTarget.json"
USE_BEST_PREDICTORS_JSON_INPUTS = True
FALLBACK_TO_SITE_AVAILABLE_VARIABLES = True

# New wide regional inputs (nowcasting pipeline) – optional alternative to per-site CSVs.
WIDE_INPUT_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/cnn_lstm_forecast/API_Input/Inputs"
USE_WIDE_REGION_INPUTS = True

OUTPUT_DIR = (
    Path(getattr(config, "OUTPUT_DIRECTORY", "."))
    / "feature_assessment"
    / "feature_selection_results_regionwise_rf_shap"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORRELATION_DIR = OUTPUT_DIR / "Region_Correlation"
CORRELATION_DIR.mkdir(parents=True, exist_ok=True)

PLOT_TOP_N_FEATURES = 10
PEARSON_COMBO_MAX_FEATURES = None  # None means use all Pearson-ranked features.
MIN_VALID_POINTS = 10
USE_TIME_FEATURES = "no"  # "yes" or "no"
TIME_FEATURE_COLUMNS = [
    "hour", "hour_sin", "hour_cos", "month_sin",
    "dayofweek", "dayofyear", "month", "month_cos",
]
SHAP_MAX_SAMPLES = 1000

# Populated at runtime when USE_WIDE_REGION_INPUTS=True
WIDE_REGION_TO_PATH = {}
WIDE_SITE_TO_REGION = {}

# Loaded at runtime
BEST_PREDICTORS_MAP = {}


def load_best_predictors_json(path):
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception as e:
        logging.warning("Failed to load best predictors JSON: %s | %s", path, e)
        return {}


def best_predictors_for_region_target(best_map, region_token, target_column):
    if not best_map or not region_token:
        return []
    region_block = best_map.get(region_token)
    if not isinstance(region_block, dict):
        # try normalized region token match
        key_norm = normalize_name(region_token)
        for k, v in best_map.items():
            if normalize_name(k) == key_norm:
                region_block = v
                break
    if not isinstance(region_block, dict):
        return []
    preds = region_block.get(target_column)
    if preds is None:
        # try a couple of common aliases
        if str(target_column).strip().upper() == "O3":
            preds = region_block.get("OZONE")
    if preds is None:
        return []
    if isinstance(preds, list):
        return [str(x).strip() for x in preds if str(x).strip()]
    return [str(preds).strip()] if str(preds).strip() else []

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


def region_token_from_name(region_name):
    """Convert a human region name into the wide-file token (underscored)."""
    token = str(region_name).strip().replace("-", "_").replace(" ", "_")
    token = re.sub(r"_+", "_", token)
    return token


def list_wide_region_files(wide_dir):
    pattern = os.path.join(wide_dir, "Allobs_processed_DPE_station_api_*_ALL.csv")
    files = sorted(glob.glob(pattern))
    region_to_path = {}
    for fp in files:
        base = os.path.basename(fp)
        if not (base.startswith("Allobs_processed_DPE_station_api_") and base.endswith("_ALL.csv")):
            continue
        token = base[len("Allobs_processed_DPE_station_api_") : -len("_ALL.csv")]
        region_to_path[token] = fp
    return region_to_path


def wide_region_sites_and_vars(wide_csv_path):
    cols = pd.read_csv(wide_csv_path, nrows=0).columns
    sites = {}
    for c in cols:
        if not isinstance(c, str):
            continue
        if c.lower() == "datetime" or c == "DateTime":
            continue
        if "_" not in c:
            continue
        var, site = c.split("_", 1)
        var = str(var).strip()
        site = str(site).strip()
        if not var or not site:
            continue
        sites.setdefault(site, set()).add(var)
    site_vars = {k: sorted(v) for k, v in sites.items()}
    return sorted(site_vars.keys()), site_vars


def build_overview_from_wide_inputs(wide_dir, target_regions=None):
    """Build the same metadata structures as overview_json, but from wide region CSVs."""
    region_to_path = list_wide_region_files(wide_dir)
    if not region_to_path:
        raise FileNotFoundError(f"No wide region files found under: {wide_dir}")

    if target_regions:
        available_tokens = sorted(region_to_path.keys())
        wanted = {normalize_name(r) for r in target_regions}
        region_to_path = {
            token: fp
            for token, fp in region_to_path.items()
            if normalize_name(token) in wanted or normalize_name(region_token_from_name(token)) in wanted or normalize_name(token.replace("_", " ")) in wanted
        }
        if not region_to_path:
            raise ValueError(
                "No wide region inputs matched TARGET_REGIONS="
                f"{target_regions}. Available regions: {available_tokens}"
            )

    sites_rows = []
    region_rows = []
    region_sites = {}
    site_region = {}
    site_available_variables = {}
    region_common_variables = {}

    for region_token, fp in sorted(region_to_path.items()):
        site_list, site_vars = wide_region_sites_and_vars(fp)
        if not site_list:
            continue
        region_sites[region_token] = site_list
        for site in site_list:
            site_region[site] = region_token
            vars_for_site = sorted(site_vars.get(site, []))
            site_available_variables[site] = vars_for_site
            sites_rows.append({
                "Region": region_token,
                "SiteName": site,
                "AvailableVariables": ", ".join(vars_for_site),
            })

        # Common variables across all sites in region
        common = None
        for site in site_list:
            svars = set(site_vars.get(site, []))
            common = svars if common is None else (common & svars)
        common_list = sorted(common) if common else []
        region_common_variables[region_token] = common_list
        region_rows.append({
            "Region": region_token,
            "Sites": ", ".join(site_list),
            "CommonVariables": ", ".join(common_list),
        })

    site_df = pd.DataFrame(sites_rows)
    region_df = pd.DataFrame(region_rows)
    return site_df, region_df, region_sites, site_region, site_available_variables, region_common_variables, region_to_path


def save_region_variable_correlation(region_name, wide_csv_path):
    """Compute a compact region-level correlogram by averaging each variable across sites."""
    try:
        df = pd.read_csv(wide_csv_path, low_memory=False)
    except Exception as e:
        logging.warning("Failed to read wide region file for correlation: %s | %s", wide_csv_path, e)
        return None

    dt_col = "datetime" if "datetime" in df.columns else "DateTime" if "DateTime" in df.columns else None
    if not dt_col:
        return None

    dt = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.drop(columns=[dt_col], errors="ignore")

    var_to_cols = {}
    for c in df.columns:
        if not isinstance(c, str) or "_" not in c:
            continue
        var, _ = c.split("_", 1)
        var = str(var).strip()
        if not var:
            continue
        var_to_cols.setdefault(var, []).append(c)

    if not var_to_cols:
        return None

    region_mean = {}
    for var, cols in var_to_cols.items():
        try:
            region_mean[var] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        except Exception:
            continue

    mean_df = pd.DataFrame(region_mean)
    mean_df.insert(0, "DateTime", dt)
    mean_df = mean_df.dropna(subset=["DateTime"]).copy()

    corr_df = mean_df.drop(columns=["DateTime"], errors="ignore").corr()
    if corr_df.empty:
        return None

    prefix = sanitize_filename(region_name)
    out_csv = CORRELATION_DIR / f"{prefix}_region_mean_variable_correlation.csv"
    corr_df.to_csv(out_csv)

    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=False, square=True)
        plt.title(f"Region mean-variable correlation: {region_name}")
        plt.tight_layout()
        out_png = CORRELATION_DIR / f"{prefix}_region_mean_variable_correlation.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logging.warning("Failed to plot correlation heatmap for %s: %s", region_name, e)

    return corr_df


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


def split_predictor_list(features):
    if features is None or (isinstance(features, float) and pd.isna(features)):
        return []
    return [feature.strip() for feature in str(features).split(",") if feature.strip()]


def best_guided_combination_records(combo_df, guidance_method, row_type):
    if combo_df is None or combo_df.empty:
        return []

    df = combo_df.copy()
    df = df[df["RowType"] == row_type].copy()
    if df.empty:
        return []

    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df = df[df["RMSE"].notna()].copy()
    if df.empty:
        return []

    group_cols = ["Guidance_Method", "Region", "Target", "RowType", "RowName"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = ""

    best_df = (
        df.sort_values(group_cols + ["RMSE", "ComboRank"])
        .groupby(group_cols, as_index=False)
        .first()
        .sort_values(["Region", "Target", "RowName"])
        .reset_index(drop=True)
    )

    records = []
    for _, row in best_df.iterrows():
        predictors = split_predictor_list(row.get("Features"))
        record = {
            "guidance_method": guidance_method,
            "region": row.get("Region"),
            "target": row.get("Target"),
            "level": "site" if row_type == "Site" else "region",
            "site": row.get("RowName") if row_type == "Site" else None,
            "region_row": row.get("RowName") if row_type == "Region" else None,
            "combo_rank": int(row["ComboRank"]) if pd.notna(row.get("ComboRank")) else None,
            "combo_label": row.get("ComboLabel"),
            "predictors": predictors,
            "n_predictors": int(row["N_Features"]) if pd.notna(row.get("N_Features")) else len(predictors),
            "metrics": {
                "RMSE": float(row["RMSE"]) if pd.notna(row.get("RMSE")) else None,
                "R": float(row["R"]) if pd.notna(row.get("R")) else None,
                "WI": float(row["WI"]) if pd.notna(row.get("WI")) else None,
                "NSE": float(row["NSE"]) if pd.notna(row.get("NSE")) else None,
                "MAE": float(row["MAE"]) if pd.notna(row.get("MAE")) else None,
            },
            "source_file": row.get("Source_File"),
        }
        records.append(record)

    return records


def save_best_guided_json(combo_df, guidance_method, row_type, filename):
    records = best_guided_combination_records(combo_df, guidance_method, row_type)
    payload = {
        "guidance_method": guidance_method,
        "level": "site" if row_type == "Site" else "region",
        "selection_metric": "lowest RMSE",
        "n_records": len(records),
        "records": records,
    }

    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info(f"Saved best guided JSON: rows={len(records)} | file={OUTPUT_DIR / filename}")
    return payload


def save_best_guided_json_outputs(pearson_df, shap_df):
    outputs = {
        "pearson_site": save_best_guided_json(
            pearson_df,
            guidance_method="Pearson",
            row_type="Site",
            filename="pearson_guided_best_site_predictor_combinations.json",
        ),
        "pearson_region": save_best_guided_json(
            pearson_df,
            guidance_method="Pearson",
            row_type="Region",
            filename="pearson_guided_best_region_predictor_combinations.json",
        ),
        "shap_site": save_best_guided_json(
            shap_df,
            guidance_method="SHAP",
            row_type="Site",
            filename="shap_guided_best_site_predictor_combinations.json",
        ),
        "shap_region": save_best_guided_json(
            shap_df,
            guidance_method="SHAP",
            row_type="Region",
            filename="shap_guided_best_region_predictor_combinations.json",
        ),
    }
    return outputs


def save_guided_combination_outputs():
    if RUN_PEARSON_GUIDED_COMBINATIONS:
        pearson_df = pd.concat(RUN_PEARSON_GUIDED_COMBINATIONS, ignore_index=True)
    else:
        pearson_df = pd.DataFrame(columns=GUIDED_COMBINATION_COLUMNS)

    if RUN_SHAP_GUIDED_COMBINATIONS:
        shap_df = pd.concat(RUN_SHAP_GUIDED_COMBINATIONS, ignore_index=True)
    else:
        shap_df = pd.DataFrame(columns=GUIDED_COMBINATION_COLUMNS)

    pearson_df.to_csv(OUTPUT_DIR / "all_pearson_guided_combinations.csv", index=False)
    shap_df.to_csv(OUTPUT_DIR / "all_shap_guided_combinations.csv", index=False)

    logging.info(
        f"Saved all Pearson-guided combinations: rows={len(pearson_df)} | "
        f"file={OUTPUT_DIR / 'all_pearson_guided_combinations.csv'}"
    )
    logging.info(
        f"Saved all SHAP-guided combinations: rows={len(shap_df)} | "
        f"file={OUTPUT_DIR / 'all_shap_guided_combinations.csv'}"
    )
    save_best_guided_json_outputs(pearson_df, shap_df)

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

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


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
    # Prefer wide region inputs when enabled.
    if USE_WIDE_REGION_INPUTS:
        site_key = normalize_name(site_name)
        region_token = WIDE_SITE_TO_REGION.get(site_name)
        if region_token is None:
            # try normalized match
            for k, v in WIDE_SITE_TO_REGION.items():
                if normalize_name(k) == site_key:
                    region_token = v
                    site_name = k
                    break

        if region_token is None:
            raise FileNotFoundError(f"No wide-region mapping found for site: {site_name}")

        wide_path = WIDE_REGION_TO_PATH.get(region_token)
        if not wide_path:
            raise FileNotFoundError(f"No wide-region file path found for region: {region_token}")

        df_wide = pd.read_csv(wide_path, low_memory=False)
        dt_col = "datetime" if "datetime" in df_wide.columns else "DateTime" if "DateTime" in df_wide.columns else None
        if not dt_col:
            raise ValueError(f"Wide input missing datetime column: {wide_path}")

        wanted_cols = [dt_col]
        for c in df_wide.columns:
            if c == dt_col or not isinstance(c, str) or "_" not in c:
                continue
            var, suffix = c.split("_", 1)
            if normalize_name(suffix) == site_key:
                wanted_cols.append(c)

        if len(wanted_cols) <= 1:
            raise FileNotFoundError(f"No columns found for site {site_name} in wide file {wide_path}")

        df = df_wide[wanted_cols].copy().rename(columns={dt_col: "DateTime"})

        # Rename VAR_SITE -> VAR
        column_mapping = {}
        for c in list(df.columns):
            if c == "DateTime":
                column_mapping[c] = c
                continue
            if "_" in c:
                base, _ = c.split("_", 1)
                column_mapping[c] = base
            else:
                column_mapping[c] = c
        df = df.rename(columns=column_mapping)

        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime"]).copy()
        df["Site"] = site_name
        df = add_time_features(df)

        if target_column is not None and str(target_column).strip() != "":
            df = ensure_target_column(df, target_column)

        return df, wide_path, column_mapping

    # Legacy per-site files
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
                f"Skipping site '{site_name}' for target '{target_column}' based on available-variables metadata."
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
    if not SHAP_AVAILABLE:
        return pd.DataFrame(columns=["Feature", "MeanAbsSHAP"])
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
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available; skipping global SHAP for region=%s target=%s", region_name, target_column)
        return pd.DataFrame()
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
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available; skipping site SHAP for region=%s target=%s", region_name, target_column)
        return pd.DataFrame()
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
        try:
            v = float(val)
        except Exception:
            v = float("nan")
        if not np.isfinite(v):
            continue
        ax.text(v, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
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
        try:
            v = float(val)
        except Exception:
            v = float("nan")
        if not np.isfinite(v):
            continue
        ax.text(v, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
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
        try:
            v = float(val)
        except Exception:
            v = float("nan")
        if not np.isfinite(v):
            continue
        ax.text(v, bar.get_y() + bar.get_height()/2, f"{v:.3f}",
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
        rmse_val = row.get("Best_RMSE")
        try:
            x = float(rmse_val)
        except Exception:
            x = float("nan")
        if not np.isfinite(x):
            x = 0.0
        try:
            label_rmse = float(rmse_val)
            label_rmse_str = f"{label_rmse:.2f}" if np.isfinite(label_rmse) else "nan"
        except Exception:
            label_rmse_str = "nan"
        label = f"{row.get('Best_Feature', '')} | {label_rmse_str}"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f" {label}",
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

        # Apply best-predictors guidance (optional): restrict input variables
        # to the region+target list from BestPredictors_ByRegionTarget.json.
        region_token = region_name
        guided_predictors = (
            best_predictors_for_region_target(BEST_PREDICTORS_MAP, region_token, current_target)
            if USE_BEST_PREDICTORS_JSON_INPUTS
            else []
        )

        if guided_predictors:
            logging.info(
                "Using BEST predictors for inputs | region=%s | target=%s | predictors=%s",
                region_name,
                current_target,
                guided_predictors,
            )

            # Filter site_available_variables to guided list (so downstream uses only these)
            if site_available_variables is not None:
                filtered_site_available = {}
                for s in site_dfs_target.keys():
                    svars = list(site_available_variables.get(s, []) or [])
                    filtered_site_available[s] = [v for v in svars if v in guided_predictors]
                site_available_variables_target = filtered_site_available
            else:
                site_available_variables_target = {s: guided_predictors for s in site_dfs_target.keys()}

            # Filter regional-common vars too
            region_common_for_target = [
                v
                for v in (region_common_variables or [])
                if v in guided_predictors
            ]
        else:
            site_available_variables_target = site_available_variables
            region_common_for_target = region_common_variables or []

        if guided_predictors:
            # Use the guided list as the region feature pool (union across sites).
            # Missing values per-site are handled via SimpleImputer in the RF workflow.
            common_features = []
            for feat in guided_predictors:
                if feat == current_target:
                    continue
                if any(feat in d.columns for d in site_dfs_target.values()):
                    common_features.append(feat)
            common_features = sorted(set(common_features))
        else:
            common_features = get_common_features_for_target(
                site_dfs_target,
                current_target,
                region_common_variables=region_common_for_target,
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
                allowed_variables=(site_available_variables_target or {}).get(site_name),
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
            site_available_variables=site_available_variables_target,
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
            site_available_variables=site_available_variables_target,
        )

        pearson_combo_df = analyze_pearson_guided_feature_combinations(
            region_name=region_name,
            site_dfs=site_dfs_target,
            common_features=common_features,
            pearson_df=pearson_df,
            target_column=current_target,
            output_prefix=prefix,
            site_available_variables=site_available_variables_target,
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
            site_available_variables=site_available_variables_target,
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
║             REGION-WISE FEATURE SELECTION USING RANDOM FOREST + SHAP        ║
║                                                                              ║
║  Examples (edit USER OPTIONS at top):                                        ║
║    TARGET_REGIONS = []                    -> all regions                     ║
║    TARGET_REGIONS = ["Upper Hunter"]      -> selected region(s)              ║
║    TARGET_COLUMN  = "PM2.5"               -> single target                   ║
║    TARGET_COLUMN  = ""                    -> all available targets           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
    )
    print(f"Configured TARGET_REGIONS = {TARGET_REGIONS}")
    print(f"Configured TARGET_COLUMN  = {TARGET_COLUMN!r} (empty means all targets)")
    print(f"Configured USE_WIDE_REGION_INPUTS = {USE_WIDE_REGION_INPUTS} | WIDE_INPUT_DIR = {WIDE_INPUT_DIR!r}")

    try:
        region_to_path = {}
        if USE_WIDE_REGION_INPUTS:
            (
                site_df,
                region_df,
                region_sites,
                site_region,
                site_available_variables,
                region_common_variables,
                region_to_path,
            ) = build_overview_from_wide_inputs(WIDE_INPUT_DIR, target_regions=TARGET_REGIONS)

            # Publish for load_site_data()
            global WIDE_REGION_TO_PATH, WIDE_SITE_TO_REGION
            WIDE_REGION_TO_PATH = dict(region_to_path)
            WIDE_SITE_TO_REGION = dict(site_region)
        else:
            if not OVERVIEW_JSON_FILE:
                raise ValueError("OVERVIEW_JSON_FILE is not set and USE_WIDE_REGION_INPUTS is False")
            site_df, region_df, region_sites, site_region, site_available_variables, region_common_variables = read_overview_metadata(
                OVERVIEW_JSON_FILE,
                target_regions=TARGET_REGIONS
            )

        # Best predictors mapping (optional)
        global BEST_PREDICTORS_MAP
        if USE_BEST_PREDICTORS_JSON_INPUTS:
            BEST_PREDICTORS_MAP = load_best_predictors_json(BEST_PREDICTORS_JSON_FILE)
        else:
            BEST_PREDICTORS_MAP = {}
    except Exception as e:
        logging.exception(f"Failed to load region/site metadata: {e}")
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
            if USE_WIDE_REGION_INPUTS:
                wide_fp = region_to_path.get(region_name)
                if wide_fp:
                    save_region_variable_correlation(region_name, wide_fp)

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
    summary_df.to_csv(OUTPUT_DIR / "region_processing_summary.csv", index=False)
    save_guided_combination_outputs()

    logging.info("\n" + "=" * 100)
    logging.info("ALL REGION PROCESSING COMPLETE")
    logging.info(f"Results saved in: {OUTPUT_DIR.resolve()}")
    logging.info("=" * 100)


if __name__ == "__main__":
    main()
