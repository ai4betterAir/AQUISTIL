"""
Random-Forest Best Individual Feature Selection for PM2.5 Imputation

Purpose
-------
This script finds the best individual feature for PM2.5 imputation using the
one consistent Random Forest evaluator across feature-selection stages.

Default experiment:
- Target: PM2.5
- Missingness: 10%
- Regime: random
- Model: Random Forest
- Feature test: one predictor at a time
- Outputs: answer CSVs + plots

This script intentionally does NOT use:
- Random Forest
- XGBRF
- SHAP as feature selection
- robustness/sensitivity models
- IDW / Kriging / spatial neighbour combinations

Main outputs
------------
1. ANSWER_best_feature_summary.csv
2. ANSWER_best_feature_by_site.csv
3. ANSWER_best_feature_by_region.csv
4. ANSWER_overall_feature_ranking.csv
5. all_random_forest_individual_feature_results.csv
6. all_random_forest_individual_feature_seed_results.csv

Plots
-----
1. plot_overall_feature_ranking.png
2. plot_best_feature_by_site.png
3. plot_best_feature_by_region.png
4. plot_site_feature_rmse_heatmap.png
5. plot_region_feature_rmse_heatmap.png

Expected project modules
------------------------
- config_spatial.py
- missingness_regimes.py

Optional project module
-----------------------
- evaluation_metrics.py

Author: Dr. Masrur
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

_SCRIPT_PATH = Path(__file__).resolve()
APP_ROOT_DIR = _SCRIPT_PATH.parents[2]
for _candidate in (_SCRIPT_PATH.parents[1], _SCRIPT_PATH.parents[2]):
    if (_candidate / "config_spatial.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import config_spatial as config
from missingness_regimes import apply_missingness

try:
    from evaluation_metrics import evaluate_metrics
except Exception:
    evaluate_metrics = None

# =============================================================================
# SETTINGS
# =============================================================================

AVAILABLE_TARGETS = [
    "PM2.5",
    "PM10",
    "OZONE",
    "NO",
    "NO2",
    "NOX",
    "CO",
]

AVAILABLE_REGIONS = [
    "Central Coast",
    "Central Tablelands",
    "Lower Hunter",
    "Newcastle Local",
    "Northern Tablelands",
    "Southern Tablelands",
    "Sydney",
    "Sydney East",
    "Sydney North west",
    "Sydney South west",
    "Upper Hunter",
]

SELECTED_TARGET = "PM2.5"
SELECTED_TARGETS = AVAILABLE_TARGETS
SELECTED_REGIONS = AVAILABLE_REGIONS

# To run all targets in-code, use:
# SELECTED_TARGETS = AVAILABLE_TARGETS[:]
#
# To run all regions in-code, use:
# SELECTED_REGIONS = AVAILABLE_REGIONS[:]

TARGET_COLUMNS_ENV = os.getenv("FS_TARGET_COLUMNS", "").strip()
TARGET_COLUMN_ENV = os.getenv("FS_TARGET_COLUMN", "").strip()
if TARGET_COLUMNS_ENV:
    TARGET_COLUMNS = [x.strip() for x in TARGET_COLUMNS_ENV.split(",") if x.strip()]
elif TARGET_COLUMN_ENV:
    TARGET_COLUMNS = [TARGET_COLUMN_ENV]
else:
    TARGET_COLUMNS = [
        x.strip()
        for x in SELECTED_TARGETS
        if x.strip()
    ] or [SELECTED_TARGET]

# For finding best feature, 10% random is a better default than 1%.
MISSINGNESS_LEVELS = [
    float(x.strip())
    for x in os.getenv("FS_MISSINGNESS_LEVELS", "0.10").split(",")
    if x.strip()
]

MISSINGNESS_REGIMES = [
    x.strip()
    for x in os.getenv("FS_MISSINGNESS_REGIMES", "random").split(",")
    if x.strip()
]
FEATURE_SELECTION_RUN_MODE = os.getenv("FEATURE_SELECTION_RUN_MODE", "").strip().lower()
if FEATURE_SELECTION_RUN_MODE == "event" and "FS_MISSINGNESS_REGIMES" not in os.environ:
    MISSINGNESS_REGIMES = ["event"]
EVENT_RUN = FEATURE_SELECTION_RUN_MODE == "event" or (
    len(MISSINGNESS_REGIMES) == 1 and MISSINGNESS_REGIMES[0].lower() == "event"
)
RUN_TYPE_LABEL = "EVENT FEATURE SELECTION RUN" if EVENT_RUN else "DEFAULT FEATURE SELECTION RUN"

SEEDS = [
    int(x.strip())
    for x in os.getenv("FS_SEEDS", "42,101,202").split(",")
    if x.strip()
]

# Keep as 0 for pure individual-feature selection.
# Set FS_ADD_TEMPORAL=1 if you want each feature to be tested as feature + time.
ADD_TEMPORAL_FEATURES = os.getenv("FS_ADD_TEMPORAL", "0").strip().lower() in {
    "1", "true", "yes"
}
INCLUDE_ALL_INPUTS = os.getenv("FS_INCLUDE_ALL_INPUTS", "0").strip().lower() in {
    "1", "true", "yes"
}

DEFAULT_WIDE_INPUT_DIR = APP_ROOT_DIR / "API_Input" / "Inputs"
INPUT_DIR = Path(
    os.getenv(
        "FS_INPUT_DIR",
        str(DEFAULT_WIDE_INPUT_DIR),
    )
).resolve()

FEATURE_SELECTION_OUTPUT_ROOT = Path(
    os.getenv(
        "FEATURE_SELECTION_OUTPUT_ROOT",
        str(APP_ROOT_DIR / "Outputs" / "Feature_Selection"),
    )
)
if EVENT_RUN:
    FEATURE_SELECTION_OUTPUT_ROOT = FEATURE_SELECTION_OUTPUT_ROOT / "feature_selection_event"
DEFAULT_OUTPUT_ROOT = FEATURE_SELECTION_OUTPUT_ROOT / "00RandomForest_Best_Individual_Feature_Selection"
OUTPUT_ROOT = Path(os.getenv("FS_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))

RESULTS_DIR = OUTPUT_ROOT / "individual_results"
PLOTS_DIR = RESULTS_DIR
PREPARED_SITE_INPUT_DIR = OUTPUT_ROOT / "_prepared_site_inputs"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PREPARED_SITE_INPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN_ROWS = int(os.getenv("FS_MIN_TRAIN_ROWS", "100"))
MIN_TEST_ROWS = int(os.getenv("FS_MIN_TEST_ROWS", "30"))

# If you only want some sites:
# export FS_TARGET_SITES=ALEXANDRIA,ALBION-PARK-SOUTH
TARGET_SITES_ENV = os.getenv("FS_TARGET_SITES", "").strip()
TARGET_REGIONS_ENV = os.getenv("FS_TARGET_REGIONS", "").strip()

# Region mapping CSV, optional:
# export SITE_REGION_CSV=/path/to/site_region.csv
# Required columns: Site, Region
SITE_REGION_CSV = os.getenv("SITE_REGION_CSV", "").strip()

# Region vs individual-sites comparison plotting.
# By default, create this comparison for the first available region only.
REGION_COMPARE_NAME = os.getenv("FS_REGION_COMPARE_NAME", "").strip()
REGION_COMPARE_LIMIT = int(os.getenv("FS_REGION_COMPARE_LIMIT", "1"))
ACTIVE_SITE_INPUT_DIR = INPUT_DIR
REGION_MAP_OVERRIDE: Dict[str, str] = {}

DEFAULT_INPUT_COLUMNS = list(getattr(config, "LOCAL_ANALYSIS_INPUTS", [
    "CO", "HUMID", "NEPH", "NO", "NO2", "NOX",
    "OZONE", "PM10", "RAIN", "TEMP", "WDR", "WSP",
]))
INPUT_COLUMNS_ENV = os.getenv("FS_INPUT_COLUMNS", "").strip()


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "lightgbm_best_feature_selection.log"),
        logging.StreamHandler(),
    ],
    force=True,
)


# =============================================================================
# MODEL
# =============================================================================

def make_random_forest_model(seed: int) -> Pipeline:
    """Create the shared RF feature-selection model used in Stages 0–3."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=int(os.getenv("RF_N_ESTIMATORS", "120")),
            max_depth=int(os.getenv("RF_MAX_DEPTH", "16")),
            min_samples_leaf=int(os.getenv("RF_MIN_SAMPLES_LEAF", "5")),
            max_features=os.getenv("RF_MAX_FEATURES", "sqrt"),
            random_state=seed,
            n_jobs=int(os.getenv("N_JOBS", "-1")),
        )),
    ])


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def normalize_token(value) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(value)).lower()


def clean_filename(value) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_")
    return out or "NA"


def infer_site_name(csv_path: Path) -> str:
    """
    Infer site name from CSV file name.

    Examples:
    ALEXANDRIA_processed.csv -> ALEXANDRIA
    ALBION-PARK-SOUTH_AQMS.csv -> ALBION-PARK-SOUTH
    """
    stem = csv_path.stem

    suffixes = [
        "_processed",
        "_aqms",
        "_station",
        "_site",
        "_data",
        "_clean",
        "_hourly",
        "_merged",
    ]

    for suffix in suffixes:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]

    if "_" in stem:
        stem = stem.split("_")[0]

    return stem.strip().upper()


def ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DateTime column exists and is parsed."""
    out = df.copy()

    if "DateTime" in out.columns:
        out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
        return out

    lower_map = {str(col).strip().lower(): col for col in out.columns}

    date_col = lower_map.get("date")
    time_col = lower_map.get("time")
    timestamp_col = lower_map.get("timestamp")
    datetime_col = lower_map.get("datetime")

    if datetime_col is not None:
        out["DateTime"] = pd.to_datetime(out[datetime_col], errors="coerce")
        return out

    if date_col is not None and time_col is not None:
        out["DateTime"] = pd.to_datetime(
            out[date_col].astype(str) + " " + out[time_col].astype(str),
            errors="coerce",
            dayfirst=True,
        )
        return out

    if timestamp_col is not None:
        out["DateTime"] = pd.to_datetime(out[timestamp_col], errors="coerce")
        return out

    raise KeyError(
        "No DateTime or Date/Time columns found. "
        f"Available columns: {list(out.columns)[:30]}"
    )


def infer_region_name(csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = "Allobs_processed_DPE_station_api_"
    suffix = "_ALL"
    if stem.startswith(prefix) and stem.endswith(suffix):
        token = stem[len(prefix) : -len(suffix)]
        return str(token).replace("_", " ").strip()
    return stem.replace("_", " ").strip()


def is_wide_region_input_dir(input_dir: Path) -> bool:
    return any(input_dir.glob("Allobs_processed_DPE_station_api_*_ALL.csv"))


def list_sites_from_wide_csv(csv_path: Path) -> List[str]:
    cols = pd.read_csv(csv_path, nrows=0).columns
    sites = set()
    for col in cols:
        if not isinstance(col, str) or col in {"DateTime", "datetime"} or "_" not in col:
            continue
        _, site = col.split("_", 1)
        if site.strip():
            sites.add(site.strip())
    return sorted(sites)


def build_per_site_csv_from_wide(wide_csv_path: Path, site_name: str, out_dir: Path) -> Path:
    df = pd.read_csv(wide_csv_path, low_memory=False)
    df = ensure_datetime_column(df)

    wanted_cols = ["DateTime"]
    for col in df.columns:
        if not isinstance(col, str) or col == "DateTime" or "_" not in col:
            continue
        var, site = col.split("_", 1)
        if normalize_token(site) == normalize_token(site_name):
            wanted_cols.append(col)

    if len(wanted_cols) <= 1:
        raise ValueError(f"No columns found for site {site_name} in {wide_csv_path}")

    sub = df[wanted_cols].copy()
    rename_map = {}
    for col in sub.columns:
        if col == "DateTime":
            continue
        var, _ = col.split("_", 1)
        rename_map[col] = var
    sub = sub.rename(columns=rename_map)
    sub = sub.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)

    out_path = out_dir / f"{site_name}.csv"
    sub.to_csv(out_path, index=False)
    return out_path


def prepare_site_inputs_from_wide(input_dir: Path) -> Dict[str, str]:
    region_map: Dict[str, str] = {}
    for old_csv in PREPARED_SITE_INPUT_DIR.glob("*.csv"):
        try:
            old_csv.unlink()
        except Exception:
            pass

    files = sorted(input_dir.glob("Allobs_processed_DPE_station_api_*_ALL.csv"))
    if not files:
        raise FileNotFoundError(f"No wide regional CSV files found in {input_dir}")

    for wide_csv_path in files:
        region_name = infer_region_name(wide_csv_path)
        for site_name in list_sites_from_wide_csv(wide_csv_path):
            build_per_site_csv_from_wide(wide_csv_path, site_name, PREPARED_SITE_INPUT_DIR)
            region_map[normalize_token(site_name)] = region_name

    logging.info(
        "Prepared %d per-site CSVs from %d wide regional file(s) into %s",
        len(region_map),
        len(files),
        PREPARED_SITE_INPUT_DIR,
    )
    return region_map


def resolve_target_column(df: pd.DataFrame, target_column: str) -> Optional[str]:
    """Resolve PM2.5 / PM25 / PM2_5 / PM2.5_SITE etc."""
    if target_column in df.columns:
        return target_column

    target_norm = normalize_token(target_column)

    aliases = {
        "pm25": {"pm25", "pm2p5", "pm2_5", "pm2.5", "pm_25", "pm_2_5"},
        "pm10": {"pm10", "pm_10"},
    }

    for col in df.columns:
        if normalize_token(col) == target_norm:
            return col

    for alias_group in aliases.values():
        alias_norms = {normalize_token(x) for x in alias_group}
        if target_norm in alias_norms:
            for col in df.columns:
                if normalize_token(col) in alias_norms:
                    return col

    # Site-specific wide column, e.g., PM2.5_ALEXANDRIA
    for col in df.columns:
        col_norm = normalize_token(col)
        if col_norm.startswith(target_norm):
            return col

    return None


def resolve_input_column(
    df: pd.DataFrame,
    input_column: str,
    site_name: Optional[str] = None,
) -> Optional[str]:
    """Resolve TEMP / TEMP_SITE / WSP_SITE etc."""
    if input_column in df.columns:
        return input_column

    input_norm = normalize_token(input_column)
    site_norm = normalize_token(site_name) if site_name else ""

    exact_matches = [col for col in df.columns if normalize_token(col) == input_norm]
    if exact_matches:
        return exact_matches[0]

    prefix_matches = [
        col for col in df.columns
        if normalize_token(col).startswith(input_norm)
    ]

    if site_norm:
        site_matches = [
            col for col in prefix_matches
            if site_norm in normalize_token(col)
        ]
        if site_matches:
            return site_matches[0]

    if prefix_matches:
        return prefix_matches[0]

    return None


def csv_has_target_column(csv_path: Path, target_column: str) -> bool:
    try:
        header = pd.read_csv(csv_path, nrows=0)
        return resolve_target_column(header, target_column) is not None
    except Exception:
        return False


# =============================================================================
# REGION MAP
# =============================================================================

def load_region_map() -> Dict[str, str]:
    """
    Load site-to-region mapping.

    Priority:
    1. SITE_REGION_CSV with columns Site, Region
    2. config.SITE_REGION_MAP / config.SITE_TO_REGION = {"ALEXANDRIA": "Sydney Metro"}
    3. config.SITE_REGIONS / config.REGION_TO_SITES = {"Sydney Metro": ["ALEXANDRIA", ...]}
    4. fallback: ALL_SITES_POOLED
    """
    if REGION_MAP_OVERRIDE:
        return dict(REGION_MAP_OVERRIDE)

    region_map: Dict[str, str] = {}

    if SITE_REGION_CSV and Path(SITE_REGION_CSV).exists():
        df = pd.read_csv(SITE_REGION_CSV)
        lower_cols = {c.lower(): c for c in df.columns}
        site_col = lower_cols.get("site")
        region_col = lower_cols.get("region")

        if site_col and region_col:
            for _, row in df.iterrows():
                site = str(row[site_col]).strip()
                region = str(row[region_col]).strip()
                if site and region:
                    region_map[normalize_token(site)] = region

            logging.info("Loaded region map from %s with %d sites", SITE_REGION_CSV, len(region_map))
            return region_map

    for attr_name in [
        "SITE_REGION_MAP",
        "REGION_MAP",
        "SITE_TO_REGION",
        "SITE_REGIONS",
        "REGION_TO_SITES",
    ]:
        obj = getattr(config, attr_name, None)

        if isinstance(obj, dict):
            # {"ALEXANDRIA": "Sydney Metro"}
            if all(not isinstance(v, (list, tuple, set)) for v in obj.values()):
                for site, region in obj.items():
                    region_map[normalize_token(site)] = str(region)
                logging.info("Loaded region map from config.%s", attr_name)
                return region_map

            # {"Sydney Metro": ["ALEXANDRIA", "ROZELLE"]}
            for region, sites in obj.items():
                if isinstance(sites, (list, tuple, set)):
                    for site in sites:
                        region_map[normalize_token(site)] = str(region)

            if region_map:
                logging.info("Loaded region map from config.%s", attr_name)
                return region_map

    logging.warning("No region mapping found. Using ALL_SITES_POOLED.")
    return region_map


def get_region_for_site(site_name: str, region_map: Dict[str, str]) -> str:
    if not region_map:
        return "ALL_SITES_POOLED"
    return region_map.get(normalize_token(site_name), "Unknown")


def requested_regions() -> List[str]:
    if TARGET_REGIONS_ENV:
        return [x.strip() for x in TARGET_REGIONS_ENV.split(",") if x.strip()]
    return [x.strip() for x in SELECTED_REGIONS if x.strip()]


def selected_input_columns() -> List[str]:
    """Resolve requested predictor columns for the run."""
    if INPUT_COLUMNS_ENV:
        requested = [x.strip() for x in INPUT_COLUMNS_ENV.split(",") if x.strip()]
    else:
        requested = DEFAULT_INPUT_COLUMNS

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(requested))


# =============================================================================
# DATA PREPARATION
# =============================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["DateTime"], errors="coerce")

    out["Hour"] = dt.dt.hour
    out["Day"] = dt.dt.day
    out["Month"] = dt.dt.month
    out["DayOfWeek"] = dt.dt.dayofweek
    out["DayOfYear"] = dt.dt.dayofyear
    out["WeekOfYear"] = dt.dt.isocalendar().week.astype(float)

    out["Hour_sin"] = np.sin(2 * np.pi * out["Hour"] / 24.0)
    out["Hour_cos"] = np.cos(2 * np.pi * out["Hour"] / 24.0)
    out["Month_sin"] = np.sin(2 * np.pi * out["Month"] / 12.0)
    out["Month_cos"] = np.cos(2 * np.pi * out["Month"] / 12.0)

    return out


def temporal_columns() -> List[str]:
    return [
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
    ]


def discover_site_files(target_column: str, region_map: Dict[str, str]) -> List[Path]:
    csvs = sorted(ACTIVE_SITE_INPUT_DIR.glob("*.csv"))
    requested_region_names = requested_regions()
    requested_region_tokens = {normalize_token(x) for x in requested_region_names}

    if TARGET_SITES_ENV:
        requested = {
            normalize_token(x.strip())
            for x in TARGET_SITES_ENV.split(",")
            if x.strip()
        }
    else:
        requested = {
            normalize_token(x)
            for x in getattr(config, "TARGET_SITES", [])
        }

    selected = []
    for csv_path in csvs:
        site_name = infer_site_name(csv_path)
        site_region = get_region_for_site(site_name, region_map)

        if requested and normalize_token(site_name) not in requested:
            continue

        if requested_region_tokens and normalize_token(site_region) not in requested_region_tokens:
            continue

        if csv_has_target_column(csv_path, target_column):
            selected.append(csv_path)

    return selected


def prepare_site_dataframe(
    csv_path: Path,
    target_column: str,
    input_columns: List[str],
    region_map: Dict[str, str],
) -> Optional[pd.DataFrame]:
    site_name = infer_site_name(csv_path)

    try:
        raw = pd.read_csv(csv_path, low_memory=False)
        raw = ensure_datetime_column(raw)
    except Exception as exc:
        logging.warning("Could not read %s: %s", csv_path, exc)
        return None

    resolved_target = resolve_target_column(raw, target_column)
    if resolved_target is None:
        logging.warning("%s: target column %s not found.", site_name, target_column)
        return None

    out = pd.DataFrame()
    out["DateTime"] = raw["DateTime"]
    out["Site"] = site_name
    out["Region"] = get_region_for_site(site_name, region_map)
    out[target_column] = pd.to_numeric(raw[resolved_target], errors="coerce")

    for var in input_columns:
        resolved = resolve_input_column(raw, var, site_name=site_name)

        if resolved is None:
            continue

        if normalize_token(resolved) == normalize_token(resolved_target):
            continue

        out[var] = pd.to_numeric(raw[resolved], errors="coerce")

    out = out.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)

    if ADD_TEMPORAL_FEATURES:
        out = add_temporal_features(out)

    return out


# =============================================================================
# MISSINGNESS
# =============================================================================

def make_reusable_missing_mask(
    df: pd.DataFrame,
    target_column: str,
    regime: str,
    frac: float,
    seed: int,
    group_by_site: bool,
) -> pd.Series:
    """
    Create one simulated missingness mask and reuse it across all variables.

    For region pooled datasets, create the mask within each site separately
    to avoid short/long gaps crossing site boundaries.
    """
    full_mask = pd.Series(False, index=df.index)

    if group_by_site and "Site" in df.columns and df["Site"].nunique() > 1:
        for i, (_, group) in enumerate(df.groupby("Site", sort=False)):
            group_seed = seed + (i * 1009)
            original_missing = group[target_column].isna()

            temp_group = group.copy()
            _, sim_mask = apply_missingness(
                temp_group,
                target_column,
                regime=regime,
                frac=frac,
                seed=group_seed,
            )

            if isinstance(sim_mask, pd.Series) and sim_mask.index.equals(group.index):
                aligned = sim_mask.astype(bool)
            else:
                aligned = pd.Series(np.asarray(sim_mask).astype(bool), index=group.index)

            full_mask.loc[group.index] = aligned & (~original_missing)

    else:
        original_missing = df[target_column].isna()

        temp_df = df.copy()
        _, sim_mask = apply_missingness(
            temp_df,
            target_column,
            regime=regime,
            frac=frac,
            seed=seed,
        )

        if isinstance(sim_mask, pd.Series) and sim_mask.index.equals(df.index):
            aligned = sim_mask.astype(bool)
        else:
            aligned = pd.Series(np.asarray(sim_mask).astype(bool), index=df.index)

        full_mask = aligned & (~original_missing)

    return full_mask.astype(bool)


# =============================================================================
# METRICS
# =============================================================================

def manual_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = np.nan

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = float(1 - np.sum((y_true - y_pred) ** 2) / denominator) if denominator > 0 else np.nan

    bias = float(np.mean(y_pred - y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R": r,
        "NSE": nse,
        "Bias": bias,
        "MAPE": mape,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Return simple stable metric names:
    RMSE, MAE, R, NSE, Bias, MAPE
    """
    if evaluate_metrics is None:
        return manual_metrics(y_true, y_pred)

    try:
        raw = evaluate_metrics(
            y_true,
            y_pred,
            handle_negative=getattr(config, "HANDLE_NEGATIVES", "none"),
        )
    except TypeError:
        try:
            raw = evaluate_metrics(y_true, y_pred)
        except Exception:
            return manual_metrics(y_true, y_pred)
    except Exception:
        return manual_metrics(y_true, y_pred)

    out = manual_metrics(y_true, y_pred)

    # Try to map common project metric names into simple columns.
    mapping = {
        "Root Mean Squared Error (RMSE)": "RMSE",
        "Mean Absolute Error (MAE)": "MAE",
        "Correlation Coefficient (R)": "R",
        "Nash-Sutcliffe Efficiency (NSE)": "NSE",
        "Mean Absolute Percentage Error (MAPE)": "MAPE",
        "Bias": "Bias",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "R": "R",
        "NSE": "NSE",
        "MAPE": "MAPE",
    }

    if isinstance(raw, dict):
        for old, new in mapping.items():
            if old in raw:
                try:
                    out[new] = float(raw[old])
                except Exception:
                    pass

    return out


# =============================================================================
# LIGHTGBM INDIVIDUAL FEATURE EVALUATION
# =============================================================================

def evaluate_feature_set(
    df: pd.DataFrame,
    scope: str,
    region: str,
    site_label: str,
    target_column: str,
    variable: str,
    features: List[str],
    reusable_mask: pd.Series,
    regime: str,
    missingness_level: float,
    seed: int,
) -> Optional[Dict]:
    """Evaluate one feature set using the shared Random Forest model."""
    if not features:
        return None

    features = [f for f in dict.fromkeys(features) if normalize_token(f) != normalize_token(target_column)]
    if not features:
        return None

    base_feature_count = len(features)

    if ADD_TEMPORAL_FEATURES:
        features.extend([c for c in temporal_columns() if c in df.columns])

    needed = ["DateTime", "Site", "Region", target_column] + features
    needed = [c for c in needed if c in df.columns]

    work = df[needed].copy()

    for col in [target_column] + features:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Hide target values according to the same mask for all variables.
    work.loc[reusable_mask, target_column] = np.nan

    train_mask = work[target_column].notna()
    test_mask = reusable_mask.copy()

    if train_mask.sum() < MIN_TRAIN_ROWS:
        logging.warning(
            "%s | %s | %s | %s | not enough training rows: %d",
            scope, region, site_label, variable, train_mask.sum()
        )
        return None

    if test_mask.sum() < MIN_TEST_ROWS:
        logging.warning(
            "%s | %s | %s | %s | not enough test rows: %d",
            scope, region, site_label, variable, test_mask.sum()
        )
        return None

    X_train = work.loc[train_mask, features]
    y_train = work.loc[train_mask, target_column]

    X_test = work.loc[test_mask, features]
    y_true = df.loc[test_mask, target_column].to_numpy(dtype=float)

    try:
        model = make_random_forest_model(seed)
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=float)
    except Exception as exc:
        logging.warning(
            "%s | %s | %s | %s | model failed: %s",
            scope, region, site_label, variable, exc
        )
        return None

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if getattr(config, "HANDLE_NEGATIVES", "none") == "exclude":
        valid = valid & (y_true >= 0) & (y_pred >= 0)

    if valid.sum() < MIN_TEST_ROWS:
        return None

    metrics = compute_metrics(y_true[valid], y_pred[valid])

    try:
        importances = model.feature_importances_
        if variable == "ALL_INPUTS":
            variable_importance = float(np.sum(importances))
        else:
            variable_importance = float(importances[features.index(variable)])
    except Exception:
        variable_importance = np.nan

    feature_set_name = (
        f"RF_IND_{variable}"
        if base_feature_count == 1 and variable != "ALL_INPUTS"
        else "RF_ALL_INPUTS"
    )
    if ADD_TEMPORAL_FEATURES:
        feature_set_name = f"{feature_set_name}_PLUS_TIME"

    return {
        "Scope": scope,
        "Region": region,
        "Site": site_label,
        "Target": target_column,
        "Model": "RandomForest",
        "Variable": variable,
        "Feature_Set": feature_set_name,
        "Feature_List": ",".join(features),
        "Regime": regime,
        "Missingness_Frac": float(missingness_level),
        "Missingness_Pct": float(missingness_level * 100.0),
        "Seed": int(seed),
        "N_Features": int(len(features)),
        "N_Train": int(train_mask.sum()),
        "N_Test_Masked": int(test_mask.sum()),
        "N_Valid": int(valid.sum()),
        "Variable_Importance": variable_importance,
        **metrics,
    }


def run_scope(
    df: pd.DataFrame,
    scope: str,
    region: str,
    site_label: str,
    target_column: str,
    input_columns: List[str],
    group_by_site_for_mask: bool,
) -> pd.DataFrame:
    """Run one-feature models for one site or pooled region."""
    available = [
        v for v in dict.fromkeys(input_columns)
        if v in df.columns and normalize_token(v) != normalize_token(target_column)
    ]

    if not available:
        logging.warning("%s | %s | %s | no available input columns.", scope, region, site_label)
        return pd.DataFrame()

    feature_specs = [{"variable": variable, "features": [variable]} for variable in available]
    if INCLUDE_ALL_INPUTS and len(available) > 1:
        feature_specs.insert(0, {"variable": "ALL_INPUTS", "features": list(available)})

    logging.info(
        "Running %s | region=%s | site=%s | n_rows=%d | variables=%s",
        scope, region, site_label, len(df), available
    )

    rows = []

    for regime in MISSINGNESS_REGIMES:
        for frac in MISSINGNESS_LEVELS:
            for seed in SEEDS:
                mask = make_reusable_missing_mask(
                    df=df,
                    target_column=target_column,
                    regime=regime,
                    frac=frac,
                    seed=seed,
                    group_by_site=group_by_site_for_mask,
                )

                logging.info(
                    "%s | %s | %s | %s | %.1f%% | seed=%d | masked=%d",
                    scope, region, site_label, regime, frac * 100, seed, int(mask.sum())
                )

                if int(mask.sum()) < MIN_TEST_ROWS:
                    logging.warning("Skipping because masked count is too small.")
                    continue

                for spec in feature_specs:
                    result = evaluate_feature_set(
                        df=df,
                        scope=scope,
                        region=region,
                        site_label=site_label,
                        target_column=target_column,
                        variable=spec["variable"],
                        features=list(spec["features"]),
                        reusable_mask=mask,
                        regime=regime,
                        missingness_level=frac,
                        seed=seed,
                    )

                    if result is not None:
                        rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def make_ranking(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds and rank by mean RMSE."""
    if results.empty:
        return pd.DataFrame()

    group_cols = [
        "Scope",
        "Region",
        "Site",
        "Target",
        "Model",
        "Variable",
        "Feature_Set",
        "Feature_List",
        "Regime",
        "Missingness_Pct",
    ]

    agg_cols = {
        "RMSE": ["mean", "std", "min", "max"],
        "MAE": ["mean", "std"],
        "R": ["mean", "std"],
        "NSE": ["mean", "std"],
        "Bias": ["mean"],
        "MAPE": ["mean"],
        "N_Train": ["mean"],
        "N_Test_Masked": ["mean"],
        "N_Valid": ["mean"],
        "Variable_Importance": ["mean"],
        "Seed": ["count"],
    }

    ranking = results.groupby(group_cols, dropna=False).agg(agg_cols)
    ranking.columns = ["_".join([str(x) for x in col if str(x)]) for col in ranking.columns]
    ranking = ranking.reset_index()

    ranking = ranking.rename(columns={
        "RMSE_mean": "Mean_RMSE",
        "RMSE_std": "SD_RMSE",
        "RMSE_min": "Min_RMSE",
        "RMSE_max": "Max_RMSE",
        "MAE_mean": "Mean_MAE",
        "MAE_std": "SD_MAE",
        "R_mean": "Mean_R",
        "R_std": "SD_R",
        "NSE_mean": "Mean_NSE",
        "NSE_std": "SD_NSE",
        "Bias_mean": "Mean_Bias",
        "MAPE_mean": "Mean_MAPE",
        "N_Train_mean": "Mean_N_Train",
        "N_Test_Masked_mean": "Mean_N_Test_Masked",
        "N_Valid_mean": "Mean_N_Valid",
        "Variable_Importance_mean": "Mean_Variable_Importance",
        "Seed_count": "N_Seeds",
    })

    ranking = ranking.sort_values(
        ["Scope", "Region", "Site", "Regime", "Missingness_Pct", "Mean_RMSE"],
        ascending=True,
    )

    ranking["Rank_By_RMSE"] = ranking.groupby(
        ["Scope", "Region", "Site", "Regime", "Missingness_Pct"],
        dropna=False,
    )["Mean_RMSE"].rank(method="dense", ascending=True).astype(int)

    return ranking


def make_answer_tables(ranking: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create final answer CSVs:
    - answer summary
    - best by site
    - best by region
    - overall feature ranking
    """
    if ranking.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    answer = ranking[ranking["Rank_By_RMSE"] == 1].copy()
    answer = answer.sort_values(["Scope", "Region", "Site", "Regime", "Missingness_Pct"])

    best_site = answer[answer["Scope"] == "Site"].copy()
    best_region = answer[answer["Scope"] == "Region"].copy()

    overall = ranking.groupby(
        ["Scope", "Variable", "Feature_Set", "Regime", "Missingness_Pct"],
        dropna=False,
    ).agg(
        Overall_Mean_RMSE=("Mean_RMSE", "mean"),
        Overall_SD_RMSE=("Mean_RMSE", "std"),
        Overall_Mean_MAE=("Mean_MAE", "mean"),
        Overall_Mean_R=("Mean_R", "mean"),
        Overall_Mean_NSE=("Mean_NSE", "mean"),
        Times_Ranked_Best=("Rank_By_RMSE", lambda x: int((x == 1).sum())),
        N_Comparisons=("Rank_By_RMSE", "count"),
    ).reset_index()

    overall["Best_Rate"] = overall["Times_Ranked_Best"] / overall["N_Comparisons"]
    overall = overall.sort_values(["Scope", "Overall_Mean_RMSE"], ascending=True)
    overall["Overall_Rank_By_RMSE"] = overall.groupby(
        ["Scope", "Regime", "Missingness_Pct"],
        dropna=False,
    )["Overall_Mean_RMSE"].rank(method="dense", ascending=True).astype(int)

    return answer, best_site, best_region, overall


# =============================================================================
# PLOTS
# =============================================================================

def save_barh(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    title: str,
    xlabel: str,
    out_path: Path,
    label_col: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> None:
    if df.empty:
        return

    plot_df = df.copy()
    plot_df = plot_df.sort_values(x_col, ascending=True)

    if max_rows is not None:
        plot_df = plot_df.head(max_rows)

    height = max(5, min(0.35 * len(plot_df) + 2, 22))
    fig, ax = plt.subplots(figsize=(11, height))

    y = np.arange(len(plot_df))
    ax.barh(y, plot_df[x_col].values)

    labels = plot_df[y_col].astype(str).tolist()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    if label_col and label_col in plot_df.columns:
        x_vals = plot_df[x_col].values
        pad = (np.nanmax(x_vals) - np.nanmin(x_vals)) * 0.01 if len(x_vals) else 0.01
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                row[x_col] + pad,
                i,
                str(row[label_col]),
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    df: pd.DataFrame,
    index_col: str,
    column_col: str,
    value_col: str,
    title: str,
    out_path: Path,
    max_rows: int = 60,
) -> None:
    if df.empty:
        return

    pivot = df.pivot_table(
        index=index_col,
        columns=column_col,
        values=value_col,
        aggfunc="mean",
    )

    if pivot.empty:
        return

    # Order rows by the best RMSE they achieved.
    pivot = pivot.loc[pivot.min(axis=1).sort_values().index]

    if len(pivot) > max_rows:
        pivot = pivot.iloc[:max_rows, :]

    fig_w = max(9, 0.8 * len(pivot.columns) + 4)
    fig_h = max(6, 0.3 * len(pivot.index) + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel(index_col)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean RMSE")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_region_vs_sites(best_site: pd.DataFrame, best_region: pd.DataFrame) -> None:
    """Plot one pooled region against its corresponding individual sites."""
    if best_site.empty or best_region.empty:
        return

    candidate_regions = [
        region
        for region in best_site["Region"].dropna().astype(str).drop_duplicates().tolist()
        if region in set(best_region["Region"].dropna().astype(str))
    ]
    if not candidate_regions:
        return

    selected_regions = candidate_regions
    if REGION_COMPARE_NAME:
        selected_regions = [r for r in candidate_regions if normalize_token(r) == normalize_token(REGION_COMPARE_NAME)]
        if not selected_regions:
            logging.warning(
                "Requested FS_REGION_COMPARE_NAME=%s was not found. Falling back to default region selection.",
                REGION_COMPARE_NAME,
            )
            selected_regions = candidate_regions

    selected_regions = selected_regions[: max(1, REGION_COMPARE_LIMIT)]

    for region in selected_regions:
        region_rows = best_region[best_region["Region"].astype(str) == str(region)].copy()
        site_rows = best_site[best_site["Region"].astype(str) == str(region)].copy()

        if region_rows.empty or site_rows.empty:
            continue

        region_row = region_rows.nsmallest(1, "Mean_RMSE").copy()
        region_row["Display_Label"] = f"{region} | ALL SITES"
        region_row["Display_Group"] = "Region"

        site_rows = site_rows.sort_values("Mean_RMSE", ascending=True).copy()
        site_rows["Display_Label"] = site_rows["Site"].astype(str)
        site_rows["Display_Group"] = "Site"

        plot_df = pd.concat([region_row, site_rows], ignore_index=True)

        height = max(5, min(0.35 * len(plot_df) + 2, 22))
        fig, ax = plt.subplots(figsize=(12, height))

        y = np.arange(len(plot_df))
        colors = ["#c44e52" if g == "Region" else "#4c72b0" for g in plot_df["Display_Group"]]
        ax.barh(y, plot_df["Mean_RMSE"].values, color=colors)

        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["Display_Label"].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Mean RMSE of selected best feature")
        ax.set_title(
            f"Best Random Forest feature: pooled region vs individual sites ({region})",
            fontsize=14,
            weight="bold",
        )
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        x_vals = plot_df["Mean_RMSE"].values
        x_span = np.nanmax(x_vals) - np.nanmin(x_vals) if len(x_vals) else 0.0
        pad = x_span * 0.01 if x_span > 0 else 0.01
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                row["Mean_RMSE"] + pad,
                i,
                str(row["Variable"]),
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        out_path = PLOTS_DIR / f"plot_region_vs_sites_{clean_filename(region)}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def create_plots(
    ranking: pd.DataFrame,
    answer: pd.DataFrame,
    best_site: pd.DataFrame,
    best_region: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    """Create all plot PNGs."""
    if ranking.empty:
        return

    # 1. Overall feature ranking, site scope first.
    for scope in ["Site", "Region"]:
        ov = overall[overall["Scope"] == scope].copy()
        if not ov.empty:
            ov = ov.sort_values("Overall_Mean_RMSE", ascending=True)
            out = PLOTS_DIR / f"plot_overall_feature_ranking_{scope.lower()}.png"
            save_barh(
                df=ov,
                y_col="Variable",
                x_col="Overall_Mean_RMSE",
                title=f"Overall Random Forest individual feature ranking ({scope})",
                xlabel="Mean RMSE across comparisons",
                out_path=out,
                label_col="Times_Ranked_Best",
                max_rows=25,
            )

    # Backward compatible single plot name: use Site if available, otherwise Region.
    if not overall.empty:
        single_scope = "Site" if (overall["Scope"] == "Site").any() else overall["Scope"].iloc[0]
        ov = overall[overall["Scope"] == single_scope].sort_values("Overall_Mean_RMSE")
        save_barh(
            df=ov,
            y_col="Variable",
            x_col="Overall_Mean_RMSE",
            title=f"Overall Random Forest individual feature ranking ({single_scope})",
            xlabel="Mean RMSE across comparisons",
            out_path=PLOTS_DIR / "plot_overall_feature_ranking.png",
            label_col="Times_Ranked_Best",
            max_rows=25,
        )

    # 2. Best feature by site.
    if not best_site.empty:
        site_plot = best_site.copy()
        site_plot["Site_Label"] = site_plot["Region"].astype(str) + " | " + site_plot["Site"].astype(str)
        save_barh(
            df=site_plot,
            y_col="Site_Label",
            x_col="Mean_RMSE",
            title="Best Random Forest individual feature by site",
            xlabel="Mean RMSE of selected best feature",
            out_path=PLOTS_DIR / "plot_best_feature_by_site.png",
            label_col="Variable",
            max_rows=60,
        )

    # 3. Best feature by region.
    if not best_region.empty:
        save_barh(
            df=best_region,
            y_col="Region",
            x_col="Mean_RMSE",
            title="Best Random Forest individual feature by region",
            xlabel="Mean RMSE of selected best feature",
            out_path=PLOTS_DIR / "plot_best_feature_by_region.png",
            label_col="Variable",
            max_rows=60,
        )

    # 4. Heatmap: site x variable.
    site_rank = ranking[ranking["Scope"] == "Site"].copy()
    if not site_rank.empty:
        site_rank["Site_Label"] = site_rank["Region"].astype(str) + " | " + site_rank["Site"].astype(str)
        plot_heatmap(
            df=site_rank,
            index_col="Site_Label",
            column_col="Variable",
            value_col="Mean_RMSE",
            title="Site-level RMSE heatmap for individual Random Forest features",
            out_path=PLOTS_DIR / "plot_site_feature_rmse_heatmap.png",
            max_rows=60,
        )

    # 5. Heatmap: region x variable.
    region_rank = ranking[ranking["Scope"] == "Region"].copy()
    if not region_rank.empty:
        plot_heatmap(
            df=region_rank,
            index_col="Region",
            column_col="Variable",
            value_col="Mean_RMSE",
            title="Region-level RMSE heatmap for individual Random Forest features",
            out_path=PLOTS_DIR / "plot_region_feature_rmse_heatmap.png",
            max_rows=60,
        )

    # 6. Pooled region vs member sites.
    plot_region_vs_sites(best_site=best_site, best_region=best_region)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    global ACTIVE_SITE_INPUT_DIR, REGION_MAP_OVERRIDE
    logging.info("=" * 100)
    logging.info("RANDOM FOREST MULTI-TARGET FEATURE EVALUATION")
    logging.info("RUN TYPE: %s", RUN_TYPE_LABEL)
    print("=== RUN TYPE: %s ===" % RUN_TYPE_LABEL, flush=True)
    logging.info("=" * 100)
    logging.info("Input directory: %s", INPUT_DIR)
    logging.info("Output root: %s", OUTPUT_ROOT)
    logging.info("Targets: %s", TARGET_COLUMNS)
    logging.info("Missingness levels: %s", MISSINGNESS_LEVELS)
    logging.info("Missingness regimes: %s", MISSINGNESS_REGIMES)
    logging.info("Seeds: %s", SEEDS)
    logging.info("Add temporal features: %s", ADD_TEMPORAL_FEATURES)

    if is_wide_region_input_dir(INPUT_DIR):
        ACTIVE_SITE_INPUT_DIR = PREPARED_SITE_INPUT_DIR
        REGION_MAP_OVERRIDE = prepare_site_inputs_from_wide(INPUT_DIR)
    else:
        ACTIVE_SITE_INPUT_DIR = INPUT_DIR
        REGION_MAP_OVERRIDE = {}

    logging.info("Active site input directory: %s", ACTIVE_SITE_INPUT_DIR)

    input_columns = selected_input_columns()
    if not input_columns:
        raise ValueError(
            "No input columns were resolved. "
            "Set FS_INPUT_COLUMNS or define config.INPUT_COLUMNS."
        )

    region_map = load_region_map()
    if requested_regions() and not region_map:
        raise ValueError(
            "FS_TARGET_REGIONS was provided, but no region map could be loaded. "
            "Provide SITE_REGION_CSV or ensure config_spatial exposes SITE_TO_REGION / REGION_TO_SITES."
        )

    all_results = []

    for target_column in TARGET_COLUMNS:
        logging.info("-" * 100)
        logging.info("Processing target: %s", target_column)

        site_files = discover_site_files(target_column, region_map)
        if not site_files:
            logging.warning("No CSV files found for target %s in %s", target_column, ACTIVE_SITE_INPUT_DIR)
            continue

        logging.info("Found %d site files for target %s.", len(site_files), target_column)

        site_dfs: List[pd.DataFrame] = []
        for csv_path in site_files:
            df = prepare_site_dataframe(
                csv_path=csv_path,
                target_column=target_column,
                input_columns=input_columns,
                region_map=region_map,
            )
            if df is not None and not df.empty:
                site_dfs.append(df)

        if not site_dfs:
            logging.warning("No valid site dataframes prepared for target %s.", target_column)
            continue

        for site_df in site_dfs:
            site = str(site_df["Site"].iloc[0])
            region = str(site_df["Region"].iloc[0])

            site_results = run_scope(
                df=site_df,
                scope="Site",
                region=region,
                site_label=site,
                target_column=target_column,
                input_columns=input_columns,
                group_by_site_for_mask=False,
            )
            if not site_results.empty:
                all_results.append(site_results)

        combined = pd.concat(site_dfs, ignore_index=True)
        for region, region_df in combined.groupby("Region", sort=True):
            region_df = region_df.copy()
            n_sites = region_df["Site"].nunique()
            if n_sites < 2:
                logging.info("Skipping region %s for target %s because it has only %d site.", region, target_column, n_sites)
                continue

            region_results = run_scope(
                df=region_df,
                scope="Region",
                region=str(region),
                site_label="ALL_SITES_IN_REGION",
                target_column=target_column,
                input_columns=input_columns,
                group_by_site_for_mask=True,
            )
            if not region_results.empty:
                all_results.append(region_results)

    if not all_results:
        raise RuntimeError("No results produced for any requested target.")

    seed_results = pd.concat(all_results, ignore_index=True)
    seed_results["_scope_order"] = seed_results["Scope"].map({"Region": 0, "Site": 1}).fillna(9).astype(int)
    seed_results["_site_order"] = np.where(
        seed_results["Scope"] == "Region",
        "",
        seed_results["Site"].astype(str),
    )
    seed_results = seed_results.sort_values(
        [
            "Target",
            "Region",
            "Regime",
            "Missingness_Pct",
            "Variable",
            "_scope_order",
            "_site_order",
            "Seed",
        ],
        ascending=True,
    ).drop(columns=["_scope_order", "_site_order"])

    ranking = make_ranking(seed_results)
    ranking["_scope_order"] = ranking["Scope"].map({"Region": 0, "Site": 1}).fillna(9).astype(int)
    ranking["_site_order"] = np.where(
        ranking["Scope"] == "Region",
        "",
        ranking["Site"].astype(str),
    )
    ranking = ranking.sort_values(
        [
            "Target",
            "Region",
            "Regime",
            "Missingness_Pct",
            "Variable",
            "_scope_order",
            "_site_order",
        ],
        ascending=True,
    ).drop(columns=["_scope_order", "_site_order"])

    answer, best_site, best_region, overall = make_answer_tables(ranking)

    answer_path = RESULTS_DIR / "ANSWER_best_feature_summary.csv"
    best_site_path = RESULTS_DIR / "ANSWER_best_feature_by_site.csv"
    best_region_path = RESULTS_DIR / "ANSWER_best_feature_by_region.csv"
    overall_path = RESULTS_DIR / "ANSWER_overall_feature_ranking.csv"
    all_results_path = RESULTS_DIR / "all_random_forest_individual_feature_results.csv"
    seed_results_path = RESULTS_DIR / "all_random_forest_individual_feature_seed_results.csv"

    answer.to_csv(answer_path, index=False)
    best_site.to_csv(best_site_path, index=False)
    best_region.to_csv(best_region_path, index=False)
    overall.to_csv(overall_path, index=False)
    ranking.to_csv(all_results_path, index=False)
    seed_results.to_csv(seed_results_path, index=False)
    create_plots(
        ranking=ranking,
        answer=answer,
        best_site=best_site,
        best_region=best_region,
        overall=overall,
    )

    logging.info("=" * 100)
    logging.info("DONE")
    logging.info("Saved answer summary CSV: %s", answer_path)
    logging.info("Saved best-by-site CSV: %s", best_site_path)
    logging.info("Saved best-by-region CSV: %s", best_region_path)
    logging.info("Saved overall ranking CSV: %s", overall_path)
    logging.info("Saved aggregated per-feature CSV: %s", all_results_path)
    logging.info("Saved seed-level CSV: %s", seed_results_path)
    logging.info("=" * 100)
    print(f"DONE: saved Random Forest feature-selection results in {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
    print("=== JOB FINISHED: Stage 0 individual-feature Random Forest ===", flush=True)
