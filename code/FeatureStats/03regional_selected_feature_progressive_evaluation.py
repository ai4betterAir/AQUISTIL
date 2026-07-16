"""
regional_selected_feature_progressive_evaluation.py

Standalone regional evaluation script.

Goal:
1. Read region-level selected local features from
   FINAL_selected_feature_combination_by_region.csv
2. For each region, evaluate pooled-site PM2.5 imputation performance with:
   - selected local features only
   - + temporal features
   - + spatial features
   - + IDW feature
   - cumulative combinations added one block at a time
3. Save raw results, summaries, feature inventories, and comparison plots.

Default output:
/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection/03Regional_Selected_Feature_Progressive_Evaluation
"""

import json
import logging
import os
import re
import sys
import argparse
import warnings
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_DIR = _SCRIPT_PATH.parents[3]
APP_ROOT_DIR = _SCRIPT_PATH.parents[2]
for _candidate in (_SCRIPT_PATH.parents[1], _SCRIPT_PATH.parents[2]):
    if (_candidate / "config_spatial.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import config_spatial as config
from spatial import add_temporal_features, load_spatial_features

try:
    from missingness_regimes import apply_missingness as project_apply_missingness
except Exception:
    project_apply_missingness = None


def env_list(name, default):
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def env_float_list(name, default):
    return [float(x) for x in env_list(name, ",".join(str(v) for v in default))]


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

SELECTED_TARGETS = AVAILABLE_TARGETS[:]
SELECTED_REGIONS = AVAILABLE_REGIONS[:]
SELECTED_TARGET = "PM2.5"


def _apply_cli_env_overrides():
    """Allow standalone runs with explicit CLI flags instead of only env vars."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input-dir")
    parser.add_argument("--selected-features-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--target")
    parser.add_argument("--regions")
    parser.add_argument("--input-features")
    parser.add_argument("--site-region-csv")
    parser.add_argument("--min-sites", type=int)
    args, _ = parser.parse_known_args()

    if args.input_dir:
        os.environ["INPUT_DIR"] = args.input_dir
    if args.selected_features_csv:
        os.environ["SELECTED_FEATURES_CSV"] = args.selected_features_csv
    if args.output_dir:
        os.environ["OUTPUT_DIR"] = args.output_dir
    if args.target:
        os.environ["TARGET_COLUMN"] = args.target
    if args.regions is not None:
        os.environ["TARGET_REGIONS"] = args.regions
    if args.input_features:
        os.environ["INPUT_FEATURES"] = args.input_features
    if args.site_region_csv:
        os.environ["SITE_REGION_CSV"] = args.site_region_csv
    if args.min_sites is not None:
        os.environ["MIN_SITES"] = str(args.min_sites)


_apply_cli_env_overrides()

FEATURE_SELECTION_RUN_MODE = os.getenv("FEATURE_SELECTION_RUN_MODE", "").strip().lower()
_requested_regimes = env_list(
    "MISSINGNESS_REGIMES",
    "event" if FEATURE_SELECTION_RUN_MODE == "event" else "random",
)
EVENT_RUN = FEATURE_SELECTION_RUN_MODE == "event" or (
    len(_requested_regimes) == 1 and _requested_regimes[0].lower() == "event"
)
RUN_TYPE_LABEL = "EVENT FEATURE SELECTION RUN" if EVENT_RUN else "DEFAULT FEATURE SELECTION RUN"


DEFAULT_INPUT_DIR = Path(
    os.getenv(
        "INPUT_DIR",
        getattr(
            config,
            "WIDE_API_INPUT_DIRECTORY",
            getattr(config, "INPUT_DIRECTORY", "."),
        ) if getattr(config, "USE_WIDE_NOWCASTING_INPUTS", False) else getattr(config, "INPUT_DIRECTORY", "."),
    )
)
INPUT_DIR = DEFAULT_INPUT_DIR
FEATURE_SELECTION_OUTPUT_ROOT = Path(
    os.getenv(
        "FEATURE_SELECTION_OUTPUT_ROOT",
        str(APP_ROOT_DIR / "Outputs" / "Feature_Selection"),
    )
)
if EVENT_RUN:
    FEATURE_SELECTION_OUTPUT_ROOT = FEATURE_SELECTION_OUTPUT_ROOT / "feature_selection_event"
DEFAULT_OUTPUT_DIR = FEATURE_SELECTION_OUTPUT_ROOT / "03Regional_Selected_Feature_Progressive_Evaluation"
OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(DEFAULT_OUTPUT_DIR),
    )
)
DEFAULT_SELECTED_FEATURES_CSV = (
    FEATURE_SELECTION_OUTPUT_ROOT
    / "02Regional_RF_SHAP_Selection"
    / "summary_outputs"
    / "FINAL_selected_feature_combination_by_region.csv"
)
SELECTED_FEATURES_CSV = Path(
    os.getenv(
        "SELECTED_FEATURES_CSV",
        str(DEFAULT_SELECTED_FEATURES_CSV),
    )
)
SITE_REGION_CSV = os.getenv("SITE_REGION_CSV", "").strip()
TARGET_COLUMN = os.getenv("TARGET_COLUMN", SELECTED_TARGET)
INPUT_FEATURES = env_list(
    "INPUT_FEATURES",
    ",".join(getattr(config, "LOCAL_ANALYSIS_INPUTS", [
        "CO", "HUMID", "NEPH", "NO", "NO2", "NOX",
        "OZONE", "PM10", "RAIN", "TEMP", "WDR", "WSP",
    ])),
)
TARGET_REGIONS = env_list("TARGET_REGIONS", ",".join(SELECTED_REGIONS))
MISSINGNESS_REGIMES = _requested_regimes
MISSINGNESS_LEVELS = env_float_list(
    "MISSINGNESS_LEVELS",
    [0.10],
)
SEEDS = [int(x) for x in env_list("SEEDS", "42,101,202")]
MIN_SITES = int(os.getenv("MIN_SITES", "2"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "100"))
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "30"))
N_JOBS = int(os.getenv("N_JOBS", "-1"))
MAX_SPATIAL_DISTANCE_KM = float(os.getenv("MAX_SPATIAL_DISTANCE_KM", "100"))

RAW_DIR = OUTPUT_DIR / "raw_results"
SUMMARY_DIR = OUTPUT_DIR / "summary_outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
for path in [OUTPUT_DIR, RAW_DIR, SUMMARY_DIR, PLOTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

PREPARED_SITE_INPUT_DIR = OUTPUT_DIR / "_prepared_site_inputs"
PREPARED_SITE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_SITE_INPUT_DIR = INPUT_DIR
REGION_MAP_OVERRIDE = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "regional_selected_feature_progressive_evaluation.log"),
        logging.StreamHandler(),
    ],
    force=True,
)


CONFIG_SPECS = [
    {"name": "Selected_Local (SL)", "temporal": False, "spatial_mode": None, "use_idw": False},
    {"name": "Selected_Local_Temporal (SLT)", "temporal": True, "spatial_mode": None, "use_idw": False},
    {"name": "Selected_Local_Spatial (SLS)", "temporal": False, "spatial_mode": "pollutants_except_target", "use_idw": False},
    {"name": "Selected_Local_Temporal_Spatial (SLTS)", "temporal": True, "spatial_mode": "pollutants_except_target", "use_idw": False},
    {"name": "Selected_Local_Temporal_IDW (SLTI)", "temporal": True, "spatial_mode": None, "use_idw": True},
    {"name": "Selected_Local_Temporal_Spatial_IDW (SLTSI)", "temporal": True, "spatial_mode": "pollutants_except_target", "use_idw": True},
    {"name": "Selected_Local_Temporal_SpatialTarget (SLTST)", "temporal": True, "spatial_mode": "all_pollutants", "use_idw": False},
    {"name": "Selected_Local_Temporal_SpatialTarget_IDW (SLTSTI)", "temporal": True, "spatial_mode": "all_pollutants", "use_idw": True},
]


TEMPORAL_COLUMNS = [
    "Hour",
    "Day",
    "DayOfWeek",
    "DayOfYear",
    "WeekOfYear",
]


def norm(value):
    return re.sub(r"[^A-Za-z0-9]+", "", str(value)).lower()


def safe(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "NA"


def normalize_site_token(value):
    return re.sub(r"[^A-Za-z0-9]+", "", str(value)).lower()


def infer_site_name(path_obj):
    stem = Path(path_obj).stem
    for suffix in ["_processed", "_aqms", "_station", "_site", "_data", "_clean", "_hourly", "_merged", "_final"]:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    if "_" in stem:
        stem = stem.split("_")[0]
    return stem.strip().upper()


def infer_region_name(path_obj):
    stem = Path(path_obj).stem
    prefix = "Allobs_processed_DPE_station_api_"
    suffix = "_ALL"
    if stem.startswith(prefix) and stem.endswith(suffix):
        token = stem[len(prefix) : -len(suffix)]
        return str(token).replace("_", " ").strip()
    return stem.replace("_", " ").strip()


def is_wide_region_input_dir(input_dir):
    return any(input_dir.glob("Allobs_processed_DPE_station_api_*_ALL.csv"))


def materialize_site_inputs_from_wide(input_dir):
    region_map = {}
    for old_csv in PREPARED_SITE_INPUT_DIR.glob("*.csv"):
        try:
            old_csv.unlink()
        except Exception:
            pass

    files = sorted(input_dir.glob("Allobs_processed_DPE_station_api_*_ALL.csv"))
    if not files:
        raise FileNotFoundError(f"No wide regional CSV files found in {input_dir}")

    site_count = 0
    for path_obj in files:
        raw = pd.read_csv(path_obj, low_memory=False)
        raw = ensure_datetime(raw)
        region_name = infer_region_name(path_obj)
        site_columns = {}
        for col in raw.columns:
            if str(col) in {"DateTime", "datetime"}:
                continue
            if "_" not in str(col):
                continue
            feature, site_name = str(col).split("_", 1)
            site_name = site_name.strip()
            feature = feature.strip()
            if not site_name or not feature:
                continue
            site_columns.setdefault(site_name, []).append((feature, col))

        for site_name, feature_pairs in site_columns.items():
            sub = pd.DataFrame({"DateTime": raw["DateTime"]})
            for feature_name, source_col in feature_pairs:
                sub[feature_name] = pd.to_numeric(raw[source_col], errors="coerce")
            sub = sub.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
            if sub.shape[1] <= 1:
                continue
            out_path = PREPARED_SITE_INPUT_DIR / f"{site_name}.csv"
            sub.to_csv(out_path, index=False)
            region_map[norm(site_name)] = region_name
            site_count += 1

    if not region_map:
        raise RuntimeError(f"No per-site files could be materialized from wide regional inputs in {input_dir}")

    logging.info(
        "Prepared %d per-site CSVs from %d wide regional input file(s) into %s",
        site_count,
        len(files),
        PREPARED_SITE_INPUT_DIR,
    )
    return PREPARED_SITE_INPUT_DIR, region_map


def ensure_datetime(df):
    out = df.copy()
    if "DateTime" in out.columns:
        out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
        return out
    lower = {str(c).strip().lower(): c for c in out.columns}
    if "datetime" in lower:
        out["DateTime"] = pd.to_datetime(out[lower["datetime"]], errors="coerce")
        return out
    if "timestamp" in lower:
        out["DateTime"] = pd.to_datetime(out[lower["timestamp"]], errors="coerce")
        return out
    if "date" in lower and "time" in lower:
        out["DateTime"] = pd.to_datetime(
            out[lower["date"]].astype(str) + " " + out[lower["time"]].astype(str),
            errors="coerce",
            dayfirst=True,
        )
        return out
    raise KeyError("No DateTime/date-time column found")


def resolve_target(df, target):
    if target in df.columns:
        return target
    target_norm = norm(target)
    aliases = {
        "pm25": {"pm25", "pm2p5", "pm2_5", "pm2.5", "pm_25", "pm_2_5"},
        "pm10": {"pm10", "pm_10"},
    }
    for c in df.columns:
        if norm(c) == target_norm:
            return c
    for group in aliases.values():
        if target_norm in {norm(x) for x in group}:
            for c in df.columns:
                if norm(c) in {norm(x) for x in group}:
                    return c
    for c in df.columns:
        if norm(c).startswith(target_norm):
            return c
    return None


def resolve_input(df, feature, site):
    if feature in df.columns:
        return feature
    f_norm = norm(feature)
    s_norm = norm(site)
    exact = [c for c in df.columns if norm(c) == f_norm]
    if exact:
        return exact[0]
    prefix = [c for c in df.columns if norm(c).startswith(f_norm)]
    site_match = [c for c in prefix if s_norm in norm(c)]
    if site_match:
        return site_match[0]
    if prefix:
        return prefix[0]
    return None


def load_region_map():
    global REGION_MAP_OVERRIDE

    if REGION_MAP_OVERRIDE:
        return dict(REGION_MAP_OVERRIDE)

    def read_region_csv(csv_path):
        frame = pd.read_csv(csv_path)
        lower = {c.lower(): c for c in frame.columns}
        site_col = None
        region_col = None
        for candidate in ["site", "sitename", "station", "stationname"]:
            if candidate in lower:
                site_col = lower[candidate]
                break
        if "region" in lower:
            region_col = lower["region"]
        if site_col is None or region_col is None:
            raise ValueError("Region CSV must contain site and region columns")
        return {
            norm(row[site_col]): str(row[region_col]).strip()
            for _, row in frame.iterrows()
            if str(row[site_col]).strip() and str(row[region_col]).strip()
        }

    if SITE_REGION_CSV and Path(SITE_REGION_CSV).exists():
        return read_region_csv(Path(SITE_REGION_CSV))

    if getattr(config, "REGION_SITE_MAPPING", None):
        mapping = getattr(config, "REGION_SITE_MAPPING")
        if isinstance(mapping, dict) and "site_to_region" in mapping:
            return {norm(k): str(v) for k, v in mapping["site_to_region"].items()}

    fallback_csv = PROJECT_ROOT_DIR / "Station_info.csv"
    if fallback_csv.exists():
        try:
            return read_region_csv(fallback_csv)
        except Exception:
            pass

    raise FileNotFoundError("No region map found. Set SITE_REGION_CSV to a valid CSV.")


def prepare_site_file(csv_path, region_map):
    site = infer_site_name(csv_path)
    region = region_map.get(norm(site))
    if region is None:
        return None
    try:
        raw = pd.read_csv(csv_path, low_memory=False)
        raw = ensure_datetime(raw)
    except Exception as exc:
        logging.warning("Could not read %s: %s", csv_path, exc)
        return None
    target_col = resolve_target(raw, TARGET_COLUMN)
    if target_col is None:
        return None

    out = pd.DataFrame(
        {
            "DateTime": raw["DateTime"],
            "Site": site,
            "Region": region,
            TARGET_COLUMN: pd.to_numeric(raw[target_col], errors="coerce"),
        }
    )
    for feature in INPUT_FEATURES:
        if norm(feature) == norm(TARGET_COLUMN):
            continue
        col = resolve_input(raw, feature, site)
        if col is None or norm(col) == norm(target_col):
            continue
        out[feature] = pd.to_numeric(raw[col], errors="coerce")
    out = out.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    return out


def load_all_data():
    global ACTIVE_SITE_INPUT_DIR, REGION_MAP_OVERRIDE

    if is_wide_region_input_dir(INPUT_DIR):
        ACTIVE_SITE_INPUT_DIR, REGION_MAP_OVERRIDE = materialize_site_inputs_from_wide(INPUT_DIR)
        config.INPUT_DIRECTORY = str(ACTIVE_SITE_INPUT_DIR)
    else:
        ACTIVE_SITE_INPUT_DIR = INPUT_DIR
        REGION_MAP_OVERRIDE = None

    region_map = load_region_map()
    files = sorted(ACTIVE_SITE_INPUT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {ACTIVE_SITE_INPUT_DIR}")
    parts = []
    for path_obj in files:
        frame = prepare_site_file(path_obj, region_map)
        if frame is not None and not frame.empty:
            parts.append(frame)
    if not parts:
        raise RuntimeError("No valid site data loaded")
    data = pd.concat(parts, ignore_index=True)
    for feature in INPUT_FEATURES:
        if feature not in data.columns and norm(feature) != norm(TARGET_COLUMN):
            data[feature] = np.nan
    data = data.sort_values(["Region", "Site", "DateTime"]).reset_index(drop=True)
    logging.info(
        "Loaded pooled regional data: %d rows, %d sites, %d regions",
        len(data),
        data["Site"].nunique(),
        data["Region"].nunique(),
    )
    return data


def fallback_missingness(df, target, regime, frac, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    obs = out.index[out[target].notna()].to_numpy()
    n = max(1, int(len(obs) * frac))
    if len(obs) == 0:
        return out, pd.Series(False, index=out.index)
    if regime == "random":
        sel = rng.choice(obs, size=min(n, len(obs)), replace=False)
    else:
        gap_len = 6 if regime == "short_gap" else 48
        picked = []
        while len(picked) < n:
            start = int(rng.choice(obs))
            picked += [x for x in range(start, start + gap_len) if x in obs]
            picked = list(dict.fromkeys(picked))
            if len(picked) >= len(obs):
                break
        sel = np.array(picked[:n])
    mask = pd.Series(False, index=out.index)
    mask.loc[sel] = True
    out.loc[mask, target] = np.nan
    return out, mask


def apply_missingness(df, target, regime, frac, seed):
    if project_apply_missingness is not None:
        try:
            return project_apply_missingness(df.copy(), target, regime=regime, frac=frac, seed=seed)
        except Exception:
            pass
    return fallback_missingness(df, target, regime, frac, seed)


def make_region_mask(region_df, regime, frac, seed):
    mask = pd.Series(False, index=region_df.index)
    for i, (_, group) in enumerate(region_df.groupby("Site", sort=False)):
        _, site_mask = apply_missingness(group.copy(), TARGET_COLUMN, regime, frac, seed + i * 1009)
        if not (isinstance(site_mask, pd.Series) and site_mask.index.equals(group.index)):
            site_mask = pd.Series(np.asarray(site_mask).astype(bool), index=group.index)
        mask.loc[group.index] = site_mask.astype(bool) & group[TARGET_COLUMN].notna()
    return mask


def metric_summary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R": np.nan, "NSE": np.nan, "N_Valid": 0}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = float(1 - np.sum((y_true - y_pred) ** 2) / den) if den > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "R": r, "NSE": nse, "N_Valid": int(len(y_true))}


def random_forest_model(seed):
    """Stage 3 progressive scoring uses the shared Random Forest."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=int(os.getenv("RF_N_ESTIMATORS", "120")),
            max_depth=int(os.getenv("RF_MAX_DEPTH", "16")),
            min_samples_leaf=int(os.getenv("RF_MIN_SAMPLES_LEAF", "5")),
            max_features=os.getenv("RF_MAX_FEATURES", "sqrt"),
            random_state=seed,
            n_jobs=N_JOBS,
        )),
    ])


def scalar_observation(value):
    if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
        numeric = pd.to_numeric(pd.Series(np.asarray(value).ravel()), errors="coerce").dropna()
        return float(numeric.mean()) if not numeric.empty else np.nan
    try:
        return float(value) if pd.notna(value) else np.nan
    except Exception:
        return np.nan


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius * c


def resolve_coordinate_key(site_name, coordinates):
    if not site_name or not coordinates:
        return None
    normalized_lookup = {normalize_site_token(coord_site): coord_site for coord_site in coordinates}
    raw = str(site_name).strip()
    candidates = [
        raw,
        raw.upper(),
        raw.split("_")[0],
        raw.upper().removesuffix("_AQMS"),
        raw.upper().removesuffix("_PROCESSED"),
        raw.upper().removesuffix("_STATION"),
        raw.upper().removesuffix("_SITE"),
    ]
    for candidate in candidates:
        if candidate in coordinates:
            return candidate
        token = normalize_site_token(candidate)
        if token in normalized_lookup:
            return normalized_lookup[token]
    token = normalize_site_token(raw)
    matches = [coord_site for coord_token, coord_site in normalized_lookup.items() if token and (token in coord_token or coord_token in token)]
    if len(matches) == 1:
        return matches[0]
    return None


def load_neighbor_data(target_site, target_variable="PM2.5"):
    neighbor_data = {}
    for filename in os.listdir(str(ACTIVE_SITE_INPUT_DIR)):
        if not filename.endswith(".csv"):
            continue
        site_name = Path(filename).stem
        if normalize_site_token(site_name) == normalize_site_token(target_site):
            continue
        try:
            filepath = os.path.join(str(ACTIVE_SITE_INPUT_DIR), filename)
            df = pd.read_csv(filepath, low_memory=False)
            df = ensure_datetime(df)
            df = df.set_index("DateTime")
            resolved_target = resolve_target(df, target_variable)
            if resolved_target is not None:
                series = pd.to_numeric(df[resolved_target], errors="coerce")
                if series.index.duplicated().any():
                    series = series.groupby(series.index).mean()
                neighbor_data[site_name] = series.sort_index()
        except Exception as exc:
            logging.debug("Could not load neighbor %s: %s", filename, exc)
    return neighbor_data


def compute_idw_feature(target_site, target_datetime_index, neighbor_data, coordinates, power=2, max_distance=100):
    target_key = resolve_coordinate_key(target_site, coordinates)
    if target_key is None:
        return pd.Series(np.nan, index=target_datetime_index)

    target_lat = coordinates[target_key]["lat"]
    target_lon = coordinates[target_key]["lon"]
    neighbor_weights = {}
    for site, data in neighbor_data.items():
        site_key = resolve_coordinate_key(site, coordinates)
        if site_key is not None and site_key != target_key:
            distance = haversine_km(
                target_lat,
                target_lon,
                coordinates[site_key]["lat"],
                coordinates[site_key]["lon"],
            )
            if 0 < distance <= max_distance:
                neighbor_weights[site] = 1.0 / (distance ** power)

    if not neighbor_weights:
        return pd.Series(np.nan, index=target_datetime_index)

    idw_values = []
    for timestamp in target_datetime_index:
        weighted_sum = 0.0
        weight_sum = 0.0
        for site, weight in neighbor_weights.items():
            if site in neighbor_data and timestamp in neighbor_data[site].index:
                value = scalar_observation(neighbor_data[site].loc[timestamp])
                if np.isfinite(value):
                    weighted_sum += value * weight
                    weight_sum += weight
        idw_values.append(weighted_sum / weight_sum if weight_sum > 0 else np.nan)
    return pd.Series(idw_values, index=target_datetime_index)


def load_selected_feature_map():
    if not SELECTED_FEATURES_CSV.exists():
        raise FileNotFoundError(f"Selected-features CSV not found: {SELECTED_FEATURES_CSV}")
    frame = pd.read_csv(SELECTED_FEATURES_CSV)
    if "Target" in frame.columns:
        frame = frame[
            frame["Target"].astype(str).map(norm) == norm(TARGET_COLUMN)
        ].copy()
    out = {}
    for _, row in frame.iterrows():
        region = str(row.get("Region", "")).strip()
        if not region:
            continue
        feature_string = str(row.get("Final_Features", "")).strip()
        features = [item.strip() for item in feature_string.split(",") if item.strip()]
        out[region] = features
    return out


def add_selected_features_to_input_columns(selected_feature_map):
    """Make the Stage 2 final feature table authoritative for local inputs."""
    added = []
    existing = {norm(feature) for feature in INPUT_FEATURES}
    for features in selected_feature_map.values():
        for feature in features:
            if norm(feature) == norm(TARGET_COLUMN) or norm(feature) in existing:
                continue
            INPUT_FEATURES.append(feature)
            existing.add(norm(feature))
            added.append(feature)
    if added:
        logging.info(
            "Added Stage 2 selected feature(s) absent from config.INPUT_COLUMNS: %s",
            added,
        )
    logging.info("Effective local input columns: %s", INPUT_FEATURES)


def describe_blocks(spec):
    blocks = ["SF"]
    if spec["temporal"]:
        blocks.append("T")
    if spec["spatial_mode"] == "pollutants_except_target":
        blocks.append("S")
    elif spec["spatial_mode"] == "all_pollutants":
        blocks.append("ST")
    if spec["use_idw"]:
        blocks.append("I")
    return "".join(blocks)


def add_spatial_block(df, target_site, spatial_mode):
    if not spatial_mode:
        return df.copy(), []
    old_mode = getattr(config, "SPATIAL_FEATURE_MODE", "pollutants_except_target")
    try:
        config.SPATIAL_FEATURE_MODE = spatial_mode
        temp = df.set_index("DateTime")
        spatial_features = load_spatial_features(
            str(ACTIVE_SITE_INPUT_DIR),
            target_site,
            TARGET_COLUMN,
            temp.index,
            max_distance=MAX_SPATIAL_DISTANCE_KM,
        )
    finally:
        config.SPATIAL_FEATURE_MODE = old_mode
    if spatial_features is None or spatial_features.empty:
        return df.copy(), []
    merged = pd.concat([temp, spatial_features], axis=1).reset_index()
    return merged, list(spatial_features.columns)


def build_site_configuration(site_df, selected_local_features, spec):
    df = site_df.copy().sort_values("DateTime").reset_index(drop=True)
    target_site = str(df["Site"].iloc[0])
    features = []

    local_features = []
    for feature in selected_local_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
            local_features.append(feature)
    features.extend(local_features)

    if spec["temporal"]:
        df = add_temporal_features(df, datetime_column="DateTime")
        features.extend([col for col in TEMPORAL_COLUMNS if col in df.columns])

    if spec["spatial_mode"]:
        df, spatial_cols = add_spatial_block(df, target_site, spec["spatial_mode"])
        features.extend(spatial_cols)

    if spec["use_idw"]:
        neighbor_data = load_neighbor_data(target_site, TARGET_COLUMN)
        if neighbor_data and getattr(config, "SITE_COORDINATES", None):
            temp = df.set_index("DateTime")
            temp["IDW_Spatial_PM25"] = compute_idw_feature(
                target_site,
                temp.index,
                neighbor_data,
                config.SITE_COORDINATES,
                power=2,
                max_distance=MAX_SPATIAL_DISTANCE_KM,
            )
            df = temp.reset_index()
            features.append("IDW_Spatial_PM25")

    features = [col for col in dict.fromkeys(features) if col in df.columns]
    if features:
        df.loc[:, features] = df.loc[:, features].apply(pd.to_numeric, errors="coerce")
        df.loc[:, features] = df.loc[:, features].ffill().bfill().fillna(0)
    return df, features


def build_region_configuration_dataset(region_df, selected_local_features, spec):
    parts = []
    feature_union = []
    for _, site_df in region_df.groupby("Site", sort=False):
        built, site_features = build_site_configuration(site_df, selected_local_features, spec)
        parts.append(built)
        feature_union.extend(site_features)
    data = pd.concat(parts, ignore_index=True)
    data = data.sort_values(["Site", "DateTime"]).reset_index(drop=True)
    feature_list = [col for col in dict.fromkeys(feature_union) if col in data.columns]
    if feature_list:
        data.loc[:, feature_list] = data.loc[:, feature_list].apply(pd.to_numeric, errors="coerce").fillna(0)
    return data, feature_list


def eval_region_configuration(region_df, features, region, config_name, regime, miss_level, seed):
    features = [f for f in features if f in region_df.columns]
    work = region_df[["DateTime", "Site", "Region", TARGET_COLUMN] + features].copy()
    for col in [TARGET_COLUMN] + features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    mask = make_region_mask(region_df[["DateTime", "Site", "Region", TARGET_COLUMN]].copy(), regime, miss_level, seed)
    work.loc[mask, TARGET_COLUMN] = np.nan
    train = work[TARGET_COLUMN].notna()
    test = mask.copy()
    if train.sum() < MIN_TRAIN_ROWS or test.sum() < MIN_TEST_ROWS or len(features) == 0:
        return {
            "Region": region,
            "Configuration": config_name,
            "Regime": regime,
            "Missingness_Level": miss_level,
            "Seed": seed,
            "N_Features": len(features),
            "N_Train": int(train.sum()),
            "N_Test": int(test.sum()),
            "RMSE": np.nan,
            "MAE": np.nan,
            "R": np.nan,
            "NSE": np.nan,
            "N_Valid": 0,
        }
    model = random_forest_model(seed)
    model.fit(work.loc[train, features], work.loc[train, TARGET_COLUMN])
    pred = model.predict(work.loc[test, features])
    metric_row = metric_summary(region_df.loc[test, TARGET_COLUMN].values, pred)
    metric_row.update(
        {
            "Region": region,
            "Configuration": config_name,
            "Regime": regime,
            "Missingness_Level": miss_level,
            "Seed": seed,
            "N_Features": len(features),
            "N_Train": int(train.sum()),
            "N_Test": int(test.sum()),
        }
    )
    return metric_row


def plot_region_rmse(summary_df, region):
    frame = summary_df[summary_df["Region"] == region].copy()
    if frame.empty:
        return
    frame = frame.sort_values("RMSE_mean")
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(frame) + 1.5)))
    ax.barh(frame["Configuration"], frame["RMSE_mean"], xerr=frame["RMSE_sd"].fillna(0), color="#1f77b4", alpha=0.85)
    ax.set_xlabel("RMSE")
    ax.set_title(f"{region}: progressive feature-block comparison")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{safe(region)}_rmse_progressive_configs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_region_r(summary_df, region):
    frame = summary_df[summary_df["Region"] == region].copy()
    if frame.empty:
        return
    frame = frame.sort_values("R_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(frame) + 1.5)))
    ax.barh(frame["Configuration"], frame["R_mean"], xerr=frame["R_sd"].fillna(0), color="#d55e00", alpha=0.85)
    ax.set_xlabel("Correlation coefficient (R)")
    ax.set_title(f"{region}: progressive feature-block correlation")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{safe(region)}_r_progressive_configs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_results(results_df):
    grouped = (
        results_df.groupby(["Region", "Configuration", "Blocks", "Selected_Local_Features", "Feature_List"], dropna=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            R_mean=("R", "mean"),
            R_sd=("R", "std"),
            NSE_mean=("NSE", "mean"),
            N_Valid_mean=("N_Valid", "mean"),
            N_Features=("N_Features", "first"),
            N_Runs=("Seed", "count"),
        )
        .reset_index()
    )
    return grouped.sort_values(["Region", "RMSE_mean", "Configuration"]).reset_index(drop=True)


def main():
    logging.info("RUN TYPE: %s", RUN_TYPE_LABEL)
    print("=== RUN TYPE: %s ===" % RUN_TYPE_LABEL, flush=True)
    logging.info("Input folder: %s", INPUT_DIR)
    logging.info("Selected features CSV: %s", SELECTED_FEATURES_CSV)
    logging.info("Output folder: %s", OUTPUT_DIR)

    selected_feature_map = load_selected_feature_map()

    if TARGET_REGIONS:
        wanted = {norm(x) for x in TARGET_REGIONS}
        selected_feature_map = {region: feats for region, feats in selected_feature_map.items() if norm(region) in wanted}

    if not selected_feature_map:
        raise RuntimeError(
            "No Stage 2 selected features matched target %s and regions %s in %s"
            % (TARGET_COLUMN, TARGET_REGIONS, SELECTED_FEATURES_CSV)
        )

    # This must happen before load_all_data(), because that function uses
    # INPUT_FEATURES to decide which columns to retain from every site file.
    add_selected_features_to_input_columns(selected_feature_map)
    all_data = load_all_data()

    raw_rows = []
    feature_rows = []

    for region, selected_local_features in selected_feature_map.items():
        region_df = all_data[all_data["Region"] == region].copy()
        if region_df.empty:
            logging.info("Skipping %s: no pooled rows found", region)
            continue
        region_df = region_df.sort_values(["Site", "DateTime"]).reset_index(drop=True)
        if region_df["Site"].nunique() < MIN_SITES:
            logging.info("Skipping %s: fewer than %d sites", region, MIN_SITES)
            continue

        unavailable = [
            feature
            for feature in selected_local_features
            if feature not in region_df.columns
            or pd.to_numeric(region_df[feature], errors="coerce").notna().sum() == 0
        ]
        if unavailable:
            raise ValueError(
                "%s Stage 2 selected feature(s) are unavailable in the regional input data: %s"
                % (region, ", ".join(unavailable))
            )

        logging.info("=== Region: %s | selected local features: %s ===", region, selected_local_features)

        for spec in CONFIG_SPECS:
            built_df, feature_list = build_region_configuration_dataset(region_df, selected_local_features, spec)
            config_name = spec["name"]
            blocks = describe_blocks(spec)
            feature_rows.append(
                {
                    "Region": region,
                    "Configuration": config_name,
                    "Blocks": blocks,
                    "Selected_Local_Features": ",".join(selected_local_features),
                    "Feature_List": ",".join(feature_list),
                    "N_Features": len(feature_list),
                }
            )
            logging.info("%s | %s | %d features", region, config_name, len(feature_list))

            for regime in MISSINGNESS_REGIMES:
                for miss_level in MISSINGNESS_LEVELS:
                    for seed in SEEDS:
                        row = eval_region_configuration(
                            built_df,
                            feature_list,
                            region,
                            config_name,
                            regime,
                            miss_level,
                            seed,
                        )
                        row["Blocks"] = blocks
                        row["Selected_Local_Features"] = ",".join(selected_local_features)
                        row["Feature_List"] = ",".join(feature_list)
                        raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    feature_df = pd.DataFrame(feature_rows).drop_duplicates().reset_index(drop=True)

    raw_path = RAW_DIR / "regional_progressive_feature_block_results.csv"
    feature_path = SUMMARY_DIR / "regional_progressive_feature_sets.csv"
    raw_df.to_csv(raw_path, index=False)
    feature_df.to_csv(feature_path, index=False)

    if raw_df.empty:
        logging.info("No results generated.")
        return

    summary_df = aggregate_results(raw_df)
    summary_path = SUMMARY_DIR / "regional_progressive_feature_block_summary.csv"
    best_path = SUMMARY_DIR / "regional_progressive_best_configuration_by_region.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_df.groupby("Region", as_index=False).first().to_csv(best_path, index=False)

    for region in summary_df["Region"].dropna().unique():
        plot_region_rmse(summary_df, region)
        plot_region_r(summary_df, region)

    run_settings = {
        "FEATURE_SELECTION_MODEL": "RandomForest",
        "INPUT_DIR": str(INPUT_DIR),
        "SELECTED_FEATURES_CSV": str(SELECTED_FEATURES_CSV),
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "TARGET_COLUMN": TARGET_COLUMN,
        "INPUT_FEATURES": INPUT_FEATURES,
        "TARGET_REGIONS": TARGET_REGIONS,
        "MISSINGNESS_REGIMES": MISSINGNESS_REGIMES,
        "MISSINGNESS_LEVELS": MISSINGNESS_LEVELS,
        "SEEDS": SEEDS,
        "MIN_SITES": MIN_SITES,
        "MIN_TRAIN_ROWS": MIN_TRAIN_ROWS,
        "MIN_TEST_ROWS": MIN_TEST_ROWS,
        "MAX_SPATIAL_DISTANCE_KM": MAX_SPATIAL_DISTANCE_KM,
        "CONFIG_SPECS": CONFIG_SPECS,
    }
    with open(OUTPUT_DIR / "run_settings.json", "w", encoding="utf-8") as handle:
        json.dump(run_settings, handle, indent=2)

    logging.info("Saved raw results: %s", raw_path)
    logging.info("Saved feature sets: %s", feature_path)
    logging.info("Saved summary: %s", summary_path)
    logging.info("Saved best-by-region summary: %s", best_path)


if __name__ == "__main__":
    main()
    print("=== JOB FINISHED: Stage 3 progressive regional evaluation ===", flush=True)
