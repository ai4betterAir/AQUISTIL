"""
regional_rf_shap_feature_selection.py

Regional feature-combination selection for PM2.5 imputation.

Workflow:
1) Build regional pooled datasets from site CSV files.
2) Train full Random Forest model and generate SHAP plots for interpretation.
3) Train the shared Random Forest model for RMSE.
4) Leave-one-variable-out Random Forest removal test.
5) Grouped removal tests, e.g., NO+NO2, WSP+WDR, TEMP+HUMID+RAIN.
6) Greedy backward Random Forest selection to find compact best combinations.
7) Save final feature subset per region.

RF + SHAP = explanation.
Random Forest RMSE = final selection.

Example HPC run:

export INPUT_DIR=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Processed_AQMS_Data
export SITE_REGION_CSV=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/site_region_map.csv
export OUTPUT_DIR=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection/02Regional_RF_SHAP_Selection
export TARGET_COLUMN=PM2.5
export INPUT_FEATURES=PM10,CO,NO2,NOX,NEPH,OZONE,TEMP,NO,WSP,WDR,HUMID,RAIN
export MIN_SITES=3
export MISSINGNESS_LEVEL=0.10
export MISSINGNESS_REGIME=random
export SEEDS=42,101,202
python3 02regional_rf_shap_feature_selection.py
"""

import os
import re
import sys
import json
import time
import argparse
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Optional project imports
_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT_DIR = _SCRIPT_PATH.parents[3]
APP_ROOT_DIR = _SCRIPT_PATH.parents[2]
for _candidate in (_SCRIPT_PATH.parents[1], _SCRIPT_PATH.parents[2]):
    if (_candidate / "config_spatial.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

try:
    import config_spatial as config
except Exception:
    config = None

try:
    from missingness_regimes import apply_missingness as project_apply_missingness
except Exception:
    project_apply_missingness = None

# =============================================================================
# Settings
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
# SELECTED_REGIONS = ["Lower Hunter"]
SELECTED_TARGETS = AVAILABLE_TARGETS[:]
SELECTED_REGIONS = AVAILABLE_REGIONS[:]
def _apply_cli_env_overrides():
    """Allow standalone runs with explicit CLI flags instead of only env vars."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--target")
    parser.add_argument("--regions")
    parser.add_argument("--input-features")
    parser.add_argument("--site-region-csv")
    parser.add_argument("--min-sites", type=int)
    args, _ = parser.parse_known_args()

    if args.input_dir:
        os.environ["INPUT_DIR"] = args.input_dir
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
EVENT_RUN = FEATURE_SELECTION_RUN_MODE == "event" or os.getenv("MISSINGNESS_REGIME", "random").strip().lower() == "event"
RUN_TYPE_LABEL = "EVENT FEATURE SELECTION RUN" if EVENT_RUN else "DEFAULT FEATURE SELECTION RUN"

def env_list(name, default):
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]

INPUT_DIR = Path(
    os.getenv(
        "INPUT_DIR",
        getattr(
            config,
            "WIDE_API_INPUT_DIRECTORY",
            getattr(config, "INPUT_DIRECTORY", ".") if config else ".",
        ) if getattr(config, "USE_WIDE_NOWCASTING_INPUTS", False) else getattr(config, "INPUT_DIRECTORY", ".") if config else ".",
    )
)
FEATURE_SELECTION_OUTPUT_ROOT = Path(
    os.getenv(
        "FEATURE_SELECTION_OUTPUT_ROOT",
        str(APP_ROOT_DIR / "Outputs" / "Feature_Selection"),
    )
)
if EVENT_RUN:
    FEATURE_SELECTION_OUTPUT_ROOT = FEATURE_SELECTION_OUTPUT_ROOT / "feature_selection_event"
DEFAULT_OUTPUT_DIR = FEATURE_SELECTION_OUTPUT_ROOT / "02Regional_RF_SHAP_Selection"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
SITE_REGION_CSV = os.getenv("SITE_REGION_CSV", "").strip()
TARGET_COLUMN = os.getenv("TARGET_COLUMN", SELECTED_TARGET)
INPUT_FEATURES = env_list(
    "INPUT_FEATURES",
    ",".join(getattr(config, "LOCAL_ANALYSIS_INPUTS", [
        "CO", "HUMID", "NEPH", "NO", "NO2", "NOX",
        "OZONE", "PM10", "RAIN", "TEMP", "WDR", "WSP",
    ])),
)
MIN_SITES = int(os.getenv("MIN_SITES", "3"))
MISSINGNESS_LEVEL = float(os.getenv("MISSINGNESS_LEVEL", "0.10"))
MISSINGNESS_REGIME = os.getenv(
    "MISSINGNESS_REGIME",
    "event" if FEATURE_SELECTION_RUN_MODE == "event" else "random",
).strip()
SEEDS = [int(x) for x in env_list("SEEDS", "42,101,202")]
RF_SHAP_SEEDS = [int(x) for x in env_list("RF_SHAP_SEEDS", str(SEEDS[0]))]
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "100"))
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "30"))
SHAP_SAMPLE_SIZE = int(os.getenv("SHAP_SAMPLE_SIZE", "500"))
RF_MAX_TRAIN_ROWS = int(os.getenv("RF_MAX_TRAIN_ROWS", "50000"))
RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "120"))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "16"))
RF_MIN_SAMPLES_LEAF = int(os.getenv("RF_MIN_SAMPLES_LEAF", "5"))
RF_MAX_FEATURES = os.getenv("RF_MAX_FEATURES", "sqrt").strip() or "sqrt"
SHAP_APPROXIMATE = os.getenv("SHAP_APPROXIMATE", "true").strip().lower() not in {"0", "false", "no"}
SAVE_COMBINED_SNAPSHOT = os.getenv("SAVE_COMBINED_SNAPSHOT", "false").strip().lower() in {"1", "true", "yes"}
BACKWARD_TOLERANCE = float(os.getenv("BACKWARD_TOLERANCE", "0.01"))
FINAL_TOLERANCE = float(os.getenv("FINAL_TOLERANCE", "0.01"))
ROBUST_SELECTION = os.getenv("ROBUST_SELECTION", "true").strip().lower() not in {"0", "false", "no"}
ROBUST_FINAL_TOLERANCE = float(os.getenv("ROBUST_FINAL_TOLERANCE", str(max(FINAL_TOLERANCE, 0.02))))
ROBUST_WEAK_THRESHOLD = float(os.getenv("ROBUST_WEAK_THRESHOLD", "0.12"))
ROBUST_SHAP_THRESHOLD = float(os.getenv("ROBUST_SHAP_THRESHOLD", "0.15"))
ROBUST_LOO_THRESHOLD = float(os.getenv("ROBUST_LOO_THRESHOLD", "0.05"))
ROBUST_CORR_THRESHOLD = float(os.getenv("ROBUST_CORR_THRESHOLD", "0.55"))
ROBUST_KEEP_MIN_FEATURES = int(os.getenv("ROBUST_KEEP_MIN_FEATURES", "2"))
TARGET_REGIONS = env_list("TARGET_REGIONS", ",".join(SELECTED_REGIONS))
N_JOBS = int(os.getenv("N_JOBS", "-1"))
JOB_ID = os.getenv("SLURM_JOB_ID", "local")

GROUP_TESTS = {
    "NO_NO2": ["NO", "NO2"],
    "NO_NO2_OZONE": ["NO", "NO2", "OZONE"],
    "WSP_WDR": ["WSP", "WDR"],
    "TEMP_HUMID_RAIN": ["TEMP", "HUMID", "RAIN"],
    "CO_NO_NO2": ["CO", "NO", "NO2"],
}

SUMMARY_DIR = OUTPUT_DIR / "summary_outputs"
DIAGNOSTIC_CSV_DIR = OUTPUT_DIR / "diagnostic_csv"
DIAGNOSTIC_PLOTS_DIR = OUTPUT_DIR / "diagnostic_plots"
CSV_DIR = DIAGNOSTIC_CSV_DIR
PLOTS_DIR = DIAGNOSTIC_PLOTS_DIR
LOG_PATH = OUTPUT_DIR / f"regional_rf_shap_feature_selection_{JOB_ID}.log"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_CSV_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PREPARED_SITE_INPUT_DIR = OUTPUT_DIR / "_prepared_site_inputs"
PREPARED_SITE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_SITE_INPUT_DIR = INPUT_DIR
REGION_MAP_OVERRIDE = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    force=True,
)

# =============================================================================
# Name and column helpers
# =============================================================================

def norm(x):
    return re.sub(r"[^A-Za-z0-9]+", "", str(x)).lower()

def safe(x):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_") or "NA"

def infer_site_name(p):
    stem = Path(p).stem
    for suffix in ["_processed", "_aqms", "_station", "_site", "_data", "_clean", "_hourly", "_merged", "_final"]:
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
    if "_" in stem:
        stem = stem.split("_")[0]
    return stem.strip().upper()

def infer_region_name(p):
    stem = Path(p).stem
    prefix = "Allobs_processed_DPE_station_api_"
    suffix = "_ALL"
    if stem.startswith(prefix) and stem.endswith(suffix):
        token = stem[len(prefix) : -len(suffix)]
        return str(token).replace("_", " ").strip()
    return stem.replace("_", " ").strip()

def is_wide_region_input_dir(input_dir):
    return any(input_dir.glob("Allobs_processed_DPE_station_api_*_ALL.csv"))

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
        out["DateTime"] = pd.to_datetime(out[lower["date"]].astype(str) + " " + out[lower["time"]].astype(str), errors="coerce", dayfirst=True)
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

# =============================================================================
# Data loading
# =============================================================================

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
            feature = feature.strip()
            site_name = site_name.strip()
            if not feature or not site_name:
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

def load_region_map():
    global REGION_MAP_OVERRIDE

    if REGION_MAP_OVERRIDE:
        return dict(REGION_MAP_OVERRIDE)

    def read_region_csv(csv_path):
        m = pd.read_csv(csv_path)
        lower = {c.lower(): c for c in m.columns}
        site_col = None
        region_col = None
        for candidate in ["site", "sitename", "station", "stationname"]:
            if candidate in lower:
                site_col = lower[candidate]
                break
        for candidate in ["region"]:
            if candidate in lower:
                region_col = lower[candidate]
                break
        if site_col is None or region_col is None:
            raise ValueError(
                f"Region CSV {csv_path} must contain a site column "
                f"(e.g. Site or SiteName) and a Region column"
            )
        out = {}
        for _, r in m.iterrows():
            site = str(r[site_col]).strip()
            region = str(r[region_col]).strip()
            if site and region and site.lower() != "nan" and region.lower() != "nan":
                out[norm(site)] = region
        if out:
            logging.info("Loaded region map: %s (%d sites)", csv_path, len(out))
        return out

    def read_region_json(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            site_to_region = data.get("site_to_region")
            if isinstance(site_to_region, dict):
                out = {
                    norm(site): str(region).strip()
                    for site, region in site_to_region.items()
                    if str(site).strip() and str(region).strip()
                }
                if out:
                    logging.info("Loaded region map JSON: %s (%d sites)", json_path, len(out))
                return out
        return {}

    mapping = {}
    if SITE_REGION_CSV and Path(SITE_REGION_CSV).exists():
        return read_region_csv(Path(SITE_REGION_CSV))
    if config is not None:
        for attr in ["SITE_REGION_MAP", "REGION_MAP", "SITE_REGIONS"]:
            obj = getattr(config, attr, None)
            if isinstance(obj, dict):
                if all(not isinstance(v, (list, tuple, set)) for v in obj.values()):
                    return {norm(k): str(v) for k, v in obj.items()}
                for region, sites in obj.items():
                    if isinstance(sites, (list, tuple, set)):
                        for site in sites:
                            mapping[norm(site)] = str(region)
                if mapping:
                    return mapping
        region_json = getattr(config, "REGION_SITE_JSON_FILE", None)
        if region_json and Path(region_json).exists():
            mapping = read_region_json(Path(region_json))
            if mapping:
                return mapping

    fallback_csv = PROJECT_ROOT_DIR / "Station_info.csv"
    if fallback_csv.exists():
        mapping = read_region_csv(fallback_csv)
        if mapping:
            return mapping

    fallback_json = PROJECT_ROOT_DIR / "Imputation_Result_Spatial_Temporal_V25_final" / "region_site_mapping.json"
    if fallback_json.exists():
        mapping = read_region_json(fallback_json)
        if mapping:
            return mapping

    raise FileNotFoundError(
        "No region map found. Set SITE_REGION_CSV to a CSV with Site/SiteName and Region columns."
    )

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
    out = pd.DataFrame({
        "DateTime": raw["DateTime"],
        "Site": site,
        "Region": region,
        TARGET_COLUMN: pd.to_numeric(raw[target_col], errors="coerce"),
    })
    for f in INPUT_FEATURES:
        if norm(f) == norm(TARGET_COLUMN):
            continue
        col = resolve_input(raw, f, site)
        if col is None or norm(col) == norm(target_col):
            continue
        out[f] = pd.to_numeric(raw[col], errors="coerce")
    out = out.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    return out

def load_all_data():
    global ACTIVE_SITE_INPUT_DIR, REGION_MAP_OVERRIDE

    if is_wide_region_input_dir(INPUT_DIR):
        ACTIVE_SITE_INPUT_DIR, REGION_MAP_OVERRIDE = materialize_site_inputs_from_wide(INPUT_DIR)
    else:
        ACTIVE_SITE_INPUT_DIR = INPUT_DIR
        REGION_MAP_OVERRIDE = None

    region_map = load_region_map()
    files = sorted(ACTIVE_SITE_INPUT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {ACTIVE_SITE_INPUT_DIR}")
    logging.info("Loading site CSV files from %s", ACTIVE_SITE_INPUT_DIR)
    logging.info("Found %d CSV files. Starting site-data preparation.", len(files))
    parts = []
    used_count = 0
    skipped_count = 0
    for i, f in enumerate(files, start=1):
        d = prepare_site_file(f, region_map)
        if d is not None and not d.empty:
            parts.append(d)
            used_count += 1
        else:
            skipped_count += 1
        if i == 1 or i % 10 == 0 or i == len(files):
            logging.info(
                "Loaded %d/%d files | usable=%d | skipped=%d | latest=%s",
                i,
                len(files),
                used_count,
                skipped_count,
                f.name,
            )
    if not parts:
        raise RuntimeError("No valid site data loaded")
    data = pd.concat(parts, ignore_index=True)
    for f in INPUT_FEATURES:
        if f not in data.columns and norm(f) != norm(TARGET_COLUMN):
            data[f] = np.nan
    logging.info(
        "Combined dataset ready: %d rows, %d sites, %d regions",
        len(data),
        data["Site"].nunique(),
        data["Region"].nunique(),
    )
    return data

# =============================================================================
# Missingness and metrics
# =============================================================================

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

def make_region_mask(region_df, seed):
    mask = pd.Series(False, index=region_df.index)
    for i, (_, g) in enumerate(region_df.groupby("Site", sort=False)):
        _, m = apply_missingness(g.copy(), TARGET_COLUMN, MISSINGNESS_REGIME, MISSINGNESS_LEVEL, seed + i * 1009)
        if not (isinstance(m, pd.Series) and m.index.equals(g.index)):
            m = pd.Series(np.asarray(m).astype(bool), index=g.index)
        mask.loc[g.index] = m.astype(bool) & g[TARGET_COLUMN].notna()
    return mask

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return dict(RMSE=np.nan, MAE=np.nan, R=np.nan, NSE=np.nan, N_Valid=0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = float(1 - np.sum((y_true - y_pred) ** 2) / den) if den > 0 else np.nan
    return dict(RMSE=rmse, MAE=mae, R=r, NSE=nse, N_Valid=int(len(y_true)))

# =============================================================================
# Models
# =============================================================================

def selection_rf(seed):
    """All Stage 2 subset scoring uses the shared Random Forest."""
    return rf(seed)

def rf(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            random_state=seed,
            n_jobs=N_JOBS,
        )),
    ])

def prepare_region_model_data(region_df, features):
    numeric = pd.DataFrame(index=region_df.index)
    numeric[TARGET_COLUMN] = pd.to_numeric(region_df[TARGET_COLUMN], errors="coerce").astype(np.float32)
    for feature in features:
        numeric[feature] = pd.to_numeric(region_df[feature], errors="coerce").astype(np.float32)
    return {
        "numeric": numeric,
        "target": numeric[TARGET_COLUMN],
        "feature_frame": numeric[features],
    }

def _sample_training_rows(Xtr, ytr, seed, max_rows):
    if max_rows <= 0 or len(Xtr) <= max_rows:
        return Xtr, ytr
    sampled_idx = Xtr.sample(n=max_rows, random_state=seed).index
    return Xtr.loc[sampled_idx], ytr.loc[sampled_idx]

def eval_rf_subset(model_data, features, mask, seed, region, experiment, removed=""):
    available_features = [f for f in features if f in model_data["feature_frame"].columns]
    y = model_data["target"]
    train = y.notna() & (~mask)
    test = mask.copy()
    X = model_data["feature_frame"]
    if train.sum() < MIN_TRAIN_ROWS or test.sum() < MIN_TEST_ROWS or len(available_features) == 0:
        return dict(Region=region, Experiment=experiment, Removed=removed, Features=",".join(available_features), N_Features=len(available_features), Seed=seed, RMSE=np.nan, MAE=np.nan, R=np.nan, NSE=np.nan, N_Train=int(train.sum()), N_Test=int(test.sum()), N_Valid=0)
    model = selection_rf(seed)
    model.fit(X.loc[train, available_features], y.loc[train])
    pred = model.predict(X.loc[test, available_features])
    m = metrics(y.loc[test].values, pred)
    return dict(Region=region, Experiment=experiment, Removed=removed, Features=",".join(available_features), N_Features=len(available_features), Seed=seed, N_Train=int(train.sum()), N_Test=int(test.sum()), **m)

def eval_rf_shap(model_data, features, mask, seed, region):
    available_features = [f for f in features if f in model_data["feature_frame"].columns]
    y = model_data["target"]
    train = y.notna() & (~mask)
    test = mask.copy()
    X = model_data["feature_frame"]
    if train.sum() < MIN_TRAIN_ROWS or test.sum() < MIN_TEST_ROWS or len(available_features) == 0:
        return {}, pd.DataFrame(), None, None
    model = rf(seed)
    Xtr = X.loc[train, available_features]
    ytr = y.loc[train]
    Xte = X.loc[test, available_features]
    Xfit, yfit = _sample_training_rows(Xtr, ytr, seed, RF_MAX_TRAIN_ROWS)
    phase_started = time.time()
    logging.info(
        "%s | RF seed=%s fitting %d rows, %d features, %d trees, max_depth=%d",
        region, seed, len(Xfit), len(available_features), RF_N_ESTIMATORS, RF_MAX_DEPTH,
    )
    model.fit(Xfit, yfit)
    logging.info("%s | RF seed=%s fit complete in %.1f seconds", region, seed, time.time() - phase_started)
    phase_started = time.time()
    pred = model.predict(Xte)
    logging.info(
        "%s | RF seed=%s predicted %d masked rows in %.1f seconds",
        region, seed, len(Xte), time.time() - phase_started,
    )
    m = metrics(y.loc[test].values, pred)
    metric_row = dict(
        Region=region,
        Model="RandomForest",
        Experiment="RF_full_all_variables",
        Seed=seed,
        Features=",".join(available_features),
        N_Features=len(available_features),
        N_Train=int(train.sum()),
        N_Train_Used=int(len(Xfit)),
        N_Test=int(test.sum()),
        **m,
    )
    if not SHAP_AVAILABLE:
        return metric_row, pd.DataFrame(), None, None
    try:
        imputer = model.named_steps["imputer"]
        rf_model = model.named_steps["model"]
        # Select the explanation sample before transforming; previously the
        # full training frame was imputed only to discard almost all of it.
        Xsample = Xfit.sample(min(SHAP_SAMPLE_SIZE, len(Xfit)), random_state=seed)
        Xs = pd.DataFrame(
            imputer.transform(Xsample),
            columns=available_features,
            index=Xsample.index,
        )
        phase_started = time.time()
        logging.info(
            "%s | SHAP seed=%s starting for %d rows (approximate=%s)",
            region, seed, len(Xs), SHAP_APPROXIMATE,
        )
        explainer = shap.TreeExplainer(rf_model)
        try:
            sv = explainer.shap_values(
                Xs,
                approximate=SHAP_APPROXIMATE,
                check_additivity=False,
            )
        except TypeError:
            logging.info("Installed SHAP does not support approximate/check_additivity arguments; using compatible call")
            sv = explainer.shap_values(Xs)
        logging.info("%s | SHAP seed=%s complete in %.1f seconds", region, seed, time.time() - phase_started)
        if isinstance(sv, list):
            sv = sv[0]
        shap_df = pd.DataFrame({"Region": region, "Model": "RandomForest", "Seed": seed, "Variable": available_features, "MeanAbsSHAP": np.abs(sv).mean(axis=0)})
        shap_df = shap_df.sort_values("MeanAbsSHAP", ascending=False)
        shap_df["SHAP_Rank"] = range(1, len(shap_df) + 1)
        return metric_row, shap_df, sv, Xs
    except Exception as exc:
        logging.warning("SHAP failed for %s seed %s: %s", region, seed, exc)
        return metric_row, pd.DataFrame(), None, None

# =============================================================================
# Plots
# =============================================================================

def plot_shap_bar(shap_summary, region):
    if shap_summary.empty: return
    d = shap_summary.sort_values("MeanAbsSHAP_mean" if "MeanAbsSHAP_mean" in shap_summary.columns else "MeanAbsSHAP", ascending=True)
    val_col = "MeanAbsSHAP_mean" if "MeanAbsSHAP_mean" in d.columns else "MeanAbsSHAP"
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(d) + 2)))
    ax.barh(d["Variable"], d[val_col])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title(f"{region}: RF-SHAP feature importance")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{safe(region)}_RF_SHAP_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_shap_beeswarm(region, sv, Xs):
    if not SHAP_AVAILABLE or sv is None or Xs is None: return
    try:
        font_scale = 1.4
        plt.figure(figsize=(11, 7))
        shap.summary_plot(sv, Xs, show=False, max_display=15)
        fig = plt.gcf()
        axes = fig.axes
        if axes:
            main_ax = axes[0]
            main_ax.set_title(f"{region}: RF-SHAP beeswarm", fontsize=14 * font_scale, fontweight="bold")
            if main_ax.get_xlabel():
                main_ax.set_xlabel(main_ax.get_xlabel(), fontsize=12 * font_scale)
            if main_ax.get_ylabel():
                main_ax.set_ylabel(main_ax.get_ylabel(), fontsize=12 * font_scale)
            main_ax.tick_params(axis="both", labelsize=12 * font_scale)
            for side in ["top", "right", "bottom", "left"]:
                main_ax.spines[side].set_visible(True)
                main_ax.spines[side].set_linewidth(1.2)
        for extra_ax in axes[1:]:
            extra_ax.tick_params(axis="both", labelsize=11 * font_scale)
            if extra_ax.get_ylabel():
                extra_ax.set_ylabel(extra_ax.get_ylabel(), fontsize=12 * font_scale)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{safe(region)}_RF_SHAP_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logging.warning("Beeswarm failed for %s: %s", region, exc)

def plot_shap_dependence(region, sv, Xs, top_features):
    if not SHAP_AVAILABLE or sv is None or Xs is None: return
    for f in top_features[:3]:
        if f not in Xs.columns: continue
        try:
            plt.figure()
            shap.dependence_plot(f, sv, Xs, show=False, interaction_index=None)
            plt.title(f"{region}: SHAP dependence for {f}")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"{safe(region)}_RF_SHAP_dependence_{safe(f)}.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            logging.warning("Dependence plot failed for %s %s: %s", region, f, exc)

def plot_correlation(feature_frame, features, region):
    features = [f for f in features if f in feature_frame.columns]
    if len(features) < 2: return
    corr = feature_frame[features].corr(method="spearman")
    corr.to_csv(CSV_DIR / f"{safe(region)}_spearman_input_correlation.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(features))); ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(features))); ax.set_yticklabels(features)
    for i in range(len(features)):
        for j in range(len(features)):
            if np.isfinite(corr.values[i, j]): ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{region}: Spearman correlation")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Spearman r")
    plt.tight_layout()
    fig.savefig(SUMMARY_DIR / f"{safe(region)}_input_correlation_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_leave_one_out(region, agg):
    full = agg[agg.Experiment == "RF_full_all_variables"]
    loo = agg[agg.Experiment == "RF_leave_one_out"].copy()
    if full.empty or loo.empty: return
    full_rmse = float(full.RMSE_mean.iloc[0])
    loo["Delta_RMSE_vs_Full"] = loo.RMSE_mean - full_rmse
    loo = loo.sort_values("Delta_RMSE_vs_Full")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(loo) + 2)))
    ax.barh(loo.Removed, loo.Delta_RMSE_vs_Full)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("ΔRMSE after removal compared with full model")
    ax.set_title(f"{region}: Random Forest leave-one-variable-out")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout(); fig.savefig(PLOTS_DIR / f"{safe(region)}_RF_leave_one_out_delta_RMSE.png", dpi=300, bbox_inches="tight"); plt.close(fig)

def plot_summary_panel(region, shap_summary, loo_summary, sv=None, Xs=None):
    if shap_summary is None or shap_summary.empty or loo_summary is None or loo_summary.empty:
        logging.info("Summary panel skipped for %s because SHAP/LOO summary data is missing.", region)
        return

    shap_value_col = "MeanAbsSHAP_mean" if "MeanAbsSHAP_mean" in shap_summary.columns else "MeanAbsSHAP"
    if shap_value_col not in shap_summary.columns or "Variable" not in shap_summary.columns:
        logging.info("Summary panel skipped for %s because SHAP summary columns are incomplete.", region)
        return
    full = loo_summary[loo_summary["Experiment"] == "RF_full_all_variables"].copy()
    loo_only = loo_summary[loo_summary["Experiment"] == "RF_leave_one_out"].copy()
    if full.empty or loo_only.empty or "Removed" not in loo_only.columns or "RMSE_mean" not in loo_only.columns:
        logging.info("Summary panel skipped for %s because leave-one-out summary is incomplete.", region)
        return

    merged = build_importance_table(shap_summary, loo_summary)
    if merged.empty:
        logging.info("Summary panel skipped for %s because merged SHAP/LOO data is empty.", region)
        return

    merged["Sort_Key"] = merged[["SHAP_Normalized", "LOO_Normalized"]].max(axis=1)
    merged = merged.sort_values(["Sort_Key", "Variable"], ascending=[False, True]).reset_index(drop=True)
    merged["Decision"] = np.where(merged["Remove_Candidate"], "Low impact: removable", "Keep / review")
    merged.to_csv(SUMMARY_DIR / f"{safe(region)}_feature_ranking_decision.csv", index=False)

    title_fs = 11
    label_fs = 8
    tick_fs = 7
    legend_fs = 6
    x_label_fs = label_fs * 1.4
    x_tick_fs = tick_fs * 1.4
    legend_right_fs = legend_fs * 1.5
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(9.0, max(4.4, 0.28 * len(merged) + 1.2)),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.0},
    )
    panel_bottom = 0.13
    panel_top = 0.90
    panel_height = panel_top - panel_bottom
    panel_gap = 0.065
    panel_left = 0.055
    panel_width = panel_height * fig.get_figheight() / fig.get_figwidth()
    panel0_pos = [panel_left, panel_bottom, panel_width, panel_height]
    panel1_pos = [panel_left + panel_width + panel_gap, panel_bottom, panel_width, panel_height]
    ax0.set_position(panel0_pos)
    ax1.set_position(panel1_pos)

    y = np.arange(len(merged))
    bar_h = 0.36
    beeswarm_drawn = False
    if SHAP_AVAILABLE and sv is not None and Xs is not None:
        try:
            plt.sca(ax0)
            shap.summary_plot(
                sv,
                Xs,
                show=False,
                max_display=min(15, len(getattr(Xs, "columns", [])) or 15),
                plot_size=None,
            )
            ax0.set_position(panel0_pos)
            ax1.set_position(panel1_pos)
            ax0.set_title(f"{region}: RF-SHAP beeswarm", fontsize=title_fs, fontweight="bold")
            ax0.set_xlabel("SHAP value", fontsize=x_label_fs)
            if ax0.get_ylabel():
                ax0.set_ylabel(ax0.get_ylabel(), fontsize=label_fs)
            ax0.tick_params(axis="x", labelsize=x_tick_fs, pad=1)
            ax0.tick_params(axis="y", labelsize=tick_fs, pad=-16)
            for lab in ax0.get_yticklabels():
                lab.set_horizontalalignment("right")
            extra_axes = [extra_ax for extra_ax in fig.axes if extra_ax not in (ax0, ax1)]
            for extra_ax in extra_axes:
                extra_ax.tick_params(axis="both", labelsize=tick_fs)
                if extra_ax.get_ylabel():
                    extra_ax.set_ylabel(extra_ax.get_ylabel(), fontsize=label_fs)
            colorbar_axes = [
                extra_ax for extra_ax in extra_axes
                if "feature value" in str(extra_ax.get_ylabel()).strip().lower()
            ]
            if colorbar_axes:
                cax = colorbar_axes[0]
                ax0.set_position(panel0_pos)
                ax1.set_position(panel1_pos)
                bbox = ax0.get_position()
                cbar_width = bbox.width * 0.010
                cbar_height = bbox.height * 0.42
                cbar_x = bbox.x1 - cbar_width - bbox.width * 0.190
                cbar_y = bbox.y0 + bbox.height * 0.28
                cax.set_position([cbar_x, cbar_y, cbar_width, cbar_height])
                cax.yaxis.set_label_position("left")
                cax.set_ylabel("Feature value", fontsize=label_fs, rotation=90, labelpad=-12)
                cax.tick_params(axis="y", labelsize=tick_fs, length=0, pad=0)
            for side in ["top", "right", "bottom", "left"]:
                ax0.spines[side].set_visible(True)
                ax0.spines[side].set_linewidth(0.8)
            ax0.set_box_aspect(1)
            beeswarm_drawn = True
        except Exception as exc:
            logging.warning("Direct beeswarm draw failed for %s: %s", region, exc)

    if not beeswarm_drawn:
        beeswarm_path = PLOTS_DIR / f"{safe(region)}_RF_SHAP_beeswarm.png"
        if not beeswarm_path.exists():
            logging.info("Summary panel skipped for %s because beeswarm plot is missing.", region)
            plt.close(fig)
            return
        beeswarm_img = plt.imread(str(beeswarm_path))
        if beeswarm_img.ndim >= 3:
            rgb = beeswarm_img[..., :3]
            nonwhite = (rgb < 0.985).any(axis=2)
            row_idx = np.where(nonwhite.any(axis=1))[0]
            col_idx = np.where(nonwhite.any(axis=0))[0]
            if len(row_idx) > 0 and len(col_idx) > 0:
                pad = 6
                r0 = max(0, row_idx[0] - pad)
                r1 = min(beeswarm_img.shape[0], row_idx[-1] + pad + 1)
                c0 = max(0, col_idx[0] - pad)
                c1 = min(beeswarm_img.shape[1], col_idx[-1] + pad + 1)
                beeswarm_img = beeswarm_img[r0:r1, c0:c1]
        ax0.imshow(beeswarm_img, aspect="auto")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_anchor("N")
        for side in ["top", "right", "bottom", "left"]:
            ax0.spines[side].set_visible(False)
        ax0.set_box_aspect(1)

    ax1.barh(
        y - bar_h / 2,
        merged["SHAP_Normalized"].values,
        height=bar_h,
        color="#d55e00",
        alpha=0.85,
        label="RF SHAP Importance",
    )
    ax1.barh(
        y + bar_h / 2,
        merged["LOO_Normalized"].values,
        height=bar_h,
        color="#0072b2",
        alpha=0.85,
        label="RF leave-one-out impact",
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(merged["Variable"].tolist(), fontsize=tick_fs)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Normalised Importance", fontsize=x_label_fs)
    ax1.set_title(f"{region}: Feature Ranking", fontsize=title_fs, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=x_tick_fs, pad=1)
    ax1.tick_params(axis="y", labelsize=tick_fs, pad=1)
    ax1.set_xticks(np.linspace(0, 1.0, 6))
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=legend_right_fs, frameon=True)
    for tick_label in ax1.get_yticklabels():
        tick_label.set_horizontalalignment("right")
    for side in ["top", "right", "bottom", "left"]:
        ax1.spines[side].set_visible(True)
        ax1.spines[side].set_linewidth(0.8)
    ax1.set_box_aspect(1)
    ax0.set_position(panel0_pos)
    ax1.set_position(panel1_pos)

    out_path = SUMMARY_DIR / f"{safe(region)}_feature_selection_summary_panel.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved summary panel: %s", out_path)

def update_feature_ranking_decision_csv(region, final):
    if not final:
        return
    csv_path = SUMMARY_DIR / f"{safe(region)}_feature_ranking_decision.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    selected = [f for f in str(final.get("Final_Features", "")).split(",") if f]

    df = df[df.get("Variable", "").astype(str) != "FINAL_SELECTED_VARIABLES"].copy()
    df["Selected_In_Final"] = df["Variable"].isin(selected)
    df["Final_Selected_Variables"] = ""
    df["Final_RMSE"] = np.nan
    df["N_Final_Features"] = np.nan
    df["Final_Decision"] = "Kept in final"

    weak_mask = df["Remove_Candidate"].fillna(False).astype(bool)
    kept_mask = df["Selected_In_Final"].fillna(False).astype(bool)
    df.loc[weak_mask & kept_mask, "Final_Decision"] = "Kept in final"
    df.loc[weak_mask & ~kept_mask, "Final_Decision"] = "Removed in final after weak SHAP/LOO screen"
    df.loc[~weak_mask & ~kept_mask, "Final_Decision"] = "Removed in final by backward/correlation step"

    summary_row = {col: "" for col in df.columns}
    for col in df.columns:
        if col in {"SHAP_Importance", "LOO_Impact", "SHAP_Normalized", "LOO_Normalized", "Combined_Score", "Sort_Key", "Final_RMSE"}:
            summary_row[col] = np.nan
        if col in {"Weak_SHAP", "Weak_LOO", "Remove_Candidate", "Selected_In_Final"}:
            summary_row[col] = False
    summary_row["Variable"] = "FINAL_SELECTED_VARIABLES"
    summary_row["Decision"] = "Final selected variables"
    summary_row["Selected_In_Final"] = True
    summary_row["Final_Selected_Variables"] = ",".join(selected)
    summary_row["Final_RMSE"] = final.get("Final_RMSE", np.nan)
    summary_row["N_Final_Features"] = final.get("N_Final_Features", np.nan)
    summary_row["Final_Decision"] = "Final selected variables"

    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)

def plot_backward_path(region, path):
    if path.empty: return
    d = path.sort_values("Step")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(d.Step, d.RMSE, marker="o")
    ax.set_xlabel("Backward selection step"); ax.set_ylabel("RMSE")
    ax.set_title(f"{region}: greedy backward selection path")
    ax.grid(True, linestyle="--", alpha=0.35)
    for _, r in d.iterrows():
        lab = r.Removed_This_Step if r.Removed_This_Step else "Full"
        ax.text(r.Step, r.RMSE, lab, fontsize=8, ha="left", va="bottom")
    plt.tight_layout(); fig.savefig(PLOTS_DIR / f"{safe(region)}_RF_backward_selection_path.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# =============================================================================
# Selection logic
# =============================================================================

def aggregate(rows):
    d = pd.DataFrame(rows)
    if d.empty: return d
    g = ["Region", "Experiment", "Removed", "Features", "N_Features"]
    return d.groupby(g, dropna=False).agg(RMSE_mean=("RMSE","mean"), RMSE_sd=("RMSE","std"), MAE_mean=("MAE","mean"), R_mean=("R","mean"), NSE_mean=("NSE","mean"), N_Valid_mean=("N_Valid","mean"), N_Seeds=("Seed","nunique")).reset_index()

def _with_subset_labels(raw, summary, experiment, removed):
    relabeled_raw = raw.copy()
    relabeled_raw["Experiment"] = experiment
    relabeled_raw["Removed"] = removed
    relabeled_summary = dict(summary)
    relabeled_summary["Experiment"] = experiment
    relabeled_summary["Removed"] = removed
    return relabeled_raw, relabeled_summary

def eval_subset(model_data, subset, masks, region, experiment, removed):
    rows = [eval_rf_subset(model_data, subset, mask, seed, region, experiment, removed) for seed, mask in masks.items()]
    raw = pd.DataFrame(rows)
    return raw, dict(Region=region, Experiment=experiment, Removed=removed, Features=",".join(subset), N_Features=len(subset), RMSE=raw.RMSE.mean(), MAE=raw.MAE.mean(), R=raw.R.mean(), NSE=raw.NSE.mean(), N_Valid=raw.N_Valid.mean(), N_Seeds=raw.Seed.nunique())

def eval_subset_cached(model_data, subset, masks, region, subset_cache, experiment, removed):
    key = tuple(subset)
    if key not in subset_cache:
        subset_cache[key] = eval_subset(model_data, list(key), masks, region, "__cache__", "")
    raw, summary = subset_cache[key]
    return _with_subset_labels(raw, summary, experiment, removed)

def run_loo(model_data, features, masks, region, subset_cache):
    rows = []
    full_raw, _ = eval_subset_cached(model_data, features, masks, region, subset_cache, "RF_full_all_variables", "")
    rows.extend(full_raw.to_dict("records"))
    for removed in features:
        subset = [f for f in features if f != removed]
        subset_raw, _ = eval_subset_cached(model_data, subset, masks, region, subset_cache, "RF_leave_one_out", removed)
        rows.extend(subset_raw.to_dict("records"))
    raw = pd.DataFrame(rows); agg = aggregate(rows)
    if not agg.empty:
        full = agg[agg.Experiment == "RF_full_all_variables"]
        if not full.empty:
            fr = float(full.RMSE_mean.iloc[0])
            agg["Delta_RMSE_vs_Full"] = agg.RMSE_mean - fr
            agg["Delta_RMSE_pct_vs_Full"] = 100 * agg.Delta_RMSE_vs_Full / fr
    return raw, agg

def run_group_removal(model_data, features, masks, region, subset_cache):
    rows = []
    for name, group in GROUP_TESTS.items():
        available = [f for f in group if f in features]
        if not available: continue
        subset = [f for f in features if f not in available]
        if not subset: continue
        subset_raw, _ = eval_subset_cached(model_data, subset, masks, region, subset_cache, "RF_group_removal", name)
        rows.extend(subset_raw.to_dict("records"))
    return pd.DataFrame(rows), aggregate(rows)

def greedy_backward(model_data, features, masks, region, subset_cache):
    tested = []
    path = []
    current = list(features)
    raw, s = eval_subset_cached(model_data, current, masks, region, subset_cache, "RF_backward_step", "")
    tested.append(raw); current_rmse = s["RMSE"]
    path.append(dict(Region=region, Step=0, Removed_This_Step="", Features=",".join(current), N_Features=len(current), RMSE=current_rmse, MAE=s["MAE"], R=s["R"], NSE=s["NSE"], Decision="Start full model"))
    step = 1
    while len(current) > 1:
        candidates = []
        for f in current:
            subset = [x for x in current if x != f]
            raw, s = eval_subset_cached(model_data, subset, masks, region, subset_cache, "RF_backward_candidate", f)
            tested.append(raw)
            candidates.append(dict(Removed_This_Step=f, Features=",".join(subset), N_Features=len(subset), RMSE=s["RMSE"], MAE=s["MAE"], R=s["R"], NSE=s["NSE"]))
        c = pd.DataFrame(candidates).sort_values("RMSE")
        best = c.iloc[0]
        if np.isfinite(best.RMSE) and best.RMSE <= current_rmse * (1 + BACKWARD_TOLERANCE):
            rem = best.Removed_This_Step
            current = [x for x in current if x != rem]
            current_rmse = float(best.RMSE)
            path.append(dict(Region=region, Step=step, Removed_This_Step=rem, Features=",".join(current), N_Features=len(current), RMSE=current_rmse, MAE=best.MAE, R=best.R, NSE=best.NSE, Decision=f"Removed {rem}"))
            step += 1
        else:
            path.append(dict(Region=region, Step=step, Removed_This_Step="", Features=",".join(current), N_Features=len(current), RMSE=current_rmse, MAE=np.nan, R=np.nan, NSE=np.nan, Decision="Stop: no acceptable removal"))
            break
    return pd.concat(tested, ignore_index=True), pd.DataFrame(path)

def choose_final(region, path):
    d = path[path.RMSE.notna()].copy()
    if d.empty: return {}
    min_rmse = float(d.RMSE.min())
    eligible = d[d.RMSE <= min_rmse * (1 + FINAL_TOLERANCE)].copy()
    chosen = eligible.sort_values(["N_Features", "RMSE"]).iloc[0]
    return dict(Region=region, Target=TARGET_COLUMN, Final_Features=chosen.Features, N_Final_Features=int(chosen.N_Features), Final_RMSE=float(chosen.RMSE), Minimum_RMSE_Observed=min_rmse, Final_Tolerance=FINAL_TOLERANCE, Selection_Rule=f"Smallest subset within {FINAL_TOLERANCE*100:.1f}% of minimum RMSE")

def build_importance_table(shap_summary, loo_summary):
    shap_df = pd.DataFrame()
    if isinstance(shap_summary, pd.DataFrame) and not shap_summary.empty:
        shap_value_col = "MeanAbsSHAP_mean" if "MeanAbsSHAP_mean" in shap_summary.columns else "MeanAbsSHAP"
        if shap_value_col in shap_summary.columns and "Variable" in shap_summary.columns:
            shap_df = shap_summary[["Variable", shap_value_col]].rename(columns={shap_value_col: "SHAP_Importance"}).copy()

    loo_df = pd.DataFrame()
    if isinstance(loo_summary, pd.DataFrame) and not loo_summary.empty:
        full = loo_summary[loo_summary["Experiment"] == "RF_full_all_variables"].copy()
        loo_only = loo_summary[loo_summary["Experiment"] == "RF_leave_one_out"].copy()
        if not full.empty and not loo_only.empty and "Removed" in loo_only.columns and "RMSE_mean" in loo_only.columns:
            full_rmse = float(full["RMSE_mean"].iloc[0])
            loo_only["LOO_Impact"] = (loo_only["RMSE_mean"] - full_rmse).clip(lower=0)
            loo_df = loo_only[["Removed", "LOO_Impact"]].rename(columns={"Removed": "Variable"}).copy()

    merged = shap_df.merge(loo_df, on="Variable", how="outer").fillna(0.0)
    if merged.empty:
        return merged

    shap_max = float(merged["SHAP_Importance"].max()) if "SHAP_Importance" in merged.columns else 0.0
    loo_max = float(merged["LOO_Impact"].max()) if "LOO_Impact" in merged.columns else 0.0
    merged["SHAP_Normalized"] = merged["SHAP_Importance"] / shap_max if shap_max > 0 else 0.0
    merged["LOO_Normalized"] = merged["LOO_Impact"] / loo_max if loo_max > 0 else 0.0
    merged["Combined_Score"] = 0.5 * (merged["SHAP_Normalized"] + merged["LOO_Normalized"])
    merged["Weak_SHAP"] = merged["SHAP_Normalized"] < ROBUST_SHAP_THRESHOLD
    merged["Weak_LOO"] = merged["LOO_Normalized"] < ROBUST_LOO_THRESHOLD
    merged["Remove_Candidate"] = merged["Weak_SHAP"] & merged["Weak_LOO"]
    return merged

def choose_final_robust(region, path, shap_summary, loo_summary, model_data, masks, subset_cache):
    base = choose_final(region, path)
    if not base:
        return {}

    if not ROBUST_SELECTION:
        base["Selection_Rule"] = f"{base['Selection_Rule']} | Robust selection disabled"
        return base

    min_rmse = float(base["Minimum_RMSE_Observed"])
    current = [f for f in str(base["Final_Features"]).split(",") if f]
    if len(current) < ROBUST_KEEP_MIN_FEATURES:
        return base

    score_df = build_importance_table(shap_summary, loo_summary)
    score_map = {}
    if not score_df.empty:
        score_map = score_df.set_index("Variable")[["SHAP_Normalized", "LOO_Normalized", "Combined_Score"]].to_dict("index")

    accepted_low = []
    accepted_corr = []
    rule_bits = [base["Selection_Rule"]]
    current_rmse = float(base["Final_RMSE"])

    def subset_ok(subset):
        nonlocal current_rmse
        if len(subset) < ROBUST_KEEP_MIN_FEATURES:
            return False
        _, s = eval_subset_cached(model_data, subset, masks, region, subset_cache, "RF_final_robust_check", "")
        rmse = float(s["RMSE"])
        if np.isfinite(rmse) and rmse <= min_rmse * (1 + ROBUST_FINAL_TOLERANCE):
            current_rmse = rmse
            return True
        return False

    def combined_score(feature):
        return float(score_map.get(feature, {}).get("Combined_Score", 0.0))

    def weak_by_both(feature):
        info = score_map.get(feature, {})
        return (
            float(info.get("SHAP_Normalized", 0.0)) < ROBUST_SHAP_THRESHOLD
            and float(info.get("LOO_Normalized", 0.0)) < ROBUST_LOO_THRESHOLD
        )

    weak_candidates = sorted(
        [f for f in current if weak_by_both(f)],
        key=combined_score,
    )
    for weak_feature in weak_candidates:
        if weak_feature not in current or len(current) <= ROBUST_KEEP_MIN_FEATURES:
            continue
        candidate = [f for f in current if f != weak_feature]
        if candidate != current and subset_ok(candidate):
            accepted_low.append(weak_feature)
            current = candidate
            rule_bits.append(
                f"Removed low-impact feature {weak_feature} "
                f"(SHAP={float(score_map.get(weak_feature, {}).get('SHAP_Normalized', 0.0)):.3f}, "
                f"LOO={float(score_map.get(weak_feature, {}).get('LOO_Normalized', 0.0)):.3f})"
            )

    if accepted_low:
        _, s = eval_subset_cached(model_data, current, masks, region, subset_cache, "RF_final_after_low_impact_forced_removal", "")
        current_rmse = float(s["RMSE"])

    progress = True
    while progress and len(current) > ROBUST_KEEP_MIN_FEATURES:
        progress = False
        corr = model_data["feature_frame"][current].corr(method="spearman").abs()
        if corr.empty:
            break
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        stacked = upper.stack().sort_values(ascending=False)
        for (f1, f2), rho in stacked.items():
            if not np.isfinite(rho) or rho < ROBUST_CORR_THRESHOLD:
                continue
            s1 = combined_score(f1)
            s2 = combined_score(f2)
            weaker, stronger = (f1, f2) if s1 <= s2 else (f2, f1)
            candidate = [f for f in current if f != weaker]
            if candidate != current and subset_ok(candidate):
                accepted_corr.append(f"{weaker}|kept={stronger}|rho={rho:.2f}")
                current = candidate
                rule_bits.append(
                    f"Removed correlated feature {weaker} "
                    f"(kept {stronger}, |rho|={rho:.2f}, weaker combined score)"
                )
                progress = True
                break

    if accepted_corr:
        _, s = eval_subset_cached(model_data, current, masks, region, subset_cache, "RF_final_after_correlation_forced_removal", "")
        current_rmse = float(s["RMSE"])

    base.update(
        Final_Features=",".join(current),
        N_Final_Features=len(current),
        Final_RMSE=current_rmse,
        Initial_Backward_Features=base["Final_Features"],
        Removed_Low_Impact=",".join(accepted_low),
        Removed_High_Correlation=";".join(accepted_corr),
        Robust_Weak_Threshold=ROBUST_WEAK_THRESHOLD,
        Robust_SHAP_Threshold=ROBUST_SHAP_THRESHOLD,
        Robust_LOO_Threshold=ROBUST_LOO_THRESHOLD,
        Robust_Correlation_Threshold=ROBUST_CORR_THRESHOLD,
        Robust_Final_Tolerance=ROBUST_FINAL_TOLERANCE,
        Robust_Note=(
            "Final subset pruned by forced low-SHAP/low-LOO removal, then forced correlation-pruning."
        ),
        Selection_Rule=" | ".join(rule_bits),
    )
    return base

# =============================================================================
# Region loop
# =============================================================================

def run_region(region, region_df):
    started = time.time()
    logging.info("=== Region: %s ===", region)
    if region_df.Site.nunique() < MIN_SITES:
        logging.info("Skip %s: fewer than %d sites", region, MIN_SITES); return None
    features = [f for f in INPUT_FEATURES if f in region_df.columns and norm(f) != norm(TARGET_COLUMN)]
    features = [f for f in features if pd.to_numeric(region_df[f], errors="coerce").notna().sum() >= MIN_TRAIN_ROWS]
    if len(features) < 2:
        logging.info("Skip %s: fewer than 2 usable features", region); return None
    logging.info("Features: %s", features)
    model_data = prepare_region_model_data(region_df, features)
    masks = {seed: make_region_mask(region_df, seed) for seed in sorted(set(SEEDS + RF_SHAP_SEEDS))}
    subset_cache = {}
    plot_correlation(model_data["feature_frame"], features, region)

    # RF + SHAP
    logging.info("%s | Starting RandomForest + SHAP across %d seed(s): %s", region, len(RF_SHAP_SEEDS), RF_SHAP_SEEDS)
    if RF_MAX_TRAIN_ROWS > 0:
        logging.info("%s | RF train rows capped at %d per seed for SHAP stage", region, RF_MAX_TRAIN_ROWS)
    rf_rows, shap_rows = [], []
    first_sv, first_Xs, top_feats = None, None, []
    for idx, seed in enumerate(RF_SHAP_SEEDS, start=1):
        logging.info("%s | RF+SHAP seed %d/%d start (seed=%s)", region, idx, len(RF_SHAP_SEEDS), seed)
        rf_metric, shap_df, sv, Xs = eval_rf_shap(model_data, features, masks[seed], seed, region)
        if rf_metric: rf_rows.append(rf_metric)
        if not shap_df.empty: shap_rows.append(shap_df)
        if seed == RF_SHAP_SEEDS[0]: first_sv, first_Xs = sv, Xs
        logging.info("%s | RF+SHAP seed %d/%d complete (seed=%s)", region, idx, len(RF_SHAP_SEEDS), seed)
    rf_metrics = pd.DataFrame(rf_rows)
    rf_metrics.to_csv(CSV_DIR / f"{safe(region)}_RF_full_metrics.csv", index=False)
    if shap_rows:
        shap_all = pd.concat(shap_rows, ignore_index=True)
        shap_agg = shap_all.groupby(["Region", "Model", "Variable"], as_index=False).agg(MeanAbsSHAP_mean=("MeanAbsSHAP","mean"), MeanAbsSHAP_sd=("MeanAbsSHAP","std"), N_Seeds=("Seed","nunique")).sort_values("MeanAbsSHAP_mean", ascending=False)
        shap_agg["SHAP_Rank"] = range(1, len(shap_agg)+1)
        shap_agg.to_csv(CSV_DIR / f"{safe(region)}_RF_SHAP_summary.csv", index=False)
        plot_shap_bar(shap_agg, region)
        top_feats = shap_agg.Variable.head(3).tolist()
        plot_shap_beeswarm(region, first_sv, first_Xs)
        plot_shap_dependence(region, first_sv, first_Xs, top_feats)
    else:
        shap_agg = pd.DataFrame()

    # LGBM leave-one-out, group removal, backward selection
    logging.info("%s | Starting Random Forest leave-one-out", region)
    loo_raw, loo_agg = run_loo(model_data, features, masks, region, subset_cache)
    loo_raw.to_csv(CSV_DIR / f"{safe(region)}_RF_leave_one_out_raw.csv", index=False)
    loo_agg.to_csv(CSV_DIR / f"{safe(region)}_RF_leave_one_out_summary.csv", index=False)
    plot_leave_one_out(region, loo_agg)
    plot_summary_panel(region, shap_agg, loo_agg, first_sv, first_Xs)

    logging.info("%s | Starting grouped feature removal", region)
    group_raw, group_agg = run_group_removal(model_data, features, masks, region, subset_cache)
    group_raw.to_csv(CSV_DIR / f"{safe(region)}_RF_group_removal_raw.csv", index=False)
    group_agg.to_csv(CSV_DIR / f"{safe(region)}_RF_group_removal_summary.csv", index=False)

    logging.info("%s | Starting greedy backward selection", region)
    back_raw, back_path = greedy_backward(model_data, features, masks, region, subset_cache)
    back_raw.to_csv(CSV_DIR / f"{safe(region)}_RF_backward_raw.csv", index=False)
    back_path.to_csv(CSV_DIR / f"{safe(region)}_RF_backward_path.csv", index=False)
    plot_backward_path(region, back_path)

    final = choose_final_robust(region, back_path, shap_agg, loo_agg, model_data, masks, subset_cache)
    update_feature_ranking_decision_csv(region, final)
    logging.info("%s | Completed in %.1f minutes", region, (time.time() - started) / 60.0)
    return dict(region=region, features=features, rf=rf_metrics, shap=shap_agg, loo=loo_agg, group=group_agg, backward=back_path, final=final)

def main():
    logging.info("RUN TYPE: %s", RUN_TYPE_LABEL)
    print("=== RUN TYPE: %s ===" % RUN_TYPE_LABEL, flush=True)
    logging.info("Job ID: %s", JOB_ID)
    logging.info("Input folder: %s", INPUT_DIR)
    logging.info("Output folder: %s", OUTPUT_DIR)
    logging.info("Target: %s", TARGET_COLUMN)
    logging.info("Features: %s", INPUT_FEATURES)
    logging.info("SHAP available: %s", SHAP_AVAILABLE)
    all_data = load_all_data()
    if SAVE_COMBINED_SNAPSHOT:
        snapshot_path = CSV_DIR / "combined_regional_input_data_snapshot.csv"
        logging.info("Writing optional combined-data snapshot: %s", snapshot_path)
        all_data.to_csv(snapshot_path, index=False)
    else:
        logging.info("Skipping combined-data snapshot (SAVE_COMBINED_SNAPSHOT=false)")
    regions = sorted(all_data.Region.dropna().unique())
    if TARGET_REGIONS:
        wanted = {norm(x) for x in TARGET_REGIONS}
        regions = [r for r in regions if norm(r) in wanted]
    logging.info("Regions selected for analysis: %d", len(regions))
    outputs, finals = [], []
    for i, region in enumerate(regions, start=1):
        logging.info("Processing region %d/%d: %s", i, len(regions), region)
        res = run_region(region, all_data[all_data.Region == region].copy())
        if res is not None:
            outputs.append(res)
            if res["final"]: finals.append(res["final"])
    final_csv_path = SUMMARY_DIR / "FINAL_selected_feature_combination_by_region.csv"
    pd.DataFrame(finals).to_csv(final_csv_path, index=False)
    for key, fname in [("loo", "ALL_REGIONS_RF_leave_one_out_summary.csv"), ("group", "ALL_REGIONS_RF_group_removal_summary.csv"), ("backward", "ALL_REGIONS_RF_backward_path.csv"), ("shap", "ALL_REGIONS_RF_SHAP_summary.csv"), ("rf", "ALL_REGIONS_RF_full_metrics.csv")]:
        frames = [o[key] for o in outputs if isinstance(o.get(key), pd.DataFrame) and not o[key].empty]
        if frames: pd.concat(frames, ignore_index=True).to_csv(CSV_DIR / fname, index=False)
    with open(OUTPUT_DIR / "run_settings.json", "w", encoding="utf-8") as f:
        json.dump(
            dict(
                INPUT_DIR=str(INPUT_DIR),
                OUTPUT_DIR=str(OUTPUT_DIR),
                SITE_REGION_CSV=SITE_REGION_CSV,
                TARGET_COLUMN=TARGET_COLUMN,
                INPUT_FEATURES=INPUT_FEATURES,
                FEATURE_SELECTION_MODEL="RandomForest",
                MIN_SITES=MIN_SITES,
                MISSINGNESS_LEVEL=MISSINGNESS_LEVEL,
                MISSINGNESS_REGIME=MISSINGNESS_REGIME,
                SEEDS=SEEDS,
                RF_SHAP_SEEDS=RF_SHAP_SEEDS,
                RF_MAX_TRAIN_ROWS=RF_MAX_TRAIN_ROWS,
                RF_N_ESTIMATORS=RF_N_ESTIMATORS,
                RF_MAX_DEPTH=RF_MAX_DEPTH,
                RF_MIN_SAMPLES_LEAF=RF_MIN_SAMPLES_LEAF,
                RF_MAX_FEATURES=RF_MAX_FEATURES,
                SHAP_SAMPLE_SIZE=SHAP_SAMPLE_SIZE,
                SHAP_APPROXIMATE=SHAP_APPROXIMATE,
                SAVE_COMBINED_SNAPSHOT=SAVE_COMBINED_SNAPSHOT,
                SHAP_AVAILABLE=SHAP_AVAILABLE,
                BACKWARD_TOLERANCE=BACKWARD_TOLERANCE,
                FINAL_TOLERANCE=FINAL_TOLERANCE,
                ROBUST_SELECTION=ROBUST_SELECTION,
                ROBUST_FINAL_TOLERANCE=ROBUST_FINAL_TOLERANCE,
                ROBUST_WEAK_THRESHOLD=ROBUST_WEAK_THRESHOLD,
                ROBUST_SHAP_THRESHOLD=ROBUST_SHAP_THRESHOLD,
                ROBUST_LOO_THRESHOLD=ROBUST_LOO_THRESHOLD,
                ROBUST_CORR_THRESHOLD=ROBUST_CORR_THRESHOLD,
                ROBUST_KEEP_MIN_FEATURES=ROBUST_KEEP_MIN_FEATURES,
            ),
            f,
            indent=2,
        )
    logging.info("Done. Final table: %s", final_csv_path)
    logging.info("Summary output folder: %s", SUMMARY_DIR)
    logging.info("Diagnostic plots folder: %s", PLOTS_DIR)
    logging.info("Diagnostic csv folder: %s", CSV_DIR)

if __name__ == "__main__":
    main()
    print("=== JOB FINISHED: Stage 2 regional RF-SHAP/RF subset selection ===", flush=True)
