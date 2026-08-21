"""
Configuration file for Spatial-Temporal Imputation Framework
Edit this file to customize your imputation settings

Author: Dr.  Masrur
Last Updated: 2026-01-21

✅ VERIFIED:   Complete configuration with: 
   - Spatial features (no data leakage)
   - Distance weighting (auto-loaded from Station_info.csv)
   - Temporal features (cyclical encoding)
   - Comprehensive evaluation support (IDW/Kriging in separate script)
"""

import os
import pandas as pd

# ============================================================================
# DIRECTORY SETTINGS
# ============================================================================

# Input directory containing CSV files for all sites
INPUT_DIRECTORY = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/input/Processed_Data_NSW_ALL"

# Base output directory for results
# main.py will build all other output trees (Model_output, Metrics, Imputed_Results, plots)
# from this root; config only declares the base and subdir names.
OUTPUT_DIRECTORY = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V25_final"

FINAL_RESULTS_SUBDIR = "FINAL_Results"
SAVE_PLACEHOLDER_VERIFICATION_FILES = False
# Backwards-compatible alias (existing code may still reference RAW_RESULTS_SUBDIR)
# Deprecated: prefer `FINAL_RESULTS_SUBDIR` in new code
RAW_RESULTS_SUBDIR = FINAL_RESULTS_SUBDIR  # deprecated alias for backwards compatibility

# Base folder (inside `OUTPUT_DIRECTORY`) for raw per-model outputs
# main.py uses:
#   MODEL_OUTPUT_ROOT = os.path.join(OUTPUT_DIRECTORY, MODEL_OUTPUT_SUBDIR)
MODEL_OUTPUT_SUBDIR = "Model_output"

# Station metadata file (contains coordinates)
# ✅ Used for distance weighting in spatial features
STATION_INFO_FILE = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQ_DATA/AquisNET_Data/Air Quality API Excel Power Query.xlsx"
MAX_SPATIAL_DISTANCE = 50  # km

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Models to run (list of module names)
# ✅ These models will use the spatial/temporal features configured below
MODELS_TO_RUN = [
    "AQUISTIL",  # ✅ Iterative LightGBM with spatial features
    "LightGBM",                  # ✅ Standard LightGBM
    # "BRITS",
    # "TFAutoencoder",
    # "SPIN",
]

# Standalone classical imputation methods
# ✅ Simple baseline for comparison
STANDALONE_MODELS = [
    "Mean",          # ✅ Mean imputation baseline
    "Median",        # ✅ Median imputation baseline
    "BaseLine",      # ✅ Custom baseline generator
    "Interpolation", # ✅ Time/linear interpolation
    # "KNN",          # ✅ K-Nearest Neighbors imputation
    # "MICE",         # Default_Mice model
    # "MICE-LGBM",    # ✅ MICE with LightGBM estimator
    # "MICE-RF",      # ✅ MICE with Random Forest estimator
    # "MICE-XGB",     # ✅ MICE with Extra Trees estimator
    # "MissForest",   # ✅ MissForest imputation
    # "SoftImpute",   # ✅ SoftImpute matrix completion
    # "XGBoost",      # ✅ MICE with XGBoost estimator
    # "MICE-KNN",     # ✅ MICE with KNN estimator
    # "RALGBM",       # ✅ Robust Adaptive LightGBM with spatial features
    # "SVT", 
    # "SPIN",
    # "GAIN",
    # "GRIN",
    # # "BRITS",
    # "DLV2"    
]

# Extend the primary models list
MODELS_TO_RUN += STANDALONE_MODELS

# MICE specific settings
MICE_MAX_ITER = 10              # Maximum iterations for iterative imputation
MICE_TOLERANCE = 0.01           # Convergence tolerance
MICE_RANDOM_STATE = 42          # Random seed for reproducibility

# Deep Learning specific settings
DL_EPOCHS = 100                 # Number of training epochs
DL_BATCH_SIZE = 32              # Batch size for training

# Monte Carlo specific settings
MC_N_ITERATIONS = 10            # Number of Monte Carlo iterations
MC_N_NEIGHBORS = 5              # Number of neighbors for KNN

# ============================================================================
# TARGET SETTINGS
# ============================================================================

# Target variable(s) to impute
TARGET_COLUMNS = ["HUMID"]  # Example: Impute these variables

# Target sites (empty = process all sites in INPUT_DIRECTORY)
# ✅ Use --all flag in main.py to process all available sites
# Set empty list to process ALL sites by default
# TARGET_SITES = []  # Process all sites in INPUT_DIRECTORY
TARGET_SITES = ["CHULLORA", "LIVERPOOL"]  # Limit to these sites

# ============================================================================
# BASE INPUT FEATURES (LOCAL SITE)
# ============================================================================

# ✅ These features are ALWAYS used from the target site itself
# ✅ The target variable is automatically excluded where needed to prevent leakage
INPUT_COLUMNS = [
    # Air Pollutants (from target site)
    # "CO",       # Carbon Monoxide
    # "NO",       # Nitric Oxide
    # "NOX",      # Nitrogen Oxides
    # "PM10",     # Particulate Matter (10 micrometers)
    # "SO2",      # Sulfur Dioxide
    
    # Meteorological Variables (from target site)
    # "HUMID",    # Humidity
    "TEMP",     # Temperature
    "WSP",      # Wind Speed
    "RAIN",     # Rainfall
    # "SOLAR",    # Solar Radiation
    "WDR",      # Wind Direction
    "WGU",      # Wind Gust
]

# ============================================================================
# SPATIAL FEATURES (FROM OTHER SITES)
# ============================================================================

# Master toggle: Use data from OTHER monitoring sites as features
# ✅ SAFE when using "pollutants_except_target" mode (recommended)
USE_SPATIAL_FEATURES = True

# -----------------------------
# SPATIAL FEATURE MODE
# -----------------------------
# ✅ CRITICAL SETTING TO PREVENT DATA LEAKAGE
#
# ✅ RECOMMENDED OPTION (CURRENTLY SELECTED):
#   "pollutants_except_target" - Load all pollutants EXCEPT target variable
#                                Example: If target=PM2.5, load CO, NO, NOX, PM10, SO2
#                                from other stations (NOT PM2.5)
#
# ⚠️  OTHER OPTIONS (USE WITH CAUTION):
#   "target_only"         - Only load target variable from other sites (leakage risk)
#   "all_pollutants"      - All pollutants including target (leakage risk)
#   "all_meteorological"  - Only weather variables from other sites (safe)
#   "all"                 - ALL INPUT_COLUMNS from other sites (may leak)
#   "custom"              - Use SPATIAL_FEATURE_COLUMNS list below
SPATIAL_FEATURE_MODE = "pollutants_except_target"

# Custom spatial features (only used if mode = "custom")
SPATIAL_FEATURE_COLUMNS = []

# Examples if using custom mode:
# SPATIAL_FEATURE_COLUMNS = ["CO", "NO", "NOX", "PM10"]  # ✅ Safe: no PM2.5
# SPATIAL_FEATURE_COLUMNS = ["PM2.5"]                    # ⚠️ Leakage risk! 

# -----------------------------
# SPATIAL CONSTRAINTS
# -----------------------------

# Maximum sites to include (0 = all available)
MAX_SPATIAL_SITES = 0

# Minimum sites required for spatial features
MIN_SPATIAL_SITES = 3

# ✅ DISTANCE WEIGHTING: Weight spatial features by inverse distance
# ✅ Requires SITE_COORDINATES (auto-loaded from Station_info.csv below)
USE_DISTANCE_WEIGHTING = True

# Maximum distance (km) for spatial features
MAX_SPATIAL_DISTANCE = 100

# ============================================================================
# TEMPORAL FEATURES (TIME PATTERNS)
# ============================================================================

# ✅ Extract time-based patterns from DateTime column
USE_TEMPORAL_FEATURES = True

# Configure which temporal features to include
TEMPORAL_FEATURES_CONFIG = {
    'hour':  True,               # Hour of day (0-23)
    'day': True,                 # Day of month (1-31)
    'month': True,               # Month of year (1-12)
    'day_of_week': True,         # Day of week (0=Monday, 6=Sunday)
    'day_of_year': True,         # Day of year (1-365)
    'week_of_year': True,        # Week of year (1-52)
    'is_weekend': False,         # Weekend indicator (optional)
    'season': False,             # Season indicator (optional)
    'cyclical_encoding': True,   # Adds: Hour_sin, Hour_cos, Month_sin, Month_cos
}

# ============================================================================
# LAGGED FEATURES (PAST VALUES)
# ============================================================================

# ✅ Currently DISABLED (handled by specific models internally if needed)
USE_LAGGED_FEATURES = False

# How many hours back to look
LAG_VALUES = [1, 2, 3, 6, 12, 24]

# Which variables to create lags for (empty = target only)
LAGGED_FEATURE_COLUMNS = []

# ============================================================================
# ROLLING WINDOW FEATURES (MOVING STATISTICS)
# ============================================================================

# ✅ Currently DISABLED (handled by some models internally)
USE_ROLLING_FEATURES = False

# Window sizes (in hours)
ROLLING_WINDOWS = [3, 6, 12, 24]

# Which statistics to compute
ROLLING_STATS = ['mean', 'std', 'min', 'max']

# Which variables to create rolling features for (empty = target only)
ROLLING_FEATURE_COLUMNS = []

# ============================================================================
# MISSINGNESS EVALUATION
# ============================================================================

# Levels of artificial missingness to evaluate (as fractions)
MISSINGNESS_LEVELS = [0.10, 0.20, 0.30, 0.50]  # 10%, 20%, 30%, 50%

# ============================================================================
# MISSINGNESS REGIME
# ============================================================================

# ✅ ALL REGIMES are tested automatically by main.py
MISSINGNESS_REGIMES = ['random', 'short_gap', 'medium_gap', 'long_gap', 'event']

# Default single regime (overridden by main.py when running all regimes)
MISSINGNESS_REGIME = None

# ============================================================================
# SITE SELECTION
# ============================================================================

EXCLUDE_SITES = []
EXCLUDE_FROM_SPATIAL = []

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

SORT_BY_HOUR = False
HANDLE_NEGATIVES = 'exclude'  # Options: 'exclude' or 'include'

INTERPOLATE_INPUTS = True
INTERPOLATION_METHOD = 'linear'  # 'linear', 'time', 'polynomial', 'spline'
MAX_INTERPOLATION_GAP = 0

# ============================================================================
# CUSTOM IMPUTATION STRATEGIES
# ============================================================================

CUSTOM_STRATEGIES = {}
APPLY_CUSTOM_TO_SPATIAL = False

# ============================================================================
# EVALUATION METRICS
# ============================================================================

METRICS_TO_CALCULATE = [
    "Nash-Sutcliffe Efficiency (NSE)",
    "Correlation Coefficient (R)",
    "Coefficient of Determination (R²)",
    "Mean Absolute Percentage Error (MAPE)",
    "Kling-Gupta Efficiency (KGE)",
    "Index of Agreement (WI)",
    "Root Mean Squared Error (RMSE)",
    "Mean Absolute Error (MAE)",
]

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

SAVE_IMPUTED_DATA = True              # Save full imputed datasets
SAVE_TARGET_COLUMN_DATA = True        # Save target column with missing type labels
SAVE_METRICS = True                   # Save evaluation metrics
SAVE_PLOTS = True                     # Generate evaluation plots

PLOT_TYPES = [
    'error_distribution',
    'residual_plot',
    'qq_plot',
    'scatterplot',
    'cdf_plot',
    'histogram',
    'correlation_heatmap',
]

PLOT_DPI = 300
SAVE_LOG = True
CREATE_SUMMARY_REPORT = True
AUTO_GENERATE_RESEARCH_PLOTS = False

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_FILE = "imputation_log.txt"
CONSOLE_VERBOSITY = "normal"  # "minimal", "normal", "detailed"

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

N_PROCESSES = 1
MEMORY_OPTIMIZATION = False
CHUNK_SIZE = 0
CACHE_SPATIAL_FEATURES = True

APPLY_MISSFOREST_PREIMPUTE = False
MISSFOREST_PREIMPUTE_N_ESTIMATORS = 100
MISSFOREST_PREIMPUTE_MAX_ITER = 5

GLOBAL_RANDOM_SEED = 42
VALIDATE_DATA = True
MIN_DATA_AVAILABILITY = 0.10

HANDLE_OUTLIERS = False
OUTLIER_METHOD = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
OUTLIER_THRESHOLD = 3.0

# ============================================================================
# EXPERIMENTAL FEATURES
# ============================================================================

USE_DL_SPATIAL_FEATURES = False
USE_ATTENTION_TEMPORAL = False
USE_ENSEMBLE = False
ENSEMBLE_MODELS = ["miceV2", "MissForest"]

# ============================================================================
# SITE METADATA (AUTO-LOADED FROM CSV)
# ============================================================================

def load_site_coordinates(station_info_file=STATION_INFO_FILE, fallback_excel=None):
    """
    Load site coordinates from Station_info.csv or from an Excel workbook.

    Behavior:
      - If `station_info_file` exists and is a CSV, try to read it.
      - Otherwise try an Excel workbook (fallback_excel or station_info_file if .xlsx).
      - Detect SiteName/Longitude/Latitude even when named like 'Column1.SiteName'.
      - Return dict with normalized site keys (uppercase): {'SITENAME': {'lat': x, 'lon': y}}
    """
    import os
    import pandas as pd
    import logging

    def _normalize_col_candidates(cols, targets):
        cols_l = [c for c in cols]
        for t in targets:
            for c in cols_l:
                if c.strip().lower() == t.strip().lower():
                    return c
        # try substring match
        for t in targets:
            for c in cols_l:
                if t.strip().lower() in c.strip().lower():
                    return c
        return None

    df = None
    if station_info_file and os.path.exists(station_info_file):
        if station_info_file.lower().endswith('.csv'):
            try:
                df = pd.read_csv(station_info_file)
            except Exception as e:
                logging.warning(f"Could not read station info CSV {station_info_file}: {e}")
                df = None
        else:
            df = None

    excel_to_try = None
    if df is None:
        if station_info_file and station_info_file.lower().endswith(('.xlsx', '.xls')) and os.path.exists(station_info_file):
            excel_to_try = station_info_file
        if fallback_excel and os.path.exists(fallback_excel):
            excel_to_try = fallback_excel

    if df is None and excel_to_try:
        try:
            df = pd.read_excel(excel_to_try, sheet_name=0,
                               engine='openpyxl' if excel_to_try.lower().endswith('.xlsx') else None)
            logging.info(f"Loaded station coordinates from Excel: {excel_to_try}")
        except Exception as e:
            logging.warning(f"Failed to read Excel station info {excel_to_try}: {e}")
            df = None

    if df is None:
        logging.warning(f"⚠️  Warning: Station info file not found or unreadable:  {station_info_file} (fallback_excel={fallback_excel})")
        logging.warning("   Distance weighting will be disabled.")
        return {}

    cols = df.columns.tolist()

    site_col = _normalize_col_candidates(cols, ['SiteName', 'Site', 'Station', 'Column1.SiteName', 'Column1.Site'])
    lon_col = _normalize_col_candidates(cols, ['Longitude', 'Lon', 'Long', 'Column1.Longitude'])
    lat_col = _normalize_col_candidates(cols, ['Latitude', 'Lat', 'Column1.Latitude'])

    if not site_col or not lon_col or not lat_col:
        logging.warning(f"Station info file missing expected columns. Available columns: {cols}")
        logging.warning(f"Detected: site_col={site_col}, lon_col={lon_col}, lat_col={lat_col}")
        return {}

    coordinates = {}
    for _, row in df.iterrows():
        try:
            site = str(row[site_col]).strip()
            if not site or site.lower() in ['nan', 'none']:
                continue
            lon_raw = row[lon_col]
            lat_raw = row[lat_col]
            try:
                lon = float(lon_raw)
                lat = float(lat_raw)
            except Exception:
                lon = pd.to_numeric(lon_raw, errors='coerce')
                lat = pd.to_numeric(lat_raw, errors='coerce')
                if pd.isna(lon) or pd.isna(lat):
                    continue
                lon = float(lon)
                lat = float(lat)
            coordinates[site.upper()] = {'lat': float(lat), 'lon': float(lon)}
        except Exception:
            continue

    logging.info(f"✅ Loaded coordinates for {len(coordinates)} sites from {station_info_file or excel_to_try}")
    return coordinates      


# ✅ AUTO-LOAD SITE COORDINATES FROM CSV
SITE_COORDINATES = load_site_coordinates()

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

# Cross-validation settings (currently not used)
PERFORM_CROSS_VALIDATION = False
CV_FOLDS = 5
CV_STRATEGY = 'kfold'  # Options: 'kfold', 'timeseries', 'spatial'

# ============================================================================
# NOTIFICATION SETTINGS (OPTIONAL)
# ============================================================================

SEND_EMAIL_NOTIFICATION = False

EMAIL_SETTINGS = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your_email@gmail.com',
    'sender_password': 'your_password',
    'recipient_email':  'recipient@email.com',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_spatial_columns():
    """
    Determine which columns to load from other sites as spatial features
    
    ✅ SAFE MODE:  Excludes target variable to prevent data leakage
    """
    if SPATIAL_FEATURE_COLUMNS:
        return SPATIAL_FEATURE_COLUMNS
    
    if SPATIAL_FEATURE_MODE == "target_only":
        # ⚠️  LEAKAGE RISK
        return TARGET_COLUMNS
    
    elif SPATIAL_FEATURE_MODE == "pollutants_except_target":
        # ✅ SAFE MODE (RECOMMENDED): All pollutants EXCEPT target
        all_pollutants = ["CO", "NO", "NOX", "PM10", "PM2.5", "SO2", "O3", "NO2"]
        return [p for p in all_pollutants if p not in TARGET_COLUMNS]
    
    elif SPATIAL_FEATURE_MODE == "all_pollutants":
        # ⚠️  LEAKAGE RISK:  Includes all pollutants (including target)
        pollutants = ["CO", "NO", "NOX", "PM10", "PM2.5", "SO2", "O3", "NO2"]
        return [col for col in pollutants if col in INPUT_COLUMNS or col in TARGET_COLUMNS]
    
    elif SPATIAL_FEATURE_MODE == "all_meteorological":
        # ✅ SAFE:  Only meteorological variables (no target)
        met_vars = ["HUMID", "TEMP", "WSP", "RAIN", "SOLAR", "WDR", "WGU",
                   "PRESSURE", "DEW_POINT", "VISIBILITY"]
        return [col for col in met_vars if col in INPUT_COLUMNS]
    
    elif SPATIAL_FEATURE_MODE == "all": 
        # ⚠️  LEAKAGE RISK:  All input columns + target columns
        return list(set(INPUT_COLUMNS + TARGET_COLUMNS))
    
    elif SPATIAL_FEATURE_MODE == "custom":
        # ✅/⚠️  Depends on SPATIAL_FEATURE_COLUMNS content
        return SPATIAL_FEATURE_COLUMNS if SPATIAL_FEATURE_COLUMNS else TARGET_COLUMNS
    
    else:
        # Default:  target only (not recommended)
        return TARGET_COLUMNS


def get_lagged_columns():
    """Determine which columns to create lagged features for"""
    if LAGGED_FEATURE_COLUMNS:
        return LAGGED_FEATURE_COLUMNS
    return TARGET_COLUMNS


def get_rolling_columns():
    """Determine which columns to create rolling features for"""
    if ROLLING_FEATURE_COLUMNS:
        return ROLLING_FEATURE_COLUMNS
    return TARGET_COLUMNS


def get_feature_summary():
    """
    Get summary of feature configuration
    """
    summary = {
        'base_features': len(INPUT_COLUMNS),
        'target_features': len(TARGET_COLUMNS),
        'spatial_enabled': USE_SPATIAL_FEATURES,
        'temporal_enabled': USE_TEMPORAL_FEATURES,
        'lagged_enabled': USE_LAGGED_FEATURES,
        'rolling_enabled': USE_ROLLING_FEATURES,
    }
    
    if USE_SPATIAL_FEATURES: 
        spatial_cols = get_spatial_columns()
        summary['spatial_variables'] = len(spatial_cols)
        summary['spatial_columns'] = spatial_cols
        summary['spatial_mode'] = SPATIAL_FEATURE_MODE
    
    if USE_TEMPORAL_FEATURES:
        temporal_count = sum(1 for v in TEMPORAL_FEATURES_CONFIG.values() if v)
        if TEMPORAL_FEATURES_CONFIG.get('cyclical_encoding', False):
            temporal_count += 6  # sin/cos encodings
        summary['temporal_features'] = temporal_count
    
    if USE_LAGGED_FEATURES:
        lag_cols = get_lagged_columns()
        summary['lagged_features'] = len(lag_cols) * len(LAG_VALUES)
        summary['lag_columns'] = lag_cols
    
    if USE_ROLLING_FEATURES:
        roll_cols = get_rolling_columns()
        summary['rolling_features'] = len(roll_cols) * len(ROLLING_WINDOWS) * len(ROLLING_STATS)
        summary['rolling_columns'] = roll_cols
    
    return summary


def estimate_total_features(num_spatial_sites=50):
    """
    Estimate total number of features that will be generated
    """
    total = 0
    
    total += len(INPUT_COLUMNS)
    
    if USE_SPATIAL_FEATURES: 
        spatial_vars = len(get_spatial_columns())
        total += spatial_vars * num_spatial_sites
    
    if USE_TEMPORAL_FEATURES: 
        temporal_count = sum(1 for v in TEMPORAL_FEATURES_CONFIG.values() if v)
        if TEMPORAL_FEATURES_CONFIG.get('cyclical_encoding', False):
            temporal_count += 6
        total += temporal_count
    
    if USE_LAGGED_FEATURES:
        total += len(get_lagged_columns()) * len(LAG_VALUES)
    
    if USE_ROLLING_FEATURES: 
        total += len(get_rolling_columns()) * len(ROLLING_WINDOWS) * len(ROLLING_STATS)
    
    return total


def get_config_summary():
    """
    Get human-readable configuration summary
    """
    summary = []
    summary.append("="*80)
    summary.append("SPATIAL-TEMPORAL IMPUTATION CONFIGURATION")
    summary.append("="*80)
    
    summary.append(f"\n📁 Directories:")
    summary.append(f"   Input: {INPUT_DIRECTORY}")
    summary.append(f"   Output: {OUTPUT_DIRECTORY}")
    
    summary.append(f"\n🤖 Models:  {len(MODELS_TO_RUN)}")
    for model in MODELS_TO_RUN:
        summary.append(f"   • {model}")
    
    summary.append(f"\n🎯 Target Variables: {', '.join(TARGET_COLUMNS)}")
    
    summary.append(f"\n📊 Feature Configuration:")
    summary.append(f"   Base Input Features: {len(INPUT_COLUMNS)}")
    summary.append(f"   Features:  {', '.join(INPUT_COLUMNS[: 5])}" +
                   (f" ...  and {len(INPUT_COLUMNS)-5} more" if len(INPUT_COLUMNS) > 5 else ""))
    
    if USE_SPATIAL_FEATURES: 
        spatial_cols = get_spatial_columns()
        summary.append(f"\n   ✅ Spatial Features:  ENABLED")
        summary.append(f"      Mode: {SPATIAL_FEATURE_MODE}")
        summary.append(f"      Variables from other sites: {', '.join(spatial_cols)}")
        summary.append(f"      Max sites: {MAX_SPATIAL_SITES if MAX_SPATIAL_SITES > 0 else 'All'}")
        summary.append(f"      Distance weighting: {'YES' if USE_DISTANCE_WEIGHTING else 'NO'}")
        if USE_DISTANCE_WEIGHTING:
            summary.append(f"      Sites with coordinates: {len(SITE_COORDINATES)}")
    else:
        summary.append(f"\n   ❌ Spatial Features: DISABLED")
    
    if USE_TEMPORAL_FEATURES:
        summary.append(f"\n   ✅ Temporal Features:  ENABLED")
        enabled_temp = [k for k, v in TEMPORAL_FEATURES_CONFIG.items() if v]
        summary.append(f"      Features: {', '.join(enabled_temp)}")
    else:
        summary.append(f"\n   ❌ Temporal Features: DISABLED")
    
    if USE_LAGGED_FEATURES:
        lag_cols = get_lagged_columns()
        summary.append(f"\n   ✅ Lagged Features: ENABLED")
        summary.append(f"      Variables: {', '.join(lag_cols)}")
        summary.append(f"      Lags (hours): {LAG_VALUES}")
    else:
        summary.append(f"\n   ❌ Lagged Features: DISABLED")
    
    if USE_ROLLING_FEATURES:
        roll_cols = get_rolling_columns()
        summary.append(f"\n   ✅ Rolling Features:  ENABLED")
        summary.append(f"      Variables: {', '.join(roll_cols)}")
        summary.append(f"      Windows (hours): {ROLLING_WINDOWS}")
        summary.append(f"      Statistics: {ROLLING_STATS}")
    else:
        summary.append(f"\n   ❌ Rolling Features: DISABLED")
    
    summary.append(f"\n📉 Missingness:")
    summary.append(f"   Levels: {[int(m*100) for m in MISSINGNESS_LEVELS]}%")
    summary.append(f"   Regimes: {', '.join(MISSINGNESS_REGIMES)}")
    
    if TARGET_SITES:
        summary.append(f"\n🗺️  Target Sites ({len(TARGET_SITES)}): {', '.join(TARGET_SITES)}")
    else:
        summary.append(f"\n🗺️  Target Sites: All available")
    
    summary.append("\n" + "="*80)
    
    estimated_features = estimate_total_features()
    summary.append(f"📈 Estimated Total Features: ~{estimated_features}")
    summary.append(f"   (assumes ~50 spatial sites available)")
    
    summary.append("="*80)
    
    return "\n".join(summary)


def validate_config():
    """
    Validate configuration settings
    """
    errors = []
    warnings = []
    
    # Check directories
    if not INPUT_DIRECTORY: 
        errors.append("INPUT_DIRECTORY is not set")
    elif not os.path.exists(INPUT_DIRECTORY):
        warnings.append(f"INPUT_DIRECTORY does not exist: {INPUT_DIRECTORY}")
    
    if not OUTPUT_DIRECTORY:
        errors.append("OUTPUT_DIRECTORY is not set")
    
    # Check models
    if not MODELS_TO_RUN: 
        errors.append("No models specified in MODELS_TO_RUN")
    
    # Check target columns
    if not TARGET_COLUMNS:
        errors.append("No target columns specified in TARGET_COLUMNS")
    
    # Check input columns
    if not INPUT_COLUMNS:
        errors.append("No input columns specified in INPUT_COLUMNS")
    
    # Check missingness levels
    if not MISSINGNESS_LEVELS:
        errors.append("No missingness levels specified")
    
    for level in MISSINGNESS_LEVELS:
        if not (0 < level < 1):
            errors.append(f"Invalid missingness level: {level} (must be between 0 and 1)")
    
    # ✅ CHECK SPATIAL CONFIGURATION FOR LEAKAGE
    if USE_SPATIAL_FEATURES: 
        valid_modes = ["target_only", "all_pollutants", "all_meteorological", "all", "custom", "pollutants_except_target"]
        if SPATIAL_FEATURE_MODE not in valid_modes: 
            errors.append(f"Invalid SPATIAL_FEATURE_MODE: '{SPATIAL_FEATURE_MODE}'.  Must be one of: {valid_modes}")
        
        # ⚠️  WARN IF LEAKAGE MODE IS SELECTED
        if SPATIAL_FEATURE_MODE in ["target_only", "all_pollutants", "all"]:
            warnings.append(
                f"⚠️  SPATIAL_FEATURE_MODE = '{SPATIAL_FEATURE_MODE}' may cause DATA LEAKAGE!\n"
                f"   This will include the target pollutant from other stations.\n"
                f"   RECOMMENDED: Set SPATIAL_FEATURE_MODE = 'pollutants_except_target'"
            )
        
        if SPATIAL_FEATURE_MODE == "custom" and not SPATIAL_FEATURE_COLUMNS:
            warnings.append("SPATIAL_FEATURE_MODE is 'custom' but SPATIAL_FEATURE_COLUMNS is empty.  Will default to target_only.")
        
        # ✅ CHECK IF CUSTOM LIST CONTAINS TARGET VARIABLE
        if SPATIAL_FEATURE_MODE == "custom":
            overlap = set(SPATIAL_FEATURE_COLUMNS) & set(TARGET_COLUMNS)
            if overlap:
                warnings.append(
                    f"⚠️  DATA LEAKAGE:  SPATIAL_FEATURE_COLUMNS contains target variable {overlap}\n"
                    f"   This will cause perfect predictions!"
                )
    
    # ✅ CHECK DISTANCE WEIGHTING CONFIGURATION
    if USE_DISTANCE_WEIGHTING:
        if not SITE_COORDINATES:
            warnings.append(
                "USE_DISTANCE_WEIGHTING is True but SITE_COORDINATES is empty.\n"
                "   Distance weighting will be disabled.  Check if Station_info.csv was loaded correctly."
            )
        elif len(SITE_COORDINATES) < 2:
            warnings.append(
                f"USE_DISTANCE_WEIGHTING is True but only {len(SITE_COORDINATES)} site(s) have coordinates.\n"
                "   Add more sites to Station_info.csv for effective distance weighting."
            )
    
    # Feature warnings
    if not USE_SPATIAL_FEATURES and not USE_TEMPORAL_FEATURES and not USE_LAGGED_FEATURES and not USE_ROLLING_FEATURES:
        warnings.append("All feature engineering options are disabled. Only base INPUT_COLUMNS will be used.")
    
    total_features = estimate_total_features()
    if total_features > 500:
        warnings.append(f"Estimated feature count is very high (~{total_features}). This may cause memory issues or slow training.")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def print_validation_results():
    """
    Print validation results in a formatted way
    """
    is_valid, errors, warnings = validate_config()
    
    print("\n" + "="*80)
    print("CONFIGURATION VALIDATION")
    print("="*80)
    
    if errors:
        print("\n❌ ERRORS:")
        for error in errors: 
            print(f"   • {error}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if is_valid:
        print("\n✅ Configuration is valid!")
    else:
        print("\n❌ Configuration has errors.  Please fix them before running.")
    
    print("="*80 + "\n")
    
    return is_valid


# ============================================================================
# DATA LEAKAGE PROTECTION
# ============================================================================

def validate_no_target_leakage():
    """
    Automatic check:  Ensure target variable is NOT in spatial features
    """
    if USE_SPATIAL_FEATURES: 
        spatial_cols = get_spatial_columns()
        target_cols = TARGET_COLUMNS
        
        # Check for overlap
        overlap = set(spatial_cols) & set(target_cols)
        
        if overlap:
            raise ValueError(
                f"⚠️  DATA LEAKAGE DETECTED!\n"
                f"Target variable {overlap} found in spatial features.\n"
                f"This will cause perfect predictions (R²=1.0) due to data leakage.\n\n"
                f"FIX: Set SPATIAL_FEATURE_MODE = 'pollutants_except_target'\n"
                f"OR: Remove {overlap} from SPATIAL_FEATURE_COLUMNS"
            )
    
    print("✅ Data leakage check passed:  Target variable NOT in spatial features")
    return True


# ✅ Run validation when config is imported (but don't stop execution for warnings)
if __name__ != "__main__":
    try:
        validate_no_target_leakage()
    except ValueError as e:
        import logging
        logging.warning(str(e))


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

def apply_preset(preset_name):
    """
    Apply a pre-configured preset
    
    Available presets:
        - "fast": Minimal features for quick testing
        - "balanced": Recommended for most use cases (SAFE - no leakage)
        - "comprehensive": All features enabled (SAFE - no leakage)
        - "spatial_only": Only spatial features (SAFE - no leakage)
        - "temporal_only": Only temporal features
    """
    global USE_SPATIAL_FEATURES, USE_TEMPORAL_FEATURES, USE_LAGGED_FEATURES, USE_ROLLING_FEATURES
    global SPATIAL_FEATURE_MODE, MODELS_TO_RUN
    
    if preset_name == "fast":
        USE_SPATIAL_FEATURES = False
        USE_TEMPORAL_FEATURES = True
        USE_LAGGED_FEATURES = False
        USE_ROLLING_FEATURES = False
        MODELS_TO_RUN = ["Standalone_Mean"]
        print("Applied 'fast' preset:  Minimal features, single model")
    
    elif preset_name == "balanced":
        USE_SPATIAL_FEATURES = True
        USE_TEMPORAL_FEATURES = True
        USE_LAGGED_FEATURES = False
        USE_ROLLING_FEATURES = False
        SPATIAL_FEATURE_MODE = "pollutants_except_target"  # ✅ SAFE
        print("Applied 'balanced' preset:  Spatial + temporal features (NO LEAKAGE)")
    
    elif preset_name == "comprehensive": 
        USE_SPATIAL_FEATURES = True
        USE_TEMPORAL_FEATURES = True
        USE_LAGGED_FEATURES = True
        USE_ROLLING_FEATURES = True
        SPATIAL_FEATURE_MODE = "pollutants_except_target"  # ✅ SAFE
        print("Applied 'comprehensive' preset:  All features enabled (NO LEAKAGE)")
    
    elif preset_name == "spatial_only": 
        USE_SPATIAL_FEATURES = True
        USE_TEMPORAL_FEATURES = False
        USE_LAGGED_FEATURES = False
        USE_ROLLING_FEATURES = False
        SPATIAL_FEATURE_MODE = "pollutants_except_target"  # ✅ SAFE
        print("Applied 'spatial_only' preset: Only spatial features (NO LEAKAGE)")
    
    elif preset_name == "temporal_only":
        USE_SPATIAL_FEATURES = False
        USE_TEMPORAL_FEATURES = True
        USE_LAGGED_FEATURES = False
        USE_ROLLING_FEATURES = False
        print("Applied 'temporal_only' preset: Only temporal features")
    
    else:
        print(f"Unknown preset: {preset_name}")
        print("Available presets: fast, balanced, comprehensive, spatial_only, temporal_only")


# ============================================================================
# MAIN (for testing configuration)
# ============================================================================

if __name__ == "__main__": 
    import sys
    
    # Check if preset requested
    if len(sys.argv) > 1:
        preset = sys.argv[1]
        apply_preset(preset)
    
    # Print configuration summary
    print(get_config_summary())
    
    # Validate configuration
    print_validation_results()
    
    # Print feature details
    print("\n" + "="*80)
    print("FEATURE DETAILS")
    print("="*80)
    feat_summary = get_feature_summary()
    for key, value in feat_summary.items():
        print(f"{key}: {value}")
    print("="*80)
    
    # ✅ NOTE: IDW and Kriging baselines are in comprehensive_model_evaluation.py
    print("\n" + "="*80)
    print("NOTE: IDW AND KRIGING BASELINES")
    print("="*80)
    print("IDW and Kriging are NOT configured here.")
    print("They are implemented in comprehensive_model_evaluation.py as:")
    print("  - Configuration 4: Spatial_IDW (Local + Temporal + IDW feature)")
    print("  - Configuration 5: Spatial_Kriging (Local + Temporal + Kriging feature)")
    print("\nThese are tested automatically when you run:")
    print("  python main.py --all")
    print("="*80)
