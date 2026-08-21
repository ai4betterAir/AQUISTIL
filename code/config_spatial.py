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
import json
import logging
import pandas as pd

# ============================================================================
# DIRECTORY SETTINGS
# ============================================================================

# Input directory containing CSV files for all sites
INPUT_DIRECTORY = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/input/Processed_Data_NSW_ALL"

# Use already-downloaded wide regional inputs from nowcasting pipeline.
# When enabled, `AI_Imputation/code/main.py` will materialize per-site CSVs
# on-the-fly from the wide regional files under NOWCASTING_WIDE_INPUT_DIR.
USE_WIDE_NOWCASTING_INPUTS = True

# Default wide regional API-input directory used by `code/main.py`.
# These files are refreshed in-place from the NSW AQ API when live-update is enabled.
WIDE_API_INPUT_DIRECTORY = os.environ.get(
    "AQUISTIL_WIDE_INPUT_DIR",
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/API_Input/Inputs",
)

# Optional: auto-download/append the missing tail "up to now" for the wide
# regional inputs before running the imputation pipeline. This can be slow
# depending on region size and API responsiveness.
# Offline by default for reproducible model evaluation. Pass
# --refresh-api-inputs explicitly when the wide inputs should be updated.
AUTO_UPDATE_WIDE_INPUTS = False

# Base output directory for results
# main.py will build all other output trees (Model_output, Metrics, Imputed_Results, plots)
# from this root; config only declares the base and subdir names.
FROZEN_OUTPUT_DIRECTORY = (
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Final_Frozen_2026_08_17"
)
DEVELOPMENT_ABLATION_OUTPUT_DIRECTORY = (
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Imputation_Results"
)
EXPERIMENT_MODE = os.environ.get(
    "AQUISTIL_EXPERIMENT_MODE", "frozen_validation"
).strip().lower()
if EXPERIMENT_MODE not in {"frozen_validation", "development_ablation"}:
    raise ValueError(
        "AQUISTIL_EXPERIMENT_MODE must be frozen_validation or development_ablation"
    )
DEFAULT_OUTPUT_DIRECTORY = (
    DEVELOPMENT_ABLATION_OUTPUT_DIRECTORY
    if EXPERIMENT_MODE == "development_ablation"
    else FROZEN_OUTPUT_DIRECTORY
)
OUTPUT_DIRECTORY = os.environ.get(
    "AQUISTIL_IMPUTATION_OUTPUT_DIR",
    DEFAULT_OUTPUT_DIRECTORY,
)

# Publication validation protocol. The runtime rejects overrides that would mix
# development regions into this held-out evaluation.
FROZEN_RELEASE_TAG = "aquistil-frozen-heldout-2026-08-17-r4"
FROZEN_VALIDATION_MODE = EXPERIMENT_MODE == "frozen_validation"
FROZEN_STAGE3_FEATURE_FILES = {
    "PM10": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "frozen_inputs",
        "aquistil_heldout_2026_08_14",
        "stage3_PM10.csv",
    ),
    "PM2.5": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "frozen_inputs",
        "aquistil_heldout_2026_08_14",
        "stage3_PM2.5.csv",
    ),
}
DEVELOPMENT_ABLATION_STAGE3_FEATURE_FILES = {
    "PM10": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "frozen_inputs",
        "aquistil_development_ablation_2026_08_16",
        "stage3_PM10.csv",
    ),
    "PM2.5": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "frozen_inputs",
        "aquistil_development_ablation_2026_08_16",
        "stage3_PM2.5.csv",
    ),
}
AQUISTIL_ABLATION_METRICS_DIRECTORY = os.environ.get(
    "AQUISTIL_ABLATION_METRICS_DIR",
    os.path.join(DEVELOPMENT_ABLATION_OUTPUT_DIRECTORY, "Metrics copy"),
)

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
STATION_INFO_SHEET = "SiteDetails"
REGION_SITE_JSON_FILE = os.path.join(OUTPUT_DIRECTORY, "region_site_mapping.json")
EXCLUDED_REGION_KEYWORDS = ["offline", "LLS"]
MAX_SPATIAL_DISTANCE = 50  # km

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Models to run (list of module names)
# ✅ These models will use the spatial/temporal features configured below
PAPER_BASELINE_MODELS = ["LightGBM"]

MODELS_TO_RUN = [
    # Robust regime-aware stack. The base experts are configured below and do
    # not need to be listed separately unless they should also be benchmarked.
    # "AQUISTIL_R",
    # Proposed model
    "AQUISTIL",
    # # "MICE_AQUISTIL",
    # "MICE_PosteriorRefined",
    # "Surrogate_AQUISTIL",
    # "Surrogate_AQUISTIL_Selector",
] + PAPER_BASELINE_MODELS

# Diagnostic AQUISTIL ablations for contiguous-gap failure analysis. Keep this
# disabled for the frozen AQUISTIL_Current/AQUISTIL/LightGBM comparison.
RUN_AQUISTIL_ABLATIONS = EXPERIMENT_MODE == "development_ablation"
AQUISTIL_ABLATION_MODELS = [
    "AQUISTIL_NoHistory",
    "AQUISTIL_NoHistoryNoEvent",
    "AQUISTIL_NoFFill",
    "AQUISTIL_NoAdaptive",
    "AQUISTIL_ExogenousOnly",
    "AQUISTIL_NoAQUISTILFeatures",
]

if RUN_AQUISTIL_ABLATIONS:
    MODELS_TO_RUN = ["AQUISTIL"] + AQUISTIL_ABLATION_MODELS + PAPER_BASELINE_MODELS

# Missingness-topology-aware AQUISTIL. The provisional 2-hour threshold keeps
# isolated missing observations on the history expert and routes sustained
# outages to the internal no-history gap expert. Freeze this only after testing
# the global candidates below on development regions.
AQUISTIL_REGIME_AWARE_ENABLED = True
AQUISTIL_GAP_EXPERT_MIN_RUN_LENGTH = 2
FROZEN_GAP_EXPERT_MIN_RUN_LENGTH = 2
AQUISTIL_GAP_EXPERT_THRESHOLD_CANDIDATES = (2, 3, 6, 12, 24)
AQUISTIL_GAP_BOUNDARY_FEATURES_ENABLED = False
# Historical/batch imputation can use the observed boundary after a gap. Set
# False for causal/real-time use; future-derived gap features then remain NaN.
AQUISTIL_GAP_ALLOW_FUTURE_CONTEXT = True
AQUISTIL_GAP_MASKED_TRAINING_ENABLED = False

# AQUISTIL-R robust stacking controls. Three regime-matched validation folds
# require 3 x len(BASE_MODELS) expert fits plus one final fit per expert. Reduce
# VALIDATION_FOLDS to 1 for a quick smoke run; use 0 to run from the priors only.
# AQUISTIL_R_BASE_MODELS = (
#     "AQUISTIL",
#     # "MICE_AQUISTIL",
#     # "MICE-BR",
#     # "AQUISTIL_A",
#     # "LightGBM",
# )
AQUISTIL_R_VALIDATION_FOLDS = 3
AQUISTIL_R_VALIDATION_FRACTION = 0.12
AQUISTIL_R_MIN_VALIDATION_POINTS = 48
AQUISTIL_R_MAX_VALIDATION_POINTS = 500
AQUISTIL_R_MIN_TRAINING_POINTS = 50
AQUISTIL_R_MIN_ROUTE_ROWS = 24
AQUISTIL_R_MIN_EXPERT_COVERAGE = 0.80
AQUISTIL_R_PRIOR_STRENGTH = 40.0
AQUISTIL_R_REGULARIZATION = 0.03
AQUISTIL_R_HUBER_DELTA = 1.5
AQUISTIL_R_EVENT_QUANTILE = 0.90

# When experts strongly disagree, blend toward their weighted median. Event
# predictions use a smaller pull so a valid high-concentration expert survives.
AQUISTIL_R_DISAGREEMENT_BLEND = 0.35
AQUISTIL_R_EVENT_DISAGREEMENT_BLEND = 0.15
AQUISTIL_R_DISAGREEMENT_START = 0.35
AQUISTIL_R_DISAGREEMENT_FULL = 1.25

# Conservative physical bounds are learned per site. Explicit bounds can be
# supplied here when a target has known measurement limits.
AQUISTIL_R_LOWER_QUANTILE = 0.001
AQUISTIL_R_UPPER_QUANTILE = 0.999
AQUISTIL_R_BOUND_IQR_FACTOR = 3.0
AQUISTIL_R_LOWER_BOUND = None
AQUISTIL_R_UPPER_BOUND = None
AQUISTIL_R_ADD_DIAGNOSTICS = True

# Active AQUISTIL event specialist. It is used for event experiments and real
# (unlabelled) missingness only; synthetic random/gap experiments use the common
# backbone so false event detections cannot dominate their point predictions.
AQUISTIL_EVENT_REFINEMENT_ENABLED = True
AQUISTIL_EVENT_REFINEMENT_REGIMES = ("event", "")
AQUISTIL_EVENT_PERCENTILE = 0.90
AQUISTIL_EVENT_PROBABILITY_THRESHOLD = 0.55
AQUISTIL_EVENT_MAX_BLEND = 0.70
AQUISTIL_EVENT_CAP_QUANTILE = 0.98

# AQUISTIL adaptive gap guardrails. These rules only alter pollutant/regime
# combinations where the pooled metrics showed weaker performance; all other
# AQUISTIL predictions keep the normal LightGBM + event-refinement output.
AQUISTIL_ADAPTIVE_GAP_GUARDRAILS_ENABLED = True
AQUISTIL_ADAPTIVE_MIN_SPATIAL_CONTRIBUTORS = 2
AQUISTIL_ADAPTIVE_BLEND_RULES = {
    # PM10 long/medium gaps had large outliers, especially Sydney North-west.
    # Pull those predictions toward the adaptive spatial/hourly reference.
    "PM10": {
        "medium_gap": 0.25,
        "long_gap": 0.40,
        "gap_medium": 0.25,
        "gap_long": 0.40,
        "gap_extreme": 0.50,
    },
    # PM2.5 medium/long gaps were weaker than BRITS/LightGBM. Use a softer
    # spatial pull so random/event performance is preserved.
    "PM2.5": {
        "medium_gap": 0.20,
        "long_gap": 0.30,
        "gap_medium": 0.20,
        "gap_long": 0.30,
        "gap_extreme": 0.35,
    },
}
AQUISTIL_ADAPTIVE_UNCERTAINTY_EXTRA_BLEND = {
    "PM2.5": 0.10,
}
AQUISTIL_ADAPTIVE_UNCERTAINTY_QUANTILE = 0.75
AQUISTIL_FINAL_CLIP_TARGETS = {"PM10": True}
AQUISTIL_ADAPTIVE_LOWER_QUANTILE = 0.01
AQUISTIL_ADAPTIVE_UPPER_QUANTILE = 0.995
AQUISTIL_ADAPTIVE_IQR_FACTOR = 1.5
AQUISTIL_ADAPTIVE_NONNEGATIVE = True

# OZONE was strong for event/random/long gaps but weaker for short/medium gaps.
# In those regimes, keep the event features but reduce the final event-expert
# correction so the backbone prediction dominates.
AQUISTIL_OZONE_SHORT_MEDIUM_EVENT_MAX_BLEND = 0.20

# Additional candidate implementations.  ``main.py`` appends these to the
# primary AQUISTIL/LightGBM pair above and evaluates every model with the same
# data, artificial masks, missingness levels and seeds.  The compact paper
# comparison remains controlled independently by ``COMPARISON_MODELS``.
STANDALONE_MODELS = [
    # "XGBoost",
    # "MICE",
    # "MICE-KNN",
    # "MissForest",
    # "KNN",
    # "Mean",
    # "Median",
    # "Mode",
    # "BaseLine",
    # "interpolation",
]

# Extend the primary models list
MODELS_TO_RUN += STANDALONE_MODELS

# Include every selected primary and standalone model in the wide comparison
# CSV. Keep the union explicit and de-duplicated so comparison output remains
# complete even if list-extension logic changes.
COMPARISON_MODELS = list(dict.fromkeys(MODELS_TO_RUN))

# ============================================================================
# MISSINGNESS EVALUATION
# ============================================================================

# Levels of artificial missingness in the frozen held-out evaluation.
MISSINGNESS_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.5, 0.6]

# ============================================================================
# MISSINGNESS REGIME
# ============================================================================

# Event-specialist model-selection run.  Restore the full list below for the
# final all-regime evaluation after selecting the winning candidate:
# ['random', 'short_gap', 'medium_gap', 'long_gap', 'event']
MISSINGNESS_REGIMES = ['random', 'short_gap', 'medium_gap', 'long_gap', 'event']
# MISSINGNESS_REGIMES = ['medium_gap', 'long_gap', 'event']

# Default single regime (overridden by main.py when running all regimes)
MISSINGNESS_REGIME = None

# ---------------------------------------------------------------------------
# REGION SCOPE
# ---------------------------------------------------------------------------
# AVAILABLE_REGIONS is the full experiment menu. Keep every region here that
# is valid for the study and has a matching wide input file.
# Use [] to allow every valid region from SiteDetails.
AVAILABLE_REGIONS = [
    "Sydney North-west",
    "Sydney South-west",
    "Upper Hunter",
    "Sydney East",
    "Northern Tablelands",
    "Southern Tablelands",
    "Lower Hunter",
    "Mid-North Coast",
    "Central Tablelands",
    "Illawarra",
    "Central Coast",
]

# These regions were used in the NoHistory ablation and/or routing development.
# They are development data and must not enter the publication validation set.
DEVELOPMENT_REGIONS = [
    "Central Coast",
    "Central Tablelands",
    "Sydney North-west",
    "Sydney South-west",
]

# Remaining configured study regions with matching regional-wide input files.
# No model or routing decisions may be changed after inspecting these results.
HELD_OUT_VALIDATION_REGIONS = [
    "Lower Hunter",
    "Northern Tablelands",
    "Southern Tablelands",
    "Sydney East",
    "Upper Hunter",
]

# These configured regions are also held out, but no matching wide input exists
# in API_Input/Inputs at freeze time, so they cannot be evaluated in this run.
HELD_OUT_REGIONS_WITHOUT_INPUTS = [
    "Mid-North Coast",
    "Illawarra",
]

# SELECTED_REGIONS is the active run list.
SELECTED_REGIONS = (
    list(DEVELOPMENT_REGIONS)
    if EXPERIMENT_MODE == "development_ablation"
    else list(HELD_OUT_VALIDATION_REGIONS)
)

# AVAILABLE_SITES is normally derived from AVAILABLE_REGIONS. Leave [] unless
# you need to restrict the region-derived station menu manually.
AVAILABLE_SITES = []

# SELECTED_SITES overrides region-derived stations. Use [] for the default
# sites discovered from SELECTED_REGIONS; otherwise provide explicit site names.
SELECTED_SITES = []

# Backward-compatible names consumed by main.py and helper functions.
TARGET_REGIONS = AVAILABLE_REGIONS
TARGET_SITES = AVAILABLE_SITES
SELECT_TARGET_REGIONS = SELECTED_REGIONS
SELECT_TARGET_SITES = SELECTED_SITES
# Optional dev-mode limits for quicker runs.

# Example: set MAX_MODELS_TO_RUN = 2 to only run the first 2 models.
MAX_MODELS_TO_RUN = 0

# MICE specific settings
MICE_MAX_ITER = 10              # Maximum iterations for iterative imputation
MICE_TOLERANCE = 0.01           # Convergence tolerance
MICE_RANDOM_STATE = 42          # Random seed for reproducibility

# Monte Carlo specific settings
MC_N_ITERATIONS = 10            # Number of Monte Carlo iterations
MC_N_NEIGHBORS = 5              # Number of neighbors for KNN

# ============================================================================
# TARGET SETTINGS
# ============================================================================

# Target variable(s) to impute
TARGET_COLUMNS = ["PM10", "PM2.5"]  # Example: Impute these variables
# TARGET_COLUMNS = ["PM10", "PM2.5", "NO2", "CO", "OZONE", "NOX", "NO"]  # Example: Impute these variables]
# TARGET_COLUMNS = ["PM10", "PM2.5", "CO", "OZONE", "NO"]  # Example: Impute these variables]

# "NOX",      # Nitrogen Oxides
#     "PM10",     # Particulate Matter (10 micrometers)
#     "PM2.5",    # Particulate Matter (2.5 micrometers)
#     "SO2",
# ---------------------------------------------------------------------------
# BEST-PREDICTORS INPUTS (region + target specific)
# ---------------------------------------------------------------------------
# Optional legacy fallback. Keep disabled unless
# BestPredictors_ByRegionTarget.json has been generated.
USE_BEST_PREDICTORS_JSON_INPUTS = False

# Use the winning region/target configuration produced by FeatureStats Stage 3.
# When a matching row exists, main.py uses its exact Feature_List in preference
# to the older best-predictors JSON and INPUT_COLUMNS fallbacks.
USE_PROGRESSIVE_BEST_FEATURES = True
# Treat Stage 3 Feature_List as the complete model input contract.  Models that
# support this flag must not append their own lag/rolling/geometry features.
# main.py enables this only when it actually resolves a matching Stage 3 row.
STRICT_PROGRESSIVE_FEATURE_LIST = False
PROGRESSIVE_BEST_FEATURES_CSV = os.environ.get(
    "PROGRESSIVE_BEST_FEATURES_CSV",
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection/03Regional_Selected_Feature_Progressive_Evaluation/summary_outputs/regional_progressive_best_configuration_by_region.csv",
)

# Train one pooled model per region and report both per-site and regional
# metrics using an equal artificial-missing count at every study site.
REGIONAL_POOLED_MODE = True
REGIONAL_EVALUATION_SEEDS = [13, 29, 42, 77, 101, 137, 211, 307, 401, 503]
# Run different regional target tasks concurrently. Native model threads are
# divided across these workers when MODEL_N_JOBS is 0.
TARGET_PARALLEL_WORKERS = 10
MODEL_N_JOBS = 0
# Keep the scheduler/combined framework log and also route each parallel
# target's model messages to Logs/By_Target/<target>.log.
SEPARATE_TARGET_LOGS = True
TARGET_LOG_SUBDIR = os.path.join("Logs", "By_Target")
BEST_PREDICTORS_JSON = os.environ.get(
    "BEST_PREDICTORS_JSON",
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/BestPredictors_ByRegionTarget.json",
)

# If predictors are missing for a (region, target) pair, fallback to INPUT_COLUMNS.
FALLBACK_TO_INPUT_COLUMNS_WHEN_MISSING = True

# ============================================================================
# BASE INPUT FEATURES (LOCAL SITE)
# ============================================================================

# Canonical local candidates shared by FeatureStats Stages 0 and 2. Stage 0
# evaluates them individually; Stage 2 searches their best regional subset.
# The active target is always excluded by the analysis code. SOLAR is
# intentionally not part of this contract.
LOCAL_ANALYSIS_INPUTS = [
    "CO", "HUMID", "NEPH", "NO", "NO2", "NOX",
    "OZONE", "PM10", "RAIN", "TEMP", "WDR", "WSP",
]

# ✅ These features are ALWAYS used from the target site itself
# ✅ The target variable is automatically excluded where needed to prevent leakage
INPUT_COLUMNS = [
    # Air Pollutants (from target site)
    "CO",       # Carbon Monoxide
    "NO",       # Nitric Oxide
    "NO2",      # Nitrogen Dioxide
    "NOX",      # Nitrogen Oxides
    "PM10",     # Particulate Matter (10 micrometers)
    "PM2.5",    # Particulate Matter (2.5 micrometers)
    "SO2",      # Sulfur Dioxide
    # Meteorological Variables (from target site)
    "HUMID",    # Humidity
    "TEMP",     # Temperature
    "WSP",      # Wind Speed
    "RAIN",     # Rainfall
    # "SOLAR",    # Solar Radiation
    "WDR",      # Wind Direction
    "WGU",      # Wind Gust
    "NEPH",     # Nephelometer / particle scattering proxy
]

# Regional pooled fallback used when a region has no Stage 3 winning row. The
# target is removed automatically and features with fewer than the configured
# number of numeric observations are discarded before model execution.
REGIONAL_GENERIC_FEATURES = list(LOCAL_ANALYSIS_INPUTS)
REGIONAL_GENERIC_MIN_FEATURE_OBSERVATIONS = 50

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
SAVE_PLOTS = False                    # Generate evaluation plots

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
# SITE / REGION METADATA (AUTO-LOADED FROM EXCEL OR CSV)
# ============================================================================

def _clean_string(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def _contains_excluded_keyword(text, excluded_keywords=None):
    text = _clean_string(text)
    if not text:
        return True
    excluded_keywords = excluded_keywords or []
    text_lower = text.lower()
    return any(k.lower() in text_lower for k in excluded_keywords)


def _find_column(columns, candidates):
    for cand in candidates:
        for c in columns:
            if c.strip().lower() == cand.strip().lower():
                return c
    for cand in candidates:
        for c in columns:
            if cand.strip().lower() in c.strip().lower():
                return c
    return None


def load_region_site_mapping(
    station_info_file=STATION_INFO_FILE,
    sheet_name=STATION_INFO_SHEET,
    excluded_region_keywords=None,
    save_json=True,
    json_output_file=REGION_SITE_JSON_FILE
):
    """
    Load region -> study sites mapping from station metadata file.

    Returns a dictionary with:
        - region_to_sites
        - site_to_region
        - site_coordinates
        - region_site_details
        - all_regions
        - all_sites
    """
    excluded_region_keywords = excluded_region_keywords or EXCLUDED_REGION_KEYWORDS

    if not station_info_file or not os.path.exists(station_info_file):
        logging.warning(f"Station info file not found: {station_info_file}")
        return {
            "region_to_sites": {},
            "site_to_region": {},
            "site_coordinates": {},
            "region_site_details": {},
            "all_regions": [],
            "all_sites": []
        }

    df = None

    try:
        if station_info_file.lower().endswith('.csv'):
            df = pd.read_csv(station_info_file)
        elif station_info_file.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(
                station_info_file,
                sheet_name=sheet_name,
                engine='openpyxl' if station_info_file.lower().endswith('.xlsx') else None
            )
        else:
            logging.warning(f"Unsupported station info file format: {station_info_file}")
    except Exception as e:
        logging.warning(f"Failed to read station metadata file: {station_info_file} | Error: {e}")

    if df is None or df.empty:
        logging.warning("Station metadata could not be loaded or is empty.")
        return {
            "region_to_sites": {},
            "site_to_region": {},
            "site_coordinates": {},
            "region_site_details": {},
            "all_regions": [],
            "all_sites": []
        }

    columns = df.columns.tolist()

    site_id_col = _find_column(columns, ["Site_Id", "SiteID", "Column1.Site_Id"])
    site_col = _find_column(columns, ["SiteName", "Site", "Station", "Column1.SiteName"])
    lon_col = _find_column(columns, ["Longitude", "Lon", "Long", "Column1.Longitude"])
    lat_col = _find_column(columns, ["Latitude", "Lat", "Column1.Latitude"])
    region_col = _find_column(columns, ["Region", "Column1.Region"])

    missing_cols = []
    if site_col is None:
        missing_cols.append("SiteName")
    if lon_col is None:
        missing_cols.append("Longitude")
    if lat_col is None:
        missing_cols.append("Latitude")
    if region_col is None:
        missing_cols.append("Region")

    if missing_cols:
        logging.warning(f"Missing expected columns in station metadata: {missing_cols}")
        logging.warning(f"Available columns: {columns}")
        return {
            "region_to_sites": {},
            "site_to_region": {},
            "site_coordinates": {},
            "region_site_details": {},
            "all_regions": [],
            "all_sites": []
        }

    df = df.copy()

    df[site_col] = df[site_col].apply(_clean_string)
    df[region_col] = df[region_col].apply(_clean_string)
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')

    valid_mask = (
        df[site_col].ne('') &
        df[region_col].ne('') &
        df[lon_col].notna() &
        df[lat_col].notna()
    )

    valid_mask &= ~df[region_col].apply(
        lambda x: _contains_excluded_keyword(x, excluded_region_keywords)
    )

    df = df.loc[valid_mask].copy()

    dedupe_cols = [site_col, region_col, lon_col, lat_col]
    if site_id_col:
        dedupe_cols = [site_id_col] + dedupe_cols

    df = df.drop_duplicates(subset=dedupe_cols)

    region_to_sites = {}
    site_to_region = {}
    site_coordinates = {}
    region_site_details = {}

    for _, row in df.iterrows():
        site_name = _clean_string(row[site_col]).upper()
        region = _clean_string(row[region_col])
        lon = float(row[lon_col])
        lat = float(row[lat_col])

        site_id = None
        if site_id_col and pd.notna(row[site_id_col]):
            try:
                site_id = int(row[site_id_col])
            except Exception:
                site_id = _clean_string(row[site_id_col])

        region_to_sites.setdefault(region, [])
        if site_name not in region_to_sites[region]:
            region_to_sites[region].append(site_name)

        site_to_region[site_name] = region
        site_coordinates[site_name] = {'lat': lat, 'lon': lon}

        region_site_details.setdefault(region, [])
        region_site_details[region].append({
            'site_id': site_id,
            'site_name': site_name,
            'longitude': lon,
            'latitude': lat,
            'region': region
        })

    all_regions = sorted(region_to_sites.keys())

    for region in all_regions:
        region_to_sites[region] = sorted(region_to_sites[region])
        region_site_details[region] = sorted(
            region_site_details[region],
            key=lambda x: x['site_name']
        )

    all_sites = sorted(site_to_region.keys())

    payload = {
        'region_to_sites': region_to_sites,
        'site_to_region': site_to_region,
        'site_coordinates': site_coordinates,
        'region_site_details': region_site_details,
        'all_regions': all_regions,
        'all_sites': all_sites
    }

    if save_json:
        try:
            os.makedirs(os.path.dirname(json_output_file), exist_ok=True)
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved region-site mapping JSON: {json_output_file}")
        except Exception as e:
            logging.warning(f"Could not save region-site mapping JSON: {e}")

    logging.info(
        f"Loaded {len(all_regions)} valid regions and {len(all_sites)} valid study sites "
        f"from {station_info_file} [{sheet_name}]"
    )

    return payload


def load_site_coordinates(
    station_info_file=STATION_INFO_FILE,
    fallback_excel=None
):
    """
    Backward-compatible loader returning only:
        {'SITENAME': {'lat': x, 'lon': y}}
    """
    excel_file = station_info_file
    if fallback_excel and os.path.exists(fallback_excel):
        excel_file = fallback_excel

    mapping = load_region_site_mapping(
        station_info_file=excel_file,
        sheet_name=STATION_INFO_SHEET,
        excluded_region_keywords=EXCLUDED_REGION_KEYWORDS,
        save_json=True,
        json_output_file=REGION_SITE_JSON_FILE
    )

    return mapping.get('site_coordinates', {})


def get_sites_from_regions(target_regions=None):
    """
    Return all sites for selected regions.
    [] or None means all valid regions.
    """
    target_regions = target_regions or []

    if not target_regions:
        return sorted(REGION_SITE_MAPPING['all_sites'])

    sites = []
    for region in target_regions:
        region_sites = REGION_SITE_MAPPING['region_to_sites'].get(region, [])
        sites.extend(region_sites)

    return sorted(set(sites))


# ✅ AUTO-LOAD ALL REGION / SITE METADATA
REGION_SITE_MAPPING = load_region_site_mapping(
    station_info_file=STATION_INFO_FILE,
    sheet_name=STATION_INFO_SHEET,
    excluded_region_keywords=EXCLUDED_REGION_KEYWORDS,
    save_json=True,
    json_output_file=REGION_SITE_JSON_FILE
)

ALL_REGIONS = REGION_SITE_MAPPING['all_regions']
REGION_TO_SITES = REGION_SITE_MAPPING['region_to_sites']
SITE_TO_REGION = REGION_SITE_MAPPING['site_to_region']
SITE_COORDINATES = REGION_SITE_MAPPING['site_coordinates']

# ✅ AUTO-POPULATE TARGET_SITES IF NOT EXPLICITLY GIVEN
if not TARGET_SITES:
    TARGET_SITES = get_sites_from_regions(TARGET_REGIONS)

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

def print_region_site_summary():
    """Print region -> study site summary for quick validation."""
    print("\n" + "=" * 80)
    print("REGION -> STUDY SITES SUMMARY")
    print("=" * 80)

    for region in ALL_REGIONS:
        sites = REGION_TO_SITES.get(region, [])
        print(f"\n{region} ({len(sites)} sites)")
        for s in sites:
            print(f"  - {s}")

    print("\n" + "=" * 80)
    print(f"Total valid regions: {len(ALL_REGIONS)}")
    print(f"Total valid study sites: {len(TARGET_SITES)}")
    print(f"JSON file: {REGION_SITE_JSON_FILE}")
    print("=" * 80)

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
        met_vars = ["HUMID", "TEMP", "WSP", "RAIN", "WDR", "WGU",
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
