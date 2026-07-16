"""regional_feature_selection.py

Region- and Site-wise Feature Selection for PM2.5

This script does TWO things (depending on --mode):

1) --mode per_site   (default)
   - For each region in config_spatial.TARGET_REGIONS (or all regions),
     and for each site in that region:
       * Load the site's CSV from config_spatial.INPUT_DIRECTORY
       * Target: PM2.5
       * Candidate features: CO, NO, NOX, PM10, HUMID, TEMP, WSP, RAIN, WDR, WGU
       * Evaluate:
            - Baseline (temporal only: no local candidates)
            - Each candidate feature alone
            - ALL candidates together
       * Use your existing LightGBM imputation pipeline (Model.LightGBM.impute_mice)
         with a simple missingness regime (random, 20%).
       * Save per-site CSV + plots under:
            regional_feature_selection_results/per_site/

2) --mode per_region
   - Reads all per-site CSVs produced above.
   - For each (Region, Target=PM2.5), aggregates per-site results:
       * Mean & std RMSE and R per feature
       * Number of sites where each feature improves over baseline
   - Saves per-region summary CSVs + RMSE and R bar plots under:
       regional_feature_selection_results/per_region/

Author: Dr. Masrur (design) + assistant (implementation)
Date: 2026-03-23
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Your existing modules
import config_spatial as config
from missingness_regimes import apply_missingness
from evaluation_metrics import evaluate_metrics

try:
    from sklearn.ensemble import RandomForestRegressor
    _HAS_SKLEARN = True
except Exception:
    RandomForestRegressor = None
    _HAS_SKLEARN = False
    logging.warning("sklearn not available; falling back to simple-mean predictors for RF calls")


# -----------------------------------------------------------------------------
# RandomForest-based prediction (not imputation)
# - Baseline predictor uses simple temporal features (hour, dayofweek, month).
# - Local-feature predictor trains RF on rows where target+features are observed
#   and predicts only rows where predictors are complete. Rows with missing
#   predictors are left as NaN (i.e., we "ignore" missing data as requested).
# -----------------------------------------------------------------------------

def _rf_train_and_predict(df: pd.DataFrame, target_column: str, features: List[str]):
    df = df.copy()
    if not features:
        raise ValueError("No features provided for RF training")

    # Training rows: target observed and all features present
    train_mask = df[target_column].notna() & df[features].notna().all(axis=1)
    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target_column]

    if len(X_train) < 10:
        raise ValueError(f"Not enough training rows for RF (found {len(X_train)})")

    if _HAS_SKLEARN and RandomForestRegressor is not None:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        # Prediction rows: target missing but predictors complete
        pred_mask = df[target_column].isna() & df[features].notna().all(axis=1)
        if pred_mask.sum() == 0:
            return df

        X_pred = df.loc[pred_mask, features]
        preds = rf.predict(X_pred)
        df.loc[pred_mask, target_column] = preds
        return df
    else:
        # Fallback: use simple conditional mean by feature combinations if possible,
        # otherwise global mean of y_train.
        pred_mask = df[target_column].isna() & df[features].notna().all(axis=1)
        if pred_mask.sum() == 0:
            return df

        try:
            # if single feature, use mean per-bin
            if len(features) == 1:
                f = features[0]
                grp = df.loc[train_mask].groupby(f)[target_column].mean()
                vals = df.loc[pred_mask, f].map(grp).fillna(y_train.mean())
                df.loc[pred_mask, target_column] = vals.values
                return df
            else:
                df.loc[pred_mask, target_column] = y_train.mean()
                return df
        except Exception:
            df.loc[pred_mask, target_column] = y_train.mean()
            return df


def _rf_baseline_predict(df: pd.DataFrame, target_column: str):
    df = df.copy()
    if "DateTime" not in df.columns:
        # try to find datetime-like column name (lowercase handled earlier)
        dt_col = next((c for c in df.columns if c.lower() == "datetime"), None)
        if dt_col is None:
            raise ValueError("No DateTime column available for baseline temporal features")
        df["DateTime"] = pd.to_datetime(df[dt_col], errors="coerce")

    # simple temporal features
    df["__hour"] = df["DateTime"].dt.hour
    df["__dow"] = df["DateTime"].dt.dayofweek
    df["__month"] = df["DateTime"].dt.month
    features = ["__hour", "__dow", "__month"]

    out = None
    try:
        out = _rf_train_and_predict(df, target_column, features)
    finally:
        # clean temporary columns if present
        if out is None:
            out = df
        for c in features:
            if c in out.columns:
                out = out.drop(columns=[c])
    return out


# =============================================================================
# CONFIGURATION SPECIFIC TO THIS SCRIPT
# =============================================================================

# Target variable for feature selection
TARGET_COLUMN = "PM2.5"

# Candidate local input features (exactly as requested)
CANDIDATE_FEATURES_PM25 = [
    "CO",   # Carbon Monoxide
    "NO",   # Nitric Oxide
    "NOX",  # Nitrogen Oxides
    "PM10", # Particulate Matter (10 micrometers)

    "HUMID",  # Humidity
    "TEMP",   # Temperature
    "WSP",    # Wind Speed
    "RAIN",   # Rainfall
    "WDR",    # Wind Direction
    "WGU",    # Wind Gust
]

# Missingness regime & fraction for screening
MISSINGNESS_REGIME = "random"
MISSINGNESS_FRAC = 0.20  # 20%

# Only use up to this many rows per site (for speed); set None for all
MAX_ROWS_PER_SITE = 5000

# Output root directory for this script
OUTPUT_ROOT = Path("regional_feature_selection_results")
OUTPUT_ROOT.mkdir(exist_ok=True)

# Sites to omit (no matching files / not available)
EXCLUDED_SITES = {"VINEYARD", "MACARTHUR"}


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=getattr(logging, getattr(config, "LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_ROOT / "regional_feature_selection.log", mode="w")
    ]
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10


# =============================================================================
# HELPERS: LOADING DATA
# =============================================================================

def find_site_file(site_name: str) -> Optional[str]:
    """Find the CSV file for a given site in config.INPUT_DIRECTORY."""
    input_dir = config.INPUT_DIRECTORY
    import re

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    target_norm = _norm(site_name)
    # 1) Prefer filenames whose first underscore-separated stem matches the site
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".csv"):
            continue
        stem = filename.split("_")[0]
        if _norm(stem) == target_norm:
            return os.path.join(input_dir, filename)

    # 2) Try any part of the filename (without extension) matching the site
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".csv"):
            continue
        name = os.path.splitext(filename)[0]
        parts = re.split(r"[_\- ]+", name)
        for p in parts:
            if _norm(p) == target_norm:
                return os.path.join(input_dir, filename)

    # 3) Fallback: filename contains the normalized site name anywhere
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".csv"):
            continue
        if target_norm in _norm(filename):
            return os.path.join(input_dir, filename)
    return None


def load_site_data(site_name: str, target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, str]:
    """
    Load data for a specific site, ensuring target_column exists.

    Returns (df, filepath)
    """
    filepath = find_site_file(site_name)
    if filepath is None:
        # provide a short debug sample to help find mapping issues
        try:
            sample_files = sorted([f for f in os.listdir(config.INPUT_DIRECTORY) if f.lower().endswith('.csv')])[:40]
        except Exception:
            sample_files = []
        logging.info(f"No CSV found for site '{site_name}' in {config.INPUT_DIRECTORY}. Example files: {sample_files}")
        raise FileNotFoundError(
            f"No CSV found for site '{site_name}' in {config.INPUT_DIRECTORY}"
        )

    df = pd.read_csv(filepath, low_memory=False)
    # Ensure we have a DateTime column. Support combined 'Date' + 'Time' columns.
    cols_lower = [c.lower() for c in df.columns]
    if "datetime" not in cols_lower:
        # try to find date and time columns (case-insensitive)
        date_col = None
        time_col = None
        for c in df.columns:
            cl = c.lower()
            if cl == "date" or cl.endswith("date") or cl.startswith("date"):
                date_col = c
            if cl == "time" or cl.endswith("time") or cl.startswith("time"):
                time_col = c
        if date_col and time_col:
            try:
                df["DateTime"] = pd.to_datetime(
                    df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
                    errors="coerce",
                )
                logging.info(f"    Created 'DateTime' by combining '{date_col}' and '{time_col}' for file: {filepath}")
            except Exception:
                raise ValueError(f"Failed to combine Date and Time into 'DateTime' for file: {filepath}")
        else:
            raise ValueError(f"'DateTime' column missing in file: {filepath}")
    else:
        # preserve original column name casing
        dt_col = next(c for c in df.columns if c.lower() == "datetime")
        df["DateTime"] = pd.to_datetime(df[dt_col], errors="coerce")

    # ------------------------------------------------------------------
    # Map site-specific suffixed column names to generic names expected
    # e.g., PM2.5_CHULLORA -> PM2.5 ; CO_CHULLORA -> CO
    # ------------------------------------------------------------------
    import re

    def normalize_token(s: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(s).upper())

    base_norm_cache = {}

    def find_site_specific_column(df_cols, base_name: str, site_name_local: str):
        # flexible matching using normalized tokens
        base_norm = normalize_token(base_name)
        site_norm = normalize_token(site_name_local)

        # 1) exact raw match
        for c in df_cols:
            if c == base_name:
                return c

        # 2) exact normalized base match (e.g., PM25 vs PM2.5)
        for c in df_cols:
            if normalize_token(c) == base_norm:
                return c

        # 3) base + site suffix match (most specific)
        for c in df_cols:
            c_norm = normalize_token(c)
            if c_norm.startswith(base_norm) and c_norm.endswith(site_norm):
                return c

        # 4) columns that start with base_norm (use if unique)
        start_matches = [c for c in df_cols if normalize_token(c).startswith(base_norm)]
        if len(start_matches) == 1:
            return start_matches[0]

        return None

    # Map target column
    target_actual = find_site_specific_column(df.columns, target_column, site_name)
    if target_actual is None:
        # try looser search for 'PM25' without punctuation
        alt = find_site_specific_column(df.columns, target_column.replace('.', ''), site_name)
        target_actual = alt

    if target_actual is None:
        logging.info(f"Target '{target_column}' not found in file: {filepath} — available columns: {list(df.columns)[:20]}")
        # return empty DataFrame so caller can skip gracefully
        return pd.DataFrame(), filepath

    # If target actual differs, create a generic column name
    if target_actual != target_column:
        df[target_column] = df[target_actual]

    # Map candidate features to generic names when possible
    for feat in CANDIDATE_FEATURES_PM25:
        found = find_site_specific_column(df.columns, feat, site_name)
        if found and found != feat:
            df[feat] = df[found]

    # Drop feature columns that are entirely NaN, but never drop the target column here
    all_nan_cols = [c for c in df.columns if df[c].isna().all() and c != target_column]
    if all_nan_cols:
        logging.debug(f"Dropping {len(all_nan_cols)} feature columns that contain NO observed values (all-NaN). Dropped columns (first 20 shown): {all_nan_cols[:20]}")
        df = df.drop(columns=all_nan_cols)

    # If target column missing at this point, nothing we can do
    if target_column not in df.columns:
        logging.info(f"After mapping, target '{target_column}' still not present in file: {filepath}")
        return pd.DataFrame(), filepath

    # If target exists but contains no valid observations, skip gracefully
    if df[target_column].notna().sum() == 0:
        logging.info(f"Target '{target_column}' exists but has NO valid observed values in file: {filepath}; skipping site. Available columns: {list(df.columns)[:20]}")
        return pd.DataFrame(), filepath

    return df, filepath


# =============================================================================
# HELPERS: MISSINGNESS & METRICS
# =============================================================================

def apply_test_missingness(
    data: pd.DataFrame,
    target_column: str,
    regime: str = MISSINGNESS_REGIME,
    frac: float = MISSINGNESS_FRAC,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply structured missingness to target_column and return
    (data_with_missing, simulated_mask).
    """
    original_missing = data[target_column].isna()
    data_with_missing, simulated_mask = apply_missingness(
        data, target_column, regime=regime, frac=frac, seed=seed
    )
    simulated_mask = simulated_mask & (~original_missing)
    return data_with_missing, simulated_mask


def evaluate_imputation(
    data_original: pd.DataFrame,
    data_imputed: pd.DataFrame,
    simulated_mask: pd.Series,
    target_column: str
) -> Dict[str, float]:
    """
    Compute metrics on simulated-missing positions only.
    """
    true_vals = data_original.loc[simulated_mask, target_column].values
    imp_vals = data_imputed.loc[simulated_mask, target_column].values

    valid = np.isfinite(true_vals) & np.isfinite(imp_vals)
    if config.HANDLE_NEGATIVES == "exclude":
        valid &= (true_vals >= 0) & (imp_vals >= 0)

    if valid.sum() < 10:
        return {}

    t = true_vals[valid]
    p = imp_vals[valid]

    metrics = evaluate_metrics(t, p, handle_negative=config.HANDLE_NEGATIVES)
    return metrics


# =============================================================================
# PER-SITE FEATURE SCREENING
# =============================================================================

def get_site_candidate_features(df: pd.DataFrame, target_column: str) -> List[str]:
    """Return the subset of CANDIDATE_FEATURES_PM25 available in this site's data."""
    return [
        c for c in CANDIDATE_FEATURES_PM25
        if c in df.columns and c != target_column
    ]


def run_site_feature_screen(
    site_name: str,
    target_column: str = TARGET_COLUMN,
    regime: str = MISSINGNESS_REGIME,
    frac: float = MISSINGNESS_FRAC,
) -> pd.DataFrame:
    """
    For a given site:
      - Baseline temporal-only
      - Each candidate feature alone
      - All candidates together

    Returns a DataFrame with columns:
      Site, Feature, RMSE, R, NSE, MAE
    """
    logging.info(f"  [SITE] {site_name} | Target={target_column}")

    try:
        df, path = load_site_data(site_name, target_column)
    except Exception as e:
        logging.info(f"    Failed to load data for {site_name}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        logging.info(f"    No usable data for {site_name}; skipping. (see earlier messages for available columns)")
        return pd.DataFrame()

    # Subsample for speed if needed
    if MAX_ROWS_PER_SITE is not None and len(df) > MAX_ROWS_PER_SITE:
        df = df.iloc[:MAX_ROWS_PER_SITE].copy()

    # Ensure numeric target & candidates
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    candidate_feats = get_site_candidate_features(df, target_column)
    for c in candidate_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[target_column].notna().sum() < 50:
        logging.info(f"    Not enough valid {target_column} values at {site_name} (found {df[target_column].notna().sum()}); skipping.")
        return pd.DataFrame()

    if not candidate_feats:
        logging.info(f"    No candidate features present for {site_name}; skipping. Available columns: {list(df.columns)[:20]}")
        return pd.DataFrame()

    results = []

    # ----------------------------------------------------------------------
    # 1) Baseline: temporal-only (no local candidate features)
    # ----------------------------------------------------------------------
    logging.info("    - Baseline (Temporal-only)")
    try:
        df_missing, sim_mask = apply_test_missingness(df.copy(), target_column, regime, frac)
        # Baseline: use RandomForest on simple temporal features (hour/day/month)
        df_imp = _rf_baseline_predict(df_missing, target_column)
        metrics = evaluate_imputation(df, df_imp, sim_mask, target_column)
        if metrics:
            results.append({
                "Site": site_name,
                "Feature": "BASELINE_TEMPORAL_ONLY",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
            })
    except Exception as e:
        logging.info(f"    Baseline FS skipped/failed at {site_name}: {e}")

    # ----------------------------------------------------------------------
    # 2) Each candidate feature alone
    # ----------------------------------------------------------------------
    for feat in candidate_feats:
        logging.info(f"    - Testing feature alone: {feat}")
        try:
            df_missing, sim_mask = apply_test_missingness(df.copy(), target_column, regime, frac)
            # Predict using RandomForest trained on this single local feature
            df_imp = _rf_train_and_predict(df_missing, target_column, [feat])
            metrics = evaluate_imputation(df, df_imp, sim_mask, target_column)
            if metrics:
                results.append({
                    "Site": site_name,
                    "Feature": feat,
                    "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                    "R": metrics.get("Correlation Coefficient (R)", np.nan),
                    "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                    "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
                })
        except Exception as e:
            logging.info(f"      Feature {feat} skipped/failed at {site_name}: {e}")

    # ----------------------------------------------------------------------
    # 3) All candidates together
    # ----------------------------------------------------------------------
    logging.info("    - ALL CANDIDATES")
    try:
        df_missing, sim_mask = apply_test_missingness(df.copy(), target_column, regime, frac)
        # Predict using RandomForest trained on all candidate features (rows with missing predictors ignored)
        df_imp = _rf_train_and_predict(df_missing, target_column, candidate_feats)
        metrics = evaluate_imputation(df, df_imp, sim_mask, target_column)
        if metrics:
            results.append({
                "Site": site_name,
                "Feature": "ALL_CANDIDATES",
                "RMSE": metrics.get("Root Mean Squared Error (RMSE)", np.nan),
                "R": metrics.get("Correlation Coefficient (R)", np.nan),
                "NSE": metrics.get("Nash-Sutcliffe Efficiency (NSE)", np.nan),
                "MAE": metrics.get("Mean Absolute Error (MAE)", np.nan),
            })
    except Exception as e:
        logging.info(f"    ALL_CANDIDATES skipped/failed at {site_name}: {e}")

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    return df_res


def plot_site_feature_performance(
    df_res: pd.DataFrame,
    site_name: str,
    out_dir: Path
):
    """Create simple horizontal bar plots for RMSE & R for a site."""
    if df_res.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # RMSE
    df_rmse = df_res.sort_values("RMSE")
    colors = [
        "red" if f == "BASELINE_TEMPORAL_ONLY"
        else "darkgreen" if f == "ALL_CANDIDATES"
        else "steelblue"
        for f in df_rmse["Feature"]
    ]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_rmse["Feature"], df_rmse["RMSE"], color=colors, edgecolor="black")
    plt.xlabel("RMSE", fontweight="bold")
    plt.title(f"{site_name} – {TARGET_COLUMN}\nRMSE by Feature", fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, df_rmse["RMSE"]):
        plt.text(val, bar.get_y() + bar.get_height()/2, f"{val:.2f}",
                 ha="left", va="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    fp_rmse = out_dir / f"{site_name}_PM25_feature_RMSE.png"
    plt.savefig(fp_rmse, dpi=300, bbox_inches="tight")
    plt.close()

    # R
    df_r = df_res.sort_values("R", ascending=False)
    colors_r = [
        "red" if f == "BASELINE_TEMPORAL_ONLY"
        else "darkgreen" if f == "ALL_CANDIDATES"
        else "steelblue"
        for f in df_r["Feature"]
    ]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_r["Feature"], df_r["R"], color=colors_r, edgecolor="black")
    plt.xlabel("Correlation Coefficient (R)", fontweight="bold")
    plt.xlim(0, 1.0)
    plt.title(f"{site_name} – {TARGET_COLUMN}\nR by Feature", fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, df_r["R"]):
        plt.text(val, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                 ha="left", va="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    fp_r = out_dir / f"{site_name}_PM25_feature_R.png"
    plt.savefig(fp_r, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"    Saved site-level plots: {fp_rmse}, {fp_r}")


# =============================================================================
# PER-REGION AGGREGATION
# =============================================================================

def aggregate_region_from_site_results(all_sites_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given concatenated site-level rows for one region+target:

    Columns expected:
      Site, Feature, RMSE, R, NSE, MAE

    Returns a DataFrame with per-feature region-level stats:
      Feature, RMSE_mean, RMSE_std, R_mean, R_std, n_sites, n_runs, Frac_Improve
    """
    if all_sites_df.empty:
        return pd.DataFrame()

    # Compute baseline RMSE per site
    baseline = (
        all_sites_df[all_sites_df["Feature"] == "BASELINE_TEMPORAL_ONLY"]
        .groupby("Site")["RMSE"]
        .mean()
        .rename("Baseline_RMSE")
        .reset_index()
    )

    merged = all_sites_df.merge(baseline, on="Site", how="left")

    # Improvement indicator: feature RMSE < baseline
    merged["Improves_vs_Baseline"] = merged["RMSE"] < merged["Baseline_RMSE"]

    # Exclude baseline row from aggregation
    merged_nonbase = merged[merged["Feature"] != "BASELINE_TEMPORAL_ONLY"].copy()

    agg = (
        merged_nonbase
        .groupby("Feature")
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R_mean=("R", "mean"),
            R_std=("R", "std"),
            n_sites=("Site", "nunique"),
            n_runs=("RMSE", "count"),
            n_improve=("Improves_vs_Baseline", "sum"),
        )
        .reset_index()
    )
    agg["Frac_Improve"] = agg["n_improve"] / agg["n_runs"].replace(0, np.nan)

    return agg


def plot_region_feature_summary(
    region_name_safe: str,
    region_label: str,
    df_agg: pd.DataFrame,
    out_dir: Path
):
    """
    Create region-level bar plots: RMSE_mean & R_mean by Feature.
    """
    if df_agg.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # RMSE
    df_rmse = df_agg.sort_values("RMSE_mean")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_rmse,
        x="RMSE_mean",
        y="Feature",
        color="steelblue",
        edgecolor="black"
    )
    plt.xlabel("Mean RMSE", fontweight="bold")
    plt.title(f"{region_label} – {TARGET_COLUMN}\nRegion-level Mean RMSE by Feature", fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fp_rmse = out_dir / f"{region_name_safe}_PM25_region_feature_RMSE.png"
    plt.savefig(fp_rmse, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved region RMSE plot: {fp_rmse}")

    # R
    df_r = df_agg.sort_values("R_mean", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_r,
        x="R_mean",
        y="Feature",
        color="darkgreen",
        edgecolor="black"
    )
    plt.xlabel("Mean R", fontweight="bold")
    plt.xlim(0, 1.0)
    plt.title(f"{region_label} – {TARGET_COLUMN}\nRegion-level Mean R by Feature", fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fp_r = out_dir / f"{region_name_safe}_PM25_region_feature_R.png"
    plt.savefig(fp_r, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved region R plot: {fp_r}")


# =============================================================================
# MODES
# =============================================================================

def run_mode_per_site():
    """
    Per-site feature screening for all sites in TARGET_REGIONS (or all regions).
    Writes per-site CSVs and plots.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           PER-SITE FEATURE SELECTION FOR PM2.5                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    per_site_dir = OUTPUT_ROOT / "per_site"
    per_site_dir.mkdir(exist_ok=True)

    region_to_sites = config.REGION_TO_SITES
    regions = config.TARGET_REGIONS or list(region_to_sites.keys())

    logging.info(f"Regions to analyze: {regions}")
    logging.info(f"Candidate features for PM2.5: {CANDIDATE_FEATURES_PM25}")
    logging.info(f"Missingness regime: {MISSINGNESS_REGIME}, fraction: {MISSINGNESS_FRAC}")

    for region in regions:
        sites = region_to_sites.get(region, [])
        # Exclude known-bad site names that don't have files
        sites = [s for s in sites if s.upper() not in EXCLUDED_SITES]
        # Filter to sites that actually have a matching CSV file
        available = []
        missing = []
        for s in sites:
            try:
                if find_site_file(s) is not None:
                    available.append(s)
                else:
                    missing.append(s)
            except Exception:
                missing.append(s)

        if missing:
            logging.warning(f"Skipping {len(missing)} unavailable sites in region '{region}': {missing}")

        sites = available
        if not sites:
            logging.warning(f"No available sites for region '{region}'")
            continue

        logging.info(f"\n=== REGION: {region} ({len(sites)} sites) ===")

        for site in sites:
            df_site = run_site_feature_screen(site_name=site, target_column=TARGET_COLUMN)
            if df_site.empty:
                continue

            # Save per-site CSV
            out_csv = per_site_dir / f"{site}_{TARGET_COLUMN.replace('.', '')}_individual_features.csv"
            df_site.to_csv(out_csv, index=False)
            logging.info(f"  Saved site feature results: {out_csv}")

            # Plots
            plot_site_feature_performance(df_site, site, per_site_dir)


def run_mode_per_region():
    """
    Aggregate over all per-site CSVs produced by run_mode_per_site and
    create region-level summaries + plots.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           PER-REGION FEATURE SELECTION SUMMARY FOR PM2.5             ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    per_site_dir = OUTPUT_ROOT / "per_site"
    per_region_dir = OUTPUT_ROOT / "per_region"
    per_region_dir.mkdir(exist_ok=True)

    region_to_sites = config.REGION_TO_SITES
    regions = config.TARGET_REGIONS or list(region_to_sites.keys())

    for region in regions:
        sites = region_to_sites.get(region, [])
        # Exclude known-bad site names that don't have files
        sites = [s for s in sites if s.upper() not in EXCLUDED_SITES]
        # Filter to sites that actually have a per-site CSV (or input CSV)
        available = []
        missing = []
        for s in sites:
            try:
                # prefer using per-site output if present, else check input file
                pattern = per_site_dir / f"{s}_{TARGET_COLUMN.replace('.', '')}_individual_features.csv"
                if pattern.exists() or find_site_file(s) is not None:
                    available.append(s)
                else:
                    missing.append(s)
            except Exception:
                missing.append(s)

        if missing:
            logging.warning(f"Skipping {len(missing)} unavailable sites in region '{region}' during aggregation: {missing}")

        sites = available
        if not sites:
            continue

        region_label = region
        region_name_safe = region_label.replace(" ", "_")

        rows = []
        for site in sites:
            pattern = per_site_dir / f"{site}_{TARGET_COLUMN.replace('.', '')}_individual_features.csv"
            if not pattern.exists():
                logging.warning(f"[{region}] No per-site CSV for {site}; run per_site mode first.")
                continue
            df_site = pd.read_csv(pattern)
            df_site["Site"] = site
            rows.append(df_site)

        if not rows:
            logging.warning(f"[{region}] No site-level results to aggregate.")
            continue

        all_sites_df = pd.concat(rows, ignore_index=True)
        all_sites_fp = per_region_dir / f"{region_name_safe}_PM25_all_site_rows.csv"
        all_sites_df.to_csv(all_sites_fp, index=False)
        logging.info(f"[{region}] Saved concatenated site rows: {all_sites_fp}")

        agg = aggregate_region_from_site_results(all_sites_df)
        if agg.empty:
            logging.warning(f"[{region}] Aggregation produced empty DataFrame.")
            continue

        agg_fp = per_region_dir / f"{region_name_safe}_PM25_region_feature_summary.csv"
        agg.to_csv(agg_fp, index=False)
        logging.info(f"[{region}] Saved region-level feature summary: {agg_fp}")

        plot_region_feature_summary(region_name_safe, region_label, agg, per_region_dir)


def report_pm25_by_region(out_fp: Optional[Path] = None):
    """Scan input files and report PM2.5 availability and variables per site/region.

    Writes a human-readable report to `out_fp` or `OUTPUT_ROOT/pm25_region_report.txt`.
    This function only reads headers (and PM2.5 column counts) and DOES NOT modify files
    or drop any columns.
    """
    if out_fp is None:
        out_fp = OUTPUT_ROOT / "pm25_region_report.txt"

    region_to_sites = config.REGION_TO_SITES
    regions = config.TARGET_REGIONS or list(region_to_sites.keys())

    lines = []
    for region in regions:
        lines.append(f"Region: {region}")
        sites = region_to_sites.get(region, [])
        sites = [s for s in sites if s.upper() not in EXCLUDED_SITES]
        for site in sites:
            try:
                filepath = find_site_file(site)
            except Exception:
                filepath = None
            if filepath is None:
                lines.append(f"  {site}: MISSING FILE")
                continue

            # read only header
            try:
                df_head = pd.read_csv(filepath, nrows=0)
                cols = list(df_head.columns)
            except Exception as e:
                lines.append(f"  {site}: ERROR reading header: {e}")
                continue

            # detect target actual column name
            target_actual = None
            try:
                target_actual = find_site_specific_column(cols, TARGET_COLUMN, site)
            except Exception:
                target_actual = None

            target_info = "NO"
            nonnull_count = 0
            if target_actual:
                # read only that column to count non-null
                try:
                    ser = pd.read_csv(filepath, usecols=[target_actual], squeeze=True)
                    nonnull_count = int(ser.notna().sum())
                    target_info = f"YES (column={target_actual}, non-null={nonnull_count})"
                except Exception:
                    target_info = f"YES (column={target_actual}, unreadable)"

            lines.append(f"  {site}: PM2.5 present: {target_info}")

            # list available mapped candidate features
            mapped = {}
            for feat in CANDIDATE_FEATURES_PM25:
                try:
                    found = find_site_specific_column(cols, feat, site)
                except Exception:
                    found = None
                if found:
                    mapped[feat] = found

            if mapped:
                lines.append(f"    Mapped features: {', '.join([f + '->' + mapped[f] for f in mapped])}")
            else:
                lines.append(f"    Mapped features: NONE of {CANDIDATE_FEATURES_PM25}")

            # also list raw columns (shortened)
            lines.append(f"    Raw columns (first 20): {cols[:20]}")

        lines.append("")

    # write report
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w") as fh:
        fh.write("\n".join(lines))

    print(f"PM2.5 report written to: {out_fp}")
    logging.info(f"PM2.5 report written to: {out_fp}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Regional & Site-wise feature selection for PM2.5."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="per_site",
        choices=["per_site", "per_region"],
        help="Mode: per_site (run site-level screening) or per_region (aggregate per-site CSVs).",
    )
    args = parser.parse_args()

    if args.mode == "per_site":
        run_mode_per_site()
    elif args.mode == "per_region":
        run_mode_per_region()
    elif args.mode == "report_pm25":
        report_pm25_by_region()
    else:
        logging.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()