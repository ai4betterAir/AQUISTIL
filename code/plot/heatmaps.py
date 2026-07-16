#!/usr/bin/env python3
"""
research_plots.py

Generate individual heatmaps (one file per Target_Site × Missingness_Regime × Target).

This variant adds "% within threshold" heatmaps (e.g., % of samples with |error| <= 5 µg/m3)
and integrates them into the existing batch generation pipeline.

Usage:
    python research_plots.py
    python research_plots.py --results_dir /path/to/results --output_dir ./heatmaps --metric RMSE --within --threshold 5.0
"""

from pathlib import Path
import argparse
import logging
import re
import os
import fnmatch
import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------------------
# Defaults (embedded in code so script can run without CLI)
# -------------------------------------------------------------------------
DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V19_final"
DEFAULT_OUTPUT_DIR = "./heatmaps"

DEFAULT_TARGET_COLUMNS = ["PM2.5"]
DEFAULT_TARGET_SITES = ["CHULLORA", "LIVERPOOL"]

DEFAULT_METRIC = "RMSE"
DEFAULT_THRESHOLD = 5.0  # µg/m3 for within-threshold metric

REGIME_ORDER = ["random", "short_gap", "medium_gap", "long_gap", "event"]
REGIME_LABELS = {
    "random": "Random (MCAR)",
    "short_gap": "Short Gap (1-23h)",
    "medium_gap": "Medium Gap (24-71h)",
    "long_gap": "Long Gap (72-240h)",
    "event": "Event-Dependent (MNAR)",
}

sns.set_theme(style="whitegrid")


# -------------------------
# Helpers: file parsing
# -------------------------
def extract_target_from_filename(stem: str):
    tokens = re.split(r"[_\-\.\s]", stem)
    pat = re.compile(r"^(pm\d+(\.\d+)?|pm\d+|no2|o3|so2|co|pm10|pm2_5)$", re.IGNORECASE)
    for t in tokens[::-1]:
        if pat.match(t.lower()):
            tt = t.upper().replace("_", ".")
            tt = re.sub(r"^PM25$", "PM2.5", tt)
            tt = re.sub(r"^PM2_5$", "PM2.5", tt)
            return tt
    for t in tokens[::-1]:
        if re.search(r"[A-Za-z]", t) and re.search(r"\d", t):
            return t
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        cols = []
        for c in df.columns:
            s = str(c)
            s = re.sub(r"\s+", " ", s).strip()
            s = re.sub(r"\.\d+$", "", s)
            cols.append(s)
        df.columns = cols
    except Exception:
        pass

    try:
        def norm(x):
            s = str(x).lower()
            s = re.sub(r"\s+", "", s)
            s = re.sub(r"[^0-9a-z]", "", s)
            return s

        groups = {}
        for c in df.columns:
            groups.setdefault(norm(c), []).append(c)
        out = {}
        for k, mem in groups.items():
            if len(mem) == 1:
                out[mem[0]] = df[mem[0]]
            else:
                out[mem[0]] = df.loc[:, mem].bfill(axis=1).iloc[:, 0]
        df = pd.DataFrame(out)
    except Exception:
        pass

    rename_map = {
        "Root Mean Squared Error (RMSE)": "RMSE",
        "Mean Absolute Error (MAE)": "MAE",
        "Missingness": "Missingness_Pct",
        "Missingness_Level": "Missingness_Pct",
        "Missingness_Regime": "MISSINGNESS_REGIMES",
    }
    df = df.rename(columns=rename_map)

    if "Missingness_Pct" in df.columns:
        try:
            df["Missingness_Pct"] = pd.to_numeric(df["Missingness_Pct"], errors="coerce")
            if df["Missingness_Pct"].max() <= 1.0:
                df["Missingness_Pct"] = df["Missingness_Pct"] * 100.0
        except Exception:
            pass

    if "MISSINGNESS_REGIMES" not in df.columns:
        df["MISSINGNESS_REGIMES"] = "random"
    if "StudySite" not in df.columns:
        df["StudySite"] = df.get("StudySite", None)
    if "Target_Site" not in df.columns:
        if "StudySite" in df.columns:
            df["Target_Site"] = df["StudySite"]
    return df


def detect_site_column(df: pd.DataFrame) -> str | None:
    candidates = list(df.columns)
    for cand in ["Target_Site", "TargetSite", "Target_Site_ID", "Target_SiteName"]:
        if cand in candidates:
            return cand
    for cand in ["StudySite", "Study_Site", "Site", "station", "location"]:
        if cand in candidates:
            return cand
    for c in candidates:
        if "target" in c.lower() and "site" in c.lower():
            return c
    for c in candidates:
        if "site" in c.lower():
            return c
    return None


def detect_target_column(df: pd.DataFrame) -> str | None:
    for cand in ["Target", "target", "Target_Var", "TargetColumn", "TargetVariable"]:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "target" in c.lower() and "site" not in c.lower():
            return c
    return None


# -------------------------
# Load / aggregate function
# -------------------------
def load_aggregated(results_dir: Path) -> pd.DataFrame | None:
    agg_file = results_dir / "all_results_summary.csv"
    if agg_file.exists():
        logging.info(f"Loading aggregated CSV: {agg_file}")
        try:
            df = pd.read_csv(agg_file)
            return standardize_columns(df)
        except Exception as e:
            logging.warning(f"Failed to read aggregated CSV: {e} — will attempt scanning.")

    logging.info("Scanning results_dir for per-model metrics CSV files...")
    frames = []
    if not results_dir.exists():
        logging.error(f"Results directory not found: {results_dir}")
        return None

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for regime_dir in sorted(model_dir.iterdir()):
            if not regime_dir.is_dir():
                continue
            regime = regime_dir.name
            metrics_dir = regime_dir / "metrics"
            if not metrics_dir.exists():
                continue
            for csv in sorted(metrics_dir.glob("*.csv")):
                try:
                    dfm = pd.read_csv(csv)
                    if dfm.shape[1] == 0:
                        continue
                    stem = csv.stem
                    site_guess = stem.split("_")[0] if "_" in stem else stem
                    target_guess = extract_target_from_filename(stem)
                    dfm["Model"] = model_name
                    dfm["MISSINGNESS_REGIMES"] = regime
                    dfm["Target_Site"] = site_guess
                    dfm["StudySite"] = site_guess
                    dfm["Target"] = target_guess
                    frames.append(dfm)
                except Exception as e:
                    logging.warning(f"Unable to read {csv}: {e}")

    # Also support the per_model_metrics directory (V21-style):
    # filenames are typically like {ModelName}_{StudySite}_metrics.csv
    per_model_dir = results_dir / "per_model_metrics"
    if per_model_dir.exists() and per_model_dir.is_dir():
        logging.info(f"Scanning per-model metrics directory: {per_model_dir}")
        for csv in sorted(per_model_dir.glob("*.csv")):
            try:
                dfm = pd.read_csv(csv)
                if dfm.shape[1] == 0:
                    continue
                stem = csv.stem
                # remove trailing _metrics if present, then split from right to separate site
                base = stem
                if stem.lower().endswith("_metrics"):
                    base = stem[: -len("_metrics")]
                parts = base.rsplit("_", 1)
                if len(parts) == 2:
                    model_guess, site_guess = parts[0], parts[1]
                else:
                    # fallback: try first token as model, remainder as site
                    toks = base.split("_")
                    model_guess = toks[0]
                    site_guess = toks[-1] if len(toks) > 1 else toks[0]
                target_guess = extract_target_from_filename(stem)
                dfm["Model"] = model_guess
                # Many V21 per-model files already include regime/missingness columns; leave them if present
                dfm["Target_Site"] = site_guess
                dfm["StudySite"] = site_guess
                dfm["Target"] = target_guess
                frames.append(dfm)
            except Exception as e:
                logging.warning(f"Unable to read per-model metrics {csv}: {e}")

    if not frames:
        logging.error("No metrics CSV files discovered while scanning results_dir.")
        return None

    df = pd.concat(frames, ignore_index=True)
    df = standardize_columns(df)
    try:
        df.to_csv(agg_file, index=False)
        logging.info(f"Saved aggregated CSV for future runs: {agg_file}")
    except Exception:
        pass
    return df


# -------------------------
# Derived-metric helpers
# -------------------------
def find_prediction_columns(df: pd.DataFrame):
    """
    Look for a (observed, predicted) column pair in df using common name patterns.
    Returns (obs_col, pred_col) or (None, None) if not found.
    """
    candidates_obs = [
        'y_true', 'y_obs', 'observed', 'obs', 'truth', 'ground_truth',
        'target_value', 'TargetValue', 'Observed', 'ObservedValue', 'OBSERVED'
    ]
    candidates_pred = [
        'y_pred', 'y_hat', 'pred', 'prediction', 'imputed', 'Imputed', 'Predicted', 'PRED'
    ]

    # normalize available columns for fuzzy matching
    col_map = {c.lower(): c for c in df.columns}

    # direct exact matches first
    for o in candidates_obs:
        for p in candidates_pred:
            if o in col_map and p in col_map:
                return col_map[o], col_map[p]

    # fuzzy contains-match: prefer columns containing both keywords
    cols = list(df.columns)
    for o_kw in ['obs', 'true', 'observ', 'target']:
        for p_kw in ['pred', 'imput', 'hat', 'prediction', 'estimate']:
            obs_candidates = [c for c in cols if o_kw in c.lower()]
            pred_candidates = [c for c in cols if p_kw in c.lower()]
            if obs_candidates and pred_candidates:
                return obs_candidates[0], pred_candidates[0]

    # also try pair patterns like ('Observed', 'Imputed'), ('Truth', 'Imputation')
    pattern_pairs = [
        ('observed', 'imputed'),
        ('truth', 'imputed'),
        ('truth', 'prediction'),
        ('target', 'imputed'),
        ('target', 'prediction'),
        ('actual', 'predicted'),
    ]
    for o_kw, p_kw in pattern_pairs:
        obs_candidates = [c for c in cols if o_kw in c.lower()]
        pred_candidates = [c for c in cols if p_kw in c.lower()]
        if obs_candidates and pred_candidates:
            return obs_candidates[0], pred_candidates[0]

    return None, None


def ensure_abs_error(df: pd.DataFrame, force=False) -> pd.DataFrame:
    """
    Ensure df contains 'abs_error' and 'error' columns if possible.

    Strategy:
      1. If 'abs_error' already present and not forcing, return df.
      2. Try to find an (obs, pred) pair via find_prediction_columns() and compute error, abs_error.
      3. If not found, look for common aggregate error columns (e.g., 'Error','AbsError','AE','Abs_Error') and standardize name.
      4. Log what was found or why not.

    Returns the (possibly modified) DataFrame.
    """
    # 1) If already present
    abs_candidates = ['abs_error', 'AbsError', 'ABS_ERROR', 'absolute_error', 'AE', 'Abs_Error']
    for c in abs_candidates:
        if c in df.columns and not force:
            # standardize to 'abs_error'
            if c != 'abs_error':
                try:
                    df['abs_error'] = pd.to_numeric(df[c], errors='coerce')
                    logging.info(f"Using existing column '{c}' as 'abs_error'.")
                except Exception:
                    df['abs_error'] = df[c]
            return df

    # 2) Find obs/pred pair
    obs_col, pred_col = find_prediction_columns(df)
    if obs_col and pred_col:
        try:
            logging.info(f"Found observed/predicted columns -> obs: '{obs_col}', pred: '{pred_col}'. Computing 'error' and 'abs_error'.")
            # compute numeric differences where possible
            df['_obs_num'] = pd.to_numeric(df[obs_col], errors='coerce')
            df['_pred_num'] = pd.to_numeric(df[pred_col], errors='coerce')
            # compute only where both numeric
            mask = df['_obs_num'].notna() & df['_pred_num'].notna()
            if mask.any():
                df.loc[mask, 'error'] = df.loc[mask, '_pred_num'] - df.loc[mask, '_obs_num']
                df.loc[mask, 'abs_error'] = df.loc[mask, 'error'].abs()
                # drop helpers
                df = df.drop(columns=['_obs_num', '_pred_num'])
                logging.info(f"Computed 'error' and 'abs_error' from columns: {obs_col}, {pred_col} (rows with numeric values: {mask.sum()})")
                return df
            else:
                # still create columns but will be NaN
                df['error'] = np.nan
                df['abs_error'] = np.nan
                df = df.drop(columns=['_obs_num', '_pred_num'])
                logging.info(f"Found obs/pred columns but none numeric: '{obs_col}', '{pred_col}'. 'abs_error' created but all NaN.")
                return df
        except Exception as e:
            logging.debug(f"Failed to compute abs_error from {obs_col}/{pred_col}: {e}")

    # 3) Try to find aggregate error columns
    err_candidates = ['error', 'Error', 'ERR', 'Residual']
    for c in err_candidates:
        if c in df.columns:
            try:
                df['error'] = pd.to_numeric(df[c], errors='coerce')
                df['abs_error'] = df['error'].abs()
                logging.info(f"Using existing column '{c}' as 'error' and computed 'abs_error'.")
                return df
            except Exception:
                pass

    abs_candidates2 = ['AbsError', 'AE', 'abs_err', 'abs_error']
    for c in abs_candidates2:
        if c in df.columns:
            try:
                df['abs_error'] = pd.to_numeric(df[c], errors='coerce')
                logging.info(f"Using existing column '{c}' as 'abs_error'.")
                return df
            except Exception:
                pass

    # 4) Nothing found
    logging.info("No obs/pred or error/abs_error columns found. Cannot compute within-threshold metrics.")
    return df


# -------------------------
# Single-heatmap function (existing)
# -------------------------
def plot_individual_heatmap(
    df: pd.DataFrame,
    site_col: str,
    site: str,
    regime: str,
    target_col: str | None,
    target: str | None,
    metric: str,
    output_dir: Path,
    annotate: bool = True,
    cmap: str = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path | None:
    d = df.copy()
    if target is not None and target_col is not None:
        d = d[d[target_col].astype(str) == str(target)]
    elif target is not None:
        logging.debug("Target requested but no target column available; continuing without target filter.")
    if site_col not in d.columns:
        logging.error("Site column not present in dataframe; cannot filter by site.")
        return None
    d = d[d[site_col].astype(str) == str(site)]
    if d.empty:
        logging.info(f"No data for Target_Site={site}; skipping.")
        return None
    d = d[d["MISSINGNESS_REGIMES"].astype(str) == str(regime)]
    if d.empty:
        logging.info(f"No data for Target_Site={site}, Regime={regime}; skipping.")
        return None
    if "Missingness_Pct" not in d.columns:
        logging.info("Missingness_Pct not present; skipping.")
        return None
    d["Missingness_Pct"] = pd.to_numeric(d["Missingness_Pct"], errors="coerce")
    d = d.dropna(subset=["Missingness_Pct"])
    if d.empty:
        logging.info("No numeric Missingness_Pct for selection; skipping.")
        return None
    if metric not in d.columns:
        logging.info(f"Metric '{metric}' not available for selection; skipping.")
        return None
    try:
        pivot = d.pivot_table(values=metric, index="Model", columns="Missingness_Pct", aggfunc="mean")
    except Exception as e:
        logging.warning(f"Pivot creation failed: {e}")
        return None
    if pivot.empty:
        logging.info("Pivot empty after grouping; skipping.")
        return None
    try:
        sorted_cols = sorted(pivot.columns, key=lambda x: float(x))
        pivot = pivot[sorted_cols]
    except Exception:
        pass
    try:
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    except Exception:
        pass
    if vmin is None or vmax is None:
        try:
            gv = pd.to_numeric(df[metric], errors="coerce")
            gv = gv[np.isfinite(gv)]
            gvmin, gvmax = (float(gv.min()), float(gv.max())) if gv.size else (None, None)
        except Exception:
            gvmin = gvmax = None
        vals = pivot.values.flatten()
        vals = vals[np.isfinite(vals)]
        pmin, pmax = (float(np.nanmin(vals)), float(np.nanmax(vals))) if vals.size else (None, None)
        vmin = vmin if vmin is not None else (gvmin if gvmin is not None else pmin)
        vmax = vmax if vmax is not None else (gvmax if gvmax is not None else pmax)
        if vmin is not None and vmax is not None and np.isclose(vmin, vmax):
            pad = 1e-6 if vmin != 0 else 0.1
            vmin -= pad
            vmax += pad
    fig_w = max(6, 0.8 * max(4, pivot.shape[1]))
    fig_h = max(4, 0.18 * max(6, pivot.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=annotate,
        fmt=".3f",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": metric},
        annot_kws={"fontsize": 8},
    )
    tgt_label = target if target is not None else "ALL"
    ax.set_title(f"{metric} — {site} — {REGIME_LABELS.get(regime, regime)} — {tgt_label}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("Model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    safe_site = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(site))
    safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(regime))
    safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(target)) if target else "ALL"
    out_name = f"heatmap_{metric}_{safe_site}_{safe_regime}_{safe_target}.png"
    out_path = output_dir / out_name
    try:
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        logging.info(f"Saved heatmap: {out_path}")
        return out_path
    except Exception as e:
        logging.error(f"Failed to save heatmap to {out_path}: {e}")
        plt.close(fig)
        return None


# -------------------------
# New: Within-threshold heatmap
# -------------------------
def plot_individual_within_threshold(
    df: pd.DataFrame,
    site_col: str,
    site: str,
    regime: str,
    target_col: str | None,
    target: str | None,
    threshold: float,
    output_dir: Path,
    annotate: bool = True,
    cmap: str = "RdYlGn_r",
) -> Path | None:
    """
    Create and save a heatmap of fraction of samples with |error| <= threshold
    Rows = Model, Columns = Missingness_Pct, Values = fraction (0..1)
    """
    d = df.copy()

    # Filter target/site/regime
    if target is not None and target_col is not None:
        d = d[d[target_col].astype(str) == str(target)]
    if site_col not in d.columns:
        logging.error("Site column not present; cannot produce within-threshold heatmap.")
        return None
    d = d[d[site_col].astype(str) == str(site)]
    if d.empty:
        logging.info(f"No data for Target_Site={site}; skipping within-threshold.")
        return None
    d = d[d["MISSINGNESS_REGIMES"].astype(str) == str(regime)]
    if d.empty:
        logging.info(f"No data for Target_Site={site}, Regime={regime}; skipping within-threshold.")
        return None

    # Ensure abs_error available (compute if possible)
    if 'abs_error' not in d.columns and ('y_true' in d.columns and 'y_pred' in d.columns):
        try:
            d['abs_error'] = (d['y_pred'] - d['y_true']).abs()
            logging.info("Computed abs_error from y_pred/y_true for within-threshold computation.")
        except Exception:
            logging.debug("Could not compute abs_error from y_pred/y_true.")

    # Try common names if abs_error absent
    abs_col = None
    for cand in ['abs_error', 'AbsError', 'ABS_ERROR', 'absolute_error', 'AE']:
        if cand in d.columns:
            abs_col = cand
            break

    if abs_col is None:
        logging.info("No abs_error/y_pred+y_true present; cannot compute within-threshold heatmap.")
        return None

    # ensure Missingness_Pct numeric
    if "Missingness_Pct" not in d.columns:
        logging.info("Missingness_Pct not present; skipping within-threshold.")
        return None
    d["Missingness_Pct"] = pd.to_numeric(d["Missingness_Pct"], errors="coerce")
    d = d.dropna(subset=["Missingness_Pct"])
    if d.empty:
        logging.info("No numeric Missingness_Pct for this selection; skipping.")
        return None

    # Compute indicator and pivot
    try:
        d['within_thr'] = (d[abs_col] <= float(threshold)).astype(float)
        pivot = d.pivot_table(values='within_thr', index='Model', columns='Missingness_Pct', aggfunc='mean')
    except Exception as e:
        logging.warning(f"Failed to compute pivot for within-threshold: {e}")
        return None

    if pivot.empty:
        logging.info("Within-threshold pivot empty; skipping.")
        return None

    try:
        sorted_cols = sorted(pivot.columns, key=lambda x: float(x))
        pivot = pivot[sorted_cols]
    except Exception:
        pass

    try:
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    except Exception:
        pass

    # Plot fraction as percentage
    fig_w = max(6, 0.8 * max(4, pivot.shape[1]))
    fig_h = max(4, 0.18 * max(6, pivot.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=annotate, fmt=".0%", vmin=0, vmax=1,
                linewidths=0.3, linecolor="white", cbar_kws={"label": f"Fraction ≤ {threshold}"}, annot_kws={"fontsize": 8})

    tgt_label = target if target is not None else "ALL"
    ax.set_title(f"% within ±{threshold} — {site} — {REGIME_LABELS.get(regime, regime)} — {tgt_label}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("Model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()

    safe_site = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(site))
    safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(regime))
    safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(target)) if target else "ALL"
    out_name = f"heatmap_within{int(threshold)}_{safe_site}_{safe_regime}_{safe_target}.png"
    out_path = output_dir / out_name
    try:
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        logging.info(f"Saved within-threshold heatmap: {out_path}")
        return out_path
    except Exception as e:
        logging.error(f"Failed to save within-threshold heatmap to {out_path}: {e}")
        plt.close(fig)
        return None


# -------------------------
# Batch wrapper (updated)
# -------------------------
def batch_generate_heatmaps(df: pd.DataFrame, output_dir: Path, metric: str = DEFAULT_METRIC,
                            sites=None, regimes=None, targets=None, annotate=True,
                            add_within=False, threshold=DEFAULT_THRESHOLD):
    site_col = detect_site_column(df)
    if site_col is None:
        logging.error("No site column detected (expected 'Target_Site' or fallback). Aborting.")
        return []

    try:
        import config_spatial as cfg
        cfg_sites = getattr(cfg, "TARGET_SITES", None)
        cfg_targets = getattr(cfg, "TARGET_COLUMNS", None)
    except Exception:
        cfg_sites = None
        cfg_targets = None

    if sites is None:
        if cfg_sites:
            sites = list(cfg_sites)
        else:
            sites = DEFAULT_TARGET_SITES if DEFAULT_TARGET_SITES else sorted(df[site_col].dropna().unique().tolist())

    if regimes is None:
        regimes = REGIME_ORDER

    if targets is None:
        if cfg_targets:
            targets = list(cfg_targets)
        else:
            tcol = detect_target_column(df)
            if tcol:
                targets = sorted(df[tcol].dropna().unique().tolist())
            else:
                targets = DEFAULT_TARGET_COLUMNS if DEFAULT_TARGET_COLUMNS else [None]

    saved = []
    tgt_col = detect_target_column(df)

    # ensure abs_error computed if y_true/y_pred exist
    df = ensure_abs_error(df)

    for tgt in targets:
        for s in sites:
            for r in regimes:
                # main metric heatmap
                try:
                    out = plot_individual_heatmap(df, site_col, s, r, tgt_col, tgt, metric, output_dir, annotate=annotate)
                    if out:
                        saved.append(out)
                except Exception as e:
                    logging.warning(f"Failed metric heatmap for site={s}, regime={r}, target={tgt}: {e}")
                # within-threshold heatmap (if requested)
                if add_within:
                    try:
                        out2 = plot_individual_within_threshold(df, site_col, s, r, tgt_col, tgt, threshold, output_dir, annotate=annotate)
                        if out2:
                            saved.append(out2)
                    except Exception as e:
                        logging.warning(f"Failed within-threshold heatmap for site={s}, regime={r}, target={tgt}: {e}")
    logging.info(f"Batch completed: {len(saved)} heatmaps saved to {output_dir}")
    return saved


def plot_regime_average_summary(df: pd.DataFrame, output_dir: Path, metric: str = DEFAULT_METRIC):
    if metric not in df.columns:
        logging.warning(f"Metric '{metric}' not in dataframe; skipping regime-average summary.")
        return
    tgt_col = detect_target_column(df)
    targets = [None]
    if tgt_col is not None:
        targets = sorted(df[tgt_col].dropna().unique().tolist()) or [None]
    try:
        gv = pd.to_numeric(df[metric], errors="coerce")
        gv = gv[np.isfinite(gv)]
        vmin = float(gv.min()) if gv.size else None
        vmax = float(gv.max()) if gv.size else None
    except Exception:
        vmin = vmax = None
    for tgt in targets:
        d = df.copy()
        if tgt is not None and tgt_col is not None:
            d = d[d[tgt_col].astype(str) == str(tgt)]
        try:
            grp = d.groupby(["Model", "MISSINGNESS_REGIMES"])[metric].mean().unstack(fill_value=np.nan)
        except Exception as e:
            logging.warning(f"Failed to compute regime averages for target={tgt}: {e}")
            continue
        if grp.empty:
            logging.info(f"No data to plot for regime-average summary target={tgt}; skipping.")
            continue
        cols = [c for c in REGIME_ORDER if c in grp.columns]
        if not cols:
            cols = list(grp.columns)
        grp = grp[cols]
        try:
            grp = grp.loc[grp.mean(axis=1).sort_values().index]
        except Exception:
            pass
        fig_h = max(4, 0.25 * max(6, grp.shape[0]))
        fig_w = max(6, 1.5 * len(grp.columns))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(grp, ax=ax, cmap="RdYlGn_r", annot=True, fmt=".3f", vmin=vmin, vmax=vmax,
                    cbar_kws={"label": metric}, linewidths=0.3)
        tgt_label = tgt if tgt is not None else "ALL"
        ax.set_title(f"{metric}: Models × Regime (averaged over missingness & sites) — Target: {tgt_label}")
        ax.set_xlabel("Missingness Regime")
        ax.set_ylabel("Model")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="center", fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)
        plt.tight_layout()
        safe_tgt = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(tgt_label))
        out = output_dir / f"Regime_Average_Heatmap_{metric}_{safe_tgt}.png"
        try:
            plt.savefig(out, dpi=300)
            plt.close(fig)
            logging.info(f"Saved regime-average heatmap: {out}")
        except Exception as e:
            logging.error(f"Failed to save regime-average heatmap to {out}: {e}")


# -------------------------
# Main
# -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate per-site × per-regime × per-target heatmaps (with optional within-threshold plots).")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help=f"Results dir (default): {DEFAULT_RESULTS_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Output dir (default): {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC, help=f"Metric to plot (default): {DEFAULT_METRIC}")
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated Target_Site values (optional)")
    parser.add_argument("--regimes", type=str, default=None, help="Comma-separated regimes (optional)")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated targets (optional)")
    parser.add_argument("--annot", action="store_true", help="Annotate heatmap cells with values")
    parser.add_argument("--within", action="store_true", help="Also generate %within-threshold heatmaps")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Threshold for within-plot (default {DEFAULT_THRESHOLD})")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Results dir: {results_dir}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Metric: {args.metric}")
    logging.info(f"Within-threshold enabled: {args.within} (threshold={args.threshold})")

    df = load_aggregated(results_dir)
    if df is None:
        logging.error("No aggregated data available; exiting.")
        return 1

    df = standardize_columns(df)

    sites = [s.strip() for s in args.sites.split(",")] if args.sites else None
    regimes = [r.strip() for r in args.regimes.split(",")] if args.regimes else None
    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None

    # Within-threshold plotting requires per-sample actual vs imputed predictions.
    # For now, heatmaps use aggregated metrics only; skip per-sample within-threshold unless
    # an explicit per-sample loader is available.
    if args.within:
        logging.warning("Within-threshold plots require per-sample Actual vs Imputed files in Imputed_Results; skipped.")

    metrics = [args.metric]
    if 'MAE' in df.columns and 'MAE' not in metrics:
        metrics.append('MAE')

    for metric in metrics:
        logging.info(f"Generating plots for metric: {metric}")
        # Generate heatmaps from aggregated metrics only (do not require per-sample imputed/actual)
        batch_generate_heatmaps(df=df, output_dir=output_dir, metric=metric, sites=sites, regimes=regimes, targets=targets, annotate=args.annot, add_within=False, threshold=args.threshold)
        try:
            plot_regime_average_summary(df=df, output_dir=output_dir, metric=metric)
        except Exception as e:
            logging.warning(f"Failed to generate regime-average summary plot for metric {metric}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())