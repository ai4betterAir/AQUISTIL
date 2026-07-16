#!/usr/bin/env python3
"""
research_heatmap.py

Creates publication-ready heatmaps of a chosen metric (Models × Missingness levels)
per Site × Regime. This variant embeds sensible defaults so you can run it
directly without CLI args:

    python /path/to/research_heatmap.py

Defaults (can be overridden with CLI args):
 - results_dir: /mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V21_final
 - metric: RMSE
 - annotate: True
 - output_dir: ./research_figs

You can still pass flags to override defaults, e.g.:
  python research_heatmap.py --results_dir /path/to/results --metric MAE --no-annotate
"""
from pathlib import Path
from typing import Optional
import argparse
import logging
import re
import sys
import os

# Use Agg backend for headless environments
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sns.set_theme(style="whitegrid")

# Default results directory embedded here
DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V21_final"


def load_aggregated(results_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load an aggregated CSV (all_results_summary.csv) if present, otherwise
    scan per-model /regime/metrics folders and assemble a DataFrame.
    """
    agg_fp = results_dir / "all_results_summary.csv"
    if agg_fp.exists():
        logging.info(f"Loading aggregated CSV: {agg_fp}")
        try:
            df = pd.read_csv(agg_fp)
            return standardize_columns(df)
        except Exception as e:
            logging.warning(f"Failed to read aggregated CSV: {e}")
    # fallback: scan metrics files
    frames = []
    if not results_dir.exists():
        logging.error("Results directory does not exist: %s", results_dir)
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
                    dfm["Model"] = model_name
                    dfm["MISSINGNESS_REGIMES"] = regime
                    dfm["StudySite"] = site_guess
                    frames.append(dfm)
                except Exception as e:
                    logging.debug(f"Unable to read {csv}: {e}")
    if not frames:
        logging.error("No metrics CSV files discovered while scanning results_dir.")
        return None
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = standardize_columns(df)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column names so downstream code can rely on standardized keys.
    """
    df = df.copy()
    # Collapse duplicate column names by taking the first non-null across duplicates
    if not df.columns.is_unique:
        try:
            cols = list(df.columns)
            unique = []
            for c in cols:
                if c not in unique:
                    unique.append(c)
            out = {}
            for name in unique:
                members = [c for c in cols if c == name]
                if len(members) == 1:
                    out[name] = df[members[0]]
                else:
                    # prefer the rightmost (later) column when duplicate names exist
                    try:
                        out[name] = df.loc[:, members].ffill(axis=1).iloc[:, -1]
                    except Exception:
                        out[name] = df.loc[:, members].bfill(axis=1).iloc[:, 0]
            df = pd.DataFrame(out)
        except Exception:
            df = df.copy()
    rename_map = {
        "Root Mean Squared Error (RMSE)": "RMSE",
        "Mean Absolute Error (MAE)": "MAE",
        "Missingness_Level": "Missingness_Pct",
        "Missingness": "Missingness_Pct",
        "Missingness_Pct": "Missingness_Pct",
        "Missingness_Regime": "MISSINGNESS_REGIMES",
        "Missingness_Regimes": "MISSINGNESS_REGIMES",
        "Missingness Regime": "MISSINGNESS_REGIMES",
    }
    df = df.rename(columns=rename_map)
    # ensure Missingness_Pct is numeric and expressed as percent
    if "Missingness_Pct" in df.columns:
        try:
            df["Missingness_Pct"] = pd.to_numeric(df["Missingness_Pct"], errors="coerce")
            if df["Missingness_Pct"].max() <= 1.0:
                df["Missingness_Pct"] = df["Missingness_Pct"] * 100.0
        except Exception:
            pass
    if "MISSINGNESS_REGIMES" not in df.columns:
        df["MISSINGNESS_REGIMES"] = df.get("Missingness_Regime", "random")
    if "StudySite" not in df.columns and "Target_Site" in df.columns:
        df["StudySite"] = df["Target_Site"]
    return df


def discover_sites_regimes_from_agg(df: pd.DataFrame):
    """
    Robustly discover StudySite and MISSINGNESS_REGIMES values from an aggregated
    DataFrame. Handles cases where df[col] returns a DataFrame (duplicate columns)
    by collapsing columns into a single value list.
    """
    def _collect_unique_values(df_local: pd.DataFrame, colname: str):
        if colname not in df_local.columns:
            return []
        col = df_local[colname]
        # If selecting the column yielded a DataFrame (duplicate column names),
        # collapse all columns into a single Series of values.
        if isinstance(col, pd.DataFrame):
            try:
                combined = pd.concat([col[c] for c in col.columns], axis=0, ignore_index=True)
            except Exception:
                combined = col.stack().reset_index(drop=True)
        else:
            combined = col
        # Ensure we pass a 1-D sequence to Series: flatten arrays / DataFrame values if needed
        try:
            if isinstance(combined, pd.DataFrame):
                arr = combined.values.ravel()
            elif isinstance(combined, pd.Series):
                arr = combined.values.ravel()
            else:
                arr = np.asarray(combined)
                if arr.ndim > 1:
                    arr = arr.ravel()
            ser = pd.Series(arr).dropna()
        except Exception:
            # fallback: coerce to string list
            ser = pd.Series([str(x) for x in combined]).dropna()
        # Convert to python list of strs, dedup and sort
        vals = sorted({str(v).strip() for v in ser.unique() if str(v).strip()})
        return vals

    sites = _collect_unique_values(df, "StudySite")
    regimes = _collect_unique_values(df, "MISSINGNESS_REGIMES")
    return sites, regimes


def plot_heatmap_for(site: str, regime: str, df: pd.DataFrame, metric: str, out_dir: Path, annotate=False, cmap="RdYlGn_r"):
    """
    Produce and save a heatmap for a single site/regime.
    """
    # Build boolean masks using numpy arrays to avoid pandas alignment issues
    # If selecting a column returned a DataFrame (duplicate column names),
    # collapse to a single Series first by taking first non-null across duplicates.
    study_col = df["StudySite"]
    if isinstance(study_col, pd.DataFrame):
        try:
            study_series = study_col.bfill(axis=1).iloc[:, 0]
        except Exception:
            study_series = study_col.stack().reset_index(drop=True)
    else:
        study_series = study_col

    reg_col = df["MISSINGNESS_REGIMES"]
    if isinstance(reg_col, pd.DataFrame):
        try:
            regime_series = reg_col.bfill(axis=1).iloc[:, 0]
        except Exception:
            regime_series = reg_col.stack().reset_index(drop=True)
    else:
        regime_series = reg_col

    study_arr = np.asarray(study_series.astype(str).str.upper().values).ravel()
    regime_arr = np.asarray(regime_series.astype(str).values).ravel()
    try:
        mask = (study_arr == site.upper()) & (regime_arr == regime)
    except Exception:
        # fallback to elementwise comparison
        mask = np.array([(s == site.upper() and r == regime) for s, r in zip(study_arr, regime_arr)])
    # Debug mask lengths for troubleshooting
    try:
        logging.debug(f"Mask lengths: df={len(df)}, study_arr={getattr(study_arr,'shape',None)}, regime_arr={getattr(regime_arr,'shape',None)}")
        logging.debug(f"Mask true count: {np.count_nonzero(mask)}")
    except Exception:
        pass
    sel = df[mask]
    if sel.empty:
        logging.info(f"No rows for site={site}, regime={regime}")
        return None
        # Debugging helper: log details for CHULLORA to diagnose missing numeric data
        if site.upper() == 'CHULLORA':
            try:
                logging.info(f"DEBUG CHULLORA: sel_rows={len(sel)}, columns={list(sel.columns)[:20]}")
                logging.info(f"DEBUG CHULLORA: Missingness_Pct head: {sel.get('Missingness_Pct').head(8).tolist() if 'Missingness_Pct' in sel.columns else 'N/A'}")
                # show metric candidates presence
                vals = {c: sel[c].head(5).tolist() for c in ['RMSE','Mean Absolute Error (MAE)','MAE'] if c in sel.columns}
                logging.info(f"DEBUG CHULLORA: metric sample vals: {vals}")
            except Exception:
                pass
    # Try several metric name variations
    metric_col = None
    for cand in (metric, metric.upper(), metric.replace(" ", ""), "RMSE", "MAE"):
        if cand in sel.columns:
            metric_col = cand
            break
    if metric_col is None:
        # fuzzy match fallback
        for c in sel.columns:
            try:
                if re.search(re.escape(metric).lower(), c.lower()):
                    metric_col = c
                    break
            except Exception:
                continue
    if metric_col is None:
        logging.warning(f"Metric '{metric}' not found in data for site={site}, regime={regime}. Columns: {list(sel.columns)[:20]}")
        return None

    # pivot models x missingness
    # Ensure metric and missingness columns are numeric where possible
    # Robustly fill Missingness_Pct from 'Missingness' when Missingness_Pct is absent or NaN
    if "Missingness_Pct" not in sel.columns:
        sel["Missingness_Pct"] = sel.get("Missingness", np.nan)
    # If Missingness_Pct exists but many values are NaN, try to populate from 'Missingness'
    try:
        miss_col = pd.to_numeric(sel.get("Missingness", pd.Series([np.nan]*len(sel))), errors="coerce")
        miss_pct_col = pd.to_numeric(sel.get("Missingness_Pct", pd.Series([np.nan]*len(sel))), errors="coerce")
        # If missingness appears as fraction (<=1) convert to percent
        try:
            if miss_col.max() <= 1.0:
                miss_col = miss_col * 100.0
        except Exception:
            pass
        # fill Missingness_Pct NAs from computed miss_col
        sel["Missingness_Pct"] = miss_pct_col.fillna(miss_col)
    except Exception:
        # last resort: leave as-is
        pass

    try:
        sel[metric_col] = pd.to_numeric(sel[metric_col], errors="coerce")
    except Exception:
        pass
    try:
        sel["Missingness_Pct"] = pd.to_numeric(sel["Missingness_Pct"], errors="coerce")
    except Exception:
        pass

    sel = sel.dropna(subset=[metric_col, "Missingness_Pct"])
    if sel.empty:
        logging.info(f"No numeric metric/levels for site={site}, regime={regime}")
        return None

    pivot = sel.pivot_table(values=metric_col, index="Model", columns="Missingness_Pct", aggfunc="mean")
    if pivot.empty:
        logging.info("Pivot empty after grouping")
        return None

    # sort columns numerically
    try:
        cols_sorted = sorted(pivot.columns, key=float)
        pivot = pivot[cols_sorted]
    except Exception:
        pass

    # sort rows by mean performance
    try:
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    except Exception:
        pass

    fig_w = max(6, 0.7 * max(6, pivot.shape[1]))
    fig_h = max(4, 0.18 * max(6, pivot.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=annotate, fmt=".3f", linewidths=0.3, cbar_kws={"label": metric_col})
    ax.set_title(f"{metric_col} — {site} — {regime}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("Model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"heatmap_{metric_col}_{site}_{regime}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved heatmap: {out_path}")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Research heatmap generator (defaults embedded)")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="Root results dir (embedded default)")
    parser.add_argument("--metric", default="RMSE", help="Metric to plot (default RMSE)")
    parser.add_argument("--sites", default=None, help="Comma-separated site list (or leave empty for all)")
    parser.add_argument("--regimes", default=None, help="Comma-separated regimes (or leave empty for all)")
    parser.add_argument("--output_dir", default="./research_figs", help="Output directory for saved heatmaps")
    # Annotate default True but allow user to pass --no-annotate
    parser.add_argument("--annotate", dest="annotate", action="store_true", help="Annotate heatmap cells (default)")
    parser.add_argument("--no-annotate", dest="annotate", action="store_false", help="Do not annotate heatmap cells")
    parser.set_defaults(annotate=True)
    args = parser.parse_args(argv)

    results_path = Path(args.results_dir)
    if not results_path.exists():
        logging.error("Results directory does not exist: %s", results_path)
        sys.exit(1)

    df = load_aggregated(results_path)
    if df is None:
        logging.error("No aggregated data available. Exiting.")
        sys.exit(1)

    # determine sites/regimes selection
    if args.sites:
        sites = [s.strip() for s in args.sites.split(",")]
    else:
        sites, _ = discover_sites_regimes_from_agg(df)

    if args.regimes:
        regimes = [r.strip() for r in args.regimes.split(",")]
    else:
        _, regimes = discover_sites_regimes_from_agg(df)

    if not sites:
        logging.error("No sites discovered to plot.")
        sys.exit(1)
    if not regimes:
        logging.error("No regimes discovered to plot.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    for site in sites:
        for regime in regimes:
            plot_heatmap_for(site, regime, df, args.metric, out_dir, annotate=args.annotate)


if __name__ == "__main__":
    main()