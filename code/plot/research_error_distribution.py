#!/usr/bin/env python3
"""
research_error_distribution.py

Plot per-model error distributions (violin + box) using per-sample Imputed_Results
(or fallback to per-run aggregated metrics). Script embeds sensible defaults so you
can run it directly without CLI args:

    python /path/to/research_error_distribution.py

Defaults:
 - results_dir: /mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V21_final
 - site: autodetected (first site found in Imputed_Results or aggregated CSV)
 - target: PM2.5
 - output_dir: ./research_figs

You can still override via CLI:
  python research_error_distribution.py --results_dir /path/to/results --site CHULLORA --target PM2.5
"""
from pathlib import Path
import argparse
import logging
import sys
import os
import re

# Use Agg backend for headless servers
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sns.set_theme(style="whitegrid")

# Embedded default results directory
DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V21_final"
DEFAULT_TARGET = "PM2.5"
DEFAULT_OUTPUT_DIR = "./research_figs"


def find_imputed_files(results_dir: Path, site: str, target: str):
    """
    Search Imputed_Results for files that include site and target tokens.
    Returns list of Path objects.
    """
    imputed_dir = results_dir / "Imputed_Results"
    if not imputed_dir.exists():
        logging.warning("Imputed_Results directory not found at: %s", imputed_dir)
        return []
    target_token = target.lower().replace(".", "").replace(" ", "")
    files = []
    for p in imputed_dir.glob("*.csv"):
        low = p.name.lower()
        if site.lower() in low and target_token in low:
            files.append(p)
    # if none found, be permissive: site present and target substring present
    if not files:
        for p in imputed_dir.glob("*.csv"):
            low = p.name.lower()
            if site.lower() in low and target.lower().split(".")[0] in low:
                files.append(p)
    return sorted(files)


def load_per_sample_from_file(fp: Path):
    """
    Load a per-site_target imputed CSV and return a DataFrame with columns:
    ['DateTime','Actual','Imputed','Comments'] if possible. Returns None if parsing fails.
    """
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        logging.debug("Failed to read %s: %s", fp, e)
        return None

    cols = list(df.columns)

    # Try to locate Actual and Imputed columns heuristically
    obs_col = next((c for c in cols if c.lower().startswith("actual_") or c.lower() == "actual" or ("actual" in c.lower() and any(k in c.lower() for k in ["pm2", "pm10", "value"])) ), None)
    imp_col = next((c for c in cols if c.lower().startswith("imputed_") or "imputed" in c.lower() or "pred" in c.lower()), None)

    # Additional heuristics
    if obs_col is None:
        # look for column that matches target name patterns (e.g., PM2.5 or PM25)
        for c in cols:
            if re.fullmatch(r"(?i)pm2[._]?5", c.replace(" ", "")):
                obs_col = c
                break

    if imp_col is None:
        # try columns containing target + 'imputed' variation
        for c in cols:
            if "imputed" in c.lower() and any(tok in c.lower() for tok in ["pm2", "pm10", "imputed"]):
                imp_col = c
                break

    # Fallback: choose first two numeric columns
    if obs_col is None or imp_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            obs_col, imp_col = numeric_cols[0], numeric_cols[1]
        else:
            logging.debug("Could not infer Actual/Imputed columns in %s", fp)
            return None

    try:
        out = pd.DataFrame({
            "DateTime": df.get("DateTime") if "DateTime" in df.columns else df.index,
            "Actual": pd.to_numeric(df[obs_col], errors="coerce"),
            "Imputed": pd.to_numeric(df[imp_col], errors="coerce"),
            "Comments": df.get("Comments")
        })
        return out
    except Exception as e:
        logging.debug("Error normalizing columns for %s: %s", fp, e)
        return None


def aggregate_errors(files):
    """
    Given a list of per-model per-site_target files, return a DataFrame with rows:
    {'Model', 'Error'} where Error = Imputed - Actual
    """
    rows = []
    for f in files:
        df = load_per_sample_from_file(f)
        if df is None:
            continue
        # Derive model name from filename: <SITE>_<TARGET>_<MODEL>_imputed.csv (best-effort)
        stem = f.stem
        parts = stem.split("_")
        model_name = None
        if len(parts) >= 3:
            # model token is everything after <SITE>_<TARGET>_
            model_name = "_".join(parts[2:])
            # strip trailing '_imputed' if present
            model_name = re.sub(r"(?i)_?imputed$", "", model_name)
        else:
            model_name = stem

        df2 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Actual", "Imputed"])
        if df2.empty:
            continue
        errs = (df2["Imputed"].to_numpy() - df2["Actual"].to_numpy()).astype(float)
        for e in errs:
            rows.append({"Model": model_name, "Error": float(e)})
    if not rows:
        return pd.DataFrame(columns=["Model", "Error"])
    return pd.DataFrame(rows)


def plot_violin(err_df: pd.DataFrame, out_path: Path, title=None):
    """
    Create violin + boxplot of errors per model and save to out_path.
    """
    if err_df.empty:
        logging.error("No per-sample error data found to plot.")
        return False

    plt.figure(figsize=(10, 6))
    # order models by median absolute error
    order = err_df.groupby("Model")["Error"].apply(lambda s: np.nanmedian(np.abs(s))).sort_values().index.tolist()
    sns.violinplot(x="Model", y="Error", data=err_df, order=order, inner=None, scale="width", cut=0)
    sns.boxplot(x="Model", y="Error", data=err_df, order=order, width=0.12, showcaps=True, boxprops={'facecolor':'none'}, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
    plt.ylabel("Imputed - Actual")
    plt.xlabel("Model")
    if title:
        plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info("Saved error distribution plot: %s", out_path)
    return True


def discover_site_from_imputed(results_dir: Path):
    """
    If site not provided, attempt to discover available sites from Imputed_Results
    directory or from aggregated all_results_summary.csv StudySite column.
    Returns list of unique site tokens (may be empty).
    """
    imputed_dir = results_dir / "Imputed_Results"
    sites = []
    if imputed_dir.exists():
        for p in imputed_dir.glob("*.csv"):
            stem = p.stem
            parts = stem.split("_")
            if parts:
                sites.append(parts[0])
    # dedupe
    sites = sorted(set([s for s in sites if s]))
    if sites:
        return sites
    # Fallback to aggregated CSV
    agg_fp = results_dir / "all_results_summary.csv"
    if agg_fp.exists():
        try:
            df = pd.read_csv(agg_fp)
            if "StudySite" in df.columns:
                vals = df["StudySite"].dropna().astype(str).unique().tolist()
                vals = sorted([v for v in vals if v])
                return vals
        except Exception:
            pass
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description="Research error distribution (defaults embedded)")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="Root results dir (embedded default)")
    parser.add_argument("--site", default=None, help="Site token (autodetected if omitted)")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target variable (default: PM2.5)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output PNG")
    args = parser.parse_args(argv)

    results_path = Path(args.results_dir)
    if not results_path.exists():
        logging.error("Results directory does not exist: %s", results_path)
        sys.exit(1)

    site = args.site
    if site is None:
        candidates = discover_site_from_imputed(results_path)
        if not candidates:
            logging.error("No site could be autodetected from Imputed_Results or aggregated CSV. Provide --site explicitly.")
            sys.exit(1)
        site = candidates[0]
        logging.info("Autodetected site: %s (use --site to override)", site)

    files = find_imputed_files(results_path, site, args.target)
    if not files:
        logging.warning("No per-sample imputed files found for site=%s target=%s under Imputed_Results.", site, args.target)
        logging.info("Attempting to fall back to aggregated per-run metrics to create a simple boxplot per model.")
        # fallback: try to build a per-model metric dataframe from aggregated CSV
        agg_fp = results_path / "all_results_summary.csv"
        if not agg_fp.exists():
            logging.error("Aggregated CSV not found at %s. Cannot produce fallback plot.", agg_fp)
            sys.exit(1)
        try:
            dfm = pd.read_csv(agg_fp)
            dfm = dfm.rename(columns={
                "Root Mean Squared Error (RMSE)": "RMSE",
                "Mean Absolute Error (MAE)": "MAE",
                "Missingness": "Missingness_Pct",
                "Missingness_Level": "Missingness_Pct",
                "Missingness_Regime": "MISSINGNESS_REGIMES"
            })
            metric_col = "RMSE" if "RMSE" in dfm.columns else next((c for c in dfm.columns if "rmse" in c.lower()), None)
            if metric_col is None:
                logging.error("No RMSE-like column found in aggregated CSV. Cannot produce fallback.")
                sys.exit(1)
            # Filter by site if possible
            if "StudySite" in dfm.columns:
                df_site = dfm[dfm["StudySite"].astype(str).str.upper() == site.upper()].copy()
            else:
                df_site = dfm.copy()
            if df_site.empty:
                logging.error("Aggregated CSV has no rows for site=%s; cannot produce plot.", site)
                sys.exit(1)
            # Prepare DataFrame with Model | Metric
            plot_df = df_site[["Model", metric_col]].dropna()
            if plot_df.empty:
                logging.error("No per-model metric rows available for plotting fallback.")
                sys.exit(1)
            # Create boxplot (per-run distribution)
            plt.figure(figsize=(10, 6))
            order = plot_df.groupby("Model")[metric_col].median().sort_values().index.tolist()
            sns.boxplot(x="Model", y=metric_col, data=plot_df, order=order)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{metric_col} per Model — {site} (fallback)")
            out_fp = Path(args.output_dir) / f"boxplot_{metric_col}_{site}.png"
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_fp, dpi=300)
            plt.close()
            logging.info("Saved fallback boxplot to %s", out_fp)
            return
        except Exception as e:
            logging.error("Failed to produce fallback plot: %s", e)
            sys.exit(1)

    # Aggregate per-sample errors
    err_df = aggregate_errors(files)
    if err_df.empty:
        logging.error("No error rows could be extracted from imputed files. Exiting.")
        sys.exit(1)

    out_path = Path(args.output_dir) / f"error_distribution_{site}_{args.target.replace('.', '')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_title = f"Error distribution — {site} — {args.target}"
    ok = plot_violin(err_df, out_path, title=plot_title)
    if not ok:
        logging.error("Failed to create error distribution plot.")
        sys.exit(1)


if __name__ == "__main__":
    main()