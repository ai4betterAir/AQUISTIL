#!/usr/bin/env python3
"""
research_error_distribution.py

Create an error-frequency bar plot (histogram-style) comparing models.
For each model we compute the error = Imputed - Actual (or |error|) and
plot frequency across bins. By default plots median-absolute-error bar chart,
but this variant produces grouped bar charts of error vs frequency.

Run without arguments (defaults embedded):
    python research_error_distribution.py

Defaults:
 - results_dir: /mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V21_final
 - site: autodetected (first site found in Imputed_Results or aggregated CSV)
 - target: PM2.5
 - output_dir: ./research_figs
 - absolute errors (|error|): True
 - bins: 20

CLI examples:
    python research_error_distribution.py
    python research_error_distribution.py --site CHULLORA --bins 30 --signed
    python research_error_distribution.py --results_dir /path/to/results --per_model --bins 25
"""
from pathlib import Path
import argparse
import logging
import sys
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

# Defaults
DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V21_final"
DEFAULT_TARGET = "PM2.5"
DEFAULT_OUTPUT_DIR = "./research_figs"


# -------------------------
# Helpers (file discovery / parsing)
# -------------------------
def find_imputed_files(results_dir: Path, site: str, target: str):
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
    if not files:
        for p in imputed_dir.glob("*.csv"):
            low = p.name.lower()
            if site.lower() in low and target.lower().split(".")[0] in low:
                files.append(p)
    return sorted(files)


def load_per_sample_from_file(fp: Path):
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        logging.debug("Failed to read %s: %s", fp, e)
        return None

    cols = list(df.columns)

    # heuristics for actual/imputed columns
    obs_col = next((c for c in cols if c.lower().startswith("actual_") or c.lower() == "actual" or ("actual" in c.lower() and any(k in c.lower() for k in ["pm2", "pm10", "value"])) ), None)
    imp_col = next((c for c in cols if c.lower().startswith("imputed_") or "imputed" in c.lower() or "pred" in c.lower()), None)

    if obs_col is None:
        for c in cols:
            if re.fullmatch(r"(?i)pm2[._]?5", c.replace(" ", "")):
                obs_col = c
                break

    if imp_col is None:
        for c in cols:
            if "imputed" in c.lower() and any(tok in c.lower() for tok in ["pm2", "pm10", "imputed"]):
                imp_col = c
                break

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
    rows = []
    for f in files:
        df = load_per_sample_from_file(f)
        if df is None:
            continue
        stem = f.stem
        parts = stem.split("_")
        model_name = "_".join(parts[2:]) if len(parts) >= 3 else stem
        model_name = re.sub(r"(?i)_?imputed$", "", model_name)

        df2 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Actual", "Imputed"])
        if df2.empty:
            continue
        errs = (df2["Imputed"].to_numpy() - df2["Actual"].to_numpy()).astype(float)
        for e in errs:
            rows.append({"Model": model_name, "Error": float(e)})
    if not rows:
        return pd.DataFrame(columns=["Model", "Error"])
    return pd.DataFrame(rows)


def discover_site_from_imputed(results_dir: Path):
    imputed_dir = results_dir / "Imputed_Results"
    sites = []
    if imputed_dir.exists():
        for p in imputed_dir.glob("*.csv"):
            stem = p.stem
            parts = stem.split("_")
            if parts:
                sites.append(parts[0])
    sites = sorted(set([s for s in sites if s]))
    if sites:
        return sites
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


# -------------------------
# Plot: grouped bar of error frequency per model
# -------------------------
def plot_error_frequency_grouped(err_df: pd.DataFrame, out_path: Path, bins: int = 20, absolute: bool = True, normalize: bool = False, title: str | None = None):
    """
    err_df: DataFrame with columns ['Model','Error'] (signed errors)
    absolute: if True use abs(errors) else signed errors
    bins: number of bins to use (same bins across models)
    normalize: if True, plot frequencies normalized to proportions
    """
    if err_df.empty:
        logging.error("No error data provided to plot.")
        return False

    working = err_df.copy()
    working["Value"] = working["Error"].abs() if absolute else working["Error"]

    # compute common bins (include 0-centered for signed if not absolute)
    vals = working["Value"].dropna().values
    if len(vals) == 0:
        logging.error("No finite error values.")
        return False

    try:
        bin_edges = np.histogram_bin_edges(vals, bins=bins)
    except Exception:
        bin_edges = np.linspace(np.nanmin(vals), np.nanmax(vals), bins + 1)

    models = working["Model"].unique().tolist()
    models_sorted = sorted(models, key=lambda m: working.loc[working["Model"] == m, "Value"].median() if len(working.loc[working["Model"] == m, "Value"])>0 else np.inf)

    # compute histogram counts per model
    hist_df = []
    for m in models_sorted:
        arr = working.loc[working["Model"] == m, "Value"].dropna().values
        if arr.size == 0:
            counts = np.zeros(len(bin_edges) - 1, dtype=float)
        else:
            counts, _ = np.histogram(arr, bins=bin_edges)
        if normalize and arr.size > 0:
            counts = counts / arr.size
        hist_df.append((m, counts))

    # prepare grouped bar positions
    n_bins = len(bin_edges) - 1
    n_models = len(models_sorted)
    width = 0.8 / max(1, n_models)  # total width per bin is 0.8
    x = np.arange(n_bins)

    fig_w = max(8, n_bins * 0.4)
    fig_h = max(4, n_models * 0.2 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    cmap = plt.get_cmap("tab10")
    for i, (m, counts) in enumerate(hist_df):
        offsets = x - 0.4 + (i + 0.5) * width
        ax.bar(offsets, counts, width=width, label=m, color=cmap(i % cmap.N), align="center", alpha=0.9)

    # x labels: show bin centers or ranges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # To label, use nicer formatting; if absolute True may want integer ticks
    tick_positions = x
    tick_labels = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if absolute:
            tick_labels.append(f"{lo:.1f}-{hi:.1f}")
        else:
            tick_labels.append(f"{lo:.2f}-{hi:.2f}")
    ax.set_xticks(tick_positions)
    # Use shorter tick labels if many bins
    if n_bins <= 12:
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    else:
        # show only every nth label
        step = max(1, n_bins // 12)
        labels_shown = [lbl if (idx % step == 0) else "" for idx, lbl in enumerate(tick_labels)]
        ax.set_xticklabels(labels_shown, rotation=45, ha="right", fontsize=8)

    ax.set_xlim(-0.6, n_bins - 0.4)
    ax.set_xlabel("Error bin (absolute)" if absolute else "Error bin (signed)")
    ax.set_ylabel("Proportion" if normalize else "Frequency")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved grouped error-frequency bar plot: %s", out_path)
    return True


# -------------------------
# Main
# -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Error vs frequency grouped bar plot (defaults embedded)")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="Root results dir (embedded default)")
    parser.add_argument("--site", default=None, help="Site token (autodetected if omitted)")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target variable (default: PM2.5)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output PNG")
    parser.add_argument("--bins", type=int, default=20, help="Number of error bins")
    parser.add_argument("--signed", action="store_true", help="Plot signed errors (Imputed - Actual) instead of absolute errors")
    parser.add_argument("--normalize", action="store_true", help="Normalize frequencies to proportions")
    parser.add_argument("--per_model", action="store_true", help="Also save individual per-model bar plots")
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
        logging.error("No per-sample imputed files found for site=%s target=%s under Imputed_Results.", site, args.target)
        sys.exit(1)

    err_df = aggregate_errors(files)
    if err_df.empty:
        logging.error("No per-sample error rows could be extracted. Exiting.")
        sys.exit(1)

    out_path = Path(args.output_dir) / f"error_frequency_grouped_{site}_{args.target.replace('.', '')}.png"
    title = f"{'Signed' if args.signed else 'Absolute'} error frequency per Model — {site} — {args.target}"
    ok = plot_error_frequency_grouped(err_df, out_path, bins=args.bins, absolute=not args.signed, normalize=args.normalize, title=title)

    if not ok:
        logging.error("Failed to create grouped error-frequency plot.")
        sys.exit(1)

    if args.per_model:
        per_dir = Path(args.output_dir) / "per_model"
        per_dir.mkdir(parents=True, exist_ok=True)
        for m in sorted(err_df["Model"].unique()):
            sub = err_df[err_df["Model"] == m].copy()
            sub["Value"] = sub["Error"].abs() if not args.signed else sub["Error"]
            # compute histogram
            counts, edges = np.histogram(sub["Value"].dropna().values, bins=args.bins)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=(edges[1]-edges[0]) * 0.9, align="center")
            ax.set_xlabel("Absolute error" if not args.signed else "Error")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{m} — error frequency")
            plt.tight_layout()
            fp = per_dir / f"{m}_error_frequency_{site}.png"
            fig.savefig(fp, dpi=300)
            plt.close(fig)
            logging.info("Saved per-model error histogram: %s", fp)


if __name__ == "__main__":
    main()