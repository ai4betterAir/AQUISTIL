#!/usr/bin/env python3
"""
CDF_from_aggregated.py

Produce ECDF (empirical CDF) overlay plots from aggregated metrics (no per-sample data required).

Modes:
 - site: produce plots per Target_Site × Regime. For each missingness level, ECDF of metric values across Models.
 - model: produce plots per Model × Regime. For each missingness level, ECDF of metric values across Target_Site.

Input:
 - results_dir containing all_results_summary.csv (default embedded)
Output:
 - PNGs saved to <output_dir>/CDF_Agg_by_Site/ or .../CDF_Agg_by_Model/

Usage examples:
  python CDF_from_aggregated.py --results_dir /path/to/results --output_dir ./agg_plots --metric RMSE --mode site --values CHULLORA,LIVERPOOL
  python CDF_from_aggregated.py --metric RMSE --mode model --values TF_BRITS --regimes random,short_gap

If --values omitted, script iterates all discovered sites/models.
"""
from pathlib import Path
import argparse
import logging
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Defaults
DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V19_final"
DEFAULT_OUTPUT_DIR = "./agg_heatmaps"
REGIME_ORDER = ["random", "short_gap", "medium_gap", "long_gap", "event"]
PREFERRED_LEVELS = [10, 20, 30, 50]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and collapse duplicate columns safely.

    - Strips whitespace, removes trailing .1/.2 suffixes, collapses duplicate logical columns
      by taking the first non-null value across duplicates.
    - Ensures df.columns are unique to avoid pandas reindex errors in later boolean indexing.
    """
    import re
    # Defensive copy
    df = df.copy()

    # Normalize names (strip spaces, remove trailing numbered suffixes added by pandas)
    try:
        new_cols = []
        for c in df.columns:
            s = str(c)
            s = re.sub(r"\s+", " ", s).strip()
            s = re.sub(r"\.\d+$", "", s)        # remove trailing .1, .2, ...
            s = re.sub(r"\(duplicate\)$", "", s).strip()
            new_cols.append(s)
        df.columns = new_cols
    except Exception:
        pass

    # Collapse exact duplicate columns by name: take first non-null across duplicates
    if not df.columns.is_unique:
        try:
            # group columns by name and collapse
            groups = {}
            for col in df.columns:
                groups.setdefault(col, []).append(col)
            # Build new dataframe by taking first non-null value across duplicates
            result = {}
            for name, members in groups.items():
                if len(members) == 1:
                    result[name] = df[members[0]]
                else:
                    # select all columns with this logical name (they may be repeated due to read_csv)
                    group_df = df.loc[:, members]
                    # bfill across columns and take first column
                    result[name] = group_df.bfill(axis=1).iloc[:, 0]
            df = pd.DataFrame(result, index=df.index)
        except Exception:
            # Last-resort: avoid duplicate labels by renaming duplicates with suffixes
            cols = list(df.columns)
            seen = {}
            newcols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    newcols.append(f"{c}.{seen[c]}")
                else:
                    seen[c] = 0
                    newcols.append(c)
            df.columns = newcols

    # Additional canonical renames (keep as in previous implementation)
    rename_map = {
        "Root Mean Squared Error (RMSE)": "RMSE",
        "Mean Absolute Error (MAE)": "MAE",
        "Missingness": "Missingness_Pct",
        "Missingness_Level": "Missingness_Pct",
        "Missingness_Regime": "MISSINGNESS_REGIMES",
    }
    try:
        df = df.rename(columns=rename_map)
    except Exception:
        pass

    # Ensure numeric Missingness_Pct in percent scale
    if 'Missingness_Pct' in df.columns:
        try:
            df['Missingness_Pct'] = pd.to_numeric(df['Missingness_Pct'], errors='coerce')
            if df['Missingness_Pct'].max() <= 1.0:
                df['Missingness_Pct'] = df['Missingness_Pct'] * 100.0
        except Exception:
            pass

    # Guarantee a regime column exists
    if 'MISSINGNESS_REGIMES' not in df.columns:
        df['MISSINGNESS_REGIMES'] = 'random'
    # Provide Target_Site fallback from StudySite if missing
    if 'Target_Site' not in df.columns and 'StudySite' in df.columns:
        df['Target_Site'] = df['StudySite']

    # Final safety: ensure unique columns (append numeric suffixes if anything still duplicates)
    if not df.columns.is_unique:
        cols = list(df.columns)
        seen = {}
        newcols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                newcols.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                newcols.append(c)
        df.columns = newcols

    return df


def emp_cdf(arr):
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.array([]), np.array([])
    x = np.sort(a)
    y = np.arange(1, x.size + 1) / float(x.size)
    return x, y


def plot_ecdf_aggregated(
    df: pd.DataFrame,
    mode: str,
    key_value: str,
    regime: str,
    metric: str,
    target: str | None,
    levels: list | None,
    output_dir: Path,
    palette="tab10",
    show_legend=True,
):
    """
    mode: 'site' or 'model'
    key_value: the specific site name (if mode=site) or model name (if mode=model)
    regime: missingness regime to filter
    metric: column to plot (e.g., RMSE)
    target: target token to filter by (or None for all)
    levels: list of missingness levels to include (numbers) or None (auto)
    """
    df_sel = df.copy()
    if target is not None and 'Target' in df_sel.columns:
        df_sel = df_sel[df_sel['Target'].astype(str) == str(target)]
    df_sel = df_sel[df_sel['MISSINGNESS_REGIMES'].astype(str) == str(regime)]

    if mode == 'site':
        # keep only rows for that site
        if 'Target_Site' not in df_sel.columns:
            logging.warning("No Target_Site column in aggregated df; cannot run mode=site.")
            return None
        df_sel = df_sel[df_sel['Target_Site'].astype(str) == str(key_value)]
        grouping_col = 'Model'  # across models we compute metric list per level
    else:
        # mode == 'model'
        if 'Model' not in df_sel.columns:
            logging.warning("No Model column in aggregated df; cannot run mode=model.")
            return None
        df_sel = df_sel[df_sel['Model'].astype(str) == str(key_value)]
        grouping_col = 'Target_Site'  # across sites

    if df_sel.empty:
        logging.info(f"No aggregated rows for {mode}={key_value}, regime={regime}, target={target}")
        return None

    # determine levels to plot
    present_levels = sorted(df_sel['Missingness_Pct'].dropna().unique())
    if not present_levels:
        logging.info(f"No Missingness_Pct values for {mode}={key_value}, regime={regime}.")
        return None

    if levels:
        levels = [float(l) for l in levels if float(l) in present_levels]
    else:
        levels = [l for l in PREFERRED_LEVELS if l in present_levels]
        if not levels:
            # pick up to 4 most common
            counts = df_sel['Missingness_Pct'].value_counts()
            levels = sorted(list(counts.index[:4]))

    if not levels:
        logging.info(f"No matching missingness levels for {mode}={key_value}, regime={regime}.")
        return None

    # prepare colors/linestyles
    colors = sns.color_palette(palette, n_colors=len(levels))
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted'] * 4

    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = False
    for i, lvl in enumerate(levels):
        part = df_sel[df_sel['Missingness_Pct'] == lvl]
        if part.empty:
            continue
        # Collect metric values across grouping_col
        if metric not in part.columns:
            logging.debug(f"Metric '{metric}' not found in selection for level {lvl}.")
            continue
        values = pd.to_numeric(part[metric], errors='coerce').dropna().values
        if values.size == 0:
            continue
        x, y = emp_cdf(values)
        if x.size == 0:
            continue
        plotted = True
        ax.step(x, y, where='post', color=colors[i], linestyle=linestyles[i], linewidth=2, label=f"{int(lvl)}%")
    if not plotted:
        plt.close(fig)
        logging.info(f"Nothing plotted for {mode}={key_value}, regime={regime}, metric={metric}")
        return None

    # reference vertical lines (optional)
    for t in [1, 5, 10]:
        ax.axvline(t, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_xlabel(metric)
    ax.set_ylabel("Empirical CDF (fraction ≤ x)")
    title_target = target if target is not None else "ALL"
    if mode == 'site':
        ax.set_title(f"CDF of {metric} across Models — Site: {key_value} — Regime: {regime} — Target: {title_target}")
    else:
        ax.set_title(f"CDF of {metric} across Sites — Model: {key_value} — Regime: {regime} — Target: {title_target}")

    if show_legend:
        ax.legend(title="Missingness level")
    ax.grid(alpha=0.25)
    plt.tight_layout()

    # safe filename
    safe_key = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(key_value))
    safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(regime))
    safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(title_target))
    subfolder = 'CDF_Agg_by_Site' if mode == 'site' else 'CDF_Agg_by_Model'
    out_dir = output_dir / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"CDF_{mode}_{safe_key}_{safe_regime}_{safe_target}.png"
    try:
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        logging.info(f"Saved aggregated CDF plot: {out_path}")
        return out_path
    except Exception as e:
        logging.error(f"Failed to save plot {out_path}: {e}")
        plt.close(fig)
        return None


def batch_from_aggregated(results_dir: Path, output_dir: Path, metric: str, mode: str, values: list | None, regimes: list | None, targets: list | None, levels: list | None):
    agg_file = results_dir / "all_results_summary.csv"
    if not agg_file.exists():
        logging.error("Aggregated file not found: %s", agg_file)
        return []

    df = pd.read_csv(agg_file)
    df = standardize_columns(df)

    # available lists
    if mode == 'site':
        if 'Target_Site' in df.columns:
            all_values = sorted(df['Target_Site'].dropna().unique().tolist())
        elif 'StudySite' in df.columns:
            all_values = sorted(df['StudySite'].dropna().unique().tolist())
        else:
            logging.error("No site column found in aggregated CSV.")
            return []
    else:  # model
        if 'Model' in df.columns:
            all_values = sorted(df['Model'].dropna().unique().tolist())
        else:
            logging.error("No Model column found in aggregated CSV.")
            return []

    if values:
        values_list = [v for v in values if v in all_values]
        if not values_list:
            logging.warning("None of requested values were found. Using all available.")
            values_list = all_values
    else:
        values_list = all_values

    if regimes:
        regimes_list = regimes
    else:
        regimes_list = REGIME_ORDER

    if targets:
        targets_list = targets
    else:
        if 'Target' in df.columns:
            targets_list = sorted(df['Target'].dropna().unique().tolist()) or [None]
        else:
            targets_list = [None]

    saved = []
    for tgt in targets_list:
        for val in values_list:
            for rg in regimes_list:
                out = plot_ecdf_aggregated(df, mode, val, rg, metric, tgt, levels, output_dir)
                if out:
                    saved.append(out)
    logging.info(f"Saved {len(saved)} aggregated CDF plots to {output_dir}")
    return saved


def parse_list_arg(s):
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Make aggregated ECDF plots from all_results_summary.csv")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", type=str, default="RMSE", help="Metric column to plot (RMSE/MAE/...)")
    parser.add_argument("--mode", type=str, choices=["site", "model"], default="site", help="Plot by 'site' or by 'model'")
    parser.add_argument("--values", type=str, default=None, help="Comma list of sites or models to plot (default: all)")
    parser.add_argument("--regimes", type=str, default=None, help="Comma-separated regimes (default: all known)")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated targets (default: all or ALL)")
    parser.add_argument("--levels", type=str, default=None, help="Comma-separated missingness levels to overlay (optional)")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values = parse_list_arg(args.values)
    regimes = parse_list_arg(args.regimes)
    targets = parse_list_arg(args.targets)
    levels = None if args.levels is None else [float(x) for x in parse_list_arg(args.levels)]

    saved = batch_from_aggregated(results_dir, output_dir, args.metric, args.mode, values, regimes, targets, levels)
    return 0


if __name__ == "__main__":
    sys.exit(main())