#!/usr/bin/env python3
"""
plot_per_combination_kde.py

Create KDE (density) plots of absolute imputation error for each combination:
  Model × Target_Site × Target × Missingness_Regime × Missingness_Level

This variant is ready to be integrated into other code (call run_kde(...))
or run stand-alone without command-line arguments — it uses DEFAULT_* paths
when no CLI args are provided.

Key changes to support "add it with the code" workflow:
 - DEFAULT_RESULTS_DIR and DEFAULT_OUTPUT_DIR set to your locations.
 - Exposed function run_kde(results_dir, output_dir, ...) for programmatic use.
 - CLI is still available, but --results_dir / --output_dir are optional and
   will default to the defined DEFAULT_* values so you don't have to call it
   from the shell.

To integrate in your code:
  from plot_per_combination_kde import run_kde
  run_kde()  # uses default paths
  # or provide explicit args:
  run_kde("/path/to/results", "/path/to/output", max_files=300, force_match=True)

To run as a script (no args required):
  python plot_per_combination_kde.py

"""

from pathlib import Path
import logging
import os
import re
import fnmatch
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------
# Configure your defaults here
# -------------------------
DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V19_final"
DEFAULT_OUTPUT_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Result_Plot/agg_heatmaps"

# target token patterns used to infer target from filenames
TARGET_PATTERNS = [r"PM2[._]?5", r"PM10", r"NO2", r"O3", r"SO2", r"\bCO\b"]
TARGET_RE = re.compile("|".join(TARGET_PATTERNS), flags=re.IGNORECASE)

REGIME_ORDER = ["random", "short_gap", "medium_gap", "long_gap", "event"]


# -------------------------
# File / detection helpers
# -------------------------
def find_all_csvs(results_dir: Path):
    all_csvs = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                all_csvs.append(Path(root) / f)
    return all_csvs


def is_imputed_file(path: Path):
    name = path.name.lower()
    return any(k in name for k in ["imput", "imputed", "_imputed_", "prediction", "_pred_", "predictions", "y_pred"])


def extract_target_from_stem(stem: str):
    m = TARGET_RE.search(stem)
    if not m:
        return None
    tok = m.group(0)
    tok = tok.upper().replace("_", ".")
    tok = re.sub(r"^PM25$", "PM2.5", tok)
    tok = re.sub(r"^PM2_5$", "PM2.5", tok)
    return tok


def prefix_up_to_target(stem: str, target_tok: str):
    parts = re.split(r"[_\-\.\s]+", stem)
    if not target_tok:
        return "_".join(parts[:3]) if len(parts) >= 3 else parts[0]
    targ = target_tok.replace(".", "").lower()
    for i, p in enumerate(parts):
        if targ in p.lower().replace("_", ""):
            return "_".join(parts[: i + 1])
    return "_".join(parts[:3]) if len(parts) >= 3 else parts[0]


def safe_read_csv(path: Path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.debug(f"Failed to read {path}: {e}")
        return None


# -------------------------
# Prepare file records & originals mapping (tolerant)
# -------------------------
def tolerant_prepare(results_dir: Path):
    all_csvs = find_all_csvs(results_dir)
    imputed_files = [p for p in all_csvs if is_imputed_file(p)]
    original_candidates = [p for p in all_csvs if not is_imputed_file(p)]
    originals_by_stem = {p.stem: p for p in original_candidates}
    records = []
    for p in imputed_files:
        stem = p.stem
        target = extract_target_from_stem(stem)
        m_miss = re.search(r"[_\-\.\s](\d{1,3})(?=[^0-9]|$)", stem)
        missingness = float(m_miss.group(1)) if m_miss else None
        regime = None
        for rg in REGIME_ORDER:
            if rg in stem.lower():
                regime = rg
                break
        parts = re.split(r"[_\-\.\s]+", stem)
        model_guess = None
        if target:
            tkey = target.replace(".", "").lower()
            for i, tok in enumerate(parts):
                if tkey in tok.lower().replace("_", ""):
                    if i + 1 < len(parts):
                        model_guess = parts[i + 1]
                    break
        prefix = prefix_up_to_target(stem, target)
        records.append({
            "path": p,
            "stem": stem,
            "target": target,
            "missingness": missingness,
            "regime": regime,
            "model": model_guess,
            "prefix": prefix
        })
    return records, originals_by_stem, original_candidates


def find_best_original(prefix: str, target: str, originals_by_stem: dict, originals_list: list, force_match=False):
    if prefix in originals_by_stem:
        return originals_by_stem[prefix]
    pref_lower = prefix.lower()
    cands = [p for s, p in originals_by_stem.items() if pref_lower in s.lower()]
    if cands:
        return sorted(cands, key=lambda x: (len(x.parts), len(x.name)))[0]
    if target:
        site_token = prefix.split("_")[0].lower()
        tkey = target.replace(".", "").lower()
        cands = [p for s, p in originals_by_stem.items() if site_token in s.lower() and tkey in s.lower()]
        if cands:
            return sorted(cands, key=lambda x: (len(x.parts), len(x.name)))[0]
    if target:
        tkey = target.replace(".", "").lower()
        cands = [p for s, p in originals_by_stem.items() if tkey in s.lower()]
        if cands:
            return sorted(cands, key=lambda x: (len(x.parts), len(x.name)))[0]
    if force_match and originals_list:
        return sorted(originals_list, key=lambda x: (len(x.parts), len(x.name)))[0]
    return None


# -------------------------
# Merge original + imputed -> compute abs_error
# -------------------------
def merge_original_imputed(original_path: Path, imputed_path: Path):
    df_orig = safe_read_csv(original_path)
    df_imp = safe_read_csv(imputed_path)
    if df_orig is None or df_imp is None:
        return None

    def normalize_datetime_col(df):
        for c in df.columns:
            if c.lower() == "datetime" or "date" in c.lower():
                return c
        return None

    dt_o = normalize_datetime_col(df_orig)
    dt_i = normalize_datetime_col(df_imp)

    # fallback: if no DateTime and equal-length, align by row index
    if not dt_o or not dt_i:
        if len(df_orig) == len(df_imp):
            try:
                df_merge = pd.DataFrame({
                    "y_true": pd.to_numeric(df_orig.iloc[:, 0], errors='coerce'),
                    "y_pred": pd.to_numeric(df_imp.iloc[:, 0], errors='coerce')
                }).dropna()
                return df_merge.reset_index(drop=True)
            except Exception:
                return None
        logging.debug(f"No DateTime column in {original_path} or {imputed_path}")
        return None

    df_orig = df_orig.rename(columns={dt_o: "DateTime"})
    df_imp = df_imp.rename(columns={dt_i: "DateTime"})

    def choose_value_col(df):
        counts = {}
        for c in df.columns:
            if c == "DateTime":
                continue
            try:
                counts[c] = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
            except Exception:
                counts[c] = 0
        if not counts:
            return None
        return max(counts.items(), key=lambda x: x[1])[0]

    col_o = choose_value_col(df_orig)
    col_i = choose_value_col(df_imp)
    if col_o is None or col_i is None:
        return None

    try:
        df_orig['DateTime_parsed'] = pd.to_datetime(df_orig['DateTime'], errors='coerce')
        df_imp['DateTime_parsed'] = pd.to_datetime(df_imp['DateTime'], errors='coerce')
        if df_orig['DateTime_parsed'].notna().sum() > 0 and df_imp['DateTime_parsed'].notna().sum() > 0:
            left = df_orig[['DateTime_parsed', col_o]].rename(columns={'DateTime_parsed': 'DateTime', col_o: 'y_true'})
            right = df_imp[['DateTime_parsed', col_i]].rename(columns={'DateTime_parsed': 'DateTime', col_i: 'y_pred'})
            dfm = pd.merge(left, right, on='DateTime', how='inner')
        else:
            dfm = pd.merge(df_orig[['DateTime', col_o]].rename(columns={col_o: 'y_true'}), df_imp[['DateTime', col_i]].rename(columns={col_i: 'y_pred'}), on='DateTime', how='inner')
    except Exception:
        try:
            dfm = pd.merge(df_orig[['DateTime', col_o]].rename(columns={col_o: 'y_true'}), df_imp[['DateTime', col_i]].rename(columns={col_i: 'y_pred'}), on='DateTime', how='inner')
        except Exception:
            return None

    dfm['y_true'] = pd.to_numeric(dfm['y_true'], errors='coerce')
    dfm['y_pred'] = pd.to_numeric(dfm['y_pred'], errors='coerce')
    dfm = dfm.dropna(subset=['y_true', 'y_pred']).reset_index(drop=True)
    if dfm.empty:
        return None
    dfm['abs_error'] = (dfm['y_pred'] - dfm['y_true']).abs()
    return dfm


# -------------------------
# KDE plotting helpers
# -------------------------
def kde_or_rug(ax, arr, color, label=None, bw_method=None, fill=True, alpha=0.25):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size >= 4:
        try:
            sns.kdeplot(arr, ax=ax, bw_method=bw_method, color=color, fill=fill, alpha=alpha, linewidth=1.8, label=label)
            return True
        except Exception:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(arr)
                xs = np.linspace(arr.min(), arr.max(), 256)
                ax.plot(xs, kde(xs), color=color, label=label, linewidth=1.5)
                if fill:
                    ax.fill_between(xs, kde(xs), color=color, alpha=alpha)
                return True
            except Exception:
                pass
    if arr.size > 0:
        ytop = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.05
        for v in arr:
            ax.plot([v, v], [0, ytop * 0.06], color=color, linestyle='-', linewidth=1.0, alpha=0.9)
        ax.plot([], [], color=color, label=f"{label} (n={len(arr)})")
    return False


def save_kde_single(dfm, out_path: Path, title: str, level_label: str, bw_method=None):
    arr = dfm['abs_error'].values
    if arr.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(8, 6))
    kde_or_rug(ax, arr, color='tab:blue', label=level_label, bw_method=bw_method, fill=True, alpha=0.25)
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def save_kde_overlay(level_to_df, out_path: Path, title: str, bw_method=None):
    colors = sns.color_palette("tab10", n_colors=len(level_to_df))
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = False
    all_vals = []
    for dfm in level_to_df.values():
        all_vals.extend(dfm['abs_error'].tolist())
    if not all_vals:
        return False
    xmin = float(np.nanmin(all_vals)); xmax = float(np.nanmax(all_vals))
    if xmin == xmax:
        xmin -= 0.5; xmax += 0.5
    xpad = (xmax - xmin) * 0.05
    ax.set_xlim(xmin - xpad, xmax + xpad)
    for i, (lvl, dfm) in enumerate(sorted(level_to_df.items())):
        arr = dfm['abs_error'].values
        label = f"{int(lvl)}%" if lvl >= 0 else "Unknown"
        color = colors[i]
        ok = kde_or_rug(ax, arr, color=color, label=label, bw_method=bw_method, fill=True, alpha=0.25)
        plotted = plotted or ok or (arr.size > 0)
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Missingness level")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


# -------------------------
# Programmatic entrypoint
# -------------------------
def run_kde(results_dir: str = None, output_dir: str = None, max_files: int = 1000, bw_method: str = None, force_match: bool = False, dry_run: bool = False):
    """
    Programmatic wrapper. Call this from your code to run the plotting logic without CLI.
    Example:
      run_kde()  # uses DEFAULT_RESULTS_DIR and DEFAULT_OUTPUT_DIR
      run_kde("/path/to/results", "/path/to/output", max_files=300, force_match=True)
    """
    results_dir = Path(results_dir or DEFAULT_RESULTS_DIR)
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)

    records, originals_map, originals_list = tolerant_prepare(results_dir)
    logging.info("Found %d imputed files and %d original candidates.", len(records), len(originals_map))

    groups = {}
    for rec in records[: max_files]:
        model = rec['model'] or "UNKNOWN_MODEL"
        key = (model, rec['prefix'], rec['regime'], rec['target'])
        groups.setdefault(key, []).append(rec)

    total_saved = 0
    for (model, prefix, regime, target), recs in groups.items():
        orig = find_best_original(prefix, target, originals_map, originals_list, force_match=force_match)
        if orig is None:
            logging.debug(f"No original found for prefix={prefix}, target={target}; skipping group (model={model})")
            continue
        level_to_df = {}
        for rec in recs:
            lvl = rec['missingness'] if rec['missingness'] is not None else -1
            dfm = merge_original_imputed(orig, rec['path'])
            if dfm is None or dfm.empty:
                logging.debug(f"Merge empty for {rec['path'].name} with original {orig.name}")
                continue
            level_to_df[lvl] = dfm
            safe_model = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(model))
            safe_site = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(prefix))
            safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(target or "ALL"))
            safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(regime or "ALL"))
            filename = output_dir / "KDE_per_combination" / safe_model / safe_site / safe_target / safe_regime / f"KDE_{safe_model}_{safe_site}_{safe_target}_{safe_regime}_{int(lvl) if lvl>=0 else 'NA'}.png"
            title = f"KDE |error| — Model:{model} Site:{prefix} Target:{target} Regime:{regime} Level:{lvl}"
            if dry_run:
                logging.info(f"DRY RUN: would save single KDE to {filename}")
            else:
                ok = save_kde_single(dfm, filename, title, level_label=f"{int(lvl)}%", bw_method=bw_method)
                if ok:
                    total_saved += 1
        if level_to_df:
            safe_model = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(model))
            safe_site = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(prefix))
            safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(target or "ALL"))
            safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(regime or "ALL"))
            overlay_path = output_dir / "KDE_overlays" / safe_model / safe_site / safe_target / safe_regime / f"KDE_overlay_{safe_model}_{safe_site}_{safe_target}_{safe_regime}.png"
            title = f"KDE overlay — Model:{model} Site:{prefix} Target:{target} Regime:{regime}"
            if dry_run:
                logging.info(f"DRY RUN: would save overlay KDE to {overlay_path}")
            else:
                ok = save_kde_overlay(level_to_df, overlay_path, title, bw_method=bw_method)
                if ok:
                    total_saved += 1

    logging.info("Done. Total plots saved (approx): %d", total_saved)
    return total_saved


# -------------------------
# CLI entrypoint (optional)
# -------------------------
def main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Plot KDE per Model/Site/Target/Regime/Level by matching imputed files to originals.")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results directory (default configured in file).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory (default configured in file).")
    parser.add_argument("--max_files", type=int, default=1000)
    parser.add_argument("--bw_method", type=str, default=None)
    parser.add_argument("--force_match", action="store_true", help="Allow forcing an original match fallback")
    parser.add_argument("--dry_run", action="store_true", help="Do not write plots; only log planned actions")
    args = parser.parse_args(argv)

    return run_kde(results_dir=args.results_dir, output_dir=args.output_dir, max_files=args.max_files, bw_method=args.bw_method, force_match=args.force_match, dry_run=args.dry_run)


if __name__ == "__main__":
    # Run with CLI defaults — so user can simply run the script without specifying arguments
    main_cli()