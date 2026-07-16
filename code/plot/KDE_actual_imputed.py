#!/usr/bin/env python3
"""
make_per_model_actual_vs_imputed.py

Match imputed files to original observed files and create per-model Actual vs Imputed KDE plots
overlaying multiple missingness levels on the same figure (same color per level: solid=Actual, dashed=Imputed).

Usage:
  python plot/make_per_model_actual_vs_imputed.py --results_dir /path/to/Imputation_Result... --output_dir ./heatmaps

Notes:
- Heuristic: detects 'target' tokens (PM2.5, PM10, NO2, etc.) in filenames and uses the prefix up to target
  to find the original (observed) CSV (a file whose stem starts with that prefix but doesn't include "imput").
- Looks for imputed files whose filename contains "imputed".
- If a group has fewer than 2 numeric samples for a curve, it will plot vertical markers instead of KDE.
"""
from pathlib import Path
import argparse
import logging
import os
import re
import fnmatch
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings_enabled = True
try:
    import warnings
    warnings.filterwarnings("ignore")
except Exception:
    warnings_enabled = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Known target tokens to detect in filenames (extend if you have other targets)
TARGET_PATTERNS = [r"PM2\.5", r"PM2_5", r"PM25", r"PM10", r"NO2", r"O3", r"SO2", r"CO"]
TARGET_RE = re.compile("|".join(TARGET_PATTERNS), flags=re.IGNORECASE)

IMPUTED_KEYWORDS = ["imput", "pred", "prediction"]

DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V19_final"


def extract_target_from_stem(stem: str):
    # return normalized target token (e.g., "PM2.5") if present
    m = TARGET_RE.search(stem)
    if not m:
        return None
    tok = m.group(0)
    tok = tok.upper().replace("_", ".")
    tok = re.sub(r"^PM25$", "PM2.5", tok)
    tok = re.sub(r"^PM2_5$", "PM2.5", tok)
    return tok


def find_all_csvs(results_dir: Path):
    all_csvs = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                all_csvs.append(Path(root) / f)
    return all_csvs


def is_imputed_file(path: Path):
    name = path.name.lower()
    return any(k in name for k in ["imput", "imputed", "_imputed_", "prediction", "_pred_"])


def stem_prefix_up_to_target(stem: str, target_tok: str):
    # tokens split by underscore/space/dash/dot
    parts = re.split(r"[_\-\.\s]", stem)
    for i, p in enumerate(parts):
        if re.fullmatch(re.escape(target_tok).replace(r"\.", r"[._]?"), p, flags=re.IGNORECASE):
            # reconstruct prefix up to and including this token
            prefix = "_".join(parts[: i + 1])
            return prefix
    # fallback: if target_tok not exactly matched, find token that contains target letters/digits
    for i, p in enumerate(parts):
        if target_tok.replace(".", "").lower() in p.lower().replace("_", ""):
            return "_".join(parts[: i + 1])
    return None


def safe_read_csv(path: Path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.debug(f"Failed to read {path}: {e}")
        return None


def prepare_groups(results_dir: Path):
    """
    Return:
     - imputed_files: list of (path, stem, model, regime, missingness_level, target, prefix)
     - original_map: dict prefix -> original_path (first match)
    """
    all_csvs = find_all_csvs(results_dir)
    imputed_files = []
    original_candidates = []

    for p in all_csvs:
        if is_imputed_file(p):
            imputed_files.append(p)
        else:
            original_candidates.append(p)

    # build original map keyed by stem prefix up to target token
    original_map = {}
    for p in original_candidates:
        stem = p.stem
        target = extract_target_from_stem(stem)
        if target:
            pref = stem_prefix_up_to_target(stem, target)
            if pref:
                # prefer the file that is shortest/full prefix (avoid picking model files)
                original_map.setdefault(pref, []).append(p)
    # choose one representative original path per prefix (prefer file in top-level or smaller path)
    chosen_originals = {}
    for pref, paths in original_map.items():
        # pick the one with shortest path string (heuristic)
        paths_sorted = sorted(paths, key=lambda x: (len(x.parts), len(x.name)))
        chosen_originals[pref] = paths_sorted[0]

    # parse imputed files into structured records
    records = []
    for p in imputed_files:
        stem = p.stem
        target = extract_target_from_stem(stem)
        # try to extract missingness level (numbers like _10 or _50)
        m_miss = re.search(r"[_\-\.](\d{1,3})(?=[^0-9]|$)", stem)
        missingness = None
        if m_miss:
            try:
                val = float(m_miss.group(1))
                # Heuristic: treat 1-digit '1' as 1% (but many of your files use 10,50)
                missingness = val
            except Exception:
                missingness = None
        # try to detect regime by known words
        regime = None
        for rg in ["random", "short_gap", "medium_gap", "long_gap", "event"]:
            if rg in stem.lower():
                regime = rg
                break
        # try to detect model token by locating tokens between target and regime or 'imputed'
        parts = re.split(r"[_\-\.\s]", stem)
        model_guess = None
        # naive: find token after the target token
        if target:
            for i, tok in enumerate(parts):
                if re.search(target.replace(".", ""), tok, flags=re.IGNORECASE):
                    if i + 1 < len(parts):
                        model_guess = parts[i + 1]
                    break
        # compute prefix to match original
        prefix = None
        if target:
            prefix = stem_prefix_up_to_target(stem, target)
        else:
            # fallback: take first 3 tokens
            prefix = "_".join(parts[:3]) if len(parts) >= 3 else parts[0]
        records.append({
            "path": p,
            "stem": stem,
            "target": target,
            "missingness": missingness,
            "regime": regime,
            "model": model_guess,
            "prefix": prefix
        })

    return records, chosen_originals


def plot_actual_vs_imputed_group(original_path: Path, imputed_paths: list, model_name: str, regime: str, site_prefix: str, target_token: str, output_dir: Path, bw_method=None, dpi=200):
    """
    imputed_paths: list of (missingness, path) pairs for same model+site+regime+target
    original_path: Path to original observed CSV
    """
    # load original
    df_orig = safe_read_csv(original_path)
    if df_orig is None:
        logging.debug(f"Cannot read original file {original_path}")
        return None
    if 'DateTime' not in df_orig.columns:
        # try common casings
        date_col = None
        for c in df_orig.columns:
            if c.lower() == 'datetime' or 'date' in c.lower():
                date_col = c; break
        if date_col is None:
            logging.info(f"No DateTime column in original file {original_path}; skipping group {site_prefix}")
            return None
        else:
            df_orig = df_orig.rename(columns={date_col: 'DateTime'})

    # detect observed column name for target in original
    obs_col = None
    for c in df_orig.columns:
        if target_token.replace('.', '').lower() in c.lower().replace('_', '').replace('.', ''):
            obs_col = c; break
    if obs_col is None:
        # fallback: take the most numeric column besides DateTime
        numeric_counts = {c: int(pd.to_numeric(df_orig[c], errors='coerce').notna().sum()) for c in df_orig.columns if c != 'DateTime'}
        if numeric_counts:
            obs_col = max(numeric_counts.items(), key=lambda x: x[1])[0]
        else:
            logging.info(f"No numeric observed column found in original {original_path}")
            return None

    # build dataset for different missingness levels
    level_series = []
    for rec in imputed_paths:
        miss = rec['missingness']
        p = rec['path']
        df_imp = safe_read_csv(p)
        if df_imp is None:
            continue
        # DateTime column handling
        if 'DateTime' not in df_imp.columns:
            for c in df_imp.columns:
                if c.lower() == 'datetime' or 'date' in c.lower():
                    df_imp = df_imp.rename(columns={c: 'DateTime'}); break
        # detect imputed column in imputed file
        imp_col = None
        for c in df_imp.columns:
            if target_token.replace('.', '').lower() in c.lower().replace('_', '').replace('.', ''):
                imp_col = c; break
        if imp_col is None:
            # fallback: choose most numeric column
            numeric_counts = {c: int(pd.to_numeric(df_imp[c], errors='coerce').notna().sum()) for c in df_imp.columns if c != 'DateTime'}
            if numeric_counts:
                imp_col = max(numeric_counts.items(), key=lambda x: x[1])[0]
            else:
                continue

        df_merge = pd.merge(df_orig[['DateTime', obs_col]], df_imp[['DateTime', imp_col]], on='DateTime', how='inner', suffixes=('_obs', '_imp'))
        if df_merge.empty:
            logging.debug(f"Merge empty for original {original_path} and imputed {p}")
            continue
        df_merge.rename(columns={obs_col: 'y_true', imp_col: 'y_pred'}, inplace=True)
        # coerce numbers
        df_merge['y_true'] = pd.to_numeric(df_merge['y_true'], errors='coerce')
        df_merge['y_pred'] = pd.to_numeric(df_merge['y_pred'], errors='coerce')
        df_merge = df_merge.dropna(subset=['y_true', 'y_pred'])
        if df_merge.empty:
            continue
        level_series.append((miss if miss is not None else -1, df_merge))

    if not level_series:
        return None

    # choose up to 4 levels (prefer 10,20,30,50 ordering), but include all available up to 8
    available_levels = sorted([l for l, _ in level_series if l is not None and l >= 0])
    preferred = [10, 20, 30, 50]
    chosen_levels = [l for l in preferred if l in available_levels]
    if not chosen_levels:
        chosen_levels = available_levels[:4]
    # map level -> df
    level_map = {l: df for l, df in level_series if l in chosen_levels}
    # if none matched chosen_levels, take first up to 4
    if not level_map:
        for l, df in level_series[:4]:
            level_map[l] = df

    # plotting
    colors = sns.color_palette("tab10", n_colors=len(level_map))
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False
    all_vals = []
    for df in level_map.values():
        all_vals.extend(df['y_true'].tolist())
        all_vals.extend(df['y_pred'].tolist())
    if not all_vals:
        return None
    xmin = float(np.nanmin(all_vals)); xmax = float(np.nanmax(all_vals))
    if xmin == xmax:
        xmin -= 0.5; xmax += 0.5
    xpad = (xmax - xmin) * 0.05
    ax.set_xlim(xmin - xpad, xmax + xpad)

    for idx, (lvl, dfm) in enumerate(sorted(level_map.items())):
        vals_true = dfm['y_true'].values
        vals_pred = dfm['y_pred'].values
        color = colors[idx]
        label_true = f"Actual {int(lvl)}%" if lvl >= 0 else "Actual"
        label_pred = f"Imputed {int(lvl)}%" if lvl >= 0 else "Imputed"
        if len(vals_true) >= 3:
            try:
                sns.kdeplot(vals_true, ax=ax, bw_method=bw_method, color=color, linestyle='solid', fill=True, alpha=0.25, label=label_true)
            except Exception:
                sns.kdeplot(vals_true, ax=ax, bw_method=bw_method, color=color, linestyle='solid', label=label_true)
            plotted_any = True
        else:
            for v in vals_true:
                ax.axvline(v, color=color, linestyle='solid', alpha=0.8)
        if len(vals_pred) >= 3:
            try:
                sns.kdeplot(vals_pred, ax=ax, bw_method=bw_method, color=color, linestyle='dashed', fill=True, alpha=0.16, label=label_pred)
            except Exception:
                sns.kdeplot(vals_pred, ax=ax, bw_method=bw_method, color=color, linestyle='dashed', label=label_pred)
            plotted_any = True
        else:
            for v in vals_pred:
                ax.axvline(v, color=color, linestyle='dashed', alpha=0.8)

    if not plotted_any:
        plt.close(fig)
        return None

    ax.set_xlabel(target_token if target_token else "Value")
    ax.set_ylabel("Density")
    title = f"Actual vs Imputed — Model: {model_name or 'UNKNOWN'} — Site: {site_prefix} — Regime: {regime or 'ALL'} — Target: {target_token or 'ALL'}"
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc='upper right')
    ax.grid(alpha=0.25)
    plt.tight_layout()

    safe_model = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(model_name or "model"))
    safe_site = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(site_prefix))
    safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(regime or "ALL"))
    safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(target_token or "ALL"))
    outdir = Path(output_dir) / "ActualVsImputed_ByModel"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"ActualVsImputed_model_{safe_model}_{safe_site}_{safe_regime}_{safe_target}.png"
    try:
        plt.savefig(outpath, dpi=dpi)
        plt.close(fig)
        logging.info(f"Saved: {outpath}")
        return outpath
    except Exception as e:
        logging.error(f"Failed to save {outpath}: {e}")
        plt.close(fig)
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create per-model Actual vs Imputed plots by matching imputed files to originals.")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", type=str, default="./heatmaps")
    parser.add_argument("--bw_method", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=1000, help="Max imputed files to scan")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, originals = prepare_groups(results_dir)
    logging.info(f"Found {len(records)} imputed files and {len(originals)} original prefixes.")

    # group records by (model, prefix, regime, target)
    groups = {}
    for rec in records[: args.max_files]:
        model = rec['model'] or "UNKNOWN"
        key = (model, rec['prefix'], rec['regime'], rec['target'])
        groups.setdefault(key, []).append(rec)

    saved_all = []
    for (model, prefix, regime, target), recs in groups.items():
        if prefix is None:
            continue
        orig = None
        # try exact prefix match
        if prefix in originals:
            orig = originals[prefix]
        else:
            # try shorter prefix variants (strip trailing tokens) to find original
            parts = prefix.split("_")
            for L in range(len(parts), 0, -1):
                cand = "_".join(parts[:L])
                if cand in originals:
                    orig = originals[cand]; break
        if orig is None:
            logging.debug(f"No original found for prefix {prefix} (model {model})")
            continue
        # collect imputed recs for this group
        saved = plot_actual_vs_imputed_group(orig, recs, model, regime, prefix, target, output_dir, bw_method=args.bw_method)
        if saved:
            saved_all.append(saved)

    logging.info(f"Done. Saved {len(saved_all)} plots to {output_dir}/ActualVsImputed_ByModel")
    return 0


if __name__ == "__main__":
    sys.exit(main())