#!/usr/bin/env python3
"""
KDE_from_aggregated.py

Produce density (KDE) overlay plots from aggregated metrics (no per-sample data required).

Modes:
 - site: for each Target_Site × Regime create one figure. For each Missingness_Pct level,
         plot the KDE of metric values across Models (one curve per level).
 - model: for each Model × Regime create one figure. For each Missingness_Pct level,
         plot the KDE of metric values across Target_Site (one curve per level).

Input:
 - results_dir containing all_results_summary.csv (default embedded)
Output:
 - PNGs saved to <output_dir>/KDE_Agg_by_Site/ or <output_dir>/KDE_Agg_by_Model/

Usage examples:
  python KDE_from_aggregated.py --results_dir /path/to/results --output_dir ./agg_plots --metric RMSE --mode site
  python KDE_from_aggregated.py --metric MAE --mode model --values TF_BRITS --regimes random,short_gap --levels 10,20,30

Notes:
 - This script follows the same "aggregated-only" strategy as your working CDF script,
   but plots smoothed densities (KDE) rather than ECDFs.
 - KDE requires several samples per curve to be meaningful. If a level has < 3 samples,
   we plot vertical markers instead of a KDE curve.
"""
from pathlib import Path
import argparse
import logging
import re
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Defaults
DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V21_final"
DEFAULT_OUTPUT_DIR = "./kde_per_sample"
REGIME_ORDER = ["random", "short_gap", "medium_gap", "long_gap", "event"]
PREFERRED_LEVELS = [10, 20, 30, 50]

Model_Name = "AQUISTIL"

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and collapse duplicate columns safely.
    Compatible with the CDF-from-aggregated script's standardize_columns.
    """
    import re
    df = df.copy()

    try:
        new_cols = []
        for c in df.columns:
            s = str(c)
            s = re.sub(r"\s+", " ", s).strip()
            s = re.sub(r"\.\d+$", "", s)
            new_cols.append(s)
        df.columns = new_cols
    except Exception:
        pass

    if not df.columns.is_unique:
        try:
            groups = {}
            for col in df.columns:
                groups.setdefault(col, []).append(col)
            result = {}
            for name, members in groups.items():
                if len(members) == 1:
                    result[name] = df[members[0]]
                else:
                    group_df = df.loc[:, members]
                    result[name] = group_df.bfill(axis=1).iloc[:, 0]
            df = pd.DataFrame(result, index=df.index)
        except Exception:
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

    if 'Missingness_Pct' in df.columns:
        try:
            df['Missingness_Pct'] = pd.to_numeric(df['Missingness_Pct'], errors='coerce')
            if df['Missingness_Pct'].max() <= 1.0:
                df['Missingness_Pct'] = df['Missingness_Pct'] * 100.0
        except Exception:
            pass

    if 'MISSINGNESS_REGIMES' not in df.columns:
        df['MISSINGNESS_REGIMES'] = 'random'
    if 'Target_Site' not in df.columns and 'StudySite' in df.columns:
        df['Target_Site'] = df['StudySite']

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


def detect_site_column(df: pd.DataFrame):
    for cand in ["Target_Site", "TargetSite", "Target_Site_ID", "Target_SiteName", "StudySite", "Study_Site", "Site", "station", "location"]:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "site" in c.lower():
            return c
    return None


def detect_target_column(df: pd.DataFrame):
    for cand in ["Target", "target", "Target_Var", "TargetColumn", "TargetVariable"]:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "target" in c.lower() and "site" not in c.lower():
            return c
    return None


# -------------------------
# Per-sample Imputed_Results loader & KDE plotting
# -------------------------
def detect_actual_imputed_cols(df: pd.DataFrame):
    act = None
    imp = None
    for c in df.columns:
        cl = c.lower()
        if act is None and any(k in cl for k in ['actual', 'observ', 'y_true', 'y_obs', 'obs']):
            act = c
        if imp is None and any(k in cl for k in ['imput', 'imputed', 'pred', 'prediction', 'y_pred']):
            imp = c
    return act, imp


def load_per_sample_imputed(results_dir: Path) -> Optional[pd.DataFrame]:
    imp_dir = results_dir / 'Imputed_Results'
    if not imp_dir.exists() or not imp_dir.is_dir():
        logging.info(f"No Imputed_Results directory at {imp_dir}")
        return None
    frames = []
    for csv in sorted(imp_dir.glob('*.csv')):
        stem = csv.stem
        if 'imput' not in stem.lower():
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            logging.warning(f"Failed read per-sample CSV {csv}: {e}")
            continue
        df = standardize_columns(df)
        # detect and normalise datetime
        for c in df.columns:
            if c.lower() == 'datetime' or 'date' in c.lower():
                if c != 'DateTime':
                    df = df.rename(columns={c: 'DateTime'})
                break
        # detect actual/imputed
        act_col, imp_col = detect_actual_imputed_cols(df)
        if act_col:
            df['Actual'] = pd.to_numeric(df[act_col], errors='coerce')
        if imp_col:
            df['Imputed'] = pd.to_numeric(df[imp_col], errors='coerce')
        # filename metadata guesses
        base = stem
        if stem.lower().endswith('_imputed'):
            base = stem[: -len('_imputed')]
        parts = base.split('_')
        site_guess = parts[0] if len(parts) >= 1 else None
        target_guess = '_'.join(parts[1:-1]) if len(parts) >= 3 else (parts[1] if len(parts) == 2 else None)
        model_guess = parts[-1] if len(parts) >= 2 else None
        if 'StudySite' not in df.columns and 'Target_Site' not in df.columns and 'Site' not in df.columns:
            df['StudySite'] = site_guess
        if 'Model' not in df.columns:
            df['Model'] = model_guess
        if 'Target' not in df.columns and target_guess:
            df['Target'] = target_guess
        # infer missingness pct from filename if present (prefer common levels first)
        inferred_level = None
        for lvl in [50, 30, 20, 10, 5, 1]:
            if re.search(fr'(?<!\d)(?:_|-|\.){lvl}(?:_|-|\.|$)', stem.lower()):
                inferred_level = float(lvl)
                break
        if inferred_level is None:
            m = re.search(r'[_\-.](\d{1,3})(?=[^0-9]|$)', stem)
            if m:
                try:
                    val = int(m.group(1))
                    # ignore 4-digit tokens (likely years)
                    if val < 1000:
                        inferred_level = float(val)
                except Exception:
                    inferred_level = None
        if inferred_level is not None:
            df['_file_missingness'] = inferred_level
            # if file's Missingness_Pct absent or all-zero/ <=1, prefer inferred
            if 'Missingness_Pct' not in df.columns or (pd.to_numeric(df.get('Missingness_Pct', pd.Series([])), errors='coerce').dropna().max() or 0) <= 1.0:
                df['Missingness_Pct'] = inferred_level
        # infer regime token
        for rg in ['random', 'short_gap', 'medium_gap', 'long_gap', 'event']:
            if rg in stem.lower():
                if 'MISSINGNESS_REGIMES' not in df.columns:
                    df['MISSINGNESS_REGIMES'] = rg
                break
        # preserve comments
        if 'Comments' not in df.columns:
            df['Comments'] = 'Original and Imputed'
        # attach source filename for diagnostics
        df['_source_file'] = str(csv.name)
        frames.append(df)
    if not frames:
        logging.info('No per-sample imputed CSVs found in Imputed_Results')
        return None
    big = pd.concat(frames, ignore_index=True)
    big = standardize_columns(big)
    return big


def plot_kde_per_sample(df: pd.DataFrame, mode: str, key_value: str, regime: str, target: Optional[str], levels: Optional[list], output_dir: Path, bw_method=None, fill=True, alpha=0.25):
    # Expect df has columns 'Actual' and 'Imputed' (or detect again)
    tgt_col = detect_target_column(df)
    site_col = None
    for cand in ['Target_Site', 'StudySite', 'Site']:
        if cand in df.columns:
            site_col = cand; break
    if site_col is None:
        logging.warning('No site column detected for per-sample KDE')
        return None

    df_sel = df.copy()
    if target is not None and tgt_col in df_sel.columns:
        df_sel = df_sel[df_sel[tgt_col].astype(str) == str(target)]
    df_sel = df_sel[df_sel['MISSINGNESS_REGIMES'].astype(str) == str(regime)] if 'MISSINGNESS_REGIMES' in df_sel.columns else df_sel

    if mode == 'site':
        df_sel = df_sel[df_sel[site_col].astype(str) == str(key_value)]
        group_label = 'Model'
    else:
        df_sel = df_sel[df_sel['Model'].astype(str) == str(key_value)]
        group_label = site_col

    if df_sel.empty:
        logging.info(f'No per-sample rows for {mode}={key_value}, regime={regime}, target={target}')
        return None

    # choose levels
    # collect available missingness levels from explicit column and filename-inferred column
    present_levels = set()
    if 'Missingness_Pct' in df_sel.columns:
        vals = pd.to_numeric(df_sel['Missingness_Pct'], errors='coerce').dropna().unique().tolist()
        present_levels.update([float(v) for v in vals if float(v) >= 0])
    if '_file_missingness' in df_sel.columns:
        vals2 = pd.to_numeric(df_sel['_file_missingness'], errors='coerce').dropna().unique().tolist()
        present_levels.update([float(v) for v in vals2 if float(v) >= 0])
    present_levels = sorted([v for v in present_levels])
    if not present_levels and levels:
        present_levels = levels
    if levels:
        levels = [float(l) for l in levels if float(l) in present_levels]
    else:
        # prefer common ordering
        pref = [10, 20, 30, 50, 5, 1]
        chosen = [v for v in pref if v in present_levels]
        if not chosen:
            chosen = present_levels[:4]
        levels = chosen
    if not levels:
        logging.info('No missingness levels found for per-sample KDE')
        return None

    # detect actual/imputed cols if not standardized
    act_col = 'Actual' if 'Actual' in df_sel.columns else None
    imp_col = 'Imputed' if 'Imputed' in df_sel.columns else None
    if not act_col or not imp_col:
        a, b = detect_actual_imputed_cols(df_sel)
        act_col = act_col or a
        imp_col = imp_col or b
    if not act_col or not imp_col:
        logging.info('No Actual/Imputed columns detected for per-sample KDE')
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette('tab10', n_colors=len(levels))
    plotted_any = False
    all_vals = []
    for lvl in levels:
        part = df_sel[df_sel['Missingness_Pct'] == lvl] if 'Missingness_Pct' in df_sel.columns else df_sel
        vals_act = pd.to_numeric(part[act_col], errors='coerce').dropna().values
        vals_imp = pd.to_numeric(part[imp_col], errors='coerce').dropna().values
        all_vals.extend(vals_act.tolist()); all_vals.extend(vals_imp.tolist())
    if not all_vals:
        logging.info('No numeric actual/imputed values for per-sample KDE')
        plt.close(fig); return None
    xmin = float(np.nanmin(all_vals)); xmax = float(np.nanmax(all_vals))
    if xmin == xmax: xmin -= 0.5; xmax += 0.5
    xpad = (xmax - xmin) * 0.05
    ax.set_xlim(xmin - xpad, xmax + xpad)

    for i, lvl in enumerate(levels):
        # select rows where either Missingness_Pct or _file_missingness equals lvl
        if 'Missingness_Pct' in df_sel.columns and '_file_missingness' in df_sel.columns:
            part = df_sel[(pd.to_numeric(df_sel['Missingness_Pct'], errors='coerce') == float(lvl)) | (pd.to_numeric(df_sel['_file_missingness'], errors='coerce') == float(lvl))]
        elif 'Missingness_Pct' in df_sel.columns:
            part = df_sel[pd.to_numeric(df_sel['Missingness_Pct'], errors='coerce') == float(lvl)]
        elif '_file_missingness' in df_sel.columns:
            part = df_sel[pd.to_numeric(df_sel['_file_missingness'], errors='coerce') == float(lvl)]
        else:
            part = df_sel
        vals_act = pd.to_numeric(part[act_col], errors='coerce').dropna().values if act_col in part.columns else np.array([])
        vals_imp = pd.to_numeric(part[imp_col], errors='coerce').dropna().values if imp_col in part.columns else np.array([])
        if vals_act.size > 0:
            try:
                sns.kdeplot(vals_act, ax=ax, bw_method=bw_method, color=colors[i], linestyle='solid', fill=fill, alpha=alpha, label=f'Actual {int(lvl)}%')
                plotted_any = True
            except Exception:
                pass
        if vals_imp.size > 0:
            try:
                sns.kdeplot(vals_imp, ax=ax, bw_method=bw_method, color=colors[i], linestyle='dashed', fill=fill, alpha=alpha*0.8, label=f'Imputed {int(lvl)}%')
                plotted_any = True
            except Exception:
                pass

    if not plotted_any:
        plt.close(fig); return None
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    tgt_label = target if target is not None else 'ALL'
    title = f'Per-sample KDE — {mode.upper()}: {key_value} — Regime: {regime} — Target: {tgt_label}'
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels(); by_label = dict(zip(labels, handles)); ax.legend(by_label.values(), by_label.keys(), fontsize=9)
    ax.grid(alpha=0.25); plt.tight_layout()
    safe_key = re.sub(r'[^0-9A-Za-z\-_\.]', '_', str(key_value))
    safe_regime = re.sub(r'[^0-9A-Za-z\-_\.]', '_', str(regime))
    safe_target = re.sub(r'[^0-9A-Za-z\-_\.]', '_', str(tgt_label))
    subfolder = 'KDE_PerSample_by_Site' if mode == 'site' else 'KDE_PerSample_by_Model'
    out_dir = output_dir / subfolder; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'KDE_per_sample_{mode}_{safe_key}_{safe_regime}_{safe_target}.png'
    try:
        plt.savefig(out_path, dpi=200); plt.close(fig); logging.info(f'Saved per-sample KDE: {out_path}'); return out_path
    except Exception as e:
        logging.error(f'Failed to save per-sample KDE {out_path}: {e}'); plt.close(fig); return None


def batch_from_per_sample(results_dir: Path, output_dir: Path, metric: str, mode: str, values: Optional[list], regimes: Optional[list], targets: Optional[list], levels: Optional[list], bw_method=None, fill=True, alpha=0.25):
    df = load_per_sample_imputed(results_dir)
    if df is None:
        logging.error('No per-sample imputed data available; aborting per-sample batch.')
        return []
    df = standardize_columns(df)
    if mode == 'site':
        if 'Target_Site' in df.columns:
            all_values = sorted(df['Target_Site'].dropna().unique().tolist())
        elif 'StudySite' in df.columns:
            all_values = sorted(df['StudySite'].dropna().unique().tolist())
        else:
            logging.error('No site column found in per-sample data.'); return []
    else:
        if 'Model' in df.columns:
            all_values = sorted(df['Model'].dropna().unique().tolist())
        else:
            logging.error('No Model column found in per-sample data.'); return []
    if values:
        values_list = [v for v in values if v in all_values]
        if not values_list:
            logging.warning('None of requested values found in per-sample; using all.'); values_list = all_values
    else:
        values_list = all_values
    regimes_list = regimes if regimes else REGIME_ORDER
    if targets:
        targets_list = targets
    else:
        tgt_col = detect_target_column(df)
        if tgt_col:
            targets_list = sorted(df[tgt_col].dropna().unique().tolist()) or [None]
        else:
            targets_list = [None]
    saved = []
    for tgt in targets_list:
        for val in values_list:
            for rg in regimes_list:
                out = plot_kde_per_sample(df, mode, val, rg, tgt, levels, output_dir, bw_method=bw_method, fill=fill, alpha=alpha)
                if out:
                    saved.append(out)
    logging.info(f'Saved {len(saved)} per-sample KDE plots to {output_dir}')
    return saved


def kde_plot_values(ax, values, color, label, bw_method=None, fill=True, alpha=0.25):
    """
    Helper to draw KDE if values large enough, otherwise draw vertical markers.
    Returns True if drawn a KDE, False if drawn markers.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size >= 4:
        try:
            sns.kdeplot(values, ax=ax, bw_method=bw_method, color=color, fill=fill, alpha=alpha, linewidth=1.8, label=label)
            return True
        except Exception:
            # fallback: use scipy gaussian_kde manually or simple histogram
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(values)
                xs = np.linspace(np.nanmin(values), np.nanmax(values), 256)
                ax.plot(xs, kde(xs), color=color, label=label, linewidth=1.5)
                if fill:
                    ax.fill_between(xs, kde(xs), color=color, alpha=alpha)
                return True
            except Exception:
                pass
    # Too few samples -> vertical marker(s)
    if values.size > 0:
        # plot small jittered rug/markers at top
        ytop = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.05
        for v in values:
            ax.plot([v, v], [0, ytop*0.06], color=color, linestyle='-', linewidth=1.0, alpha=0.9)
        # add a label entry (marker-only)
        ax.plot([], [], color=color, label=f"{label} (n={len(values)})")
    return False


def plot_kde_aggregated(
    df: pd.DataFrame,
    mode: str,
    key_value: str,
    regime: str,
    metric: str,
    target: Optional[str],
    levels: Optional[list],
    output_dir: Path,
    palette="tab10",
    bw_method=None,
    fill=True,
    alpha=0.25,
):
    """
    mode: 'site' or 'model'
    key_value: site name (mode=site) or model name (mode=model)
    regime: missingness regime to filter
    metric: numeric metric column to plot distribution for
    target: filter by Target value or None
    levels: list of numeric Missingness_Pct levels to include
    """
    df_sel = df.copy()
    if target is not None and 'Target' in df_sel.columns:
        df_sel = df_sel[df_sel['Target'].astype(str) == str(target)]
    df_sel = df_sel[df_sel['MISSINGNESS_REGIMES'].astype(str) == str(regime)]

    if mode == 'site':
        if 'Target_Site' not in df_sel.columns:
            logging.warning("No Target_Site column in aggregated df; cannot run mode=site.")
            return None
        df_sel = df_sel[df_sel['Target_Site'].astype(str) == str(key_value)]
        across_label = "Model"
    else:
        if 'Model' not in df_sel.columns:
            logging.warning("No Model column in aggregated df; cannot run mode=model.")
            return None
        df_sel = df_sel[df_sel['Model'].astype(str) == str(key_value)]
        across_label = "Target_Site"

    if df_sel.empty:
        logging.info(f"No aggregated rows for {mode}={key_value}, regime={regime}, target={target}")
        return None

    if metric not in df_sel.columns:
        logging.info(f"Metric '{metric}' not present in aggregated CSV.")
        return None

    present_levels = sorted(df_sel['Missingness_Pct'].dropna().unique())
    if not present_levels:
        logging.info(f"No Missingness_Pct values for {mode}={key_value}, regime={regime}.")
        return None

    if levels:
        # keep only levels that exist
        levels = [float(l) for l in levels if float(l) in present_levels]
    else:
        levels = [l for l in PREFERRED_LEVELS if l in present_levels]
        if not levels:
            counts = df_sel['Missingness_Pct'].value_counts()
            levels = sorted(list(counts.index[:4]))

    if not levels:
        logging.info(f"No matching missingness levels for {mode}={key_value}, regime={regime}.")
        return None

    colors = sns.color_palette(palette, n_colors=len(levels))
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted'] * 4

    fig, ax = plt.subplots(figsize=(9, 6))
    plotted_any = False
    # Determine global x-limits using all selected values concatenated
    all_vals = []
    for lvl in levels:
        part = df_sel[df_sel['Missingness_Pct'] == lvl]
        vals = pd.to_numeric(part[metric], errors='coerce').dropna().values
        all_vals.extend(vals.tolist())
    if len(all_vals) == 0:
        logging.info("No numeric metric values available for plotting.")
        plt.close(fig)
        return None
    xmin = float(np.nanmin(all_vals))
    xmax = float(np.nanmax(all_vals))
    if xmin == xmax:
        xmin = xmin - 0.5
        xmax = xmax + 0.5
    xpad = (xmax - xmin) * 0.05
    ax.set_xlim(xmin - xpad, xmax + xpad)

    for i, lvl in enumerate(levels):
        part = df_sel[df_sel['Missingness_Pct'] == lvl]
        values = pd.to_numeric(part[metric], errors='coerce').dropna().values
        if values.size == 0:
            continue
        label = f"{int(lvl)}% missing"
        color = colors[i]
        drawn = kde_plot_values(ax, values, color=color, label=label, bw_method=bw_method, fill=fill, alpha=alpha)
        plotted_any = plotted_any or drawn or (values.size > 0)

    if not plotted_any:
        plt.close(fig)
        logging.info(f"No distributions plotted for {mode}={key_value}, regime={regime}.")
        return None

    for t in [1, 5, 10]:
        ax.axvline(t, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    tgt_label = target if target is not None else "ALL"
    if mode == 'site':
        ax.set_title(f"KDE of {metric} across Models — Site: {key_value} — Regime: {REGIME_ORDER[REGIME_ORDER.index(regime)] if regime in REGIME_ORDER else regime} — Target: {tgt_label}")
    else:
        ax.set_title(f"KDE of {metric} across Sites — Model: {key_value} — Regime: {REGIME_ORDER[REGIME_ORDER.index(regime)] if regime in REGIME_ORDER else regime} — Target: {tgt_label}")

    # tidy legend (avoid duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Missingness level", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    safe_key = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(key_value))
    safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(regime))
    safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(tgt_label))
    subfolder = 'KDE_Agg_by_Site' if mode == 'site' else 'KDE_Agg_by_Model'
    out_dir = output_dir / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"KDE_{mode}_{safe_key}_{safe_regime}_{safe_target}.png"
    try:
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        logging.info(f"Saved KDE aggregated plot: {out_path}")
        return out_path
    except Exception as e:
        logging.error(f"Failed to save plot {out_path}: {e}")
        plt.close(fig)
        return None


def batch_from_aggregated(results_dir: Path, output_dir: Path, metric: str, mode: str, values: Optional[list], regimes: Optional[list], targets: Optional[list], levels: Optional[list], bw_method=None, fill=True, alpha=0.25):
    agg_file = results_dir / "all_results_summary.csv"
    if not agg_file.exists():
        logging.error("Aggregated file not found: %s", agg_file)
        return []

    df = pd.read_csv(agg_file)
    df = standardize_columns(df)

    if mode == 'site':
        if 'Target_Site' in df.columns:
            all_values = sorted(df['Target_Site'].dropna().unique().tolist())
        elif 'StudySite' in df.columns:
            all_values = sorted(df['StudySite'].dropna().unique().tolist())
        else:
            logging.error("No site column found in aggregated CSV.")
            return []
    else:
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

    regimes_list = regimes if regimes else REGIME_ORDER

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
                out = plot_kde_aggregated(df, mode, val, rg, metric, tgt, levels, output_dir, bw_method=bw_method, fill=fill, alpha=alpha)
                if out:
                    saved.append(out)
    logging.info(f"Saved {len(saved)} aggregated KDE plots to {output_dir}")
    return saved


def parse_list_arg(s):
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Make aggregated KDE plots from all_results_summary.csv")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", type=str, default="RMSE", help="Metric column to plot (RMSE/MAE/...)")
    parser.add_argument("--mode", type=str, choices=["site", "model"], default="model", help="Plot by 'site' or by 'model'")
    parser.add_argument("--values", type=str, default=None, help="Comma list of sites or models to plot (default: all)")
    parser.add_argument("--regimes", type=str, default=None, help="Comma-separated regimes (default: all known)")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated targets (default: all or ALL)")
    parser.add_argument("--levels", type=str, default=None, help="Comma-separated missingness levels to overlay (optional)")
    parser.add_argument("--bw_method", type=str, default=None, help="Bandwidth method for KDE (pass to seaborn.kdeplot bw_method)")
    parser.add_argument("--no_fill", action="store_true", help="Do not fill under KDE curves")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha for fills")
    parser.add_argument("--per_sample", action="store_true", default=True, help="Use per-sample Imputed_Results CSVs to plot Actual vs Imputed (requires Imputed_Results/*.csv)")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values = parse_list_arg(args.values)
    # If a hard-coded Model_Name is provided in the file, prefer it
    try:
        if 'Model_Name' in globals() and isinstance(Model_Name, str) and Model_Name:
            values = [Model_Name]
    except Exception:
        pass
    regimes = parse_list_arg(args.regimes)
    targets = parse_list_arg(args.targets)
    levels = None if args.levels is None else [float(x) for x in parse_list_arg(args.levels)]
    if args.per_sample:
        logging.info("Per-sample mode: scanning Imputed_Results for Actual vs Imputed CSVs")
        saved = batch_from_per_sample(results_dir, output_dir, args.metric, args.mode, values, regimes, targets, levels, bw_method=args.bw_method, fill=(not args.no_fill), alpha=args.alpha)
    else:
        saved = batch_from_aggregated(results_dir, output_dir, args.metric, args.mode, values, regimes, targets, levels, bw_method=args.bw_method, fill=(not args.no_fill), alpha=args.alpha)
    return 0


if __name__ == "__main__":
    sys.exit(main())