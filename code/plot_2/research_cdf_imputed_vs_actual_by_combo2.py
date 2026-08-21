#!/usr/bin/env python3
"""
research_cdf_combined_by_combo.py

For each combination of StudySite × Model × Regime × Target, create a single
overlay plot that shows the CDFs (Empirical or KDE) of Actual vs Imputed values
for each missingness level (10%, 20%, 30%, 50%).

Conventions:
 - Solid line = Actual
 - Dashed line = Imputed
 - Each missingness level uses a distinct color (same color for Actual+Imputed)
 - If --absolute is used, plots |value|; otherwise signed values
 - X-limits computed from the combined data (per plot), with small padding
 - Default mode = empirical step ECDF; use --ecdf_mode kde for smoothed cumulative KDE

Output:
  <OUTPUT_DIR>/<StudySite>/<Model>/<Regime>/<Target>_imputed_vs_actual_cdf_combined.png

Run:
  python research_cdf_combined_by_combo.py

Options:
  --results_dir   root results dir (default embedded)
  --output_dir    where to save PNGs
  --ecdf_mode     'empirical' or 'kde'
  --absolute      plot absolute values |value|
  --require_actual require Actual present for imputed rows (default True)
  --max_files     debug: limit number of files processed
"""
from pathlib import Path
import argparse
import logging
import sys
import re
from typing import Optional, Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sns.set_theme(style="whitegrid")

DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V21_final"
DEFAULT_OUTPUT_DIR = "./research_figs"
DESIRED_MISSINGNESS = [10, 20, 30, 50]

# Color mapping for missingness levels (can be extended)
MISSINGNESS_COLORS: Dict[int, str] = {
    10: "#1f77b4",  # blue
    20: "#ff7f0e",  # orange
    30: "#2ca02c",  # green
    50: "#d62728",  # red
}
COLOR_ACTUAL = None  # not used here (colors per level used)


def find_imputed_csvs(results_dir: Path):
    imputed_dir = results_dir / "Imputed_Results"
    if not imputed_dir.exists() or not imputed_dir.is_dir():
        logging.error("Imputed_Results directory not found at: %s", imputed_dir)
        return []
    return sorted(imputed_dir.glob("*.csv"))


def _find_column(df: pd.DataFrame, candidates):
    cols_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_map:
            return cols_map[key]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def parse_imputed_file(fp: Path, require_actual=True):
    """
    Parse an imputed CSV and return entries describing the imputed rows:
    [{
      'StudySite','Model','Regime','Missingness_Level','Target',
      'actuals'(np.array), 'imputeds'(np.array), 'source_file'
    }]
    """
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception as e:
        logging.warning("Failed to read %s: %s", fp, e)
        return []

    site_col = _find_column(df, ["Site", "StudySite", "Study_Site"])
    model_col = _find_column(df, ["Model"])
    regime_col = _find_column(df, ["Regime", "Missingness_Regime"])
    missing_col = _find_column(df, ["Missingness_Level", "Missingness", "Missingness_Pct"])
    target_col = _find_column(df, ["Target"])
    actual_col = _find_column(df, ["Actual", "Observed", "True", "y_true"])
    imputed_col = _find_column(df, ["Imputed", "Prediction", "Predicted", "y_pred"])
    comments_col = _find_column(df, ["Comments", "Missing_Type", "MissingType", "Missingness_Type"])

    if imputed_col is None:
        logging.debug("No imputed column in %s -> skipping", fp)
        return []

    # fallback tokens from filename
    stem = fp.stem
    parts = stem.split("_")
    fname_site = parts[0] if len(parts) >= 1 else ""
    fname_target = parts[1] if len(parts) >= 2 else ""
    fname_model = "_".join(parts[2:-1]) if len(parts) > 3 else (parts[2] if len(parts) >= 3 else "")

    mask = pd.Series(True, index=df.index)

    # Prefer rows marked as Imputed
    if comments_col is not None:
        com = df[comments_col].astype(str).fillna("").str.strip().str.lower()
        mask &= com.str.contains("imput", na=False)
    else:
        logging.debug("No Comments column in %s; using heuristic", fp)

    # require imputed numeric
    im_vals = pd.to_numeric(df[imputed_col], errors="coerce")
    mask &= im_vals.notna()

    # require actual numeric if requested
    if require_actual:
        if actual_col is None:
            logging.debug("No Actual column in %s and require_actual=True -> skipping", fp)
            return []
        a_vals = pd.to_numeric(df[actual_col], errors="coerce")
        mask &= a_vals.notna()

    # heuristic fallback when Comments missing
    if comments_col is None and actual_col is not None:
        a_vals = pd.to_numeric(df[actual_col], errors="coerce")
        heur = (~a_vals.isna()) & (~im_vals.isna()) & (a_vals != im_vals)
        if heur.sum() == 0:
            heur = (~a_vals.isna()) & (~im_vals.isna())
        mask &= heur

    sel = df.loc[mask]
    if sel.empty:
        return []

    actuals = pd.to_numeric(sel[actual_col], errors="coerce") if actual_col is not None else pd.Series([], dtype=float)
    imputeds = pd.to_numeric(sel[imputed_col], errors="coerce")
    valid = actuals.notna() & imputeds.notna()
    actuals = actuals.loc[valid].to_numpy(dtype=float)
    imputeds = imputeds.loc[valid].to_numpy(dtype=float)
    if actuals.size == 0 or imputeds.size == 0:
        return []

    def first(col, fallback):
        if col is None:
            return fallback
        try:
            return str(df[col].dropna().astype(str).iloc[0])
        except Exception:
            return fallback

    site_val = first(site_col, fname_site)
    model_val = first(model_col, fname_model)
    regime_val = first(regime_col, "")
    missing_val = first(missing_col, "")
    target_val = first(target_col, fname_target)

    return [{
        "StudySite": site_val.strip(),
        "Model": model_val.strip(),
        "Regime": regime_val.strip(),
        "Missingness_Level": missing_val.strip(),
        "Target": target_val.strip(),
        "actuals": actuals,
        "imputeds": imputeds,
        "source_file": str(fp)
    }]


def ecdf_empirical(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if values is None or len(values) == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    n = x.size
    y = np.arange(1, n + 1) / float(n)
    x_steps = np.concatenate(([x[0] - 1e-8], x))
    y_steps = np.concatenate(([0.0], y))
    return x_steps, y_steps


def compute_xlim(all_actuals: np.ndarray, all_imputeds: np.ndarray, absolute: bool, pad_frac: float = 0.05, min_pad: float = 1.0) -> Tuple[float, float]:
    vals = np.array([], dtype=float)
    if all_actuals is not None and len(all_actuals) > 0:
        vals = np.concatenate((vals, np.abs(all_actuals) if absolute else all_actuals))
    if all_imputeds is not None and len(all_imputeds) > 0:
        vals = np.concatenate((vals, np.abs(all_imputeds) if absolute else all_imputeds))
    if vals.size == 0:
        return (-1.0, 1.0) if not absolute else (0.0, 1.0)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if absolute:
        vmin = max(0.0, vmin)
    if np.isclose(vmin, vmax):
        pad = max(min_pad, abs(vmax) * pad_frac if vmax != 0 else min_pad)
        return (vmin - pad, vmax + pad) if not absolute else (0.0, vmax + pad)
    rng = vmax - vmin
    pad = max(min_pad, rng * pad_frac)
    left = vmin - pad if not absolute else 0.0
    right = vmax + pad
    return (left, right)


def plot_combined_cdf(miss_map: Dict[str, Tuple[np.ndarray, np.ndarray]], out_fp: Path, absolute: bool, ecdf_mode: str, title: str):
    """
    miss_map: dict mapping missingness_key -> (actuals, imputeds)
    Plot all levels in the same figure, color-coded by missingness.
    Solid line = Actual, dashed = Imputed
    """
    # combine all values to compute global x-limits
    all_actuals = np.concatenate([v[0] for v in miss_map.values() if v[0] is not None and len(v[0]) > 0]) if miss_map else np.array([])
    all_imputeds = np.concatenate([v[1] for v in miss_map.values() if v[1] is not None and len(v[1]) > 0]) if miss_map else np.array([])

    xl = compute_xlim(all_actuals, all_imputeds, absolute=absolute)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False
    # iterate desired order but include any other found keys as well
    # normalize keys to percent ints where possible
    def key_to_pct(k: str) -> Optional[int]:
        s = str(k).strip()
        if s.endswith("%"):
            s = s[:-1]
        try:
            v = float(s)
            if 0 < v <= 1:
                v = v * 100
            return int(round(v))
        except Exception:
            m = re.search(r"(\d+)", s)
            return int(m.group(1)) if m else None

    # prefer DESIRED_MISSINGNESS ordering, then any remaining keys
    ordered_keys = []
    for p in DESIRED_MISSINGNESS:
        for k in miss_map.keys():
            if key_to_pct(k) == p:
                ordered_keys.append(k)
                break
    # append remaining keys not matched
    for k in miss_map.keys():
        if k not in ordered_keys:
            ordered_keys.append(k)

    for k in ordered_keys:
        a_vals, i_vals = miss_map[k]
        if (a_vals is None or len(a_vals) == 0) and (i_vals is None or len(i_vals) == 0):
            continue
        pct = key_to_pct(k) or k
        color = MISSINGNESS_COLORS.get(pct if isinstance(pct, int) else None, None) or MISSINGNESS_COLORS.get(10)
        # prepare arrays
        a = np.abs(a_vals) if absolute else a_vals
        i = np.abs(i_vals) if absolute else i_vals
        # trim to x-limits
        a = a[(a >= xl[0]) & (a <= xl[1])] if len(a) > 0 else a
        i = i[(i >= xl[0]) & (i <= xl[1])] if len(i) > 0 else i

        # plot Actual (solid)
        if len(a) > 0:
            if ecdf_mode == "kde":
                try:
                    sns.kdeplot(a, ax=ax, cumulative=True, bw_method="scott", color=color, linestyle="-", linewidth=1.8, label=f"{pct}% Actual")
                except Exception:
                    xa, ya = ecdf_empirical(a)
                    ax.step(xa, ya, where="post", color=color, linestyle="-", linewidth=1.4, label=f"{pct}% Actual")
            else:
                xa, ya = ecdf_empirical(a)
                ax.step(xa, ya, where="post", color=color, linestyle="-", linewidth=1.4, label=f"{pct}% Actual")
            plotted_any = True

        # plot Imputed (dashed)
        if len(i) > 0:
            if ecdf_mode == "kde":
                try:
                    sns.kdeplot(i, ax=ax, cumulative=True, bw_method="scott", color=color, linestyle="--", linewidth=1.6, label=f"{pct}% Imputed")
                except Exception:
                    xi, yi = ecdf_empirical(i)
                    ax.step(xi, yi, where="post", color=color, linestyle="--", linewidth=1.2, label=f"{pct}% Imputed")
            else:
                xi, yi = ecdf_empirical(i)
                ax.step(xi, yi, where="post", color=color, linestyle="--", linewidth=1.2, label=f"{pct}% Imputed")
            plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "no data for selected missingness levels", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1.05)
    else:
        ax.set_xlim(xl)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("|Value|" if absolute else "Value")
        ax.set_ylabel("CDF")
        ax.grid(True)
        # build a legend with grouped entries (one pair per missingness)
        handles, labels = ax.get_legend_handles_labels()
        # keep label order as plotted, but we may want to shorten labels in legend
        ax.legend(loc="lower right", fontsize="small")

    plt.title(title)
    plt.tight_layout()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Combined CDF plot: Imputed vs Actual per Site/Model/Regime/Target")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ecdf_mode", choices=("empirical", "kde"), default="empirical")
    parser.add_argument("--absolute", action="store_true", help="Plot absolute values instead of signed")
    parser.add_argument("--require_actual", action="store_true", default=True)
    parser.add_argument("--max_files", type=int, default=None, help="debug limit")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logging.error("Results dir not found: %s", results_dir)
        sys.exit(1)

    csvs = find_imputed_csvs(results_dir)
    if not csvs:
        logging.error("No imputed CSVs found under Imputed_Results")
        sys.exit(1)
    if args.max_files is not None:
        csvs = csvs[: args.max_files]

    # collect data grouped by (site, model, regime, target) -> missingness -> (actuals, imputeds)
    grouped = {}
    for fp in csvs:
        entries = parse_imputed_file(fp, require_actual=args.require_actual)
        if not entries:
            continue
        for e in entries:
            site = e["StudySite"] or "UNKNOWN_SITE"
            model = e["Model"] or "UNKNOWN_MODEL"
            regime = e["Regime"] or "unknown_regime"
            target = e["Target"] or "target"
            miss = e["Missingness_Level"] or "missing"
            actuals = e["actuals"]
            imputeds = e["imputeds"]
            key = (site, model, regime, target)
            grouped.setdefault(key, {})[str(miss)] = (actuals, imputeds)

    total = 0
    skipped = 0
    for (site, model, regime, target), miss_map in grouped.items():
        safe_site = re.sub(r"[^0-9A-Za-z\-_\.]", "_", site)
        safe_model = re.sub(r"[^0-9A-Za-z\-_\.]", "_", model)
        safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", regime)
        safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", target)
        out_dir = Path(args.output_dir) / safe_site / safe_model / safe_regime
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / f"{safe_target}_imputed_vs_actual_cdf_combined.png"
        title = f"{site} | {model} | {regime} | {target}"

        # Filter miss_map to only include presence
        available_levels = {k: v for k, v in miss_map.items() if (v[0] is not None and len(v[0]) > 0) or (v[1] is not None and len(v[1]) > 0)}
        if not available_levels:
            skipped += 1
            continue

        plot_combined_cdf(available_levels, out_fp, absolute=args.absolute, ecdf_mode=args.ecdf_mode, title=title)
        total += 1

    logging.info("Done. Combined CDF plots generated: %d, skipped groups: %d", total, skipped)


if __name__ == "__main__":
    main()