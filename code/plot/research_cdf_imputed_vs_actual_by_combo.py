#!/usr/bin/env python3
"""
research_cdf_imputed_vs_actual_by_combo.py

Create per-StudySite × Model × Regime × Target panels showing the empirical
Cumulative Distribution Function (CDF) of Actual vs Imputed values at the
timestamps that were imputed (i.e. rows flagged 'Imputed').

Features:
- Computes x-limits from the data (no fixed negative range).
- If --absolute is used, X starts at 0; otherwise uses min(actual, imputed).
- Optionally uses KDE cumulative smoothing with --ecdf_mode kde.
- One PNG per (Site,Model,Regime,Target) with 4 panels for missingness 10/20/30/50.

Run:
  python research_cdf_imputed_vs_actual_by_combo.py
"""
from pathlib import Path
import argparse
import logging
import sys
import re
from typing import Optional, Tuple

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
DESIRED_MISSINGNESS = [10, 20, 30, 50]  # order of panels
COLOR_ACTUAL = "#1f77b4"
COLOR_IMPUTED = "#ff7f0e"


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

    stem = fp.stem
    parts = stem.split("_")
    fname_site = parts[0] if len(parts) >= 1 else ""
    fname_target = parts[1] if len(parts) >= 2 else ""
    fname_model = "_".join(parts[2:-1]) if len(parts) > 3 else (parts[2] if len(parts) >= 3 else "")

    mask = pd.Series(True, index=df.index)

    # Prefer explicit "Imputed" marks in Comments
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

    # heuristic fallback when Comments absent
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


def compute_xlim_from_data(actuals: np.ndarray, imputeds: np.ndarray, absolute: bool, pad_frac: float = 0.05, min_pad: float = 1.0) -> Tuple[float, float]:
    # combine arrays, ignore nan/empty
    vals = np.array([], dtype=float)
    if actuals is not None and len(actuals) > 0:
        vals = np.concatenate((vals, np.abs(actuals) if absolute else actuals))
    if imputeds is not None and len(imputeds) > 0:
        vals = np.concatenate((vals, np.abs(imputeds) if absolute else imputeds))
    if vals.size == 0:
        # fallback defaults
        return (-1.0, 1.0) if not absolute else (0.0, 1.0)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    # for absolute mode ensure lower bound zero
    if absolute:
        vmin = max(0.0, vmin)
    # if vmin == vmax expand small window
    if np.isclose(vmin, vmax):
        pad = max(min_pad, abs(vmax) * pad_frac if vmax != 0 else min_pad)
        return (vmin - pad, vmax + pad) if not absolute else (0.0, vmax + pad)
    # apply fractional padding
    rng = vmax - vmin
    pad = max(min_pad, rng * pad_frac)
    left = vmin - pad if not absolute else 0.0
    right = vmax + pad
    return (left, right)


def plot_cdf_panel(actuals: np.ndarray, imputeds: np.ndarray, ax: plt.Axes, absolute: bool, mode: str):
    # compute x-limits from data
    xl = compute_xlim_from_data(actuals, imputeds, absolute=absolute)
    # if no data at all -> annotate and return
    if (actuals is None or len(actuals) == 0) and (imputeds is None or len(imputeds) == 0):
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(xl)
        ax.set_ylim(0, 1.05)
        return

    a_vals = np.abs(actuals) if absolute else actuals
    i_vals = np.abs(imputeds) if absolute else imputeds

    # trim values to x-limits to keep plots clean
    a_vals = a_vals[(a_vals >= xl[0]) & (a_vals <= xl[1])] if len(a_vals) > 0 else a_vals
    i_vals = i_vals[(i_vals >= xl[0]) & (i_vals <= xl[1])] if len(i_vals) > 0 else i_vals

    if mode == "kde":
        if len(a_vals) > 0:
            try:
                sns.kdeplot(a_vals, ax=ax, cumulative=True, bw_method="scott", color=COLOR_ACTUAL, label="Actual", linewidth=1.4)
            except Exception:
                xa, ya = ecdf_empirical(a_vals)
                ax.step(xa, ya, where="post", color=COLOR_ACTUAL, label="Actual")
        if len(i_vals) > 0:
            try:
                sns.kdeplot(i_vals, ax=ax, cumulative=True, bw_method="scott", color=COLOR_IMPUTED, label="Imputed", linewidth=1.4)
            except Exception:
                xi, yi = ecdf_empirical(i_vals)
                ax.step(xi, yi, where="post", color=COLOR_IMPUTED, label="Imputed")
    else:
        if len(a_vals) > 0:
            xa, ya = ecdf_empirical(a_vals)
            ax.step(xa, ya, where="post", color=COLOR_ACTUAL, label="Actual")
        if len(i_vals) > 0:
            xi, yi = ecdf_empirical(i_vals)
            ax.step(xi, yi, where="post", color=COLOR_IMPUTED, label="Imputed")

    ax.set_xlim(xl)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("|Value|" if absolute else "Value")
    ax.set_ylabel("CDF")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize="small")


def main(argv=None):
    parser = argparse.ArgumentParser(description="CDF comparison of Imputed vs Actual per Site/Model/Regime/Target")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ecdf_mode", choices=("empirical", "kde"), default="empirical")
    parser.add_argument("--absolute", action="store_true", help="Plot CDFs of absolute values instead of signed")
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
        out_fp = out_dir / f"{safe_target}_imputed_vs_actual_cdf.png"

        fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 4), sharey=True)
        any_plot = False
        for i, pct in enumerate(DESIRED_MISSINGNESS):
            ax = axes[i]
            found_pair = None
            for k, (a_vals, i_vals) in miss_map.items():
                # attempt to coerce k to percent
                s = str(k)
                if s.endswith("%"):
                    s2 = s[:-1]
                else:
                    s2 = s
                try:
                    v = float(s2)
                    if 0 < v <= 1:
                        v = v * 100
                    k_pct = int(round(v))
                except Exception:
                    m = re.search(r"(\d+)", s)
                    k_pct = int(m.group(1)) if m else None
                if k_pct == pct:
                    found_pair = (a_vals, i_vals)
                    break
            title = f"{pct}%"
            if found_pair is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, 1.05)
                ax.set_title(title)
                continue
            any_plot = True
            actuals, imputeds = found_pair
            plot_cdf_panel(actuals, imputeds, ax, absolute=args.absolute, mode=args.ecdf_mode)
            ax.set_title(title)

        if not any_plot:
            skipped += 1
            plt.close(fig)
            continue

        fig.suptitle(f"{site} | {model} | {regime} | {target}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_fp, dpi=300)
        plt.close(fig)
        total += 1

    logging.info("Done. CDF panels generated: %d, skipped groups: %d", total, skipped)


if __name__ == "__main__":
    main()