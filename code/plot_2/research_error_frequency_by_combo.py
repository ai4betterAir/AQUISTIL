#!/usr/bin/env python3
"""
research_error_frequency_by_combo.py

Scan Imputed_Results CSV files and create one error-frequency histogram per:
  - StudySite
  - Model
  - Regime
  - Missingness_Level
  - Target

Defaults: signed (normal) error plots (Imputed - Actual).
If you prefer absolute errors, pass --absolute.

Run:
  python research_error_frequency_by_combo.py

Outputs:
  PNGs saved to:
    <output_dir>/<StudySite>/<Model>/<Regime>/<Missingness_Level>/<Target>_error_freq.png
"""
from pathlib import Path
import argparse
import logging
import sys
import re
from typing import Optional

# Headless backend
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
DEFAULT_BINS = 20

DEFAULT_XLIMIT = (-50, 50)
# color palette for up to four missingness levels (10,20,30,50)
MISSINGNESS_COLORS = {
    10: "#1f77b4",  # blue
    20: "#ff7f0e",  # orange
    30: "#2ca02c",  # green
    50: "#d62728",  # red
}


def _missingness_to_pct(miss_label):
    """Try to coerce a missingness label to an integer percent (10,20,30,50).
    Accepts strings like '0.1', '0.1 (10%)', '10', '10.0', '10%'. Returns int or None.
    """
    if miss_label is None:
        return None
    s = str(miss_label).strip()
    if s.endswith("%"):
        s = s[:-1]
    # try float
    try:
        v = float(s)
        if 0.0 < v <= 1.0:
            v = v * 100.0
        return int(round(v))
    except Exception:
        # try to extract first integer
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
    return None


def get_color_for_missingness(miss_label):
    pct = _missingness_to_pct(miss_label)
    if pct is None:
        return "tab:gray"
    return MISSINGNESS_COLORS.get(pct, "tab:gray")


def find_imputed_csvs(results_dir: Path):
    imputed_dir = results_dir / "Imputed_Results"
    if not imputed_dir.exists() or not imputed_dir.is_dir():
        logging.error("Imputed_Results directory not found at: %s", imputed_dir)
        return []
    return sorted(imputed_dir.glob("*.csv"))


def _find_column(df: pd.DataFrame, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def load_and_extract_errors(fp: Path, comments_values=("imputed",), require_actual=True):
    """
    Read file and return a list of info dicts with 'errors' array and metadata.
    Uses only rows where Comments indicates 'imputed' (case-insensitive) and Actual present.
    """
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception as e:
        logging.warning("Failed to read %s: %s", fp, e)
        return []

    site_col = _find_column(df, ["Site", "StudySite", "Study_Site"])
    model_col = _find_column(df, ["Model"])
    regime_col = _find_column(df, ["Regime", "Missingness_Regime"])
    missing_level_col = _find_column(df, ["Missingness_Level", "Missingness", "Missingness_Pct"])
    target_col = _find_column(df, ["Target"])
    actual_col = _find_column(df, ["Actual", "Observed", "True"])
    imputed_col = _find_column(df, ["Imputed", "Prediction", "Predicted", "Imputed_Value"])
    comments_col = _find_column(df, ["Comments", "Missing_Type", "MissingType", "Missingness_Type"])

    if imputed_col is None:
        logging.debug("No imputed column detected in %s -> skipping", fp)
        return []

    # filename fallback parsing
    stem = fp.stem
    parts = stem.split("_")
    fname_site = parts[0] if len(parts) >= 1 else ""
    fname_target = parts[1] if len(parts) >= 2 else ""
    fname_model = "_".join(parts[2:-1]) if len(parts) > 3 else (parts[2] if len(parts) >= 3 else "")
    fname_last = parts[-1] if parts else ""

    meta_site = None if site_col is not None else (fname_site or "")
    meta_model = None if model_col is not None else (fname_model or "")
    meta_target = None if target_col is not None else (fname_target or "")

    mask = pd.Series(True, index=df.index)

    # Filter to rows flagged as Imputed (Comments)
    if comments_col is not None:
        comments_ser = df[comments_col].astype(str).fillna("").str.strip().str.lower()
        wanted = [v.lower() for v in comments_values]
        mask_comments = comments_ser.isin(wanted) | comments_ser.str.contains("|".join([re.escape(w) for w in wanted]), na=False)
        mask &= mask_comments
    else:
        # If no Comments column, try heuristic: rows where Imputed != Actual or Actual is NaN but Imputed present
        logging.debug("No Comments column in %s; will use heuristic selection", fp)

    # Must have imputed numeric
    try:
        mask_imputed = pd.to_numeric(df[imputed_col], errors="coerce").notna()
    except Exception:
        mask_imputed = df[imputed_col].notna()
    mask &= mask_imputed

    # Require Actual present (default)
    if require_actual:
        if actual_col is None:
            logging.debug("Actual column missing in %s; cannot compute errors -> skipping", fp)
            return []
        mask_actual = pd.to_numeric(df[actual_col], errors="coerce").notna()
        mask &= mask_actual

    # If Comments missing, try heuristic selection now
    if comments_col is None:
        # prefer rows where actual present and imputed differs OR actual was NaN and imputed present
        if actual_col is not None:
            a = pd.to_numeric(df[actual_col], errors="coerce")
            b = pd.to_numeric(df[imputed_col], errors="coerce")
            heuristic_mask = (~a.isna()) & (~b.isna()) & (a != b)
            if heuristic_mask.sum() == 0:
                heuristic_mask = (a.isna()) & (~b.isna())
            mask &= heuristic_mask

    sel = df.loc[mask]
    if sel.empty:
        logging.debug("No qualifying rows in %s after filtering", fp)
        return []

    # numeric imputed/actual values
    imputed_vals = pd.to_numeric(sel[imputed_col], errors="coerce")
    actual_vals = pd.to_numeric(sel[actual_col], errors="coerce") if actual_col is not None else pd.Series([np.nan]*len(imputed_vals), index=imputed_vals.index)
    valid_pair = imputed_vals.notna() & actual_vals.notna()
    if not valid_pair.any():
        logging.debug("No numeric imputed+actual pairs in %s after filtering", fp)
        return []

    # compute per-row errors (Series aligned to sel.index)
    errors_series = (imputed_vals - actual_vals).loc[valid_pair]

    def _first_nonnull(col, fallback):
        if col is None:
            return fallback
        try:
            v = df[col].dropna().astype(str).iloc[0]
            return v
        except Exception:
            return fallback

    site_val = _first_nonnull(site_col, meta_site or "")
    model_val = _first_nonnull(model_col, meta_model or "")
    regime_val = _first_nonnull(regime_col, "")
    target_val = _first_nonnull(target_col, meta_target or "")

    # Build a per-row missingness label series (try Missingness column; else extract from Comments; else filename)
    if missing_level_col is not None:
        raw_miss = sel.loc[valid_pair.index, missing_level_col].astype(str).fillna("").str.strip()
    else:
        raw_miss = pd.Series([""] * len(errors_series), index=errors_series.index)
        if comments_col is not None:
            comments_ser = sel.loc[valid_pair.index, comments_col].astype(str).fillna("")
            def _extract_from_text(s: str):
                # look for patterns like '10%', '0.1', '10 (10%)', 'missingness=10'
                m = re.search(r"(\d+\.?\d*)\s*%", s)
                if m:
                    return m.group(1)
                m = re.search(r"(\d+\.?\d*)", s)
                if m:
                    return m.group(1)
                return ""
            raw_miss = comments_ser.apply(_extract_from_text).astype(str).str.strip()

    # If still empty labels, try filename tail
    if raw_miss.isnull().all() or (raw_miss == "").all():
        candidate = fname_last
        pct = _missingness_to_pct(candidate)
        fname_label = str(int(pct)) if pct is not None else (candidate or "")
        raw_miss = pd.Series([fname_label] * len(errors_series), index=errors_series.index)

    results = []
    # group by normalized missingness label
    for raw_label in sorted(raw_miss.unique()):
        if raw_label is None:
            continue
        if str(raw_label).strip() == "":
            label_norm = ""
        else:
            pct = _missingness_to_pct(raw_label)
            label_norm = str(int(pct)) if pct is not None else str(raw_label).strip()
        idx = raw_miss[raw_miss == raw_label].index
        group_errors = errors_series.loc[idx].to_numpy(dtype=float)
        if group_errors.size == 0:
            continue
        results.append({
            "StudySite": str(site_val).strip(),
            "Model": str(model_val).strip(),
            "Regime": str(regime_val).strip(),
            "Missingness_Level": str(label_norm).strip(),
            "Target": str(target_val).strip(),
            "errors": group_errors,
            "source_file": str(fp)
        })

    return results


def plot_histogram_signed(errors: np.ndarray, out_fp: Path = None, bins: int = 20, normalize: bool = False, title: Optional[str] = None, ylim_top: Optional[int] = None, color: Optional[str] = None, ax: Optional[plt.Axes] = None):
    """
    Plot signed error histogram. Bins are symmetric around zero so negative/positive errors visible.
    """
    if errors is None or len(errors) == 0:
        return False
    vals = errors
    # symmetric bins around zero
    max_abs = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)))
    if max_abs == 0:
        edges = np.linspace(-1.0, 1.0, bins + 1)
    else:
        edges = np.linspace(-max_abs, max_abs, bins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    if normalize:
        total = counts.sum()
        counts = counts / total if total > 0 else counts
    centers = 0.5 * (edges[:-1] + edges[1:])
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True
    bar_color = color or "tab:blue"
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=bar_color, alpha=0.9)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Error (Imputed - Actual)")
    ax.set_ylabel("Proportion" if normalize else "Frequency")
    # enforce global x limits and make y limits optional (caller can set ylim)
    ax.set_xlim(*DEFAULT_XLIMIT)
    # set sensible xticks
    ax.set_xticks(np.linspace(DEFAULT_XLIMIT[0], DEFAULT_XLIMIT[1], 5))
    if title:
        ax.set_title(title)
    if ylim_top is not None:
        try:
            ax.set_ylim(0, int(ylim_top))
            ax.set_yticks(np.linspace(0, int(ylim_top), 5))
        except Exception:
            pass
    if created_fig:
        plt.tight_layout()
        if out_fp is not None:
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_fp, dpi=300)
        plt.close(fig)
    return True


def plot_histogram_absolute(errors: np.ndarray, out_fp: Path = None, bins: int = 20, normalize: bool = False, title: Optional[str] = None, ylim_top: Optional[int] = None, color: Optional[str] = None, ax: Optional[plt.Axes] = None):
    if errors is None or len(errors) == 0:
        return False
    vals = np.abs(errors)
    # use same symmetric edges so x ticks stay consistent (-50..50)
    edges = np.linspace(DEFAULT_XLIMIT[0], DEFAULT_XLIMIT[1], bins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    if normalize:
        total = counts.sum()
        counts = counts / total if total > 0 else counts
    centers = 0.5 * (edges[:-1] + edges[1:])
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True
    bar_color = color or "tab:green"
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=bar_color, alpha=0.9)
    ax.set_xlabel("|Error| = |Imputed - Actual|")
    ax.set_ylabel("Proportion" if normalize else "Frequency")
    ax.set_xlim(*DEFAULT_XLIMIT)
    ax.set_xticks(np.linspace(DEFAULT_XLIMIT[0], DEFAULT_XLIMIT[1], 5))
    if title:
        ax.set_title(title)
    if ylim_top is not None:
        try:
            ax.set_ylim(0, int(ylim_top))
            ax.set_yticks(np.linspace(0, int(ylim_top), 5))
        except Exception:
            pass
    if created_fig:
        plt.tight_layout()
        if out_fp is not None:
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_fp, dpi=300)
        plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create error-frequency histograms per Site/Model/Regime/Missingness/Target")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="Root results dir")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save plots")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Histogram bins")
    parser.add_argument("--absolute", action="store_true", help="Plot absolute error instead of signed (default is signed)")
    parser.add_argument("--normalize", action="store_true", help="Normalize counts to proportions")
    parser.add_argument("--require_actual", action="store_true", default=True, help="Require Actual present for Imputed rows (default True)")
    parser.add_argument("--max_files", type=int, default=None, help="If set, process only this many imputed CSVs (debug)")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logging.error("Results directory does not exist: %s", results_dir)
        sys.exit(1)

    csvs = find_imputed_csvs(results_dir)
    if not csvs:
        logging.error("No imputed CSV files found under %s/Imputed_Results", results_dir)
        sys.exit(1)

    total_plots = 0
    skipped = 0

    if args.max_files is not None:
        csvs = csvs[: args.max_files]

    # First pass: compute per-site max histogram height so y-axis can be consistent per site
    site_max = {}
    edges = np.linspace(DEFAULT_XLIMIT[0], DEFAULT_XLIMIT[1], args.bins + 1)
    for fp in csvs:
        entries = load_and_extract_errors(fp, comments_values=("imputed",), require_actual=args.require_actual)
        if not entries:
            continue
        for info in entries:
            site = info["StudySite"] or "UNKNOWN_SITE"
            errors = info["errors"]
            if errors is None or len(errors) == 0:
                continue
            vals = np.abs(errors) if args.absolute else errors
            counts, _ = np.histogram(vals, bins=edges)
            maxc = int(counts.max()) if counts.size else 0
            site_max[site] = max(site_max.get(site, 0), maxc)

    # Second pass: generate plots with consistent y-limits
    # Build grouped mapping: (site, model, regime, target) -> {miss_level: errors}
    grouped = {}
    for fp in csvs:
        entries = load_and_extract_errors(fp, comments_values=("imputed",), require_actual=args.require_actual)
        if not entries:
            continue
        for info in entries:
            site = info["StudySite"] or "UNKNOWN_SITE"
            model = info["Model"] or "UNKNOWN_MODEL"
            regime = info["Regime"] or "unknown_regime"
            target = info["Target"] or "target"
            miss = info["Missingness_Level"] or "missing"
            errors = info["errors"]
            if errors is None or len(errors) == 0:
                continue
            key = (site, model, regime, target)
            grouped.setdefault(key, {})[str(miss)] = errors

    # Desired missingness order and labels
    desired = [10, 20, 30, 50]
    for (site, model, regime, target), miss_map in grouped.items():
        safe_site = re.sub(r"[^0-9A-Za-z\-_\.]", "_", site)
        safe_model = re.sub(r"[^0-9A-Za-z\-_\.]", "_", model)
        safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", regime)
        safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", target)
        out_dir = Path(args.output_dir) / safe_site / safe_model / safe_regime
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / f"{safe_target}_missingness_panels.png"

        # create 4-column subplot figure
        fig, axes = plt.subplots(1, 4, figsize=(4 * 5, 4), sharey=True)
        ylim_top = site_max.get(site, None)
        any_plotted = False
        for i, pct in enumerate(desired):
            miss_key_variants = [str(pct), f"{pct}", f"{pct}.0", f"0.{int(pct/10)}"]
            # find first matching missingness in miss_map
            errors = None
            for k in miss_key_variants:
                if k in miss_map:
                    errors = miss_map[k]
                    break
            title = f"{pct}%"
            color = get_color_for_missingness(pct)
            ax = axes[i]
            if errors is None:
                # blank axes label
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_xlim(*DEFAULT_XLIMIT)
                ax.set_xticks(np.linspace(DEFAULT_XLIMIT[0], DEFAULT_XLIMIT[1], 5))
                if ylim_top is not None:
                    ax.set_ylim(0, int(ylim_top))
                ax.set_title(title)
                continue
            any_plotted = True
            if args.absolute:
                plot_histogram_absolute(errors, out_fp=None, bins=args.bins, normalize=args.normalize, title=title, ylim_top=ylim_top, color=color, ax=ax)
            else:
                plot_histogram_signed(errors, out_fp=None, bins=args.bins, normalize=args.normalize, title=title, ylim_top=ylim_top, color=color, ax=ax)

        if any_plotted:
            fig.suptitle(f"{site} | {model} | {regime} | {target}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(out_fp, dpi=300)
            plt.close(fig)
            total_plots += 1
        else:
            skipped += 1

    logging.info("Finished. Plots generated: %d, skipped: %d (files processed: %d)", total_plots, skipped, len(csvs))


if __name__ == "__main__":
    main()