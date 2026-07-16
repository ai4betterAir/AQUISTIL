#!/usr/bin/env python3
"""
research_plots.py

Generate individual heatmaps (one file per Target_Site × Missingness_Regime × Target).

Behavior:
- Uses an embedded default RESULTS_DIR and OUTPUT_DIR so the script can be run without CLI args.
- Prefers Target_Site (or TargetSite) column names; falls back to StudySite.
- Uses TARGET_COLUMNS and TARGET_SITES from config_spatial if available, otherwise uses the embedded defaults:
    TARGET_COLUMNS = ["PM2.5"]
    TARGET_SITES   = ["CHULLORA", "LIVERPOOL"]
- Aggregates / loads all_results_summary.csv if present; otherwise scans model/*/<regime>/metrics/*.csv and infers Target and Target_Site from filenames.
- Produces one PNG per combination, saved in OUTPUT_DIR.

Usage:
    python research_plots.py
    (or optionally pass CLI args --results_dir /path --output_dir /path --metric RMSE)
"""

from pathlib import Path
import argparse
import logging
import re
import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------------------
# Defaults (embedded in code so script can run without CLI)
# -------------------------------------------------------------------------
DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V19_final"
DEFAULT_OUTPUT_DIR = "./heatmaps"

# If config_spatial isn't present or doesn't provide these, these are used
DEFAULT_TARGET_COLUMNS = ["PM2.5"]
DEFAULT_TARGET_SITES = ["CHULLORA", "LIVERPOOL"]

DEFAULT_METRIC = "RMSE"

REGIME_ORDER = ["random", "short_gap", "medium_gap", "long_gap", "event"]
REGIME_LABELS = {
    "random": "Random (MCAR)",
    "short_gap": "Short Gap (1-23h)",
    "medium_gap": "Medium Gap (24-71h)",
    "long_gap": "Long Gap (72-240h)",
    "event": "Event-Dependent (MNAR)",
}

sns.set_theme(style="whitegrid")


# -------------------------
# Helpers: file parsing
# -------------------------
def extract_target_from_filename(stem: str):
    """Heuristic to extract pollutant/target token from filename stem (e.g., PM2.5)."""
    tokens = re.split(r"[_\-\.\s]", stem)
    pat = re.compile(r"^(pm\d+(\.\d+)?|pm\d+|no2|o3|so2|co|pm10|pm2_5)$", re.IGNORECASE)
    for t in tokens[::-1]:
        if pat.match(t.lower()):
            tt = t.upper().replace("_", ".")
            tt = re.sub(r"^PM25$", "PM2.5", tt)
            tt = re.sub(r"^PM2_5$", "PM2.5", tt)
            return tt
    for t in tokens[::-1]:
        if re.search(r"[A-Za-z]", t) and re.search(r"\d", t):
            return t
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and collapse duplicate columns; map common names to canonical names."""
    try:
        cols = []
        for c in df.columns:
            s = str(c)
            s = re.sub(r"\s+", " ", s).strip()
            s = re.sub(r"\.\d+$", "", s)
            cols.append(s)
        df.columns = cols
    except Exception:
        pass

    # collapse duplicate-like groups by normalized key
    try:
        def norm(x):
            s = str(x).lower()
            s = re.sub(r"\s+", "", s)
            s = re.sub(r"[^0-9a-z]", "", s)
            return s

        groups = {}
        for c in df.columns:
            groups.setdefault(norm(c), []).append(c)
        out = {}
        for k, mem in groups.items():
            if len(mem) == 1:
                out[mem[0]] = df[mem[0]]
            else:
                out[mem[0]] = df.loc[:, mem].bfill(axis=1).iloc[:, 0]
        df = pd.DataFrame(out)
    except Exception:
        pass

    rename_map = {
        "Root Mean Squared Error (RMSE)": "RMSE",
        "Mean Absolute Error (MAE)": "MAE",
        "Missingness": "Missingness_Pct",
        "Missingness_Level": "Missingness_Pct",
        "Missingness_Regime": "MISSINGNESS_REGIMES",
    }
    df = df.rename(columns=rename_map)

    if "Missingness_Pct" in df.columns:
        try:
            df["Missingness_Pct"] = pd.to_numeric(df["Missingness_Pct"], errors="coerce")
            if df["Missingness_Pct"].max() <= 1.0:
                df["Missingness_Pct"] = df["Missingness_Pct"] * 100.0
        except Exception:
            pass

    if "MISSINGNESS_REGIMES" not in df.columns:
        df["MISSINGNESS_REGIMES"] = "random"
    if "StudySite" not in df.columns:
        df["StudySite"] = df.get("StudySite", None)
    if "Target_Site" not in df.columns:
        # keep compatibility: if StudySite exists, copy to Target_Site
        if "StudySite" in df.columns:
            df["Target_Site"] = df["StudySite"]
    return df


def detect_site_column(df: pd.DataFrame) -> str | None:
    """Prefer 'Target_Site'/'TargetSite' over 'StudySite'."""
    candidates = list(df.columns)
    for cand in ["Target_Site", "TargetSite", "Target_Site_ID", "Target_SiteName"]:
        if cand in candidates:
            return cand
    for cand in ["StudySite", "Study_Site", "Site", "station", "location"]:
        if cand in candidates:
            return cand
    for c in candidates:
        if "target" in c.lower() and "site" in c.lower():
            return c
    for c in candidates:
        if "site" in c.lower():
            return c
    return None


def detect_target_column(df: pd.DataFrame) -> str | None:
    """Detect a 'Target' column if present."""
    for cand in ["Target", "target", "Target_Var", "TargetColumn", "TargetVariable"]:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "target" in c.lower() and "site" not in c.lower():
            return c
    return None


# -------------------------
# Load / aggregate function
# -------------------------
def load_aggregated(results_dir: Path) -> pd.DataFrame | None:
    """Load all_results_summary.csv if present, otherwise scan model/<regime>/metrics/*.csv."""
    agg_file = results_dir / "all_results_summary.csv"
    if agg_file.exists():
        logging.info(f"Loading aggregated CSV: {agg_file}")
        try:
            df = pd.read_csv(agg_file)
            return standardize_columns(df)
        except Exception as e:
            logging.warning(f"Failed to read aggregated CSV: {e} — will attempt scanning.")

    logging.info("Scanning results_dir for per-model metrics CSV files...")
    frames = []
    if not results_dir.exists():
        logging.error(f"Results directory not found: {results_dir}")
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
                    target_guess = extract_target_from_filename(stem)
                    dfm["Model"] = model_name
                    dfm["MISSINGNESS_REGIMES"] = regime
                    dfm["Target_Site"] = site_guess
                    dfm["StudySite"] = site_guess
                    dfm["Target"] = target_guess
                    frames.append(dfm)
                except Exception as e:
                    logging.warning(f"Unable to read {csv}: {e}")

    if not frames:
        logging.error("No metrics CSV files discovered while scanning results_dir.")
        return None

    df = pd.concat(frames, ignore_index=True)
    df = standardize_columns(df)
    try:
        df.to_csv(agg_file, index=False)
        logging.info(f"Saved aggregated CSV for future runs: {agg_file}")
    except Exception:
        pass
    return df


# -------------------------
# Single-heatmap function
# -------------------------
def plot_individual_heatmap(
    df: pd.DataFrame,
    site_col: str,
    site: str,
    regime: str,
    target_col: str | None,
    target: str | None,
    metric: str,
    output_dir: Path,
    annotate: bool = True,
    cmap: str = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path | None:
    """
    Build & save one heatmap for (site, regime, target).
    Rows = Model, Columns = Missingness_Pct, Values = mean(metric).
    """
    d = df.copy()

    # Target filter
    if target is not None and target_col is not None:
        d = d[d[target_col].astype(str) == str(target)]
    elif target is not None:
        logging.debug("Target requested but no target column available; continuing without target filter.")

    # Site filter
    if site_col not in d.columns:
        logging.error("Site column not present in dataframe; cannot filter by site.")
        return None
    d = d[d[site_col].astype(str) == str(site)]
    if d.empty:
        logging.info(f"No data for Target_Site={site}; skipping.")
        return None

    # Regime filter
    d = d[d["MISSINGNESS_REGIMES"].astype(str) == str(regime)]
    if d.empty:
        logging.info(f"No data for Target_Site={site}, Regime={regime}; skipping.")
        return None

    # Ensure missingness numeric
    if "Missingness_Pct" not in d.columns:
        logging.info("Missingness_Pct not present; skipping.")
        return None
    d["Missingness_Pct"] = pd.to_numeric(d["Missingness_Pct"], errors="coerce")
    d = d.dropna(subset=["Missingness_Pct"])
    if d.empty:
        logging.info("No numeric Missingness_Pct for selection; skipping.")
        return None

    if metric not in d.columns:
        logging.info(f"Metric '{metric}' not available for selection; skipping.")
        return None

    # Pivot
    try:
        pivot = d.pivot_table(values=metric, index="Model", columns="Missingness_Pct", aggfunc="mean")
    except Exception as e:
        logging.warning(f"Pivot creation failed: {e}")
        return None
    if pivot.empty:
        logging.info("Pivot empty after grouping; skipping.")
        return None

    # Sort columns numerically
    try:
        sorted_cols = sorted(pivot.columns, key=lambda x: float(x))
        pivot = pivot[sorted_cols]
    except Exception:
        pass

    # Sort rows by mean
    try:
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    except Exception:
        pass

    # Determine color scale if not provided: prefer global metric range
    if vmin is None or vmax is None:
        try:
            gv = pd.to_numeric(df[metric], errors="coerce")
            gv = gv[np.isfinite(gv)]
            if gv.size:
                gvmin, gvmax = float(gv.min()), float(gv.max())
            else:
                gvmin = gvmax = None
        except Exception:
            gvmin = gvmax = None
        vals = pivot.values.flatten()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            pmin, pmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        else:
            pmin = pmax = None
        vmin = vmin if vmin is not None else (gvmin if gvmin is not None else pmin)
        vmax = vmax if vmax is not None else (gvmax if gvmax is not None else pmax)
        if vmin is not None and vmax is not None and np.isclose(vmin, vmax):
            pad = 1e-6 if vmin != 0 else 0.1
            vmin -= pad
            vmax += pad

    # Plot
    fig_w = max(6, 0.8 * max(4, pivot.shape[1]))
    fig_h = max(4, 0.18 * max(6, pivot.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=annotate,
        fmt=".3f",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": metric},
        annot_kws={"fontsize": 8},
    )

    tgt_label = target if target is not None else "ALL"
    ax.set_title(
        f"{metric} — {site} — {REGIME_LABELS.get(regime, regime)} — {tgt_label}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Missingness (%)")
    ax.set_ylabel("Model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()

    safe_site = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(site))
    safe_regime = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(regime))
    safe_target = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(target)) if target else "ALL"
    out_name = f"heatmap_{metric}_{safe_site}_{safe_regime}_{safe_target}.png"
    out_path = output_dir / out_name
    try:
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        logging.info(f"Saved heatmap: {out_path}")
        return out_path
    except Exception as e:
        logging.error(f"Failed to save heatmap to {out_path}: {e}")
        plt.close(fig)
        return None


# -------------------------
# Batch wrapper
# -------------------------
def batch_generate_heatmaps(df: pd.DataFrame, output_dir: Path, metric: str = DEFAULT_METRIC, sites=None, regimes=None, targets=None, annotate=True):
    site_col = detect_site_column(df)
    if site_col is None:
        logging.error("No site column detected (expected 'Target_Site' or fallback). Aborting.")
        return []

    # try config_spatial defaults
    try:
        import config_spatial as cfg
        cfg_sites = getattr(cfg, "TARGET_SITES", None)
        cfg_targets = getattr(cfg, "TARGET_COLUMNS", None)
    except Exception:
        cfg_sites = None
        cfg_targets = None

    if sites is None:
        if cfg_sites:
            sites = list(cfg_sites)
        else:
            # fall back to defaults embedded in code
            sites = DEFAULT_TARGET_SITES if DEFAULT_TARGET_SITES else sorted(df[site_col].dropna().unique().tolist())

    if regimes is None:
        regimes = REGIME_ORDER

    if targets is None:
        if cfg_targets:
            targets = list(cfg_targets)
        else:
            # prefer explicit Target column if present
            tcol = detect_target_column(df)
            if tcol:
                targets = sorted(df[tcol].dropna().unique().tolist())
            else:
                targets = DEFAULT_TARGET_COLUMNS if DEFAULT_TARGET_COLUMNS else [None]

    saved = []
    tgt_col = detect_target_column(df)
    for tgt in targets:
        for s in sites:
            for r in regimes:
                try:
                    out = plot_individual_heatmap(df, site_col, s, r, tgt_col, tgt, metric, output_dir, annotate=annotate)
                    if out:
                        saved.append(out)
                except Exception as e:
                    logging.warning(f"Failed for site={s}, regime={r}, target={tgt}: {e}")
    logging.info(f"Batch completed: {len(saved)} heatmaps saved to {output_dir}")
    return saved


def plot_regime_average_summary(df: pd.DataFrame, output_dir: Path, metric: str = DEFAULT_METRIC):
    """Create a Models × Regime heatmap where each cell is the average of the metric
    across all missingness levels and sites for that (Model, Regime).

    Saves one PNG per target (or a single ALL file when no target column exists).
    """
    if metric not in df.columns:
        logging.warning(f"Metric '{metric}' not in dataframe; skipping regime-average summary.")
        return

    tgt_col = detect_target_column(df)
    targets = [None]
    if tgt_col is not None:
        targets = sorted(df[tgt_col].dropna().unique().tolist()) or [None]

    # global vmin/vmax from metric
    try:
        gv = pd.to_numeric(df[metric], errors="coerce")
        gv = gv[np.isfinite(gv)]
        vmin = float(gv.min()) if gv.size else None
        vmax = float(gv.max()) if gv.size else None
    except Exception:
        vmin = vmax = None

    for tgt in targets:
        d = df.copy()
        if tgt is not None and tgt_col is not None:
            d = d[d[tgt_col].astype(str) == str(tgt)]
        # average across Missingness_Pct and StudySite by grouping only by Model and Regime
        try:
            grp = d.groupby(["Model", "MISSINGNESS_REGIMES"])[metric].mean().unstack(fill_value=np.nan)
        except Exception as e:
            logging.warning(f"Failed to compute regime averages for target={tgt}: {e}")
            continue
        if grp.empty:
            logging.info(f"No data to plot for regime-average summary target={tgt}; skipping.")
            continue

        # order regimes per REGIME_ORDER if present
        cols = [c for c in REGIME_ORDER if c in grp.columns]
        if not cols:
            cols = list(grp.columns)
        grp = grp[cols]

        # sort models by mean across regimes for visual clarity
        try:
            grp = grp.loc[grp.mean(axis=1).sort_values().index]
        except Exception:
            pass

        fig_h = max(4, 0.25 * max(6, grp.shape[0]))
        fig_w = max(6, 1.5 * len(grp.columns))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sns.heatmap(grp, ax=ax, cmap="RdYlGn_r", annot=True, fmt=".3f", vmin=vmin, vmax=vmax,
                    cbar_kws={"label": metric}, linewidths=0.3)

        tgt_label = tgt if tgt is not None else "ALL"
        ax.set_title(f"{metric}: Models × Regime (averaged over missingness & sites) — Target: {tgt_label}")
        ax.set_xlabel("Missingness Regime")
        ax.set_ylabel("Model")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="center", fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)
        plt.tight_layout()

        safe_tgt = re.sub(r"[^0-9A-Za-z\-_\.]", "_", str(tgt_label))
        out = output_dir / f"Regime_Average_Heatmap_{metric}_{safe_tgt}.png"
        try:
            plt.savefig(out, dpi=300)
            plt.close(fig)
            logging.info(f"Saved regime-average heatmap: {out}")
        except Exception as e:
            logging.error(f"Failed to save regime-average heatmap to {out}: {e}")


# -------------------------
# Main (no required CLI)
# -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate per-site × per-regime × per-target heatmaps (defaults embedded in code).")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help=f"Results dir (default in code): {DEFAULT_RESULTS_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Output dir (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC, help=f"Metric to plot (default: {DEFAULT_METRIC})")
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated list of Target_Site values (optional; defaults in config_spatial or embedded list)")
    parser.add_argument("--regimes", type=str, default=None, help="Comma-separated regimes (optional; default all)")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated targets (optional; defaults in config_spatial or embedded)")
    parser.add_argument("--annot", action="store_true", help="Annotate heatmap cells with values")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Results dir: {results_dir}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Metric: {args.metric}")

    df = load_aggregated(results_dir)
    if df is None:
        logging.error("No aggregated data available; exiting.")
        return 1

    df = standardize_columns(df)

    # parse optional comma lists
    sites = [s.strip() for s in args.sites.split(",")] if args.sites else None
    regimes = [r.strip() for r in args.regimes.split(",")] if args.regimes else None
    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None

    # prepare metrics list: include the requested metric and MAE (mean absolute error) if available
    metrics = [args.metric]
    if 'MAE' in df.columns and 'MAE' not in metrics:
        metrics.append('MAE')

    for metric in metrics:
        logging.info(f"Generating plots for metric: {metric}")
        try:
            batch_generate_heatmaps(df=df, output_dir=output_dir, metric=metric, sites=sites, regimes=regimes, targets=targets, annotate=args.annot)
        except Exception as e:
            logging.warning(f"Batch generation failed for metric {metric}: {e}")
        # produce a regime-average summary (models × regime) averaged across missingness levels and sites
        try:
            plot_regime_average_summary(df=df, output_dir=output_dir, metric=metric)
        except Exception as e:
            logging.warning(f"Failed to generate regime-average summary plot for metric {metric}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())