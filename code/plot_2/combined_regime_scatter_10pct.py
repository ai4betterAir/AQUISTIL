#!/usr/bin/env python3
"""Plot region-merged observed-vs-imputed scatter panels for one model."""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aquistil_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/"
    "Imputation_Result"
)

REGIME_STYLE = {
    "random": {"label": "Random", "color": "#2458A6", "marker": "o", "background": "#EDF3FB"},
    "short_gap": {"label": "Short gap", "color": "#C58A1B", "marker": "s", "background": "#FCF6E9"},
    "medium_gap": {"label": "Medium gap", "color": "#7A68A6", "marker": "P", "background": "#F3F0F8"},
    "long_gap": {"label": "Long gap", "color": "#637A35", "marker": "^", "background": "#F1F4EA"},
    "event": {"label": "Event", "color": "#B54775", "marker": "D", "background": "#FAEEF3"},
}

CHARCOAL = "#2F3437"
GRID = "#E7EAED"
PREDICTION_FILE = "masked_predictions_by_site.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--target", default="PM25")
    parser.add_argument("--model", default="AQUISTIL")
    parser.add_argument("--sites", nargs="+", default=None, help="Optional station names to include.")
    parser.add_argument("--scope-label", default=None, help="Spatial label appended to the figure title.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the output directory.")
    parser.add_argument(
        "--plot-types",
        nargs="+",
        choices=["scatter", "hexbin", "density"],
        default=["scatter", "hexbin", "density"],
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Regions to include. The default discovers and merges every available region.",
    )
    parser.add_argument(
        "--missingness-percent",
        type=float,
        nargs="+",
        default=None,
        help="Optional percentages to include. The default plots every available level.",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=list(REGIME_STYLE),
        choices=list(REGIME_STYLE),
    )
    parser.add_argument(
        "--max-points-per-panel",
        type=int,
        default=5000,
        help="Maximum displayed points per panel; metrics still use every valid row.",
    )
    parser.add_argument(
        "--axis-percentile",
        type=float,
        default=99.9,
        help="Upper pooled Observed/Imputed percentile used for both axes (default: 99.9). Use 100 for the full range.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Scatter sampling seed.")
    return parser.parse_args()


def _prediction_files(args: argparse.Namespace) -> List[Path]:
    model_root = args.results_root / "Regional_Pooled_Imputation" / args.model
    files = sorted(model_root.glob(f"*/{args.target}/{PREDICTION_FILE}"))
    if args.regions:
        wanted = set(args.regions)
        files = [path for path in files if path.parent.parent.name in wanted]
    if not files:
        region_text = ", ".join(args.regions) if args.regions else "all regions"
        raise FileNotFoundError(
            f"No prediction CSVs found for model={args.model}, target={args.target}, regions={region_text} under {model_root}"
        )
    return files


def _load_predictions(files: List[Path], regimes: List[str]) -> pd.DataFrame:
    frames = []
    columns = ["Site", "Regime", "Missingness_Level", "Observed", "Imputed"]
    for path in files:
        part = pd.read_csv(path, usecols=columns)
        part = part.loc[part["Regime"].isin(regimes)].copy()
        part["Region"] = path.parent.parent.name
        frames.append(part)
    data = pd.concat(frames, ignore_index=True)
    for column in ["Missingness_Level", "Observed", "Imputed"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["Missingness_Level", "Observed", "Imputed"])


def _metrics(observed: pd.Series, imputed: pd.Series) -> Tuple[float, float]:
    obs = observed.to_numpy(dtype=float)
    imp = imputed.to_numpy(dtype=float)
    if len(obs) == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((imp - obs) ** 2)))
    if len(obs) > 1 and np.std(obs) > 0 and np.std(imp) > 0:
        correlation = float(np.corrcoef(obs, imp)[0, 1])
    else:
        correlation = float("nan")
    return rmse, correlation


def _format_pollutant(target: str) -> str:
    return {"PM25": "PM2.5", "PM10": "PM10", "NO2": "NO₂", "OZONE": "O₃"}.get(target, target)


def _draw_panel(
    ax,
    part,
    style,
    lower,
    upper,
    rng,
    max_points,
    metric_fontsize=7.5,
    tick_fontsize=7,
    plot_type="scatter",
):
    rmse, correlation = _metrics(part["Observed"], part["Imputed"])
    shown = part
    if max_points > 0 and len(part) > max_points:
        indices = rng.choice(len(part), max_points, replace=False)
        shown = part.iloc[indices]
    x = shown["Observed"].to_numpy(dtype=float)
    y = shown["Imputed"].to_numpy(dtype=float)
    density_cmap = LinearSegmentedColormap.from_list(
        f"density_{style['label'].replace(' ', '_')}", [style["background"], style["color"], "#241F24"]
    )
    if plot_type == "hexbin":
        ax.hexbin(
            part["Observed"],
            part["Imputed"],
            gridsize=55,
            extent=(lower, upper, lower, upper),
            mincnt=1,
            bins="log",
            cmap=density_cmap,
            linewidths=0,
            rasterized=True,
        )
    elif plot_type == "density":
        counts, x_edges, y_edges = np.histogram2d(
            x, y, bins=55, range=[[lower, upper], [lower, upper]]
        )
        x_bin = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, len(x_edges) - 2)
        y_bin = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, len(y_edges) - 2)
        point_density = np.log1p(counts[x_bin, y_bin])
        order = np.argsort(point_density)
        ax.scatter(
            x[order],
            y[order],
            c=point_density[order],
            cmap=density_cmap,
            s=9,
            edgecolors="none",
            alpha=0.72,
            rasterized=True,
        )
    else:
        ax.scatter(
            x,
            y,
            s=8,
            marker="o",
            color=style["color"],
            edgecolors="none",
            alpha=0.32,
            rasterized=True,
        )
    ax.plot([lower, upper], [lower, upper], color=CHARCOAL, linestyle="--", linewidth=0.8)
    ax.text(
        0.04,
        0.96,
        f"RMSE={rmse:.2f}\nR={correlation:.2f}\nn={len(part):,}",
        transform=ax.transAxes,
        va="top",
        fontsize=metric_fontsize,
        bbox={"facecolor": "none", "edgecolor": "none", "pad": 0},
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(style["background"])
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(CHARCOAL)
    ax.tick_params(colors=CHARCOAL, labelsize=tick_fontsize)


def main() -> None:
    args = _parse_args()
    files = _prediction_files(args)
    data = _load_predictions(files, args.regimes)
    if args.sites:
        wanted_sites = {site.strip().upper() for site in args.sites}
        data = data.loc[data["Site"].astype(str).str.strip().str.upper().isin(wanted_sites)].copy()
    # Concentrations below zero are physically invalid. Exclude them from both
    # the displayed points and all panel statistics.
    data = data.loc[data["Observed"].ge(0) & data["Imputed"].ge(0)].copy()

    if args.missingness_percent:
        wanted_levels = np.asarray(args.missingness_percent, dtype=float) / 100.0
        keep = np.zeros(len(data), dtype=bool)
        for level in wanted_levels:
            keep |= np.isclose(data["Missingness_Level"].to_numpy(), level)
        data = data.loc[keep]
    levels = sorted(data["Missingness_Level"].unique())
    regimes = [regime for regime in args.regimes if data["Regime"].eq(regime).any()]
    if data.empty or not levels or not regimes:
        raise ValueError("No rows match the selected model, target, regimes, and missingness levels.")

    if not 90 <= args.axis_percentile <= 100:
        raise ValueError("--axis-percentile must be between 90 and 100.")
    pooled_values = np.concatenate(
        [data["Observed"].to_numpy(dtype=float), data["Imputed"].to_numpy(dtype=float)]
    )
    lower = 0.0
    upper = float(np.percentile(pooled_values, args.axis_percentile))
    pad = max((upper - lower) * 0.04, 0.5)
    upper += pad

    nrows, ncols = len(regimes), len(levels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.05 * ncols, 2.85 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    rng = np.random.default_rng(args.seed)
    for row, regime in enumerate(regimes):
        style = REGIME_STYLE[regime]
        for col, level in enumerate(levels):
            ax = axes[row, col]
            part = data.loc[data["Regime"].eq(regime) & np.isclose(data["Missingness_Level"], level)]
            _draw_panel(ax, part, style, lower, upper, rng, args.max_points_per_panel)
            if row == 0:
                ax.set_title(f"{level * 100:g}% missing", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{style['label']}\nImputed {_format_pollutant(args.target)}", fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel(f"Observed {_format_pollutant(args.target)}", fontsize=9)

    regions = [path.parent.parent.name for path in files]
    fig.suptitle(
        f"{args.model}: observed vs imputed {_format_pollutant(args.target)} across all regions\n"
        f"Merged regions ({len(regions)}); axes show pooled 0.1–{args.axis_percentile:g} percentile range",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=12,
        fontweight="bold",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.045, top=0.91, wspace=0.12, hspace=0.09)

    output_dir = args.output_dir or (
        args.results_root / "plots_by_type" / "scatterplot" / "all_regions" / args.target / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = output_dir / "all_missingness_regimes"
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Publication-sized A4 landscape figure for each regime. Six standard
    # missingness levels are arranged 2 x 3 so individual panels remain legible.
    for plot_type in args.plot_types:
        for regime in regimes:
            style = REGIME_STYLE[regime]
            regime_fig, regime_axes = plt.subplots(
                2, 3, figsize=(11.69, 8.27), sharex=True, sharey=True, squeeze=False
            )
            flat_axes = regime_axes.ravel()
            for index, level in enumerate(levels):
                ax = flat_axes[index]
                part = data.loc[
                    data["Regime"].eq(regime) & np.isclose(data["Missingness_Level"], level)
                ]
                _draw_panel(
                    ax,
                    part,
                    style,
                    lower,
                    upper,
                    rng,
                    args.max_points_per_panel,
                    metric_fontsize=13.2,
                    tick_fontsize=15.6,
                    plot_type=plot_type,
                )
                ax.set_aspect("auto")
                ax.text(
                    0.50,
                    0.96,
                    f"{level * 100:g}%",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=19.2,
                    fontweight="bold",
                    bbox={"facecolor": "none", "edgecolor": "none", "pad": 0},
                )
            for ax in flat_axes[len(levels):]:
                ax.set_visible(False)
            for row_axes in regime_axes:
                for ax in row_axes[1:]:
                    ticks = ax.xaxis.get_major_ticks()
                    if ticks:
                        ticks[0].label1.set_visible(False)
            pollutant = _format_pollutant(args.target)
            regime_fig.text(0.5, 0.02, f"Observed {pollutant}", ha="center", fontsize=20.4)
            regime_fig.text(
                0.006, 0.5, f"Imputed {pollutant}", va="center", rotation="vertical", fontsize=20.4
            )
            scope_suffix = f" — {args.scope_label}" if args.scope_label else ""
            regime_fig.suptitle(
                f"{args.model}: {style['label'].title()}{scope_suffix}",
                fontsize=21.6,
                fontweight="bold",
                y=0.965,
            )
            regime_fig.subplots_adjust(
                left=0.065, right=0.99, bottom=0.085, top=0.90, wspace=0.025, hspace=0.10
            )
            regime_stem = output_dir / f"regime_{regime}_six_missingness_A4_{plot_type}"
            regime_fig.savefig(f"{regime_stem}.png", dpi=300, bbox_inches="tight")
            regime_fig.savefig(f"{regime_stem}.pdf", bbox_inches="tight")
            plt.close(regime_fig)
            print(f"Saved {regime_stem}.png")
            print(f"Saved {regime_stem}.pdf")

    print(f"Merged {len(files)} regions and {len(data):,} valid prediction rows")
    print(f"Saved {out_stem}.png")
    print(f"Saved {out_stem}.pdf")


if __name__ == "__main__":
    main()
