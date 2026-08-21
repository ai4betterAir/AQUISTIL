#!/usr/bin/env python3
"""Create all AQUISTIL scatter plots in one run (PNG only).

For every available pollutant, this standalone script creates six-missingness
A4 figures for all regions merged, every region, and every monitoring station.
Each regime is saved as a separate PNG. Negative Observed/Imputed values are
excluded from both the plots and the reported statistics.
"""

import argparse
import gc
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aquistil_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Imputation_Result"
)
PREDICTION_FILE = "masked_predictions_by_site.csv"
REGIME_STYLE = {
    "random": ("Random", "#2458A6", "#EDF3FB"),
    "short_gap": ("Short Gap", "#C58A1B", "#FCF6E9"),
    "medium_gap": ("Medium Gap", "#7A68A6", "#F3F0F8"),
    "long_gap": ("Long Gap", "#637A35", "#F1F4EA"),
    "event": ("Event", "#B54775", "#FAEEF3"),
}
REGIMES = list(REGIME_STYLE)
CHARCOAL = "#2F3437"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model", default="AQUISTIL")
    parser.add_argument("--targets", nargs="+", default=None, help="Default: all available pollutants.")
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=["all_regions", "region", "station"],
        default=["all_regions", "region", "station"],
    )
    parser.add_argument("--axis-percentile", type=float, default=99.9)
    parser.add_argument("--max-points-per-panel", type=int, default=5000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def slug(value):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_")


def pollutant_label(target):
    return {"PM25": "PM2.5", "PM10": "PM10", "NO2": "NO₂", "OZONE": "O₃"}.get(target, target)


def discover_targets(results_root, model):
    model_root = results_root / "Regional_Pooled_Imputation" / model
    return sorted({p.parent.name for p in model_root.glob("*/*/{}".format(PREDICTION_FILE))})


def load_target(results_root, model, target):
    model_root = results_root / "Regional_Pooled_Imputation" / model
    files = sorted(model_root.glob("*/{}/{}".format(target, PREDICTION_FILE)))
    frames = []
    columns = ["Site", "Regime", "Missingness_Level", "Observed", "Imputed"]
    for path in files:
        part = pd.read_csv(path, usecols=columns)
        part["Region"] = path.parent.parent.name
        frames.append(part)
    if not frames:
        return pd.DataFrame(), []
    data = pd.concat(frames, ignore_index=True)
    for column in ["Missingness_Level", "Observed", "Imputed"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Site", "Regime", "Missingness_Level", "Observed", "Imputed"])
    data = data.loc[
        data["Regime"].isin(REGIMES) & data["Observed"].ge(0) & data["Imputed"].ge(0)
    ].copy()
    return data, files


def metrics(part):
    observed = part["Observed"].to_numpy(dtype=float)
    imputed = part["Imputed"].to_numpy(dtype=float)
    if len(observed) == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((imputed - observed) ** 2)))
    if len(observed) > 1 and np.std(observed) > 0 and np.std(imputed) > 0:
        correlation = float(np.corrcoef(observed, imputed)[0, 1])
    else:
        correlation = float("nan")
    return rmse, correlation


def draw_panel(ax, part, colour, background, lower, upper, rng, max_points):
    rmse, correlation = metrics(part)
    shown = part
    if max_points > 0 and len(part) > max_points:
        shown = part.iloc[rng.choice(len(part), max_points, replace=False)]
    ax.scatter(
        shown["Observed"],
        shown["Imputed"],
        s=8,
        marker="o",
        color=colour,
        edgecolors="none",
        alpha=0.32,
        rasterized=True,
    )
    ax.plot([lower, upper], [lower, upper], color=CHARCOAL, linestyle="--", linewidth=0.9)
    ax.text(
        0.04,
        0.96,
        "RMSE={:.2f}\nR={:.2f}\nn={:,}".format(rmse, correlation, len(part)),
        transform=ax.transAxes,
        va="top",
        fontsize=13.2,
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("auto")
    ax.set_facecolor(background)
    ax.grid(False)
    ax.tick_params(colors=CHARCOAL, labelsize=15.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(CHARCOAL)


def render_scope(data, target, model, scope_label, output_dir, args):
    if data.empty:
        return 0
    levels = sorted(data["Missingness_Level"].unique())
    if len(levels) > 6:
        raise ValueError("Expected at most six missingness levels; found {}".format(len(levels)))
    pooled = np.concatenate(
        [data["Observed"].to_numpy(dtype=float), data["Imputed"].to_numpy(dtype=float)]
    )
    lower = 0.0
    upper = float(np.percentile(pooled, args.axis_percentile))
    upper += max(upper * 0.04, 0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    pollutant = pollutant_label(target)
    made = 0

    for regime in REGIMES:
        regime_data = data.loc[data["Regime"].eq(regime)]
        if regime_data.empty:
            continue
        regime_label, colour, background = REGIME_STYLE[regime]
        output = output_dir / "regime_{}_six_missingness_A4_scatter.png".format(regime)
        if output.exists() and not args.force:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), sharex=True, sharey=True, squeeze=False)
        flat_axes = axes.ravel()
        rng = np.random.RandomState(args.seed)
        for index, level in enumerate(levels):
            ax = flat_axes[index]
            part = regime_data.loc[np.isclose(regime_data["Missingness_Level"], level)]
            draw_panel(ax, part, colour, background, lower, upper, rng, args.max_points_per_panel)
            ax.text(
                0.50,
                0.96,
                "{:g}%".format(level * 100),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=19.2,
                fontweight="bold",
            )
        for ax in flat_axes[len(levels):]:
            ax.set_visible(False)
        for row_axes in axes:
            for ax in row_axes[1:]:
                ticks = ax.xaxis.get_major_ticks()
                if ticks:
                    ticks[0].label1.set_visible(False)
        fig.text(0.5, 0.02, "Observed {}".format(pollutant), ha="center", fontsize=20.4)
        fig.text(
            0.006,
            0.5,
            "Imputed {}".format(pollutant),
            va="center",
            rotation="vertical",
            fontsize=20.4,
        )
        fig.suptitle(
            "{}: {} — {}".format(model, regime_label, scope_label),
            fontsize=21.6,
            fontweight="bold",
            y=0.965,
        )
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.085, top=0.90, wspace=0.025, hspace=0.10)
        fig.savefig(str(output), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print("Saved {}".format(output), flush=True)
        made += 1
    return made


def main():
    args = parse_args()
    if not 90 <= args.axis_percentile <= 100:
        raise ValueError("--axis-percentile must be between 90 and 100")
    targets = args.targets or discover_targets(args.results_root, args.model)
    plot_root = args.results_root / "plots_by_type" / "scatterplot"
    total = 0
    for target in targets:
        print("Loading {}".format(target), flush=True)
        data, files = load_target(args.results_root, args.model, target)
        if data.empty:
            print("No valid data for {}; skipping".format(target), flush=True)
            continue
        if "all_regions" in args.scopes:
            total += render_scope(
                data,
                target,
                args.model,
                "All Regions",
                plot_root / "all_regions" / target / args.model,
                args,
            )
        if "region" in args.scopes:
            for region, region_data in data.groupby("Region", sort=True):
                total += render_scope(
                    region_data,
                    target,
                    args.model,
                    str(region).replace("_", " ").title(),
                    plot_root / "by_region" / target / args.model / str(region),
                    args,
                )
        if "station" in args.scopes:
            for (region, site), site_data in data.groupby(["Region", "Site"], sort=True):
                total += render_scope(
                    site_data,
                    target,
                    args.model,
                    str(site).title(),
                    plot_root / "by_station" / target / args.model / str(region) / slug(site),
                    args,
                )
        del data
        gc.collect()
    print("Completed: {} new PNG figures".format(total), flush=True)


if __name__ == "__main__":
    main()
