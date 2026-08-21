#!/usr/bin/env python3
"""Create six-panel regional RMSE box plots for model comparison.

Each figure is one pollutant, region, and masking regime.  The six panels are
the available missingness levels; within each panel, models are shown on the
x-axis and boxes contain site/run-level RMSE values for that region.
"""

import argparse
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
REGIME_LABELS = {
    "random": "Random",
    "short_gap": "Short Gap",
    "medium_gap": "Medium Gap",
    "long_gap": "Long Gap",
    "event": "Event",
}
MODEL_COLOURS = ["#2458A6", "#C58A1B", "#7A68A6", "#637A35", "#B54775", "#378A82"]
CHARCOAL = "#2F3437"
GRID = "#D9DEE2"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--targets", nargs="+", default=None, help="Default: all pollutants.")
    parser.add_argument("--models", nargs="+", default=None, help="Default: all models.")
    parser.add_argument(
        "--regimes", nargs="+", choices=list(REGIME_LABELS), default=list(REGIME_LABELS)
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def slug(value):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_")


def pollutant_label(target):
    return {
        "PM25": "PM2.5", "PM2.5": "PM2.5", "PM10": "PM10",
        "NO2": "NO₂", "OZONE": "O₃",
    }.get(target, target)


def region_label(region):
    return str(region).replace("_", " ").title()


def load_metrics(path, targets=None, models=None):
    columns = [
        "Region", "Site", "Target", "Model", "Regime",
        "Missingness_Level", "Seed", "Scope", "RMSE",
    ]
    data = pd.read_csv(path, usecols=columns)
    data["Missingness_Level"] = pd.to_numeric(data["Missingness_Level"], errors="coerce")
    data["RMSE"] = pd.to_numeric(data["RMSE"], errors="coerce")
    data = data.loc[data["Scope"].astype(str).str.casefold().eq("site")].copy()
    data = data.dropna(subset=["Region", "Site", "Target", "Model", "Regime", "Missingness_Level", "RMSE"])
    data = data.loc[np.isfinite(data["RMSE"]) & data["RMSE"].ge(0)]
    if targets:
        wanted = {"PM2.5" if target == "PM25" else target for target in targets}
        data = data.loc[data["Target"].isin(wanted)]
    if models:
        data = data.loc[data["Model"].isin(models)]
    return data


def draw_panel(ax, data, level, models):
    part = data.loc[np.isclose(data["Missingness_Level"], level)]
    present = [model for model in models if part["Model"].eq(model).any()]
    values = [part.loc[part["Model"].eq(model), "RMSE"].to_numpy(dtype=float) for model in present]
    if not present:
        ax.set_visible(False)
        return
    result = ax.boxplot(
        values,
        labels=present,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": CHARCOAL, "markersize": 4.5},
        medianprops={"color": CHARCOAL, "linewidth": 1.5},
        whiskerprops={"color": CHARCOAL, "linewidth": 1.0},
        capprops={"color": CHARCOAL, "linewidth": 1.0},
        flierprops={"marker": "o", "markeredgecolor": "none", "alpha": 0.45, "markersize": 3.5},
    )
    for index, box in enumerate(result["boxes"]):
        colour = MODEL_COLOURS[models.index(present[index]) % len(MODEL_COLOURS)]
        box.set(facecolor=colour, edgecolor=CHARCOAL, alpha=0.76, linewidth=1.0)
    for index, flier in enumerate(result["fliers"]):
        colour = MODEL_COLOURS[models.index(present[index]) % len(MODEL_COLOURS)]
        flier.set_markerfacecolor(colour)
    # Show exact observations when a box contains only a few sites/runs.
    for position, group in enumerate(values, start=1):
        if len(group) <= 10:
            offsets = np.linspace(-0.07, 0.07, len(group)) if len(group) > 1 else [0.0]
            ax.scatter(position + np.asarray(offsets), group, s=13, color=CHARCOAL, alpha=0.70, zorder=3)
    ax.set_title("{:g}% missing".format(level * 100), fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8.5)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.tick_params(axis="y", labelsize=9)
    ax.set_facecolor("#F6F8FA")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_figure(data, target, region, regime, models, output, args):
    if output.exists() and not args.force:
        return 0
    levels = sorted(data["Missingness_Level"].unique())
    if not levels:
        return 0
    if len(levels) > 6:
        raise ValueError("Expected at most six missingness levels; found {}".format(len(levels)))
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), sharey=True, squeeze=False)
    for index, level in enumerate(levels):
        draw_panel(axes.ravel()[index], data, level, models)
    for ax in axes.ravel()[len(levels):]:
        ax.set_visible(False)
    fig.suptitle(
        "Model RMSE by missingness — {} — {} — {}".format(
            pollutant_label(target), region_label(region), REGIME_LABELS.get(regime, regime)
        ),
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    fig.text(0.012, 0.5, "RMSE ({})".format(pollutant_label(target)), va="center", rotation="vertical", fontsize=14)
    fig.text(0.99, 0.01, "Boxes use site/run RMSE; diamond = mean", ha="right", fontsize=9, color=CHARCOAL)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.91, wspace=0.10, hspace=0.32)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output), flush=True)
    return 1


def main():
    args = parse_args()
    metrics_file = args.metrics_file or (args.results_root / "Metrics" / "regional_pooled_metrics.csv")
    data = load_metrics(metrics_file, args.targets, args.models)
    if data.empty:
        raise ValueError("No site-level RMSE metrics match the requested filters.")
    targets = sorted(data["Target"].unique())
    models = args.models or sorted(data["Model"].unique())
    output_root = args.results_root / "plots_by_type" / "boxplot" / "regional_six_missingness"
    made = 0
    for target in targets:
        target_data = data.loc[data["Target"].eq(target)]
        for region, region_data in target_data.groupby("Region", sort=True):
            for regime in args.regimes:
                selection = region_data.loc[region_data["Regime"].eq(regime)]
                output = (
                    output_root / slug(target) / slug(region)
                    / "{}_six_missingness_rmse_boxplots.png".format(regime)
                )
                made += render_figure(selection, target, region, regime, models, output, args)
    print("Completed: {} new PNG figures".format(made), flush=True)


if __name__ == "__main__":
    main()
