#!/usr/bin/env python3
"""Create regional model RMSE heatmaps for every pollutant.

Each figure represents one pollutant and masking regime.  Its panels are the
available missingness levels, with regions on rows, models on columns, and
RMSE encoded by cell colour and printed exactly in every available cell.
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Default: <results-root>/Metrics/regional_pooled_metrics.csv",
    )
    parser.add_argument("--targets", nargs="+", default=None, help="Default: all pollutants.")
    parser.add_argument("--models", nargs="+", default=None, help="Default: all models.")
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=list(REGIME_LABELS),
        default=list(REGIME_LABELS),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="Overwrite existing figures.")
    return parser.parse_args()


def slug(value):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_")


def pollutant_label(target):
    return {
        "PM25": "PM2.5",
        "PM2.5": "PM2.5",
        "PM10": "PM10",
        "NO2": "NO₂",
        "OZONE": "O₃",
    }.get(target, target)


def region_label(region):
    return str(region).replace("_", " ").title()


def load_metrics(path, targets=None, models=None):
    columns = [
        "Region",
        "Target",
        "Model",
        "Regime",
        "Missingness_Level",
        "Scope",
        "RMSE",
    ]
    data = pd.read_csv(path, usecols=columns)
    data["Missingness_Level"] = pd.to_numeric(data["Missingness_Level"], errors="coerce")
    data["RMSE"] = pd.to_numeric(data["RMSE"], errors="coerce")
    data = data.loc[data["Scope"].astype(str).str.casefold().eq("region_macro")].copy()
    data = data.dropna(subset=["Region", "Target", "Model", "Regime", "Missingness_Level", "RMSE"])
    data = data.loc[np.isfinite(data["RMSE"]) & data["RMSE"].ge(0)]
    if targets:
        wanted = {"PM2.5" if target == "PM25" else target for target in targets}
        data = data.loc[data["Target"].isin(wanted)]
    if models:
        data = data.loc[data["Model"].isin(models)]
    return data


def matrix_for_level(data, level, regions, models):
    part = data.loc[np.isclose(data["Missingness_Level"], level)]
    # Mean safely handles duplicate seeds/runs while retaining one cell for each
    # region-model experimental condition.
    table = part.pivot_table(index="Region", columns="Model", values="RMSE", aggfunc="mean")
    return table.reindex(index=regions, columns=models).to_numpy(dtype=float)


def render_figure(data, target, regime, models, output, args):
    if output.exists() and not args.force:
        return 0
    levels = sorted(data["Missingness_Level"].unique())
    regions = sorted(data["Region"].unique())
    present_models = [model for model in models if data["Model"].eq(model).any()]
    if not levels or not regions or not present_models:
        return 0

    ncols = min(3, len(levels))
    nrows = int(np.ceil(len(levels) / float(ncols)))
    fig_width = max(11.69, 2.15 * len(present_models) * ncols)
    fig_height = max(8.27, 0.42 * len(regions) * nrows + 2.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    image = None
    finite_rmse = data["RMSE"].to_numpy(dtype=float)
    colour_max = float(np.max(finite_rmse))
    if colour_max <= 0:
        colour_max = 1.0

    for index, level in enumerate(levels):
        ax = axes.ravel()[index]
        matrix = matrix_for_level(data, level, regions, present_models)
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(
            masked, cmap="YlOrRd", vmin=0, vmax=colour_max,
            aspect="auto", interpolation="nearest"
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isfinite(value):
                    text_colour = "white" if value / colour_max >= 0.58 else "#222222"
                    ax.text(
                        column, row, "{:.4g}".format(value),
                        ha="center", va="center", fontsize=8.5,
                        color=text_colour, fontweight="bold",
                    )
        ax.set_title("{:g}% missing".format(level * 100), fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(len(present_models)))
        ax.set_xticklabels(present_models, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(np.arange(len(regions)))
        if index % ncols == 0:
            ax.set_yticklabels([region_label(region) for region in regions], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(-0.5, len(present_models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(regions), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes.ravel()[len(levels):]:
        ax.set_visible(False)

    fig.suptitle(
        "Regional model RMSE — {} — {}".format(
            pollutant_label(target), REGIME_LABELS.get(regime, regime)
        ),
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.12, right=0.91, bottom=0.10, top=0.92, wspace=0.10, hspace=0.30)
    colour_ax = fig.add_axes([0.93, 0.20, 0.015, 0.60])
    colour_bar = fig.colorbar(image, cax=colour_ax)
    colour_bar.set_label("RMSE (lower is better)", fontsize=12)
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
        raise ValueError("No regional RMSE metrics match the requested filters.")
    targets = sorted(data["Target"].unique())
    models = args.models or sorted(data["Model"].unique())
    output_root = args.results_root / "plots_by_type" / "rmse_heatmap"
    made = 0
    for target in targets:
        target_data = data.loc[data["Target"].eq(target)]
        for regime in args.regimes:
            regime_data = target_data.loc[target_data["Regime"].eq(regime)]
            output = output_root / slug(target) / "regional_model_{}_rmse_heatmap.png".format(regime)
            made += render_figure(regime_data, target, regime, models, output, args)
    print("Completed: {} new PNG figures".format(made), flush=True)


if __name__ == "__main__":
    main()
