#!/usr/bin/env python3
"""Create all-region six-model observed-vs-imputed scatter comparisons.

Each figure is one pollutant, missingness level, and masking regime.  Its six
panels compare models after pooling every available region.  CSVs are processed
in chunks: panel metrics use every valid row while only a bounded random sample
is retained for display.
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
PREDICTION_FILE = "masked_predictions_by_site.csv"
REGIME_STYLE = {
    "random": ("Random", "#EDF3FB"),
    "short_gap": ("Short Gap", "#FCF6E9"),
    "medium_gap": ("Medium Gap", "#F3F0F8"),
    "long_gap": ("Long Gap", "#F1F4EA"),
    "event": ("Event", "#FAEEF3"),
}
MODEL_COLOURS = ["#2458A6", "#C58A1B", "#7A68A6", "#637A35", "#B54775", "#378A82"]
CHARCOAL = "#2F3437"
DEFAULT_MODELS = ["AQUISTIL", "LightGBM", "XGBoost", "MICE", "MICE-KNN", "KNN"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to display. Default: {}".format(", ".join(DEFAULT_MODELS)),
    )
    parser.add_argument("--targets", nargs="+", default=None, help="Default: all available pollutants.")
    parser.add_argument("--max-points-per-panel", type=int, default=5000)
    parser.add_argument("--axis-percentile", type=float, default=99.9)
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def pollutant_label(target):
    return {"PM25": "PM2.5", "PM2.5": "PM2.5", "PM10": "PM10", "NO2": "NO₂", "OZONE": "O₃"}.get(target, target)


def discover_models(results_root):
    root = results_root / "Regional_Pooled_Imputation"
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def discover_targets(results_root, models):
    root = results_root / "Regional_Pooled_Imputation"
    return sorted({path.parent.name for model in models for path in (root / model).glob("*/*/{}".format(PREDICTION_FILE))})


def empty_accumulator():
    return {
        "n": 0, "sx": 0.0, "sy": 0.0, "sx2": 0.0,
        "sy2": 0.0, "sxy": 0.0, "sse": 0.0,
        "x": np.empty(0), "y": np.empty(0), "key": np.empty(0),
    }


def update_accumulator(acc, x, y, rng, max_points):
    if len(x) == 0:
        return
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    acc["n"] += len(x)
    acc["sx"] += float(x.sum())
    acc["sy"] += float(y.sum())
    acc["sx2"] += float(np.dot(x, x))
    acc["sy2"] += float(np.dot(y, y))
    acc["sxy"] += float(np.dot(x, y))
    acc["sse"] += float(np.dot(y - x, y - x))
    if max_points <= 0:
        return
    keys = rng.random_sample(len(x))
    all_keys = np.concatenate([acc["key"], keys])
    all_x = np.concatenate([acc["x"], x])
    all_y = np.concatenate([acc["y"], y])
    if len(all_keys) > max_points:
        keep = np.argpartition(all_keys, -max_points)[-max_points:]
        all_keys, all_x, all_y = all_keys[keep], all_x[keep], all_y[keep]
    acc["key"], acc["x"], acc["y"] = all_keys, all_x, all_y


def stream_target(results_root, models, target, args):
    accumulators = {}
    rng = np.random.RandomState(args.seed)
    root = results_root / "Regional_Pooled_Imputation"
    columns = ["Regime", "Missingness_Level", "Observed", "Imputed"]
    for model in models:
        files = sorted((root / model).glob("*/{}/{}".format(target, PREDICTION_FILE)))
        for path in files:
            for chunk in pd.read_csv(path, usecols=columns, chunksize=args.chunksize):
                for column in ["Missingness_Level", "Observed", "Imputed"]:
                    chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
                chunk = chunk.dropna(subset=columns)
                chunk = chunk.loc[
                    chunk["Regime"].isin(REGIME_STYLE)
                    & chunk["Observed"].ge(0) & chunk["Imputed"].ge(0)
                ]
                for (regime, level), part in chunk.groupby(["Regime", "Missingness_Level"], sort=False):
                    key = (model, str(regime), float(level))
                    acc = accumulators.setdefault(key, empty_accumulator())
                    update_accumulator(acc, part["Observed"].to_numpy(), part["Imputed"].to_numpy(), rng, args.max_points_per_panel)
    return accumulators


def panel_metrics(acc):
    n = acc["n"]
    if n == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(acc["sse"] / n))
    numerator = n * acc["sxy"] - acc["sx"] * acc["sy"]
    denominator = np.sqrt(
        max(n * acc["sx2"] - acc["sx"] ** 2, 0.0)
        * max(n * acc["sy2"] - acc["sy"] ** 2, 0.0)
    )
    correlation = float(numerator / denominator) if denominator > 0 else float("nan")
    return rmse, correlation


def render_condition(accumulators, models, target, regime, level, output, args):
    if output.exists() and not args.force:
        return 0
    present = [model for model in models if (model, regime, level) in accumulators]
    if not present:
        return 0
    pooled = []
    for model in present:
        acc = accumulators[(model, regime, level)]
        pooled.extend([acc["x"], acc["y"]])
    pooled = np.concatenate([values for values in pooled if len(values)])
    upper = float(np.percentile(pooled, args.axis_percentile))
    upper += max(upper * 0.04, 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), sharex=True, sharey=True, squeeze=False)
    regime_label, background = REGIME_STYLE[regime]
    for index, model in enumerate(models[:6]):
        ax = axes.ravel()[index]
        key = (model, regime, level)
        if key not in accumulators:
            ax.set_title(model, fontsize=15, fontweight="bold")
            ax.set_facecolor("#F2F2F2")
            ax.text(0.5, 0.5, "No prediction data", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="#666666")
            ax.tick_params(labelbottom=False, labelleft=False, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        acc = accumulators[key]
        colour = MODEL_COLOURS[index % len(MODEL_COLOURS)]
        ax.scatter(acc["x"], acc["y"], s=8, color=colour, edgecolors="none", alpha=0.30, rasterized=True)
        ax.plot([0, upper], [0, upper], color=CHARCOAL, linestyle="--", linewidth=0.9)
        rmse, correlation = panel_metrics(acc)
        ax.text(0.04, 0.96, "RMSE={:.3g}\nR={:.3f}\nn={:,}".format(rmse, correlation, acc["n"]), transform=ax.transAxes, va="top", fontsize=11)
        ax.set_title(model, fontsize=15, fontweight="bold")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_facecolor(background)
        ax.grid(False)
        ax.tick_params(labelsize=10, colors=CHARCOAL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes.ravel()[len(models[:6]):]:
        ax.set_visible(False)
    pollutant = pollutant_label(target)
    fig.suptitle("All Regions — {} — {} — {:g}% missing".format(pollutant, regime_label, level * 100), fontsize=19, fontweight="bold", y=0.97)
    fig.text(0.5, 0.02, "Observed {}".format(pollutant), ha="center", fontsize=16)
    fig.text(0.008, 0.5, "Imputed {}".format(pollutant), va="center", rotation="vertical", fontsize=16)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.085, top=0.91, wspace=0.08, hspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output), flush=True)
    return 1


def main():
    args = parse_args()
    if len(args.models or []) > 6:
        raise ValueError("A maximum of six models can be displayed.")
    if not 90 <= args.axis_percentile <= 100:
        raise ValueError("--axis-percentile must be between 90 and 100")
    available_models = discover_models(args.results_root)
    models = args.models or DEFAULT_MODELS
    missing_models = [model for model in models if model not in available_models]
    if missing_models:
        raise ValueError(
            "Selected model output(s) not found: {}. Available models: {}".format(
                ", ".join(missing_models), ", ".join(available_models)
            )
        )
    if len(models) != 6:
        raise ValueError("Expected six models; found {}: {}".format(len(models), ", ".join(models)))
    targets = args.targets or discover_targets(args.results_root, models)
    output_root = args.results_root / "plots_by_type" / "scatterplot" / "all_regions_six_models"
    total = 0
    for target in targets:
        print("Streaming all models / {}".format(target), flush=True)
        accumulators = stream_target(args.results_root, models, target, args)
        conditions = sorted({(key[1], key[2]) for key in accumulators})
        for regime, level in conditions:
            output = output_root / target / regime / "{:g}pct_missing_six_models_scatter.png".format(level * 100)
            total += render_condition(accumulators, models, target, regime, level, output, args)
    print("Completed: {} new PNG figures".format(total), flush=True)


if __name__ == "__main__":
    main()
