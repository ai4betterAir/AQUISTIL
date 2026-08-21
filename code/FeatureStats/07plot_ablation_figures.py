#!/usr/bin/env python3
"""Create publication-style figures for the AQUISTIL ablation study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aquistil_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_ABLATION_ROOT = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/"
    "Imputation_Result/AQUISTIL_Ablation"
)

MODEL_ORDER = [
    "AQUISTIL_Full",
    "AQUISTIL_WithoutHistory",
    "AQUISTIL_WithoutSpatial",
    "AQUISTIL_WithoutGap",
    "AQUISTIL_WithoutEvent",
    "AQUISTIL_WithoutCyclicCalendar",
    "AQUISTIL_WithoutPosterior",
    "AQUISTIL_BackboneOnly",
]

MODEL_LABELS = {
    "AQUISTIL_Full": "Full",
    "AQUISTIL_WithoutHistory": "No history",
    "AQUISTIL_WithoutSpatial": "No spatial",
    "AQUISTIL_WithoutGap": "No gap",
    "AQUISTIL_WithoutEvent": "No event",
    "AQUISTIL_WithoutCyclicCalendar": "No cyclic calendar",
    "AQUISTIL_WithoutPosterior": "No posterior",
    "AQUISTIL_BackboneOnly": "Backbone only",
}

REGIME_ORDER = ["random", "short_gap", "medium_gap", "long_gap", "event"]

BLUE = "#2458A6"
GOLD = "#C58A1B"
ORANGE = "#D26A2C"
OLIVE = "#637A35"
PINK = "#B54775"
TEAL = "#2D7F7B"
VIOLET = "#6A5A9E"
SLATE = "#6E7781"
CHARCOAL = "#2F3437"
GREY = "#D7DCE0"
LIGHT_GREY = "#F3F5F6"

MODEL_COLORS = {
    "AQUISTIL_Full": BLUE,
    "AQUISTIL_WithoutHistory": GOLD,
    "AQUISTIL_WithoutSpatial": ORANGE,
    "AQUISTIL_WithoutGap": OLIVE,
    "AQUISTIL_WithoutEvent": PINK,
    "AQUISTIL_WithoutCyclicCalendar": VIOLET,
    "AQUISTIL_WithoutPosterior": TEAL,
    "AQUISTIL_BackboneOnly": SLATE,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=DEFAULT_ABLATION_ROOT,
        help="Directory containing ablation_metrics_long.csv.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional explicit metrics CSV path.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Output directory for figures. Default: <ablation-root>/figures.",
    )
    return parser.parse_args()


def _format_model(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("AQUISTIL_", "").replace("_", " "))


def _ordered_models(data: pd.DataFrame) -> list[str]:
    present = set(data["Model"].dropna())
    ordered = [model for model in MODEL_ORDER if model in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_metrics(metrics_csv: Path) -> pd.DataFrame:
    data = pd.read_csv(metrics_csv)
    data = data.loc[data["Scope"].eq("Region_Micro")].copy()
    if data.empty:
        raise ValueError(f"No Region_Micro rows found in {metrics_csv}")
    data["Model_Label"] = data["Model"].map(_format_model)
    data["Missingness_Percent"] = pd.to_numeric(data["Missingness_Percent"], errors="coerce")
    for metric in ["RMSE", "MAE", "R2", "NSE"]:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
    return data


def _plot_overall_ranking(data: pd.DataFrame, figures_dir: Path) -> None:
    summary = (
        data.groupby("Model", as_index=False)
        .agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), R2=("R2", "mean"), N=("RMSE", "size"))
        .sort_values("RMSE", ascending=True)
    )
    summary["Model_Label"] = summary["Model"].map(_format_model)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [BLUE if model == "AQUISTIL_Full" else GREY for model in summary["Model"]]
    bars = ax.barh(summary["Model_Label"], summary["RMSE"], color=colors, edgecolor=CHARCOAL, linewidth=0.6)
    ax.invert_yaxis()
    ax.set_title("Mean regional RMSE by ablation variant", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("RMSE, lower is better")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#ECEFF1", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, summary["RMSE"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f}", va="center", fontsize=9)
    sns.despine(ax=ax, left=True)
    _save(fig, figures_dir, "fig1_ablation_mean_rmse")


def _plot_delta_vs_full(data: pd.DataFrame, figures_dir: Path) -> None:
    keys = ["Region", "Target", "Regime", "Missingness_Level", "Seed"]
    full = data.loc[data["Model"].eq("AQUISTIL_Full"), keys + ["RMSE"]].rename(columns={"RMSE": "Full_RMSE"})
    paired = data.merge(full, on=keys, how="inner")
    paired["Delta_RMSE"] = paired["RMSE"] - paired["Full_RMSE"]
    summary = (
        paired.groupby("Model", as_index=False)
        .agg(Delta_RMSE=("Delta_RMSE", "mean"))
        .sort_values("Delta_RMSE", ascending=True)
    )
    summary["Model_Label"] = summary["Model"].map(_format_model)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [
        BLUE if model == "AQUISTIL_Full" else OLIVE if value < 0 else ORANGE
        for model, value in zip(summary["Model"], summary["Delta_RMSE"])
    ]
    bars = ax.barh(summary["Model_Label"], summary["Delta_RMSE"], color=colors, edgecolor=CHARCOAL, linewidth=0.6)
    ax.axvline(0, color=CHARCOAL, linewidth=0.9)
    ax.invert_yaxis()
    ax.set_title("Mean regional DEL RMSE versus full AQUISTIL", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("DEL RMSE relative to full model")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#ECEFF1", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, summary["Delta_RMSE"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.015 if value >= 0 else -0.015
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center", ha=ha, fontsize=9)
    sns.despine(ax=ax, left=True)
    _save(fig, figures_dir, "fig2_ablation_delta_rmse_vs_full")


def _plot_delta_boxplot_vs_full(data: pd.DataFrame, figures_dir: Path) -> None:
    keys = ["Region", "Target", "Regime", "Missingness_Level", "Seed"]
    full = data.loc[data["Model"].eq("AQUISTIL_Full"), keys + ["RMSE"]].rename(columns={"RMSE": "Full_RMSE"})
    paired = data.merge(full, on=keys, how="inner")
    paired["Delta_RMSE"] = paired["RMSE"] - paired["Full_RMSE"]

    order = (
        paired.groupby("Model")["Delta_RMSE"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    paired["Model_Label"] = paired["Model"].map(_format_model)
    label_order = [_format_model(model) for model in order]
    palette = {_format_model(model): MODEL_COLORS.get(model, GREY) for model in order}

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.boxplot(
        data=paired,
        y="Model_Label",
        x="Delta_RMSE",
        hue="Model_Label",
        order=label_order,
        hue_order=label_order,
        palette=palette,
        legend=False,
        width=0.58,
        linewidth=1.0,
        fliersize=0,
        saturation=0.82,
        ax=ax,
        boxprops={"edgecolor": CHARCOAL},
        medianprops={"color": CHARCOAL, "linewidth": 1.4},
        whiskerprops={"color": CHARCOAL, "linewidth": 1.0},
        capprops={"color": CHARCOAL, "linewidth": 1.0},
    )
    sns.stripplot(
        data=paired,
        y="Model_Label",
        x="Delta_RMSE",
        order=label_order,
        color=CHARCOAL,
        alpha=0.45,
        size=3,
        jitter=0.17,
        ax=ax,
    )
    ax.axvline(0, color=CHARCOAL, linewidth=0.9)
    ax.set_title("Regional DEL RMSE distribution versus full AQUISTIL", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("DEL RMSE relative to full model")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E7EAED", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    sns.despine(ax=ax, left=True)
    _save(fig, figures_dir, "fig5_ablation_delta_rmse_boxplot")


def _plot_regime_heatmap(data: pd.DataFrame, figures_dir: Path) -> None:
    models = _ordered_models(data)
    regime_order = [regime for regime in REGIME_ORDER if regime in set(data["Regime"])]
    pivot = (
        data.groupby(["Regime", "Model"])["RMSE"]
        .mean()
        .unstack("Model")
        .reindex(index=regime_order, columns=models)
    )
    pivot.columns = [_format_model(model) for model in pivot.columns]
    pivot.index = [label.replace("_", " ").title() for label in pivot.index]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=sns.light_palette(BLUE, as_cmap=True),
        annot=True,
        fmt=".2f",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Mean RMSE"},
    )
    ax.set_title("Mean regional RMSE by missingness regime and variant", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    _save(fig, figures_dir, "fig3_ablation_rmse_regime_heatmap")


def _plot_missingness_curves(data: pd.DataFrame, figures_dir: Path) -> None:
    summary = (
        data.groupby(["Model", "Missingness_Percent"], as_index=False)
        .agg(RMSE=("RMSE", "mean"))
    )
    models = _ordered_models(data)
    palette = {
        "AQUISTIL_Full": BLUE,
        "AQUISTIL_WithoutHistory": GOLD,
        "AQUISTIL_WithoutSpatial": ORANGE,
        "AQUISTIL_WithoutGap": OLIVE,
        "AQUISTIL_WithoutEvent": PINK,
        "AQUISTIL_WithoutCyclicCalendar": "#6A5A9E",
        "AQUISTIL_WithoutPosterior": "#4E7D78",
        "AQUISTIL_BackboneOnly": "#7B7F84",
    }

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for model in models:
        part = summary.loc[summary["Model"].eq(model)].sort_values("Missingness_Percent")
        if part.empty:
            continue
        line_width = 2.7 if model == "AQUISTIL_Full" else 1.7
        ax.plot(
            part["Missingness_Percent"],
            part["RMSE"],
            marker="o",
            linewidth=line_width,
            color=palette.get(model, GREY),
            label=_format_model(model),
        )
    ax.set_title("Mean regional RMSE across missingness levels", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Artificial missingness (%)")
    ax.set_ylabel("RMSE, lower is better")
    ax.grid(color="#ECEFF1", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    sns.despine(ax=ax)
    _save(fig, figures_dir, "fig4_ablation_rmse_by_missingness")


def main() -> None:
    args = _parse_args()
    metrics_csv = args.metrics_csv or args.ablation_root / "ablation_metrics_long.csv"
    figures_dir = args.figures_dir or args.ablation_root / "figures"

    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "text.color": CHARCOAL,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    data = _load_metrics(metrics_csv)
    _plot_overall_ranking(data, figures_dir)
    _plot_delta_vs_full(data, figures_dir)
    _plot_delta_boxplot_vs_full(data, figures_dir)
    _plot_regime_heatmap(data, figures_dir)
    _plot_missingness_curves(data, figures_dir)
    print(f"Saved ablation figures to {figures_dir}")


if __name__ == "__main__":
    main()
