#!/usr/bin/env python3
"""Plot paired AQUISTIL ablation improvements from the comparison metrics table."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aquistil_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List


DEFAULT_METRICS_DIR = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/"
    "Imputation_Results/Metrics_with_Ablation"
)

MODEL_ORDER = [
    "AQUISTIL_NoHistory",
    "AQUISTIL_NoHistoryNoEvent",
    "AQUISTIL_NoFFill",
    "AQUISTIL_NoAdaptive",
    "AQUISTIL_ExogenousOnly",
    "AQUISTIL_NoAQUISTILFeatures",
]

MODEL_LABELS = {
    "AQUISTIL_NoHistory": "No history",
    "AQUISTIL_NoHistoryNoEvent": "No history + no event",
    "AQUISTIL_NoFFill": "No forward fill",
    "AQUISTIL_NoAdaptive": "No adaptive routing",
    "AQUISTIL_ExogenousOnly": "Exogenous only",
    "AQUISTIL_NoAQUISTILFeatures": "No AQUISTIL features",
}

REGIME_ORDER = ["random", "short_gap", "medium_gap", "long_gap", "event"]

BLUE = "#2458A6"
GOLD = "#C58A1B"
ORANGE = "#D26A2C"
OLIVE = "#637A35"
PINK = "#B54775"
CHARCOAL = "#2F3437"
GREY = "#D7DCE0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Directory containing aquistil_ablation_comparison.csv.",
    )
    parser.add_argument(
        "--scope",
        default="Region_Micro",
        choices=["Region_Macro", "Region_Micro", "Site"],
        help="Evaluation scope to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory. Default: <metrics-dir>/ablation_improvement_plots.",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("AQUISTIL_", "").replace("_", " "))


def load_paired_deltas(metrics_dir: Path, scope: str) -> pd.DataFrame:
    path = metrics_dir / "aquistil_ablation_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison CSV: {path}")

    frame = pd.read_csv(path)
    frame = frame.loc[frame["Scope"].eq(scope)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for Scope={scope!r} in {path}")

    base_col = "RMSE_AQUISTIL"
    if base_col not in frame.columns:
        raise ValueError(f"Missing required baseline column: {base_col}")

    rows = []  # type: List[pd.DataFrame]
    for model in MODEL_ORDER:
        col = f"RMSE_{model}"
        if col not in frame.columns:
            continue
        part = frame[
            [
                "Region",
                "Site",
                "Target",
                "Regime",
                "Missingness_Level",
                "Missingness_Percent",
                "Seed",
                "Scope",
                base_col,
                col,
            ]
        ].copy()
        part = part.rename(columns={base_col: "RMSE_AQUISTIL", col: "RMSE_Ablated"})
        part["Model"] = model
        rows.append(part)

    if not rows:
        raise ValueError("No AQUISTIL ablation RMSE columns were found.")

    paired = pd.concat(rows, ignore_index=True)
    paired["RMSE_AQUISTIL"] = pd.to_numeric(paired["RMSE_AQUISTIL"], errors="coerce")
    paired["RMSE_Ablated"] = pd.to_numeric(paired["RMSE_Ablated"], errors="coerce")
    paired["Missingness_Percent"] = pd.to_numeric(
        paired["Missingness_Percent"], errors="coerce"
    )
    paired = paired.dropna(subset=["RMSE_AQUISTIL", "RMSE_Ablated"])
    paired["Ablation_Improvement_RMSE"] = paired["RMSE_AQUISTIL"] - paired["RMSE_Ablated"]
    paired["Ablation_Improvement_Percent"] = np.where(
        paired["RMSE_AQUISTIL"].ne(0),
        100.0 * paired["Ablation_Improvement_RMSE"] / paired["RMSE_AQUISTIL"],
        np.nan,
    )
    paired["Model_Label"] = paired["Model"].map(model_label)
    return paired


def build_summary(paired: pd.DataFrame) -> pd.DataFrame:
    summary = (
        paired.groupby(["Model", "Model_Label"], as_index=False)
        .agg(
            Mean_Ablation_Improvement_RMSE=("Ablation_Improvement_RMSE", "mean"),
            Median_Ablation_Improvement_RMSE=("Ablation_Improvement_RMSE", "median"),
            Mean_Ablation_Improvement_Percent=(
                "Ablation_Improvement_Percent",
                "mean",
            ),
            Pct_Pairs_Ablation_Better=(
                "Ablation_Improvement_RMSE",
                lambda s: 100.0 * (s > 0).mean(),
            ),
            N_Pairs=("Ablation_Improvement_RMSE", "size"),
        )
        .sort_values("Mean_Ablation_Improvement_RMSE", ascending=True)
    )
    return summary


def plot_overall_delta(summary: pd.DataFrame, output_dir: Path, scope: str) -> None:
    plot_df = summary.sort_values("Mean_Ablation_Improvement_RMSE", ascending=True).copy()
    y = np.arange(len(plot_df))
    colors = [
        BLUE if value >= 0 else ORANGE
        for value in plot_df["Mean_Ablation_Improvement_RMSE"]
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    bars = ax.barh(
        y,
        plot_df["Mean_Ablation_Improvement_RMSE"],
        color=colors,
        edgecolor=CHARCOAL,
        linewidth=0.7,
    )
    ax.axvline(0, color=CHARCOAL, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Model_Label"])
    fig.suptitle(
        "Mean RMSE improvement from AQUISTIL ablations",
        x=0.25,
        y=0.96,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.25,
        0.91,
        f"{scope}; positive values mean the ablated variant has lower RMSE than full AQUISTIL.",
        ha="left",
        va="top",
        fontsize=9,
        color=CHARCOAL,
    )
    ax.set_xlabel("Full AQUISTIL RMSE - ablated RMSE")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#ECEFF1", linewidth=0.8)
    ax.set_axisbelow(True)

    values = plot_df["Mean_Ablation_Improvement_RMSE"].to_numpy(dtype=float)
    value_range = max(abs(values).max(), 0.01)
    ax.set_xlim(values.min() - value_range * 0.22, values.max() + value_range * 0.22)
    for bar, value, pct in zip(
        bars,
        plot_df["Mean_Ablation_Improvement_RMSE"],
        plot_df["Mean_Ablation_Improvement_Percent"],
    ):
        if abs(value) < value_range * 0.01:
            label_x = value_range * 0.035
            ha = "left"
        else:
            ha = "left" if value >= 0 else "right"
            label_x = value + value_range * (0.035 if value >= 0 else -0.035)
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f} RMSE ({pct:+.2f}%)",
            ha=ha,
            va="center",
            fontsize=9,
            color=CHARCOAL,
        )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.25, right=0.96, top=0.82, bottom=0.16)
    save_figure(fig, output_dir, f"{scope.lower()}_overall_ablation_delta_rmse")


def plot_regime_heatmap(paired: pd.DataFrame, output_dir: Path, scope: str) -> None:
    pivot = (
        paired.groupby(["Regime", "Model_Label"])["Ablation_Improvement_RMSE"]
        .mean()
        .unstack("Model_Label")
    )
    regimes = [regime for regime in REGIME_ORDER if regime in pivot.index]
    labels = [model_label(model) for model in MODEL_ORDER if model_label(model) in pivot.columns]
    pivot = pivot.reindex(index=regimes, columns=labels)

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    values = pivot.to_numpy(dtype=float)
    limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    image = ax.imshow(values, cmap="PuOr_r", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v).replace("_", " ").title() for v in pivot.index])
    ax.set_title(
        "Mean RMSE improvement by missingness regime",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=CHARCOAL,
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Full AQUISTIL RMSE - ablated RMSE")
    save_figure(fig, output_dir, f"{scope.lower()}_regime_ablation_delta_heatmap")


def plot_missingness_heatmap(paired: pd.DataFrame, output_dir: Path, scope: str) -> None:
    pivot = (
        paired.groupby(["Missingness_Percent", "Model_Label"])[
            "Ablation_Improvement_RMSE"
        ]
        .mean()
        .unstack("Model_Label")
        .sort_index()
    )
    labels = [
        model_label(model)
        for model in MODEL_ORDER
        if model_label(model) in pivot.columns
    ]
    pivot = pivot.reindex(columns=labels)

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    values = pivot.to_numpy(dtype=float)
    limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    image = ax.imshow(values, cmap="PuOr_r", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}%" for value in pivot.index])
    ax.set_title(
        "Mean RMSE improvement by missingness percent",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Artificial missingness")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=CHARCOAL,
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Full AQUISTIL RMSE - ablated RMSE")
    save_figure(fig, output_dir, f"{scope.lower()}_missingness_percent_ablation_delta_heatmap")


def plot_missingness_lines(paired: pd.DataFrame, output_dir: Path, scope: str) -> None:
    summary = (
        paired.groupby(["Missingness_Percent", "Model", "Model_Label"], as_index=False)
        .agg(Mean_Ablation_Improvement_RMSE=("Ablation_Improvement_RMSE", "mean"))
    )
    palette = {
        "AQUISTIL_NoHistory": BLUE,
        "AQUISTIL_NoHistoryNoEvent": GOLD,
        "AQUISTIL_NoFFill": ORANGE,
        "AQUISTIL_NoAdaptive": OLIVE,
        "AQUISTIL_ExogenousOnly": PINK,
        "AQUISTIL_NoAQUISTILFeatures": GREY,
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for model in MODEL_ORDER:
        part = summary.loc[summary["Model"].eq(model)].sort_values(
            "Missingness_Percent"
        )
        if part.empty:
            continue
        ax.plot(
            part["Missingness_Percent"],
            part["Mean_Ablation_Improvement_RMSE"],
            marker="o",
            linewidth=2.0,
            color=palette.get(model, GREY),
            label=model_label(model),
        )

    ax.axhline(0, color=CHARCOAL, linewidth=1.0)
    ax.set_title(
        "Mean RMSE improvement across missingness levels",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Artificial missingness (%)")
    ax.set_ylabel("Full AQUISTIL RMSE - ablated RMSE")
    ax.grid(color="#ECEFF1", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save_figure(fig, output_dir, f"{scope.lower()}_missingness_ablation_delta_lines")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.metrics_dir / "ablation_improvement_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "text.color": CHARCOAL,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    paired = load_paired_deltas(args.metrics_dir, args.scope)
    summary = build_summary(paired)
    summary.to_csv(output_dir / f"{args.scope.lower()}_ablation_delta_summary.csv", index=False)
    paired.to_csv(output_dir / f"{args.scope.lower()}_paired_ablation_deltas.csv", index=False)

    plot_overall_delta(summary, output_dir, args.scope)
    plot_regime_heatmap(paired, output_dir, args.scope)
    plot_missingness_heatmap(paired, output_dir, args.scope)
    print(f"Saved ablation improvement plots to {output_dir}")


if __name__ == "__main__":
    main()
