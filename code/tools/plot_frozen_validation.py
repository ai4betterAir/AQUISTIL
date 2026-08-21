#!/usr/bin/env python3
"""Create publication figures from validated frozen-comparison CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIMES = ("random", "short_gap", "medium_gap", "long_gap", "event")
REGIME_LABELS = ("Random", "Short", "Medium", "Long", "Event")
TARGETS = ("PM10", "PM2.5")
COLORS = {"AQUISTIL": "#276FBF", "LightGBM": "#D49A00"}


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overall_performance(pairs: pd.DataFrame, output_dir: Path) -> None:
    data = pairs.loc[pairs["Scope"].eq("Region_Micro")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    positions = np.arange(len(REGIMES), dtype=float)
    for axis, target in zip(axes, TARGETS):
        target_data = data.loc[data["Target"].eq(target)]
        for model, offset in (("AQUISTIL", -0.18), ("LightGBM", 0.18)):
            values = [
                target_data.loc[target_data["Regime"].eq(regime), f"RMSE_{model}"].dropna()
                for regime in REGIMES
            ]
            box = axis.boxplot(
                values,
                positions=positions + offset,
                widths=0.30,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#20242A", "linewidth": 1.2},
            )
            for patch in box["boxes"]:
                patch.set(facecolor=COLORS[model], edgecolor=COLORS[model], alpha=0.72)
        axis.set_title(target)
        axis.set_ylabel("RMSE")
        axis.set_xticks(positions, REGIME_LABELS)
        _style_axis(axis)
    handles = [plt.Line2D([0], [0], color=color, linewidth=8) for color in COLORS.values()]
    fig.suptitle("Frozen held-out performance by missingness regime", y=0.98)
    fig.legend(
        handles,
        COLORS.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    _save(fig, output_dir, "01_overall_performance_by_regime")


def plot_delta_rmse(pairs: pd.DataFrame, output_dir: Path) -> None:
    data = pairs.loc[pairs["Scope"].eq("Region_Micro")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    positions = np.arange(len(REGIMES), dtype=float)
    for axis, target in zip(axes, TARGETS):
        target_data = data.loc[data["Target"].eq(target)]
        values = [
            target_data.loc[target_data["Regime"].eq(regime), "Delta_RMSE"].dropna()
            for regime in REGIMES
        ]
        box = axis.boxplot(
            values,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#20242A", "linewidth": 1.3},
        )
        for patch in box["boxes"]:
            patch.set(facecolor="#7BA7D8", edgecolor="#276FBF", alpha=0.78)
        axis.axhline(0, color="#20242A", linewidth=1.0)
        axis.set_title(target)
        axis.set_ylabel("Delta RMSE (AQUISTIL - LightGBM)")
        axis.set_xticks(positions, REGIME_LABELS)
        _style_axis(axis)
    fig.suptitle("Paired frozen-model RMSE differences", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, output_dir, "02_paired_delta_rmse")


def plot_robustness(site_robustness: pd.DataFrame, output_dir: Path) -> None:
    data = site_robustness.loc[
        site_robustness["Regime"].isin(["short_gap", "medium_gap", "long_gap"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, target in zip(axes, TARGETS):
        part = data.loc[data["Target"].eq(target)].copy()
        normal = part.loc[~part["Difficult_Case"].astype(bool)]
        difficult = part.loc[part["Difficult_Case"].astype(bool)]
        axis.scatter(
            normal["Delta_RMSE_Median"], normal["RMSE_Ratio_Median"],
            s=24, color="#276FBF", alpha=0.62, label="Other site/regime",
        )
        axis.scatter(
            difficult["Delta_RMSE_Median"], difficult["RMSE_Ratio_Median"],
            s=48, color="#C65D21", marker="D", label="Difficult case",
        )
        labels = difficult.copy()
        if labels.empty:
            labels = part.nlargest(3, "RMSE_Ratio_Median")
        label_offsets = ((5, 10), (5, 0), (5, -10))
        for label_index, (_, row) in enumerate(labels.iterrows()):
            axis.annotate(
                f"{row['Site']} ({row['Regime'].replace('_gap', '')})",
                (row["Delta_RMSE_Median"], row["RMSE_Ratio_Median"]),
                xytext=label_offsets[label_index % len(label_offsets)],
                textcoords="offset points",
                fontsize=7,
            )
        axis.axvline(0, color="#20242A", linewidth=1.0)
        axis.axhline(1, color="#777D86", linewidth=0.9, linestyle="--")
        axis.set_title(target)
        axis.set_xlabel("Median delta RMSE")
        axis.set_ylabel("Median AQUISTIL / LightGBM RMSE")
        _style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Held-out site robustness during contiguous gaps", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    _save(fig, output_dir, "03_region_site_robustness")


def plot_event_performance(pairs: pd.DataFrame, output_dir: Path) -> None:
    data = pairs.loc[
        pairs["Scope"].eq("Region_Micro") & pairs["Regime"].eq("event")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for axis, target in zip(axes, TARGETS):
        part = data.loc[data["Target"].eq(target)]
        for model in COLORS:
            grouped = part.groupby("Missingness_Percent")[f"RMSE_{model}"]
            summary = grouped.agg(
                Mean="mean",
                Q1=lambda values: values.quantile(0.25),
                Q3=lambda values: values.quantile(0.75),
            ).reset_index()
            axis.plot(
                summary["Missingness_Percent"], summary["Mean"],
                marker="o", linewidth=1.8, color=COLORS[model], label=model,
            )
            axis.fill_between(
                summary["Missingness_Percent"], summary["Q1"], summary["Q3"],
                color=COLORS[model], alpha=0.14,
            )
        axis.set_title(target)
        axis.set_xlabel("Event observations masked (%)")
        axis.set_ylabel("Region-Micro RMSE")
        _style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Held-out event-dependent reconstruction", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    _save(fig, output_dir, "04_event_performance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("Outputs/Final_Frozen"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_dir = args.results_root / "Statistical_Comparison"
    pairs = pd.read_csv(comparison_dir / "paired_aquistil_lightgbm_metrics.csv")
    site_robustness = pd.read_csv(comparison_dir / "site_robustness.csv")
    output_dir = args.results_root / "Paper_Figures"
    plot_overall_performance(pairs, output_dir)
    plot_delta_rmse(pairs, output_dir)
    plot_robustness(site_robustness, output_dir)
    plot_event_performance(pairs, output_dir)
    print(f"Saved frozen-validation figures to {output_dir}")


if __name__ == "__main__":
    main()
