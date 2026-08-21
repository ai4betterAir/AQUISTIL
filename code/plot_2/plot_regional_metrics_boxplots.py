#!/usr/bin/env python3
"""Create pollutant-wise model boxplots from regional imputation metrics."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


DEFAULT_METRICS = [
    "RMSE",
    "MAE",
    "R2",
    "NRMSE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot each pollutant as model-wise metric boxplots, with overlaid "
            "markers colored by regime and missingness percent."
        )
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path(
            "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/"
            "Outputs/Imputation_Result/Metrics/regional_pooled_metrics.csv"
        ),
        help="Path to regional_pooled_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/"
            "Outputs/Imputation_Result/plots_by_type/model_metric_boxplots"
        ),
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric column(s) to plot.",
    )
    parser.add_argument(
        "--scope",
        default="Region_Macro",
        help="Scope value to filter before plotting. Use 'ALL' for no filter.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional pollutant/target list, e.g. PM10 OZONE.",
    )
    parser.add_argument(
        "--by-region",
        action="store_true",
        help="Also save one model boxplot per pollutant and region.",
    )
    return parser.parse_args()


def clean_metrics(df: pd.DataFrame, metric: str, scope: str) -> pd.DataFrame:
    required = {"Target", "Model", "Regime", "Missingness_Percent", metric}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    plot_df = df.copy()
    if scope.upper() != "ALL":
        if "Scope" not in plot_df.columns:
            raise ValueError("--scope was set but the CSV has no Scope column")
        plot_df = plot_df.loc[plot_df["Scope"].eq(scope)].copy()

    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df["Missingness_Percent"] = pd.to_numeric(
        plot_df["Missingness_Percent"], errors="coerce"
    )
    plot_df = plot_df.dropna(
        subset=["Target", "Model", "Regime", "Missingness_Percent", metric]
    )
    plot_df["Regime_Missingness"] = (
        plot_df["Regime"].astype(str)
        + " "
        + plot_df["Missingness_Percent"].round(0).astype(int).astype(str)
        + "%"
    )
    return plot_df


def save_metric_plots(df, metric, scope, output_dir, targets):
    plot_df = clean_metrics(df, metric, scope=scope)
    if targets:
        plot_df = plot_df.loc[plot_df["Target"].isin(targets)].copy()

    if plot_df.empty:
        raise ValueError(f"No rows available to plot for metric {metric}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    target_order = sorted(plot_df["Target"].unique())
    model_order = sorted(plot_df["Model"].unique())
    hue_order = sorted(
        plot_df["Regime_Missingness"].unique(),
        key=lambda item: (
            item.rsplit(" ", 1)[0],
            int(item.rsplit(" ", 1)[1].rstrip("%")),
        ),
    )
    palette = dict(zip(hue_order, sns.color_palette("tab20", len(hue_order))))

    for target in target_order:
        target_df = plot_df.loc[plot_df["Target"].eq(target)].copy()
        if target_df.empty:
            continue

        height = 6.5
        width = max(10.0, 1.2 * len(model_order) + 6.0)
        fig, ax = plt.subplots(figsize=(width, height))

        box_values = [
            target_df.loc[target_df["Model"].eq(model), metric].dropna().to_numpy()
            for model in model_order
        ]
        ax.boxplot(
            box_values,
            positions=range(len(model_order)),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#d9dee7", "edgecolor": "#58606f", "linewidth": 1.2},
            medianprops={"color": "#111827", "linewidth": 1.3},
            whiskerprops={"color": "#58606f", "linewidth": 1.0},
            capprops={"color": "#58606f", "linewidth": 1.0},
        )

        model_to_x = {model: idx for idx, model in enumerate(model_order)}
        hue_to_offset = {
            hue: offset
            for hue, offset in zip(
                hue_order,
                pd.Series(range(len(hue_order))).map(
                    lambda idx: -0.28 + (0.56 * idx / max(len(hue_order) - 1, 1))
                ),
            )
        }
        for hue in hue_order:
            hue_df = target_df.loc[target_df["Regime_Missingness"].eq(hue)]
            if hue_df.empty:
                continue
            x_values = [
                model_to_x[model] + hue_to_offset[hue]
                for model in hue_df["Model"].astype(str)
            ]
            ax.scatter(
                x_values,
                hue_df[metric],
                s=18,
                color=palette[hue],
                alpha=0.78,
                edgecolors="#20242a",
                linewidths=0.25,
                label=hue,
            )

        ax.set_title(f"{target}: {metric} by model")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=35, ha="right")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=palette[hue],
                markeredgecolor="#20242a",
                markersize=5,
                label=hue,
            )
            for hue in hue_order
        ]
        legend = ax.legend(
            handles=handles,
            title="Regime + missingness",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=True,
            ncol=1,
        )
        legend.get_frame().set_linewidth(0.6)

        fig.tight_layout()
        out_path = output_dir / f"{target}_{metric}_model_boxplot.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def save_metric_region_plots(df, metric, scope, output_dir, targets):
    plot_df = clean_metrics(df, metric, scope=scope)
    if targets:
        plot_df = plot_df.loc[plot_df["Target"].isin(targets)].copy()

    if plot_df.empty:
        raise ValueError(f"No rows available to plot for metric {metric}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    target_order = sorted(plot_df["Target"].unique())
    region_order = sorted(plot_df["Region"].dropna().unique())
    model_order = sorted(plot_df["Model"].unique())
    hue_order = sorted(
        plot_df["Regime_Missingness"].unique(),
        key=lambda item: (
            item.rsplit(" ", 1)[0],
            int(item.rsplit(" ", 1)[1].rstrip("%")),
        ),
    )
    palette = dict(zip(hue_order, sns.color_palette("tab20", len(hue_order))))

    for target in target_order:
        for region in region_order:
            region_df = plot_df.loc[
                plot_df["Target"].eq(target) & plot_df["Region"].eq(region)
            ].copy()
            if region_df.empty:
                continue

            present_models = [m for m in model_order if region_df["Model"].eq(m).any()]
            height = 6.5
            width = max(10.0, 1.2 * len(present_models) + 6.0)
            fig, ax = plt.subplots(figsize=(width, height))

            box_values = [
                region_df.loc[region_df["Model"].eq(model), metric].dropna().to_numpy()
                for model in present_models
            ]
            ax.boxplot(
                box_values,
                positions=range(len(present_models)),
                widths=0.58,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": "#d9dee7",
                    "edgecolor": "#58606f",
                    "linewidth": 1.2,
                },
                medianprops={"color": "#111827", "linewidth": 1.3},
                whiskerprops={"color": "#58606f", "linewidth": 1.0},
                capprops={"color": "#58606f", "linewidth": 1.0},
            )

            model_to_x = {model: idx for idx, model in enumerate(present_models)}
            hue_to_offset = {
                hue: offset
                for hue, offset in zip(
                    hue_order,
                    pd.Series(range(len(hue_order))).map(
                        lambda idx: -0.28
                        + (0.56 * idx / max(len(hue_order) - 1, 1))
                    ),
                )
            }
            for hue in hue_order:
                hue_df = region_df.loc[region_df["Regime_Missingness"].eq(hue)]
                if hue_df.empty:
                    continue
                x_values = [
                    model_to_x[model] + hue_to_offset[hue]
                    for model in hue_df["Model"].astype(str)
                ]
                ax.scatter(
                    x_values,
                    hue_df[metric],
                    s=18,
                    color=palette[hue],
                    alpha=0.78,
                    edgecolors="#20242a",
                    linewidths=0.25,
                    label=hue,
                )

            ax.set_title(f"{target} - {region}: {metric} by model")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(present_models)))
            ax.set_xticklabels(present_models, rotation=35, ha="right")
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
            ax.set_axisbelow(True)

            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=palette[hue],
                    markeredgecolor="#20242a",
                    markersize=5,
                    label=hue,
                )
                for hue in hue_order
            ]
            legend = ax.legend(
                handles=handles,
                title="Regime + missingness",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
                frameon=True,
                ncol=1,
            )
            legend.get_frame().set_linewidth(0.6)

            fig.tight_layout()
            safe_region = region.replace(" ", "_").replace("/", "_")
            out_path = output_dir / target / f"{safe_region}_{metric}_model_boxplot.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(out_path)

    return saved_paths


if __name__ == "__main__":
    args = parse_args()
    sns.set_theme(style="whitegrid", context="notebook")

    metrics_df = pd.read_csv(args.metrics_csv)
    all_saved = []
    for metric_name in args.metrics:
        all_saved.extend(
            save_metric_plots(
                metrics_df,
                metric=metric_name,
                scope=args.scope,
                output_dir=args.output_dir / metric_name,
                targets=args.targets,
            )
        )
        if args.by_region:
            all_saved.extend(
                save_metric_region_plots(
                    metrics_df,
                    metric=metric_name,
                    scope=args.scope,
                    output_dir=args.output_dir / "by_region" / metric_name,
                    targets=args.targets,
                )
            )

    print(f"Saved {len(all_saved)} plot(s):")
    for path in all_saved:
        print(path)
