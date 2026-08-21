#!/usr/bin/env python3
"""Compare model RMSE and R distributions at all-region, region, and site scopes.

For each spatial scope and model, each metric is averaged within every masking
regime x missingness combination.  Those condition-level averages form the
boxplot distribution, so models are compared fairly on the x axis without
large sites or conditions contributing disproportionate numbers of rows.
"""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aquistil_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


DEFAULT_RESULTS_ROOT = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Imputation_Result"
)
DEFAULT_METRICS_FILE = DEFAULT_RESULTS_ROOT / "Metrics" / "regional_pooled_metrics.csv"
MODEL_COLOURS = [
    "#3F6FAE",
    "#D9A441",
    "#8D78B7",
    "#7F985A",
    "#CC6D96",
    "#5FA9A4",
    "#5B84C4",
    "#D6A64A",
    "#8C8F98",
    "#789457",
    "#C85F88",
]
REGIME_COLOURS = {
    "event": "#D95F02",
    "long_gap": "#7570B3",
    "medium_gap": "#1B9E77",
    "random": "#E7298A",
    "short_gap": "#66A61E",
}
MISSINGNESS_COLOURS = {
    0.05: "#2166AC",
    0.10: "#67A9CF",
    0.20: "#1B9E77",
    0.30: "#F1A340",
    0.50: "#D95F02",
    0.60: "#B2182B",
}
CHARCOAL = "#2F3437"
GRID = "#D9DEE2"
PANEL_BG = "#F7F9FB"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Default: <results-root>/Metrics/regional_pooled_metrics.csv",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Default: every available model.")
    parser.add_argument("--targets", nargs="+", default=None, help="Default: every available pollutant.")
    parser.add_argument(
        "--metrics", nargs="+", choices=["RMSE", "R"], default=["RMSE", "R"]
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=["all_regions", "region", "station"],
        default=["all_regions", "region", "station"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def slug(value):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_")


def pollutant_label(target):
    return {"PM25": "PM2.5", "PM2.5": "PM2.5", "PM10": "PM10", "NO2": "NO₂", "OZONE": "O₃"}.get(target, target)


def is_pm10(target):
    return str(target).replace(".", "").upper() == "PM10"


def is_pm25(target):
    return str(target).replace(".", "").upper() == "PM25"


def load_metrics(path, models=None, targets=None):
    columns = ["Region", "Site", "Target", "Model", "Regime", "Missingness_Level", "Seed", "Scope", "RMSE", "R"]
    data = pd.read_csv(path, usecols=columns)
    for column in ["Missingness_Level", "Seed", "RMSE", "R"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Region", "Site", "Target", "Model", "Regime", "Missingness_Level"])
    if models:
        data = data.loc[data["Model"].isin(models)]
    if targets:
        wanted = {"PM2.5" if target == "PM25" else target for target in targets}
        data = data.loc[data["Target"].isin(wanted)]
    return data


def condition_means(data, metric):
    """Return one equally weighted metric observation per model/condition."""
    return data.groupby(
        ["Model", "Regime", "Missingness_Level"], as_index=False, observed=True
    )[metric].mean()


def missingness_label(value):
    return "{:g}% missingness".format(float(value) * 100.0)


def missingness_slug(value):
    return "{:g}_percent".format(float(value) * 100.0).replace(".", "_")


def regime_label(value):
    return str(value).replace("_", " ").title()


def metric_title(metric):
    return "RMSE" if metric == "RMSE" else "R"


def metric_axis_label(metric, target):
    if metric == "RMSE":
        return "RMSE ({})".format(pollutant_label(target))
    return "R"


def metric_data(data, metric):
    data = data.loc[np.isfinite(data[metric])].copy()
    if metric == "RMSE":
        data = data.loc[data[metric].ge(0)]
    return data


def set_metric_ylim(ax, values, metric):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return
    if metric == "RMSE":
        upper = np.nanpercentile(finite, 98)
        if upper > 0:
            ax.set_ylim(bottom=0, top=upper * 1.15)
        return
    lower = np.nanpercentile(finite, 2)
    upper = np.nanpercentile(finite, 98)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return
    if lower == upper:
        pad = 0.05
    else:
        pad = max((upper - lower) * 0.12, 0.03)
    ax.set_ylim(max(-1.0, lower - pad), min(1.0, upper + pad))


def hdr_threshold(grid, density, mass):
    """Return the density threshold enclosing approximately ``mass`` probability."""
    if len(grid) < 2 or not np.any(np.isfinite(density)):
        return None
    dx = float(np.mean(np.diff(grid)))
    total = float(np.sum(density) * dx)
    if total <= 0:
        return None
    density = density / total
    order = np.argsort(density)[::-1]
    cumulative = np.cumsum(density[order] * dx)
    cutoff = np.searchsorted(cumulative, mass, side="left")
    cutoff = min(cutoff, len(order) - 1)
    return float(density[order[cutoff]])


def hdr_intervals(grid, density, threshold):
    if threshold is None:
        return []
    mask = density >= threshold
    intervals = []
    start = None
    for index, keep in enumerate(mask):
        if keep and start is None:
            start = index
        if start is not None and (not keep or index == len(mask) - 1):
            end = index if keep and index == len(mask) - 1 else index - 1
            intervals.append((grid[start], grid[end]))
            start = None
    return intervals


def outside_intervals(values, intervals):
    if not intervals:
        return np.asarray([], dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.zeros(values.shape, dtype=bool)
    for low, high in intervals:
        keep |= (values >= low) & (values <= high)
    return values[~keep]


def render_all_regions_hdr_summary(data, target, metric, output, models, args):
    """Render one pooled HDR density boxplot by model across all regimes/missingness."""
    if output.exists() and not args.force:
        return 0
    data = metric_data(data, metric)
    if data.empty:
        return 0

    present_models = [model for model in models if data["Model"].eq(model).any()]
    if not present_models:
        return 0

    values_by_model = [
        data.loc[data["Model"].eq(model), metric].dropna().to_numpy()
        for model in present_models
    ]
    values_by_model = [values[np.isfinite(values)] for values in values_by_model]
    all_values = np.concatenate([values for values in values_by_model if len(values)])
    if not len(all_values):
        return 0

    lower = np.nanpercentile(all_values, 2)
    upper = np.nanpercentile(all_values, 98)
    pad = max((upper - lower) * 0.12, 0.03)
    if metric == "RMSE":
        y_min = max(0.0, lower - pad)
        y_max = upper + pad
    else:
        y_min = max(-1.0, lower - pad)
        y_max = min(1.0, upper + pad)
    if y_min >= y_max:
        y_min, y_max = float(np.nanmin(all_values)), float(np.nanmax(all_values))
        y_min, y_max = y_min - 0.05, y_max + 0.05

    fig_width = max(11.69, 0.58 * len(present_models) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_width, 8.27))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    positions = np.arange(len(present_models), dtype=float) + 1.0
    max_half_width = 0.34
    for index, (position, model, values) in enumerate(zip(positions, present_models, values_by_model)):
        if len(values) < 5 or len(np.unique(values)) < 3:
            result = ax.boxplot(
                [values],
                positions=[position],
                widths=0.42,
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": CHARCOAL,
                    "markeredgewidth": 1.1,
                    "markersize": 5.0,
                },
                medianprops={"color": CHARCOAL, "linewidth": 1.8},
                whiskerprops={"color": CHARCOAL, "linewidth": 1.15},
                capprops={"color": CHARCOAL, "linewidth": 1.15},
                flierprops={"marker": "o", "markeredgecolor": "none", "alpha": 0.28, "markersize": 3.1},
            )
            result["boxes"][0].set(
                facecolor=MODEL_COLOURS[index % len(MODEL_COLOURS)],
                edgecolor=CHARCOAL,
                alpha=0.42,
                linewidth=1.15,
            )
            continue

        grid = np.linspace(y_min, y_max, 512)
        try:
            kde = gaussian_kde(values)
            density = kde(grid)
        except Exception:
            continue
        if not np.any(np.isfinite(density)) or np.nanmax(density) <= 0:
            continue
        density_norm = density / np.trapz(density, grid)
        threshold_99 = hdr_threshold(grid, density_norm, 0.99)
        threshold_50 = hdr_threshold(grid, density_norm, 0.50)
        density_scaled = density / np.nanmax(density) * max_half_width
        colour = MODEL_COLOURS[index % len(MODEL_COLOURS)]

        ax.fill_betweenx(
            grid,
            position - density_scaled,
            position + density_scaled,
            where=density_norm >= threshold_99,
            facecolor=colour,
            alpha=0.22,
            edgecolor="none",
            interpolate=True,
        )
        ax.fill_betweenx(
            grid,
            position - density_scaled,
            position + density_scaled,
            where=density_norm >= threshold_50,
            facecolor=colour,
            alpha=0.62,
            edgecolor="none",
            interpolate=True,
        )
        density_left = np.where(density_norm >= threshold_99, position - density_scaled, np.nan)
        density_right = np.where(density_norm >= threshold_99, position + density_scaled, np.nan)
        ax.plot(density_left, grid, color=CHARCOAL, linewidth=0.7, alpha=0.55)
        ax.plot(density_right, grid, color=CHARCOAL, linewidth=0.7, alpha=0.55)

        intervals_99 = hdr_intervals(grid, density_norm, threshold_99)
        outliers = outside_intervals(values, intervals_99)
        if len(outliers):
            jitter = np.linspace(-0.045, 0.045, len(outliers)) if len(outliers) > 1 else np.array([0.0])
            jitter = np.resize(jitter, len(outliers))
            ax.scatter(
                np.full(len(outliers), position) + jitter,
                outliers,
                s=11,
                color=colour,
                alpha=0.25,
                edgecolors="none",
                zorder=3,
            )

        median = float(np.nanmedian(values))
        mean = float(np.nanmean(values))
        ax.plot([position - 0.22, position + 0.22], [median, median], color=CHARCOAL, linewidth=1.8, zorder=4)
        ax.scatter(
            [position],
            [mean],
            marker="D",
            s=36,
            facecolor="white",
            edgecolor=CHARCOAL,
            linewidth=1.1,
            zorder=5,
        )

    ax.set_xlim(0.45, len(present_models) + 0.55)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(positions)
    ax.set_xticklabels(present_models, rotation=45, ha="right", fontsize=11.2)
    ax.set_ylabel(metric_axis_label(metric, target), fontsize=14)
    ax.set_title(
        "{} HDR Density by Model - All Regimes and Missingness - {}".format(
            metric_title(metric),
            pollutant_label(target),
        ),
        fontsize=18,
        fontweight="bold",
        pad=16,
    )
    ax.tick_params(axis="y", labelsize=11, colors=CHARCOAL)
    ax.tick_params(axis="x", colors=CHARCOAL, pad=1)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHARCOAL)
    ax.spines["bottom"].set_color(CHARCOAL)
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    ax.text(
        0.995,
        0.99,
        "Dark fill = 50% HDR; light fill = 99% HDR; line = median; diamond = mean",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        color=CHARCOAL,
    )
    fig.tight_layout(rect=(0.015, 0.02, 0.995, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output), flush=True)
    return 1


def render_all_regions_hdr_regime_overlay(data, target, metric, output, models, args):
    """Render pooled model HDR violins with regime-specific density overlays."""
    if output.exists() and not args.force:
        return 0
    data = metric_data(data, metric)
    if data.empty:
        return 0

    present_models = [model for model in models if data["Model"].eq(model).any()]
    regimes = sorted(data["Regime"].dropna().unique())
    if not present_models or not regimes:
        return 0

    all_values = data[metric].dropna().to_numpy()
    all_values = all_values[np.isfinite(all_values)]
    if not len(all_values):
        return 0

    lower = np.nanpercentile(all_values, 2)
    upper = np.nanpercentile(all_values, 98)
    pad = max((upper - lower) * 0.12, 0.03)
    if metric == "RMSE":
        y_min = max(0.0, lower - pad)
        y_max = upper + pad
    else:
        y_min = max(-1.0, lower - pad)
        y_max = min(1.0, upper + pad)

    fig_width = max(11.69, 0.64 * len(present_models) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_width, 8.27))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PANEL_BG)

    grid = np.linspace(y_min, y_max, 512)
    positions = np.arange(len(present_models), dtype=float) + 1.0
    max_half_width = 0.32
    regime_offsets = np.linspace(-0.20, 0.20, len(regimes)) if len(regimes) > 1 else np.array([0.0])

    for position, model in zip(positions, present_models):
        model_values = data.loc[data["Model"].eq(model), metric].dropna().to_numpy()
        model_values = model_values[np.isfinite(model_values)]
        if len(model_values) >= 5 and len(np.unique(model_values)) >= 3:
            try:
                model_density = gaussian_kde(model_values)(grid)
            except Exception:
                model_density = None
            if model_density is not None and np.nanmax(model_density) > 0:
                model_density_norm = model_density / np.trapz(model_density, grid)
                model_threshold_99 = hdr_threshold(grid, model_density_norm, 0.99)
                model_scaled = model_density / np.nanmax(model_density) * max_half_width
                ax.fill_betweenx(
                    grid,
                    position - model_scaled,
                    position + model_scaled,
                    where=model_density_norm >= model_threshold_99,
                    facecolor="#F7D9D9",
                    alpha=0.42,
                    edgecolor="none",
                    interpolate=True,
                    zorder=1,
                )

        for offset, regime in zip(regime_offsets, regimes):
            values = data.loc[
                data["Model"].eq(model) & data["Regime"].eq(regime),
                metric,
            ].dropna().to_numpy()
            values = values[np.isfinite(values)]
            if len(values) < 2:
                continue
            center = position + offset
            colour = REGIME_COLOURS.get(str(regime), MODEL_COLOURS[0])

            median = float(np.nanmedian(values))
            intervals_50 = []
            try:
                if len(values) >= 5 and len(np.unique(values)) >= 3:
                    density = gaussian_kde(values)(grid)
                    if np.any(np.isfinite(density)) and np.nanmax(density) > 0:
                        density_norm = density / np.trapz(density, grid)
                        intervals_50 = hdr_intervals(
                            grid,
                            density_norm,
                            hdr_threshold(grid, density_norm, 0.50),
                        )
            except Exception:
                intervals_50 = []
            if not intervals_50:
                intervals_50 = [(float(np.nanpercentile(values, 25)), float(np.nanpercentile(values, 75)))]

            for low, high in intervals_50:
                ax.plot(
                    [center, center],
                    [low, high],
                    color=colour,
                    linewidth=3.2,
                    alpha=0.88,
                    solid_capstyle="round",
                    zorder=3,
                )
            ax.plot(
                [center - 0.045, center + 0.045],
                [median, median],
                color=CHARCOAL,
                linewidth=1.15,
                zorder=4,
            )

        model_mean = float(np.nanmean(model_values)) if len(model_values) else np.nan
        if np.isfinite(model_mean):
            ax.scatter(
                [position],
                [model_mean],
                marker="D",
                s=34,
                facecolor="white",
                edgecolor=CHARCOAL,
                linewidth=1.1,
                zorder=6,
            )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=REGIME_COLOURS.get(str(regime), MODEL_COLOURS[0]),
            linewidth=4,
            alpha=0.88,
            label=regime_label(regime),
        )
        for regime in regimes
    ]
    if not is_pm10(target):
        legend_kwargs = {
            "handles": handles,
            "title": "Regime",
            "frameon": not is_pm25(target),
            "fontsize": 20,
            "title_fontsize": 22,
        }
        if is_pm25(target):
            legend_kwargs.update(
                loc="center right",
                bbox_to_anchor=(0.99, 0.19),
                ncol=2,
                columnspacing=0.9,
                handlelength=1.6,
                borderaxespad=0.35,
            )
        else:
            legend_kwargs.update(loc="upper right")
        ax.legend(
            **legend_kwargs
        )
    ax.set_xlim(0.45, len(present_models) + 0.55)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(positions)
    ax.set_xticklabels(present_models, rotation=45, ha="right", fontsize=22.4)
    if is_pm25(target):
        ax.set_ylabel("")
    else:
        ax.set_ylabel(metric_axis_label(metric, target), fontsize=28)
    ax.set_title(
        "{} by Model and Regime - All Missingness - {}".format(
            metric_title(metric),
            pollutant_label(target),
        ),
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    if metric == "RMSE":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.tick_params(axis="y", labelsize=22, colors=CHARCOAL)
    ax.tick_params(axis="x", colors=CHARCOAL, pad=1)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHARCOAL)
    ax.spines["bottom"].set_color(CHARCOAL)
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    fig.text(
        0.99,
        0.018,
        "Grey violin = pooled 99% HDR per model; colored bars = regime 50% HDR; tick = regime median; diamond = pooled mean",
        ha="right",
        va="bottom",
        fontsize=12,
        color=CHARCOAL,
    )
    fig.tight_layout(rect=(0.015, 0.06, 0.995, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output), flush=True)
    return 1


def render_all_regions_hdr_missingness_overlay(data, target, metric, output, models, args):
    """Render pooled model HDR violins with missingness-specific HDR summaries."""
    if output.exists() and not args.force:
        return 0
    data = metric_data(data, metric)
    if data.empty:
        return 0

    present_models = [model for model in models if data["Model"].eq(model).any()]
    missingness_levels = sorted(data["Missingness_Level"].dropna().unique())
    if not present_models or not missingness_levels:
        return 0

    all_values = data[metric].dropna().to_numpy()
    all_values = all_values[np.isfinite(all_values)]
    if not len(all_values):
        return 0

    lower = np.nanpercentile(all_values, 2)
    upper = np.nanpercentile(all_values, 98)
    pad = max((upper - lower) * 0.12, 0.03)
    if metric == "RMSE":
        y_min = max(0.0, lower - pad)
        y_max = upper + pad
    else:
        y_min = max(-1.0, lower - pad)
        y_max = min(1.0, upper + pad)

    fig_width = max(11.69, 0.64 * len(present_models) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_width, 8.27))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    grid = np.linspace(y_min, y_max, 512)
    positions = np.arange(len(present_models), dtype=float) + 1.0
    max_half_width = 0.32
    offsets = np.linspace(-0.22, 0.22, len(missingness_levels)) if len(missingness_levels) > 1 else np.array([0.0])

    for position, model in zip(positions, present_models):
        model_values = data.loc[data["Model"].eq(model), metric].dropna().to_numpy()
        model_values = model_values[np.isfinite(model_values)]
        if len(model_values) >= 5 and len(np.unique(model_values)) >= 3:
            try:
                model_density = gaussian_kde(model_values)(grid)
            except Exception:
                model_density = None
            if model_density is not None and np.nanmax(model_density) > 0:
                model_density_norm = model_density / np.trapz(model_density, grid)
                model_threshold_99 = hdr_threshold(grid, model_density_norm, 0.99)
                model_scaled = model_density / np.nanmax(model_density) * max_half_width
                ax.fill_betweenx(
                    grid,
                    position - model_scaled,
                    position + model_scaled,
                    where=model_density_norm >= model_threshold_99,
                    facecolor="#DDEFD8",
                    alpha=0.48,
                    edgecolor="none",
                    interpolate=True,
                    zorder=1,
                )

        for offset, missingness in zip(offsets, missingness_levels):
            values = data.loc[
                data["Model"].eq(model) & data["Missingness_Level"].eq(missingness),
                metric,
            ].dropna().to_numpy()
            values = values[np.isfinite(values)]
            if len(values) < 2:
                continue
            center = position + offset
            colour = MISSINGNESS_COLOURS.get(round(float(missingness), 2), MODEL_COLOURS[0])

            median = float(np.nanmedian(values))
            intervals_50 = []
            try:
                if len(values) >= 5 and len(np.unique(values)) >= 3:
                    density = gaussian_kde(values)(grid)
                    if np.any(np.isfinite(density)) and np.nanmax(density) > 0:
                        density_norm = density / np.trapz(density, grid)
                        intervals_50 = hdr_intervals(
                            grid,
                            density_norm,
                            hdr_threshold(grid, density_norm, 0.50),
                        )
            except Exception:
                intervals_50 = []
            if not intervals_50:
                intervals_50 = [(float(np.nanpercentile(values, 25)), float(np.nanpercentile(values, 75)))]

            for low, high in intervals_50:
                ax.plot(
                    [center, center],
                    [low, high],
                    color=colour,
                    linewidth=3.0,
                    alpha=0.88,
                    solid_capstyle="round",
                    zorder=3,
                )
            ax.plot(
                [center - 0.04, center + 0.04],
                [median, median],
                color=CHARCOAL,
                linewidth=1.1,
                zorder=4,
            )

        model_mean = float(np.nanmean(model_values)) if len(model_values) else np.nan
        if np.isfinite(model_mean):
            ax.scatter(
                [position],
                [model_mean],
                marker="D",
                s=34,
                facecolor="white",
                edgecolor=CHARCOAL,
                linewidth=1.1,
                zorder=6,
            )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=MISSINGNESS_COLOURS.get(round(float(missingness), 2), MODEL_COLOURS[0]),
            linewidth=4,
            alpha=0.88,
            label="{:g}%".format(float(missingness) * 100.0),
        )
        for missingness in missingness_levels
    ]
    if not is_pm10(target):
        legend_kwargs = {
            "handles": handles,
            "title": "Missingness",
            "frameon": not is_pm25(target),
            "fontsize": 20,
            "title_fontsize": 22,
        }
        if is_pm25(target):
            legend_kwargs.update(
                loc="center right",
                bbox_to_anchor=(0.99, 0.19),
                ncol=2,
                columnspacing=0.9,
                handlelength=1.6,
                borderaxespad=0.35,
            )
        else:
            legend_kwargs.update(loc="upper right")
        ax.legend(
            **legend_kwargs
        )
    ax.set_xlim(0.45, len(present_models) + 0.55)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(positions)
    ax.set_xticklabels(present_models, rotation=45, ha="right", fontsize=22.4)
    ax.set_ylabel(metric_axis_label(metric, target), fontsize=28)
    ax.set_title(
        "{} by Model and Missingness - All Regimes - {}".format(
            metric_title(metric),
            pollutant_label(target),
        ),
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    if metric == "RMSE":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.tick_params(axis="y", labelsize=22, colors=CHARCOAL)
    ax.tick_params(axis="x", colors=CHARCOAL, pad=1)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHARCOAL)
    ax.spines["bottom"].set_color(CHARCOAL)
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    fig.text(
        0.99,
        0.018,
        "Grey violin = pooled 99% HDR per model; colored bars = missingness 50% HDR; tick = missingness median; diamond = pooled mean",
        ha="right",
        va="bottom",
        fontsize=12,
        color=CHARCOAL,
    )
    fig.tight_layout(rect=(0.015, 0.06, 0.995, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output), flush=True)
    return 1


def render_hdr_combined_panel(missingness_png, regime_png, output, metric, target, args):
    """Combine missingness-overlay and regime-overlay HDR summaries into one 1x2 panel."""
    if output.exists() and not args.force:
        return 0
    if not missingness_png.exists() or not regime_png.exists():
        return 0

    left = plt.imread(missingness_png)
    right = plt.imread(regime_png)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.3))
    fig.patch.set_facecolor("white")
    for ax, image in zip(axes, [left, right]):
        ax.imshow(image)
        ax.axis("off")

    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99), w_pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output), flush=True)
    return 1


def render_all_regions_regime_missingness_subplots(data, target, regime, metric, output, models, args):
    """Render one all-regions metric panel for a regime, with missingness subplots."""
    if output.exists() and not args.force:
        return 0
    data = metric_data(data.loc[data["Regime"].eq(regime)], metric)
    if data.empty:
        return 0

    # Equally weight each region/site within a missingness level for this regime.
    summary = data.groupby(
        ["Model", "Region", "Site", "Missingness_Level"],
        as_index=False,
        observed=True,
    )[metric].mean()
    missingness_levels = sorted(summary["Missingness_Level"].dropna().unique())
    present_models = [model for model in models if summary["Model"].eq(model).any()]
    if not missingness_levels or not present_models:
        return 0

    ncols = 2
    nrows = int(np.ceil(len(missingness_levels) / ncols))
    width = 8.27
    height = 9.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor("white")
    axes_flat = axes.ravel()
    positions = np.arange(len(present_models), dtype=float) * 0.72 + 1.0

    y_values = summary[metric].to_numpy()

    for ax_index, (ax, missingness) in enumerate(zip(axes_flat, missingness_levels)):
        part = summary.loc[summary["Missingness_Level"].eq(missingness)]
        values = [
            part.loc[part["Model"].eq(model), metric].dropna().to_numpy()
            for model in present_models
        ]
        result = ax.boxplot(
            values,
            positions=positions,
            widths=0.30,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": CHARCOAL,
                "markeredgewidth": 1.1,
                "markersize": 5.0,
            },
            medianprops={"color": CHARCOAL, "linewidth": 1.8},
            whiskerprops={"color": CHARCOAL, "linewidth": 1.15},
            capprops={"color": CHARCOAL, "linewidth": 1.15},
            flierprops={
                "marker": "o",
                "markeredgecolor": "none",
                "alpha": 0.28,
                "markersize": 3.1,
            },
        )
        for index, box in enumerate(result["boxes"]):
            colour = MODEL_COLOURS[index % len(MODEL_COLOURS)]
            box.set(facecolor=colour, edgecolor=CHARCOAL, alpha=0.84, linewidth=1.15)
        for index, flier in enumerate(result["fliers"]):
            flier.set_markerfacecolor(MODEL_COLOURS[index % len(MODEL_COLOURS)])

        ax.set_title(missingness_label(missingness), fontsize=12.0, fontweight="bold", pad=2)
        ax.set_facecolor(PANEL_BG)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xticks(positions)
        if ax_index // ncols == nrows - 1:
            ax.set_xticklabels(present_models, rotation=45, ha="right", fontsize=11.2)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        ax.set_xlim(positions[0] - 0.38, positions[-1] + 0.38)
        ax.tick_params(axis="x", colors=CHARCOAL, pad=1)
        ax.tick_params(axis="y", labelsize=10.2, colors=CHARCOAL)
        if ax_index % ncols == 0:
            ax.tick_params(axis="y", labelleft=True, left=True)
            ax.set_ylabel(metric_axis_label(metric, target), fontsize=11.5)
        else:
            ax.tick_params(axis="y", labelleft=False, left=False, length=0)
            ax.set_ylabel("")
        set_metric_ylim(ax, y_values, metric)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(CHARCOAL)
        ax.spines["bottom"].set_color(CHARCOAL)
        ax.spines["left"].set_linewidth(0.95)
        ax.spines["bottom"].set_linewidth(0.95)

    for ax in axes_flat[len(missingness_levels):]:
        ax.axis("off")

    fig.suptitle(
        "{} by Model and Missingness - {} - {}".format(
            metric_title(metric),
            str(regime).replace("_", " ").title(),
            pollutant_label(target)
        ),
        fontsize=18,
        fontweight="bold",
        y=0.984,
    )
    fig.subplots_adjust(left=0.06, right=0.992, bottom=0.13, top=0.915, wspace=0.045, hspace=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output), flush=True)
    return 1


def render_all_regions_missingness_regime_subplots(data, target, missingness, metric, output, models, args):
    """Render one all-regions metric panel for a missingness level, with regime subplots."""
    if output.exists() and not args.force:
        return 0
    data = metric_data(data.loc[data["Missingness_Level"].eq(missingness)], metric)
    if data.empty:
        return 0

    # Equally weight each region/site within a regime for this missingness level.
    summary = data.groupby(
        ["Model", "Region", "Site", "Regime"],
        as_index=False,
        observed=True,
    )[metric].mean()
    regimes = sorted(summary["Regime"].dropna().unique())
    present_models = [model for model in models if summary["Model"].eq(model).any()]
    if not regimes or not present_models:
        return 0

    ncols = 2
    nrows = int(np.ceil(len(regimes) / ncols))
    width = 8.27
    height = 9.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor("white")
    axes_flat = axes.ravel()
    positions = np.arange(len(present_models), dtype=float) * 0.72 + 1.0

    for ax_index, (ax, regime) in enumerate(zip(axes_flat, regimes)):
        part = summary.loc[summary["Regime"].eq(regime)]
        values = [
            part.loc[part["Model"].eq(model), metric].dropna().to_numpy()
            for model in present_models
        ]
        finite_part_y = part[metric].to_numpy()
        result = ax.boxplot(
            values,
            positions=positions,
            widths=0.30,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": CHARCOAL,
                "markeredgewidth": 1.1,
                "markersize": 5.0,
            },
            medianprops={"color": CHARCOAL, "linewidth": 1.8},
            whiskerprops={"color": CHARCOAL, "linewidth": 1.15},
            capprops={"color": CHARCOAL, "linewidth": 1.15},
            flierprops={
                "marker": "o",
                "markeredgecolor": "none",
                "alpha": 0.28,
                "markersize": 3.1,
            },
        )
        for index, box in enumerate(result["boxes"]):
            colour = MODEL_COLOURS[index % len(MODEL_COLOURS)]
            box.set(facecolor=colour, edgecolor=CHARCOAL, alpha=0.84, linewidth=1.15)
        for index, flier in enumerate(result["fliers"]):
            flier.set_markerfacecolor(MODEL_COLOURS[index % len(MODEL_COLOURS)])

        ax.set_title(regime_label(regime), fontsize=12.0, fontweight="bold", pad=2)
        ax.set_facecolor(PANEL_BG)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xticks(positions)
        show_model_labels = ax_index // ncols == nrows - 1 or str(regime).casefold() == "random"
        if show_model_labels:
            ax.set_xticklabels(present_models, rotation=45, ha="right", fontsize=11.2)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        ax.set_xlim(positions[0] - 0.38, positions[-1] + 0.38)
        ax.tick_params(axis="x", colors=CHARCOAL, pad=1)
        ax.tick_params(axis="y", labelsize=10.2, colors=CHARCOAL)
        ax.tick_params(axis="y", labelleft=True, left=True)
        if ax_index % ncols == 0:
            ax.set_ylabel(metric_axis_label(metric, target), fontsize=11.5)
        else:
            ax.set_ylabel("")
        set_metric_ylim(ax, finite_part_y, metric)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(CHARCOAL)
        ax.spines["bottom"].set_color(CHARCOAL)
        ax.spines["left"].set_linewidth(0.95)
        ax.spines["bottom"].set_linewidth(0.95)

    for ax in axes_flat[len(regimes):]:
        ax.axis("off")

    fig.suptitle(
        "{} by Model and Regime - {} - {}".format(
            metric_title(metric),
            missingness_label(missingness).replace(" missingness", " Missingness"),
            pollutant_label(target)
        ),
        fontsize=18,
        fontweight="bold",
        y=0.984,
    )
    fig.subplots_adjust(left=0.06, right=0.992, bottom=0.13, top=0.915, wspace=0.11, hspace=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output), flush=True)
    return 1


def render_scope(data, target, metric, scope_label, output, models, args):
    if output.exists() and not args.force:
        return 0
    data = data.loc[np.isfinite(data[metric])].copy()
    if metric == "RMSE":
        data = data.loc[data[metric].ge(0)]
    summary = condition_means(data, metric)
    present_models = [model for model in models if summary["Model"].eq(model).any()]
    if not present_models:
        return 0
    values = [summary.loc[summary["Model"].eq(model), metric].to_numpy() for model in present_models]
    means = [float(np.mean(group)) for group in values]

    width = max(11.69, 1.55 * len(present_models) + 3.0)
    fig, ax = plt.subplots(figsize=(width, 8.27))
    result = ax.boxplot(
        values,
        labels=present_models,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": CHARCOAL, "markersize": 6},
        medianprops={"color": CHARCOAL, "linewidth": 1.8},
        whiskerprops={"color": CHARCOAL, "linewidth": 1.1},
        capprops={"color": CHARCOAL, "linewidth": 1.1},
        flierprops={"marker": "o", "markeredgecolor": "none", "alpha": 0.5, "markersize": 4},
    )
    for index, box in enumerate(result["boxes"]):
        colour = MODEL_COLOURS[index % len(MODEL_COLOURS)]
        box.set(facecolor=colour, edgecolor=CHARCOAL, alpha=0.76, linewidth=1.1)
    for index, flier in enumerate(result["fliers"]):
        flier.set_markerfacecolor(MODEL_COLOURS[index % len(MODEL_COLOURS)])

    y_min, y_max = ax.get_ylim()
    offset = max((y_max - y_min) * 0.025, 1e-6)
    for position, mean in enumerate(means, start=1):
        ax.text(position, mean + offset, "mean={:.3g}".format(mean), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_facecolor("#F6F8FA")
    ax.set_xlabel("Model", fontsize=16)
    y_label = "Mean RMSE ({})".format(pollutant_label(target)) if metric == "RMSE" else "Mean correlation (R)"
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title("Overall model {} — {} — {}".format(metric, pollutant_label(target), scope_label), fontsize=20, fontweight="bold", pad=18)
    ax.tick_params(axis="both", labelsize=12, colors=CHARCOAL)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHARCOAL)
    ax.spines["bottom"].set_color(CHARCOAL)
    fig.text(
        0.99,
        0.01,
        "Each box contains mean {} for every regime × missingness combination; diamond and label = overall mean".format(metric),
        ha="right",
        fontsize=9,
        color=CHARCOAL,
    )
    fig.tight_layout(rect=(0.02, 0.035, 0.99, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output), flush=True)
    return 1


def main():
    args = parse_args()
    metrics_file = args.metrics_file or (args.results_root / "Metrics" / "regional_pooled_metrics.csv")
    data = load_metrics(metrics_file, args.models, args.targets)
    if data.empty:
        raise ValueError("No metric rows match the requested models and targets.")
    models = args.models or sorted(data["Model"].unique())
    targets = sorted(data["Target"].unique())
    plot_root = args.results_root / "plots_by_type" / "boxplot"
    hdr_root = args.results_root / "plots_by_type" / "hdr_boxplot"
    total = 0

    # Site rows are the common base unit for all scopes. Aggregation below first
    # averages sites/runs within each experimental condition.
    site_data = data.loc[data["Scope"].astype(str).str.casefold().eq("site")].copy()
    for metric in args.metrics:
        metric_root = plot_root / metric
        for target in targets:
            target_data = site_data.loc[site_data["Target"].eq(target)]
            target_dir = slug(target)
            filename = "overall_{}_by_model_boxplot.png".format(metric.lower())
            if "all_regions" in args.scopes:
                hdr_dir = hdr_root / metric / "all_regions" / target_dir
                hdr_summary_png = hdr_dir / "all_regime_missingness_{}_hdr_by_model.png".format(metric.lower())
                hdr_missingness_overlay_png = hdr_dir / "all_regime_missingness_overlay_{}_hdr_by_model.png".format(metric.lower())
                hdr_regime_overlay_png = hdr_dir / "all_missingness_regime_overlay_{}_hdr_by_model.png".format(metric.lower())
                total += render_all_regions_hdr_summary(
                    target_data,
                    target,
                    metric,
                    hdr_summary_png,
                    models,
                    args,
                )
                total += render_all_regions_hdr_missingness_overlay(
                    target_data,
                    target,
                    metric,
                    hdr_missingness_overlay_png,
                    models,
                    args,
                )
                total += render_all_regions_hdr_regime_overlay(
                    target_data,
                    target,
                    metric,
                    hdr_regime_overlay_png,
                    models,
                    args,
                )
                total += render_hdr_combined_panel(
                    hdr_missingness_overlay_png,
                    hdr_regime_overlay_png,
                    hdr_dir / "all_regime_and_missingness_overlay_{}_hdr_by_model.png".format(metric.lower()),
                    metric,
                    target,
                    args,
                )
                total += render_scope(
                    target_data,
                    target,
                    metric,
                    "All Regions",
                    metric_root / "all_regions" / target_dir / filename,
                    models,
                    args,
                )
                metric_slug = metric.lower()
                for regime in sorted(target_data["Regime"].dropna().unique()):
                    total += render_all_regions_regime_missingness_subplots(
                        target_data,
                        target,
                        regime,
                        metric,
                        metric_root
                        / "all_regions"
                        / target_dir
                        / "by_regime"
                        / "{}_{}_by_missingness_subplots.png".format(
                            slug(regime).lower(),
                            metric_slug,
                        ),
                        models,
                        args,
                    )
                for missingness in sorted(target_data["Missingness_Level"].dropna().unique()):
                    total += render_all_regions_missingness_regime_subplots(
                        target_data,
                        target,
                        missingness,
                        metric,
                        metric_root
                        / "all_regions"
                        / target_dir
                        / "by_missingness"
                        / "{}_{}_by_regime_subplots.png".format(
                            missingness_slug(missingness),
                            metric_slug,
                        ),
                        models,
                        args,
                    )
            if "region" in args.scopes:
                for region, region_data in target_data.groupby("Region", sort=True):
                    total += render_scope(
                        region_data, target, metric,
                        str(region).replace("_", " ").title(),
                        metric_root / "by_region" / target_dir / slug(region) / filename,
                        models, args,
                    )
            if "station" in args.scopes:
                for (region, site), station_data in target_data.groupby(["Region", "Site"], sort=True):
                    total += render_scope(
                        station_data, target, metric, str(site).title(),
                        metric_root / "by_station" / target_dir / slug(region) / slug(site) / filename,
                        models, args,
                    )
    print("Completed: {} new PNG figures".format(total), flush=True)


if __name__ == "__main__":
    main()
