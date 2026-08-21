#!/usr/bin/env python3
"""Batch six-missingness distribution plots for all AQUISTIL spatial scopes.

Plot families:
  cdf                ECDF of absolute imputation error
  cdf_density        density of absolute imputation error
  kde_actual_imputed observed and imputed concentration densities
  error_distribution signed-error violin plus box plot
  qq                 observed-versus-imputed quantile comparison

Every figure represents one model, pollutant, spatial scope, and regime.  Its
six panels are the available missingness levels.  Each family is saved under a
separate directory in ``plots_by_type``.
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

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None


DEFAULT_RESULTS_ROOT = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Imputation_Result"
)
PREDICTION_FILE = "masked_predictions_by_site.csv"
PLOT_TYPES = ["cdf", "cdf_density", "kde_actual_imputed", "error_distribution", "qq"]
PLOT_FOLDERS = {
    "cdf": "CDF",
    "cdf_density": "CDF_density",
    "kde_actual_imputed": "KDE_actual_imputed",
    "error_distribution": "research_error_distribution",
    "qq": "QQ",
}
PLOT_LABELS = {
    "cdf": "CDF",
    "cdf_density": "CDF Density",
    "kde_actual_imputed": "KDE Actual vs Imputed",
    "error_distribution": "Error Distribution",
    "qq": "Q–Q",
}
REGIME_STYLE = {
    "random": ("Random", "#2458A6", "#EDF3FB"),
    "short_gap": ("Short Gap", "#C58A1B", "#FCF6E9"),
    "medium_gap": ("Medium Gap", "#7A68A6", "#F3F0F8"),
    "long_gap": ("Long Gap", "#637A35", "#F1F4EA"),
    "event": ("Event", "#B54775", "#FAEEF3"),
}
CHARCOAL = "#2F3437"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model", default="AQUISTIL")
    parser.add_argument("--targets", nargs="+", default=None, help="Default: all available pollutants.")
    parser.add_argument("--plot-types", nargs="+", choices=PLOT_TYPES, default=PLOT_TYPES)
    parser.add_argument(
        "--scopes", nargs="+", choices=["all_regions", "region", "station"],
        default=["all_regions", "region", "station"],
    )
    parser.add_argument("--max-samples-per-panel", type=int, default=10000)
    parser.add_argument("--axis-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def slug(value):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_")


def pollutant_label(target):
    return {"PM25": "PM2.5", "PM2.5": "PM2.5", "PM10": "PM10", "NO2": "NO₂", "OZONE": "O₃"}.get(target, target)


def discover_targets(results_root, model):
    root = results_root / "Regional_Pooled_Imputation" / model
    return sorted({path.parent.name for path in root.glob("*/*/{}".format(PREDICTION_FILE))})


def load_target(results_root, model, target):
    root = results_root / "Regional_Pooled_Imputation" / model
    files = sorted(root.glob("*/{}/{}".format(target, PREDICTION_FILE)))
    frames = []
    columns = ["Site", "Regime", "Missingness_Level", "Observed", "Imputed"]
    for path in files:
        part = pd.read_csv(path, usecols=columns)
        part["Region"] = path.parent.parent.name
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    for column in ["Missingness_Level", "Observed", "Imputed"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Site", "Regime", "Missingness_Level", "Observed", "Imputed"])
    return data.loc[
        data["Regime"].isin(REGIME_STYLE)
        & data["Observed"].ge(0) & data["Imputed"].ge(0)
    ].copy()


def sampled(part, maximum, seed):
    if maximum > 0 and len(part) > maximum:
        return part.sample(n=maximum, random_state=seed)
    return part


def density_curve(values, upper, points=300):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0) & (values <= upper)]
    grid = np.linspace(0, upper, points)
    if len(values) >= 4 and np.std(values) > 0 and gaussian_kde is not None:
        return grid, gaussian_kde(values)(grid)
    counts, edges = np.histogram(values, bins=40, range=(0, upper), density=True)
    return (edges[:-1] + edges[1:]) / 2.0, counts


def signed_density_curve(values, lower, upper, points=300):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= lower) & (values <= upper)]
    grid = np.linspace(lower, upper, points)
    if len(values) >= 4 and np.std(values) > 0 and gaussian_kde is not None:
        return grid, gaussian_kde(values)(grid)
    counts, edges = np.histogram(values, bins=40, range=(lower, upper), density=True)
    return (edges[:-1] + edges[1:]) / 2.0, counts


def style_panel(ax, background, tick_size=9):
    ax.set_facecolor(background)
    ax.grid(False)
    ax.tick_params(labelsize=tick_size, colors=CHARCOAL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_cdf(ax, part, colour, background, upper):
    errors = np.sort(np.abs(part["Imputed"].to_numpy() - part["Observed"].to_numpy()))
    errors = errors[errors <= upper]
    if len(errors):
        probability = np.arange(1, len(errors) + 1) / float(len(errors))
        ax.step(errors, probability, where="post", color=colour, linewidth=1.8)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, 1.01)
    style_panel(ax, background)


def draw_cdf_density(ax, part, colour, background, upper):
    errors = np.abs(part["Imputed"].to_numpy() - part["Observed"].to_numpy())
    grid, density = density_curve(errors, upper)
    ax.plot(grid, density, color=colour, linewidth=1.8)
    ax.fill_between(grid, 0, density, color=colour, alpha=0.28)
    ax.set_xlim(0, upper)
    ax.set_ylim(bottom=0)
    style_panel(ax, background)


def draw_actual_imputed(ax, part, colour, background, upper):
    observed = part["Observed"].to_numpy(dtype=float)
    imputed = part["Imputed"].to_numpy(dtype=float)
    grid_o, density_o = density_curve(observed, upper)
    grid_i, density_i = density_curve(imputed, upper)
    ax.plot(grid_o, density_o, color=CHARCOAL, linewidth=1.7, label="Observed")
    ax.plot(grid_i, density_i, color=colour, linestyle="--", linewidth=1.8, label="Imputed")
    ax.fill_between(grid_i, 0, density_i, color=colour, alpha=0.18)
    ax.set_xlim(0, upper)
    ax.set_ylim(bottom=0)
    style_panel(ax, background)


def draw_error_distribution(ax, part, colour, background, limit):
    errors = part["Imputed"].to_numpy(dtype=float) - part["Observed"].to_numpy(dtype=float)
    errors = errors[np.abs(errors) <= limit]
    if len(errors):
        violin = ax.violinplot([errors], positions=[1], widths=0.72, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(colour)
            body.set_edgecolor(CHARCOAL)
            body.set_alpha(0.55)
        ax.boxplot(
            [errors], positions=[1], widths=0.18, patch_artist=True, showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": CHARCOAL},
            medianprops={"color": CHARCOAL, "linewidth": 1.5},
            whiskerprops={"color": CHARCOAL}, capprops={"color": CHARCOAL},
        )
    ax.axhline(0, color=CHARCOAL, linestyle="--", linewidth=0.8)
    ax.set_xlim(0.45, 1.55)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([])
    style_panel(ax, background)


def draw_qq(ax, part, colour, background, upper):
    observed = part["Observed"].to_numpy(dtype=float)
    imputed = part["Imputed"].to_numpy(dtype=float)
    count = min(len(observed), 500)
    if count >= 2:
        probabilities = np.linspace(0.001, 0.999, count)
        observed_quantiles = np.quantile(observed, probabilities)
        imputed_quantiles = np.quantile(imputed, probabilities)
        keep = (observed_quantiles <= upper) & (imputed_quantiles <= upper)
        ax.scatter(
            observed_quantiles[keep], imputed_quantiles[keep], s=12,
            color=colour, edgecolors="none", alpha=0.65, rasterized=True,
        )
    ax.plot([0, upper], [0, upper], color=CHARCOAL, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    style_panel(ax, background)


def draw_qq_overlay(ax, part, upper, max_samples, seed):
    """Draw one Q-Q panel with all regimes overlaid for a fixed missingness level."""
    for index, (regime, (regime_label, colour, _background)) in enumerate(REGIME_STYLE.items()):
        regime_part = part.loc[part["Regime"].eq(regime)]
        if regime_part.empty:
            continue
        shown = sampled(regime_part, max_samples, seed + index)
        observed = shown["Observed"].to_numpy(dtype=float)
        imputed = shown["Imputed"].to_numpy(dtype=float)
        count = min(len(observed), 500)
        if count < 2:
            continue
        probabilities = np.linspace(0.001, 0.999, count)
        observed_quantiles = np.quantile(observed, probabilities)
        imputed_quantiles = np.quantile(imputed, probabilities)
        keep = (observed_quantiles <= upper) & (imputed_quantiles <= upper)
        ax.scatter(
            observed_quantiles[keep],
            imputed_quantiles[keep],
            s=14,
            color=colour,
            edgecolors="none",
            alpha=0.62,
            rasterized=True,
            label=regime_label,
        )
    ax.plot([0, upper], [0, upper], color=CHARCOAL, linestyle="--", linewidth=1.05)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    style_panel(ax, "white", tick_size=12)


def axis_limit(data, plot_type, percentile):
    if plot_type in ["kde_actual_imputed", "qq"]:
        values = np.concatenate([data["Observed"].to_numpy(), data["Imputed"].to_numpy()])
    else:
        values = np.abs(data["Imputed"].to_numpy() - data["Observed"].to_numpy())
    upper = float(np.percentile(values, percentile))
    return max(upper * 1.04, 1e-6)


def labels_for(plot_type, pollutant):
    if plot_type == "cdf":
        return "Absolute error", "Empirical CDF"
    if plot_type == "cdf_density":
        return "Absolute error", "Density"
    if plot_type == "kde_actual_imputed":
        return "{} concentration".format(pollutant), "Density"
    if plot_type == "qq":
        return "Observed {} quantiles".format(pollutant), "Imputed {} quantiles".format(pollutant)
    return "", "Imputed - observed {}".format(pollutant)


def render_qq_regime_overlay_by_missingness(data, target, model, scope_label, scope_path, plot_root, args):
    """Render one Q-Q plot per missingness level with all regimes overlaid."""
    if data.empty:
        return 0
    levels = sorted(data["Missingness_Level"].unique())
    pollutant = pollutant_label(target)
    made = 0
    output_dir = plot_root / PLOT_FOLDERS["qq"] / scope_path / target / model / "by_missingness_regime_overlay"
    for index, level in enumerate(levels):
        level_data = data.loc[np.isclose(data["Missingness_Level"], level)]
        if level_data.empty:
            continue
        output = output_dir / "missingness_{:02.0f}_all_regimes_qq.png".format(level * 100)
        if output.exists() and not args.force:
            continue
        limit = axis_limit(level_data, "qq", args.axis_percentile)
        fig, ax = plt.subplots(figsize=(8.27, 8.27))
        fig.patch.set_facecolor("white")
        draw_qq_overlay(ax, level_data, limit, args.max_samples_per_panel, args.seed + index)
        x_label, y_label = labels_for("qq", pollutant)
        ax.set_xlabel(x_label, fontsize=15)
        ax.set_ylabel(y_label, fontsize=15)
        ax.set_title(
            "{}: all regimes - {} - {:g}% missing - Q-Q".format(
                model,
                scope_label,
                level * 100,
            ),
            fontsize=17,
            fontweight="bold",
            pad=12,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                title="Regime",
                frameon=False,
                fontsize=11,
                title_fontsize=12,
                loc="upper left",
            )
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print("Saved {}".format(output), flush=True)
        made += 1
    return made


def render_scope(data, target, model, scope_label, scope_path, plot_root, args):
    if data.empty:
        return 0
    levels = sorted(data["Missingness_Level"].unique())
    if len(levels) > 6:
        raise ValueError("Expected at most six missingness levels; found {}".format(len(levels)))
    made = 0
    pollutant = pollutant_label(target)
    for regime, (regime_label, colour, background) in REGIME_STYLE.items():
        regime_data = data.loc[data["Regime"].eq(regime)]
        if regime_data.empty:
            continue
        for plot_type in args.plot_types:
            output = (
                plot_root / PLOT_FOLDERS[plot_type] / scope_path / target / model
                / "regime_{}_six_missingness_{}.png".format(regime, plot_type)
            )
            if output.exists() and not args.force:
                continue
            limit = axis_limit(regime_data, plot_type, args.axis_percentile)
            fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), sharex=True, sharey=True, squeeze=False)
            for index, level in enumerate(levels):
                ax = axes.ravel()[index]
                part = regime_data.loc[np.isclose(regime_data["Missingness_Level"], level)]
                shown = sampled(part, args.max_samples_per_panel, args.seed + index)
                if plot_type == "cdf":
                    draw_cdf(ax, shown, colour, background, limit)
                elif plot_type == "cdf_density":
                    draw_cdf_density(ax, shown, colour, background, limit)
                elif plot_type == "kde_actual_imputed":
                    draw_actual_imputed(ax, shown, colour, background, limit)
                elif plot_type == "qq":
                    draw_qq(ax, shown, colour, background, limit)
                else:
                    draw_error_distribution(ax, shown, colour, background, limit)
                ax.set_title("{:g}% missing".format(level * 100), fontsize=13, fontweight="bold")
            for ax in axes.ravel()[len(levels):]:
                ax.set_visible(False)
            x_label, y_label = labels_for(plot_type, pollutant)
            fig.text(0.5, 0.025, x_label, ha="center", fontsize=15)
            fig.text(0.012, 0.5, y_label, va="center", rotation="vertical", fontsize=15)
            fig.suptitle("{}: {} — {} — {}".format(model, regime_label, scope_label, PLOT_LABELS[plot_type]), fontsize=18, fontweight="bold", y=0.97)
            if plot_type == "kde_actual_imputed":
                axes.ravel()[0].legend(frameon=False, fontsize=9)
            fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.91, wspace=0.10, hspace=0.20)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print("Saved {}".format(output), flush=True)
            made += 1
    if "qq" in args.plot_types:
        made += render_qq_regime_overlay_by_missingness(
            data, target, model, scope_label, scope_path, plot_root, args
        )
    return made


def main():
    args = parse_args()
    if not 90 <= args.axis_percentile <= 100:
        raise ValueError("--axis-percentile must be between 90 and 100")
    targets = args.targets or discover_targets(args.results_root, args.model)
    plot_root = args.results_root / "plots_by_type"
    total = 0
    for target in targets:
        print("Loading {}".format(target), flush=True)
        data = load_target(args.results_root, args.model, target)
        if data.empty:
            continue
        if "all_regions" in args.scopes:
            total += render_scope(data, target, args.model, "All Regions", Path("all_regions"), plot_root, args)
        if "region" in args.scopes:
            for region, region_data in data.groupby("Region", sort=True):
                total += render_scope(region_data, target, args.model, str(region).replace("_", " ").title(), Path("by_region") / slug(region), plot_root, args)
        if "station" in args.scopes:
            for (region, site), site_data in data.groupby(["Region", "Site"], sort=True):
                total += render_scope(site_data, target, args.model, str(site).title(), Path("by_station") / slug(region) / slug(site), plot_root, args)
        del data
        gc.collect()
    print("Completed: {} new PNG figures".format(total), flush=True)


if __name__ == "__main__":
    main()
