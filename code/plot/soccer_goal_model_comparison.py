#!/usr/bin/env python3
"""Create soccer-goal plots comparing AQUISTIL with other imputation models.

The plot follows the model-performance style used in Yumimoto et al. Fig. 5:
each point is a model summary, x is normalized mean bias (NMB), y is
normalized mean error (NME), and point color shows Pearson R.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "Outputs" / "Imputation_Result"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "plots_by_type" / "soccer_goal"
PREDICTION_GLOB = "Regional_Pooled_Imputation/*/*/*/masked_predictions_by_site.csv"
EPS = 1e-10


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(".", "")
    )


def partial_summaries(frame: pd.DataFrame, group_cols) -> pd.DataFrame:
    frame = frame.copy()
    frame["Observed"] = pd.to_numeric(frame["Observed"], errors="coerce")
    frame["Imputed"] = pd.to_numeric(frame["Imputed"], errors="coerce")
    frame = frame[np.isfinite(frame["Observed"]) & np.isfinite(frame["Imputed"])]
    frame = frame[frame["Observed"].abs() >= EPS]
    if frame.empty:
        return pd.DataFrame()
    frame["_err"] = frame["Imputed"] - frame["Observed"]
    frame["_abs_err"] = frame["_err"].abs()
    frame["_obs2"] = frame["Observed"] * frame["Observed"]
    frame["_imp2"] = frame["Imputed"] * frame["Imputed"]
    frame["_obs_imp"] = frame["Observed"] * frame["Imputed"]
    return (
        frame.groupby(group_cols, dropna=False)
        .agg(
            sum_obs=("Observed", "sum"),
            sum_imp=("Imputed", "sum"),
            sum_err=("_err", "sum"),
            sum_abs_err=("_abs_err", "sum"),
            sum_obs2=("_obs2", "sum"),
            sum_imp2=("_imp2", "sum"),
            sum_obs_imp=("_obs_imp", "sum"),
            N=("Observed", "size"),
        )
        .reset_index()
    )


def combine_partials(partials, group_cols) -> pd.DataFrame:
    if not partials:
        return pd.DataFrame()
    data = pd.concat(partials, ignore_index=True)
    sum_cols = [
        "sum_obs",
        "sum_imp",
        "sum_err",
        "sum_abs_err",
        "sum_obs2",
        "sum_imp2",
        "sum_obs_imp",
        "N",
    ]
    return data.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()


def finalize_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["NMB"] = 100.0 * out["sum_err"] / out["sum_obs"]
    out["NME"] = 100.0 * out["sum_abs_err"] / out["sum_obs"]
    cov = out["sum_obs_imp"] - (out["sum_obs"] * out["sum_imp"] / out["N"])
    var_o = out["sum_obs2"] - (out["sum_obs"] * out["sum_obs"] / out["N"])
    var_s = out["sum_imp2"] - (out["sum_imp"] * out["sum_imp"] / out["N"])
    var_o = np.maximum(var_o, 0.0)
    var_s = np.maximum(var_s, 0.0)
    denom = np.sqrt(var_o * var_s)
    out["R"] = np.where(denom > EPS, cov / denom, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["NMB", "NME"])


def load_summaries(results_dir: Path, chunksize: int = 500000):
    files = sorted(results_dir.glob(PREDICTION_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No masked prediction files found under {results_dir / PREDICTION_GLOB}"
        )

    overall_partials = []
    target_partials = []
    detail_partials = []
    usecols = [
        "Region",
        "Site",
        "Target",
        "Model",
        "Regime",
        "Missingness_Level",
        "Seed",
        "Observed",
        "Imputed",
        "Was_Artificially_Masked",
    ]
    for file_number, file_path in enumerate(files, start=1):
        try:
            reader = pd.read_csv(file_path, usecols=lambda c: c in usecols, chunksize=chunksize)
        except Exception as exc:
            print(f"Skipping unreadable file: {file_path} ({exc})")
            continue
        for frame in reader:
            if frame.empty:
                continue
            if "Was_Artificially_Masked" in frame.columns:
                frame = frame[frame["Was_Artificially_Masked"].astype(bool)]
            if frame.empty:
                continue
            overall_partials.append(partial_summaries(frame, ["Model"]))
            target_partials.append(partial_summaries(frame, ["Target", "Model"]))
            detail_partials.append(
                partial_summaries(
                    frame,
                    ["Target", "Model", "Region", "Site", "Regime", "Missingness_Level", "Seed"],
                )
            )
        if file_number % 10 == 0 or file_number == len(files):
            print(f"Processed {file_number}/{len(files)} prediction files", flush=True)

    if not overall_partials:
        raise ValueError("Prediction files were found, but no masked rows were usable.")

    overall = finalize_metrics(combine_partials(overall_partials, ["Model"]))
    by_target = finalize_metrics(combine_partials(target_partials, ["Target", "Model"]))
    detail = finalize_metrics(
        combine_partials(
            detail_partials,
            ["Target", "Model", "Region", "Site", "Regime", "Missingness_Level", "Seed"],
        )
    )
    return overall, by_target, detail


def plot_soccer(summary: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    finite = summary[np.isfinite(summary["NMB"]) & np.isfinite(summary["NME"])].copy()
    if finite.empty:
        raise ValueError(f"No finite rows available for {title}")

    x_abs = max(40.0, float(np.nanpercentile(np.abs(finite["NMB"]), 98)) * 1.15)
    y_max = max(50.0, float(np.nanpercentile(finite["NME"], 98)) * 1.15)
    x_abs = min(x_abs, max(40.0, float(np.nanmax(np.abs(finite["NMB"]))) * 1.05))
    y_max = min(y_max, max(50.0, float(np.nanmax(finite["NME"])) * 1.05))

    ax.axvline(0, color="0.25", linewidth=1.2, zorder=0)
    ax.axhline(0, color="0.25", linewidth=0.8, zorder=0)
    ax.add_patch(
        plt.Rectangle((-30, 0), 60, 50, fill=False, linestyle="-.", linewidth=1.8, color="0.25")
    )
    ax.add_patch(
        plt.Rectangle((-15, 0), 30, 30, fill=False, linestyle="-", linewidth=2.0, color="0.1")
    )

    models = list(finite["Model"].drop_duplicates())
    markers = {
        "AQUISTIL": "*",
        "LightGBM": "o",
        "XGBoost": "s",
        "MICE": "^",
        "MICE-KNN": "D",
        "KNN": "P",
        "Mean": "X",
        "Median": "v",
    }
    for model in models:
        part = finite[finite["Model"] == model]
        marker = markers.get(model, "o")
        size = 270 if model == "AQUISTIL" else 95
        edge = "black" if model == "AQUISTIL" else "white"
        ax.scatter(
            part["NMB"],
            part["NME"],
            c=part["R"],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=size,
            marker=marker,
            edgecolors=edge,
            linewidths=1.0,
            alpha=0.92,
            label=model,
        )

    for _, row in finite.iterrows():
        if row["Model"] == "AQUISTIL":
            ax.annotate(
                "AQUISTIL",
                (row["NMB"], row["NME"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.015)
    cbar.set_label("Pearson correlation (R)")

    ax.set_xlim(-x_abs, x_abs)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Normalized mean bias, NMB (%)")
    ax.set_ylabel("Normalized mean error, NME (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.text(-29, 47, "criterion", fontsize=8, color="0.25")
    ax.text(-14, 27, "goal", fontsize=8, color="0.1")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_soccer_individual(summary: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.8))

    finite = summary[np.isfinite(summary["NMB"]) & np.isfinite(summary["NME"])].copy()
    if finite.empty:
        raise ValueError(f"No finite rows available for {title}")

    x_abs = max(40.0, float(np.nanpercentile(np.abs(finite["NMB"]), 98)) * 1.20)
    y_max = max(50.0, float(np.nanpercentile(finite["NME"], 98)) * 1.20)

    ax.axvline(0, color="0.25", linewidth=1.1, zorder=0)
    ax.axhline(0, color="0.25", linewidth=0.8, zorder=0)
    ax.add_patch(
        plt.Rectangle((-30, 0), 60, 50, fill=False, linestyle="-.", linewidth=1.8, color="0.25")
    )
    ax.add_patch(
        plt.Rectangle((-15, 0), 30, 30, fill=False, linestyle="-", linewidth=2.0, color="0.1")
    )

    markers = {
        "AQUISTIL": "*",
        "LightGBM": "o",
        "XGBoost": "s",
        "MICE": "^",
        "MICE-KNN": "D",
    }
    for model in sorted(finite["Model"].dropna().unique()):
        part = finite[finite["Model"] == model]
        is_aquistil = model == "AQUISTIL"
        ax.scatter(
            part["NMB"],
            part["NME"],
            c=part["R"],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=120 if is_aquistil else 34,
            marker=markers.get(model, "o"),
            edgecolors="black" if is_aquistil else "none",
            linewidths=0.7 if is_aquistil else 0.0,
            alpha=0.88 if is_aquistil else 0.42,
            label=f"{model} (n={len(part)})",
        )

    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.015)
    cbar.set_label("Pearson correlation (R)")

    ax.set_xlim(-x_abs, x_abs)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Normalized mean bias, NMB (%)")
    ax.set_ylabel("Normalized mean error, NME (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.text(-29, 47, "criterion", fontsize=8, color="0.25")
    ax.text(-14, 27, "goal", fontsize=8, color="0.1")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_soccer_categorical(
    summary: pd.DataFrame,
    title: str,
    output_path: Path,
    color_col: str,
    color_label: str,
) -> None:
    """Soccer-goal scatter where marker shape is model and color is a category."""
    fig, ax = plt.subplots(figsize=(11.4, 8.2))

    finite = summary[np.isfinite(summary["NMB"]) & np.isfinite(summary["NME"])].copy()
    finite = finite.dropna(subset=["Model", color_col])
    if finite.empty:
        raise ValueError(f"No finite rows available for {title}")

    if color_col == "Missingness_Level":
        finite["_color_key"] = finite[color_col].map(missingness_label)
        category_order = [
            missingness_label(v)
            for v in sorted(finite[color_col].dropna().unique(), key=lambda x: float(x))
        ]
    else:
        finite["_color_key"] = finite[color_col].astype(str)
        category_order = sorted(finite["_color_key"].dropna().unique())

    palette = plt.get_cmap("tab10")
    color_map = {cat: palette(i % 10) for i, cat in enumerate(category_order)}

    x_abs = max(40.0, float(np.nanpercentile(np.abs(finite["NMB"]), 98)) * 1.20)
    y_max = max(50.0, float(np.nanpercentile(finite["NME"], 98)) * 1.20)

    ax.axvline(0, color="0.25", linewidth=1.1, zorder=0)
    ax.axhline(0, color="0.25", linewidth=0.8, zorder=0)
    ax.add_patch(
        plt.Rectangle((-30, 0), 60, 50, fill=False, linestyle="-.", linewidth=1.8, color="0.25")
    )
    ax.add_patch(
        plt.Rectangle((-15, 0), 30, 30, fill=False, linestyle="-", linewidth=2.0, color="0.1")
    )

    markers = {
        "AQUISTIL": "*",
        "LightGBM": "o",
        "XGBoost": "s",
        "MICE": "^",
        "MICE-KNN": "D",
        "KNN": "P",
        "Mean": "X",
        "Median": "v",
    }
    for model in sorted(finite["Model"].dropna().unique()):
        model_part = finite[finite["Model"] == model]
        is_aquistil = model == "AQUISTIL"
        for category in category_order:
            part = model_part[model_part["_color_key"] == category]
            if part.empty:
                continue
            ax.scatter(
                part["NMB"],
                part["NME"],
                color=color_map[category],
                s=140 if is_aquistil else 42,
                marker=markers.get(model, "o"),
                edgecolors="black" if is_aquistil else "white",
                linewidths=0.75 if is_aquistil else 0.35,
                alpha=0.88 if is_aquistil else 0.55,
            )

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=markers.get(model, "o"),
            color="none",
            markerfacecolor="0.45",
            markeredgecolor="black",
            markersize=11 if model == "AQUISTIL" else 8,
            label=model,
        )
        for model in sorted(finite["Model"].dropna().unique())
    ]
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_map[category],
            markeredgecolor="white",
            markersize=8,
            label=category,
        )
        for category in category_order
    ]

    ax.set_xlim(-x_abs, x_abs)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Normalized mean bias, NMB (%)")
    ax.set_ylabel("Normalized mean error, NME (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.text(-29, 47, "criterion", fontsize=8, color="0.25")
    ax.text(-14, 27, "goal", fontsize=8, color="0.1")

    model_legend = ax.legend(
        handles=model_handles,
        title="Model marker",
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=color_handles,
        title=color_label,
        loc="center right",
        bbox_to_anchor=(1.0, 0.55),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def aggregate_metrics(summary: pd.DataFrame, group_cols) -> pd.DataFrame:
    sum_cols = [
        "sum_obs",
        "sum_imp",
        "sum_err",
        "sum_abs_err",
        "sum_obs2",
        "sum_imp2",
        "sum_obs_imp",
        "N",
    ]
    available = [col for col in sum_cols if col in summary.columns]
    if len(available) != len(sum_cols):
        missing = sorted(set(sum_cols) - set(available))
        raise ValueError(f"Cannot aggregate soccer-goal metrics; missing columns: {missing}")
    grouped = summary.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    return finalize_metrics(grouped)


def missingness_label(value) -> str:
    try:
        pct = float(value) * 100.0 if float(value) <= 1.0 else float(value)
        if abs(pct - round(pct)) < 1e-8:
            return f"{int(round(pct))}pct"
        return f"{pct:g}pct".replace(".", "p")
    except Exception:
        return safe_name(value)


def add_missingness_plots(detail: pd.DataFrame, output_dir: Path, min_n: int) -> list:
    outputs = []
    for target, target_detail in sorted(detail.groupby("Target", dropna=False), key=lambda x: str(x[0])):
        target_slug = safe_name(target)
        for missingness, miss_detail in sorted(
            target_detail.groupby("Missingness_Level", dropna=False),
            key=lambda x: float(x[0]) if pd.notna(x[0]) else -1.0,
        ):
            miss_detail = miss_detail[miss_detail["N"] >= min_n]
            if miss_detail.empty:
                continue
            miss_slug = missingness_label(missingness)
            csv_path = output_dir / f"soccer_goal_{target_slug}_missingness_{miss_slug}_individual_metrics.csv"
            png_path = output_dir / f"soccer_goal_{target_slug}_missingness_{miss_slug}_individual.png"
            miss_detail.to_csv(csv_path, index=False)
            plot_soccer_individual(
                miss_detail,
                f"AQUISTIL Soccer-Goal Individual Evaluations: {target}, missingness {miss_slug}",
                png_path,
            )
            outputs.extend([csv_path, png_path])
    return outputs


def add_missingness_regime_plots(detail: pd.DataFrame, output_dir: Path, min_n: int) -> list:
    outputs = []
    for target, target_detail in sorted(detail.groupby("Target", dropna=False), key=lambda x: str(x[0])):
        target_slug = safe_name(target)
        grouped = target_detail.groupby(["Missingness_Level", "Regime"], dropna=False)
        for (missingness, regime), split_detail in sorted(
            grouped,
            key=lambda x: (
                float(x[0][0]) if pd.notna(x[0][0]) else -1.0,
                str(x[0][1]),
            ),
        ):
            split_detail = split_detail[split_detail["N"] >= min_n]
            if split_detail.empty:
                continue
            miss_slug = missingness_label(missingness)
            regime_slug = safe_name(regime)
            stem = f"soccer_goal_{target_slug}_missingness_{miss_slug}_regime_{regime_slug}_individual"
            csv_path = output_dir / f"{stem}_metrics.csv"
            png_path = output_dir / f"{stem}.png"
            split_detail.to_csv(csv_path, index=False)
            plot_soccer_individual(
                split_detail,
                f"AQUISTIL Soccer-Goal: {target}, {miss_slug}, {regime}",
                png_path,
            )
            outputs.extend([csv_path, png_path])
    return outputs


def add_aggregated_target_plots(detail: pd.DataFrame, output_dir: Path, min_n: int) -> list:
    """Save the three paper-summary soccer-goal plot types for every pollutant."""
    outputs = []
    for target, target_detail in sorted(detail.groupby("Target", dropna=False), key=lambda x: str(x[0])):
        target_detail = target_detail[target_detail["N"] >= min_n].copy()
        if target_detail.empty:
            continue
        target_slug = safe_name(target)

        by_model = aggregate_metrics(target_detail, ["Target", "Model"])
        by_model_csv = output_dir / f"soccer_goal_{target_slug}_overall_by_model_metrics.csv"
        by_model_png = output_dir / f"soccer_goal_{target_slug}_overall_by_model.png"
        by_model.to_csv(by_model_csv, index=False)
        plot_soccer(
            by_model,
            f"{target} Soccer-Goal: overall by model",
            by_model_png,
        )
        outputs.extend([by_model_csv, by_model_png])

        by_model_regime = aggregate_metrics(target_detail, ["Target", "Model", "Regime"])
        by_model_regime_csv = output_dir / f"soccer_goal_{target_slug}_by_model_regime_avg_missingness_metrics.csv"
        by_model_regime_png = output_dir / f"soccer_goal_{target_slug}_by_model_regime_avg_missingness.png"
        by_model_regime.to_csv(by_model_regime_csv, index=False)
        plot_soccer_categorical(
            by_model_regime,
            f"{target} Soccer-Goal: marker = model, color = regime; pooled across missingness",
            by_model_regime_png,
            color_col="Regime",
            color_label="Regime color",
        )
        outputs.extend([by_model_regime_csv, by_model_regime_png])

        by_model_missingness = aggregate_metrics(target_detail, ["Target", "Model", "Missingness_Level"])
        by_model_missingness_csv = output_dir / f"soccer_goal_{target_slug}_by_model_missingness_avg_regime_metrics.csv"
        by_model_missingness_png = output_dir / f"soccer_goal_{target_slug}_by_model_missingness_avg_regime.png"
        by_model_missingness.to_csv(by_model_missingness_csv, index=False)
        plot_soccer_categorical(
            by_model_missingness,
            f"{target} Soccer-Goal: marker = model, color = missingness; pooled across regimes",
            by_model_missingness_png,
            color_col="Missingness_Level",
            color_label="Missingness color",
        )
        outputs.extend([by_model_missingness_csv, by_model_missingness_png])

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument(
        "--aggregated-only",
        action="store_true",
        help="Save only the three paper-summary plot types per pollutant.",
    )
    args = parser.parse_args()

    overall, by_target, detail = load_summaries(args.results_dir, chunksize=args.chunksize)
    outputs = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregated_only:
        overall = overall[overall["N"] >= args.min_n]
        overall_csv = args.output_dir / "soccer_goal_overall_metrics.csv"
        overall.to_csv(overall_csv, index=False)
        overall_png = args.output_dir / "soccer_goal_overall.png"
        plot_soccer(overall, "AQUISTIL Soccer-Goal Model Comparison: all targets", overall_png)
        outputs.extend([overall_csv, overall_png])

        for target, summary in sorted(by_target.groupby("Target", dropna=False), key=lambda x: str(x[0])):
            summary = summary[summary["N"] >= args.min_n]
            if summary.empty:
                continue
            target_slug = safe_name(target)
            csv_path = args.output_dir / f"soccer_goal_{target_slug}_metrics.csv"
            png_path = args.output_dir / f"soccer_goal_{target_slug}.png"
            summary.to_csv(csv_path, index=False)
            plot_soccer(summary, f"AQUISTIL Soccer-Goal Model Comparison: {target}", png_path)
            outputs.extend([csv_path, png_path])

            detail_summary = detail[(detail["Target"] == target) & (detail["N"] >= args.min_n)]
            if not detail_summary.empty:
                detail_csv = args.output_dir / f"soccer_goal_{target_slug}_individual_metrics.csv"
                detail_png = args.output_dir / f"soccer_goal_{target_slug}_individual.png"
                detail_summary.to_csv(detail_csv, index=False)
                plot_soccer_individual(
                    detail_summary,
                    f"AQUISTIL Soccer-Goal Individual Evaluations: {target}",
                    detail_png,
                )
                outputs.extend([detail_csv, detail_png])

        outputs.extend(add_missingness_plots(detail, args.output_dir, args.min_n))
        outputs.extend(add_missingness_regime_plots(detail, args.output_dir, args.min_n))

    outputs.extend(add_aggregated_target_plots(detail, args.output_dir, args.min_n))

    print("Generated files:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
