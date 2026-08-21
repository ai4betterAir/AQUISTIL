#!/usr/bin/env python3
"""Create a PM2.5 critical-difference style model-rank plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "Outputs" / "Imputation_Result"
METRICS_CSV = RESULTS_DIR / "Metrics" / "regional_pooled_metrics.csv"
OUTPUT_DIR = RESULTS_DIR / "plots_by_type" / "rank_plot"
OUTPUT_PNG = OUTPUT_DIR / "pm25_critical_difference_rank_rmse.png"
OUTPUT_CSV = OUTPUT_DIR / "pm25_critical_difference_rank_rmse_metrics.csv"
COMBINED_OUTPUT_PNG = OUTPUT_DIR / "pm25_rank_overall_regime_missingness_rmse.png"
COMBINED_OUTPUT_CSV = OUTPUT_DIR / "pm25_rank_overall_regime_missingness_rmse_metrics.csv"

# Demsar/Nemenyi critical values for alpha=0.05, infinite degrees of freedom.
# Values are q_alpha / sqrt(2), matching CD = q * sqrt(k(k+1)/(6N)).
NEMENYI_Q_ALPHA_05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
    16: 3.426,
    17: 3.458,
    18: 3.489,
    19: 3.517,
    20: 3.544,
}


def bootstrap_ci(rank_matrix: pd.DataFrame, n_boot: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rank_matrix.to_numpy(dtype=float)
    n = values.shape[0]
    samples = np.empty((n_boot, values.shape[1]), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i, :] = values[idx, :].mean(axis=0)
    return pd.DataFrame(
        {
            "Model": rank_matrix.columns,
            "Rank_CI_Low": np.percentile(samples, 2.5, axis=0),
            "Rank_CI_High": np.percentile(samples, 97.5, axis=0),
        }
    )


def missingness_label(value) -> str:
    pct = float(value) * 100.0 if float(value) <= 1.0 else float(value)
    if abs(pct - round(pct)) < 1e-8:
        return f"{int(round(pct))}%"
    return f"{pct:g}%"


def mean_rank_summary(
    frame: pd.DataFrame,
    key_cols: list,
    group_col: str = None,
    group_type: str = "overall",
    group_label: str = "Overall",
) -> pd.DataFrame:
    """Rank models by RMSE inside scenarios, then average ranks by optional split."""
    rows = []
    if group_col is None:
        groups = [(group_label, frame)]
    else:
        groups = sorted(frame.groupby(group_col, dropna=False), key=lambda x: str(x[0]))

    for value, part in groups:
        pivot = part.pivot_table(index=key_cols, columns="Model", values="RMSE", aggfunc="mean")
        pivot = pivot.dropna(axis=0, how="any")
        if pivot.empty:
            continue
        ranks = pivot.rank(axis=1, method="average", ascending=True)
        label = missingness_label(value) if group_col == "Missingness_Level" else str(value)
        for model, rank in ranks.mean(axis=0).items():
            rows.append(
                {
                    "Model": model,
                    "Group_Type": group_type,
                    "Group_Label": label,
                    "Mean_Rank": rank,
                    "N_Scenarios": len(ranks),
                    "Mean_RMSE": pivot[model].mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_combined_rank_summary(summary: pd.DataFrame, model_order: list, output_path: Path) -> None:
    fig_height = max(7.2, 0.56 * len(model_order) + 2.4)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))

    y_lookup = {model: i for i, model in enumerate(model_order)}
    offsets = {"overall": 0.0, "regime": -0.16, "missingness": 0.16}
    markers = {"overall": "D", "regime": "o", "missingness": "^"}

    regime_labels = ["event", "long_gap", "medium_gap", "random", "short_gap"]
    missingness_labels = ["5%", "10%", "20%", "30%", "50%", "60%"]
    palette = plt.get_cmap("tab10")
    colors = {"Overall": "black"}
    colors.update({label: palette(i) for i, label in enumerate(regime_labels)})
    colors.update({label: palette(i + len(regime_labels)) for i, label in enumerate(missingness_labels)})

    for _, row in summary.iterrows():
        model = row["Model"]
        if model not in y_lookup:
            continue
        group_type = row["Group_Type"]
        label = row["Group_Label"]
        is_overall = group_type == "overall"
        ax.scatter(
            row["Mean_Rank"],
            y_lookup[model] + offsets.get(group_type, 0.0),
            marker=markers.get(group_type, "o"),
            s=115 if is_overall else 48,
            color=colors.get(label, "0.5"),
            edgecolors="black" if is_overall else "white",
            linewidths=0.8 if is_overall else 0.35,
            alpha=0.95 if is_overall else 0.78,
            zorder=3 if is_overall else 2,
        )

    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order)
    ax.invert_yaxis()
    ax.set_xlim(0.5, len(model_order) + 0.7)
    ax.set_xlabel("Mean rank by RMSE (1 = best)")
    ax.set_title("PM2.5 Rank Summary: overall, by regime, and by missingness")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    marker_handles = [
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="black", markeredgecolor="black", markersize=8, label="Overall"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markeredgecolor="white", markersize=8, label="Regime"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="0.45", markeredgecolor="white", markersize=8, label="Missingness"),
    ]
    category_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[label], markeredgecolor="white", markersize=7, label=label)
        for label in ["Overall"] + regime_labels + missingness_labels
        if label in set(summary["Group_Label"])
    ]

    marker_legend = ax.legend(
        handles=marker_handles,
        title="Marker type",
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(marker_legend)
    ax.legend(
        handles=category_handles,
        title="Color category",
        loc="center right",
        bbox_to_anchor=(1.0, 0.50),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    ax.text(
        0.5,
        1.03,
        "Overall uses all comparable scenarios; regime points pool missingness; missingness points pool regimes.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.25",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    usecols = [
        "Region",
        "Target",
        "Model",
        "Regime",
        "Missingness_Level",
        "Seed",
        "Scope",
        "RMSE",
    ]
    frame = pd.read_csv(METRICS_CSV, usecols=usecols)
    frame = frame[
        (frame["Target"] == "PM2.5")
        & (frame["Scope"] == "Region_Macro")
    ].copy()
    frame["RMSE"] = pd.to_numeric(frame["RMSE"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    key_cols = ["Region", "Regime", "Missingness_Level", "Seed"]
    counts = frame.groupby("Model")[key_cols].apply(lambda x: x.drop_duplicates().shape[0])
    complete_models = counts[counts == counts.max()].index.tolist()
    frame = frame[frame["Model"].isin(complete_models)]

    pivot = frame.pivot_table(index=key_cols, columns="Model", values="RMSE", aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")
    ranks = pivot.rank(axis=1, method="average", ascending=True)

    mean_ranks = ranks.mean(axis=0).rename("Mean_Rank").reset_index()
    rmse_mean = pivot.mean(axis=0).rename("Mean_RMSE").reset_index()
    ci = bootstrap_ci(ranks)
    summary = mean_ranks.merge(rmse_mean, on="Model").merge(ci, on="Model")
    summary = summary.sort_values("Mean_Rank").reset_index(drop=True)

    k = len(summary)
    n = len(ranks)
    q_alpha = NEMENYI_Q_ALPHA_05.get(k, NEMENYI_Q_ALPHA_05[20])
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
    summary["N_Scenarios"] = n
    summary["N_Models"] = k
    summary["Critical_Difference"] = cd
    summary["Included_In_CD"] = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    fig_height = max(6.0, 0.48 * k + 2.2)
    fig, ax = plt.subplots(figsize=(9.8, fig_height))
    y = np.arange(k)
    x = summary["Mean_Rank"].to_numpy()
    xerr = np.vstack(
        [
            x - summary["Rank_CI_Low"].to_numpy(),
            summary["Rank_CI_High"].to_numpy() - x,
        ]
    )

    colors = ["#1f77b4" if model == "AQUISTIL" else "#595959" for model in summary["Model"]]
    sizes = [110 if model == "AQUISTIL" else 62 for model in summary["Model"]]
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="0.60",
        elinewidth=1.4,
        capsize=3,
        zorder=1,
    )
    ax.scatter(x, y, s=sizes, c=colors, edgecolors="black", linewidths=0.6, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(summary["Model"])
    ax.invert_yaxis()
    ax.set_xlim(0.5, k + 0.65)
    ax.set_xlabel("Mean rank by RMSE (1 = best)")
    ax.set_title("PM2.5 Critical-Difference Rank Plot")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    cd_y = k + 0.1
    cd_x0 = 1.0
    cd_x1 = cd_x0 + cd
    ax.plot([cd_x0, cd_x1], [cd_y, cd_y], color="black", linewidth=2.0, clip_on=False)
    ax.plot([cd_x0, cd_x0], [cd_y - 0.08, cd_y + 0.08], color="black", linewidth=1.6, clip_on=False)
    ax.plot([cd_x1, cd_x1], [cd_y - 0.08, cd_y + 0.08], color="black", linewidth=1.6, clip_on=False)
    ax.text(
        (cd_x0 + cd_x1) / 2,
        cd_y + 0.28,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        0.5,
        1.03,
        f"Region_Macro scenarios: N={n}; complete models: k={k}; metric: RMSE",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.25",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    combined = pd.concat(
        [
            mean_rank_summary(frame, key_cols, group_type="overall", group_label="Overall"),
            mean_rank_summary(frame, key_cols, group_col="Regime", group_type="regime"),
            mean_rank_summary(
                frame,
                key_cols,
                group_col="Missingness_Level",
                group_type="missingness",
            ),
        ],
        ignore_index=True,
    )
    model_order = summary["Model"].tolist()
    combined["Model"] = pd.Categorical(combined["Model"], categories=model_order, ordered=True)
    combined = combined.sort_values(["Model", "Group_Type", "Group_Label"]).reset_index(drop=True)
    combined.to_csv(COMBINED_OUTPUT_CSV, index=False)
    plot_combined_rank_summary(combined, model_order, COMBINED_OUTPUT_PNG)

    print(OUTPUT_CSV)
    print(OUTPUT_PNG)
    print(COMBINED_OUTPUT_CSV)
    print(COMBINED_OUTPUT_PNG)
    print(summary[["Model", "Mean_Rank", "Rank_CI_Low", "Rank_CI_High", "Mean_RMSE"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
