import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300
})


# ============================================================
# 1) GAP-WISE BOXPLOTS
# ============================================================
def plot_gapwise_boxplots(
    metrics_csvs,
    metric="RMSE",
    out_dir="plots",
    title_suffix=""
):
    """
    metrics_csvs: dict
        {
            "LightGBM": "path/to/lgbm_gapwise_metrics.csv",
            "MissForest": "...",
        }
    """

    os.makedirs(out_dir, exist_ok=True)

    dfs = []
    for model, path in metrics_csvs.items():
        df = pd.read_csv(path)
        df["model"] = model
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    plt.figure(figsize=(6.5, 4))
    ax = sns.boxplot(
        data=df_all,
        x="gap_regime",
        y=metric,
        hue="model",
        order=["short", "medium", "long"],
        showfliers=False
    )

    ax.set_xlabel("Gap length regime")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by gap regime {title_suffix}".strip())

    plt.legend(title="Model", frameon=True)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"boxplot_{metric}_by_gap.png")
    plt.savefig(out_path)
    plt.close()

    print(f"✅ Saved boxplot: {out_path}")


# ============================================================
# 2) REGIME × MODEL HEATMAP
# ============================================================
def plot_gapwise_heatmap(
    metrics_csvs,
    metric="RMSE",
    out_dir="plots",
    title_suffix=""
):
    """
    Creates a heatmap:
        rows   → gap regime
        cols   → model
        values → metric (mean)
    """

    os.makedirs(out_dir, exist_ok=True)

    records = []

    for model, path in metrics_csvs.items():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            records.append({
                "model": model,
                "gap_regime": r["gap_regime"],
                metric: r[metric]
            })

    df = pd.DataFrame(records)

    pivot = df.pivot(
        index="gap_regime",
        columns="model",
        values=metric
    ).reindex(["short", "medium", "long"])

    plt.figure(figsize=(6, 3.8))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": metric},
        linewidths=0.5
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Gap regime")
    ax.set_title(f"{metric} heatmap by gap regime {title_suffix}".strip())

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"heatmap_{metric}_gap_model.png")
    plt.savefig(out_path)
    plt.close()

    print(f"✅ Saved heatmap: {out_path}")


# ============================================================
# EXAMPLE DRIVER (OPTIONAL)
# ============================================================
if __name__ == "__main__":

    metrics_csvs = {
        "LightGBM": "results/LightGBM_PM25_gapwise_metrics.csv",
        "MissForest": "results/MissForest_PM25_gapwise_metrics.csv",
        "GATI": "results/GATI_PM25_gapwise_metrics.csv",
    }

    for metric in ["RMSE", "MAE", "R", "NSE"]:
        plot_gapwise_boxplots(
            metrics_csvs,
            metric=metric,
            out_dir="plots",
            title_suffix="(PM₂.₅)"
        )

        plot_gapwise_heatmap(
            metrics_csvs,
            metric=metric,
            out_dir="plots",
            title_suffix="(PM₂.₅)"
        )
