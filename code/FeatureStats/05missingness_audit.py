#!/usr/bin/env python3
"""Missingness audit for publication-ready data characterization.

Outputs summary CSVs and figures for:
- missingness by region/site/pollutant
- region x pollutant missingness heatmap
- site x pollutant missingness heatmap
- temporal missingness patterns by hour and month
- real observed gap-length distribution
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL")
DEFAULT_INPUTS_DIR = PROJECT_ROOT / "Outputs/Imputation_Result/Inputs_PerSite"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "Outputs/Imputation_Result/region_site_mapping.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Outputs/Publication_Figures/Missingness_Audit"
DEFAULT_POLLUTANTS = ["CO", "NO", "NO2", "NOX", "OZONE", "PM10", "PM2.5"]


def canon(value):
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()


def safe_token(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def load_site_region_mapping(mapping_path):
    with mapping_path.open() as handle:
        mapping = json.load(handle)
    site_to_region = {}
    for region, sites in mapping.get("region_to_sites", {}).items():
        for site in sites:
            site_to_region[canon(site)] = region
    return site_to_region


def gap_rows_for_series(region, site, pollutant, timestamps, missing):
    rows = []
    missing = np.asarray(missing, dtype=bool)
    if not missing.any():
        return rows
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
    for start, end in zip(starts, ends):
        rows.append(
            {
                "Region": region,
                "Site": site,
                "Pollutant": pollutant,
                "Gap_Length_Hours": int(end - start + 1),
                "Gap_Start": timestamps.iloc[start],
                "Gap_End": timestamps.iloc[end],
            }
        )
    return rows


def load_missingness_records(inputs_dir, mapping_path, pollutants, include_unknown=False):
    site_to_region = load_site_region_mapping(mapping_path)
    records = []
    gap_records = []

    for path in sorted(inputs_dir.glob("*.csv")):
        site = path.stem
        region = site_to_region.get(canon(site), "Unknown")
        if region == "Unknown" and not include_unknown:
            continue

        wanted = set(["DateTime"] + list(pollutants))
        try:
            frame = pd.read_csv(path, usecols=lambda col: col in wanted)
        except Exception as exc:
            print("Skipping %s: %s" % (path, exc))
            continue

        if "DateTime" not in frame.columns:
            continue
        frame["DateTime"] = pd.to_datetime(frame["DateTime"], errors="coerce")
        frame = frame.dropna(subset=["DateTime"]).sort_values("DateTime")
        if frame.empty:
            continue

        for pollutant in pollutants:
            if pollutant not in frame.columns:
                continue
            values = pd.to_numeric(frame[pollutant], errors="coerce")
            missing = values.isna()
            gap_records.extend(
                gap_rows_for_series(region, site, pollutant, frame["DateTime"], missing.to_numpy())
            )
            temp = pd.DataFrame(
                {
                    "DateTime": frame["DateTime"],
                    "Region": region,
                    "Site": site,
                    "Pollutant": pollutant,
                    "Is_Missing": missing.astype(bool),
                }
            )
            temp["Hour"] = temp["DateTime"].dt.hour
            temp["Month"] = temp["DateTime"].dt.month
            temp["Date"] = temp["DateTime"].dt.date
            records.append(temp)

    if not records:
        return (
            pd.DataFrame(
            columns=["DateTime", "Region", "Site", "Pollutant", "Is_Missing", "Hour", "Month", "Date"]
            ),
            pd.DataFrame(),
        )
    return pd.concat(records, ignore_index=True), pd.DataFrame(gap_records)


def build_summary_tables(long_df, output_dir):
    site_pollutant = (
        long_df.groupby(["Region", "Site", "Pollutant"], as_index=False)
        .agg(N_Total=("Is_Missing", "size"), N_Missing=("Is_Missing", "sum"))
    )
    site_pollutant["Missingness_Percent"] = (
        100.0 * site_pollutant["N_Missing"] / site_pollutant["N_Total"].where(site_pollutant["N_Total"] != 0)
    )

    region_pollutant = (
        site_pollutant.groupby(["Region", "Pollutant"], as_index=False)
        .agg(
            Site_Count=("Site", "nunique"),
            N_Total=("N_Total", "sum"),
            N_Missing=("N_Missing", "sum"),
            Median_Site_Missingness_Percent=("Missingness_Percent", "median"),
        )
    )
    region_pollutant["Missingness_Percent"] = (
        100.0 * region_pollutant["N_Missing"] / region_pollutant["N_Total"].where(region_pollutant["N_Total"] != 0)
    )

    pollutant = (
        site_pollutant.groupby(["Pollutant"], as_index=False)
        .agg(Site_Count=("Site", "nunique"), N_Total=("N_Total", "sum"), N_Missing=("N_Missing", "sum"))
    )
    pollutant["Missingness_Percent"] = (
        100.0 * pollutant["N_Missing"] / pollutant["N_Total"].where(pollutant["N_Total"] != 0)
    )

    region = (
        site_pollutant.groupby(["Region"], as_index=False)
        .agg(Site_Count=("Site", "nunique"), N_Total=("N_Total", "sum"), N_Missing=("N_Missing", "sum"))
    )
    region["Missingness_Percent"] = (
        100.0 * region["N_Missing"] / region["N_Total"].where(region["N_Total"] != 0)
    )

    site_pollutant.to_csv(output_dir / "missingness_by_site_pollutant.csv", index=False)
    region_pollutant.to_csv(output_dir / "missingness_by_region_pollutant.csv", index=False)
    pollutant.to_csv(output_dir / "missingness_by_pollutant.csv", index=False)
    region.to_csv(output_dir / "missingness_by_region.csv", index=False)
    return site_pollutant, region_pollutant, pollutant, region


def plot_region_pollutant_box(region_pollutant, output_dir):
    region_order = (
        region_pollutant.groupby("Region")["Missingness_Percent"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [
        region_pollutant.loc[region_pollutant.Region.eq(region), "Missingness_Percent"].dropna().to_numpy()
        for region in region_order
    ]

    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(region_order)), 6.5))
    ax.boxplot(
        data,
        positions=np.arange(len(region_order)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        boxprops={"facecolor": "#D8E2EA", "edgecolor": "#4A5568", "linewidth": 1.1},
        whiskerprops={"color": "#4A5568", "linewidth": 1.0},
        capprops={"color": "#4A5568", "linewidth": 1.0},
    )

    pollutants = sorted(region_pollutant["Pollutant"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(pollutants), 3)))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
    color = {p: palette[i % len(palette)] for i, p in enumerate(pollutants)}
    marker = {p: markers[i % len(markers)] for i, p in enumerate(pollutants)}

    for index, region_name in enumerate(region_order):
        part = region_pollutant.loc[region_pollutant.Region.eq(region_name)]
        for _, row in part.iterrows():
            ax.scatter(
                index,
                row["Missingness_Percent"],
                s=46,
                marker=marker[row["Pollutant"]],
                c=[color[row["Pollutant"]]],
                edgecolors="white",
                linewidths=0.55,
                zorder=3,
            )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker[p],
            color="none",
            markerfacecolor=color[p],
            markeredgecolor="white",
            markersize=7,
            label=p,
        )
        for p in pollutants
    ]
    ax.legend(handles=handles, title="Pollutant", frameon=False, ncol=min(len(handles), 7), loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Observed Missingness by Region and Pollutant", fontsize=14, fontweight="bold")
    ax.set_ylabel("Missingness (%)")
    ax.set_xlabel("Region")
    ax.set_xticks(np.arange(len(region_order)))
    ax.set_xticklabels(region_order, rotation=45, ha="right")
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_missingness_region_pollutant_box.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_missingness_region_pollutant_box.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(table, index_col, column_col, value_col, title, output_path):
    pivot = table.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(pivot.columns)), max(5, 0.32 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGn", vmin=0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Missingness (%)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, "%.1f" % val, ha="center", va="center", fontsize=7, color="#111827")
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_temporal_patterns(long_df, output_dir):
    hourly = (
        long_df.groupby(["Pollutant", "Hour"], as_index=False)
        .agg(N_Total=("Is_Missing", "size"), N_Missing=("Is_Missing", "sum"))
    )
    hourly["Missingness_Percent"] = 100.0 * hourly["N_Missing"] / hourly["N_Total"].where(hourly["N_Total"] != 0)
    monthly = (
        long_df.groupby(["Pollutant", "Month"], as_index=False)
        .agg(N_Total=("Is_Missing", "size"), N_Missing=("Is_Missing", "sum"))
    )
    monthly["Missingness_Percent"] = 100.0 * monthly["N_Missing"] / monthly["N_Total"].where(monthly["N_Total"] != 0)
    hourly.to_csv(output_dir / "missingness_by_hour_pollutant.csv", index=False)
    monthly.to_csv(output_dir / "missingness_by_month_pollutant.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=False)
    pollutants = sorted(long_df["Pollutant"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(pollutants), 3)))
    color = {p: palette[i % len(palette)] for i, p in enumerate(pollutants)}

    for pollutant in pollutants:
        h = hourly.loc[hourly.Pollutant.eq(pollutant)]
        axes[0].plot(h["Hour"], h["Missingness_Percent"], marker="o", linewidth=1.5, markersize=3, color=color[pollutant], label=pollutant)
        m = monthly.loc[monthly.Pollutant.eq(pollutant)]
        axes[1].plot(m["Month"], m["Missingness_Percent"], marker="o", linewidth=1.5, markersize=3, color=color[pollutant], label=pollutant)

    axes[0].set_title("Hourly Missingness Pattern", fontweight="bold")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Missingness (%)")
    axes[0].set_xticks(range(0, 24, 3))
    axes[1].set_title("Monthly Missingness Pattern", fontweight="bold")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Missingness (%)")
    axes[1].set_xticks(range(1, 13))
    for ax in axes:
        ax.grid(axis="y", color="#E5E7EB")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, title="Pollutant", bbox_to_anchor=(1.03, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_temporal_missingness_hour_month.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_temporal_missingness_hour_month.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gap_lengths(gaps, output_dir):
    gaps.to_csv(output_dir / "missingness_gap_lengths.csv", index=False)
    if gaps.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bins = [1, 2, 4, 8, 12, 24, 48, 72, 168, 336, max(337, int(gaps["Gap_Length_Hours"].max()) + 1)]
    ax.hist(gaps["Gap_Length_Hours"], bins=bins, color="#2F855A", edgecolor="white", alpha=0.85)
    ax.set_xscale("log")
    ax.set_title("Observed Missing-Data Gap Length Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Contiguous missing gap length (hours, log scale)")
    ax.set_ylabel("Number of gaps")
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_gap_length_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_gap_length_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run missingness audit analyses for publication.")
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pollutants", default=",".join(DEFAULT_POLLUTANTS))
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument(
        "--save-long-records",
        action="store_true",
        help="Write the expanded timestamp x site x pollutant missingness table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pollutants = [item.strip() for item in args.pollutants.split(",") if item.strip()]

    long_df, gaps = load_missingness_records(args.inputs_dir, args.mapping, pollutants, args.include_unknown)
    if long_df.empty:
        raise RuntimeError("No missingness records were loaded")

    if args.save_long_records:
        long_df.to_csv(args.output_dir / "missingness_long_records.csv", index=False)
    site_pollutant, region_pollutant, pollutant, region = build_summary_tables(long_df, args.output_dir)

    plot_region_pollutant_box(region_pollutant, args.output_dir)
    plot_heatmap(
        region_pollutant,
        "Region",
        "Pollutant",
        "Missingness_Percent",
        "Region x Pollutant Missingness",
        args.output_dir / "fig_heatmap_region_pollutant",
    )
    plot_heatmap(
        site_pollutant,
        "Site",
        "Pollutant",
        "Missingness_Percent",
        "Site x Pollutant Missingness",
        args.output_dir / "fig_heatmap_site_pollutant",
    )
    plot_temporal_patterns(long_df, args.output_dir)
    plot_gap_lengths(gaps, args.output_dir)

    print("Missingness audit complete")
    print("Output directory: %s" % args.output_dir)
    print("Records: %d" % len(long_df))
    print("Region-pollutant rows: %d" % len(region_pollutant))
    print("Site-pollutant rows: %d" % len(site_pollutant))
    print("Observed gaps: %d" % len(gaps))
    print("\nTop pollutants by missingness:")
    print(pollutant.sort_values("Missingness_Percent", ascending=False).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
