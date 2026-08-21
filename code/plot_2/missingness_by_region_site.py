#!/usr/bin/env python3
"""Plot observed pollutant missingness by region.

Default output:
    Outputs/Publication_Figures/fig_missingness_percent_by_region_pollutant.png
    Outputs/Publication_Figures/missingness_percent_by_region_pollutant.csv
"""

import argparse
import json
import re
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL"
)
DEFAULT_INPUTS_DIR = (
    DEFAULT_PROJECT_ROOT / "Outputs/Imputation_Result/Inputs_PerSite"
)
DEFAULT_MAPPING_PATH = (
    DEFAULT_PROJECT_ROOT / "Outputs/Imputation_Result/region_site_mapping.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "Outputs/Publication_Figures"
DEFAULT_POLLUTANTS = ["CO", "NO", "NO2", "NOX", "OZONE", "PM10", "PM2.5"]


def canon(value: object) -> str:
    """Normalize site names so file stems can be matched to mapping names."""
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()


def target_token(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", target.replace("2.5", "25"))


def load_site_region_mapping(mapping_path: Path) -> dict:
    with mapping_path.open() as handle:
        mapping = json.load(handle)

    site_to_region = {}
    for region, sites in mapping.get("region_to_sites", {}).items():
        for site in sites:
            site_to_region[canon(site)] = region
    return site_to_region


def compute_site_missingness(
    inputs_dir: Path,
    mapping_path: Path,
    targets: list,
    include_unknown: bool = False,
) -> pd.DataFrame:
    site_to_region = load_site_region_mapping(mapping_path)
    rows = []

    for path in sorted(inputs_dir.glob("*.csv")):
        site = path.stem
        region = site_to_region.get(canon(site), "Unknown")

        try:
            wanted = set(targets)
            frame = pd.read_csv(path, usecols=lambda col: col in wanted)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        for target in targets:
            if target not in frame.columns:
                continue
            values = pd.to_numeric(frame[target], errors="coerce")
            n_total = int(len(values))
            n_missing = int(values.isna().sum())
            missingness = 100.0 * n_missing / n_total if n_total else np.nan
            rows.append(
                {
                    "Region": region,
                    "Site": site,
                    "Pollutant": target,
                    "N_Total": n_total,
                    "N_Missing": n_missing,
                    "Missingness_Percent": missingness,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    if not include_unknown:
        summary = summary.loc[summary["Region"].ne("Unknown")].copy()
    return summary.sort_values(
        ["Region", "Pollutant", "Missingness_Percent", "Site"], kind="stable"
    ).reset_index(drop=True)


def aggregate_region_pollutants(site_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        site_summary.groupby(["Region", "Pollutant"], as_index=False)
        .agg(
            Site_Count=("Site", "nunique"),
            N_Total=("N_Total", "sum"),
            N_Missing=("N_Missing", "sum"),
            Median_Site_Missingness_Percent=("Missingness_Percent", "median"),
        )
    )
    grouped["Missingness_Percent"] = (
        100.0 * grouped["N_Missing"] / grouped["N_Total"].where(grouped["N_Total"] != 0)
    )
    return grouped.sort_values(
        ["Region", "Missingness_Percent", "Pollutant"], kind="stable"
    ).reset_index(drop=True)


def plot_missingness(summary: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    if summary.empty:
        raise ValueError("No missingness rows available to plot")

    output_dir.mkdir(parents=True, exist_ok=True)

    region_order = (
        summary.groupby("Region")["Missingness_Percent"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    plot_data = [
        summary.loc[summary["Region"].eq(region), "Missingness_Percent"].dropna().to_numpy()
        for region in region_order
    ]

    fig_width = max(10.0, 0.58 * len(region_order))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
    ax.boxplot(
        plot_data,
        positions=np.arange(len(region_order)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        boxprops={"facecolor": "#D8E2EA", "edgecolor": "#4A5568", "linewidth": 1.1},
        whiskerprops={"color": "#4A5568", "linewidth": 1.0},
        capprops={"color": "#4A5568", "linewidth": 1.0},
    )

    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    pollutant_to_color = {
        pollutant: palette[index % len(palette)]
        for index, pollutant in enumerate(sorted(summary["Pollutant"].unique()))
    }
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8", "p"]
    pollutant_to_marker = {
        pollutant: markers[index % len(markers)]
        for index, pollutant in enumerate(sorted(summary["Pollutant"].unique()))
    }

    for index, region in enumerate(region_order):
        part = summary.loc[summary["Region"].eq(region)].reset_index(drop=True)
        for _, row in part.iterrows():
            ax.scatter(
                index,
                row["Missingness_Percent"],
                s=42,
                c=[pollutant_to_color[row["Pollutant"]]],
                marker=pollutant_to_marker[row["Pollutant"]],
                alpha=0.86,
                edgecolors="white",
                linewidths=0.55,
                zorder=3,
            )

    ax.set_title(
        "Observed Missingness by Region and Pollutant",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    ax.set_ylabel("Missingness (%)", fontsize=11)
    ax.set_xlabel("Region", fontsize=11)
    ax.set_xticks(np.arange(len(region_order)))
    ax.set_xticklabels(region_order, rotation=45, ha="right")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = float(np.nanmax(summary["Missingness_Percent"]))
    ax.set_ylim(0, ymax * 1.14 + 0.5)
    for index, region in enumerate(region_order):
        n_sites = int(summary.loc[summary["Region"].eq(region), "Site_Count"].max())
        ax.text(
            index,
            ymax * 1.08 + 0.25,
            f"sites={n_sites}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#4B5563",
        )

    fig.tight_layout()
    handles = []
    for pollutant in sorted(summary["Pollutant"].unique()):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=pollutant_to_marker[pollutant],
                color="none",
                markerfacecolor=pollutant_to_color[pollutant],
                markeredgecolor="white",
                markersize=7,
                label=pollutant,
            )
        )
    ax.legend(
        handles=handles,
        title="Pollutant",
        frameon=False,
        ncol=min(len(handles), 7),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
    )

    png_path = output_dir / "fig_missingness_percent_by_region_pollutant.png"
    pdf_path = output_dir / "fig_missingness_percent_by_region_pollutant.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-style boxplot of observed missingness by region/site."
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_POLLUTANTS),
        help="Comma-separated pollutant columns to assess.",
    )
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include sites that are not found in the region-site mapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    site_summary = compute_site_missingness(
        inputs_dir=args.inputs_dir,
        mapping_path=args.mapping,
        targets=targets,
        include_unknown=args.include_unknown,
    )
    summary = aggregate_region_pollutants(site_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    site_csv_path = args.output_dir / "missingness_percent_by_region_site_pollutant.csv"
    site_summary.to_csv(site_csv_path, index=False)
    csv_path = args.output_dir / "missingness_percent_by_region_pollutant.csv"
    summary.to_csv(csv_path, index=False)
    png_path, pdf_path = plot_missingness(summary, args.output_dir)

    print(f"Region-pollutant rows: {len(summary)}")
    print(f"Site-pollutant CSV: {site_csv_path}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print("\nTop regions by median missingness:")
    print(
        summary.groupby("Region")["Missingness_Percent"]
        .agg(["count", "median", "min", "max"])
        .sort_values("median", ascending=False)
        .round(2)
        .head(10)
        .to_string()
    )


if __name__ == "__main__":
    main()
