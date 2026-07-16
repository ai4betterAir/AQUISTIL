"""
plot_mirror_region_vs_site.py

Mirror-style regional pooled vs individual-site RMSE figure.

Purpose
-------
This script creates the figure style requested:

    Left side  = regional pooled RMSE
    Right side = individual-site RMSE
    Centre line separates regional and site analysis

For each target-input pair, it creates:
1. One mirror-style PNG/PDF for the first requested target-input pair.
2. A full multi-page PDF for all requested inputs for the selected target.
3. CSV files containing the plotted values.

Default setting
---------------
Target: PM2.5
Inputs: PM10, CO, NO2, OZONE, TEMP, NO, WSP, WDR, HUMID, RAIN
Regions: only regions with at least 3 sites
Metric: RMSE

Expected input CSV
------------------
The CSV should be your Random Forest individual-feature result file, such as:

all_random_forest_individual_feature_results.csv

Required columns:
- Target
- Variable
- Scope
- Region
- Site
- Mean_RMSE

Optional columns:
- Mean_MAE
- Mean_R
- Mean_NSE
- Mean_N_Valid

How to run on HPC
-----------------

Example:

export RESULTS_CSV=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection/00RandomForest_Best_Individual_Feature_Selection/individual_results/all_random_forest_individual_feature_results.csv

export OUTPUT_DIR=/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection/01Plot_Mirror_Region_vs_Site

export TARGETS=PM2.5
export INPUTS=PM10,CO,NO2,NOX,NEPH,OZONE,TEMP,NO,WSP,WDR,HUMID,RAIN
export MIN_SITES=3

python3 plot_mirror_region_vs_site.py
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# USER SETTINGS
# =============================================================================

AVAILABLE_TARGETS = [
    "PM2.5",
    "PM10",
    "OZONE",
    "NO",
    "NO2",
    "NOX",
    "CO",
]

AVAILABLE_REGIONS = [
    "Central Coast",
    "Central Tablelands",
    "Lower Hunter",
    "Newcastle Local",
    "Northern Tablelands",
    "Southern Tablelands",
    "Sydney",
    "Sydney East",
    "Sydney North west",
    "Sydney South west",
    "Upper Hunter",
]

SELECTED_TARGET = "PM2.5"
SELECTED_TARGETS = AVAILABLE_TARGETS[:]
SELECTED_REGIONS = AVAILABLE_REGIONS[:]

def _apply_cli_env_overrides() -> None:
    """Allow standalone CLI usage while keeping env-var compatibility."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--target")
    parser.add_argument("--targets")
    parser.add_argument("--regions")
    parser.add_argument("--inputs")
    parser.add_argument("--min-sites", type=int)
    parser.add_argument("--save-full-pdf")
    parser.add_argument("--save-individual-figures")
    parser.add_argument("--save-individual-png")
    parser.add_argument("--save-individual-pdf")
    args, _ = parser.parse_known_args()

    if args.results_csv:
        os.environ["RESULTS_CSV"] = args.results_csv
    if args.output_dir:
        os.environ["OUTPUT_DIR"] = args.output_dir
    if args.target:
        os.environ["TARGETS"] = args.target
    elif args.targets:
        os.environ["TARGETS"] = args.targets
    if args.regions is not None:
        os.environ["TARGET_REGIONS"] = args.regions
    if args.inputs:
        os.environ["INPUTS"] = args.inputs
    if args.min_sites is not None:
        os.environ["MIN_SITES"] = str(args.min_sites)
    if args.save_full_pdf is not None:
        os.environ["SAVE_FULL_PDF"] = str(args.save_full_pdf)
    if args.save_individual_figures is not None:
        os.environ["SAVE_INDIVIDUAL_FIGURES"] = str(args.save_individual_figures)
    if args.save_individual_png is not None:
        os.environ["SAVE_INDIVIDUAL_PNG"] = str(args.save_individual_png)
    if args.save_individual_pdf is not None:
        os.environ["SAVE_INDIVIDUAL_PDF"] = str(args.save_individual_pdf)


_apply_cli_env_overrides()

FEATURE_SELECTION_RUN_MODE = os.getenv("FEATURE_SELECTION_RUN_MODE", "default").strip().lower()
EVENT_RUN = FEATURE_SELECTION_RUN_MODE == "event"
RUN_TYPE_LABEL = "EVENT FEATURE SELECTION RUN" if EVENT_RUN else "DEFAULT FEATURE SELECTION RUN"
FEATURE_SELECTION_ROOT = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Feature_Selection"
)
if EVENT_RUN:
    FEATURE_SELECTION_ROOT = FEATURE_SELECTION_ROOT / "feature_selection_event"

RESULTS_CSV = Path(
    os.getenv(
        "RESULTS_CSV",
        str(FEATURE_SELECTION_ROOT / "00RandomForest_Best_Individual_Feature_Selection" / "individual_results" / "all_random_forest_individual_feature_results.csv"),
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(FEATURE_SELECTION_ROOT / "01Plot_Mirror_Region_vs_Site"),
    )
)

TARGETS = [
    x.strip()
    for x in os.getenv("TARGETS", ",".join(SELECTED_TARGETS)).split(",")
    if x.strip()
]

INPUTS = [
    x.strip()
    for x in os.getenv(
        "INPUTS",
        "PM10,CO,NO2,NOX,NEPH,OZONE,TEMP,NO,WSP,WDR,HUMID,RAIN",
    ).split(",")
    if x.strip()
]

MIN_SITES = int(os.getenv("MIN_SITES", "3"))
TARGET_REGIONS = [
    x.strip()
    for x in os.getenv("TARGET_REGIONS", ",".join(SELECTED_REGIONS)).split(",")
    if x.strip()
]

# Set to 1 if you want one PNG/PDF per target-input pair.
SAVE_INDIVIDUAL_FIGURES = os.getenv("SAVE_INDIVIDUAL_FIGURES", "1").strip().lower() in {
    "1", "true", "yes"
}

# Fine-grained control for per-input outputs
SAVE_INDIVIDUAL_PNG = os.getenv("SAVE_INDIVIDUAL_PNG", "1").strip().lower() in {
    "1", "true", "yes"
}

SAVE_INDIVIDUAL_PDF = os.getenv("SAVE_INDIVIDUAL_PDF", "0").strip().lower() in {
    "1", "true", "yes"
}

# Set to 1 if you want a full PDF for each target.
SAVE_FULL_PDF = os.getenv("SAVE_FULL_PDF", "0").strip().lower() in {
    "1", "true", "yes"
}

# Figure style
REGION_COLOR = "#83C6E2"
SITE_PALETTE = [
    "#4D4D4D",
    "#1F9ACF",
    "#E41A1C",
    "#CC66CC",
    "#80C7E8",
    "#F28E2B",
    "#59A14F",
    "#B07AA1",
    "#9C755F",
    "#FF9DA6",
]

REGION_LABEL_FONT_SIZE = 10 * 1.30
STATION_LABEL_FONT_SIZE = 8.5 * 1.20
VALUE_LABEL_FONT_SIZE = 9 * 1.30
RMSE_FONT_SIZE = 12 * 1.50
RMSE_TICK_FONT_SIZE = 10 * 1.50
SECTION_HEADER_FONT_SIZE = 11 * 1.50
SUBTITLE_FONT_SIZE = 12 * 1.50
TITLE_FONT_SIZE = 14 * 1.50
FOOTNOTE_FONT_SIZE = 9 * 1.30


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_name(value: str) -> str:
    """Create a safe file name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def require_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Stop early if required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + f"\n\nAvailable columns:\n{list(df.columns)}"
        )


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and type-convert the result table."""
    df = df.copy()

    for col in ["Mean_RMSE", "Mean_MAE", "Mean_R", "Mean_NSE", "Mean_N_Valid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Target", "Variable", "Scope", "Region", "Site"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def deduplicate_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deduplicate to one row per target-region-variable for regional rows
    and one row per target-region-site-variable for site rows.

    If duplicate rows exist, the lowest RMSE row is kept.
    """
    region_all = (
        df[df["Scope"] == "Region"]
        .sort_values(["Target", "Region", "Variable", "Mean_RMSE"])
        .groupby(["Target", "Region", "Variable"], as_index=False)
        .first()
    )

    site_all = (
        df[df["Scope"] == "Site"]
        .sort_values(["Target", "Region", "Site", "Variable", "Mean_RMSE"])
        .groupby(["Target", "Region", "Site", "Variable"], as_index=False)
        .first()
    )

    return region_all, site_all


def get_eligible_regions(site_all: pd.DataFrame, target: str, min_sites: int) -> List[str]:
    """Return regions with at least min_sites sites for the selected target."""
    site_counts = (
        site_all[site_all["Target"] == target]
        .groupby("Region")["Site"]
        .nunique()
        .reset_index(name="N_Sites")
    )

    if TARGET_REGIONS:
        wanted = {safe_name(x).lower() for x in TARGET_REGIONS}
        site_counts = site_counts[
            site_counts["Region"].astype(str).map(lambda x: safe_name(x).lower()).isin(wanted)
        ]

    eligible = site_counts.loc[site_counts["N_Sites"] >= min_sites, "Region"].tolist()
    return eligible


def build_plot_table(
    region_all: pd.DataFrame,
    site_all: pd.DataFrame,
    target: str,
    input_variable: str,
    min_sites: int,
) -> pd.DataFrame:
    """
    Build long-format data for the mirror plot.

    Site rows are plotted on the right.
    Regional pooled rows are plotted on the left.
    """
    reg = region_all[
        (region_all["Target"] == target)
        & (region_all["Variable"] == input_variable)
    ].copy()

    sites = site_all[
        (site_all["Target"] == target)
        & (site_all["Variable"] == input_variable)
    ].copy()

    eligible_regions = get_eligible_regions(site_all, target, min_sites)

    reg = reg[reg["Region"].isin(eligible_regions)].copy()
    sites = sites[sites["Region"].isin(eligible_regions)].copy()

    if sites.empty:
        return pd.DataFrame()

    # Region order: use regional RMSE where available, then append remaining regions.
    region_order = []
    if not reg.empty:
        region_order = reg.sort_values("Mean_RMSE")["Region"].tolist()

    for region in sorted(sites["Region"].unique()):
        if region not in region_order:
            region_order.append(region)

    rows = []
    y = 0.0
    group_gap = 1.0

    for region in region_order:
        site_rows = sites[sites["Region"] == region].sort_values("Mean_RMSE")
        reg_row = reg[reg["Region"] == region]

        if site_rows.empty:
            continue

        region_rmse = float(reg_row["Mean_RMSE"].iloc[0]) if not reg_row.empty else np.nan

        group_start = y

        for i, (_, row) in enumerate(site_rows.iterrows()):
            rows.append(
                {
                    "Target": target,
                    "Input": input_variable,
                    "Region": region,
                    "Entity": row["Site"],
                    "Type": "Site",
                    "RMSE": float(row["Mean_RMSE"]),
                    "y": y,
                    "Color": SITE_PALETTE[i % len(SITE_PALETTE)],
                }
            )
            y += 1.0

        group_end = y - 1.0
        group_mid = (group_start + group_end) / 2.0

        rows.append(
            {
                "Target": target,
                "Input": input_variable,
                "Region": region,
                "Entity": "Region pooled",
                "Type": "Region",
                "RMSE": region_rmse,
                "y": group_mid,
                "Color": REGION_COLOR,
            }
        )

        y += group_gap

    return pd.DataFrame(rows)


def make_mirror_plot(
    plot_df: pd.DataFrame,
    target: str,
    input_variable: str,
    save_png: Optional[Path] = None,
    save_pdf: Optional[Path] = None,
):
    """
    Create mirror-style plot.

    Left bars  = regional pooled RMSE
    Right bars = individual-site RMSE
    """
    if plot_df.empty:
        return None

    site_df = plot_df[plot_df["Type"] == "Site"].copy()
    reg_df = plot_df[plot_df["Type"] == "Region"].copy()

    if site_df.empty:
        return None

    max_right = site_df["RMSE"].max()
    max_left = reg_df["RMSE"].max(skipna=True) if not reg_df.empty else np.nan
    max_val = np.nanmax([max_right, max_left])
    left_bar_max = float(max_left) if np.isfinite(max_left) else 0.0
    right_bar_max = float(max_right) if np.isfinite(max_right) else 0.0

    if not np.isfinite(max_val):
        return None

    n_site_rows = len(site_df)

    fig_h = max(7.5, 0.38 * n_site_rows + 2.8)
    fig, ax = plt.subplots(figsize=(16.5, fig_h))

    # Left: regional pooled bars
    for _, row in reg_df.iterrows():
        if np.isfinite(row["RMSE"]):
            ax.barh(
                row["y"],
                -row["RMSE"],
                height=0.72,
                color=REGION_COLOR,
                edgecolor="none",
                alpha=0.95,
            )

    # Right: individual site bars
    for _, row in site_df.iterrows():
        ax.barh(
            row["y"],
            row["RMSE"],
            height=0.72,
            color=row["Color"],
            edgecolor="none",
            alpha=0.98,
        )

    # Centre separator
    ax.axvline(0, color="black", linewidth=1.1)

    # Region labels and separators
    region_order = list(dict.fromkeys(plot_df["Region"].tolist()))

    for region in region_order:
        sub = site_df[site_df["Region"] == region]

        if sub.empty:
            continue

        y_mid = (sub["y"].min() + sub["y"].max()) / 2.0

        ax.text(
            0.02,
            y_mid,
            region,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=REGION_LABEL_FONT_SIZE,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )

        ax.axhline(
            sub["y"].min() - 0.52,
            color="0.82",
            linewidth=0.8,
        )

    # Site labels
    pad = max_val * 0.012

    for _, row in site_df.iterrows():
        ax.text(
            row["RMSE"] + pad,
            row["y"],
            row["Entity"],
            va="center",
            ha="left",
            fontsize=STATION_LABEL_FONT_SIZE,
        )

    # Axis limits
    left_limit = -(left_bar_max * 1.10) if left_bar_max > 0 else -(max_val * 0.12)
    right_limit = right_bar_max * 1.12
    ax.set_xlim(left_limit, right_limit)
    ax.set_ylim(site_df["y"].max() + 0.9, -0.6)

    # Absolute tick labels on both sides
    tick_max = int(np.ceil(max(abs(left_limit), abs(right_limit))))
    ticks = np.arange(-tick_max, tick_max + 1, 1)

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(t)) for t in ticks])
    ax.tick_params(axis="x", labelsize=RMSE_TICK_FONT_SIZE)

    ax.set_yticks([])
    ax.set_xlabel("RMSE", fontsize=RMSE_FONT_SIZE, fontweight="bold")

    # Header labels in axes/figure coordinates so they do not drift into the plot.
    fig.suptitle(
        f"Target: {target}  |  Input: {input_variable}",
        fontsize=SUBTITLE_FONT_SIZE,
        fontweight="bold",
        y=0.955,
    )

    ax.text(
        0.25,
        1.002,
        "Regional pooled",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=SECTION_HEADER_FONT_SIZE,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    ax.text(
        0.75,
        1.002,
        "Individual sites",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=SECTION_HEADER_FONT_SIZE,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    # Clean frame
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.text(
        0.01,
        0.01,
        "Metric: RMSE; lower is better. Left bars show regional pooled model RMSE. Right bars show individual-site model RMSE.",
        fontsize=FOOTNOTE_FONT_SIZE,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.935])

    if save_png is not None:
        fig.savefig(save_png, dpi=300, bbox_inches="tight")

    if save_pdf is not None:
        fig.savefig(save_pdf, bbox_inches="tight")

    return fig


def make_cover_page(pdf, target: str, input_order: List[str], min_sites: int):
    """Add cover page to full PDF."""
    fig_cover = plt.figure(figsize=(11.69, 8.27))

    lines = [
        "Mirror-style region vs site RMSE figures",
        "",
        f"Target: {target}",
        "",
        "Inputs included:",
    ]

    lines += [f"  - {v}" for v in input_order]

    lines += [
        "",
        "Layout:",
        "  Left side  = regional pooled RMSE",
        "  Right side = individual-site RMSE",
        f"  Regions shown only when at least {min_sites} sites are available",
        "",
        "Metric: RMSE; lower is better.",
    ]

    fig_cover.text(0.06, 0.94, "\n".join(lines), va="top", fontsize=14)
    pdf.savefig(fig_cover, bbox_inches="tight")
    plt.close(fig_cover)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== RUN TYPE: %s ===" % RUN_TYPE_LABEL, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find input CSV:\n{RESULTS_CSV}\n\n"
            "Set the path like this:\n"
            "export RESULTS_CSV=/path/to/all_random_forest_individual_feature_results.csv"
        )

    print(f"Reading: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)

    required_cols = ["Target", "Variable", "Scope", "Region", "Site", "Mean_RMSE"]
    require_columns(df, required_cols)

    df = clean_results(df)
    region_all, site_all = deduplicate_results(df)

    # Save a simple summary of eligible regions
    eligible_summary_rows = []

    for target in TARGETS:
        eligible_regions = get_eligible_regions(site_all, target, MIN_SITES)
        for region in eligible_regions:
            n_sites = site_all[
                (site_all["Target"] == target)
                & (site_all["Region"] == region)
            ]["Site"].nunique()

            eligible_summary_rows.append(
                {
                    "Target": target,
                    "Region": region,
                    "N_Sites": n_sites,
                }
            )

    eligible_summary = pd.DataFrame(eligible_summary_rows)
    eligible_summary.to_csv(OUTPUT_DIR / "eligible_regions_summary.csv", index=False)

    # Build target/input index
    index_rows = []

    for target in TARGETS:
        available_inputs = (
            site_all.loc[site_all["Target"] == target, "Variable"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        input_order = [v for v in INPUTS if v in available_inputs]

        for v in sorted(available_inputs):
            if v not in input_order:
                input_order.append(v)

        # Full PDF per target
        full_pdf_path = OUTPUT_DIR / f"Mirror_{safe_name(target)}_all_inputs_region_vs_sites_full.pdf"

        if SAVE_FULL_PDF:
            with PdfPages(full_pdf_path) as pdf:
                make_cover_page(pdf, target, input_order, MIN_SITES)

                for input_var in input_order:
                    plot_df = build_plot_table(
                        region_all=region_all,
                        site_all=site_all,
                        target=target,
                        input_variable=input_var,
                        min_sites=MIN_SITES,
                    )

                    if plot_df.empty:
                        continue

                    csv_path = OUTPUT_DIR / f"Mirror_{safe_name(target)}_input_{safe_name(input_var)}_data.csv"
                    plot_df.to_csv(csv_path, index=False)

                    fig = make_mirror_plot(
                        plot_df=plot_df,
                        target=target,
                        input_variable=input_var,
                    )

                    if fig is not None:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

                    index_rows.append(
                        {
                            "Target": target,
                            "Input": input_var,
                            "Data_CSV": str(csv_path),
                            "Included_in_full_PDF": str(full_pdf_path),
                        }
                    )

        # Individual PNG/PDF per target-input
        if SAVE_INDIVIDUAL_FIGURES:
            for input_var in input_order:
                plot_df = build_plot_table(
                    region_all=region_all,
                    site_all=site_all,
                    target=target,
                    input_variable=input_var,
                    min_sites=MIN_SITES,
                )

                if plot_df.empty:
                    continue

                png_path = (
                    OUTPUT_DIR / f"Mirror_{safe_name(target)}_target_{safe_name(input_var)}_input_region_vs_sites.png"
                    if SAVE_INDIVIDUAL_PNG
                    else None
                )
                pdf_path = (
                    OUTPUT_DIR / f"Mirror_{safe_name(target)}_target_{safe_name(input_var)}_input_region_vs_sites.pdf"
                    if SAVE_INDIVIDUAL_PDF
                    else None
                )
                csv_path = OUTPUT_DIR / f"Mirror_{safe_name(target)}_target_{safe_name(input_var)}_input_region_vs_sites_data.csv"

                plot_df.to_csv(csv_path, index=False)

                fig = make_mirror_plot(
                    plot_df=plot_df,
                    target=target,
                    input_variable=input_var,
                    save_png=png_path,
                    save_pdf=pdf_path,
                )

                if fig is not None:
                    plt.close(fig)

                index_rows.append(
                    {
                        "Target": target,
                        "Input": input_var,
                        "PNG": str(png_path) if png_path is not None else "",
                        "PDF": str(pdf_path) if pdf_path is not None else "",
                        "Data_CSV": str(csv_path),
                    }
                )

    index_df = pd.DataFrame(index_rows)
    index_path = OUTPUT_DIR / "mirror_region_vs_site_output_index.csv"
    index_df.to_csv(index_path, index=False)

    print("\nDone.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Index file: {index_path}")
    print(f"Eligible regions file: {OUTPUT_DIR / 'eligible_regions_summary.csv'}")

    if SAVE_FULL_PDF:
        print("\nFull PDFs:")
        for target in TARGETS:
            print(f"  {OUTPUT_DIR / f'Mirror_{safe_name(target)}_all_inputs_region_vs_sites_full.pdf'}")


if __name__ == "__main__":
    main()
    print("=== JOB FINISHED: Stage 1 region-versus-site plots ===", flush=True)
