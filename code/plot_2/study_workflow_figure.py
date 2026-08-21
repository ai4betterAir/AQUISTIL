#!/usr/bin/env python3
"""Create Figure 1: study workflow for the AQUISTIL publication pipeline."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_PROJECT_ROOT = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL"
)
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "Outputs/Publication_Figures"


def add_box(ax, x, y, w, h, title, body, facecolor, edgecolor="#334155"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.25,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.07,
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        x + w / 2,
        y + h - 0.18,
        body,
        ha="center",
        va="top",
        fontsize=8.6,
        color="#334155",
        linespacing=1.22,
    )


def add_arrow(ax, start, end, color="#475569"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.35,
        color=color,
        shrinkA=5,
        shrinkB=5,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def create_workflow(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Figure 1. AQUISTIL Study Workflow",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.925,
        "From API observations to controlled missingness experiments, model benchmarking, and publication outputs",
        ha="center",
        va="top",
        fontsize=10,
        color="#475569",
    )

    boxes = [
        (
            0.05,
            0.72,
            0.19,
            0.14,
            "1. API Data",
            "AquisNet/API hourly records\npollutants + meteorology\nsite metadata",
            "#DBEAFE",
        ),
        (
            0.30,
            0.72,
            0.19,
            0.14,
            "2. Preprocessing",
            "timestamp harmonisation\nsite-region mapping\nnumeric quality checks",
            "#E0F2FE",
        ),
        (
            0.55,
            0.72,
            0.19,
            0.14,
            "3. Missingness Audit",
            "observed missingness by\nregion, site, pollutant\npublication descriptive plots",
            "#ECFDF5",
        ),
        (
            0.05,
            0.48,
            0.19,
            0.14,
            "4. Feature Engineering",
            "local predictors\ncalendar features\nspatial IDW proxy",
            "#F0FDFA",
        ),
        (
            0.30,
            0.48,
            0.19,
            0.14,
            "5. Feature Selection",
            "Stage-3 progressive\nregion-target feature contract\nsame inputs for benchmarks",
            "#FEF3C7",
        ),
        (
            0.55,
            0.48,
            0.19,
            0.14,
            "6. Missingness Regimes",
            "random + gap regimes\nevent daily maxima\n10, 20, 30, 50%",
            "#FFEDD5",
        ),
        (
            0.80,
            0.48,
            0.15,
            0.14,
            "7. Models",
            "AQUISTIL\nLightGBM\nMICE baselines",
            "#FCE7F3",
        ),
        (
            0.18,
            0.22,
            0.21,
            0.15,
            "8. AQUISTIL Imputation",
            "LightGBM backbone\nhistory + event correction\nq10/q90 uncertainty\nposterior-style sampling",
            "#EDE9FE",
        ),
        (
            0.47,
            0.22,
            0.21,
            0.15,
            "9. Evaluation",
            "RMSE, MAE, RMAE\nR, R2, NSE, WI\nbias and event-peak recovery",
            "#F1F5F9",
        ),
        (
            0.76,
            0.22,
            0.19,
            0.15,
            "10. Outputs",
            "main figures\nappendix CSV tables\nall models, regimes, levels",
            "#DCFCE7",
        ),
    ]

    for box in boxes:
        add_box(ax, *box)

    arrows = [
        ((0.24, 0.79), (0.30, 0.79)),
        ((0.49, 0.79), (0.55, 0.79)),
        ((0.645, 0.72), (0.145, 0.62)),
        ((0.24, 0.55), (0.30, 0.55)),
        ((0.49, 0.55), (0.55, 0.55)),
        ((0.74, 0.55), (0.80, 0.55)),
        ((0.875, 0.48), (0.285, 0.37)),
        ((0.39, 0.295), (0.47, 0.295)),
        ((0.68, 0.295), (0.76, 0.295)),
    ]
    for start, end in arrows:
        add_arrow(ax, start, end)

    ax.text(
        0.5,
        0.09,
        "Controlled evaluation unit: same region/site/target/regime/missingness mask is used for every benchmark model.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    png_path = output_dir / "fig1_study_workflow.png"
    pdf_path = output_dir / "fig1_study_workflow.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create Figure 1 study workflow diagram.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    png_path, pdf_path = create_workflow(args.output_dir)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
