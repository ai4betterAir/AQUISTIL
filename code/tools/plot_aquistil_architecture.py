#!/usr/bin/env python3
"""Render the frozen AQUISTIL two-expert architecture diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def _box(axis, xy, width, height, title, lines, facecolor, edgecolor):
    patch = Rectangle(xy, width, height, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.4)
    axis.add_patch(patch)
    x, y = xy
    title_y = y + height - min(0.035, height * 0.28)
    axis.text(x + width / 2, title_y, title, ha="center", va="top", weight="bold", fontsize=11)
    if lines:
        axis.text(
            x + 0.04,
            title_y - 0.075,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=8.2,
            linespacing=1.28,
        )


def _arrow(axis, start, end, label=""):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#343A40"))
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, label, ha="center", fontsize=8)


def render(output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 7))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _box(axis, (0.38, 0.86), 0.24, 0.07, "AQUISTIL", [], "#EEF3F8", "#34495E")
    axis.text(0.50, 0.825, "Frozen missingness-topology-aware framework", ha="center", fontsize=9)
    _box(axis, (0.34, 0.65), 0.32, 0.11, "Mask topology router", [], "#F5F5F2", "#666B73")
    axis.text(0.50, 0.688, "Actual per-site missing-run length | Global threshold: 2 hours", ha="center", fontsize=8.2)
    _box(axis, (0.05, 0.27), 0.40, 0.27, "History-rich expert", ["Target lags and rolling statistics", "Temporal derivatives", "Event specialist/refinement", "Selected exogenous and spatial predictors", "Calendar and site/region context", "Adaptive guardrails and uncertainty"], "#E7F0FA", "#276FBF")
    _box(axis, (0.55, 0.27), 0.40, 0.27, "Contiguous-gap expert", ["No target-history predictors", "Selected exogenous predictors", "Co-pollutants and meteorology", "Spatial and adaptive spatial information", "Calendar and site/region context", "Gap-position and boundary context"], "#FFF3D6", "#D49A00")
    _box(axis, (0.31, 0.08), 0.38, 0.09, "Reconstructed concentration", [], "#EEF3F8", "#34495E")
    axis.text(0.50, 0.035, "One AQUISTIL output; no LightGBM fallback", ha="center", fontsize=9)
    _arrow(axis, (0.50, 0.82), (0.50, 0.76))
    _arrow(axis, (0.42, 0.65), (0.25, 0.54), "isolated / event")
    _arrow(axis, (0.58, 0.65), (0.75, 0.54), "contiguous gap")
    _arrow(axis, (0.25, 0.27), (0.43, 0.17))
    _arrow(axis, (0.75, 0.27), (0.57, 0.17))
    axis.set_title("Frozen AQUISTIL architecture", fontsize=15, pad=12)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "05_aquistil_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "05_aquistil_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("Outputs/Final_Frozen/Paper_Figures"))
    args = parser.parse_args()
    render(args.output_dir)
    print(f"Saved AQUISTIL architecture diagram to {args.output_dir}")


if __name__ == "__main__":
    main()
