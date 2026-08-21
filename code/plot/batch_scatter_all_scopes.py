#!/usr/bin/env python3
"""Generate AQUISTIL scatter figures for all pollutants and spatial scopes."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_ROOT = Path(
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQUISTIL/Outputs/Imputation_Result"
)
PLOT_SCRIPT = Path(__file__).with_name("combined_regime_scatter_10pct.py")
PREDICTION_FILE = "masked_predictions_by_site.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model", default="AQUISTIL")
    parser.add_argument("--targets", nargs="+", default=None, help="Default: every available pollutant.")
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=["all_regions", "region", "station"],
        default=["all_regions", "region", "station"],
    )
    parser.add_argument("--force", action="store_true", help="Regenerate outputs that already exist.")
    return parser.parse_args()


def slug(value):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_")


def available_files(results_root, model, target):
    root = results_root / "Regional_Pooled_Imputation" / model
    return sorted(root.glob("*/{}/{}".format(target, PREDICTION_FILE)))


def stations_in_file(path):
    stations = set()
    for chunk in pd.read_csv(path, usecols=["Site"], chunksize=500000):
        stations.update(chunk["Site"].dropna().astype(str).str.strip().tolist())
    return sorted(station for station in stations if station)


def complete(output_dir):
    return (output_dir / "regime_event_six_missingness_A4_scatter.png").exists()


def run_plot(args, target, output_dir, scope_label, regions=None, sites=None):
    if not args.force and complete(output_dir):
        print("Skip existing {}".format(output_dir), flush=True)
        return
    command = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--results-root",
        str(args.results_root),
        "--model",
        args.model,
        "--target",
        target,
        "--plot-types",
        "scatter",
        "--scope-label",
        scope_label,
        "--output-dir",
        str(output_dir),
    ]
    if regions:
        command.extend(["--regions"] + list(regions))
    if sites:
        command.extend(["--sites"] + list(sites))
    print("Generate {}".format(output_dir), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    model_root = args.results_root / "Regional_Pooled_Imputation" / args.model
    targets = args.targets or sorted(
        {path.parent.name for path in model_root.glob("*/*/{}".format(PREDICTION_FILE))}
    )
    plot_root = args.results_root / "plots_by_type" / "scatterplot"

    for target in targets:
        files = available_files(args.results_root, args.model, target)
        if not files:
            print("No files for {}; skipping".format(target), flush=True)
            continue
        if "all_regions" in args.scopes:
            run_plot(
                args,
                target,
                plot_root / "all_regions" / target / args.model,
                "All Regions",
            )
        for path in files:
            region = path.parent.parent.name
            region_label = region.replace("_", " ").title()
            if "region" in args.scopes:
                run_plot(
                    args,
                    target,
                    plot_root / "by_region" / target / args.model / region,
                    region_label,
                    regions=[region],
                )
            if "station" in args.scopes:
                for station in stations_in_file(path):
                    run_plot(
                        args,
                        target,
                        plot_root / "by_station" / target / args.model / region / slug(station),
                        station.title(),
                        regions=[region],
                        sites=[station],
                    )


if __name__ == "__main__":
    main()
