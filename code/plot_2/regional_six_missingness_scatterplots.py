#!/usr/bin/env python3
"""Generate six-panel observed-vs-imputed scatter plots for every region.

Each figure represents one model, pollutant, region, and masking regime.  Its
six panels show the available missingness levels.  The plotting calculations
and visual style are shared with ``scatter_all_scopes.py``.
"""

import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

from scatter_all_scopes import (
    DEFAULT_RESULTS_ROOT,
    PREDICTION_FILE,
    discover_targets,
    load_target,
    render_scope,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--models", nargs="+", default=None, help="Default: all available models.")
    parser.add_argument("--targets", nargs="+", default=None, help="Default: all available pollutants.")
    parser.add_argument("--axis-percentile", type=float, default=99.9)
    parser.add_argument("--max-points-per-panel", type=int, default=5000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def discover_models(results_root):
    root = results_root / "Regional_Pooled_Imputation"
    return sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and next(path.glob("*/*/{}".format(PREDICTION_FILE)), None) is not None
    )


def main():
    args = parse_args()
    if not 90 <= args.axis_percentile <= 100:
        raise ValueError("--axis-percentile must be between 90 and 100")
    models = args.models or discover_models(args.results_root)
    output_root = args.results_root / "plots_by_type" / "scatterplot" / "regional_six_missingness"
    total = 0

    for model in models:
        targets = args.targets or discover_targets(args.results_root, model)
        render_args = SimpleNamespace(
            axis_percentile=args.axis_percentile,
            max_points_per_panel=args.max_points_per_panel,
            dpi=args.dpi,
            seed=args.seed,
            force=args.force,
        )
        for target in targets:
            print("Loading {} / {}".format(model, target), flush=True)
            data, unused_files = load_target(args.results_root, model, target)
            if data.empty:
                print("No valid predictions; skipping", flush=True)
                continue
            for region, region_data in data.groupby("Region", sort=True):
                total += render_scope(
                    region_data,
                    target,
                    model,
                    str(region).replace("_", " ").title(),
                    output_root / target / model / str(region),
                    render_args,
                )
            del data
            gc.collect()
    print("Completed: {} new PNG figures".format(total), flush=True)


if __name__ == "__main__":
    main()
