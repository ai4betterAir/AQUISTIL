#!/usr/bin/env python3
"""Run a compact, publication-focused ablation study of AQUISTIL.

The experiment keeps the regional data, Stage-3 selected predictors, artificial
missingness masks, missingness levels, and seeds fixed. It changes only one
AQUISTIL architecture module at a time, plus one deliberately weak
LightGBM-backbone combination. This is a component ablation, not a second
feature-selection search.

Default variants
----------------
1. Full AQUISTIL
2. Without observed-history features
3. Without spatial information/enhancement
4. Without gap-aware features
5. Without event-aware features and refinement
6. Without AQUISTIL's cyclic calendar encoding
7. Backbone-only weak combination

The script is resumable at variant/region/target level and writes a compact
paper table, paired deltas from full AQUISTIL, and the recommended reduced
variant. It intentionally does not search every possible module combination.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aquistil_matplotlib")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config_spatial as config
import main as pipeline
from Model import AQUISTIL
from regional_imputation import run_balanced_regional_task


LOGGER = logging.getLogger("aquistil_ablation")
DEFAULT_OUTPUT_ROOT = Path(config.OUTPUT_DIRECTORY) / "AQUISTIL_Ablation"
DEFAULT_VARIANT_KEYS = (
    "full",
    "no_history",
    "no_spatial",
    "no_gap",
    "no_event",
    "no_cyclic_calendar",
    "backbone_only",
)

BASE_SWITCHES = {
    "history_features": True,
    "spatial_features": True,
    "gap_features": True,
    "event_refinement": True,
    "calendar_features": True,
    "site_features": True,
    "uncertainty_models": True,
}

VARIANTS = OrderedDict(
    [
        (
            "full",
            {
                "model": "AQUISTIL_Full",
                "type": "Reference",
                "removed": "None",
                "description": "Complete AQUISTIL architecture",
                "switches": {},
            },
        ),
        (
            "no_history",
            {
                "model": "AQUISTIL_WithoutHistory",
                "type": "Leave-one-module-out",
                "removed": "Observed-history module",
                "description": "Removes leakage-safe lag, rolling, trend, and persistence features",
                "switches": {"history_features": False},
            },
        ),
        (
            "no_spatial",
            {
                "model": "AQUISTIL_WithoutSpatial",
                "type": "Leave-one-module-out",
                "removed": "Spatial module",
                "description": "Removes selected IDW/spatial predictors and adaptive spatial aggregation",
                "switches": {"spatial_features": False},
            },
        ),
        (
            "no_gap",
            {
                "model": "AQUISTIL_WithoutGap",
                "type": "Leave-one-module-out",
                "removed": "Gap-aware module",
                "description": "Removes gap length, gap band, and missing-row context features",
                "switches": {"gap_features": False},
            },
        ),
        (
            "no_event",
            {
                "model": "AQUISTIL_WithoutEvent",
                "type": "Leave-one-module-out",
                "removed": "Event-aware module",
                "description": "Removes event-score features, event classifier, and event regressor refinement",
                "switches": {"event_refinement": False},
            },
        ),
        (
            "no_cyclic_calendar",
            {
                "model": "AQUISTIL_WithoutCyclicCalendar",
                "type": "Leave-one-module-out",
                "removed": "Cyclic calendar encoding",
                "description": "Removes AQUISTIL sine/cosine calendar encoding; shared Stage-3 calendar inputs remain",
                "switches": {"calendar_features": False},
            },
        ),
        (
            "backbone_only",
            {
                "model": "AQUISTIL_BackboneOnly",
                "type": "Weak combination",
                "removed": "All AQUISTIL-specific predictive modules",
                "description": "Shared Stage-3 predictors and pooled LightGBM backbone only",
                "switches": {
                    "history_features": False,
                    "spatial_features": False,
                    "gap_features": False,
                    "event_refinement": False,
                    "calendar_features": False,
                    "uncertainty_models": False,
                },
            },
        ),
    ]
)

SUMMARY_METRICS = ("RMSE", "MAE", "RMAE", "R", "R2", "NSE", "WI")
SCENARIO_KEYS = ("Region", "Target", "Regime", "Missingness_Level", "Seed", "Scope")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run compact AQUISTIL component ablations using the same regional "
            "inputs and missingness masks as the main publication pipeline."
        )
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Region names. Default: config.SELECT_TARGET_REGIONS, then config.TARGET_REGIONS.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Target columns. Default: config.TARGET_COLUMNS.",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=None,
        choices=["random", "short_gap", "medium_gap", "long_gap", "event"],
        help="Missingness regimes. Default: config.MISSINGNESS_REGIMES.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=None,
        help="Missingness fractions or percentages, e.g. 0.1 0.2 or 10 20.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Evaluation seeds. Default: config.REGIONAL_EVALUATION_SEEDS.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(DEFAULT_VARIANT_KEYS),
        help="Ablation variants to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Ablation output directory.",
    )
    parser.add_argument(
        "--wide-input-dir",
        type=Path,
        default=Path(config.WIDE_API_INPUT_DIRECTORY),
        help="Directory containing wide regional API-input CSV files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute completed variant/region/target tasks.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="Print the ablation variants and exit.",
    )
    return parser.parse_args()


def _normalise_levels(values: list[float]) -> list[float]:
    levels = []
    for value in values:
        level = value / 100.0 if value > 1 else value
        if not 0 < level < 1:
            raise ValueError("Missingness levels must be between 0 and 1, or 1 and 100")
        levels.append(float(level))
    return list(dict.fromkeys(levels))


def _variant_switches(variant_key: str) -> dict:
    switches = dict(BASE_SWITCHES)
    switches.update(VARIANTS[variant_key]["switches"])
    return switches


def _make_imputer(variant_key: str) -> Callable:
    switches = _variant_switches(variant_key)

    def impute(data, target_column, input_columns, custom_strategies=None, **kwargs):
        runtime = dict(kwargs)
        runtime.update(switches)
        runtime["random_state"] = int(runtime.get("seed", 42))
        return AQUISTIL.impute_mice(
            data,
            target_column,
            input_columns,
            custom_strategies=custom_strategies,
            **runtime,
        )

    return impute


def _resolve_regions(args: argparse.Namespace) -> list[str]:
    if args.regions:
        return list(args.regions)
    selected = list(getattr(config, "SELECT_TARGET_REGIONS", []) or [])
    return selected or list(getattr(config, "TARGET_REGIONS", []) or [])


def _prepare_regional_datasets(
    regions: list[str],
    targets: list[str],
    wide_input_dir: Path,
    cache_root: Path,
) -> dict:
    region_files = pipeline._resolve_region_wide_files(regions, wide_dir=str(wide_input_dir))
    if not region_files:
        raise RuntimeError("No requested regional wide-input files were found")

    all_region_files = pipeline._resolve_region_wide_files(None, wide_dir=str(wide_input_dir))
    site_index = pipeline._build_site_region_index(region_files)
    prepared_sites = pipeline._materialize_per_site_cache(
        region_files, str(cache_root / "selected_sites")
    )
    neighbor_sites = pipeline._materialize_per_site_cache(
        all_region_files, str(cache_root / "idw_neighbors")
    )
    progressive_map = pipeline._load_progressive_best_features(
        config.PROGRESSIVE_BEST_FEATURES_CSV
    )

    requested_site_tokens = {
        pipeline._canon_token(site)
        for site in (getattr(config, "SELECT_TARGET_SITES", []) or [])
    }
    datasets = {}

    for region_token, _ in region_files:
        for target in targets:
            choice = progressive_map.get(
                (pipeline._canon_token(region_token), pipeline._canon_token(target))
            )
            choice = (
                dict(choice)
                if choice
                else pipeline._borrow_stage3_regional_feature_choice(
                    progressive_map, region_token, target
                )
            )
            if not choice:
                LOGGER.warning("No Stage-3 feature contract for %s/%s", region_token, target)
                continue

            site_parts = []
            for site_meta in site_index.values():
                if pipeline._canon_token(site_meta["region_token"]) != pipeline._canon_token(
                    region_token
                ):
                    continue
                site_name = site_meta["site_name"]
                if requested_site_tokens and pipeline._canon_token(site_name) not in requested_site_tokens:
                    continue
                site_path = prepared_sites.get(pipeline._canon_token(site_name))
                if not site_path:
                    continue
                site_data = pd.read_csv(site_path, low_memory=False)
                site_data["DateTime"] = pd.to_datetime(
                    site_data["DateTime"], errors="coerce"
                )
                site_data = pipeline._add_progressive_derived_features(
                    site_data,
                    site_name,
                    target,
                    choice["features"],
                    neighbor_sites or prepared_sites,
                )
                site_data["Site"] = site_name
                site_data["Region"] = region_token.replace("_", " ")
                site_parts.append(site_data)

            if not site_parts:
                LOGGER.warning("No site data constructed for %s/%s", region_token, target)
                continue

            regional_data = pd.concat(site_parts, ignore_index=True)
            if target not in regional_data.columns:
                LOGGER.warning("Target %s is absent for %s", target, region_token)
                continue

            usable_features = []
            for feature in choice["features"]:
                if feature not in regional_data.columns:
                    continue
                if pd.to_numeric(regional_data[feature], errors="coerce").notna().sum() >= 50:
                    usable_features.append(feature)
            if not usable_features:
                LOGGER.warning("No usable selected predictors for %s/%s", region_token, target)
                continue

            choice = dict(choice)
            choice["features"] = usable_features
            datasets[(region_token, target)] = (regional_data, choice)

    if not datasets:
        raise RuntimeError("No regional ablation datasets could be constructed")
    return datasets


def _task_paths(
    raw_root: Path, model_name: str, region_token: str, target: str
) -> tuple[Path, Path]:
    task_root = (
        raw_root
        / model_name
        / region_token.replace(" ", "_")
        / target.replace(".", "")
    )
    return (
        task_root / "metrics_by_site_and_region.csv",
        task_root / "masked_predictions_by_site.csv",
    )


def _is_complete(
    metrics: pd.DataFrame,
    regimes: list[str],
    levels: list[float],
    seeds: list[int],
) -> bool:
    if metrics.empty or "Scope" not in metrics.columns:
        return False
    micro = metrics.loc[metrics["Scope"] == "Region_Micro"]
    expected = len(regimes) * len(levels) * len(seeds)
    scenario_columns = ["Regime", "Missingness_Level", "Seed"]
    return len(micro.loc[:, scenario_columns].drop_duplicates()) == expected


def _run_task(
    variant_key: str,
    region_token: str,
    target: str,
    regional_data: pd.DataFrame,
    choice: dict,
    regimes: list[str],
    levels: list[float],
    seeds: list[int],
    raw_root: Path,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant = VARIANTS[variant_key]
    model_name = variant["model"]
    metrics_path, predictions_path = _task_paths(
        raw_root, model_name, region_token, target
    )

    if not force and metrics_path.is_file() and predictions_path.is_file():
        metrics = pd.read_csv(metrics_path)
        if _is_complete(metrics, regimes, levels, seeds):
            LOGGER.info("Resume: %s/%s/%s", model_name, region_token, target)
            return metrics, pd.read_csv(predictions_path)

    switches = _variant_switches(variant_key)
    LOGGER.info(
        "Run: %s | %s/%s | switches=%s",
        model_name,
        region_token,
        target,
        switches,
    )
    result = run_balanced_regional_task(
        regional_data=regional_data,
        region=region_token.replace("_", " "),
        target=target,
        features=choice["features"],
        model_name=model_name,
        impute_callable=_make_imputer(variant_key),
        regimes=regimes,
        missingness_levels=levels,
        seeds=seeds,
        output_root=str(raw_root),
        plots_root=None,
        plot_types=(),
        parameters={
            "ablation_key": variant_key,
            "ablation_type": variant["type"],
            "removed_component": variant["removed"],
            "description": variant["description"],
            "switches": switches,
            "stage3_configuration": choice.get("configuration", ""),
            "stage3_blocks": choice.get("blocks", ""),
            "feature_source": choice.get("feature_source", "stage3"),
        },
    )
    return result["metrics"], result["predictions"]


def _build_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    micro = metrics.loc[metrics["Scope"] == "Region_Micro"].copy()
    available_metrics = [metric for metric in SUMMARY_METRICS if metric in micro.columns]
    aggregation = {}
    for metric in available_metrics:
        aggregation[f"{metric}_Mean"] = (metric, "mean")
        aggregation[f"{metric}_SD"] = (metric, "std")
    aggregation["N_Scenarios"] = ("RMSE", "size")

    summary = (
        micro.groupby("Model", as_index=False)
        .agg(**aggregation)
        .sort_values(["RMSE_Mean", "R2_Mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "Rank", np.arange(1, len(summary) + 1))

    metadata_rows = []
    for key, variant in VARIANTS.items():
        metadata_rows.append(
            {
                "Model": variant["model"],
                "Variant_Key": key,
                "Ablation_Type": variant["type"],
                "Removed_Component": variant["removed"],
                "Description": variant["description"],
                "Switches": json.dumps(_variant_switches(key), sort_keys=True),
            }
        )
    summary = summary.merge(pd.DataFrame(metadata_rows), on="Model", how="left")

    full_name = VARIANTS["full"]["model"]
    full = micro.loc[micro["Model"] == full_name, list(SCENARIO_KEYS) + available_metrics]
    full = full.rename(columns={metric: f"{metric}_Full" for metric in available_metrics})
    paired_parts = []
    for model_name, model_rows in micro.groupby("Model", sort=False):
        paired = model_rows.merge(full, on=list(SCENARIO_KEYS), how="inner")
        if paired.empty:
            continue
        for metric in available_metrics:
            paired[f"Delta_{metric}"] = paired[metric] - paired[f"{metric}_Full"]
        denominator = paired["RMSE_Full"].replace(0.0, np.nan)
        paired["RMSE_Change_Percent"] = (
            100.0 * paired["Delta_RMSE"] / denominator
        )
        paired_parts.append(paired)
    paired_deltas = (
        pd.concat(paired_parts, ignore_index=True) if paired_parts else pd.DataFrame()
    )

    if not paired_deltas.empty:
        delta_columns = [
            column
            for column in paired_deltas.columns
            if column.startswith("Delta_") or column == "RMSE_Change_Percent"
        ]
        delta_summary = (
            paired_deltas.groupby("Model", as_index=False)[delta_columns]
            .mean()
            .rename(
                columns={
                    column: f"{column}_Mean"
                    for column in delta_columns
                }
            )
        )
        summary = summary.merge(delta_summary, on="Model", how="left")

    summary["Recommended_Reduced"] = False
    reduced = summary.loc[
        (summary["Model"] != full_name)
        & (summary["Ablation_Type"] != "Weak combination")
    ]
    if reduced.empty:
        reduced = summary.loc[summary["Model"] != full_name]
    if not reduced.empty:
        recommended_index = reduced.sort_values(
            ["RMSE_Mean", "R2_Mean"], ascending=[True, False]
        ).index[0]
        summary.loc[recommended_index, "Recommended_Reduced"] = True

    return summary, paired_deltas


def _write_recommendation(summary: pd.DataFrame, output_root: Path) -> None:
    full_name = VARIANTS["full"]["model"]
    overall = summary.sort_values(["RMSE_Mean", "R2_Mean"], ascending=[True, False]).iloc[0]
    reduced_rows = summary.loc[summary["Recommended_Reduced"]]
    reduced = reduced_rows.iloc[0] if not reduced_rows.empty else None

    payload = {
        "selection_rule": "Lowest mean regional-micro RMSE; mean R2 breaks ties",
        "best_overall_model": overall["Model"],
        "best_overall_rmse_mean": float(overall["RMSE_Mean"]),
        "full_model": full_name,
        "recommended_reduced_model": None if reduced is None else reduced["Model"],
        "recommended_removed_component": (
            None if reduced is None else reduced["Removed_Component"]
        ),
        "recommended_reduced_rmse_mean": (
            None if reduced is None else float(reduced["RMSE_Mean"])
        ),
    }
    with (output_root / "recommended_variant.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = _parse_args()
    if args.list_variants:
        for key, variant in VARIANTS.items():
            print(f"{key:20s} {variant['model']:34s} {variant['removed']}")
        return

    output_root = args.output_root.resolve()
    raw_root = output_root / "raw"
    cache_root = output_root / "cache"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(output_root / "ablation_progress.log", mode="a"),
        ],
        force=True,
    )

    regions = _resolve_regions(args)
    targets = list(args.targets or config.TARGET_COLUMNS)
    regimes = list(args.regimes or config.MISSINGNESS_REGIMES)
    levels = _normalise_levels(list(args.levels or config.MISSINGNESS_LEVELS))
    seeds = list(args.seeds or getattr(config, "REGIONAL_EVALUATION_SEEDS", [42]))
    variant_keys = list(dict.fromkeys(args.variants))

    run_settings = {
        "regions": regions,
        "targets": targets,
        "regimes": regimes,
        "missingness_levels": levels,
        "seeds": seeds,
        "variants": variant_keys,
        "wide_input_dir": str(args.wide_input_dir.resolve()),
        "stage3_features_csv": str(config.PROGRESSIVE_BEST_FEATURES_CSV),
        "selection_rule": "Lowest mean regional-micro RMSE; mean R2 breaks ties",
    }
    with (output_root / "run_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(run_settings, handle, indent=2)

    LOGGER.info("Preparing controlled regional datasets")
    datasets = _prepare_regional_datasets(
        regions=regions,
        targets=targets,
        wide_input_dir=args.wide_input_dir.resolve(),
        cache_root=cache_root,
    )

    # Match the main regional pipeline: Stage-3 features are materialized before
    # model execution, so generic spatial/temporal expansion is disabled.
    config.USE_SPATIAL_FEATURES = False
    config.USE_TEMPORAL_FEATURES = False

    metric_parts = []
    prediction_parts = []
    for variant_key in variant_keys:
        for (region_token, target), (regional_data, choice) in datasets.items():
            config.STRICT_PROGRESSIVE_FEATURE_LIST = bool(
                choice.get("strict_progressive", False)
            )
            task_metrics, task_predictions = _run_task(
                variant_key=variant_key,
                region_token=region_token,
                target=target,
                regional_data=regional_data,
                choice=choice,
                regimes=regimes,
                levels=levels,
                seeds=seeds,
                raw_root=raw_root,
                force=args.force,
            )
            metric_parts.append(task_metrics)
            prediction_parts.append(task_predictions)

    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics = metrics.loc[metrics["Scope"] == "Region_Micro"].reset_index(drop=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics.to_csv(output_root / "ablation_metrics_long.csv", index=False)
    predictions.to_csv(output_root / "ablation_masked_predictions.csv", index=False)

    summary, paired_deltas = _build_summary(metrics)
    summary.to_csv(output_root / "ablation_publication_table.csv", index=False)
    paired_deltas.to_csv(output_root / "ablation_paired_deltas.csv", index=False)
    _write_recommendation(summary, output_root)

    LOGGER.info("Ablation study complete: %s", output_root)
    LOGGER.info(
        "Best model by mean RMSE: %s",
        summary.sort_values(["RMSE_Mean", "R2_Mean"], ascending=[True, False])
        .iloc[0]["Model"],
    )


if __name__ == "__main__":
    main()
