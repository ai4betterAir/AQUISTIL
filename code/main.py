"""
Main Script for Spatial-Temporal Imputation Framework
Orchestrates multiple imputation models across multiple sites
Enhanced with comprehensive feature tracking and reporting
NOW SUPPORTS ALL MISSINGNESS REGIMES AUTOMATICALLY

Author: Dr.  Masrur
Last Updated: 2026-03-03 (centralized outputs in main.py)
"""

import os
import argparse
import subprocess
import sys
import pandas as pd
import re
from sortdata import sort_and_impute_by_hour
from spatial import get_available_sites
import importlib
import logging
import config_spatial as config
import traceback
from datetime import datetime
import aggregate_metrics
import numpy as np
from pandas.errors import EmptyDataError
from typing import Callable
import shutil
import inspect
import warnings
import fcntl
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)

# Ensure local imports work when running from repo root (common in SLURM).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# ------------------------------------------------------------------------------
# WIDE NOWCASTING INPUTS (regional-wide CSVs)
# ------------------------------------------------------------------------------

DEFAULT_WIDE_INPUT_DIR = os.path.abspath(
    getattr(
        config,
        "WIDE_API_INPUT_DIRECTORY",
        os.path.join(os.path.dirname(THIS_DIR), "API_Input", "Inputs"),
    )
)
NOWCASTING_WIDE_INPUT_DIR = os.environ.get("AQUISTIL_WIDE_INPUT_DIR", DEFAULT_WIDE_INPUT_DIR)


def _canon_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s).upper())


def _safe_output_token(value) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return token or "UNKNOWN"


def _target_run_token() -> str:
    targets = list(getattr(config, "TARGET_COLUMNS", []) or [])
    if len(targets) == 1:
        return _safe_output_token(targets[0])
    return "all_targets"


def _aquistil_ablation_models() -> list[str]:
    return list(dict.fromkeys(getattr(config, "AQUISTIL_ABLATION_MODELS", []) or []))


def _aquistil_ablation_assessment_models() -> list[str]:
    return list(
        dict.fromkeys(
            ["AQUISTIL"] + _aquistil_ablation_models() + ["LightGBM"]
        )
    )


def _aquistil_ablation_metrics_dir(default_metrics_dir: str) -> str:
    if not _ablation_run_requested():
        return default_metrics_dir
    configured = getattr(config, "AQUISTIL_ABLATION_METRICS_DIRECTORY", "")
    return os.path.abspath(configured or default_metrics_dir)


def _without_aquistil_ablations(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty or "Model" not in metrics.columns:
        return metrics
    return metrics.loc[~metrics["Model"].isin(_aquistil_ablation_models())].copy()


def _normal_comparison_models() -> list[str]:
    configured = getattr(config, "COMPARISON_MODELS", config.MODELS_TO_RUN)
    ablations = set(_aquistil_ablation_models())
    return [model for model in dict.fromkeys(configured) if model not in ablations]


def _ablation_run_requested() -> bool:
    configured = set(getattr(config, "MODELS_TO_RUN", []) or [])
    return bool(configured.intersection(_aquistil_ablation_models()))


def _validate_frozen_validation_protocol(args) -> None:
    """Reject run-time drift or development-region contamination."""
    if not getattr(config, "FROZEN_VALIDATION_MODE", False):
        return

    expected_models = ["AQUISTIL", "LightGBM"]
    expected_targets = {"PM10", "PM2.5"}
    expected_regimes = {"random", "short_gap", "medium_gap", "long_gap", "event"}
    expected_levels = {0.05, 0.10, 0.20, 0.30}
    expected_seeds = {13, 29, 42, 77, 101, 137, 211, 307, 401, 503}
    expected_regions = set(getattr(config, "HELD_OUT_VALIDATION_REGIONS", []) or [])
    development_regions = set(getattr(config, "DEVELOPMENT_REGIONS", []) or [])
    selected_regions = set(getattr(config, "SELECT_TARGET_REGIONS", []) or [])
    selected_targets = set(getattr(config, "TARGET_COLUMNS", []) or [])
    errors = []

    if list(getattr(config, "MODELS_TO_RUN", []) or []) != expected_models:
        errors.append("models must be exactly AQUISTIL and LightGBM")
    if not selected_targets or not selected_targets.issubset(expected_targets):
        errors.append("targets must be PM10 and/or PM2.5")
    if selected_regions != expected_regions:
        errors.append("selected regions must exactly match HELD_OUT_VALIDATION_REGIONS")
    overlap = selected_regions.intersection(development_regions)
    if overlap:
        errors.append("development regions selected: %s" % sorted(overlap))
    if set(getattr(config, "MISSINGNESS_REGIMES", []) or []) != expected_regimes:
        errors.append("missingness regimes differ from the frozen five-regime protocol")
    if set(map(float, getattr(config, "MISSINGNESS_LEVELS", []) or [])) != expected_levels:
        errors.append("missingness levels differ from the frozen four-level protocol")
    if set(map(int, getattr(config, "REGIONAL_EVALUATION_SEEDS", []) or [])) != expected_seeds:
        errors.append("evaluation seeds differ from the frozen ten-seed protocol")
    if getattr(config, "RUN_AQUISTIL_ABLATIONS", False):
        errors.append("AQUISTIL ablations must be disabled")
    if not getattr(config, "AQUISTIL_REGIME_AWARE_ENABLED", False):
        errors.append("the frozen AQUISTIL router is disabled")
    if getattr(config, "AQUISTIL_GAP_EXPERT_MIN_RUN_LENGTH", None) != getattr(
        config, "FROZEN_GAP_EXPERT_MIN_RUN_LENGTH", None
    ):
        errors.append("AQUISTIL gap-expert threshold differs from the frozen value")
    if os.path.abspath(config.OUTPUT_DIRECTORY) != os.path.abspath(
        getattr(config, "FROZEN_OUTPUT_DIRECTORY", config.OUTPUT_DIRECTORY)
    ):
        errors.append("output directory differs from FROZEN_OUTPUT_DIRECTORY")
    if args.refresh_api_inputs:
        errors.append("API refresh is forbidden during frozen validation")

    if errors:
        raise RuntimeError("Frozen validation protocol violation: " + "; ".join(errors))


def _upsert_aquistil_ablation_metrics(
    current_metrics: pd.DataFrame,
    metrics_dir: str,
) -> pd.DataFrame:
    """Persist ablation assessment rows without contaminating paper metrics."""
    ablation_models = _aquistil_ablation_models()
    if (
        current_metrics is None
        or current_metrics.empty
        or "Model" not in current_metrics.columns
        or not ablation_models
    ):
        return pd.DataFrame()

    assessment_models = (
        _aquistil_ablation_assessment_models()
        if _ablation_run_requested()
        else ablation_models
    )
    ablation_metrics = current_metrics.loc[
        current_metrics["Model"].isin(assessment_models)
    ].copy()
    if ablation_metrics.empty:
        return ablation_metrics

    metrics_dir = _aquistil_ablation_metrics_dir(metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)
    output_path = os.path.join(metrics_dir, "aquistil_ablation_metrics.csv")
    with _exclusive_file_lock(output_path + ".lock"):
        merged = aggregate_metrics.upsert_regional_metrics(output_path, ablation_metrics)
        merged.to_csv(output_path, index=False)

    if "Target" in ablation_metrics.columns:
        for target, target_metrics in ablation_metrics.groupby(
            "Target", sort=True, dropna=False
        ):
            target_path = os.path.join(
                metrics_dir,
                f"aquistil_ablation_metrics_{_safe_output_token(target)}.csv",
            )
            with _exclusive_file_lock(target_path + ".lock"):
                target_merged = aggregate_metrics.upsert_regional_metrics(
                    target_path, target_metrics
                )
                target_merged.to_csv(target_path, index=False)

    logging.info(
        "Saved dedicated AQUISTIL ablation metrics: %s "
        "(%d current rows, %d total rows)",
        output_path,
        len(ablation_metrics),
        len(merged),
    )
    return merged


def _write_aquistil_ablation_comparison(
    regional_metrics: pd.DataFrame,
    metrics_dir: str,
) -> pd.DataFrame:
    """Write paired full-model, ablation, and LightGBM metrics when complete."""
    models = _aquistil_ablation_assessment_models()
    if regional_metrics is None or regional_metrics.empty or len(models) < 3:
        return pd.DataFrame()

    metrics_dir = _aquistil_ablation_metrics_dir(metrics_dir)
    dedicated_path = os.path.join(metrics_dir, "aquistil_ablation_metrics.csv")
    comparison_source = regional_metrics.copy()
    if os.path.exists(dedicated_path):
        comparison_source = pd.concat(
            [comparison_source, pd.read_csv(dedicated_path)],
            ignore_index=True,
            sort=False,
        )
        comparison_source = aggregate_metrics.upsert_regional_metrics(
            pd.DataFrame(), comparison_source
        )

    comparison_path = os.path.join(
        metrics_dir, "aquistil_ablation_comparison.csv"
    )
    comparison = aggregate_metrics.write_models_comparison(
        comparison_source,
        comparison_path,
        models,
    )
    logging.info(
        "Saved paired AQUISTIL ablation comparison: %s (%d rows)",
        comparison_path,
        len(comparison),
    )
    return comparison


def _load_persisted_standard_metrics(metrics_dir: str) -> pd.DataFrame:
    metrics_path = os.path.join(metrics_dir, "regional_pooled_metrics.csv")
    if os.path.exists(metrics_path):
        return _without_aquistil_ablations(pd.read_csv(metrics_path))

    frames = []
    for path in sorted(Path(metrics_dir).glob("regional_pooled_metrics_*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return _without_aquistil_ablations(
        pd.concat(frames, ignore_index=True, sort=False)
    )


def _load_aquistil_ablation_resume_metrics(metrics_dir: str) -> pd.DataFrame:
    """Seed and load the seven-model assessment dataset for task resumption."""
    if not _ablation_run_requested():
        return pd.DataFrame()

    ablation_metrics_dir = _aquistil_ablation_metrics_dir(metrics_dir)
    standard_path = os.path.join(
        ablation_metrics_dir, "regional_pooled_metrics.csv"
    )
    if os.path.exists(standard_path):
        standard_metrics = _without_aquistil_ablations(pd.read_csv(standard_path))
        _upsert_aquistil_ablation_metrics(standard_metrics, metrics_dir)

    dedicated_path = os.path.join(
        ablation_metrics_dir, "aquistil_ablation_metrics.csv"
    )
    if not os.path.exists(dedicated_path):
        return pd.DataFrame()
    return pd.read_csv(dedicated_path)


def _ablation_task_is_complete(
    metrics: pd.DataFrame,
    region: str,
    target: str,
    model: str,
) -> bool:
    """Return True only when every configured evaluation key is persisted."""
    if metrics is None or metrics.empty:
        return False
    required = {
        "Region", "Site", "Target", "Model", "Regime",
        "Missingness_Level", "Seed", "Scope",
    }
    if not required.issubset(metrics.columns):
        return False

    selected = metrics.loc[
        metrics["Region"].map(_canon_token).eq(_canon_token(region))
        & metrics["Target"].map(_canon_token).eq(_canon_token(target))
        & metrics["Model"].astype(str).eq(str(model))
    ].copy()
    if selected.empty or set(selected["Scope"].dropna()) != {
        "Site", "Region_Macro", "Region_Micro"
    }:
        return False

    reference = metrics.loc[
        metrics["Region"].map(_canon_token).eq(_canon_token(region))
        & metrics["Target"].map(_canon_token).eq(_canon_token(target))
        & metrics["Model"].astype(str).eq("AQUISTIL")
        & metrics["Scope"].eq("Site")
    ]
    if model != "AQUISTIL" and reference.empty:
        return False
    if not reference.empty:
        expected_sites = set(reference["Site"].dropna().astype(str))
        actual_sites = set(
            selected.loc[selected["Scope"].eq("Site"), "Site"]
            .dropna()
            .astype(str)
        )
        if actual_sites != expected_sites:
            return False

    expected_grid = {
        (str(regime), round(float(level), 12), int(seed))
        for regime in getattr(config, "MISSINGNESS_REGIMES", [])
        for level in getattr(config, "MISSINGNESS_LEVELS", [])
        for seed in getattr(config, "REGIONAL_EVALUATION_SEEDS", [])
    }
    if not expected_grid:
        return False

    identity = [
        "Region", "Site", "Target", "Model", "Regime",
        "Missingness_Level", "Seed", "Scope",
    ]
    if selected.duplicated(identity).any():
        return False
    for (_, _), group in selected.groupby(["Scope", "Site"], dropna=False):
        actual_grid = {
            (str(row.Regime), round(float(row.Missingness_Level), 12), int(row.Seed))
            for row in group.itertuples()
        }
        if actual_grid != expected_grid:
            return False
    return True


def _finalize_aquistil_ablation_outputs(metrics_dir: str, results_root: str) -> None:
    if not _ablation_run_requested():
        return
    ablation_metrics_dir = _aquistil_ablation_metrics_dir(metrics_dir)
    dedicated_path = os.path.join(
        ablation_metrics_dir, "aquistil_ablation_metrics.csv"
    )
    standard_frames = [_load_persisted_standard_metrics(metrics_dir)]
    if os.path.exists(dedicated_path):
        standard_frames.append(
            _without_aquistil_ablations(pd.read_csv(dedicated_path))
        )
    standard_frames = [frame for frame in standard_frames if not frame.empty]
    standard_metrics = (
        aggregate_metrics.upsert_regional_metrics(
            pd.DataFrame(),
            pd.concat(standard_frames, ignore_index=True, sort=False),
        )
        if standard_frames
        else pd.DataFrame()
    )
    if standard_metrics.empty:
        raise ValueError(
            "Cannot finalize AQUISTIL ablations without persisted AQUISTIL and "
            "LightGBM regional metrics"
        )
    _write_aquistil_ablation_comparison(standard_metrics, metrics_dir)

    from tools.summarize_aquistil_ablation import write_summary_files

    written_summaries = write_summary_files(
        Path(results_root),
        scope="All",
        metrics_dir=Path(ablation_metrics_dir),
    )
    logging.info(
        "Saved complete AQUISTIL ablation summaries: %s",
        ", ".join(str(path) for path in written_summaries.values()),
    )


def _write_regional_metrics_by_target(regional_metrics: pd.DataFrame, metrics_dir: str) -> None:
    if regional_metrics is None or regional_metrics.empty or "Target" not in regional_metrics.columns:
        return
    regional_metrics = _without_aquistil_ablations(regional_metrics)
    os.makedirs(metrics_dir, exist_ok=True)
    for target, target_metrics in regional_metrics.groupby("Target", sort=True, dropna=False):
        target_path = os.path.join(
            metrics_dir,
            f"regional_pooled_metrics_{_safe_output_token(target)}.csv",
        )
        target_metrics.to_csv(target_path, index=False)
        logging.info(
            "Saved regional metrics for %s: %s (%d rows)",
            target,
            target_path,
            len(target_metrics),
        )


def _upsert_regional_metrics_by_target(current_metrics: pd.DataFrame, metrics_dir: str) -> None:
    if current_metrics is None or current_metrics.empty or "Target" not in current_metrics.columns:
        return
    os.makedirs(metrics_dir, exist_ok=True)
    for target, target_metrics in current_metrics.groupby("Target", sort=True, dropna=False):
        target_path = os.path.join(
            metrics_dir,
            f"regional_pooled_metrics_{_safe_output_token(target)}.csv",
        )
        with _exclusive_file_lock(target_path + ".lock"):
            merged = aggregate_metrics.upsert_regional_metrics(target_path, target_metrics)
            merged = _without_aquistil_ablations(merged)
            merged.to_csv(target_path, index=False)
        logging.info(
            "Upserted regional metrics for %s: %s (%d current rows, %d total rows)",
            target,
            target_path,
            len(target_metrics),
            len(merged),
        )


@contextmanager
def _exclusive_file_lock(lock_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _region_token_from_config_region(region_name: str) -> str:
    """Convert config region label (e.g., 'Sydney North-west') into token used by inputs/JSON.

    The repo convention uses underscores in file/JSON keys:
      Sydney_North_west
    """
    token = str(region_name).strip()
    token = token.replace("-", "_")
    token = token.replace(" ", "_")
    # keep double-underscore out
    token = re.sub(r"_+", "_", token)
    return token


def _load_best_predictors_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        logging.info("BEST_PREDICTORS_JSON not found; using configured feature fallbacks: %s", path)
        return {}
    try:
        import json

        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.warning("Failed to load BEST_PREDICTORS_JSON=%s: %s", path, e)
        return {}


def _get_predictors_for_region_target(best_map: dict, region_token: str, target: str) -> list:
    if not best_map:
        return []
    region_block = best_map.get(region_token) or best_map.get(_region_token_from_config_region(region_token))
    if not isinstance(region_block, dict):
        return []
    # targets in JSON are like 'PM2.5', 'PM10', 'OZONE', 'NO', 'NO2'
    return list(region_block.get(target, []) or [])


def _target_output_token(target: str) -> str:
    return str(target).replace(".", "_")


def _allocated_cpu_count() -> int:
    """Return the scheduler/cpuset CPU allocation visible to this process."""
    for env_name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        try:
            value = int(os.environ.get(env_name, ""))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_target_parallelism(
    target_count: int,
    task_count: int,
    requested_workers: int,
    requested_model_n_jobs: int,
    available_cpus: int,
) -> tuple:
    """Bound target workers and divide the CPU allocation between them."""
    target_count = max(1, int(target_count))
    task_count = max(1, int(task_count))
    available_cpus = max(1, int(available_cpus))
    requested_workers = max(1, int(requested_workers or 1))
    # A task is one independent region-target evaluation. Multiple regions for
    # the same target can run concurrently without changing masks or models.
    workers = min(requested_workers, task_count, available_cpus)

    requested_model_n_jobs = int(requested_model_n_jobs or 0)
    if requested_model_n_jobs > 0:
        model_n_jobs = min(requested_model_n_jobs, available_cpus)
    else:
        model_n_jobs = max(1, available_cpus // workers)
    return workers, model_n_jobs


def _stage3_best_csv_candidates_for_target(target: str) -> list:
    target_names = list(dict.fromkeys([str(target), _target_output_token(target)]))
    return [
        os.path.join(
            os.path.dirname(THIS_DIR),
            "Outputs",
            "Feature_Selection",
            "by_target",
            target_name,
            "03Regional_Selected_Feature_Progressive_Evaluation",
            "summary_outputs",
            "regional_progressive_best_configuration_by_region.csv",
        )
        for target_name in target_names
    ]


def _stage3_best_csv_for_target(target: str) -> str:
    if getattr(config, "RUN_AQUISTIL_ABLATIONS", False):
        ablation_files = getattr(
            config, "DEVELOPMENT_ABLATION_STAGE3_FEATURE_FILES", {}
        ) or {}
        ablation_path = ablation_files.get(str(target))
        if not ablation_path or not os.path.isfile(ablation_path):
            raise FileNotFoundError(
                "Development ablation Stage 3 feature selection is missing for %s: %s"
                % (target, ablation_path)
            )
        return ablation_path
    if getattr(config, "FROZEN_VALIDATION_MODE", False):
        frozen_files = getattr(config, "FROZEN_STAGE3_FEATURE_FILES", {}) or {}
        frozen_path = frozen_files.get(str(target))
        if not frozen_path or not os.path.isfile(frozen_path):
            raise FileNotFoundError(
                "Frozen Stage 3 feature selection is missing for %s: %s"
                % (target, frozen_path)
            )
        return frozen_path
    candidates = _stage3_best_csv_candidates_for_target(target)
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[-1]


def _stage3_best_csv_display_path_for_target(target: str) -> str:
    return os.path.join(
        os.path.dirname(THIS_DIR),
        "Outputs",
        "Feature_Selection",
        "by_target",
        _target_output_token(target),
        "03Regional_Selected_Feature_Progressive_Evaluation",
        "summary_outputs",
        "regional_progressive_best_configuration_by_region.csv",
    )


def _load_progressive_best_features(path: str, implicit_target: str = None) -> dict:
    """Load Stage 3 winning configurations keyed by normalized region/target."""
    if not path or not os.path.isfile(path):
        logging.warning("Stage 3 best-configuration CSV not found: %s", path)
        return {}
    frame = pd.read_csv(path)
    required = {"Region", "Feature_List"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Stage 3 best-configuration CSV missing columns: %s" % sorted(missing))
    result = {}
    configured_targets = list(getattr(config, "TARGET_COLUMNS", []) or [])
    if implicit_target is None:
        implicit_target = configured_targets[0] if len(configured_targets) == 1 else None
    for _, row in frame.iterrows():
        target = str(row.get("Target", implicit_target or "")).strip()
        if not target:
            raise ValueError(
                "Stage 3 CSV has no Target column and config.TARGET_COLUMNS does not identify exactly one target"
            )
        features = [x.strip() for x in str(row["Feature_List"]).split(",") if x.strip()]
        result[(_canon_token(row["Region"]), _canon_token(target))] = {
            "region": str(row["Region"]).strip(),
            "configuration": str(row.get("Configuration", "")).strip(),
            "blocks": str(row.get("Blocks", "")).strip(),
            "features": features,
            "feature_source": "stage3",
            "strict_progressive": True,
        }
    return result


def _load_progressive_best_features_for_targets(targets: list) -> dict:
    """Load Stage 3 best configurations, preferring target-specific outputs."""
    explicit_path = getattr(config, "PROGRESSIVE_BEST_FEATURES_CSV", "")

    combined = {}
    loaded_paths = []
    for target in targets:
        target_path = _stage3_best_csv_for_target(target)
        target_map = _load_progressive_best_features(target_path, implicit_target=target)
        combined.update(target_map)
        if target_map:
            loaded_paths.append(target_path)

    if combined:
        logging.info(
            "Loaded Stage 3 best features from %d target-specific CSV(s): %s",
            len(loaded_paths),
            loaded_paths,
        )
        return combined

    if os.environ.get("PROGRESSIVE_BEST_FEATURES_CSV"):
        logging.info(
            "No target-specific Stage 3 CSVs were loaded; trying explicit PROGRESSIVE_BEST_FEATURES_CSV=%s",
            explicit_path,
        )
    return _load_progressive_best_features(explicit_path)


def _generic_regional_feature_choice(region_token: str, target: str) -> dict:
    """Build the configured last-resort fallback contract for a region missing Stage 3."""
    if not getattr(config, "FALLBACK_TO_INPUT_COLUMNS_WHEN_MISSING", True):
        return {}
    candidates = list(
        getattr(
            config,
            "REGIONAL_GENERIC_FEATURES",
            getattr(config, "LOCAL_ANALYSIS_INPUTS", getattr(config, "INPUT_COLUMNS", [])),
        )
        or []
    )
    features = list(
        dict.fromkeys(
            feature
            for feature in candidates
            if str(feature).strip() and _canon_token(feature) != _canon_token(target)
        )
    )
    if not features:
        return {}
    return {
        "region": str(region_token).replace("_", " "),
        "configuration": "Generic_Local_Feature_Fallback",
        "blocks": "LOCAL_GENERIC",
        "features": features,
        "feature_source": "generic_fallback",
        "strict_progressive": False,
    }


def _borrow_stage3_regional_feature_choice(
    progressive_best_map: dict,
    region_token: str,
    target: str,
) -> dict:
    """Reuse an available Stage 3 feature set for a region missing Stage 3."""
    if not getattr(config, "FALLBACK_TO_INPUT_COLUMNS_WHEN_MISSING", True):
        return {}

    target_key = _canon_token(target)
    candidates = [
        choice for (choice_region, choice_target), choice in sorted(progressive_best_map.items())
        if choice_target == target_key and choice.get("features")
    ]
    if not candidates:
        return _generic_regional_feature_choice(region_token, target)

    source = dict(candidates[0])
    borrowed = dict(source)
    borrowed["region"] = str(region_token).replace("_", " ")
    borrowed["source_region"] = source.get("region", "")
    borrowed["configuration"] = "Borrowed_Stage3_{region}_{configuration}".format(
        region=_canon_token(source.get("region", "available_region")),
        configuration=source.get("configuration", "") or "configuration",
    )
    borrowed["feature_source"] = "stage3_borrowed"
    borrowed["strict_progressive"] = True
    return borrowed


def _add_progressive_derived_features(
    data: pd.DataFrame,
    site_name: str,
    target: str,
    requested_features: list,
    prepared_site_inputs: dict,
) -> pd.DataFrame:
    """Add the temporal and IDW columns selected by Stage 3."""
    out = data.copy()
    dt = pd.to_datetime(out.get("DateTime"), errors="coerce")
    temporal_builders = {
        "Hour": lambda x: x.dt.hour,
        "Day": lambda x: x.dt.day,
        "Month": lambda x: x.dt.month,
        "DayOfWeek": lambda x: x.dt.dayofweek,
        "DayOfYear": lambda x: x.dt.dayofyear,
        "WeekOfYear": lambda x: x.dt.isocalendar().week.astype(float),
    }
    for feature, builder in temporal_builders.items():
        if feature in requested_features and feature not in out.columns:
            out[feature] = builder(dt)

    idw_col = "IDW_Spatial_PM25" if target == "PM2.5" else "IDW_Spatial_%s" % target
    if idw_col not in requested_features:
        return out

    coordinates = getattr(config, "SITE_COORDINATES", {}) or {}
    coord_map = {_canon_token(name): value for name, value in coordinates.items()}
    target_coord = coord_map.get(_canon_token(site_name))
    if not target_coord:
        logging.warning("Cannot build %s: coordinates unavailable for %s", idw_col, site_name)
        out[idw_col] = np.nan
        return out

    from math import atan2, cos, radians, sin, sqrt

    def distance_km(a, b):
        lat1, lon1, lat2, lon2 = map(radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371.0 * 2 * atan2(sqrt(value), sqrt(1 - value))

    numerator = pd.Series(0.0, index=out.index)
    denominator = pd.Series(0.0, index=out.index)
    target_index = pd.DatetimeIndex(dt)
    max_distance = float(getattr(config, "MAX_SPATIAL_DISTANCE", 100))
    for neighbor_token, neighbor_path in prepared_site_inputs.items():
        if neighbor_token == _canon_token(site_name):
            continue
        neighbor_coord = coord_map.get(neighbor_token)
        if not neighbor_coord:
            continue
        distance = distance_km(target_coord, neighbor_coord)
        if not (0 < distance <= max_distance):
            continue
        try:
            neighbor = pd.read_csv(neighbor_path, usecols=lambda c: c in {"DateTime", target})
        except EmptyDataError:
            logging.warning("Skipping empty IDW neighbor cache file: %s", neighbor_path)
            continue
        if "DateTime" not in neighbor.columns or target not in neighbor.columns:
            continue
        neighbor["DateTime"] = pd.to_datetime(neighbor["DateTime"], errors="coerce")
        series = pd.to_numeric(neighbor[target], errors="coerce")
        series.index = neighbor["DateTime"]
        series = series.loc[~series.index.duplicated(keep="last")].reindex(target_index)
        values = pd.Series(series.to_numpy(), index=out.index)
        valid = values.notna()
        weight = 1.0 / (distance ** 2)
        numerator.loc[valid] += values.loc[valid] * weight
        denominator.loc[valid] += weight
    out[idw_col] = numerator.div(denominator.where(denominator > 0))
    logging.info("Built %s for %s using %d valid timestamps", idw_col, site_name, out[idw_col].notna().sum())
    return out


def _find_region_wide_csv(region_name: str, wide_dir: str = NOWCASTING_WIDE_INPUT_DIR) -> str:
    """Return the wide regional CSV path for a region name.

    Matches files like:
      Allobs_processed_DPE_station_api_Sydney_North_west_ALL.csv
    and config regions like:
      "Sydney North-west"
    """
    if not os.path.isdir(wide_dir):
        raise FileNotFoundError("Wide input directory not found: %s" % wide_dir)

    # Try exact normalized token match
    want = _canon_token(region_name)
    candidates = []
    for fn in os.listdir(wide_dir):
        if not (fn.startswith("Allobs_processed_DPE_station_api_") and fn.endswith("_ALL.csv")):
            continue
        region_token = fn[len("Allobs_processed_DPE_station_api_") : -len("_ALL.csv")]
        if _canon_token(region_token) == want:
            return os.path.join(wide_dir, fn)
        candidates.append((region_token, fn))

    # Fallback: allow substring match
    for region_token, fn in candidates:
        if want in _canon_token(region_token) or _canon_token(region_token) in want:
            return os.path.join(wide_dir, fn)

    raise FileNotFoundError("No wide regional CSV found for region '%s' under %s" % (region_name, wide_dir))


def _wide_region_token_from_path(csv_path: str) -> str:
    fn = os.path.basename(csv_path)
    prefix = "Allobs_processed_DPE_station_api_"
    suffix = "_ALL.csv"
    if fn.startswith(prefix) and fn.endswith(suffix):
        return fn[len(prefix) : -len(suffix)]
    return os.path.splitext(fn)[0]


def _list_wide_region_files(wide_dir: str = NOWCASTING_WIDE_INPUT_DIR) -> list:
    if not os.path.isdir(wide_dir):
        return []
    files = []
    for fn in sorted(os.listdir(wide_dir)):
        if fn.startswith("Allobs_processed_DPE_station_api_") and fn.endswith("_ALL.csv"):
            files.append(os.path.join(wide_dir, fn))
    return files


def _resolve_region_wide_files(requested_regions=None, wide_dir: str = NOWCASTING_WIDE_INPUT_DIR) -> list:
    files = _list_wide_region_files(wide_dir)
    if not requested_regions:
        return [(_wide_region_token_from_path(fp), fp) for fp in files]

    resolved = []
    seen = set()
    missing = []
    for region_name in requested_regions:
        try:
            fp = _find_region_wide_csv(region_name, wide_dir=wide_dir)
        except FileNotFoundError:
            missing.append(region_name)
            continue
        if fp in seen:
            continue
        seen.add(fp)
        resolved.append((_wide_region_token_from_path(fp), fp))

    if missing:
        logging.warning(
            "Configured regions missing from wide input directory %s: %s",
            wide_dir,
            ", ".join(str(region) for region in missing),
        )
    return resolved


def _list_sites_from_wide_csv(wide_csv_path: str) -> list:
    cols = pd.read_csv(wide_csv_path, nrows=0).columns
    sites = set()
    for c in cols:
        if not isinstance(c, str):
            continue
        if c.lower() == "datetime" or c == "DateTime":
            continue
        if "_" not in c:
            continue
        _, site = c.split("_", 1)
        site = site.strip()
        if site:
            sites.add(site)
    return sorted(sites)


def _reindex_to_complete_hourly_grid(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Make one row represent exactly one hour, preserving existing columns."""
    out = df.copy()
    original_rows = len(out)
    out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
    out = out.dropna(subset=["DateTime"]).sort_values("DateTime")
    duplicate_count = int(out["DateTime"].duplicated(keep="last").sum())
    if duplicate_count:
        numeric_cols = [c for c in out.columns if c != "DateTime"]
        out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
        out = out.groupby("DateTime", as_index=False, sort=True).mean(numeric_only=True)

    if out.empty:
        logging.warning("Hourly grid skipped for empty data: %s", context)
        return out

    original_diffs = out["DateTime"].diff().dropna()
    max_gap = original_diffs.max() if not original_diffs.empty else pd.Timedelta(0)
    original_unique_timestamps = int(out["DateTime"].nunique())
    hourly_grid = pd.date_range(out["DateTime"].min(), out["DateTime"].max(), freq="h")
    out = out.set_index("DateTime").reindex(hourly_grid)
    out.index.name = "DateTime"
    out = out.reset_index()
    inserted = len(out) - original_unique_timestamps
    logging.info(
        "Hourly grid %s | original_rows=%d hourly_grid_rows=%d inserted_timestamps=%d "
        "duplicate_timestamps=%d max_original_timestamp_gap=%s",
        context,
        original_rows,
        len(out),
        max(0, int(inserted)),
        duplicate_count,
        max_gap,
    )
    return out


def _largest_missing_episode(mask) -> tuple:
    values = np.asarray(mask, dtype=bool)
    if not values.any():
        return 0, 0
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False])
    lengths = ends - starts + 1
    return int(lengths.max()), int(len(lengths))


def _write_air_quality_qc_report(regional_data: pd.DataFrame, output_root: str) -> None:
    """Write source-QA diagnostics without mutating or filtering observations."""
    targets = [t for t in getattr(config, "TARGET_COLUMNS", []) if t in regional_data.columns]
    if not targets or regional_data.empty:
        return
    qc_dir = os.path.join(output_root, "Data_QC")
    os.makedirs(qc_dir, exist_ok=True)
    summary_rows, flag_rows = [], []
    data = regional_data.copy()
    data["DateTime"] = pd.to_datetime(data["DateTime"], errors="coerce")

    for (region, site), site_data in data.groupby(["Region", "Site"], sort=False):
        site_data = site_data.sort_values("DateTime")
        for target in targets:
            series = pd.to_numeric(site_data[target], errors="coerce")
            largest_episode, episode_count = _largest_missing_episode(series.isna().to_numpy())
            observed = series.dropna()
            summary_rows.append(
                {
                    "Region": region,
                    "Site": site,
                    "Target": target,
                    "N": int(len(series)),
                    "Missing_Percent": float(series.isna().mean() * 100) if len(series) else np.nan,
                    "Minimum": float(observed.min()) if len(observed) else np.nan,
                    "Mean": float(observed.mean()) if len(observed) else np.nan,
                    "Median": float(observed.median()) if len(observed) else np.nan,
                    "SD": float(observed.std()) if len(observed) > 1 else np.nan,
                    "P95": float(observed.quantile(0.95)) if len(observed) else np.nan,
                    "P99": float(observed.quantile(0.99)) if len(observed) else np.nan,
                    "P99_5": float(observed.quantile(0.995)) if len(observed) else np.nan,
                    "Maximum": float(observed.max()) if len(observed) else np.nan,
                    "Negative_Observations": int((observed < 0).sum()),
                    "Largest_Natural_Missing_Episode": largest_episode,
                    "Missing_Episodes": episode_count,
                }
            )
            if len(observed) < 8:
                continue
            median = observed.median()
            mad = (observed - median).abs().median()
            q1, q3 = observed.quantile([0.25, 0.75])
            iqr = q3 - q1
            p995 = observed.quantile(0.995)
            robust_z = 0.6745 * (series - median) / mad if mad and mad > 0 else pd.Series(np.nan, index=series.index)
            iqr_limit = q3 + 6 * iqr if iqr and iqr > 0 else np.inf
            quantile_limit = max(p995, observed.quantile(0.99) * 1.5)
            flags = (robust_z > 12) | (series > iqr_limit) | (series > quantile_limit)
            for idx in site_data.index[flags.fillna(False)]:
                row_pos = site_data.index.get_loc(idx)
                flag_rows.append(
                    {
                        "Region": region,
                        "Site": site,
                        "Target": target,
                        "DateTime": site_data.loc[idx, "DateTime"],
                        "Value": series.loc[idx],
                        "Robust_Z": robust_z.loc[idx],
                        "IQR_High_Limit": iqr_limit,
                        "P99_5": p995,
                        "Same_Site_PM2_5": site_data.loc[idx, "PM2.5"] if "PM2.5" in site_data.columns else np.nan,
                        "Same_Site_NEPH": site_data.loc[idx, "NEPH"] if "NEPH" in site_data.columns else np.nan,
                        "Previous_Target_Value": series.iloc[row_pos - 1] if row_pos > 0 else np.nan,
                        "Next_Target_Value": series.iloc[row_pos + 1] if row_pos + 1 < len(series) else np.nan,
                    }
                )

    summary_path = os.path.join(qc_dir, "site_target_summary.csv")
    flags_path = os.path.join(qc_dir, "extreme_value_flags.csv")
    summary = pd.DataFrame(summary_rows)
    flags = pd.DataFrame(flag_rows)
    if os.path.exists(summary_path):
        summary = pd.concat([pd.read_csv(summary_path), summary], ignore_index=True, sort=False)
        summary = summary.drop_duplicates(["Region", "Site", "Target"], keep="last")
    if os.path.exists(flags_path) and not flags.empty:
        flags = pd.concat([pd.read_csv(flags_path), flags], ignore_index=True, sort=False)
        flags = flags.drop_duplicates(["Region", "Site", "Target", "DateTime", "Value"], keep="last")
    summary.to_csv(summary_path, index=False)
    flags.to_csv(flags_path, index=False)
    logging.info(
        "Saved air-quality QC diagnostics: %s (%d summaries, %d extreme flags)",
        qc_dir,
        len(summary_rows),
        len(flag_rows),
    )


def _build_site_region_index(region_files: list) -> dict:
    site_map = {}
    for region_token, wide_fp in region_files:
        try:
            sites_in_region = _list_sites_from_wide_csv(wide_fp)
        except Exception as e:
            logging.warning("Failed reading wide input %s: %s", wide_fp, e)
            continue
        for site_name in sites_in_region:
            key = _canon_token(site_name)
            if key not in site_map:
                site_map[key] = {
                    "site_name": site_name,
                    "region_token": region_token,
                    "wide_path": wide_fp,
                }
    return site_map


def _materialize_per_site_cache(region_files: list, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    site_input_paths = {}
    for region_token, wide_fp in region_files:
        try:
            sites_in_region = _list_sites_from_wide_csv(wide_fp)
        except Exception as e:
            logging.warning("Failed listing sites from %s: %s", wide_fp, e)
            continue
        for site_name in sites_in_region:
            try:
                out_path = _build_per_site_csv_from_wide(wide_fp, site_name, out_dir)
                site_input_paths[_canon_token(site_name)] = out_path
            except Exception as e:
                logging.warning("Failed building per-site CSV for %s from %s: %s", site_name, wide_fp, e)
    logging.info(
        "Prepared %d per-site CSVs from %d wide regional file(s) into %s",
        len(site_input_paths),
        len(region_files),
        out_dir,
    )
    return site_input_paths


def _build_per_site_csv_from_wide(
    wide_csv_path: str,
    site_name: str,
    out_dir: str,
) -> str:
    """Materialize a per-site CSV (DateTime + local variables) from a wide region CSV.

    This keeps downstream code unchanged (sortdata.py expects input_file path).
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(wide_csv_path, low_memory=False)
    dt_col = "datetime" if "datetime" in df.columns else "DateTime" if "DateTime" in df.columns else None
    if not dt_col:
        raise ValueError("Wide CSV missing datetime column: %s" % wide_csv_path)

    # Select columns for this site (normalize for dashes/underscores/spaces)
    wanted_cols = [dt_col]
    wanted_site = _canon_token(site_name)
    for c in df.columns:
        if not isinstance(c, str):
            continue
        if c == dt_col:
            continue
        if "_" not in c:
            continue
        var, st = c.split("_", 1)
        if _canon_token(st) == wanted_site:
            wanted_cols.append(c)

    if len(wanted_cols) <= 1:
        raise ValueError("No columns found for site '%s' in %s" % (site_name, wide_csv_path))

    sub = df[wanted_cols].copy()
    sub = sub.rename(columns={dt_col: "DateTime"})

    # Rename VAR_SITE -> VAR
    rename_map = {}
    for c in sub.columns:
        if c == "DateTime":
            continue
        if "_" in c:
            var, _ = c.split("_", 1)
            rename_map[c] = var
    sub = sub.rename(columns=rename_map)

    # Keep stable column order
    col_order = ["DateTime"] + [c for c in sub.columns if c != "DateTime"]
    sub = sub[col_order]

    # Clean + enforce datetime
    sub["DateTime"] = pd.to_datetime(sub["DateTime"], errors="coerce")
    sub = _reindex_to_complete_hourly_grid(
        sub,
        context="%s/%s" % (_wide_region_token_from_path(wide_csv_path), site_name),
    )

    safe_site = re.sub(r"[^A-Za-z0-9]+", "_", site_name.strip())
    out_path = os.path.join(out_dir, f"{safe_site}.csv")
    tmp_path = f"{out_path}.{os.getpid()}.tmp"
    sub.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    return out_path

# ------------------------------------------------------------------------------
# CENTRAL OUTPUT LAYOUT (single source of truth)
# ------------------------------------------------------------------------------

# Base results root (from config)
RESULTS_ROOT = config.OUTPUT_DIRECTORY

# Per-model raw outputs (tree where imputed_data / metrics / target_column_data live)
MODEL_OUTPUT_ROOT = os.path.join(RESULTS_ROOT, config.MODEL_OUTPUT_SUBDIR)

# Central aggregated metrics for all models/sites/regimes
CENTRAL_METRICS_ROOT = os.path.join(RESULTS_ROOT, "Metrics")

# Central per-sample imputed master files
CENTRAL_IMPUTED_ROOT = os.path.join(RESULTS_ROOT, "Imputed_Results")

# Central plots (cross-model, cross-regime)
CENTRAL_PLOTS_ROOT = os.path.join(RESULTS_ROOT, "plots_by_type")

# Research-grade plots (e.g. research_plots.py outputs)
RESEARCH_PLOTS_DIR = os.path.join(RESULTS_ROOT, "Research_Plots")

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
log_file = os.path.join(RESULTS_ROOT, 'imputation_framework.log')
os.makedirs(RESULTS_ROOT, exist_ok=True)

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_CURRENT_LOG_TARGET = ContextVar("aquistil_log_target", default=None)


class _TargetLogFilter(logging.Filter):
    """Route records emitted inside one target worker to its target log."""

    def __init__(self, target):
        super().__init__()
        self.target = str(target)

    def filter(self, record):
        return _CURRENT_LOG_TARGET.get() == self.target


@contextmanager
def _target_logging_context(target):
    token = _CURRENT_LOG_TARGET.set(str(target))
    try:
        yield
    finally:
        _CURRENT_LOG_TARGET.reset(token)


def _add_target_log_handlers(targets, target_log_root):
    """Attach one filtered file handler per target and return newly added handlers."""
    os.makedirs(target_log_root, exist_ok=True)
    root_logger = logging.getLogger()
    formatter = logging.Formatter(_LOG_FORMAT)
    existing_paths = {
        os.path.abspath(handler.baseFilename)
        for handler in root_logger.handlers
        if isinstance(handler, logging.FileHandler)
    }
    added_handlers = []

    for target in targets:
        safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", str(target)).strip("._")
        target_log_path = os.path.abspath(
            os.path.join(target_log_root, f"{safe_target or 'target'}.log")
        )
        if target_log_path in existing_paths:
            continue

        handler = logging.FileHandler(target_log_path)
        handler.setLevel(getattr(logging, getattr(config, "LOG_LEVEL", "INFO")))
        handler.setFormatter(formatter)
        handler.addFilter(_TargetLogFilter(target))
        root_logger.addHandler(handler)
        existing_paths.add(target_log_path)
        added_handlers.append(handler)

    return added_handlers


log_level = getattr(logging, getattr(config, "LOG_LEVEL", "INFO"))
logging.basicConfig(
    level=log_level,
    format=_LOG_FORMAT,
)
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

# Imported modules may have already called basicConfig, so add the framework
# file handler explicitly instead of relying on another basicConfig call.
if not any(
    isinstance(handler, logging.FileHandler)
    and os.path.abspath(handler.baseFilename) == os.path.abspath(log_file)
    for handler in root_logger.handlers
):
    framework_handler = logging.FileHandler(log_file)
    framework_handler.setLevel(log_level)
    framework_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(framework_handler)

TARGET_LOG_ROOT = os.path.join(
    RESULTS_ROOT,
    getattr(config, "TARGET_LOG_SUBDIR", os.path.join("Logs", "By_Target")),
)
if getattr(config, "SEPARATE_TARGET_LOGS", True):
    _add_target_log_handlers(
        getattr(config, "TARGET_COLUMNS", []),
        TARGET_LOG_ROOT,
    )
    logging.info("Per-target model logs enabled: %s", TARGET_LOG_ROOT)

# ------------------------------------------------------------------------------
# Safety wrapper for imputer callables
# ------------------------------------------------------------------------------
def _safe_imputer_wrapper(fn: Callable):
    """
    Wrap a model impute callable so its return is always a pd.DataFrame aligned
    with the input DataFrame's columns and index.

    The wrapped callable has the same signature:
        wrapped(data, target_column, input_columns, custom_strategies=None, **kwargs)
    and returns a pd.DataFrame (never raw ndarray).
    """

    def wrapped(data, target_column, input_columns, custom_strategies=None, **kwargs):
        try:
            res = fn(data.copy(), target_column, input_columns, custom_strategies=custom_strategies, **kwargs)
        except Exception as e:
            logging.exception("Imputer function raised an exception: %s", e)
            # return input data with target untouched as safe fallback
            out_df = data.copy()
            # if target missing create a column of NaNs so downstream saves still happen
            if target_column not in out_df.columns:
                out_df[target_column] = np.nan
            return out_df

        # If numpy array returned, convert to DataFrame preserving index/order
        if isinstance(res, np.ndarray):
            try:
                res = pd.DataFrame(
                    res,
                    index=data.index,
                    columns=[c for c in data.columns if c != 'DateTime'][:res.shape[1]]
                )
            except Exception:
                res = pd.DataFrame(res, index=data.index)

        # If Series returned, place as the target column
        if isinstance(res, pd.Series):
            out_df = data.copy()
            out_df[target_column] = pd.to_numeric(res.reindex(data.index), errors='coerce')
            return out_df

        # If DataFrame returned, ensure it contains target_column
        if isinstance(res, pd.DataFrame):
            out = res.copy()
            # If no target column present, try to infer from same-named column in input_columns
            if target_column not in out.columns:
                # if res has single column, assume it's the target
                if out.shape[1] == 1:
                    out[target_column] = pd.to_numeric(
                        out.iloc[:, 0].reindex(data.index), errors='coerce'
                    )
                else:
                    out = out.reindex(index=data.index)
                    out[target_column] = out.get(target_column, np.nan)

            # Reindex to input index if necessary
            try:
                out = out.reindex(index=data.index)
            except Exception:
                out.index = data.index

            # Ensure DateTime column is preserved from original data if not present
            if 'DateTime' not in out.columns and 'DateTime' in data.columns:
                out['DateTime'] = data['DateTime']

            return out

        # For any other return type, coerce to DataFrame and ensure target_column
        try:
            out = pd.DataFrame(res)
            if target_column not in out.columns:
                out[target_column] = np.nan
            out = out.reindex(index=data.index)
        except Exception:
            out = data.copy()
            if target_column not in out.columns:
                out[target_column] = np.nan

        return out

    return wrapped

def check_dependencies():
    """Return a dict of optional dependency availability (True/False).

    This is used to decide which heavy ML backends are available.
    """
    deps = {
        'torch': False, 'xgboost': False, 'lightgbm': False, 'sklearn': False,
        'pandas': False, 'numpy': False, 'matplotlib': False, 'seaborn': False, 'scipy': False
    }
    for d in list(deps.keys()):
        try:
            importlib.import_module(d)
            deps[d] = True
        except Exception:
            deps[d] = False
    return deps


def get_available_models(dependencies):
    """
    Determine which configured models are available to run.
    """
    pytorch_models = {"BRITS", "DLV2", "GAIN", "GRIN", "TransformerImpute", "SPIN"}
    xgboost_models = {"XGBoost"}
    lightgbm_models = {"LightGBM"}

    sklearn_models = {
        "miceV2", "SoftImpute", "OptSpace", "SVT",
        "MissForest", "GaussianProcess", "MICE", "GATI_AQ", "Mean", "Median"
    }

    configured_models = list(config.MODELS_TO_RUN) if getattr(config, "MODELS_TO_RUN", None) else []
    if not configured_models:
        fallback = getattr(config, "STANDALONE_MODELS", [])
        if fallback:
            logging.warning("MODELS_TO_RUN is empty — falling back to STANDALONE_MODELS for execution.")
            configured_models = list(fallback)
        else:
            logging.error("No models configured in MODELS_TO_RUN or STANDALONE_MODELS. Nothing to run.")
            return [], []

    available = []
    skipped = []

    for model in configured_models:
        if model in sklearn_models:
            available.append(model)
        elif model in pytorch_models:
            if dependencies.get('torch'):
                available.append(model)
            else:
                logging.warning(
                    f"PyTorch not found: {model} will still be scheduled and is expected "
                    f"to use sklearn fallback (if implemented in the module)."
                )
                available.append(model)
        elif model in xgboost_models:
            if dependencies.get('xgboost'):
                available.append(model)
            else:
                logging.warning(f"xgboost not available, skipping model: {model}")
                skipped.append(model)
        elif model in lightgbm_models:
            if dependencies.get('lightgbm'):
                available.append(model)
            else:
                logging.warning(f"lightgbm not available, skipping model: {model}")
                skipped.append(model)
        else:
            available.append(model)

    return list(available), list(skipped)


# ------------------------------------------------------------------------------
# Dynamic model loader (imputer discovery + fallback names)
# ------------------------------------------------------------------------------
def load_model_module(model_name):
    """
    Load model module from Model.<model_name> or direct import.
    Provides detailed logging about what was found (module file, attributes).
    """
    try:
        mod = importlib.import_module(f"Model.{model_name}")
    except Exception as e1:
        try:
            mod = importlib.import_module(model_name)
        except Exception as e2:
            try:
                spec = importlib.util.find_spec('Model')
                if spec and spec.submodule_search_locations:
                    model_pkg_dir = spec.submodule_search_locations[0]
                    from pathlib import Path
                    candidates = []
                    for p in Path(model_pkg_dir).glob('*.py'):
                        stem = p.stem.lower()
                        if model_name.lower() in stem or stem in model_name.lower():
                            candidates.append(p.stem)
                    if candidates:
                        cand = sorted(candidates, key=lambda s: (s != model_name, s))[0]
                        mod = importlib.import_module(f"Model.{cand}")
                    else:
                        raise ImportError(f"No matching module under Model/ for '{model_name}'")
                else:
                    raise
            except Exception as e3:
                logging.error(
                    f"❌ Could not load model '{model_name}': "
                    f"ModelImportError: {e1} | FallbackImportError: {e2} | DirSearchError: {e3}"
                )
                return None

    mod_file = getattr(mod, "__file__", "builtin/unknown")
    logging.info(f"Loaded model module '{model_name}' from: {mod_file}")

    has_impute = hasattr(mod, "impute_mice")
    model_name_attr = getattr(mod, "MODEL_NAME", None)
    logging.info(f"  - exposes impute_mice: {has_impute}")
    logging.info(f"  - MODEL_NAME attribute: {model_name_attr}")

    if not has_impute:
        logging.warning(
            f"Module '{model_name}' does not implement 'impute_mice(data, target, input_cols, ...)' — will be skipped."
        )

    return mod


def get_impute_callable(model_module):
    """
    Return a callable that performs imputation for the given model module.
    Tries a set of common function names.
    """
    candidates = [
        "impute_mice", "impute", "run_impute", "impute_values", "predict", "transform"
    ]
    for name in candidates:
        fn = getattr(model_module, name, None)
        if callable(fn):
            logging.info(
                f"Using imputation function '{name}' from module "
                f"'{getattr(model_module, 'MODEL_NAME', getattr(model_module, '__name__', 'unknown'))}'"
            )
            def wrapper(data, target_column, input_columns, custom_strategies=None, **kwargs):
                optional_kwargs = {"custom_strategies": custom_strategies, **kwargs}
                try:
                    parameters = inspect.signature(fn).parameters
                    accepts_arbitrary_kwargs = any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                    if not accepts_arbitrary_kwargs:
                        optional_kwargs = {
                            key: value for key, value in optional_kwargs.items()
                            if key in parameters
                        }
                except (TypeError, ValueError):
                    # Some extension callables do not expose an inspectable
                    # signature. Preserve the historical call for those.
                    pass
                return fn(data.copy(), target_column, input_columns, **optional_kwargs)
            return wrapper

    for attr in dir(model_module):
        obj = getattr(model_module, attr)
        if hasattr(obj, "impute") and callable(getattr(obj, "impute")):
            logging.info(
                f"Using class method 'impute' from '{attr}' in module '{model_module.__name__}'"
            )
            def wrapper(data, target_column, input_columns, custom_strategies=None, **kwargs):
                instance = obj()
                return instance.impute(
                    data.copy(), target_column, input_columns,
                    custom_strategies=custom_strategies, **kwargs
                )
            return wrapper

    return None


def validate_data_for_imputation(data, target_column, input_columns):
    """
    Validate that data is suitable for imputation.
    Prevents errors from all-NaN columns and missing features.
    """
    errors = []

    if target_column not in data.columns:
        errors.append(f"Target column '{target_column}' not found")
        return errors

    target_obs = data[target_column].notna().sum()
    if target_obs == 0:
        errors.append(
            f"Target column '{target_column}' is 100% missing "
            f"(no observations to learn from). Cannot impute."
        )

    for col in input_columns:
        if col not in data.columns:
            errors.append(f"Input column '{col}' not found in data")
        elif data[col].notna().sum() == 0:
            errors.append(f"Input column '{col}' is 100% missing")

    try:
        available_rows = data[[target_column] + input_columns].notna().all(axis=1).sum()
    except Exception:
        available_rows = 0

    if available_rows < 10:
        errors.append(
            f"Only {available_rows} complete rows available for training "
            f"(need at least 10)"
        )

    return errors


def check_data_availability(input_directory):
    """Check data availability across all sites and targets."""
    logging.info("="*80)
    logging.info("CHECKING DATA AVAILABILITY")
    logging.info("="*80)

    for filename in os.listdir(input_directory):
        if not filename.endswith('.csv'):
            continue

        try:
            df = pd.read_csv(os.path.join(input_directory, filename), nrows=100)
            logging.info(f"\n{filename}:")
            logging.info(f"  Rows: {len(df)}")
            logging.info(f"  Columns: {list(df.columns)}")

            for target in ['CO', 'PM2.5', 'PM10']:
                if target in df.columns:
                    obs = df[target].notna().sum()
                    total = len(df)
                    pct = (obs / total) * 100 if total > 0 else 0.0
                    status = "✅" if obs > 0 else "❌"
                    logging.info(f"  {status} {target}: {obs}/{total} ({pct:.1f}% observed)")
        except Exception as e:
            logging.error(f"  Error reading {filename}: {e}")


def verify_run_outputs(model_out_dir, site_file, target, model_name, regimes, missingness_levels):
    """
    Verify that expected outputs exist for each regime and missingness level.

    Logs warnings for any missing files. Returns False if any expected file is missing.

    This version is aligned with sortdata.py naming:
      metrics: <site>_<target>_<MODEL>_<regime>_<miss_pct>_all_metrics.csv
      imputed: <site>_<target>_<MODEL>_<regime>_imputed_<miss_pct>.csv
      target:  <site>_<target>_<MODEL>_<regime>_target_column_<miss_pct>.csv

    It first checks for those exact names before falling back to fuzzy glob checks.
    Placeholder creation can be disabled via config.SAVE_PLACEHOLDER_VERIFICATION_FILES.
    """
    import config_spatial as cfg

    success = True
    site_basename = os.path.basename(site_file).split('.')[0]

    def dir_has_tolerant_file(dirpath, substrings):
        """Return True if any filename in `dirpath` contains all substrings (case-insensitive)."""
        try:
            if not os.path.isdir(dirpath):
                return False
            for fn in os.listdir(dirpath):
                low = fn.lower()
                if all(s.lower() in low for s in substrings):
                    return True
            return False
        except Exception:
            return False

    # allow model_name to be passed as MODEL_SITE; derive base model token
    model_base = model_name.split('_')[0] if isinstance(model_name, str) else model_name
    # short site token (first token before underscore) to match filenames like 'CHULLORA' vs 'CHULLORA_AQMS_Processed'
    short_site = site_basename.split('_')[0] if isinstance(site_basename, str) and '_' in site_basename else site_basename
    # Choose a display_model that does not duplicate the site token if present
    display_model = model_name
    try:
        if isinstance(model_name, str) and isinstance(site_basename, str) and site_basename.lower() in model_name.lower():
            display_model = model_base
    except Exception:
        display_model = model_base

    save_placeholders = getattr(cfg, "SAVE_PLACEHOLDER_VERIFICATION_FILES", False)

    for regime in regimes:
        metrics_dir = os.path.join(model_out_dir, regime, "metrics")
        imputed_dir = os.path.join(model_out_dir, regime, "imputed_data")
        target_col_dir = os.path.join(model_out_dir, regime, "target_column_data")
        central_master_used = False

        for miss in missingness_levels:
            miss_pct = int(miss * 100)
            missing_files = []

            # ------------------------------------------------------------------
            # METRICS: exact filename first, then tolerant fallback
            # ------------------------------------------------------------------
            expected_metrics_name = f"{site_basename}_{target}_{model_base}_{regime}_{miss_pct}_all_metrics.csv"
            expected_metrics_path = os.path.join(metrics_dir, expected_metrics_name)

            metrics_missing = False
            if os.path.isfile(expected_metrics_path):
                # Real per-level metrics exists; this is the primary success condition
                metrics_missing = False
                logging.info(
                    f"verify_run_outputs: found metrics file {expected_metrics_name} "
                    f"for {model_base}_{short_site} | {regime} | {miss_pct}%"
                )
            else:
                # Fallback: tolerant substring/glob checks (older logic)
                metrics_subs_candidates = [
                    [site_basename, target, model_name, regime, str(miss_pct)],
                    [site_basename, target, model_base, regime, str(miss_pct)],
                    [short_site, target, model_name, regime, str(miss_pct)],
                    [short_site, target, model_base, regime, str(miss_pct)],
                ]
                if not dir_has_tolerant_file(metrics_dir, [site_basename, model_base, regime, str(miss_pct)]):
                    if not any(dir_has_tolerant_file(metrics_dir, subs) for subs in metrics_subs_candidates):
                        metrics_missing = True

            # ------------------------------------------------------------------
            # IMPUTED: same tolerant logic as before
            # ------------------------------------------------------------------
            imputed_subs_candidates = [
                [site_basename, target, model_name, 'imputed', str(miss_pct)],
                [site_basename, target, model_base, 'imputed', str(miss_pct)],
                [short_site, target, model_name, 'imputed', str(miss_pct)],
                [short_site, target, model_base, 'imputed', str(miss_pct)],
            ]
            def any_dir_has(candidates, d):
                for subs in candidates:
                    if dir_has_tolerant_file(d, subs):
                        return True
                return False

            imputed_found = any_dir_has(imputed_subs_candidates, imputed_dir)
            if not imputed_found:
                alt_imputed_subs = [target, model_base, 'imputed', str(miss_pct)]
                if dir_has_tolerant_file(imputed_dir, alt_imputed_subs):
                    imputed_found = True

            # Central master fallback (unchanged from your previous code)
            if not imputed_found:
                try:
                    central_imputed_dir = os.path.join(cfg.OUTPUT_DIRECTORY, 'Imputed_Results')
                    central_subs = [site_basename, target, 'master_imputed']
                    short_site_token = site_basename.split('_')[0] if isinstance(site_basename, str) and '_' in site_basename else site_basename
                    central_subs_alt = [short_site_token, target, 'master_imputed']
                    if dir_has_tolerant_file(central_imputed_dir, central_subs) or dir_has_tolerant_file(central_imputed_dir, central_subs_alt):
                        logging.info(f"Found centralized per-site_target master for {site_basename} {target}; accepting as imputed output")
                        src_fp = None
                        try:
                            for fn in os.listdir(central_imputed_dir):
                                low = fn.lower()
                                if all(s.lower() in low for s in central_subs) or all(s.lower() in low for s in central_subs_alt):
                                    src_fp = os.path.join(central_imputed_dir, fn)
                                    break
                        except Exception:
                            src_fp = None

                        if src_fp and os.path.exists(src_fp):
                            try:
                                os.makedirs(imputed_dir, exist_ok=True)
                                dest_fn = f"{model_base}_{site_basename}_{target}_{regime}_imputed_{miss_pct}.csv"
                                dest_fp = os.path.join(imputed_dir, dest_fn)
                                shutil.copy2(src_fp, dest_fp)
                                logging.info(f"Copied central master {os.path.basename(src_fp)} -> {dest_fp}")
                                imputed_found = True
                                central_master_used = True
                            except Exception as e:
                                logging.debug(f"Failed to copy central master: {e}")
                                imputed_found = True
                                central_master_used = True
                        else:
                            imputed_found = True
                            central_master_used = True
                except Exception:
                    pass

            # ------------------------------------------------------------------
            # Optionally create placeholders (now controlled by config flag)
            # ------------------------------------------------------------------
            if imputed_found and metrics_missing and save_placeholders:
                try:
                    # representative imputed file
                    candidate_fp = None
                    if os.path.isdir(imputed_dir):
                        for fn in os.listdir(imputed_dir):
                            low = fn.lower()
                            if (site_basename.lower() in low and
                                target.lower() in low and
                                regime.lower() in low and
                                str(miss_pct) in low):
                                candidate_fp = os.path.join(imputed_dir, fn)
                                break
                        if candidate_fp is None:
                            for fn in os.listdir(imputed_dir):
                                if fn.lower().endswith('.csv'):
                                    candidate_fp = os.path.join(imputed_dir, fn)
                                    break

                    # metrics placeholder
                    try:
                        os.makedirs(metrics_dir, exist_ok=True)
                        metrics_dest = os.path.join(metrics_dir, f"{site_basename}_{model_base}_{regime}_{miss_pct}_metrics.csv")
                        if not os.path.exists(metrics_dest):
                            with open(metrics_dest, 'w') as mf:
                                mf.write('metric,value\n')
                                mf.write('generated_from_imputed,True\n')
                            logging.info(f"Wrote placeholder metrics: {metrics_dest}")
                    except Exception as e:
                        logging.debug(f"Could not write placeholder metrics: {e}")

                    # target_column placeholder
                    try:
                        os.makedirs(target_col_dir, exist_ok=True)
                        target_dest = os.path.join(
                            target_col_dir,
                            f"StudySites_{model_base}_{site_basename}_{target}_{regime}_target_column_{miss_pct}.csv"
                        )
                        if candidate_fp and os.path.exists(candidate_fp):
                            try:
                                df_imp = pd.read_csv(candidate_fp, low_memory=False)
                                dt_col = 'DateTime' if 'DateTime' in df_imp.columns else next(
                                    (c for c in df_imp.columns if 'date' in c.lower()), None
                                )
                                cols_to_save = []
                                if dt_col:
                                    cols_to_save.append(dt_col)
                                if target in df_imp.columns:
                                    cols_to_save.append(target)
                                elif target.replace('.', '').lower() in [c.replace('.', '').lower() for c in df_imp.columns]:
                                    match = next(
                                        (c for c in df_imp.columns
                                         if target.replace('.', '').lower() == c.replace('.', '').lower()),
                                        None
                                    )
                                    if match:
                                        cols_to_save.append(match)

                                if cols_to_save:
                                    df_imp[cols_to_save].to_csv(target_dest, index=False)
                                    logging.info(f"Wrote target_column placeholder: {target_dest}")
                                else:
                                    with open(target_dest, 'w') as tf:
                                        tf.write('DateTime,%s\n' % target)
                                    logging.info(f"Wrote empty target_column placeholder: {target_dest}")
                            except Exception as e:
                                logging.debug(f"Failed to extract target column from {candidate_fp}: {e}")
                                if not os.path.exists(target_dest):
                                    with open(target_dest, 'w') as tf:
                                        tf.write('DateTime,%s\n' % target)
                        else:
                            if not os.path.exists(target_dest):
                                with open(target_dest, 'w') as tf:
                                    tf.write('DateTime,%s\n' % target)
                                logging.info(f"Wrote empty target_column placeholder: {target_dest}")
                    except Exception as e:
                        logging.debug(f"Could not write target_column placeholder: {e}")
                except Exception:
                    pass

            # ------------------------------------------------------------------
            # Final missingness decision for this (model,site,target,regime,miss)
            # ------------------------------------------------------------------
            if not imputed_found:
                missing_files.append(os.path.join(imputed_dir, f"*{site_basename}*{model_base}*imputed*{miss_pct}*.csv"))

            if not central_master_used:
                if metrics_missing:
                    missing_files.append(expected_metrics_path)
                target_subs_candidates = [
                    [site_basename, target, model_name, 'target_column', str(miss_pct)],
                    [site_basename, target, model_base, 'target_column', str(miss_pct)],
                    [short_site, target, model_name, 'target_column', str(miss_pct)],
                    [short_site, target, model_base, 'target_column', str(miss_pct)],
                ]
                if not any_dir_has(target_subs_candidates, target_col_dir):
                    alt_target_subs = [target, model_base, 'target_column', str(miss_pct)]
                    if not dir_has_tolerant_file(target_col_dir, alt_target_subs):
                        missing_files.append(os.path.join(
                            target_col_dir,
                            f"*{site_basename}*{model_base}*target_column*{miss_pct}*.csv"
                        ))

            if missing_files:
                logging.warning(
                    f"Missing output files for {model_base}_{short_site} | {site_basename} | "
                    f"{regime} | {miss_pct}%:\n  " + "\n  ".join(missing_files)
                )
                success = False
            else:
                logging.info(
                    f"Outputs present for {model_base}_{short_site} | {site_basename} | "
                    f"{regime} | {miss_pct}%"
                )

    return success

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():

    start_time = datetime.now()
    logging.info("=" * 80)
    logging.info("SPATIAL–TEMPORAL IMPUTATION FRAMEWORK")
    logging.info("WITH MULTIPLE MISSINGNESS REGIMES")
    logging.info("=" * 80)

    # Ensure central roots exist. Plot roots are created only when plot output is enabled.
    central_roots = [MODEL_OUTPUT_ROOT, CENTRAL_METRICS_ROOT, CENTRAL_IMPUTED_ROOT]
    if getattr(config, "SAVE_PLOTS", False):
        central_roots.append(CENTRAL_PLOTS_ROOT)
    for d in central_roots:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--sites', nargs='+')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--generate-report', action='store_true')
    parser.add_argument('--generate-research', action='store_true',
                       help='Generate research-grade plots from results (requires research_plots.py)')
    parser.add_argument('--regime', type=str, default=None,
                       help='Run specific regime only (random, short_gap, medium_gap, long_gap, event)')
    parser.add_argument(
        '--refresh-api-inputs',
        action='store_true',
        help='Download the missing live AQ API tail into the wide regional CSVs before imputation.',
    )
    parser.add_argument(
        '--skip-api-refresh',
        action='store_true',
        help='Use the existing regional CSVs without contacting the AQ API (offline runs only).',
    )
    parser.add_argument(
        '--wide-input-dir',
        default=NOWCASTING_WIDE_INPUT_DIR,
        help='Directory containing Allobs_processed_DPE_station_api_*_ALL.csv wide API inputs.',
    )
    parser.add_argument(
        '--target-workers',
        type=int,
        default=None,
        help='Regional pooled target tasks to run concurrently (default: config setting).',
    )
    parser.add_argument(
        '--model-n-jobs',
        type=int,
        default=None,
        help='Native model threads per target worker; 0 divides allocated CPUs automatically.',
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        default=None,
        help='Override config.TARGET_COLUMNS for this run, e.g. --targets PM10 PM2.5 NO2.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Override config.MODELS_TO_RUN for this run, e.g. --models AQUISTIL LightGBM.',
    )
    parser.add_argument(
        '--run-aquistil-ablations',
        action='store_true',
        help=(
            'Run full AQUISTIL, every configured AQUISTIL ablation, and LightGBM; '
            'save their metrics and paired summaries separately.'
        ),
    )
    parser.add_argument(
        '--regions',
        nargs='+',
        default=None,
        help='Override selected regions for this run, e.g. --regions "Sydney North-west".',
    )
    parser.add_argument(
        '--missingness-levels',
        nargs='+',
        type=float,
        default=None,
        help='Override missingness fractions, e.g. --missingness-levels 0.05 0.10.',
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=None,
        help='Override regional evaluation seeds for a smoke or reproducibility run.',
    )
    parser.add_argument(
        '--gap-expert-min-run-length',
        type=int,
        choices=[2, 3, 6, 12, 24],
        default=None,
        help='Development-only global AQUISTIL gap-router threshold in hours.',
    )
    args = parser.parse_args()

    if args.run_aquistil_ablations and args.models:
        raise ValueError(
            "--run-aquistil-ablations cannot be combined with --models"
        )

    if args.targets:
        config.TARGET_COLUMNS = [str(target).strip() for target in args.targets if str(target).strip()]
        if not config.TARGET_COLUMNS:
            raise ValueError("--targets was supplied but no valid target names were found")
        if getattr(config, "SEPARATE_TARGET_LOGS", True):
            _add_target_log_handlers(config.TARGET_COLUMNS, TARGET_LOG_ROOT)
        logging.info("Overriding TARGET_COLUMNS from command line: %s", config.TARGET_COLUMNS)

    if args.models:
        config.MODELS_TO_RUN = [str(model).strip() for model in args.models if str(model).strip()]
        config.STANDALONE_MODELS = []
        config.COMPARISON_MODELS = list(dict.fromkeys(config.MODELS_TO_RUN))
        if not config.MODELS_TO_RUN:
            raise ValueError("--models was supplied but no valid model names were found")
        logging.info("Overriding MODELS_TO_RUN from command line: %s", config.MODELS_TO_RUN)

    if args.run_aquistil_ablations:
        config.RUN_AQUISTIL_ABLATIONS = True
        config.MODELS_TO_RUN = list(
            dict.fromkeys(
                ["AQUISTIL"]
                + _aquistil_ablation_models()
                + list(getattr(config, "PAPER_BASELINE_MODELS", ["LightGBM"]))
            )
        )
        config.STANDALONE_MODELS = []
        config.COMPARISON_MODELS = list(config.MODELS_TO_RUN)
        logging.info(
            "AQUISTIL ablation run enabled; models: %s", config.MODELS_TO_RUN
        )

    if args.regions:
        config.SELECT_TARGET_REGIONS = [
            str(region).strip() for region in args.regions if str(region).strip()
        ]
        if not config.SELECT_TARGET_REGIONS:
            raise ValueError("--regions was supplied but no valid region names were found")
        logging.info(
            "Overriding SELECT_TARGET_REGIONS from command line: %s",
            config.SELECT_TARGET_REGIONS,
        )

    if args.missingness_levels is not None:
        if not args.missingness_levels or any(
            level <= 0 or level > 1 for level in args.missingness_levels
        ):
            raise ValueError("--missingness-levels values must be in (0, 1]")
        config.MISSINGNESS_LEVELS = list(dict.fromkeys(args.missingness_levels))
        logging.info(
            "Overriding MISSINGNESS_LEVELS from command line: %s",
            config.MISSINGNESS_LEVELS,
        )
    if args.seeds is not None:
        if not args.seeds:
            raise ValueError("--seeds requires at least one integer")
        config.REGIONAL_EVALUATION_SEEDS = list(dict.fromkeys(args.seeds))
        logging.info(
            "Overriding REGIONAL_EVALUATION_SEEDS from command line: %s",
            config.REGIONAL_EVALUATION_SEEDS,
        )
    if args.gap_expert_min_run_length is not None:
        config.AQUISTIL_GAP_EXPERT_MIN_RUN_LENGTH = args.gap_expert_min_run_length
        logging.info(
            "Overriding AQUISTIL gap-expert threshold: %d hours",
            config.AQUISTIL_GAP_EXPERT_MIN_RUN_LENGTH,
        )

    _validate_frozen_validation_protocol(args)

    wide_input_dir = os.path.abspath(args.wide_input_dir)

    # ----------------------------------------------------------------------
    # Wide nowcasting inputs: keep them up to date before materializing the
    # per-site model inputs. --skip-api-refresh is the explicit offline escape.
    # ----------------------------------------------------------------------
    if getattr(config, "USE_WIDE_NOWCASTING_INPUTS", True) and (
        args.refresh_api_inputs
        or (getattr(config, "AUTO_UPDATE_WIDE_INPUTS", False) and not args.skip_api_refresh)
    ):
        try:
            configured_regions = list(
                getattr(config, "SELECT_TARGET_REGIONS", []) or getattr(config, "TARGET_REGIONS", []) or []
            )
            from nsw_air_quality.incremental_region_input_updater import update_nsw_region_input_files

            if configured_regions:
                logging.info(
                    "Updating wide API inputs in %s for configured regions: %s",
                    wide_input_dir,
                    ", ".join(configured_regions),
                )
                update_results = update_nsw_region_input_files(
                    wide_input_dir,
                    only_region_tokens=configured_regions,
                    raise_on_error=True,
                )
            else:
                logging.info("Updating all wide API inputs under %s", wide_input_dir)
                update_results = update_nsw_region_input_files(wide_input_dir, raise_on_error=True)

            if not update_results:
                raise RuntimeError(
                    "No regional wide-input CSVs matched the requested regions under %s"
                    % wide_input_dir
                )

            updated_count = sum(1 for changed in update_results.values() if changed)
            logging.info(
                "Wide API refresh finished: %d/%d file(s) updated under %s",
                updated_count,
                len(update_results),
                wide_input_dir,
            )
        except Exception as e:
            # The NSW endpoint can reject this host (for example, HTTP 403
            # "Ip Forbidden"). In that case continue with a validated local
            # snapshot rather than aborting the entire forecast pipeline.
            from nsw_air_quality.incremental_region_input_updater import (
                validate_existing_wide_region_files,
            )

            try:
                fallback_coverage = validate_existing_wide_region_files(
                    wide_input_dir,
                    only_region_tokens=configured_regions or None,
                )
            except Exception:
                logging.error(
                    "Wide input update failed and no valid local fallback is available: %s",
                    e,
                )
                raise

            logging.warning(
                "Wide input API refresh failed; continuing with %d validated local "
                "file(s) under %s. API error: %s",
                len(fallback_coverage),
                wide_input_dir,
                e,
            )
            for fallback_path, (first_ts, last_ts, row_count) in fallback_coverage.items():
                logging.warning(
                    "Local wide-input fallback: %s | coverage=%s to %s | rows=%d",
                    fallback_path,
                    first_ts.isoformat(),
                    last_ts.isoformat(),
                    row_count,
                )

    # Diagnostic: log basic data availability for first-pass troubleshooting
    try:
        input_dir_for_check = wide_input_dir if getattr(config, "USE_WIDE_NOWCASTING_INPUTS", True) else config.INPUT_DIRECTORY
        check_data_availability(input_dir_for_check)
    except Exception as e:
        logging.debug(f"check_data_availability failed: {e}")

    dependencies = check_dependencies()
    available_models, skipped_models = get_available_models(dependencies)

    # Determine models_to_run from config, preferring explicit MODELS_TO_RUN.
    models_to_run = []
    if getattr(config, "MODELS_TO_RUN", None):
        models_to_run = list(config.MODELS_TO_RUN)
    else:
        models_to_run = list(getattr(config, "STANDALONE_MODELS", []))

    # Ensure STANDALONE_MODELS are appended if not already present
    if getattr(config, "STANDALONE_MODELS", None):
        for m in config.STANDALONE_MODELS:
            if m not in models_to_run:
                models_to_run.append(m)

    if models_to_run:
        filtered = [m for m in models_to_run if m in available_models]
        if filtered:
            available_models = filtered
        else:
            logging.warning(
                "Requested models from config not available; using dependency-detected models: %s",
                available_models
            )

    # Optional cap for quicker runs
    try:
        max_models = int(getattr(config, "MAX_MODELS_TO_RUN", 0) or 0)
    except Exception:
        max_models = 0
    if max_models and max_models > 0:
        available_models = list(available_models)[:max_models]
        logging.info("Capping models to first %d: %s", max_models, available_models)

    if not available_models:
        logging.error("No models available.")
        return

    # ----------------------------------------------------------------------
    # Resolve selected regions/sites from config overrides
    # ----------------------------------------------------------------------
    target_regions = list(getattr(config, "TARGET_REGIONS", []) or [])
    selected_regions = list(getattr(config, "SELECT_TARGET_REGIONS", []) or [])
    using_selected_regions = bool(selected_regions)
    if using_selected_regions:
        target_regions = selected_regions

    selected_sites_override = list(getattr(config, "SELECT_TARGET_SITES", []) or [])

    # Load best-predictors mapping once (optional)
    best_predictors_map = {}
    if getattr(config, "USE_BEST_PREDICTORS_JSON_INPUTS", False):
        best_predictors_map = _load_best_predictors_json(getattr(config, "BEST_PREDICTORS_JSON", ""))

    progressive_best_map = {}
    if getattr(config, "USE_PROGRESSIVE_BEST_FEATURES", False):
        progressive_best_map = _load_progressive_best_features_for_targets(
            list(getattr(config, "TARGET_COLUMNS", []) or [])
        )
        logging.info("Loaded %d Stage 3 best region/target configuration(s)", len(progressive_best_map))

    # ----------------------------------------------------------------------
    # Wide-input region/site inventory
    # ----------------------------------------------------------------------
    wide_region_files = []
    wide_site_index = {}
    prepared_site_inputs = {}
    progressive_neighbor_inputs = {}
    if getattr(config, "USE_WIDE_NOWCASTING_INPUTS", True):
        wide_region_files = _resolve_region_wide_files(target_regions, wide_dir=wide_input_dir)
        if not wide_region_files:
            wide_region_files = _resolve_region_wide_files(None, wide_dir=wide_input_dir)
            if wide_region_files:
                logging.info(
                    "Falling back to all available wide regional inputs in %s because none matched the configured region selection.",
                    wide_input_dir,
                )

        wide_site_index = _build_site_region_index(wide_region_files)
        available_sites = sorted({meta["site_name"] for meta in wide_site_index.values()})

        if not available_sites:
            logging.warning(
                "No sites were discovered from wide inputs under %s; falling back to legacy INPUT_DIRECTORY",
                wide_input_dir,
            )
            available_sites = get_available_sites(config.INPUT_DIRECTORY)
        else:
            target_cache_token = _target_run_token()
            prepared_wide_site_dir = os.path.join(
                RESULTS_ROOT,
                "Inputs_PerSite",
                target_cache_token,
            )
            prepared_site_inputs = _materialize_per_site_cache(wide_region_files, prepared_wide_site_dir)
            if prepared_site_inputs:
                config.INPUT_DIRECTORY = prepared_wide_site_dir
                logging.info("Spatial feature input directory set to prepared per-site cache: %s", prepared_wide_site_dir)

            needs_idw = any(
                any(str(feature).startswith("IDW_Spatial_") for feature in choice.get("features", []))
                for choice in progressive_best_map.values()
            )
            if needs_idw:
                all_region_files = _resolve_region_wide_files(None, wide_dir=wide_input_dir)
                idw_neighbor_dir = os.path.join(
                    RESULTS_ROOT,
                    "Inputs_IDW_Neighbors",
                    target_cache_token,
                )
                progressive_neighbor_inputs = _materialize_per_site_cache(
                    all_region_files,
                    idw_neighbor_dir,
                )
                logging.info(
                    "Prepared %d all-region site inputs for Stage 3-compatible IDW features",
                    len(progressive_neighbor_inputs),
                )
    else:
        available_sites = get_available_sites(config.INPUT_DIRECTORY)

    # --------------------------------------------------------------------------
    # Site selection
    # --------------------------------------------------------------------------
    if args.all:
        sites_to_process = available_sites
    elif args.sites:
        wanted = [str(x).strip() for x in args.sites if str(x).strip()]
        wanted_norm = {_canon_token(x) for x in wanted}
        sites_to_process = [s for s in available_sites if _canon_token(s) in wanted_norm]
        # If the requested sites are not discoverable (e.g., limited region list),
        # run the explicit site names as provided.
        if not sites_to_process and wanted:
            sites_to_process = wanted
    else:
        if selected_sites_override:
            sites_to_process = selected_sites_override
        else:
            # If SELECT_TARGET_REGIONS is set, prefer the sites discovered from
            # those selected region-wide input files (avoids pulling stale/
            # broader TARGET_SITES computed at import-time from full TARGET_REGIONS).
            if using_selected_regions:
                sites_to_process = available_sites
            else:
                sites_to_process = config.TARGET_SITES or available_sites

    # --------------------------------------------------------------------------
    # Regime selection
    # --------------------------------------------------------------------------
    if args.regime:
        regimes_to_run = [args.regime]
        logging.info(f"Running single regime: {args.regime}")
    else:
        regimes_to_run = config.MISSINGNESS_REGIMES
        logging.info(f"Running ALL regimes: {regimes_to_run}")

    logging.info(f"Sites:  {sites_to_process}")
    logging.info(f"Models: {available_models}")
    logging.info(f"Missingness levels: {[int(m*100) for m in config.MISSINGNESS_LEVELS]}%")

    # ----------------------------------------------------------------------
    # Regional pooled execution: one model per region, with exactly the same
    # artificial-missing count contributed by every eligible study site.
    # ----------------------------------------------------------------------
    if getattr(config, "REGIONAL_POOLED_MODE", False):
        from regional_imputation import run_balanced_regional_task

        regional_output_root = os.path.join(RESULTS_ROOT, "Regional_Pooled_Imputation")
        os.makedirs(regional_output_root, exist_ok=True)
        regional_datasets = {}
        for region_token, _ in wide_region_files:
            for target in config.TARGET_COLUMNS:
                choice = progressive_best_map.get(
                    (_canon_token(region_token), _canon_token(target))
                )
                choice = dict(choice) if choice else _borrow_stage3_regional_feature_choice(
                    progressive_best_map, region_token, target
                )
                if not choice:
                    logging.warning(
                        "No Stage 3 feature selection for region %s / target %s; skipping regional model",
                        region_token,
                        target,
                    )
                    continue
                if choice.get("feature_source") == "stage3_borrowed":
                    logging.warning(
                        "No Stage 3 feature selection for region %s / target %s; borrowing Stage 3 features from %s: %s",
                        region_token,
                        target,
                        choice.get("source_region", "available region"),
                        choice["features"],
                    )
                if choice.get("feature_source") == "generic_fallback":
                    logging.warning(
                        "No Stage 3 feature selection for any region/target match; using generic regional features for %s / %s: %s",
                        region_token,
                        target,
                        choice["features"],
                    )
                site_parts = []
                for site in sites_to_process:
                    meta = wide_site_index.get(_canon_token(site))
                    if not meta or _canon_token(meta["region_token"]) != _canon_token(region_token):
                        continue
                    path = prepared_site_inputs.get(_canon_token(site))
                    if not path:
                        continue
                    try:
                        site_data = pd.read_csv(path)
                    except EmptyDataError:
                        logging.warning(
                            "Skipping empty per-site cache file for %s / %s / %s: %s",
                            region_token,
                            site,
                            target,
                            path,
                        )
                        continue
                    site_data["DateTime"] = pd.to_datetime(site_data["DateTime"], errors="coerce")
                    site_data = _add_progressive_derived_features(
                        site_data,
                        site,
                        target,
                        choice["features"],
                        progressive_neighbor_inputs or prepared_site_inputs,
                    )
                    site_data["Site"] = site
                    site_data["Region"] = region_token.replace("_", " ")
                    site_parts.append(site_data)
                if site_parts:
                    regional_data = pd.concat(site_parts, ignore_index=True)
                    _write_air_quality_qc_report(regional_data, RESULTS_ROOT)
                    if choice.get("feature_source") in {"generic_fallback", "stage3_borrowed"}:
                        minimum_observations = int(
                            getattr(config, "REGIONAL_GENERIC_MIN_FEATURE_OBSERVATIONS", 50)
                        )
                        usable_features = []
                        coverage = {}
                        for feature in choice["features"]:
                            if feature not in regional_data.columns:
                                continue
                            observed_values = int(
                                pd.to_numeric(regional_data[feature], errors="coerce").notna().sum()
                            )
                            coverage[feature] = observed_values
                            if observed_values >= minimum_observations:
                                usable_features.append(feature)
                        choice["features"] = usable_features
                        if not usable_features:
                            logging.warning(
                                "Regional fallback for %s / %s has no usable borrowed/generic features; skipping regional model",
                                region_token,
                                target,
                            )
                            continue
                        logging.info(
                            "Regional fallback feature coverage for %s / %s: %s",
                            region_token,
                            target,
                            coverage,
                        )
                    regional_datasets[(region_token, target)] = (regional_data, choice)

        if not regional_datasets:
            raise RuntimeError("No regional pooled datasets could be constructed")

        requested_target_workers = (
            args.target_workers
            if args.target_workers is not None
            else getattr(config, "TARGET_PARALLEL_WORKERS", 1)
        )
        requested_model_n_jobs = (
            args.model_n_jobs
            if args.model_n_jobs is not None
            else getattr(config, "MODEL_N_JOBS", 0)
        )
        target_workers, model_n_jobs = _resolve_target_parallelism(
            target_count=len({target for _, target in regional_datasets}),
            task_count=len(regional_datasets),
            requested_workers=requested_target_workers,
            requested_model_n_jobs=requested_model_n_jobs,
            available_cpus=_allocated_cpu_count(),
        )
        config.MODEL_N_JOBS = model_n_jobs
        config.USE_SPATIAL_FEATURES = False
        config.USE_TEMPORAL_FEATURES = False
        config.STRICT_PROGRESSIVE_FEATURE_LIST = False
        logging.info(
            "Regional target parallelism: workers=%d, model threads/worker=%d, tasks=%d",
            target_workers,
            model_n_jobs,
            len(regional_datasets),
        )

        regional_metric_frames = []
        regional_prediction_frames = []

        def record_regional_result(regional_result):
            """Publish one completed worker result from the main thread."""
            task_metrics = regional_result.get("metrics") if regional_result else None
            if task_metrics is not None and not task_metrics.empty:
                os.makedirs(CENTRAL_METRICS_ROOT, exist_ok=True)
                _upsert_aquistil_ablation_metrics(task_metrics, CENTRAL_METRICS_ROOT)
                standard_metrics = _without_aquistil_ablations(task_metrics)
                task_metrics = standard_metrics
                if not task_metrics.empty:
                    regional_metric_frames.append(task_metrics)
                    _upsert_regional_metrics_by_target(task_metrics, CENTRAL_METRICS_ROOT)
            if task_metrics is not None and not task_metrics.empty:
                regional_metrics_path = os.path.join(
                    CENTRAL_METRICS_ROOT, "regional_pooled_metrics.csv"
                )
                with _exclusive_file_lock(regional_metrics_path + ".lock"):
                    regional_metrics = aggregate_metrics.upsert_regional_metrics(
                        regional_metrics_path,
                        task_metrics,
                    )
                    regional_metrics = _without_aquistil_ablations(regional_metrics)
                    regional_metrics.to_csv(regional_metrics_path, index=False)
                    _write_regional_metrics_by_target(regional_metrics, CENTRAL_METRICS_ROOT)
                    logging.info(
                        "Incrementally saved central regional metrics: %s "
                        "(%d current rows, %d total rows)",
                        regional_metrics_path,
                        len(task_metrics),
                        len(regional_metrics),
                    )
                    comparison_path = os.path.join(
                        CENTRAL_METRICS_ROOT, "models_to_run_comparison.csv"
                    )
                    try:
                        comparison = aggregate_metrics.write_models_comparison(
                            regional_metrics,
                            comparison_path,
                            _normal_comparison_models(),
                        )
                        logging.info(
                            "Incrementally saved MODELS_TO_RUN comparison: %s (%d matched rows)",
                            comparison_path,
                            len(comparison),
                        )
                    except ValueError as exc:
                        logging.info(
                            "Central regional metrics were saved; comparison will be completed "
                            "after all requested models have metrics: %s",
                            exc,
                        )
                    if _ablation_run_requested():
                        try:
                            _write_aquistil_ablation_comparison(
                                regional_metrics, CENTRAL_METRICS_ROOT
                            )
                        except ValueError as exc:
                            logging.info(
                                "AQUISTIL ablation metrics were saved; paired comparison "
                                "will be completed after all variants finish: %s",
                                exc,
                            )

            task_predictions = regional_result.get("predictions") if regional_result else None
            if task_predictions is not None and not task_predictions.empty:
                regional_prediction_frames.append(task_predictions)
                os.makedirs(CENTRAL_IMPUTED_ROOT, exist_ok=True)
                regional_imputed_path = os.path.join(
                    CENTRAL_IMPUTED_ROOT, "regional_pooled_imputed_results.csv"
                )
                with _exclusive_file_lock(regional_imputed_path + ".lock"):
                    if os.path.exists(regional_imputed_path):
                        previous_predictions = pd.read_csv(regional_imputed_path)
                        current_predictions = pd.concat(
                            [previous_predictions, task_predictions],
                            ignore_index=True,
                            sort=False,
                        )
                    else:
                        current_predictions = task_predictions.copy()
                    prediction_key = [
                        column
                        for column in [
                            "DateTime",
                            "Region",
                            "Site",
                            "Target",
                            "Model",
                            "Regime",
                            "Missingness_Level",
                            "Seed",
                        ]
                        if column in current_predictions.columns
                    ]
                    if prediction_key:
                        current_predictions = current_predictions.drop_duplicates(
                            prediction_key,
                            keep="last",
                        )
                    current_predictions.to_csv(regional_imputed_path, index=False)
                    logging.info(
                        "Incrementally saved central regional imputed results: %s "
                        "(%d current rows, %d total rows)",
                        regional_imputed_path,
                        len(task_predictions),
                        len(current_predictions),
                    )

            for key, filename in (
                ("diagnostics", "gap_mask_diagnostics.csv"),
                ("exclusions", "site_target_exclusions.csv"),
            ):
                frame = regional_result.get(key) if regional_result else None
                if frame is None or frame.empty:
                    continue
                os.makedirs(CENTRAL_METRICS_ROOT, exist_ok=True)
                path = os.path.join(CENTRAL_METRICS_ROOT, filename)
                with _exclusive_file_lock(path + ".lock"):
                    if os.path.exists(path):
                        previous = pd.read_csv(path)
                        current = pd.concat([previous, frame], ignore_index=True, sort=False)
                    else:
                        current = frame.copy()
                    dedupe_key = [
                        column
                        for column in [
                            "Region",
                            "Site",
                            "Target",
                            "Regime",
                            "Seed",
                            "Requested_Missingness",
                        ]
                        if column in current.columns
                    ]
                    if dedupe_key:
                        current = current.drop_duplicates(dedupe_key, keep="last")
                    current.to_csv(path, index=False)
                logging.info("Incrementally saved central %s: %s (%d current rows)", key, path, len(frame))

        ablation_resume_metrics = _load_aquistil_ablation_resume_metrics(
            CENTRAL_METRICS_ROOT
        )
        if not ablation_resume_metrics.empty:
            logging.info(
                "Loaded %d persisted rows for resumable AQUISTIL ablation tasks",
                len(ablation_resume_metrics),
            )

        for model_name in available_models:
            model_module = load_model_module(model_name)
            if model_module is None:
                logging.error("Skipping unavailable regional model: %s", model_name)
                continue
            impute_callable = get_impute_callable(model_module)
            if impute_callable is None:
                logging.error("Skipping regional model without imputation callable: %s", model_name)
                continue
            impute_callable = _safe_imputer_wrapper(impute_callable)
            canonical_model = getattr(model_module, "MODEL_NAME", model_name)

            model_jobs = []
            for (region_token, target), (regional_data, choice) in regional_datasets.items():
                if target not in regional_data.columns:
                    logging.warning("Target %s unavailable for region %s", target, region_token)
                    continue
                if _ablation_task_is_complete(
                    ablation_resume_metrics,
                    region_token,
                    target,
                    canonical_model,
                ):
                    logging.info(
                        "Skipping complete ablation task: %s/%s/%s",
                        region_token,
                        target,
                        canonical_model,
                    )
                    continue
                model_jobs.append(
                    (region_token, target, regional_data, choice)
                )

            if not model_jobs:
                logging.info(
                    "All requested regional tasks are already complete for %s",
                    canonical_model,
                )
                continue

            def run_regional_job(job):
                region_token, target, regional_data, choice = job
                with _target_logging_context(target):
                    result = run_balanced_regional_task(
                        regional_data=regional_data,
                        region=region_token.replace("_", " "),
                        target=target,
                        features=choice["features"],
                        model_name=canonical_model,
                        impute_callable=impute_callable,
                        regimes=regimes_to_run,
                        missingness_levels=config.MISSINGNESS_LEVELS,
                        seeds=getattr(config, "REGIONAL_EVALUATION_SEEDS", [42]),
                        output_root=regional_output_root,
                        plots_root=CENTRAL_PLOTS_ROOT if getattr(config, "SAVE_PLOTS", False) else None,
                        plot_types=getattr(config, "PLOT_TYPES", []),
                        plot_dpi=getattr(config, "PLOT_DPI", 300),
                        parameters={
                            "configuration": choice.get("configuration", ""),
                            "blocks": choice.get("blocks", ""),
                            "feature_source": choice.get("feature_source", "stage3"),
                        },
                        strict_feature_list=bool(
                            choice.get("strict_progressive", False)
                        ),
                        model_n_jobs=model_n_jobs,
                        handle_negatives=getattr(config, "HANDLE_NEGATIVES", "exclude"),
                    )
                return (region_token, target), result

            model_workers = min(target_workers, max(1, len(model_jobs)))
            if model_workers == 1:
                for job in model_jobs:
                    try:
                        job_key, regional_result = run_regional_job(job)
                    except Exception:
                        with _target_logging_context(job[1]):
                            logging.exception(
                                "Regional task failed for %s/%s/%s",
                                job[0],
                                job[1],
                                canonical_model,
                            )
                        continue
                    with _target_logging_context(job_key[1]):
                        logging.info(
                            "Collecting completed regional result: %s/%s/%s",
                            job_key[0],
                            job_key[1],
                            canonical_model,
                        )
                        record_regional_result(regional_result)
            else:
                logging.info(
                    "Running %d regional %s tasks with %d target workers",
                    len(model_jobs),
                    canonical_model,
                    model_workers,
                )
                with ThreadPoolExecutor(max_workers=model_workers) as executor:
                    future_jobs = {
                        executor.submit(run_regional_job, job): job
                        for job in model_jobs
                    }
                    for future in as_completed(future_jobs):
                        job = future_jobs[future]
                        try:
                            job_key, regional_result = future.result()
                        except Exception:
                            with _target_logging_context(job[1]):
                                logging.exception(
                                    "Regional task failed for %s/%s/%s",
                                    job[0],
                                    job[1],
                                    canonical_model,
                                )
                            continue
                        with _target_logging_context(job_key[1]):
                            logging.info(
                                "Collecting completed regional result: %s/%s/%s",
                                job_key[0],
                                job_key[1],
                                canonical_model,
                            )
                            record_regional_result(regional_result)

        # Regional mode returns before the standard per-site aggregation below,
        # so explicitly publish its metrics to the central Metrics directory.
        os.makedirs(CENTRAL_METRICS_ROOT, exist_ok=True)
        regional_metrics_path = os.path.join(
            CENTRAL_METRICS_ROOT, "regional_pooled_metrics.csv"
        )
        if regional_metric_frames:
            current_regional_metrics = pd.concat(regional_metric_frames, ignore_index=True)
            _upsert_aquistil_ablation_metrics(
                current_regional_metrics, CENTRAL_METRICS_ROOT
            )
            _upsert_regional_metrics_by_target(current_regional_metrics, CENTRAL_METRICS_ROOT)
            with _exclusive_file_lock(regional_metrics_path + ".lock"):
                regional_metrics = aggregate_metrics.upsert_regional_metrics(
                    regional_metrics_path,
                    current_regional_metrics,
                )
                regional_metrics = _without_aquistil_ablations(regional_metrics)
                regional_metrics.to_csv(regional_metrics_path, index=False)
                _write_regional_metrics_by_target(regional_metrics, CENTRAL_METRICS_ROOT)
                logging.info(
                    "Upserted central regional metrics: %s "
                    "(%d current rows, %d total rows)",
                    regional_metrics_path,
                    len(current_regional_metrics),
                    len(regional_metrics),
                )
                comparison_path = os.path.join(
                    CENTRAL_METRICS_ROOT, "models_to_run_comparison.csv"
                )
                try:
                    comparison = aggregate_metrics.write_models_comparison(
                        regional_metrics,
                        comparison_path,
                        _normal_comparison_models(),
                    )
                    logging.info(
                        "Saved MODELS_TO_RUN comparison: %s (%d matched rows)",
                        comparison_path,
                        len(comparison),
                    )
                except ValueError as exc:
                    logging.error(
                        "Regional metrics were saved, but the model comparison "
                        "could not be created: %s",
                        exc,
                    )
                if _ablation_run_requested():
                    try:
                        _finalize_aquistil_ablation_outputs(
                            CENTRAL_METRICS_ROOT, RESULTS_ROOT
                        )
                    except ValueError as exc:
                        logging.error(
                            "Ablation metrics were saved, but the paired ablation "
                            "comparison could not be created: %s",
                            exc,
                        )
        else:
            logging.warning("No regional pooled metrics were produced; central metrics CSV not written")

        if _ablation_run_requested():
            try:
                _finalize_aquistil_ablation_outputs(
                    CENTRAL_METRICS_ROOT, RESULTS_ROOT
                )
            except ValueError as exc:
                logging.error(
                    "AQUISTIL ablation run remains incomplete; raw completed rows "
                    "were preserved but final summaries were not replaced: %s",
                    exc,
                )

        # Publish all artificially masked observations and their imputed values
        # in one central file for downstream analysis.
        os.makedirs(CENTRAL_IMPUTED_ROOT, exist_ok=True)
        regional_imputed_path = os.path.join(
            CENTRAL_IMPUTED_ROOT, "regional_pooled_imputed_results.csv"
        )
        if regional_prediction_frames:
            current_predictions = pd.concat(regional_prediction_frames, ignore_index=True)
            with _exclusive_file_lock(regional_imputed_path + ".lock"):
                if os.path.exists(regional_imputed_path):
                    previous_predictions = pd.read_csv(regional_imputed_path)
                    current_predictions = pd.concat(
                        [previous_predictions, current_predictions],
                        ignore_index=True,
                        sort=False,
                    )
                prediction_key = [
                    column
                    for column in [
                        "DateTime",
                        "Region",
                        "Site",
                        "Target",
                        "Model",
                        "Regime",
                        "Missingness_Level",
                        "Seed",
                    ]
                    if column in current_predictions.columns
                ]
                if prediction_key:
                    current_predictions = current_predictions.drop_duplicates(
                        prediction_key,
                        keep="last",
                    )
                current_predictions.to_csv(regional_imputed_path, index=False)
                logging.info("Saved central regional imputed results: %s", regional_imputed_path)
        else:
            logging.warning(
                "No regional pooled predictions were produced; central imputed CSV not written"
            )
        logging.info("REGIONAL POOLED IMPUTATION FINISHED: %s", regional_output_root)
        print("=== JOB FINISHED: regional pooled imputation ===", flush=True)
        return

    # Ensure model/regime folders under MODEL_OUTPUT_ROOT
    for mn in available_models:
        for r in regimes_to_run:
            model_out_dir = os.path.join(MODEL_OUTPUT_ROOT, mn, r)
            for sub in ["imputed_data", "metrics", "target_column_data"]:
                os.makedirs(os.path.join(model_out_dir, sub), exist_ok=True)
    logging.info("Ensured output directories for all models and regimes exist.")

    # --------------------------------------------------------------------------
    # Execution counters
    # --------------------------------------------------------------------------
    total_tasks = (len(available_models) * len(sites_to_process) * 
                  len(config.TARGET_COLUMNS) * len(regimes_to_run))
    completed_tasks, failed_tasks = 0, 0
    progressive_site_file_cache = {}
    default_use_spatial = getattr(config, "USE_SPATIAL_FEATURES", False)
    default_use_temporal = getattr(config, "USE_TEMPORAL_FEATURES", True)

    # ==========================================================================
    # REGIME LOOP
    # ==========================================================================
    for regime_idx, regime in enumerate(regimes_to_run, 1):
        
        logging.info("\n" + "=" * 80)
        logging.info(f"REGIME {regime_idx}/{len(regimes_to_run)}: {regime.upper()}")
        logging.info("=" * 80)
        
        config.MISSINGNESS_REGIME = regime

        # ----------------------------------------------------------------------
        # MODEL LOOP
        # ----------------------------------------------------------------------
        for model_name in available_models: 

            logging.info("=" * 80)
            logging.info(f"MODEL: {model_name} | REGIME: {regime}")
            logging.info("=" * 80)

            model_module = load_model_module(model_name)
            if model_module is None:
                logging.error(f"Skipping model {model_name} because module could not be loaded.")
                failed_tasks += len(sites_to_process) * len(config.TARGET_COLUMNS)
                continue

            impute_callable = get_impute_callable(model_module)
            if impute_callable is None:
                logging.warning(
                    f"Model module '{model_name}' does not expose a recognized imputer function. Skipping."
                )
                failed_tasks += len(sites_to_process) * len(config.TARGET_COLUMNS)
                continue

            impute_callable = _safe_imputer_wrapper(impute_callable)

            MODEL_NAME = getattr(model_module, "MODEL_NAME", model_name)

            if MODEL_NAME == "GATI_AQ":
                logging.info("🧠 Running GATI-AQ (Gated Adaptive Imputation)")

            # Per-model, per-regime output directory under MODEL_OUTPUT_ROOT
            model_out_dir = os.path.join(MODEL_OUTPUT_ROOT, MODEL_NAME, regime)
            os.makedirs(model_out_dir, exist_ok=True)
            for sub in ["imputed_data", "metrics", "target_column_data"]:
                os.makedirs(os.path.join(model_out_dir, sub), exist_ok=True)

            # ------------------------------------------------------------------
            # SITE LOOP
            # ------------------------------------------------------------------
            for site in sites_to_process:

                # ------------------------------------------------------------------
                # Resolve input file (wide inputs -> materialize per-site CSV)
                # ------------------------------------------------------------------
                if getattr(config, "USE_WIDE_NOWCASTING_INPUTS", True):
                    site_meta = wide_site_index.get(_canon_token(site))
                    if not site_meta:
                        logging.warning(
                            "Site '%s' not found in wide inputs under %s; skipping",
                            site,
                            wide_input_dir,
                        )
                        failed_tasks += len(config.TARGET_COLUMNS)
                        continue

                    site_region = site_meta["region_token"]
                    wide_path = site_meta["wide_path"]
                    try:
                        site_file_path = prepared_site_inputs.get(_canon_token(site))
                        if not site_file_path or not os.path.exists(site_file_path):
                            per_site_dir = os.path.join(RESULTS_ROOT, "Inputs_PerSite")
                            site_file_path = _build_per_site_csv_from_wide(wide_path, site, per_site_dir)
                            prepared_site_inputs[_canon_token(site)] = site_file_path
                        site_file = os.path.basename(site_file_path)
                        input_file_path = site_file_path
                    except Exception as e:
                        logging.warning("Failed to build per-site CSV for %s from %s: %s", site, wide_path, e)
                        failed_tasks += len(config.TARGET_COLUMNS)
                        continue
                else:
                    site_file = next(
                        (f for f in os.listdir(config.INPUT_DIRECTORY)
                         if f.lower().startswith(site.lower()) and f.endswith('.csv')),
                        None
                    )

                    if site_file is None:
                        failed_tasks += len(config.TARGET_COLUMNS)
                        continue

                    input_file_path = os.path.join(config.INPUT_DIRECTORY, site_file)

                    data = pd.read_csv(input_file_path)
                if "DateTime" in data.columns:
                    data["DateTime"] = pd.to_datetime(data["DateTime"], errors="coerce")

                for target in config.TARGET_COLUMNS: 

                    if target not in data.columns:
                        failed_tasks += 1
                        continue

                    # Robust selection of input columns: allow fuzzy/normalized matches
                    import re as _re

                    def _normalize_name(s):
                        return _re.sub(r'[^A-Za-z0-9]', '', str(s)).upper()

                    data_col_map = { _normalize_name(col): col for col in data.columns }

                    # Choose predictors:
                    # 1) Stage 3 winning regional configuration (authoritative).
                    # 2) Older region+target best-predictors JSON.
                    # 3) config.INPUT_COLUMNS fallback.
                    input_cols = []
                    candidate_predictors = []
                    progressive_choice = progressive_best_map.get(
                        (_canon_token(site_region), _canon_token(target))
                    ) if site_region else None

                    if progressive_choice:
                        candidate_predictors = list(progressive_choice["features"])
                        configuration_name = progressive_choice["configuration"]
                        # The exact Stage 3 Feature_List is materialized below;
                        # prevent individual model modules from adding extra
                        # temporal/spatial columns outside the winning set.
                        config.USE_TEMPORAL_FEATURES = False
                        config.USE_SPATIAL_FEATURES = False
                        config.STRICT_PROGRESSIVE_FEATURE_LIST = True

                        cache_key = (input_file_path, target, tuple(candidate_predictors))
                        selected_input_path = progressive_site_file_cache.get(cache_key)
                        if not selected_input_path:
                            enriched = _add_progressive_derived_features(
                                data,
                                site,
                                target,
                                candidate_predictors,
                                progressive_neighbor_inputs or prepared_site_inputs,
                            )
                            selected_dir = os.path.join(RESULTS_ROOT, "Inputs_SelectedFeatures")
                            os.makedirs(selected_dir, exist_ok=True)
                            selected_input_path = os.path.join(
                                selected_dir,
                                "%s_%s.csv" % (_canon_token(site), _canon_token(target)),
                            )
                            enriched.to_csv(selected_input_path, index=False)
                            progressive_site_file_cache[cache_key] = selected_input_path
                        input_file_path = selected_input_path
                        data = pd.read_csv(input_file_path)
                        data["DateTime"] = pd.to_datetime(data["DateTime"], errors="coerce")
                        data_col_map = {_normalize_name(col): col for col in data.columns}
                        logging.info(
                            "Using Stage 3 best configuration for %s/%s: %s | features=%s",
                            site_region,
                            target,
                            configuration_name,
                            candidate_predictors,
                        )
                    else:
                        config.USE_SPATIAL_FEATURES = default_use_spatial
                        config.USE_TEMPORAL_FEATURES = default_use_temporal
                        config.STRICT_PROGRESSIVE_FEATURE_LIST = False

                    if not candidate_predictors and getattr(config, "USE_BEST_PREDICTORS_JSON_INPUTS", False) and site_region:
                        region_token = _region_token_from_config_region(site_region)
                        candidate_predictors = _get_predictors_for_region_target(
                            best_predictors_map, region_token, target
                        )

                    if candidate_predictors:
                        for cfg_col in candidate_predictors:
                            norm = _normalize_name(cfg_col)
                            if norm in data_col_map and data_col_map[norm] != target:
                                input_cols.append(data_col_map[norm])
                    else:
                        for cfg_col in config.INPUT_COLUMNS:
                            norm = _normalize_name(cfg_col)
                            if norm in data_col_map and data_col_map[norm] != target:
                                input_cols.append(data_col_map[norm])

                    if not input_cols:
                        if candidate_predictors:
                            # try exact columns for candidate predictors
                            input_cols = [c for c in candidate_predictors if c in data.columns and c != target]
                        if not input_cols:
                            if getattr(config, "FALLBACK_TO_INPUT_COLUMNS_WHEN_MISSING", True):
                                input_cols = [c for c in config.INPUT_COLUMNS if c in data.columns and c != target]

                    if progressive_choice:
                        selected_norms = {_normalize_name(col) for col in input_cols}
                        unavailable = [
                            feature
                            for feature in candidate_predictors
                            if _normalize_name(feature) not in selected_norms
                        ]
                        if unavailable:
                            logging.error(
                                "Stage 3 winning features unavailable for %s/%s: %s",
                                site,
                                target,
                                unavailable,
                            )
                            failed_tasks += 1
                            continue

                    logging.info(f"Selected input columns for {site}: {input_cols}")
                    if not input_cols:
                        failed_tasks += 1
                        continue

                    validation_errors = validate_data_for_imputation(data, target, input_cols)
                    if validation_errors:
                        logging.error(f"Data validation failed for {site}/{target}:")
                        for err in validation_errors:
                            logging.error(f"  - {err}")
                        failed_tasks += 1
                        continue

                    try:
                        sort_and_impute_by_hour(
                            input_file=input_file_path,
                            imputed_data_path=os.path.join(model_out_dir, "imputed_data"),
                            target_column_data_path=os.path.join(model_out_dir, "target_column_data"),
                            metrics_save_path=os.path.join(model_out_dir, "metrics"),
                            target_column=target,
                            input_columns=input_cols,
                            missingness_levels=config.MISSINGNESS_LEVELS,
                            handle_negatives=config.HANDLE_NEGATIVES,
                            impute_function=lambda d, t, i, custom_strategies=None, **kwargs: 
                                impute_callable(d, t, i, custom_strategies=custom_strategies, **kwargs),
                            model_name=f"{MODEL_NAME}_{site}",
                            sort_by_hour=config.SORT_BY_HOUR,
                            missingness_regime=regime,
                            evaluation_seed=getattr(config, "GLOBAL_RANDOM_SEED", 42),
                            # Central paths from main.py
                            central_imputed_root=CENTRAL_IMPUTED_ROOT,
                            central_metrics_root=CENTRAL_METRICS_ROOT,
                            central_plots_root=(
                                CENTRAL_PLOTS_ROOT
                                if getattr(config, "SAVE_PLOTS", False)
                                else None
                            ),
                            results_root=RESULTS_ROOT,
                        )
                        
                        model_out_root_for_verify = os.path.join(MODEL_OUTPUT_ROOT, MODEL_NAME)
                        verification_ok = verify_run_outputs(
                            model_out_root_for_verify,
                            site_file,
                            target,
                            f"{MODEL_NAME}_{site}",
                            [regime],  # only verify the regime we just ran
                            config.MISSINGNESS_LEVELS
                        )
                        if not verification_ok:
                            logging.warning(
                                f"Post-run verification found missing outputs for {MODEL_NAME} | {site}"
                            )

                        completed_tasks += 1
                        logging.info(
                            f"✅ Completed:  {MODEL_NAME} | {site} | {target} | "
                            f"{regime} | {completed_tasks}/{total_tasks}"
                        )
                        
                    except Exception as e:
                        logging.error(
                            f"❌ Failed:  {MODEL_NAME} | {site} | {target} | {regime}"
                        )
                        traceback.print_exc()
                        failed_tasks += 1

    # --------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------
    logging.info("\n" + "=" * 80)
    logging.info("COMPLETED ALL REGIMES")
    logging.info("=" * 80)
    logging.info(f"Total tasks: {total_tasks}")
    logging.info(f"Completed:  {completed_tasks}")
    logging.info(f"Failed: {failed_tasks}")
    if total_tasks > 0:
        logging.info(f"Success rate: {(completed_tasks/total_tasks*100):.1f}%")
    else:
        logging.info("Success rate: N/A (no tasks scheduled)")
    logging.info(f"Duration: {datetime.now() - start_time}")
    logging.info(f"\nResults saved to: {RESULTS_ROOT}")

    if getattr(config, "CREATE_SUMMARY_REPORT", False):
        try:
            logging.info("Aggregating per-run metrics into overall summary CSV...")
            aggregate_metrics.aggregate(RESULTS_ROOT)
            logging.info("✅ Aggregation completed.")
        except Exception:
            logging.error("Aggregation failed:")
            traceback.print_exc()

    try_generate = args.generate_research or getattr(config, "AUTO_GENERATE_RESEARCH_PLOTS", False)
    if try_generate and not getattr(config, "SAVE_PLOTS", False):
        logging.info("Skipping research plot generation because SAVE_PLOTS is False.")
        try_generate = False
    if try_generate:
        logging.info("Preparing to generate research-grade plots...")
        plot_out = RESEARCH_PLOTS_DIR
        try:
            os.makedirs(plot_out, exist_ok=True)
        except Exception:
            logging.warning(f"Could not create plot output directory: {plot_out}")

        # Refresh the aggregate before plotting so a one-model follow-up run
        # contributes its rows instead of reusing a stale all_results_summary.csv.
        try:
            logging.info("Refreshing aggregated summary for research plots...")
            aggregate_metrics.aggregate(RESULTS_ROOT)
            logging.info("✅ Aggregation completed for research plots.")
        except Exception:
            logging.warning(
                "Aggregation for research plots failed; proceeding to attempt plotting from available metrics."
            )
            traceback.print_exc()

        try:
            logging.info("Generating research-grade plots...")
            import research_plots as rp
            target = (
                config.TARGET_COLUMNS[0]
                if hasattr(config, 'TARGET_COLUMNS') and config.TARGET_COLUMNS
                else 'PM2.5'
            )
            plotter = rp.ResearchPlotter(
                results_dir=RESULTS_ROOT,
                output_dir=plot_out,
                target=target
            )
            plotter.generate_all_plots()
            logging.info(f"✅ Research plots saved to: {plot_out}")
        except Exception as e:
            logging.error(f"Failed to generate research plots: {e}")
            traceback.print_exc()
    logging.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    finally:
        print("!!!!!!!!!!!!!!!!!!!!Finished!!!!!!!!!!!!!!!!")
