#!/usr/bin/env python3
"""Evaluate candidate imputers specifically on pollution-event missingness.

This is a thin experiment driver around ``main.py``.  It deliberately reuses
the production data preparation, balanced regional masks, model interface and
metrics so candidates receive exactly the same event rows at each missingness
level and seed.

Examples
--------
List compatible modules without running them::

    python main_eval.py --list-models

Run the complete default event experiment (Lower Hunter, 10/20/30/50%,
cached inputs, no plots)::

    python main_eval.py

Run an explicit shortlist::

    python main_eval.py --models AQUISTIL LightGBM LGBMPlus RALGBM \
        --regions "Lower Hunter" --skip-api-refresh
"""

import argparse
import ast
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR / "Model"
DEFAULT_OUTPUT = (
    THIS_DIR.parent / "Outputs" / "Event_Model_Evaluation"
)
DEFAULT_REGIONS = ["Lower Hunter"]
DEFAULT_LEVELS = [10, 20, 30, 50]
DEFAULT_SEEDS = [42]

# A practical default shortlist of LightGBM-based and hybrid candidates.  Use
# ``--models all`` to schedule every module exposing a supported imputer API.
DEFAULT_EVENT_CANDIDATES = [
    "AQUISTIL",
    "LightGBM",
    "LGBMPlus",
    "LGBM_AQ_Plus_Adaptive",
    "LGBMPlusSpatialIterOptimized",
    "LGBM_AQ_Plus_SpatialIter_Optimized_V2",
    "RALGBM",
    "HybridMFXGB_LGB",
    "HybridMFXGB_LGB_Pro",
    "HybridMissForest_Advanced",
    "HybridMissForest_Only",
    "HybridMissForest_Probabilistic",
    "HybridMissForest_XGB_Simple",
    "HybridModelSequential",
    "SuperLearner",
]

SUPPORTED_CALLABLES = {
    "impute_mice",
    "impute",
    "run_impute",
    "impute_values",
    "predict",
    "transform",
}
METRICS = ("RMSE", "MAE", "R", "NSE")


def discover_model_modules() -> List[str]:
    """Find model files that statically expose a supported imputation call."""
    found = []
    for path in sorted(MODEL_DIR.glob("*.py")):
        if path.name == "__init__.py" or " copy" in path.stem:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        functions = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        classes_with_impute = any(
            isinstance(node, ast.ClassDef)
            and any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "impute"
                for child in node.body
            )
            for node in tree.body
        )
        if functions.intersection(SUPPORTED_CALLABLES) or classes_with_impute:
            found.append(path.stem)
    return found


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def validate_models(requested: List[str]) -> List[str]:
    discovered = set(discover_model_modules())
    missing = [name for name in requested if name not in discovered]
    if missing:
        raise SystemExit(
            "Unknown or incompatible model module(s): " + ", ".join(missing)
            + ". Run --list-models to inspect valid names."
        )
    return requested


def canonical_model_name(module_name: str) -> str:
    """Read a module's static MODEL_NAME without importing heavy dependencies."""
    path = MODEL_DIR / (module_name + ".py")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "MODEL_NAME" for target in node.targets):
                continue
            if isinstance(node.value, ast.Str):
                return node.value.s
    except (OSError, SyntaxError):
        pass
    return module_name


def build_rankings(metrics_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """Create overall and regime-level rankings from regional metrics."""
    if not metrics_path.is_file():
        raise FileNotFoundError("Event metrics were not produced: %s" % metrics_path)

    data = pd.read_csv(metrics_path)
    required = {"Model", "Regime", "Scope", "RMSE", "MAE", "R", "NSE"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Metrics file is missing columns: %s" % ", ".join(missing))

    data = data.loc[data["Regime"].astype(str).str.lower().eq("event")].copy()
    # Region_Micro represents every masked observation once. Site and macro
    # rows are retained in the raw CSV but excluded from model selection.
    micro = data.loc[data["Scope"].eq("Region_Micro")].copy()
    if micro.empty:
        raise ValueError("No Region_Micro event rows found in %s" % metrics_path)

    summary = micro.groupby("Model", as_index=False).agg(
        Evaluations=("RMSE", "size"),
        RMSE=("RMSE", "mean"),
        MAE=("MAE", "mean"),
        R=("R", "mean"),
        NSE=("NSE", "mean"),
        Worst_NSE=("NSE", "min"),
    )

    # Rank each criterion independently, then use their mean as a transparent
    # balanced selection score. Lower final rank is better.
    summary["Rank_RMSE"] = summary["RMSE"].rank(method="min", ascending=True)
    summary["Rank_MAE"] = summary["MAE"].rank(method="min", ascending=True)
    summary["Rank_R"] = summary["R"].rank(method="min", ascending=False)
    summary["Rank_NSE"] = summary["NSE"].rank(method="min", ascending=False)
    summary["Mean_Rank"] = summary[
        ["Rank_RMSE", "Rank_MAE", "Rank_R", "Rank_NSE"]
    ].mean(axis=1)

    baseline = summary.loc[summary["Model"].eq("LightGBM")]
    if not baseline.empty:
        base = baseline.iloc[0]
        summary["RMSE_Improvement_vs_LightGBM_pct"] = (
            (base["RMSE"] - summary["RMSE"]) / base["RMSE"] * 100.0
        )
        summary["MAE_Improvement_vs_LightGBM_pct"] = (
            (base["MAE"] - summary["MAE"]) / base["MAE"] * 100.0
        )
        summary["Delta_R_vs_LightGBM"] = summary["R"] - base["R"]
        summary["Delta_NSE_vs_LightGBM"] = summary["NSE"] - base["NSE"]

    summary = summary.sort_values(
        ["Mean_Rank", "RMSE", "MAE"], kind="stable"
    ).reset_index(drop=True)
    summary.insert(0, "Overall_Position", np.arange(1, len(summary) + 1))

    by_level = micro.groupby(
        ["Missingness_Percent", "Model"], as_index=False
    ).agg(
        Evaluations=("RMSE", "size"),
        RMSE=("RMSE", "mean"),
        MAE=("MAE", "mean"),
        R=("R", "mean"),
        NSE=("NSE", "mean"),
    )
    by_level["Rank_RMSE"] = by_level.groupby("Missingness_Percent")["RMSE"].rank(
        method="min", ascending=True
    )
    by_level["Rank_MAE"] = by_level.groupby("Missingness_Percent")["MAE"].rank(
        method="min", ascending=True
    )
    by_level["Rank_R"] = by_level.groupby("Missingness_Percent")["R"].rank(
        method="min", ascending=False
    )
    by_level["Rank_NSE"] = by_level.groupby("Missingness_Percent")["NSE"].rank(
        method="min", ascending=False
    )
    by_level["Mean_Rank"] = by_level[
        ["Rank_RMSE", "Rank_MAE", "Rank_R", "Rank_NSE"]
    ].mean(axis=1)
    by_level = by_level.sort_values(
        ["Missingness_Percent", "Mean_Rank", "RMSE"], kind="stable"
    )
    by_level["Level_Position"] = (
        by_level.groupby("Missingness_Percent").cumcount() + 1
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    overall_path = output_dir / "event_model_ranking_overall.csv"
    level_path = output_dir / "event_model_ranking_by_missingness.csv"
    summary.to_csv(overall_path, index=False)
    by_level.to_csv(level_path, index=False)
    return overall_path, level_path


def build_blend_screen(predictions_path: Path, output_dir: Path, anchor="AQUISTIL") -> Path:
    """Screen fixed AQUISTIL/candidate blends on matched event predictions.

    This is exploratory model development output, not an unbiased final test:
    any selected partner and weight must be frozen and re-evaluated separately.
    """
    data = pd.read_csv(predictions_path)
    required = {
        "DateTime", "Region", "Site", "Target", "Model", "Regime",
        "Missingness_Level", "Seed", "Observed", "Imputed",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Prediction file is missing columns: %s" % ", ".join(missing))
    data = data.loc[data["Regime"].astype(str).str.lower().eq("event")].copy()
    keys = [
        "DateTime", "Region", "Site", "Target", "Regime",
        "Missingness_Level", "Seed",
    ]
    anchor_data = data.loc[data["Model"].eq(anchor), keys + ["Observed", "Imputed"]].rename(
        columns={"Imputed": "Anchor_Imputed"}
    )
    if anchor_data.empty:
        raise ValueError("Cannot screen blends because anchor model %r has no predictions" % anchor)

    def metrics(observed, predicted):
        observed = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(observed) & np.isfinite(predicted)
        observed, predicted = observed[valid], predicted[valid]
        if not len(observed):
            return dict(N_Valid=0, RMSE=np.nan, MAE=np.nan, R=np.nan, NSE=np.nan)
        residual = predicted - observed
        denominator = np.sum((observed - observed.mean()) ** 2)
        correlation = np.corrcoef(observed, predicted)[0, 1] if len(observed) > 1 else np.nan
        return dict(
            N_Valid=len(observed),
            RMSE=float(np.sqrt(np.mean(residual ** 2))),
            MAE=float(np.mean(np.abs(residual))),
            R=float(correlation),
            NSE=float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else np.nan,
        )

    rows = []
    candidates = sorted(set(data["Model"].dropna()) - {anchor})
    for candidate in candidates:
        other = data.loc[data["Model"].eq(candidate), keys + ["Imputed"]].rename(
            columns={"Imputed": "Candidate_Imputed"}
        )
        matched = anchor_data.merge(other, on=keys, how="inner", validate="one_to_one")
        if matched.empty:
            continue
        for level, part in matched.groupby("Missingness_Level", sort=True):
            for anchor_weight in (0.25, 0.50, 0.75):
                blended = (
                    anchor_weight * pd.to_numeric(part["Anchor_Imputed"], errors="coerce")
                    + (1.0 - anchor_weight)
                    * pd.to_numeric(part["Candidate_Imputed"], errors="coerce")
                )
                rows.append(dict(
                    Scope="By_Missingness",
                    Missingness_Level=level,
                    Anchor_Model=anchor,
                    Candidate_Model=candidate,
                    Anchor_Weight=anchor_weight,
                    Candidate_Weight=1.0 - anchor_weight,
                    **metrics(part["Observed"], blended)
                ))
        for anchor_weight in (0.25, 0.50, 0.75):
            blended = (
                anchor_weight * pd.to_numeric(matched["Anchor_Imputed"], errors="coerce")
                + (1.0 - anchor_weight)
                * pd.to_numeric(matched["Candidate_Imputed"], errors="coerce")
            )
            rows.append(dict(
                Scope="Overall",
                Missingness_Level=np.nan,
                Anchor_Model=anchor,
                Candidate_Model=candidate,
                Anchor_Weight=anchor_weight,
                Candidate_Weight=1.0 - anchor_weight,
                **metrics(matched["Observed"], blended)
            ))

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No matched candidate predictions were available for blend screening")
    result["Rank_RMSE"] = result.groupby(["Scope", "Missingness_Level"], dropna=False)["RMSE"].rank(
        method="min", ascending=True
    )
    result["Rank_MAE"] = result.groupby(["Scope", "Missingness_Level"], dropna=False)["MAE"].rank(
        method="min", ascending=True
    )
    result["Rank_R"] = result.groupby(["Scope", "Missingness_Level"], dropna=False)["R"].rank(
        method="min", ascending=False
    )
    result["Rank_NSE"] = result.groupby(["Scope", "Missingness_Level"], dropna=False)["NSE"].rank(
        method="min", ascending=False
    )
    result["Mean_Rank"] = result[
        ["Rank_RMSE", "Rank_MAE", "Rank_R", "Rank_NSE"]
    ].mean(axis=1)
    result = result.sort_values(
        ["Scope", "Missingness_Level", "Mean_Rank", "RMSE"], kind="stable"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "event_blend_screening_exploratory.csv"
    result.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark candidate imputers on event-driven missingness only."
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model module names, or the single value 'all'.",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--levels", nargs="+", type=float, default=DEFAULT_LEVELS,
        help="Missingness percentages; accepts 10 20 30 50 or fractions.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--regions", nargs="+", default=DEFAULT_REGIONS,
        help="Regions to evaluate (default: Lower Hunter).",
    )
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--target", default="PM2.5")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--wide-input-dir", default=None)
    parser.add_argument(
        "--skip-api-refresh", action="store_true", default=True,
        help="Use cached wide inputs (default).",
    )
    parser.add_argument("--refresh-api-inputs", action="store_true")
    parser.add_argument(
        "--no-plots", action="store_true", default=True,
        help="Disable expensive per-model plots (default).",
    )
    parser.add_argument(
        "--plots", action="store_false", dest="no_plots",
        help="Enable plots for this experiment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Configure logging before importing config_spatial/main.py. Otherwise the
    # first warning installs Python's default WARNING-only handler and all
    # model/progress INFO messages disappear, making a healthy batch job look
    # stuck.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
    else:
        for handler in root_logger.handlers:
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )

    discovered = discover_model_modules()
    if args.list_models:
        print("\n".join(discovered))
        return 0

    if args.models == ["all"]:
        models = discovered
    else:
        models = _unique(args.models or DEFAULT_EVENT_CANDIDATES)
    models = validate_models(models)
    if "LightGBM" not in models:
        models.append("LightGBM")

    levels = [value / 100.0 if value > 1 else value for value in args.levels]
    if any(not 0 < value < 1 for value in levels):
        raise SystemExit("Every missingness level must be between 0 and 100 percent.")

    output_dir = Path(args.output_dir).resolve()

    # Override configuration before importing main.py because it derives its
    # output constants at import time.
    import config_spatial as config

    config.OUTPUT_DIRECTORY = str(output_dir)
    config.MODELS_TO_RUN = models
    config.STANDALONE_MODELS = []
    config.COMPARISON_MODELS = _unique(canonical_model_name(name) for name in models)
    config.MAX_MODELS_TO_RUN = 0
    config.MISSINGNESS_REGIMES = ["event"]
    config.MISSINGNESS_REGIME = "event"
    config.MISSINGNESS_LEVELS = levels
    config.REGIONAL_EVALUATION_SEEDS = list(dict.fromkeys(args.seeds))
    config.TARGET_COLUMNS = [args.target]
    config.REGIONAL_POOLED_MODE = True
    # This experiment uses the Stage 3 winning feature contract.  The older
    # BestPredictors_ByRegionTarget.json is optional and is not present in this
    # repository, so do not attempt to load it or silently mix fallback inputs.
    config.USE_PROGRESSIVE_BEST_FEATURES = True
    config.USE_BEST_PREDICTORS_JSON_INPUTS = False
    config.AUTO_UPDATE_WIDE_INPUTS = bool(args.refresh_api_inputs)
    config.SAVE_PLOTS = not args.no_plots
    config.AUTO_GENERATE_RESEARCH_PLOTS = False
    config.CREATE_SUMMARY_REPORT = False
    if args.regions:
        config.SELECT_TARGET_REGIONS = args.regions
    if args.sites:
        config.SELECT_TARGET_SITES = args.sites

    # main.py owns the experiment; only pass through arguments it understands.
    runner_argv = [str(THIS_DIR / "main.py"), "--regime", "event"]
    if args.skip_api_refresh or not args.refresh_api_inputs:
        runner_argv.append("--skip-api-refresh")
    if args.refresh_api_inputs:
        runner_argv.append("--refresh-api-inputs")
    if args.wide_input_dir:
        runner_argv.extend(["--wide-input-dir", args.wide_input_dir])
    if args.sites:
        runner_argv.extend(["--sites", *args.sites])

    old_argv = sys.argv
    try:
        sys.argv = runner_argv
        spec = importlib.util.spec_from_file_location("aquistil_main_runner", THIS_DIR / "main.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load main.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner.main()
    finally:
        sys.argv = old_argv

    metrics_path = output_dir / "Metrics" / "regional_pooled_metrics.csv"
    overall, by_level = build_rankings(metrics_path, output_dir / "Metrics")
    predictions_path = (
        output_dir / "Imputed_Results" / "regional_pooled_imputed_results.csv"
    )
    blend_path = None
    if predictions_path.is_file() and "AQUISTIL" in models:
        blend_path = build_blend_screen(
            predictions_path, output_dir / "Metrics", anchor="AQUISTIL"
        )
    ranking = pd.read_csv(overall)
    print("\nEvent-model ranking (Region_Micro):")
    print(
        ranking[["Overall_Position", "Model", "RMSE", "MAE", "R", "NSE", "Mean_Rank"]]
        .to_string(index=False)
    )
    print("\nOverall ranking: %s" % overall)
    print("By-missingness ranking: %s" % by_level)
    if blend_path:
        print("Exploratory blend screening: %s" % blend_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
