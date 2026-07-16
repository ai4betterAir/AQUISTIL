import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
APP_ROOT_DIR = CODE_DIR.parent

DEFAULT_WIDE_INPUT_DIR = APP_ROOT_DIR / "API_Input" / "Inputs"
DEFAULT_OUTPUT_ROOT = APP_ROOT_DIR / "Outputs" / "Feature_Selection"

# -----------------------------------------------------------------------------
# Editable main selection block
# Leave SELECTED_REGIONS empty to run all available regions in the input folder.
# -----------------------------------------------------------------------------
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
SELECTED_REGIONS = ["Lower Hunter"]

INDIVIDUAL_FEATURE_SCRIPT = SCRIPT_DIR / "00run_random_forest_best_individual_feature.py"
MIRROR_SCRIPT = SCRIPT_DIR / "01plot_mirror_region_vs_site_edited.py"
REGIONAL_SELECTION_SCRIPT = SCRIPT_DIR / "02regional_rf_shap_feature_selection.py"
PROGRESSIVE_SCRIPT = SCRIPT_DIR / "03regional_selected_feature_progressive_evaluation.py"
BASICS_SCRIPT = SCRIPT_DIR / "04basics_stats.py"


def canon(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value).upper())


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "NA"


def env_with_pythonpath() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(CODE_DIR) if not existing else f"{CODE_DIR}:{existing}"
    return env


def list_wide_region_files(input_dir: Path) -> list:
    return sorted(input_dir.glob("Allobs_processed_DPE_station_api_*_ALL.csv"))


def region_token_from_wide_csv(csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = "Allobs_processed_DPE_station_api_"
    suffix = "_ALL"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return stem


def resolve_region_wide_files(input_dir: Path, selected_regions: list) -> list:
    files = list_wide_region_files(input_dir)
    if not selected_regions:
        return files

    wanted = {canon(region) for region in selected_regions}
    selected = [
        csv_path for csv_path in files
        if canon(region_token_from_wide_csv(csv_path)) in wanted
    ]
    return selected


def ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "DateTime" in out.columns:
        out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
        return out
    if "datetime" in out.columns:
        out["DateTime"] = pd.to_datetime(out["datetime"], errors="coerce")
        return out
    raise KeyError("Wide regional CSV is missing a DateTime/datetime column")


def list_sites_from_wide_csv(csv_path: Path) -> list:
    cols = pd.read_csv(csv_path, nrows=0).columns
    sites = set()
    for col in cols:
        if not isinstance(col, str) or col in {"DateTime", "datetime"} or "_" not in col:
            continue
        _, site = col.split("_", 1)
        if site.strip():
            sites.add(site.strip())
    return sorted(sites)


def build_per_site_csv_from_wide(wide_csv_path: Path, site_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(wide_csv_path, low_memory=False)
    df = ensure_datetime_column(df)

    wanted_cols = ["DateTime"]
    for col in df.columns:
        if not isinstance(col, str) or col == "DateTime" or "_" not in col:
            continue
        var, site = col.split("_", 1)
        if canon(site) == canon(site_name):
            wanted_cols.append(col)

    if len(wanted_cols) <= 1:
        raise ValueError(f"No columns found for site {site_name} in {wide_csv_path}")

    sub = df[wanted_cols].copy()
    rename_map = {}
    for col in sub.columns:
        if col == "DateTime":
            continue
        var, _ = col.split("_", 1)
        rename_map[col] = var
    sub = sub.rename(columns=rename_map)
    sub = sub.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)

    out_path = out_dir / f"{site_name}.csv"
    sub.to_csv(out_path, index=False)
    return out_path


def prepare_site_input_cache(input_dir: Path, cache_dir: Path, selected_regions: list) -> dict:
    region_files = resolve_region_wide_files(input_dir, selected_regions)
    if not region_files:
        region_text = ", ".join(selected_regions) if selected_regions else "<all>"
        raise FileNotFoundError(f"No wide regional CSVs found in {input_dir} for regions {region_text}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    for old_csv in cache_dir.glob("*.csv"):
        try:
            old_csv.unlink()
        except Exception:
            pass

    site_paths = {}
    for wide_csv_path in region_files:
        for site_name in list_sites_from_wide_csv(wide_csv_path):
            out_path = build_per_site_csv_from_wide(wide_csv_path, site_name, cache_dir)
            site_paths[canon(site_name)] = str(out_path)

    logging.info(
        "Prepared %d per-site CSVs from %d wide regional file(s) into %s",
        len(site_paths),
        len(region_files),
        cache_dir,
    )
    return site_paths


def run_stage(stage_name: str, script_path: Path, env: dict, extra_args=None) -> None:
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    logging.info("Starting %s", stage_name)
    logging.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR), env=env)
    logging.info("Completed %s", stage_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the numbered FeatureStats pipeline in order.")
    parser.add_argument(
        "--target",
        default=SELECTED_TARGET,
        help=f"Single target variable to process. Available targets: {', '.join(AVAILABLE_TARGETS)}",
    )
    parser.add_argument(
        "--regions",
        default=",".join(SELECTED_REGIONS),
        help=(
            "Comma-separated region names. Empty means all available regions in the wide input folder. "
            f"Available regions: {', '.join(AVAILABLE_REGIONS)}"
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_WIDE_INPUT_DIR),
        help="Wide regional input folder containing Allobs_processed_DPE_station_api_*_ALL.csv",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output folder for all Feature_Selection stages.",
    )
    parser.add_argument(
        "--min-sites",
        type=int,
        default=2,
        help="Minimum number of sites required for mirror plots.",
    )
    parser.add_argument(
        "--use-existing-individual-results",
        action="store_true",
        help="Reuse existing Random Forest individual-feature results instead of regenerating them.",
    )
    parser.add_argument("--skip-mirror", action="store_true")
    parser.add_argument("--skip-regional-selection", action="store_true")
    parser.add_argument("--skip-progressive", action="store_true")
    parser.add_argument("--skip-basics", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if os.getenv("FEATURE_SELECTION_RUN_MODE", "").strip().lower() == "event" and output_root.name != "feature_selection_event":
        output_root = output_root / "feature_selection_event"
    regions = [x.strip() for x in args.regions.split(",") if x.strip()]

    if args.target not in AVAILABLE_TARGETS:
        raise ValueError(
            f"Unsupported target '{args.target}'. Choose from: {', '.join(AVAILABLE_TARGETS)}"
        )

    invalid_regions = [region for region in regions if region not in AVAILABLE_REGIONS]
    if invalid_regions:
        raise ValueError(
            "Unsupported region(s): "
            + ", ".join(invalid_regions)
            + f". Choose from: {', '.join(AVAILABLE_REGIONS)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "_pipeline_cache" / safe_name("_".join(regions) if regions else "all_regions")
    prepared_site_dir = cache_dir / "site_inputs"
    prepared_site_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_root / "main_stats.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )

    logging.info("=" * 100)
    logging.info("FEATURE STATS PIPELINE")
    logging.info("=" * 100)
    logging.info("Available targets: %s", AVAILABLE_TARGETS)
    logging.info("Available regions: %s", AVAILABLE_REGIONS)
    logging.info("Target: %s", args.target)
    logging.info("Regions: %s", regions if regions else "ALL_AVAILABLE")
    logging.info("Wide input directory: %s", input_dir)
    logging.info("Output root: %s", output_root)

    site_paths = prepare_site_input_cache(input_dir, prepared_site_dir, regions)
    if not site_paths:
        raise RuntimeError(f"No per-site cache could be prepared from {input_dir}")

    random_forest_output_root = output_root / "00RandomForest_Best_Individual_Feature_Selection"
    random_forest_results_csv = random_forest_output_root / "individual_results" / "all_random_forest_individual_feature_results.csv"
    mirror_output_dir = output_root / "01Plot_Mirror_Region_vs_Site"
    regional_selection_output_dir = output_root / "02Regional_RF_SHAP_Selection"
    progressive_output_dir = output_root / "03Regional_Selected_Feature_Progressive_Evaluation"
    basics_output_root = output_root / "04Basics_Stats"
    selected_features_csv = regional_selection_output_dir / "summary_outputs" / "FINAL_selected_feature_combination_by_region.csv"

    common_env = env_with_pythonpath()
    common_env["FEATURE_SELECTION_OUTPUT_ROOT"] = str(output_root)
    common_env["AQUISTIL_WIDE_INPUT_DIR"] = str(input_dir)
    common_env["FEATURE_SELECTION_REGION_INPUT_DIR"] = str(input_dir)
    common_env["TARGET_COLUMN"] = args.target
    common_env["TARGETS"] = args.target
    common_env["FS_TARGET_COLUMN"] = args.target
    common_env["MIN_SITES"] = str(args.min_sites)
    if regions:
        common_env["TARGET_REGIONS"] = ",".join(regions)
        common_env["FS_TARGET_REGIONS"] = ",".join(regions)
    else:
        common_env.pop("TARGET_REGIONS", None)
        common_env.pop("FS_TARGET_REGIONS", None)

    if not args.use_existing_individual_results or not random_forest_results_csv.exists():
        random_forest_env = dict(common_env)
        random_forest_env["FS_INPUT_DIR"] = str(prepared_site_dir)
        random_forest_env["FS_OUTPUT_ROOT"] = str(random_forest_output_root)
        run_stage(
            "00. Random Forest Best Individual Feature Selection",
            INDIVIDUAL_FEATURE_SCRIPT,
            random_forest_env,
        )
    else:
        logging.info("Reusing existing individual-feature results: %s", random_forest_results_csv)

    if not args.skip_mirror:
        mirror_env = dict(common_env)
        mirror_env["RESULTS_CSV"] = str(random_forest_results_csv)
        mirror_env["OUTPUT_DIR"] = str(mirror_output_dir)
        mirror_env["SAVE_FULL_PDF"] = "0"
        mirror_env["SAVE_INDIVIDUAL_PDF"] = "0"
        mirror_env["SAVE_INDIVIDUAL_PNG"] = "1"
        run_stage("01. Mirror Region vs Site Plot", MIRROR_SCRIPT, mirror_env)

    if not args.skip_regional_selection:
        stage2_env = dict(common_env)
        stage2_env["INPUT_DIR"] = str(prepared_site_dir)
        stage2_env["OUTPUT_DIR"] = str(regional_selection_output_dir)
        run_stage("02. Regional RF-SHAP Feature Selection", REGIONAL_SELECTION_SCRIPT, stage2_env)

    if not args.skip_progressive:
        stage3_env = dict(common_env)
        stage3_env["INPUT_DIR"] = str(prepared_site_dir)
        stage3_env["OUTPUT_DIR"] = str(progressive_output_dir)
        stage3_env["SELECTED_FEATURES_CSV"] = str(selected_features_csv)
        run_stage("03. Regional Selected Feature Progressive Evaluation", PROGRESSIVE_SCRIPT, stage3_env)

    if not args.skip_basics:
        stage4_env = dict(common_env)
        stage4_env["FEATURE_SELECTION_OUTPUT_ROOT"] = str(basics_output_root)
        basics_args = ["--mode", "simple_rf", "--target", args.target]
        if regions:
            basics_args.extend(["--regions", ",".join(regions)])
        run_stage("04. Basics Stats", BASICS_SCRIPT, stage4_env, extra_args=basics_args)

    settings = {
        "input_dir": str(input_dir),
        "prepared_site_dir": str(prepared_site_dir),
        "output_root": str(output_root),
        "target": args.target,
        "regions": regions,
        "min_sites": args.min_sites,
        "random_forest_results_csv": str(random_forest_results_csv),
        "mirror_output_dir": str(mirror_output_dir),
        "regional_selection_output_dir": str(regional_selection_output_dir),
        "progressive_output_dir": str(progressive_output_dir),
        "basics_output_root": str(basics_output_root),
        "selected_features_csv": str(selected_features_csv),
    }
    with open(output_root / "main_stats_run_settings.json", "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    logging.info("Pipeline finished. Settings written to %s", output_root / "main_stats_run_settings.json")


if __name__ == "__main__":
    main()
