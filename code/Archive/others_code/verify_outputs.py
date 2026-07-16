"""
Verify existence of expected output artifacts for each Model × Regime × Site × Missingness.

Checks:
 - metrics CSVs (per-level and combined all-level)
 - imputed_data CSVs
 - target_column_data CSVs
 - scatter plots (plots/Scatterplot)
 - saved run manifest JSON (metrics folder)

Special-case: BaseLine model often writes baseline_all_<site>.csv files at model-level
without the per-regime subfolder structure. This script will treat those baseline files
as present for every regime × missingness combination so they appear in the verification
report (matching your request).

Usage:
    python verify_outputs.py --results_dir /path/to/Imputation_Result_Spatial_Temporal_V19_final \
                             --regimes random short_gap medium_gap long_gap event \
                             --missingness_levels 10 20 30 50 \
                             --out_csv verification_report.csv
"""
import argparse
import os
import glob
import pandas as pd
from typing import List, Tuple, Dict, Optional

# Default results directory (set to the path you provided)
DEFAULT_RESULTS_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V19_final"

def find_models(results_dir: str) -> List[str]:
    try:
        return [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    except Exception:
        return []

def _list_files_safe(dirpath: str, pattern: str) -> List[str]:
    try:
        return glob.glob(os.path.join(dirpath, pattern))
    except Exception:
        return []

def _parse_metrics_filename(fn: str, model: str, regime: str) -> Optional[Tuple[str, str, str]]:
    """
    Given a basename like:
      CHULLORA_AQMS_Processed_PM2.5_Mean_random_10_all_metrics.csv
    and model='Mean', regime='random', return (site, target, missingness_str) where:
      site = 'CHULLORA_AQMS_Processed'
      target = 'PM2.5'
      missingness_str = '10' or 'all'
    If parsing fails, return None.
    """
    basename = os.path.basename(fn)
    token = f"_{model}_{regime}_"
    if token not in basename:
        return None
    prefix = basename.split(token, 1)[0]  # e.g. "CHULLORA_AQMS_Processed_PM2.5"
    rest = basename.split(token, 1)[1]
    parts = rest.split("_all_metrics", 1)
    if not parts:
        return None
    miss_part = parts[0]  # e.g. "10" or "all"
    if "_" in prefix:
        site, target = prefix.rsplit("_", 1)
    else:
        site, target = prefix, ""
    return site, target, miss_part

def _expected_paths_for(site: str, target: str, model: str, regime: str, miss: str, base_dir: str) -> Dict[str, str]:
    """
    Build expected file paths for imputed_data, target_column_data, metrics, scatter plot.
    miss: e.g. '10' or 'all'
    """
    metrics_dir = os.path.join(base_dir, model, regime, "metrics")
    imputed_dir = os.path.join(base_dir, model, regime, "imputed_data")
    target_col_dir = os.path.join(base_dir, model, regime, "target_column_data")
    plot_scatter_dir = os.path.join(base_dir, model, regime, "plots", "Scatterplot")

    # Two observed metrics filename styles:
    # per-level: <site>_<target>_<Model>_<regime>_<miss>_all_metrics.csv
    # combined: <site>_<target>_<Model>_<regime>_all_metrics.csv
    if miss == "all":
        metrics_name = f"{site}_{target}_{model}_{regime}_all_metrics.csv"
    else:
        metrics_name = f"{site}_{target}_{model}_{regime}_{miss}_all_metrics.csv"

    imputed_name = f"{site}_{target}_{model}_{regime}_imputed_{miss}.csv"
    target_name = f"{site}_{target}_{model}_{regime}_target_column_{miss}.csv"
    scatter_name = f"{site}_{target}_{model}_{regime}_Scatterplots_{miss}.png"

    return {
        "metrics": os.path.join(metrics_dir, metrics_name),
        "imputed": os.path.join(imputed_dir, imputed_name),
        "target_column": os.path.join(target_col_dir, target_name),
        "scatter": os.path.join(plot_scatter_dir, scatter_name),
        "manifest": os.path.join(metrics_dir, f"saved_runs_manifest_{model}_{regime}.json")
    }

def scan(results_dir: str, regimes: List[str], missingness_levels: List[int]) -> pd.DataFrame:
    """
    Scan results_dir for models/regimes and report presence/absence of expected artifacts.

    Special-case handling:
      - If a model directory has no regime subfolders but contains baseline_all_<site>.csv files
        (typical for BaseLine), the script will generate rows for every regime × missingness level
        marking the baseline as present (imputed_present=True) for each combination.
    """
    rows = []
    models = find_models(results_dir)
    if not models:
        raise SystemExit(f"No model folders found in results_dir: {results_dir}")

    miss_levels = [str(int(m)) for m in missingness_levels] if missingness_levels else []

    for model in sorted(models):
        model_dir = os.path.join(results_dir, model)

        # list regime subfolders under this model dir
        regime_subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

        # Special-case: BaseLine or any model that writes baseline_all_<site>.csv directly under model_dir
        # Treat baseline files as present even if regime subfolders exist (they are model-level artifacts)
        baseline_files = _list_files_safe(model_dir, "baseline_all_*.csv")

        if baseline_files:
            # Create entries for each discovered baseline file across all regimes and missingness levels
            for bf in baseline_files:
                bn = os.path.basename(bf)
                # basename pattern: baseline_all_<site>.csv
                site = bn.replace("baseline_all_", "").rsplit(".csv", 1)[0]
                # For baseline we don't have a 'target' token in filename; leave empty or None
                target = None
                for regime in regimes:
                    # report per missingness levels + combined 'all'
                    for miss in miss_levels + ["all"]:
                        row = {
                            "model": model,
                            "regime": regime,
                            "site": site,
                            "target": target,
                            "missingness": miss,
                            # baseline file is treated as the 'imputed' artifact (present)
                            "metrics_present": False,
                            "imputed_present": True,
                            "target_column_present": False,
                            "scatter_present": False,
                            # no manifest for baseline typically
                            "manifest_present": False
                        }
                        rows.append(row)
            # continue to next model (we treat baseline-models as handled)
            continue

        # Otherwise, treat as standard model with regime subfolders
        for regime in regimes:
            metrics_dir = os.path.join(results_dir, model, regime, "metrics")
            imputed_dir = os.path.join(results_dir, model, regime, "imputed_data")
            plot_scatter_dir = os.path.join(results_dir, model, regime, "plots", "Scatterplot")
            target_col_dir = os.path.join(results_dir, model, regime, "target_column_data")
            manifest_path = os.path.join(metrics_dir, f"saved_runs_manifest_{model}_{regime}.json")

            # discover runs by inspecting metrics first, then imputed_data fallback
            present_metrics = _list_files_safe(metrics_dir, "*_all_metrics.csv")
            discovered = []
            if present_metrics:
                for mf in present_metrics:
                    parsed = _parse_metrics_filename(mf, model, regime)
                    if parsed:
                        site, target, miss = parsed
                        discovered.append((site, target, miss))
                    else:
                        base = os.path.basename(mf)
                        if f"_{model}_{regime}_" in base:
                            prefix = base.split(f"_{model}_{regime}_")[0]
                            if "_" in prefix:
                                site, target = prefix.rsplit("_", 1)
                            else:
                                site, target = prefix, ""
                            discovered.append((site, target, "all"))
                        else:
                            discovered.append((None, None, "unknown"))
            else:
                # fallback to imputed_data filenames
                imputed_files = _list_files_safe(imputed_dir, "*_imputed_*.csv")
                for imf in imputed_files:
                    b = os.path.basename(imf)
                    token = f"_{model}_{regime}_imputed_"
                    if token in b:
                        prefix = b.split(token)[0]
                        if "_" in prefix:
                            site, target = prefix.rsplit("_", 1)
                        else:
                            site, target = prefix, ""
                        miss = b.split(token)[1].split(".csv")[0]
                        discovered.append((site, target, miss))

            site_target_set = set((s, t) for (s, t, m) in discovered if s)
            if not site_target_set:
                # no discovered runs for this model/regime
                row = {
                    "model": model,
                    "regime": regime,
                    "site": None,
                    "target": None,
                    "missingness": None,
                    "metrics_dir_exists": os.path.isdir(metrics_dir),
                    "imputed_dir_exists": os.path.isdir(imputed_dir),
                    "plots_scatter_exists": os.path.isdir(plot_scatter_dir),
                    "target_col_dir_exists": os.path.isdir(target_col_dir),
                    "manifest_exists": os.path.isfile(manifest_path),
                    "artifact": "NO_DISCOVERED_RUNS"
                }
                rows.append(row)
                continue

            # For each discovered site/target, check expected files for each missingness level and combined 'all'
            for (site, target) in sorted(site_target_set):
                for miss in miss_levels + ["all"]:
                    paths = _expected_paths_for(site, target, model, regime, miss, results_dir)
                    row = {
                        "model": model,
                        "regime": regime,
                        "site": site,
                        "target": target,
                        "missingness": miss,
                        "metrics_present": os.path.isfile(paths["metrics"]),
                        "imputed_present": os.path.isfile(paths["imputed"]),
                        "target_column_present": os.path.isfile(paths["target_column"]),
                        "scatter_present": os.path.isfile(paths["scatter"]),
                        "manifest_present": os.path.isfile(paths["manifest"])
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    # normalize boolean columns to explicit True/False where present
    bool_cols = ["metrics_present", "imputed_present", "target_column_present", "scatter_present", "manifest_present",
                 "metrics_dir_exists", "imputed_dir_exists", "plots_scatter_exists", "target_col_dir_exists"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=False, default=DEFAULT_RESULTS_DIR,
                    help=f"Path to results directory (default: {DEFAULT_RESULTS_DIR})")
    ap.add_argument("--regimes", nargs="+", default=["random","short_gap","medium_gap","long_gap","event"])
    ap.add_argument("--missingness_levels", nargs="*", type=int, default=[10,20,30,50],
                    help="Missingness levels (percent) to check; examples: 10 20 30 50")
    ap.add_argument("--out_csv", required=False, default=None, help="Optional CSV path to save verification report")
    args = ap.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        raise SystemExit(f"Results directory not found: {results_dir}")

    df = scan(results_dir, args.regimes, args.missingness_levels)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))

    # Provide a quick summary by model/regime of missing artifacts
    summary = []
    if not df.empty and "model" in df.columns and "regime" in df.columns:
        grouped = df.groupby(["model", "regime"], dropna=False)
        for (model, regime), group in grouped:
            total = len(group)
            missing_metrics = int((~group["metrics_present"].fillna(False)).sum()) if "metrics_present" in group else 0
            missing_imputed = int((~group["imputed_present"].fillna(False)).sum()) if "imputed_present" in group else 0
            missing_scatter = int((~group["scatter_present"].fillna(False)).sum()) if "scatter_present" in group else 0
            summary.append({
                "model": model,
                "regime": regime,
                "rows_checked": total,
                "missing_metrics_count": missing_metrics,
                "missing_imputed_count": missing_imputed,
                "missing_scatter_count": missing_scatter
            })
    if summary:
        print("\nSUMMARY (per model/regime):")
        print(pd.DataFrame(summary).to_string(index=False))

    if args.out_csv:
        try:
            df.to_csv(args.out_csv, index=False)
            print(f"\nWrote verification CSV to {args.out_csv}")
        except Exception as e:
            print(f"Failed to write CSV: {e}")

if __name__ == "__main__":
    main()