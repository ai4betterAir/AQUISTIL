"""Aggregate per-run metrics CSVs into a single summary CSV.

Saves output to: Imputation_Result_Spatial_Temporal_V7/all_results_summary.csv

Heuristics used to extract metadata:
- `Model` and `Regime` are inferred from the file path: .../<Model>/<Regime>/metrics/...
- `StudySite` is inferred from the filename before `_PM` if present, else a filename prefix.

Run:
    python aggregate_metrics.py
"""
import os
import glob
import pandas as pd


COMPARISON_INDEX_COLUMNS = [
    "Region",
    "Site",
    "Target",
    "Regime",
    "Missingness_Level",
    "Missingness_Percent",
    "Seed",
    "Scope",
]
COMPARISON_METRIC_COLUMNS = ["RMSE", "MAE", "R", "NSE", "N_Valid"]


def write_models_comparison(metrics, output_path, models):
    """Write one wide row per matched evaluation for the requested models.

    The source may be either a DataFrame or the path to the central regional
    metrics CSV. Model names remain case-sensitive so output columns match the
    configured ``MODELS_TO_RUN`` names exactly.
    """
    data = pd.read_csv(metrics) if isinstance(metrics, (str, os.PathLike)) else metrics.copy()
    models = list(dict.fromkeys(models))
    required = COMPARISON_INDEX_COLUMNS + ["Model"] + COMPARISON_METRIC_COLUMNS
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Cannot create model comparison; regional metrics are missing columns: "
            + ", ".join(missing_columns)
        )

    selected = data.loc[data["Model"].isin(models), required].copy()
    present_models = set(selected["Model"].dropna().unique())
    missing_models = [model for model in models if model not in present_models]
    if missing_models:
        raise ValueError(
            "Cannot create a complete model comparison; no metrics found for: "
            + ", ".join(missing_models)
        )

    key = COMPARISON_INDEX_COLUMNS + ["Model"]
    duplicates = selected.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(
            "Cannot create an exact model comparison because duplicate "
            f"evaluations were found ({int(duplicates.sum())} rows)"
        )

    wide = selected.pivot(
        index=COMPARISON_INDEX_COLUMNS,
        columns="Model",
        values=COMPARISON_METRIC_COLUMNS,
    )
    ordered_columns = pd.MultiIndex.from_product(
        [COMPARISON_METRIC_COLUMNS, models]
    )
    wide = wide.reindex(columns=ordered_columns)
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()

    comparison_columns = [
        f"{metric}_{model}"
        for metric in COMPARISON_METRIC_COLUMNS
        for model in models
    ]
    incomplete = wide[comparison_columns].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(
            "Cannot create a complete model comparison; "
            f"{int(incomplete.sum())} evaluation keys do not contain every model"
        )

    wide = wide.sort_values(
        ["Region", "Target", "Regime", "Missingness_Percent", "Scope", "Site", "Seed"],
        kind="stable",
    ).reset_index(drop=True)
    wide.to_csv(output_path, index=False)
    return wide


def infer_metadata_from_path(fp, results_root_name=None):
    """Return (model, regime) inferred from the path.

    If `results_root_name` is None the function will try to detect a
    folder starting with `Imputation_Result_Spatial_Temporal_V`.
    """
    parts = fp.replace("\\", "/").split("/")

    # Auto-detect results root folder if not provided
    if results_root_name is None:
        detected = next((p for p in parts if p.startswith("Imputation_Result_Spatial_Temporal_V")), None)
        results_root_name = detected

    if results_root_name and results_root_name in parts:
        i = parts.index(results_root_name)
        # Handle cases where an extra container folder like 'Model Results' or 'Model_results'
        # exists between the results root and the actual model folder.
        next_part = parts[i + 1] if i + 1 < len(parts) else ""
        if next_part in ("Model Results", "Model_results") and i + 2 < len(parts):
            model = parts[i + 2]
            regime = parts[i + 3] if i + 3 < len(parts) else ""
        else:
            model = next_part
            regime = parts[i + 2] if i + 2 < len(parts) else ""
    else:
        model = ""
        regime = ""

    return model, regime


def infer_study_site_from_filename(fname):
    base = os.path.basename(fname)
    if base.endswith("_all_metrics.csv"):
        base = base[: -len("_all_metrics.csv")]

    # StudySite: use the first token before the first underscore
    site = base.split("_")[0]
    return site


def aggregate(results_root):
    pattern = os.path.join(results_root, "**", "*_*_all_metrics.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("No per-run metrics files found under:", results_root)
        return

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Skipping {fp}: failed to read ({e})")
            continue

        model, regime = infer_metadata_from_path(fp, os.path.basename(results_root))
        study_site = infer_study_site_from_filename(fp)

        # Normalize column names
        df = df.copy()
        # Prefer path-inferred regime for MISSINGNESS_REGIMES; fallback to file column if absent
        if regime:
            df["MISSINGNESS_REGIMES"] = regime
        elif "Missingness_Regime" in df.columns:
            df["MISSINGNESS_REGIMES"] = df["Missingness_Regime"]
        else:
            df["MISSINGNESS_REGIMES"] = pd.NA

        # Missingness level value may be 'Missingness' or 'Missingness_Level'
        if "Missingness" in df.columns:
            df["MISSINGNESS_LEVEL"] = df["Missingness"]
        elif "Missingness_Level" in df.columns:
            df["MISSINGNESS_LEVEL"] = df["Missingness_Level"]
        else:
            df["MISSINGNESS_LEVEL"] = pd.NA

        df["Model"] = model
        df["StudySite"] = study_site
        df["Source_File"] = os.path.relpath(fp, results_root)

        # Ensure consistent column ordering later
        dfs.append(df)

    big = pd.concat(dfs, ignore_index=True, sort=False)

    # Reorder columns: Model, MISSINGNESS_REGIMES, MISSINGNESS_LEVEL, StudySite, Source_File, metrics...
    meta_cols = [c for c in ["Model", "MISSINGNESS_REGIMES", "MISSINGNESS_LEVEL", "StudySite", "Source_File"] if c in big.columns]
    other_cols = [c for c in big.columns if c not in meta_cols]
    big = big[meta_cols + other_cols]

    out_fp = os.path.join(results_root, "all_results_summary.csv")
    big.to_csv(out_fp, index=False)
    print(f"Wrote aggregated CSV to: {out_fp} (rows={len(big)})")

    # Do not produce per_model_metrics outputs.
    print("Skipped per-model split output.")
    print("all module finished !!!!!!!!!!")


if __name__ == "__main__":
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.join(workspace_root, "Imputation_Result_Spatial_Temporal_V7")
    if not os.path.isdir(results_root):
        print("Expected results directory not found:", results_root)
    else:
        aggregate(results_root)
