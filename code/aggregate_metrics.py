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
COMPARISON_METRIC_COLUMNS = [
    "RMSE", "RMAE", "R", "R2",
]


def upsert_regional_metrics(existing, current):
    """Merge current-run metrics into persisted metrics by evaluation/model key.

    Current rows replace older rows with the same evaluation identity and
    model. Unrelated historical models and evaluations are retained. The
    function is intentionally side-effect free; callers decide where to save.
    """
    current_data = current.copy()
    if isinstance(existing, (str, os.PathLike)):
        existing_data = pd.read_csv(existing) if os.path.exists(existing) else pd.DataFrame()
    else:
        existing_data = existing.copy() if existing is not None else pd.DataFrame()

    key = COMPARISON_INDEX_COLUMNS + ["Model"]
    missing = [column for column in key if column not in current_data.columns]
    if missing:
        raise ValueError(
            "Cannot upsert regional metrics; current rows are missing key columns: "
            + ", ".join(missing)
        )
    if existing_data.empty:
        merged = current_data
    else:
        existing_missing = [column for column in key if column not in existing_data.columns]
        if existing_missing:
            raise ValueError(
                "Cannot upsert regional metrics; existing file is missing key columns: "
                + ", ".join(existing_missing)
            )
        # Concat existing first and current last so the new run wins conflicts.
        merged = pd.concat([existing_data, current_data], ignore_index=True, sort=False)

    merged = merged.drop_duplicates(key, keep="last")
    sort_columns = [column for column in COMPARISON_INDEX_COLUMNS + ["Model"] if column in merged]
    return merged.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def write_models_comparison(metrics, output_path, models, preserve_existing=True):
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

    configured_by_key = {str(model).casefold(): model for model in models}
    model_keys = data["Model"].astype(str).str.casefold()
    selected = data.loc[model_keys.isin(configured_by_key), required].copy()
    # Module MODEL_NAME values may differ in capitalization from configuration
    # (for example Interpolation vs interpolation). Match robustly while keeping
    # configured spelling in comparison column names.
    selected["Model"] = selected["Model"].astype(str).str.casefold().map(configured_by_key)
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

    # Check row/model coverage independently of metric values. Some statistics
    # are mathematically undefined for constant or very small samples and may
    # correctly be NaN; that must not suppress the complete comparison CSV.
    model_counts = selected.groupby(COMPARISON_INDEX_COLUMNS, dropna=False)["Model"].nunique()
    incomplete_keys = model_counts[model_counts != len(models)]
    if not incomplete_keys.empty:
        raise ValueError(
            "Cannot create a complete model comparison; "
            f"{len(incomplete_keys)} evaluation keys do not contain every model"
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
    wide = wide.sort_values(
        ["Region", "Target", "Regime", "Missingness_Percent", "Scope", "Site", "Seed"],
        kind="stable",
    ).reset_index(drop=True)

    if preserve_existing and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        existing_missing = [
            column for column in COMPARISON_INDEX_COLUMNS
            if column not in existing.columns
        ]
        if existing_missing:
            raise ValueError(
                "Cannot preserve existing model comparison; existing file is "
                "missing key columns: " + ", ".join(existing_missing)
            )

        update_columns = [column for column in wide.columns if column not in COMPARISON_INDEX_COLUMNS]
        old_metric_columns = [
            column for column in existing.columns
            if column not in COMPARISON_INDEX_COLUMNS and column not in update_columns
        ]
        merged = existing.merge(
            wide,
            on=COMPARISON_INDEX_COLUMNS,
            how="outer",
            suffixes=("", "__new"),
        )
        for column in update_columns:
            new_column = "%s__new" % column
            if new_column in merged.columns:
                merged[column] = merged[new_column].combine_first(merged[column])
                merged = merged.drop(columns=[new_column])
        ordered_columns = COMPARISON_INDEX_COLUMNS + old_metric_columns + update_columns
        ordered_columns = list(dict.fromkeys([c for c in ordered_columns if c in merged.columns]))
        wide = merged[ordered_columns].sort_values(
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
