"""Summarize targeted AQUISTIL ablation metrics against LightGBM."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parents[2] / "Outputs" / "AQUISTIL_Ablation"
)
SCOPES = ("Site", "Region_Macro", "Region_Micro")
ABLATION_MODELS = (
    "AQUISTIL_NoHistory",
    "AQUISTIL_NoHistoryNoEvent",
    "AQUISTIL_NoFFill",
    "AQUISTIL_NoAdaptive",
    "AQUISTIL_ExogenousOnly",
    "AQUISTIL_NoAQUISTILFeatures",
)
VARIANT_MODELS = ("AQUISTIL", *ABLATION_MODELS)
REQUIRED_MODELS = (*VARIANT_MODELS, "LightGBM")


def _load_metrics(
    results_root: Path,
    metrics_dir: Path | None = None,
) -> pd.DataFrame:
    metrics_dir = metrics_dir or results_root / "Metrics"
    frames = []
    master_path = metrics_dir / "regional_pooled_metrics.csv"
    if master_path.is_file():
        frames.append(pd.read_csv(master_path))
    for path in sorted(metrics_dir.glob("regional_pooled_metrics_*.csv")):
        frames.append(pd.read_csv(path))
    dedicated_path = metrics_dir / "aquistil_ablation_metrics.csv"
    if dedicated_path.is_file():
        frames.append(pd.read_csv(dedicated_path))
    if not frames:
        raise FileNotFoundError(
            f"No regional or dedicated ablation metrics files under {metrics_dir}"
        )
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    key = [
        "Region", "Site", "Target", "Model", "Regime",
        "Missingness_Level", "Missingness_Percent", "Seed", "Scope",
    ]
    key = [column for column in key if column in metrics.columns]
    return metrics.drop_duplicates(key, keep="last")


def build_summary(
    results_root: Path,
    scope: str = "All",
    metrics_dir: Path | None = None,
) -> pd.DataFrame:
    frame = _load_metrics(results_root, metrics_dir=metrics_dir)
    if scope != "All" and scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; expected All or one of {SCOPES}")
    scope_mask = frame["Scope"].isin(SCOPES) if scope == "All" else frame["Scope"].eq(scope)
    frame = frame.loc[
        scope_mask
        & frame["Model"].isin(REQUIRED_MODELS)
    ].copy()
    missing_models = [model for model in REQUIRED_MODELS if model not in set(frame["Model"])]
    if missing_models:
        raise ValueError(
            "Cannot summarize AQUISTIL ablations because source metrics are missing: "
            + ", ".join(missing_models)
            + ". Run main.py with --run-aquistil-ablations first."
        )
    keys = ["Scope", "Target", "Region", "Regime", "Missingness_Level", "Seed"]
    coverage = frame.groupby(keys, dropna=False)["Model"].nunique()
    incomplete = coverage[coverage != len(REQUIRED_MODELS)]
    if not incomplete.empty:
        raise ValueError(
            "Cannot summarize an incomplete ablation run: "
            f"{len(incomplete)} evaluation keys do not contain all "
            f"{len(REQUIRED_MODELS)} required models."
        )
    averaged = (
        frame.groupby(keys + ["Model"], dropna=False)["RMSE"]
        .mean()
        .reset_index()
    )
    pivot = averaged.pivot_table(index=keys, columns="Model", values="RMSE", aggfunc="mean")
    if "LightGBM" not in pivot.columns:
        raise ValueError("LightGBM rows are required for ablation comparison")

    rows = []
    for variant in VARIANT_MODELS:
        comparison = pivot[[variant, "LightGBM"]].dropna().reset_index()
        comparison["Variant"] = variant
        comparison["AQUISTIL_RMSE"] = comparison[variant]
        comparison["LightGBM_RMSE"] = comparison["LightGBM"]
        comparison["Delta"] = comparison["AQUISTIL_RMSE"] - comparison["LightGBM_RMSE"]
        rows.append(
            comparison[
                [
                    "Scope",
                    "Target",
                    "Region",
                    "Regime",
                    "Missingness_Level",
                    "Seed",
                    "Variant",
                    "AQUISTIL_RMSE",
                    "LightGBM_RMSE",
                    "Delta",
                ]
            ]
        )
    detail = pd.concat(rows, ignore_index=True, sort=False)
    summary = (
        detail.groupby(
            ["Scope", "Target", "Region", "Regime", "Variant"], dropna=False
        )
        .agg(
            AQUISTIL_RMSE=("AQUISTIL_RMSE", "mean"),
            LightGBM_RMSE=("LightGBM_RMSE", "mean"),
            Delta=("Delta", "mean"),
            N_Pairs=("Delta", "size"),
        )
        .reset_index()
        .sort_values(["Scope", "Target", "Region", "Regime", "Variant"])
    )
    return summary


def write_summary_files(
    results_root: Path,
    scope: str = "All",
    output: Path | None = None,
    metrics_dir: Path | None = None,
) -> dict[str, Path]:
    metrics_dir = metrics_dir or results_root / "Metrics"
    summary = build_summary(results_root, scope=scope, metrics_dir=metrics_dir)
    default_name = (
        "aquistil_ablation_summary_all_scopes.csv"
        if scope == "All"
        else f"aquistil_ablation_summary_{scope}.csv"
    )
    combined_output = output or metrics_dir / default_name
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(combined_output, index=False)
    written = {scope: combined_output}

    if scope == "All":
        for selected_scope in SCOPES:
            scoped_output = (
                metrics_dir
                / f"aquistil_ablation_summary_{selected_scope}.csv"
            )
            summary.loc[summary["Scope"].eq(selected_scope)].to_csv(
                scoped_output, index=False
            )
            written[selected_scope] = scoped_output
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--scope",
        default="All",
        choices=["All", *SCOPES],
        help="Default All writes a combined summary and one file per scope.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Optional metrics directory, for example 'Outputs/Imputation_Results/Metrics copy'.",
    )
    args = parser.parse_args()

    summary = build_summary(
        args.results_root,
        scope=args.scope,
        metrics_dir=args.metrics_dir,
    )
    written = write_summary_files(
        args.results_root,
        args.scope,
        args.output,
        metrics_dir=args.metrics_dir,
    )
    for output_scope, path in written.items():
        row_count = len(summary) if output_scope == args.scope else int(
            summary["Scope"].eq(output_scope).sum()
        )
        print(f"Saved {row_count} rows: {path}")
    print(summary.to_string(index=False, max_rows=80))


if __name__ == "__main__":
    main()
