"""Create paired development tables for topology-aware AQUISTIL variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODELS = (
    "AQUISTIL_Current",
    "AQUISTIL",
    "AQUISTIL_GapTrained",
    "LightGBM",
)
METRICS = ("RMSE", "MAE", "R", "NSE", "KGE", "N_Valid")
KEYS = (
    "Target",
    "Region",
    "Site",
    "Scope",
    "Regime",
    "Missingness_Level",
    "Seed",
)


def load_metrics(results_root: Path) -> pd.DataFrame:
    metrics_dir = results_root / "Metrics"
    paths = sorted(metrics_dir.glob("regional_pooled_metrics_*.csv"))
    master = metrics_dir / "regional_pooled_metrics.csv"
    if master.is_file():
        paths.append(master)
    if not paths:
        raise FileNotFoundError(f"No regional pooled metrics under {metrics_dir}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False)
    key = [column for column in (*KEYS, "Model") if column in frame]
    return frame.drop_duplicates(key, keep="last")


def build_tables(
    results_root: Path,
    models=DEFAULT_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_metrics(results_root)
    models = [model for model in models if model in set(frame["Model"])]
    required = {"AQUISTIL_Current", "AQUISTIL", "LightGBM"}
    missing = sorted(required - set(models))
    if missing:
        raise ValueError("Development metrics are missing: " + ", ".join(missing))

    available_metrics = [metric for metric in METRICS if metric in frame]
    detail = frame.loc[frame["Model"].isin(models), [*KEYS, "Model", *available_metrics]].copy()
    reference = detail.loc[
        detail["Model"].eq("LightGBM"), [*KEYS, "RMSE"]
    ].rename(columns={"RMSE": "LightGBM_RMSE"})
    current = detail.loc[
        detail["Model"].eq("AQUISTIL_Current"), [*KEYS, "RMSE"]
    ].rename(columns={"RMSE": "Current_AQUISTIL_RMSE"})
    detail = detail.merge(reference, on=list(KEYS), how="left", validate="many_to_one")
    detail = detail.merge(current, on=list(KEYS), how="left", validate="many_to_one")
    detail["Delta_RMSE_vs_LightGBM"] = detail["RMSE"] - detail["LightGBM_RMSE"]
    detail["Delta_RMSE_vs_Current_AQUISTIL"] = (
        detail["RMSE"] - detail["Current_AQUISTIL_RMSE"]
    )

    group_columns = ["Target", "Region", "Site", "Scope", "Regime", "Missingness_Level", "Model"]
    summary = (
        detail.groupby(group_columns, dropna=False)
        .agg(
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Median=("RMSE", "median"),
            RMSE_SD=("RMSE", "std"),
            RMSE_Q1=("RMSE", lambda values: values.quantile(0.25)),
            RMSE_Q3=("RMSE", lambda values: values.quantile(0.75)),
            Delta_vs_LightGBM_Mean=("Delta_RMSE_vs_LightGBM", "mean"),
            Delta_vs_LightGBM_Median=("Delta_RMSE_vs_LightGBM", "median"),
            Win_Pct_vs_LightGBM=(
                "Delta_RMSE_vs_LightGBM",
                lambda values: 100.0 * np.mean(values < 0),
            ),
            Delta_vs_Current_Mean=("Delta_RMSE_vs_Current_AQUISTIL", "mean"),
            Delta_vs_Current_Median=("Delta_RMSE_vs_Current_AQUISTIL", "median"),
            Win_Pct_vs_Current=(
                "Delta_RMSE_vs_Current_AQUISTIL",
                lambda values: 100.0 * np.mean(values < 0),
            ),
            N_Seeds=("Seed", "nunique"),
        )
        .reset_index()
    )
    summary["RMSE_IQR"] = summary["RMSE_Q3"] - summary["RMSE_Q1"]
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    args = parser.parse_args()

    detail, summary = build_tables(args.results_root, models=args.models)
    metrics_dir = args.results_root / "Metrics"
    detail_path = metrics_dir / "aquistil_regime_aware_development_detail.csv"
    summary_path = metrics_dir / "aquistil_regime_aware_development_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {len(detail)} rows: {detail_path}")
    print(f"Saved {len(summary)} rows: {summary_path}")


if __name__ == "__main__":
    main()
