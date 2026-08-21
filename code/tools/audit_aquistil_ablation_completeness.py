#!/usr/bin/env python3
"""Validate exact model coverage for the development ablation study."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODELS = (
    "AQUISTIL",
    "AQUISTIL_NoHistory",
    "AQUISTIL_NoHistoryNoEvent",
    "AQUISTIL_NoFFill",
    "AQUISTIL_NoAdaptive",
    "AQUISTIL_ExogenousOnly",
    "AQUISTIL_NoAQUISTILFeatures",
    "LightGBM",
)
REGIONS = (
    "Central Coast",
    "Central Tablelands",
    "Sydney North west",
    "Sydney South west",
)
TARGETS = ("PM10", "PM2.5")
REGIMES = ("random", "short_gap", "medium_gap", "long_gap", "event")
LEVELS = (0.05, 0.10, 0.20, 0.30)
SEEDS = (13, 29, 42, 77, 101, 137, 211, 307, 401, 503)
SCOPES = ("Site", "Region_Macro", "Region_Micro")
EXPECTED_ROWS_PER_MODEL = 9_200
KEY = (
    "Region",
    "Site",
    "Target",
    "Regime",
    "Missingness_Level",
    "Missingness_Percent",
    "Seed",
    "Scope",
)


def _canon_region(value: object) -> str:
    return " ".join(str(value).replace("-", " ").split()).casefold()


def audit(metrics: pd.DataFrame) -> pd.DataFrame:
    required = set(KEY) | {"Model", "RMSE"}
    missing_columns = sorted(required.difference(metrics.columns))
    if missing_columns:
        raise ValueError("Metrics are missing columns: " + ", ".join(missing_columns))

    data = metrics.loc[metrics["Model"].isin(MODELS)].copy()
    data["_Region"] = data["Region"].map(_canon_region)
    data["_Level"] = pd.to_numeric(data["Missingness_Level"], errors="coerce").round(12)
    expected_regions = {_canon_region(region) for region in REGIONS}
    protocol = data.loc[
        data["_Region"].isin(expected_regions)
        & data["Target"].isin(TARGETS)
        & data["Regime"].isin(REGIMES)
        & data["_Level"].isin(LEVELS)
        & data["Seed"].isin(SEEDS)
        & data["Scope"].isin(SCOPES)
    ].copy()

    normalized_key = list(KEY)
    normalized_key[0] = "_Region"
    normalized_key[4] = "_Level"
    duplicate_key = normalized_key + ["Model"]
    duplicates = protocol.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Ablation metrics contain {int(duplicates.sum())} duplicate evaluation/model rows"
        )

    reference = protocol.loc[protocol["Model"].eq("AQUISTIL"), normalized_key]
    if reference.empty:
        raise ValueError("AQUISTIL reference rows are missing")
    reference = reference.drop_duplicates()
    reference_index = pd.MultiIndex.from_frame(reference)
    reference_protocol_missing = max(0, EXPECTED_ROWS_PER_MODEL - len(reference_index))

    rows = []
    for model in MODELS:
        model_rows = protocol.loc[protocol["Model"].eq(model)]
        model_keys = model_rows[normalized_key].drop_duplicates()
        model_index = pd.MultiIndex.from_frame(model_keys)
        missing = reference_index.difference(model_index)
        extra = model_index.difference(reference_index)
        missing_count = len(missing) + reference_protocol_missing
        rows.append(
            {
                "Model": model,
                "Rows": len(model_rows),
                "Expected_Rows": EXPECTED_ROWS_PER_MODEL,
                "Completion_Percent": 100.0 * len(model_index) / EXPECTED_ROWS_PER_MODEL,
                "Missing_Keys": missing_count,
                "Extra_Keys": len(extra),
                "Null_RMSE": int(model_rows["RMSE"].isna().sum()),
                "Complete": len(model_index) == EXPECTED_ROWS_PER_MODEL
                and not missing_count and not len(extra)
                and not model_rows["RMSE"].isna().any(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "Outputs/Imputation_Results/Metrics copy/aquistil_ablation_metrics.csv"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = audit(pd.read_csv(args.metrics))
    output = args.output or args.metrics.with_name(
        "aquistil_ablation_completeness.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved completeness audit: {output}")
    if not result["Complete"].all():
        raise SystemExit("AQUISTIL ablation assessment is incomplete")


if __name__ == "__main__":
    main()
