#!/usr/bin/env python3
"""Validate and summarize the frozen paired AQUISTIL/LightGBM experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("AQUISTIL", "LightGBM")
METRICS = ("RMSE", "MAE", "R", "NSE")
PAIR_KEYS = (
    "Region",
    "Site",
    "Target",
    "Regime",
    "Missingness_Level",
    "Missingness_Percent",
    "Seed",
    "Scope",
)
EXPECTED_TARGETS = ("PM10", "PM2.5")
EXPECTED_REGIMES = ("random", "short_gap", "medium_gap", "long_gap", "event")
EXPECTED_LEVELS = (0.05, 0.10, 0.20, 0.30)
EXPECTED_SEEDS = (13, 29, 42, 77, 101, 137, 211, 307, 401, 503)
EXPECTED_SCOPES = ("Site", "Region_Macro", "Region_Micro")
EXPECTED_REGIONS = (
    "Lower Hunter",
    "Northern Tablelands",
    "Southern Tablelands",
    "Sydney East",
    "Upper Hunter",
)


def _canonical_text(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("_", " ", regex=False).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()


def load_metrics(results_root: Path) -> pd.DataFrame:
    path = results_root / "Metrics" / "regional_pooled_metrics.csv"
    if not path.exists():
        target_paths = [
            results_root / "Metrics" / f"regional_pooled_metrics_{target}.csv"
            for target in EXPECTED_TARGETS
        ]
        missing = [str(item) for item in target_paths if not item.exists()]
        if missing:
            raise FileNotFoundError("Missing frozen metric files: " + ", ".join(missing))
        data = pd.concat([pd.read_csv(item) for item in target_paths], ignore_index=True)
    else:
        data = pd.read_csv(path)

    required = set(PAIR_KEYS) | {"Model", "Mask_SHA256", "N_Masked", "N_Valid"} | set(METRICS)
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError("Metrics are missing required columns: " + ", ".join(missing_columns))

    data = data.copy()
    data["Region"] = _canonical_text(data["Region"])
    data["Regime"] = data["Regime"].astype(str).str.strip()
    data["Target"] = data["Target"].astype(str).str.strip()
    data["Model"] = data["Model"].astype(str).str.strip()
    data["Scope"] = data["Scope"].astype(str).str.strip()
    data["Missingness_Level"] = pd.to_numeric(data["Missingness_Level"], errors="coerce")
    data["Seed"] = pd.to_numeric(data["Seed"], errors="coerce").astype("Int64")
    return data.loc[data["Model"].isin(MODELS)].copy()


def validate_protocol(data: pd.DataFrame) -> pd.DataFrame:
    errors = []
    for label, actual, expected in (
        ("regions", set(data["Region"].dropna()), set(EXPECTED_REGIONS)),
        ("targets", set(data["Target"].dropna()), set(EXPECTED_TARGETS)),
        ("models", set(data["Model"].dropna()), set(MODELS)),
        ("regimes", set(data["Regime"].dropna()), set(EXPECTED_REGIMES)),
        ("levels", set(data["Missingness_Level"].dropna().astype(float)), set(EXPECTED_LEVELS)),
        ("seeds", set(data["Seed"].dropna().astype(int)), set(EXPECTED_SEEDS)),
        ("scopes", set(data["Scope"].dropna()), set(EXPECTED_SCOPES)),
    ):
        if actual != expected:
            errors.append(
                f"{label}: missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
            )

    duplicate_key = list(PAIR_KEYS) + ["Model"]
    duplicates = data.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        errors.append(f"duplicate model/evaluation rows={int(duplicates.sum())}")
    if errors:
        raise ValueError("Frozen validation protocol is incomplete or contaminated: " + "; ".join(errors))
    return data


def build_pairs(data: pd.DataFrame) -> pd.DataFrame:
    model_frames = {}
    retained = list(PAIR_KEYS) + ["Mask_SHA256", "N_Masked", "N_Valid"] + list(METRICS)
    for model in MODELS:
        frame = data.loc[data["Model"].eq(model), retained].copy()
        frame = frame.rename(
            columns={
                column: f"{column}_{model}"
                for column in retained
                if column not in PAIR_KEYS
            }
        )
        model_frames[model] = frame

    pairs = model_frames["AQUISTIL"].merge(
        model_frames["LightGBM"],
        on=list(PAIR_KEYS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unpaired = pairs.loc[pairs["_merge"].ne("both")]
    if not unpaired.empty:
        raise ValueError(f"Found {len(unpaired)} unpaired model evaluations")
    pairs = pairs.drop(columns="_merge")

    mask_mismatch = pairs["Mask_SHA256_AQUISTIL"].ne(pairs["Mask_SHA256_LightGBM"])
    count_mismatch = pairs["N_Masked_AQUISTIL"].ne(pairs["N_Masked_LightGBM"])
    if mask_mismatch.any() or count_mismatch.any():
        raise ValueError(
            "AQUISTIL and LightGBM did not use identical masks: "
            f"hash mismatches={int(mask_mismatch.sum())}, count mismatches={int(count_mismatch.sum())}"
        )

    for metric in METRICS:
        pairs[f"Delta_{metric}"] = pairs[f"{metric}_AQUISTIL"] - pairs[f"{metric}_LightGBM"]
    pairs["RMSE_Ratio_AQUISTIL_to_LightGBM"] = (
        pairs["RMSE_AQUISTIL"] / pairs["RMSE_LightGBM"].replace(0, np.nan)
    )
    pairs["AQUISTIL_RMSE_Win"] = pairs["Delta_RMSE"] < 0
    return pairs


def _cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    value_column: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    finite = frame.loc[np.isfinite(pd.to_numeric(frame[value_column], errors="coerce"))].copy()
    if finite.empty:
        return np.nan, np.nan
    clusters = finite["Region"].dropna().unique()
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    if len(clusters) >= 2:
        grouped = {region: finite.loc[finite["Region"].eq(region), value_column].to_numpy(float)
                   for region in clusters}
        for index in range(iterations):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            estimates[index] = np.mean(np.concatenate([grouped[region] for region in sampled]))
    else:
        values = finite[value_column].to_numpy(float)
        for index in range(iterations):
            estimates[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def summarize_pairs(pairs: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rows = []
    for group_number, ((target, regime, scope), frame) in enumerate(
        pairs.groupby(["Target", "Regime", "Scope"], sort=True, dropna=False)
    ):
        delta = pd.to_numeric(frame["Delta_RMSE"], errors="coerce").dropna()
        lower, upper = _cluster_bootstrap_mean_ci(
            frame, "Delta_RMSE", iterations, seed + group_number * 1009
        )
        row = {
            "Target": target,
            "Regime": regime,
            "Scope": scope,
            "N_Pairs": int(len(delta)),
            "N_Regions": int(frame["Region"].nunique()),
            "Delta_RMSE_Mean": float(delta.mean()),
            "Delta_RMSE_Median": float(delta.median()),
            "Delta_RMSE_Q1": float(delta.quantile(0.25)),
            "Delta_RMSE_Q3": float(delta.quantile(0.75)),
            "Delta_RMSE_IQR": float(delta.quantile(0.75) - delta.quantile(0.25)),
            "Delta_RMSE_Mean_CI95_Lower": lower,
            "Delta_RMSE_Mean_CI95_Upper": upper,
            "AQUISTIL_RMSE_Win_Percent": float((delta < 0).mean() * 100),
            "RMSE_Tie_Percent": float((delta == 0).mean() * 100),
        }
        for metric in METRICS:
            for model in MODELS:
                values = pd.to_numeric(frame[f"{metric}_{model}"], errors="coerce")
                row[f"{metric}_{model}_Mean"] = float(values.mean())
                row[f"{metric}_{model}_Median"] = float(values.median())
            values = pd.to_numeric(frame[f"Delta_{metric}"], errors="coerce")
            row[f"Delta_{metric}_Mean"] = float(values.mean())
            row[f"Delta_{metric}_Median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_tables(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        rows = []
        for keys, part in frame.groupby(groups, sort=True, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            ratio = pd.to_numeric(part["RMSE_Ratio_AQUISTIL_to_LightGBM"], errors="coerce")
            delta = pd.to_numeric(part["Delta_RMSE"], errors="coerce")
            row = dict(zip(groups, keys))
            row.update(
                N_Pairs=int(delta.notna().sum()),
                Delta_RMSE_Mean=float(delta.mean()),
                Delta_RMSE_Median=float(delta.median()),
                RMSE_Ratio_Median=float(ratio.median()),
                AQUISTIL_RMSE_Win_Percent=float((delta < 0).mean() * 100),
                RMSE_AQUISTIL_Median=float(part["RMSE_AQUISTIL"].median()),
                RMSE_LightGBM_Median=float(part["RMSE_LightGBM"].median()),
            )
            rows.append(row)
        result = pd.DataFrame(rows)
        is_gap = result["Regime"].isin(["short_gap", "medium_gap", "long_gap"])
        result["Difficult_Case"] = (
            is_gap
            & result["RMSE_Ratio_Median"].ge(2.0)
            & result["AQUISTIL_RMSE_Win_Percent"].le(20.0)
        )
        return result.sort_values(
            ["Difficult_Case", "RMSE_Ratio_Median", "Delta_RMSE_Median"],
            ascending=[False, False, False],
            kind="stable",
        ).reset_index(drop=True)

    site = aggregate(
        pairs.loc[pairs["Scope"].eq("Site")],
        ["Target", "Region", "Site", "Regime"],
    )
    region = aggregate(
        pairs.loc[pairs["Scope"].eq("Region_Micro")],
        ["Target", "Region", "Regime"],
    )
    return site, region


def coverage_table(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["Target", "Region", "Regime", "Scope", "Model"], dropna=False)
        .agg(
            Rows=("RMSE", "size"),
            Missingness_Levels=("Missingness_Level", "nunique"),
            Seeds=("Seed", "nunique"),
            Valid_Metric_Rows=("N_Valid", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("Outputs/Final_Frozen"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 1000:
        raise ValueError("Use at least 1000 bootstrap iterations")
    data = validate_protocol(load_metrics(args.results_root))
    pairs = build_pairs(data)
    summary = summarize_pairs(pairs, args.bootstrap_iterations, args.bootstrap_seed)
    site_robustness, region_robustness = robustness_tables(pairs)
    coverage = coverage_table(data)

    output_dir = args.results_root / "Statistical_Comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paired_aquistil_lightgbm_metrics.csv": pairs,
        "paired_delta_rmse_summary.csv": summary,
        "site_robustness.csv": site_robustness,
        "region_robustness.csv": region_robustness,
        "protocol_coverage.csv": coverage,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    difficult = int(site_robustness["Difficult_Case"].sum())
    print(f"Validated {len(pairs):,} exact AQUISTIL/LightGBM pairs")
    print(f"Saved statistical outputs to {output_dir}")
    print(f"Pre-specified difficult site/regime cases: {difficult}")


if __name__ == "__main__":
    main()
