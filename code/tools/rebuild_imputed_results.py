#!/usr/bin/env python3
"""
Rebuild continuous Actual/Imputed time-series CSVs from regional masked predictions.

The regional pooled evaluator stores predictions only for artificially masked rows.
This utility merges those predictions back onto each full per-site input series and
writes one continuous CSV per model/site/target/regime/missingness combination under:

    Outputs/Imputation_Result/Imputed_Results
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import pandas as pd


TARGET_ALIASES = {
    "PM25": "PM2.5",
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "NO2": "NO2",
    "NO": "NO",
    "OZONE": "OZONE",
}

_SITE_FRAMES = {}


def safe_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    return token.strip("_") or "unknown"


def target_to_input_column(target: str) -> str:
    return TARGET_ALIASES.get(str(target).upper().replace(".", ""), str(target))


def read_site_inputs(inputs_dir: Path) -> Dict[str, pd.DataFrame]:
    site_frames = {}
    for path in sorted(inputs_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if "DateTime" not in df.columns:
            continue
        df = df.copy()
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce", utc=True)
        df = (
            df.dropna(subset=["DateTime"])
            .sort_values("DateTime")
            .drop_duplicates(subset=["DateTime"], keep="last")
        )
        if not df.empty:
            complete_hours = pd.date_range(
                start=df["DateTime"].min(),
                end=df["DateTime"].max(),
                freq="h",
                tz="UTC",
            )
            df = (
                df.set_index("DateTime")
                .reindex(complete_hours)
                .rename_axis("DateTime")
                .reset_index()
            )
        site_frames[safe_token(path.stem).upper()] = df
    return site_frames


def rebuild_file(pred_path: Path, site_frames: Dict[str, pd.DataFrame], output_dir: Path, skip_existing: bool) -> int:
    pred = pd.read_csv(pred_path)
    required = {"DateTime", "Site", "Target", "Model", "Regime", "Missingness_Level", "Observed", "Imputed"}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"{pred_path} is missing columns: {sorted(missing)}")

    pred = pred.copy()
    pred["DateTime"] = pd.to_datetime(pred["DateTime"], errors="coerce", utc=True)
    pred = pred.dropna(subset=["DateTime", "Site", "Target", "Model", "Regime", "Missingness_Level"])

    written = 0
    group_cols = ["Site", "Target", "Model", "Regime", "Missingness_Level"]
    for (site, target, model, regime, missingness), group in pred.groupby(group_cols, dropna=False):
        filename = (
            f"{safe_token(target)}_{safe_token(site)}_{safe_token(model)}_"
            f"{safe_token(regime)}_{int(round(float(missingness) * 100))}_imputed.csv"
        )
        out_path = output_dir / filename
        if skip_existing and out_path.exists():
            continue

        site_key = safe_token(site).upper()
        if site_key not in site_frames:
            continue

        input_col = target_to_input_column(str(target))
        site_df = site_frames[site_key]
        if input_col not in site_df.columns:
            continue

        base = site_df[["DateTime", input_col]].copy()
        base = base.rename(columns={input_col: "Actual"})

        preds = (
            group[["DateTime", "Imputed"]]
            .dropna(subset=["DateTime"])
            .sort_values("DateTime")
            .drop_duplicates(subset=["DateTime"], keep="last")
        )
        merged = base.merge(preds, on="DateTime", how="left", suffixes=("", "_Prediction"))
        prediction_col = "Imputed_Prediction" if "Imputed_Prediction" in merged.columns else "Imputed"
        has_prediction = merged[prediction_col].notna()
        has_actual = merged["Actual"].notna()
        merged["Imputed"] = merged[prediction_col].where(has_prediction, merged["Actual"])
        merged["Comments"] = "Missing_Unimputed"
        merged.loc[has_actual, "Comments"] = "Original"
        merged.loc[has_prediction, "Comments"] = "Imputed"

        out = pd.DataFrame(
            {
                "DateTime": merged["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S%z"),
                "Missingness": int(round(float(missingness) * 100)),
                "Missingness_Regime": regime,
                "Model": model,
                "StudySite": site,
                "Actual": merged["Actual"],
                "Imputed": merged["Imputed"],
                "Comments": merged["Comments"],
            }
        )

        temp_path = out_path.with_suffix(f".{os.getpid()}.tmp")
        out.to_csv(temp_path, index=False)
        temp_path.replace(out_path)
        written += 1

    return written


def rebuild_path(pred_path: Path, output_dir: Path, skip_existing: bool) -> int:
    return rebuild_file(pred_path, _SITE_FRAMES, output_dir, skip_existing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2] / "Outputs" / "Imputation_Result"
    parser.add_argument("--results-dir", type=Path, default=default_root)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    results_dir = args.results_dir
    inputs_dir = results_dir / "Inputs_PerSite"
    predictions_root = results_dir / "Regional_Pooled_Imputation"
    output_dir = results_dir / "Imputed_Results"
    output_dir.mkdir(parents=True, exist_ok=True)

    global _SITE_FRAMES
    _SITE_FRAMES = read_site_inputs(inputs_dir)
    total = 0
    pred_paths = sorted(predictions_root.rglob("masked_predictions_by_site.csv"))
    workers = max(1, args.workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_paths = {
            executor.submit(
                rebuild_path,
                pred_path,
                output_dir,
                args.skip_existing,
            ): pred_path
            for pred_path in pred_paths
        }
        for i, future in enumerate(as_completed(future_paths), start=1):
            pred_path = future_paths[future]
            total += future.result()
            print(f"[{i}/{len(pred_paths)}] processed {pred_path}", flush=True)

    print(f"Wrote {total} continuous imputed result files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
