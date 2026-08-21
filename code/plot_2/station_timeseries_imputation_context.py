#!/usr/bin/env python3
"""Station-level 6-month time-series plot with imputation context.

The plot overlays:
  - continuous actual observations for one station
  - selected model imputed values at artificially masked timestamps
  - BaseLine imputed values at the same timestamps
  - regional average excluding the station
  - nearest station in the same region, when available
"""

import argparse
from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Outputs" / "Imputation_Result"
API_INPUT_DIR = PROJECT_ROOT / "API_Input" / "Inputs"
STATION_INFO_FILE = PROJECT_ROOT.parent / "AQ_DATA" / "AquisNET_Data" / "Air Quality API Excel Power Query.xlsx"
OUTPUT_DIR = RESULTS_DIR / "plots_by_type" / "time_series_station"


def region_token(region: str) -> str:
    return re.sub(r"_+", "_", str(region).strip().replace("-", "_").replace(" ", "_"))


def site_token(site: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(site).strip().upper()).strip("_")


def target_folder(target: str) -> str:
    target = str(target).strip()
    return "PM25" if target.upper() in {"PM2.5", "PM25", "PM2_5"} else target.replace(".", "_")


def target_column_prefix(target: str) -> str:
    target = str(target).strip()
    return "PM2.5" if target.upper() in {"PM25", "PM2_5"} else target


def find_wide_input(region: str) -> Path:
    token = region_token(region)
    candidates = [
        API_INPUT_DIR / f"Allobs_processed_DPE_station_api_{token}_ALL.csv",
        API_INPUT_DIR / f"Allobs_processed_DPE_station_api_{region}_ALL.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(API_INPUT_DIR.glob(f"*{token}*ALL.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No wide input CSV found for region={region!r} under {API_INPUT_DIR}")


def read_wide_region(path: Path, target: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    prefix = target_column_prefix(target)
    header = pd.read_csv(path, nrows=0)
    date_col = "datetime" if "datetime" in header.columns else "DateTime"
    target_cols = [c for c in header.columns if c == prefix or c.startswith(f"{prefix}_")]
    usecols = [date_col] + target_cols
    if not target_cols:
        raise ValueError(f"No {prefix} station columns found in {path}")
    frame = pd.read_csv(path, usecols=usecols)
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    frame = frame.dropna(subset=[date_col]).rename(columns={date_col: "DateTime"})
    frame = frame[(frame["DateTime"] >= start) & (frame["DateTime"] <= end)].copy()
    frame = frame.sort_values("DateTime")
    return frame


def station_column(frame: pd.DataFrame, target: str, site: str) -> str:
    prefix = target_column_prefix(target)
    exact = f"{prefix}_{site_token(site)}"
    by_canon = {
        site_token(c.replace(f"{prefix}_", "")): c
        for c in frame.columns
        if c.startswith(f"{prefix}_")
    }
    if site_token(site) in by_canon:
        return by_canon[site_token(site)]
    if exact in frame.columns:
        return exact
    raise ValueError(f"No column for target={target!r}, site={site!r}; expected something like {exact}")


def load_masked_predictions(
    model: str,
    region: str,
    target: str,
    site: str,
    regime: str,
    missingness: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    path = (
        RESULTS_DIR
        / "Regional_Pooled_Imputation"
        / model
        / region_token(region)
        / target_folder(target)
        / "masked_predictions_by_site.csv"
    )
    if not path.exists():
        return pd.DataFrame(columns=["DateTime", "Observed", "Imputed"])

    chunks = []
    usecols = ["DateTime", "Site", "Regime", "Missingness_Level", "Observed", "Imputed", "Was_Artificially_Masked"]
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=250000):
        chunk = chunk[
            (chunk["Site"].astype(str).str.upper() == str(site).upper())
            & (chunk["Regime"].astype(str) == str(regime))
            & np.isclose(pd.to_numeric(chunk["Missingness_Level"], errors="coerce"), float(missingness))
        ].copy()
        if chunk.empty:
            continue
        if "Was_Artificially_Masked" in chunk.columns:
            chunk = chunk[chunk["Was_Artificially_Masked"].astype(bool)]
        chunk["DateTime"] = pd.to_datetime(chunk["DateTime"], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["DateTime"])
        chunk = chunk[(chunk["DateTime"] >= start) & (chunk["DateTime"] <= end)]
        if not chunk.empty:
            chunks.append(chunk[["DateTime", "Observed", "Imputed"]])
    if not chunks:
        return pd.DataFrame(columns=["DateTime", "Observed", "Imputed"])
    out = pd.concat(chunks, ignore_index=True).sort_values("DateTime")
    out["Observed"] = pd.to_numeric(out["Observed"], errors="coerce")
    out["Imputed"] = pd.to_numeric(out["Imputed"], errors="coerce")
    return out


def load_station_metadata() -> pd.DataFrame:
    if not STATION_INFO_FILE.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_excel(STATION_INFO_FILE, sheet_name="SiteDetails")
    except Exception:
        return pd.DataFrame()
    rename = {
        "Column1.SiteName": "Site",
        "Column1.Region": "Region",
        "Column1.Longitude": "Longitude",
        "Column1.Latitude": "Latitude",
    }
    frame = frame.rename(columns=rename)
    needed = ["Site", "Region", "Longitude", "Latitude"]
    if not set(needed).issubset(frame.columns):
        return pd.DataFrame()
    frame = frame[needed].copy()
    frame["SiteKey"] = frame["Site"].map(site_token)
    frame["RegionKey"] = frame["Region"].map(region_token)
    return frame


def nearest_station(region: str, site: str, available_sites: list) -> str:
    available = [s for s in available_sites if site_token(s) != site_token(site)]
    if not available:
        return ""
    meta = load_station_metadata()
    if meta.empty:
        return available[0]
    region_meta = meta[meta["RegionKey"] == region_token(region)].copy()
    target = region_meta[region_meta["SiteKey"] == site_token(site)]
    if target.empty:
        return available[0]
    target_lat = float(target.iloc[0]["Latitude"])
    target_lon = float(target.iloc[0]["Longitude"])
    region_meta = region_meta[region_meta["SiteKey"].isin([site_token(s) for s in available])]
    if region_meta.empty:
        return available[0]
    lat = np.radians(region_meta["Latitude"].astype(float).to_numpy())
    lon = np.radians(region_meta["Longitude"].astype(float).to_numpy())
    tlat = np.radians(target_lat)
    tlon = np.radians(target_lon)
    dlat = lat - tlat
    dlon = lon - tlon
    a = np.sin(dlat / 2.0) ** 2 + np.cos(tlat) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    region_meta["DistanceKm"] = 6371.0 * 2.0 * np.arcsin(np.sqrt(a))
    nearest_key = region_meta.sort_values("DistanceKm").iloc[0]["SiteKey"]
    for candidate in available:
        if site_token(candidate) == nearest_key:
            return candidate
    return available[0]


def continuous_hourly(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame
    index = pd.date_range(start=start, end=end, freq="H")
    out = frame.set_index("DateTime").sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.reindex(index)
    out.index.name = "DateTime"
    return out.reset_index()


def discover_regions(model: str, target: str) -> list:
    root = RESULTS_DIR / "Regional_Pooled_Imputation" / model
    return sorted(
        path.parent.parent.name
        for path in root.glob(f"*/{target_folder(target)}/masked_predictions_by_site.csv")
    )


def available_sites_for_selection(model: str, region: str, target: str, regime: str, missingness: float) -> list:
    path = (
        RESULTS_DIR
        / "Regional_Pooled_Imputation"
        / model
        / region_token(region)
        / target_folder(target)
        / "masked_predictions_by_site.csv"
    )
    if not path.exists():
        return []
    counts = {}
    usecols = ["Site", "Regime", "Missingness_Level", "Was_Artificially_Masked"]
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=250000):
        chunk = chunk[
            (chunk["Regime"].astype(str) == str(regime))
            & np.isclose(pd.to_numeric(chunk["Missingness_Level"], errors="coerce"), float(missingness))
        ].copy()
        if chunk.empty:
            continue
        if "Was_Artificially_Masked" in chunk.columns:
            chunk = chunk[chunk["Was_Artificially_Masked"].astype(bool)]
        for site_name, n_rows in chunk["Site"].value_counts().items():
            counts[str(site_name)] = counts.get(str(site_name), 0) + int(n_rows)
    return [site for site, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def choose_site(model: str, region: str, target: str, requested_site: str, regime: str, missingness: float) -> str:
    sites = available_sites_for_selection(model, region, target, regime, missingness)
    if not sites:
        return requested_site
    requested_key = site_token(requested_site)
    for site_name in sites:
        if site_token(site_name) == requested_key:
            return site_name
    return sites[0]


def generate_plot(args, region: str, site: str, start: pd.Timestamp, end: pd.Timestamp, start_label: str) -> tuple:
    wide_path = find_wide_input(region)
    wide = read_wide_region(wide_path, args.target, start, end)
    actual_col = station_column(wide, args.target, site)
    prefix = target_column_prefix(args.target)
    station_cols = [c for c in wide.columns if c.startswith(f"{prefix}_")]
    other_cols = [c for c in station_cols if c != actual_col]
    other_sites = [c.replace(f"{prefix}_", "") for c in other_cols]
    near_site = nearest_station(region, site, other_sites)
    near_col = station_column(wide, args.target, near_site) if near_site else None

    plot_data = wide[["DateTime", actual_col] + other_cols].copy()
    plot_data["Actual"] = pd.to_numeric(plot_data[actual_col], errors="coerce")
    plot_data["Regional_Average_Excl_Station"] = plot_data[other_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if other_cols else np.nan
    plot_data["Nearest_Station"] = pd.to_numeric(plot_data[near_col], errors="coerce") if near_col else np.nan
    plot_data = continuous_hourly(plot_data[["DateTime", "Actual", "Regional_Average_Excl_Station", "Nearest_Station"]], start, end)

    model_pred = load_masked_predictions(
        args.model, region, args.target, site, args.regime, args.missingness, start, end
    )
    baseline_pred = load_masked_predictions(
        "BaseLine", region, args.target, site, args.regime, args.missingness, start, end
    )

    fig, ax = plt.subplots(figsize=(15, 6.8))
    if args.masked_only_markers:
        if not model_pred.empty:
            ax.scatter(
                model_pred["DateTime"],
                model_pred["Observed"],
                color="black",
                s=22,
                alpha=0.82,
                label="Actual at masked timestamps",
                zorder=4,
            )
            ax.scatter(
                model_pred["DateTime"],
                model_pred["Imputed"],
                facecolors="none",
                edgecolors="#e45756",
                s=42,
                linewidths=1.15,
                alpha=0.95,
                label=f"{args.model} imputed",
                zorder=5,
            )
    else:
        ax.plot(plot_data["DateTime"], plot_data["Actual"], color="black", linewidth=1.25, label="Actual")
    if args.aquistil_only_lines and not args.masked_only_markers:
        if not model_pred.empty:
            ax.plot(
                model_pred["DateTime"],
                model_pred["Imputed"],
                color="#e45756",
                linewidth=1.15,
                marker="o",
                markersize=2.8,
                alpha=0.90,
                label=f"{args.model} imputed",
                zorder=5,
            )
    elif (not args.masked_only_markers) and other_cols:
        ax.plot(
            plot_data["DateTime"],
            plot_data["Regional_Average_Excl_Station"],
            color="#4c78a8",
            linewidth=1.0,
            alpha=0.70,
            label="Regional average excluding station",
        )
    if (not args.aquistil_only_lines) and (not args.masked_only_markers) and near_col:
        ax.plot(
            plot_data["DateTime"],
            plot_data["Nearest_Station"],
            color="#72b7b2",
            linewidth=0.95,
            alpha=0.70,
            label=f"Nearest station: {near_site}",
        )
    if (not args.aquistil_only_lines) and (not args.masked_only_markers) and not baseline_pred.empty:
        ax.scatter(
            baseline_pred["DateTime"],
            baseline_pred["Imputed"],
            color="#f58518",
            s=14,
            alpha=0.78,
            label="BaseLine imputed at masked timestamps",
            zorder=4,
        )
    if (not args.aquistil_only_lines) and (not args.masked_only_markers) and not model_pred.empty:
        ax.scatter(
            model_pred["DateTime"],
            model_pred["Imputed"],
            color="#e45756",
            s=18,
            alpha=0.88,
            label=f"{args.model} imputed at masked timestamps",
            zorder=5,
        )
        ax.scatter(
            model_pred["DateTime"],
            model_pred["Observed"],
            facecolors="none",
            edgecolors="#7f3c8d",
            s=24,
            linewidths=0.8,
            alpha=0.75,
            label="Held-out actual at masked timestamps",
            zorder=6,
        )

    ax.set_title(
        f"{site} {args.target} | {region} | {args.model}, {args.regime}, "
        f"{args.missingness:g} missingness | {start.date()} to {end.date()}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(args.target)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    miss_label = f"{int(args.missingness * 100)}pct" if args.missingness <= 1 else f"{int(args.missingness)}pct"
    out_name = (
        f"timeseries_{target_folder(args.target)}_{region_token(region)}_{site_token(site)}_"
        f"{args.model}_{args.regime}_{miss_label}_{start_label}_{args.months}mo.png"
    )
    if args.aquistil_only_lines:
        out_name = out_name.replace(".png", "_actual_imputed_lines.png")
    if args.masked_only_markers:
        out_name = out_name.replace(".png", "_masked_actual_imputed_markers.png")
    out_path = args.output_dir / out_name
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sidecar = out_path.with_suffix(".csv")
    merged = plot_data.copy()
    if not model_pred.empty:
        merged = merged.merge(
            model_pred.rename(columns={"Imputed": f"{args.model}_Imputed", "Observed": "Heldout_Actual"})[
                ["DateTime", "Heldout_Actual", f"{args.model}_Imputed"]
            ],
            on="DateTime",
            how="left",
        )
    if not baseline_pred.empty:
        merged = merged.merge(
            baseline_pred.rename(columns={"Imputed": "BaseLine_Imputed"})[["DateTime", "BaseLine_Imputed"]],
            on="DateTime",
            how="left",
        )
    merged.to_csv(sidecar, index=False)

    print(out_path)
    print(sidecar)
    print(f"wide_input={wide_path}")
    print(f"region={region}")
    print(f"site={site}")
    print(f"masked_points_model={len(model_pred)}")
    print(f"masked_points_baseline={len(baseline_pred)}")
    print(f"nearest_station={near_site or 'none'}")
    return out_path, sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="Lower Hunter")
    parser.add_argument("--site", default="NEWCASTLE")
    parser.add_argument("--target", default="PM2.5")
    parser.add_argument("--model", default="AQUISTIL")
    parser.add_argument("--regime", default="random")
    parser.add_argument("--missingness", type=float, default=0.20)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--all-regions", action="store_true", help="Generate one station plot for every available model/target region.")
    parser.add_argument("--aquistil-only-lines", action="store_true", help="Plot only Actual and selected-model imputed values as lines.")
    parser.add_argument("--masked-only-markers", action="store_true", help="Plot only masked timestamps: actual values and hollow edge-only selected-model imputed markers.")
    parser.add_argument("--seed", type=int, default=42, help="Reserved for labels; prediction file is already seed-filtered by run.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    start_label = pd.Timestamp(args.start).strftime("%Y%m%d")
    start = pd.Timestamp(args.start)
    if start.tzinfo is None:
        start = start.tz_localize("Australia/Sydney")
    start = start.tz_convert("UTC")
    end = start + pd.DateOffset(months=args.months) - pd.Timedelta(hours=1)

    if args.all_regions:
        outputs = []
        for region in discover_regions(args.model, args.target):
            site = choose_site(args.model, region, args.target, args.site, args.regime, args.missingness)
            try:
                outputs.append(generate_plot(args, region, site, start, end, start_label))
            except Exception as exc:
                print(f"Skipping region={region}: {exc}")
        print(f"Generated {len(outputs)} region plot(s)")
        return 0

    generate_plot(args, args.region, args.site, start, end, start_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
