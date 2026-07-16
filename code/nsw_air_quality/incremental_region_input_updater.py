"""Utilities to keep the already-downloaded *wide* regional API-input CSVs up to date.

The nowcasting inputs live under (example):
  AI_Nowcasting/cnn_lstm_forecast/API_Input/Inputs/

Files are wide-format with columns like:
  datetime, CO_ALEXANDRIA, CO_COOK AND PHILLIP, ...

This module appends the missing tail ("download remaining data up to now")
by querying the NSW Air Quality API via `NSWAirQualityAPIClient`.
"""

import glob
import logging
import os
import re
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .api_client import NSWAirQualityAPIClient


AEST_TZ = "Australia/Brisbane"  # match existing wide CSV offsets (+10:00, no DST)


def _canon(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(s).upper())


def _fmt_ts(ts) -> str:
    if ts is None or pd.isna(ts):
        return "NA"
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return str(ts)


def _as_aest_datetime(values) -> pd.Series:
    """Parse timestamps and return one consistently tz-aware AEST series."""
    try:
        parsed = pd.to_datetime(values, errors="coerce")
    except ValueError:
        # Concatenating the CSV strings with newly downloaded tz-aware
        # Timestamps creates a mixed representation; UTC parsing normalizes it.
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(AEST_TZ, nonexistent="shift_forward", ambiguous="NaT")
    return parsed.dt.tz_convert(AEST_TZ)


def _atomic_write_csv(frame: pd.DataFrame, csv_path: str) -> None:
    """Replace a CSV only after the complete new file has been written."""
    output_dir = os.path.dirname(os.path.abspath(csv_path))
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(csv_path)}.", suffix=".tmp", dir=output_dir
    )
    os.close(fd)
    try:
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, csv_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _split_var_station(col: str) -> Optional[Tuple[str, str]]:
    if not isinstance(col, str) or "_" not in col:
        return None
    var, station = col.split("_", 1)
    var = var.strip()
    station = station.strip()
    if not var or not station:
        return None
    return var, station


def _var_alias_to_api_code(var: str) -> str:
    v = str(var).strip()
    up = v.upper()
    if up in {"O3", "OZONE"}:
        return "OZONE"
    if up in {"PM25", "PM2_5", "PM_25", "PM2.5"}:
        return "PM2.5"
    if up in {"PM10"}:
        return "PM10"
    return v


def infer_wide_schema(columns: Sequence[str]) -> Dict[str, object]:
    datetime_col = "datetime" if "datetime" in columns else "DateTime" if "DateTime" in columns else None
    if not datetime_col:
        raise ValueError("Wide input CSV is missing a datetime column (expected 'datetime' or 'DateTime')")

    stations: List[str] = []
    variables: List[str] = []
    canon_to_station: Dict[str, str] = {}

    for col in columns:
        if col == datetime_col:
            continue
        parsed = _split_var_station(col)
        if not parsed:
            continue
        var, station = parsed
        if station not in stations:
            stations.append(station)
        if var not in variables:
            variables.append(var)
        canon_to_station.setdefault(_canon(station), station)

    return {
        "datetime_col": datetime_col,
        "stations": tuple(stations),
        "variables": tuple(variables),
        "station_canon_to_name": canon_to_station,
    }


def list_wide_region_files(input_dir: str) -> List[str]:
    pattern = os.path.join(input_dir, "Allobs_processed_DPE_station_api_*_ALL.csv")
    return sorted(glob.glob(pattern))


def validate_existing_wide_region_files(
    input_dir: str,
    only_region_tokens: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Validate local fallback inputs and return their coverage information."""
    files = list_wide_region_files(input_dir)
    if only_region_tokens:
        norm_tokens = {_canon(t) for t in only_region_tokens}
        files = [
            fp for fp in files
            if _canon(
                re.match(
                    r"Allobs_processed_DPE_station_api_(.+?)_ALL\.csv$",
                    os.path.basename(fp),
                ).group(1)
            ) in norm_tokens
        ]
    if not files:
        raise RuntimeError("No existing regional wide-input CSVs found under %s" % input_dir)

    coverage: Dict[str, Tuple[pd.Timestamp, pd.Timestamp, int]] = {}
    for fp in files:
        frame = pd.read_csv(fp)
        schema = infer_wide_schema(frame.columns)
        if frame.empty:
            raise ValueError("Existing wide-input CSV is empty: %s" % fp)
        timestamps = _as_aest_datetime(frame[schema["datetime_col"]]).dropna()
        if timestamps.empty:
            raise ValueError("Existing wide-input CSV has no valid timestamps: %s" % fp)
        coverage[fp] = (timestamps.min(), timestamps.max(), len(frame))
    return coverage


def _build_site_maps() -> Tuple[Dict[str, int], Dict[int, str]]:
    aqms = NSWAirQualityAPIClient()
    sites = pd.json_normalize(aqms.get_site_details().json())
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    for _, row in sites.iterrows():
        name = str(row.get("SiteName", "")).strip()
        sid = row.get("Site_Id")
        if not name or pd.isna(sid):
            continue
        try:
            sid_int = int(sid)
        except Exception:
            continue
        name_to_id[_canon(name)] = sid_int
        id_to_name[sid_int] = name
    return name_to_id, id_to_name


def _available_parameter_codes() -> set:
    aqms = NSWAirQualityAPIClient()
    params = pd.json_normalize(aqms.get_parameters_details().json())
    codes = set(str(x).strip() for x in params.get("ParameterCode", []).tolist())
    return codes


def _api_datetime_from_date_hour(date_series: pd.Series, hour_series: pd.Series) -> pd.Series:
    # API uses Date='YYYY-MM-DD' and Hour=1..24 (HourDescription says "12 am - 1 am" for Hour=1)
    date = pd.to_datetime(date_series, errors="coerce")
    hour = pd.to_numeric(hour_series, errors="coerce").astype("Int64")
    dt = date + pd.to_timedelta((hour - 1).astype("float"), unit="h")
    # Align to AEST (+10) like the existing files.
    try:
        dt = dt.dt.tz_localize(AEST_TZ, nonexistent="shift_forward", ambiguous="NaT")
    except Exception:
        # if already tz-aware, keep as-is
        pass
    return dt


def _obs_to_wide_append(
    obs_df: pd.DataFrame,
    schema: Dict[str, object],
    station_id_to_name: Dict[int, str],
) -> pd.DataFrame:
    if obs_df.empty:
        return pd.DataFrame(columns=[schema["datetime_col"]])

    if "Parameter.ParameterCode" not in obs_df.columns:
        return pd.DataFrame(columns=[schema["datetime_col"]])

    obs_df = obs_df.copy()
    obs_df["_dt"] = _api_datetime_from_date_hour(obs_df.get("Date"), obs_df.get("Hour"))
    obs_df["_var"] = obs_df["Parameter.ParameterCode"].astype(str)
    obs_df["_sid"] = pd.to_numeric(obs_df.get("Site_Id"), errors="coerce").astype("Int64")
    obs_df["_station_api"] = obs_df["_sid"].map(lambda x: station_id_to_name.get(int(x)) if pd.notna(x) else None)
    canon_map = schema.get("station_canon_to_name", {})
    obs_df["_station"] = obs_df["_station_api"].map(
        lambda x: canon_map.get(_canon(x), x) if isinstance(x, str) else x
    )

    obs_df = obs_df.dropna(subset=["_dt", "_var", "_station"], how="any")
    if obs_df.empty:
        return pd.DataFrame(columns=[schema["datetime_col"]])

    obs_df["_col"] = obs_df.apply(lambda r: f"{r['_var']}_{r['_station']}", axis=1)
    wide = (
        obs_df.pivot_table(index="_dt", columns="_col", values="Value", aggfunc="mean")
        .sort_index()
    )

    wide = wide.reset_index().rename(columns={"_dt": schema["datetime_col"]})
    return wide


def update_nsw_region_input_file(csv_path: str, now: Optional[pd.Timestamp] = None, site_batch_size: int = 4) -> bool:
    """Append missing tail observations to a single wide input CSV.

    Returns True if the file was updated (new rows were appended/merged), else False.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    logging.info("Updating wide input: %s", csv_path)

    # Load existing
    df_existing = pd.read_csv(csv_path, low_memory=False)
    schema = infer_wide_schema(df_existing.columns)
    dtcol = schema["datetime_col"]
    dt_existing = _as_aest_datetime(df_existing[dtcol])
    if dt_existing.isna().all():
        raise ValueError(f"Could not parse datetime column '{dtcol}' in {csv_path}")

    min_dt = dt_existing.min()
    max_dt = dt_existing.max()
    if now is None:
        now = pd.Timestamp.now(tz=AEST_TZ)

    if pd.isna(max_dt):
        raise ValueError(f"No max datetime found in {csv_path}")

    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize(AEST_TZ)
    else:
        now = now.tz_convert(AEST_TZ)

    # Only closed hourly averages are safe to persist. For example, at 14:37
    # the latest complete observation is the hour labelled 13:00.
    target_hour = now.floor("h") - pd.Timedelta(hours=1)
    tail_start = max_dt + pd.Timedelta(hours=1)
    if max_dt >= target_hour:
        logging.info(
            "Wide input already up to date: %s | coverage=%s to %s | now=%s",
            csv_path,
            _fmt_ts(min_dt),
            _fmt_ts(max_dt),
            _fmt_ts(target_hour),
        )
        return False

    # Build request inputs
    station_name_to_id, station_id_to_name = _build_site_maps()
    available_param_codes = _available_parameter_codes()

    # Variables to download from API (skip derived columns such as NOX)
    file_vars = [_var_alias_to_api_code(v) for v in schema["variables"]]
    api_vars = sorted({v for v in file_vars if v in available_param_codes})

    if not api_vars:
        logging.warning("No API parameter codes found for %s; skipping update", csv_path)
        return False

    # Resolve station IDs
    station_ids: List[int] = []
    missing_station_names: List[str] = []
    for station in schema["stations"]:
        sid = station_name_to_id.get(_canon(station))
        if sid is None:
            missing_station_names.append(station)
            continue
        station_ids.append(int(sid))
    station_ids = sorted(set(station_ids))
    if missing_station_names:
        logging.warning("Stations missing from API site list (skipped): %s", ", ".join(sorted(set(missing_station_names))))

    if not station_ids:
        logging.warning("No valid station IDs resolved for %s; skipping update", csv_path)
        return False

    # Determine date window (API only takes dates).
    # Pull in small windows (day-by-day) to avoid very large responses.
    start_day = tail_start.floor("D")
    end_day = target_hour.floor("D")
    if start_day > end_day:
        logging.info(
            "Wide input already covers requested period: %s | coverage=%s to %s | requested_start=%s | requested_end=%s",
            csv_path,
            _fmt_ts(min_dt),
            _fmt_ts(max_dt),
            _fmt_ts(start_day),
            _fmt_ts(end_day),
        )
        return False

    aqms = NSWAirQualityAPIClient()
    base_req = aqms.ObsRequest_init()
    base_req["Categories"] = ["Averages"]
    base_req["SubCategories"] = ["Hourly"]
    base_req["Frequency"] = ["Hourly average"]
    base_req["Parameters"] = api_vars
    # We will set StartDate/EndDate per day chunk below.

    # Batch stations to keep responses manageable
    batches = [station_ids[i : i + site_batch_size] for i in range(0, len(station_ids), site_batch_size)]
    all_batches: List[pd.DataFrame] = []

    # Day-chunked downloads to keep payload sizes manageable
    day = start_day
    while day <= end_day:
        day_start = day.date().isoformat()
        day_end = (day + pd.Timedelta(days=1)).date().isoformat()
        for bi, batch in enumerate(batches, 1):
            req = dict(base_req)
            req["Sites"] = list(batch)
            req["StartDate"] = day_start
            req["EndDate"] = day_end
            resp = aqms.get_historical_obs(req)
            if resp.status_code != 200:
                # Do not advance a regional file after a partial download. If
                # one batch fails, the next pipeline run must retry the same
                # tail for every station.
                raise RuntimeError(
                    "API batch %d/%d failed for %s..%s (status=%s)"
                    % (bi, len(batches), day_start, day_end, resp.status_code)
                )
            try:
                js = resp.json()
            except Exception as exc:
                raise RuntimeError(
                    "API batch %d/%d returned invalid JSON for %s..%s (status=%s)"
                    % (bi, len(batches), day_start, day_end, resp.status_code)
                ) from exc
            if not js:
                continue
            all_batches.append(pd.json_normalize(js))
        day = day + pd.Timedelta(days=1)

    if not all_batches:
        logging.info(
            "No new API observations returned for %s | requested tail=%s to %s | existing coverage=%s to %s",
            csv_path,
            _fmt_ts(tail_start),
            _fmt_ts(target_hour),
            _fmt_ts(min_dt),
            _fmt_ts(max_dt),
        )
        return False

    obs_all = pd.concat(all_batches, ignore_index=True)
    wide_new = _obs_to_wide_append(obs_all, schema, station_id_to_name)

    if wide_new.empty or dtcol not in wide_new.columns:
        logging.info(
            "No usable new wide rows produced for %s | requested tail=%s to %s | existing coverage=%s to %s",
            csv_path,
            _fmt_ts(tail_start),
            _fmt_ts(target_hour),
            _fmt_ts(min_dt),
            _fmt_ts(max_dt),
        )
        return False

    # The API accepts dates rather than datetimes, so a request can return a
    # whole boundary day. Retain only the genuinely missing closed-hour tail.
    new_times = _as_aest_datetime(wide_new[dtcol])
    keep = new_times.between(tail_start, target_hour, inclusive="both")
    wide_new = wide_new.loc[keep].copy()
    wide_new[dtcol] = new_times.loc[keep]
    if wide_new.empty:
        logging.info(
            "API returned no observations in missing tail for %s | requested=%s to %s",
            csv_path,
            _fmt_ts(tail_start),
            _fmt_ts(target_hour),
        )
        return False

    # Ensure all original columns exist in the new frame
    # (missing station-variable combos will be NaN)
    for c in df_existing.columns:
        if c not in wide_new.columns:
            wide_new[c] = pd.NA
    wide_new = wide_new[df_existing.columns]

    # Derived NOX columns (API does not provide NOX): compute when possible
    try:
        no_cols = [c for c in df_existing.columns if isinstance(c, str) and c.startswith("NO_")]
        no2_cols = [c for c in df_existing.columns if isinstance(c, str) and c.startswith("NO2_")]
        nox_cols = [c for c in df_existing.columns if isinstance(c, str) and c.startswith("NOX_")]
        if nox_cols and (no_cols or no2_cols):
            # compute per station token
            stations = schema["stations"]
            for st in stations:
                c_no = f"NO_{st}"
                c_no2 = f"NO2_{st}"
                c_nox = f"NOX_{st}"
                if c_nox in df_existing.columns and c_no in wide_new.columns and c_no2 in wide_new.columns:
                    wide_new[c_nox] = pd.to_numeric(wide_new[c_no], errors="coerce") + pd.to_numeric(wide_new[c_no2], errors="coerce")
    except Exception:
        logging.debug("Failed to compute derived NOX columns", exc_info=True)

    # New data is tail-only, so existing timestamps and values are never
    # overwritten by a partial API response.
    merged = pd.concat([df_existing, wide_new], ignore_index=True)
    dtcol = schema["datetime_col"]
    merged[dtcol] = _as_aest_datetime(merged[dtcol])
    merged = merged.dropna(subset=[dtcol])
    merged = merged.drop_duplicates(subset=[dtcol], keep="first")
    merged = merged.sort_values(dtcol).reset_index(drop=True)

    new_dt = _as_aest_datetime(wide_new[dtcol]).dropna()
    merged_dt = _as_aest_datetime(merged[dtcol]).dropna()

    # Write back
    _atomic_write_csv(merged, csv_path)
    logging.info(
        "API download confirmed for %s | requested tail=%s to %s | downloaded coverage=%s to %s | file coverage now=%s to %s | rows=%d",
        csv_path,
        _fmt_ts(tail_start),
        _fmt_ts(target_hour),
        _fmt_ts(new_dt.min() if not new_dt.empty else None),
        _fmt_ts(new_dt.max() if not new_dt.empty else None),
        _fmt_ts(merged_dt.min() if not merged_dt.empty else None),
        _fmt_ts(merged_dt.max() if not merged_dt.empty else None),
        len(merged),
    )
    return True


def update_nsw_region_input_files(
    input_dir: str,
    only_region_tokens: Optional[Sequence[str]] = None,
    site_batch_size: int = 4,
    raise_on_error: bool = False,
) -> Dict[str, bool]:
    """Update all wide region files in a directory.

    Returns a dict: {csv_path: updated_bool}.
    """
    files = list_wide_region_files(input_dir)
    if only_region_tokens:
        norm_tokens = {_canon(t) for t in only_region_tokens}
        filtered = []
        for fp in files:
            token = os.path.basename(fp)
            # token between api_ and _ALL
            m = re.match(r"Allobs_processed_DPE_station_api_(.+?)_ALL\.csv$", token)
            region_token = m.group(1) if m else token
            if _canon(region_token) in norm_tokens:
                filtered.append(fp)
        files = filtered

    results: Dict[str, bool] = {}
    for fp in files:
        try:
            results[fp] = bool(update_nsw_region_input_file(fp, site_batch_size=site_batch_size))
        except Exception as e:
            logging.warning("Failed to update %s: %s", fp, e)
            results[fp] = False
            if raise_on_error:
                raise
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update wide nowcasting input CSVs by downloading the missing tail from the NSW AQ API")
    parser.add_argument("--input-dir", default=None, help="Directory containing Allobs_processed_DPE_station_api_*_ALL.csv")
    parser.add_argument("--regions", nargs="+", default=None, help="Optional region tokens/names to update (e.g., 'Sydney North-west')")
    parser.add_argument("--site-batch-size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    root = args.input_dir or os.environ.get("WIDE_INPUT_DIR") or ""
    if not root:
        raise SystemExit("Provide --input-dir or set WIDE_INPUT_DIR")

    res = update_nsw_region_input_files(root, only_region_tokens=args.regions, site_batch_size=args.site_batch_size)
    changed = [k for k, v in res.items() if v]
    logging.info("Updated %d/%d files", len(changed), len(res))
