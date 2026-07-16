"""
regional_pm25_inventory.py

PM2.5 availability inventory by region and site.

What this script does
---------------------
1. For each region in config_spatial.TARGET_REGIONS (or all regions in REGION_TO_SITES):
   - checks each site
   - finds whether that site has a PM2.5 column
   - lists all available variables for that site
   - creates a region-level variable -> sites summary

2. Outputs:
   - TXT report
   - CSV summary

Important:
----------
- This script does NOT drop any columns.
- This script does NOT run imputation / feature selection.
- This script is only for inventory/reporting.

Author: Masrur + assistant
Date: 2026-03-24
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import pandas as pd

import config_spatial as config


# =============================================================================
# USER SETTINGS
# =============================================================================

TARGET_COLUMN = "PM2.5"

OUTPUT_ROOT = Path("regional_feature_selection_results")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Optional site aliases if config site names do not match filenames exactly.
# Edit these if needed.
SITE_ALIASES: Dict[str, str] = {
    # "VINEYARD": "VINEYARDS",
    # "MACARTHUR": "CAMDEN",
}

# Sites to skip entirely, if you want
EXCLUDED_SITES = set()


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_ROOT / "regional_pm25_inventory.log", mode="w", encoding="utf-8"),
    ],
)


# =============================================================================
# HELPERS
# =============================================================================


def normalize_token(s: str) -> str:
    """Normalize strings for loose matching."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def find_site_file(site_name: str) -> Optional[str]:
    """
    Find the CSV file for a given site in config.INPUT_DIRECTORY.

    Matching strategy:
    1) first underscore-separated token
    2) any token in filename
    3) normalized site name anywhere in normalized filename
    """
    input_dir = config.INPUT_DIRECTORY

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"INPUT_DIRECTORY does not exist: {input_dir}")

    target_norm = re.sub(r"[^a-z0-9]", "", site_name.lower())

    # 1) Prefer filenames whose first underscore-separated stem matches the site
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".csv"):
            continue
        stem = filename.split("_")[0]
        stem_norm = re.sub(r"[^a-z0-9]", "", stem.lower())
        if stem_norm == target_norm:
            return os.path.join(input_dir, filename)

    # 2) Try any token in filename
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".csv"):
            continue
        name_no_ext = os.path.splitext(filename)[0]
        parts = re.split(r"[_\-\s]+", name_no_ext)
        for part in parts:
            part_norm = re.sub(r"[^a-z0-9]", "", part.lower())
            if part_norm == target_norm:
                return os.path.join(input_dir, filename)

    # 3) Fallback: site name appears anywhere in normalized filename
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".csv"):
            continue
        filename_norm = re.sub(r"[^a-z0-9]", "", filename.lower())
        if target_norm in filename_norm:
            return os.path.join(input_dir, filename)

    return None


def find_site_specific_column(df_cols: List[str], base_name: str, site_name: str) -> Optional[str]:
    """
    Find a column like:
      PM2.5
      PM2.5_SITE
      PM25_SITE
      PM2_5_SITE
    using flexible normalized matching.
    """
    base_norm = normalize_token(base_name)
    site_norm = normalize_token(site_name)

    # 1) exact raw match
    for c in df_cols:
        if c == base_name:
            return c

    # 2) exact normalized match
    for c in df_cols:
        if normalize_token(c) == base_norm:
            return c

    # 3) base + site suffix match
    for c in df_cols:
        c_norm = normalize_token(c)
        if c_norm.startswith(base_norm) and c_norm.endswith(site_norm):
            return c

    # 4) starts with base only
    start_matches = [c for c in df_cols if normalize_token(c).startswith(base_norm)]
    if len(start_matches) == 1:
        return start_matches[0]

    # 5) contains base anywhere
    any_matches = [c for c in df_cols if base_norm in normalize_token(c)]
    if len(any_matches) == 1:
        return any_matches[0]

    return None


def get_available_variables(cols: List[str]) -> List[str]:
    """
    Keep all raw variables except obvious time/index helper columns.
    No dropping based on NaN.
    """
    excluded = {
        "datetime", "date", "time", "year", "month", "day", "hour", "minute",
        "unnamed: 0", "index"
    }

    out = []
    for c in cols:
        if str(c).strip().lower() not in excluded:
            out.append(c)
    return out


def inspect_site(site_name: str) -> Dict:
    """
    Inspect one site file and return inventory details.
    """
    site_lookup = SITE_ALIASES.get(site_name, site_name)
    filepath = find_site_file(site_lookup)

    result = {
        "requested_site": site_name,
        "lookup_site": site_lookup,
        "file_found": False,
        "filepath": "",
        "pm25_present": False,
        "pm25_column": "",
        "all_columns": [],
        "available_variables": [],
        "error": "",
    }

    if filepath is None:
        result["error"] = "FILE NOT FOUND"
        return result

    result["file_found"] = True
    result["filepath"] = filepath

    try:
        df_head = pd.read_csv(filepath, nrows=0)
        cols = list(df_head.columns)
        result["all_columns"] = cols
        result["available_variables"] = get_available_variables(cols)

        target_actual = find_site_specific_column(cols, TARGET_COLUMN, site_lookup)
        if target_actual is not None:
            result["pm25_present"] = True
            result["pm25_column"] = target_actual

    except Exception as e:
        result["error"] = f"HEADER READ ERROR: {e}"

    return result


# =============================================================================
# REPORT GENERATION
# =============================================================================


def build_pm25_inventory() -> Tuple[List[str], pd.DataFrame]:
    """
    Build report text lines and summary dataframe.
    """
    region_to_sites = config.REGION_TO_SITES
    target_regions = getattr(config, "TARGET_REGIONS", None)

    if target_regions:
        regions = list(target_regions)
    else:
        regions = list(region_to_sites.keys())

    lines: List[str] = []
    summary_rows: List[Dict] = []

    for region in regions:
        sites = region_to_sites.get(region, [])
        sites = [s for s in sites if s.upper() not in {x.upper() for x in EXCLUDED_SITES}]

        lines.append("=" * 120)
        lines.append(f"Region: {region}")
        lines.append("=" * 120)

        pm25_sites: List[str] = []
        site_to_variables: Dict[str, List[str]] = {}
        variable_to_sites: Dict[str, List[str]] = {}

        for site in sites:
            info = inspect_site(site)

            summary_rows.append({
                "Region": region,
                "Requested_Site": info["requested_site"],
                "Lookup_Site": info["lookup_site"],
                "File_Found": "YES" if info["file_found"] else "NO",
                "FilePath": info["filepath"],
                "PM25_Present": "YES" if info["pm25_present"] else "NO",
                "PM25_Column": info["pm25_column"],
                "Variable_Count": len(info["available_variables"]),
                "Available_Variables": ", ".join(info["available_variables"]),
                "Error": info["error"],
            })

            if info["pm25_present"]:
                pm25_sites.append(site)

            site_to_variables[site] = info["available_variables"]

            for var in info["available_variables"]:
                variable_to_sites.setdefault(var, []).append(site)

        # ---------------------------------------------------------------------
        # Section 1: sites with PM2.5
        # ---------------------------------------------------------------------
        lines.append("Sites with PM2.5:")
        if pm25_sites:
            for s in pm25_sites:
                lines.append(f"  - {s}")
        else:
            lines.append("  NONE")
        lines.append("")

        # ---------------------------------------------------------------------
        # Section 2: site -> available variables
        # ---------------------------------------------------------------------
        lines.append("Site -> Available variables:")
        if site_to_variables:
            for site in sites:
                vars_list = site_to_variables.get(site, [])
                lines.append(f"  {site}:")
                if vars_list:
                    lines.append(f"    {', '.join(vars_list)}")
                else:
                    lines.append("    NONE")
        else:
            lines.append("  NONE")
        lines.append("")

        # ---------------------------------------------------------------------
        # Section 3: variable -> sites
        # ---------------------------------------------------------------------
        lines.append("Variable -> Sites:")
        if variable_to_sites:
            for var in sorted(variable_to_sites.keys(), key=lambda x: x.upper()):
                sites_for_var = sorted(variable_to_sites[var], key=lambda x: x.upper())
                lines.append(f"  {var}: {', '.join(sites_for_var)}")
        else:
            lines.append("  NONE")
        lines.append("")

        # ---------------------------------------------------------------------
        # Section 4: file problems
        # ---------------------------------------------------------------------
        file_problem_rows = [r for r in summary_rows if r["Region"] == region and r["File_Found"] == "NO"]
        if file_problem_rows:
            lines.append("Sites with missing files:")
            for r in file_problem_rows:
                lines.append(f"  - {r['Requested_Site']}")
            lines.append("")

    df_summary = pd.DataFrame(summary_rows)
    return lines, df_summary


def save_outputs(lines: List[str], df_summary: pd.DataFrame) -> None:
    txt_fp = OUTPUT_ROOT / "pm25_region_inventory_report.txt"
    csv_fp = OUTPUT_ROOT / "pm25_region_inventory_summary.csv"

    with open(txt_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    df_summary.to_csv(csv_fp, index=False)

    print(f"\nReport written to:\n  {txt_fp}")
    print(f"Summary CSV written to:\n  {csv_fp}\n")

    logging.info(f"TXT report written: {txt_fp}")
    logging.info(f"CSV summary written: {csv_fp}")


def print_quick_preview(df_summary: pd.DataFrame) -> None:
    """
    Print a short console preview so you can immediately see that it worked.
    """
    print("\n" + "=" * 100)
    print("QUICK PREVIEW")
    print("=" * 100)

    if df_summary.empty:
        print("No rows found.")
        return

    grouped = df_summary.groupby("Region", dropna=False)

    for region, g in grouped:
        pm25_sites = g.loc[g["PM25_Present"] == "YES", "Requested_Site"].tolist()
        print(f"\nRegion: {region}")
        if pm25_sites:
            print("  Sites with PM2.5: " + ", ".join(pm25_sites))
        else:
            print("  Sites with PM2.5: NONE")


# =============================================================================
# MAIN
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Regional & Site-wise feature selection / PM2.5 inventory."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="report_pm25",
        choices=["per_site", "per_region", "report_pm25"],
        help="Mode: per_site, per_region, or report_pm25.",
    )
    args = parser.parse_args()

    if args.mode == "per_site":
        run_mode_per_site()
    elif args.mode == "per_region":
        run_mode_per_region()
    elif args.mode == "report_pm25":
        report_pm25_by_region()
    else:
        logging.error(f"Unknown mode: {args.mode}")


def report_pm25_by_region(out_fp: Optional[Path] = None):
    """
    Scan input files and report PM2.5 availability and variables per site/region.

    Outputs:
      1) Region -> sites with PM2.5
      2) Region -> Site -> available variables
      3) Region -> Variable -> sites
      4) CSV summary

    This function only reads file headers (and optionally the PM2.5 column count).
    It does NOT modify files and does NOT drop columns.
    """
    if out_fp is None:
        out_fp = OUTPUT_ROOT / "pm25_region_report.txt"

    csv_fp = OUTPUT_ROOT / "pm25_region_report_summary.csv"

    region_to_sites = config.REGION_TO_SITES
    regions = config.TARGET_REGIONS or list(region_to_sites.keys())

    lines = []
    summary_rows = []

    for region in regions:
        lines.append("=" * 120)
        lines.append(f"Region: {region}")
        lines.append("=" * 120)

        sites = region_to_sites.get(region, [])
        sites = [s for s in sites if s.upper() not in EXCLUDED_SITES]

        pm25_sites = []
        site_variable_map = {}
        variable_site_map = defaultdict(list)

        for site in sites:
            site_lookup = SITE_ALIASES.get(site, site)

            try:
                filepath = find_site_file(site_lookup)
            except Exception:
                filepath = None

            if filepath is None:
                lines.append(f"Site: {site}")
                lines.append("  File: NOT FOUND")
                lines.append("")
                summary_rows.append({
                    "Region": region,
                    "Site": site,
                    "Lookup_Site": site_lookup,
                    "File_Found": "NO",
                    "FilePath": "",
                    "PM25_Present": "NO",
                    "PM25_Column": "",
                    "PM25_NonNull_Count": "",
                    "Variable_Count": 0,
                    "Available_Variables": "",
                })
                continue

            try:
                df_head = pd.read_csv(filepath, nrows=0)
                cols = list(df_head.columns)
            except Exception as e:
                lines.append(f"Site: {site}")
                lines.append(f"  File: {filepath}")
                lines.append(f"  ERROR reading header: {e}")
                lines.append("")
                summary_rows.append({
                    "Region": region,
                    "Site": site,
                    "Lookup_Site": site_lookup,
                    "File_Found": "YES",
                    "FilePath": filepath,
                    "PM25_Present": "NO",
                    "PM25_Column": "",
                    "PM25_NonNull_Count": "",
                    "Variable_Count": 0,
                    "Available_Variables": "",
                })
                continue

            available_vars = get_available_variables(cols)
            site_variable_map[site] = available_vars

            for var in available_vars:
                variable_site_map[var].append(site)

            target_actual = find_site_specific_column(cols, TARGET_COLUMN, site_lookup)

            pm25_present = False
            pm25_nonnull_count = ""
            if target_actual is not None:
                pm25_present = True
                pm25_sites.append(site)

                try:
                    ser = pd.read_csv(filepath, usecols=[target_actual])[target_actual]
                    pm25_nonnull_count = int(ser.notna().sum())
                except Exception:
                    pm25_nonnull_count = "unreadable"

            summary_rows.append({
                "Region": region,
                "Site": site,
                "Lookup_Site": site_lookup,
                "File_Found": "YES",
                "FilePath": filepath,
                "PM25_Present": "YES" if pm25_present else "NO",
                "PM25_Column": target_actual if target_actual else "",
                "PM25_NonNull_Count": pm25_nonnull_count,
                "Variable_Count": len(available_vars),
                "Available_Variables": ", ".join(available_vars),
            })

        # ------------------------------------------------------------
        # Section 1: Sites with PM2.5
        # ------------------------------------------------------------
        lines.append("Sites with PM2.5:")
        if pm25_sites:
            for s in pm25_sites:
                lines.append(f"  - {s}")
        else:
            lines.append("  NONE")
        lines.append("")

        # ------------------------------------------------------------
        # Section 2: Site -> Available variables
        # ------------------------------------------------------------
        lines.append("Site -> Available variables:")
        if site_variable_map:
            for site in sites:
                vars_list = site_variable_map.get(site, [])
                lines.append(f"  {site}:")
                if vars_list:
                    lines.append(f"    {', '.join(vars_list)}")
                else:
                    lines.append("    NONE")
        else:
            lines.append("  NONE")
        lines.append("")

        # ------------------------------------------------------------
        # Section 3: Variable -> Sites
        # ------------------------------------------------------------
        lines.append("Variable -> Sites:")
        if variable_site_map:
            for var in sorted(variable_site_map.keys(), key=lambda x: x.upper()):
                lines.append(f"  {var}: {', '.join(sorted(variable_site_map[var]))}")
        else:
            lines.append("  NONE")
        lines.append("")

    # write outputs
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_fp, index=False)

    # print preview to terminal
    print("\n" + "=" * 120)
    print("PM2.5 REGION REPORT PREVIEW")
    print("=" * 120)
    for region in regions:
        subset = df_summary[df_summary["Region"] == region]
        pm25_yes = subset.loc[subset["PM25_Present"] == "YES", "Site"].tolist()
        print(f"\nRegion: {region}")
        if pm25_yes:
            print("  Sites with PM2.5: " + ", ".join(pm25_yes))
        else:
            print("  Sites with PM2.5: NONE")

    print(f"\nPM2.5 report written to: {out_fp}")
    print(f"PM2.5 summary CSV written to: {csv_fp}")

    logging.info(f"PM2.5 report written to: {out_fp}")
    logging.info(f"PM2.5 summary CSV written to: {csv_fp}")


if __name__ == "__main__":
    main()
