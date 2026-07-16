#!/usr/bin/env python3
"""
diagnose_merges.py

Scan imputed files and their matched original files and report detailed diagnostics
about why merges succeed/fail.

Output:
 - <output_dir>/merge_diagnostics.csv  : one row per imputed file diagnostic
 - <output_dir>/diagnostics/<safe_name>_orig_head.csv  : first rows of the chosen original (if readable)
 - <output_dir>/diagnostics/<safe_name>_imp_head.csv   : first rows of the imputed file (if readable)
 - <output_dir>/diagnostics/<safe_name>_merged_head.csv: first rows of the merged result (if merge succeeded)

Run:
  python plot/diagnose_merges.py --results_dir /path/to/results --output_dir /path/to/output --max 50
"""
from pathlib import Path
import argparse
import logging
import os
import re
import fnmatch
import sys
import csv

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Patterns / heuristics (same as the plotting script)
TARGET_PATTERNS = [r"PM2[._]?5", r"PM10", r"NO2", r"O3", r"SO2", r"\bCO\b"]
TARGET_RE = re.compile("|".join(TARGET_PATTERNS), flags=re.IGNORECASE)
REGIME_ORDER = ["random", "short_gap", "medium_gap", "long_gap", "event"]


def find_all_csvs(results_dir: Path):
    all_csvs = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                all_csvs.append(Path(root) / f)
    return all_csvs


def is_imputed_file(path: Path):
    name = path.name.lower()
    return any(k in name for k in ["imput", "imputed", "_imputed_", "prediction", "_pred_", "predictions", "y_pred"])


def extract_target_from_stem(stem: str):
    m = TARGET_RE.search(stem)
    if not m:
        return None
    tok = m.group(0)
    tok = tok.upper().replace("_", ".")
    tok = re.sub(r"^PM25$", "PM2.5", tok)
    tok = re.sub(r"^PM2_5$", "PM2.5", tok)
    return tok


def prefix_up_to_target(stem: str, target_tok: str):
    parts = re.split(r"[_\-\.\s]+", stem)
    if not target_tok:
        return "_".join(parts[:3]) if len(parts) >= 3 else parts[0]
    targ = target_tok.replace(".", "").lower()
    for i, p in enumerate(parts):
        if targ in p.lower().replace("_", ""):
            return "_".join(parts[: i + 1])
    return "_".join(parts[:3]) if len(parts) >= 3 else parts[0]


def safe_read_csv(path: Path, nrows=10):
    try:
        df = pd.read_csv(path, nrows=nrows)
        return df
    except Exception as e:
        logging.debug(f"Failed to read {path}: {e}")
        return None


def choose_value_col(df):
    counts = {}
    for c in df.columns:
        try:
            counts[c] = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
        except Exception:
            counts[c] = 0
    if not counts:
        return None, {}
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    return sorted_items[0][0], dict(sorted_items)


def normalize_datetime_col(df):
    for c in df.columns:
        if c.lower() == "datetime" or "date" in c.lower():
            return c
    return None


def merge_original_imputed_for_diag(original_path: Path, imputed_path: Path):
    """
    Attempt a merge and return diagnostic dict:
      { 'orig_readable', 'imp_readable', 'orig_cols', 'imp_cols',
        'orig_top_col', 'imp_top_col', 'orig_top_counts', 'imp_top_counts',
        'orig_dt_col', 'imp_dt_col', 'merge_rows', 'merge_preview' }
    """
    diag = {
        'orig_readable': False,
        'imp_readable': False,
        'orig_cols': None,
        'imp_cols': None,
        'orig_top_col': None,
        'imp_top_col': None,
        'orig_top_counts': None,
        'imp_top_counts': None,
        'orig_dt_col': None,
        'imp_dt_col': None,
        'merge_rows': 0,
        'merge_error': None,
        'merged_preview_path': None
    }
    df_orig = safe_read_csv(original_path, nrows=200)
    if df_orig is None:
        diag['merge_error'] = f"cannot read original {original_path}"
        return diag
    diag['orig_readable'] = True
    diag['orig_cols'] = list(df_orig.columns)
    orig_top_col, orig_counts = choose_value_col(df_orig)
    diag['orig_top_col'] = orig_top_col
    diag['orig_top_counts'] = orig_counts
    diag['orig_dt_col'] = normalize_datetime_col(df_orig)

    df_imp = safe_read_csv(imputed_path, nrows=200)
    if df_imp is None:
        diag['merge_error'] = f"cannot read imputed {imputed_path}"
        return diag
    diag['imp_readable'] = True
    diag['imp_cols'] = list(df_imp.columns)
    imp_top_col, imp_counts = choose_value_col(df_imp)
    diag['imp_top_col'] = imp_top_col
    diag['imp_top_counts'] = imp_counts
    diag['imp_dt_col'] = normalize_datetime_col(df_imp)

    # Try to merge using DateTime_parsed if available
    try:
        df_full_orig = pd.read_csv(original_path)
        df_full_imp = pd.read_csv(imputed_path)
    except Exception as e:
        diag['merge_error'] = f"failed to read full files: {e}"
        return diag

    dt_o = normalize_datetime_col(df_full_orig)
    dt_i = normalize_datetime_col(df_full_imp)

    # fallback by index if no DateTime but length equal
    merged_df = None
    try:
        if dt_o and dt_i:
            df_full_orig = df_full_orig.rename(columns={dt_o: "DateTime"})
            df_full_imp = df_full_imp.rename(columns={dt_i: "DateTime"})
            # try parse datetimes
            df_full_orig['DateTime_parsed'] = pd.to_datetime(df_full_orig['DateTime'], errors='coerce')
            df_full_imp['DateTime_parsed'] = pd.to_datetime(df_full_imp['DateTime'], errors='coerce')
            if df_full_orig['DateTime_parsed'].notna().sum() > 0 and df_full_imp['DateTime_parsed'].notna().sum() > 0:
                left = df_full_orig[['DateTime_parsed', orig_top_col]].rename(columns={'DateTime_parsed': 'DateTime', orig_top_col: 'y_true'})
                right = df_full_imp[['DateTime_parsed', imp_top_col]].rename(columns={'DateTime_parsed': 'DateTime', imp_top_col: 'y_pred'})
                merged_df = pd.merge(left, right, on='DateTime', how='inner')
            else:
                # merge on raw string DateTime
                left = df_full_orig[['DateTime', orig_top_col]].rename(columns={orig_top_col: 'y_true'})
                right = df_full_imp[['DateTime', imp_top_col]].rename(columns={imp_top_col: 'y_pred'})
                merged_df = pd.merge(left, right, on='DateTime', how='inner')
        else:
            if len(df_full_orig) == len(df_full_imp):
                merged_df = pd.DataFrame({
                    'y_true': pd.to_numeric(df_full_orig[orig_top_col], errors='coerce'),
                    'y_pred': pd.to_numeric(df_full_imp[imp_top_col], errors='coerce')
                }).dropna().reset_index(drop=True)
            else:
                diag['merge_error'] = "no DateTime and lengths differ"
                merged_df = None
    except Exception as e:
        diag['merge_error'] = f"merge exception: {e}"
        merged_df = None

    if merged_df is None or merged_df.empty:
        diag['merge_rows'] = 0
        return diag

    merged_df['y_true'] = pd.to_numeric(merged_df['y_true'], errors='coerce')
    merged_df['y_pred'] = pd.to_numeric(merged_df['y_pred'], errors='coerce')
    merged_df = merged_df.dropna(subset=['y_true', 'y_pred']).reset_index(drop=True)
    diag['merge_rows'] = int(len(merged_df))
    return diag, merged_df.head(50)


def safe_name_from_paths(orig: Path, imp: Path):
    s = f"{imp.stem}__orig__{orig.stem}"
    s = re.sub(r"[^0-9A-Za-z\-_\.]", "_", s)
    return s[:200]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose merges between imputed and original CSVs.")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max", type=int, default=50, help="Max imputed files to inspect")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    all_csvs = find_all_csvs(results_dir)
    imputed_files = [p for p in all_csvs if is_imputed_file(p)]
    original_candidates = [p for p in all_csvs if not is_imputed_file(p)]
    originals_by_stem = {p.stem: p for p in original_candidates}

    logging.info("Found %d imputed files and %d original candidates", len(imputed_files), len(original_candidates))

    rows = []
    inspected = 0
    for imp in imputed_files:
        if inspected >= args.max:
            break
        inspected += 1
        stem = imp.stem
        target = extract_target_from_stem(stem)
        prefix = prefix_up_to_target(stem, target)
        # find best original similar to prefix (tolerant)
        orig = None
        if prefix in originals_by_stem:
            orig = originals_by_stem[prefix]
        else:
            # try contains
            pref_lower = prefix.lower()
            cands = [p for s, p in originals_by_stem.items() if pref_lower in s.lower()]
            if cands:
                orig = sorted(cands, key=lambda x: (len(x.parts), len(x.name)))[0]
            else:
                # try any file that contains target token
                if target:
                    tkey = target.replace(".", "").lower()
                    cands = [p for s, p in originals_by_stem.items() if tkey in s.lower()]
                    if cands:
                        orig = sorted(cands, key=lambda x: (len(x.parts), len(x.name)))[0]
        if orig is None:
            rows.append({
                'imputed_path': str(imp),
                'original_path': '',
                'prefix': prefix,
                'target': target,
                'merge_rows': 0,
                'merge_error': 'no original found'
            })
            continue

        diag_result = merge_original_imputed_for_diag(orig, imp)
        if isinstance(diag_result, tuple):
            diag, merged_head = diag_result
        else:
            diag = diag_result
            merged_head = None

        safe_name = safe_name_from_paths(orig, imp)
        # save heads
        try:
            dfo = safe_read_csv(orig, nrows=200)
            if dfo is not None:
                dfo.head(20).to_csv(diag_dir / f"{safe_name}_orig_head.csv", index=False)
        except Exception:
            pass
        try:
            dfi = safe_read_csv(imp, nrows=200)
            if dfi is not None:
                dfi.head(20).to_csv(diag_dir / f"{safe_name}_imp_head.csv", index=False)
        except Exception:
            pass
        if merged_head is not None:
            try:
                merged_head.to_csv(diag_dir / f"{safe_name}_merged_head.csv", index=False)
            except Exception:
                pass

        row = {
            'imputed_path': str(imp),
            'original_path': str(orig),
            'prefix': prefix,
            'target': target,
            'orig_cols': ";".join(diag.get('orig_cols') or []),
            'imp_cols': ";".join(diag.get('imp_cols') or []),
            'orig_top_col': diag.get('orig_top_col'),
            'imp_top_col': diag.get('imp_top_col'),
            'orig_dt_col': diag.get('orig_dt_col'),
            'imp_dt_col': diag.get('imp_dt_col'),
            'merge_rows': diag.get('merge_rows', 0),
            'merge_error': diag.get('merge_error', '')
        }
        rows.append(row)

    # write diagnostics CSV
    out_csv = output_dir / "merge_diagnostics.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    logging.info("Wrote diagnostics: %s (and per-file heads under %s)", out_csv, diag_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())