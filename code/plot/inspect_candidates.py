#!/usr/bin/env python3
"""
inspect_candidates.py

Scan a results directory for candidate per-sample CSV files (filenames containing
keywords like 'pred', 'imput', 'imputed', etc.), print each file path, columns,
first rows, and attempt to detect observed/predicted columns and numeric counts.

Usage:
  python inspect_candidates.py /path/to/results_dir --max 12

Output to paste here:
  - For 2-4 representative files: the FILE line, the "columns:" list,
    the "detected obs/pred:" line, and numeric counts printed below.
"""
from pathlib import Path
import os
import fnmatch
import argparse
import pandas as pd
import re
import sys
import json

PATTERNS = ['*pred*.csv', '*imput*.csv', '*imputed*.csv', '*prediction*.csv', '*imputation*.csv', '*impute*.csv']

def find_candidate_files(results_dir: Path, patterns, max_files=None):
    files = []
    for root, _, filenames in os.walk(results_dir):
        for f in filenames:
            fl = f.lower()
            for pat in patterns:
                if fnmatch.fnmatch(fl, pat):
                    files.append(Path(root) / f)
                    break
            if max_files and len(files) >= max_files:
                break
        if max_files and len(files) >= max_files:
            break
    return files

def detect_obs_pred(cols):
    obs_keys=['y_true','y_obs','observed','obs','truth','ground_truth','actual','value','observed_value','observedvalue']
    pred_keys=['y_pred','y_hat','pred','prediction','imputed','imputed_value','imputedvalue','predicted','estimate','value_pred']
    cols_l=[c.lower() for c in cols]
    # exact match pairs
    for ok in obs_keys:
        for pk in pred_keys:
            if ok in cols_l and pk in cols_l:
                return cols[cols_l.index(ok)], cols[cols_l.index(pk)]
    # contains-match
    obs = [c for c in cols if any(k in c.lower() for k in ['obs','true','actual','observ','value'])]
    pred = [c for c in cols if any(k in c.lower() for k in ['pred','imput','prediction','estimate','hat'])]
    if obs and pred:
        return obs[0], pred[0]
    # fallback: try some likely pair patterns
    pairs = [('observed','imputed'), ('truth','imputed'), ('actual','predicted'), ('observed','prediction')]
    for o_kw,p_kw in pairs:
        o_c = [c for c in cols if o_kw in c.lower()]
        p_c = [c for c in cols if p_kw in c.lower()]
        if o_c and p_c:
            return o_c[0], p_c[0]
    return None, None

def top_numeric_columns(df, top_n=6):
    counts = {}
    for c in df.columns:
        try:
            s = pd.to_numeric(df[c], errors='coerce')
            counts[c] = int(s.notna().sum())
        except Exception:
            counts[c] = 0
    items = sorted(counts.items(), key=lambda x:-x[1])
    return items[:top_n]

def inspect_file(fpath: Path, rows=10):
    print("="*120)
    print("FILE:", fpath)
    try:
        df = pd.read_csv(fpath, nrows=rows)
    except Exception as e:
        print("  ERROR reading file:", e)
        try:
            size = fpath.stat().st_size
            print("  filesize (bytes):", size)
        except Exception:
            pass
        return

    cols = list(df.columns)
    print("  columns:", cols)
    obs_col, pred_col = detect_obs_pred(cols)
    print("  detected obs/pred:", (obs_col, pred_col))
    # print first rows (JSON safe)
    try:
        preview = df.head(5).fillna("").astype(str).to_dict(orient='records')
        print("  first rows (up to 5):")
        print(json.dumps(preview, indent=2)[:4000])  # truncate if huge
    except Exception as e:
        print("  error printing rows:", e)

    # numeric counts for detected obs/pred
    if obs_col and pred_col:
        try:
            obs_num = pd.to_numeric(df[obs_col], errors='coerce')
            pred_num = pd.to_numeric(df[pred_col], errors='coerce')
            n_obs_num = int(obs_num.notna().sum())
            n_pred_num = int(pred_num.notna().sum())
            n_both = int((obs_num.notna() & pred_num.notna()).sum())
            print(f"  numeric counts -> obs: {n_obs_num}, pred: {n_pred_num}, both numeric: {n_both}")
        except Exception as e:
            print("  error computing numeric counts for detected obs/pred:", e)
    else:
        # show top numeric-like columns
        top = top_numeric_columns(df, top_n=8)
        print("  top numeric-like columns (col, count):", top)
    try:
        print("  filesize (bytes):", fpath.stat().st_size)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Inspect candidate per-sample CSV files in results dir.")
    parser.add_argument("results_dir", type=str, nargs="?", default="/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/Imputation_Result_Spatial_Temporal_V19_final", help="Path to results dir")
    parser.add_argument("--max", type=int, default=12, help="Max candidate files to inspect")
    parser.add_argument("--rows", type=int, default=10, help="Rows to read for preview")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print("results_dir not found:", results_dir)
        return 2

    cands = find_candidate_files(results_dir, PATTERNS, max_files=args.max)
    print("Found candidate files:", len(cands))
    if not cands:
        print("No candidate files found by patterns. Consider different filename patterns or provide --per_sample_csv.")
        return 0

    for f in cands[:args.max]:
        inspect_file(f, rows=args.rows)

if __name__ == "__main__":
    main()