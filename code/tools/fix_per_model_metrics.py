#!/usr/bin/env python3
"""Fix per-model metrics CSVs:
- remove `Source_File` column (case-insensitive)
- for `Missingness_Regime` and `Missingness_Level` keep only the first matching column (case-insensitive) and drop the rest

Saves files in-place and prints a summary.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "Imputation_Result_Spatial_Temporal_V25_final"
PM_DIR = ROOT / "per_model_metrics"

if not PM_DIR.exists():
    print("per_model_metrics folder not found:", PM_DIR)
    sys.exit(1)

changed = []

def first_match_keep(cols, canonical):
    # return list of columns to drop: all that contain canonical (case-insensitive) except the first
    matches = [c for c in cols if canonical in c.lower()]
    if len(matches) <= 1:
        return []
    return matches[1:]

for fp in sorted(PM_DIR.glob("*_metrics.csv")):
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        print(f"Skipping {fp} (read error): {e}")
        continue

    orig_cols = df.columns.tolist()
    to_drop = []

    # Drop any Source_File (case-insensitive)
    to_drop += [c for c in df.columns if c.lower() == 'source_file']

    # Deduplicate missingness columns: keep first occurrence of any column containing these substrings
    for canonical in ('missingness_regime', 'missingness_level'):
        to_drop += first_match_keep(df.columns.tolist(), canonical)

    # Remove exact duplicate column names created by some writers (e.g., 'Missingness' and 'Missingness.1')
    # For any repeated exact name, keep the first
    seen = set()
    for c in df.columns:
        nm = c
        if nm in seen:
            to_drop.append(c)
        else:
            seen.add(nm)

    # Unique-ify to_drop preserving order
    seen_drop = set()
    final_drop = []
    for c in to_drop:
        if c not in seen_drop and c in df.columns:
            final_drop.append(c)
            seen_drop.add(c)

    if final_drop:
        df = df.drop(columns=final_drop)
        df.to_csv(fp, index=False)
        changed.append((fp.name, final_drop, orig_cols, df.columns.tolist()))

print("Processed per_model_metrics files:")
for name, dropped, before, after in changed:
    print(f"- {name}: dropped {dropped}")

if not changed:
    print("No changes made (no Source_File or duplicate missingness columns found).")
