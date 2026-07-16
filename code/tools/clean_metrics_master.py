#!/usr/bin/env python3
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "Imputation_Result_Spatial_Temporal_V25_final"
TARGET = ROOT / "metrics_master.csv"
BACKUP = ROOT / "metrics_master.csv.bak"
REMOVED = ROOT / "metrics_master_removed_rows.csv"

print(f"Target: {TARGET}")
if not TARGET.exists():
    raise SystemExit("metrics_master.csv not found")

shutil.copy2(TARGET, BACKUP)
print(f"Backed up to {BACKUP}")

with TARGET.open('r', newline='') as f:
    reader = csv.reader(f)
    rows = list(reader)

if not rows:
    raise SystemExit("Empty CSV")

header = rows[0]
hlen = len(header)
print(f"Header columns: {hlen}")

kept = [header]
removed = []

for i, row in enumerate(rows[1:], start=2):
    # treat rows that are fully empty as removed
    if all((not c.strip()) for c in row):
        removed.append((i, row))
        continue
    if len(row) != hlen:
        removed.append((i, row))
    else:
        kept.append(row)

# Write cleaned file
with TARGET.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(kept)

print(f"Wrote cleaned CSV with {len(kept)-1} data rows (removed {len(removed)})")

# Save removed rows for inspection
with REMOVED.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["line_number","raw_fields_count","row_data"])
    for ln, row in removed:
        writer.writerow([ln, len(row), "|".join(row)])

print(f"Saved removed rows to {REMOVED}")
print("Done.")
