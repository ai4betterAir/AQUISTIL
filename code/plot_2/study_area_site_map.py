#!/usr/bin/env python3
"""Create a study-area site map with sites separated by region."""

import argparse
import json
from pathlib import Path
import struct
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    import config_spatial as config
except Exception:
    config = None


DEFAULT_STATION_INFO_FILE = (
    getattr(config, "STATION_INFO_FILE", "")
    if config is not None
    else "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AQ_DATA/AquisNET_Data/Air Quality API Excel Power Query.xlsx"
)
DEFAULT_STATION_INFO_SHEET = (
    getattr(config, "STATION_INFO_SHEET", "SiteDetails")
    if config is not None
    else "SiteDetails"
)
DEFAULT_OUTPUT_DIR = (
    Path(getattr(config, "OUTPUT_DIRECTORY", CODE_DIR.parent / "Outputs"))
    / "plots"
    / "study_area"
    if config is not None
    else CODE_DIR.parent / "Outputs" / "plots" / "study_area"
)
DEFAULT_EXCLUDED_REGION_KEYWORDS = (
    list(getattr(config, "EXCLUDED_REGION_KEYWORDS", ["offline", "LLS"]))
    if config is not None
    else ["offline", "LLS"]
)
DEFAULT_EXCLUDED_REGION_KEYWORDS += [
    "SA ",
    "Test Region",
    "Roadside Monitoring",
    "Research Monitoring",
    "Mallee CMA",
    "Gunnedah Emergency",
    "Cadia Monitoring DRX",
    "North Central CMA",
    "Incident Monitoring",
]
DEFAULT_BOUNDARY_FILE = (
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/NSW_AI_Dashboard/dashboard/assets/australian-states.json"
)
DEFAULT_LGA_BOUNDARY_FILE = (
    "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4MEMS/NSW_WHEMInventory_2026V1/Inputs/ShapeFile/LGA_2021_AUST_GDA2020_SHP/LGA_2021_AUST_GDA2020.shp"
)


REGION_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        direct = normalized.get(candidate.strip().lower())
        if direct is not None:
            return direct
    for candidate in candidates:
        needle = candidate.strip().lower()
        for column in columns:
            if needle in str(column).strip().lower():
                return column
    return None


def clean_string(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_station_table(path: Path, sheet_name: str) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raw = pd.read_csv(path)

    columns = list(raw.columns)
    site_col = find_column(columns, ["SiteName", "Site", "Station", "Column1.SiteName"])
    lon_col = find_column(columns, ["Longitude", "Lon", "Long", "Column1.Longitude"])
    lat_col = find_column(columns, ["Latitude", "Lat", "Column1.Latitude"])
    region_col = find_column(columns, ["Region", "Column1.Region"])

    missing = [
        name
        for name, column in {
            "site": site_col,
            "longitude": lon_col,
            "latitude": lat_col,
            "region": region_col,
        }.items()
        if column is None
    ]
    if missing:
        raise ValueError("Station table is missing required column(s): " + ", ".join(missing))

    data = pd.DataFrame(
        {
            "Site": raw[site_col].map(clean_string),
            "Longitude": pd.to_numeric(raw[lon_col], errors="coerce"),
            "Latitude": pd.to_numeric(raw[lat_col], errors="coerce"),
            "Region": raw[region_col].map(clean_string),
        }
    )
    data = data.loc[
        data["Site"].ne("")
        & data["Region"].ne("")
        & data["Longitude"].notna()
        & data["Latitude"].notna()
    ].copy()
    data = data.drop_duplicates(["Site", "Region", "Longitude", "Latitude"])
    return data.sort_values(["Region", "Site"], kind="stable").reset_index(drop=True)


def filter_regions(data: pd.DataFrame, excluded_keywords: List[str], include_regions: List[str]) -> pd.DataFrame:
    filtered = data.copy()
    if excluded_keywords:
        pattern = "|".join(excluded_keywords)
        filtered = filtered.loc[
            ~filtered["Region"].str.contains(pattern, case=False, na=False, regex=True)
        ].copy()
    if include_regions:
        wanted = {region.strip().lower() for region in include_regions if region.strip()}
        filtered = filtered.loc[filtered["Region"].str.lower().isin(wanted)].copy()
    return filtered.reset_index(drop=True)


def iter_polygon_rings(geometry: dict):
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        for ring in coordinates:
            yield ring
    elif geom_type == "MultiPolygon":
        for polygon in coordinates:
            for ring in polygon:
                yield ring


def feature_matches_state(feature: dict, state_name: str) -> bool:
    wanted = state_name.strip().lower()
    properties = feature.get("properties", {})
    return any(str(value).strip().lower() == wanted for value in properties.values())


def draw_state_boundary(ax, boundary_file: str, state_name: str = "New South Wales") -> bool:
    path = Path(boundary_file)
    if not path.exists():
        return False
    if path.suffix.lower() == ".shp":
        return draw_shapefile_boundary(ax, path, state_name)

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    features = data.get("features", []) if isinstance(data, dict) else []
    state_features = [feature for feature in features if feature_matches_state(feature, state_name)]
    if not state_features:
        return False

    for feature in state_features:
        geometry = feature.get("geometry") or {}
        for ring in iter_polygon_rings(geometry):
            if not ring:
                continue
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            ax.fill(xs, ys, facecolor="#f2f2f2", edgecolor="#666666", linewidth=0.9, zorder=1)
    return True


def read_dbf_records(dbf_path: Path) -> List[dict]:
    raw = dbf_path.read_bytes()
    record_count = struct.unpack("<I", raw[4:8])[0]
    header_length = struct.unpack("<H", raw[8:10])[0]
    record_length = struct.unpack("<H", raw[10:12])[0]

    fields = []
    offset = 32
    while offset < header_length - 1:
        descriptor = raw[offset : offset + 32]
        if not descriptor or descriptor[0] == 0x0D:
            break
        name = descriptor[:11].split(b"\0", 1)[0].decode("ascii", "ignore").strip()
        length = descriptor[16]
        fields.append((name, length))
        offset += 32

    records = []
    offset = header_length
    for _ in range(record_count):
        record = raw[offset : offset + record_length]
        offset += record_length
        if not record or record[0:1] == b"*":
            records.append({})
            continue
        cursor = 1
        values = {}
        for name, length in fields:
            value = record[cursor : cursor + length].decode("utf-8", "ignore").strip()
            values[name] = value
            cursor += length
        records.append(values)
    return records


def iter_shp_polygon_parts(shp_path: Path):
    with shp_path.open("rb") as handle:
        handle.seek(100)
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            _, content_length_words = struct.unpack(">2i", header)
            content = handle.read(content_length_words * 2)
            if len(content) < 44:
                yield []
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                yield []
                continue
            if shape_type not in {5, 15, 25, 31}:
                yield []
                continue

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_offset = 44
            points_offset = parts_offset + 4 * num_parts
            parts = list(struct.unpack("<%di" % num_parts, content[parts_offset:points_offset]))
            points = [
                struct.unpack("<2d", content[points_offset + i * 16 : points_offset + (i + 1) * 16])
                for i in range(num_points)
            ]
            part_ends = parts[1:] + [num_points]
            yield [points[start:end] for start, end in zip(parts, part_ends)]


def draw_shapefile_boundary(ax, shp_path: Path, state_name: str) -> bool:
    dbf_path = shp_path.with_suffix(".dbf")
    if not dbf_path.exists():
        return False

    records = read_dbf_records(dbf_path)
    wanted = state_name.strip().lower()
    drew_any = False

    for record, parts in zip(records, iter_shp_polygon_parts(shp_path)):
        if record.get("STE_NAME21", "").strip().lower() != wanted:
            continue
        for part in parts:
            if len(part) < 2:
                continue
            xs = [point[0] for point in part]
            ys = [point[1] for point in part]
            ax.plot(xs, ys, color="#8a8a8a", linewidth=0.35, alpha=0.85, zorder=1)
            drew_any = True
    return drew_any


def add_region_labels(ax, data: pd.DataFrame) -> None:
    for region, group in data.groupby("Region", sort=True):
        ax.text(
            group["Longitude"].mean(),
            group["Latitude"].mean(),
            region,
            fontsize=8,
            fontweight="bold",
            color="#222222",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.75},
        )


def plot_study_area(
    data: pd.DataFrame,
    output_dir: Path,
    label_sites: bool = False,
    boundary_file: str = "",
    state_name: str = "New South Wales",
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    has_boundary = draw_state_boundary(ax, boundary_file, state_name) if boundary_file else False

    regions = sorted(data["Region"].unique())
    cmap = plt.get_cmap("tab20")
    handles = []

    for index, region in enumerate(regions):
        group = data.loc[data["Region"].eq(region)]
        color = cmap(index % 20)
        marker = REGION_MARKERS[index % len(REGION_MARKERS)]
        ax.scatter(
            group["Longitude"],
            group["Latitude"],
            s=58,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            zorder=4,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=region,
                markersize=8,
            )
        )

    if label_sites:
        for _, row in data.iterrows():
            ax.annotate(
                row["Site"],
                (row["Longitude"], row["Latitude"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.5,
                color="#333333",
                alpha=0.85,
            )

    if has_boundary:
        ax.set_xlim(140.8, 154.3)
        ax.set_ylim(-37.8, -28.0)
    else:
        lon_pad = max((data["Longitude"].max() - data["Longitude"].min()) * 0.08, 0.08)
        lat_pad = max((data["Latitude"].max() - data["Latitude"].min()) * 0.08, 0.08)
        ax.set_xlim(data["Longitude"].min() - lon_pad, data["Longitude"].max() + lon_pad)
        ax.set_ylim(data["Latitude"].min() - lat_pad, data["Latitude"].max() + lat_pad)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Study Area Monitoring Sites by Region", loc="center", fontsize=16, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#bbbbbb", alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Region",
        fontsize=10,
        title_fontsize=12,
    )
    fig.tight_layout()

    png_path = output_dir / "study_area_sites_by_region.png"
    pdf_path = output_dir / "study_area_sites_by_region.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-info-file", default=DEFAULT_STATION_INFO_FILE)
    parser.add_argument("--sheet-name", default=DEFAULT_STATION_INFO_SHEET)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--boundary-file", default=DEFAULT_BOUNDARY_FILE)
    parser.add_argument("--state-name", default="New South Wales")
    parser.add_argument(
        "--use-lga-boundary",
        action="store_true",
        help="Draw the NSW LGA shapefile boundary behind the site markers.",
    )
    parser.add_argument("--regions", default="", help="Comma-separated region names to include. Empty means all.")
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include regions matching config.EXCLUDED_REGION_KEYWORDS.",
    )
    parser.add_argument("--label-sites", action="store_true", help="Label each site instead of region centroids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_regions = [item.strip() for item in args.regions.split(",") if item.strip()]
    excluded_keywords = [] if args.include_excluded else DEFAULT_EXCLUDED_REGION_KEYWORDS

    data = read_station_table(Path(args.station_info_file), args.sheet_name)
    data = filter_regions(data, excluded_keywords, include_regions)
    if data.empty:
        raise ValueError("No stations remain after filtering.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "study_area_sites_by_region.csv"
    data.to_csv(table_path, index=False)
    png_path, pdf_path = plot_study_area(
        data,
        output_dir,
        label_sites=args.label_sites,
        boundary_file=DEFAULT_LGA_BOUNDARY_FILE if args.use_lga_boundary else args.boundary_file,
        state_name=args.state_name,
    )

    print(f"Sites plotted: {len(data)}")
    print(f"Regions plotted: {data['Region'].nunique()}")
    print(f"CSV: {table_path}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
