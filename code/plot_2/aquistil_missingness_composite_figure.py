#!/usr/bin/env python3
"""Create a Figure-1-style missingness characterization composite.

The figure is built from local AQUISTIL per-site input CSVs plus the station
metadata workbook. It intentionally avoids geopandas/cartopy so it can run in
the current project environment.
"""

import argparse
import contextlib
import importlib.util
import io
import math
import sys
import textwrap
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
AQUISTIL_ROOT = CODE_DIR.parent
WORKSPACE_ROOT = AQUISTIL_ROOT.parent

MISSINGNESS_AUDIT = CODE_DIR / "FeatureStats" / "05missingness_audit.py"
STUDY_MAP = SCRIPT_DIR / "study_area_site_map.py"

DEFAULT_INPUTS_DIR = AQUISTIL_ROOT / "Outputs/Imputation_Result/Inputs_PerSite"
DEFAULT_MAPPING = AQUISTIL_ROOT / "Outputs/Imputation_Result/region_site_mapping.json"
DEFAULT_OUTPUT_DIR = AQUISTIL_ROOT / "Outputs/Publication_Figures/Missingness_Composite"
DEFAULT_AUDIT_DIR = AQUISTIL_ROOT / "Outputs/Publication_Figures/Missingness_Audit"
DEFAULT_STATION_INFO = WORKSPACE_ROOT / "AQ_DATA/AquisNET_Data/Air Quality API Excel Power Query.xlsx"
DEFAULT_BOUNDARY_FILE = (
    WORKSPACE_ROOT
    / "AI_Nowcasting/NSW_AI_Dashboard/dashboard/assets/australian-states.json"
)
DEFAULT_POLLUTANTS = ["PM2.5", "PM10", "OZONE"]
SATELLITE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
    "MapServer/tile/{z}/{y}/{x}"
)
STATION_CLASS_COLOURS = {
    "Metropolitan": "#E31A1C",
    "Regional": "#2463D4",
    "Rural": "#1B9E4B",
}
REGION_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#A65628", "#F781BF", "#17BECF",
]
PANEL_BACKGROUNDS = {
    "map": "#F1FAEE",
    "regional": "#FFF8E6",
    "matrix": "#EEF5FF",
    "gap": "#F7F1FF",
    "seasonal": "#F2FAF4",
    "hourly": "#FFF0F0",
    "pm25": "#EEF5FF",
    "pm10": "#FFF3E8",
    "ozone": "#EDF8ED",
    "default": "#F7F9FC",
}
AUSTRALIAN_SEASON_MONTHS = [
    ("Summer", 12, "Dec"),
    ("Summer", 1, "Jan"),
    ("Summer", 2, "Feb"),
    ("Autumn", 3, "Mar"),
    ("Autumn", 4, "Apr"),
    ("Autumn", 5, "May"),
    ("Winter", 6, "Jun"),
    ("Winter", 7, "Jul"),
    ("Winter", 8, "Aug"),
    ("Spring", 9, "Sep"),
    ("Spring", 10, "Oct"),
    ("Spring", 11, "Nov"),
]
SEASON_COLOURS = {
    "Summer": "#D95F02",
    "Autumn": "#A65628",
    "Winter": "#377EB8",
    "Spring": "#4DAF4A",
}
SEASON_LABELS = {
    "Summer": "Summer: Dec, Jan, Feb",
    "Autumn": "Autumn: Mar, Apr, May",
    "Winter": "Winter: Jun, Jul, Aug",
    "Spring": "Spring: Sep, Oct, Nov",
}


def wrap_title(title: str, width: int = 42) -> str:
    lines = []
    for part in str(title).splitlines():
        lines.extend(textwrap.wrap(part, width=width) or [""])
    return "\n".join(lines)


def set_panel_background(ax, colour="#F7F9FC"):
    ax.set_facecolor(colour)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def add_inside_title(
    ax,
    title,
    *,
    fontsize=14,
    color="#111111",
    x=0.02,
    y=0.98,
    ha="left",
    width=42,
    box_face="white",
    box_alpha=0.88,
):
    ax.text(
        x,
        y,
        wrap_title(title, width=width),
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        linespacing=1.05,
        bbox={
            "facecolor": box_face,
            "edgecolor": "none",
            "alpha": box_alpha,
            "boxstyle": "round,pad=0.22",
        },
        zorder=20,
    )


def set_above_title(ax, title, *, fontsize=14, loc="left", color="#111111", width=70):
    ax.set_title(
        wrap_title(title, width=width),
        loc=loc,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        pad=8,
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


audit = load_module(MISSINGNESS_AUDIT, "aquistil_missingness_audit")
study_map = load_module(STUDY_MAP, "aquistil_study_area_site_map")


def canon(value):
    return audit.canon(value)


def pollutant_label(pollutant: str) -> str:
    return {"PM2.5": r"PM$_{2.5}$", "PM10": r"PM$_{10}$", "OZONE": r"O$_3$"}.get(
        pollutant, pollutant
    )


def normalise_station_name(value: str) -> str:
    return str(value).strip().replace("_", "-").upper()


def load_station_metadata(args, sites):
    data = study_map.read_station_table(args.station_info_file, args.station_sheet)
    data = study_map.filter_regions(data, args.excluded_region_keywords, [])
    wanted = {canon(site) for site in sites}
    data = data.loc[data["Site"].map(canon).isin(wanted)].copy()
    if data.empty:
        raise RuntimeError("No station metadata matched the per-site CSV inputs.")
    data["Station_Class"] = data["Region"].map(classify_region)
    return data.sort_values(["Region", "Site"], kind="stable").reset_index(drop=True)


def classify_region(region: str) -> str:
    text = str(region).lower()
    if "sydney" in text or "newcastle local" in text or "roadside" in text:
        return "Metropolitan"
    if "hunter" in text or "illawarra" in text or "central coast" in text:
        return "Regional"
    if "cadia" in text:
        return "Industrial"
    if "research" in text:
        return "Background"
    return "Rural"


def make_availability_matrix(long_df, pollutants, station_order):
    records = []
    subset = long_df.loc[long_df["Pollutant"].isin(pollutants)].copy()
    subset["SiteKey"] = subset["Site"].map(canon)
    for site in station_order:
        part = subset.loc[subset["SiteKey"].eq(canon(site))]
        if part.empty:
            continue
        by_time = part.groupby("DateTime")["Is_Missing"].max().sort_index()
        records.append((site, by_time))
    if not records:
        raise RuntimeError("Could not build availability matrix.")
    timestamps = pd.date_range(
        min(series.index.min() for _, series in records),
        max(series.index.max() for _, series in records),
        freq="h",
    )
    matrix = []
    for _, series in records:
        hourly = series.reindex(timestamps)
        matrix.append(hourly.astype(float).fillna(1.0).to_numpy())
    return timestamps, np.vstack(matrix)


def gap_category(length):
    if length == 1:
        return "Isolated\n(1 hour)"
    if length <= 6:
        return "Short\n(1-6 h)"
    if length <= 24:
        return "Medium\n(7-24 h)"
    if length <= 72:
        return "Long\n(25-72 h)"
    return "Extended\n(>72 h)"


def _lonlat_to_tile(lon, lat, zoom):
    lat = max(-85.05112878, min(85.05112878, lat))
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _tile_to_lonlat(x, y, zoom):
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def draw_satellite_basemap(ax, bounds, zoom, cache_dir):
    """Draw a cached Esri World Imagery tile mosaic in lon/lat coordinates."""
    west, east, south, north = bounds
    x0f, y0f = _lonlat_to_tile(west, north, zoom)
    x1f, y1f = _lonlat_to_tile(east, south, zoom)
    x0, x1 = int(math.floor(x0f)), int(math.floor(x1f))
    y0, y1 = int(math.floor(y0f)), int(math.floor(y1f))
    cache_dir.mkdir(parents=True, exist_ok=True)
    mosaic = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    try:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                tile_path = cache_dir / str(zoom) / str(x) / f"{y}.jpg"
                if not tile_path.exists():
                    tile_path.parent.mkdir(parents=True, exist_ok=True)
                    request = Request(
                        SATELLITE_URL.format(z=zoom, x=x, y=y),
                        headers={"User-Agent": "AQUISTIL publication figure/1.0"},
                    )
                    with urlopen(request, timeout=20) as response:
                        tile_path.write_bytes(response.read())
                with Image.open(tile_path) as tile:
                    mosaic.paste(tile.convert("RGB"), ((x - x0) * 256, (y - y0) * 256))
    except Exception as exc:
        print(f"Warning: satellite basemap unavailable ({exc}); using boundary background.", file=sys.stderr)
        return False

    image_west, image_north = _tile_to_lonlat(x0, y0, zoom)
    image_east, image_south = _tile_to_lonlat(x1 + 1, y1 + 1, zoom)
    ax.imshow(
        np.asarray(mosaic),
        extent=[image_west, image_east, image_south, image_north],
        origin="upper",
        interpolation="bilinear",
        zorder=0,
    )
    return True


def draw_map(ax, stations, boundary_file, satellite_cache, region_colors=None):
    set_panel_background(ax, PANEL_BACKGROUNDS["map"])
    station_pad_x, station_pad_y = 0.55, 0.45
    station_bounds = (
        stations["Longitude"].min() - station_pad_x,
        stations["Longitude"].max() + station_pad_x,
        stations["Latitude"].min() - station_pad_y,
        stations["Latitude"].max() + station_pad_y,
    )
    has_satellite = draw_satellite_basemap(ax, station_bounds, 7, satellite_cache)
    if not has_satellite:
        study_map.draw_state_boundary(ax, str(boundary_file), "New South Wales")

    color_field = "Region" if region_colors else "Station_Class"
    for group_name, group in stations.groupby(color_field, sort=True):
        marker_color = (
            region_colors.get(canon(group_name), "#777777")
            if region_colors
            else STATION_CLASS_COLOURS.get(group_name, "#6A3D9A")
        )
        ax.scatter(
            group["Longitude"],
            group["Latitude"],
            s=78,
            c=marker_color,
            marker="o",
            edgecolor="black",
            linewidth=1.1,
            zorder=5,
        )
    ax.set_xlim(station_bounds[0], station_bounds[1])
    ax.set_ylim(station_bounds[2], station_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", linewidth=0.45, linestyle=":", alpha=0.55)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=7, colors="white" if has_satellite else "#333333")

    # NSW-wide locator: the rectangle identifies the station-network extent.
    # Keep the locator large enough to remain legible in the composite figure.
    inset = ax.inset_axes([0.02, 0.65, 0.54, 0.33])
    nsw_bounds = (140.7, 154.2, -37.8, -28.0)
    inset_satellite = draw_satellite_basemap(inset, nsw_bounds, 5, satellite_cache)
    study_map.draw_state_boundary(inset, str(boundary_file), "New South Wales")
    if inset_satellite:
        for patch in inset.patches:
            patch.set_facecolor("none")
            patch.set_edgecolor("white")
            patch.set_linewidth(0.8)
    inset.add_patch(
        Rectangle(
            (station_bounds[0], station_bounds[2]),
            station_bounds[1] - station_bounds[0],
            station_bounds[3] - station_bounds[2],
            fill=False,
            edgecolor="#FFD400",
            linewidth=1.5,
            zorder=7,
        )
    )
    inset.set_xlim(nsw_bounds[0], nsw_bounds[1])
    inset.set_ylim(nsw_bounds[2], nsw_bounds[3])
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    add_inside_title(
        inset,
        "NSW",
        fontsize=7,
        color="white" if inset_satellite else "#222222",
        x=0.04,
        y=0.96,
        width=10,
        box_face="black" if inset_satellite else "white",
        box_alpha=0.45 if inset_satellite else 0.82,
    )
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)

    ax.text(
        0.985, 0.015, f"n = {len(stations)}  |  Esri World Imagery",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        color="white" if has_satellite else "#333333",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 2},
    )


def draw_missingness_bars(
    fig, outer_spec, site_pollutant, pollutants, axis_title_fontsize=13.2
):
    axes = outer_spec.subgridspec(1, len(pollutants), wspace=0.05).subplots()
    if len(pollutants) == 1:
        axes = [axes]
    ranking = (
        site_pollutant.loc[site_pollutant["Pollutant"].isin(pollutants)]
        .groupby("Site")["Missingness_Percent"]
        .mean()
        .sort_values(ascending=False)
    )
    max_sites = min(14, len(ranking))
    sites = ranking.head(max_sites).index.tolist()
    palette = {"PM2.5": "#2D6CDF", "PM10": "#FF7A1A", "OZONE": "#2CA02C"}
    for i, (ax, pollutant) in enumerate(zip(axes, pollutants)):
        set_panel_background(
            ax,
            {
                "PM2.5": PANEL_BACKGROUNDS["pm25"],
                "PM10": PANEL_BACKGROUNDS["pm10"],
                "OZONE": PANEL_BACKGROUNDS["ozone"],
            }.get(pollutant, PANEL_BACKGROUNDS["default"]),
        )
        values = (
            site_pollutant.loc[
                site_pollutant["Pollutant"].eq(pollutant)
                & site_pollutant["Site"].isin(sites),
                ["Site", "Missingness_Percent"],
            ]
            .set_index("Site")
            .reindex(sites)["Missingness_Percent"]
            .fillna(0)
        )
        y = np.arange(len(sites))
        ax.barh(y, values.to_numpy(), color=palette[pollutant], alpha=0.45, edgecolor=palette[pollutant])
        ax.set_xlim(0, 20 if pollutant in {"PM2.5", "PM10"} else 100)
        set_above_title(
            ax,
            pollutant_label(pollutant),
            fontsize=12,
            color=palette[pollutant],
            loc="center",
            width=14,
        )
        ax.grid(axis="x", color="#D9D9D9", linestyle="--", linewidth=0.7)
        ax.invert_yaxis()
        ax.set_xlabel(
            "Missing observations (%)" if i == 1 else "",
            fontsize=axis_title_fontsize,
        )
        ax.tick_params(axis="x", labelsize=12)
        if i == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([normalise_station_name(s).title() for s in sites], fontsize=10.8)
            ax.set_ylabel("Stations", fontsize=axis_title_fontsize)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    return axes


def draw_availability_matrix(ax, dates, availability):
    set_panel_background(ax, PANEL_BACKGROUNDS["matrix"])
    ax.imshow(
        availability,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#BFE6FA", "#12246D"]),
        vmin=0,
        vmax=1,
    )
    years = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="YS")
    positions = [np.searchsorted(dates, y) for y in years]
    ax.set_xticks(positions)
    ax.set_xticklabels([str(y.year) for y in years], fontsize=13.8)
    ax.set_yticks([])
    ax.set_ylabel("")
    for pos in positions:
        ax.axvline(pos, color="#333333", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(
        handles=[Patch(facecolor="#BFE6FA", label="Observed"), Patch(facecolor="#12246D", label="Missing")],
        loc="lower left",
        bbox_to_anchor=(0.66, 1.015),
        ncol=2,
        frameon=False,
        fontsize=13.2,
        borderaxespad=0,
        handlelength=2.0,
        columnspacing=2.0,
    )


def draw_gap_distribution(ax, gaps):
    set_panel_background(ax, PANEL_BACKGROUNDS["gap"])
    cats = ["Isolated\n(1 hour)", "Short\n(1-6 h)", "Medium\n(7-24 h)", "Long\n(25-72 h)", "Extended\n(>72 h)"]
    counts = gaps["Gap_Length_Hours"].map(gap_category).value_counts().reindex(cats, fill_value=0)
    ax.bar(np.arange(len(cats)), counts.values, color="#AAA7D8", edgecolor="#3E3A84", width=0.62)
    for x, y in enumerate(counts.values):
        ax.text(x, y * 1.08 if y else 1, f"{int(y):,}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    if len(counts) and counts.max() > 0:
        ax.set_ylim(top=counts.max() * 1.7)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats, fontsize=10.8)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Number of gaps", fontsize=13.2)
    ax.set_xlabel("Gap duration", fontsize=13.2)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_seasonal(ax, long_df, pollutants):
    set_panel_background(ax, PANEL_BACKGROUNDS["seasonal"])
    colors = {"PM2.5": "#2D83D5", "PM10": "#FF7F0E", "OZONE": "#2CA02C"}
    month_order = [month for _, month, _ in AUSTRALIAN_SEASON_MONTHS]
    month_labels = [label for _, _, label in AUSTRALIAN_SEASON_MONTHS]
    month_seasons = [season for season, _, _ in AUSTRALIAN_SEASON_MONTHS]
    x = np.arange(len(month_order))
    monthly_site = (
        long_df.loc[long_df["Pollutant"].isin(pollutants)]
        .groupby(["Pollutant", "Site", "Month"], as_index=False)
        .agg(N_Total=("Is_Missing", "size"), N_Missing=("Is_Missing", "sum"))
    )
    monthly_site["Missingness_Percent"] = 100 * monthly_site["N_Missing"] / monthly_site["N_Total"]
    for pollutant in pollutants:
        pivot = monthly_site.loc[monthly_site["Pollutant"].eq(pollutant)].pivot_table(
            index="Month", columns="Site", values="Missingness_Percent"
        )
        mean = pivot.mean(axis=1).reindex(month_order)
        low = pivot.min(axis=1).reindex(month_order)
        high = pivot.max(axis=1).reindex(month_order)
        ax.fill_between(x, low, high, color=colors[pollutant], alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=colors[pollutant], marker="o", linewidth=1.5, markersize=3, label=pollutant_label(pollutant))
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, fontsize=11.9)
    for label, season in zip(ax.get_xticklabels(), month_seasons):
        label.set_color(SEASON_COLOURS[season])
        label.set_fontweight("normal")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Month", fontsize=13.2)
    ax.set_ylabel("Missing observations (%)", fontsize=13.2)
    ax.set_ylim(0, 30)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    pollutant_legend = ax.legend(
        fontsize=10.4, frameon=False, ncol=3,
        loc="upper right",
        borderaxespad=0.6, handlelength=1.56, handletextpad=0.45, columnspacing=0.9,
    )
    ax.add_artist(pollutant_legend)
    season_handles = [
        Patch(facecolor=colour, edgecolor=colour, label=SEASON_LABELS[season])
        for season, colour in SEASON_COLOURS.items()
    ]
    ax.legend(
        handles=season_handles,
        title="Australian season",
        fontsize=7.4,
        title_fontsize=8.2,
        frameon=True,
        facecolor="none",
        framealpha=0.0,
        edgecolor="none",
        ncol=1,
        loc="upper left",
        borderpad=0.45,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_hourly(ax, long_df, pollutants):
    set_panel_background(ax, PANEL_BACKGROUNDS["hourly"])
    colors = {"PM2.5": "#2D83D5", "PM10": "#FF7F0E", "OZONE": "#2CA02C"}
    hourly_data = long_df.loc[long_df["Pollutant"].isin(pollutants)].copy()
    hourly_data["Hour"] = hourly_data["DateTime"].dt.hour
    hourly_site = (
        hourly_data
        .groupby(["Pollutant", "Site", "Hour"], as_index=False)
        .agg(N_Total=("Is_Missing", "size"), N_Missing=("Is_Missing", "sum"))
    )
    hourly_site["Missingness_Percent"] = 100 * hourly_site["N_Missing"] / hourly_site["N_Total"]
    for pollutant in pollutants:
        pivot = hourly_site.loc[hourly_site["Pollutant"].eq(pollutant)].pivot_table(
            index="Hour", columns="Site", values="Missingness_Percent"
        )
        mean = pivot.mean(axis=1).reindex(range(24))
        low = pivot.min(axis=1).reindex(range(24))
        high = pivot.max(axis=1).reindex(range(24))
        x = np.arange(24)
        ax.fill_between(x, low, high, color=colors[pollutant], alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=colors[pollutant], marker="o", linewidth=1.5, markersize=3, label=pollutant_label(pollutant))
    tick_hours = [0, 4, 8, 12, 16, 20]
    ax.set_xticks(tick_hours)
    ax.set_xticklabels(
        [pd.Timestamp(hour=hour, year=2000, month=1, day=1).strftime("%-I %p") for hour in tick_hours],
        fontsize=11.9,
    )
    ax.set_xlim(0, 23)
    ax.set_xlabel("Hour of day", fontsize=13.2)
    ax.set_ylabel("Missing observations (%)", fontsize=13.2)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    ax.legend(
        fontsize=10.4, frameon=False, ncol=3,
        loc="upper right",
        borderaxespad=0.6, handlelength=1.56, handletextpad=0.45, columnspacing=0.9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_region_pollutant_box(ax, region_pollutant, region_colors=None):
    set_panel_background(ax, PANEL_BACKGROUNDS["regional"])
    region_order = (
        region_pollutant.groupby("Region")["Missingness_Percent"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [
        region_pollutant.loc[
            region_pollutant["Region"].eq(region), "Missingness_Percent"
        ].dropna().to_numpy()
        for region in region_order
    ]
    boxplot = ax.boxplot(
        data,
        positions=np.arange(len(region_order)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        boxprops={"facecolor": "#D8E2EA", "edgecolor": "#4A5568", "linewidth": 1.1},
        whiskerprops={"color": "#4A5568", "linewidth": 1.0},
        capprops={"color": "#4A5568", "linewidth": 1.0},
    )
    if region_colors:
        for box, region_name in zip(boxplot["boxes"], region_order):
            box.set_facecolor(region_colors.get(canon(region_name), "#D8E2EA"))
            box.set_alpha(0.42)
    pollutant_names = sorted(region_pollutant["Pollutant"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(pollutant_names), 3)))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
    colors = {p: palette[i % len(palette)] for i, p in enumerate(pollutant_names)}
    marker_map = {p: markers[i % len(markers)] for i, p in enumerate(pollutant_names)}
    for index, region_name in enumerate(region_order):
        part = region_pollutant.loc[region_pollutant["Region"].eq(region_name)]
        for _, row in part.iterrows():
            ax.scatter(
                index,
                row["Missingness_Percent"],
                s=62,
                marker=marker_map[row["Pollutant"]],
                c=[colors[row["Pollutant"]]],
                alpha=0.58,
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
    handles = [
        Line2D(
            [0], [0], marker=marker_map[p], color="none",
            markerfacecolor=colors[p], markeredgecolor="black",
            markeredgewidth=0.8, markersize=11.7, label=p,
        )
        for p in pollutant_names
    ]
    ax.legend(
        handles=handles,
        title="Pollutant",
        frameon=True,
        facecolor="white",
        framealpha=0.90,
        ncol=4,
        loc="upper right",
        fontsize=11.7,
        title_fontsize=13,
        borderpad=0.7,
        handletextpad=0.55,
        columnspacing=1.2,
    )
    ax.set_ylabel("Missingness (%)", fontsize=13.2)
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(region_order)))
    region_labels = ["\n".join(str(region).rsplit(" ", 1)) for region in region_order]
    ax.set_xticklabels(region_labels, rotation=0, ha="center", fontsize=10.9)
    if region_colors:
        for label, region_name in zip(ax.get_xticklabels(), region_order):
            label.set_color(region_colors.get(canon(region_name), "#333333"))
            label.set_fontweight("normal")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure(args):
    pollutants = [p.strip() for p in args.pollutants.split(",") if p.strip()]
    long_df, gaps = audit.load_missingness_records(args.inputs_dir, args.mapping, pollutants, args.include_unknown)
    if long_df.empty:
        raise RuntimeError("No missingness records loaded.")
    site_pollutant, region_pollutant, _, _ = audit.build_summary_tables(long_df, args.output_dir)
    full_region_pollutant_csv = DEFAULT_AUDIT_DIR / "missingness_by_region_pollutant.csv"
    if full_region_pollutant_csv.exists():
        region_pollutant = pd.read_csv(full_region_pollutant_csv)
    region_order = (
        region_pollutant.groupby("Region")["Missingness_Percent"]
        .median().sort_values(ascending=False).index.tolist()
    )
    region_colors = {
        canon(region): REGION_PALETTE[i % len(REGION_PALETTE)]
        for i, region in enumerate(region_order)
    }
    stations = load_station_metadata(args, long_df["Site"].unique())
    station_order = (
        site_pollutant.groupby("Site")["Missingness_Percent"].mean().sort_values(ascending=False).index.tolist()
    )
    dates, availability = make_availability_matrix(long_df, pollutants, station_order)

    fig = plt.figure(figsize=(18.0, 12.6), constrained_layout=False)
    grid = fig.add_gridspec(
        3, 3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 0.82, 0.82],
        wspace=0.30,
        hspace=0.46,
    )
    fig.suptitle(
        "Figure 1. Spatial, temporal and gap-duration characteristics of missing air-quality observations\nused in the AQUISTIL study",
        fontsize=16,
        fontweight="bold",
        color="#08245C",
        y=0.98,
    )

    ax_map = fig.add_subplot(grid[0, 0])
    draw_map(ax_map, stations, args.boundary_file, args.satellite_cache, region_colors)
    set_above_title(
        ax_map,
        "(a) Monitoring-station network",
        fontsize=13.0,
        color="#111111",
        width=24,
    )

    ax_matrix = fig.add_subplot(grid[0, 1:])
    draw_availability_matrix(ax_matrix, dates, availability)
    set_above_title(
        ax_matrix,
        f"(b) Data availability matrix ({dates.min().year}-{dates.max().year})",
        fontsize=15.5,
        width=90,
    )

    ax_gap = fig.add_subplot(grid[1, 0])
    draw_gap_distribution(ax_gap, gaps)
    set_above_title(
        ax_gap,
        "(c) Natural gap-duration distribution\n(all stations & pollutants)",
        fontsize=12.4,
        width=100,
    )

    ax_region = fig.add_subplot(grid[1, 1:])
    draw_region_pollutant_box(ax_region, region_pollutant, region_colors)
    set_above_title(ax_region, "(d) Regional missingness", fontsize=14.5, width=70)

    bar_axes = draw_missingness_bars(
        fig,
        grid[2, 0],
        site_pollutant,
        pollutants,
        axis_title_fontsize=10.8,
    )
    for ax in bar_axes:
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", labelsize=7.8)
        ax.xaxis.label.set_size(9.5)
        ax.yaxis.label.set_size(9.5)
    bar_axes[0].text(
        0.0,
        1.13,
        "(e) Percentage of missing observations by station",
        transform=bar_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=12.4,
        fontweight="bold",
        color="#111111",
    )

    ax_hourly = fig.add_subplot(grid[2, 1])
    draw_hourly(ax_hourly, long_df, pollutants)
    set_above_title(ax_hourly, "(f) Hourly missingness", fontsize=13.2, width=70)

    ax_season = fig.add_subplot(grid[2, 2])
    draw_seasonal(ax_season, long_df, pollutants)
    set_above_title(ax_season, "(g) Seasonal missingness", fontsize=13.2, width=70)
    data = {
        "availability": availability,
        "dates": dates,
        "gaps": gaps,
        "long_df": long_df,
        "pollutants": pollutants,
        "site_pollutant": site_pollutant,
        "region_pollutant": region_pollutant,
        "region_colors": region_colors,
        "stations": stations,
    }
    return fig, data


def save_figure(fig, path_base, transparent=False):
    fig.savefig(
        path_base.with_suffix(".png"), dpi=300, bbox_inches="tight",
        transparent=transparent,
    )
    fig.savefig(
        path_base.with_suffix(".pdf"), bbox_inches="tight",
        transparent=transparent,
    )
    plt.close(fig)


def save_component_figures(args, data):
    component_dir = args.output_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    draw_map(ax, data["stations"], args.boundary_file, args.satellite_cache, data["region_colors"])
    set_above_title(
        ax,
        "(a) Monitoring-station network",
        fontsize=15.0,
        color="#111111",
        width=70,
    )
    save_figure(fig, component_dir / "a_monitoring_station_network")

    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    draw_availability_matrix(ax, data["dates"], data["availability"])
    set_above_title(
        ax,
        f"(b) Data availability matrix ({data['dates'].min().year}-{data['dates'].max().year})",
        fontsize=15.5,
        width=90,
    )
    save_figure(fig, component_dir / "c_data_availability_matrix")

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    draw_gap_distribution(ax, data["gaps"])
    set_above_title(
        ax,
        "(c) Natural gap-duration distribution\n(all stations & pollutants)",
        fontsize=12.6,
        width=100,
    )
    save_figure(fig, component_dir / "d_gap_duration_distribution")

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    draw_region_pollutant_box(ax, data["region_pollutant"], data["region_colors"])
    set_above_title(ax, "(d) Regional missingness", fontsize=15.0, width=70)
    save_figure(fig, component_dir / "b_regional_missingness")

    fig = plt.figure(figsize=(10.5, 5.4))
    spec = fig.add_gridspec(1, 1)[0]
    bar_axes = draw_missingness_bars(
        fig,
        spec,
        data["site_pollutant"],
        data["pollutants"],
        axis_title_fontsize=14.52,
    )
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1)
    fig.text(
        0.5,
        0.985,
        "(e) Percentage of missing observations by station",
        ha="center",
        va="top",
        fontsize=15.0,
        fontweight="bold",
    )
    save_figure(
        fig,
        component_dir / "c_missing_observations_by_station",
        transparent=False,
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    draw_hourly(ax, data["long_df"], data["pollutants"])
    set_above_title(ax, "(f) Hourly missingness", fontsize=13.0, width=70)
    save_figure(fig, component_dir / "f_hourly_missingness_pattern")

    fig, ax = plt.subplots(figsize=(5.7, 4.6))
    draw_seasonal(ax, data["long_df"], data["pollutants"])
    set_above_title(ax, "(g) Seasonal missingness", fontsize=13.0, width=70)
    save_figure(fig, component_dir / "e_seasonal_missingness_pattern")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--station-info-file", type=Path, default=DEFAULT_STATION_INFO)
    parser.add_argument("--station-sheet", default="SiteDetails")
    parser.add_argument("--boundary-file", type=Path, default=DEFAULT_BOUNDARY_FILE)
    parser.add_argument(
        "--satellite-cache",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "satellite_tile_cache",
        help="Directory used to cache Esri World Imagery tiles.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pollutants", default=",".join(DEFAULT_POLLUTANTS))
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument(
        "--excluded-region-keywords",
        nargs="*",
        default=study_map.DEFAULT_EXCLUDED_REGION_KEYWORDS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, data = build_figure(args)
    png = args.output_dir / "fig1_missingness_characteristics.png"
    pdf = args.output_dir / "fig1_missingness_characteristics.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    save_component_figures(args, data)
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    print(f"Components: {args.output_dir / 'components'}")
    print(f"Stations: {data['stations']['Site'].nunique()}")
    print(f"Records: {len(data['long_df']):,}")
    print(f"Gaps: {len(data['gaps']):,}")


if __name__ == "__main__":
    main()
