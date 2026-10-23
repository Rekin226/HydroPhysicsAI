"""Fixed paths and constants of WP X3 (public well depths and the pumping split by layer)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGING = REPO / "dataprep_staging"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from common.paths import DATAPREP, L0_NEW  # noqa: E402

PKG = STAGING / "x3_well_depths"
REPORT_DIR = DATAPREP / "reports" / "X3_well_depths"
PRODUCT_ROOT = "x3_well_depths"          # results/dataprep/x3_well_depths/<name>/v<N>

FETCH_DAY = "2026-10-20"                 # UTC date of the X3 fetches (local 2026-10-21)
L0 = {
    "thesis": L0_NEW / "ntu_thesis_lin2012_choushui_pumping" / FETCH_DAY / "ntu-101-1.pdf",
    "wra_well_mgmt": L0_NEW / "wra_lib_x3_pages_b" / FETCH_DAY / "marc74288.pdf",
    "wra_lib_82451": L0_NEW / "wra_lib_x3_pages" / FETCH_DAY / "books_82451.html",
    "moa_dir": L0_NEW / "moa_ia_well_depth_tables" / FETCH_DAY,
    "cy53001": L0_NEW / "wra_reports_misc" / "cy53001.pdf",
}
#: MOA agricultural statistics yearbook pages (file -> ROC yearbook year = data year).
MOA_PAGES = {
    "17892_316.pdf": 96, "20588_342.pdf": 97, "2445260_098308.pdf": 98,
    "2445218_099312.pdf": 99, "2447157_2447157_3.pdf": 100, "2501682_23-3_102-IA.pdf": 102,
    "2503263_13-3.pdf": 103,
}
MOA_DEPTH_CLASSES = ["lt60", "60_80", "80_100", "100_120", "120_140", "ge140"]
#: depth-class edges (m) of the MOA classes, for mapping to aquifers
MOA_EDGES = [(0, 60), (60, 80), (80, 100), (100, 120), (120, 140), (140, 200)]

# AMP_V2 inputs (read-only vendor cache) and repository products (read-only)
STATIONS = REPO / "AMP_V2" / "data" / "fan_stations.parquet"
STATION_COLS = ["sid", "CountyName", "TownName", "GroundwaterLayerCode", "WellDepth",
                "FinishDepth", "LocationByTWD97"]
CENSUS_L1 = DATAPREP / "pump_electricity" / "census_L1" / "v3"
CENSUS_COLS = ["county", "town", "x_twd97", "y_twd97", "cell", "in_grid", "kept", "pump_hp",
               "purpose_class", "tube_dia_raw", "install_month", "meter_duty_2012_2022",
               "hp_share"]
ELEC_PREORIGIN = DATAPREP / "pump_electricity" / "preorigin" / "v3"
ZONE_COUNTY = DATAPREP / "zone_county_map" / "v2"
RUN = {"R0": REPO / "results" / "twin_runs" / "stage3_datum_gate_cwa",
       "R1c": REPO / "results" / "twin_runs" / "stage3_datum_gate_ccz"}

COUNTY_ZH = {"changhua": "彰化縣", "yunlin": "雲林縣"}
YUNLIN_TOWNS = ["二崙", "口湖", "土庫", "大埤", "元長", "斗六", "斗南", "水林", "北港", "古坑",
                "台西", "四湖", "西螺", "東勢", "林內", "虎尾", "崙背", "麥寮", "莿桐", "褒忠"]
CHANGHUA_TOWNS = ["二水", "二林", "大村", "大城", "北斗", "永靖", "田中", "田尾", "竹塘", "伸港",
                  "秀水", "和美", "社頭", "芬園", "花壇", "芳苑", "員林", "埔心", "埔鹽", "埤頭",
                  "鹿港", "溪州", "溪湖", "彰化", "福興", "線西"]
#: fan zones by easting (km), as in the I1 diagnostics: distal (west), mid, proximal (east)
ZONE_EDGES_KM = (185.0, 205.0)
ORIGIN = "2022-12"
DESIGN_START = "2012-01"


def town_key(name: str) -> str:
    """Township name without its suffix (鄉/鎮/市), 臺 -> 台, for joins across sources."""
    s = str(name).strip().replace("臺", "台")
    if s.endswith(("鄉", "鎮")) or (s.endswith("市") and len(s) > 2):
        s = s[:-1]
    return s


def county_of_town(key: str) -> str | None:
    if key in YUNLIN_TOWNS:
        return "yunlin"
    if key in CHANGHUA_TOWNS:
        return "changhua"
    return None


def zone_of_x(x_m: float) -> str:
    xk = x_m / 1000.0
    return "distal" if xk < ZONE_EDGES_KM[0] else ("mid" if xk < ZONE_EDGES_KM[1] else "proximal")
