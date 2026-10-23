"""Fixed settings of WP W1 (surface water v4). Everything a result depends on lives here, so
the pre-registration can hash it.

Units: volumes in 10^8 m3, discharge in m3/s (cms), dates in calendar months.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from common.paths import DATAPREP, L0_NEW, REPO  # noqa: E402

WP = "W1"
PRODUCT_BASE = "surface_water_v4"

# ----------------------------------------------------------------------------- inputs (L0)
YEARBOOK_DIR = L0_NEW / "wra_hydro_yearbook"
YEARBOOK_YEARS = tuple(range(2012, 2026))
JIJI_CSV = REPO / "chou-shui-data" / "data" / "water" / "jiji_weir_annual_supply.csv"
JIJI_HTML = L0_NEW / "jiji_weir_pages" / "annual.html"
BRIEFING_110_PDF = L0_NEW / "wra_reports_misc" / "wra_cy_subsidence_results_110.pdf"
BRIEFING_110_PAGE = 19                       # "1~5月供水量" table, slide 19
MOA_691_CSV = L0_NEW / "moa_irrigation_stats" / "moa_actual_691.csv"
V3_NPZ = REPO / "results" / "twin" / "surface_water_v3.npz"
TWIN_GRID_NPZ = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"
WELL_META = REPO / "AMP_V2" / "data" / "fan_stations.parquet"
WELL_DIR = REPO / "AMP_V2" / "data" / "wells"
RAIN_CSV = REPO / "chou-shui-data" / "data" / "rf_timeseries.csv"

# ----------------------------------------------------------------------------- gauges
BASIN = "1510"                                # Choushui basin number in the yearbook
GAUGE_NAMES = {"H049": "內茅埔", "H050": "延平橋", "H057": "彰雲橋", "H063": "玉峰橋",
               "H064": "水里", "H071": "溪州大橋", "H075": "寶石橋", "H079": "龍門橋",
               "H083": "南雲大橋"}
#: Inflow composite I(day) = 寶石橋 + 內茅埔 + 水里 (plan W1 step 2).
INFLOW_GAUGES = ("H075", "H049", "H064")
GAP_FILL_DAYS = 15                            # linear interpolation of gaps up to 15 days
DONOR_RATIO_TOL = 0.5                         # annual ratio > 50 % off its median -> donor
#: Explicit per-year exclusions (plan W1 step 1): station -> list of (year, months or None).
#: 玉峰橋 H063 is not in the composite; its 2024-25 exclusion matters only for reporting.
QC_EXCLUDE = {"H063": [(2024, None), (2025, None)], "H049": [(2024, (8,))]}
PRINTED_MONTHLY_TOL = 0.05                    # report station-months off by more than 5 %

# ----------------------------------------------------------------------------- diversion
#: Rotation calendar: month -> (crop, relative daily-rate weight); absent months get no
#: water unless the winter weight is positive (Dec, Jan).
CALENDAR = {2: (1, 1.5), 3: (1, 1.5), 4: (1, 1.0), 5: (1, 1.0), 6: (1, 0.5),
            7: (2, 1.5), 8: (2, 1.0), 9: (2, 1.0), 10: (2, 1.0), 11: (2, 0.5)}
CAL_DAYS = {2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30}
SHARE_GRID = tuple(round(0.10 + 0.05 * k, 2) for k in range(12))   # 0.10 .. 0.65
WINTER_GRID = (0.0, 0.5, 1.0)
DISTRICTS = ("chc", "yl")
DISTRICT_NAME = {"chc": "changhua", "yl": "yunlin"}
JIJI_COL = {"chc": "changhua_irrigation_1e8_m3", "yl": "yunlin_irrigation_1e8_m3"}

# ----------------------------------------------------------------------------- variants
#: Pre-origin variant (U3): beta, a_chc, a_yl and every fitted QC parameter see 2012-2022
#: only; the Denton benchmark stops at 2022. Full variant: 2012-2025 (U4: recorded arm only).
VARIANTS = {
    "preorigin": {"years": tuple(range(2012, 2023)), "fit_window_end": "2022-12"},
    "full": {"years": tuple(range(2012, 2026)), "fit_window_end": "2025-12"},
}
BENCH_YEARS = tuple(range(2015, 2022))        # Jan-May held-out benchmark (WRA 110 slide 19)

# ----------------------------------------------------------------------------- validation
BOOT_B = 10000
BOOT_SEED = 20261012
BOOT_LEVEL = 0.90

# ----------------------------------------------------------------------------- state space
#: Observation errors (1 sigma, 10^8 m3 unless noted), fixed before any fit:
#: annual totals are printed to 0.01 (rounding sd 0.003) -> 0.01 allows small accounting
#: noise; Jan-May totals are printed to 0.0001 but are a different report (scope or timing
#: mismatch) -> 3 % of the value, floor 0.03; crop-1 share of the MOA office totals (whole
#: irrigation office, all sources, 2012-2015 only) -> share sd 0.10 (weak).
SS_SIGMA_ANNUAL = 0.01
SS_SIGMA_JANMAY_REL = 0.03
SS_SIGMA_JANMAY_MIN = 0.03
SS_SIGMA_CROPSHARE = 0.10
SS_RHO_GRID = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
SS_SIGMA_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.45)
SS_N_DRAWS = 200
SS_SEED = 1000
SS_ESS_THIN = 5                               # elliptical slice sampling: keep every 5th
SS_ESS_BURN = 200

# ----------------------------------------------------------------------------- products
def product_dir_base(variant: str) -> Path:
    return DATAPREP / PRODUCT_BASE / variant



