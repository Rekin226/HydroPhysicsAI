"""Constants of the X5 energy-to-volume route (Q4), fixed before any registered computation.

Route 1 of the 2026-10-25 data audit: three estimators of the volume pumped per kWh of
census electricity, per county, and a stage-2 twin rung that fixes the twin's conversion.

* (a) WRA 水電比 volumes (2025 trial report, Tables 4-1.4..4-1.7, 4-3.19, 4-3.22) divided by
  census irrigation kWh, on the kept (twin) and the deduplicated bases;
* (b) pump physics per pump (rated horsepower, discharge-pipe diameter, lift);
* (c) the diurnal drawdown amplitude (AMP v1 of AMP_V2) inverted through the periodic Theis
  kernel of the duty-cycled kept irrigation pumps around a monitoring well. (c) uses no WRA
  volume and no electricity-to-volume ratio: electricity enters only as the time pattern and
  the relative size of each pump's pumping.

No data value is written here: WRA table cells are parsed from the report text at run time;
only declared physical constants and method settings are fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, L0_NEW, REPO, REPORTS_DIR  # noqa: E402

WP = "X5"
CODE_DIR = Path(__file__).resolve().parent
OUT = DATAPREP / "x5_energy_volume"
WORK = OUT / "work"
LOCK = DATAPREP / "preregistration" / ".x5.lock"

PREREG = {
    "step0_ab": "X5.energy_volume.step0_ab.v1",
    "synthetic": "X5.energy_volume.synthetic_c.v1",
    "test": "X5.energy_volume.c_vs_a.v1",
}
TEST_ID = "X5EV.c_vs_a_confined_2010_2011"
REPORT_DIR = {k: REPORTS_DIR / v for k, v in PREREG.items()}

# ------------------------------------------------------------------ inputs (read-only)
CENSUS_L1 = DATAPREP / "pump_electricity" / "census_L1" / "v3" / "census_pumps.parquet"
L0_PARQ = L0_NEW / "tpc_changhua_yunlin_chiayi_pingtung_2026" / "parquet"
WRA_TXT = REPO / "results" / "field_data_acquisition" / "wra_2025_trial_report_97847.txt"
PUMPING_PY = REPO / "hydrophysics" / "twin" / "pumping.py"
STATIONS = REPO / "AMP_V2" / "data" / "fan_stations.parquet"
LS_CACHE = REPO / "chou-shui-data" / "data" / "ls_cache"
AMP_WELLS = REPO / "AMP_V2" / "data" / "wells"
AMP_PY = REPO / "AMP_V2" / "amp.py"
GAFD_PY = REPO / "AMP_V2" / "gafd.py"
BE_CSV = DATAPREP / "heads" / "L1_clean" / "v3" / "barometric_efficiency.csv"
X3_CELL_SHARES = DATAPREP / "x3_well_depths" / "depth_layers" / "v1" / "cell_layer_shares.csv"
X3_COUNTY_SPLIT = DATAPREP / "x3_well_depths" / "depth_layers" / "v1" / "county_split_combined.csv"
GRID_NPZ = REPO / "results" / "twin_forward" / "datum_gate_innov.npz"
Q0_THETA = REPO / "results" / "twin_runs" / "stage3_datum_gate_q0" / "stage3_theta.json"

COUNTIES = ("changhua", "yunlin")
COUNTY_ZH = {"changhua": "彰化縣", "yunlin": "雲林縣"}
CLASSES = ("irrigation", "aquaculture", "livestock", "domestic", "industry", "other")
FORBIDDEN_COLS = ("戶名1", "用電地址2", "郵遞區號")

# ------------------------------------------------------------------ WRA 2025 report anchors
# The cells are parsed from WRA_TXT at run time (wra.py), never typed here. Table 4-1.4 /
# 4-1.6: 2011-24 (ROC 100-113) annual means by purpose and source; Table 4-1.5 / 4-1.7:
# Jan-May means of the same years; Tables 4-3.19 / 4-3.22: Jan-May irrigation groundwater
# pumping of 2023 (ROC 112) and 2024 (ROC 113). Units 億 m3 (1e8 m3).
WRA_TABLES = {
    "changhua": {"annual": "表 4-1.4 彰化歷年平均", "janmay": "表 4-1.5 彰化枯水期",
                 "recent": "表 4-3.19 彰化地區近 2 年"},
    "yunlin": {"annual": "表 4-1.6 雲林歷年平均", "janmay": "表 4-1.7 雲林枯水期",
               "recent": "表 4-3.22 雲林地區近 2 年"},
}
WRA_COLS = ("domestic", "industry", "irrigation", "aquaculture", "livestock", "agri_subtotal",
            "total")
WRA_YEARS = (2011, 2024)          # ROC 100-113
N_WRA_YEARS = 14
REF_YEARS = (2011, 2022)          # pre-2023 reference window of the scored comparison (U7)

# ------------------------------------------------------------------ the twin's conversion
ETA_TWIN = 0.5
HEAD_EXTRA_TWIN = 40.0
LIFT_MEDIAN = 6.65                # m, pumping.energy_to_volume docstring
RHO_G = 9800.0
J_PER_KWH = 3.6e6
M3_PER_KWH_PER_M = J_PER_KWH / RHO_G          # 367.35 m3 x m per kWh at eta 1
KW_PER_HP = 0.746

# ------------------------------------------------------------------ estimator (b)
B_V_PRIMARY = 1.5                 # m/s design velocity in the discharge pipe
B_V_RANGE = (1.0, 2.0)
B_HP_RANGE = (0.0, 50.0)          # (lo, hi] horsepower kept for (b)
B_DIA_RANGE = (1.5, 12.0)         # inches kept for (b)
B2_ETA = (0.4, 0.5, 0.6)
B2_HDYN = (5.0, 10.0, 20.0)       # m dynamic head on top of the static lift

# ------------------------------------------------------------------ estimator (c)
T_LIT = (58.0, 6034.0)            # m2/day, Liu et al. (2002) Choushui pumping tests
T_PRIMARY = (T_LIT[0] * T_LIT[1]) ** 0.5     # geometric centre, ~592 m2/day
POROSITY = 0.3
BETA_W = 4.4e-10                  # 1/Pa, water compressibility
AQ_THICKNESS = 50.0               # m, declared thickness of the screened aquifer
SY_UNCONFINED = 0.1               # reported-only storage of unconfined wells
DUTY_H = 9.3                      # h/day on pumping days (AMP-G network median, AMP_V2 README)
DUTY_RANGE_H = (4.0, 12.0)
R_MIN = 30.0                      # m, distance floor (pump coordinate precision)
R_MAX = 4000.0                    # m, pumps beyond carry |K0| < ~5 % of the near field
R_COUNT = 2000.0                  # m, radius of the minimum-pump-count rule
MIN_PUMPS = 20                    # kept irrigation pumps within R_COUNT
MIN_VALID_FRAC = 0.8              # valid hours per month
MIN_JM_MONTHS = 3                 # valid Jan-May months per well
FLAT_HOURS = 168                  # runs of identical hourly values this long are removed (1-cm sensors)
K1P1_OVER_O1 = 1.49               # |K1| (1.41) and |P1| (0.47) over |O1| of the tidal potential
ATM_S1_M = 0.002                  # m, declared atmospheric S1 floor
AMP_BW = 0.05
AMP_TREND_CUT = 0.1
AMP_SMOOTH_DAYS = 30.0
TEST_WINDOW = ("2010-01-01", "2012-01-01")
TEST_SCOPE = "heads:2010-01..2011-12"
JANMAY = (1, 2, 3, 4, 5)
RATIO_BAND = 1.5                  # pass if 1/1.5 <= median ratio <= 1.5
SYNTH_C_TRUE = (6.0, 3.0)         # m3/kWh used to build the synthetic heads
SYNTH_BAND = 0.25                 # +-25 %
SEED_SYNTH = 20261025
SEED_BOOT = 20261026
