"""Fixed design constants of WP X5 (route 5: synthetic DiD on the 2019 Yunlin shutdown).

Everything here is part of the declaration ``X5.shutdown_sdid.v1``; the declaration hashes
this file. Change nothing after registration: a changed design needs a new id.
"""
from __future__ import annotations

from pathlib import Path

WP = "X5"
PREREG_ID = "X5.shutdown_sdid.v1"
CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parents[1]
OUT_DIR = REPO / "results/dataprep/x5_shutdown_did"
REPORT_DIR = REPO / "results/dataprep/reports" / PREREG_ID

# ---------------------------------------------------------------- inputs (read-only)
LEVELING_DATA_DIR = REPO / "chou-shui-data/data"          # hydrophysics.twin.leveling.PANEL
WELLS_LOCATED = REPO / "results/dataprep/public_wells/v2/wells_located.csv"
INTERVENTIONS = REPO / "results/dataprep/natural_experiments/v1/interventions.csv"
PUBLIC_WELL_JANMAY = REPO / "results/dataprep/natural_experiments/v1/public_well_janmay.csv"
ZONE_MAP = REPO / "results/dataprep/zone_county_map/v2/zone_county_map.npz"
CENSUS_CELLS = REPO / "results/dataprep/pump_electricity/full/v3/cells_total.npz"
CENSUS_VAR = "value_all"
RINGS = REPO / "results/dataprep/rings_bridged/v3/ring_bands_monthly_bridged.parquet"

# ---------------------------------------------------------------- outcome and sign
# S = subsidence rate, cm/yr, POSITIVE = SINKING: S = -(elev(t1) - elev(t0)) / dt * 100.
# ATT < 0 means LESS sinking at treated units after the event (the expected direction).
SIGN = "S = -(elevation change)/dt in cm/yr; positive = sinking; ATT < 0 = less sinking"
SURVEY_TARGET_DOY = 135            # 15 May: one survey per benchmark-year, nearest to it
DT_MIN, DT_MAX = 0.6, 1.6          # admissible interval length, years
GROSS_RATE = 20.0                  # cm/yr: a unit with any |S| above it is dropped (datum
#                                    artefact; the fan's fastest benchmarks sink < 10 cm/yr)
ONGRID_M = 1000.0                  # benchmark within 1 km of an active twin-cell centroid

# ---------------------------------------------------------------- periods
YEARS = list(range(2009, 2022))    # interval ending in the May survey of year y
PRE = list(range(2009, 2019))      # 2009..2018
POST = [2019, 2020, 2021]          # primary post window
POST_ALT = [2019, 2020]            # declared alternative: drops the 2021 interval (TWC
#                                    drought reactivation Jan-May 2021 at ~65 % of pre dose)
LEAD_PRE = list(range(2009, 2018))  # lead placebo: fake event 2018, real post dropped
LEAD_POST = [2018]
PAIR_PRE = list(range(2009, 2014))  # alternative period pair 2009-13 vs 2015-19
PAIR_POST = list(range(2015, 2020))

# ---------------------------------------------------------------- treatment geometry
# Primary arm T2: within R of a Yunlin Taiwan Water well of the 2022 drought-well inventory
# (the located proxy of the wells that stopped in 2019 and were switched back on in 2021).
TWC_AGENCY = "Taiwan Water Corporation (台水)"
R_TREAT = 3000.0
R_BUFFER = 6000.0                  # 3-6 km annulus excluded from donors
R_ALTS = {"R2km": (2000.0, 5000.0), "R5km": (5000.0, 8000.0)}   # reported, not gating
# Alternative arm T1: the B02 townships (quote: no Jan-May TWC pumping since 2019).
T1_TOWNS = ("虎尾鎮", "土庫鎮", "元長鄉")
T1_DONOR_GAP = 3000.0              # donors farther than 3 km from any T1 benchmark
# Golden Corridor 2020 extension townships (M11): 虎尾, 土庫, 元長, 北港.
GC_TOWNS = ("虎尾鎮", "土庫鎮", "元長鄉", "北港鎮")

# ---------------------------------------------------------------- inference
BLOCK_M = 5000.0                   # placebo-in-space blocks (5 km tiles of donor units)
N_PLACEBO = 500
SEED = 20261025
P_SUCCESS = 0.10                   # one-sided block-placebo p for the primary ATT
P_LEAD_NULL = 0.10                 # lead placebo is null if two-sided p >= 0.10
Z90 = 1.6448536269514722

# ---------------------------------------------------------------- secondary (long difference)
LD_RADIUS = 3000.0
LD_EARLY, LD_LATE = (2010, 2014), (2017, 2021)       # the audit's pair (seen: design-time)
LD_ALT_EARLY, LD_ALT_LATE = (2009, 2013), (2015, 2019)
LD_MIN_INTERVALS = 3               # per window
LD_BOOT = 1000

# ---------------------------------------------------------------- rings (reported only)
RING_SERIES = "primary"
RING_BANDS = ("band0", "band1", "band2", "total")    # 0-100, 100-200, 200-bottom m, total
RING_YEARS = list(range(2015, 2022))                 # May(y-1) -> May(y); 2014-05 first
RING_PRE = [2015, 2016, 2017, 2018]
RING_POST = [2019, 2020, 2021]
RING_MAY_TOL_MONTHS = 1

# ---------------------------------------------------------------- synthetic check
SYN_N_PANELS = 100
SYN_N_PLACEBO = 200
SYN_DELTA = -0.5                    # cm/yr rate change from 2019-01-01 (less sinking)
SYN_DELTA_SMALL = -0.25             # power only
SYN_SEED = 7_2026_1025
SYN_SIGMA_SURVEY = 0.5              # cm, per survey (independent, shared by two intervals)
SYN_FACTOR_SD = 1.0                 # cm/yr, common year shocks
SYN_BLOCK_RW_SD = 0.3               # cm/yr, 5 km block shocks
SYN_COUNTY_SD = 0.3                 # cm/yr, Yunlin-only shocks (stress scenario B only)
SYN_BIAS_TOL = 0.25                 # |mean tau - truth| <= 0.25 |truth|
SYN_COVER = (0.80, 0.98)            # placebo-interval 90 % coverage
SYN_SIZE_MAX = 0.15                 # false-positive rate of p < 0.10 under no effect
