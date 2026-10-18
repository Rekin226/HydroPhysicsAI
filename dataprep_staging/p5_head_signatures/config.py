"""WP P5 configuration (plan WP P5, 2.4-2.5, 4.1, 4.4-4.5; task update 2026-10-12).

Every threshold comes from the plan text (WP P5 validation and acceptance), from the
head-signature pilot (``_pilots/head_pumping_signatures``: index definitions, month
eligibility, the epoch check, the shallow-layer CV design), from WP H1a (barometric response,
day flags, O1 coherence rule) or from WP A1 (seed convention, coverage rule). Nothing here was
chosen on data dated 2023-01 or later (U7), and no threshold was set after looking at a P5
result: the values were written before the extraction and before the pre-registration
``P5.signatures.v1``.

No path here names a user or a scratchpad (CLAUDE.md Rule 1).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
WP = "P5"
PRODUCT = "head_signatures"

# ----------------------------------------------------------------------------- inputs
L1_DIR = REPO / "results/dataprep/heads/L1_clean/v3"             # H1a product for use
C0_PRE_DIR = REPO / "results/dataprep/pump_electricity/preorigin/v3"       # C0 pump arrays
C0_CENSUS_DIR = REPO / "results/dataprep/pump_electricity/census_L1/v3"    # C0 pump table
ET0_DIR = REPO / "results/dataprep/et0/v1"                       # W2
SW_PRE_DIR = REPO / "results/dataprep/surface_water_v4/preorigin/v2"      # W1
UNITS_DIR = REPO / "results/dataprep/validation_units/v3"        # A1 units (158 model wells)
PILOT_DIR = REPO / "results/dataprep/_pilots/head_pumping_signatures"
#: L0, read-only: the CWA-CODIS-rebuilt daily rain of the 26 gauges (evidence doc section 11)
RAIN_CSV = REPO / "chou-shui-data/data/rf_timeseries.csv"
RAIN_STATIONS = REPO / "chou-shui-data/data/rf_stations.csv"

# ----------------------------------------------------------------------------- outputs
WORK = Path(os.environ.get("P5_WORK", REPO / "results/dataprep/head_signatures/_work"))
EVIDENCE = Path(os.environ.get("P5_EVIDENCE", REPO / "results/dataprep/reports/P5_evidence"))

# ----------------------------------------------------------------------------- time
ORIGIN = "2022-12"
PRE_MONTHS = ("2012-01", "2022-12")
POST_MONTHS = ("2023-01", "2026-09")
ALL_MONTHS = ("2012-01", "2026-09")
COUNTIES = {"彰化縣": "Changhua", "雲林縣": "Yunlin", "嘉義縣": "Chiayi"}

# ----------------------------------------------------------------------------- extractor
# Pilot definitions (well_features.py / well_features2.py), unchanged:
MONTH_MIN_HOURS = 24 * 10        # a month needs >= 10 days of hourly means
HARM_MIN_HOURS = 24 * 20         # harmonic fit and the 24-h profile need >= 20 days
DAY_MIN_HOURS = 20               # a day enters the daily statistics with >= 20 hourly means
DAILY_STATS_MIN_DAYS = 10        # daily-statistic medians need >= 10 such days in the month
EV_FRAC10_MIN = 0.8              # 30-min events only in months with > 80 % 10-min sampling
EV_THRESHOLDS = ((0.05, "5"), (0.20, "20"))
#: Day mask: H1a L1 v3 daily flag in {0, 1} (observed or barometric-corrected); samples must
#: also be clean in H1a's stage-1 screen (kind == 0). This replaces the pilot's own cleaning
#: (12 m outliers, flat dropouts, 1 m spikes), a declared deviation.
GOOD_DAY_FLAGS = (0, 1)

# ----------------------------------------------------------------------------- local kWh
LOCAL_RADII_M = (1000.0, 2000.0, 3000.0, 5000.0)   # pilot E_r1, E_r2, E_r3, E_r5
#: Pilot periodic-Theis kernel E_K0_<delta>: |K0(max(r, 30 m) sqrt(i) / delta)|, zero for
#: r > 4 delta + 500 m (local_kernel.py), delta = 1000 m.
K0_DELTA_M = 1000.0
K0_RMIN_M = 30.0
SECTOR_RADIUS_M = 3000.0         # pilot: 8 sectors of the 3 km disk
SECTORS_MIN = 7                  # pilot: census coverage around the well (>= 7 of 8 sectors)
KWH_MONTHS = ("2012-01", "2022-12")   # pre-origin window of every kWh statistic (U7)

# ----------------------------------------------------------------------------- tests
SEED = 2027                      # A1 seed convention for new P5 draws
N_PERM = 1000                    # feature-permutation null draws
N_BOOT = 2000                    # well-cluster bootstrap draws
REPRO_SPEARMAN_MIN = 0.95        # plan P5 acceptance (pilot: 0.968 dr_med, 0.955 S1c)
EPOCH_HOURLY = (2016, 2018)      # pilot epoch check: hourly logger years ...
EPOCH_10MIN = (2019, 2021)       # ... against 10-minute logger years
FOLD_PASS_MIN = 4                # plan: in at least 4 of 5 folds
N_FOLDS = 5
BLOCK_M = 5000.0                 # plan: 5 km blocks
PLACEBO_MIN_M = 15000.0          # plan: kWh more than 15 km away
PLACEBO_MAX_M = 25000.0          # outer edge of the distant annulus (cross-well placebo)
PRIMARY_TARGET = "E_r2"          # pilot cv_shallow: best single target, declared before results
PRIMARY_FEATURE = "A_S1c_strain"  # corrected S1 (strain admittance), barometrically corrected
SECONDARY_FEATURES = ("DN", "dr_med")
LAYERS_SHALLOW = ("1", "2")      # plan: the cross-well signal sits in layers 1-2
WELL_MIN_MONTHS = 24             # pilot: n_months >= 24 for a well-level index
WITHIN_MIN_MONTHS = 36           # pilot placebo.py: >= 36 months for a within-well correlation
FEATURE_LOG_FLOOR = 1e-4         # pilot: log10(clip(x, 1e-4)) for amplitude-type features
CLIM_MIN_YEARS = 2               # seasonal CV: a calendar month needs >= 2 years
P4_SEASONALITY_CUT = 0.75        # plan P4.3: "Changhua seasonal variation >= 0.75 of Yunlin"

# ----------------------------------------------------------------------------- transfer functions
TF_MEMORY_MONTHS = 36            # gamma block response truncated at 36 months
TF_MIN_MONTHS = 60               # a well needs >= 60 observed (flag 0/1) pre-origin months
TF_N_BOUNDS = (0.2, 6.0)         # gamma shape n
TF_A_BOUNDS = (0.2, 24.0)        # gamma scale a (months)

# ----------------------------------------------------------------------------- EEMD / EOF
EEMD_ENSEMBLE = 100
EEMD_NOISE_SD = 0.2              # added white noise, in units of the series sd (Wu & Huang 2009)
EEMD_MAX_IMFS = 6
EEMD_SIFT_ITERS = 10             # fixed sifting number (Wu & Huang 2009)
EEMD_MIN_MONTHS = 108            # >= 9 of the 11 pre-origin years observed (monthly series)
EOF_MIN_COVERAGE = 0.9           # wells with >= 90 % of the 131 monthly differences
EOF_N_MODES = 3
