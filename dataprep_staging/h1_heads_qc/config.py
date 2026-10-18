"""Frozen H1a QC configuration (pre-registered before the acceptance tests ran).

Every threshold below comes from the plan (WP H1, section 2.4/2.5), from the audit's
artefact detector (``_pilots/audit_2026-10-11/artefacts.py``), from the gap-fill pilot's
clean rule (``_pilots/head_gapfill_pilot/s1_build.py``, ``s5b_cleantruth.py``), from the
head-signature pilot (floor dropouts, 12 m outliers), or from A0's pre-origin rho(k). The
barometric design, the logger-epoch split and the se calibration were chosen on 2012-22 data
only (``dev_baro_methods.py``, ``dev_baro_epochs.py``; design-time evidence). Nothing here was
tuned on data dated 2023-01 or later (U7); the audit that proposed the jump thresholds viewed
records of both periods (disclosed in the pre-registration). The sha256 of this file is
recorded in the pre-registration and in the product manifest.
"""
from __future__ import annotations

ORIGIN_SPLIT = "2023-01-01"          # first post-origin day

# ---------------------------------------------------------------- sample screens
SENTINEL_ABS = 1000.0                # |v| >= 1000 (the -999998 blocks) -> flag 8
JUMP_M = 1.0                         # audit: |dv| > 1 m between consecutive samples
JUMP_MAX_DT_S = 70 * 60              # audit: consecutive samples <= 70 min apart
DEAD_TIME_S = 6 * 3600               # audit: skip jumps < 6 h after the previous event
PERSIST_H = (6.0, 48.0)              # audit: post-jump window for offset persistence
PERSIST_MIN_SAMPLES = 5
PERSIST_TOL = 0.3                    # |post - pre - j| < 0.3 |j| -> offset
REVERT_TOL = 0.3                     # |post - pre| < 0.3 |j| -> spike
SPIKE_MASK_FRAC = 0.5                # spike samples: |v - pre| > 0.5 |j| inside 6 h

PAIR_MAX_DAYS = 365                  # gap-fill pilot s5b: opposite jump within 365 days
PAIR_SIZE_TOL = 0.3                  # |j1 + j2| < 0.3 max(|j1|, |j2|)
STEP_MIN_M = 3.0                     # plan: unpaired steps >= 3 m
EPISODE_MIN_M = 1.0                  # plan: paired jumps >= 1 m

FLAT_RUN_MIN_H = 72.0                # audit: identical value >= 3 days -> stuck sensor
MAD_K = 15.0                         # plan / heads.py: 15 x MAD despike (pre-origin median)
ROLL_WINDOW_H = 15 * 24              # 15-day centred rolling median of hourly medians
FAR_DEV_M = 12.0                     # head-signature pilot: |dev| > 12 m from rolling median
DROPOUT_DEV_M = 3.0                  # dropout: > 3 m below the rolling median ...
DROPOUT_FLAT_N = {"10min": 6, "hourly": 2}   # ... inside a run of identical values
BOTTOM_MARGIN_M = 1.0                # below (WellElevation - WellDepth) - 1 m: impossible
TOP_MARGIN_M = 5.0                   # above WellElevation + 5 m: impossible (step sides)
DROPOUT_EPISODE_FLAT_SD = 0.02       # episode/spike segment this flat and >3 m below -> dropout

# ---------------------------------------------------------------- daily screen
DAY_MIN_HOURS = 12                   # C20: a day needs >= 12 h
MONTH_MIN_DAYS = 20                  # C20: a month needs >= 20 days
DAILY_ROLL_DAYS = 15                 # gap-fill clean rule: centred 15-day rolling median
DAILY_DEP_MIN_M = 3.0                # threshold max(3 m, 8 x MAD(daily diff) x sqrt(7))
DAILY_DEP_MAD_MULT = 8.0

# ---------------------------------------------------------------- sigma_obs (U2)
# A0 repair round, pre-origin rho(k) (results/dataprep/reports/A0_evidence/repair/reruns/
# rho_k_result.json, "preorigin" table): RMSE of a k-clean-day partial mean vs the full month.
RHO_K = {3: 0.8616494466712785, 5: 0.8041236004796088, 10: 0.6505993940198698,
         15: 0.4944701483871803, 20: 0.331143272089593}
SIGMA_FLOOR = 0.10                   # sigma_obs = max(0.10 m, rho(k)); complete month -> 0.10

# ---------------------------------------------------------------- barometric
BARO_LAGS_H = 48                     # regression deconvolution lags 0..48 h
BARO_DESIGN = "hourly_grouped"       # production design (baro.DESIGNS; dev_baro_methods.py)
BARO_EPOCHS = True                   # fit per logger epoch (dev_baro_epochs.py)
#: Calibration of the block-bootstrap se(A_K): within-epoch split halves give a median |z| of
#: 0.829 against 0.6745 expected (dev_baro_epochs.py, 2012-22, 191 wells), so se x 1.23.
BARO_SE_INFLATION = 1.23
BARO_DAY_MIN_HOURS = 20              # daily design: a day mean needs >= 20 clean hours
BARO_IRLS_ITERS = 4                  # Huber IRLS iterations
BARO_BLOCK_DAYS = 30                 # within-block harmonics projected out (Frisch-Waugh)
BARO_HARMONICS_DEG_H = {"S1": 15.0, "S2": 30.0, "S3": 45.0, "S4": 60.0,
                        "O1": 13.9430356, "M2": 28.9841042, "N2": 28.4397295,
                        "Q1": 13.3986609}
BARO_HUBER_C = 1.345
BARO_MIN_PAIRS = 2000
BARO_MIN_BLOCKS = 12
BARO_BOOT = 200                      # block bootstrap resamples for se(A_K)
BARO_SIG = 3.0                       # accept if |A_K| > 3 se (calibrated se)
BARO_RANGE = (-1.2, 1.2)             # physically plausible response (incl. absolute sensors)
BARO_SEED = 1000
HPA_TO_M = 100.0 / (1000.0 * 9.80665)   # 1 hPa of air pressure in metres of water

# ---------------------------------------------------------------- corrections / backing
COLOC_WINDOW_S = 2 * 3600            # sibling jump within +-2 h counts as simultaneous
COLOC_SIB_FRAC = 0.3                 # sibling jump > 0.3 |j| -> site-wide, not backed
META_DATE_DAYS = 7                   # metadata date within +-7 days of a step
CARRY_MAX_GAP_DAYS = 90              # carry a pre-origin step only across a record gap <= 90 d
NEIGH_MAX_KM = 25.0                  # neighbour regression (plan H2 neighbour rule)
NEIGH_MIN_R = 0.6
NEIGH_K = 4
NEIGH_MIN_OVERLAP_D = 365
NEIGH_RIDGE = 1e-3
NEIGH_SIG = 4.0                      # plan: residual above 4 calibrated sd -> flag 3
NEIGH_PLACEBOS = 200
NEIGH_SIZE_TOL = 0.5                 # |delta - j| < 0.5 |j|
NEIGH_REF_DAYS = 15                  # reference residual window before / after
NEIGH_STEP_DAYS = 60                 # step test windows

# ---------------------------------------------------------------- validation
PRESSURE_DROP_HPA = 20.0             # plan: no flagged episode on a >= 20 hPa pressure drop
PRESSURE_DROP_WINDOW_H = 24          # around the episode start / step
INJECT_REALIZATIONS = 10             # A1 family O, r = 0..9 (seed 1000*ord('O')+r)
INJECT_HOUR = 10                     # injected jumps at 10:00 of the generator's day
AUDIT_TOL_DAYS = 1.0                 # agreement: same well, within +-1 day
RECALL_MIN = 0.9
AUDIT_AGREE_MIN = 0.95

#: Known real events that must stay unflagged (plan WP H1 acceptance; audit section 1).
#: Windows fixed before the post-origin records were opened.
REAL_EVENTS = [
    {"sid": "09200321", "start": "2024-05-28", "end": "2024-06-04",
     "what": "post-season recovery about 1 m/day across a 3-day logger gap"},
    {"sid": "09190132", "start": "2024-05-28", "end": "2024-06-04",
     "what": "post-season recovery about 1 m/day across a 3-day logger gap"},
    {"sid": "09070311", "start": "2024-07-20", "end": "2024-08-03",
     "what": "typhoon recharge +4.9 m at the end of July 2024"},
]
#: Event kinds that count as an artefact flag for the real-event check.
ARTEFACT_KINDS = ("spike", "offset_episode", "dropout", "step", "step_carried", "flat_run",
                  "mad_outlier", "far_outlier", "dropout_flat", "below_well_bottom",
                  "daily_departure")
