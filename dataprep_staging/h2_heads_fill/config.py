"""WP H2 configuration: head gap filling with calibrated uncertainty (plan WP H2, 2.4-2.5, 4.4).

Every constant below comes from the plan text (WP H2 technique steps 1-7 and validation), from
the head gap-fill pilot (``results/dataprep/_pilots/head_gapfill_pilot/lib.py``, design-time
evidence), from A1's harness (seeds, coverage rule gate-3) or from A0's pre-origin rho(k). The
values that the design stage of this WP derives (gap-class sd factors, bridge sd classes, the
per-class fall-backs) are not here: ``s2_design.py`` writes them to the work directory and the
pre-registration hashes that file. Nothing in this file was chosen on data dated 2023-01 or
later (U7). The sha256 of this file is recorded in the pre-registration and in both manifests.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- inputs
L1_DIR = REPO / "results/dataprep/heads/L1_clean/v3"          # H1a product for use
UNITS_DIR = REPO / "results/dataprep/validation_units/v3"     # the 158 model wells
GRID_NPZ = REPO / "results/twin_forward/datum_gate_innov.npz"   # model grid (task)
BASEMAP_NPZ = REPO / "results/twin/basemap.npz"               # SRTM on active cells (twin input)

# ----------------------------------------------------------------------------- outputs
PRODUCT_PRE = "heads/L2_filled_preorigin"
PRODUCT_FULL = "heads/L2_filled_full"
EVIDENCE_DIR = REPO / "results/dataprep/reports/H2_evidence"
WORKDIR = Path(os.environ.get(
    "H2_WORKDIR",
    str(REPO / "results/dataprep/_scratch/h2work")))  # gitignored scratch; override with H2_WORKDIR
WP = "H2"

# ----------------------------------------------------------------------------- calendar
START = "2012-01"
PRE_END = "2022-12"                  # forecast origin (plan 4.2)
FULL_END = "2026-09"                 # last month of the L1 record
OUTAGE_2026 = ("2026-02-11", "2026-03-19")   # the real network outage (plan 4.4)

# ----------------------------------------------------------------------------- observed set
# An observed month is an L1 C20 month (>= 20 clean days) with L1 flag 0 or 1 (plan H1
# aggregates; U2). Months with fewer clean days are partial and are candidates for the daily
# bridge (flag 2). L1 flag 3 (post-origin neighbour-corroborated step, 2 months) is treated
# as a gap and filled (its value is neighbour-derived, never an observation).
OBS_FLAGS = (0, 1)

# ----------------------------------------------------------------------------- tested envelope
# Plan step 4: fill only wells with >= 36 observed months, internal runs <= 24 months and
# edge extrapolation <= 66 months; everything else flag 8. The trailing edge was tested only
# to 48 months (family E), so trailing fills stop at 48 (a tightening of the plan's 66, which
# is the leading-edge test length; recorded as a deviation).
MIN_OBS_MONTHS = 36
MAX_INTERNAL_RUN = 24
MAX_LEAD = 66
MAX_TRAIL = 48
SD_FLOOR = 0.05                      # plan step 3: floor 0.05 m

# ----------------------------------------------------------------------------- router (pilot)
NEIGH_K = 4                          # top 4 wells
NEIGH_MIN_CORR = 0.6                 # r >= 0.6
NEIGH_MAX_KM = 25.0                  # within 25 km
NEIGH_MIN_OVERLAP = 24               # months of overlap (pilot)
NEIGH_ALPHA = 1e-3                   # ridge penalty (pilot)
SI_GRID = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016)   # x top singular value (pilot)
SI_ITERS = 300
SI_CV_FRAC = 0.05                    # 6-month block CV mask share (pilot)
SI_CV_BLOCK = 6
SI_CV_SEED = 0
GP_MIN_OBS = 24                      # per-well GP needs >= 24 observed months (pilot)
IDW_MAX_KM = 15.0                    # same-layer IDW of anomalies baseline (pilot)
CLIM_SHRINK = 3.0                    # climatology shrinkage (pilot)

# ----------------------------------------------------------------------------- space-time kriging
STK_NEIGHBOURS = 40                  # nearest space-time neighbours per prediction
STK_MAX_LAG_MONTHS = 12              # candidate window in time
STK_MAX_KM = 30.0                    # candidate window in space
STK_VARIO_KM_BINS = (0, 2, 4, 6, 8, 10, 14, 18, 24, 30)
STK_VARIO_LAGS = (0, 1, 2, 3, 4, 6, 9, 12)

# ----------------------------------------------------------------------------- daily bridge
BRIDGE_EDGE_DAYS = 3                 # 3-day means at each edge (plan step 1)
BRIDGE_EDGE_MIN = 2                  # of which at least 2 days observed (pilot s4)
BRIDGE_MAX_GAP_DAYS = 60             # tested envelope of the bridge (family d lengths 5-60)
BRIDGE_SHAPE_MIN_YEARS = 3           # deviation curve needs >= 3 other years (pilot s4)
BRIDGE_SHAPE_OBS_FRAC = 0.8          # a year's deviation curve needs > 80 % of its days
BRIDGE_SHAPE_GRID = 37               # resampling grid of the deviation curve (pilot)
BRIDGE_MIN_CLEAN_DAYS_MONTH = 0      # an outage month fully inside a bridged gap is allowed
#: bridged-day share classes of a month for the bridge sd (design derives one sd per class).
#: Set from the distribution of the real pre-origin bridged months (shares 0.32-1.0, quartiles
#: 0.48 / 0.67 / 0.90; no error was looked at), so that each class holds real gaps.
BRIDGE_FRAC_CLASSES = ((0.0, 0.5), (0.5, 0.8), (0.8, 1.0001))
BRIDGE_D_LEN = (5, 60)               # family d: random daily gaps of 5..60 days
#: bridge form: 'shape' (linear bridge + the well's mean deviation curve, the plan's form) or
#: 'lin' (linear bridge only). Decided on pre-origin design realizations by
#: ``s1_bridge_form.py`` (rule declared there; evidence H2_evidence/design/bridge_form.json).
BRIDGE_FORM = "shape"

# ----------------------------------------------------------------------------- gap classes
# plan step 3 (gap type -> sd factor). Design-time refinements (second design pass, before
# any fresh realization): 'thin' (months where fewer than THIN_FRAC of the median number of
# observed wells are observed); the network-outage class split by run length (one month /
# two or more); the plan's internal 3-23-month class split at 6 and 12 months, because the
# native sd's scaling with gap length differs between 3- and 12-month gaps (first design
# pass: one pooled factor over-covered 3-month and under-covered 12-month gaps).
GAP_CLASSES = ("int_le2", "int_3_5", "int_6_11", "int_12_23", "int_ge24", "lead", "trail",
               "no_obs_month", "no_obs_run2", "thin")
THIN_FRAC = 0.6

# ----------------------------------------------------------------------------- artificial gaps
# Families of plan 4.4 (harness.gaps / gaps_guarded; seeds 1000*ord(F)+r, T: 7000+r) plus
# H2's own random daily-gap family 'd' (seed 1000*ord('d')+r). Design-time realizations: the
# pilot's (N 0-2, S 0-3, E 0-1, H 0-2, T 0-2 with a post-origin window) and this WP's design
# draws r <= 4. Fresh realizations (registered before any is generated): r >= 10.
DESIGN_R = {"N": range(0, 5), "S": range(0, 5), "E": range(0, 5), "H": range(0, 5),
            "T": range(0, 5), "d": range(0, 5)}
FRESH_R = {"N": range(10, 20), "S": range(10, 20), "E": range(10, 20), "H": range(10, 30),
           "T": range(10, 30), "d": range(10, 20)}
TARGET_MIN_COV = 0.5                 # pilot: target wells have >= 50 % observed months
T_WINDOW_MONTHS = 45                 # test-period replica inside the last 45 pre-origin months
D_YEARS_PRE = tuple(range(2013, 2023))   # family D (Feb-Mar window) pre-origin years
D_YEARS_FULL = tuple(range(2013, 2026))  # after-the-fact (full variant only)

# ----------------------------------------------------------------------------- scoring
Z90 = 1.6448536269514722
COVERAGE_LEVEL = 0.90
#: pumping-cycle wells (H1a well_qc_summary: 205 of 272 flagged pre-origin episodes)
PUMP_CYCLE_WELLS = ("07080132", "07080122", "10050211")
SEED_SPLIT = 20261013                # split-conformal half splits inside the fresh set (report)

PRODUCT_VERSION_PRE = "heads/L2_filled_preorigin/v1"
PRODUCT_VERSION_FULL = "heads/L2_filled_full/v1"
