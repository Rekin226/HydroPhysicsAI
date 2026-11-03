"""Constants of X5.gnss_seasonal_family.v1 (round 8 rank 11): one consolidated GNSS seasonal
declaration, the pre-look redesign of X5.gnss_seasonal_column.v1 (never registered; its G0-OC
failed under the scored two-scale rule, g0oc.json 35ebf516, no GNSS value read).

Everything a registration or a gate binds lives here, in code (dataprep_staging/.gitignore
ignores *.json). Nothing in this module reads a file.

Vehicle: the frozen v1 model side (model_side.json 65b927cb, 14 arms) read through
``dataprep_staging/sd_column`` (imported read-only, its files hash-bound), SD v2's cut30
values (report.json 5cfc9ea7) and one model-only supplement (the page arm and every other
arm at the TKJS/NTUH deep-pile cells over 2016-06..2019-12, built from the same sealed SD
forwards). Primary statistic T1 (distal/mid median annual amplitude ratio, frame scale
cancels); named secondaries T2 (median distal amplitude against the frame interval), v1's
directional arm (unchanged criteria) and U2 (deep-pile one-sided frame bound).

The files live in dataprep_staging/gnss_family/ by the lead's choice (the plan named
dataprep_staging/sd_column/, whose files stay untouched and hash-bound).

HONESTY NOTE (review B3): on the frozen model side, readings (a)-(c) are decided by T2, the
frame-carrying statistic whose outcome the disclosures call close to known (N1's 6.1 mm InSAR
median against a ~20 mm cluster). T1, the frame-free statistic, can exclude the far member
only with a 90 % half-width below about 0.08 (review figure; planner model T1 1.80 vs the
nearest cluster member 2.57 gives log 0.36 minus the 0.223 allowance = 0.13), far below the
expected ~0.4-0.5. The pass label therefore names T2 (PASS_LABEL).
"""
from __future__ import annotations

import math

X = "dataprep_staging/gnss_family"
SDC = "dataprep_staging/sd_column"
WP = "X5"
ID = "X5.gnss_seasonal_family.v1"
#: the gates declaration (kind 'declaration'), registered by the lead BEFORE G-sep runs
#: (X5.n2c_reading / N2CS.osse_design precedent)
DECL_ID = "X5.gnss_seasonal_family.gates.v1"
V1_ID = "X5.gnss_seasonal_column.v1"

OUT = f"results/dataprep/reports/{ID}"
ATTEMPTS = f"{OUT}/attempts.jsonl"
ENGINEERING = f"{OUT}/engineering.json"
PILE_MODEL_JSON = f"{OUT}/pile_model.json"
PILE_MODEL_NPZ = f"{OUT}/pile_model.npz"
GSEP_OUT = f"{OUT}/gsep.json"
G0_OUT = f"{OUT}/g0.json"
REPORT = f"{OUT}/report.json"
PREREG_DIR_REL = "results/dataprep/preregistration"
LOOKS = f"{PREREG_DIR_REL}/looks.jsonl"

# --------------------------------------------------------------------------- bound inputs
V1_MODEL_DIR = f"results/dataprep/reports/{V1_ID}/model_side"
V1_MODEL_JSON = f"{V1_MODEL_DIR}/model_side.json"
V1_MODEL_SHA = "65b927cb13071e9c5330a850eba4ccc60d4bb9c6d97024d06369a67461d6c525"
V1_SERIES_NPZ = f"{V1_MODEL_DIR}/model_series.npz"
V1_SERIES_SHA = "487e0d755543278f08152774e5ea86f1b32608323d4641ed746ab41ee5edcb92"
V1_G0 = f"results/dataprep/reports/{V1_ID}/g0oc.json"
V1_G0_SHA = "35ebf51674ee6e357e14aa332699d75b0c60d80000586898f5552c3bcc1cf7eb"
V1_ATTEMPTS = f"results/dataprep/reports/{V1_ID}/attempts.jsonl"
SD_REPORT = "results/dataprep/reports/SD.cut_response_spread.v2/report.json"
SD_REPORT_SHA = "5cfc9ea7aac797c6374ff07c9f87d4c75838c10bc27f02907471b0e68143b569"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULINGS_SHA = "ec610b692835d955141e9261e6d6046ee1984f39891bcf1c2e77591aad1c69eb"
RULING_ID = "R8.5"
RULING_LABEL = "S (design-time, window flagged, outcome partly known)"
#: the label a screen pass (a)/(b) and DIR 'larger' carry (lead ruling on review B3)
PASS_LABEL = ("S (design-time, window flagged, outcome partly known; decided by the "
              "frame-carrying T2)")
DECIDED_BY_T2 = ("readings (a)-(c) are decided by T2, the frame-carrying statistic whose "
                 "outcome is close to known; T1 (frame-free) can exclude the far member only "
                 "if its 90 % half-width is below about 0.08, far below the expected ~0.4-0.5")
RULING_PHRASES = ("Extend R7.gnss_seasonal_S", "outcome partly known",
                  "design-time narrowing, not selection", "BESIDE the full range")

GNSS_PRODUCT = "results/dataprep/gnss_clean/preorigin/v4"
PILE_PRODUCT = "results/dataprep/deep_benchmarks/v2"
PILE_MAPPING = f"{PILE_PRODUCT}/mapping.csv"
#: the only mapping.csv columns read (metadata; r_printed_feed and the identity shares are
#: never opened)
PILE_MAPPING_COLUMNS = ("sensor_id", "pile", "depth_m", "grid_cell", "on_grid",
                        "first_month", "last_month")
ELEC_PRODUCT = "results/dataprep/pump_electricity/preorigin/v3"
ELEC_FILE = "cells_total.npz"
USE = "design_time_evidence"
#: gnss_clean/preorigin/v4 and deep_benchmarks/v2 failed their gates: only these uses open
GATE_EXEMPT = ("design_time_evidence", "after_the_fact_analysis")

# --------------------------------------------------------------------------- members
#: SD v2 tiers (report.json 'tiers'); the screen family is Tier A+B, deduplicated
TIER_AB = ("cwa", "cwa_tau30", "page", "page_aquitard", "page_tau30", "q0", "q0_cs",
           "q0_csk01_s29", "q0_csk1", "q0_nh", "q0_s29")
CH_TIER = ("ccz", "q0c_csk1", "x5r1_cs")
PAGE = "page"
#: duplicate -> kept arm: byte-identical model series at every primary station and both
#: pile cells (checked by the engineering record; q0_s29 is also SD v2's distinct_of q0;
#: q0_csk1 differs from q0_cs only at Changhua cells)
DEDUP = {"q0_s29": "q0", "q0_csk1": "q0_cs"}
#: Tier A+B arms kept out of the screen (report-only)
SCREEN_REPORT_ONLY = {"q0_nh": "nudged-head pairing unverified (its 2014-22 model side runs "
                               "on nudged heads, hindcast gain 1.0; SD UNVERIFIED)"}
SCREEN = tuple(sorted(set(TIER_AB) - set(DEDUP) - set(SCREEN_REPORT_ONLY)))
REPORT_ONLY_ARMS = tuple(sorted(set(SCREEN_REPORT_ONLY) | set(CH_TIER)))
#: cut30 of a column: SD v2 per_column[c].summary[CUT30_HORIZON][CUT30_GROUP][CUT30_FIELD]
#: (Yunlin mean avoided subsidence, cm, at SD's primary horizon)
CUT30_HORIZON = "2028-12"
CUT30_GROUP = "yunlin"
CUT30_FIELD = "avoided_cm"
#: members whose cut30 is within x CLUSTER_X of the page column form the cluster; the rest
#: are 'far'; a G0 member is 'far off' the truth when its cut30 differs by more than x FAR_X
CLUSTER_X = 1.5
FAR_X = 1.5

# --------------------------------------------------------------------------- units
ZONES = ("distal", "mid")
NGL_STRATUM = ("DNAN", "HUWE")
REPORT_STATIONS = ("ANES", "CHSG")
UNIT_PRIMARY = "x5gsf.gnss.yunlin_primary"
UNIT_NGL = "x5gsf.gnss.yunlin_ngl_huwe_dnan"
UNIT_PILES = "x5gsf.piles.tkjs_ntuh_300m"
UNIT_REPORT = "x5gsf.gnss.report_only_anes_chsg"
T1_ID = "X5.gsf.T1_distal_mid_ratio"
T2_ID = "X5.gsf.T2_distal_amplitude"
DIR_ID = "X5.gsf.DIR_v1_directional"
U2_ID = "X5.gsf.U2_pile_frame_bound"

# --------------------------------------------------------------------------- U2 piles
#: deepest pre-origin sensors at the two Yunlin feed piles (deep_benchmarks/v2 mapping);
#: TKJS(3) 400 m starts 2023-09 and is defective before it (flag 9)
PILES = {"TKJS(1)": {"gnss": "TKJS", "depth_m": 300.0, "pile_cell": 541, "gnss_cell": 541},
         "NTUH(1)": {"gnss": "NTUH", "depth_m": 300.0, "pile_cell": 769, "gnss_cell": 771}}
PILE_WINDOW = ("2016-06", "2019-12")
PILE_FLAGS = (0, 1, 2)
PILE_FEEDS = ("vendor", "raw")
PILE_VARIANT = "preorigin"
#: spring months (March-May, calendar month index 2-4) are masked on the pile AND the model
#: side alike (G4 spent pile springs); the annual fit uses the other nine months
SPRING_MONTHS = (2, 3, 4)
#: a sensor enters the bound with >= 2 years (NTUH(1) has one pre-origin year in the window:
#: report-only by this rule); GNSS and pile pair by site, one bound per site
MIN_YEARS_U2 = 2

# --------------------------------------------------------------------------- estimator
MIN_MONTHS = 8
MIN_YEARS = 2
MIN_DAYS = 10
LEVEL = 0.90
ALLOW = 1.25                       # declared model-error allowance (ratio)
FRAME_MAX = 1.42                   # GNSS over-read of fast sinking (leveling reference)
LOG_ALLOW = math.log(ALLOW)
LOG_FRAME = math.log(FRAME_MAX)
#: T1 keeps member m unless the 90 % interval of log(T1_obs / T1_m) lies wholly outside
#: [-log 1.25, log 1.25]; T2 unless the interval of log(T2_obs / T2_m) lies wholly outside
#: [-log 1.25, log 1.42 + log 1.25] (the frame interval [1/1.42, 1] widened by log 1.25)
T1_BAND = (-LOG_ALLOW, LOG_ALLOW)
T2_BAND = (-LOG_ALLOW, LOG_FRAME + LOG_ALLOW)
B_SCORE = 2000
#: REPORT-ONLY qc_jumps variant (review S1, U7): the product's jump-cluster constants
#: (s3_gnss/v4 products.py) were set in the S3 repair round after 2023-25 jump counts were
#: seen, so the gated estimator applies NO extra jump rule beyond the product's own flags
#: (v1's flags 0-1); the variant dropping WRA station-months with >= QC_JUMPS_PER_MONTH
#: consecutive-day |delta value_raw| > QC_JUMP_M is report-only
QC_JUMP_M = 0.03
QC_JUMPS_PER_MONTH = 3
GNSS_FLAGS = (0, 1)
REMOVAL = "ngl_offfan"             # v1's declared S.REMOVAL, unchanged
#: report-only F1 electricity reduced form (C-R2)
F1_DROP_YEAR = 2021

# --------------------------------------------------------------------------- seeds
#: seeds that may not be declared (v1's estimator and G0-OC seeds; a fresh seed is required)
REFUSED_SEEDS = frozenset({20261101, 20261102})


def check_seed(seed) -> int:
    """A declared seed: an int (bool refused), not one of REFUSED_SEEDS. No default."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int (no default)")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if seed in REFUSED_SEEDS:
        raise ValueError(f"seed {seed} is v1's; declare a fresh seed")
    return seed


# --------------------------------------------------------------------------- G0 grid
LOADING_MM = (0.0, 3.0, 6.0)
THERMAL_MM = (0.0, 2.0)
SIGMA_MM = (2.0, 4.0)
JUMPS = ("off", "tkjs")
MODEL_ERR = (0.2, 0.4)             # per-station lognormal model error sigma
RHO = 0.5
P_JUMP = 0.25
JUMP_MM = (30.0, 120.0)
SURVIVE = (0.3, 1.0)
P_MISS = 0.10
THERMAL_PEAK_MONTH = 6
FRAME_UNIFORM = (1.0, FRAME_MAX)   # base cells: one s ~ U(1, 1.42) per world, all stations
#: proximal_seasonal_truth = the spec's "proximal seasonal truth": real seasonal subsidence of
#: 2-4 mm at the reference (common-mode) stations, removed with the common mode
STRESS = ("proximal_seasonal_truth", "zone_frame", "record_low_2021", "coast_loading")
STRESS_BASE = {"loading_mm": 3.0, "thermal_mm": 2.0, "jumps": "tkjs"}
REF_SEASONAL_MM = (2.0, 4.0)       # real seasonal subsidence inside the removed common mode
ZONE_FRAME = {"distal": FRAME_MAX, "mid": 1.0}
RECORD_LOW_MM = {"distal": (5.0, 20.0), "mid": (2.5, 10.0)}
RECORD_LOW_RAMP = ("2021-03", "2021-06")
COAST_LOADING_MM = 6.0
#: >= 1000 per cell (review B1: at 200 the every-cell 0.90 rule fails by chance ~half the time)
N_REP = 1000
B_G0 = 400
G0_PASS = {"true_retained_min_every_cell": 0.90,
           "far_excluded_min_pooled": 0.80,
           "far_excluded_min_worst_cell": 0.60,
           "p_all_excluded_max_every_cell": 0.10}
G0_CELL = "a cell is one nuisance cell x one true member (64 x 8 = 512 cells)"
#: G-sep: noise half-width from page-truth worlds on these reference cells (the maximum of
#: the per-cell median 90 % half-widths). Pass (LEAD's design choice, review B2, not the
#: plan's): on model-only points the far member(s) separate from EVERY cluster member by more
#: than hw + log 1.25 on T1, or on T2 by more than hw + log 1.25 + log 1.42 (the screen's T2
#: band)
GSEP_CELLS = tuple({"sigma_mm": s, "model_err": e, **STRESS_BASE} for s in SIGMA_MM
                   for e in MODEL_ERR)
GSEP_NREP = 200

# --------------------------------------------------------------------------- random streams
#: every random stream is numpy SeedSequence([declared seed, stream]); the gates declaration's
#: seed drives G-sep and G0, the registration's (different) seed drives the scoring streams
STREAM_GSEP = 1
STREAM_G0 = 2
STREAM_SCREEN = 11
STREAM_U2 = 12
STREAM_SCREEN_QC = 13

# --------------------------------------------------------------------------- disclosures
#: printed in BOTH the gates declaration and the registration (review minor)
FOUR_DISCLOSURES = (
    "outcome partly known: S4b examined WRA/InSAR seasonal amplitude against easting in Yunlin "
    "2020-22 (looks s4b.verify.amp_census_easting.yunlin and "
    "s4b.repair.amp_census_easting.yunlin): the coast-to-inland contrast T1 reads was seen",
    "N1 lists the observed median InSAR seasonal amplitude 6.1 mm (S4b) as designer "
    "knowledge: T2's outcome against the ~20 mm cluster is close to known (outcome (b) is the "
    "likely one)",
    "stations.csv carries a per-station annual_amp_m computed at product build: the station "
    "amplitude pre-exists; no builder of v1 or this redesign read that column",
    "round-8 accidental look: the planner read the header and the WRA ANES and CHSG rows "
    "(rate_m_yr, annual_amp_m, 2020-01..2025-03) of gnss_clean/full/v4/stations.csv, logged "
    "as R8:gnss_anes_chsg_stations_disclosure at 2026-11-02T15:53Z (after the round-7 "
    "R7:gnss_anes_chsg_disclosure of 2026-11-01T11:14Z); ANES and CHSG are report-only")
