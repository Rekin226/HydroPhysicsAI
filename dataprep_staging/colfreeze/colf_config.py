"""Frozen configuration of COLF.column_family_2028.v1 (round-8 rank 7): the column-family
decidability OSSE and freeze.

Everything the member loader (members.py), the OSSE engine (osse.py) and the scorer
(score_colf.py) depend on lives here as Python literals (dataprep_staging/.gitignore ignores
*.json). Values copied from other registrations are bound by the sha256 of their source file;
members.py and osse.py re-check those bindings whenever the local results/ tree exists.

Spec: docs/TWIN_ROUND8_PLAN.local.md rank #7; user rulings R8.4a/b
(results/dataprep/rulings/user_rulings_2026-11-03.json).
"""
from __future__ import annotations

import datetime as dt

UTC = dt.timezone.utc

ID = "COLF.column_family_2028.v1"
WP = "COLF"
E = "dataprep_staging/colfreeze"
CODE_FILES = (f"{E}/colf_config.py", f"{E}/members.py", f"{E}/osse.py", f"{E}/score_colf.py")

REG_DEADLINE = dt.datetime(2026, 11, 30, 16, 0, tzinfo=UTC)
NOT_BEFORE = dt.datetime(2030, 1, 15, 0, 0, tzinfo=UTC)        # the gating arm's months end
NO_LATER_THAN = dt.datetime(2031, 6, 30, 23, 59, tzinfo=UTC)   # 2029-11; later: not evaluable
D_PENDING_AFTER = "2030-01-15"                                 # R8.4b: counts as D-pending

RULINGS = {"path": "results/dataprep/rulings/user_rulings_2026-11-03.json",
           "sha256": "ec610b692835d955141e9261e6d6046ee1984f39891bcf1c2e77591aad1c69eb",
           "ids": ("R8.4a", "R8.4b")}

# ----------------------------------------------------------------------------- members
SD_V2 = {
    "id": "SD.cut_response_spread.v2",
    "registration": "results/dataprep/preregistration/SD.cut_response_spread.v2.json",
    "registration_sha256": "aecb5a81d066ff4b141b619435bb1d3822ebb1370e4cf7bf96b11729a8d76628",
    "report": "results/dataprep/reports/SD.cut_response_spread.v2/report.json",
    "report_sha256": "5cfc9ea7aac797c6374ff07c9f87d4c75838c10bc27f02907471b0e68143b569",
    "tier_key": "A+B",
}
FORWARDS = {
    "page": ("results/twin_forward/sd_page.npz",
             "09beddf71b2871ee9561a1f1eaf7a5e37c378dbac2f6d6be89bf5b3099760f5e"),
    "cwa": ("results/twin_forward/sd_cwa.npz",
            "02ae2586940654aa59fadc69135f8041f8e4322bf7cb5c18bf42b170bf1c0079"),
    "q0": ("results/twin_forward/sd_q0.npz",
           "44ee0e6eb62001a67fd653ab7c45a86093ffa36513efbf08ca7a16c7d66ba858"),
    "q0cs": ("results/twin_forward/sd_q0cs.npz",
             "a9dd2e17e9eb99eb9c88c61cd826d66235b1a336f9bedfdf1dfdf279d4b7ac91"),
    "q0nh": ("results/twin_forward/sd_q0nh.npz",
             "722c5cd5fcd8fc842ca9eea6353d1685a624b98432cfa54fdce808d880cd8ea4"),
}
UNVERIFIED_TRAJECTORY = "unverified trajectory (shared heads)"
UNVERIFIED_PAIRING = "unverified pairing (no clean reference)"
PILE_PRODUCT_COVERS = "2029-11"     # the gating arm's last month (review Q8)
# SD v2 Tier A+B (report.tiers['A+B']), copied from the SD v2 registration 'columns' and the
# report 'forwards' (rheology_index); members.verify_sd_v2 re-checks every field
MEMBER_SPEC = {
    "page": {"tier": "A", "forward": "page", "rheology_index": 0, "label": None,
             "column_sha256": "d290c59209f758389b7133c3f5b9aadcc71b42ad55f9089d56645fafc547b162",
             "theta_sha256": "036a7a17dcbd383fba9eaab6a12e716ab0cdffcdae0385adaf5349a57c64de77"},
    "page_tau30": {
        "tier": "A", "forward": "page", "rheology_index": 1, "label": None,
        "column_sha256": "7f799b590fb8e82043204424aa9af001206bb6a542415197606baf08d8518d81",
        "theta_sha256": "036a7a17dcbd383fba9eaab6a12e716ab0cdffcdae0385adaf5349a57c64de77"},
    "page_aquitard": {
        "tier": "A", "forward": "page", "rheology_index": 2, "label": UNVERIFIED_TRAJECTORY,
        "column_sha256": "d0f6511e1d3aa52b19888729969ffb6566dbb87473025891e8b005cc13c37ee0",
        "theta_sha256": "036a7a17dcbd383fba9eaab6a12e716ab0cdffcdae0385adaf5349a57c64de77"},
    "cwa": {"tier": "B", "forward": "cwa", "rheology_index": 0, "label": None,
            "column_sha256": "a80a541bbfd247b334a014365ef489181f218c303a60ba06ad668442c78b2b11",
            "theta_sha256": "789d0e459a3b35eb3e16002869405c889a86131815fcebfef84a9ffd8370704c"},
    "cwa_tau30": {
        "tier": "B", "forward": "cwa", "rheology_index": 1, "label": None,
        "column_sha256": "25700022155805bcf42305fd1d8c64231dbe1e0accae231ae9dd6809c329c988",
        "theta_sha256": "789d0e459a3b35eb3e16002869405c889a86131815fcebfef84a9ffd8370704c"},
    "q0": {"tier": "B", "forward": "q0", "rheology_index": 0, "label": None,
           "column_sha256": "86a35e070362cda5942f351d41b8518958c1918d4a616837510716ce3b3e37d0",
           "theta_sha256": "567dd90d8d96c8127024dffa8352afdb00b575ade162b9ea34b006d3a307e41c"},
    "q0_s29": {
        "tier": "B", "forward": "q0", "rheology_index": 1, "label": None,
        "column_sha256": "477460c1678a2461b8c097a1649e241f84f1a92e33d8d4528562689384b01b4c",
        "theta_sha256": "567dd90d8d96c8127024dffa8352afdb00b575ade162b9ea34b006d3a307e41c"},
    "q0_cs": {
        "tier": "B", "forward": "q0cs", "rheology_index": 0, "label": None,
        "column_sha256": "d0dff27c1321f0483f81a17fa212fbe1a0b8bc48a431dda07ce69ad3f5db9148",
        "theta_sha256": "4671a4ce337c912d766fdb6599bf47eb6776a1a10e9a9ff0b7379c7d3cb2ec26"},
    "q0_csk1": {
        "tier": "B", "forward": "q0cs", "rheology_index": 1, "label": None,
        "column_sha256": "e1844d376a43768ad4ac43278496928fd2b5ca8506d71768b268a5af0150838b",
        "theta_sha256": "4671a4ce337c912d766fdb6599bf47eb6776a1a10e9a9ff0b7379c7d3cb2ec26"},
    "q0_csk01_s29": {
        "tier": "B", "forward": "q0cs", "rheology_index": 2, "label": None,
        "column_sha256": "fe241557ec93abcd27b47a11fbdea42a1d60302569709bedc7e6049a93b8ff4f",
        "theta_sha256": "4671a4ce337c912d766fdb6599bf47eb6776a1a10e9a9ff0b7379c7d3cb2ec26"},
    "q0_nh": {
        "tier": "B", "forward": "q0nh", "rheology_index": 0, "label": UNVERIFIED_PAIRING,
        "column_sha256": "e208d29137a4f846441c7fed6a10bc67aecd4e0c1b7fc72487a22a3b3934eb5c",
        "theta_sha256": "567dd90d8d96c8127024dffa8352afdb00b575ade162b9ea34b006d3a307e41c"},
}
TIER_AB = tuple(sorted(MEMBER_SPEC))
# declared duplicates (spec): alias -> canonical. members.dedup collapses an alias on an arm
# only where its predictions are byte-identical to the canonical member's on that arm's cells
ALIASES = {"q0_s29": "q0", "q0_csk1": "q0_cs"}
# q0_nh: only from a fresh origin-2022-12 forward (spec). sd_q0nh.npz is that forward (run
# 2026-11-01 under SD v1 at main 4c2723e: origin 2022-12, --hindcast-gain 1.0 --hindcast-every
# 1, never --extend-record-to); members.load_forward re-checks it
Q0NH_FORWARD = {"origin_month": "2022-12", "hindcast_gain": 1.0, "hindcast_every": 1}

FORWARD_DATES = ("2012-01", "2032-12")
ORIGIN_MONTH = "2022-12"
SCENARIOS = ("baseline", "cut30")
STATUS_QUO = "baseline"

# ----------------------------------------------------------------------------- units
N3 = {
    "id": "N3.piles_prospective.v1",
    "registration": "results/dataprep/preregistration/N3.piles_prospective.v1.json",
    "registration_sha256": "afa017d76f974c4edb3b34bba38aa23ab9242d76b6d384f6a4d9e786b62df5c5",
    "mapping": "results/dataprep/reports/N3/mapping/mapping.json",
    "mapping_sha256": "eea0d2d8f8f8d4c0a24a2e3fb8df59b0bb6d7cf949434959af7711d8a6543a38",
    "model_side": "results/dataprep/reports/N3/mapping/model_side.json",
    "model_side_sha256": "1b40afe910a9ab5f020b5fc4557bfd75e97529eaf1e07c41b221755cd44e2149",
    "g0_result": "results/dataprep/reports/N3/g0/result.json",
    "g0_result_sha256": "80e260d005c58eac8d6d112bc91c15e94009ae496180797877c982b2d1d25fcf",
    "g0_code": "dataprep_staging/n3_piles/g0_n3.py",
    "g0_code_sha256": "3ecaad8ba3fe2949cf0b312d541dc9ffc9a29aa7eff6d6f14e81222ab7b5060f",
    "n3lib": "dataprep_staging/n3_piles/n3lib.py",
    "n3lib_sha256": "47cfebd3987adef1a43f053786000c66bb6b1be9f73c122ce7de48e63d36ad56",
    "scorer": "dataprep_staging/n3_piles/score_n3.py",
    "scorer_sha256": "2a06642f6b26783fd8bff0cb9e5d24e7fc3eaf3dc38494a4a29b9f80a45dd8eb",
    "unit": "N3:piles_printed_prospective_2026-12_2028-11",
    "level_24m_test": "N3P.level_24m",
}
N3SL = {
    "id": "N3SL.township_prospective_roc116-117.v1",
    "registration": "results/dataprep/preregistration/N3SL.township_prospective_roc116-117.v1.json",
    "registration_sha256": "be95c6e25dc5047670161d0adfa067f1c6a4c1ada9ec0a0cb5682eb505c9d6c8",
    "cells": "results/dataprep/reports/N3S-L/freeze/cells.csv",
    "cells_sha256": "918da3dac2ee1bab31020fc48f94764d7afde95036fcb79c763e336dbb471c50",
    "scorer": "dataprep_staging/n3_township/score_n3s_l.py",
    "scorer_sha256": "c72da0404f33cbec71ee74e0e567cfe3aa1cc2a3988dfc9839566f052d00ae9c",
    "units": {116: "N3SL:wra_township_summary_roc116", 117: "N3SL:wra_township_summary_roc117"},
    "l1_tests": {116: "N3SLP.L1_coast_roc116", 117: "N3SLP.L1_coast_roc117"},
}
# N3's primary piles (n3lib.PILES, n3lib sha 47cfebd3; cells from N3 mapping.json eea0d2d8)
PILES = {
    "tuku": {"pile": "土庫國中", "cell": 541, "depth_m": 400, "fallback_depth_m": 300},
    "huwei": {"pile": "中科虎尾", "cell": 769, "depth_m": 300, "fallback_depth_m": None},
}
PILE_KEYS = ("tuku", "huwei")
XIUTAN = {"pile": "秀潭國小", "depth_m": 70,
          "rule": "excluded unless deep_benchmarks_printed v3 grants prospective_scorer (R8.4a)",
          "product_root": "results/dataprep/deep_benchmarks_printed", "min_version": 3}
# N3S-L's 11 coastal townships and nominal survey intervals (score_n3s_l.py c72da040)
COAST = ("麥寮鄉", "北港鎮", "東勢鄉", "水林鄉", "臺西鄉", "四湖鄉", "口湖鄉", "褒忠鄉",
         "元長鄉", "崙背鄉", "大城鄉")
COAST_COUNTY = {t: ("彰化縣" if t == "大城鄉" else "雲林縣") for t in COAST}
SURVEYS = (116, 117)
NOMINAL = {116: ("2026-05", "2027-05"), 117: ("2027-05", "2028-05")}

# frozen prediction windows (monthly baseline subsidence, metres, at the unit cells)
FREEZE_PILE_MONTHS = ("2026-11", "2029-11")
FREEZE_TOWN_MONTHS = ("2025-01", "2029-06")      # widened (review S6)

# ----------------------------------------------------------------------------- arms
ARMS = {
    "r_piles": {
        "test_id": "COLFP.rank_piles_2026-12_2028-11", "role": "report_only", "kind": "piles",
        "window": ("2026-12", "2028-11"), "ref_month": "2026-11", "min_months": 16,
        "data_scope": "deep_piles_printed:2026-12..2028-11", "owner": N3["id"],
        "fresh_units": ["COLF:piles_printed_report_only_2026-12_2028-11"],
        "owner_units": [N3["unit"]],
        "why_role": "N3's months (N3P.level_24m report-only): R8.4a report-only ranking role; "
                    "one role per number"},
    "r_towns": {
        "test_id": "COLFP.rank_towns_roc116-117", "role": "report_only", "kind": "towns",
        "surveys": SURVEYS, "data_scope": "wra_township_summaries:2026-06..2028-05",
        "owner": N3SL["id"],
        "fresh_units": ["COLF:wra_township_summary_roc116_report_only",
                        "COLF:wra_township_summary_roc117_report_only"],
        "owner_units": [N3SL["units"][116], N3SL["units"][117]],
        "why_role": "N3S-L's survey years: report-only ranking role (R8.4a, one role per "
                    "number)"},
    "g_piles": {
        "test_id": "COLFP.screen_piles_2028-12_2029-11", "role": "gating", "kind": "piles",
        "window": ("2028-12", "2029-11"), "ref_month": "2028-11", "min_months": 8,
        "data_scope": "deep_piles_printed:2028-12..2029-11", "owner": None,
        "fresh_units": ["COLF:piles_printed_prospective_2028-12_2029-11"],
        "why_role": "the unowned months 2028-12..2029-11: R8.4a gating screen; not evaluable "
                    "with fewer than 8 verified months at either primary pile"},
}
ARM_ORDER = ("r_piles", "r_towns", "g_piles")
# The report-only arms read the SAME printed numbers as N3's and N3S-L's units. They carry
# their own unit ids (members copied by value from the owners' declarations, osse.py) because
# the owners' unit ids are blocked for any NEW registration by the look
# 'R8:fv_stat_v2_builder_reads_disclosure' (2026-11-03T01:39Z; universal members, window
# 2026-11..2027-10; a read of E04 model outputs, no pile or township value; corrected in
# prose by LOOK_SCOPE_CORRECTION, which cannot change its members): prereg3's rule (b)
# blocks a unit looked at after it was declared fresh, whatever the estimand. Disclosed;
# USE_OWNER_UNIT_IDS = True restores the owners' ids (then write_prereg refuses).
USE_OWNER_UNIT_IDS = False
G_UNIT = ARMS["g_piles"]["fresh_units"][0]
G_UNIT_MEMBERS = {
    "description": "WRA printed deep-pile months (table 3 of the monthly subsidence report; "
                   "007001 table 1 as second source) 2028-12..2029-11 at the two Yunlin piles "
                   "anchored at >= 300 m: 土庫國中 400 m (300 m fallback sensor) and 中科虎尾 "
                   "300 m; months no registration owns (N3's end 2028-11)",
    "ids": ["dbm_printed:土庫國中:400", "dbm_printed:土庫國中:300", "dbm_printed:中科虎尾:300"],
    "window": "2028-12..2029-11"}

# ----------------------------------------------------------------------------- OSSE design
# N3 G0's pile-side noise, copied (g0_n3.py 3ecaad8b, n3lib.py 47cfebd3, result 80e260d0):
# the nest-vs-cell head error AR(1) on the head level (marginal sd x persistence grid, one
# series per pile) and the 0.1 cm print rounding of every monthly increment. G0 puts the head
# error on dD; here it reaches the pile through N3's frozen g_mod_12 (model_side.json 1b40afe9,
# cm of column increment per metre of displayed head change): obs increment =
# round_0.1(share x (truth increment + G_MOD x diff(e)))
G0_SD_GRID_M = (0.75, 1.5)
G0_PHI_GRID = (0.5, 0.9)
PRINT_RES_CM = 0.1
G_MOD_CM_PER_M = 0.7613843383557103
# report-only sensitivity cells (review S3): N3 model_side g_mod_12.g_mod_cell_report_only
# (the cell's own layer-mean head) in place of g_mod, and G0's other conversion truths
G_MOD_CELL_CM_PER_M = 0.932312795670687
RHO_SENS = (0.4, 2.0)
# the anchor-change cell (300 m printing): 土庫國中's 400 m sensor prints only the first
# AC_400_MONTHS months (< the arm's min_months), so the declared sensor rule (N3's: 400 m;
# 300 m only if 400 m has fewer than min_months and 300 m at least that many; never mixed)
# scores the 300 m series, whose truth is F300 x the truth column (N3's report-only
# anchor-depth share 0.8); members stay unscaled (N3's depth convention)
F300 = 0.8
AC_400_MONTHS = 5
ANCHOR_CELLS = ("nominal", "ac300")      # ac300 is report-only (review S2)
# availability cells (review S4, nominal set): every month verified, or exactly the arm's
# min_months per pile (a random subset per replicate and pile)
AVAIL_CELLS = ("full", "minimum")
# N3S-L's township noise and bound rules (score_n3s_l.py c72da040): N(0, sigma) on each
# township peak (its passability grid, sigma > 0), listing floor 3 cm/yr, tolerance 0.5
# cm/yr, upper bound = the year's maximum listed peak when listed else 3, informative only
# with >= 3 listed coastal townships
TOWN_SIGMA_GRID = (0.5, 1.0, 1.5)
LISTED_FLOOR = 3.0
TOL = 0.5
L1_MIN_LISTED = 3

REPS = 2000
P_D = 0.80                    # pair decidable: D sub-label (gating) / ranking line (report-only)
P_NARROW = 0.50               # pair not separable: narrowing line now
SELF_CHECK_TOL = 0.05         # |P_sym - 0.5| on the equidistant (midpoint) truth
TIE_ABS = 1e-12               # |RMSE_a - RMSE_b| <= this is a tie (cm or cm/yr)

# fixed paths (review S8): the engineering record (seed, freeze sha) is written once before
# any OSSE run; attempts, the result and self-check failure records live beside it
COLF_DIR = "results/dataprep/reports/COLF"
ENGINEERING_RECORD = f"{COLF_DIR}/engineering_record.json"
ATTEMPTS = f"{COLF_DIR}/attempts.jsonl"
OSSE_RESULT = f"{COLF_DIR}/osse_result.json"
SELF_CHECK_FAILURE = f"{COLF_DIR}/osse_selfcheck_failure_{{k}}.json"
# the lead's correction of the universal disclosure rows (looks.jsonl metadata only)
LOOK_SCOPE_CORRECTION = {
    "unit": "R8:look_scope_correction_20261103", "time": "2026-11-03T03:37:16.405497Z",
    "row": 632, "prev": "7db36223cac627b6b2490920c25b24f1e03f2b7668dabcb67d8b096f99bb3d32",
    "chain_hash": "bef64e1ad13ca7e9d89162a455601193094b731e0958ac163b90b853acb482c3",
    "chain_hash_source": "the 'prev' of looks.jsonl row 633"}

TEST_SEED = 987654321                        # the unit tests' seed; refused for real runs
REFUSED_SEEDS = (20261101, 20261130)         # N3 G0's seed and N3's bootstrap seed
REAL_REFUSED_SEEDS = (TEST_SEED,)

CLAIMS = {"E24": "model-form spread row (SD.cut_response_spread.v2)",
          "C09": "pile-anchored status-quo compaction at the 2 Yunlin pile cells (sub-claim)",
          "C11": "Townships tab: the column-dependent ranking (no leader named)"}
CLAIMS_BY_ARM = {"r_piles": ("E24", "C09", "C11"), "r_towns": ("E24", "C11"),
                 "g_piles": ("E24", "C09", "C11")}
