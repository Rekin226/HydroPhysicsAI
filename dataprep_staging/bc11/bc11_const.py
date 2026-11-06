"""Frozen constants of I1.backcast_2010_11.v1 (round 8 plan #12; user rulings R8.6a-c).

Every number, path, seed and unit id the declaration freezes lives here, so the product build,
the model runner, the G0 studies, the declaration builder, the scorer and the tests read one
definition. The declaration hashes this file (frozen_code): changing anything here after
registration needs a new version id.

Nothing in this module reads data. The availability look's own constants stay in
``dataprep_staging/ch_x/avail_bc11.py`` (PR #84; its constants hash is in the logged look
note) and are imported, never copied.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_CHX = HERE.parent / "ch_x"
for _p in (str(HERE), str(_CHX), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import avail_bc11 as A  # noqa: E402  (the logged availability look's design; never edited)

X = "dataprep_staging/bc11"
CHX = "dataprep_staging/ch_x"
WP = "I1"

# ----------------------------------------------------------------------------- ids and units
ID = A.BC11_ID                                   # "I1.backcast_2010_11.v1"
AVAIL_ID = A.AVAIL_ID
AVAIL_UNIT = A.UNIT                              # I1BC11:ls_cache_dates_2009_2011 (logged)
AVAIL_OUT = A.OUT                                # results/twin_runs/chx/backcast_2010_11/...
#: the test's BRAND-NEW fresh units (never looked at under these ids): the availability look
#: (AVAIL_UNIT), the product_build look (PRODUCT_UNIT), E26's I1CHB units and every registry
#: fresh.* unit are different ids, so none of those looks blocks the test by id; their member
#: overlap is disclosed, and they all predate the registration
UNIT_YL = "I1BC11T:heads_2010_2011_yunlin"
UNIT_CH = "I1BC11T:heads_2010_2011_changhua"
UNITS = (UNIT_YL, UNIT_CH)
UNIT_OF_COUNTY = {"yunlin": UNIT_YL, "changhua": UNIT_CH}
#: the product_build look's own unit (not a test unit, not the availability unit, not E26's)
PRODUCT_UNIT = "I1BC11P:l1pre2012_v2_build_2010_2011"
PRODUCT_PURPOSE = "product_build"
SCOPE = "heads:2010-01..2011-12"
WINDOW_TEXT = "2010-01..2011-12"

TEST_G1 = "I1BC11T.G1_stat_vs_cl_2011"           # primary (G1 + G1b)
TEST_G2 = "I1BC11T.G2_physics_fair_cl_2011"      # secondary, per county
TEST_YR = "I1BC11T.yunlin_response_2010_2011"    # secondary, Yunlin only
TEST_G3 = "I1BC11T.G3_pumping_value_2011"        # report-only unless the swap gate passes
TESTS = (TEST_G1, TEST_G2, TEST_YR, TEST_G3)
COUNTIES = ("yunlin", "changhua")

# ----------------------------------------------------------------------------- months
BC_RUN = ("2007-01", "2011-12")                  # model axis; spin-up 2007-01..2009-12
SPINUP_END = "2009-12"
WINDOW = ("2010-01", "2011-12")                  # the product's and the panel's months
PRE = ("2010-01", "2010-12")                     # the 12 pre-origin months
ORIGIN = "2010-12"                               # the only origin (2009-12: 0 qualifying)
SCORED = ("2011-01", "2011-12")                  # the scored year
FIT = ("2012-01", "2022-12")                     # CL, rho, sd, datum: 2012-22 only (U7)
PATTERN_YEAR = 2007                              # Yunlin response arm: Yunlin at 2007

# ----------------------------------------------------------------------------- data
STATIONS = "AMP_V2/data/fan_stations.parquet"
PRODUCT_ROOT = "results/dataprep/heads/L1_pre2012"
PRODUCT_NAME = "heads/L1_pre2012"
PRODUCT_VERSION = "v2"
PRODUCT = f"{PRODUCT_ROOT}/{PRODUCT_VERSION}"
PRODUCT_USE = "design_time_evidence"             # the scorer's only use of the product
PRODUCT_ALLOWED = ("validator", "design_time_evidence", "initial_conditions")
PRODUCT_FORBIDDEN = ("scoring_target", "calibration_target", "datum_fit", "baseline",
                     "selection", "prior", "prior_width_input", "model_design", "idw_field",
                     "forcing_frozen", "forcing_recorded_arm", "prospective_scorer",
                     "statistical_layer_history")
#: 2012-22 heads for CL, rho, sd and the physics datum (spent; L1_clean v3 starts 2012-01)
CL_PRODUCT = "results/dataprep/heads/L1_clean/v3"
CL_USE = "baseline"
CL_WINDOW_END = "2022-12"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULING_PRODUCT, RULING_S, RULING_ARM = "R8.6a", "R8.6b", "R8.6c"
E26_ID = "I1.CH_backcast.v1"
E26_UNIT = "I1CHB:changhua_l1pre2012_2010_2011"
X5EV_ID = "X5.energy_volume.c_vs_a.v1"
X5EV_UNITS = A.X5EV_UNITS
I1CHB_UNITS = A.I1CHB_UNITS
PRINTED_WELL = A.PRINTED_WELL

# ----------------------------------------------------------------------------- page model
PAGE_RUN = "results/twin_runs/stage3_datum_gate"
PAGE_THETAS = (f"{PAGE_RUN}/stage3_theta.json", f"{PAGE_RUN}/stage3_fold_thetas.json",
               f"{PAGE_RUN}/stage3_posterior_tn.json")
PAGE_N_SETS = 18                                 # page theta + 5 folds + 12 posterior draws
PAGE_RF = "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv"   # the page run's rain
PAGE_FORWARD = "results/twin_forward/datum_gate_innov.npz"
PAGE_FORWARD_SHA = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
PAGE_FIT_MONTHS = 132                            # months 0..131 = 2012-01..2022-12
#: E0 runs at ONE declared code commit (lead ruling): 4c2723e, where FS.start_invariance.v1
#: reproduced the page recipe and scored. (730f65b, FS's second rung, cannot import
#: forward.extra_hist / zone_county, which this runner needs, so it is not a rung here.) An
#: E0 failure at 4c2723e (e0.json written with passed false, exit 3, logged) means the
#: backcast is NOT EVALUABLE: stop, no registration, ledger line only.
E0_LADDER = ("4c2723e",)
WT_HEAD = E0_LADDER[0]
WT_NAME_FMT = "HydroPhysicsAI-bc11run-{head}"
WT_NAME = WT_NAME_FMT.format(head=WT_HEAD)
COUNTY_GPKG = "chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg"
COUNTY_ZH = {"yunlin": "雲林縣", "changhua": "彰化縣"}
CENSUS_V3 = "results/dataprep/pump_electricity/preorigin/v3"
E0_TOL_M = 0.01

# ----------------------------------------------------------------------------- outputs
RUN_DIR = "results/twin_runs/chx/backcast_2010_11"
E0_OUT_FMT = RUN_DIR + "/e0_{head}/e0.json"          # named by the E0 code commit
E0_OUT = E0_OUT_FMT.format(head=WT_HEAD)
SWAP_OUT = f"{RUN_DIR}/swap/swap.json"
PANEL = f"{RUN_DIR}/hindcast/bc11_panel.npz"
PANEL_JSON = f"{RUN_DIR}/hindcast/bc11_panel.json"
BASELINES = f"{RUN_DIR}/baselines/bc11_baselines.npz"
BASELINES_JSON = f"{RUN_DIR}/baselines/bc11_baselines.json"
G0_DIR = f"{RUN_DIR}/g0"
G0_FILES = {"g1": f"{G0_DIR}/g0_g1.json", "g2": f"{G0_DIR}/g0_g2.json",
            "yr": f"{G0_DIR}/g0_yr.json", "g3": f"{G0_DIR}/g0_g3.json"}
ATTEMPTS = f"{RUN_DIR}/attempts.jsonl"
#: every output-writing step appends its OWN attempt row (attempts_bc11.run_step: time,
#: step, rc, the sha256 of each output file, code shas). Before registration each step must
#: have exactly ONE rc = 0 row whose output shas equal the files now on disk (a deleted and
#: rerun output leaves two rc 0 rows or a sha mismatch: refused); every other row is
#: disclosed; any logged E0 failure (step e0, rc 3) makes the backcast not evaluable.
STEP_OUTPUTS = {
    "g0_g1": (G0_FILES["g1"],),
    "build": (f"{PRODUCT}/MANIFEST.json",),
    "e0": (E0_OUT,),
    "swap": (SWAP_OUT,),
    "hindcast": (PANEL, PANEL_JSON),
    "baselines": (BASELINES, BASELINES_JSON),
    "g0_g2": (G0_FILES["g2"],), "g0_g3": (G0_FILES["g3"],), "g0_yr": (G0_FILES["yr"],)}
ATTEMPT_STEPS = tuple(STEP_OUTPUTS)
E0_FAIL_RC = 3
REPORT = f"results/dataprep/reports/{ID}/report.json"
PREREG_FILE = f"results/dataprep/preregistration/{ID}.json"

# ----------------------------------------------------------------------------- G1 / G1b
#: E04's recipe (twin.hybrid.register), refitted on 2012-22 only (E04's frozen rho unused)
MAX_GAP = 3                                      # last pre-origin month within 3 of 2010-12
MIN_FIT_MONTHS = 24                              # hybrid.register's count >= 24
LEVEL = 90                                       # central 90 % interval, +-Z[90] sd
MIN_WINDOW_MONTHS = 6                            # E04's min_months, per scored well
MIN_POOLED_WELLS = 25                            # the availability rule's pooled floor
COUNTY_LINE_MIN_WELLS = 10
G1_MAX_RATIO = 1.0                               # point ratio <= 1.0 (ties pass)
G1_UB_BELOW = 1.0                                # 90 % upper bound strictly < 1.0
COVERAGE_BAND = (0.85, 0.95)                     # inclusive
BOOT_B = 2000
BOOT_SEED = 71120615
BOOT_Q = 0.95                                    # upper end of the central 90 % interval

# ----------------------------------------------------------------------------- G2 / G3
FAIR_K = 1.25                                    # drift_diag.FAIR_RMSE_K (E07)
FAIR_SHAPE_TOL = 0.05                            # drift_diag.FAIR_SHAPE_TOL (E07)
FAIR_MIN_FIT_MONTHS = 24                         # drift_diag.FAIR_MIN_FIT_MONTHS (E07)
G2_SHAPE_CLAUSE = True                           # E07's fair rule has both clauses
G3_MAX_RATIO = 1.0                               # F1 / F0 equal-station ratio < 1.0 ...
G3_UB_BELOW = 1.0                                # ... with the 90 % upper bound < 1.0
#: the pre-declared Yunlin+Changhua v3-vs-native energy swap gate (chx_const S0 bands, both
#: counties, every member of the page ensemble pooled as the ensemble mean)
SWAP_ENERGY_BAND = (0.85, 1.15)
SWAP_CORR_MIN = 0.90
SWAP_RMS_BAND = (0.80, 1.25)
SWAP_PATTERN_YEAR = 2012

# ----------------------------------------------------------------------------- Yunlin arm
YR_FORM = "calmonth_twoway"                      # well x calendar month + month of sample
YR_MIN_WELLS = 10
YR_MIN_MONTHS = 18

# ----------------------------------------------------------------------------- G0
G0_POWER_MIN = 0.60                              # R8.6c, applied to every gating G0
G0_FALSE_MAX = 0.20
G0_SEEDS = {"g1": 71120611, "g2": 71120612, "yr": 71120613, "g3": 71120614}
#: G1: FV.stat.v2's AR(1) generator (stat_fv.anomalies) and grid
G0_RHO_GRID = (0.80, 0.86, 0.93)
G0_SHARE_GRID = (0.0, 0.15, 0.30)
G0_NULL_RHO = 0.0
G0_K_G1 = 500
G0_K_G2 = 500
G0_K_G3 = 500
G0_K_YR = 1000
#: report-only G1 cells: a per-well level offset of the 2010-11 months (sd in units of the
#: anomaly sd) on top of the AR world at rho 0.86, share 0.15 (the time-reversed CL)
G0_G1_OFFSET_SDS = (0.5, 1.0)
#: Yunlin arm: N2C's declared noise and confounder (synth_n2c.PARAMS, Changhua-calibrated,
#: applied to Yunlin; no Yunlin set is published) and chx_lib.world's heterogeneous loading
G0_YR_K = (0.0, 1.0)
G0_YR_RHO = (0.0, -0.83)
G0_YR_REPORT_NOISE = 1.5                         # report-only cell: noise x 1.5 at k = 1
G0_PARAMS_FROM = "dataprep_staging/n2c_reading/synth_n2c.py"

# ----------------------------------------------------------------------------- frozen code
#: every file the declaration hashes (frozen_code); every model / G0 / baselines output
#: records these shas and the declaration refuses unless they equal frozen_code
FROZEN = (f"{X}/bc11_const.py", f"{X}/bc11_lib.py", f"{X}/build_l1pre_v2.py",
          f"{X}/page_run.py", f"{X}/baselines_bc11.py", f"{X}/g0_bc11.py",
          f"{X}/score_bc11.py", f"{X}/run_bc11.sh", f"{X}/prereg_bc11.py",
          f"{X}/attempts_bc11.py",
          f"{X}/tests/test_bc11.py",
          f"{CHX}/avail_bc11.py", f"{CHX}/avail_lib.py", f"{CHX}/avail_l1pre.py",
          f"{CHX}/bc_lib.py", f"{CHX}/build_l1pre.py", f"{CHX}/chx_const.py",
          f"{CHX}/chx_io.py", f"{CHX}/chx_lib.py", f"{CHX}/hindcast.py",
          "dataprep_staging/n2_policy/bootstrap_scorer.py", G0_PARAMS_FROM,
          "dataprep_staging/fv_decidability_v2/stat_fv.py",
          "dataprep_staging/fv_decidability_v2/osse_fv2.py",
          "dataprep_staging/fv_decidability/osse_fv.py",
          "dataprep_staging/h1_heads_qc/screens.py", "dataprep_staging/h1_heads_qc/config.py",
          "hydrophysics/twin/hybrid.py", "hydrophysics/twin/coverage.py",
          "hydrophysics/twin/drift_diag.py")


def code_shas() -> dict:
    """sha256 of every frozen file in this checkout (recorded by every output)."""
    import hashlib

    root = HERE.parents[1]
    out = {}
    for p in FROZEN:
        f = root / p
        out[p] = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else "missing"
    return out
