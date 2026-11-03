"""Frozen constants of round-8 rank #17, E02z.level_years23: E02c's strict level rule (beat CL
AND CLT, per county, baselines from pre-origin months only) on E02y's year-2/3 windows of the
frozen head forecast cf2c7154 (2027-11..2028-10 and 2028-11..2029-10), on new unit ids.

Every number the registration, the decidability engine, the scorer and the tests share lives
here. The registration hashes this file (frozen_code): changing anything after it is written
needs a new version id (the retry rule never lets a gate be weakened).

Governance. User ruling R8.14 (results/dataprep/rulings/user_rulings_2026-11-03.json,
answered 2026-11-03T01:26Z) confirms that E02z is covered by R7.heads_years23 (sixth set,
user_rulings_2026-11-01b.json) through the registered GOV.heads_window_product.v1, without
editing common/uses.py. The MANIFEST precondition is E02z's own (the N2CS precedent, lead
ruling B1): E02c's frozen ``score_e02c.product_grants_scorer`` AND
``heads_window.covering_rule(m, window)`` of the GOV module pinned at sha 9a953039.

Reading (FM.origin_memory.v1's start-memory note): at leads 59-82 months the frozen run's
head level is the model's average-weather (climatology-forced) scenario, its parameters and
datum; E02z tests that level, not the 2022-12 observation update. User ruling R8.7 splits C30
into C30-scen (the physics average-weather scenario, N) and C30-fc (the first-year forecast
of record, D via E04): E02z maps to C30-scen only.

Nothing in this module reads data.
"""
from __future__ import annotations

X = "dataprep_staging/e02z"
WP = "E02z"

# ----------------------------------------------------------------------------- identity
ID = "E02z.level_years23.v1"
TITLE = ("E02z: E02c's strict level rule per county (beat CL and CLT, pre-origin baselines) "
         "on years 2 and 3 of the frozen head forecast cf2c7154 (2027-11..2028-10, "
         "2028-11..2029-10): the model's average-weather level")
WINDOWS = {"B": ("2027-11", "2028-10"), "C": ("2028-11", "2029-10")}
TEST = {"B": "E02zP.level_2027-11", "C": "E02zP.level_2028-11"}
UNIT = {"B": "E02z:heads_2027-11_2028-10_wells158", "C": "E02z:heads_2028-11_2029-10_wells158"}
SCOPE = {w: f"heads:{a}..{b}" for w, (a, b) in WINDOWS.items()}
N_WELLS = 158
COUNTIES = ("Yunlin", "Changhua")                 # station code 09 / 07 (score_e02c.COUNTY)
#: register before this instant (2027-10-31 end of day UTC) and before any window product
REG_DEADLINE = "2027-11-01T00:00:00Z"
#: E02y's scoring dates (one session per window, E02z after E02y's window report)
NOT_BEFORE = {"B": "2029-01-15T00:00:00Z", "C": "2030-01-15T00:00:00Z"}
NO_LATER_THAN = {"B": "2029-02-01T00:00:00Z", "C": "2030-02-01T00:00:00Z"}   # end of 01-31

# ----------------------------------------------------------------------------- E02y (rows)
E02Y_ID = "E02y.years23_2027-11_2029-10.v1"
E02Y_SHA = "4bf7d91f928c8017176c5f53a60abffe5f86a9cc93386d521e5fd8bc7e0c6d12"
E02Y_UNIT = {"B": "E02y:heads_2027-11_2028-10_wells158",
             "C": "E02y:heads_2028-11_2029-10_wells158"}
#: E02y's score-look test ids per window (its scorer logs relative, then absolute)
E02Y_TESTS = {"B": ("E02yP.rel_2027-11", "E02yP.abs_2027-11"),
              "C": ("E02yP.rel_2028-11", "E02yP.abs_2028-11")}
E02Y_WINDOW_KEY = {"B": "B", "C": "C"}            # score_e02d.py --window
#: the frozen forecast's absolute registrations of years 2 and 3 (E02y artefacts.absolute)
ABS_DIRS = {"B": "results/twin/prospective_2027-11_y2",
            "C": "results/twin/prospective_2028-11_y3"}
ABS_SHA = {
    "B": {"manifest.json": "27dd28aaf3c1804af3326111ab520811de84c95bc09f5149c2dde4d39a01b5d9",
          "predictions.csv": "49e23b7e7f667f7aef7a64c4cbb5228863a2d1c7631b812f9c00516ee5930653",
          "protocol.json": "9d23c03144cb82dcbcd11028c1da143c024b394406dd79dec46fcb0dc2f47810"},
    "C": {"manifest.json": "0d129e2c3bc571b73170667c1a03b0f8ebe5135999b54838374341937476616e",
          "predictions.csv": "c494fa040e487b6912edd30a2facc99746e853125e7e61d4c3f94fbd4f473eaa",
          "protocol.json": "34b718f88f80b3776f0ccf3c12ebeeb46e245beaa49787e4b80d69c9082a6afe"}}
MODEL_SHA = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
#: a lead-written record naming E02y and a window with this verdict replaces E02y's report
NOT_SCORABLE = "not_scorable"

# ----------------------------------------------------------------------------- E02c (rule)
E02C_ID = "E02c.prospective_2026-11.v1"
E02C_REG_SHA = "cc5b970f4bd009cd1f84ae3c702e7d057bd6325f724c38ecdd710ab2a62dd468"
E02C_SCORER = "dataprep_staging/e02c/score_e02c.py"
E02C_SCORER_SHA = "1c4e092a43ae42c9e57ec96ebf4120b0911155d1f7d3216b27c42a3f41b6bc3d"
E02C_EXPORT = "dataprep_staging/e02c/export_baselines.py"
E02C_EXPORT_SHA = "a9ee3610a042d6122a30be73ef14c0f0d373020064e804faf2654f9922a6be94"
#: E02c's frozen well table (county, TWD97 5 km block, forward cell and layer): model and
#: station metadata only, no head value
E02C_WELLS = "results/dataprep/reports/E02c/baselines/wells_158.csv"
E02C_WELLS_SHA = "8ea086330f8a9ecf5ff76313c9dc3e617e8e500775d70d7cc180f077340ac7d3"

# ----------------------------------------------------------------------------- GOV, rulings
GOV = "dataprep_staging/gov_heads_window/heads_window.py"
GOV_ID = "GOV.heads_window_product.v1"
GOV_SHA = "9a953039d75fc812fa3d537c80d17cd8c8fa9f90f2b4f70af5d90ddcfa7a4f37"
GOV_REG_SHA = "141c012899a67c69eecfd630a6e9a92de3c057f5cfa2d67412adf42608167855"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULINGS_SHA = "ec610b692835d955141e9261e6d6046ee1984f39891bcf1c2e77591aad1c69eb"
RULINGS_R7 = "results/dataprep/rulings/user_rulings_2026-11-01b.json"
RULINGS_R7_SHA = "5f53909bcc9154d8a22da902402fcf41264facccd1bf7cf29eb28c48c8f6b6ab"
FM_ID = "FM.origin_memory.v1"
FM_SHA = "e2d261fb6cbe080d51048241ecce3c58b0f1d8b9cba842d8dfd899cabacbc28d"

# ----------------------------------------------------------------------------- the rule
#: E02c's T-level thresholds, unchanged; the minimum is per county (E02c's county envelope
#: MIN_K_COUNTY 20, the same minimum E02y's per-county gate uses), not E02c's pooled 100
MIN_MONTHS = 6
MIN_WELLS_COUNTY = 20
#: review S5 (lead): a county also needs >= 10 distinct 5 km blocks among its scored wells
#: (the bootstrap's clusters); below it the county is not evaluable. Only tightens the rule.
MIN_BLOCKS_COUNTY = 10
BOOT_B = 2000
BOOT_CI = 0.90
BLOCK_KM = 5.0

# ----------------------------------------------------------------------------- baselines
BASE_DIR = "results/dataprep/reports/E02z/baselines"
BASE_FILE = "baselines_e02z.csv"
BASE_FILES = (BASE_FILE, "summary.json", "verify.json")

# ----------------------------------------------------------------------------- outputs
REPORT_DIR = f"results/dataprep/reports/{ID}"
SCORE_REPORT = {w: f"{REPORT_DIR}/report_{w}.json" for w in WINDOWS}
SCORE_ATTEMPTS = f"{REPORT_DIR}/score_attempts.jsonl"
DECIDE_REPORT = f"{REPORT_DIR}/decidability/report.json"
DECIDE_ATTEMPTS = f"{REPORT_DIR}/decidability/attempts.jsonl"
RUNS = "results/twin_runs/e02z"
RUN_ATTEMPTS = f"{RUNS}/attempts.jsonl"

# ----------------------------------------------------------------------------- decidability
#: FV v1's design-time noise inputs (registered by FV.decidability.v1; per-well AR(1) sd and
#: lag-1 of the page model's pre-origin fitted residuals; availability mask 2012-22)
FV_INPUTS = "results/dataprep/reports/FV.decidability.v1/inputs"
FV_INPUT_SHA = {
    "noise_ar1.csv": "73965a08914149af2132c96f617268701277b49411cf17cfbf57d41fbeaa9d55",
    "obs_mask.csv": "605a35de5d07cab40cef3a1e86fae719502ff3afcd0f6ea36310ccf5425d0c89"}
FV1_ID = "FV.decidability.v1"
FV1_SHA = "cf2adf0d0c9b9fe5f70f5e6ee6cd1bc133964811c2b944eaf2cce344b4d3f61d"
FV2_ID = "FV.decidability.v2_postfailure"
FV2_SHA = "f7e35a17482cce307c6be84013f9a2919f16f4be66e9a01cd7eeacbd08fb49d4"
FVSTAT2_ID = "FV.stat.v2"
FVSTAT2_SHA = "a15e468d1292bb9d170fa981d96ee52bfadc80051c867496dbdb94ff2c8de7e6"
FVSTAT2_SEED = 37355759                     # FV.stat.v2's registered design seed
#: the truths of FV v1/v2 (distinct-recipe theta sets on disk) and the same-code reference
TRUTHS = (("R0", "stage3_datum_gate_cwa"), ("Q0", "stage3_datum_gate_q0"),
          ("R1c", "stage3_datum_gate_ccz"), ("X5R1", "stage3_datum_gate_x5r1"))
TAGS = tuple(t for t, _ in TRUTHS)
ZR0 = "ZR0"
ZR0_DIR = "stage3_datum_gate"
ZR0_THETAS = ("stage3_theta.json", "stage3_fold_thetas.json", "stage3_posterior_tn.json")
COLUMN = "coupled_leveling/vep_zonal_leveling.json"
#: the forward axis: 2012-01 .. 2029-10 (origin 2022-12 = index 131, horizon 82)
AXIS_START = "2012-01"
T_FIT = 132
ORIGIN_IDX = T_FIT - 1
HORIZON = 82
T_RUN = T_FIT + HORIZON                     # 214
FWD_WIN = {"B": (190, 202), "C": (202, 214)}
N_WIN = 12
#: FV v1's commit (its ZR0 and truth forwards); E02z's forwards rerun it at horizon 82,
#: baseline only, so each hindcast must equal FV v1's to 0.0 m at the 158 wells
WT_HEAD = "5f3af18"
WT_NAME = "HydroPhysicsAI-e02zrun"
FWD_FLAGS = ("--horizon 82 --gain 1.0 --ic-members 0 --update-method innovation "
             "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
             "--column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members none")
HINDCAST_TOL_M = 0.0


def fwd_npz(tag: str) -> str:
    return ("results/twin_forward/e02z_zr0.npz" if tag == ZR0
            else f"results/twin_forward/e02z_truth_{tag}.npz")


#: review S4: FV v1's gate-E reference forwards, pinned by the npz_sha256 that FV v1's
#: results/twin_runs/fv_decidability/attempts.jsonl records (attempts 1-5, 2026-11-01; equal to
#: osse_fv2.V1_FORWARD_SHA)
FV1_FWD_SHA = {
    "ZR0": "a1833d5f861817223ebafb5a651444239c8877d21a4d4752e91dbbbd356172cf",
    "R0": "2b1363bec1581e76bf98f422a30a023cbc8d21f4dabba82256b5670d6d019fcf",
    "Q0": "1a6a1ef7d1c9d1c6fff21f0a2e26bd2fb8ad36e073ec36f03e96f6bde99d74b1",
    "R1c": "faff857e7ffae99d9cbcf5ed440546482c9d9d1276cf6fad5871a5426b87fdf1",
    "X5R1": "9d8e6311d5e1ece3dc54a4c40719a13e18b1a440cfe87770707649276c4c9566"}
#: review Q3 (lead): no CPU fallback
CUDA_ONLY = ("the forwards run on cuda (FV v1's device); if the 0.0 m gate-E hindcast identity "
             "cannot be reproduced (a driver, GPU or platform change), the decidability "
             "reading is 'not_run' (its declared suffix and grade); there is no CPU fallback")


def fv1_npz(tag: str) -> str:
    """FV v1's forward of the same run (horizon 58), the hindcast-identity reference."""
    return ("results/twin_forward/fv_zr0.npz" if tag == ZR0
            else f"results/twin_forward/fv_truth_{tag}.npz")


K_TRUTH = 250
K_NULL = 500
P_WEAK = 0.5
MC_HALF_WIDTH = 0.07
STREAM_NOISE, STREAM_NULL, STREAM_SELFCHECK = 71, 72, 73       # FV used 1-4, FV.stat 31-43
N_SELFCHECK = 2
SMOKE_SEED = 17170017                     # E02z engineering smokes only
#: seeds a registration may not use: previewed, smoked or already registered elsewhere
REFUSED_SEEDS = {
    20261101: "FV.decidability.v1/v2's seed (and N2's pinned bootstrap seed)",
    20261031: "E02c's registered bootstrap seed (E02z must draw fresh resamples)",
    20261103: "FV.stat.v1's seed (previewed at K = 8 before its registration)",
    77770001: "FV.stat.v1's smoke seed",
    51120017: "FV.stat.v2's smoke seed",
    FVSTAT2_SEED: "FV.stat.v2's registered seed",
    SMOKE_SEED: "E02z's own engineering smoke seed",
}

# ----------------------------------------------------------------------------- RC5, wording
RC5 = {"frozen_run_rmse_m": 2.048, "cl_rmse_m": 1.667, "record": "2023-26",
       "pooled": "pooled over both counties (equal-well RMSE); not per county",
       "source": "the published 2023-26 record (round-8 audit RC5; ledger F2 disclosure)",
       "use": "quoted to pre-word the expected outcome only; it chose no gate, threshold, "
              "baseline, well or seed, and no 2023-26 residual is read or recomputed"}

WORDING = {
    "level_supported": (
        "{county}: the frozen model's average-weather level scenario (cf2c7154, 2022-12 start) "
        "beat climatology and climatology-plus-trend on {first}..{last} (E02z, prospective; "
        "an average-weather scenario, not a forecast)"),
    "level_not_supported": (
        "{county}: the frozen model's average-weather level scenario was not better than both "
        "simple pre-origin baselines on {first}..{last} (E02z, prospective); expected from the "
        "published 2023-26 record (pooled: frozen run 2.048 m against climatology 1.667 m), so "
        "the physics head level after 2023-11 stays an average-weather scenario, not a "
        "forecast"),
    "level_not_evaluable": "{county}: undecided on {first}..{last} ({why})",
    "engineering_failure_after_look": (
        "{county}: undecided on {first}..{last} (engineering failure after the score look; "
        "disclosed; never re-scored)"),
}
#: appended by the decidability reading (per county and window); 'not run' if no report
DECIDE_SUFFIX = {
    "weak_by_design": ("; synthetic decidability (E02z, first attempt): a right-ish model "
                       "passes this rule with probability {p:.2f}, so a fail does not refute "
                       "the model's average-weather level"),
    "not_shown_weak": ("; synthetic decidability (E02z, first attempt): a right-ish model "
                       "passes this rule with probability {p:.2f} (optimistic: average-weather "
                       "truths, wells independent, no gaps; not a strict bound)"),
    "not_evaluable": "; synthetic decidability not evaluable ({why})",
    "not_run": "; synthetic decidability not run before scoring",
}
#: the round-8 plan's grading of a decided row by its decidability reading (bookkeeping only)
GRADE = {"not_shown_weak": 0.7, "weak_by_design": 0.5, "not_evaluable": 0.5, "not_run": 0.5}
NEVER = ["calibrated", "校準", "forecast of record"]
