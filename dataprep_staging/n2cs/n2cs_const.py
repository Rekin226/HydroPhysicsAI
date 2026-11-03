"""Frozen constants of round-8 rank #6, N2CS: the prospective, one-sided Changhua first-year
anomaly-response size-class test on N2C's 74 wells over 2026-12..2027-10, gated by an OSSE.

Every number the OSSE record or the registration freezes lives here, so the builder, the
OSSE engine, the draft registration, the scorer and the tests read one definition. The OSSE
record and the registration hash this file: changing anything after either is written needs a
new version id (the retry rule never lets a gate be weakened).

User ruling R8.2 (results/dataprep/rulings/user_rulings_2026-11-03.json, answered
2026-11-03T01:10Z): approved for the 11-month window only, the 24-month variant declined;
C20/C21 size clause N -> N+D plus new row E27; C13a, E20, the N2C clauses and the F < 0.5 cap
untouched; no correction factor follows (ruling 2026-10-29 (4)).

Nothing in this module reads data.
"""
from __future__ import annotations

X = "dataprep_staging/n2cs"
WP = "N2"

# ----------------------------------------------------------------------------- identity
ID = "N2CS.changhua_size_class.prospective_2026-12.v1"
TEST = "N2CS.changhua_first_year_size_class"
UNIT = "N2CS:heads_changhua_2026-12_2027-10"
SCOPE = "heads:2026-12..2027-10"
WINDOW = ("2026-12", "2027-10")
N_WINDOW = 11
#: forward month index of the window (dates 2012-01 .. 2032-12, 252 months; 2026-12 = 179)
FWD_WINDOW = (179, 190)
N_T = 252
REG_DEADLINE = "2026-11-30T16:00:00Z"
NOT_BEFORE = "2028-01-15T00:00:00Z"
NO_LATER_THAN = "2028-02-01T00:00:00Z"          # end of 2028-01-31 (N2C's session end)

N2C_ID = "N2C.changhua_response.prospective_2026-11.v1"
N2C_SHA = "76853696bcd00c97c33c4dff1ac4d01a14e62a241a0b02938a76af6f20c44ee1"
N2C_UNIT = "N2C:heads_changhua_prospective_2026-11_2027-10"
N2C_TEST = "N2CP.heads_changhua_response"
#: N2C's step-(2) report (the reports/<id>/report.json convention); its 'verdict' field must
#: be one of the pinned scorer's VERDICTS
N2C_REPORT = f"results/dataprep/reports/{N2C_ID}/report.json"
N_WELLS = 74

RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULINGS_SHA = "ec610b692835d955141e9261e6d6046ee1984f39891bcf1c2e77591aad1c69eb"
RULING_ID = "R8.2"
RULING_FACTOR = "ruling 2026-10-29 (4)"
WORDING = "first-year anomaly response size class; no correction factor follows " \
          "(ruling 2026-10-29 (4))"

# ----------------------------------------------------------------------------- model, forward
#: R1c (N2C's primary model): stage3_datum_gate_ccz main parameter set + its column
R1C_RUN = "results/twin_runs/stage3_datum_gate_ccz"
R1C_THETA = f"{R1C_RUN}/stage3_theta.json"
R1C_COLUMN = f"{R1C_RUN}/coupled_leveling/vep_zonal_leveling.json"
R1C_THETA_SHA = "03aa7563caa5fab3de0cbcce87118be5221de44d024c31305af071704743d98b"
R1C_COLUMN_SHA = "63e53cfd69786c45fa70edc8d50374f93e8e4fdad3ac5d561c5a0844e1e94bff"
#: N2C's registered code commit; the clean detached worktree CHX already runs it from
WT = "$HOME/HydroPhysicsAI-chx"
WT_HEAD = "1b08fb6"
#: N2C's registered forward flags, unchanged (CPU, main set, one IC member)
FWD_FLAGS = ("--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation "
             "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
             "--column-hpc0-fast-days 365 --device cpu --save-members none")
FWD_MEMBER = "stage3_datum_gate_ccz"
FWD_OPTIONS = {"column_heads": "free", "update_method": "innovation",
               "apex_hold": "calibrated", "restart_taper_km": 5.0,
               "column_hpc0_fast_days": 365.0, "save_members": "none"}
ORIGIN, HORIZON = 131, 120                      # 2022-12; 120 months
RECORDED_START = "2023-01"

LOG = "results/twin_runs/n2cs"
ATTEMPTS = f"{LOG}/attempts.jsonl"
FILES = f"{LOG}/r_files"                        # pre-gate unit files + OSSE replays
UNIT_CH = f"{FILES}/unit_changhua.npz"
UNIT_YL = f"{FILES}/unit_yunlin.npz"
REPLAYS = f"{FILES}/replays"
FWD_A = "results/twin_forward/n2cs_osse_a_r1c"  # baseline, UCH, UYL, RCH2012..RCH2018
FWD_B = "results/twin_forward/n2cs_osse_b_r1c"  # baseline, RYL2012..RYL2018
PANEL = f"{LOG}/osse_panel.npz"
PANEL_RECORD = f"{LOG}/osse_panel.json"
OSSE_OUT = "results/dataprep/reports/N2CS_osse"
SCORE_OUT = f"results/dataprep/reports/{ID}"
#: the scoring-time forward (built and run in the 2028 session, never before)
SCORE_ARMS = ("NCSCH", "NCSYL")

PREGATE_TOL_M = 1e-6
PREGATE_ARMS = ("UCH", "UYL")
REPLAY_YEARS = (2012, 2013, 2014, 2015, 2016, 2017, 2018)

# ----------------------------------------------------------------------------- inputs
COUNTY_MAP = "results/dataprep/zone_county_map/v2/zone_county_map.npz"
R1C_COUNTY_SHA = "6ca7840401a5f77ace183acc605f38c936b79de94df54bec8b1134bbfd86529e"
WELLS = "dataprep_staging/x5_class_share/wells_frozen.json"
WELLS_SHA = "fc2d8257dbe729238a96faf96f2e0b8e3ff82facc7faddb03794b453c2fffcd6"
WELLS_XY = "results/twin_runs/stage3_datum_gate_ccz/stage3_wells.csv"
ZR_DIR = "results/twin/prospective_2026-11_n2c_r1c"     # N2C's registered R1c ZR
#: OSSE forcing variability: the pre-origin census only (X1 forbids design_time_evidence)
V3 = "results/dataprep/pump_electricity/preorigin/v3"
V3_TABLE = "county_class_monthly.parquet"
V3_USE = "design_time_evidence"
#: the scoring-time recorded forcing (X1, allowed use forcing_recorded_arm), N2C's rule
X1_DEFAULT = "results/dataprep/electricity_x1_tpc/full/v1/county_monthly.parquet"

# ----------------------------------------------------------------------------- pinned code
SCORER = "dataprep_staging/n2_policy/bootstrap_scorer.py"
SCORER_SHA = "f8990d6182b7b080608f77b7e5f7a449ea722f97f845cd83478997cdc9ae75bd"
BUILD_R_BOTH = "dataprep_staging/n2_policy/build_r_both.py"
BUILD_R_BOTH_SHA = "7d498b9c8b4054b9813a4e7460ae9fe3e8fefd9a6e30dc683fc89fb89feac734"
BUILD_R = "dataprep_staging/n2_policy/build_r.py"
BUILD_R_SHA = "9ed2793fb562c8c58d73ea70facdfce12b1d654921ada39013f718f90cd8ab13"
REPLAY = "dataprep_staging/x5_changhua_lever/replay.py"
REPLAY_SHA = "898279457d490479ad3b9e0468c36ae8d4f6f2ed565f6396af8f769dfc68a8ad"
SYNTH = "dataprep_staging/n2c_reading/synth_n2c.py"     # the published noise parameters
SYNTH_SHA = "3bd79f713e9ad0bc57a5415c4ccf89b95472cd6fcc8124de7be6d88c2ff993c8"
GOV = "dataprep_staging/gov_heads_window/heads_window.py"
GOV_ID = "GOV.heads_window_product.v1"            # registered 2026-11-03T01:46Z (binds GOV_SHA)
GOV_SHA = "9a953039d75fc812fa3d537c80d17cd8c8fa9f90f2b4f70af5d90ddcfa7a4f37"
#: the MANIFEST precondition is N2CS's own: E02c's frozen product_grants_scorer (the
#: prospective_scorer grant, flags 0-1) AND heads_window.covering_rule(m, WINDOW); no GOV entry
#: is added (GOV is registered); both files are in N2CS's frozen_code, checked at scoring
E02C_SCORER = "dataprep_staging/e02c/score_e02c.py"
E02C_SCORER_SHA = "1c4e092a43ae42c9e57ec96ebf4120b0911155d1f7d3216b27c42a3f41b6bc3d"
#: the heads window table's column contract (E02c's registered observation columns
#: station_id, date, observed_m; flag optional, flags 0-1 only), a parquet table listed in the
#: product's MANIFEST outputs; checked from the schema and the id/date/flag columns BEFORE
#: assert_can_score (no observed_m value is read before it)
HEADS_CONTRACT = {
    "format": "parquet, listed in the window product's MANIFEST outputs",
    "columns": {"station_id": "string (8 characters, leading zeros kept)",
                "date": "string YYYY-MM or YYYY-MM-DD, or timestamp; parsed to its month",
                "observed_m": "float (metres)",
                "flag": "integer, optional; only flags 0-1 are scored"},
    "source": "E02c.prospective_2026-11.v1 observation columns (score_e02c.join_obs: "
              "station_id, date, observed_m)",
    "pre_look_checks": ["schema (pyarrow.parquet.read_schema): the three columns present, "
                        "station_id a string type, observed_m a floating type, flag (if "
                        "present) an integer type",
                        "station_id values are strings; none of the 74 ids appears with its "
                        "leading zeros stripped",
                        "every date of the 74 wells parses to a month",
                        "at least one scorable row (one of the 74 ids, a window month, flag "
                        "0-1)"],
    "after_look": "any exception after assert_can_score writes the final report "
                  "'engineering_failure_after_look' (disclosed; never re-scored)"}

# ----------------------------------------------------------------------------- estimator
#: at most three pre-declared estimator forms of the twoway regression of e = obs - ZR on
#: (c_CH, c_YL); the OSSE chooses among them by CHOICE_RULE only
FORMS = ("twoway", "twoway_level", "twoway_fd")
FORM_TEXT = {
    "twoway": "per-well intercept and linear trend + month-of-sample effects (CHX's primary "
              "form), then OLS of e~ on (c_CH~, c_YL~)",
    "twoway_level": "per-well intercept + month-of-sample effects, then OLS of e~ on "
                    "(c_CH~, c_YL~)",
    "twoway_fd": "per-well first differences of consecutive scored months, then per-well "
                 "intercept (a level trend) + month-of-sample effects on the differences, then "
                 "OLS of de~ on (dc_CH~, dc_YL~)"}
ESTIMAND = "the Changhua coefficient beta_CH of the two-contrast regression"
INTERVAL = ("Frisch-Waugh: c_CH~ is residualised on c_YL~ over the scored rows; beta_CH and "
            "its 90 % interval are the pinned scorer's beta_response(e~, c_CH~ perp, wells, "
            "blocks) (per-well slope, 5 km block cluster bootstrap, B 2000, seed 20261101, 5 / "
            "95 % quantiles); its per-well demeaning is a no-op on these residuals; the "
            "fixed-effect projection (month effects included) and the partialling are "
            "computed once on the scored rows and held fixed across draws")
MIN_WELL_MONTHS = 6                             # a well enters with >= 6 scored months
SIZE_X = 0.5                                    # 'overstated >= x2' <=> beta_CH < 0.5
READINGS = ("overstated_x2", "not_overstated_x2", "inconclusive", "not_informative",
            "not_evaluable", "engineering_failure_after_look")
READING_TEXT = {
    "overstated_x2": "first-year anomaly response overstated >= x2 (upper 90 % bound of "
                     "beta_CH < 0.5)",
    "not_overstated_x2": "first-year anomaly response not overstated by x2 (lower 90 % "
                         "bound of beta_CH > 0.5)",
    "inconclusive": "first-year anomaly response size class inconclusive (90 % interval of "
                    "beta_CH contains 0.5)",
    "not_informative": "not informative (the model-only informative gate failed)",
    "not_evaluable": "not evaluable (no heads window product granting prospective_scorer "
                     "for 2026-12..2027-10)",
    "engineering_failure_after_look": "engineering failure after the score look (the reading "
                                      "could not be computed; disclosed; never re-scored)"}
#: model-only informative gate at scoring time (contrast panel of every well and window
#: month, no head), under the CHOSEN form (lead ruling Q2): RMS of the Frisch-Waugh regressor
#: (c_CH~ residualised on c_YL~) >= 0.05 m, |corr(c_CH~, c_YL~)| <= 0.8 as an additional check,
#: plus the pinned gate's >= 5 clusters and >= 6 recorded Changhua window months
MAX_ABS_CORR = 0.8
INFORMATIVE_RMS_M = 0.05

# ----------------------------------------------------------------------------- OSSE
OSSE_DECL_ID = "N2CS.osse_design.v1"            # prereg3 declaration, registered BEFORE files
K_GRID = (0.2, 1.0 / 3.0, 0.5, 1.0, 1.2)
K_TRUE = 1.0
K_OVER = 1.0 / 3.0
#: the canal-like omitted term g_t (unit sd over the window; corr RHO with the replay's
#: Changhua pumping anomaly), scaled by kappa (synth_n2c.PARAMS kappa_m), loading per well:
#: 'uniform' 1 (county-uniform), 'shaped' 1 + HET_MULT z_i, z_i = LOAD_RHO za_i +
#: sqrt(1 - LOAD_RHO^2) N(0, 1), za_i the standardised per-well RMS of the replay's
#: well-demeaned c_CH (CHX's heterogeneous loading). Every form carries month-of-sample
#: effects, so the county-uniform term is absorbed EXACTLY by construction (as a real
#: county-uniform term would be); the binding confounder test is 'shaped' (lead ruling S4).
RHO = -0.83
LOAD_RHO = 0.5
HET_MULT = 1.0
CONFOUNDERS = {"uniform": {"rho": RHO, "loading": "uniform"},
               "shaped": {"rho": RHO, "loading": "shaped", "load_rho": LOAD_RHO,
                          "het_mult": HET_MULT},
               "none": {"rho": 0.0, "loading": "uniform", "kappa_mult": 0.0}}
#: G0 variants (lead ruling Q3): the two confounders on the full panel, and the shaped
#: confounder on a 25 % thinned panel (each well-month missing with probability 0.25) and on a
#: whole-well dropout panel (15 of the 74 wells removed)
DROP_WELLS = 15
MISS_FRAC = 0.25
GATE_VARIANTS = {"uniform": {"conf": "uniform"},
                 "shaped": {"conf": "shaped"},
                 "shaped_thinned25": {"conf": "shaped", "miss_frac": MISS_FRAC},
                 "shaped_drop15": {"conf": "shaped", "drop_wells": DROP_WELLS}}
#: world: e = k c_CH + k_YL c_YL + block AR(1) + well AR(1) + kappa load_i g_t, k_YL = k
#: (one scale error of the model's response) unless a report cell says otherwise
REPORT_CELLS = (
    {"k": 0.2, "conf": "none"}, {"k": 1.0 / 3.0, "conf": "none"},
    {"k": 0.5, "conf": "none"}, {"k": 1.0, "conf": "none"}, {"k": 1.2, "conf": "none"},
    {"k": 1.0 / 3.0, "conf": "shaped", "k_yl": 1.0},
    {"k": 1.0 / 3.0, "conf": "shaped", "k_yl": 3.0},
    {"k": 1.0, "conf": "shaped", "k_yl": 3.0},
    {"k": 1.0 / 3.0, "conf": "shaped", "kappa_mult": 2.0},
)
REPLICATES = 2000
#: G0 uses k = 1 and k = 1/3 only (the other k are report-only); every condition under EVERY
#: gate variant; P(not overstated | k = 1) is unconditional (a not-informative world counts as
#: a miss); the two false rates are the stricter of unconditional and given-informative
G0 = {"p_not_overstated_k1_min": 0.8, "p_not_overstated_k1_3_max": 0.2,
      "p_overstated_k1_max": 0.1}
CHOICE_RULE = (
    "among the forms that pass G0 under every gate variant: the largest min over gate "
    "variants of P(not overstated | k = 1); exact tie -> the smaller max over gate variants "
    "of P(not overstated | k = 1/3); still tied -> the declared order twoway, twoway_level, "
    "twoway_fd. No form passes -> the OSSE fails: the N line reads 'size class not decidable "
    "prospectively (OSSE power X)', X = the largest min over gate variants of P(not "
    "overstated | k = 1) over the three forms, and it is bookkeeping, not evidence")
#: replay quality (lead ruling S8): at most this fraction of a replay's 11 window SOURCE months
#: may be filled with r = 1 (no kept census row), per county; over it the build refuses
MAX_FILLED_FRACTION = 0.2
PLUMBING_TOL = 1e-9
SEEDING = ("numpy SeedSequence(world_seed).spawn(n_gate_cells + n_report_cells): children in "
           "cell order (k outer in K_GRID order, gate variant inner in GATE_VARIANTS order, "
           "then REPORT_CELLS in order); replicate r of a cell uses replay r mod 7 and that "
           "cell's generator in order: block AR(1) (n_blocks x 11), well AR(1) (74 x 11), "
           "confounder xi (11), loading z (74), then the missing mask (74 x 11) in a "
           "miss_frac cell or the dropped-well permutation (74) in a drop_wells cell; every "
           "form is scored on the same world; the bootstrap is the pinned scorer's (seed "
           "20261101) in every world")
