"""Frozen constants of round 7 ranks 13 and 18 (builder chB): I1.CHX.v1 and I1.CH_backcast.v1.

Every number a registration freezes lives here, so the declaration scripts, the runners, the
scorers and the tests read one definition. Changing anything here after a registration changes
this file's sha256, which the registrations hash (frozen_code): a changed design needs a new
version id.

Nothing in this module reads data.
"""
from __future__ import annotations

X = "dataprep_staging/ch_x"
WP = "I1"

# ----------------------------------------------------------------------------- I1.CHX.v1
CHX_ID = "I1.CHX.v1"
CHX_TEST = "I1CHX.direction_R1c_ext23"
CHX_UNIT = "I1CHX:ext_changhua_23"
CHJ_TEST = "I1CHX.chj_externals"                 # secondary (CH-J), same session
CHX_UNIT_YL = "I1CHX:ext_yunlin"                 # Yunlin externals: CH/YL ratio only
CHX_SCOPE = "heads:2013-01..2022-12"
CHX_WINDOW = ("2013-01", "2022-12")
RUN_MONTHS = ("2012-01", "2022-12")            # the hindcast axis (132 months)
PATTERN_YEAR = 2012                            # ZR: Changhua at its 2012 monthly pattern

#: R1c, the model N2C scores (stage3_datum_gate_ccz, main parameter set, free run)
R1C_RUN = "results/twin_runs/stage3_datum_gate_ccz"
R1C_THETA = f"{R1C_RUN}/stage3_theta.json"
R1C_COLUMN = f"{R1C_RUN}/coupled_leveling/vep_zonal_leveling.json"
#: Q0 (CH-J's frozen parent; Yunlin and every shared parameter pinned at its values)
Q0_THETA = "results/twin_runs/stage3_datum_gate_q0/stage3_theta.json"
#: the county polygons and R1c's county map: the Changhua cells (R1c's map, sha verified
#: against R1c's meta) and the Yunlin cells of the placebo arm (cell centroid in 雲林縣)
COUNTY_GPKG = "chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg"
COUNTY_MAP = "results/dataprep/zone_county_map/v2/zone_county_map.npz"
YUNLIN_ZH = "雲林縣"

#: the frozen clean detached worktree the hindcasts run in (N2C's registered code commit, at
#: which X5.changhua_lever_model.v1 and X5.n2c_reading.v1 reproduced R1c's archived members)
WT_NAME = "HydroPhysicsAI-chx"
WT_HEAD = "1b08fb6"

#: N3F's export of the never-calibrated external wells (kept rule applied to pre-origin rows
#: only, before any availability look); the 23 Changhua kept wells are the unit
EXT_DIR = "results/twin/prospective_2026-11_ext"
EXT_WELLS = f"{EXT_DIR}/wells.csv"
#: the only columns of wells.csv any ch_x code reads (metadata; the file also carries
#: N3F's pre-origin datum and residual sd, which ch_x never reads)
EXT_COLUMNS = ("station_id", "county", "x", "y", "layer_code", "layer", "cell", "block5",
               "kept")
#: report-only extra column for the G0 availability mask (N3F's logged pre-origin month count)
EXT_COUNT_COLUMN = "n_fit_months"
N_EXT_CHANGHUA = 23
#: the 07100111-141 nest is one bootstrap block whatever its 5 km squares are
NEST = ("07100111", "07100121", "07100131", "07100141")
NEST_BLOCK = "nest_071001"

L1 = "results/dataprep/heads/L1_clean/v3"
L1_USE = "design_time_evidence"
ORIGIN = "2022-12"
MIN_WELL_MONTHS = 12                           # a well enters with >= 12 scored window months

#: the pinned shared N2/N2C scorer (cite, never edit)
SCORER = "dataprep_staging/n2_policy/bootstrap_scorer.py"
SCORER_SHA = "f8990d6182b7b080608f77b7e5f7a449ea722f97f845cd83478997cdc9ae75bd"
DIRECTION_VERDICTS = ("response_supported", "direction_supported_size_not")
#: the primary's fixed effects, frozen with this file (the lead may switch BEFORE registering):
#:   "twoway"    per-well intercept and linear trend + month-of-sample effects (the default:
#:               a common confounder and the common trend are absorbed exactly; the slope is
#:               identified from differences ACROSS screens in R1c's contrast);
#:   "detrended" per-well intercept and linear trend only (the judge's slope_fe(detrend=True);
#:               on the stand-in its confounded false pass is 1.0, so G0 cannot pass).
CHX_PRIMARY_FORM = "twoway"
FORMS = ("detrended", "twoway", "calmonth", "calmonth_twoway")

# ----------------------------------------------------------------------------- G0
G0_SEED = 20261107
G0_REPLICATES = 2000
G0_K = (0.0, 0.5, 1.0, 2.0)
G0_RHO = (0.0, -0.83)
G0_POWER_MIN = 0.60            # P(direction | k = 1, rho = -0.83, common trend) >= 0.60
G0_FALSE_MAX = 0.20            # max over rho of P(direction | k = 0, common trend) <= 0.20
#: the common-trend term: a hinge common to every well, 0 before TREND_BREAK and rising
#: linearly to TREND_AMP_M at the window's last month (the mid-2016 census install snapshot);
#: its sign is the one that biases the slope upward (toward a false pass)
TREND_BREAK = "2016-07"
TREND_AMP_M = 1.0
#: noise and confounder parameters are N2C's published, already declared values
#: (dataprep_staging/n2c_reading/synth_n2c.PARAMS: sigma_tot, block_share, phi, kappa)
G0_PARAMS_FROM = "dataprep_staging/n2c_reading/synth_n2c.py"
G0_PLUMBING_TOL = 1e-9
#: heterogeneous confounder loading: kappa g_t (1 + HET z_i), z_i = HET_CORR za_i +
#: sqrt(1 - HET_CORR^2) N(0, 1) drawn per replicate, za_i the standardised per-screen RMS of
#: R1c's well-demeaned contrast (a common-only confounder is absorbed exactly by the
#: two-way form, so the gate would otherwise be vacuous)
G0_HET_MULT = 1.0
G0_HET_CORR = 0.0
#: report-only cells after the grid (never gate): loading correlated with the contrast
#: amplitude, the common trend reversed, and no trend
G0_REPORT_CELLS = (
    {"k": 0.0, "rho": -0.83, "het_corr": 0.5},
    {"k": 1.0, "rho": -0.83, "het_corr": 0.5},
    {"k": 1.0, "rho": -0.83, "trend_mult": -1.0},
    {"k": 0.0, "rho": -0.83, "trend_mult": 0.0},
    {"k": 1.0, "rho": -0.83, "trend_mult": 0.0},
)

# ----------------------------------------------------------------------------- CH-J
CHJ_SEEDS = (2027, 2029)
CHJ_COUNTY_COPY_SD = 0.5
CHJ_SHARES_CHANGHUA = (0.84, 0.11, 0.025, 0.025)      # X3 measured shares, layers 0-3
Q0_PUMP_LAYER = 1                                     # Q0's pump layer (meta pump_layer)
#: Q0 places a sigmoid(pump_split_logit) share of every cell's volume in layer 0 and the
#: rest in Q0_PUMP_LAYER (meta pump_split True); CH-J keeps exactly that outside Changhua
CHJ_SW_COMPONENTS = 2                                 # deliveries, percolation
CHJ_SW_INIT = (0.15, 0.25)                            # calibrate_flow.SW_SCALE_INIT_2
CHJ_SW_LAYER = 0
CHJ_SW_FRACTION_BOUND = 0.01                          # lower bound of the Changhua fraction
CHJ_COLUMN_CAP = 0.1                                  # Changhua Ske/Skv <= 0.1
CHJ_FREEZE_REPRO_TOL_M = 1e-6
CHJ_AMP_BAND = (0.67, 1.5)
CHJ_NOHARM = {"yunlin_kfold": 0.01, "yunlin_county_corr": 0.01, "oof_leveling": 0.02,
              "county_mean_corr_ext": 0.05}
CHJ_FORWARD_WINDOW = ("2023-01", "2025-12")
CHJ_ENGINEERING_DEADLINE_DAYS = 21                    # the judge's 'about 3 weeks'

# ----------------------------------------------------------------------------- backcast
BC_ID = "I1.CH_backcast.v1"
BC_TEST = "I1CHB.direction_R1c_2010_2011"
BC_UNIT = "I1CHB:changhua_l1pre2012_2010_2011"
BC_SCOPE = "heads:2010-01..2011-12"
BC_AVAIL_ID = "I1.CH_backcast.availability.v1"
BC_AVAIL_UNIT = "I1CHB:ls_cache_rowcounts_2010_2011"
BC_RUN = ("2007-01", "2011-12")                       # spin-up 2007-01..2009-12, 36 months
BC_WINDOW = ("2010-01", "2011-12")
BC_PATTERN_YEAR = 2007                                # Changhua held at its 2007 pattern
BC_PRIMARY_FORM = "calmonth_twoway"     # well x calendar-month + month-of-sample effects
BC_MIN_WELLS = 10
BC_MIN_MONTHS = 18
BC_DAY_MIN_HOURS = 12                                 # H1a C20: a day needs >= 12 h
BC_MONTH_MIN_DAYS = 20                                # H1a C20: a month needs >= 20 days
BC_NEST_RADIUS_M = 1000.0                             # mechanism.NEST_RADIUS_M
LS_CACHE = "chou-shui-data/data/ls_cache"
LS_RAW = "gw__{sid}__raw.parquet"
STATIONS = "AMP_V2/data/fan_stations.parquet"
MLCW_STATIONS = "chou-shui-data/data/mlcw_stations.csv"
CENSUS_V3 = "results/dataprep/pump_electricity/preorigin/v3"
CENSUS_V3_FILE = f"{CENSUS_V3}/cells_by_class.npz"
L1PRE = "results/dataprep/heads/L1_pre2012"
L1PRE_PRODUCT = "heads/L1_pre2012"
L1PRE_V1 = f"{L1PRE}/v1"
BC_DIR = "results/twin_runs/chx/backcast"
L1PRE_ALLOWED = ("validator", "design_time_evidence", "initial_conditions")
L1PRE_FORBIDDEN = ("scoring_target", "calibration_target", "datum_fit", "baseline",
                   "selection", "prior", "prior_width_input", "model_design", "idw_field",
                   "forcing_frozen", "forcing_recorded_arm", "prospective_scorer",
                   "statistical_layer_history")
S0_TOL_M = 0.01
#: S0's declared forcing-swap pre-gate on 2012-22 (census preorigin v3 energy in place of
#: R1c's model_census v1): Changhua total energy ratio, and R1c's Changhua-only contrast
#: (Changhua at its 2012 pattern) at every Changhua cell and layer over 2013-22 under the two
#: energies: pooled correlation of the per-cell-demeaned contrasts and their RMS ratio
S0_SWAP_ENERGY_BAND = (0.85, 1.15)
S0_SWAP_CORR_MIN = 0.90
S0_SWAP_RMS_BAND = (0.80, 1.25)
#: report-only synthetic power of the backcast on its own panel (model only, before heads)
BC_POWER_REPS = 500
BC_POWER_SEED = 20261108

#: user rulings this work rests on (never edit a rulings file; registrations hash them)
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-01b.json"
RULING_L1PRE = "R7.l1_pre2012"
