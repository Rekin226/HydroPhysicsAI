"""X5.n2_reading.v1: synthetic operating characteristics of N2's decision rule (Yunlin), run
with the hashed shared N2/N2C scorer (dataprep_staging/n2_policy/bootstrap_scorer.py, sha256
f8990d61..., bound by X5.n2c_reading.v1) exactly as it will run in the 2028 session.

N2 (N2.policy_response.prospective_2026-11.v1) decides the page model's Yunlin head response
on 2026-11..2027-10 with the per-well-demeaned slope beta of e = obs - ZR on c = PR - ZR over
the 84 Yunlin wells, a 5 km block bootstrap 90 % interval and the rule 'response supported'
iff the interval excludes 0, beta in [1/1.5, 1.5] and RMSE(PR) < RMSE(ZR). This module
measures how often that rule passes under declared synthetic worlds; it mirrors the
X5.n2c_reading.v1 harness (dataprep_staging/n2c_reading/synth_n2c.py, imported read-only) and
changes only what Yunlin and the page model require:

    obs_syn - ZR = e = k * c_j + n + kappa * g_j     (84 Yunlin wells x 12 window months)

- c_j: the page model's contrast at each well's cell and layer (RPj - baseline of ONE CPU
  forward of the page model's main set, N2's registered command with the PR arm replaced by
  seven arms RP2012..RP2018 replaying contiguous pre-origin Yunlin in-grid electricity
  anomaly blocks of pump_electricity/preorigin/v3, used as design_time_evidence). ZR cancels
  from e, so the panel is scored with ZR = 0 and obs = e.
- n: residual noise, a county-wide AR(1) + a 5 km block AR(1) + a per-well AR(1), stationary
  from the first month, total sd SIGMA_TOT (R0's Yunlin anomaly RMSE: no published statistic
  of the page model's own residuals exists; R0 is N2's declared secondary model). Unlike
  Changhua (pairs beyond 20 km correlate -0.08, so N2C has no county-wide part), R0's Yunlin
  residual pairs beyond 20 km correlate +0.27: that median is the county-wide share, the 0-5 km
  median (0.79) the county-plus-block share. Report-only cells drop the county-wide part
  (N2C's structure, same total sd).
- g_j: an omitted canal-like term common to the 84 wells, correlation RHO with replay j's
  Yunlin pumping anomaly a_YL,j(t) (exactly, by orthogonalisation; synth_n2c.confounder).
  RHO = -0.87 is Yunlin's documented co-movement (observed regional head vs census energy
  -0.868 and vs canal deliveries +0.877), the drought-year direction that biases beta UPWARD
  (c ~ -a). KAPPA (review B1, lead ruling) is the deseasonalised UNIQUE canal share, capped at
  the county-wide residual variance and taken OUT of the county-wide noise (total variance
  stays sigma^2); g's within-window sd is the one per-well demeaning leaves of an AR(1)
  county part of marginal sd KAPPA. Report-only cells: raw kappa, kappa x0.5 / x0.25, N2C's
  noise structure, the page model's pre-origin k-fold sigma, a forcing-error cell (lambda
  0.8) and rho -0.52.

Every parameter cites a published report field (``PARAMS``); ``prereg_n2_reading.py`` checks
the cited values against the files before registration. The seed has NO default: the lead
picks it at registration and ``run`` refuses any other.

Order of the registered run (``run``): (1) exact plumbing (synth_n2c.exact_plumbing): with no
noise and kappa 0 the scorer returns beta == k for every replay; a failure is a code defect
(ENGINEERING FAILURE, no reading). (2) the low-noise plumbing cell (k = 1, rho = 0, kappa 0,
noise x0.02, 200 replicates): median beta in [0.9, 1.1], else the TERMINAL reading
'not_estimable'. (3) the grid and the report-only cells; (4) the declared wording (weak /
informative / pass_controlled_low_power / decisive, power floor 0.80).
report.json and report.md are written once (mode 'x'); every attempt is appended to
attempts.jsonl.

Usage (repository root; run_n2_reading.sh calls these):
    python dataprep_staging/x5_n2_reading/synth_n2.py replays --out <dir>   (BEFORE registration)
    python dataprep_staging/x5_n2_reading/synth_n2.py engineering           (engineering gate)
    python dataprep_staging/x5_n2_reading/synth_n2.py check                 (registered?)
    python dataprep_staging/x5_n2_reading/synth_n2.py extract               (after the forward)
    python dataprep_staging/x5_n2_reading/synth_n2.py run --seed S --prereg-id X5.n2_reading.v1
    python dataprep_staging/x5_n2_reading/synth_n2.py smoke --out <scratch dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO / "dataprep_staging" / "n2_policy"),
           str(REPO / "dataprep_staging" / "n2c_reading"),
           str(REPO / "dataprep_staging" / "x5_changhua_lever"),
           str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bootstrap_scorer as BS  # noqa: E402  (shared N2/N2C scorer, hash-bound; never edited)
import synth_n2c as SC  # noqa: E402  (X5.n2c_reading.v1 harness, read-only)

DECL_ID = "X5.n2_reading.v1"
N2_ID = "N2.policy_response.prospective_2026-11.v1"
N2_SHA = "bfbbd6b3b5e0580c8b4e9025c18db966a7f728420f5a2d07fb4ccb047cff6b0c"
N2C_READING_ID = SC.DECL_ID
N2C_READING_SHA = "9da07c371f0028a8dac828f621097d5061694bbfacf69e495b0767a9468f2aef"
SCORER = "dataprep_staging/n2_policy/bootstrap_scorer.py"
SCORER_SHA = "f8990d6182b7b080608f77b7e5f7a449ea722f97f845cd83478997cdc9ae75bd"
X = "dataprep_staging/x5_n2_reading"
LOG = "results/twin_runs/n2_reading"
REPLAYS = f"{LOG}/replays"
PANEL = f"{LOG}/panel.npz"
FWD_STEM = "results/twin_forward/n2_reading_page"
OUT_DIR = f"results/dataprep/reports/{DECL_ID}"
REPORT = f"{OUT_DIR}/report.json"
ATTEMPTS = f"{OUT_DIR}/attempts.jsonl"
WELLS = SC.WELLS                               # 158 screens (sid, cell, layer, county)
WELLS_SHA = SC.WELLS_SHA
WELLS_XY = "results/twin_runs/stage3_datum_gate/stage3_wells.csv"     # N2's registered wells
WELLS_XY_SHA = "14bc30ec0f93e38a1ea68a23ea9a78d0de0cb0201f73544d5690402ef67451e6"
PRODUCT = "results/dataprep/pump_electricity/preorigin/v3"
TABLE = "county_class_monthly.parquet"
USE = "design_time_evidence"
REHEARSAL = "results/dataprep/reports/N2_policy/rehearsal/forward_pr_page.npz"
# pinned 2026-11-03 (N2's registration names the rehearsal dir but hashes no rehearsal file)
REHEARSAL_SHA = "6795e125cefe09e273bb7d9777f466bfe2d8ff23d480fd8a2fda527feb99ed4c"
PAGE_FORWARD = "results/twin_forward/datum_gate_innov.npz"            # N2 inputs: page_forward
PAGE_FORWARD_SHA = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
R1C_REPLAY = "results/twin_forward/n2c_reading_r1c.npz"               # X5.n2c_reading.v1's
CODE_FILES = (f"{X}/synth_n2.py", f"{X}/prereg_n2_reading.py", f"{X}/run_n2_reading.sh",
              SCORER, "dataprep_staging/n2c_reading/synth_n2c.py",
              "dataprep_staging/x5_changhua_lever/replay.py", WELLS)
REPLAY_YEARS = (2012, 2013, 2014, 2015, 2016, 2017, 2018)
N2_WINDOW = SC.N2C_WINDOW                      # forward months 178..189 = 2026-11 .. 2027-10
N_T = SC.N_T
N_YUNLIN = 84

# published design-time fields (prereg_n2_reading.published_problems re-reads every one)
_R2_R0_YL = 0.29                               # I1_changhua_drivers/report.md table, Yunlin
_RMSE_R0_YL = 1.8561680599835215
_R2_CANAL_YL = 0.41855159786701807             # raw:canal_w1_cell (report-only kappa)
_R_ENERGY_YL = -0.8683200707914109
_R_CANAL_YL = 0.8771508726475611
_R2_DS_ALL_YL = 0.7435338902539914             # deseasonalised:all_drivers (median well)
_R2_DS_ALL_MINUS_CANAL_YL = -0.001664693967455777  # deseasonalised:all_minus_canal_w1_cell
_SHARE_0_5 = 0.7881534218270372
_SHARE_COUNTY = 0.2678172591368074
_SIGMA_PAGE_KFOLD_YL = 9.204463975181127       # mean rmse_model_m over the 84 Yunlin wells
_ANOM_SD_YL = _RMSE_R0_YL / math.sqrt(1.0 - _R2_R0_YL)
_R2_CANAL_UNIQUE_YL = max(0.0, _R2_DS_ALL_YL - _R2_DS_ALL_MINUS_CANAL_YL)
_KAPPA_UNCAPPED = math.sqrt(_R2_CANAL_UNIQUE_YL) * _ANOM_SD_YL
_KAPPA_CAP = math.sqrt(_SHARE_COUNTY) * _RMSE_R0_YL
FORCING_ERROR_LAMBDA = 0.8
RHO_LOW = -0.52
N2_PUBLISHED_WITHIN_RMS_M = (0.65, 0.83)       # N2 power_check.model_dynamics_replay p50, p90
N2_PUBLISHED_TEXT = "within-window (well-demeaned) RMS p50 0.65 m (p90 0.83)"


def corr_bound(r1: float, r2: float) -> tuple[float, float]:
    """The range of corr(A, B) given corr(H, A) = r1 and corr(H, B) = r2 (the 3 x 3
    correlation matrix must be positive semi-definite)."""
    s = math.sqrt((1.0 - r1 * r1) * (1.0 - r2 * r2))
    return r1 * r2 - s, r1 * r2 + s


def ar1_within_factor(phi: float, T: int) -> float:
    """E[within-window variance] / marginal variance of a stationary AR(1) over T months (what
    per-well demeaning leaves of it): 1 - sum_ij phi^|i-j| / T^2."""
    i = np.arange(T)
    return float(1.0 - (phi ** np.abs(i[:, None] - i[None, :])).sum() / T ** 2)


PARAMS = {
    "sigma_tot_m": {
        "value": _RMSE_R0_YL,
        "source": "results/dataprep/reports/I1_changhua_local/report.json",
        "field": "part_a_local.yunlin.rmse_r0_mean",
        "note": "R0's out-of-fold Yunlin own-mean head-anomaly RMSE (84 wells); design-time, "
                "spent heads. R0 is N2's declared secondary model; the page model's own "
                "pre-origin per-well RMSE (sigma_page_kfold_m) is an absolute-head RMSE and "
                "enters report-only cells; its post-origin RMSE is excluded under U7"},
    "share_0_5km": {
        "value": _SHARE_0_5,
        "source": "results/dataprep/reports/I1_changhua_interannual/report.json",
        "field": "counties.yunlin.residual.R0.by_distance.0-5 km.median_r",
        "note": "median correlation of R0's Yunlin residual series for well pairs within 5 km "
                "(201 pairs) = the county-wide plus 5 km block variance share"},
    "share_county": {
        "value": _SHARE_COUNTY,
        "source": "results/dataprep/reports/I1_changhua_interannual/report.json",
        "field": "counties.yunlin.residual.R0.by_distance.>20 km.median_r",
        "note": "median correlation of R0's Yunlin residual series for well pairs beyond 20 km "
                "(1435 pairs; a FULL-RECORD pair correlation, not a within-12-month one) = the "
                "county-wide variance share; kappa^2 is taken out of it for the confounder; "
                "the block share is share_0_5km - share_county, the well share "
                "1 - share_0_5km"},
    "phi": {
        "value": SC.PARAMS["phi"]["value"],
        "source": SC.PARAMS["phi"]["source"],
        "field": SC.PARAMS["phi"]["field"],
        "note": "as X5.n2c_reading.v1: median per-well lag-1 autocorrelation of the OBSERVED "
                "monthly head anomaly over 158 wells in both counties; used for all three "
                "AR(1) parts"},
    "kappa_m": {
        "value": min(_KAPPA_UNCAPPED, _KAPPA_CAP),
        "source": "results/dataprep/reports/I1_changhua_drivers/report.json; "
                  "results/dataprep/reports/I1_changhua_local/report.json; "
                  "results/dataprep/reports/I1_changhua_drivers/report.md; "
                  "results/dataprep/reports/I1_changhua_interannual/report.json",
        "field": "min(sqrt(max(0, summary.yunlin.median_cv_r2.deseasonalised:all_drivers - "
                 "deseasonalised:all_minus_canal_w1_cell)) x part_a_local.yunlin.rmse_r0_mean "
                 "/ sqrt(1 - R0 Yunlin median-well out-of-fold R2 0.29 (report.md table)), "
                 "sqrt(share_county) x sigma_tot_m)",
        "inputs": {"r2_deseasonalised_all_drivers": _R2_DS_ALL_YL,
                   "r2_deseasonalised_all_minus_canal": _R2_DS_ALL_MINUS_CANAL_YL,
                   "r2_canal_unique": _R2_CANAL_UNIQUE_YL, "rmse_r0_m": _RMSE_R0_YL,
                   "r2_r0": _R2_R0_YL, "kappa_uncapped_m": _KAPPA_UNCAPPED,
                   "kappa_cap_m": _KAPPA_CAP, "cap_binds": _KAPPA_UNCAPPED > _KAPPA_CAP},
        "note": "lead ruling (review B1): the deseasonalised UNIQUE canal share (beyond every "
                "other driver; a difference of two median-well R2s, not a median of "
                "differences), capped so that kappa^2 <= share_county sigma^2, and taken OUT "
                "of the county-wide noise so the total variance stays sigma^2. The cap binds "
                "(uncapped 1.90 m), so the county-wide AR part is 0 in the primary cells"},
    "kappa_raw_m": {
        "value": math.sqrt(_R2_CANAL_YL) * _ANOM_SD_YL,
        "source": "results/dataprep/reports/I1_changhua_drivers/report.json",
        "field": "sqrt(summary.yunlin.median_cv_r2.raw:canal_w1_cell) x rmse_r0_mean / "
                 "sqrt(1 - 0.29) (X5.n2c_reading.v1's derivation)",
        "inputs": {"canal_r2_raw": _R2_CANAL_YL},
        "note": "report-only cells: kappa^2 2.03 m^2 exceeds share_county sigma^2, so the "
                "county part is 0 there and the total variance exceeds sigma^2"},
    "sigma_page_kfold_m": {
        "value": _SIGMA_PAGE_KFOLD_YL,
        "source": "results/twin_runs/stage3_datum_gate/stage3_kfold_wells.csv",
        "field": "mean of rmse_model_m over the 84 Yunlin wells (one row each)",
        "note": "report-only (review S1): the page model's PRE-origin out-of-fold per-well "
                "RMSE. It is a held-out-well ABSOLUTE-head RMSE (level included, which N2's "
                "per-well demeaning removes), so it is an extreme upper bound, not an anomaly "
                "sigma"},
    "confounder_within_factor": {
        "value": ar1_within_factor(SC.PARAMS["phi"]["value"], 12),
        "source": "derived from phi and the 12 window months",
        "field": "1 - sum_ij phi^|i-j| / 144",
        "note": "g is scaled to within-window sd sqrt(factor), not 1: the expected "
                "within-window variance per-well demeaning leaves of an AR(1) county "
                "component of marginal variance kappa^2 (review B1)"},
    "rho_abs": {
        "value": 0.87,
        "source": "results/dataprep/reports/I1_changhua_interannual/report.json",
        "field": "counties.yunlin.observed.regional_vs_drivers_r (census_energy "
                 "-0.8683, canal_deliveries +0.8772)",
        "inputs": {"census_energy": _R_ENERGY_YL, "canal_deliveries": _R_CANAL_YL},
        "note": "the mean of the two magnitudes, rounded to 2 decimals (X5.n2c_reading.v1 "
                "took 0.83, both Changhua magnitudes rounding to it). Proxy: no canal-vs-"
                "pumping correlation is published; the cited ones are INTERANNUAL (regional "
                "series) correlations transferred to monthly window co-movement. Sign: rho = "
                "corr(g, a_YL), the drought-year direction that biases beta upward"},
    "rho_low": {
        "value": RHO_LOW,
        "source": "derived: corr_bound(census_energy, canal_deliveries)",
        "field": "lower end of corr(canal, -energy) implied by the two cited correlations",
        "inputs": {"bound": list(corr_bound(-_R_ENERGY_YL, _R_CANAL_YL))},
        "note": "report-only cells (review S4): corr(canal, energy) magnitude is only bounded "
                "to [0.52, 1.00] by the two published correlations"},
    "forcing_error_lambda": {
        "value": FORCING_ERROR_LAMBDA, "source": "fixed in the draft (review S2)",
        "field": "scored contrast lambda c_j + sqrt(1 - lambda^2) c_(j+1 mod 7), rescaled to "
                 "c_j's within-window RMS; the world is generated with c_j",
        "note": "report-only cells"},
}
GRID = {"k": (0.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0), "rho": (0.0, -0.87)}
_RAW = PARAMS["kappa_raw_m"]["value"]
REPORT_ONLY_CELLS = (
    {"k": 0.0, "rho": -0.87, "kappa_m": _RAW, "label": "kappa raw (canal alone, 1.43 m)"},
    {"k": 1.0, "rho": -0.87, "kappa_m": _RAW, "label": "kappa raw (canal alone, 1.43 m)"},
    {"k": 0.0, "rho": -0.87, "kappa_mult": 0.5, "label": "kappa x0.5"},
    {"k": 1.0, "rho": -0.87, "kappa_mult": 0.5, "label": "kappa x0.5"},
    {"k": 0.0, "rho": -0.87, "kappa_mult": 0.25, "label": "kappa x0.25"},
    {"k": 1.0, "rho": -0.87, "kappa_mult": 0.25, "label": "kappa x0.25"},
    {"k": 0.0, "rho": 0.0, "county_mult": 0.0, "label": "N2C noise structure (no county part)"},
    {"k": 1.0, "rho": 0.0, "county_mult": 0.0, "label": "N2C noise structure (no county part)"},
    {"k": 0.0, "rho": -0.87, "sigma_m": _SIGMA_PAGE_KFOLD_YL,
     "label": "sigma = page model pre-origin k-fold rmse_model_m (absolute heads)"},
    {"k": 1.0, "rho": 0.0, "sigma_m": _SIGMA_PAGE_KFOLD_YL,
     "label": "sigma = page model pre-origin k-fold rmse_model_m (absolute heads)"},
    {"k": 1.0, "rho": 0.0, "forcing_error_lambda": FORCING_ERROR_LAMBDA,
     "label": "forcing error, lambda 0.8"},
    {"k": 0.5, "rho": 0.0, "forcing_error_lambda": FORCING_ERROR_LAMBDA,
     "label": "forcing error, lambda 0.8"},
    {"k": 0.0, "rho": RHO_LOW, "label": "rho -0.52 (lower bound)"},
    {"k": 1.0, "rho": RHO_LOW, "label": "rho -0.52 (lower bound)"},
)
CELL_DEFAULTS = {"kappa_mult": 1.0, "county_mult": 1.0}
REPLICATES = 500
PLUMBING_BAND = SC.PLUMBING_BAND
PLUMBING_NOISE_SCALE = SC.PLUMBING_NOISE_SCALE
PLUMBING_REPLICATES = SC.PLUMBING_REPLICATES
SMOKE_SEED = 990001                            # the smoke's own seed; refused for the study
TEST_SEED = 123456789                          # the unit tests' seed; refused for real runs
REFUSED_SEEDS = (SMOKE_SEED, BS.SEED, SC.SEED)  # smoke, the scorer's bootstrap, N2C reading's
REAL_REFUSED_SEEDS = (TEST_SEED,)              # also refused by draft_prereg, simulate, run
# the X5.n2c_reading.v1 reading thresholds, copied (synth_n2c.READING; registered there as
# thresholds.reading: false_pass_max 0.25, wrong_size_max 0.25, coverage_min 0.8), plus the
# lead's power floor (review B2; not from X5.n2c_reading.v1, which registered none)
READING = {"false_pass_max": 0.25, "wrong_size_max": 0.25, "coverage_min": 0.80}
POWER_MIN = 0.80
READING_SOURCE = (f"{N2C_READING_ID} (registration sha256 {N2C_READING_SHA[:8]}, "
                  "thresholds.reading; dataprep_staging/n2c_reading/synth_n2c.py READING): "
                  "R1 false pass >= 0.25 -> reported_not_shown; R2 wrong-size pass >= 0.25 "
                  "or coverage < 0.80 -> rates_printed. X5.n2c_reading.v1 never used the word "
                  "'decisive' (its display keys were reported_not_shown, rates_printed and "
                  "power); the tier names here and the power floor 0.80 are the lead's "
                  "(review B2)")
_FE = ("; no forcing error between X1 and the true pumping is modelled in the gating cells "
       "(report-only forcing-error cell, lambda 0.8: power {pwf:.2f})")
_PW = "; power {pw:.2f} ({pwc:.2f} with the canal confounder)"
WORDING = {
    "weak": "N2's rule is weak by design for Yunlin (X5.n2_reading.v1): the confounded "
            "false-pass rate max over rho of P(supported | k = 0) = {fp:.2f} >= 0.25 (the "
            "stricter of the unconditional rate and the rate given N2's informative gate); "
            "an N2 'response supported' would be reported, not shown" + _PW,
    "informative": "N2's rule is informative but not decisive for Yunlin (X5.n2_reading.v1): "
                   "false pass {fp:.2f} < 0.25, but wrong-size pass at k = 0.5 / 2 "
                   "{ws05:.2f} / {ws2:.2f} or 90 % interval coverage at k = 1 {cov:.2f} misses "
                   "its threshold (0.25 / 0.80); an N2 verdict is printed with these rates"
                   + _PW + _FE,
    "pass_controlled_low_power": "N2's rule is pass-controlled but low-power for Yunlin "
                                 "(X5.n2_reading.v1): false pass {fp:.2f} < 0.25, wrong-size "
                                 "pass at k = 0.5 / 2 {ws05:.2f} / {ws2:.2f} < 0.25, coverage "
                                 "{cov:.2f} >= 0.80, but power {pw:.2f} < 0.80 ({pwc:.2f} with "
                                 "the canal confounder); an N2 non-pass says little" + _FE,
    "decisive": "N2's rule is decisive by design for Yunlin (X5.n2_reading.v1): false pass "
                "{fp:.2f} < 0.25, wrong-size pass at k = 0.5 / 2 {ws05:.2f} / {ws2:.2f} < "
                "0.25, 90 % interval coverage at k = 1 {cov:.2f} >= 0.80, power {pw:.2f} >= "
                "0.80 ({pwc:.2f} with the canal confounder)" + _FE,
    "not_estimable": "X5.n2_reading.v1: N2's operating characteristics were not estimable "
                     "(the low-noise plumbing cell's median beta {mb} fell outside [0.9, 1.1] "
                     "with exact plumbing passing); no rate is printed and C01's D stays "
                     "'decidability unknown'",
    "verdict_word": "the N2 verdict word never changes; no N2 threshold changes; the page line "
                    "needs ruling R8.8",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rp(s: str | Path) -> Path:
    return Path(s) if Path(s).is_absolute() else REPO / s


def frozen_code(repo: Path = REPO) -> dict:
    return {p: sha256(repo / p) if (repo / p).exists() else "missing" for p in CODE_FILES}


def check_seed(seed, real: bool = False) -> int:
    """The study seed: an int the lead picks at registration (no default; never a refused
    seed). ``real`` (draft_prereg, simulate, the run CLI) also refuses the unit tests' seed."""
    if seed is None:
        raise ValueError("the seed is required (no default): the lead picks it at registration")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"the seed must be an int, not {type(seed).__name__}")
    seed = int(seed)
    if not 0 <= seed < 2 ** 63:
        raise ValueError("the seed must be in [0, 2**63)")
    if seed in REFUSED_SEEDS:
        raise ValueError(f"seed {seed} is refused (smoke, scorer or N2C-reading seed)")
    if real and seed in REAL_REFUSED_SEEDS:
        raise ValueError(f"seed {seed} is refused for real runs (the unit tests' seed)")
    return seed


def design_spec(seed) -> dict:
    """Everything the registered run depends on (frozen into the declaration)."""
    seed = check_seed(seed)
    return {
        "world": "e = k * c_j + n + kappa * g_j over N2's 84 Yunlin wells x 12 window months; "
                 "ZR = 0, obs = e; scored by bootstrap_scorer.score (flags 0, no CL, detrended "
                 "secondary off) with n_recorded_months 12 (synth_n2c.score_world)",
        "params": PARAMS, "grid": {k: list(v) for k, v in GRID.items()},
        "report_only_cells": list(REPORT_ONLY_CELLS), "replicates_per_cell": REPLICATES,
        "seed": seed,
        "seeding": "numpy SeedSequence(seed).spawn(n_cells + 1): children 0..n_cells-1 in "
                   "cell order (k outer, rho inner, then the report-only cells), child "
                   "n_cells for the plumbing cell; replicate r of a cell uses replay r mod 7 "
                   "and that cell's generator in order: county noise (1 x 12), block noise "
                   "(n_blocks x 12), well noise (84 x 12), xi (12)",
        "replays": [f"RP{y}" for y in REPLAY_YEARS],
        "replay_source": f"{PRODUCT} {TABLE} through common.guard for the use {USE} (allowed by "
                         "its MANIFEST); rows scope in_grid, class all, dated <= 2022-12 "
                         "(x5_changhua_lever/replay.raw_ratios: ratio to the 2012-2022 "
                         "month-of-year mean, a source month without a kept row takes 1); "
                         "the Yunlin ratio only, fan-wide (N2's rule: E_PR = E_ZR x r(t) in "
                         "every cell); RPY maps 2023-01+j onto Y-01+j, j = 0..57; never X1",
        "noise": "x(t) = phi x(t-1) + sqrt(1 - phi^2) s eps, x(0) ~ N(0, s^2) "
                 "(synth_n2c.ar1); with sigma = the cell's sigma (sigma_tot_m unless "
                 "overridden), kappa_eff = the cell's kappa_m (default kappa_m) x kappa_mult "
                 "and avail = max(0, share_county sigma^2 - kappa_eff^2): county part s^2 = "
                 "county_mult avail (one series for all 84 wells), block part s^2 = "
                 "(share_0_5km - share_county) sigma^2 + (1 - county_mult) avail (5 km TWD97 "
                 "blocks, bootstrap_scorer.block_ids of the wells' x, y), well part s^2 = "
                 "(1 - share_0_5km) sigma^2; the total is sigma^2 whenever kappa_eff^2 <= "
                 "share_county sigma^2",
        "confounder": "kappa_eff sqrt(confounder_within_factor) synth_n2c.confounder: g = rho "
                      "a~ + sqrt(1 - rho^2) xi~, a~ = the window's Yunlin anomaly demeaned "
                      "and scaled to unit sd, xi~ an N(0, 1) draw demeaned, orthogonalised "
                      "against a~ and scaled to unit sd; common to all wells; its "
                      "within-window variance is the one per-well demeaning leaves of an "
                      "AR(1) county part of marginal variance kappa_eff^2",
        "forcing_error": "cells with forcing_error_lambda generate e with c_j and score it "
                         "with c'_j = lambda c_j + sqrt(1 - lambda^2) c_(j+1 mod 7), rescaled "
                         "so its RMS of the well-demeaned contrast equals c_j's",
        "cell_defaults": CELL_DEFAULTS,
        "scorer": {"path": SCORER, "sha256": SCORER_SHA, "version": BS.VERSION,
                   "B": BS.B_BOOT, "seed": BS.SEED, "level": BS.LEVEL},
        "plumbing": {"exact": "no noise, kappa 0: beta == k (|diff| <= 1e-9) for every "
                              "replay with an informative contrast (synth_n2c."
                              "exact_plumbing); failure = engineering (code defect), no "
                              "reading",
                     "cell": {"k": 1.0, "rho": 0.0, "kappa_mult": 0.0, "county_mult": 1.0,
                              "noise_scale": PLUMBING_NOISE_SCALE,
                              "replicates": PLUMBING_REPLICATES, "band": list(PLUMBING_BAND),
                              "outside_band": "terminal reading 'not_estimable'; report.json "
                                              "is written; no grid cell is computed; final"},
                     "full_noise_k1_rho0": "an ordinary grid cell; gates nothing"},
        "reading": {**READING, "power_min": POWER_MIN, "source": READING_SOURCE,
                    "wording": WORDING,
                    "rules": [
                        "every pass rate P below is max(P(response_supported), "
                        "P(response_supported | the replicate passes N2's informative gate)) "
                        "(the stricter; the unconditional one when no replicate is "
                        "informative); both are reported",
                        "false pass = max over rho (0, -0.87) of P(k = 0), as wrong size "
                        "already is (review S3)",
                        "weak: false pass >= 0.25 (X5.n2c_reading.v1 R1)",
                        "informative: not weak, and max over rho of P(k = 0.5) or of "
                        "P(k = 2) >= 0.25, or the 90 % interval coverage of k = 1 at rho = 0 "
                        "< 0.80 (X5.n2c_reading.v1 R2)",
                        "pass_controlled_low_power: neither, and power < 0.80 (lead, review "
                        "B2)",
                        "decisive: none of the above (false pass < 0.25, wrong size < 0.25, "
                        "coverage >= 0.80, power >= 0.80)",
                        "power = P(response_supported | k = 1, rho = 0) (unconditional; the "
                        "conditional one beside it), printed with rho = -0.87 and the "
                        "report-only forcing-error power beside it in every wording",
                        "plumbing cell outside its band (exact plumbing passing) -> the "
                        "terminal reading not_estimable instead of the above",
                        "the reading is reported whichever way it falls; the verdict word "
                        "never changes; the report-only cells never change a reading"]},
        "window": ["2026-11", "2027-10"], "n2_window_forward_months": list(N2_WINDOW),
    }


# ----------------------------------------------------------------------------- replays
def read_source(product_dir: Path):
    """The pre-origin county table through the guard, for the declared use only."""
    from common import guard

    p = guard.open_product(product_dir, USE)
    df = p.read_table(TABLE, columns=["county", "scope", "class", "date", "value"])
    rec = {"product": PRODUCT, "use": USE, "table": TABLE,
           "manifest_sha256": sha256(Path(product_dir) / "MANIFEST.json"),
           "table_sha256": p.outputs[TABLE]["sha256"], "guard_notes": p.notes,
           "rows_kept_by_guard": int(len(df)),
           "read": "columns county, scope, class, date, value (and the flag column the guard "
                   "always reads); rows scope in_grid, class all, Changhua and Yunlin, dated "
                   "<= 2022-12 enter the ratio (replay.raw_ratios needs both counties' "
                   "climatology); only the Yunlin ratio is written"}
    return df, rec


def build_replays(out: Path, product_dir: Path) -> dict:
    """rpY.csv (month,multiplier: N2's fan-wide r(t) format, build_r's layout) for the seven
    replays and replays.json. Refuses a non-empty directory."""
    import replay as RPL  # x5_changhua_lever/replay.py, read-only

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; replays are written once")
    out.mkdir(parents=True, exist_ok=True)
    df, src = read_source(product_dir)
    raw = RPL.raw_ratios(df)
    ratios = raw.fillna(1.0)
    rec = {"builder": f"{X}/synth_n2.py build_replays (x5_changhua_lever/replay.py raw_ratios "
                      "and replay, read-only)",
           "builder_sha256": sha256(Path(__file__)),
           "replay_py_sha256": sha256(REPO / "dataprep_staging/x5_changhua_lever/replay.py"),
           "source": src,
           "rule": "r_YL(s) = Yunlin in_grid all-class value / its 2012-2022 month-of-year "
                   "mean; a missing source month -> 1; applied to every cell (N2's rule)",
           "first": RPL.FIRST, "n_months": RPL.N_MONTHS, "window": list(RPL.WINDOW),
           "filled_with_r1": {"yunlin": RPL.filled_months(raw)["yunlin"]}, "replays": {}}
    for y in REPLAY_YEARS:
        rep = RPL.replay(ratios, y)
        p = out / f"rp{y}.csv"
        with p.open("x") as fh:
            rep.rename(columns={"r_yunlin": "multiplier"})[["month", "multiplier"]].to_csv(
                fh, index=False)
        w = RPL.window_anomalies(rep)
        rec["replays"][f"rp{y}"] = {
            "file": p.name, "sha256": sha256(p), "source_first": rep["source"].iloc[0],
            "source_last": rep["source"].iloc[-1],
            "filled_with_r1": {k: v for k, v in RPL.filled_months(
                raw, rep["source"].tolist())["yunlin"].items() if k != "months"},
            "window": {"months": w["months"], "source": w["source"],
                       "a_yunlin": w["a_yunlin"]}}
    with (out / "replays.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    return rec


# ----------------------------------------------------------------------------- panel
def yunlin_screens(repo: Path = REPO) -> dict:
    p = repo / WELLS
    if sha256(p) != WELLS_SHA:
        raise SystemExit(f"{WELLS} differs from {WELLS_SHA[:8]}")
    rows = [r for r in json.loads(p.read_text())["rows"] if r["county"] == "yunlin"]
    if len(rows) != N_YUNLIN:
        raise SystemExit(f"expected {N_YUNLIN} Yunlin screens")
    return {"sid": np.array([r["sid"] for r in rows]),
            "cell": np.array([r["cell"] for r in rows], int),
            "layer": np.array([r["layer"] for r in rows], int)}


def n2_yunlin_ids(repo: Path = REPO) -> list[str]:
    """N2's 84 Yunlin wells: its external unit's ids that are Yunlin screens."""
    d = json.loads((repo / f"results/dataprep/preregistration/{N2_ID}.json").read_text())
    ids = d["tests"][0]["external_units"]["N2:heads_prospective_2026-11_2027-10"]["ids"]
    return list(ids)


PAGE_MEMBER = "stage3_datum_gate"
FWD_OPTIONS = {"column_heads": "free", "update_method": "innovation", "apex_hold": "calibrated",
               "restart_taper_km": 5.0, "column_hpc0_fast_days": 365.0, "save_members": "none"}
ORIGIN, HORIZON = SC.ORIGIN, SC.HORIZON
REPRO_TOL_M = 1e-3


def forward_meta_problems(f) -> list[str]:
    """N2's registered forward command as run: member label, gains, origin, horizon, options."""
    e = []
    dates = [str(s)[:7] for s in f["dates"]]
    if not dates or dates[0] != "2012-01" or dates[-1] != "2032-12" or len(dates) != N_T:
        e.append(f"dates {dates[:1]}..{dates[-1:]}")
    if int(f["origin"]) != ORIGIN or int(f["horizon"]) != HORIZON:
        e.append(f"origin {int(f['origin'])} horizon {int(f['horizon'])}")
    labels = [str(s) for s in f["member_labels"]]
    if labels != [PAGE_MEMBER]:
        e.append(f"member_labels {labels}")
    if float(f["gain"]) != 1.0 or float(f["hindcast_gain"]) != 0.0:
        e.append(f"gain {float(f['gain'])} hindcast_gain {float(f['hindcast_gain'])}")
    fo = json.loads(str(f["forward_options"]))
    bad = {k: fo.get(k) for k, v in FWD_OPTIONS.items() if fo.get(k) != v}
    if bad:
        e.append(f"forward_options differ: {bad}")
    return e


def baseline_at(f, sc: dict) -> np.ndarray:
    """The baseline arm's heads (wells x all months) at the screens' cells and layers. Only
    the baseline scenario is indexed."""
    names = [str(s) for s in f["scenario_names"]]
    hm = f["heads_mean"]
    return np.asarray(hm[names.index("baseline")][sc["layer"], sc["cell"], :], float)


def reproduction(run_base: np.ndarray, ref_base: np.ndarray) -> dict:
    """Engineering: the run's baseline arm is the rehearsal's (N2's registered command, page
    main set, same horizon) within REPRO_TOL_M at the 84 screens over every month."""
    if run_base.shape != ref_base.shape:
        return {"ok": False, "why": f"shapes {run_base.shape} vs {ref_base.shape}"}
    if not (np.isfinite(run_base).all() and np.isfinite(ref_base).all()):
        return {"ok": False, "why": "non-finite baseline heads"}
    d = float(np.max(np.abs(run_base - ref_base)))
    return {"ok": d <= REPRO_TOL_M, "max_abs_diff_m": d, "tol_m": REPRO_TOL_M}


def check_rehearsal(path: Path) -> None:
    """Refuse a reference forward that is not the pinned N2 rehearsal (review M1)."""
    got = sha256(path) if Path(path).exists() else "missing"
    if got != REHEARSAL_SHA:
        raise SystemExit(f"REFUSED: {path} sha256 {got} is not the pinned rehearsal "
                         f"{REHEARSAL_SHA}")


def extract(fwd: Path, replays_json: Path, out: Path, rehearsal: Path) -> dict:
    """The panel: c_j (7, 84, 12) at the Yunlin screens, a_YL,j (7, 12), blocks (84,)."""
    sc = yunlin_screens()
    if sorted(sc["sid"].tolist()) != sorted(set(sc["sid"]) & set(n2_yunlin_ids())):
        raise SystemExit("the Yunlin screens are not N2's wells")
    f = np.load(fwd, allow_pickle=False)
    names = [str(s) for s in f["scenario_names"]]
    want = ["baseline"] + [f"RP{y}" for y in REPLAY_YEARS]
    if names != want:
        raise SystemExit(f"{fwd}: scenarios {names} != {want}")
    dates = [str(s)[:7] for s in f["dates"]]
    if len(dates) != N_T or dates[N2_WINDOW[0]] != "2026-11" or \
            dates[N2_WINDOW[1] - 1] != "2027-10":
        raise SystemExit(f"{fwd}: window months misplaced")
    probs = forward_meta_problems(f)
    if probs:
        raise SystemExit(f"{fwd}: " + "; ".join(probs))
    rep = json.loads(replays_json.read_text())
    rec = (json.loads(str(f["forward_options"]))).get("recorded_pumping") or {}
    hm = f["heads_mean"]
    t0, t1 = N2_WINDOW
    base_all = baseline_at(f, sc)
    base = base_all[:, t0:t1]
    c, a = [], []
    for y in REPLAY_YEARS:
        r = rep["replays"][f"rp{y}"]
        if rec.get(f"RP{y}", {}).get("sha256") != r["sha256"]:
            raise SystemExit(f"{fwd}: RP{y} was not run on {r['file']} as built")
        c.append(np.asarray(hm[names.index(f"RP{y}")][sc["layer"], sc["cell"], t0:t1],
                            float) - base)
        a.append(np.asarray(r["window"]["a_yunlin"], float))
    check_rehearsal(rehearsal)
    g = np.load(rehearsal, allow_pickle=False)
    gp = forward_meta_problems(g)
    if gp:
        raise SystemExit(f"{rehearsal}: " + "; ".join(gp))
    repro = reproduction(base_all, baseline_at(g, sc))
    repro.update(reference=REHEARSAL, reference_sha256=sha256(rehearsal))
    if not repro["ok"]:
        raise SystemExit(f"ENGINEERING: the run's baseline does not reproduce the N2 "
                         f"rehearsal's baseline ({repro})")
    blocks = SC.well_blocks(sc["sid"], rp(WELLS_XY))
    panel = {"c": np.stack(c), "a_yl": np.stack(a), "sid": sc["sid"], "block": blocks,
             "replays": np.array([f"RP{y}" for y in REPLAY_YEARS]),
             "forward_sha256": np.str_(sha256(fwd)),
             "replays_json_sha256": np.str_(sha256(replays_json))}
    if not np.isfinite(panel["c"]).all():
        raise SystemExit("non-finite contrast")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as fh:
        np.savez(fh, **panel)
    return {"panel": str(out), "panel_sha256": sha256(out), "shape_c": list(panel["c"].shape),
            "n_blocks": int(len(set(blocks))), "reproduction": repro}


def load_panel(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def synthetic_panel(seed: int = 0, n_blocks: int = 24, scale: float = 0.3) -> dict:
    """A stand-in panel for smoke runs and tests (no model output)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.1, (7, 12))
    lam = rng.uniform(0.5, 1.5, N_YUNLIN) * scale / 0.1
    lag = np.array([[0.3 * a[j, max(t - 1, 0)] + 0.7 * a[j, t] for t in range(12)]
                    for j in range(7)])
    c = -lam[None, :, None] * lag[:, None, :] + rng.normal(0, 0.2, (7, N_YUNLIN, 1))
    blocks = np.array([f"{b}_0" for b in rng.integers(0, n_blocks, N_YUNLIN)])
    return {"c": c, "a_yl": a, "sid": np.array([f"07{i:06d}" for i in range(N_YUNLIN)]),
            "block": blocks, "replays": np.array([f"RP{y}" for y in REPLAY_YEARS])}


# ----------------------------------------------------------------------------- worlds
def cell_kappa(cell: dict) -> float:
    return float(cell.get("kappa_m", PARAMS["kappa_m"]["value"])) * float(
        cell.get("kappa_mult", 1.0))


def noise_variances(cell: dict) -> dict:
    """The three AR(1) parts' marginal variances for a cell (see design_spec 'noise')."""
    sig = float(cell.get("sigma_m", PARAMS["sigma_tot_m"]["value"]))
    s05, scty = PARAMS["share_0_5km"]["value"], PARAMS["share_county"]["value"]
    cm = float(cell.get("county_mult", 1.0))
    avail = max(0.0, scty * sig * sig - cell_kappa(cell) ** 2)
    return {"county": cm * avail, "block": (s05 - scty) * sig * sig + (1.0 - cm) * avail,
            "well": (1.0 - s05) * sig * sig, "kappa_sq": cell_kappa(cell) ** 2,
            "sigma_sq": sig * sig}


def world(panel: dict, j: int, cell: dict, rng, noise_scale: float = 1.0) -> np.ndarray:
    c = panel["c"][j]
    nw, T = c.shape
    ub, binv = np.unique(panel["block"], return_inverse=True)
    phi = PARAMS["phi"]["value"]
    v = noise_variances(cell)
    cty = SC.ar1(rng, 1, T, noise_scale * math.sqrt(v["county"]), phi)
    b = SC.ar1(rng, len(ub), T, noise_scale * math.sqrt(v["block"]), phi)[binv]
    w = SC.ar1(rng, nw, T, noise_scale * math.sqrt(v["well"]), phi)
    g = SC.confounder(rng, panel["a_yl"][j], cell["rho"])
    gsd = cell_kappa(cell) * math.sqrt(ar1_within_factor(phi, T))
    return cell["k"] * c + cty + b + w + gsd * g[None, :]


def forcing_error_panel(panel: dict, lam: float) -> dict:
    """The contrast N2 would score under forcing error: lambda c_j + sqrt(1 - lambda^2)
    c_(j+1 mod n), rescaled to c_j's RMS of the well-demeaned contrast."""
    c = np.asarray(panel["c"], float)
    n = c.shape[0]

    def rms(x):
        return float(np.sqrt(np.mean((x - x.mean(axis=1, keepdims=True)) ** 2)))

    out = []
    for j in range(n):
        m = lam * c[j] + math.sqrt(1.0 - lam * lam) * c[(j + 1) % n]
        r = rms(m)
        out.append(m * (rms(c[j]) / r) if r > 0 else m)
    return {**panel, "c": np.stack(out)}


def cells() -> list[dict]:
    out = [{"k": k, "rho": r, **CELL_DEFAULTS, "report_only": False}
           for k in GRID["k"] for r in GRID["rho"]]
    return out + [{**CELL_DEFAULTS, **c, "report_only": True} for c in REPORT_ONLY_CELLS]


def run_cell(panel: dict, cell: dict, rng, replicates: int) -> dict:
    n_rep = panel["c"].shape[0]
    lam = cell.get("forcing_error_lambda")
    scored = forcing_error_panel(panel, lam) if lam is not None else panel
    v = {x: 0 for x in BS.VERDICTS}
    betas, cover, ratio, n_inf = [], 0, [], 0
    for r in range(replicates):
        j = r % n_rep
        e = world(panel, j, cell, rng, cell.get("noise_scale", 1.0))
        s = SC.score_world(scored, j, e)
        v[s["verdict"]] += 1
        n_inf += int(bool(s["informative_gate"]["passes"]))
        b = s["beta_response"]
        betas.append(b["beta"])
        lo, hi = b["ci90"]
        cover += int(np.isfinite(lo) and lo <= cell["k"] <= hi)
        ratio.append(s["rmse_m"]["PR"]["equal_well"] / s["rmse_m"]["ZR"]["equal_well"])
    p = {x: n / replicates for x, n in v.items()}
    sup_inf = v["response_supported"] / n_inf if n_inf else None
    return {**cell, "replicates": replicates, "noise_variances": noise_variances(cell),
            "p": p, "p_informative": n_inf / replicates,
            "p_supported_given_informative": sup_inf,
            "p_supported_stricter": (max(p["response_supported"], sup_inf)
                                     if sup_inf is not None else p["response_supported"]),
            "mc_se": {x: math.sqrt(q * (1 - q) / replicates) for x, q in p.items()},
            "median_beta": float(np.nanmedian(betas)),
            "beta_p10_p90": [float(np.nanpercentile(betas, 10)),
                             float(np.nanpercentile(betas, 90))],
            "coverage90_of_k": cover / replicates,
            "median_rmse_ratio_pr_over_zr": float(np.median(ratio))}


def replay_amplitudes(panel: dict) -> dict:
    out = {}
    for j, name in enumerate(panel["replays"]):
        c = panel["c"][j]
        ct = c - c.mean(axis=1, keepdims=True)
        out[str(name)] = {"rms_well_demeaned_c_m": float(np.sqrt(np.mean(ct ** 2))),
                          "window_mean_c_m": float(c.mean()),
                          "sd_a_yunlin": float(np.std(panel["a_yl"][j]))}
    r = [v["rms_well_demeaned_c_m"] for v in out.values()]
    return {"per_replay": out, "rms_range_m": [min(r), max(r)],
            "rms_median_m": float(np.median(r)),
            "n2_published_within_window_rms_p50_p90_m": list(N2_PUBLISHED_WITHIN_RMS_M),
            "n2_published_source": f"{N2_ID} power_check.model_dynamics_replay "
                                   f"('{N2_PUBLISHED_TEXT}'; an X1-forced rollout over the "
                                   "record, 10 Nov-Oct windows)",
            "note": "report only: the replays' within-window RMS beside N2's published one; "
                    "gates nothing"}


def _cell(cs: list[dict], k: float, rho: float) -> dict:
    return next(c for c in cs if not c.get("report_only") and abs(c["k"] - k) < 1e-12
                and c["rho"] == rho)


def _ro_cell(cs: list[dict], k: float, key: str) -> dict | None:
    return next((c for c in cs if c.get("report_only") and abs(c["k"] - k) < 1e-12
                 and c.get(key) is not None), None)


def reading(cs: list[dict]) -> dict:
    """The declared wording from the grid cells; report-only cells set no tier (the
    forcing-error cell's power is printed beside it)."""
    rho_c = GRID["rho"][1]
    fp_by_rho = {r: SC._strict(_cell(cs, 0.0, r)) for r in GRID["rho"]}
    fp = max(fp_by_rho.values())
    ws = {k: max(SC._strict(_cell(cs, k, r)) for r in GRID["rho"]) for k in (0.5, 2.0)}
    cov = _cell(cs, 1.0, 0.0)["coverage90_of_k"]
    pw = _cell(cs, 1.0, 0.0)["p"]["response_supported"]
    pwc = _cell(cs, 1.0, rho_c)["p"]["response_supported"]
    fe = _ro_cell(cs, 1.0, "forcing_error_lambda")
    pwf = fe["p"]["response_supported"] if fe is not None else float("nan")
    if fp >= READING["false_pass_max"]:
        key = "weak"
    elif (ws[0.5] >= READING["wrong_size_max"] or ws[2.0] >= READING["wrong_size_max"]
          or cov < READING["coverage_min"]):
        key = "informative"
    elif pw < POWER_MIN:
        key = "pass_controlled_low_power"
    else:
        key = "decisive"
    v = {"fp": fp, "ws05": ws[0.5], "ws2": ws[2.0], "cov": cov, "pw": pw, "pwc": pwc,
         "pwf": pwf}
    c0 = _cell(cs, 0.0, rho_c)
    return {"status": "estimated", "wording": key,
            "false_pass_max_over_rho": fp,
            "false_pass_by_rho": {str(r): x for r, x in fp_by_rho.items()},
            "false_pass_unconditional_rho_m087": c0["p"]["response_supported"],
            "false_pass_given_informative_rho_m087": c0.get("p_supported_given_informative"),
            "wrong_size_pass": {"k0.5": ws[0.5], "k2": ws[2.0]},
            "coverage90_k1_rho0": cov, "power_k1_rho0": pw, "power_k1_rho_m087": pwc,
            "power_floor": POWER_MIN, "power_k1_forcing_error_report_only": pwf,
            "power_k1_rho0_given_informative":
                _cell(cs, 1.0, 0.0).get("p_supported_given_informative"),
            "display_text": [WORDING[key].format(**v), WORDING["verdict_word"]]}


def simulate(panel: dict, seed, replicates: int = REPLICATES, log=print,
             allow_refused_seed: bool = False) -> dict:
    seed = int(seed) if allow_refused_seed else check_seed(seed, real=True)
    t0 = time.time()
    res = {"exact_plumbing": SC.exact_plumbing(panel)}
    if not res["exact_plumbing"]["ok"]:
        res["engineering_failure"] = "exact plumbing: beta != k with no noise"
        return res
    res["replay_amplitudes"] = replay_amplitudes(panel)
    cl = cells()
    gens = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(cl) + 1)]
    pc = {"k": 1.0, "rho": 0.0, "kappa_mult": 0.0, "county_mult": 1.0,
          "noise_scale": PLUMBING_NOISE_SCALE, "report_only": True}
    c1 = run_cell(panel, pc, gens[len(cl)], PLUMBING_REPLICATES)
    mb = c1["median_beta"]
    res["plumbing_cell"] = {**c1, "band": list(PLUMBING_BAND),
                            "ok": bool(np.isfinite(mb) and PLUMBING_BAND[0] <= mb
                                       <= PLUMBING_BAND[1])}
    log(f"plumbing cell (k=1, rho=0, low noise): ok={res['plumbing_cell']['ok']}")
    if not res["plumbing_cell"]["ok"]:
        txt = WORDING["not_estimable"].format(mb=f"{mb:.3f}" if np.isfinite(mb) else "nan")
        res["reading"] = {"status": "not_estimable", "wording": "not_estimable",
                          "display_text": [txt, WORDING["verdict_word"]]}
        res["runtime_s"] = round(time.time() - t0, 1)
        return res
    out = []
    for i, c in enumerate(cl):
        out.append(run_cell(panel, c, gens[i], replicates))
        log(f"  cell {i + 1}/{len(cl)} done ({time.time() - t0:.0f} s)")
    res["cells"] = out
    res["full_noise_k1_rho0_median_beta_report_only"] = _cell(out, 1.0, 0.0)["median_beta"]
    res["reading"] = reading(out)
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


# ----------------------------------------------------------------------------- engineering
def reference_statistic(e, c, wells, block_of: dict, n_months: int, B: int = BS.B_BOOT,
                        seed: int = BS.SEED) -> dict:
    """See :func:`_reference_statistic`; BLAS pinned to one thread (thousands of small
    least-squares fits oversubscribe a shared machine otherwise)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                          # pragma: no cover - optional
        return _reference_statistic(e, c, wells, block_of, n_months, B, seed)
    with threadpool_limits(limits=1):
        return _reference_statistic(e, c, wells, block_of, n_months, B, seed)


def _reference_statistic(e, c, wells, block_of: dict, n_months: int, B: int = BS.B_BOOT,
                         seed: int = BS.SEED) -> dict:
    """N2's registered statistic computed from its registered wording, independently of the
    scorer: OLS of e on c with one dummy per well (numpy lstsq), the 90 % percentile interval
    of the same refit on block-resampled panels (each drawn copy of a well a new well; the
    draw stream X5.n2c_reading.v1 fixes), the informative gate (RMS of well-demeaned c >= 0.05
    m, >= 5 blocks, >= 6 months), equal-well RMSEs and the decision rule's wording."""
    import pandas as pd

    e, c, wells = np.asarray(e, float), np.asarray(c, float), np.asarray(wells).astype(str)

    def dummy_ols(ee, cc, ww):          # the literal dummy-variable regression (no FWL)
        u, inv = np.unique(ww, return_inverse=True)
        X = np.zeros((len(ee), 1 + len(u)))
        X[:, 0] = cc
        X[np.arange(len(ee)), 1 + inv] = 1.0
        return float(np.linalg.lstsq(X, ee, rcond=None)[0][0])

    beta = dummy_ols(e, c, wells)
    blocks = sorted({block_of[w] for w in wells})
    row_block = np.array([block_of[w] for w in wells])
    rows = {b: np.flatnonzero(row_block == b) for b in blocks}
    _, wcode = np.unique(wells, return_inverse=True)
    nwc = int(wcode.max()) + 1
    draws = np.random.default_rng(seed).integers(0, len(blocks), size=(B, len(blocks)))
    bb = []
    for d in draws:
        idx = [rows[blocks[bi]] for bi in d]
        lab = np.concatenate([wcode[r] + nwc * n for n, r in enumerate(idx)])
        idx = np.concatenate(idx)
        cc = c[idx]
        _, inv = np.unique(lab, return_inverse=True)
        cnt = np.bincount(inv)
        dm = (np.bincount(inv, weights=cc) / cnt)[inv]
        if float(np.sum((cc - dm) ** 2)) > 0:
            bb.append(dummy_ols(e[idx], cc, lab))
    lo, hi = (float(v) for v in np.quantile(np.array(bb), (0.05, 0.95)))
    df = pd.DataFrame({"w": wells, "e": e, "c": c})
    ct = df["c"] - df.groupby("w")["c"].transform("mean")
    rms = float(np.sqrt(np.mean(ct.to_numpy() ** 2)))
    inform = rms >= 0.05 and len(blocks) >= 5 and n_months >= 6
    mse_zr = (df["e"] ** 2).groupby(df["w"]).mean()
    mse_pr = ((df["e"] - df["c"]) ** 2).groupby(df["w"]).mean()
    r_zr, r_pr = float(np.sqrt(mse_zr.mean())), float(np.sqrt(mse_pr.mean()))
    if not inform:
        verdict = "not_informative"
    elif lo <= 0 <= hi or beta <= 0:
        verdict = "not_supported"
    elif 1 / 1.5 <= beta <= 1.5 and r_pr < r_zr:
        verdict = "response_supported"
    else:
        verdict = "direction_supported_size_not"
    return {"beta": beta, "lo": lo, "hi": hi, "rms": rms, "rmse_pr": r_pr, "rmse_zr": r_zr,
            "verdict": verdict}


def harness_statistic(panel: dict, j: int, e: np.ndarray) -> dict:
    s = SC.score_world(panel, j, e)
    return {"beta": s["beta_response"]["beta"], "lo": s["beta_response"]["ci90"][0],
            "hi": s["beta_response"]["ci90"][1],
            "rms": s["informative_gate"]["rms_well_demeaned_contrast_m"],
            "rmse_pr": s["rmse_m"]["PR"]["equal_well"],
            "rmse_zr": s["rmse_m"]["ZR"]["equal_well"], "verdict": s["verdict"]}


def engineering_worlds(c: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
    """Deterministic synthetic e on a contrast (no RNG, no observation): an exact k = 1
    response; k = 0.5 plus per-well offsets plus a smooth well-specific wave; k = 2 plus
    offsets and a common wave; k = 0 plus the wave."""
    nw, T = c.shape
    i = np.arange(nw)[:, None]
    t = np.arange(T)[None, :]
    off = 3.0 * np.sin(1.7 * i + 0.3)
    wave = 0.4 * np.sin(0.9 * t + 1.3 * i) + 0.25 * np.cos(0.5 * t * (1 + i % 3))
    common = 0.3 * np.sin(0.6 * t + 0.2) * np.ones((nw, 1))
    return [("k1_exact", 1.0, 1.0 * c), ("k0.5_offsets_wave", 0.5, 0.5 * c + off + wave),
            ("k2_offsets_common", 2.0, 2.0 * c + off + common), ("k0_wave", 0.0, off + wave)]


def frozen_contrast_panels(repo: Path = REPO) -> list[tuple[str, dict]]:
    """Contrast panels at the 84 Yunlin screens over N2's window from frozen forwards (model
    output only): the page forward's cut30 and retire_aqua arms minus its baseline
    (datum_gate_innov, N2's page_forward), and X5.n2c_reading.v1's R1c replay arms RP2012..
    RP2018 minus their baseline. Blocks from N2's registered wells table."""
    sc = yunlin_screens(repo)
    blocks = SC.well_blocks(sc["sid"], repo / WELLS_XY)
    t0, t1 = N2_WINDOW
    out = []
    for path, arms in ((PAGE_FORWARD, ("cut30", "retire_aqua")),
                       (R1C_REPLAY, tuple(f"RP{y}" for y in REPLAY_YEARS))):
        z = np.load(repo / path, allow_pickle=False)
        names = [str(s) for s in z["scenario_names"]]
        dates = [str(s)[:7] for s in z["dates"]]
        if dates[t0] != "2026-11" or dates[t1 - 1] != "2027-10":
            raise SystemExit(f"{path}: window misplaced")
        hm = z["heads_mean"]
        base = np.asarray(hm[names.index("baseline")][sc["layer"], sc["cell"], t0:t1], float)
        cs = np.stack([np.asarray(hm[names.index(a)][sc["layer"], sc["cell"], t0:t1], float)
                       - base for a in arms])
        out.append((path, {"c": cs, "sid": sc["sid"], "block": blocks,
                           "replays": np.array(arms)}))
    return out


def gate_code_sha(repo: Path = REPO) -> dict:
    """The code the engineering gate result is bound to."""
    return {p: sha256(repo / p) for p in (f"{X}/synth_n2.py",
                                          "dataprep_staging/n2c_reading/synth_n2c.py", SCORER)}


def engineering_gate(panels: list[tuple[str, dict]], B: int = BS.B_BOOT) -> dict:
    """The harness (synth_n2c.score_world -> bootstrap_scorer.score) against the reference
    computation of N2's registered statistic, and exact plumbing, on every contrast."""
    keys = ("beta", "lo", "hi", "rms", "rmse_pr", "rmse_zr")
    mx = dict.fromkeys(keys, 0.0)
    n, agree, plumb = 0, 0, []
    for _name, pan in panels:
        T = pan["c"].shape[2]
        wells = np.repeat(pan["sid"], T)
        bof = dict(zip(pan["sid"], pan["block"], strict=True))
        plumb.append(SC.exact_plumbing(pan)["ok"])
        for j in range(pan["c"].shape[0]):
            for _lab, _k, e in engineering_worlds(pan["c"][j]):
                h = harness_statistic(pan, j, e)
                r = reference_statistic(e.ravel(), pan["c"][j].ravel(), wells, bof, T, B=B)
                for k in keys:
                    mx[k] = max(mx[k], abs(h[k] - r[k]))
                agree += int(h["verdict"] == r["verdict"])
                n += 1
    tol = 1e-8
    return {"n_panels": len(panels), "n_worlds": n, "verdict_agreement": f"{agree}/{n}",
            "max_abs_diff": mx, "tol": tol, "exact_plumbing_ok": all(plumb),
            "scorer_sha256": BS.file_sha256(), "harness_sha256": gate_code_sha(),
            "ok": bool(all(plumb) and agree == n and max(mx.values()) <= tol
                       and BS.file_sha256() == SCORER_SHA)}


# ----------------------------------------------------------------------------- registry
def registered_declaration(prereg_id: str, seed=None, prereg_dir: Path | None = None,
                           repo: Path = REPO) -> tuple[dict, str]:
    """Refuse unless ``prereg_id`` is this declaration, is in INDEX.jsonl at its file's
    sha256, its frozen_code is the code now, and (when given) ``seed`` is its seed."""
    from harness import prereg3

    if prereg_id != DECL_ID:
        raise SystemExit(f"REFUSED: {prereg_id!r} is not {DECL_ID}")
    d = Path(prereg_dir) if prereg_dir is not None else repo / "results/dataprep/preregistration"
    rows = [r for r in prereg3.read_chained(d / "INDEX.jsonl") if r["id"] == DECL_ID]
    if not rows:
        raise SystemExit(f"REFUSED: {DECL_ID} is not registered in {d}/INDEX.jsonl")
    path = d / f"{DECL_ID}.json"
    if not path.exists():
        raise SystemExit(f"REFUSED: {path} is missing")
    sha = sha256(path)
    if sha != rows[-1]["sha256"]:
        raise SystemExit(f"REFUSED: {path} sha256 {sha} != INDEX {rows[-1]['sha256']}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    reg_seed = (doc.get("design") or {}).get("seed")
    if seed is not None and int(seed) != reg_seed:
        raise SystemExit(f"REFUSED: seed {seed} is not the registered seed")
    try:
        want = json.loads(json.dumps(design_spec(reg_seed)))
    except ValueError as err:
        raise SystemExit(f"REFUSED: registered seed: {err}") from err
    if doc.get("design") != want:
        raise SystemExit("REFUSED: the registered design differs from synth_n2.design_spec()")
    now = frozen_code(repo)
    if doc.get("frozen_code") != now:
        bad = sorted(k for k in now if (doc.get("frozen_code") or {}).get(k) != now[k])
        raise SystemExit(f"REFUSED: code changed since registration: {bad}")
    return doc, sha


def last_forward_sha(attempts: Path) -> str | None:
    return SC.last_forward_sha(attempts)


def append_attempt(attempts: Path, rec: dict) -> None:
    attempts.parent.mkdir(parents=True, exist_ok=True)
    with attempts.open("a") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def render_md(res: dict) -> str:
    lines = [f"# {DECL_ID}: synthetic operating characteristics of N2's rule (Yunlin)", "",
             f"Created {res['created']}. Scorer {res['scorer']['sha256'][:12]} "
             f"(bootstrap_scorer {BS.VERSION}). Seed {res.get('seed')}. Synthetic worlds "
             "only; no observation read.", ""]
    if "engineering_failure" in res:
        return "\n".join(lines + [f"ENGINEERING FAILURE: {res['engineering_failure']}", ""])
    if res["reading"]["status"] == "not_estimable":
        return "\n".join(lines + ["Reading (terminal, declared):"]
                         + [f"- {t}" for t in res["reading"]["display_text"]] + [""])
    am = res.get("replay_amplitudes") or {}
    if am:
        lines += [f"Replay within-window RMS of the contrast (report only): median "
                  f"{am['rms_median_m']:.2f} m, range {am['rms_range_m'][0]:.2f}-"
                  f"{am['rms_range_m'][1]:.2f} m; N2 published p50 / p90 "
                  f"{am['n2_published_within_window_rms_p50_p90_m'][0]} / "
                  f"{am['n2_published_within_window_rms_p50_p90_m'][1]} m.", ""]
    lines += ["| k | rho | kappa (m) | cell | P(supported) | P(direction only) | "
              "P(not supported) | P(not informative) | median beta | coverage of k |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        p = c["p"]
        lab = f"report only: {c.get('label', '')}" if c["report_only"] else "grid"
        lines.append(f"| {c['k']:.3g} | {c['rho']:g} | {cell_kappa(c):.2f} | {lab} | "
                     f"{p['response_supported']:.3f} | {p['direction_supported_size_not']:.3f} | "
                     f"{p['not_supported']:.3f} | {p['not_informative']:.3f} | "
                     f"{c['median_beta']:.2f} | {c['coverage90_of_k']:.3f} |")
    lines += ["", f"Reading ({res['reading']['wording']}):"]
    lines += [f"- {t}" for t in res["reading"]["display_text"]] + [""]
    return "\n".join(lines)


def write_report(res: dict, out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "report.json"
    with p.open("x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    with (out_dir / "report.md").open("x") as fh:
        fh.write(render_md(res))
    return p, sha256(p)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_run(seed: int, prereg_id: str, panel_path: Path, out_dir: Path,
            prereg_dir: Path | None = None, attempts: Path | None = None,
            simulate_fn=simulate) -> int:
    """The registered run, once. Refusals happen before any replicate is drawn."""
    seed = check_seed(seed)
    doc, decl_sha = registered_declaration(prereg_id, seed, prereg_dir)
    attempts = attempts if attempts is not None else out_dir / "attempts.jsonl"
    if (out_dir / "report.json").exists():
        raise SystemExit("REFUSED: report.json exists; the declaration is run once")
    panel = load_panel(panel_path)
    if str(panel["replays_json_sha256"]) != (doc.get("replays") or {}).get(
            "replays_json_sha256"):
        raise SystemExit("REFUSED: the panel was not built from the registered replays")
    if str(panel["forward_sha256"]) != last_forward_sha(attempts):
        raise SystemExit("REFUSED: the panel's forward is not the last successful forward "
                         "in attempts.jsonl")
    stamp = _utc()
    base = {"event": "run", "time": stamp, "seed": seed, "declaration_sha256": decl_sha,
            "frozen_code": frozen_code()}
    try:
        res = simulate_fn(panel, seed)
    except Exception as err:                    # engineering: logged, no reading
        append_attempt(attempts, {**base, "rc": 2, "error": repr(err)[:500]})
        raise
    res.update(id=DECL_ID, seed=seed, declaration_sha256=decl_sha, design=design_spec(seed),
               frozen_code=frozen_code(), scorer=BS.describe(),
               panel={"path": str(panel_path), "sha256": sha256(panel_path),
                      "forward_sha256": str(panel["forward_sha256"])}, created=stamp)
    if "engineering_failure" in res:
        f = out_dir / f"engineering_failure_{stamp.replace(':', '')}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        with f.open("x") as fh:
            json.dump(res, fh, indent=1, sort_keys=True, default=float)
        append_attempt(attempts, {**base, "rc": 3, "engineering_failure": str(f)})
        print(f"ENGINEERING FAILURE: {res['engineering_failure']}; logged {f} (no reading)")
        return 3
    p, h = write_report(res, out_dir)
    append_attempt(attempts, {**base, "rc": 0, "report_sha256": h})
    print(f"written {p} sha256 {h}; wording {res['reading']['wording']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rb = sub.add_parser("replays", help="build the seven replays (BEFORE registration)")
    rb.add_argument("--out", default=REPLAYS)
    rb.add_argument("--product", default=PRODUCT)
    sub.add_parser("engineering", help="engineering gate on frozen forwards (no observation)")
    sub.add_parser("check", help="the registration verifies (prints ok or refuses)")
    ex = sub.add_parser("extract", help="the c panel from the CPU replay forward (registered)")
    ex.add_argument("--forward", default=FWD_STEM + ".npz")
    ex.add_argument("--replays", default=f"{REPLAYS}/replays.json")
    ex.add_argument("--out", default=PANEL)
    ru = sub.add_parser("run", help="the registered synthetic run (once)")
    ru.add_argument("--seed", type=int, required=True)
    ru.add_argument("--prereg-id", required=True)
    ru.add_argument("--panel", default=PANEL)
    ru.add_argument("--out-dir", default=OUT_DIR)
    sm = sub.add_parser("smoke", help="tiny plumbing run on a synthetic panel; prints 'ran ok'")
    sm.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "replays":
        decl = REPO / f"results/dataprep/preregistration/{DECL_ID}.json"
        if decl.exists():
            raise SystemExit(f"REFUSED: {DECL_ID} is registered; its replays are frozen in it")
        rec = build_replays(rp(a.out), rp(a.product))
        print(json.dumps({r: x["sha256"] for r, x in rec["replays"].items()}, indent=1))
        print("replays.json sha256", sha256(rp(a.out) / "replays.json"))
        return 0
    if a.cmd == "engineering":
        g = engineering_gate(frozen_contrast_panels())
        print(json.dumps(g, indent=1))
        return 0 if g["ok"] else 3
    if a.cmd == "check":
        doc, sha = registered_declaration(DECL_ID)
        print(f"registered {DECL_ID} sha256 {sha}; seed {doc['design']['seed']}; code ok")
        return 0
    if a.cmd == "extract":
        doc, _ = registered_declaration(DECL_ID)
        rj = rp(a.replays)
        if sha256(rj) != doc["replays"]["replays_json_sha256"]:
            raise SystemExit("REFUSED: replays.json differs from the registered one")
        info = extract(rp(a.forward), rj, rp(a.out), rp(REHEARSAL))
        append_attempt(rp(ATTEMPTS), {"event": "extract", "time": _utc(),
                                      "panel_sha256": info["panel_sha256"],
                                      "reproduction": info["reproduction"]})
        print(json.dumps(info, indent=1))
        return 0
    if a.cmd == "smoke":
        out = rp(a.out)
        if str(out.resolve()).startswith(str((REPO / "results").resolve())):
            raise SystemExit("smoke output must not go under results/")
        res = simulate(synthetic_panel(), SMOKE_SEED, replicates=2, log=lambda *_: None,
                       allow_refused_seed=True)
        ok = res["exact_plumbing"]["ok"] and "reading" in res
        out.mkdir(parents=True, exist_ok=True)
        (out / "smoke_ran.txt").write_text("ran ok\n" if ok else "smoke failed\n")
        print("ran ok" if ok else "smoke failed")
        return 0 if ok else 3
    check_seed(a.seed, real=True)
    return cmd_run(a.seed, a.prereg_id, rp(a.panel), rp(a.out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
