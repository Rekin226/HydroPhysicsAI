"""Score E02c.prospective_2026-11.v1 (T-cov, T-level) and the external-well level test
N3F.ext_level.prospective_<window>.v1, once, after the window closes.

E02c is the one companion of the frozen E02 head forecast (results/twin/prospective_2026-11,
registered by manifest hash 2026-10-11) on I1's unit I1:prospective_heads_2026-11_2027-10_R0wells
(158 wells). It only tightens E02; E02's own verdict is reported unchanged beside it.

Observations. E02c computes nothing from a heads table of its own: its rows are E02's own
step-(5) output (``python -m hydrophysics.twin.prospective score``: evaluation.csv and
report.json), so G0 IS E02's verdict and T-cov / T-level sit on exactly E02's well-months.
The scorer checks that report.json carries E02's registered manifest, that every
evaluation row equals the registered prediction row (predicted_m, lower_m, upper_m,
baseline_m), and that ``twin.validation.evaluate`` on that evaluation.csv reproduces
report.json; any disagreement is an engineering failure, never a verdict.

* T-cov (claim C14a), per county (station code 07 Changhua, 09 Yunlin):
  G0  checks.interval_coverage of E02's step-(5) report.json, unchanged and conjunctive;
  G1  harness.coverage3.coverage_gate (rule gate-3), clusters = well, target 0.90, level 0.95,
      point window 0.85-0.95 when n_eff_lower >= 30;
  G1b month clusters (k = 12), report-only, labelled 'uncalibrated (k = 12 < 20)';
  G2  miss_below <= 0.10 AND miss_above <= 0.10, gates only the word 'balanced'.
* T-level (claim C30, heads part): ZR (E02's registered predicted_m) beats BOTH the
  climatology CL and climatology-plus-trend CLT (pre-origin baselines, frozen CSV) in
  equal-well RMSE, each with the 5 km block-bootstrap 90 % interval of the MSE difference
  below 0 (B 2000, fixed seed); month-clustered interval is a declared secondary;
  drift_diag.fair_temporal_verdict (K = 1.25) through a gap adapter is report-only.
* Secondaries (hybrid, r0, n2c_r1c), report-only: per-county coverage and equal-well RMSE
  against the same CL / CLT on the same rows; never gate, never substitute.
* EXT (rank 5): E02's protocol at the never-calibrated external wells, scored by E02's own
  ``twin.prospective.score`` on the external registration (RMSE ratio to own climatology
  <= 1.0 AND pooled 90 % coverage in [0.85, 0.95]); informative iff >= 30 wells with >= 6
  window months.

Order of a real run (E02c): (1) the window product's MANIFEST must grant prospective_scorer
(metadata only, no table read; otherwise not_evaluable, no look); (2) N2C's step-(1) score
look must be logged and N2C's step-(2) report must exist (otherwise REFUSED: nothing read,
no look); (3) prereg3.assert_can_score (logs the look); (4) only then E02's step-(5) rows
are read. Nothing here runs before 2028-01-15 except ``--rehearsal`` on synthetic
observations (a directory carrying the SYNTHETIC marker; no look is logged).

Usage (repository root):
    python -I dataprep_staging/e02c/score_e02c.py score --window-product <heads product dir>
        --e02-score-dir <E02 step-(5) --out> --e02-new-wells <E02 step-(5) --new-wells>
        --n2c-report <N2C report.json> --out results/dataprep/reports/E02c/score
    python -I dataprep_staging/e02c/score_e02c.py score-ext --window 2026-11
        --window-product <dir> --new-wells <dir> --e02c-report <E02c report.json>
        --out results/dataprep/reports/N3F_ext/score
    python -I dataprep_staging/e02c/score_e02c.py score --rehearsal --synthetic <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCORER_VERSION = "1.1.0"
E = "dataprep_staging/e02c"

# ----------------------------------------------------------------------------- identities
ID = "E02c.prospective_2026-11.v1"
WP = "E02c"
TEST_COV = "E02cP.T_cov_county_2026-11"
TEST_LEVEL = "E02cP.T_level_strict_2026-11"
UNIT = "I1:prospective_heads_2026-11_2027-10_R0wells"
I1_PREREG = "I1.hybrid_tf.prospective_2026-11.v1"
N1_PREREG = "N1.cpu_response.v1"
N1_EXT_UNIT = "N1:ext_wells"
N2C_ID = "N2C.changhua_response.prospective_2026-11.v1"
N2C_TEST = "N2CP.heads_changhua_response"
POWER_ID = "U1.HB.power.v1"
NO_BAND_ID = "U1.no_band.v1"
WINDOW = ("2026-11", "2027-10")
DATA_SCOPE = "heads:2026-11..2027-10"
DEADLINE = dt.datetime(2026, 10, 31, 16, 0, tzinfo=dt.timezone.utc)
NOT_BEFORE = dt.datetime(2028, 1, 15, 0, 0, tzinfo=dt.timezone.utc)
NO_LATER_THAN = "2028-01-31"
ORIGIN = "2022-12"
AXIS_START = "2012-01"                       # forward run and L1 record start
T_FIT = 132                                  # 2012-01 .. 2022-12 (fitted months)

E02_DIR = "results/twin/prospective_2026-11"
E02_PRED_SHA = "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32"
E02_PROTOCOL_SHA = "8a96c0ea9330fd79432eb17a666640ad526728c838b62bd8459cd41de0d908b9"
MODEL_SHA = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
FORWARD = "results/twin_forward/datum_gate_innov.npz"
SECONDARIES = {   # report-only, never gate, never substitute for the primary
    "hybrid": ("results/twin/prospective_2026-11_hybrid",
               "982f4bfa91280539bd47147285dadaee322940c8f140f9fbba3405783ad3abed"),
    "r0": ("results/twin/prospective_2026-11_r0",
           "730cc156b09a86279f00f9f9d4f908886c506028abcec1885c228b36dd2368ec"),
    "n2c_r1c": ("results/twin/prospective_2026-11_n2c_r1c",
                "58f938062d99194add8430a5ade62820db5fa4e5a6e645df8c895cc6f4a0c2ed"),
}
E02_STEP5_FILES = ("report.json", "evaluation.csv")
EVAL_CHECK_COLS = ("predicted_m", "lower_m", "upper_m", "baseline_m")
EVAL_REPRO_KEYS = ("checks", "passed", "rmse_ratio", "coverage", "stations", "rows",
                   "predictions_sha256")

# ----------------------------------------------------------------------------- constants
COUNTY = {"07": "Changhua", "08": "Nantou", "09": "Yunlin"}
COUNTIES_COV = ("Yunlin", "Changhua")
TARGET, GATE_LEVEL = 0.90, 0.95
POINT_WINDOW = (0.85, 0.95)
N_EFF_WINDOW_MIN = 30.0
MIN_K_COUNTY = 20                 # coverage3 ENVELOPE min_k: below it a county is undecided
MAX_MISS_SIDE = 0.10              # G2, wording 'balanced' only
MIN_MONTHS = 6
MIN_WELLS_LEVEL = 100
MIN_WELLS_EXT = 30
BOOT_B = 2000
BOOT_SEED = 20261031
BOOT_CI = 0.90
BLOCK_KM = 5.0
FAIR_K_REPORT = 1.25
SYNTHETIC_MARKER = "SYNTHETIC"
NOT_SCORABLE = "not_scorable"     # verdict of a lead-written record for a test never scored

# the order of the 2028-01-15..31 session (declared in both registrations)
SESSION_ORDER = (
    "(1) prereg3.assert_can_score(N2C.changhua_response.prospective_2026-11.v1) succeeds",
    "(2) N2.policy_response.prospective_2026-11.v1 Yunlin primary (statistics on the 84 "
    "Yunlin wells only)",
    "(3) N2C step (2): N2C's primary and declared statistics computed, its report written",
    "(4) N2's Changhua report-only statistics",
    "(5) E02: python -m hydrophysics.twin.prospective score on results/twin/prospective_2026-11 "
    "(--new-wells inside the window product directory); its evaluation.csv and report.json "
    "are E02c's rows",
    "(6) the hybrid manifest results/twin/prospective_2026-11_hybrid",
    "(7) I1.hybrid_tf.prospective_2026-11.v1 T1/T2",
    "(8) X1.hybrid_tf_tpc.prospective_2026-11.v1 T3a/T3b",
    f"(9) {ID}: MANIFEST check, order check, assert_can_score, then E02's step-(5) rows are "
    "read; T-cov (Yunlin, then Changhua), T-level, report-only secondaries",
    "(10) N3F.ext_level.prospective_<window>.v1 (the external wells), after E02c's report "
    "(or a lead-written not-scorable record for E02c)",
)

# ----------------------------------------------------------------------------- wording
# claim ids follow U1.no_band.v1's C14 scheme (C14a: per-well registered 90 % head interval
# at the monitoring wells, in the window); wording 'passed its registered gate', never
# 'calibrated'. The two power-dependent replacements come from U1.HB.power.v1 (see
# prereg_e02c.power_flags).
WORDING = {
    "cov_supported": "{county}: the registered 90 % head interval passed its registered "
                     "gate on 2026-11..2027-10 (coverage {cov:.2f}; E02c, prospective)",
    "cov_supported_weak_power": "{county}: no gross under-coverage detected by the "
                                "registered test on 2026-11..2027-10 (coverage {cov:.2f}; "
                                "E02c, prospective; U1.HB.power.v1: power < 0.8 at 0.75)",
    "cov_balanced_suffix": "; misses balanced (below {below:.2f}, above {above:.2f})",
    "cov_pooled_failed_county_ok": "registered pooled gate failed; {county} stratum "
                                   "within 0.85-0.95 (report-only)",
    "cov_pooled_failed_county_gate3_only": "registered pooled gate failed; {county} stratum "
                                           "coverage {cov:.2f} passed gate-3 outside "
                                           "0.85-0.95 (n_eff_lower < 30; report-only)",
    "cov_not_supported": "{county}: the registered 90 % interval covered {cov:.2f} of "
                         "{n} well-months, {side} the registered range; the 'predictive "
                         "interval' wording and the half-width summary are withdrawn",
    "cov_not_supported_shock_suffix": "; outside the registered range; may reflect one "
                                      "year's shock (U1.HB.power.v1: false-fail rate > 0.10 "
                                      "at a true 0.90)",
    "cov_not_evaluable": "{county}: undecided ({why})",
    "level_supported": "the frozen 2022-12 level forecast beat climatology and "
                       "climatology-plus-trend on 2026-11..2027-10 (E02c, prospective)",
    "level_not_supported": "prospective test: the frozen level forecast was not better "
                           "than both simple pre-origin baselines on 2026-11..2027-10",
    "level_not_evaluable": "undecided (fewer than {n_min} wells with >= {m_min} window "
                           "months, or no window product granting prospective_scorer)",
    "ext_supported": "frozen forecast vs own climatology at wells never used in "
                     "calibration: passed (ratio {ratio:.2f}, coverage {cov:.2f}; prospective)",
    "ext_not_supported": "frozen forecast does not beat own climatology at unseen wells "
                         "(ratio {ratio:.2f}, coverage {cov:.2f}; prospective)",
    "ext_not_informative": "undecided: fewer than {n_min} external wells with >= {m_min} "
                           "window months",
}


class ScoringRefused(RuntimeError):
    """The scorer refuses to run (date, order, provenance or rehearsal rules)."""


class EngineeringFailure(ScoringRefused):
    """An input of a scoring run is inconsistent (no verdict; fix, log and re-run)."""


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def county_of(sid: str) -> str:
    return COUNTY.get(str(sid)[:2], "other")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _month(s) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


# ----------------------------------------------------------------------------- frames
def window_months(first: str, last: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{first}-01", f"{last}-01", freq="MS")


def annotate(f: pd.DataFrame) -> pd.DataFrame:
    """County, hit and miss side of scored rows (``observed_m``, ``lower_m``, ``upper_m``)."""
    f = f.copy()
    f["station_id"] = f["station_id"].astype(str)
    f["date"] = _month(f["date"])
    f["county"] = f["station_id"].map(county_of)
    f["hit"] = ((f["observed_m"] >= f["lower_m"]) & (f["observed_m"] <= f["upper_m"])
                ).astype(float)
    f["side"] = np.where(f["observed_m"] < f["lower_m"], "below",
                         np.where(f["observed_m"] > f["upper_m"], "above", "in"))
    return f.reset_index(drop=True)


def join_obs(pred: pd.DataFrame, obs: pd.DataFrame, min_months: int = MIN_MONTHS
             ) -> pd.DataFrame:
    """Prediction rows joined with observed monthly heads (``station_id``, ``date``,
    ``observed_m``); keeps rows with an observation at wells with >= ``min_months`` observed
    window months (E02's ``prospective.score`` rule). Rehearsal only (real rows come from
    E02's step (5))."""
    p = pred.copy()
    p["station_id"] = p["station_id"].astype(str)
    p["_m"] = _month(p["date"])
    o = obs[["station_id", "date", "observed_m"]].copy()
    o["station_id"] = o["station_id"].astype(str)
    o["_m"] = _month(o["date"])
    if o.duplicated(["station_id", "_m"]).any():
        raise ValueError("duplicate station-month observations")
    f = p.merge(o.drop(columns="date"), on=["station_id", "_m"], how="left")
    f = f[np.isfinite(f["observed_m"].to_numpy(dtype=float))]
    n = f.groupby("station_id")["_m"].nunique()
    return f[f["station_id"].isin(n[n >= int(min_months)].index)].drop(
        columns="_m").reset_index(drop=True)


def rehearse_e02_step5(pred: pd.DataFrame, obs: pd.DataFrame, protocol: Path,
                       manifest: dict, out: Path) -> Path:
    """REHEARSAL ONLY: what ``twin.prospective.score`` writes (evaluation.csv = prediction
    rows plus ``observed_m`` at wells with >= min_months; report.json = validation.evaluate
    plus the registration manifest), from a synthetic observation table."""
    from hydrophysics.twin.validation import evaluate

    out.mkdir(parents=True, exist_ok=False)
    f = join_obs(pred, obs)
    f[list(pred.columns) + ["observed_m"]].to_csv(out / "evaluation.csv", index=False)
    r = evaluate(out / "evaluation.csv", protocol)
    r["registration_manifest"] = manifest
    (out / "report.json").write_text(json.dumps(r, indent=2, default=str) + "\n")
    return out


def load_e02_step5(score_dir: Path, pred: pd.DataFrame, protocol: Path, manifest: dict
                   ) -> tuple[pd.DataFrame, dict, dict]:
    """E02's step-(5) rows and report, verified against the registered E02 files. Returns
    (annotated rows, report.json unchanged, provenance). Raises EngineeringFailure on any
    disagreement (never a verdict)."""
    from hydrophysics.twin.validation import evaluate

    score_dir = Path(score_dir)
    rep = json.loads((score_dir / "report.json").read_text(encoding="utf-8"))
    if rep.get("registration_manifest") != manifest:
        raise EngineeringFailure("E02 step-(5) report.json does not carry E02's registered "
                                 "manifest")
    ev = pd.read_csv(score_dir / "evaluation.csv", dtype={"station_id": str,
                                                         "model_sha256": str})
    p = pred.copy()
    p["station_id"] = p["station_id"].astype(str)
    p["_m"] = _month(p["date"])
    e = ev.copy()
    e["_m"] = _month(e["date"])
    if e.duplicated(["station_id", "_m"]).any():
        raise EngineeringFailure("duplicate station-months in E02's evaluation.csv")
    m = e.merge(p[["station_id", "_m", *EVAL_CHECK_COLS]], on=["station_id", "_m"],
                how="left", suffixes=("", "_reg"))
    if m[[f"{c}_reg" for c in EVAL_CHECK_COLS]].isna().any().any():
        raise EngineeringFailure("E02's evaluation.csv has rows that are not registered "
                                 "prediction rows")
    for c in EVAL_CHECK_COLS:
        if not np.allclose(m[c].to_numpy(float), m[f"{c}_reg"].to_numpy(float), rtol=0,
                           atol=1e-9):
            raise EngineeringFailure(f"E02's evaluation.csv column {c} differs from the "
                                     "registered predictions")
    r2 = evaluate(score_dir / "evaluation.csv", protocol)
    bad = [k for k in EVAL_REPRO_KEYS if r2.get(k) != rep.get(k)]
    if bad:
        raise EngineeringFailure(f"validation.evaluate on E02's evaluation.csv does not "
                                 f"reproduce report.json: {bad}")
    prov = {"dir": str(score_dir),
            **{f"{n}_sha256": sha256(score_dir / n) for n in E02_STEP5_FILES},
            "rows": int(len(ev)), "wells": int(ev["station_id"].nunique()),
            "reproduced_keys": list(EVAL_REPRO_KEYS)}
    return annotate(ev), rep, prov


# ----------------------------------------------------------------------------- T-cov
def _side_rates(sub: pd.DataFrame) -> tuple[float, float]:
    n = max(len(sub), 1)
    return float((sub["side"] == "below").sum() / n), float((sub["side"] == "above").sum() / n)


def county_coverage(frame: pd.DataFrame, county: str) -> dict:
    """G1 (gate-3, clusters = well), G1b (month clusters, report-only) and G2 for one county."""
    from harness import coverage3

    from hydrophysics.twin import bands

    sub = frame[frame["county"] == county]
    out: dict = {"county": county, "n_rows": int(len(sub)),
                 "n_wells": int(sub["station_id"].nunique())}
    if sub.empty:
        out.update({"evaluable": False, "why": "no scored well-month"})
        return out
    g1 = coverage3.coverage_gate(sub["hit"].to_numpy(), sub["station_id"].to_numpy(),
                                 target=TARGET, level=GATE_LEVEL, window=POINT_WINDOW,
                                 n_eff_window_min=N_EFF_WINDOW_MIN)
    month = sub["date"].dt.strftime("%Y-%m").to_numpy()
    g1b = coverage3.coverage_gate(sub["hit"].to_numpy(), month, target=TARGET,
                                  level=GATE_LEVEL, window=POINT_WINDOW,
                                  n_eff_window_min=N_EFF_WINDOW_MIN)
    g1b_bands = bands.coverage_report(sub["hit"].to_numpy(), month,
                                      side=sub["side"].to_numpy(), target=TARGET,
                                      level=GATE_LEVEL, window=POINT_WINDOW)
    below, above = _side_rates(sub)
    out.update({
        "evaluable": g1["k_clusters"] >= MIN_K_COUNTY,
        "why": (None if g1["k_clusters"] >= MIN_K_COUNTY else
                f"k = {g1['k_clusters']} wells < {MIN_K_COUNTY} (gate-3 envelope)"),
        "G1": g1,
        "G1b_report_only": {"label": f"uncalibrated (k = {g1b['k_clusters']} < 20)",
                            "coverage3": g1b, "bands_coverage_report": g1b_bands},
        "G2": {"miss_below": below, "miss_above": above,
               "balanced": below <= MAX_MISS_SIDE and above <= MAX_MISS_SIDE}})
    return out


def cov_outcome(g0_pass: bool | None, c: dict, power: dict) -> dict:
    """Pre-declared pooled-vs-county outcome and wording for one county."""
    county = c["county"]
    if g0_pass is None:
        return {"verdict": "not_evaluable",
                "wording": WORDING["cov_not_evaluable"].format(
                    county=county, why="E02's pooled check not evaluable")}
    if not c.get("evaluable"):
        return {"verdict": "not_evaluable",
                "wording": WORDING["cov_not_evaluable"].format(county=county, why=c["why"])}
    g1 = c["G1"]
    cov = g1["coverage"]
    if g0_pass and g1["pass"]:
        key = "cov_supported_weak_power" if power["pass_no_gross_only"] else "cov_supported"
        w = WORDING[key].format(county=county, cov=cov)
        if c["G2"]["balanced"]:
            w += WORDING["cov_balanced_suffix"].format(below=c["G2"]["miss_below"],
                                                       above=c["G2"]["miss_above"])
        return {"verdict": "coverage_supported", "wording": w}
    if (not g0_pass) and g1["pass"]:
        if POINT_WINDOW[0] <= cov <= POINT_WINDOW[1]:
            w = WORDING["cov_pooled_failed_county_ok"].format(county=county)
        else:   # gate-3 passed without the point window (n_eff_lower < 30): declared variant
            w = WORDING["cov_pooled_failed_county_gate3_only"].format(county=county, cov=cov)
        return {"verdict": "pooled_gate_failed_county_report_only", "wording": w}
    side = "below" if cov < TARGET else "above"
    w = WORDING["cov_not_supported"].format(county=county, cov=cov, n=g1["n"], side=side)
    if power["fail_may_reflect_shock"]:
        w += WORDING["cov_not_supported_shock_suffix"]
    return {"verdict": "coverage_not_supported", "wording": w}


def g0_of(e02_report: dict) -> bool | None:
    """G0 = checks.interval_coverage of E02's step-(5) report, None when E02's station or
    month checks fail (E02 not evaluable)."""
    ch = e02_report.get("checks") or {}
    if not (ch.get("enough_stations") and ch.get("enough_months_per_station")):
        return None
    return bool(ch.get("interval_coverage"))


def t_cov(frame: pd.DataFrame, e02_report: dict, power: dict) -> dict:
    g0 = g0_of(e02_report)
    out = {"test_id": TEST_COV,
           "G0": {"pass": g0, "coverage_pooled": e02_report.get("coverage"),
                  "source": "checks.interval_coverage of E02's step-(5) report.json, unchanged"},
           "counties": {}}
    for county in COUNTIES_COV:
        c = county_coverage(frame, county)
        c.update(cov_outcome(g0, c, power.get(county, power)))
        out["counties"][county] = c
    return out


# ----------------------------------------------------------------------------- T-level
def block_ids(x, y, km: float = BLOCK_KM) -> np.ndarray:
    """5 km square blocks on TWD97 metres (fixed origin 0, 0)."""
    s = km * 1000.0
    bx = np.floor(np.asarray(x, float) / s).astype(np.int64)
    by = np.floor(np.asarray(y, float) / s).astype(np.int64)
    return np.array([f"{a}_{b}" for a, b in zip(bx, by, strict=True)])


def _well_month_se(frame: pd.DataFrame, cols) -> tuple[list, list, dict]:
    wells = sorted(frame["station_id"].unique())
    months = sorted(frame["date"].unique())
    wi = {w: i for i, w in enumerate(wells)}
    mi = {m: j for j, m in enumerate(months)}
    ii = frame["station_id"].map(wi).to_numpy()
    jj = frame["date"].map(mi).to_numpy()
    se = {}
    for c in cols:
        a = np.full((len(wells), len(months)), np.nan)
        a[ii, jj] = (frame[c].to_numpy(float) - frame["observed_m"].to_numpy(float)) ** 2
        se[c] = a
    return wells, months, se


def _equal_well_mse(se: np.ndarray, well_w: np.ndarray | None = None,
                    month_w: np.ndarray | None = None) -> np.ndarray:
    """Equal-well MSE (E02 convention: mean over wells of each well's mean squared error),
    vectorised over bootstrap weights: ``well_w`` (B, W) and/or ``month_w`` (B, M)."""
    ok = np.isfinite(se)
    s = np.where(ok, se, 0.0)
    if month_w is None:
        per = s.sum(1) / ok.sum(1)                              # (W,)
        if well_w is None:
            return np.asarray(per.mean())
        return (well_w * per[None]).sum(1) / well_w.sum(1)
    num = month_w @ s.T                                          # (B, W)
    den = month_w @ ok.T.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return np.nanmean(per, axis=1)


def paired_bootstrap(frame: pd.DataFrame, a: str, b: str, blocks: dict, B: int = BOOT_B,
                     seed: int = BOOT_SEED, ci: float = BOOT_CI) -> dict:
    """MSE(a) - MSE(b) (equal-well), its block-bootstrap interval (wells resampled by 5 km
    block, with replacement) and the month-clustered secondary interval."""
    wells, months, se = _well_month_se(frame, (a, b))
    d0 = float(_equal_well_mse(se[a]) - _equal_well_mse(se[b]))
    blk = np.array([blocks[w] for w in wells])
    ub, codes = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ub), size=(B, len(ub)))
    cnt = np.zeros((B, len(ub)))
    np.add.at(cnt, (np.repeat(np.arange(B), len(ub)), draw.ravel()), 1.0)
    ww = cnt[:, codes]                                           # (B, W) well weights
    d_blk = _equal_well_mse(se[a], well_w=ww) - _equal_well_mse(se[b], well_w=ww)
    rng2 = np.random.default_rng(seed + 1)
    mdraw = rng2.integers(0, len(months), size=(B, len(months)))
    mcnt = np.zeros((B, len(months)))
    np.add.at(mcnt, (np.repeat(np.arange(B), len(months)), mdraw.ravel()), 1.0)
    d_mon = _equal_well_mse(se[a], month_w=mcnt) - _equal_well_mse(se[b], month_w=mcnt)
    q = ((1 - ci) / 2, 1 - (1 - ci) / 2)
    lo, hi = np.nanquantile(d_blk, q)
    mlo, mhi = np.nanquantile(d_mon, q)
    return {"mse_diff": d0, "block_ci": [float(lo), float(hi)], "n_blocks": int(len(ub)),
            "month_ci_secondary": [float(mlo), float(mhi)], "n_months": len(months),
            "B": B, "seed": seed, "ci": ci, "a": a, "b": b,
            "a_better_block": bool(hi < 0), "a_better_month_secondary": bool(mhi < 0)}


def with_baselines(frame: pd.DataFrame, baselines: pd.DataFrame, blocks: dict) -> pd.DataFrame:
    """Rows with finite CL and CLT at blocked wells with >= MIN_MONTHS such months."""
    b = baselines[["station_id", "date", "cl_m", "clt_m"]].copy()
    b["station_id"] = b["station_id"].astype(str)
    b["date"] = _month(b["date"])
    f = frame.merge(b, on=["station_id", "date"], how="left")
    f = f[np.isfinite(f[["cl_m", "clt_m"]].to_numpy(float)).all(1)]
    f = f[f["station_id"].isin(blocks)]
    n = f.groupby("station_id")["date"].nunique()
    return f[f["station_id"].isin(n[n >= MIN_MONTHS].index)].reset_index(drop=True)


def _rmse3(f: pd.DataFrame, pcol: str = "predicted_m") -> dict:
    _, _, se = _well_month_se(f, (pcol, "cl_m", "clt_m"))
    return {k: float(np.sqrt(_equal_well_mse(se[c]))) for k, c in
            (("ZR", pcol), ("CL", "cl_m"), ("CLT", "clt_m"))}


def t_level(frame: pd.DataFrame, baselines: pd.DataFrame, blocks: dict) -> dict:
    """Strict level rule: ZR beats CL AND CLT, each with a block interval below 0."""
    f = with_baselines(frame, baselines, blocks)
    n_wells = int(f["station_id"].nunique())
    out = {"test_id": TEST_LEVEL, "n_wells": n_wells, "n_rows": int(len(f)),
           "min_wells": MIN_WELLS_LEVEL, "min_months": MIN_MONTHS}
    if n_wells < MIN_WELLS_LEVEL:
        out.update({"verdict": "not_evaluable",
                    "wording": WORDING["level_not_evaluable"].format(n_min=MIN_WELLS_LEVEL,
                                                                     m_min=MIN_MONTHS)})
        return out
    rmse = _rmse3(f)
    vs_cl = paired_bootstrap(f, "predicted_m", "cl_m", blocks)
    vs_clt = paired_bootstrap(f, "predicted_m", "clt_m", blocks)
    ok = (rmse["ZR"] < rmse["CL"] and rmse["ZR"] < rmse["CLT"] and vs_cl["a_better_block"]
          and vs_clt["a_better_block"])
    out.update({"rmse_equal_well_m": rmse, "vs_CL": vs_cl, "vs_CLT": vs_clt,
                "verdict": "level_forecast_supported" if ok else "level_forecast_not_supported",
                "wording": WORDING["level_supported" if ok else "level_not_supported"],
                "by_county_report_only": {
                    c: _rmse3(f[f["county"] == c]) for c in COUNTIES_COV
                    if (f["county"] == c).any()}})
    return out


def fair_verdict_gap(pred_full: np.ndarray, obs_fit: np.ndarray, obs_window: np.ndarray,
                     t_fit: int = T_FIT, gap: int = 46) -> dict:
    """Report-only: ``drift_diag.fair_temporal_verdict`` (K = 1.25, constants unchanged) with
    a gap adapter. ``pred_full`` (W, t_fit + gap + M) is the raw model on the whole axis,
    ``obs_fit`` (W, t_fit) the pre-origin record, ``obs_window`` (W, M) the window. The
    ``gap`` months between the origin and the window (2023-01..2026-10) are passed as NaN,
    never loaded. Can never move C30 to S."""
    from hydrophysics.twin.drift_diag import fair_temporal_verdict

    W, M = obs_window.shape
    if obs_fit.shape != (W, t_fit) or pred_full.shape != (W, t_fit + gap + M):
        raise ValueError("shapes do not match the gap adapter")
    obs = np.concatenate([obs_fit, np.full((W, gap), np.nan), obs_window], axis=1)
    r = fair_temporal_verdict(pred_full, obs, t_fit, k=FAIR_K_REPORT, month_offset=0)
    r["label"] = "report-only; can never move C30 to S"
    return r


def fair_inputs_real(wells: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Arrays for :func:`fair_verdict_gap` at scoring time: the forward run (sha-checked) at
    the wells whose cell matched E02's (export cell check), the pre-origin record from
    heads/L1_clean v3 (use 'baseline', window_end 2022-12) and the window rows of E02's
    step (5)."""
    import export_baselines as X

    w = wells[(wells["cell"] >= 0) & (wells["layer"] >= 0)
              & wells["cell_matches_e02"].astype(bool)]
    ids = sorted(w["station_id"].astype(str))
    fw = X._load_forward()
    long, _, _ = X._open_l1("baseline")
    ax = X.axis(WINDOW[1])
    obs_fit = X.obs_matrix(long, ids, ax)[:, :T_FIT]
    wi = w.set_index(w["station_id"].astype(str))
    pred = np.stack([X.model_series(fw, int(wi.loc[s, "cell"]), int(wi.loc[s, "layer"]))[0]
                     [:len(ax)] for s in ids])
    months = window_months(*WINDOW)
    ow = (frame[frame["station_id"].isin(ids)]
          .pivot_table(index="station_id", columns="date", values="observed_m",
                       aggfunc="first")
          .reindex(index=ids, columns=months).to_numpy(float))
    return {"pred_full": pred, "obs_fit": obs_fit, "obs_window": ow,
            "gap": len(ax) - T_FIT - len(months)}


# ----------------------------------------------------------------------------- secondaries
def secondaries_report_only(frame: pd.DataFrame, baselines: pd.DataFrame, blocks: dict
                            ) -> dict:
    """Report-only lines for the bound secondaries on E02's scored well-months: per county,
    pooled coverage, gate-3 (clusters = well) and equal-well RMSE against the same CL / CLT.
    n2c_r1c is computed only because the order check (N2C step (2) report) has passed."""
    from harness import coverage3

    obs = frame[["station_id", "date", "observed_m"]]
    out = {}
    for k, (d, want) in SECONDARIES.items():
        p = REPO / d / "predictions.csv"
        if not p.is_file() or sha256(p) != want:
            out[k] = {"error": "predictions.csv missing or not the bound sha"}
            continue
        sp = pd.read_csv(p, dtype={"station_id": str})
        sp["date"] = _month(sp["date"])
        f = annotate(sp.merge(obs, on=["station_id", "date"], how="inner"))
        lv = with_baselines(f, baselines, blocks)
        rec = {"label": "report-only; never gates, never substitutes for E02", "counties": {}}
        for c in COUNTIES_COV:
            sub = f[f["county"] == c]
            if sub.empty:
                rec["counties"][c] = {"n_rows": 0}
                continue
            g = coverage3.coverage_gate(sub["hit"].to_numpy(), sub["station_id"].to_numpy(),
                                        target=TARGET, level=GATE_LEVEL, window=POINT_WINDOW,
                                        n_eff_window_min=N_EFF_WINDOW_MIN)
            lc = lv[lv["county"] == c]
            rec["counties"][c] = {"n_rows": int(len(sub)),
                                  "n_wells": int(sub["station_id"].nunique()),
                                  "coverage": float(sub["hit"].mean()),
                                  "gate3_report_only": g,
                                  "rmse_equal_well_m": _rmse3(lc) if len(lc) else None}
        out[k] = rec
    return out


# ----------------------------------------------------------------------------- EXT
def ext_outcome(frame: pd.DataFrame, report: dict | None) -> dict:
    """The external-well verdict from E02's evaluation (``twin.prospective.score`` on the
    external registration): passed iff beats_baseline AND interval_coverage."""
    n = int(frame["station_id"].nunique()) if len(frame) else 0
    out = {"n_wells": n, "min_wells": MIN_WELLS_EXT, "min_months": MIN_MONTHS}
    if n < MIN_WELLS_EXT or report is None:
        out.update({"verdict": "not_informative",
                    "wording": WORDING["ext_not_informative"].format(n_min=MIN_WELLS_EXT,
                                                                     m_min=MIN_MONTHS)})
        return out
    ok = bool(report["checks"]["beats_baseline"] and report["checks"].get("interval_coverage"))
    out.update({"evaluate": report, "verdict": "passed" if ok else "not_supported",
                "wording": WORDING["ext_supported" if ok else "ext_not_supported"].format(
                    ratio=report["rmse_ratio"], cov=report["coverage"]),
                "by_county_report_only": {
                    c: {"n_wells": int(frame.loc[frame.county == c, "station_id"].nunique()),
                        "coverage": float(frame.loc[frame.county == c, "hit"].mean())}
                    for c in sorted(frame["county"].unique())}})
    return out


# ----------------------------------------------------------------------------- guards
def assert_order(required, prereg_dir) -> list:
    """The (prereg_id, test_id) score looks that must already be logged; returns missing."""
    from harness import prereg3

    looks = prereg3.read_chained(Path(prereg_dir) / "looks.jsonl")
    have = {(lk.get("prereg_id"), lk.get("test_id")) for lk in looks
            if lk.get("purpose") == "score"}
    return [r for r in required if tuple(r) not in have]


def _read_record(path, prereg_id: str) -> tuple[dict | None, str]:
    if path is None:
        return None, "no record path given"
    p = Path(path)
    if not p.is_file():
        return None, f"{p} does not exist"
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{p} is not JSON ({exc})"
    if not isinstance(r, dict) or r.get("prereg_id") != prereg_id:
        return None, f"{p} does not name {prereg_id}"
    return r, "ok"


def n2c_order_status(prereg_dir, n2c_report) -> dict:
    """N2C's registered same-session clause, before any Changhua statistic: step (1) = N2C's
    score look in looks.jsonl, step (2) = N2C's report written (a JSON naming N2C's id).
    A lead-written record with verdict 'not_scorable' (N2C refused or never scorable)
    replaces both. Reads N2C's own report only, never a window value."""
    rec, why = _read_record(n2c_report, N2C_ID)
    look = not assert_order([(N2C_ID, N2C_TEST)], prereg_dir)
    st = {"n2c_score_look": look, "n2c_report": str(n2c_report) if n2c_report else None,
          "n2c_report_status": why}
    if rec is None:
        st.update({"ok": False, "why": f"N2C step (2) report missing: {why}"})
    elif rec.get("verdict") == NOT_SCORABLE:
        st.update({"ok": True, "why": "N2C declared not scorable by a lead-written record",
                   "n2c_report_sha256": sha256(n2c_report)})
    elif not look:
        st.update({"ok": False, "why": "N2C step (1) score look not logged"})
    else:
        st.update({"ok": True, "why": "N2C steps (1) and (2) done",
                   "n2c_report_sha256": sha256(n2c_report)})
    return st


def e02c_done_status(prereg_dir, e02c_report) -> dict:
    """EXT runs after E02c: E02c's report.json (any verdict, written by this scorer) or a
    lead-written not-scorable record must exist; a report that scored must have its two
    score looks logged."""
    rec, why = _read_record(e02c_report, ID)
    if rec is None:
        return {"ok": False, "why": f"E02c report missing: {why}"}
    scored = "T_cov" in rec
    missing = assert_order([(ID, TEST_COV), (ID, TEST_LEVEL)], prereg_dir) if scored else []
    if missing:
        return {"ok": False, "why": f"E02c report claims a score but looks are missing: "
                                    f"{missing}"}
    return {"ok": True, "e02c_report": str(e02c_report), "e02c_report_sha256":
            sha256(e02c_report), "e02c_verdict": rec.get("verdict", "scored")}


def product_grants_scorer(product_dir) -> tuple[bool, str]:
    """Whether the window product's MANIFEST grants prospective_scorer (flags 0-1): the
    manifest's metadata only, no table is read."""
    from common import guard
    from common.manifest import load_manifest

    try:
        m, _ = load_manifest(product_dir)
    except (OSError, ValueError) as exc:
        return False, f"no readable MANIFEST ({exc})"
    try:
        guard.check_use(m, "prospective_scorer")
    except (guard.UseNotAllowed, guard.GateNotPassed, guard.LeakageError, KeyError) as exc:
        return False, str(exc)
    return True, "MANIFEST grants prospective_scorer"


def _inside(child, parent) -> bool:
    c, p = Path(child).resolve(), Path(parent).resolve()
    return c == p or p in c.parents


def assert_rehearsal_dir(d: Path) -> None:
    d = Path(d).resolve()
    if not (d / SYNTHETIC_MARKER).is_file():
        raise ScoringRefused(f"--rehearsal needs a synthetic directory carrying a "
                             f"{SYNTHETIC_MARKER} marker file; {d} has none")
    for real in (REPO / "results", REPO / "data_fetch_api", REPO / "AMP_V2"):
        if d == real.resolve() or real.resolve() in d.parents:
            raise ScoringRefused(f"--rehearsal refuses a directory inside {real}")


def read_synthetic_obs(d: Path) -> pd.DataFrame:
    assert_rehearsal_dir(d)
    o = pd.read_csv(Path(d) / "observations.csv", dtype={"station_id": str})
    if "synthetic" not in o.columns or not o["synthetic"].astype(bool).all():
        raise ScoringRefused("every rehearsal row must carry synthetic = True")
    return o


def _write_once(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=True, default=str)
        f.write("\n")


def _refuse_if_exists(p: Path) -> None:
    if p.exists():
        raise FileExistsError(f"{p} exists: each report is written once")


# ----------------------------------------------------------------------------- runners
def score_e02c(doc: dict, frame: pd.DataFrame, e02_report: dict, *, baselines: pd.DataFrame,
               wells: pd.DataFrame) -> dict:
    """Both E02c tests on E02's step-(5) rows (pure; no look, no file)."""
    power = doc["power"]["flags"]
    blocks = dict(zip(wells["station_id"].astype(str), wells["block5"].astype(str),
                      strict=True))
    return {"prereg_id": doc["id"], "scorer_version": SCORER_VERSION,
            "e02_own_verdict_unchanged": e02_report,
            "T_cov": t_cov(frame, e02_report, power),
            "T_level": t_level(frame, baselines, blocks),
            "secondaries_report_only": secondaries_report_only(frame, baselines, blocks),
            "power_flags": power}


def _e02_files() -> tuple[pd.DataFrame, Path, dict]:
    pred = pd.read_csv(REPO / E02_DIR / "predictions.csv", dtype={"station_id": str})
    manifest = json.loads((REPO / E02_DIR / "manifest.json").read_text(encoding="utf-8"))
    return pred, REPO / E02_DIR / "protocol.json", manifest


def run_score(a, prereg_dir=None, rows_dir=None) -> dict:
    from harness import prereg3

    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    rdir = Path(rows_dir) if rows_dir else prereg3.LEDGER_ROWS
    out = Path(a.out)
    if a.rehearsal:
        import prereg_e02c as P
        _refuse_if_exists(out / "rehearsal_report.json")
        sd = Path(a.synthetic)
        obs = read_synthetic_obs(sd)
        pred, protocol, manifest = _e02_files()
        step5 = rehearse_e02_step5(pred, obs, protocol, manifest, out / "rehearsal_e02_step5")
        frame, e02_rep, prov = load_e02_step5(step5, pred, protocol, manifest)
        doc = P.doc(baselines_dir=sd, power_report=None, rehearsal=True)
        rep = score_e02c(doc, frame, e02_rep,
                         baselines=pd.read_csv(sd / "baselines_e02c.csv",
                                               dtype={"station_id": str}),
                         wells=pd.read_csv(sd / "wells_158.csv", dtype={"station_id": str}))
        rep.update({"mode": "REHEARSAL on synthetic observations: not a score, no look "
                            "logged", "e02_step5": prov})
        _write_once(out / "rehearsal_report.json", rep)
        return rep
    _refuse_if_exists(out / "report.json")
    if _now() < NOT_BEFORE:
        raise ScoringRefused(f"not before {NOT_BEFORE:%Y-%m-%d}; use --rehearsal with "
                             f"synthetic data")
    doc, reg_sha = prereg3.load_prereg(ID, prereg_dir=pdir, rows_dir=rdir)
    if doc.get("frozen_code", {}).get(f"{E}/score_e02c.py") != sha256(Path(__file__)):
        raise ScoringRefused("this scorer is not the one the registration froze")
    # (1) MANIFEST metadata only: no table read, no look
    granted, why = product_grants_scorer(a.window_product)
    if not granted:
        rep = {"prereg_id": ID, "verdict": "not_evaluable",
               "why": f"the window heads product does not grant prospective_scorer ({why}); "
                      "no value read, no look logged",
               "window_product": str(a.window_product)}
        _write_once(out / "report.json", rep)
        return rep
    if not a.e02_new_wells or not _inside(a.e02_new_wells, a.window_product):
        raise ScoringRefused("E02's step-(5) --new-wells must lie inside the window product "
                             "directory whose MANIFEST was checked")
    # (2) N2C's same-session clause: refuse (nothing read, no look) until steps (1)-(2)
    order = n2c_order_status(pdir, a.n2c_report)
    if not order["ok"]:
        raise ScoringRefused(f"session order: {order['why']}; nothing read, no look logged")
    sdir = Path(a.e02_score_dir)
    gone = [n for n in E02_STEP5_FILES if not (sdir / n).is_file()]
    if gone:
        raise ScoringRefused(f"E02 step (5) has not run: {gone} missing in {sdir}")
    # (3) the design is intact; logs the score looks
    ok = prereg3.assert_can_score(ID, WP, prereg_dir=pdir, rows_dir=rdir)
    # (4) only now are window rows read
    pred, protocol, manifest = _e02_files()
    frame, e02_rep, prov = load_e02_step5(sdir, pred, protocol, manifest)
    bdir = Path(doc["baselines"]["dir"])
    bdir = bdir if bdir.is_absolute() else REPO / bdir
    wells = pd.read_csv(bdir / "wells_158.csv", dtype={"station_id": str})
    rep = score_e02c(doc, frame, e02_rep,
                     baselines=pd.read_csv(bdir / "baselines_e02c.csv",
                                           dtype={"station_id": str}), wells=wells)
    try:     # report-only; an engineering failure here never touches the two verdicts
        rep["T_level_fair_rule_report_only"] = fair_verdict_gap(
            **fair_inputs_real(wells, frame))
    except Exception as exc:  # noqa: BLE001
        rep["T_level_fair_rule_report_only"] = {"error": repr(exc)}
    rep.update({"registration_sha256": reg_sha, "order": order, "e02_step5": prov,
                "post_registration_looks": ok["post_registration_looks"],
                "unverifiable_prior_looks": ok["unverifiable_prior_looks"],
                "window_product": str(a.window_product), "manifest_check": why})
    _write_once(out / "report.json", rep)
    return rep


def run_score_ext(a, prereg_dir=None, rows_dir=None) -> dict:
    import prereg_e02c_ext as PX
    from harness import prereg3
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    rdir = Path(rows_dir) if rows_dir else prereg3.LEDGER_ROWS
    out = Path(a.out)
    if a.rehearsal:
        _refuse_if_exists(out / "rehearsal_report_ext.json")
        obs = read_synthetic_obs(Path(a.synthetic))
        ed = Path(a.synthetic) / "ext"
        man = (json.loads((ed / "manifest.json").read_text())
               if (ed / "manifest.json").is_file() else {})
        pred = pd.read_csv(ed / "predictions.csv", dtype={"station_id": str})
        step5 = rehearse_e02_step5(pred, obs, ed / "protocol.json", man,
                                   out / "rehearsal_ext_step5")
        frame, rep5, _ = load_e02_step5(step5, pred, ed / "protocol.json", man)
        rep = ext_outcome(frame, rep5)
        rep["mode"] = "REHEARSAL on synthetic observations: not a score, no look logged"
        _write_once(out / "rehearsal_report_ext.json", rep)
        return rep
    _refuse_if_exists(out / "report_ext.json")
    if _now() < NOT_BEFORE:
        raise ScoringRefused(f"not before {NOT_BEFORE:%Y-%m-%d}")
    pid = PX.prereg_id(a.window)
    doc, reg_sha = prereg3.load_prereg(pid, prereg_dir=pdir, rows_dir=rdir)
    if doc.get("frozen_code", {}).get(f"{E}/score_e02c.py") != sha256(Path(__file__)):
        raise ScoringRefused("this scorer is not the one the registration froze")
    granted, why = product_grants_scorer(a.window_product)
    if not granted:
        rep = {"prereg_id": pid, "verdict": "not_informative",
               "why": f"the window heads product does not grant prospective_scorer ({why}); "
                      "no value read, no look logged"}
        _write_once(out / "report_ext.json", rep)
        return rep
    if not a.new_wells or not _inside(a.new_wells, a.window_product):
        raise ScoringRefused("--new-wells must lie inside the window product directory")
    done = e02c_done_status(pdir, a.e02c_report)
    if not done["ok"]:
        raise ScoringRefused(f"session order: E02c must be scored first ({done['why']})")
    ok = prereg3.assert_can_score(pid, PX.WP, prereg_dir=pdir, rows_dir=rdir)
    edir = Path(doc["ext_predictions"]["dir"])
    edir = edir if edir.is_absolute() else REPO / edir
    rep = run_e02_protocol_ext(edir, Path(a.new_wells), out / "e02_protocol_run")
    rep.update({"prereg_id": pid, "registration_sha256": reg_sha, "order": done,
                "post_registration_looks": ok["post_registration_looks"]})
    _write_once(out / "report_ext.json", rep)
    return rep


def run_e02_protocol_ext(edir: Path, new_wells: Path, run_out: Path) -> dict:
    """E02's own scorer (``twin.prospective.score``) on the external registration, then the
    declared outcome. An empty or too-small evaluation is not informative; any other error
    is an engineering failure."""
    from hydrophysics.twin.prospective import score as e02_score

    try:
        r = e02_score(edir, new_wells, run_out)
    except ValueError:
        ev = run_out / "evaluation.csv"
        if not ev.is_file():
            raise
        f = pd.read_csv(ev, dtype={"station_id": str})
        if f["station_id"].nunique() >= MIN_WELLS_EXT:
            raise
        return ext_outcome(f, None)
    f = annotate(pd.read_csv(run_out / "evaluation.csv", dtype={"station_id": str}))
    return ext_outcome(f, r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "score-ext"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--window-product", default=None)
        s.add_argument("--rehearsal", action="store_true")
        s.add_argument("--synthetic", default=None)
        if name == "score":
            s.add_argument("--e02-score-dir", default=None,
                           help="E02 step (5): the --out of hydrophysics.twin.prospective score")
            s.add_argument("--e02-new-wells", default=None,
                           help="the --new-wells E02 step (5) read (inside the window product)")
            s.add_argument("--n2c-report", default=None,
                           help="N2C's step-(2) report (or a lead-written not-scorable record)")
        else:
            s.add_argument("--window", choices=("2026-11", "2026-12"), default="2026-11")
            s.add_argument("--new-wells", default=None)
            s.add_argument("--e02c-report", default=None)
    a = ap.parse_args(argv)
    if a.rehearsal and not a.synthetic:
        ap.error("--rehearsal needs --synthetic <dir>")
    if not a.rehearsal:
        need = (("window_product", "e02_score_dir", "e02_new_wells", "n2c_report")
                if a.cmd == "score" else ("window_product", "new_wells", "e02c_report"))
        miss = [n for n in need if not getattr(a, n)]
        if miss:
            ap.error(f"outside --rehearsal these are required: {miss}")
    try:
        rep = run_score(a) if a.cmd == "score" else run_score_ext(a)
    except ScoringRefused as exc:
        print("REFUSED:", exc)
        return 2
    print(json.dumps({k: v for k, v in rep.items() if k in ("verdict", "mode", "prereg_id")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
