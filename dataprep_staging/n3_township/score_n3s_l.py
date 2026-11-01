"""Score N3SL.township_prospective_roc116-117.v1 (rank 13, N3S-L), once per WRA survey year.

The prospective township-summary test: WRA's printed leveling summaries for survey years ROC116
(2026-05..2027-05) and ROC117 (2027-05..2028-05), read only from the product version
``subsidence_public_reports/prospective`` (user ruling 2026-11-01, fourth set: future survey
intervals only, prospective_scorer allowed; ROC115 and earlier excluded). Three frozen arms:

* ZR  the page model's frozen forward run (results/twin_forward/datum_gate_innov.npz, the E02
      forward, sha cf2c7154), baseline-scenario ensemble-mean subsidence, cm/yr (+ = sinking);
* SL  the statistical background layer mean3_eb (freeze_n3s_l.py): each benchmark's mean of
      its last 3 annual intervals ending by 2021-09, empirical-Bayes shrunk toward its 3 km
      neighbours, persistence (decay 0), no head terms;
* Z   zero.

Tests (per survey year; decisions fixed in the registration, never selected afterwards):

* L1 coast (claims C26 and the coastal part of C30; SL's page role fixed at registration).
  Over the 11 coastal townships (X5NH's list: >= 50 % of their twin cells west of 182 km),
  each arm's township peak rate against the printed bounds (lower 3 cm/yr when listed,
  upper the printed bound), tolerance 0.5 cm/yr. Informative only if >= 3 coastal townships
  are listed. An arm is 'supported' if it meets >= ceil(0.5 x listed) lower bounds, breaks
  <= 2 upper bounds and does at least as well as Z on both counts taken together (net bound
  score lower_met - upper_broken >= Z's; see L1_Z_CLAUSE). The physics column barely moves
  on the coast (it is near Z there), so L1 is in effect a test of SL against Z.
* L2 area (claim C05). Per county, the model's area sinking > 3 cm/yr over the printed
  survey interval (the page's cell count) against WRA's printed significant-subsidence area:
  printed > 0 -> ratio in [1/1.5, 1.5] and closer (in |log ratio|) than SL; printed = 0 ->
  model area <= the county's pre-origin tolerance (km2) and smaller than SL's.
* L3 HSR piers: report-only, never gating.

Order of a real run, survey year Y (the scorer refuses any other order):
(1) a fresh availability record (prereg_n3s_l.py availability: listing titles and dates only)
    shows the ROC Y summary or briefing listed (series 007005 / 007007), first listed by the
    year's deadline (``open-session`` checks this); (2) N3 (rank 6, N3.piles_prospective*) is
    scored or recorded not_scorable, or was never registered; (3) ``open-session --survey-roc
    Y``: prereg3.assert_can_score for both tests of Y (logs the score looks); (4) only then
    the lead fetches and transcribes the reports and builds the product
    (build_prospective_product.py refuses before step 3); (5) ``score --survey-roc Y`` reads
    the product through common.guard (prospective_scorer).
ROC117's run combines both years (``--stage1-report``) or, if ROC116 was never published by
its deadline, a lead-written not_published record. If ROC117 is never published by its
deadline, ``not-published --survey-roc 117 --stage1-report <ROC116 report>`` writes the
record AND the final combination (ROC116 alone) with the page mapping. Nothing runs before
2028-01-15 except ``--rehearsal`` on a synthetic product (SYNTHETIC marker; no look).

Usage (repository root):
    python -I dataprep_staging/n3_township/score_n3s_l.py open-session --survey-roc 116
        --availability <availability_<date>.json> --n3-report <N3 report.json or not_scorable
        record> --out <dir>
    python -I dataprep_staging/n3_township/score_n3s_l.py score --survey-roc 116
        --product <prospective/vN> --freeze-dir <frozen predictions> --session <dir>
        --out <dir> [--stage1-report <ROC116 report.json>]
    python -I dataprep_staging/n3_township/score_n3s_l.py not-published --survey-roc 116 --out <dir>
    python -I dataprep_staging/n3_township/score_n3s_l.py not-published --survey-roc 117
        --stage1-report <ROC116 report.json or its not_published record> --out <dir>
    python -I dataprep_staging/n3_township/score_n3s_l.py score --rehearsal --survey-roc 116
        --product <synthetic> --freeze-dir <synthetic> --out <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
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
E = "dataprep_staging/n3_township"

# ----------------------------------------------------------------------------- identities
ID = "N3SL.township_prospective_roc116-117.v1"
WP = "N3S"
PRODUCT = "subsidence_public_reports/prospective"
SURVEYS = (116, 117)
EXCLUDED_ROC_MAX = 115            # ROC115 and earlier: never scored (ROC115 partly looked)
TESTS = {y: {"L1": f"N3SLP.L1_coast_roc{y}", "L2": f"N3SLP.L2_area_roc{y}"} for y in SURVEYS}
UNITS = {y: f"N3SL:wra_township_summary_roc{y}" for y in SURVEYS}
NOMINAL = {116: ("2026-05", "2027-05"), 117: ("2027-05", "2028-05")}
DATA_SCOPE = {116: "wra_township_summaries:2026-06..2027-05",
              117: "wra_township_summaries:2027-06..2028-05"}
REG_DEADLINE = dt.datetime(2026, 11, 30, 16, 0, tzinfo=dt.timezone.utc)
REG_MONTH = "2026-11"             # survey intervals must end after this month
NOT_BEFORE = dt.datetime(2028, 1, 15, 0, 0, tzinfo=dt.timezone.utc)
NO_LATER_THAN = {116: "2029-12-31", 117: "2030-12-31"}
N3_PREFIX = "N3.piles_prospective"       # rank 6's id (the lead's registered id must start so)
FORWARD = "results/twin_forward/datum_gate_innov.npz"
MODEL_SHA = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"

# ----------------------------------------------------------------------------- constants
COUNTIES = ("雲林縣", "彰化縣")
SHORT = {"雲林縣": "雲林", "彰化縣": "彰化"}
# X5NH's coastal townships (>= 50 % of their twin cells west of 182 km), unchanged
COAST = ("麥寮鄉", "北港鎮", "東勢鄉", "水林鄉", "臺西鄉", "四湖鄉", "口湖鄉", "褒忠鄉",
         "元長鄉", "崙背鄉", "大城鄉")
COAST_COUNTY = {t: ("彰化縣" if t == "大城鄉" else "雲林縣") for t in COAST}
DISTAL_KM = 182.0
LISTED_FLOOR = 3.0                # cm/yr, WRA's listing threshold
TOL = 0.5                         # cm/yr on every township bound (X5NH's)
L1_MIN_LISTED = 3                 # informative gate: listed coastal townships in the year
L1_LOWER_FRAC = 0.5               # lower bounds met >= ceil(0.5 x listed)
L1_MAX_UPPER_BROKEN = 2
# The plan's 'at least as well as Z on both counts' read as ONE comparison on the net bound
# score (lower_met - upper_broken >= Z's). Read as two separate comparisons it would require
# upper_broken <= Z's, normally 0 (Z breaks an upper bound only if a printed county
# maximum is below -0.5 cm/yr), which silently cancels the explicit '<= 2 broken' allowance; the specific numeric
# allowance governs, the Z comparison keeps both counts. Fixed from the rule's logic alone.
L1_Z_CLAUSE = "net"
AVAIL_MAX_AGE_DAYS = 14
AVAIL_SERIES = ("007005", "007007", "007001")
SUMMARY_SERIES = ("007005", "007007")   # summary report, annual briefing (not monthly 007001)
AREA_RATE = 3.0                   # cm/yr: a cell counts when its rate exceeds this
AREA_RATIO = 1.5                  # L2 band [1/1.5, 1.5]
L2_MIN_PASS = 3                   # of the 4 county-years
ARMS = ("ZR", "SL", "Z")
SYNTHETIC_MARKER = "SYNTHETIC"
NOT_SCORABLE = "not_scorable"
NOT_PUBLISHED = "not_published"
SL_ROLES = ("report_only", "eligible_ds")

WORDING = {
    "l1_supported": "{arm}: coastal township bounds met on WRA survey {years} (N3S-L L1, "
                    "prospective)",
    "l1_not_supported": "{arm}: failed the registered coastal-township test on WRA survey "
                        "{years} (N3S-L L1, prospective)",
    "l1_undecided": "{arm}: coastal test undecided ({why})",
    "l2_supported": "county area sinking > 3 cm/yr within x1.5 of WRA's printed area and "
                    "closer than the statistical layer in {n_pass} of {n} county-years "
                    "(ROC116/117; N3S-L L2, prospective)",
    "l2_not_supported": "the model's county area sinking > 3 cm/yr failed the registered "
                        "test against WRA's printed area ({n_pass} of {n} county-years; N3S-L "
                        "L2, prospective)",
    "l2_undecided": "area test undecided (fewer than 3 evaluable county-years)",
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


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _flag(v) -> bool:
    """A CSV boolean that may be missing (NaN reads as False, never as True)."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v) if pd.notna(v) else False


# ----------------------------------------------------------------------------- L1
def coast_rows(bounds: pd.DataFrame, survey_roc: int) -> pd.DataFrame:
    """The product's township_bounds rows of the 11 coastal townships for one survey year."""
    if bounds is None or bounds.empty:
        return pd.DataFrame(columns=["township", "county"])
    b = bounds[(bounds["survey_year"].astype(int) == survey_roc + 1911)
               & bounds["township"].isin(COAST)]
    return b.drop_duplicates(["county", "township"]).reset_index(drop=True)


def check_bounds(peaks: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """X5NH's bound semantics: a listed, unambiguous township carries a lower bound (peak >=
    3 - TOL); every unambiguous township an upper bound (peak <= upper + TOL). A township
    with no prediction (SL: no benchmark inside) misses its lower bound and keeps its upper."""
    out = []
    for r in rows.itertuples():
        peak = peaks.get(r.township, np.nan)
        listed, anyl = _flag(r.listed_by_all), _flag(r.listed_by_any)
        ambiguous = _flag(getattr(r, "ambiguous", False)) or (anyl and not listed)
        upper = float(r.peak_rate_upper_cm_yr)
        have = peak is not None and np.isfinite(peak)
        lower_met = None
        if listed and not ambiguous:
            lower_met = bool(have and peak >= LISTED_FLOOR - TOL)
        upper_met = None if ambiguous else bool((not have) or peak <= upper + TOL)
        out.append({"township": r.township, "listed": listed, "ambiguous": ambiguous,
                    "upper_cm_yr": upper, "peak_cm_yr": float(peak) if have else None,
                    "lower_met": lower_met, "upper_met": upper_met})
    return pd.DataFrame(out, columns=["township", "listed", "ambiguous", "upper_cm_yr",
                                      "peak_cm_yr", "lower_met", "upper_met"])


def bound_counts(df: pd.DataFrame) -> dict:
    lo = df["lower_met"].dropna()
    up = df["upper_met"].dropna()
    return {"lower_n": int(len(lo)), "lower_met": int(lo.astype(bool).sum()),
            "upper_n": int(len(up)), "upper_broken": int((~up.astype(bool)).sum())}


def z_clause(c: dict, z: dict, reading: str = L1_Z_CLAUSE) -> bool:
    """'At least as well as Z on both counts'. 'net' (registered): lower_met - upper_broken
    >= Z's. 'literal' (report-only, for the passability table): lower_met >= Z's AND
    upper_broken <= Z's."""
    if reading == "net":
        return c["lower_met"] - c["upper_broken"] >= z["lower_met"] - z["upper_broken"]
    return c["lower_met"] >= z["lower_met"] and c["upper_broken"] <= z["upper_broken"]


def l1_passability(n_listed: int, sigma: float, reading: str = L1_Z_CLAUSE, n_sim: int = 2000,
                   seed: int = 20261101) -> float:
    """Design-time synthetic check (no data): P(an arm is 'supported' in one year) when the
    arm equals a synthetic truth field plus N(0, sigma) error on each township peak. Truth:
    11 coastal townships, ``n_listed`` of them listed with peaks U(3, 6) cm/yr, the rest
    U(0, 3); upper bound = the maximum listed peak when listed, else 3 (lev_build.bounds);
    tolerance TOL. Generic ranges, not taken from any WRA summary."""
    rng = np.random.default_rng(seed)
    n = len(COAST)
    listed = np.arange(n) < n_listed
    truth = np.where(listed, rng.uniform(3.0, 6.0, (n_sim, n)), rng.uniform(0.0, 3.0,
                                                                            (n_sim, n)))
    upper = np.where(listed, truth[:, listed].max(axis=1, keepdims=True), LISTED_FLOOR)
    arm = truth + sigma * rng.standard_normal((n_sim, n))
    lower_met = ((arm >= LISTED_FLOOR - TOL) & listed).sum(axis=1)
    broken = (arm > upper + TOL).sum(axis=1)
    need = math.ceil(L1_LOWER_FRAC * n_listed)
    z = {"lower_met": 0, "upper_broken": 0}
    ok = [lm >= need and ub <= L1_MAX_UPPER_BROKEN
          and z_clause({"lower_met": lm, "upper_broken": ub}, z, reading)
          for lm, ub in zip(lower_met.tolist(), broken.tolist(), strict=True)]
    return float(np.mean(ok))


def l1_year(rows: pd.DataFrame, peaks_by_arm: dict, yunlin_list_ok: bool) -> dict:
    """One survey year: per-arm counts and verdicts (supported / not_supported /
    not_informative / not_evaluable). Evaluable only if Yunlin's township list is verified
    for the year (all 10 Yunlin coastal townships carry a bound row)."""
    yl = [t for t in COAST if COAST_COUNTY[t] == "雲林縣"]
    have = set(rows["township"]) if len(rows) else set()
    if not yunlin_list_ok or not set(yl) <= have:
        return {"status": "not_evaluable", "why": "Yunlin's township list for the year is not "
                "verified (two transcriptions disagree, or not printed)",
                "arms": {a: {"verdict": "not_evaluable"} for a in peaks_by_arm}}
    counts = {a: bound_counts(check_bounds(p, rows)) for a, p in peaks_by_arm.items()}
    n_listed = counts["Z"]["lower_n"]
    out = {"status": "informative" if n_listed >= L1_MIN_LISTED else "not_informative",
           "n_listed": n_listed, "n_rows": int(len(rows)), "arms": {}}
    need = math.ceil(L1_LOWER_FRAC * n_listed)
    z = counts["Z"]
    for a, c in counts.items():
        if n_listed < L1_MIN_LISTED:
            v = "not_informative"
        else:
            ok = (c["lower_met"] >= need and c["upper_broken"] <= L1_MAX_UPPER_BROKEN
                  and z_clause(c, z))
            v = "supported" if ok else "not_supported"
        out["arms"][a] = {**c, "need_lower": need, "net": c["lower_met"] - c["upper_broken"],
                          "z_clause": z_clause(c, z), "verdict": v}
    return out


def l1_combine(years: dict) -> dict:
    """Per arm over the survey years: 'supported' iff supported in every decided year and at
    least one year decided; 'undecided' if no year is decided; else 'not_supported'."""
    arms = sorted({a for y in years.values() for a in (y.get("arms") or {})})
    out = {}
    for a in arms:
        vs = {k: (y.get("arms") or {}).get(a, {}).get("verdict", y.get("status"))
              for k, y in years.items()}
        dec = {k: v for k, v in vs.items() if v in ("supported", "not_supported")}
        if not dec:
            v = "undecided"
        elif all(x == "supported" for x in dec.values()):
            v = "supported"
        else:
            v = "not_supported"
        out[a] = {"verdict": v, "by_year": vs}
    return out


# ----------------------------------------------------------------------------- L2
def printed_area(area: pd.DataFrame, intervals: pd.DataFrame, statements: pd.DataFrame,
                 county: str, survey_roc: int) -> dict:
    """WRA's printed significant-subsidence area (km2), first verified source of: table 1-1
    (county_area_by_year, year label = ROC year), table 2-x (county_intervals area_gt3_km2,
    survey interval ending in the year), the statement's significant area. Disagreements
    between sources are reported, never resolved by choice."""
    short = SHORT[county]
    cands = []
    if area is not None and len(area):
        a = area[area["county"].astype(str).str.contains(short[0])
                 & area["year_label"].astype(str).str.match(rf"^\s*{survey_roc}\b")]
        a = a.sort_values("edition_roc") if "edition_roc" in a else a
        v = pd.to_numeric(a["area_km2"], errors="coerce").dropna()
        if len(v):
            cands.append(("table_1_1", float(v.iloc[-1])))
    if intervals is not None and len(intervals):
        i = intervals[(intervals["county"] == short)
                      & (intervals["survey_year"].astype(int) == survey_roc + 1911)]
        v = pd.to_numeric(i.get("area_gt3_km2", pd.Series(dtype=float)), errors="coerce")
        v = v.dropna()
        if len(v):
            cands.append(("table_2_x", float(v.iloc[-1])))
    if statements is not None and len(statements):
        s = statements[(statements["county"] == short)
                       & (statements["survey_year"].astype(int) == survey_roc + 1911)]
        v = pd.to_numeric(s.get("significant_area_km2", pd.Series(dtype=float)),
                          errors="coerce").dropna()
        if len(v):
            cands.append(("statement", float(v.iloc[-1])))
    if not cands:
        return {"value": None, "source": None, "all": {}}
    allv = {k: v for k, v in cands}
    return {"value": cands[0][1], "source": cands[0][0], "all": allv,
            "sources_disagree": len({round(v, 6) for v in allv.values()}) > 1}


def _logr(a: float, p: float) -> float:
    return math.inf if a <= 0 else abs(math.log(a / p))


def l2_county_year(p: float | None, m: float, s: float, tol: float | None) -> dict:
    """One county-year. p printed, m model (ZR), s statistical layer, tol the county's
    pre-origin tolerance (km2), used only when the printed area is zero."""
    if p is None or not np.isfinite(p) or p < 0:
        return {"evaluable": False, "why": "no verified printed area", "pass": None}
    if p > 0:
        ratio = m / p
        in_band = bool(m > 0 and 1 / AREA_RATIO <= ratio <= AREA_RATIO)
        closer = bool(_logr(m, p) < _logr(s, p))
        return {"evaluable": True, "branch": "ratio", "printed_km2": p, "model_km2": m,
                "sl_km2": s, "ratio": ratio, "in_band": in_band, "closer_than_sl": closer,
                "sl_in_band": bool(s > 0 and 1 / AREA_RATIO <= s / p <= AREA_RATIO),
                "pass": in_band and closer}
    if tol is None or not np.isfinite(tol):
        return {"evaluable": False, "why": "printed area is zero and no pre-origin tolerance",
                "pass": None}
    within = bool(m <= tol)
    closer = bool(abs(m) < abs(s))
    return {"evaluable": True, "branch": "zero", "printed_km2": 0.0, "model_km2": m,
            "sl_km2": s, "tolerance_km2": tol, "within_tolerance": within,
            "closer_than_sl": closer, "pass": within and closer}


def l2_combine(flags: dict) -> dict:
    """{(roc, county): county-year dict} -> verdict over the (up to) 4 county-years."""
    ev = {k: v for k, v in flags.items() if v.get("evaluable")}
    n_pass = sum(bool(v["pass"]) for v in ev.values())
    if n_pass >= L2_MIN_PASS:
        verdict = "supported"
    elif len(ev) < L2_MIN_PASS:
        verdict = "undecided"
    else:
        verdict = "not_supported"
    return {"verdict": verdict, "n_evaluable": len(ev), "n_pass": n_pass,
            "n_county_years": len(flags)}


# ----------------------------------------------------------------------------- L3
def l3_piers(hsr: pd.DataFrame | None, pier_xy: pd.DataFrame | None, model_rate,
             survey_roc: int) -> dict:
    """Report-only: Spearman over printed pier rates and the model rate at each pier's cell,
    and whether the maximum pier rate is within a factor 1.5. Needs a pier location table
    (TK chainage -> TWD97 x, y, from metadata only) registered with the predictions."""
    if pier_xy is None:
        return {"status": "not_evaluable", "why": "no pier location frame registered (the "
                "OSM centreline carries no TK kilometrage)", "label": "report-only"}
    if hsr is None or not len(hsr):
        return {"status": "not_evaluable", "why": "no verified pier table",
                "label": "report-only"}
    h = hsr[pd.to_numeric(hsr["roc_year"], errors="coerce") == survey_roc]
    h = h.merge(pier_xy, on="chainage", how="inner")
    h = h[pd.to_numeric(h["settlement_cm"], errors="coerce").notna()]
    if len(h) < 5:
        return {"status": "not_evaluable", "why": "fewer than 5 located piers",
                "label": "report-only"}
    obs = -pd.to_numeric(h["settlement_cm"]).to_numpy(float)   # printed sign: - = sinking
    mod = np.array([model_rate(x, y) for x, y in zip(h["x"], h["y"], strict=True)])
    ok = np.isfinite(mod)
    rho = pd.Series(obs[ok]).rank().corr(pd.Series(mod[ok]).rank())
    mx_o, mx_m = float(np.max(obs[ok])), float(np.max(mod[ok]))
    within = bool(mx_o > 0 and mx_m > 0 and 1 / AREA_RATIO <= mx_m / mx_o <= AREA_RATIO)
    return {"status": "computed", "label": "report-only", "n": int(ok.sum()),
            "spearman": float(rho), "max_printed_cm_yr": mx_o, "max_model_cm_yr": mx_m,
            "max_within_1_5": within}


# ----------------------------------------------------------------------------- frozen inputs
def load_freeze(freeze_dir) -> dict:
    """The frozen predictions directory (freeze_n3s_l.py): files, summary and verify."""
    d = Path(freeze_dir)
    need = ("predictions.csv", "cells.csv", "summary.json")
    miss = [n for n in need if not (d / n).is_file()]
    if miss:
        raise EngineeringFailure(f"{d}: missing frozen files {miss}")
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    bad = [n for n, h in (s.get("files") or {}).items()
           if not (d / n).is_file() or sha256(d / n) != h]
    if bad:
        raise EngineeringFailure(f"{d}: frozen files changed since the freeze: {bad}")
    pred = pd.read_csv(d / "predictions.csv")
    cells = pd.read_csv(d / "cells.csv")
    tol = s.get("area_tolerance_km2") or {}
    pier = pd.read_csv(d / "pier_xy.csv", dtype={"chainage": str}) if \
        (d / "pier_xy.csv").is_file() else None
    return {"dir": d, "summary": s, "pred": pred, "cells": cells, "tol": tol, "pier": pier}


def frozen_values(pred: pd.DataFrame, arm: str, roc: int, level: str, quantity: str) -> dict:
    p = pred[(pred["arm"] == arm) & (pred["survey_roc"] == roc) & (pred["level"] == level)
             & (pred["quantity"] == quantity)]
    key = "township" if level == "township" else "county"
    return {str(r[key]): float(r["value"]) for _, r in p.iterrows()}


def zr_quantities(run, cells: pd.DataFrame, start: str, end: str) -> dict:
    """ZR township peaks/means and county areas over one interval, from the forward run."""
    rate = run.rate(start, end)
    town = {}
    for (cty, t), g in cells.groupby(["county", "township"]):
        r = rate[g["cell"].to_numpy(int)]
        dist = g["distal"].astype(bool).to_numpy()
        town[(cty, t)] = {"peak": float(np.max(r)) if len(r) else np.nan,
                          "mean": float(np.mean(r)) if len(r) else np.nan,
                          "distal_peak": float(np.max(r[dist])) if dist.any() else np.nan}
    cell_km2 = (float(run.dx) / 1000.0) ** 2
    area = {}
    for cty, g in cells.groupby("county"):
        c = np.unique(g["cell"].to_numpy(int))
        area[cty] = float((rate[c] > AREA_RATE).sum() * cell_km2)
    return {"town": town, "area": area}


def check_zr_frozen(run, fz: dict, roc: int) -> None:
    """Engineering check: the forward recomputes the frozen nominal-interval values exactly."""
    s, e = NOMINAL[roc]
    q = zr_quantities(run, fz["cells"], s, e)
    want_t = frozen_values(fz["pred"], "ZR", roc, "township", "peak_cm_yr")
    want_a = frozen_values(fz["pred"], "ZR", roc, "county", "area_gt3_km2")
    got_t = {t: v["peak"] for (_, t), v in q["town"].items() if t in want_t}
    bad = [t for t in want_t if not np.isclose(got_t.get(t, np.nan), want_t[t], rtol=0,
                                               atol=1e-9, equal_nan=True)]
    bad += [c for c in want_a if not np.isclose(q["area"].get(c, np.nan), want_a[c], rtol=0,
                                                atol=1e-9, equal_nan=True)]
    if bad:
        raise EngineeringFailure(f"ZR does not reproduce the frozen ROC{roc} values: {bad}")


# ----------------------------------------------------------------------------- one year
def interval_of(intervals: pd.DataFrame | None, county: str, roc: int) -> tuple[str, str, str]:
    """The printed survey interval of a county-year, else the nominal one (declared)."""
    if intervals is not None and len(intervals):
        i = intervals[(intervals["county"] == SHORT[county])
                      & (intervals["survey_year"].astype(int) == roc + 1911)]
        if len(i) and pd.notna(i["start_ym"].iloc[-1]) and pd.notna(i["end_ym"].iloc[-1]):
            return str(i["start_ym"].iloc[-1]), str(i["end_ym"].iloc[-1]), "printed"
    s, e = NOMINAL[roc]
    return s, e, "nominal"


def score_year(roc: int, tables: dict, fz: dict, run) -> dict:
    """L1, L2 (county-years) and L3 for one survey year from the product's tables."""
    iv = tables.get("county_intervals.csv")
    yl_s, yl_e, yl_how = interval_of(iv, "雲林縣", roc)
    ch_s, ch_e, ch_how = interval_of(iv, "彰化縣", roc)
    zr = {"雲林縣": zr_quantities(run, fz["cells"], yl_s, yl_e),
          "彰化縣": zr_quantities(run, fz["cells"], ch_s, ch_e)}
    rows = coast_rows(tables.get("township_bounds.csv"), roc)
    peaks = {"ZR": {t: zr[COAST_COUNTY[t]]["town"].get((COAST_COUNTY[t], t), {}).get("peak",
                                                                                      np.nan)
                    for t in COAST},
             "SL": frozen_values(fz["pred"], "SL", roc, "township", "peak_cm_yr"),
             "Z": {t: 0.0 for t in COAST}}
    st = tables.get("township_status.csv")
    yl_ok = bool(st is not None and len(st) and (
        (st["county"] == "雲林") & (st["survey_year"].astype(int) == roc + 1911)).any())
    l1 = l1_year(rows, peaks, yl_ok)
    l1["intervals"] = {"雲林縣": [yl_s, yl_e, yl_how], "彰化縣": [ch_s, ch_e, ch_how]}
    l1["rows"] = {a: check_bounds(p, rows).to_dict("records") for a, p in peaks.items()}
    sl_area = frozen_values(fz["pred"], "SL", roc, "county", "area_gt3_km2")
    l2 = {}
    for cty in COUNTIES:
        pa = printed_area(tables.get("county_area_by_year.csv"), iv,
                          tables.get("township_statements.csv"), cty, roc)
        rec = l2_county_year(pa["value"], zr[cty]["area"][cty], sl_area.get(cty, 0.0),
                             fz["tol"].get(cty))
        rec["printed_source"] = pa
        rec["z_km2"] = 0.0
        l2[cty] = rec

    def model_rate(x, y):
        c = run.cell_of(x, y)
        return float(run.rate(yl_s, yl_e)[c]) if c >= 0 else np.nan
    l3 = l3_piers(tables.get("hsr_piers.csv"), fz["pier"], model_rate, roc)
    return {"survey_roc": roc, "L1": l1, "L2_county_years": l2, "L3_report_only": l3}


# ----------------------------------------------------------------------------- availability
_ROC_IN_TITLE = (re.compile(r"(?<!\d)(1\d{2})\s*(?:年|年度)"),
                 re.compile(r"(?<!\d)(1\d{2})\s*[~～\-至]\s*(1\d{2})\s*(?:年|年度)"),
                 re.compile(r"^\s*(1\d{2})(?!\d)"))


def roc_status(titles: list) -> dict:
    """Which of ROC115-117 a listing names, from the TITLES only (never the publication
    date column): 'NNN年' / 'NNN年度', a range 'NNN~MMM年', or a title starting with 'NNN'."""
    return {str(r): ("listed" if r in rocs_named(titles) else "not_listed")
            for r in (115, 116, 117)}


def rocs_named(titles: list) -> set:
    seen = set()
    for t in titles:
        t = str(t.get("title", "") if isinstance(t, dict) else t)
        for rx in _ROC_IN_TITLE:
            for m in rx.finditer(t):
                seen.update(int(g) for g in m.groups() if g)
    return seen


def published_ym(s) -> str | None:
    """A listing's publication date ('發布年月', ROC or Gregorian) as YYYY-MM, else None."""
    s = str(s or "")
    m = re.search(r"(?<!\d)(20\d{2})\D{1,3}(\d{1,2})(?!\d)", s)
    if m:
        return f"{int(m[1]):04d}-{int(m[2]):02d}"
    m = re.search(r"(?<!\d)(1\d{2})\D{1,3}(\d{1,2})(?!\d)", s)
    if m and 1 <= int(m[2]) <= 12:
        return f"{int(m[1]) + 1911:04d}-{int(m[2]):02d}"
    return None


def first_listed(rows: list, roc: int) -> str | None:
    """Earliest publication month (YYYY-MM) of a listing row whose title names ``roc``."""
    yms = [published_ym(r.get("published")) for r in rows
           if isinstance(r, dict) and roc in rocs_named([r])]
    yms = [y for y in yms if y]
    return min(yms) if yms else None


def availability_record(listings: dict, checked: str | None = None) -> dict:
    """{series: [titles, or {'title', 'published'} rows]} -> the record (no values): per
    series the ROC115-117 status from titles and the first publication month naming each."""
    per = {}
    for s, rows in listings.items():
        per[s] = {"n_rows": len(rows), "status": roc_status(rows),
                  "first_listed": {str(r): first_listed(rows, r) for r in (115, 116, 117)}}
    ok = all(v["status"][str(r)] == "not_listed" for v in per.values() for r in SURVEYS)
    return {"id": ID, "kind": "availability_check", "checked": checked or _now().isoformat(),
            "series": per, "unpublished_116_117": bool(ok),
            "method": "WRA portal listing pages (titles and publication dates only); no PDF "
                      "fetched, no value read; ROC115 is recorded and excluded whatever its "
                      "status"}


def _checked(rec: dict) -> dt.datetime:
    return dt.datetime.fromisoformat(str(rec.get("checked")).replace("Z", "+00:00"))


def availability_for_session(path, roc: int) -> dict:
    """Step (1): the record is this registration's, at most AVAIL_MAX_AGE_DAYS old, lists
    the ROC ``roc`` summary or briefing (SUMMARY_SERIES), and the first listing month (else
    the check date) is on or before the year's deadline. Raises ScoringRefused otherwise."""
    if path is None or not Path(path).is_file():
        raise ScoringRefused("give --availability: a fresh record showing the ROC summary listed")
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec.get("id") != ID or rec.get("kind") != "availability_check":
        raise ScoringRefused("the availability record is not this registration's")
    age = (_now() - _checked(rec)).days
    if not 0 <= age <= AVAIL_MAX_AGE_DAYS:
        raise ScoringRefused(f"availability record is {age} days old (max "
                             f"{AVAIL_MAX_AGE_DAYS}): re-run availability")
    ser = rec.get("series") or {}
    listed = [s for s in SUMMARY_SERIES
              if (ser.get(s) or {}).get("status", {}).get(str(roc)) == "listed"]
    if not listed:
        raise ScoringRefused(f"ROC{roc}'s summary is not listed in {SUMMARY_SERIES}: wait")
    firsts = [ser[s].get("first_listed", {}).get(str(roc)) for s in listed]
    firsts = [f for f in firsts if f]
    deadline = NO_LATER_THAN[roc]
    when = min(firsts) if firsts else _checked(rec).date().isoformat()[:7]
    basis = "first_listed" if firsts else "record_checked"
    if when > deadline[:7]:
        raise ScoringRefused(f"ROC{roc} was first listed {when} ({basis}), after its deadline "
                             f"{deadline}: write the not-published record instead")
    return {"record_sha256": sha256(path), "checked": rec.get("checked"), "listed_in": listed,
            "first_listed": when, "basis": basis, "deadline": deadline}


# ----------------------------------------------------------------------------- order, session
def index_rows(prereg_dir) -> list[dict]:
    from harness import prereg3
    p = Path(prereg_dir) / "INDEX.jsonl"
    return prereg3.read_chained(p) if p.exists() else []


def score_looks(prereg_dir, prereg_id_pred, test_ids=None) -> list[dict]:
    from harness import prereg3
    p = Path(prereg_dir) / "looks.jsonl"
    if not p.exists():
        return []
    return [lk for lk in prereg3.read_chained(p)
            if lk.get("purpose") == "score" and prereg_id_pred(str(lk.get("prereg_id") or ""))
            and (test_ids is None or lk.get("test_id") in test_ids)]


def n3_order_status(prereg_dir, n3_report: Path | None) -> dict:
    """N3 first: if an N3.piles_prospective* registration exists, its score look must be
    logged and its report (or a lead-written not_scorable record) must name it."""
    regs = [r["id"] for r in index_rows(prereg_dir) if str(r["id"]).startswith(N3_PREFIX)]
    if not regs:
        return {"ok": True, "n3": None, "note": "no N3 registration in the index: no order "
                "constraint (rank 6 not registered as gating)"}
    n3 = regs[-1]
    rec = None
    if n3_report is not None and Path(n3_report).is_file():
        rec = json.loads(Path(n3_report).read_text(encoding="utf-8"))
    names = rec is not None and (rec.get("prereg_id") == n3 or rec.get("id") == n3)
    not_scorable = bool(names and rec.get("verdict") == NOT_SCORABLE)
    looks = score_looks(prereg_dir, lambda x: x == n3)
    ok = bool(names and (looks or not_scorable))
    return {"ok": ok, "n3": n3, "n3_report": None if rec is None else sha256(n3_report),
            "n3_score_look": bool(looks), "n3_not_scorable": not_scorable}


def session_status(prereg_dir, roc: int) -> dict:
    tids = set(TESTS[roc].values())
    looks = score_looks(prereg_dir, lambda x: x == ID, tids)
    return {"ok": {lk["test_id"] for lk in looks} == tids,
            "looks": sorted({(lk["test_id"], lk["time"]) for lk in looks})}


def open_session(roc: int, n3_report, out, prereg_dir=None, rows_dir=None,
                 availability=None) -> dict:
    """Steps (1)-(3): checks the date, the availability record (ROC ``roc`` listed by its
    deadline; the deadline applies to the first listing, not to the session date) and N3's
    order, then logs both score looks of year ``roc`` (prereg3.assert_can_score). Writes
    ``session_roc<Y>.json``; no report value is read."""
    from harness import prereg3
    if roc not in SURVEYS:
        raise ScoringRefused(f"survey ROC{roc} is not registered (ROC116/117 only)")
    now = _now()
    if now < NOT_BEFORE:
        raise ScoringRefused(f"not before {NOT_BEFORE:%Y-%m-%d}")
    av = availability_for_session(availability, roc)
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    order = n3_order_status(pdir, Path(n3_report) if n3_report else None)
    if not order["ok"]:
        raise ScoringRefused(f"N3 has not been scored (or recorded not_scorable): {order}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"session_roc{roc}.json"
    if f.exists():
        raise FileExistsError(f"{f} exists: the session of ROC{roc} is opened once")
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    res = {}
    for t in TESTS[roc].values():
        r = prereg3.assert_can_score(ID, WP, test_id=t, **kw)
        res[t] = {"sha256": r["sha256"], "post_registration_looks": r["post_registration_looks"],
                  "unverifiable_prior_looks": r["unverifiable_prior_looks"]}
    rec = {"id": ID, "survey_roc": roc, "opened": now.isoformat(), "availability": av,
           "order": order, "assert_can_score": res}
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return rec


# ----------------------------------------------------------------------------- product
PRODUCT_TABLES = ("township_bounds.csv", "township_status.csv", "county_intervals.csv",
                  "county_area_by_year.csv", "township_statements.csv", "hsr_piers.csv")


def product_status(product_dir) -> dict:
    """Metadata only (no table read): the product line, the use, the gate and the years."""
    m = json.loads((Path(product_dir) / "MANIFEST.json").read_text(encoding="utf-8"))
    par = m.get("parameters") or {}
    ok = (m.get("product") == PRODUCT and "prospective_scorer" in (m.get("allowed_uses") or [])
          and "prospective_scorer" not in (m.get("forbidden_uses") or [])
          and bool((m.get("gate") or {}).get("passed")) and par.get("registration_id") == ID)
    return {"ok": bool(ok), "product": m.get("product"), "version": m.get("version"),
            "survey_rocs": par.get("survey_rocs"), "gate": m.get("gate")}


def read_tables(product_dir, rehearsal: bool) -> dict:
    d = Path(product_dir)
    if rehearsal:
        return {n: pd.read_csv(d / n) for n in PRODUCT_TABLES if (d / n).is_file()}
    from common import guard
    prod = guard.open_product(d, "prospective_scorer")
    out = {}
    for n in PRODUCT_TABLES:
        if n in prod.outputs:
            out[n] = prod.read_table(n)
    return out


def check_product_years(tables: dict, roc: int) -> None:
    """A product row of a survey year <= 115 is an engineering failure (the builder drops
    them); rows of other future years are ignored here."""
    for n, df in tables.items():
        if "survey_year" in df.columns and len(df):
            yrs = pd.to_numeric(df["survey_year"], errors="coerce").dropna().astype(int)
            if (yrs <= EXCLUDED_ROC_MAX + 1911).any():
                raise EngineeringFailure(f"{n} holds a survey year <= ROC{EXCLUDED_ROC_MAX}")


# ----------------------------------------------------------------------------- scoring
def final_mapping(l1c: dict, l2c: dict, sl_role: str) -> dict:
    """The page consequences fixed in the registration, applied to the combined verdicts."""
    zr = l1c.get("ZR", {}).get("verdict", "undecided")
    sl = l1c.get("SL", {}).get("verdict", "undecided")
    out = {"C26": {"supported": "N, decided: 'the frozen coastal forecast met the registered "
                                "coastal-township bounds on WRA ROC116/117 (no S: the claim is "
                                "a narrowing)'",
                   "not_supported": "N, decided: 'coastal forecasts failed a registered "
                                    "prospective test (N3S-L L1)'",
                   "undecided": "N, 'undecided (N3S-L L1)'"}[zr],
           "C30_coastal": {"supported": "N with decided sub-label 'coastal part passed N3S-L "
                                        "L1'", "not_supported": "N with decided sub-label "
                           "'coastal part failed N3S-L L1'",
                           "undecided": "N, 'coastal part undecided (N3S-L L1)'"}[zr],
           "C05": {"supported": "returns as a narrowed S (prospective): the county area "
                                "sinking > 3 cm/yr for ROC116/117 only; no cut number, no "
                                "policy difference",
                   "not_supported": "W, decided (N3S-L L2 failed)",
                   "undecided": "W, 'undecided (N3S-L L2)'"}[l2c["verdict"]],
           "C25": "unchanged", "C27": "unchanged (L4 dropped)", "C06": "unchanged (L3 "
           "report-only)"}
    if sl_role == "eligible_ds":
        out["C26b"] = {"supported": "S (prospective, statistical layer, ROC116/117 only): "
                                    "'coastal sinking continued at its pre-2022 background'",
                       "not_supported": "W, decided: the statistical background layer failed "
                                        "N3S-L L1", "undecided": "D removed; 'undecided'"}[sl]
    else:
        out["C26b"] = f"report-only: SL's L1 verdict ({sl}) is reported beside ZR; no claim"
    return out


def run_score(a, prereg_dir=None, rows_dir=None) -> dict:
    roc = int(a.survey_roc)
    if roc not in SURVEYS:
        raise ScoringRefused(f"survey ROC{roc} is not registered (ROC116/117 only)")
    out = Path(a.out)
    if a.rehearsal:
        if not (Path(a.product) / SYNTHETIC_MARKER).is_file():
            raise ScoringRefused("--rehearsal needs a synthetic product (SYNTHETIC marker)")
        sl_role = getattr(a, "sl_role", None) or "report_only"
    else:
        if _now() < NOT_BEFORE:
            raise ScoringRefused(f"not before {NOT_BEFORE:%Y-%m-%d}")
        from harness import prereg3
        pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
        ss = session_status(pdir, roc)
        if not ss["ok"]:
            raise ScoringRefused(f"the score looks of ROC{roc} are not logged: run "
                                 "open-session first")
        if (Path(a.product) / SYNTHETIC_MARKER).exists():
            raise ScoringRefused("a synthetic product cannot be scored in real mode")
        ps = product_status(a.product)
        if not ps["ok"] or roc not in (ps["survey_rocs"] or []):
            raise ScoringRefused(f"product does not grant prospective_scorer for ROC{roc}: "
                                 f"{ps}")
        kw = {"prereg_dir": pdir} if rows_dir is None else {"prereg_dir": pdir,
                                                              "rows_dir": rows_dir}
        doc, _ = prereg3.load_prereg(ID, **kw)
        sl_role = doc["sl_page_role"]["role"]
    out.mkdir(parents=True, exist_ok=True)
    rep_path = out / f"report_roc{roc}.json"
    if rep_path.exists():
        raise FileExistsError(f"{rep_path} exists: ROC{roc} is scored once")
    fz = load_freeze(a.freeze_dir)
    run = a.run if getattr(a, "run", None) is not None else _forward(fz)
    check_zr_frozen(run, fz, roc)
    tables = read_tables(a.product, a.rehearsal)
    check_product_years(tables, roc)
    yr = score_year(roc, tables, fz, run)
    rep = {"id": ID, "scorer_version": SCORER_VERSION, "rehearsal": bool(a.rehearsal),
           "survey_roc": roc, "product": str(a.product), "freeze_summary_sha256":
           sha256(Path(a.freeze_dir) / "summary.json"), "year": yr,
           "test_ids": TESTS[roc], "sl_role": sl_role}
    if roc == 117:
        rep["combined"] = combine(yr, getattr(a, "stage1_report", None), sl_role,
                                  rehearsal=bool(a.rehearsal))
    rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json),
                        encoding="utf-8")
    return rep


def combine(year117: dict | None, stage1, sl_role: str, rehearsal: bool = False) -> dict:
    """ROC116 (its report, or a not_published record) and ROC117 (its scored year, or None
    when ROC117 was not published by its deadline) -> the claim verdicts and the page
    mapping. With neither year scored every verdict is 'undecided'."""
    years = {} if year117 is None else {117: year117}
    if stage1 is None:
        raise ScoringRefused("ROC117 combines with ROC116: give --stage1-report (its report or "
                             "the not_published record)")
    s1 = json.loads(Path(stage1).read_text(encoding="utf-8"))
    if s1.get("id") != ID or s1.get("survey_roc") != 116:
        raise ScoringRefused("--stage1-report does not name this registration's ROC116")
    if bool(s1.get("rehearsal")) and not rehearsal:
        raise ScoringRefused("a rehearsal ROC116 report cannot enter a real combination")
    if s1.get("verdict") != NOT_PUBLISHED:
        years[116] = s1["year"]
    l1 = l1_combine({k: v["L1"] for k, v in years.items()})
    flags = {f"{k}:{c}": v for k, y in years.items() for c, v in y["L2_county_years"].items()}
    l2 = l2_combine(flags)
    return {"L1": l1, "L2": l2, "years_scored": sorted(years),
            "not_published": sorted({116, 117} - set(years)),
            "stage1_sha256": sha256(stage1),
            "page_mapping_applied": final_mapping(l1, l2, sl_role)}


def not_published(roc: int, out, stage1=None, prereg_dir=None, rows_dir=None) -> dict:
    """The lead-written record of a survey year not published by its deadline. For ROC117
    it also writes the FINAL combination (ROC116's report, or its own not_published record,
    alone) with the registered page mapping, so the claims always reach a verdict."""
    if roc not in SURVEYS:
        raise ScoringRefused(f"survey ROC{roc} is not registered (ROC116/117 only)")
    out = Path(out)
    f = out / f"report_roc{roc}.json"
    if f.exists():
        raise FileExistsError(f"{f} exists")
    if _now().date() <= dt.date.fromisoformat(NO_LATER_THAN[roc]):
        raise ScoringRefused(f"ROC{roc}'s deadline {NO_LATER_THAN[roc]} has not passed")
    rec = {"id": ID, "survey_roc": roc, "verdict": NOT_PUBLISHED,
           "written": _now().isoformat(), "note": "lead-written: no ROC summary published "
           "by the deadline; both tests of the year are not evaluable"}
    if roc == 117:
        from harness import prereg3
        if stage1 is None:
            raise ScoringRefused("ROC117 not published: give --stage1-report (ROC116's report "
                                 "or its not_published record) so the final verdict is written")
        pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
        kw = {"prereg_dir": pdir} if rows_dir is None else {"prereg_dir": pdir,
                                                              "rows_dir": rows_dir}
        doc, _ = prereg3.load_prereg(ID, **kw)
        rec["sl_role"] = doc["sl_page_role"]["role"]
        rec["combined"] = combine(None, stage1, rec["sl_role"])
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    return rec


def _forward(fz: dict):
    from freeze_n3s_l import ForwardRun
    p = REPO / FORWARD
    if sha256(p) != MODEL_SHA:
        raise EngineeringFailure("forward npz sha changed")
    return ForwardRun.from_npz(p)


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("open-session")
    o.add_argument("--survey-roc", type=int, required=True)
    o.add_argument("--n3-report", default=None)
    o.add_argument("--availability", required=True)
    o.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--survey-roc", type=int, required=True)
    s.add_argument("--product", required=True)
    s.add_argument("--freeze-dir", required=True)
    s.add_argument("--stage1-report", default=None)
    s.add_argument("--rehearsal", action="store_true")
    s.add_argument("--sl-role", choices=SL_ROLES, default=None, help="rehearsal only")
    s.add_argument("--out", required=True)
    n = sub.add_parser("not-published")
    n.add_argument("--survey-roc", type=int, required=True)
    n.add_argument("--stage1-report", default=None, help="required for ROC117")
    n.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "open-session":
        print(json.dumps(open_session(a.survey_roc, a.n3_report, a.out, prereg_dir, rows_dir,
                                      availability=a.availability),
                         ensure_ascii=False, indent=1, default=str))
    elif a.cmd == "score":
        a.run = None
        rep = run_score(a, prereg_dir, rows_dir)
        print(json.dumps({"survey_roc": rep["survey_roc"], "written": True}))
    else:
        print(json.dumps(not_published(a.survey_roc, a.out, a.stage1_report, prereg_dir,
                                       rows_dir), ensure_ascii=False, default=_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
