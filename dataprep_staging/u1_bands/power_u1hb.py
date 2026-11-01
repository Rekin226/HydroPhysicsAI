"""U1.HB.power.v1: synthetic power of the coverage gate set E02c will register (rank 2).

What a single 12-month window at 84 Yunlin and 74 Changhua wells can say about the frozen
E02 head interval (results/twin/prospective_2026-11, combined sigma, central 90 %), before
E02c is written. Synthetic hits only: no observation, no residual and no product is opened,
so nothing here is a look. The declaration ``U1.HB.power.v1`` (prereg3 kind 'declaration',
written by ``prereg_u1hb_power.py``) fixes everything below BEFORE the registered run draws
its first replicate; :func:`design_spec` is the one source of the declared design, and
``run`` refuses unless the registered declaration carries exactly this design and the code
files still have the sha256 it bound.

Generator (per replicate; wells 0-83 Yunlin, 84-157 Changhua; months 0-11). A Gaussian
liability with crossed effects and a year-level shift, thresholded so that the marginal
hit probability is exactly the declared true coverage p::

    U[w, t] = sqrt(a_y) C + sqrt(1 - a_y) (sqrt(a_w) A[w] + sqrt(a_m) B[t]
                                            + sqrt(1 - a_w - a_m) E[w, t])
    hit[w, t] = U[w, t] <= Phi^-1(p)

C (one per replicate: the window's year shock, common to every well-month), B (one per month,
common to both counties), A (one per well) and E are independent N(0, 1), drawn in that order
(C, B, A, E) as float32 from ``np.random.default_rng(SeedSequence(SEED, spawn_key=(cell,)))``
in chunks of ``CHUNK`` replicates. a_w, a_m and a_y are liability-scale variance shares (the
realised binary-scale ICCs are reported per cell). The heavy-tailed well arm replaces A by a
two-point well type: each well is independently 'bad' with probability
pi = (1 - p) / (1 - p_bad), p_bad = 0.10; a good well always hits, a bad well hits when
U[w, t] <= Phi^-1(p_bad) with a_w = 0 (month and year shocks kept). Its marginal is again p.

Gates (exactly E02c's T-cov set, rank 4):
- G0: the manifest's pooled point check, coverage of all 1896 well-months in [0.85, 0.95]
  (``twin.validation.evaluate`` interval_coverage; bounds = protocol.json coverage_bounds).
- G1 per county: ``harness.coverage3`` rule gate-3 (method upper90), clusters = wells, target
  0.90, level 0.95, point window [0.85, 0.95] when n_eff_lower >= 30 (window conf 0.95).
- G1b per county: the same rule on month clusters (k = 12), REPORT-ONLY, 'uncalibrated
  (k = 12 < 20)'.
- County outcome: 'coverage supported' = G0 AND that county's G1; anything else is a FAIL.

Wording rules (two failure modes, declared): see :data:`WORDING` and :func:`wording`. The
FAIL rule never removes a failure: 'fail_shock' is 'fail_plain' (the word 'failed' and the
coverage number) followed by a context suffix. The PASS rule only weakens a pass.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "dataprep_staging") not in sys.path:
    sys.path.insert(0, str(REPO / "dataprep_staging"))

from harness import coverage3  # noqa: E402
from harness.coverage import icc_anova_batch  # noqa: E402

DECL_ID = "U1.HB.power.v1"
COUNTIES = (("yunlin", 84), ("changhua", 74))
N_WELLS = sum(n for _, n in COUNTIES)
N_MONTHS = 12
ICC_WELL = (0.0, 0.1, 0.3, 0.5, 0.7, "heavy")
ICC_MONTH = (0.0, 0.05, 0.15, 0.3)
ICC_YEAR = (0.0, 0.1, 0.2)
COVERAGE = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)
HEAVY_P_BAD = 0.10
REPLICATES = 10_000
CHUNK = 2_000
SEED = 20261030
G0_BOUNDS = (0.85, 0.95)
GATE = {"method": "upper90", "target": 0.90, "level": 0.95, "window": (0.85, 0.95),
        "n_eff_window_min": 30.0, "conf": 0.95}
FALSE_FAIL_MAX = 0.10
POWER_MIN = 0.80
POWER_AT = 0.75
POWER_ALSO_AT = 0.80
POWER_ICC_MONTH_MAX = 0.15
G1B_LABEL = "uncalibrated (k = 12 < 20)"
# smoke: a tiny grid on a DIFFERENT seed, engineering only (its numbers set nothing)
SMOKE_GRID = {"icc_well": (0.0, "heavy"), "icc_month": (0.15,), "icc_year": (0.1,),
              "coverage": (0.75, 0.90)}
SMOKE_REPLICATES = 400
SMOKE_SEED = 1
SELF_CHECK_SEED = 20261031
RESULT_DIR = "results/dataprep/reports/U1.HB.power.v1"
ENGINEERING_FAILURE_NOTE = (
    "ENGINEERING FAILURE: the self-check or the consistency check failed; the result is "
    "written flagged 'engineering_failure' and gates nothing. The self-check is "
    "deterministic and 'run' refuses any change to the bound code, so a fix that touches "
    "power_u1hb.py, harness/coverage*.py or bands.py needs a new declaration id (not a "
    "re-run of this one). Only a cause outside the bound code (e.g. a missing or changed "
    "protocol.json) can be fixed and re-run under this id; the failed result is then "
    "archived and must be reported.")
CODE_FILES = ("dataprep_staging/u1_bands/power_u1hb.py",
              "dataprep_staging/harness/coverage3.py",
              "dataprep_staging/harness/coverage2.py",
              "dataprep_staging/harness/coverage.py",
              "hydrophysics/twin/bands.py")
PROTOCOL = "results/twin/prospective_2026-11/protocol.json"

_FAIL_PLAIN = {"en": "failed its registered gate (coverage {x})",
               "zh": "未通過其已登錄的檢驗門檻（涵蓋率 {x}）"}
FAIL_SHOCK_SUFFIX = {"en": "; outside the registered range; may reflect one year's shock",
                     "zh": "；超出已登錄的範圍；可能反映單一年份的衝擊"}
WORDING = {
    "pass_plain": {"en": "passed its registered gate",
                   "zh": "通過其已登錄的檢驗門檻"},
    "pass_weak": {"en": "no gross under-coverage detected",
                  "zh": "未偵測到嚴重的涵蓋不足"},
    "fail_plain": dict(_FAIL_PLAIN),
    # a suffix to fail_plain, never a replacement: 'failed' and the coverage number stay
    "fail_shock": {k: _FAIL_PLAIN[k] + FAIL_SHOCK_SUFFIX[k] for k in ("en", "zh")},
    "fail_shock_suffix": dict(FAIL_SHOCK_SUFFIX),
    "never": ["calibrated", "校準"],
}
WORDING_RULES = {
    "fail": f"per county: FF = the largest P(FAIL) over every grid cell with true coverage "
            f"0.90 (all ICC_well incl. heavy, ICC_month, ICC_year); if FF > {FALSE_FAIL_MAX} "
            f"a FAIL of that county reads 'fail_shock' (= 'fail_plain' + 'fail_shock_suffix': "
            f"the word 'failed' and the coverage number are kept), otherwise 'fail_plain'",
    "pass": f"per county: PW = the smallest P(FAIL) over every grid cell with true coverage "
            f"{POWER_AT} and ICC_month <= {POWER_ICC_MONTH_MAX} (all ICC_well incl. heavy, "
            f"all ICC_year); if PW < {POWER_MIN} a PASS of that county reads 'pass_weak', "
            f"otherwise 'pass_plain'; no outcome ever uses the words in 'never'",
    "independent": "the two rules are evaluated separately; both can apply",
    "report_only": f"power at {POWER_ALSO_AT} (same cells), the detectable set (true "
                   f"coverages <= 0.85 whose smallest power over the same cells is >= "
                   f"{POWER_MIN}), the false-fail maximum and the power minimum of each "
                   f"ICC_year slice (shows how much the year axis drives the two rules), G1b "
                   f"power, n_eff, n_eff_lower and the window share are reported and gate no "
                   f"wording",
    "direction": "the PASS rule only weakens a pass ('pass_weak'); the FAIL rule only adds "
                 "context to a FAIL ('fail_shock' keeps 'failed' and the coverage number); "
                 "neither turns a FAIL into a pass, a pass into 'calibrated', or un-withdraws "
                 "the interval wording after a FAIL",
    "comparison": "point estimates over 10,000 replicates are compared with the thresholds "
                  "as they are; the Monte Carlo standard error is reported beside each",
}


def _jsonable(x):
    return json.loads(json.dumps(x))


def design_spec() -> dict:
    """The declared design (the declaration embeds this dict; ``run`` compares them)."""
    return _jsonable({
        "counties": [{"name": c, "wells": n} for c, n in COUNTIES], "months": N_MONTHS,
        "grid": {"icc_well": list(ICC_WELL), "icc_month": list(ICC_MONTH),
                 "icc_year": list(ICC_YEAR), "coverage": list(COVERAGE)},
        "cell_order": "icc_well (outer), icc_month, icc_year, coverage (inner); cell index "
                      "0..n_cells-1 in that order",
        "heavy_p_bad": HEAVY_P_BAD, "replicates_per_cell": REPLICATES, "chunk": CHUNK,
        "seed": SEED, "rng": "np.random.default_rng(np.random.SeedSequence(seed, "
                             "spawn_key=(cell_index,))); per chunk draws C (D,1,1), B (D,1,12),"
                             " then A (D,158,1) or the heavy well type rng.random((D,158,1)),"
                             " then E (D,158,12); standard normals float32",
        "liability_scale": "a_w, a_m, a_y are liability-scale variance shares; a_w and a_m "
                           "are shares of the non-year variance (1 - a_y)",
        "g0": {"bounds": list(G0_BOUNDS), "rows": "all well-months of both counties",
               "source": PROTOCOL + " coverage_bounds (twin.validation.evaluate)"},
        "g1": dict(GATE, clusters="well", per="county", module="harness.coverage3",
                   rule=coverage3.RULE),
        "g1b": dict(GATE, clusters="month", per="county", module="harness.coverage3",
                    rule=coverage3.RULE, role="report-only", label=G1B_LABEL),
        "outcome": "county 'coverage supported' = G0 AND G1(county); FAIL otherwise",
        "thresholds": {"false_fail_max": FALSE_FAIL_MAX, "power_min": POWER_MIN,
                       "power_at": POWER_AT, "power_also_at": POWER_ALSO_AT,
                       "power_icc_month_max": POWER_ICC_MONTH_MAX},
        "wording": WORDING, "wording_rules": WORDING_RULES,
        "self_check_seed": SELF_CHECK_SEED,
        "smoke": {"grid": SMOKE_GRID, "replicates": SMOKE_REPLICATES, "seed": SMOKE_SEED},
    })


def cells(grid: dict | None = None) -> list[dict]:
    g = grid or {"icc_well": ICC_WELL, "icc_month": ICC_MONTH, "icc_year": ICC_YEAR,
                 "coverage": COVERAGE}
    out = []
    for aw in g["icc_well"]:
        for am in g["icc_month"]:
            for ay in g["icc_year"]:
                for p in g["coverage"]:
                    out.append({"cell": len(out), "icc_well": aw, "icc_month": float(am),
                                "icc_year": float(ay), "coverage": float(p)})
    return out


# ----------------------------------------------------------------------------- generator
def draw_hits(rng: np.random.Generator, n: int, p: float, icc_well, icc_month: float,
              icc_year: float) -> np.ndarray:
    """(n, 158, 12) boolean hits from the declared generator (see the module docstring)."""
    if not 0 < p < 1:
        raise ValueError(f"coverage must lie in (0, 1), not {p}")
    heavy = icc_well == "heavy"
    aw = 0.0 if heavy else float(icc_well)
    am, ay = float(icc_month), float(icc_year)
    if aw < 0 or am < 0 or not 0 <= ay < 1 or aw + am > 1 + 1e-12:
        raise ValueError(f"invalid shares a_w={aw}, a_m={am}, a_y={ay}")
    f32 = np.float32
    c = rng.standard_normal((n, 1, 1), dtype=f32)
    b = rng.standard_normal((n, 1, N_MONTHS), dtype=f32)
    if heavy:
        pi = (1.0 - p) / (1.0 - HEAVY_P_BAD)
        if not 0 <= pi <= 1:
            raise ValueError(f"heavy arm needs p >= {HEAVY_P_BAD}")
        bad = rng.random((n, N_WELLS, 1)) < pi
        thr = np.where(bad, f32(ndtri(HEAVY_P_BAD)), f32(np.inf))
        a = None
    else:
        a = rng.standard_normal((n, N_WELLS, 1), dtype=f32)
        thr = f32(ndtri(p))
    e = rng.standard_normal((n, N_WELLS, N_MONTHS), dtype=f32)
    s_res = math.sqrt(max(0.0, 1.0 - aw - am))
    inner = f32(math.sqrt(am)) * b + f32(s_res) * e
    if a is not None and aw > 0:
        inner += f32(math.sqrt(aw)) * a
    u = f32(math.sqrt(ay)) * c + f32(math.sqrt(1.0 - ay)) * inner
    return u <= thr


def county_slices() -> list[tuple[str, slice]]:
    out, i = [], 0
    for name, n in COUNTIES:
        out.append((name, slice(i, i + n)))
        i += n
    return out


def _gate(Y: np.ndarray, m: np.ndarray) -> dict:
    return coverage3.gate_from_totals_batch(
        Y, m, GATE["method"], GATE["target"], GATE["level"], GATE["window"],
        GATE["n_eff_window_min"], GATE["conf"])


def score_hits(hits: np.ndarray) -> dict:
    """Gate arrays (n,) of a batch of hit cubes (n, 158, 12)."""
    p_all = hits.mean(axis=(1, 2), dtype=np.float64)
    g0 = (p_all >= G0_BOUNDS[0]) & (p_all <= G0_BOUNDS[1])
    out = {"p_all": p_all, "g0": g0}
    for name, sl in county_slices():
        h = hits[:, sl, :]
        nw = h.shape[1]
        yw = h.sum(axis=2, dtype=np.float64)
        mw = np.full(nw, float(N_MONTHS))
        ym = h.sum(axis=1, dtype=np.float64)
        mm = np.full(N_MONTHS, float(nw))
        g1, g1b = _gate(yw, mw), _gate(ym, mm)
        out[name] = {"p": g1["p"], "g1": g1["pass"], "g1b": g1b["pass"],
                     "supported": g0 & g1["pass"], "n_eff": g1["n_eff"],
                     "n_eff_lower": g1["n_eff_lower"], "window_applies": g1["window_applies"],
                     "in_ci": g1["in_ci"], "lo": g1["lo"], "hi": g1["hi"],
                     "g1b_n_eff": g1b["n_eff"], "rho_well": icc_anova_batch(yw, mw),
                     "rho_month": icc_anova_batch(ym, mm)}
    return out


# ----------------------------------------------------------------------------- one cell
def _q(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "q05": float(np.quantile(x, 0.05)),
            "q50": float(np.quantile(x, 0.50)), "q95": float(np.quantile(x, 0.95))}


def _prob(x: np.ndarray) -> dict:
    p = float(np.mean(x))
    return {"p": p, "mc_se": math.sqrt(max(p * (1 - p), 0.0) / x.size)}


def run_cell(cell: dict, replicates: int, seed: int, chunk: int = CHUNK) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(cell["cell"],)))
    parts = []
    done = 0
    while done < replicates:
        n = min(chunk, replicates - done)
        hits = draw_hits(rng, n, cell["coverage"], cell["icc_well"], cell["icc_month"],
                         cell["icc_year"])
        parts.append(score_hits(hits))
        done += n
    cat = {"p_all": np.concatenate([s["p_all"] for s in parts]),
           "g0": np.concatenate([s["g0"] for s in parts])}
    out = dict(cell)
    out.update({"replicates": replicates, "mean_coverage_hat": float(cat["p_all"].mean()),
                "sd_coverage_hat": float(cat["p_all"].std(ddof=1)) if replicates > 1 else 0.0,
                "P_g0": _prob(cat["g0"])})
    for name, _ in COUNTIES:
        c = {k: np.concatenate([s[name][k] for s in parts]) for k in parts[0][name]}
        out[name] = {
            "P_g1": _prob(c["g1"]), "P_g1b": _prob(c["g1b"]),
            "P_supported": _prob(c["supported"]), "P_fail": _prob(~c["supported"]),
            "P_g1b_fail": _prob(~c["g1b"]), "P_contains_target": _prob(c["in_ci"]),
            "window_applies": float(c["window_applies"].mean()),
            "n_eff": _q(c["n_eff"]), "n_eff_lower": _q(c["n_eff_lower"]),
            "g1b_n_eff": _q(c["g1b_n_eff"]),
            "mean_coverage_hat": float(c["p"].mean()),
            "rho_anova_well_mean": float(c["rho_well"].mean()),
            "rho_anova_month_mean": float(c["rho_month"].mean())}
    return out


# ----------------------------------------------------------------------------- wording
def wording(cell_results: list[dict]) -> dict:
    """Apply the declared wording rules (WORDING_RULES) to the cell results."""
    out = {"rules": WORDING_RULES}
    for name, _ in COUNTIES:
        ff = [c for c in cell_results if abs(c["coverage"] - 0.90) < 1e-9]
        pw = [c for c in cell_results if abs(c["coverage"] - POWER_AT) < 1e-9
              and c["icc_month"] <= POWER_ICC_MONTH_MAX + 1e-12]
        pw2 = [c for c in cell_results if abs(c["coverage"] - POWER_ALSO_AT) < 1e-9
               and c["icc_month"] <= POWER_ICC_MONTH_MAX + 1e-12]
        rec: dict = {}
        if ff:
            worst = max(ff, key=lambda c: c[name]["P_fail"]["p"])
            v = worst[name]["P_fail"]["p"]
            rec["false_fail_max"] = {"value": v, "mc_se": worst[name]["P_fail"]["mc_se"],
                                     "cell": _cell_id(worst)}
            rec["false_fail_cells_above"] = [_cell_id(c) for c in ff
                                             if c[name]["P_fail"]["p"] > FALSE_FAIL_MAX]
            rec["fail_wording"] = "fail_shock" if v > FALSE_FAIL_MAX else "fail_plain"
        else:
            rec["fail_wording"] = "not determined (no cell at true coverage 0.90)"
        if pw:
            weakest = min(pw, key=lambda c: c[name]["P_fail"]["p"])
            v = weakest[name]["P_fail"]["p"]
            rec["power_min_at_0.75"] = {"value": v, "mc_se": weakest[name]["P_fail"]["mc_se"],
                                        "cell": _cell_id(weakest)}
            rec["pass_wording"] = "pass_weak" if v < POWER_MIN else "pass_plain"
            g1b = [c[name]["P_g1b_fail"]["p"] for c in pw]
            rec["g1b_power_at_0.75"] = {"min": min(g1b), "max": max(g1b), "label": G1B_LABEL,
                                        "reading": "near-uninformative; kept as a report-only "
                                                   "guard" if max(g1b) < 0.5 else
                                                   "report-only"}
        else:
            rec["pass_wording"] = f"not determined (no cell at true coverage {POWER_AT})"
        if pw2:
            weakest = min(pw2, key=lambda c: c[name]["P_fail"]["p"])
            rec["power_min_at_0.80"] = {"value": weakest[name]["P_fail"]["p"],
                                        "cell": _cell_id(weakest)}
        det = []
        for p in sorted({c["coverage"] for c in cell_results if c["coverage"] <= 0.85 + 1e-9}):
            sub = [c for c in cell_results if abs(c["coverage"] - p) < 1e-9
                   and c["icc_month"] <= POWER_ICC_MONTH_MAX + 1e-12]
            if sub and min(c[name]["P_fail"]["p"] for c in sub) >= POWER_MIN:
                det.append(p)
        rec["detectable_coverages_report_only"] = det
        rec["by_icc_year_report_only"] = _by_icc_year(cell_results, name)
        for k in ("pass_wording", "fail_wording"):
            if rec[k] in WORDING:
                rec[k + "_text"] = {"en": WORDING[rec[k]]["en"], "zh": WORDING[rec[k]]["zh"]}
        out[name] = rec
    return out


def _by_icc_year(cell_results: list[dict], name: str) -> dict:
    """Report-only: the two rule statistics within each ICC_year slice."""
    out = {}
    for ay in sorted({c["icc_year"] for c in cell_results}):
        sl = [c for c in cell_results if abs(c["icc_year"] - ay) < 1e-12]
        ff = [c[name]["P_fail"]["p"] for c in sl if abs(c["coverage"] - 0.90) < 1e-9]
        pw = [c[name]["P_fail"]["p"] for c in sl if abs(c["coverage"] - POWER_AT) < 1e-9
              and c["icc_month"] <= POWER_ICC_MONTH_MAX + 1e-12]
        out[f"{ay:g}"] = {"false_fail_max": max(ff) if ff else None,
                          "power_min_at_0.75": min(pw) if pw else None}
    return out


def _cell_id(c: dict) -> dict:
    return {k: c[k] for k in ("cell", "icc_well", "icc_month", "icc_year", "coverage")}


# ----------------------------------------------------------------------------- self-check
def _rel_close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def self_check(n_parity: int = 40, n_bias: int = 4000) -> dict:
    """G1/G1b parity with harness.coverage3.coverage_gate (rel 1e-12) on generated hits, and
    the coverage estimator's bias at ICC = 0 and under heavy clustering (within 4 MC se)."""
    rng = np.random.default_rng(np.random.SeedSequence(SELF_CHECK_SEED))
    out: dict = {"seed": SELF_CHECK_SEED, "parity": [], "bias": []}
    ok = True
    for aw, am, ay, p in ((0.5, 0.15, 0.1, 0.85), ("heavy", 0.3, 0.0, 0.9), (0.0, 0.0, 0.0, 0.9)):
        hits = draw_hits(rng, n_parity, p, aw, am, ay)
        s = score_hits(hits)
        worst = 0.0
        mism = 0
        for name, sl in county_slices():
            nw = sl.stop - sl.start
            wells = np.repeat(np.arange(nw), N_MONTHS)
            months = np.tile(np.arange(N_MONTHS), nw)
            for d in range(n_parity):
                flat = hits[d, sl, :].reshape(-1).astype(float)
                for clusters, key in ((wells, "g1"), (months, "g1b")):
                    g = coverage3.coverage_gate(flat, clusters, GATE["method"], GATE["target"],
                                                GATE["level"], GATE["window"],
                                                GATE["n_eff_window_min"], GATE["conf"])
                    if key == "g1":
                        pairs = ((g["coverage"], s[name]["p"][d]),
                                 (g["n_eff"], s[name]["n_eff"][d]),
                                 (g["n_eff_lower"], s[name]["n_eff_lower"][d]),
                                 (g["lo"], s[name]["lo"][d]), (g["hi"], s[name]["hi"][d]))
                        same = (g["pass"] == bool(s[name]["g1"][d])
                                and g["window_applies"] == bool(s[name]["window_applies"][d]))
                    else:
                        pairs = ((g["n_eff"], s[name]["g1b_n_eff"][d]),)
                        same = g["pass"] == bool(s[name]["g1b"][d])
                    for a, b in pairs:
                        worst = max(worst, abs(a - b) / max(1.0, abs(a), abs(b)))
                        if not _rel_close(a, b):
                            mism += 1
                    mism += 0 if same else 1
        rec = {"cell": [aw, am, ay, p], "replicates": n_parity, "max_rel_diff": worst,
               "mismatches": mism, "ok": mism == 0}
        ok &= rec["ok"]
        out["parity"].append(rec)
    for aw, am, ay, p in ((0.0, 0.0, 0.0, 0.90), (0.7, 0.3, 0.2, 0.90), ("heavy", 0.3, 0.2, 0.80)):
        vals = np.concatenate([score_hits(draw_hits(rng, 1000, p, aw, am, ay))["p_all"]
                               for _ in range(n_bias // 1000)])
        if aw == 0.0 and am == 0.0 and ay == 0.0:
            se = math.sqrt(p * (1 - p) / (N_WELLS * N_MONTHS * vals.size))
        else:
            se = float(vals.std(ddof=1) / math.sqrt(vals.size))
        rec = {"cell": [aw, am, ay, p], "replicates": int(vals.size),
               "mean": float(vals.mean()), "mc_se": se,
               "ok": bool(abs(vals.mean() - p) <= 4 * se)}
        ok &= rec["ok"]
        out["bias"].append(rec)
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- binding
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_sha256() -> dict:
    return {f: sha256_file(REPO / f) for f in CODE_FILES}


def bands_constants() -> dict:
    """Read hydrophysics/twin/bands.py by path (read-only, no package import)."""
    name = "_u1_bands_ro"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, REPO / "hydrophysics/twin/bands.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod            # dataclasses resolve the module by name
        spec.loader.exec_module(mod)
    return {"TARGET": mod.TARGET, "GATE_LEVEL": mod.GATE_LEVEL,
            "POINT_WINDOW": list(mod.POINT_WINDOW), "min_units_signed_0.90": mod.min_units(0.90)}


def consistency_problems() -> list[str]:
    """The constants this study declares agree with the code and the registered protocol."""
    out = []
    b = bands_constants()
    if b["TARGET"] != GATE["target"] or b["GATE_LEVEL"] != GATE["level"]:
        out.append(f"bands.py target/level {b['TARGET']}/{b['GATE_LEVEL']} differ from GATE")
    if tuple(b["POINT_WINDOW"]) != tuple(GATE["window"]) or tuple(GATE["window"]) != G0_BOUNDS:
        out.append("bands.POINT_WINDOW, GATE window and G0 bounds differ")
    if GATE["method"] != coverage3.DEFAULT_METHOD or coverage3.RULE != "gate-3":
        out.append(f"harness.coverage3 default {coverage3.DEFAULT_METHOD}/{coverage3.RULE}")
    if GATE["n_eff_window_min"] != coverage3.N_EFF_WINDOW_MIN:
        out.append("coverage3.N_EFF_WINDOW_MIN differs from GATE")
    proto = REPO / PROTOCOL
    if proto.exists():
        cb = json.loads(proto.read_text(encoding="utf-8")).get("coverage_bounds")
        if tuple(cb or ()) != G0_BOUNDS:
            out.append(f"{PROTOCOL} coverage_bounds {cb} != G0 {G0_BOUNDS}")
    else:
        out.append(f"{PROTOCOL} missing")
    return out


def registered_declaration(prereg_dir: Path | None = None) -> tuple[dict, str]:
    """(declaration, sha256) of the registered U1.HB.power.v1; raises unless it is in the
    registry INDEX with the same file sha, carries exactly :func:`design_spec`, and binds the
    current sha256 of every code file."""
    from common.paths import PREREG_DIR
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir is not None else PREREG_DIR
    rows = [r for r in prereg3.read_chained(d / "INDEX.jsonl") if r["id"] == DECL_ID]
    if not rows:
        raise SystemExit(f"REFUSED: {DECL_ID} is not registered in {d}; register it first "
                         "(prereg_u1hb_power.py --write)")
    path = d / f"{DECL_ID}.json"
    sha = sha256_file(path)
    if sha != rows[-1]["sha256"]:
        raise SystemExit(f"REFUSED: {path} sha256 {sha} != INDEX {rows[-1]['sha256']}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("kind") != "declaration" or doc.get("design") != design_spec():
        raise SystemExit("REFUSED: the registered design differs from power_u1hb.design_spec()")
    now = code_sha256()
    if doc.get("code_sha256") != now:
        bad = sorted(k for k in now if (doc.get("code_sha256") or {}).get(k) != now[k])
        raise SystemExit(f"REFUSED: code changed since registration: {bad}")
    return doc, sha


# ----------------------------------------------------------------------------- run / report
def run(grid: dict | None, replicates: int, seed: int, *, smoke: bool,
        declaration: dict | None = None, log=print) -> dict:
    t0 = time.time()
    probs = consistency_problems()
    chk = self_check()
    log(f"self-check ok={chk['ok']}; consistency problems: {probs or 'none'}")
    res: dict = {
        "id": DECL_ID, "kind": "synthetic_power", "smoke": smoke,
        "label": ("SMOKE (engineering only; tiny grid, different seed; sets nothing)" if smoke
                  else "registered synthetic power study; no observation read; it restricts "
                       "E02c's wording only"),
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "declaration": declaration, "design": design_spec(), "code_sha256": code_sha256(),
        "grid_run": _jsonable(grid or design_spec()["grid"]), "replicates_per_cell": replicates,
        "seed": seed, "self_check": chk, "consistency_problems": probs,
        "engineering_failure": bool(probs or not chk["ok"])}
    if res["engineering_failure"]:
        res["cells"], res["wording"] = [], {}
        return res
    out = []
    for c in cells(grid):
        out.append(run_cell(c, replicates, seed))
        if len(out) % 24 == 0 or len(out) == len(cells(grid)):
            log(f"  {len(out)}/{len(cells(grid))} cells, {time.time() - t0:.0f} s")
    res["cells"] = out
    res["wording"] = wording(out)
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def report_md(res: dict) -> str:
    lines = [f"# {res['id']} {'SMOKE ' if res['smoke'] else ''}result", "",
             f"- label: {res['label']}", f"- created: {res['created']}",
             f"- replicates per cell: {res['replicates_per_cell']}, seed {res['seed']}, "
             f"cells {len(res['cells'])}",
             f"- self-check ok: {res['self_check']['ok']}; engineering failure: "
             f"{res['engineering_failure']}"]
    if res.get("declaration"):
        lines.append(f"- declaration: {res['declaration']['id']} sha256 "
                     f"{res['declaration']['sha256']}")
    lines += ["", "## Wording (declared rules)", ""]
    for name, _ in COUNTIES:
        w = res["wording"].get(name, {})
        lines.append(f"- {name}: PASS -> {w.get('pass_wording')}, FAIL -> "
                     f"{w.get('fail_wording')}; false-fail max "
                     f"{_fmt((w.get('false_fail_max') or {}).get('value'))}, power min at "
                     f"{POWER_AT} {_fmt((w.get('power_min_at_0.75') or {}).get('value'))}, "
                     f"at {POWER_ALSO_AT} {_fmt((w.get('power_min_at_0.80') or {}).get('value'))};"
                     f" detectable (report-only) {w.get('detectable_coverages_report_only')}")
        for ay, v in (w.get("by_icc_year_report_only") or {}).items():
            lines.append(f"  - ICC_year {ay} (report-only): false-fail max "
                         f"{_fmt(v['false_fail_max'])}, power min at {POWER_AT} "
                         f"{_fmt(v['power_min_at_0.75'])}")
    lines += ["", "## Cells", "",
              "| cell | ICC_well | ICC_month | ICC_year | p | P(G0) | YL P(fail) | CH P(fail) | "
              "YL n_eff_lower q50 | YL window | YL rho_well | YL G1b fail |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        y, h = c["yunlin"], c["changhua"]
        lines.append(f"| {c['cell']} | {c['icc_well']} | {c['icc_month']} | {c['icc_year']} | "
                     f"{c['coverage']} | {c['P_g0']['p']:.3f} | {y['P_fail']['p']:.3f} | "
                     f"{h['P_fail']['p']:.3f} | {y['n_eff_lower']['q50']:.1f} | "
                     f"{y['window_applies']:.2f} | {y['rho_anova_well_mean']:.3f} | "
                     f"{y['P_g1b_fail']['p']:.3f} |")
    return "\n".join(lines) + "\n"


def write_result(res: dict, out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "smoke_result" if res["smoke"] else "result"
    p = out_dir / f"{name}.json"
    data = (json.dumps(res, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode()
    p.write_bytes(data)
    (out_dir / f"{name}.md").write_text(report_md(res), encoding="utf-8")
    return p, hashlib.sha256(data).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="tiny grid on the smoke seed (engineering only)")
    s.add_argument("--out", required=True, type=Path)
    r = sub.add_parser("run", help="the registered grid (refuses unless registered)")
    r.add_argument("--out", type=Path, default=REPO / RESULT_DIR)
    r.add_argument("--prereg-dir", type=Path, default=None)
    rp = sub.add_parser("report", help="re-render the markdown of a result.json")
    rp.add_argument("--result", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        res = json.loads(a.result.read_text(encoding="utf-8"))
        md = a.result.with_suffix(".md")
        md.write_text(report_md(res), encoding="utf-8")
        print(f"wrote {md}; result sha256 {sha256_file(a.result)}")
        return 0
    if a.cmd == "smoke":
        out = a.out.resolve()
        if out == (REPO / RESULT_DIR).resolve() or (REPO / RESULT_DIR).resolve() in out.parents:
            print("REFUSED: a smoke run never writes into the registered result directory")
            return 1
        res = run(SMOKE_GRID, SMOKE_REPLICATES, SMOKE_SEED, smoke=True)
    else:
        doc, sha = registered_declaration(a.prereg_dir)
        out = a.out.resolve()
        prev = out / "result.json"
        if prev.exists():
            old = json.loads(prev.read_text(encoding="utf-8"))
            if not old.get("engineering_failure"):
                print(f"REFUSED: {prev} exists and is not an engineering failure (run once)")
                return 1
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            prev.rename(out / f"result.engineering_failure_{stamp}.json")
            print(f"archived the earlier engineering failure as result.engineering_failure_"
                  f"{stamp}.json (report it)")
        res = run(None, REPLICATES, SEED, smoke=False,
                  declaration={"id": DECL_ID, "sha256": sha, "created": doc.get("created")})
    path, h = write_result(res, out)
    print(f"wrote {path} sha256 {h}")
    if res["engineering_failure"]:
        print(ENGINEERING_FAILURE_NOTE)
        return 1
    for name, _ in COUNTIES:
        w = res["wording"][name]
        print(f"{name}: PASS -> {w['pass_wording']}, FAIL -> {w['fail_wording']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
