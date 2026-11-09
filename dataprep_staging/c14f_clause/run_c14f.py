"""C14f.coverage_clause.v1 (round 11, user ruling R11.7; post-failure, exploratory; model-only,
CPU): how often does N3F's COVERAGE CLAUSE ALONE pass in FV.ext.v1's own synthetic worlds?

STATUS: design and code (code 51d7908 plus review fixes; user rulings R11.7a-d: run once, on
FV.ext.v1's worlds, FV.ext.v1's null decides, no new look). Not registered; nothing has been
scored. ``run`` refuses until the declaration (prereg_c14f.py) is in the registry; register
and run from the same build checkout (see prereg_c14f.py).

Why. C14f (claims ledger) is "the frozen forecast's per-well 90 % head interval ... at N3F's 66
never-calibrated external wells, 2026-11..2027-10", decided by the coverage clause of N3F's
registered rule (N3F.ext_level.prospective_2026-11.v1, test N3FP.ext_level_2026-11:
"passed: RMSE ratio <= 1.0 AND coverage in [0.85, 0.95]"; not_informative: "fewer than 30
wells with >= 6 window months"). Its only decidability reading, FV.ext.v1 (declaration sha256
50108315, report sha256 b29f6cfc), reads N3F's WHOLE rule (P 0.592, null false pass 0.0); by
user ruling R10.5 (R9.4's thresholds) C14f became D with decidability unknown. The clause's own
P and null pass rate were never computed: FV.ext.v1's report keeps per-truth pass rates of the
whole rule and median coverage, no per-world clause outcome. R11.7 allows one synthetic
reading of the clause alone, labelled post-failure exploratory, code committed before it runs,
run once, no fresh units; C14f's grade moves only if a later user ruling accepts it.

Worlds: EXACTLY FV.ext.v1's. The run calls FV.ext.v1's registered scorer
``ext_fv.score_ext`` unchanged (its six files pinned by sha256 to FV.ext.v1's frozen_code),
with FV.ext.v1's seed, streams, K and plan (fixed in that code), and records every call of its
rule function ``ext_fv.rule`` (a pass-through wrapper: it returns the frozen rule's own output
and stores the pooled coverage, RMSE ratio and pass of that call in memory). World k of truth T
at external well w is then FV.ext.v1's::

    obs(w, t) = ZR_ext(w, t) + D_T(w, t) + R_T,k(w, t) + e_k(w, t)      t in 2026-11..2027-10
    null:  obs0(w, t) = baseline_reg(w, t) + e_k(w, t)                    (stream 22, K_NULL)

Gate I (identity, before any clause statistic leaves memory): the frozen scorer's report on
this run must equal FV.ext.v1's stored report.json (sha256 b29f6cfc) on every key except
``created`` and ``frozen_code`` (floats within 1e-9). Equality shows the worlds are FV.ext.v1's.

The clause (N3F's, as ``osse_fv.g0_e02`` and ``twin.validation.evaluate`` compute it):
pooled central 90 % coverage over every window row in [0.85, 0.95] (closed), with >= 30 wells
of >= 6 months (always met: 66 x 12). Coverage does not depend on the baseline, so the clause
outcome of a truth world is read from its primary call (own pre-origin climatology) and checked
equal on its report-only call (registered baseline).

  P  = equal-weight mean over the four truths (FV v2's R0, Q0, R1c, X5R1) of the share of their
       250 worlds in which the clause passes;
  P0 = the share of FV.ext.v1's 500 null worlds in which the clause passes.

Outcome (R9.4's rule as R10.5 applied it; point estimates decide, compared as EXACT fractions
of integer pass counts, never float means; applied once):
'measured' if P >= 0.8 and P0 <= 0.2; otherwise 'weak' if P < 0.5 or P0 > 0.5; otherwise
'unchanged'. A failed Monte Carlo check (1.645 SE <= 0.07; non-binding at this K) or a
non-finite estimate gives 'not_evaluable', mapped as 'unchanged'. The outcome is a PROPOSAL for
C14f's D qualifier: it moves nothing until a later user ruling accepts it (R11.7).

Known before this design (disclosure; sets nothing): per world the clause passes whenever the
whole rule passes, so on these worlds P >= FV.ext.v1's whole-rule P (0.592; by truth Q0 0.568,
R0 0.544, R1c 0.620, X5R1 0.636) and 'weak' can arise only through P0 > 0.5. FV.ext.v1's median
coverage per truth (Q0 0.946, R0 0.944, R1c 0.941, X5R1 0.936) lies just below the clause's
upper bound 0.95: the binding side is over-coverage, so whether P reaches 0.8 depends on the
across-world spread of pooled coverage, which no stored output holds.

Report-only, pre-declared (never decides): the clause's pass rate in a midpoint null
(obs = (ZR_ext + baseline_reg) / 2 + e_k on FV.ext.v1's null noise draws, as
E02.decidability.v1's equal-skill null); coverage quantiles and the share of worlds failing
low (< 0.85) and high (> 0.95) per truth and in each null.

Model-only: no observed head is read, before or in the window. The run reads model forwards
(FV v2 batches, FV v1 truth forwards, ZR0, the registered forward), N3F's registered export
(predictions, interval, registered baseline, per-well datum and residual sd: pre-origin derived,
FV.ext.v1's logged design-time look FV:ext_preorigin_derived_N3Fwells) and FV v1's noise and
anomaly inputs. CPU only; no forward, no twin solver, no GPU.

CLI (repository root, hydro env):
  run_c14f.py smoke --out <new dir outside results/>   synthetic end to end
  run_c14f.py verify                                   registrations, code and worlds intact
  run_c14f.py run                                      once, after registration; report written once
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "fv_decidability_v2"),
           str(REPO / "dataprep_staging" / "fv_decidability"),
           str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ext_fv as XF  # noqa: E402  (FV.ext.v1, frozen; never edited)
import osse_fv as O  # noqa: E402  (FV v1, frozen)
import osse_fv2 as V2  # noqa: E402  (FV v2, frozen)

ID = "C14f.coverage_clause.v1"
WP = "FV"
VERSION = "1.0.0"
F = "dataprep_staging/c14f_clause"
SELF = f"{F}/run_c14f.py"
PREREG = f"{F}/prereg_c14f.py"
TESTS = "tests/test_c14f_clause.py"
LABEL = ("post-failure, exploratory (user ruling R11.7; follows R10.5's regrade of C14f on "
         "FV.ext.v1; never validation)")
EngineeringFailure = O.EngineeringFailure

# ----------------------------------------------------------------------------- FV.ext.v1
FVEXT_ID = XF.ID                                           # "FV.ext.v1"
FVEXT_DECL = f"results/dataprep/preregistration/{FVEXT_ID}.json"
FVEXT_DECL_SHA = "5010831534996cd4a68cc6613731c9e1b97851d71f1925dd7fe2e5a5abab549e"
FVEXT_REPORT = XF.REPORT                                   # results/dataprep/reports/FV.ext.v1
FVEXT_REPORT_SHA = "b29f6cfc92e61c6fc8e12963e953d38b0adf5cb81bea471721bbfe5d9b149c2e"
#: FV.ext.v1's registered frozen_code; imported unchanged, a changed file refuses the run
REUSED = {
    "dataprep_staging/e02c/export_baselines.py":
        "a9ee3610a042d6122a30be73ef14c0f0d373020064e804faf2654f9922a6be94",
    "dataprep_staging/e02c/prereg_e02c_ext.py":
        "9b9dc6a2ec3819b96fe3865d6e4529ef3f0089b2aa4b0b37e9c94834a2aa6a16",
    "dataprep_staging/e02c/score_e02c.py":
        "1c4e092a43ae42c9e57ec96ebf4120b0911155d1f7d3216b27c42a3f41b6bc3d",
    "dataprep_staging/fv_decidability/osse_fv.py":
        "8b0d5a3c110a9f31275a7d15bc42b4666f10907c11f37ab10147ebd2707a5b34",
    "dataprep_staging/fv_decidability_v2/ext_fv.py":
        "f8819beb9c192dddc69031b783d77e931a916807677cf049ab50240dc23dd6cb",
    "dataprep_staging/fv_decidability_v2/osse_fv2.py":
        "3fc5bc7f5a584e4be1276d5cde561610b2c870f846aaa5401b0a527a9a6b36c1",
}
FROZEN = (SELF, *REUSED)
#: published in FV.ext.v1's report and the ledger before this design: disclosure only
FVEXT_KNOWN = {
    "whole_rule_P": 0.592,
    "whole_rule_by_truth": {"Q0": 0.568, "R0": 0.544, "R1c": 0.62, "X5R1": 0.636},
    "median_coverage_by_truth": {"Q0": 0.9457, "R0": 0.9444, "R1c": 0.9407, "X5R1": 0.9356},
    "null_false_pass_whole_rule": 0.0,
}
IDENTITY_SKIP = ("created", "frozen_code")
IDENTITY_TOL = 1e-9

# ----------------------------------------------------------------------------- the clause
#: N3F's registered thresholds (coverage clause and its informative gate; the ratio bound is
#: read only to check each recorded call's pass); the export protocol must state exactly these
N3F_RULE = {"coverage_bounds": [0.85, 0.95], "min_stations": 30, "min_months": 6,
            "max_rmse_ratio": 1.0}
CLAUSE_TEXT = ("pooled central 90 % combined-sigma coverage of the frozen forecast's per-well "
               "interval over every scored well-month in [0.85, 0.95] (closed), with >= 30 "
               "wells of >= 6 window months (N3F.ext_level.prospective_2026-11.v1, test "
               "N3FP.ext_level_2026-11, thresholds.coverage_bounds; decision_rule 'passed: "
               "RMSE ratio <= 1.0 AND coverage in [0.85, 0.95]')")

# ----------------------------------------------------------------------------- design
TAGS = O.TAGS                        # R0, Q0, R1c, X5R1 (FV v1/v2's order)
PLAN = V2.PLAN                       # FV.ext.v1's: 250 worlds per truth
K_TRUTH = XF.K_TRUTH                 # 250
K_NULL = XF.K_NULL                   # 500
SEED = O.SEED                        # FV.ext.v1's (20261101): the same draws, by design
STREAMS = {"truth_noise": XF.STREAM_NOISE, "null": XF.STREAM_NULL,
           "selfcheck": XF.STREAM_SELFCHECK}          # 21, 22, 23 (FV.ext.v1's)
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

#: thresholds as exact fractions: the outcome is decided on integer pass counts (sum of the
#: four truths' counts over 4 x K_TRUTH, and the null count over K_NULL), never on a float
#: mean (150 + 151 + 250 + 249 = 800 of 1000 is exactly 0.8, though its float mean is
#: 0.7999999999999999); floats are for display only
P_MEASURED = Fraction(4, 5)
P0_MEASURED = Fraction(1, 5)
P_WEAK = Fraction(1, 2)
P0_WEAK = Fraction(1, 2)
MC_HALF_WIDTH = V2.MC_HALF_WIDTH     # 0.07
Z_MC = V2.Z_MC
OUTCOMES = ("measured", "weak", "unchanged")
NOT_EVALUABLE = "not_evaluable"
#: the D qualifier each outcome would PROPOSE for C14f (progress rubric); nothing moves
#: without a later user ruling (R11.7)
PROPOSED_GRADE = {"measured": "D-measured", "weak": "D-weak", "unchanged": "D-unknown",
                  NOT_EVALUABLE: "D-unknown"}
OUTCOME_RULE = ("'measured' if P >= 0.8 and P0 <= 0.2; otherwise 'weak' if P < 0.5 or "
                "P0 > 0.5; otherwise 'unchanged' (R9.4's rule, E02.decidability.v1's "
                "pre-declared thresholds, as R10.5 applied them to C14f). P = equal-weight mean "
                "over FV.ext.v1's four truths of the share of their worlds in which N3F's "
                "coverage clause passes; P0 = the share of FV.ext.v1's null worlds in which it "
                "passes. Point estimates decide (applied once), compared as exact fractions of "
                "integer pass counts (P = total clause passes over 4 x 250 worlds; P0 = passes "
                "over 500), never as floats; a 1.645 SE > 0.07 Monte Carlo "
                "half-width on P or P0, or a non-finite estimate, gives 'not_evaluable', mapped "
                "as 'unchanged'")

REPORT_DIR = f"results/dataprep/reports/{ID}"
REPORT = f"{REPORT_DIR}/report.json"
ATTEMPTS_NAME = "attempts.jsonl"
RULINGS = {
    "R11.7": ("results/dataprep/rulings/user_rulings_2026-11-08f.json",
              "656d778bba87a7e0e36696ea46ebe0885e6974e7e505f0bfe8d118810057eb8c"),
    "R10.5": ("results/dataprep/rulings/user_rulings_2026-11-08c.json",
              "8f67973f7d3479fa9784ac3b71c54670c428731c705298d12b4ce011fcf282ac"),
    "R9.4": ("results/dataprep/rulings/user_rulings_2026-11-07c.json",
             "338f2de4cfdbf3811c9a871d3bf4972d18c88f8f9da41000f73d287385d7223f"),
    "R9.4b": ("results/dataprep/rulings/user_rulings_2026-11-07d.json",
              "bfeed6d628a6ca96b289711a722fe03c778438384ffbf3a14562fbcbc47c16bc"),
    **{rid: ("results/dataprep/rulings/user_rulings_2026-11-09.json",
             "7c7219b85ef5ee83d7e1fd88b391367272c81f72b9edb985f2bd325052dc05d1")
       for rid in ("R11.7a", "R11.7b", "R11.7c", "R11.7d")},
}

_HEAD = {
    "en": "Synthetic reading of C14f's coverage clause alone ({id}, post-failure exploratory, "
          "model-only; FV.ext.v1's worlds): a model of the right kind meets N3F's pooled 90 % "
          "coverage clause ([0.85, 0.95]) at the 66 never-calibrated external wells with "
          "probability {p:.3f}, and FV.ext.v1's null (truth at the registered climatology) "
          "meets it with probability {p0:.3f}; ",
    "zh": "僅就 C14f 之涵蓋率條款的合成解讀（{id}，失敗後探索性，純模型；沿用 FV.ext.v1 的"
          "世界）：正確類型的模型在 66 口未參與率定的外部井達到 N3F 合併 90 % 區間涵蓋率條款"
          "（[0.85, 0.95]）的機率為 {p:.3f}，而 FV.ext.v1 的虛無世界（真值為登錄之氣候值）"
          "達到的機率為 {p0:.3f}；"}
_TAIL = {
    "en": " Optimistic (rain at climatology, noise independent between wells, every month "
          "observed); not a strict bound; not validation. C14f's grade changes only if a later "
          "user ruling accepts this reading (R11.7). Model-only; nothing here is evidence "
          "about the aquifer.",
    "zh": "偏樂觀（降雨取氣候值、各井雜訊彼此獨立、每月皆有觀測）；並非嚴格界限；不屬驗證。"
          "僅當日後使用者裁定接受本解讀時，C14f 的等級才會改變（R11.7）。純模型；此處沒有任何"
          "關於含水層的證據。"}
WORDING = {
    "measured": {
        "en": _HEAD["en"] + "by the declared thresholds (P >= 0.8 and P0 <= 0.2) the clause "
              "can decide." + _TAIL["en"],
        "zh": _HEAD["zh"] + "依預先宣告的門檻（P >= 0.8 且 P0 <= 0.2），該條款能夠判定。"
              + _TAIL["zh"]},
    "weak": {
        "en": _HEAD["en"] + "by the declared thresholds (P < 0.5 or P0 > 0.5) the clause is "
              "weak by design: {why}." + _TAIL["en"],
        "zh": _HEAD["zh"] + "依預先宣告的門檻（P < 0.5 或 P0 > 0.5），該條款在設計上屬弱"
              "檢驗：{why}。" + _TAIL["zh"]},
    "unchanged": {
        "en": _HEAD["en"] + "this meets neither the declared 'measured' thresholds (P >= 0.8 "
              "and P0 <= 0.2) nor the 'weak' ones (P < 0.5 or P0 > 0.5), so the clause's "
              "decidability stays unknown." + _TAIL["en"],
        "zh": _HEAD["zh"] + "未達預先宣告的「已量測」門檻（P >= 0.8 且 P0 <= 0.2），亦未達"
              "「弱」門檻（P < 0.5 或 P0 > 0.5），因此該條款的可判定性仍屬未知。" + _TAIL["zh"]},
    NOT_EVALUABLE: {
        "en": "The synthetic reading of C14f's coverage clause alone ({id}, post-failure "
              "exploratory, model-only) is not evaluable ({why}); C14f's decidability stays "
              "unknown.",
        "zh": "僅就 C14f 之涵蓋率條款的合成解讀（{id}，失敗後探索性，純模型）無法評估"
              "（{why}）；C14f 的可判定性仍屬未知。"},
}
WEAK_WHY = {
    "p": {"en": "a model of the right kind misses the band more often than it meets it, so a "
                "fail would not refute the interval",
          "zh": "正確類型的模型落在區間外的機率高於落在區間內，因此未通過不足以否定該區間"},
    "p0": {"en": "a truth at climatology meets the band more often than not, so a pass would "
                 "not show the interval is right",
           "zh": "真值為氣候值時達到該區間的機率高於未達到，因此通過不能顯示該區間正確"},
}
LEDGER_MAPPING = {
    "measured": "proposal: C14f stays D; its qualifier would become D-measured (0.7)",
    "weak": "proposal: C14f stays D; its qualifier would become D-weak (0.5)",
    "unchanged": "C14f stays D; its qualifier stays D-unknown (0.6)",
    NOT_EVALUABLE: "C14f stays D; its qualifier stays D-unknown (0.6)",
    "condition": "R11.7: C14f's grade moves only if a later user ruling accepts this reading; "
                 "until then the ledger may cite the report in C14f's notes and nothing else "
                 "changes (progress_map_v1.json untouched)",
    "never": ["no status change from this reading", "never S", "never 'validation'",
              "E22 (N3F's whole rule) is not re-read: FV.ext.v1's reading stands",
              "no page line without a user ruling",
              "P is not the chance of a real pass; nothing about the aquifer"],
}


def weak_why(p: float, p0: float, lang: str) -> str:
    keys = [k for k, on in (("p", p < P_WEAK), ("p0", p0 > P0_WEAK)) if on]
    sep = "; " if lang == "en" else "；"
    return sep.join(WEAK_WHY[k][lang] for k in keys)


def _count(n, k) -> Fraction:
    if isinstance(n, bool) or isinstance(k, bool) or not isinstance(n, (int, np.integer)) \
            or not isinstance(k, (int, np.integer)) or k <= 0 or not 0 <= n <= k:
        raise EngineeringFailure(f"a pass count must be an int in [0, k], k > 0: {n!r}/{k!r}")
    return Fraction(int(n), int(k))


def exact_p(n_by_truth, k_truth: int) -> Fraction:
    """P as an exact fraction: the equal-weight mean over the truths of n_T / K equals
    sum(n_T) / (len x K) because every truth has the same K."""
    ns = [int(_count(n, k_truth) * k_truth) for n in n_by_truth]
    if len(ns) != len(TAGS):
        raise EngineeringFailure(f"{len(ns)} truth counts, expected {len(TAGS)}")
    return Fraction(sum(ns), len(ns) * int(k_truth))


def outcome_of(n_by_truth, k_truth: int, n0: int, k_null: int, half_p, half_p0) -> str:
    """The pre-declared outcome (see :data:`OUTCOME_RULE`), decided on integer pass counts as
    exact fractions; the Monte Carlo half-widths (floats) only gate 'not_evaluable'."""
    try:
        p, p0 = exact_p(n_by_truth, k_truth), _count(n0, k_null)
    except (EngineeringFailure, TypeError, ValueError):
        return NOT_EVALUABLE
    if any(v is None or not np.isfinite(v) for v in (half_p, half_p0)):
        return NOT_EVALUABLE
    if half_p > MC_HALF_WIDTH or half_p0 > MC_HALF_WIDTH:
        return NOT_EVALUABLE
    if p >= P_MEASURED and p0 <= P0_MEASURED:
        return "measured"
    if p < P_WEAK or p0 > P0_WEAK:
        return "weak"
    return "unchanged"


def wording_for(outcome: str, *, p=None, p0=None, why: str = "") -> dict:
    """The pre-written line of an outcome, filled (EN and ZH)."""
    if outcome == NOT_EVALUABLE:
        return {k: v.format(id=ID, why=why) for k, v in WORDING[NOT_EVALUABLE].items()}
    if outcome not in OUTCOMES:
        raise EngineeringFailure(f"no declared wording for outcome {outcome!r}")
    return {k: v.format(id=ID, p=float(p), p0=float(p0),
                        why=weak_why(p, p0, k) if outcome == "weak" else "")
            for k, v in WORDING[outcome].items()}


def design() -> dict:
    return {"worlds": f"{FVEXT_ID}'s, through its frozen scorer ext_fv.score_ext",
            "seed": SEED, "K_TRUTH": K_TRUTH, "K_NULL": K_NULL, "batch": PLAN.batch,
            "truth_order": list(TAGS), "streams": dict(STREAMS),
            "thresholds": {"P_measured": str(P_MEASURED), "P0_measured": str(P0_MEASURED),
                           "P_weak": str(P_WEAK), "P0_weak": str(P0_WEAK),
                           "compared_as": "exact fractions of integer pass counts",
                           "mc_half_width_90": MC_HALF_WIDTH},
            "clause": dict(N3F_RULE), "quantiles": list(QUANTILES)}


# ----------------------------------------------------------------------------- code pins
def reused_code_problems(root: Path = REPO) -> list[str]:
    out = []
    for p, h in REUSED.items():
        q = Path(root) / p
        if not q.is_file():
            out.append(f"{p} missing")
        elif O.sha256(q) != h:
            out.append(f"{p} differs from {FVEXT_ID}'s registered frozen_code")
    return out


# ----------------------------------------------------------------------------- recorder
@contextlib.contextmanager
def recording():
    """Wrap ``ext_fv.rule`` (looked up as a module global by the frozen ``score_ext``): every
    call returns the frozen rule's own output unchanged and appends its pass, ratio and pooled
    coverage to the yielded list, in call order."""
    calls: list[dict] = []
    orig = XF.rule

    def rec(ctx, obs, baseline):
        out = orig(ctx, obs, baseline)
        calls.append({"pass": bool(out["pass"]), "ratio": float(out["ratio"]),
                      "coverage": float(out["coverage"])})
        return out

    XF.rule = rec
    try:
        yield calls
    finally:
        XF.rule = orig


def split_calls(calls: list[dict], k_null: int, k_truth: int) -> dict:
    """score_ext's call order: the K_NULL null worlds, then per truth (TAGS order) world by
    world the primary call (own climatology) and the report-only call (registered baseline)."""
    want = k_null + len(TAGS) * 2 * k_truth
    if len(calls) != want:
        raise EngineeringFailure(f"recorded {len(calls)} rule calls, expected {want}")
    out = {"null": calls[:k_null]}
    for i, tag in enumerate(TAGS):
        blk = calls[k_null + i * 2 * k_truth: k_null + (i + 1) * 2 * k_truth]
        out[tag] = {"prim": blk[0::2], "reg": blk[1::2]}
    return out


def enough(ctx: dict) -> bool:
    """N3F's informative gate on the frame scored in every world (all rows observed)."""
    f = ctx["frame0"]
    g = f.groupby("station_id")["date"].nunique()
    return bool(len(g) >= N3F_RULE["min_stations"] and (g >= N3F_RULE["min_months"]).all())


def clause_pass(cov: float, ok: bool) -> bool:
    lo, hi = N3F_RULE["coverage_bounds"]
    return bool(ok and lo <= cov <= hi)                   # osse_fv.g0_e02's comparison


def protocol_rule(protocol: dict) -> dict:
    stated = {k: protocol.get(k) for k in N3F_RULE}
    stated["coverage_bounds"] = list(stated["coverage_bounds"] or [])
    if stated != N3F_RULE:
        raise EngineeringFailure(f"the export protocol states {stated}, not N3F's {N3F_RULE}")
    return stated


def check_calls(parts: dict, ok: bool, rep_fv: dict) -> None:
    """Every recorded pass is N3F's whole rule of its own ratio and coverage; primary and
    report-only calls of a world share their coverage; the recorded passes and medians
    reproduce the frozen scorer's own per-truth summary."""
    mx = N3F_RULE["max_rmse_ratio"]
    allc = parts["null"] + [c for t in TAGS for k in ("prim", "reg") for c in parts[t][k]]
    for c in allc:
        if c["pass"] != (c["ratio"] <= mx and clause_pass(c["coverage"], ok)):
            raise EngineeringFailure("a recorded pass is not N3F's rule of its ratio and "
                                     "coverage")
    for t in TAGS:
        a = [c["coverage"] for c in parts[t]["prim"]]
        b = [c["coverage"] for c in parts[t]["reg"]]
        if a != b:
            raise EngineeringFailure(f"truth {t}: coverage differs between a world's two calls")
        tw = rep_fv["truth_worlds"][t]
        if O._rate([c["pass"] for c in parts[t]["prim"]]) != tw["passed"] \
                or float(np.median(a)) != tw["median_coverage"]:
            raise EngineeringFailure(f"truth {t}: recorded calls do not reproduce the scorer's "
                                     "summary")


def _diff(a, b, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: key on one side only" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += _diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [x for i, (u, v) in enumerate(zip(a, b, strict=True))
                for x in _diff(u, v, f"{path}[{i}]")]
    if isinstance(a, bool) or isinstance(b, bool) or not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [] if a == b else [f"{path}: {a!r} vs {b!r}"]
    if math.isnan(a) and math.isnan(b):
        return []
    return [] if abs(a - b) <= IDENTITY_TOL else [f"{path}: {a!r} vs {b!r}"]


def identity_problems(rep_fv: dict, reference: dict) -> list[str]:
    """Gate I: the frozen scorer's report on this run against the reference (FV.ext.v1's
    stored report in a real run; an unwrapped scorer run on the same inputs in a smoke)."""
    a = json.loads(json.dumps({k: v for k, v in rep_fv.items() if k not in IDENTITY_SKIP},
                              default=O._jsonable))
    b = {k: v for k, v in reference.items() if k not in IDENTITY_SKIP}
    return _diff(a, b)


# ----------------------------------------------------------------------------- statistics
def _quantiles(x) -> dict:
    x = np.asarray(x, float)
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(x, q)) for q in QUANTILES}


def _sides(cov) -> dict:
    lo, hi = N3F_RULE["coverage_bounds"]
    c = np.asarray(cov, float)
    return {"below_0.85": float((c < lo).mean()), "above_0.95": float((c > hi).mean())}


def summarise(cov: list[float], ok: bool) -> dict:
    """Clause pass count (int, decides) and rate (float, display); coverage quantiles and the
    fail-side shares are report-only and never change the outcome or the proposal."""
    passes = [clause_pass(c, ok) for c in cov]
    return {"clause_passed": {**O._rate(passes), "n": int(sum(passes))},
            "coverage_quantiles": _quantiles(cov), "fail_side_share": _sides(cov)}


def pooled(per: dict, k_truth: int) -> dict:
    ps = [(per[t]["clause_passed"]["p"], per[t]["clause_passed"]["mc_se"]) for t in TAGS]
    ns = [per[t]["clause_passed"]["n"] for t in TAGS]
    exact = exact_p(ns, k_truth)
    se = math.sqrt(sum(s ** 2 for _, s in ps)) / len(ps)
    return {"p": float(exact), "exact": f"{exact.numerator}/{exact.denominator}",
            "n_by_truth": dict(zip(TAGS, ns, strict=True)), "n_total": int(sum(ns)),
            "k_total": len(ns) * int(k_truth),
            "mc_se": se, "mc_half_width_90": Z_MC * se,
            "by_truth": {t: p for t, (p, _) in zip(TAGS, ps, strict=True)},
            "min_over_truths": float(min(p for p, _ in ps))}


def _in_se(v: float, thr: float, se: float) -> float | None:
    return (float(v) - float(thr)) / se if se > 0 else None


def midpoint_null(ctx: dict, k_null: int) -> list[float]:
    """Report-only: obs = (ZR_ext + baseline_reg) / 2 + e_k on FV.ext.v1's null draws."""
    mid = 0.5 * (ctx["zr"] + ctx["base_reg"])
    return [XF.rule(ctx, mid + O.simulate_ar1(ctx["sd"], ctx["phi"], O.N_WIN,
                                              O.rng_for(XF.STREAM_NULL, 0, k)),
                    ctx["base_reg"])["coverage"] for k in range(k_null)]


# ----------------------------------------------------------------------------- scoring
@dataclass
class ClausePaths:
    ext: XF.ExtPaths = field(default_factory=lambda: XF.ExtPaths(report=REPORT))
    fvext_decl: str = FVEXT_DECL
    fvext_report: str = FVEXT_REPORT
    report: str = REPORT
    rulings: dict = field(default_factory=lambda: dict(RULINGS))

    def all_paths(self) -> list[str]:
        e = self.ext
        return [e.ext_dir, e.v2.arms_dir, e.v2.attempts, e.v2.v1.zr0, e.v2.v1.zr_forward,
                e.v2.v1.inputs_dir, *[p for ps in e.v2.batches.values() for p in ps],
                *e.v2.v1_truths.values(), self.fvext_decl, self.fvext_report, self.report]


def _score(P: ClausePaths, *, smoke: bool = False, plan: V2.Plan | None = None,
           k_null: int | None = None, reference: dict | None = None, prereg_dir=None,
           rows_dir=None, log=print) -> dict:
    """FV.ext.v1's frozen scorer under the recorder, gate I, then P, P0 and the outcome.
    Returns the report (not written). Every gate raises EngineeringFailure before a clause
    statistic is formed (nothing written: a free engineering retry)."""
    plan = PLAN if plan is None else plan
    k_null = K_NULL if k_null is None else k_null
    decl = dsha = fv_doc = fv_sha = None
    look = {"unit": XF.EXT_LOOK_UNIT, "new_look_logged": False, "ruling": "R11.7d",
            "logged_look": None}
    if smoke:
        if any(_under_results(p) for p in P.all_paths()):
            raise EngineeringFailure("a smoke runs on synthetic worlds outside results/ only")
        if reference is None:
            raise EngineeringFailure("a smoke needs an unwrapped reference run")
    else:
        if plan != PLAN or k_null != K_NULL or reference is not None:
            raise EngineeringFailure(f"{ID} scores only on FV.ext.v1's plan and K, against "
                                     f"FV.ext.v1's stored report")
        decl, dsha = load_registered(ID, prereg_dir, rows_dir)
        fv_doc, fv_sha = V2.load_registered(FVEXT_ID, XF.FROZEN)
        if fv_sha != FVEXT_DECL_SHA:
            raise EngineeringFailure(f"{FVEXT_ID} sha256 {fv_sha} is not {FVEXT_DECL_SHA}")
        if O.sha256(O.rp(P.fvext_report)) != FVEXT_REPORT_SHA:
            raise EngineeringFailure(f"{FVEXT_ID}'s report is not the one at {FVEXT_REPORT_SHA}")
        reference = json.loads(O.rp(P.fvext_report).read_text(encoding="utf-8"))
        look["logged_look"] = XF.look_logged(prereg_dir, P.ext.ext_dir)
        if look["logged_look"] is None:
            raise EngineeringFailure(f"{XF.EXT_LOOK_UNIT} is not logged for these export files: "
                                     "R11.7d cites it and allows no new look")
        V2.check_attempts(O.rp(P.ext.v2.attempts), {
            (t, b): O.rp(p) for t, ps in P.ext.v2.batches.items() for b, p in enumerate(ps)})
    bad = reused_code_problems()
    if bad:
        raise EngineeringFailure(f"reused code changed: {bad}")
    # ---- FV.ext.v1's scorer, unchanged, recorded
    with recording() as calls:
        rep_fv = XF.score_ext(P.ext, plan=plan, smoke=smoke, k_null=k_null, decl=fv_doc,
                              decl_sha=fv_sha, log=lambda *_: None)
    if rep_fv.get("status") != "scored":
        raise EngineeringFailure(f"FV.ext.v1's scorer did not score ({rep_fv.get('why')})")
    # ---- gate I and the call checks
    bad = identity_problems(rep_fv, reference)
    if bad:
        raise EngineeringFailure(f"gate I: the worlds are not {FVEXT_ID}'s: {bad[:5]}")
    ctx = XF.load_ext(P.ext)
    rule = protocol_rule(ctx["protocol"])
    ok = enough(ctx)
    if not ok:
        raise EngineeringFailure("the scored frame fails N3F's >= 30 wells x >= 6 months gate")
    parts = split_calls(calls, k_null, plan.k_truth)
    check_calls(parts, ok, rep_fv)
    log("gates E and I passed")
    # ---- the clause
    per = {t: summarise([c["coverage"] for c in parts[t]["prim"]], ok) for t in TAGS}
    for t in TAGS:
        per[t]["whole_rule_passed_fvext"] = rep_fv["truth_worlds"][t]["passed"]
    null = summarise([c["coverage"] for c in parts["null"]], ok)
    P1 = pooled(per, plan.k_truth)
    n0 = null["clause_passed"]["n"]
    P0 = {"p": float(_count(n0, k_null)), "exact": f"{n0}/{k_null}", "n": n0, "k": k_null,
          "mc_se": null["clause_passed"]["mc_se"],
          "mc_half_width_90": Z_MC * null["clause_passed"]["mc_se"]}
    mid = summarise(midpoint_null(ctx, k_null), ok)
    rep: dict = {
        "id": ID, "version": VERSION, "label": LABEL, "smoke": smoke, "created": O._now(),
        "declaration_sha256": dsha, "design": design(), "clause": CLAUSE_TEXT,
        "rule_checked": rule, "model_only": True,
        "worlds": {"source": FVEXT_ID, "declaration_sha256": fv_sha,
                   "report_sha256": FVEXT_REPORT_SHA if not smoke else None,
                   "gate_I": "the frozen scorer's report equals the reference on every key "
                             f"except {list(IDENTITY_SKIP)} (floats within {IDENTITY_TOL})",
                   "n_wells": ctx["W"], "rows_per_world": int(len(ctx["frame0"]))},
        "known_before": FVEXT_KNOWN,
        "truth_worlds": per,
        "null_world": {**null, "definition": "FV.ext.v1's: obs0 = registered baseline + e_k "
                       "(stream 22)"},
        "P": P1, "P0": P0,
        "report_only": {"midpoint_null": {**mid, "definition": "obs = (ZR_ext + "
                                          "baseline_reg) / 2 + e_k on FV.ext.v1's null draws; "
                                          "never decides"}},
        "thresholds": design()["thresholds"],
        "distance_in_mc_se": {
            "P_minus_0.8": _in_se(P1["p"], P_MEASURED, P1["mc_se"]),
            "P_minus_0.5": _in_se(P1["p"], P_WEAK, P1["mc_se"]),
            "P0_minus_0.2": _in_se(P0["p"], P0_MEASURED, P0["mc_se"]),
            "P0_minus_0.5": _in_se(P0["p"], P0_WEAK, P0["mc_se"])},
        "mc_gate": {"max_half_width": MC_HALF_WIDTH,
                    "P": P1["mc_half_width_90"] <= MC_HALF_WIDTH,
                    "P0": P0["mc_half_width_90"] <= MC_HALF_WIDTH,
                    "note": V2.mc_note(len(TAGS), plan.k_truth)},
        "design_time_look": look,
        "report_only_note": "coverage quantiles, fail_side_share and the midpoint null never "
                            "change the outcome or the proposal",
        "engineering": {"fvext_engineering": rep_fv["engineering"],
                        "rule_calls": len(calls), "reused_code": dict(REUSED),
                        "versions": versions()}}
    out = outcome_of([P1["n_by_truth"][t] for t in TAGS], plan.k_truth, n0, k_null,
                     P1["mc_half_width_90"], P0["mc_half_width_90"])
    if out == NOT_EVALUABLE:
        why = "Monte Carlo half-width above 0.07 or a non-finite estimate"
        rep.update(status=NOT_EVALUABLE, outcome=out, why=why,
                   wording=wording_for(out, why=why))
    else:
        rep.update(status="scored", outcome=out,
                   wording=wording_for(out, p=exact_p(list(P1["n_by_truth"].values()),
                                                      plan.k_truth),
                                       p0=_count(n0, k_null)))
    rep.update(rule_applied=OUTCOME_RULE,
               ledger_mapping={"outcome": LEDGER_MAPPING[out],
                               "proposed_grade_key": PROPOSED_GRADE[out],
                               "condition": LEDGER_MAPPING["condition"],
                               "never": LEDGER_MAPPING["never"]})
    if decl is not None:
        rep["frozen_code"] = decl["frozen_code"]
    return rep


def versions() -> dict:
    import pandas as pd

    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__}


def write_once_atomic(path: Path, obj: dict) -> None:
    """Write ``path`` once: a temporary file in the same directory, then a no-clobber link
    into place (fails if ``path`` exists), so a crash never leaves a partial report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=True, default=O._jsonable)
            f.write("\n")
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


# ----------------------------------------------------------------------------- registration
def index_record(prereg_id: str = ID, prereg_dir=None) -> dict | None:
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    p = d / "INDEX.jsonl"
    if not p.is_file():
        return None
    rows = [r for r in prereg3.read_chained(p) if r["id"] == prereg_id]
    return rows[-1] if rows else None


def load_registered(prereg_id: str = ID, prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    """Refuse unless this declaration is in INDEX.jsonl, loads intact at that sha256, its
    frozen code is byte-identical and its design is this code's."""
    from harness import prereg3

    if prereg_id != ID:
        raise EngineeringFailure(f"this runner scores {ID} only, not {prereg_id!r}")
    rec = index_record(prereg_id, prereg_dir)
    if rec is None:
        raise EngineeringFailure(f"{prereg_id} is not in INDEX.jsonl: register it first")
    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    try:
        doc, sha = prereg3.load_prereg(prereg_id, **kw)
    except Exception as err:      # noqa: BLE001 - any failure is a refusal
        raise EngineeringFailure(f"{prereg_id} is not registered or not intact: {err}") from err
    if sha != rec.get("sha256"):
        raise EngineeringFailure(f"{prereg_id}: the document's sha256 is not INDEX.jsonl's")
    fc = doc.get("frozen_code") or {}
    if set(fc) != set(FROZEN):
        raise EngineeringFailure(f"frozen_code does not list exactly {ID}'s files")
    bad = [p for p, h in fc.items() if not O.rp(p).is_file() or O.sha256(O.rp(p)) != h]
    if bad:
        raise EngineeringFailure(f"code sha256 differs from the registration: {bad}")
    changed = prereg3.changed_inputs(doc)
    if changed:
        raise EngineeringFailure(f"registered inputs changed: {changed}")
    d = doc.get("design") or {}
    bad = sorted(k for k, v in json.loads(json.dumps(design())).items() if d.get(k) != v)
    if bad:
        raise EngineeringFailure(f"the registered design differs from this code on {bad}")
    return doc, sha


def attempts_path(report: str) -> Path:
    return O.rp(report).parent / ATTEMPTS_NAME


def _log_attempt(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": ID, "time": O._now(), **rec}, sort_keys=True,
                            ensure_ascii=False, default=O._jsonable) + "\n")


def _attempt_no(path: Path) -> int:
    if not path.is_file():
        return 1
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return 1 + sum(r.get("event") == "start" for r in rows)


def run(P: ClausePaths | None = None, *, prereg_dir=None, rows_dir=None, log=print) -> dict:
    """Score once. Refuses (nothing logged, nothing read) unless registered and intact; then
    logs 'start' and 'engineering_failure', 'error' or 'success' (with the report sha256) to
    attempts.jsonl beside the report, and writes report.json once (temp file + no-clobber
    link). No statistic is logged; a failed attempt writes no clause number anywhere."""
    P = P or ClausePaths()
    load_registered(ID, prereg_dir, rows_dir)          # before anything is logged or read
    outp = O.rp(P.report)
    if outp.exists():
        raise EngineeringFailure(f"{outp} exists: {ID} is scored once")
    ap = attempts_path(P.report)
    n = _attempt_no(ap)
    _log_attempt(ap, {"event": "start", "attempt": n, "report": str(P.report),
                      "code_sha256": O.sha256(Path(__file__))})
    try:
        rep = _score(P, prereg_dir=prereg_dir, rows_dir=rows_dir, log=log)
        rep["attempt"] = n
        write_once_atomic(outp, rep)
    except EngineeringFailure as err:
        _log_attempt(ap, {"event": "engineering_failure", "attempt": n, "message": str(err)})
        raise
    except Exception as err:
        _log_attempt(ap, {"event": "error", "attempt": n,
                          "message": f"{type(err).__name__}: {err}"})
        raise
    _log_attempt(ap, {"event": "success", "attempt": n, "report_sha256": O.sha256(outp),
                      "declaration_sha256": rep.get("declaration_sha256")})
    return rep


# ----------------------------------------------------------------------------- synthetic
def _under_results(p) -> bool:
    """True for any path that is, or may be, part of the real results tree (lexically,
    resolved, or through a symlinked results/ in a build worktree)."""
    q = Path(p)
    q = q if q.is_absolute() else REPO / q
    root = REPO / "results"
    for cand in (Path(os.path.abspath(q)), q.resolve()):
        for r in (root, root.resolve()):
            try:
                cand.relative_to(r)
                return True
            except ValueError:
                pass
    return "results" in q.resolve().parts


SYNTH_PLAN = V2.Plan(k_truth=6, batch=4)


def build_synthetic(out: Path, plan: V2.Plan = SYNTH_PLAN) -> ClausePaths:
    """SYNTHETIC inputs only: FV.ext.v1's own synthetic builder (FV v2's synthetic worlds plus a
    synthetic N3F-style export of 40 wells)."""
    out = Path(out).resolve()
    if _under_results(out):
        raise EngineeringFailure("synthetic inputs never live under results/")
    ext = XF.build_synthetic(out, plan)
    rep = str(out / "report" / "report.json")
    ext.report = rep
    rulings = {}
    for rid, (_, sha) in RULINGS.items():
        q = out / f"rulings_{rid}.json"
        q.write_text(json.dumps({"rulings": [{"id": rid, "decision": "synthetic stand-in"}]})
                     + "\n")
        rulings[rid] = (str(q), sha)
    return ClausePaths(ext=ext, fvext_decl=str(out / "fvext_decl.json"),
                       fvext_report=str(out / "fvext_report.json"), report=rep,
                       rulings=rulings)


def run_smoke(out: Path, k_null: int = 20) -> dict:
    """Synthetic end to end: an unwrapped FV.ext.v1 scorer run is the gate-I reference, then
    the recorded run. Writes out/smoke_report.json once; numbers mean nothing about the twin."""
    out = Path(out).resolve()
    if _under_results(out):
        raise EngineeringFailure("the smoke never writes under results/")
    P = build_synthetic(out / "inputs_synthetic")
    ref = XF.score_ext(P.ext, plan=SYNTH_PLAN, smoke=True, k_null=k_null, log=lambda *_: None)
    ref = json.loads(json.dumps(ref, default=O._jsonable))
    rep = _score(P, smoke=True, plan=SYNTH_PLAN, k_null=k_null, reference=ref,
                 log=lambda *_: None)
    rep["note"] = "SMOKE: synthetic inputs only; numbers mean nothing about the twin"
    O._write_json_once(out / "smoke_report.json", rep)
    return rep


# ----------------------------------------------------------------------------- CLI
def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke", help="synthetic end to end")
    sm.add_argument("--out", required=True, help="a NEW directory outside results/")
    sub.add_parser("verify", help="registrations and code intact (no statistic)")
    sub.add_parser("run", help="score once, after registration")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out))
            print(f"smoke {rep['status']} (synthetic)")
            return 0
        if a.cmd == "verify":
            _, dsha = load_registered(ID, prereg_dir, rows_dir)
            V2.load_registered(FVEXT_ID, XF.FROZEN)
            bad = reused_code_problems()
            if bad:
                raise EngineeringFailure("; ".join(bad))
            print(f"{ID} registered (sha256 {dsha[:16]}); {FVEXT_ID} and code intact")
            return 0
        rep = run(prereg_dir=prereg_dir, rows_dir=rows_dir)
        print(f"wrote {REPORT} (status {rep['status']}; outcome {rep.get('outcome')})")
        return 0
    except EngineeringFailure as err:
        print(f"REFUSED / ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
