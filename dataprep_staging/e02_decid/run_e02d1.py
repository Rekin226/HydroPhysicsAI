"""E02.decidability.v1 (round 9, user ruling R9.4; model-only, CPU): can E02's OWN pooled rule
decide?

STATUS: not registered by user ruling R9.4b (seventeenth set,
results/dataprep/rulings/user_rulings_2026-11-07d.json, sha256 bfeed6d6). FV.decidability.v2
reading 1 already gives P = 0.736 on E02's own rule on these same forwards, so the outcome would
almost surely be 'unchanged'; no registration is spent on a known answer. This file is kept as
unregistered code: ``run`` refuses without a registration, and none is to be made.

E02 (results/twin/prospective_2026-11, registered 2026-10-11) is decided by
``twin.validation.evaluate`` on its registered protocol: pooled central 90 % coverage in
[0.85, 0.95] AND equal-station RMSE ratio to the month-of-year climatology (``baseline_m``) <=
1.0, with >= 100 stations of >= 6 window months. FV.decidability.v2_postfailure scored that
rule only as a report-only side rate (P 0.736) and gave it no null reading, so E02's
decidability is unknown in the progress rubric. This reading measures it, with both sides:

* P, the power: the share of RIGHT-KIND worlds in which E02's rule passes. The worlds are FV
  v2's own truth families (R0, Q0, R1c, X5R1), each world a full forward already on disk::

      obs(w, t) = ZR(w, t) + D_T(w, t) + R_T,k(w, t) + e_k(w, t)       t in 2026-11..2027-10

  with ZR E02's registered prediction, D_T and R_T,k exactly as FV v2's ``score_worlds`` builds
  them (``osse_fv.departure`` on the batch baseline against ZR0; the batch world minus its
  baseline), and e_k FV v1's per-well AR(1) noise on THIS reading's fresh seed and stream.
  P = equal-weight mean over the four truths of each truth's pass rate (K_TRUTH = 250 worlds).
* P0, the null pass rate: the share of NO-SKILL worlds in which the rule passes. As FV.stat.v2's
  null is the world in which the tested forecast has no skill over climatology (rho = 0: its
  layer reverts to climatology), this null puts zero skill in the truth, because E02's forecast
  is frozen and cannot revert::

      obs0(w, t) = (ZR(w, t) + B(w, t)) / 2 + e_k(w, t)

  B is E02's own baseline (``baseline_m`` of the registered predictions). The truth mean is
  equidistant from the forecast and the baseline at every well-month, so their expected squared
  errors are equal: the frozen forecast has exactly no skill over E02's baseline in
  expectation. The noise is the truth worlds' (FV v1's AR(1)), on its own stream; K_NULL worlds.

Three outcomes, pre-declared (``outcome_of``; point estimates decide, as in every precedent):
'measured' (P >= 0.8 and P0 <= 0.2: E02 D measured), 'weak' (P < 0.5 or P0 > 0.5: E02 D weak),
otherwise 'unchanged' (E02 stays D unknown); a failed Monte Carlo check (1.645 SE <= 0.07 on P or
P0; non-binding at this K) gives 'not_evaluable', mapped as 'unchanged'. No status changes: the
outcome sets only E02's D qualifier in the progress rubric (progress_map_v1.json
``reading_outcomes``), once the ledger cites the registration and this report and records the
outcome.

Code reused UNCHANGED by import and pinned by sha256 (:data:`REUSED`): FV v2's registered
frozen_code for osse_fv2.py, osse_fv.py, score_e02c.py, export_baselines.py, coverage3.py and
bands.py, and FV.stat.v2's for hydrophysics/twin/validation.py and release.py (evaluate). A
changed file refuses the run.

Model-only: no observed head is read. The reading reads model forwards, E02's registered
predictions and FV v1's / E02c's pre-origin observation-derived statistics (the AR(1) sd and
lag-1, the observed-month mask, CL/CLT, which ``osse_fv.load_context`` loads and E02's rule
does not use).

Gates, in order, before any statistic:
  E  engineering: registration intact (real run), frozen code and inputs unchanged, every world
     file at its declared sha256 and each FV v2 batch the output of its last successful forward
     attempt (FV v2's attempts.jsonl); FV v2's checks unchanged (arms, batch metadata, hindcast
     identity 0.0 m against FV v1's truth forwards, X5R1 reproduction within 0.02 m, ZR0
     metadata); the frame order; E02's rule as the protocol states it; on three self-check
     worlds (two right-kind, one null) this reading's E02 rule == twin.validation.evaluate on
     the same rows (osse_fv.evaluate_selfcheck) == FV v2's own ``world_stats`` E02 rate. A
     failure raises EngineeringFailure: nothing is written (free retry before any statistic).
  MC 1.645 x Monte Carlo SE of P and of P0 <= 0.07 (non-binding: largest possible 0.026 and
     0.018); a failure gives 'not_evaluable' (a possible bug).

Scoring is private: :func:`run` loads the registration itself; ``_score`` refuses a real run
unless the declaration in INDEX.jsonl loads intact with this code's design, and a smoke unless
every path lies outside results/ (lexically, resolved, or through a symlinked results/).

CLI (repository root, hydro env):
  run_e02d1.py smoke --out <new dir outside results/>   synthetic end to end
  run_e02d1.py verify                                   registration, code and worlds intact
  run_e02d1.py run                                      once, after registration; report mode 'x'
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_PATHS = (str(HERE), str(REPO / "dataprep_staging" / "fv_decidability_v2"),
          str(REPO / "dataprep_staging" / "fv_decidability"),
          str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"), str(REPO))


def _paths() -> None:
    for p in _PATHS:
        if p not in sys.path:
            sys.path.insert(0, p)


_paths()

import osse_fv as O  # noqa: E402  (FV v1, frozen; never edited)
import osse_fv2 as V2  # noqa: E402  (FV v2, frozen; never edited)
import score_e02c as S  # noqa: E402  (E02c, frozen; annotate, window)

ID = "E02.decidability.v1"
WP = "FV"
VERSION = "1.0.0"
F = "dataprep_staging/e02_decid"
SELF = f"{F}/run_e02d1.py"
PREREG = f"{F}/prereg_e02d1.py"
TESTS = "tests/test_e02_decid.py"
EngineeringFailure = O.EngineeringFailure

# ----------------------------------------------------------------------------- reused code
FV2_ID = V2.ID                                   # FV.decidability.v2_postfailure
FVS2_ID = "FV.stat.v2"
#: registered frozen_code sha256 of every reused file, and the declaration it is pinned to
REUSED = {
    "dataprep_staging/fv_decidability_v2/osse_fv2.py":
        "3fc5bc7f5a584e4be1276d5cde561610b2c870f846aaa5401b0a527a9a6b36c1",
    "dataprep_staging/fv_decidability/osse_fv.py":
        "8b0d5a3c110a9f31275a7d15bc42b4666f10907c11f37ab10147ebd2707a5b34",
    "dataprep_staging/e02c/score_e02c.py":
        "1c4e092a43ae42c9e57ec96ebf4120b0911155d1f7d3216b27c42a3f41b6bc3d",
    "dataprep_staging/e02c/export_baselines.py":
        "a9ee3610a042d6122a30be73ef14c0f0d373020064e804faf2654f9922a6be94",
    "dataprep_staging/harness/coverage3.py":
        "ffe13c84ee137920e239e69772b69bff30efb9005c050bb711c536c01bda7d17",
    "hydrophysics/twin/bands.py":
        "99a88018fc91824442222a5420ef701174b0b748787cdd9f5d0d464c3771b3d0",
    "hydrophysics/twin/validation.py":
        "d827cace444dff7d26f1c1254daa8684ccad6c671f3a51e704b0f2cb2c7b849c",
    "hydrophysics/twin/release.py":
        "4c40a17ef5214bfc7dee6f3296e5007fb447686c8ae9e31d111f3dbe34e965ba",
}
REUSED_FROM = {p: (FVS2_ID if p.startswith("hydrophysics/twin/") and not p.endswith("bands.py")
                   else FV2_ID) for p in REUSED}
FROZEN = (SELF, *REUSED)
#: FV v1's export as FV v2 registered it (inputs_frozen)
FV1_INPUTS = {
    "anomaly_pool.csv": "acab736764fd635ecd21eb62f77d26dd3ea8132a184678e7ac0a0fd1fc94f290",
    "noise_ar1.csv": "73965a08914149af2132c96f617268701277b49411cf17cfbf57d41fbeaa9d55",
    "obs_mask.csv": "605a35de5d07cab40cef3a1e86fae719502ff3afcd0f6ea36310ccf5425d0c89"}

# ----------------------------------------------------------------------------- E02's rule
#: E02's registered rule (results/twin/prospective_2026-11/protocol.json, sha 8a96c0ea); the
#: run checks the protocol states exactly this
E02_RULE = {"max_rmse_ratio": 1.0, "min_stations": 100, "min_months": 6,
            "coverage_bounds": [0.85, 0.95]}
E02_PROTOCOL_SHA = S.E02_PROTOCOL_SHA
E02_PRED_SHA = S.E02_PRED_SHA
E02_MANIFEST_SHA = "d7286551245644b7932fac98751bcd0e837611ffdfbc9a403dd096c8c51af056"
#: FV v2 reading 1's report-only rate of this rule on the same forwards (other noise draws),
#: published in the claims ledger (E02 notes) before this declaration: disclosure only
FV2_REPORT_ONLY_P = {"pooled": 0.736, "Q0": 0.468, "R0": 0.492, "R1c": 0.992, "X5R1": 0.992}

# ----------------------------------------------------------------------------- design
TAGS = O.TAGS                                    # R0, Q0, R1c, X5R1 (FV v1/v2's order)
PLAN = V2.PLAN                                   # 250 worlds per truth, batches of 50
K_TRUTH = V2.K_TRUTH                             # 250 (every FV v2 world, one noise draw each)
K_NULL = 2000                                    # one null cell
SEED = 60417293                                  # fixed in code before registration (fresh)
SMOKE_SEED = 18830527                            # engineering smokes only
STREAM_TRUTH, STREAM_NULL, STREAM_SELFCHECK = 71, 72, 73   # FV v1 1-4, v2 11-13, FW.rider 61-62
N_SELFCHECK_TRUTH = 2
REFUSED_SEEDS = {
    SMOKE_SEED: "this reading's own engineering smoke seed",
    O.SEED: "FV v1/v2's seed (20261101): its noise stream drew reading 1's worlds, whose "
            "report-only E02 rate 0.736 is known",
}
if SEED in REFUSED_SEEDS or SEED in (46190527, 31750093):     # FW.rider's seeds too
    raise ImportError("the registered seed must be fresh")

# ----------------------------------------------------------------------------- thresholds
P_MEASURED = 0.8
P0_MEASURED = 0.2
P_WEAK = 0.5
P0_WEAK = 0.5
MC_HALF_WIDTH = V2.MC_HALF_WIDTH                 # 0.07
Z_MC = V2.Z_MC
OUTCOMES = ("measured", "weak", "unchanged")
NOT_EVALUABLE = "not_evaluable"
#: the D qualifier each outcome gives E02 in the progress rubric (progress_map_v1.json)
LEDGER_GRADE = {"measured": "D-measured", "weak": "D-weak", "unchanged": "D-unknown",
                NOT_EVALUABLE: "D-unknown"}
OUTCOME_RULE = ("'measured' if P >= 0.8 and P0 <= 0.2; otherwise 'weak' if P < 0.5 or P0 > 0.5; "
                "otherwise 'unchanged'. P = equal-weight mean over FV v2's four truth families "
                "of the share of their worlds that pass E02's pooled rule; P0 = the share of the "
                "equal-skill null worlds that pass it. Point estimates decide (applied once); a "
                "1.645 SE > 0.07 Monte Carlo half-width on P or P0, or a non-finite estimate, "
                "gives 'not_evaluable', mapped as 'unchanged'")

REPORT_DIR = f"results/dataprep/reports/{ID}"
REPORT = f"{REPORT_DIR}/report.json"
ATTEMPTS_NAME = "attempts.jsonl"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-07c.json"
RULINGS_SHA = "338f2de4cfdbf3811c9a871d3bf4972d18c88f8f9da41000f73d287385d7223f"
RULING_ID = "R9.4"
NOT_REGISTERED = ("not registered by user ruling R9.4b (seventeenth set, "
                  "results/dataprep/rulings/user_rulings_2026-11-07d.json, sha256 bfeed6d6): FV v2 "
                  "reading 1's P = 0.736 on E02's own rule already leaves E02 not shown decidable "
                  "(D unknown); no registration is spent on a known answer")

_TAIL = {"en": " Model-only; nothing here is evidence about the aquifer.",
         "zh": "僅模型；此處沒有任何關於含水層的證據。"}
_HEAD = {
    "en": "Synthetic decidability reading of E02's own pooled rule (E02.decidability.v1, "
          "model-only, registered before it ran): in FV.decidability.v2's four truth families "
          "(full forwards, rain at climatology) a model of the right kind passes E02's rule "
          "(pooled 90 % coverage in [0.85, 0.95] and equal-station RMSE ratio to the "
          "month-of-year climatology <= 1.0) with probability {p:.2f}, and a model with no "
          "skill over that climatology passes it with probability {p0:.2f}; ",
    "zh": "E02 自身合併規則的合成可判定性解讀（E02.decidability.v1，僅模型，於執行前登錄）："
          "在 FV.decidability.v2 的四個真值族（完整正演，降雨取氣候值）中，正確類型的模型通過 "
          "E02 規則（合併 90 % 區間涵蓋率落在 [0.85, 0.95] 且等站權重 RMSE 對逐月氣候值之比值 "
          "<= 1.0）的機率為 {p:.2f}，而相對該氣候值無預報能力的模型通過的機率為 {p0:.2f}；"}
_CAVEAT = {
    "en": " Optimistic (noise independent between wells, no county-common shock); not a strict "
          "bound; the right-kind worlds are FV v2's own forwards (its report-only rate 0.736 "
          "was known), only the noise is fresh.",
    "zh": "偏樂觀（井間雜訊獨立，無全縣共同衝擊）；並非嚴格界限；正確類型的世界即 FV v2 "
          "既有的正演（其僅供報告的通過率 0.736 事先已知），僅雜訊為新抽樣。"}
WORDING = {
    "measured": {
        "en": _HEAD["en"] + "by the declared thresholds (P >= 0.8 and P0 <= 0.2) E02's rule can "
              "decide." + _CAVEAT["en"] + _TAIL["en"],
        "zh": _HEAD["zh"] + "依預先宣告的門檻（P >= 0.8 且 P0 <= 0.2），E02 的規則能夠判定。"
              + _CAVEAT["zh"] + _TAIL["zh"]},
    "weak": {
        "en": _HEAD["en"] + "by the declared thresholds (P < 0.5 or P0 > 0.5) E02's rule is weak "
              "by design: {why}." + _CAVEAT["en"] + _TAIL["en"],
        "zh": _HEAD["zh"] + "依預先宣告的門檻（P < 0.5 或 P0 > 0.5），E02 的規則在設計上屬弱"
              "檢驗：{why}。" + _CAVEAT["zh"] + _TAIL["zh"]},
    "unchanged": {
        "en": _HEAD["en"] + "this meets neither the declared 'measured' thresholds (P >= 0.8 and "
              "P0 <= 0.2) nor the 'weak' ones (P < 0.5 or P0 > 0.5), so E02's decidability "
              "stays unknown." + _CAVEAT["en"] + _TAIL["en"],
        "zh": _HEAD["zh"] + "未達預先宣告的「已量測」門檻（P >= 0.8 且 P0 <= 0.2），亦未達"
              "「弱」門檻（P < 0.5 或 P0 > 0.5），因此 E02 的可判定性仍屬未知。"
              + _CAVEAT["zh"] + _TAIL["zh"]},
    NOT_EVALUABLE: {
        "en": "The synthetic decidability reading of E02's own pooled rule (E02.decidability.v1, "
              "model-only) is not evaluable ({why}); E02's decidability stays unknown.",
        "zh": "E02 自身合併規則的合成可判定性解讀（E02.decidability.v1，僅模型）無法評估"
              "（{why}）；E02 的可判定性仍屬未知。"},
}
WEAK_WHY = {
    "p": {"en": "a model of the right kind fails it more often than it passes, so a 2028 fail "
                "would not refute",
          "zh": "正確類型的模型未通過的機率高於通過，因此 2028 年未通過不足以否定"},
    "p0": {"en": "a model with no skill passes it more often than not, so a 2028 pass would "
                 "not show skill",
           "zh": "無預報能力的模型通過的機率高於未通過，因此 2028 年通過不能顯示預報能力"},
}
LEDGER_MAPPING = {
    "measured": "E02 stays D; its progress qualifier becomes D-measured (0.7)",
    "weak": "E02 stays D; its progress qualifier becomes D-weak (0.5)",
    "unchanged": "E02 stays D; its progress qualifier stays D-unknown (0.6)",
    NOT_EVALUABLE: "E02 stays D; its progress qualifier stays D-unknown (0.6)",
    "condition": "applies once the ledger cites E02.decidability.v1 in registry_hashes with its "
                 "real sha256, lists this report in E02's cites.reports and records the outcome "
                 "in E02's decidability_reading (progress_map_v1.json reading_outcomes; "
                 "hydrophysics/twin/progress.py)",
    "never": ["no status change from this reading", "never S",
              "no page line without a user ruling (ledger and docs only)",
              "P is not the chance of a 2028 E02 pass; nothing about the aquifer"],
}


def weak_why(p: float, p0: float, lang: str) -> str:
    keys = [k for k, on in (("p", p < P_WEAK), ("p0", p0 > P0_WEAK)) if on]
    sep = "; " if lang == "en" else "；"
    return sep.join(WEAK_WHY[k][lang] for k in keys)


def outcome_of(p, p0, half_p, half_p0) -> str:
    """The pre-declared outcome of point estimates P and P0 (see :data:`OUTCOME_RULE`)."""
    vals = (p, p0, half_p, half_p0)
    if any(v is None or not np.isfinite(v) for v in vals):
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
        return {k: v.format(why=why) for k, v in WORDING[NOT_EVALUABLE].items()}
    if outcome not in OUTCOMES:
        raise EngineeringFailure(f"no declared wording for outcome {outcome!r}")
    return {k: v.format(p=p, p0=p0, why=weak_why(p, p0, k) if outcome == "weak" else "")
            for k, v in WORDING[outcome].items()}


# ----------------------------------------------------------------------------- code pins
def reused_code_problems(root: Path = REPO) -> list[str]:
    """Reused files whose bytes are not the registered ones (empty = unchanged)."""
    out = []
    for p, h in REUSED.items():
        q = Path(root) / p
        if not q.is_file():
            out.append(f"{p} missing")
        elif O.sha256(q) != h:
            out.append(f"{p} differs from {REUSED_FROM[p]}'s registered frozen_code")
    return out


def check_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EngineeringFailure(f"seed must be a non-negative int, not {seed!r}")
    return seed


def rng_for(stream: int, group: int, k: int, *, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream, group, k]))


# ----------------------------------------------------------------------------- inputs
@dataclass
class DecidPaths:
    """Every file the reading or its declaration builder reads or writes (repo-relative by
    default; a synthetic smoke gives its own, all outside results/)."""
    fv2: V2.Paths2 = field(default_factory=V2.Paths2)
    fv2_decl: str = f"results/dataprep/preregistration/{FV2_ID}.json"
    fvs2_decl: str = f"results/dataprep/preregistration/{FVS2_ID}.json"
    rulings: str = RULINGS
    dataprep: str = "results/dataprep"
    report: str = REPORT

    @property
    def v1(self) -> O.Paths:
        return self.fv2.v1

    def e02_manifest(self) -> str:
        return str(Path(self.v1.pred).parent / "manifest.json")

    def all_paths(self) -> list[str]:
        v = self.v1
        return [self.fv2_decl, self.fvs2_decl, self.rulings, self.dataprep, self.report,
                self.fv2.arms_dir, self.fv2.attempts, v.pred, v.protocol, v.baselines, v.wells,
                v.inputs_dir, v.zr_forward, v.zr0, self.e02_manifest(),
                *[p for ps in self.fv2.batches.values() for p in ps],
                *self.fv2.v1_truths.values(), *[f for f, _ in v.repro.values()]]


def world_files(P: DecidPaths) -> dict[str, str]:
    """{name: path} of every model-output file the worlds are built from. FV v2's attempts log
    is not pinned (other FV v2 scorers append to it); gate E reads it to check that each batch
    is the output of its last successful forward attempt."""
    out = {f"{t}_b{b}": p for t, ps in P.fv2.batches.items() for b, p in enumerate(ps)}
    out.update({f"v1_{t}": p for t, p in P.fv2.v1_truths.items()})
    out.update({"ZR0": P.v1.zr0, "ZR": P.v1.zr_forward,
                "arms_json": str(Path(P.fv2.arms_dir) / "arms.json")})
    out.update({f"repro_{t}": f for t, (f, _) in P.v1.repro.items()})
    return out


def world_shas(P: DecidPaths) -> dict[str, str]:
    """sha256 of every world file (no value is read)."""
    return {k: (O.sha256(O.rp(v)) if O.rp(v).is_file() else "MISSING")
            for k, v in world_files(P).items()}


def world_problems(P: DecidPaths, declared: dict | None) -> list[str]:
    """World files missing, or (with ``declared``) at another sha256 than declared."""
    got = world_shas(P)
    out = [f"{k}: missing" for k, v in got.items() if v == "MISSING"]
    if declared is not None:
        out += [f"{k}: sha256 differs from the declaration" for k, v in got.items()
                if v != "MISSING" and v != declared.get(k)]
        out += [f"{k}: declared but not a world file" for k in declared if k not in got]
    return out


def protocol_rule(real: dict) -> dict:
    """E02's rule as the protocol states it; must equal :data:`E02_RULE`."""
    missing = [k for k in E02_RULE if k not in real]
    if missing:
        raise EngineeringFailure(f"E02 protocol lacks {missing}")
    rule = {k: real[k] for k in E02_RULE}
    if rule != E02_RULE:
        raise EngineeringFailure("E02's protocol states another rule than the declared one")
    return rule


def check_frame_order(ctx: dict) -> np.ndarray:
    """The window rows must run well by well (ctx ids) and month by month, so that a (W, 12)
    matrix maps onto them by reshape (as osse_fv.world_stats assumes). Returns E02's baseline
    B (W, 12) from ``baseline_m``."""
    f = ctx["frame0"]
    win = S.window_months(*S.WINDOW)
    ids = np.repeat(np.asarray(ctx["ids"], dtype=object), len(win))
    dates = np.tile(np.asarray(win), len(ctx["ids"]))
    if len(f) != len(ids) or not (f["station_id"].to_numpy(dtype=object) == ids).all() or \
            not (f["date"].to_numpy() == dates).all():
        raise EngineeringFailure("prediction rows are not in well x month order")
    B = f["baseline_m"].to_numpy(float).reshape(len(ctx["ids"]), len(win))
    if not np.isfinite(B).all():
        raise EngineeringFailure("non-finite baseline_m in the window")
    return B


def load_worlds(P: DecidPaths, ctx: dict, plan: V2.Plan = PLAN) -> dict:
    """FV v2's truth worlds, built by FV v2's own functions, unchanged: D_T (W, 12) and the
    window responses R_T (K, W, 12) per truth, with FV v2's engineering checks (arms, batch
    metadata, hindcast identity against FV v1's truth forwards, X5R1 reproduction, ZR0)."""
    eng: dict = {"truths": {}, "repro": {}}
    arms = V2.check_arms(P.fv2.arms_dir, ctx["pool"], plan)
    zr_meta, zr_heads = O.well_heads(O.rp(P.v1.zr_forward), ctx["cells"], ctx["layers"])
    del zr_heads
    zr0_meta, zr0_heads = O.well_heads(O.rp(P.v1.zr0), ctx["cells"], ctx["layers"])
    eng["zr0_meta"] = O.check_zr0_meta(zr0_meta, zr0_heads, zr_meta)
    h_zr0 = zr0_heads[0]
    del zr0_heads
    T = {}
    for i, tag in enumerate(TAGS):
        ref, _ = V2.v1_reference(P.fv2.v1_truths[tag], ctx["cells"], ctx["layers"], tag)
        lt = V2.load_truth(P.fv2.batches[tag], tag, arms, ctx["cells"], ctx["layers"], ref,
                           plan)
        eng["truths"][tag] = lt["batches"]
        D = O.departure(lt["base"], h_zr0, ctx["mask"])
        if tag in P.v1.repro:
            path, gating = P.v1.repro[tag]
            _, rf = O.well_heads(O.rp(path), ctx["cells"], ctx["layers"])
            err = float(np.max(np.abs(lt["base"][:, :O.T_FIT] - rf[0][:, :O.T_FIT])))
            eng["repro"][tag] = {"max_abs_hindcast_m": err, "tolerance_m": O.REPRO_TOL_M,
                                 "gating": gating, "pass": err <= O.REPRO_TOL_M}
            if gating and err > O.REPRO_TOL_M:
                raise EngineeringFailure(f"{tag} hindcast does not reproduce its forward "
                                         f"(max {err:.4f} m > {O.REPRO_TOL_M} m)")
        T[tag] = {"i": i, "D": D, "R": lt["R"]}
        del ref, lt
    return {"T": T, "eng": eng}


# ----------------------------------------------------------------------------- statistics
def e02_rule(ctx: dict, obs: np.ndarray) -> dict:
    """E02's own pooled checks on one world (osse_fv.g0_e02, the evaluate arithmetic)."""
    f = ctx["frame0"].copy()
    f["observed_m"] = np.asarray(obs, float).reshape(-1)
    g = O.g0_e02(S.annotate(f), ctx["protocol"])
    lo, hi = ctx["protocol"]["coverage_bounds"]
    return {"pass": bool(g["e02_passed"]), "coverage": float(g["coverage"]),
            "ratio": float(g["ratio"]), "in_bounds": bool(lo <= g["coverage"] <= hi),
            "ratio_ok": bool(g["ratio"] <= float(ctx["protocol"]["max_rmse_ratio"]))}


def truth_obs(ctx: dict, t: dict, k: int, *, seed: int, stream: int = STREAM_TRUTH
              ) -> np.ndarray:
    e = O.simulate_ar1(ctx["sd"], ctx["phi"], O.N_WIN, rng_for(stream, t["i"], k, seed=seed))
    return ctx["zr"] + t["D"] + t["R"][k] + e


def null_obs(ctx: dict, B: np.ndarray, k: int, *, seed: int, stream: int = STREAM_NULL
             ) -> np.ndarray:
    e = O.simulate_ar1(ctx["sd"], ctx["phi"], O.N_WIN, rng_for(stream, 0, k, seed=seed))
    return 0.5 * (ctx["zr"] + B) + e


def selfcheck(ctx: dict, W: dict, B: np.ndarray, protocol_path: Path, tmp_root: Path, *,
              seed: int) -> list[dict]:
    """Gate E: on two right-kind worlds (truth R0) and one null world, this reading's rule ==
    twin.validation.evaluate (osse_fv.evaluate_selfcheck) == FV v2's own world_stats rate."""
    worlds = [("truth_R0", truth_obs(ctx, W["T"]["R0"], k, seed=seed, stream=STREAM_SELFCHECK))
              for k in range(N_SELFCHECK_TRUTH)]
    worlds.append(("null", null_obs(ctx, B, 0, seed=seed, stream=STREAM_SELFCHECK)))
    out = []
    for name, obs in worlds:
        mine = e02_rule(ctx, obs)
        ev = O.evaluate_selfcheck(ctx, obs, protocol_path, tmp_root)
        fv2 = O.world_stats(ctx, obs)
        if bool(fv2["e02_passed"]) != mine["pass"] or \
                abs(fv2["coverage_pooled"] - mine["coverage"]) > 1e-12 or \
                abs(fv2["e02_ratio"] - mine["ratio"]) > 1e-12:
            raise EngineeringFailure("E02's rule here disagrees with FV v2's world_stats")
        out.append({"world": name, "evaluate_identical": ev["identical"],
                    "fv2_world_stats_identical": True})
    return out


def summarise(records: list[dict]) -> dict:
    return {"passed": O._rate([r["pass"] for r in records]),
            "ratio_ok": float(np.mean([r["ratio_ok"] for r in records])),
            "coverage_in_bounds": float(np.mean([r["in_bounds"] for r in records])),
            "coverage_above_upper": float(np.mean([r["coverage"] > E02_RULE["coverage_bounds"][1]
                                                   for r in records])),
            "coverage_below_lower": float(np.mean([r["coverage"] < E02_RULE["coverage_bounds"][0]
                                                   for r in records])),
            "median_coverage": float(np.median([r["coverage"] for r in records])),
            "median_ratio": float(np.median([r["ratio"] for r in records]))}


def pooled(per: dict) -> dict:
    ps = [per[t]["passed"] for t in TAGS]
    se = math.sqrt(sum(x["mc_se"] ** 2 for x in ps)) / len(ps)
    p = float(np.mean([x["p"] for x in ps]))
    return {"p": p, "mc_se": se, "mc_half_width_90": Z_MC * se,
            "by_truth": {t: x["p"] for t, x in zip(TAGS, ps, strict=True)},
            "min_over_truths": float(min(x["p"] for x in ps)),
            "max_over_truths": float(max(x["p"] for x in ps))}


def _in_se(v: float, thr: float, se: float) -> float | None:
    return (v - thr) / se if se > 0 else None


def close(rep: dict, why: str) -> dict:
    rep.update(status=NOT_EVALUABLE, why=why, outcome=NOT_EVALUABLE,
               wording=wording_for(NOT_EVALUABLE, why=why),
               ledger_mapping={"outcome": LEDGER_MAPPING[NOT_EVALUABLE],
                               "grade_key": LEDGER_GRADE[NOT_EVALUABLE]})
    return rep


def _registry_guard(P: DecidPaths, prereg_dir=None, rows_dir=None) -> None:
    """A run on real paths (any under results/) uses the real registry and ledger rows only:
    a temporary registry holding a copy of the declaration must never unlock the real worlds."""
    from harness import prereg3

    if not any(_under_results(p) for p in P.all_paths()):
        return
    for given, real, name in ((prereg_dir, prereg3.PREREG_DIR, "prereg_dir"),
                              (rows_dir, prereg3.LEDGER_ROWS, "rows_dir")):
        if given is not None and Path(given).resolve() != Path(real).resolve():
            raise EngineeringFailure(f"{ID} on real paths uses the real registry only: "
                                     f"{name} {given} is not {real}")


def _score(P: DecidPaths, *, seed: int, plan: V2.Plan | None = None, k_null: int | None = None,
           smoke: bool = False, prereg_dir=None, rows_dir=None, log=print) -> dict:
    """Gate E, the worlds, P and P0, then the outcome. Returns the report (not written).

    A scored run loads the registration itself (load_registered) and uses only the registered
    seed, plan and K (the module's PLAN and K_NULL when not given); a smoke needs SMOKE_SEED and
    every path outside results/."""
    plan = PLAN if plan is None else plan
    k_null = K_NULL if k_null is None else k_null
    decl = dsha = None
    seed = check_seed(seed)
    if smoke:
        if seed != SMOKE_SEED or any(_under_results(p) for p in P.all_paths()):
            raise EngineeringFailure("a smoke runs on SMOKE_SEED and synthetic worlds outside "
                                     "results/ only")
    else:
        if seed != SEED or plan != PLAN or k_null != K_NULL:
            raise EngineeringFailure(f"{ID} scores only on its registered seed, plan and K")
        _registry_guard(P, prereg_dir, rows_dir)
        decl, dsha = load_registered(ID, prereg_dir, rows_dir)
    rep: dict = {"id": ID, "version": VERSION, "smoke": smoke, "created": O._now(),
                 "declaration_sha256": dsha, "seed": seed,
                 "k_truth": plan.k_truth, "k_null": k_null, "batch": plan.batch,
                 "streams": {"truth": STREAM_TRUTH, "null": STREAM_NULL,
                             "selfcheck": STREAM_SELFCHECK},
                 "truths": dict(O.TRUTHS), "rule": dict(E02_RULE), "model_only": True,
                 "fv2_report_only_known": FV2_REPORT_ONLY_P,
                 "mc_gate_note": V2.mc_note(len(TAGS), plan.k_truth)}
    # ---- gate E
    if decl is not None:
        for name, h in FV1_INPUTS.items():
            if O.sha256(O.rp(P.v1.inputs_dir) / name) != h:
                raise EngineeringFailure(f"{name} differs from FV v2's registered sha256")
    bad = world_problems(P, ((decl or {}).get("worlds") or {}).get("files"))
    if bad:
        raise EngineeringFailure("world files: " + "; ".join(bad))
    npz = {(t, b): O.rp(p) for t, ps in P.fv2.batches.items() for b, p in enumerate(ps)}
    att = V2.check_attempts(O.rp(P.fv2.attempts), npz)
    ctx = O.load_context(P.v1)
    rule = protocol_rule(ctx["protocol"])
    B = check_frame_order(ctx)
    W = load_worlds(P, ctx, plan)
    eng = {**W["eng"], "fv2_attempts": att, "rule": rule,
           "selfcheck": selfcheck(ctx, W, B, O.rp(P.v1.protocol), O.rp(P.report).parent,
                                  seed=seed)}
    rep["engineering"] = eng
    log("gate E passed (engineering)")
    # ---- worlds
    per = {}
    for tag in TAGS:
        t = W["T"][tag]
        recs = [e02_rule(ctx, truth_obs(ctx, t, k, seed=seed)) for k in range(plan.k_truth)]
        per[tag] = {**summarise(recs), "rms_departure_D_m": float(np.sqrt((t["D"] ** 2).mean()))}
    nrecs = [e02_rule(ctx, null_obs(ctx, B, k, seed=seed)) for k in range(k_null)]
    null = summarise(nrecs)
    gap = 0.5 * (ctx["zr"] - B)
    P1 = pooled(per)
    P0 = {"p": null["passed"]["p"], "mc_se": null["passed"]["mc_se"],
          "mc_half_width_90": Z_MC * null["passed"]["mc_se"]}
    rep.update({
        "truth_worlds": per,
        "null_world": {**null, "definition": "obs0 = (ZR + B) / 2 + e: the truth mean is "
                       "equidistant from E02's forecast and its baseline at every well-month",
                       "rms_half_gap_m": float(np.sqrt((gap ** 2).mean())),
                       "report_only_note": "by symmetry the ratio half passes with probability "
                                           "0.5 in expectation (ties pass)"},
        "P": P1, "P0": P0,
        "thresholds": {"P_measured": P_MEASURED, "P0_measured": P0_MEASURED,
                       "P_weak": P_WEAK, "P0_weak": P0_WEAK, "mc_half_width_90": MC_HALF_WIDTH},
        "distance_in_mc_se": {
            "P_minus_0.8": _in_se(P1["p"], P_MEASURED, P1["mc_se"]),
            "P_minus_0.5": _in_se(P1["p"], P_WEAK, P1["mc_se"]),
            "P0_minus_0.2": _in_se(P0["p"], P0_MEASURED, P0["mc_se"]),
            "P0_minus_0.5": _in_se(P0["p"], P0_WEAK, P0["mc_se"])},
        "mc_gate": {"max_half_width": MC_HALF_WIDTH,
                    "P": P1["mc_half_width_90"] <= MC_HALF_WIDTH,
                    "P0": P0["mc_half_width_90"] <= MC_HALF_WIDTH}})
    out = outcome_of(P1["p"], P0["p"], P1["mc_half_width_90"], P0["mc_half_width_90"])
    if out == NOT_EVALUABLE:
        return close(rep, "Monte Carlo half-width above 0.07 or a non-finite estimate")
    rep.update(status="scored", outcome=out, rule_applied=OUTCOME_RULE,
               wording=wording_for(out, p=P1["p"], p0=P0["p"]),
               ledger_mapping={"outcome": LEDGER_MAPPING[out], "grade_key": LEDGER_GRADE[out],
                               "condition": LEDGER_MAPPING["condition"],
                               "never": LEDGER_MAPPING["never"]})
    if decl is not None:
        rep["frozen_code"] = decl["frozen_code"]
    return rep


# ----------------------------------------------------------------------------- registration
def index_record(prereg_id: str = ID, prereg_dir=None) -> dict | None:
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    p = d / "INDEX.jsonl"
    if not p.is_file():
        return None
    rows = [r for r in prereg3.read_chained(p) if r["id"] == prereg_id]
    return rows[-1] if rows else None


def design() -> dict:
    return {"seed": SEED, "K_TRUTH": K_TRUTH, "K_NULL": K_NULL, "batch": PLAN.batch,
            "truth_order": list(TAGS),
            "streams": {"truth": STREAM_TRUTH, "null": STREAM_NULL,
                        "selfcheck": STREAM_SELFCHECK},
            "thresholds": {"P_measured": P_MEASURED, "P0_measured": P0_MEASURED,
                           "P_weak": P_WEAK, "P0_weak": P0_WEAK,
                           "mc_half_width_90": MC_HALF_WIDTH}}


def load_registered(prereg_id: str = ID, prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    """Refuse unless the declaration is in INDEX.jsonl, loads intact through
    prereg3.load_prereg at that entry's sha256, its frozen code is byte-identical and its
    design is this code's."""
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
    bad = reused_code_problems()
    if bad:
        raise EngineeringFailure(f"reused code changed: {bad}")
    changed = prereg3.changed_inputs(doc)
    if changed:
        raise EngineeringFailure(f"registered inputs changed: {changed}")
    d = doc.get("design") or {}
    bad = sorted(k for k, v in design().items() if d.get(k) != v)
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


def run(P: DecidPaths | None = None, *, prereg_dir=None, rows_dir=None, log=print) -> dict:
    """Score once. Refuses (nothing logged, nothing read) unless registered and intact; then
    logs 'start' and 'engineering_failure', 'error' or 'success' (with the report sha256) to
    attempts.jsonl beside the report, and writes report.json once (mode 'x'). No statistic
    is logged."""
    P = P or DecidPaths()
    _registry_guard(P, prereg_dir, rows_dir)          # real paths: the real registry only
    load_registered(ID, prereg_dir, rows_dir)         # before anything is logged or read
    outp = O.rp(P.report)
    ap = attempts_path(P.report)
    n = _attempt_no(ap)
    _log_attempt(ap, {"event": "start", "attempt": n, "report": str(P.report),
                      "code_sha256": O.sha256(Path(__file__))})
    try:
        if outp.exists():
            raise EngineeringFailure(f"{outp} exists: {ID} is scored once")
        rep = _score(P, seed=SEED, prereg_dir=prereg_dir, rows_dir=rows_dir, log=log)
        O._write_json_once(outp, rep)
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
    """True for any path that is, or may be, part of the real results tree: lexically under
    <repo>/results, resolved under it, or with a 'results' directory anywhere in its resolved
    form (results/ entries are symlinks into another checkout in build worktrees, so the
    resolved path alone misses them). Synthetic smokes live outside every such path."""
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


SYNTH_PRODUCTS = {"heads/L1_clean/v3": ["baseline", "design_time_evidence"],
                  "pump_electricity/preorigin/v3": ["design_time_evidence"]}
SYNTH_PLAN = V2.Plan(k_truth=12, batch=5)


def build_synthetic(out: Path, plan: V2.Plan = SYNTH_PLAN, nonlinear: float = 40.0
                    ) -> DecidPaths:
    """SYNTHETIC inputs only (no real data, nothing under results/): FV v2's synthetic set
    (osse_fv2.build_synthetic: predictions, protocol, baselines, wells, noise, mask, pool, ZR,
    ZR0, FV v1 truth forwards, arms and batch forwards) plus an FV v2-style attempts log, the two
    source declarations' frozen_code, rulings, E02's manifest and product MANIFESTs."""
    out = Path(out).resolve()
    if _under_results(out):
        raise EngineeringFailure("synthetic inputs never live under results/")
    fv2 = V2.build_synthetic(out / "fv2", plan, nonlinear=nonlinear)
    rows = [{"event": "forward", "truth": t, "batch": b, "rc": 0, "attempt": 1,
             "npz_sha256": O.sha256(Path(p))}
            for t, ps in fv2.batches.items() for b, p in enumerate(ps)]
    Path(fv2.attempts).write_text("".join(json.dumps(r) + "\n" for r in rows))
    P = DecidPaths(fv2=fv2, fv2_decl=str(out / "fv2_decl.json"),
                   fvs2_decl=str(out / "fvs2_decl.json"), rulings=str(out / "rulings.json"),
                   dataprep=str(out / "dataprep"), report=str(out / "report" / "report.json"))
    for path, did in ((P.fv2_decl, FV2_ID), (P.fvs2_decl, FVS2_ID)):
        Path(path).write_text(json.dumps({"id": did, "frozen_code": {
            p: h for p, h in REUSED.items() if REUSED_FROM[p] == did}}) + "\n")
    Path(P.rulings).write_text(json.dumps({"rulings": [
        {"id": RULING_ID, "decision": "synthetic stand-in"}]}) + "\n")
    for prod, uses in SYNTH_PRODUCTS.items():
        d = Path(P.dataprep) / prod
        d.mkdir(parents=True)
        (d / "MANIFEST.json").write_text(json.dumps({"allowed_uses": uses,
                                                     "forbidden_uses": []}) + "\n")
    Path(P.e02_manifest()).write_text(json.dumps({"synthetic": True}) + "\n")
    return P


def run_smoke(out: Path, k_null: int = 40) -> dict:
    """Synthetic end to end on SMOKE_SEED; writes out/smoke_report.json once. Numbers mean
    nothing about the twin."""
    out = Path(out).resolve()
    if _under_results(out):
        raise EngineeringFailure("the smoke never writes under results/")
    P = build_synthetic(out / "inputs_synthetic")
    rep = _score(P, seed=SMOKE_SEED, plan=SYNTH_PLAN, k_null=k_null, smoke=True,
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
    sub.add_parser("verify", help="registration, code and worlds intact (no statistic)")
    sub.add_parser("run", help="score once, after registration")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out))
            print(f"smoke {rep['status']} (synthetic)")
            return 0
        if a.cmd == "verify":
            _registry_guard(DecidPaths(), prereg_dir, rows_dir)
            doc, dsha = load_registered(ID, prereg_dir, rows_dir)
            bad = world_problems(DecidPaths(), (doc.get("worlds") or {}).get("files"))
            if bad:
                raise EngineeringFailure("; ".join(bad))
            print(f"{ID} registered (sha256 {dsha[:16]}); code, inputs and worlds intact")
            return 0
        rep = run(prereg_dir=prereg_dir, rows_dir=rows_dir)
        print(f"wrote {DecidPaths().report} (status {rep['status']}; outcome "
              f"{rep.get('outcome')})")
        return 0
    except EngineeringFailure as err:
        print(f"REFUSED / ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
