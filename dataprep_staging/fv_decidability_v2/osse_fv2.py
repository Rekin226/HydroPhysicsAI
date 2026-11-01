"""FV.decidability.v2_postfailure: the single post-failure exploratory follow-up of
FV.decidability.v1, merged for the forecast and bands parts (round-7 rank 12).

v1 (registered 2026-11-01, sha cf2adf0d) was scored NOT EVALUABLE: its engineering
shortcut, linear superposition of 58 one-month impulse arms, missed the full forwards by
0.2845 m against a 0.02 m tolerance. v1's attempt_policy allows exactly one follow-up,
labelled 'post-failure, exploratory', that discloses v1. This is it. User ruling
R7.fv_v2_page (2026-11-01, sixth set): its reading may reach the page, labelled
'post-failure, exploratory'.

What changes (the METHOD only): every synthetic truth world is its own full forward
scenario. World k of truth T carries exactly v1's multiplier draw
``draw_multipliers(pool, rng_for(STREAM_ANOM, T, k))`` as a ``--recorded-pumping`` arm, and

    obs(w, t) = ZR(w, t) + D_T(w, t) + [h_T(world k) - h_T(baseline)](w, t) + e(w, t)

on 2026-11..2027-10 at E02c's 158 wells. ZR, D_T (against v1's same-code reference ZR0),
the AR(1) noise e, its stream, the inputs (v1's export, verified and logged), the truths,
K_TRUTH 250, K_NULL 500, the seed, P_weak 0.5, the null tolerance 0.006 and the one-sided
readings are v1's, byte for byte. Gate S (superposition) has nothing to check and is gone.

Reading 1 (v1's, unchanged): E02c's strict level rule and its per-county coverage gates;
P(pass | right-ish) = equal-weight mean over the four truths; P <= 0.5 -> 'weak_by_design',
otherwise 'not_shown_weak'. No 'a fail is informative' reading exists.

Reading 2 (NEW, declared as new; forcing_year.py): forcing-year completeness of the frozen
combined sigma, on round-7 rank 11's replay worlds (FW.weather_replay.v1 engine, detrended
pumping). Scored separately (report_forcing_year.json) because those forwards come from
another declaration's runner.

Gates, in order, before any truth-world statistic:
  E  engineering: registration, frozen code, v1 inputs, arm files (each the declared draw),
     forward metadata, every batch baseline's 2012-22 hindcast identical (0.0 m) to v1's truth
     forward at the 158 wells, X5R1 against datum_gate_x5r1.npz within v1's 0.02 m (Q0
     report-only), ZR0 = v1's reference (sha pinned), v1's null tolerance recomputed
     identically, g0_e02 self-checked against twin.validation.evaluate;
  N  v1's null world: E02c level false pass <= the declared tolerance;
  MC each reading's 90 % Monte Carlo half-width (1.645 SE) <= 0.07. NON-BINDING at the frozen
     K: the largest possible half-width (every rate at 0.5) is 0.026 for reading 1 and FV.ext
     (4 truths x 250), 0.012 for FV.stat (9 cells x 500) and 0.016 for reading 2's RC (11 x
     250), so this check cannot fail and has no power; it is kept as a bug check, and each
     report states P - 0.5 in Monte Carlo SE units beside the reading.
An E failure writes nothing (engineering retry); N writes a not_evaluable report (final).

Subcommands (repository root, hydro env):
    osse_fv2.py arms                             after registration: world CSVs (CPU)
    osse_fv2.py specs --truth T --batch B        runner: --recorded-pumping specs
    osse_fv2.py verify-registration              runner preconditions
    osse_fv2.py score-worlds                     reading 1, once, after all 20 forwards
    osse_fv2.py replay-index --out P --clim N --seq N ...   (forcing_year.py; metadata only)
    osse_fv2.py score-forcing --replay-index P | --replay-unavailable REASON   reading 2, once
                                                 (REASON only on a pre-declared ground)
    osse_fv2.py smoke --out DIR                  synthetic end to end (anyone)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "fv_decidability"),
           str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osse_fv as O  # noqa: E402  (v1, frozen; bound by sha256)
import score_e02c as S  # noqa: E402

VERSION = "2.0.0"
ID = "FV.decidability.v2_postfailure"
V1_ID = O.ID
WP = "FV"
F = "dataprep_staging/fv_decidability_v2"
SELF = f"{F}/osse_fv2.py"
FY = f"{F}/forcing_year.py"
RUNNER = f"{F}/run_fv2.sh"
PREREG = f"{F}/prereg_fv2.py"
# v1's frozen set without v1's runner, plus this version's code
FROZEN = (SELF, FY, RUNNER, *[p for p in O.FROZEN if p != O.RUNNER])
EngineeringFailure = O.EngineeringFailure

# ----------------------------------------------------------------------------- frozen by v1
P_WEAK = O.P_WEAK                       # 0.5
K_TRUTH = O.K_TRUTH                     # 250
K_NULL = O.K_NULL                       # 500
SEED = O.SEED
REPRO_TOL_M = O.REPRO_TOL_M             # 0.02 (X5R1 against datum_gate_x5r1.npz, gating)
V1_NULL_TOLERANCE = 0.006000375877185671    # FV.decidability.v1 null_world.tolerance
V1_DECL_SHA = "cf2adf0d0c9b9fe5f70f5e6ee6cd1bc133964811c2b944eaf2cce344b4d3f61d"
# v1's forwards (attempts.jsonl of FV.decidability.v1, all at 5f3af18): ZR0 is reused as the
# same-code reference, the truth forwards' baselines are the hindcast-identity references
V1_FORWARD_SHA = {
    "ZR0": "a1833d5f861817223ebafb5a651444239c8877d21a4d4752e91dbbbd356172cf",
    "R0": "2b1363bec1581e76bf98f422a30a023cbc8d21f4dabba82256b5670d6d019fcf",
    "Q0": "1a6a1ef7d1c9d1c6fff21f0a2e26bd2fb8ad36e073ec36f03e96f6bde99d74b1",
    "R1c": "faff857e7ffae99d9cbcf5ed440546482c9d9d1276cf6fad5871a5426b87fdf1",
    "X5R1": "9d8e6311d5e1ece3dc54a4c40719a13e18b1a440cfe87770707649276c4c9566"}

# ----------------------------------------------------------------------------- new in v2
HINDCAST_TOL_M = 0.0                    # judge gate E: each truth's hindcast within 0.0 m
MC_HALF_WIDTH = 0.07                    # G-MC: 1.645 x Monte Carlo SE of a P
Z_MC = 1.6448536269514722


def mc_max_half_width(n_groups: int, k: int) -> float:
    """The largest possible 90 % Monte Carlo half-width of an equal-weight mean of
    ``n_groups`` pass rates over ``k`` worlds each (every rate at 0.5)."""
    return Z_MC * math.sqrt(n_groups * 0.25 / k) / n_groups


def mc_note(n_groups: int, k: int) -> str:
    return (f"non-binding at the frozen K ({n_groups} x {k}): the largest possible 90 % "
            f"half-width is {mc_max_half_width(n_groups, k):.3f} < {MC_HALF_WIDTH}, so the "
            "check cannot fail and carries no power; it is a bug check only, no page branch "
            "depends on it, and the report states P - 0.5 in Monte Carlo SE units")


BATCH = 50                              # worlds per forward (+ baseline = 51 scenarios)

# ----------------------------------------------------------------------------- run layout
WT_NAME = "HydroPhysicsAI-fv2run"       # fresh clean detached worktree at v1's commit
WT_HEAD = O.WT_HEAD                     # 5f3af18: same code as v1's ZR0 and truth forwards
FWD_FLAGS = O.FWD_FLAGS                 # E02's recipe, one member, horizon 58 (as v1)
RUNS = "results/twin_runs/fv2"
PY_SH = f"{RUNS}/py.sh"
ARMS_DIR = f"{RUNS}/arms"
ATTEMPTS = f"{RUNS}/attempts.jsonl"
REPORT_DIR = f"results/dataprep/reports/{ID}"
REPORT_WORLDS = f"{REPORT_DIR}/report_worlds.json"
REPORT_FORCING = f"{REPORT_DIR}/report_forcing_year.json"
DECL = f"results/dataprep/preregistration/{ID}.json"
V1_DECL = f"results/dataprep/preregistration/{V1_ID}.json"


def batch_npz(tag: str, b: int) -> str:
    return f"results/twin_forward/fv2_truth_{tag}_b{b}.npz"


def v1_truth_npz(tag: str) -> str:
    return O.truth_npz(tag)


def world_name(k: int) -> str:
    return f"w{k:03d}"


@dataclass(frozen=True)
class Plan:
    """World layout: K worlds per truth in forwards of ``batch`` worlds plus a baseline."""
    k_truth: int = K_TRUTH
    batch: int = BATCH

    @property
    def n_batch(self) -> int:
        return math.ceil(self.k_truth / self.batch)

    def worlds(self, b: int) -> range:
        return range(b * self.batch, min((b + 1) * self.batch, self.k_truth))

    def names(self, b: int) -> list[str]:
        return ["baseline"] + [world_name(k) for k in self.worlds(b)]


PLAN = Plan()


# ----------------------------------------------------------------------------- wording
_LBL_EN = ("post-failure exploratory follow-up (FV.decidability.v2; v1 was not evaluable "
           "because its superposition shortcut failed)")
_LBL_ZH = "失敗後探索性後續研究（FV.decidability.v2；v1 因疊加近似失敗而無法評估）"
WORDING = {
    "level": {
        "weak_by_design": {
            "en": f"Synthetic decidability study, {_LBL_EN}: with every synthetic world run as "
                  "a full forward, a model of the right kind passes E02c's level rule with "
                  "probability {p:.2f} (optimistic: rain at climatology, noise independent "
                  "between wells; not a strict bound), so the level test is weak by design. A "
                  "fail would not refute the forecast, and the baseline projection's head "
                  "part stays narrowed (N) whatever the 2028 level outcome.",
            "zh": f"合成可判定性研究，{_LBL_ZH}：每個合成情境皆以完整正演模擬計算，正確類型的模型"
                  "通過 E02c 水位規則的機率為 {p:.2f}（偏樂觀：降雨取氣候值，各井雜訊彼此獨立；"
                  "並非嚴格界限），因此該水位檢驗在設計上即屬弱檢驗。未通過也不足以否定預測，"
                  "且不論 2028 年水位結果如何，基準推估的水位部分都維持限縮措辭（N）。"},
        "not_shown_weak": {
            "en": f"Synthetic decidability study, {_LBL_EN}: with every synthetic world run as "
                  "a full forward, a model of the right kind passes E02c's level rule with "
                  "probability {p:.2f} (optimistic: rain at climatology, noise independent "
                  "between wells; not a strict bound); the study does not show the level test "
                  "weak and adds no other reading.",
            "zh": f"合成可判定性研究，{_LBL_ZH}：每個合成情境皆以完整正演模擬計算，正確類型的模型"
                  "通過 E02c 水位規則的機率為 {p:.2f}（偏樂觀：降雨取氣候值，各井雜訊彼此獨立；"
                  "並非嚴格界限）；本研究未顯示該水位檢驗為弱檢驗，也不提供其他解讀。"}},
    "coverage": {
        "weak_by_design": {
            "en": "{county}: in the synthetic study (" + _LBL_EN + ") a model of the right "
                  "kind passes E02c's coverage gate with probability {p:.2f} (not a bound: "
                  "rain variability and correlated noise could move coverage either way), so "
                  "the coverage test is weak by design there; a fail would not refute the "
                  "interval.",
            "zh": "{county}：在合成研究中（" + _LBL_ZH + "），正確類型的模型通過 E02c 涵蓋率檢定"
                  "的機率為 {p:.2f}（並非界限：降雨變異與井間相關雜訊可使涵蓋率往任一方向變動），"
                  "因此該處的涵蓋率檢驗在設計上即屬弱檢驗；未通過也不足以否定此區間。"},
        "not_shown_weak": {
            "en": "{county}: in the synthetic study (" + _LBL_EN + ") a model of the right "
                  "kind passes E02c's coverage gate with probability {p:.2f} (not a bound: "
                  "rain variability and correlated noise could move coverage either way); not "
                  "shown weak, no other reading.",
            "zh": "{county}：在合成研究中（" + _LBL_ZH + "），正確類型的模型通過 E02c 涵蓋率檢定"
                  "的機率為 {p:.2f}（並非界限：降雨變異與井間相關雜訊可使涵蓋率往任一方向變動）；"
                  "未顯示為弱檢驗，不提供其他解讀。"}},
    "not_evaluable": {
        "en": "Synthetic decidability study, post-failure exploratory follow-up "
              "(FV.decidability.v2) not evaluable ({why}); E02 and E02c gain no interpretation "
              "line from it, and no further follow-up exists.",
        "zh": "合成可判定性研究之失敗後探索性後續研究（FV.decidability.v2）無法評估（{why}）；"
              "E02 與 E02c 不因此增加解讀說明，且不再有後續研究。"},
}


# ----------------------------------------------------------------------------- arms
def world_series(pool: np.ndarray, tag: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """v1's draw k of truth ``tag`` (stream STREAM_ANOM), unchanged."""
    return O.draw_multipliers(pool, O.rng_for(O.STREAM_ANOM, O.TAGS.index(tag), k))


def write_arms(out: Path, pool: np.ndarray, plan: Plan = PLAN) -> dict:
    """One multiplier CSV per (truth, world), written once, with arms.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    files: dict = {}
    for tag in O.TAGS:
        (out / tag).mkdir()
        files[tag] = {}
        for k in range(plan.k_truth):
            a, pick = world_series(pool, tag, k)
            name = world_name(k)
            p = out / tag / f"{name}.csv"
            O.write_series_csv(p, a)
            files[tag][name] = {"file": f"{tag}/{name}.csv", "start": f"{O.ANOM_MONTHS[0]:%Y-%m}",
                                "sha256": O.sha256(p), "pool_rows_drawn": [int(x) for x in pick]}
    rec = {"id": ID, "created": O._now(), "county": O.COUNTY_SCOPE, "k_truth": plan.k_truth,
           "batch": plan.batch, "files": files, "osse_fv2_sha256": O.sha256(Path(__file__))}
    O._write_json_once(out / "arms.json", rec)
    return rec


def recorded_specs(tag: str, b: int, arms_dir: str = ARMS_DIR, plan: Plan = PLAN) -> list[str]:
    """``NAME=PATH@START#yunlin`` for the worlds of batch ``b`` of truth ``tag``, in the
    scenario order of :meth:`Plan.names`."""
    meta = json.loads((O.rp(arms_dir) / "arms.json").read_text(encoding="utf-8"))
    out = []
    for name in plan.names(b)[1:]:
        r = meta["files"][tag][name]
        out.append(f"{name}={arms_dir}/{r['file']}@{r['start']}#{O.COUNTY_SCOPE}")
    return out


def check_arms(arms_dir, pool: np.ndarray, plan: Plan = PLAN) -> dict:
    """Every world file holds exactly its declared draw and its arms.json sha256."""
    d = O.rp(arms_dir)
    arms = json.loads((d / "arms.json").read_text(encoding="utf-8"))
    for tag in O.TAGS:
        for k in range(plan.k_truth):
            r = arms["files"][tag][world_name(k)]
            p = d / r["file"]
            if O.sha256(p) != r["sha256"]:
                raise EngineeringFailure(f"{r['file']}: sha256 differs from arms.json")
            a, _ = world_series(pool, tag, k)
            if not np.allclose(O.read_series_csv(p), a, rtol=1e-12, atol=0.0):
                raise EngineeringFailure(f"{r['file']} is not draw {k} of truth {tag}")
    return arms


# ----------------------------------------------------------------------------- forwards
def check_batch_meta(tag: str, b: int, meta: dict, heads: np.ndarray, arms: dict,
                     plan: Plan = PLAN) -> dict:
    """Engineering checks of one truth batch forward before any of its values is used."""
    names = [str(x) for x in meta["scenario_names"]]
    want = plan.names(b)
    probs = []
    if names != want:
        probs.append(f"scenario names differ from the declared batch ({len(names)} vs "
                     f"{len(want)})")
    dates = pd.DatetimeIndex(pd.to_datetime(meta["dates"])).to_period("M").to_timestamp()
    if len(dates) != O.T_RUN or dates[0] != pd.Timestamp(f"{O.AXIS_START}-01"):
        probs.append(f"dates are not {O.AXIS_START}..2027-10 ({len(dates)} months)")
    if int(meta["origin"]) != O.ORIGIN_IDX or int(meta["horizon"]) != O.HORIZON:
        probs.append("origin or horizon differs")
    labels = [str(x) for x in meta.get("member_labels", [])]
    if int(meta["n_members"]) != 1 or labels != [dict(O.TRUTHS)[tag]]:
        probs.append(f"members {int(meta['n_members'])} {labels}, expected one "
                     f"{dict(O.TRUTHS)[tag]}")
    opts = json.loads(str(meta["forward_options"]))
    rec = opts.get("recorded_pumping") or {}
    if set(rec) != set(want[1:]):
        probs.append("recorded-pumping arms differ from the batch's worlds")
    for name in want[1:]:
        r, a = rec.get(name), arms["files"][tag].get(name)
        if r is None or a is None or r.get("sha256") != a["sha256"] or \
                r.get("county") != O.COUNTY_SCOPE or r.get("start") != a["start"]:
            probs.append(f"world {name}: recorded series does not match arms.json")
            break
    if float(meta["gain"]) != 1.0 or float(meta["hindcast_gain"]) != 0.0:
        probs.append("gain / hindcast gain differ from E02's recipe")
    if opts.get("update_method") != "innovation" or opts.get("column_heads") != "free":
        probs.append("forward options differ from E02's recipe")
    if not np.isfinite(heads).all():
        probs.append("non-finite heads at a well cell")
    if probs:
        raise EngineeringFailure(f"truth {tag} batch {b}: " + "; ".join(probs))
    return {"batch": b, "scenarios": len(names),
            "gate_git": json.loads(str(meta["gate"])).get("git_commit")}


def load_truth(paths: list, tag: str, arms: dict, cells: np.ndarray, layers: np.ndarray,
               ref: np.ndarray, plan: Plan = PLAN, keep_full=()) -> dict:
    """All batches of truth ``tag`` at (cells, layers): the batch baseline (first batch), the
    window responses R (K, W, 12) = world - its own batch baseline, and engineering records.
    ``ref`` (S_v1, W, T) is v1's truth forward at the same wells: every batch baseline's
    2012-22 hindcast must equal ref[0]'s within HINDCAST_TOL_M (0.0 m). ``keep_full``: world
    indices whose full series are kept (report-only comparison with v1's check forwards)."""
    if len(paths) != plan.n_batch:
        raise EngineeringFailure(f"truth {tag}: {len(paths)} batch files, expected "
                                 f"{plan.n_batch}")
    W = len(cells)
    R = np.full((plan.k_truth, W, O.N_WIN), np.nan)
    full, base0, eng = {}, None, []
    for b, p in enumerate(paths):
        meta, h = O.well_heads(O.rp(p), cells, layers)
        rec = check_batch_meta(tag, b, meta, h, arms, plan)
        hind = float(np.max(np.abs(h[0][:, :O.T_FIT] - ref[0][:, :O.T_FIT])))
        if hind > HINDCAST_TOL_M:
            raise EngineeringFailure(f"truth {tag} batch {b}: baseline hindcast differs from "
                                     f"v1's truth forward by {hind:.3g} m (> {HINDCAST_TOL_M})")
        if base0 is None:
            base0 = h[0][:, :O.T_RUN].copy()
        rec.update({"hindcast_max_abs_vs_v1_m": hind,
                    "report_only_baseline_projection_max_abs_vs_batch0_m":
                        float(np.max(np.abs(h[0][:, :O.T_RUN] - base0))),
                    "report_only_baseline_projection_max_abs_vs_v1_m":
                        float(np.max(np.abs(h[0][:, O.T_FIT:O.T_RUN]
                                            - ref[0][:, O.T_FIT:O.T_RUN])))})
        for j, k in enumerate(plan.worlds(b)):
            R[k] = h[1 + j][:, O.WIN[0]:O.WIN[1]] - h[0][:, O.WIN[0]:O.WIN[1]]
            if k in keep_full:
                full[k] = h[1 + j][:, :O.T_RUN].copy()
        eng.append(rec)
        del h
    if not np.isfinite(R).all():
        raise EngineeringFailure(f"truth {tag}: a world response is missing")
    return {"base": base0, "R": R, "full": full, "batches": eng}


def v1_reference(path, cells, layers, tag: str) -> tuple[np.ndarray, dict]:
    """v1's truth forward at the wells: (S_v1, W, T) heads and its check-scenario index."""
    meta, h = O.well_heads(O.rp(path), cells, layers)
    names = [str(x) for x in meta["scenario_names"]]
    if names[:1] != ["baseline"]:
        raise EngineeringFailure(f"{path}: first scenario is not the baseline")
    return h, {n: i for i, n in enumerate(names) if n.startswith("chk_")}


def v1_check_agreement(tag: str, ref: np.ndarray, chk_idx: dict, full: dict) -> list[dict]:
    """REPORT-ONLY: v1's full check forwards (the same multiplier draws) against v2's worlds
    over 2023-01..2027-10. No threshold; it enters no gate and no world."""
    out = []
    for j, t, d in O.check_plan():
        if t != tag or d not in full or f"chk_{j}" not in chk_idx:
            continue
        diff = np.abs(full[d][:, O.T_FIT:O.T_RUN] - ref[chk_idx[f"chk_{j}"]][:, O.T_FIT:O.T_RUN])
        out.append({"v1_check": j, "world": d, "max_abs_projection_m": float(diff.max()),
                    "report_only": True})
    return out


def check_attempts(att: Path, npz: dict) -> dict:
    """Each batch npz must be the output of its last successful forward attempt
    (attempts.jsonl rows: event forward, truth, batch, rc, npz_sha256)."""
    if not att.exists():
        raise EngineeringFailure(f"{att} missing")
    recs = [json.loads(s) for s in att.read_text().splitlines() if s.strip()]
    out = {}
    for (tag, b), p in npz.items():
        fw = [r for r in recs if r.get("event") == "forward" and r.get("truth") == tag
              and str(r.get("batch")) == str(b)]
        if not fw or int(fw[-1].get("rc", 1)) != 0:
            raise EngineeringFailure(f"no successful forward attempt for {tag} batch {b}")
        if fw[-1].get("npz_sha256") != O.sha256(p):
            raise EngineeringFailure(f"{p} is not the output of {tag} batch {b}'s last attempt")
        out[f"{tag}_b{b}"] = {"attempt": fw[-1].get("attempt"), "npz_sha256": fw[-1]["npz_sha256"]}
    return out


# ----------------------------------------------------------------------------- paths
@dataclass
class Paths2:
    v1: O.Paths = field(default_factory=O.Paths)
    arms_dir: str = ARMS_DIR
    attempts: str = ATTEMPTS
    batches: dict = field(default_factory=lambda: {
        t: [batch_npz(t, b) for b in range(PLAN.n_batch)] for t in O.TAGS})
    v1_truths: dict = field(default_factory=lambda: {t: v1_truth_npz(t) for t in O.TAGS})
    report_worlds: str = REPORT_WORLDS
    report_forcing: str = REPORT_FORCING


# ----------------------------------------------------------------------------- statistics
def _pooled(per: dict, get) -> dict:
    ps = [get(per[t]) for t in O.TAGS]
    se = math.sqrt(sum(x[1] ** 2 for x in ps)) / len(ps)
    p = float(np.mean([x[0] for x in ps]))
    return {"p": p, "mc_se": se, "mc_half_width_90": Z_MC * se,
            "p_minus_threshold_in_se": (p - P_WEAK) / se if se > 0 else None,
            "by_truth": {t: x[0] for t, x in zip(O.TAGS, ps, strict=True)},
            "min_over_truths": float(min(x[0] for x in ps))}


def mc_ok(pooled: dict) -> bool:
    return pooled["mc_half_width_90"] <= MC_HALF_WIDTH


def reading(pooled: dict) -> str | None:
    """v1's one-sided rule on the point estimate, if the Monte Carlo precision is met."""
    return O.reading(pooled["p"]) if mc_ok(pooled) else None


def score_worlds(P: Paths2, *, plan: Plan = PLAN, smoke: bool = False, k_null: int = K_NULL,
                 decl: dict | None = None, decl_sha: str | None = None, log=print) -> dict:
    """Gates E and N, then the truth worlds and reading 1. Returns the report (not
    written). Raises EngineeringFailure (nothing written) on E."""
    rep: dict = {"id": ID, "version": VERSION, "part": "reading 1 (v1's readings, full "
                 "forwards)", "smoke": smoke, "created": O._now(),
                 "declaration_sha256": decl_sha, "k_truth": plan.k_truth, "k_null": k_null,
                 "batch": plan.batch, "seed": SEED, "truths": dict(O.TRUTHS),
                 "v1": {"id": V1_ID, "status": "not_evaluable", "why": "superposition failed",
                        "declaration_sha256": V1_DECL_SHA}}
    ctx = O.load_context(P.v1)
    if decl is not None:
        for name, h in (decl.get("inputs_frozen") or {}).items():
            if O.sha256(O.rp(P.v1.inputs_dir) / name) != h:
                raise EngineeringFailure(f"{name} differs from the declared sha256")
        for t, p in {"ZR0": P.v1.zr0, **P.v1_truths}.items():
            if O.sha256(O.rp(p)) != (decl.get("v1_forwards") or {}).get(t):
                raise EngineeringFailure(f"v1 forward {t} differs from the declared sha256")
    arms = check_arms(P.arms_dir, ctx["pool"], plan)
    eng: dict = {"truths": {}, "repro": {}, "v1_check_agreement_report_only": {}}
    zr_meta, zr_heads = O.well_heads(O.rp(P.v1.zr_forward), ctx["cells"], ctx["layers"])
    h_zr = zr_heads[[str(x) for x in zr_meta["scenario_names"]].index("baseline")]
    del zr_heads
    zr0_meta, zr0_heads = O.well_heads(O.rp(P.v1.zr0), ctx["cells"], ctx["layers"])
    eng["zr0_meta"] = O.check_zr0_meta(zr0_meta, zr0_heads, zr_meta)
    h_zr0 = zr0_heads[0]
    del zr0_heads
    T = {}
    for i, tag in enumerate(O.TAGS):
        ref, chk = v1_reference(P.v1_truths[tag], ctx["cells"], ctx["layers"], tag)
        keep = {d for _, t, d in O.check_plan() if t == tag}
        lt = load_truth(P.batches[tag], tag, arms, ctx["cells"], ctx["layers"], ref, plan,
                        keep_full=keep)
        eng["truths"][tag] = lt["batches"]
        eng["v1_check_agreement_report_only"][tag] = v1_check_agreement(tag, ref, chk,
                                                                        lt["full"])
        T[tag] = {"i": i, "R": lt["R"], "D": O.departure(lt["base"], h_zr0, ctx["mask"])}
        if tag in P.v1.repro:
            path, gating = P.v1.repro[tag]
            _, rf = O.well_heads(O.rp(path), ctx["cells"], ctx["layers"])
            err = float(np.max(np.abs(lt["base"][:, :O.T_FIT] - rf[0][:, :O.T_FIT])))
            eng["repro"][tag] = {"file": str(path), "max_abs_hindcast_m": err,
                                 "tolerance_m": REPRO_TOL_M, "gating": gating,
                                 "pass": err <= REPRO_TOL_M}
            if gating and err > REPRO_TOL_M:
                raise EngineeringFailure(f"{tag} hindcast does not reproduce {path} "
                                         f"(max {err:.4f} m > {REPRO_TOL_M} m)")
        del ref, lt
    tol = O.null_tolerance(ctx["zr"] - ctx["cl"], ctx["sd"], ctx["phi"], k_null)
    if decl is not None:
        dtol = (decl.get("null_world") or {}).get("tolerance")
        if dtol is None or abs(dtol - tol["tolerance"]) > 1e-12:
            raise EngineeringFailure(f"null tolerance {tol['tolerance']} differs from the "
                                     f"declared {dtol}")
    eng["evaluate_selfcheck"] = [
        O.evaluate_selfcheck(ctx, ctx["cl"] + O.simulate_ar1(
            ctx["sd"], ctx["phi"], O.N_WIN, O.rng_for(O.STREAM_SELFCHECK, 0, k)),
            O.rp(P.v1.protocol), O.rp(P.report_worlds).parent)
        for k in range(O.N_SELFCHECK)]
    rep["engineering"] = eng
    log("gate E passed (engineering)")
    rep["zr0_vs_registered_report_only"] = O.zr0_diagnostic(h_zr0, h_zr, ctx["mask"])
    del h_zr, h_zr0

    # ---- gate N: v1's null world, unchanged
    nulls = [O.world_stats(ctx, ctx["cl"] + O.simulate_ar1(
        ctx["sd"], ctx["phi"], O.N_WIN, O.rng_for(O.STREAM_NULL, 0, k))) for k in range(k_null)]
    fp = O._rate([w["level"] == "level_forecast_supported" for w in nulls])
    rep["null_world"] = {
        "tolerance": tol, "level_false_pass": fp,
        "report_only": {
            "beat_cl": O._rate([w.get("beat_cl", False) for w in nulls]),
            "e02_own_passed": O._rate([w["e02_passed"] for w in nulls]),
            "coverage_supported": {c: O._rate([w["cov"][c]["verdict"] == "coverage_supported"
                                               for w in nulls]) for c in S.COUNTIES_COV}},
        "pass": fp["p"] <= tol["tolerance"]}
    if fp["p"] > tol["tolerance"]:
        log(f"gate N FAILED: false pass {fp['p']:.3f} > {tol['tolerance']:.3f}")
        return close(rep, "null-world false-pass rate above the declared tolerance")
    log(f"gate N passed (false pass {fp['p']:.3f} <= {tol['tolerance']:.3f})")

    # ---- truth worlds (full forwards)
    per: dict = {}
    for tag in O.TAGS:
        t = T[tag]
        ws, rms_r = [], []
        for k in range(plan.k_truth):
            r = t["R"][k]
            e = O.simulate_ar1(ctx["sd"], ctx["phi"], O.N_WIN, O.rng_for(O.STREAM_NOISE,
                                                                          t["i"], k))
            ws.append(O.world_stats(ctx, ctx["zr"] + t["D"] + r + e))
            rms_r.append(float(np.sqrt((r ** 2).mean())))
        per[tag] = {
            "level_supported": O._rate([w["level"] == "level_forecast_supported" for w in ws]),
            "beat_cl": O._rate([w.get("beat_cl", False) for w in ws]),
            "beat_clt": O._rate([w.get("beat_clt", False) for w in ws]),
            "coverage_supported": {c: O._rate([w["cov"][c]["verdict"] == "coverage_supported"
                                               for w in ws]) for c in S.COUNTIES_COV},
            "g0_pass": O._rate([bool(w["g0"]) for w in ws]),
            "e02_own_passed": O._rate([w["e02_passed"] for w in ws]),
            "median_rmse_m": {m: float(np.median([w["rmse"][m] for w in ws if "rmse" in w]))
                              for m in ("ZR", "CL", "CLT")},
            "rms_departure_D_m": float(np.sqrt((t["D"] ** 2).mean())),
            "mean_rms_pumping_response_m": float(np.mean(rms_r))}
        log(f"truth {tag}: P(level) {per[tag]['level_supported']['p']:.3f}")
    rep["truth_worlds"] = per
    rep["noise_report_only"] = {"median_sd_m": float(np.median(ctx["sd"])),
                                "median_phi": float(np.median(ctx["phi"]))}
    P_level = _pooled(per, lambda r: (r["level_supported"]["p"], r["level_supported"]["mc_se"]))
    P_cov = {c: _pooled(per, lambda r, c=c: (r["coverage_supported"][c]["p"],
                                             r["coverage_supported"][c]["mc_se"]))
             for c in S.COUNTIES_COV}
    P_e02 = _pooled(per, lambda r: (r["e02_own_passed"]["p"], r["e02_own_passed"]["mc_se"]))
    rep["P"] = {"level": P_level, "coverage": P_cov, "e02_own_report_only": P_e02,
                "definition": "equal-weight mean over the four truths of each truth's pass "
                              f"rate over {plan.k_truth} full-forward worlds; the reading uses "
                              "the point estimate"}
    rep["mc_gate"] = {"max_half_width": MC_HALF_WIDTH,
                      "level": mc_ok(P_level), **{c: mc_ok(P_cov[c]) for c in S.COUNTIES_COV}}
    rl = reading(P_level)
    rc = {c: reading(P_cov[c]) for c in S.COUNTIES_COV}
    why_mc = "Monte Carlo precision not met"
    rep["readings"] = {
        "threshold": P_WEAK, "label": "post-failure, exploratory",
        "rule": "P <= 0.5 -> weak_by_design; otherwise not_shown_weak (one-sided; applied "
                "once, as v1)",
        "level": rl, "coverage": rc,
        "wording": {
            "level": ({k: v.format(p=P_level["p"]) for k, v in WORDING["level"][rl].items()}
                      if rl else {k: v.format(why=why_mc)
                                  for k, v in WORDING["not_evaluable"].items()}),
            "coverage": {c: ({k: v.format(p=P_cov[c]["p"], county=c)
                              for k, v in WORDING["coverage"][rc[c]].items()} if rc[c] else None)
                         for c in S.COUNTIES_COV}}}
    rep["status"] = "scored"
    return rep


def close(rep: dict, why: str) -> dict:
    rep["status"] = "not_evaluable"
    rep["why"] = why
    rep["readings"] = {"level": None, "coverage": None, "label": "post-failure, exploratory",
                       "wording": {k: v.format(why=why)
                                   for k, v in WORDING["not_evaluable"].items()}}
    return rep


# ----------------------------------------------------------------------------- registration
def load_registered(prereg_id: str = ID, frozen=FROZEN, prereg_dir=None, rows_dir=None
                    ) -> tuple[dict, str]:
    from harness import prereg3

    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    try:
        doc, sha = prereg3.load_prereg(prereg_id, **kw)
    except Exception as err:      # noqa: BLE001 - any failure is a refusal
        raise EngineeringFailure(f"{prereg_id} is not registered or not intact: {err}") from err
    bad = [p for p, h in (doc.get("frozen_code") or {}).items() if O.sha256(O.rp(p)) != h]
    if set(doc.get("frozen_code") or {}) != set(frozen):
        bad.append("frozen_code does not list exactly the declared files")
    if bad:
        raise EngineeringFailure(f"frozen code changed since registration: {bad}")
    changed = prereg3.changed_inputs(doc)
    if changed:
        raise EngineeringFailure(f"registered inputs changed: {changed}")
    return doc, sha


# ----------------------------------------------------------------------------- synthetic
def _batch_world(base: np.ndarray, mult: np.ndarray, gain: np.ndarray, tau: np.ndarray,
                 nonlinear: float) -> np.ndarray:
    """A synthetic full forward: exponential impulse responses of every multiplier month
    plus a quadratic term, so a superposition shortcut would fail but full forwards do not
    care."""
    tr = np.arange(O.T_RUN, dtype=float)
    r = np.zeros_like(base)
    for m in range(O.HORIZON):
        lag = tr[None] - (O.T_FIT + m)
        r += (mult[m] - 1.0) * np.where(lag >= 0, gain[:, None]
                                        * np.exp(-np.maximum(lag, 0) / tau[:, None]), 0.0)
    return base + r + nonlinear * r ** 2


def build_synthetic(out: Path, plan: Plan, nonlinear: float = 40.0, seed: int = 11) -> Paths2:
    """SYNTHETIC inputs with the real shapes (no real data): v1's synthetic set (predictions,
    baselines, wells, noise, mask, pool, ZR, ZR0, v1 truth forwards) plus v2's arm files and
    one npz per (truth, batch) whose baselines equal v1's synthetic truth baselines."""
    out = Path(out)
    v1 = O.build_synthetic(out / "v1", nonlinear=0.01)
    pool = O.pool_matrix(pd.read_csv(Path(v1.inputs_dir) / "anomaly_pool.csv"))
    arms = write_arms(out / "arms", pool, plan)
    rng = np.random.default_rng(seed)
    batches = {}
    for tag in O.TAGS:
        with np.load(v1.truths[tag], allow_pickle=False) as z:
            H1 = np.asarray(z["heads_mean"], float)
            label = str(z["member_labels"][0])
        base = H1[0]                                        # (4, C, T)
        L, C, Tn = base.shape
        flat = base.reshape(L * C, Tn)
        gain = rng.uniform(-0.3, -0.05, L * C)
        tau = rng.uniform(3.0, 18.0, L * C)
        batches[tag] = []
        for b in range(plan.n_batch):
            names = plan.names(b)
            H = np.zeros((len(names), L, C, Tn))
            H[0] = base
            rec = {}
            for j, name in enumerate(names[1:]):
                a = arms["files"][tag][name]
                rec[name] = {"sha256": a["sha256"], "county": O.COUNTY_SCOPE,
                             "start": a["start"]}
                mult = O.read_series_csv(out / "arms" / a["file"])
                H[1 + j] = _batch_world(flat, mult, gain, tau, nonlinear).reshape(L, C, Tn)
            p = out / f"fv2_truth_{tag}_b{b}.npz"
            O._save_npz(p, H, names, label, rec)
            batches[tag].append(str(p))
    return Paths2(v1=v1, arms_dir=str(out / "arms"), attempts=str(out / "attempts.jsonl"),
                  batches=batches, v1_truths=dict(v1.truths),
                  report_worlds=str(out / "report_worlds.json"),
                  report_forcing=str(out / "report_forcing_year.json"))


def run_smoke(out: Path, k_truth: int = 6, batch: int = 4, k_null: int = 12,
              nonlinear: float = 40.0) -> dict:
    """Synthetic end to end (no registry, no real data): writes out/smoke_worlds.json once."""
    out = Path(out)
    plan = Plan(k_truth=k_truth, batch=batch)
    P = build_synthetic(out / "inputs_synthetic", plan, nonlinear=nonlinear)
    rep = score_worlds(P, plan=plan, smoke=True, k_null=k_null)
    rep["note"] = "SMOKE: synthetic inputs only; numbers mean nothing about the twin"
    O._write_json_once(out / "smoke_worlds.json", rep)
    return rep


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    import forcing_year as FYM

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("arms", help="after registration: world CSVs")
    sp = sub.add_parser("specs", help="runner: --recorded-pumping specs of one batch")
    sp.add_argument("--truth", required=True, choices=O.TAGS)
    sp.add_argument("--batch", required=True, type=int)
    sub.add_parser("verify-registration")
    sw = sub.add_parser("score-worlds", help="reading 1, once")
    sw.add_argument("--out", default=REPORT_WORLDS)
    ri = sub.add_parser("replay-index", help="lead: index rank 11's replay forwards "
                                             "(metadata only)")
    FYM.add_index_args(ri)
    sf = sub.add_parser("score-forcing", help="reading 2, once")
    g = sf.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay-index")
    g.add_argument("--replay-unavailable", metavar="REASON")
    sf.add_argument("--out", default=REPORT_FORCING)
    sm = sub.add_parser("smoke", help="synthetic end to end")
    sm.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out))
            frep = FYM.run_smoke(Path(a.out) / "forcing")
            print(json.dumps({"worlds": {"status": rep["status"], "readings": rep["readings"]},
                              "forcing": {"status": frep["status"],
                                          "readings": frep.get("readings")}},
                             indent=1, ensure_ascii=False, default=O._jsonable))
            return 0
        if a.cmd == "specs":
            print("\n".join(recorded_specs(a.truth, a.batch)))
            return 0
        if a.cmd == "replay-index":
            rec = FYM.build_index_cli(a)
            print(json.dumps({k: v for k, v in rec.items() if k != "sequences"}, indent=1))
            return 0
        doc, dsha = load_registered()
        if a.cmd == "verify-registration":
            print(f"{ID} registered (sha256 {dsha[:16]}); frozen code and inputs intact")
            return 0
        if a.cmd == "arms":
            if O.rp(ARMS_DIR).exists():
                raise EngineeringFailure(f"{ARMS_DIR} exists (archive it first)")
            pool = O.pool_matrix(pd.read_csv(O.rp(O.INPUTS_DIR) / "anomaly_pool.csv"))
            r = write_arms(O.rp(ARMS_DIR), pool)
            n = sum(len(v) for v in r["files"].values())
            print(f"wrote {n} world files to {ARMS_DIR} (arms.json sha256 "
                  f"{O.sha256(O.rp(ARMS_DIR) / 'arms.json')})")
            return 0
        if a.cmd == "score-worlds":
            out = O.rp(a.out)
            if out.exists():
                raise EngineeringFailure(f"{out} exists: reading 1 is scored once")
            P = Paths2()
            att = check_attempts(O.rp(P.attempts), {
                (t, b): O.rp(p) for t, ps in P.batches.items() for b, p in enumerate(ps)})
            rep = score_worlds(P, decl=doc, decl_sha=dsha)
            rep.update({"attempts": att, "frozen_code": doc["frozen_code"]})
            O._write_json_once(out, rep)
            print(f"wrote {a.out} (status {rep['status']}; level reading "
                  f"{rep['readings']['level']})")
            return 0
        if a.cmd == "score-forcing":
            out = O.rp(a.out)
            if out.exists():
                raise EngineeringFailure(f"{out} exists: reading 2 is scored once")
            try:
                rep = FYM.score_forcing(FYM.ForcingPaths(), index=a.replay_index,
                                        unavailable=a.replay_unavailable, decl=doc,
                                        decl_sha=dsha, attempts=ATTEMPTS)
            except FYM.ReplayGateE as err:
                FYM.log_replay_gate_e(O.rp(ATTEMPTS), a.replay_index, str(err))
                raise
            rep["frozen_code"] = doc["frozen_code"]
            O._write_json_once(out, rep)
            print(f"wrote {a.out} (status {rep['status']})")
            return 0
    except EngineeringFailure as err:
        print(f"ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
