"""FV.stat.v2 (round-8 rank 3; engineering v2 of FV.stat.v1, CPU, synthetic only): how decisive
is E04, the registered prospective test of the statistical layer (twin.hybrid, 2026-11..2027-10)?

FV.stat.v1 stopped in its self-check before any statistic was computed
(results/dataprep/reports/FV.stat.v1/engineering_failure.json, sha256 ac83bd46...): stat_fv.py
writes its synthetic evaluation rows with model_sha256 = 'synthetic' (stat_fv.py:232) and, when
scoring, hands ``twin.validation.evaluate`` the REAL E04 protocol (stat_fv.py:424,
``protocol_path=pp``), whose model_sha256 is the hash of E04's parameters.json, so evaluate
refuses the rows ('Predictions do not match the registered model'). The synthetic protocol
stat_fv.py:321-325 would have written is never used on that path, and even there ``**protocol``
(line 324) would let a caller's model_sha256 override 'synthetic'. stat_fv.py and prereg_fv2.py
are hash-bound (FV v2, FV.ext), so this v2 lives in new files and imports v1 unchanged.

The fix (gate E, :func:`selfcheck_world`): each self-check world's rows are written as a
synthetic predictions file whose sha256 becomes their model_sha256; they are evaluated against a
SYNTHETIC protocol copy (the real protocol's evaluate-read fields, model_sha256 replaced by that
sha) and must match ``O.g0_e02`` exactly; and, as an engineering CONTROL, the same rows are
handed to the REAL protocol, which must refuse them for the sha mismatch and nothing else.

Design copied verbatim from v1 (RHO_GRID, SHARE_GRID, K 500, E04's layout and rule: equal-station
RMSE ratio to climatology <= 1.0, pooled coverage in [0.85, 0.95], >= 100 stations with >= 6
months), with the registered layer's arithmetic re-run per world by v1's own functions
(``stat_fv.e04_frame`` / ``e04_pass``). Changes, each declared: the self-check above; a FRESH seed
chosen by the lead at registration (a parameter with no default; v1's declared seed 20261103,
previewed at K = 8 with pooled P about 0.92, is refused, as are both smoke seeds) and new stream
ids; a declared REPORT-ONLY rho = 0 null reading (an AR world without persistence, K_NULL per
share value); wording that states what a 2028 E04 pass can and cannot show given that null rate;
I1.hybrid_tf T1/T2 (v1's report-only rider) are not carried; v1's frozen_code omitted
osse_fv2.py and score_e02c.py (both imported on v1's path), v2 freezes both and
hydrophysics/twin/release.py (imported by validation); every ``run`` attempt is appended to
attempts.jsonl next to the report.

Reading (one-sided, applied once): P = equal-weight mean of P(E04 pass) over the nine AR cells;
P > 0.5 -> 'not_shown_weak', otherwise 'weak_by_design'. The null reading sets no branch.

CLI (repository root, hydro env):
  stat_v2.py smoke --out <dir outside results/>     engineering only; prints 'ran ok', no rate
  stat_v2.py verify --seed N --prereg-id FV.stat.v2 registration, code and protocol intact
  stat_v2.py run --seed N --prereg-id FV.stat.v2    once, after registration; report mode 'x';
                                                    every attempt logged in attempts.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

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

import osse_fv as O  # noqa: E402  (FV v1, frozen; g0_e02, helpers)
import osse_fv2 as V2  # noqa: E402  (FV v2, frozen; Monte Carlo constants)
import stat_fv as ST  # noqa: E402  (FV.stat.v1, frozen; never edited)

ID = "FV.stat.v2"
WP = "FV"
VERSION = "2.0.0"
F = "dataprep_staging/fv_stat_v2"
SELF = f"{F}/stat_v2.py"
PREREG = f"{F}/prereg_v2.py"
TESTS = "tests/test_fv_stat_v2.py"
FROZEN = (SELF, ST.SELF, O.SELF, f"{V2.F}/osse_fv2.py", "dataprep_staging/e02c/score_e02c.py",
          "hydrophysics/twin/hybrid.py", "hydrophysics/twin/coverage.py",
          "hydrophysics/twin/drift_diag.py", "hydrophysics/twin/validation.py",
          "hydrophysics/twin/release.py")

# ------------------------------------------------------------------ v1's design, verbatim
RHO_GRID = (0.80, 0.86, 0.93)
SHARE_GRID = (0.0, 0.15, 0.30)
K_STAT = 500
E04_PROTOCOL = {"max_rmse_ratio": 1.0, "min_stations": 100, "min_months": 6,
                "coverage_bounds": [0.85, 0.95]}
E04_ORIGINS = {"2026-09": 133, "2026-08": 6, "2026-07": 8}
E04_COUNTIES = {"Yunlin": 78, "Changhua": 69}
P_WEAK = 0.5
RHO_RECOVERY_TOL = 0.03
N_RECOVERY = 50
N_SELFCHECK_AR = 2
if (RHO_GRID, SHARE_GRID, K_STAT, E04_PROTOCOL, E04_ORIGINS, E04_COUNTIES, P_WEAK,
        RHO_RECOVERY_TOL, N_RECOVERY, N_SELFCHECK_AR) != (
        ST.RHO_GRID, ST.SHARE_GRID, ST.K_STAT, ST.E04_PROTOCOL, ST.E04_ORIGINS,
        ST.E04_COUNTIES, ST.P_WEAK, ST.RHO_RECOVERY_TOL, ST.N_RECOVERY, O.N_SELFCHECK):
    raise ImportError("FV.stat.v2's copy of v1's design differs from stat_fv.py")

# ------------------------------------------------------------------ v2's declared changes
NULL_RHO = 0.0
K_NULL = 500
STREAM_E04, STREAM_NULL, STREAM_SELFCHECK = 41, 42, 43     # v1 used 31/32/33
SMOKE_SEED = 51120017                                      # v2 engineering smokes only
REFUSED_SEEDS = {
    ST.SEED: "FV.stat.v1's declared seed: previewed at K = 8 per cell before registration "
             "(pooled P about 0.92)",
    ST.SMOKE_SEED: "FV.stat.v1's smoke seed (engineering smokes at K = 8 per cell)",
    SMOKE_SEED: "FV.stat.v2's own engineering smoke seed",
}
REFUSAL = "Predictions do not match the registered model"
#: the protocol fields twin.validation.evaluate reads (and schema/task); the synthetic copy
#: carries exactly these, with model_sha256 replaced
EVAL_KEYS = ("schema", "task", "test_start", "test_end", "registered_before",
             "independent_holdout", "baseline", "max_rmse_ratio", "min_stations", "min_months",
             "coverage_bounds")
HYB = ST.HYB
E04_PROTOCOL_PATH = f"{HYB}/protocol.json"
E04_MANIFEST_PATH = f"{HYB}/manifest.json"
E04_PROTOCOL_SHA = "18a87f270c47853c7f07bf700df2b3b8bf484394a3c548b6ee45f03593d42264"
E04_MANIFEST_SHA = "8e28ac21a7d590da93875bb02a639399ab9b196661e779aa9dff3538b777b512"
E04_SCORABLE_FROM = ST.E04_SCORABLE_FROM
REPORT = f"results/dataprep/reports/{ID}/report.json"
ATTEMPTS_NAME = "attempts.jsonl"           # beside the report: every ``run`` attempt, appended
EngineeringFailure = O.EngineeringFailure

V1_PREVIEW_NOTE = ("FV.stat.v1's identical design was previewed at K = 8 per cell on v1's "
                   "declared seed 20261103 before v1's registration (pooled P about 0.92, "
                   "'not_shown_weak'); v2 draws fresh worlds (fresh seed, new streams), but the "
                   "preview informs the expected outcome and is disclosed")

WORDING = {
    "weak_by_design": {
        "en": "Synthetic decidability study (FV.stat.v2, model-free; engineering v2 of "
              "FV.stat.v1 on a fresh seed): if anomalies truly persist as the statistical layer "
              "assumes (lag-1 0.80-0.93), its registered test (E04) passes with probability "
              "{p:.2f} (optimistic: no gaps, drift or datum steps; not a strict bound; v1's "
              "identical design was previewed at K = 8 on another seed before registration), so "
              "the test is weak by design: a 2028 fail would not refute the layer. In a world "
              "with no persistence (lag-1 0) the same rule passes with probability {p0:.2f}.",
        "zh": "合成可判定性研究（FV.stat.v2，不依賴模型；FV.stat.v1 的工程修正版，使用新種子）："
              "若異常確如統計層所假設般延續（一階自相關 0.80-0.93），其登錄檢驗（E04）的通過"
              "機率為 {p:.2f}（偏樂觀：無缺測、漂移或基準跳動；並非嚴格界限；v1 的同一設計曾於"
              "登錄前以另一種子、K = 8 預覽），因此該檢驗在設計上即屬弱檢驗；2028 年未通過也"
              "不足以否定統計層。在異常不延續的世界（一階自相關 0）中，同一規則的通過機率為 "
              "{p0:.2f}。"},
    "not_shown_weak": {
        "en": "Synthetic decidability study (FV.stat.v2, model-free; engineering v2 of "
              "FV.stat.v1 on a fresh seed): if anomalies truly persist as the statistical layer "
              "assumes (lag-1 0.80-0.93), its registered test (E04) passes with probability "
              "{p:.2f} (optimistic: no gaps, drift or datum steps; not a strict bound; v1's "
              "identical design was previewed at K = 8 on another seed before registration); "
              "the study does not show the test weak. In a world with no persistence (lag-1 0) "
              "the same rule passes with probability {p0:.2f}.",
        "zh": "合成可判定性研究（FV.stat.v2，不依賴模型；FV.stat.v1 的工程修正版，使用新種子）："
              "若異常確如統計層所假設般延續（一階自相關 0.80-0.93），其登錄檢驗（E04）的通過"
              "機率為 {p:.2f}（偏樂觀：無缺測、漂移或基準跳動；並非嚴格界限；v1 的同一設計曾於"
              "登錄前以另一種子、K = 8 預覽）；本研究未顯示該檢驗為弱檢驗。在異常不延續的世界"
              "（一階自相關 0）中，同一規則的通過機率為 {p0:.2f}。"},
    "not_evaluable": {
        "en": "Synthetic decidability study FV.stat.v2 not evaluable ({why}); E04 gains no "
              "interpretation line from it.",
        "zh": "合成可判定性研究 FV.stat.v2 無法評估（{why}）；E04 不因此增加解讀說明。"},
}
#: carried beside either reading (never with not_evaluable): what a 2028 E04 pass can show
PASS_CLAUSE = {
    "en": "What a 2028 E04 pass can show: over 2026-11..2027-10 the layer was no worse than "
          "climatology on equal-station RMSE (ratio <= 1.0, no margin; ties pass), with pooled "
          "90 % interval coverage in [0.85, 0.95]. What it cannot show: a pass alone does not "
          "establish persistence; a no-persistence world passes with probability {p0:.2f} "
          "(against {p:.2f} with persistence, both synthetic and optimistic); nor anything about "
          "the physics model, or skill in other years. The outcomes are not symmetric: in the "
          "AR worlds a fail has probability about {q:.2f} (1 - P), so a 2028 fail is the more "
          "informative outcome.",
    "zh": "2028 年 E04 通過所能顯示的：在 2026-11 至 2027-10 期間，統計層以等站權重 RMSE 計"
          "不劣於氣候值（比值 <= 1.0，無餘裕；相等亦算通過），且合併 90 % 區間涵蓋率落在 "
          "[0.85, 0.95] 內。所不能顯示的：僅憑通過不能確立異常延續；異常不延續的世界通過的機率"
          "為 {p0:.2f}（延續時為 {p:.2f}，兩者皆為合成且偏樂觀的數值）；亦不能顯示任何關於物理"
          "模型的結論或其他年份的預報能力。兩種結果並不對稱：在 AR 世界中未通過的機率約為 "
          "{q:.2f}（1 - P），因此 2028 年未通過是資訊量較高的結果。",
}
READING_RULE = ("P > 0.5 -> 'not_shown_weak'; otherwise 'weak_by_design' (one-sided, applied "
                "once; the 1.645 SE <= 0.07 Monte Carlo check is non-binding at this K, and a "
                "failed check gives 'not_evaluable'); the rho = 0 null reading is report-only and "
                "sets no branch")


# ----------------------------------------------------------------------------- seeds, worlds
def check_seed(seed) -> int:
    """The registered run's seed: an int chosen by the lead, never a previewed or smoke seed."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise EngineeringFailure(f"seed must be a non-negative int, got {seed!r}")
    if int(seed) in REFUSED_SEEDS:
        raise EngineeringFailure(f"seed {seed} refused: {REFUSED_SEEDS[int(seed)]}")
    return int(seed)


def rng_for(stream: int, cell: int, k: int, *, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream, cell, k]))


def cells() -> list[dict]:
    return ST.cells()


def null_cells() -> list[dict]:
    return [{"cell": j, "rho": NULL_RHO, "share": c} for j, c in enumerate(SHARE_GRID)]


def world(lay, rho: float, share: float, stream: int, cell: int, k: int, *, seed: int):
    return ST.anomalies(lay.county, rho, share, rng_for(stream, cell, k, seed=seed))


# ----------------------------------------------------------------------------- gate E
def protocol_rule(real: dict) -> dict:
    """E04's rule as the real protocol states it; must equal v1's copy."""
    missing = [k for k in E04_PROTOCOL if k not in real]
    if missing:
        raise EngineeringFailure(f"E04 protocol lacks {missing}")
    rule = {k: real[k] for k in E04_PROTOCOL}
    if rule != E04_PROTOCOL:
        raise EngineeringFailure("E04 protocol's rule differs from FV.stat.v1's copy")
    return rule


def synthetic_protocol(real: dict, syn_sha: str) -> dict:
    """The real protocol's evaluate-read fields with model_sha256 = the synthetic rows' sha."""
    need = [k for k in EVAL_KEYS if k not in ("schema", "task")]
    missing = [k for k in need if k not in real]
    if missing:
        raise EngineeringFailure(f"E04 protocol lacks evaluate fields {missing}")
    d = {k: real[k] for k in EVAL_KEYS if k in real}
    d["model_sha256"] = syn_sha
    return d


def synthetic_sha(path: Path) -> str:
    """model_sha256 of synthetic rows: the sha256 of their predictions file (as
    twin.hybrid.register hashes its parameter file)."""
    return O.sha256(path)


def _origins(lay) -> dict:
    return {i: str(ST.AXIS[t].date()) for i, t in zip(lay.ids, lay.last_fit, strict=True)}


def selfcheck_world(f: pd.DataFrame, origin_by_id: dict, real_protocol_path: Path, real: dict,
                    rule: dict, tmp_root: Path) -> dict:
    """g0_e02 == twin.validation.evaluate on one world against the SYNTHETIC protocol copy;
    then the CONTROL: the real E04 protocol must refuse the same rows for the sha mismatch."""
    from hydrophysics.twin.validation import evaluate

    mine = ST.e04_pass(f, rule)
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root, prefix=".selfcheck_v2_") as td:
        pred = f.drop(columns=["observed_m"]).assign(
            date=f["date"].dt.strftime("%Y-%m-%d"),
            prediction_origin=f["station_id"].map(origin_by_id))
        pp = Path(td) / "predictions.csv"
        pred.to_csv(pp, index=False)
        syn = synthetic_sha(pp)
        if syn == real.get("model_sha256"):
            raise EngineeringFailure("the synthetic rows' sha equals E04's model_sha256: the "
                                     "control would be vacuous")
        ev = Path(td) / "evaluation.csv"
        pred.assign(observed_m=f["observed_m"].to_numpy(), model_sha256=syn).to_csv(ev,
                                                                                    index=False)
        sp = Path(td) / "protocol_synthetic.json"
        sp.write_text(json.dumps(synthetic_protocol(real, syn), sort_keys=True) + "\n")
        r = evaluate(ev, sp)
        ok = (r["coverage"] is not None and abs(r["coverage"] - mine["coverage"]) < 1e-12
              and r["rmse_ratio"] is not None and abs(r["rmse_ratio"] - mine["ratio"]) < 1e-12
              and bool(r["passed"]) == bool(mine["pass"]))
        if not ok:
            raise EngineeringFailure("g0_e02 disagrees with validation.evaluate on the "
                                     "synthetic protocol copy")
        try:
            evaluate(ev, real_protocol_path)
        except ValueError as err:
            if str(err) != REFUSAL:
                raise EngineeringFailure(f"the real E04 protocol refused the synthetic rows for "
                                         f"another reason: {err}") from err
        else:
            raise EngineeringFailure("the real E04 protocol accepted rows with a mismatched "
                                     "model_sha256")
    return {"identical": True, "rows": int(len(f)), "control_refused": True,
            "control_message": REFUSAL}


def engineering(lay, real_protocol_path: Path, tmp_root: Path, *, seed: int,
                n_recovery: int = N_RECOVERY) -> dict:
    """Gate E: the fixed self-check with its control (two AR worlds as v1, plus one null world)
    and v1's rho-recovery check. Raises EngineeringFailure (nothing written) on any failure."""
    real_protocol_path = Path(real_protocol_path)
    real = json.loads(real_protocol_path.read_text(encoding="utf-8"))
    rule = protocol_rule(real)
    orig = _origins(lay)
    checks = []
    for k, (rho, share) in enumerate([(0.86, 0.15)] * N_SELFCHECK_AR + [(NULL_RHO, 0.15)]):
        f, _ = ST.e04_frame(lay, world(lay, rho, share, STREAM_SELFCHECK, 0, k, seed=seed))
        checks.append({"rho": rho, "share": share,
                       **selfcheck_world(f, orig, real_protocol_path, real, rule, tmp_root)})
    rec = next(c for c in cells() if c["rho"] == 0.86 and c["share"] == 0.0)
    rhos = [ST.e04_frame(lay, world(lay, 0.86, 0.0, STREAM_E04, rec["cell"], k, seed=seed)
                         )[1]["rho_mean"] for k in range(n_recovery)]
    m = float(np.mean(rhos))
    recovery = {"mean_fitted_rho": m, "true": 0.86, "tolerance": RHO_RECOVERY_TOL,
                "worlds": n_recovery, "pass": abs(m - 0.86) <= RHO_RECOVERY_TOL}
    if not recovery["pass"]:
        raise EngineeringFailure(f"the simulation does not recover rho: {m:.3f}")
    return {"evaluate_selfcheck": checks, "rho_recovery": recovery, "rule": rule,
            "synthetic_protocol_keys": sorted(EVAL_KEYS) + ["model_sha256"]}


# ----------------------------------------------------------------------------- worlds, P
def cell_records(lay, cell_list: list[dict], stream: int, k: int, rule: dict, *, seed: int
                 ) -> list[list[dict]]:
    """Per cell, per world: E04's pass, ratio and coverage (no aggregation)."""
    return [[ST.e04_pass(ST.e04_frame(lay, world(lay, c["rho"], c["share"], stream, c["cell"],
                                                 i, seed=seed))[0], rule)
             for i in range(k)] for c in cell_list]


def summarise(cell_list: list[dict], records: list[list[dict]]) -> list[dict]:
    out = []
    for c, ws in zip(cell_list, records, strict=True):
        out.append({**c, "passed": O._rate([w["pass"] for w in ws]),
                    "median_ratio": float(np.median([w["ratio"] for w in ws])),
                    "median_coverage": float(np.median([w["coverage"] for w in ws])),
                    "coverage_above_upper": float(np.mean([w["coverage"] > 0.95 for w in ws])),
                    "coverage_below_lower": float(np.mean([w["coverage"] < 0.85 for w in ws])),
                    "ratio_above_1": float(np.mean([w["ratio"] > 1.0 for w in ws]))})
    return out


def pooled(per: list[dict]) -> dict:
    ps = [x["passed"] for x in per]
    se = math.sqrt(sum(x["mc_se"] ** 2 for x in ps)) / len(ps)
    p = float(np.mean([x["p"] for x in ps]))
    return {"p": p, "mc_se": se, "mc_half_width_90": V2.Z_MC * se,
            "p_minus_threshold_in_se": (p - P_WEAK) / se if se > 0 else None,
            "min_over_cells": float(min(x["p"] for x in ps)),
            "max_over_cells": float(max(x["p"] for x in ps))}


def reading(P: dict) -> str | None:
    """P > 0.5 -> not_shown_weak, otherwise weak_by_design; None if the MC check fails."""
    if P["mc_half_width_90"] > V2.MC_HALF_WIDTH or not np.isfinite(P["p"]):
        return None
    return "not_shown_weak" if P["p"] > P_WEAK else "weak_by_design"


def readings(P: dict, P0: dict) -> dict:
    rd = reading(P)
    if rd is None:
        wording = {k: v.format(why="Monte Carlo precision not met")
                   for k, v in WORDING["not_evaluable"].items()}
        clause = None
    else:
        wording = {k: v.format(p=P["p"], p0=P0["p"]) for k, v in WORDING[rd].items()}
        clause = {k: v.format(p=P["p"], p0=P0["p"], q=1.0 - P["p"])
                  for k, v in PASS_CLAUSE.items()}
    return {"threshold": P_WEAK, "rule": READING_RULE, "e04": rd, "wording": wording,
            "what_a_2028_pass_shows": clause,
            "pass_likelihood_ratio_report_only": (P["p"] / P0["p"] if P0["p"] > 0 else None)}


# ----------------------------------------------------------------------------- score
def _score(*, seed: int, protocol_path: Path, tmp_root: Path, decl_sha: str,
           k_stat: int = K_STAT, k_null: int = K_NULL, log=print) -> dict:
    """Gate E, then the nine AR cells and the null cells, then the reading. Private: only
    ``run`` calls it, after :func:`verify` has loaded the registration (``decl_sha`` is the
    registered declaration's sha256 and has no default); the smoke never does."""
    if not (isinstance(decl_sha, str) and len(decl_sha) == 64):
        raise EngineeringFailure(f"{ID} scores only under its registration: run "
                                 "'stat_v2.py run --seed N --prereg-id FV.stat.v2'")
    seed = check_seed(seed)
    lay = ST.e04_layout()
    rep: dict = {"id": ID, "version": VERSION, "created": O._now(),
                 "declaration_sha256": decl_sha, "seed": seed, "k_stat": k_stat,
                 "k_null": k_null, "streams": {"e04": STREAM_E04, "null": STREAM_NULL,
                                               "selfcheck": STREAM_SELFCHECK},
                 "grid": {"rho": list(RHO_GRID), "share": list(SHARE_GRID)},
                 "v1_preview": V1_PREVIEW_NOTE,
                 "mc_gate_note": V2.mc_note(len(cells()), k_stat),
                 "e04_layout": {"stations": len(lay.ids), "origins": E04_ORIGINS,
                                "counties": E04_COUNTIES}}
    rep["engineering"] = engineering(lay, protocol_path, tmp_root, seed=seed)
    rule = rep["engineering"]["rule"]
    log("gate E passed (engineering)")
    per = summarise(cells(), cell_records(lay, cells(), STREAM_E04, k_stat, rule, seed=seed))
    per0 = summarise(null_cells(), cell_records(lay, null_cells(), STREAM_NULL, k_null, rule,
                                                seed=seed))
    P, P0 = pooled(per), pooled(per0)
    P["by_rho"] = {str(r): float(np.mean([x["passed"]["p"] for x in per if x["rho"] == r]))
                   for r in RHO_GRID}
    rep.update({"e04_cells": per, "P": P,
                "null_reading": {"report_only": True, "rho": NULL_RHO, "cells": per0, "P": P0,
                                 "note": "AR worlds without persistence; sets no branch; the "
                                         "planner's expectation (about 0.4-0.5) is a prior, "
                                         "not a gate"},
                "readings": readings(P, P0), "status": "scored"})
    return rep


# ----------------------------------------------------------------------------- registration
def index_record(prereg_id: str, prereg_dir=None) -> dict | None:
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    p = d / "INDEX.jsonl"
    if not p.is_file():
        return None
    rows = [r for r in prereg3.read_chained(p) if r["id"] == prereg_id]
    return rows[-1] if rows else None


def load_registered(prereg_id: str, prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    """INDEX.jsonl entry, file sha and ledger row intact; frozen code byte-identical."""
    from harness import prereg3

    if index_record(prereg_id, prereg_dir) is None:
        raise EngineeringFailure(f"{prereg_id} is not in INDEX.jsonl: register it first")
    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    try:
        doc, sha = prereg3.load_prereg(prereg_id, **kw)
    except Exception as err:      # noqa: BLE001 - any failure is a refusal
        raise EngineeringFailure(f"{prereg_id} is not registered or not intact: {err}") from err
    fc = doc.get("frozen_code") or {}
    if set(fc) != set(FROZEN):
        raise EngineeringFailure("frozen_code does not list exactly FV.stat.v2's files")
    bad = [p for p, h in fc.items() if not O.rp(p).is_file() or O.sha256(O.rp(p)) != h]
    if bad:
        raise EngineeringFailure(f"code sha256 differs from the registration: {bad}")
    changed = prereg3.changed_inputs(doc)
    if changed:
        raise EngineeringFailure(f"registered inputs changed: {changed}")
    return doc, sha


def check_registered_design(doc: dict, seed: int) -> None:
    d = doc.get("design") or {}
    want = {"seed": seed, "K_STAT": K_STAT, "K_NULL": K_NULL, "rho_grid": list(RHO_GRID),
            "share_grid": list(SHARE_GRID), "null_rho": NULL_RHO,
            "streams": {"e04": STREAM_E04, "null": STREAM_NULL, "selfcheck": STREAM_SELFCHECK}}
    bad = sorted(k for k, v in want.items() if d.get(k) != v)
    if bad:
        raise EngineeringFailure(f"the registered design differs from this run on {bad} "
                                 "(the seed must be the registered one)")
    if (doc.get("e04") or {}).get("protocol_sha256") != E04_PROTOCOL_SHA:
        raise EngineeringFailure("the registration's E04 protocol sha256 is not E04's")


def verify(seed, prereg_id: str, *, prereg_dir=None, rows_dir=None,
           protocol: str = E04_PROTOCOL_PATH) -> tuple[dict, str, Path]:
    seed = check_seed(seed)
    if prereg_id != ID:
        raise EngineeringFailure(f"this runner scores {ID} only, not {prereg_id!r}")
    doc, dsha = load_registered(prereg_id, prereg_dir, rows_dir)
    check_registered_design(doc, seed)
    pp = O.rp(protocol)
    if not pp.is_file() or O.sha256(pp) != E04_PROTOCOL_SHA:
        raise EngineeringFailure("E04 protocol.json missing or differs from its sha256")
    return doc, dsha, pp


def attempts_path(out: str = REPORT) -> Path:
    return O.rp(out).parent / ATTEMPTS_NAME


def _log_attempt(path: Path, rec: dict) -> None:
    """Append one attempt record (never rewritten); engineering retries are logged here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": ID, "time": O._now(), **rec}, sort_keys=True,
                            ensure_ascii=False, default=O._jsonable) + "\n")


def _attempt_no(path: Path) -> int:
    if not path.is_file():
        return 1
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return 1 + sum(r.get("event") == "start" for r in rows)


def run(seed, prereg_id: str, *, out: str = REPORT, prereg_dir=None, rows_dir=None,
        protocol: str = E04_PROTOCOL_PATH, log=print) -> dict:
    """Score once: refuses unless registered, intact and on the registered seed; mode 'x'.
    Every attempt appends to attempts.jsonl beside the report: 'start', then
    'engineering_failure' (with its message), 'error' (any other exception) or 'success'
    (with the report's sha256). No statistic is written to the log."""
    outp = O.rp(out)
    ap = attempts_path(out)
    n = _attempt_no(ap)
    _log_attempt(ap, {"event": "start", "attempt": n, "seed": seed, "prereg_id": prereg_id,
                      "report": str(out), "code_sha256": O.sha256(Path(__file__))})
    try:
        check_seed(seed)
        if outp.exists():
            raise EngineeringFailure(f"{outp} exists: {ID} is scored once")
        doc, dsha, pp = verify(seed, prereg_id, prereg_dir=prereg_dir, rows_dir=rows_dir,
                               protocol=protocol)
        rep = _score(seed=seed, protocol_path=pp, tmp_root=outp.parent, decl_sha=dsha, log=log)
        rep["frozen_code"] = doc["frozen_code"]
        O._write_json_once(outp, rep)
    except EngineeringFailure as err:
        _log_attempt(ap, {"event": "engineering_failure", "attempt": n, "message": str(err)})
        raise
    except Exception as err:
        _log_attempt(ap, {"event": "error", "attempt": n,
                          "message": f"{type(err).__name__}: {err}"})
        raise
    _log_attempt(ap, {"event": "success", "attempt": n, "report_sha256": O.sha256(outp),
                      "declaration_sha256": dsha})
    return rep


# ----------------------------------------------------------------------------- smoke
def _under_results(p: Path) -> bool:
    try:
        Path(p).resolve().relative_to((REPO / "results").resolve())
        return True
    except ValueError:
        return False


FRAME_COLUMNS = ("station_id", "date", "predicted_m", "baseline_m", "lower_m", "upper_m",
                 "observed_m")


def frame_shape_ok(lay, f: pd.DataFrame) -> bool:
    """Shape only: E04's columns, every station x every window month, finite bounds ordered."""
    n_win = ST.WIN[1] - ST.WIN[0]
    return (tuple(f.columns) == FRAME_COLUMNS and len(f) == len(lay.ids) * n_win
            and f["station_id"].nunique() == len(lay.ids) and f["date"].nunique() == n_win
            and bool(np.isfinite(f[["predicted_m", "lower_m", "upper_m"]].to_numpy()).all())
            and bool((f["lower_m"] < f["upper_m"]).all()))


def run_smoke(out: Path, protocol: str = E04_PROTOCOL_PATH, k: int = 2) -> dict:
    """Engineering only, on SMOKE_SEED: gate E with its control, every cell's world generator and
    layer at a tiny K (the E04 FRAME checked for shape only; E04's rule is never applied to a
    cell world), and the reading/wording plumbing on FABRICATED records. Writes and returns no
    pass rate, P or world statistic."""
    out = Path(out)
    if _under_results(out):
        raise EngineeringFailure("the smoke never writes under results/")
    out.mkdir(parents=True, exist_ok=False)
    lay = ST.e04_layout()
    eng = engineering(lay, O.rp(protocol), out / "tmp", seed=SMOKE_SEED, n_recovery=k)
    rule = eng["rule"]
    del rule
    shape_ok = all(frame_shape_ok(lay, ST.e04_frame(lay, world(lay, c["rho"], c["share"],
                                                               stream, c["cell"], i,
                                                               seed=SMOKE_SEED))[0])
                   for cl, stream in ((cells(), STREAM_E04), (null_cells(), STREAM_NULL))
                   for c in cl for i in range(k))
    fab = [[{"pass": True, "ratio": 0.9, "coverage": 0.9}] * 4] * len(cells())
    fab0 = [[{"pass": False, "ratio": 1.1, "coverage": 0.9}] * 4] * len(null_cells())
    rd = readings(pooled(summarise(cells(), fab)), pooled(summarise(null_cells(), fab0)))
    plumbing_ok = shape_ok and rd["e04"] in ("not_shown_weak", None) and bool(rd["wording"])
    rep = {"id": ID, "smoke": True, "status": "ran ok" if plumbing_ok else "plumbing failed",
           "seed": SMOKE_SEED, "k": k,
           "engineering": {"selfcheck_identical": [c["identical"] for c in
                                                   eng["evaluate_selfcheck"]],
                           "control_refused": [c["control_refused"] for c in
                                               eng["evaluate_selfcheck"]],
                           "control_message": REFUSAL,
                           "rho_recovery_gate_passed": eng["rho_recovery"]["pass"]},
           "note": "SMOKE: engineering only; no pass rate, P or world statistic is written"}
    O._write_json_once(out / "smoke.json", rep)
    return rep


# ----------------------------------------------------------------------------- CLI
def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke", help="engineering only; prints 'ran ok'")
    sm.add_argument("--out", required=True, help="a NEW directory outside results/")
    sm.add_argument("--protocol", default=E04_PROTOCOL_PATH)
    for name in ("verify", "run"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, required=True, help="the registered (fresh) seed")
        p.add_argument("--prereg-id", required=True)
        if name == "run":
            p.add_argument("--out", default=REPORT)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out), a.protocol)
            print(rep["status"])
            return 0 if rep["status"] == "ran ok" else 1
        if a.cmd == "verify":
            _, dsha, _ = verify(a.seed, a.prereg_id, prereg_dir=prereg_dir, rows_dir=rows_dir)
            print(f"{ID} registered (sha256 {dsha[:16]}); code, inputs and E04 protocol intact")
            return 0
        rep = run(a.seed, a.prereg_id, out=a.out, prereg_dir=prereg_dir, rows_dir=rows_dir)
        print(f"wrote {a.out} (status {rep['status']}; E04 reading {rep['readings']['e04']})")
        return 0
    except EngineeringFailure as err:
        print(f"REFUSED / ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
