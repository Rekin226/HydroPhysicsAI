"""FV.stat.v1 (fresh declaration, CPU, synthetic only; round-7 rank 12 rider): would a 2028
fail of the registered statistical layer (E04) refute it? And how do I1.hybrid_tf's T1/T2
rules behave when the AR(1) layer is the truth?

Truth worlds: every well's monthly head anomaly is a stationary AR(1) with lag-1 coefficient
rho in RHO_GRID = {0.80, 0.86, 0.93} (the round-7 judge's grid around the E04 fit's median,
0.86; the post-origin in-sample 1.8 persistence weight is NOT used, U7), unit marginal
variance, made of a well part and a county-common part with share c in SHARE_GRID =
{0.0, 0.15, 0.30} (values of U1.HB.power.v1's registered ICC_month grid). Nine cells, K_STAT
worlds each. Every month 2012-01..2026-09 observed except where E04's registered origins say
a well's last observation was 2026-07 or 2026-08 (optimistic: no gaps, no drift, no datum
step; the scale is common to all wells, stated).

E04 (primary): the registered layer's own procedure, re-run on each world exactly as
``twin.hybrid.register`` does it (fair_baselines climatology, lag1_rho with the pooled
fallback, persisted_anomaly with max gap 3, _residual_sd, central 90 % = +-1.645 sd),
scored with E02's evaluation arithmetic (equal-station RMSE ratio to climatology <= 1.0 AND
pooled coverage in [0.85, 0.95], >= 100 stations with >= 6 months; O.g0_e02, self-checked
against twin.validation.evaluate). P(pass | AR(1) layer right) = equal-weight mean over the
nine cells. One-sided reading as FV: P <= 0.5 -> 'weak_by_design' (a 2028 E04 fail would not
refute the layer), otherwise 'not_shown_weak'.

I1.hybrid_tf T1/T2 (REPORT-ONLY; no reading, no page line): 158 wells (84 Yunlin, 74
Changhua, one well per site), eight county-common null drivers (AR(1), lag-1 0.9, no effect on
heads), the frozen hyperparameters and code of i1.hybrid_tf.model (features, Ridge, lag1)
fitted on 2012-02..2026-09; T1 decided by i1.hybrid_tf.scorer.decide exactly; drivers taken
as recorded through the window. Reported: the rate of 'adds skill' per county when no driver
carries information (T1's false-pass rate), and T2's RMSE ratio to climatology.

Gates: E (self-check against twin.validation.evaluate; the simulation recovers rho: in cell
(0.86, 0.0) the mean fitted rho over the first 50 worlds lies within 0.03 of 0.86); MC
(1.645 SE of P <= 0.07; non-binding at 9 x 500, the largest possible half-width being 0.012).

Previewed before registration: the builder ran this study once at K = 8 per cell on the
DECLARED seed (pooled P about 0.92); those are the first 8 worlds per cell of the K = 500
score. The seed is NOT changed (that would be a reroll); the preview is disclosed, carried
in the report (``builder_preview``) and in every wording line, and logged as a look by the
lead.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "fv_decidability"),
           str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osse_fv as O  # noqa: E402
import osse_fv2 as V2  # noqa: E402

ID = "FV.stat.v1"
WP = "FV"
VERSION = "1.0.0"
SELF = f"{V2.F}/stat_fv.py"
HYB = "results/twin/prospective_2026-11_hybrid"
I1_ID = "I1.hybrid_tf.prospective_2026-11.v1"
FROZEN = (SELF, O.SELF, "hydrophysics/twin/hybrid.py", "hydrophysics/twin/coverage.py",
          "hydrophysics/twin/drift_diag.py", "hydrophysics/twin/validation.py",
          "dataprep_staging/i1/hybrid_tf/model.py", "dataprep_staging/i1/hybrid_tf/scorer.py",
          "dataprep_staging/i1/hybrid_tf/config.py")
RHO_GRID = (0.80, 0.86, 0.93)
SHARE_GRID = (0.0, 0.15, 0.30)
K_STAT = 500
K_T1 = 100
SEED = 20261103
SMOKE_SEED = 77770001        # smoke only: a DIFFERENT seed, engineering only (sets nothing)
STREAM_E04, STREAM_T1, STREAM_SELFCHECK = 31, 32, 33
P_WEAK = V2.P_WEAK
RHO_RECOVERY_TOL = 0.03
N_RECOVERY = 50
DRIVER_RHO = 0.9
# E04's registered layout (manifest.json of results/twin/prospective_2026-11_hybrid: 147
# stations; origins 2026-09 133, 2026-08 6, 2026-07 8). County split of the 147 in the
# 158-well proportion (84/74), declared: the manifest has no county field.
E04_ORIGINS = {"2026-09": 133, "2026-08": 6, "2026-07": 8}
E04_COUNTIES = {"Yunlin": 78, "Changhua": 69}
E04_PROTOCOL = {"max_rmse_ratio": 1.0, "min_stations": 100, "min_months": 6,
                "coverage_bounds": [0.85, 0.95]}
T1_COUNTIES = {"yunlin": 84, "changhua": 74}
AXIS = pd.date_range("2012-01-01", "2027-10-01", freq="MS")
T_ALL = len(AXIS)                                   # 190
T_FIT = int((AXIS < "2026-11-01").sum())            # 178: 2012-01..2026-10 (window starts 178)
WIN = (T_FIT, T_ALL)
REPORT = f"results/dataprep/reports/{ID}/report.json"
BUILDER_PREVIEW = ("previewed at K = 8 per cell on the declared seed before registration "
                   "(pooled P about 0.92, 'not_shown_weak'); those are the first 8 worlds per "
                   "cell of this score; nothing was changed after the preview and the seed "
                   "was not rerolled")
E04_SCORABLE_FROM = "2027-11-01"     # E04's window closes 2027-10-31 (protocol test_end)
EngineeringFailure = O.EngineeringFailure

WORDING = {
    "weak_by_design": {
        "en": "Synthetic decidability study (FV.stat.v1, model-free): if anomalies truly "
              "persist as the statistical layer assumes (lag-1 0.80-0.93), its registered test "
              "passes with probability {p:.2f} (optimistic: no gaps, drift or datum steps; "
              "not a strict bound; previewed at K = 8 before registration), so the test is "
              "weak by design; a 2028 fail would not refute the layer.",
        "zh": "合成可判定性研究（FV.stat.v1，不依賴模型）："
              "若異常確如統計層所假設般延續（一階自相關 0.80-0.93），"
              "其登錄檢驗的通過機率為 {p:.2f}（偏樂觀：無缺測、漂移或基準跳動；並非嚴格"
              "界限；登錄前曾以 K = 8 預覽），因此該檢驗在設計上即屬弱檢驗；"
              "2028 年未通過也不足以否定統計層。"},
    "not_shown_weak": {
        "en": "Synthetic decidability study (FV.stat.v1, model-free): if anomalies truly "
              "persist as the statistical layer assumes (lag-1 0.80-0.93), its registered test "
              "passes with probability {p:.2f} (optimistic: no gaps, drift or datum steps; "
              "not a strict bound; previewed at K = 8 before registration); the study does "
              "not show the test weak and adds no other reading.",
        "zh": "合成可判定性研究（FV.stat.v1，不依賴模型）："
              "若異常確如統計層所假設般延續（一階自相關 0.80-0.93），"
              "其登錄檢驗的通過機率為 {p:.2f}（偏樂觀：無缺測、漂移或基準跳動；並非嚴格"
              "界限；登錄前曾以 K = 8 預覽）；本研究未顯示該檢驗為弱檢驗，也不提供其他解讀。"},
    "not_evaluable": {
        "en": "Synthetic decidability study FV.stat.v1 not evaluable ({why}); E04 gains no "
              "interpretation line from it.",
        "zh": "合成可判定性研究 FV.stat.v1 無法評估（{why}）；E04 不因此增加解讀說明。"},
}


def rng_for(stream: int, cell: int, k: int, seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream, cell, k]))


def cells() -> list[dict]:
    return [{"cell": i * len(SHARE_GRID) + j, "rho": r, "share": c}
            for i, r in enumerate(RHO_GRID) for j, c in enumerate(SHARE_GRID)]


def ar1_unit(n: int, T: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """(n, T) stationary AR(1) series with unit marginal variance."""
    z = rng.standard_normal((n, T))
    x = np.empty_like(z)
    x[:, 0] = z[:, 0]
    s = math.sqrt(1.0 - rho * rho)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + s * z[:, t]
    return x


def anomalies(county: np.ndarray, rho: float, share: float, rng: np.random.Generator
              ) -> np.ndarray:
    """(W, T_ALL) anomaly = sqrt(1 - c) well part + sqrt(c) county-common part."""
    W = len(county)
    u = ar1_unit(W, T_ALL, rho, rng)
    names = sorted(set(county))
    v = ar1_unit(len(names), T_ALL, rho, rng)
    common = v[[names.index(c) for c in county]]
    return math.sqrt(1.0 - share) * u + math.sqrt(share) * common


# ----------------------------------------------------------------------------- E04
@dataclass
class E04Layout:
    ids: list
    county: np.ndarray
    last_fit: np.ndarray            # index of each well's last observed month before 2026-11


def e04_layout() -> E04Layout:
    county = np.array([c for c, n in E04_COUNTIES.items() for _ in range(n)])
    last = np.concatenate([np.full(n, AXIS.get_loc(pd.Timestamp(f"{m}-01")))
                           for m, n in E04_ORIGINS.items()])
    ids = [f"{'09' if c == 'Yunlin' else '07'}{i:06d}" for i, c in enumerate(county)]
    return E04Layout(ids=ids, county=county, last_fit=last)


def e04_frame(lay: E04Layout, y: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """The registered layer re-run on one world (twin.hybrid.register's arithmetic) ->
    window frame (predicted, baseline, lower, upper, observed) and fitted ingredients."""
    from hydrophysics.twin.coverage import Z, _residual_sd
    from hydrophysics.twin.drift_diag import fair_baselines
    from hydrophysics.twin.hybrid import lag1_rho, persisted_anomaly

    obs = y.copy()
    for w, t in enumerate(lay.last_fit):
        obs[w, t + 1:T_FIT] = np.nan
    obs[:, T_FIT:] = np.nan
    clim = fair_baselines(obs, T_FIT, month_offset=0)["clim"]
    anom = obs - clim
    rho, _ = lag1_rho(anom, T_FIT)
    pooled = float(np.nanmedian(rho))
    rho_used = np.where(np.isfinite(rho), rho, pooled)
    a_stat, last = persisted_anomaly(anom, T_FIT, rho_used, 3)
    sd = _residual_sd(obs, clim, T_FIT)
    count = np.isfinite(obs[:, :T_FIT]).sum(axis=1)
    keep = (count >= 24) & np.isfinite(sd) & (last >= 0)
    pred = clim + a_stat
    z = Z[90]
    rows = []
    for w in np.flatnonzero(keep):
        for t in range(*WIN):
            rows.append((lay.ids[w], AXIS[t], pred[w, t], clim[w, t], pred[w, t] - z * sd[w],
                         pred[w, t] + z * sd[w], y[w, t]))
    f = pd.DataFrame(rows, columns=["station_id", "date", "predicted_m", "baseline_m",
                                    "lower_m", "upper_m", "observed_m"])
    return f, {"rho_mean": float(np.nanmean(rho)), "n_kept": int(keep.sum())}


def e04_pass(f: pd.DataFrame, protocol: dict) -> dict:
    g = O.g0_e02(f.assign(hit=((f["observed_m"] >= f["lower_m"])
                               & (f["observed_m"] <= f["upper_m"])).astype(float)), protocol)
    return {"pass": bool(g["e02_passed"]), "ratio": g["ratio"], "coverage": g["coverage"]}


def e04_selfcheck(lay: E04Layout, protocol_path: Path, tmp_root: Path, protocol: dict,
                  seed: int = SEED) -> list:
    """O.g0_e02 against twin.validation.evaluate on N_SELFCHECK worlds (own stream)."""
    from hydrophysics.twin.validation import evaluate

    out = []
    for k in range(O.N_SELFCHECK):
        y = anomalies(lay.county, 0.86, 0.15, rng_for(STREAM_SELFCHECK, 0, k, seed))
        f, _ = e04_frame(lay, y)
        mine = e04_pass(f, protocol)
        Path(tmp_root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root, prefix=".selfcheck_") as td:
            p = Path(td) / "evaluation.csv"
            f.assign(date=f["date"].dt.strftime("%Y-%m-%d"),
                     prediction_origin="2026-09-01", model_sha256="synthetic"
                     ).to_csv(p, index=False)
            r = evaluate(p, protocol_path)
        ok = (abs(r["coverage"] - mine["coverage"]) < 1e-12
              and abs(r["rmse_ratio"] - mine["ratio"]) < 1e-12 and r["passed"] == mine["pass"])
        if not ok:
            raise EngineeringFailure("g0_e02 disagrees with validation.evaluate (E04 worlds)")
        out.append({"identical": True, "rows": int(len(f))})
    return out


# ----------------------------------------------------------------------------- T1 / T2
def _hyb():
    from i1.hybrid_tf import config as HC
    from i1.hybrid_tf import model as HM
    from i1.hybrid_tf import scorer as HS
    return HC, HM, HS


def t1_world(rho: float, share: float, rng: np.random.Generator) -> dict:
    """One AR(1) world for I1.hybrid_tf: T1 decided by scorer.decide; T2 ratio reported."""
    HC, HM, HS = _hyb()
    county = np.array([c for c, n in T1_COUNTIES.items() for _ in range(n)])
    y = anomalies(county, rho, share, rng)                         # (W, 190) on 2012-01..
    cal_d = np.asarray(AXIS.month - 1)
    ref_sel = pd.Timestamp("2022-12-01") >= AXIS
    X = {}
    for c in T1_COUNTIES:
        series = {k: ar1_unit(1, T_ALL, DRIVER_RHO, rng)[0] for k in HC.DRIVERS}
        ref = {k: float(v[ref_sel].mean()) for k, v in series.items()}
        X[c] = np.c_[HM.features(series, HC.DRIVERS, ref)[1:], HM.month_dummies(cal_d[1:])]
    yh = y[:, 1:]                                                  # head months 2012-02..
    cal_h = cal_d[1:]
    fit = np.asarray(AXIS[1:] <= pd.Timestamp("2026-09-01"))
    W, T = yh.shape
    pred = {m: np.full((W, T), np.nan) for m in ("arx1", "stat_ar1_1", "persist1", "clim")}
    tf = np.full((W, T), np.nan)
    rhos, anom = [], []
    for w in range(W):
        clim = np.array([yh[w, fit & (cal_h == m)].mean() for m in range(12)])[cal_h]
        anom.append(yh[w] - clim)
        r, _ = HM.lag1(np.where(fit, anom[w], np.nan), fit)
        rhos.append(r)
        pred["clim"][w] = clim
    pooled = float(np.nanmedian(rhos))
    for w in range(W):
        Xw = X[county[w]]
        ok = fit
        a = anom[w]
        clim = pred["clim"][w]
        r = rhos[w] if np.isfinite(rhos[w]) else pooled
        tr = np.flatnonzero(ok[1:] & ok[:-1]) + 1
        arx = HM.Ridge().fit(Xw[tr], a[tr] - r * a[tr - 1])
        tf[w] = HM.Ridge().fit(Xw[ok], yh[w, ok]).predict(Xw)
        yp = np.r_[np.nan, yh[w, :-1]]
        cp = np.r_[np.nan, clim[:-1]]
        base = clim + r * (yp - cp)
        pred["persist1"][w] = yp
        pred["stat_ar1_1"][w] = base
        pred["arx1"][w] = base + arx.predict(Xw)
    window = np.asarray(AXIS[1:] >= pd.Timestamp("2026-11-01"))
    res = HS.decide(pred, yh, window, county, np.arange(W).astype(str))
    t2 = {}
    for c in T1_COUNTIES:
        sel = county == c
        e_tf = ((tf[sel] - yh[sel])[:, window] ** 2).sum()
        e_cl = ((pred["clim"][sel] - yh[sel])[:, window] ** 2).sum()
        t2[c] = float(np.sqrt(e_tf / e_cl))
    return {"outcome": res["outcome"],
            "pass": {c: bool(res["counties"][c].get("pass", False)) for c in T1_COUNTIES},
            "t2_ratio_vs_clim": t2}


# ----------------------------------------------------------------------------- score
def score_stat(*, k_stat: int = K_STAT, k_t1: int = K_T1, protocol: dict | None = None,
               protocol_path: Path | None = None, tmp_root: Path, smoke: bool = False,
               seed: int = SEED, decl_sha: str | None = None, log=print) -> dict:
    protocol = {**E04_PROTOCOL, **(protocol or {})}
    lay = e04_layout()
    rep: dict = {"id": ID, "version": VERSION, "smoke": smoke, "created": O._now(),
                 "declaration_sha256": decl_sha, "k_stat": k_stat, "k_t1": k_t1,
                 "seed": seed, "grid": {"rho": list(RHO_GRID), "share": list(SHARE_GRID)},
                 "builder_preview": BUILDER_PREVIEW if seed == SEED else None,
                 "mc_gate_note": V2.mc_note(len(cells()), k_stat),
                 "e04_layout": {"stations": len(lay.ids), "origins": E04_ORIGINS,
                                "counties": E04_COUNTIES}}
    if protocol_path is None:
        tmp_root.mkdir(parents=True, exist_ok=True)
        protocol_path = tmp_root / "protocol_e04.json"
        protocol_path.write_text(json.dumps({
            "schema": 1, "task": "free-running-head-validation", "test_start": "2026-11-01",
            "test_end": "2027-10-31", "registered_before": "2026-10-11",
            "independent_holdout": True, "model_sha256": "synthetic",
            "baseline": "synthetic climatology", **protocol}) + "\n")
    eng = {"evaluate_selfcheck": e04_selfcheck(lay, protocol_path, tmp_root, protocol, seed)}
    rec_cell = next(c for c in cells() if c["rho"] == 0.86 and c["share"] == 0.0)
    rhos = [e04_frame(lay, anomalies(lay.county, 0.86, 0.0,
                                     rng_for(STREAM_E04, rec_cell["cell"], k, seed))
                      )[1]["rho_mean"]
            for k in range(min(N_RECOVERY, k_stat))]
    eng["rho_recovery"] = {"mean_fitted_rho": float(np.mean(rhos)), "true": 0.86,
                           "tolerance": RHO_RECOVERY_TOL,
                           "pass": abs(float(np.mean(rhos)) - 0.86) <= RHO_RECOVERY_TOL}
    if not eng["rho_recovery"]["pass"]:
        raise EngineeringFailure(f"the simulation does not recover rho: {np.mean(rhos):.3f}")
    rep["engineering"] = eng
    log("gate E passed (engineering)")
    per = []
    for c in cells():
        ws = [e04_pass(e04_frame(lay, anomalies(lay.county, c["rho"], c["share"],
                                                rng_for(STREAM_E04, c["cell"], k, seed)))[0],
                       protocol) for k in range(k_stat)]
        per.append({**c, "passed": O._rate([w["pass"] for w in ws]),
                    "median_ratio": float(np.median([w["ratio"] for w in ws])),
                    "median_coverage": float(np.median([w["coverage"] for w in ws])),
                    "coverage_above_upper": float(np.mean([w["coverage"] > 0.95 for w in ws])),
                    "coverage_below_lower": float(np.mean([w["coverage"] < 0.85 for w in ws]))})
        log(f"cell rho {c['rho']} share {c['share']}: P {per[-1]['passed']['p']:.3f}")
    ps = [x["passed"] for x in per]
    se = math.sqrt(sum(x["mc_se"] ** 2 for x in ps)) / len(ps)
    p_all = float(np.mean([x["p"] for x in ps]))
    P = {"p": p_all, "mc_se": se, "mc_half_width_90": V2.Z_MC * se,
         "p_minus_threshold_in_se": (p_all - P_WEAK) / se if se > 0 else None,
         "min_over_cells": float(min(x["p"] for x in ps)),
         "by_rho": {str(r): float(np.mean([x["passed"]["p"] for x in per if x["rho"] == r]))
                    for r in RHO_GRID}}
    rep["e04_cells"] = per
    rep["P"] = P
    rd = V2.reading(P)
    rep["readings"] = {"threshold": P_WEAK, "rule": "P <= 0.5 -> weak_by_design; otherwise "
                       "not_shown_weak (one-sided, applied once)", "e04": rd,
                       "mc_ok": V2.mc_ok(P),
                       "wording": ({k: v.format(p=P["p"]) for k, v in WORDING[rd].items()}
                                   if rd else {k: v.format(why="Monte Carlo precision not met")
                                               for k, v in WORDING["not_evaluable"].items()})}
    t1 = []
    for c in cells():
        ws = [t1_world(c["rho"], c["share"], rng_for(STREAM_T1, c["cell"], k, seed))
              for k in range(k_t1)]
        t1.append({**c, "false_pass": {cc: O._rate([w["pass"][cc] for w in ws])
                                       for cc in T1_COUNTIES},
                   "outcome_not_no_skill": O._rate([w["outcome"] != "no added skill"
                                                    for w in ws]),
                   "t2_median_ratio_vs_clim": {cc: float(np.median([w["t2_ratio_vs_clim"][cc]
                                                                    for w in ws]))
                                               for cc in T1_COUNTIES}})
    rep["i1_t1_t2_report_only"] = {
        "cells": t1, "note": "T1's rate of 'adds skill' when no driver carries information "
                             "(its false-pass rate in AR(1) worlds) and T2's ratio to "
                             "climatology; report-only, no reading, no page line"}
    rep["status"] = "scored"
    return rep


def run_smoke(out: Path, k_stat: int = 8, k_t1: int = 2) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rep = score_stat(k_stat=k_stat, k_t1=k_t1, tmp_root=out / "tmp", smoke=True,
                     seed=SMOKE_SEED)
    rep["note"] = ("SMOKE: a different seed from the declared one, engineering only; numbers "
                   "at this K set nothing")
    O._write_json_once(out / "smoke_stat.json", rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-registration")
    sc = sub.add_parser("score", help="once (CPU, about 20-40 min)")
    sc.add_argument("--out", default=REPORT)
    sm = sub.add_parser("smoke")
    sm.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out))
            print(json.dumps({"status": rep["status"], "readings": rep["readings"]}, indent=1,
                             ensure_ascii=False, default=O._jsonable))
            return 0
        doc, dsha = V2.load_registered(ID, FROZEN)
        if a.cmd == "verify-registration":
            print(f"{ID} registered (sha256 {dsha[:16]}); frozen code and inputs intact")
            return 0
        if a.cmd == "score":
            out = O.rp(a.out)
            if out.exists():
                raise EngineeringFailure(f"{out} exists: {ID} is scored once")
            pp = O.rp(f"{HYB}/protocol.json")
            if O.sha256(pp) != (doc.get("e04") or {}).get("protocol_sha256"):
                raise EngineeringFailure("E04 protocol.json differs from the declared sha256")
            proto = json.loads(pp.read_text(encoding="utf-8"))
            rep = score_stat(protocol={k: proto[k] for k in E04_PROTOCOL}, protocol_path=pp,
                             tmp_root=out.parent, decl_sha=dsha)
            rep["frozen_code"] = doc["frozen_code"]
            O._write_json_once(out, rep)
            print(f"wrote {a.out} (status {rep['status']}; E04 reading "
                  f"{rep['readings']['e04']})")
            return 0
    except EngineeringFailure as err:
        print(f"ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
