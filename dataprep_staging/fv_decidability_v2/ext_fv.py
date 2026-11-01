"""FV.ext.v1 (fresh declaration, not post-failure; round-7 rank 12 rider): how often would
N3F's external-well level rule pass if a model of the right kind were the truth?

The worlds are FV.decidability.v2_postfailure's full forwards (same truths, same world draws,
same batch files), read at the 66 never-calibrated external wells kept by N3F's registered
rule (results/twin/prospective_2026-11_ext/wells.csv: cell and layer of the forward grid).
No extra forward is needed: heads_mean holds the full field.

World k of truth T at external well w:

    window obs(w, t) = ZR_ext(w, t) + D_T(w, t) + [h_T(world k) - h_T(baseline)](w, t) + e(w, t)
    pre-origin y(w, t) = M(w, t) + D_T(w, t) + e(w, t)                       t in 2012-22

ZR_ext = N3F's registered predictions; M = the registered forward's baseline hindcast at the
well's cell and layer plus its registered datum (wells.csv datum_m); D_T = (h_T - h_ZR0)
minus its mean over ALL 2012-22 months (the external wells have no observed-month mask
without a new look: declared difference from v1's anchor); e = one stationary AR(1) path
over 2012-01..2027-10 per world, sd = the registered residual sd (wells.csv residual_sd_m),
lag-1 = the county median of v1's noise inputs (overall median for Nantou).

Rule (N3F's, primary): E02's evaluation (equal-station RMSE ratio to the baseline <= 1.0 AND
pooled coverage in [0.85, 0.95], >= 30 wells with >= 6 months) with the baseline = each
world's OWN pre-origin month-of-year climatology (every pre-origin month observed: an
optimism, stated). Report-only: the same rule against the registered real-record baseline.
P(pass | right-ish) = equal-weight mean over the four truths; P <= 0.5 -> 'weak_by_design',
otherwise 'not_shown_weak' (one-sided, as FV v1).

Gates: E (registration, frozen code, N3F's export files as registered, kept wells, v2's
arm files, batch metadata and hindcast identity at the external cells, the declared null
tolerance recomputed, g0_e02 self-checked against twin.validation.evaluate with N3F's
protocol); N (null world obs = registered baseline + e, scored against it: false pass <= the
tolerance declared from RMS(ZR_ext - baseline) and the noise); MC (1.645 SE <= 0.07;
non-binding at 4 x 250, the largest possible half-width being 0.026).

Product use and design-time look: residual_sd_m, datum_m (wells.csv) and baseline_m
(predictions.csv) are derived from pre-origin heads (heads/L1_clean v3, through N3F's
registered export). This study uses them as 'design_time_evidence' (checked against the
product MANIFEST's allowed_uses at declaration). The lead logs that read as a design-time look
(``ext_fv.py log-look``, unit EXT_LOOK_UNIT) BEFORE the declaration's dry run, which is the
first step that reads those values (the null tolerance); until then the dry run leaves the
tolerance pending. N3F is pinned to its 2026-11 registration (the export directory and
the window are that registration's).
"""
from __future__ import annotations

import argparse
import json
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

import osse_fv as O  # noqa: E402
import osse_fv2 as V2  # noqa: E402
import score_e02c as S  # noqa: E402

ID = "FV.ext.v1"
WP = "FV"
VERSION = "1.0.0"
SELF = f"{V2.F}/ext_fv.py"
FROZEN = (SELF, V2.SELF, O.SELF, f"{S.E}/score_e02c.py", f"{S.E}/prereg_e02c_ext.py",
          f"{S.E}/export_baselines.py")
N3F_ID = "N3F.ext_level.prospective_2026-11.v1"   # pinned: EXT_DIR and S.WINDOW are its own
EXT_DIR = "results/twin/prospective_2026-11_ext"
PRODUCT = O.L1                                    # heads/L1_clean v3 (through N3F's export)
PRODUCT_USE = "design_time_evidence"
EXT_LOOK_UNIT = "FV:ext_preorigin_derived_N3Fwells"
EXT_LOOK_SCOPE = "heads:2012-01..2022-12"
EXT_LOOK_PURPOSE = "design_time_noise_inputs"
EXT_FILES = ("predictions.csv", "wells.csv", "protocol.json")
K_TRUTH = V2.K_TRUTH
K_NULL = V2.K_NULL
P_WEAK = V2.P_WEAK
STREAM_NOISE, STREAM_NULL, STREAM_SELFCHECK = 21, 22, 23
REPORT = f"results/dataprep/reports/{ID}/report.json"
EngineeringFailure = O.EngineeringFailure

WORDING = {
    "weak_by_design": {
        "en": "Synthetic decidability study (FV.ext.v1, model-only): a model of the right kind "
              "passes the external-well level rule with probability {p:.2f} (optimistic: rain "
              "at climatology, noise independent between wells, every month observed; not a "
              "strict bound), so this test is weak by design; a fail would not refute the "
              "forecast at the never-calibrated wells.",
        "zh": "合成可判定性研究（FV.ext.v1，純模型）：正確類型的模型通過外部井水位規則的機率為 "
              "{p:.2f}（偏樂觀：降雨取氣候值、各井雜訊彼此獨立、每月皆有觀測；並非嚴格界限），因此"
              "該檢驗在設計上即屬弱檢驗；未通過也不足以否定在未參與率定之井的預測。"},
    "not_shown_weak": {
        "en": "Synthetic decidability study (FV.ext.v1, model-only): a model of the right kind "
              "passes the external-well level rule with probability {p:.2f} (optimistic: rain "
              "at climatology, noise independent between wells, every month observed; not a "
              "strict bound); the study does not show this test weak and adds no other "
              "reading.",
        "zh": "合成可判定性研究（FV.ext.v1，純模型）：正確類型的模型通過外部井水位規則的機率為 "
              "{p:.2f}（偏樂觀：降雨取氣候值、各井雜訊彼此獨立、每月皆有觀測；並非嚴格界限）；本研究"
              "未顯示該檢驗為弱檢驗，也不提供其他解讀。"},
    "not_evaluable": {
        "en": "Synthetic decidability study FV.ext.v1 not evaluable ({why}); the external-well "
              "test (E22) gains no interpretation line from it.",
        "zh": "合成可判定性研究 FV.ext.v1 無法評估（{why}）；"
              "外部井檢驗（E22）不因此增加解讀說明。"},
}


@dataclass
class ExtPaths:
    v2: V2.Paths2 = field(default_factory=V2.Paths2)
    ext_dir: str = EXT_DIR
    report: str = REPORT


def load_ext(P: ExtPaths) -> dict:
    """N3F's registered export: window rows of the kept wells, their cell/layer, sd and
    datum, the protocol, and the lag-1 coefficients from v1's noise inputs."""
    d = O.rp(P.ext_dir)
    pred = pd.read_csv(d / "predictions.csv", dtype={"station_id": str})
    pred["date"] = S._month(pred["date"])
    win = S.window_months(*S.WINDOW)
    pred = pred[pred["date"].isin(win)].sort_values(["station_id", "date"])
    ids = sorted(pred["station_id"].unique())
    W = len(ids)
    if len(pred) != W * O.N_WIN:
        raise EngineeringFailure(f"ext predictions: {len(pred)} window rows, expected {W} x 12")
    wells = pd.read_csv(d / "wells.csv", dtype={"station_id": str})
    wells = wells[wells["kept"].astype(bool)].set_index("station_id")
    if sorted(wells.index) != ids:
        raise EngineeringFailure("ext wells.csv kept wells differ from the prediction wells")
    wells = wells.loc[ids]
    if (wells["cell"] < 0).any() or not wells["layer"].isin([0, 1, 2, 3]).all():
        raise EngineeringFailure("an external well has no forward cell or layer")
    sd = wells["residual_sd_m"].to_numpy(float)
    if not (np.isfinite(sd).all() and (sd > 0).all()):
        raise EngineeringFailure("an external well has no residual sd")
    noise = pd.read_csv(O.rp(P.v2.v1.inputs_dir) / "noise_ar1.csv", dtype={"station_id": str})
    county = np.array([S.county_of(s) for s in ids])
    overall = float(np.median(noise["phi"]))
    by_c = {c: float(np.median(noise.loc[noise["station_id"].map(S.county_of) == c, "phi"]))
            for c in S.COUNTIES_COV}
    phi = np.array([by_c.get(c, overall) for c in county])

    def mat(c):
        return pred.pivot(index="station_id", columns="date")[c].reindex(
            index=ids, columns=win).to_numpy(float)

    frame0 = pred[["station_id", "date", "prediction_origin", "predicted_m", "baseline_m",
                   "lower_m", "upper_m", "model_sha256"]].reset_index(drop=True)
    return {"ids": ids, "W": W, "county": county, "cells": wells["cell"].to_numpy(int),
            "layers": wells["layer"].to_numpy(int), "datum": wells["datum_m"].to_numpy(float),
            "sd": sd, "phi": phi, "zr": mat("predicted_m"), "base_reg": mat("baseline_m"),
            "frame0": frame0,
            "protocol": json.loads((d / "protocol.json").read_text(encoding="utf-8"))}


def synthetic_cl(y_pre: np.ndarray) -> np.ndarray:
    """(W, 12) month-of-year means of a (W, 132) pre-origin record starting 2012-01, laid
    out on the window months 2026-11..2027-10."""
    moy = np.arange(O.T_FIT) % 12
    clim = np.stack([y_pre[:, moy == m].mean(axis=1) for m in range(12)], axis=1)
    return clim[:, np.arange(O.WIN[0], O.WIN[1]) % 12]


def rule(ctx: dict, obs: np.ndarray, baseline: np.ndarray) -> dict:
    f = ctx["frame0"].copy()
    f["observed_m"] = np.asarray(obs, float).reshape(-1)
    f["baseline_m"] = np.asarray(baseline, float).reshape(-1)
    g = O.g0_e02(S.annotate(f), ctx["protocol"])
    return {"pass": bool(g["e02_passed"]), "ratio": g["ratio"], "coverage": g["coverage"]}


def score_ext(P: ExtPaths, *, plan: V2.Plan = V2.PLAN, smoke: bool = False,
              k_null: int = K_NULL, decl: dict | None = None, decl_sha: str | None = None,
              log=print) -> dict:
    rep: dict = {"id": ID, "version": VERSION, "smoke": smoke, "created": O._now(),
                 "declaration_sha256": decl_sha, "k_truth": plan.k_truth, "k_null": k_null,
                 "worlds": V2.ID, "mc_gate_note": V2.mc_note(len(O.TAGS), plan.k_truth)}
    ctx = load_ext(P)
    if decl is not None:
        for name, h in (decl.get("ext_files") or {}).items():
            if O.sha256(O.rp(P.ext_dir) / name) != h:
                raise EngineeringFailure(f"ext {name} differs from N3F's registered sha256")
        if decl.get("kept_ids") != ctx["ids"]:
            raise EngineeringFailure("kept external wells differ from the declaration")
    pool = O.pool_matrix(pd.read_csv(O.rp(P.v2.v1.inputs_dir) / "anomaly_pool.csv"))
    arms = V2.check_arms(P.v2.arms_dir, pool, plan)
    cells, layers = ctx["cells"], ctx["layers"]
    _, zr0 = O.well_heads(O.rp(P.v2.v1.zr0), cells, layers)
    h_zr0 = zr0[0]
    zm, zh = O.well_heads(O.rp(P.v2.v1.zr_forward), cells, layers)
    M = zh[[str(x) for x in zm["scenario_names"]].index("baseline")][:, :O.T_FIT] \
        + ctx["datum"][:, None]
    del zr0, zh
    T, eng = {}, {}
    for i, tag in enumerate(O.TAGS):
        ref, _ = V2.v1_reference(P.v2.v1_truths[tag], cells, layers, tag)
        lt = V2.load_truth(P.v2.batches[tag], tag, arms, cells, layers, ref, plan)
        diff = lt["base"][:, :O.T_RUN] - h_zr0[:, :O.T_RUN]
        D = diff - diff[:, :O.T_FIT].mean(axis=1, keepdims=True)
        T[tag] = {"i": i, "R": lt["R"], "D_pre": D[:, :O.T_FIT],
                  "D_win": D[:, O.WIN[0]:O.WIN[1]]}
        eng[tag] = lt["batches"]
        del ref, lt
    tol = O.null_tolerance(ctx["zr"] - ctx["base_reg"], ctx["sd"], ctx["phi"], k_null)
    if decl is not None:
        dtol = (decl.get("null_world") or {}).get("tolerance")
        if dtol is None or abs(dtol - tol["tolerance"]) > 1e-12:
            raise EngineeringFailure(f"null tolerance {tol['tolerance']} differs from the "
                                     f"declared {dtol}")
    sc = {"frame0": ctx["frame0"], "protocol": ctx["protocol"]}
    selfcheck = [O.evaluate_selfcheck(sc, ctx["base_reg"] + O.simulate_ar1(
        ctx["sd"], ctx["phi"], O.N_WIN, O.rng_for(STREAM_SELFCHECK, 0, k)),
        O.rp(P.ext_dir) / "protocol.json", O.rp(P.report).parent) for k in range(2)]
    rep["engineering"] = {"truths": eng, "evaluate_selfcheck": selfcheck,
                          "n_wells": ctx["W"], "by_county": {
                              c: int((ctx["county"] == c).sum())
                              for c in sorted(set(ctx["county"]))}}
    log("gate E passed (engineering)")
    nulls = [rule(ctx, ctx["base_reg"] + O.simulate_ar1(
        ctx["sd"], ctx["phi"], O.N_WIN, O.rng_for(STREAM_NULL, 0, k)), ctx["base_reg"])["pass"]
             for k in range(k_null)]
    fp = O._rate(nulls)
    rep["null_world"] = {"tolerance": tol, "false_pass": fp, "pass": fp["p"] <= tol["tolerance"]}
    if not rep["null_world"]["pass"]:
        return close(rep, "null-world false-pass rate above the declared tolerance")
    per = {}
    for tag in O.TAGS:
        t = T[tag]
        prim, reg = [], []
        for k in range(plan.k_truth):
            e = O.simulate_ar1(ctx["sd"], ctx["phi"], O.T_RUN,
                               O.rng_for(STREAM_NOISE, t["i"], k))
            cl = synthetic_cl(M + t["D_pre"] + e[:, :O.T_FIT])
            obs = ctx["zr"] + t["D_win"] + t["R"][k] + e[:, O.WIN[0]:O.WIN[1]]
            prim.append(rule(ctx, obs, cl))
            reg.append(rule(ctx, obs, ctx["base_reg"]))
        per[tag] = {"passed": O._rate([w["pass"] for w in prim]),
                    "median_ratio": float(np.median([w["ratio"] for w in prim])),
                    "median_coverage": float(np.median([w["coverage"] for w in prim])),
                    "report_only_registered_baseline": {
                        "passed": O._rate([w["pass"] for w in reg]),
                        "median_ratio": float(np.median([w["ratio"] for w in reg]))}}
        log(f"truth {tag}: P(pass) {per[tag]['passed']['p']:.3f}")
    rep["truth_worlds"] = per
    pooled = V2._pooled(per, lambda r: (r["passed"]["p"], r["passed"]["mc_se"]))
    rep["P"] = pooled
    rd = V2.reading(pooled)
    rep["readings"] = {"threshold": P_WEAK, "rule": "P <= 0.5 -> weak_by_design; otherwise "
                       "not_shown_weak (one-sided, applied once)", "ext_level": rd,
                       "mc_ok": V2.mc_ok(pooled),
                       "wording": ({k: v.format(p=pooled["p"]) for k, v in WORDING[rd].items()}
                                   if rd else {k: v.format(why="Monte Carlo precision not met")
                                               for k, v in WORDING["not_evaluable"].items()})}
    rep["status"] = "scored"
    return rep


def close(rep: dict, why: str) -> dict:
    rep.update({"status": "not_evaluable", "why": why,
                "readings": {"ext_level": None, "wording": {
                    k: v.format(why=why) for k, v in WORDING["not_evaluable"].items()}}})
    return rep


# ----------------------------------------------------------------------------- look (lead)
def _ext_shas(ext_dir: str) -> dict:
    d = O.rp(ext_dir)
    return {n: O.sha256(d / n) for n in ("wells.csv", "predictions.csv", "baselines.csv")}


def look_note(ext_dir: str = EXT_DIR) -> str:
    sh = _ext_shas(ext_dir)
    return (f"{ID} design-time read of N3F's registered export {ext_dir}: wells.csv "
            f"(residual_sd_m, datum_m) sha256 {sh['wells.csv']}; predictions.csv (baseline_m, "
            f"predicted_m) sha256 {sh['predictions.csv']}; baselines.csv sha256 "
            f"{sh['baselines.csv']}; use {PRODUCT_USE} of {PRODUCT}")


def look_logged(prereg_dir=None, ext_dir: str = EXT_DIR) -> dict | None:
    """The lead's design-time look on the export, if logged for these exact files."""
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    try:
        sha = _ext_shas(ext_dir)["wells.csv"]
    except OSError:
        return None
    for lk in prereg3.read_chained(d / "looks.jsonl"):
        if (lk.get("units") == [EXT_LOOK_UNIT] and lk.get("purpose") == EXT_LOOK_PURPOSE
                and sha in str(lk.get("note", ""))):
            return {k: lk.get(k) for k in ("time", "wp", "purpose", "units", "data_scope")}
    return None


def log_look(prereg_dir=None, rows_dir=None, ext_dir: str = EXT_DIR) -> dict:
    """Log the design-time look on N3F's export (the lead; real registry by default). Reads
    the kept station ids only."""
    from harness import prereg3

    w = pd.read_csv(O.rp(ext_dir) / "wells.csv", usecols=["station_id", "kept"],
                    dtype={"station_id": str})
    ids = sorted(w.loc[w["kept"].astype(bool), "station_id"])
    members = {EXT_LOOK_UNIT: {
        "ids": ids, "window": "2012-01..2022-12",
        "description": "per-well quantities derived from pre-origin heads (heads/L1_clean v3) "
                       "at N3F's 66 kept external wells, read from N3F's registered export "
                       "(residual_sd_m, datum_m, baseline_m) by FV.ext.v1's dry run (null "
                       "tolerance) and scorer (noise, pre-origin record); design-time "
                       "evidence, spent months (2012-22)"}}
    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    return prereg3.record_look([EXT_LOOK_UNIT], EXT_LOOK_SCOPE, WP, EXT_LOOK_PURPOSE,
                               members=members, note=look_note(ext_dir), **kw)


# ----------------------------------------------------------------------------- synthetic
def build_synthetic(out: Path, plan: V2.Plan, n_ext: int = 40, seed: int = 9) -> ExtPaths:
    """v2's synthetic worlds plus a SYNTHETIC N3F-style export at random cells and layers."""
    out = Path(out)
    p2 = V2.build_synthetic(out / "worlds", plan)
    rng = np.random.default_rng(seed)
    with np.load(p2.v1.zr_forward, allow_pickle=False) as z:
        H = np.asarray(z["heads_mean"], float)[0]
    W = n_ext
    ids = sorted([f"09{900000 + i:06d}" for i in range(W // 2)]
                 + [f"07{900000 + i:06d}" for i in range(W - W // 2)])
    cells = rng.integers(0, H.shape[1], W)
    layers = rng.integers(0, 4, W)
    datum = rng.normal(0.0, 1.0, W)
    sd = rng.uniform(0.3, 0.9, W)
    d = out / "ext"
    d.mkdir(parents=True)
    pd.DataFrame({"station_id": ids, "county": [S.county_of(s) for s in ids], "cell": cells,
                  "layer": layers, "residual_sd_m": sd, "datum_m": datum, "kept": True}
                 ).to_csv(d / "wells.csv", index=False)
    months = S.window_months(*S.WINDOW)
    rows = []
    t_fit = np.arange(O.T_FIT) % 12
    for w, s in enumerate(ids):
        h = H[layers[w], cells[w]]
        cl = np.array([h[:O.T_FIT][t_fit == m].mean() for m in range(12)]) + datum[w]
        for j, m in enumerate(months):
            t = O.WIN[0] + j
            zr = h[t] + datum[w]
            sig = np.sqrt(sd[w] ** 2 + 0.04)
            rows.append({"station_id": s, "layer": int(layers[w]) + 1,
                         "date": f"{m:%Y-%m-%d}", "prediction_origin": "2022-12-01",
                         "predicted_m": zr, "baseline_m": cl[t % 12] + 0.3,
                         "lower_m": zr - O.Z90 * sig, "upper_m": zr + O.Z90 * sig,
                         "model_sha256": "synthetic"})
    pd.DataFrame(rows).to_csv(d / "predictions.csv", index=False)
    proto = json.loads(Path(p2.v1.protocol).read_text())
    proto["min_stations"] = 30
    (d / "protocol.json").write_text(json.dumps(proto, indent=1) + "\n")
    return ExtPaths(v2=p2, ext_dir=str(d), report=str(out / "report_ext.json"))


def run_smoke(out: Path, k_truth: int = 6, batch: int = 4, k_null: int = 20) -> dict:
    out = Path(out)
    plan = V2.Plan(k_truth=k_truth, batch=batch)
    P = build_synthetic(out / "inputs_synthetic", plan)
    rep = score_ext(P, plan=plan, smoke=True, k_null=k_null)
    rep["note"] = "SMOKE: synthetic inputs only; numbers mean nothing about the twin"
    O._write_json_once(out / "smoke_ext.json", rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-registration")
    sub.add_parser("log-look", help="the lead, BEFORE the declaration's dry run: log the "
                                    "design-time read of N3F's export")
    sc = sub.add_parser("score", help="once, after FV.decidability.v2's forwards")
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
        if a.cmd == "log-look":
            if look_logged():
                raise EngineeringFailure("the design-time look is already logged")
            rec = log_look()
            print(f"logged look {EXT_LOOK_UNIT} at {rec.get('time')}")
            return 0
        doc, dsha = V2.load_registered(ID, FROZEN)
        if a.cmd == "verify-registration":
            print(f"{ID} registered (sha256 {dsha[:16]}); frozen code and inputs intact")
            return 0
        if a.cmd == "score":
            out = O.rp(a.out)
            if out.exists():
                raise EngineeringFailure(f"{out} exists: {ID} is scored once")
            P = ExtPaths()
            V2.check_attempts(O.rp(P.v2.attempts), {
                (t, b): O.rp(p) for t, ps in P.v2.batches.items() for b, p in enumerate(ps)})
            rep = score_ext(P, decl=doc, decl_sha=dsha)
            rep["frozen_code"] = doc["frozen_code"]
            O._write_json_once(out, rep)
            print(f"wrote {a.out} (status {rep['status']}; reading "
                  f"{rep['readings']['ext_level']})")
            return 0
    except EngineeringFailure as err:
        print(f"ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
