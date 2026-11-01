"""U1.decomp_power.v1: synthetic P-gate of E02d's per-county G-idio test (round 7, rank 5)
and its 12-month re-run for the year-2/3 windows (rank 6).

Synthetic only: no observation, residual or window head is opened. The design is derived by
a declared formula from two pre-origin artefacts of DR.relative_drift.v1 (its report and
the frozen wells_e02d.csv: groups, sub-groups and the relative scale of sd_fit_rel at the
158 wells), so the ICC grid is BOUNDED by the pre-origin estimates plus declared margins and
is not chosen by the designers (who know the 2023 distal drift). The top sub-group share is
max(rem + AS_MARGIN, DR's well x month-block bootstrap upper-90 % of rem), capped: the
margin floor stands for in-sample damping (which biases rem low, as it biases tau low), the
bootstrap for sampling uncertainty. ``design_spec`` is the one
source of the design; ``run`` refuses unless the registered declaration carries exactly it.

World (per county, independently; replicate d, well w, month t):

    x[w, t] = s_w ( sqrt(a_s) S[sub(w), t] + sqrt(a_w) A[w] + sqrt(1 - a_s - a_w) U[w, t] )
              + COMMON_SCALE * C[t]

C a county-month shock (cancels exactly in the LOO operator; kept as an engineering check),
S a sub-group-month shock shared by the wells of the other partition's group (what survives
the chosen grouping), A a persistent per-well offset over the window (lead drift), U a
stationary AR(1) with phi; all N(0, 1). s_w = DR's sd_fit_rel (its relative pattern). The
heavy arm (report-only) replaces A by a well type: each well 'bad' with P_BAD, its scale
times SCALE_BAD. Wells drop out for the whole window with probability ``dropout``.

Scorer: EXACTLY E02d's (e02dlib.relative_hits on the DR groups, MIN_OTHERS, then
e02dlib.county_verdict: >= 6 months, >= 20 wells, coverage3 gate-3 clusters = wells AND
point coverage in [0.85, 0.95]); the vectorised path is checked against county_verdict
(self-check). True coverage p: the frozen half-width z s_w k_p with k_p the p-quantile of
|e| / (z s_w) over a calibration sample of the same cell WITHOUT dropout (so dropout's extra
LOO noise is a genuine source of false fails).

Decision (per county, per month count): FF = max P(FAIL) over gating cells at p = 0.90;
PW = min P(FAIL) over gating cells at p = 0.75; P-gate passes iff FF <= 0.20 and PW >= 0.80.
Window A (11 months): E02d is registered only if BOTH counties pass, else C14a-r is N
('undecidable even after removing the common mode'). 12 months (B, C): the relative year-2/3
test of a county is primary only if that county passes, otherwise report-only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02dio as io  # noqa: E402
import e02dlib as L  # noqa: E402

VERSION = "1.0.0"
CODE_FILES = (f"{io.X}/power_e02d.py", f"{io.X}/e02dlib.py", f"{io.X}/e02dio.py",
              "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/coverage2.py",
              "dataprep_staging/harness/coverage.py")
RESULT_DIR = f"results/dataprep/reports/{L.POWER_ID}"
DR_DIR = f"results/dataprep/reports/{L.DR_ID}"
MONTHS = (11, 12)                 # window A; windows B and C (same wells, same months)
COVERAGE = (0.75, 0.80, 0.90)
DROPOUT = (0.0, 0.2, 0.4)
AS_MARGIN, AS_CAP = 0.05, 0.5
AW_MARGIN, AW_CAP = 0.2, 0.7
PHI_CAP = 0.95
P_BAD, SCALE_BAD = 0.10, 3.0
COMMON_SCALE = 3.0
FALSE_FAIL_MAX = 0.20
POWER_MIN = 0.80
POWER_AT = 0.75
REPLICATES = 2000
CAL_REPLICATES = 500
SEED = 20261103
SELF_CHECK_SEED = 20261104
SMOKE = {"replicates": 60, "cal_replicates": 40, "seed": 1}


def _uniq(xs) -> list[float]:
    out = []
    for x in xs:
        x = round(float(x), 6)
        if x not in out:
            out.append(x)
    return sorted(out)


def county_design(dr: dict, wells: pd.DataFrame, county: str) -> dict:
    """Declared derivation of one county's grid from DR (pre-origin)."""
    c = dr["counties"][county]
    w = wells[(wells["county"] == county) & np.isfinite(wells["sd_fit_rel_m"])]
    s = w["sd_fit_rel_m"].to_numpy(float)
    rem = float(c["icc_remaining"]["icc"]) if math.isfinite(
        c["icc_remaining"]["icc"]) else 0.0
    up = c["icc_remaining"].get("boot_upper90")
    up = float(up) if up is not None and math.isfinite(float(up)) else 0.0
    a_s_top = min(AS_CAP, max(rem + AS_MARGIN, up))
    aw = float(c.get("window_well_share_hat") or 0.0)
    phi = float(c["drift"].get("phi_median") or 0.0)
    return {
        "wells": int(len(w)), "groups": w["group"].astype(str).tolist(),
        "subgroups": w["subgroup"].astype(str).tolist(),
        "scale": (s / np.median(s)).round(6).tolist(),
        "a_s": _uniq([0.0, min(AS_CAP, rem), a_s_top]),
        "a_w": _uniq([0.0, min(AW_CAP, aw), min(AW_CAP, aw + AW_MARGIN)]),
        "phi": _uniq([0.0, min(PHI_CAP, max(0.0, phi))]),
        "from_dr": {"icc_remaining": rem, "icc_remaining_boot_upper90": up,
                    "window_well_share_hat": aw, "phi_median": phi,
                    "grouping": c["grouping"]["grouping"]}}


def design_spec(dr: dict, wells: pd.DataFrame, dr_sha: str = "", wells_sha: str = "") -> dict:
    return io.jsonable({
        "version": VERSION, "dr_report_sha256": dr_sha, "wells_e02d_sha256": wells_sha,
        "counties": {c: county_design(dr, wells, c) for c in L.COUNTIES},
        "months": list(MONTHS), "coverage": list(COVERAGE), "dropout": list(DROPOUT),
        "margins": {"a_s": [AS_MARGIN, AS_CAP], "a_w": [AW_MARGIN, AW_CAP], "phi_cap": PHI_CAP},
        "heavy_report_only": {"p_bad": P_BAD, "scale_bad": SCALE_BAD,
                              "cells": "a_s = max, phi = max, dropout {0, 0.4}, coverage "
                                       "{0.75, 0.90}"},
        "common_scale": COMMON_SCALE, "replicates": REPLICATES,
        "cal_replicates": CAL_REPLICATES, "seed": SEED, "self_check_seed": SELF_CHECK_SEED,
        "scorer": {"min_others": L.MIN_OTHERS, "min_months": L.MIN_MONTHS,
                   "min_wells": L.MIN_WELLS_COUNTY, "gate": L.gate_kwargs(),
                   "point_window_required": True, "z": L.Z90},
        "decision": {"false_fail_max": FALSE_FAIL_MAX, "power_min": POWER_MIN,
                     "power_at": POWER_AT, "fail": "evaluable AND NOT (gate-3 pass AND point "
                                                   "coverage in [0.85, 0.95])",
                     "not_evaluable": "reported; it is neither a fail nor a pass"},
        "rng": "np.random.default_rng(SeedSequence(seed, spawn_key=(county index, months, "
               "cell index, 0 draws | 1 calibration)))"})


def cells(cd: dict) -> list[dict]:
    out = []
    for a_s in cd["a_s"]:
        for a_w in cd["a_w"]:
            for phi in cd["phi"]:
                for drop in DROPOUT:
                    for p in COVERAGE:
                        out.append({"a_s": a_s, "a_w": a_w, "phi": phi, "dropout": drop,
                                    "coverage": p, "heavy": False, "gating": True})
    for drop in (0.0, 0.4):
        for p in (0.75, 0.90):
            out.append({"a_s": max(cd["a_s"]), "a_w": 0.0, "phi": max(cd["phi"]),
                        "dropout": drop, "coverage": p, "heavy": True, "gating": False})
    for i, c in enumerate(out):
        c["cell"] = i
    return out


# ----------------------------------------------------------------------------- world
def draw(rng, D: int, T: int, cd: dict, cell: dict, dropout: float) -> np.ndarray:
    """(D, W, T) synthetic residuals of one county (NaN = dropped well)."""
    s = np.asarray(cd["scale"], float)[None, :, None]
    W = s.shape[1]
    subs = pd.factorize(np.asarray(cd["subgroups"]))[0]
    a_s, a_w, phi = cell["a_s"], cell["a_w"], cell["phi"]
    C = rng.standard_normal((D, 1, T))
    S = rng.standard_normal((D, subs.max() + 1, T))[:, subs, :]
    if cell["heavy"]:
        bad = rng.random((D, W, 1)) < P_BAD
        A = np.zeros((D, W, 1))
        mult = np.where(bad, SCALE_BAD, 1.0)
        a_w = 0.0
    else:
        A = rng.standard_normal((D, W, 1))
        mult = 1.0
    eps = rng.standard_normal((D, W, T))
    U = np.empty_like(eps)
    U[..., 0] = eps[..., 0]
    q = math.sqrt(max(0.0, 1.0 - phi ** 2))
    for t in range(1, T):
        U[..., t] = phi * U[..., t - 1] + q * eps[..., t]
    r = math.sqrt(max(0.0, 1.0 - a_s - a_w))
    x = s * mult * (math.sqrt(a_s) * S + math.sqrt(a_w) * A + r * U) + COMMON_SCALE * C
    if dropout > 0:
        keep = rng.random((D, W, 1)) >= dropout
        x = np.where(keep, x, np.nan)
    return x


def calibrate(rng, T: int, cd: dict, cell: dict, n: int) -> float:
    """k_p: the p-quantile of |e| / (z s_w) without dropout."""
    x = draw(rng, n, T, cd, cell, 0.0)
    e = L.relative(x, np.asarray(cd["groups"]))
    s = np.asarray(cd["scale"], float)[None, :, None]
    u = (np.abs(e) / (L.Z90 * s))[np.isfinite(e)]
    return float(np.quantile(u, cell["coverage"]))


def score_batch(x: np.ndarray, cd: dict, k: float) -> dict:
    """Vectorised E02d verdicts of a batch (D, W, T): every scored well has all T months
    (whole-well dropout), so gate-3 runs on per-well totals, batched by the well count."""
    from harness import coverage3

    s = np.asarray(cd["scale"], float)[None, :, None]
    hits, _, _ = L.relative_hits(x, s * k + 0 * x, np.asarray(cd["groups"]))
    D, W, T = hits.shape
    nm = np.isfinite(hits).sum(axis=2)
    scored = nm >= L.MIN_MONTHS
    kw = scored.sum(axis=1)
    fail = np.zeros(D, bool)
    ne = kw < L.MIN_WELLS_COUNTY
    cov = np.full(D, np.nan)
    g = L.gate_kwargs()
    for kk in np.unique(kw[~ne]):
        ds = np.flatnonzero((kw == kk) & ~ne)
        tot = np.nansum(np.where(scored[ds][:, :, None], hits[ds], 0.0), axis=2)
        Y = np.stack([tot[i][scored[d]] for i, d in enumerate(ds)])
        m = np.stack([nm[d][scored[d]] for d in ds]).astype(float)
        if not np.all(m == m[0]):
            raise io.EngineeringFailure("unequal month counts in a batch")
        r = coverage3.gate_from_totals_batch(Y, m[0], g["method"], g["target"], g["level"],
                                            g["window"], g["n_eff_window_min"], g["conf"])
        inw = (r["p"] >= L.POINT_WINDOW[0]) & (r["p"] <= L.POINT_WINDOW[1])
        fail[ds] = ~(r["pass"] & inw)
        cov[ds] = r["p"]
    return {"fail": fail & ~ne, "not_evaluable": ne, "coverage": cov, "hits": hits}


def run_cell(cd: dict, cell: dict, T: int, ci: int, reps: int, cal: int, seed: int) -> dict:
    rc = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(ci, T, cell["cell"], 1)))
    k = calibrate(rc, T, cd, cell, cal)
    rd = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(ci, T, cell["cell"], 0)))
    fails, nes, covs = [], [], []
    done = 0
    while done < reps:
        n = min(500, reps - done)
        sb = score_batch(draw(rd, n, T, cd, cell, cell["dropout"]), cd, k)
        fails.append(sb["fail"])
        nes.append(sb["not_evaluable"])
        covs.append(sb["coverage"])
        done += n
    f, ne, cv = np.concatenate(fails), np.concatenate(nes), np.concatenate(covs)
    p = float(f.mean())
    return {**cell, "months": T, "k_p": k, "replicates": reps, "P_fail": p,
            "mc_se": math.sqrt(max(p * (1 - p), 0.0) / reps),
            "P_not_evaluable": float(ne.mean()), "mean_coverage_hat": float(np.nanmean(cv))}


def self_check(cd: dict, T: int = 11, n: int = 12) -> dict:
    """The vectorised verdicts equal e02dlib.county_verdict on the same replicates, and the
    county-month shock cancels (C scaled x0 vs x3 gives identical hits)."""
    rng = np.random.default_rng(SELF_CHECK_SEED)
    cell = {"a_s": max(cd["a_s"]), "a_w": max(cd["a_w"]), "phi": max(cd["phi"]),
            "heavy": False, "coverage": 0.85}
    x = draw(rng, n, T, cd, cell, 0.3)
    sb = score_batch(x, cd, 1.0)
    mism = 0
    for d in range(n):
        v = L.county_verdict(sb["hits"][d], np.ones(x.shape[1], bool))
        fail_ref = v["verdict"] == "fail"
        ne_ref = v["verdict"] == "not_evaluable"
        differ = (fail_ref != bool(sb["fail"][d]) or ne_ref != bool(sb["not_evaluable"][d])
                  or (not ne_ref and abs(v["coverage"] - sb["coverage"][d]) > 1e-12))
        mism += int(differ)
    rng2 = np.random.default_rng(SELF_CHECK_SEED + 1)
    y = draw(rng2, 4, T, cd, cell, 0.0)
    shift = 7.0 * np.random.default_rng(3).standard_normal((4, 1, T))
    h1 = score_batch(y, cd, 1.0)["hits"]
    h2 = score_batch(y + shift, cd, 1.0)["hits"]
    same = bool(np.array_equal(np.isnan(h1), np.isnan(h2))
                and np.array_equal(np.nan_to_num(h1), np.nan_to_num(h2)))
    return {"parity_mismatches": mism, "replicates": n, "common_mode_cancels": same,
            "ok": mism == 0 and same}


def decide(rows: list[dict]) -> dict:
    g = [r for r in rows if r["gating"]]
    ff = [r for r in g if abs(r["coverage"] - 0.90) < 1e-9]
    pw = [r for r in g if abs(r["coverage"] - POWER_AT) < 1e-9]
    wf = max(ff, key=lambda r: r["P_fail"])
    wp = min(pw, key=lambda r: r["P_fail"])
    ok = wf["P_fail"] <= FALSE_FAIL_MAX and wp["P_fail"] >= POWER_MIN
    return {"false_fail_max": wf["P_fail"], "false_fail_cell": wf["cell"],
            "power_min_at_0.75": wp["P_fail"], "power_cell": wp["cell"],
            "power_min_at_0.80_report_only": min(
                (r["P_fail"] for r in g if abs(r["coverage"] - 0.80) < 1e-9), default=None),
            "heavy_report_only": {r["cell"]: r["P_fail"] for r in rows if r["heavy"]},
            "p_gate_pass": bool(ok)}


def run(design: dict, reps: int, cal: int, seed: int, *, smoke: bool, log=print) -> dict:
    t0 = time.time()
    res = {"id": L.POWER_ID, "smoke": smoke, "design": design, "seed": seed,
           "replicates": reps, "cal_replicates": cal,
           "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "code_sha256": {f: io.sha256(io.REPO / f) for f in CODE_FILES},
           "self_check": {}, "results": {}, "decisions": {}}
    for c in L.COUNTIES:
        res["self_check"][c] = self_check(design["counties"][c])
    res["engineering_failure"] = not all(v["ok"] for v in res["self_check"].values())
    if res["engineering_failure"]:
        return res
    for ci, c in enumerate(L.COUNTIES):
        cd = design["counties"][c]
        for T in MONTHS:
            rows = [run_cell(cd, cell, T, ci, reps, cal, seed) for cell in cells(cd)]
            res["results"].setdefault(c, {})[str(T)] = rows
            res["decisions"].setdefault(c, {})[str(T)] = decide(rows)
            log(f"{c} {T} months: {res['decisions'][c][str(T)]['p_gate_pass']} "
                f"({time.time() - t0:.0f} s)")
    res["e02d_registrable"] = all(res["decisions"][c]["11"]["p_gate_pass"]
                                  for c in L.COUNTIES)
    res["years23_relative_primary"] = {c: res["decisions"][c]["12"]["p_gate_pass"]
                                       for c in L.COUNTIES}
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def load_dr(dr_dir: Path) -> tuple[dict, pd.DataFrame, str, str]:
    rp, wp = dr_dir / "report.json", dr_dir / "wells_e02d.csv"
    dr = json.loads(rp.read_text(encoding="utf-8"))
    if dr.get("engineering_failure") or "counties" not in dr:
        raise io.Refused("DR.relative_drift.v1 has no valid report")
    if io.sha256(wp) != dr.get("wells_e02d_sha256"):
        raise io.EngineeringFailure("wells_e02d.csv differs from DR's report")
    return dr, pd.read_csv(wp, dtype={"station_id": str}), io.sha256(rp), io.sha256(wp)


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="tiny replicate counts, smoke seed (sets nothing)")
    s.add_argument("--dr-dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="the registered study (refuses unless registered)")
    r.add_argument("--dr-dir", type=Path, default=io.REPO / DR_DIR)
    r.add_argument("--out", type=Path, default=io.REPO / RESULT_DIR)
    a = ap.parse_args(argv)
    try:
        dr, wells, dsha, wsha = load_dr(a.dr_dir)
        design = design_spec(dr, wells, dsha, wsha)
        if a.cmd == "smoke":
            out = a.out.resolve()
            if out == (io.REPO / RESULT_DIR).resolve() or \
                    (io.REPO / RESULT_DIR).resolve() in out.parents:
                raise io.Refused("a smoke run never writes into the registered result dir")
            res = run(design, SMOKE["replicates"], SMOKE["cal_replicates"], SMOKE["seed"],
                      smoke=True)
        else:
            doc, sha = io.registered(L.POWER_ID, prereg_dir, ())
            if doc.get("design") != design:
                raise io.Refused("the registered design differs from design_spec(DR)")
            now = {f: io.sha256(io.REPO / f) for f in CODE_FILES}
            if doc.get("code_sha256") != now:
                raise io.Refused("bound code changed since registration")
            if (a.out / "result.json").exists():
                raise io.Refused("result.json exists (run once)")
            res = run(design, REPLICATES, CAL_REPLICATES, SEED, smoke=False)
            res["declaration"] = {"id": L.POWER_ID, "sha256": sha}
        h = io.write_once(a.out / ("smoke_result.json" if a.cmd == "smoke" else "result.json"),
                          res)
        print(f"wrote {a.out} sha256 {h}; engineering failure {res['engineering_failure']}")
        for c, d in res.get("decisions", {}).items():
            print(c, {T: v["p_gate_pass"] for T, v in d.items()})
        return 1 if res["engineering_failure"] else 0
    except io.Refused as exc:
        print(f"REFUSED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
