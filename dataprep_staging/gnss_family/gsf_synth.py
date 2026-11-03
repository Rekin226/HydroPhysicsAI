"""G-sep and G0 engines of X5.gnss_seasonal_family.v1: model-only, synthetic. Nothing here
reads a file or a GNSS value; the runner (gsf_gates.py) passes the frozen model side's
series. Nothing runs on import.

Stations: {station: {"zone": "distal"|"mid", "family": "ngl"|"wra", "m": months,
"mod": {member: monthly model subsidence (m)}}} - the primary unit of the frozen v1 model side.

World (one true member t, one cell):

    obs_s(t) = f_s exp(e_s) mod_t,s(t)                    frame scale and model error
               + L u_s cos(w t - phi)                     regional loading (coast-only cell:
                                                          distal stations only)
               - T cos(w t - w_July)                      monument thermal term
               - P cos(w t - phi_t)                       the spec's proximal seasonal truth:
                                                          real seasonal subsidence at the
                                                          reference stations, removed with the
                                                          common mode (stress cell)
               + D_s ramp_2021(t)                         2021 record-low extra compaction
                                                          (stress cell; WRA stations)
               + AR(1) noise [+ an uncorrected WRA jump]

f_s: one s ~ U(1, 1.42) per world for every station (base cells), or distal 1.42 / mid 1.0
(zone_frame cell); e_s ~ N(0, sigma_me) per station and world; u_s ~ U(0.3, 1.0); phi
uniform per world; phi_t the phase of the true member's median mid-zone annual vector;
P ~ U(2, 4) mm; D_s ~ U(5, 20) mm distal, U(2.5, 10) mm mid, ramping over 2021-03..2021-06
and held. Each station-month is dropped with probability 0.10 once per cell and truth (obs and
model then use the kept months, as the scorer does). No jump rule is applied (the gated
estimator has none beyond the product's flags; worlds keep every jump).

G0 cells: v1's grid (loading 0/3/6 mm x thermal 0/2 mm x sigma 2/4 mm x jumps off/TKJS) x
model error 0.2/0.4 (48 cells, frame U(1, 1.42)), plus four stress cells x sigma 2/4 x model
error 0.2/0.4 on loading 3 mm, thermal 2 mm, TKJS jumps (16 cells): proximal_seasonal_truth,
zone_frame, record_low_2021, coast_loading (6 mm, distal stations only). Each cell runs every
screen member as the truth.

G0 PASS (declared before it runs; gsf_const.G0_PASS): true member retained >= 0.90 in every
cell (nuisance cell x truth); members more than x1.5 off the truth in cut30 excluded >= 0.80
pooled (mean over cells with a far member, each weighted equally) and >= 0.60 in the worst
such cell; P(all members excluded) <= 0.10 in every cell (implied by the first criterion,
checked anyway). n_rep 1000 per cell (review B1).

G-sep PASS (the LEAD's design choice after review B2, not the plan's): on noise-free model
points, every far member differs from EVERY cluster member by more than hw_T1 + log 1.25 on
T1, or by more than hw_T2 + log 1.25 + log 1.42 on T2 (the screen's own T2 band); hw is the
largest, over GSEP_CELLS, of the median 90 % half-width of log(T_obs / T_page) in page-truth
worlds. On the frozen model side the T1 route needs hw_T1 below about 0.08-0.13, so a pass
is expected through T2 only (the frame-carrying statistic; gsf_const DECIDED_BY_T2).
"""
from __future__ import annotations

import itertools
import math

import gsf_const as K
import gsf_estimator as F
import numpy as np


def _E():
    import gsc_estimator as E

    return E


#: independent random streams of the one declared seed (SeedSequence([seed, stream]))
STREAM_GSEP = K.STREAM_GSEP
STREAM_G0 = K.STREAM_G0


def ar1(rng, n_rep: int, n: int, sigma: float, rho: float = K.RHO) -> np.ndarray:
    e = rng.normal(0.0, sigma * math.sqrt(1 - rho ** 2), size=(n_rep, n))
    x = np.empty_like(e)
    x[:, 0] = rng.normal(0.0, sigma, size=n_rep)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def cells() -> list[dict]:
    """The declared G0 nuisance cells (64)."""
    out = []
    for L, T, sg, jp, me in itertools.product(K.LOADING_MM, K.THERMAL_MM, K.SIGMA_MM, K.JUMPS,
                                              K.MODEL_ERR):
        out.append({"stress": "none", "loading_mm": L, "thermal_mm": T, "sigma_mm": sg,
                    "jumps": jp, "model_err": me})
    for st, sg, me in itertools.product(K.STRESS, K.SIGMA_MM, K.MODEL_ERR):
        out.append({"stress": st, **K.STRESS_BASE, "sigma_mm": sg, "model_err": me})
    return out


def cell_key(c: dict) -> str:
    return (f"{c['stress']}|L{c['loading_mm']:g}|T{c['thermal_mm']:g}|s{c['sigma_mm']:g}|"
            f"{c['jumps']}|me{c['model_err']:g}")


def _ym(ym: str) -> int:
    return 12 * int(ym[:4]) + int(ym[5:7]) - 1


def truth_phase(stations: dict, truth: str) -> float:
    """Phase (radians) of the true member's median annual vector over mid stations."""
    vs = []
    for st in stations.values():
        if st["zone"] != "mid":
            continue
        _, V = F.year_vectors(st["m"], st["mod"][truth])
        if len(V):
            vs.append(F.median_vector(V))
    a, b = np.median(np.array(vs), axis=0) if vs else (1.0, 0.0)
    return math.atan2(b, a)


def worlds(stations: dict, truth: str, cell: dict, rng, n_rep: int, keep: dict,
           phi_t: float) -> dict:
    """{station: (n_rep, n_kept) synthetic observed subsidence (m)} for one cell."""
    phase = rng.uniform(0, 2 * math.pi, size=n_rep)
    s_world = rng.uniform(*K.FRAME_UNIFORM, size=n_rep)
    P = rng.uniform(*K.REF_SEASONAL_MM, size=n_rep) / 1000.0
    tpk = 2 * math.pi * (K.THERMAL_PEAK_MONTH + 0.5) / 12.0
    r0, r1 = _ym(K.RECORD_LOW_RAMP[0]), _ym(K.RECORD_LOW_RAMP[1])
    out = {}
    for s in sorted(keep):
        st = stations[s]
        m = np.asarray(st["m"])[keep[s]]
        ang = 2 * math.pi * ((m % 12) + 0.5) / 12.0
        base = np.asarray(st["mod"][truth], float)[keep[s]]
        zf = cell["stress"] == "zone_frame"
        f = np.full(n_rep, K.ZONE_FRAME[st["zone"]]) if zf else s_world
        err = np.exp(rng.normal(0.0, cell["model_err"], size=n_rep))
        y = (f * err)[:, None] * base[None, :]
        L = cell["loading_mm"] / 1000.0
        if cell["stress"] == "coast_loading":
            L = K.COAST_LOADING_MM / 1000.0 if st["zone"] == "distal" else 0.0
        surv = rng.uniform(*K.SURVIVE, size=n_rep)
        y += L * surv[:, None] * np.cos(ang[None, :] - phase[:, None])
        y -= (cell["thermal_mm"] / 1000.0) * np.cos(ang - tpk)[None, :]
        if cell["stress"] == "proximal_seasonal_truth":
            y -= P[:, None] * np.cos(ang - phi_t)[None, :]
        if cell["stress"] == "record_low_2021" and st["family"] == "wra":
            lo, hi = K.RECORD_LOW_MM[st["zone"]]
            D = rng.uniform(lo, hi, size=n_rep) / 1000.0
            ramp = np.clip((m - r0) / (r1 - r0), 0.0, 1.0) * (m >= r0)
            y += D[:, None] * ramp[None, :]
        y += ar1(rng, n_rep, m.size, cell["sigma_mm"] / 1000.0)
        if cell["jumps"] == "tkjs" and st["family"] == "wra":
            has = rng.random(n_rep) < K.P_JUMP
            size = rng.uniform(*K.JUMP_MM, size=n_rep) / 1000.0 * rng.choice([-1, 1], n_rep)
            at = rng.integers(1, m.size, size=n_rep)
            y += (np.arange(m.size)[None, :] >= at[:, None]) * (has * size)[:, None]
        out[s] = y
    return out


def run_cell(stations: dict, members: list[str], truth: str, cell: dict, rng, n_rep: int,
             B: int, far: list[str], cluster_far: dict) -> dict:
    """One cell x truth: retention of the truth, exclusion of far members, P(all excluded),
    survivor-set readings (report-only)."""
    E = _E()
    keep = {s: rng.random(len(st["m"])) >= K.P_MISS for s, st in stations.items()}
    preps = {s: E.prepare(np.asarray(st["m"])[keep[s]]) for s, st in stations.items()}
    ok = [s for s in sorted(stations) if len(preps[s]["years"]) >= K.MIN_YEARS]
    keep = {s: keep[s] for s in ok}
    phi_t = truth_phase(stations, truth)
    W = worlds(stations, truth, cell, rng, n_rep, keep, phi_t)
    obs = {s: E.coefficients(preps[s], W[s]) for s in ok}               # (n_rep, Y, 2)
    mod = {s: {mb: E.coefficients(preps[s], np.asarray(stations[s]["mod"][mb],
                                                         float)[keep[s]])
               for mb in members} for s in ok}
    pairs = {s: {"years": preps[s]["years"], "obs": obs[s][0], "mod": mod[s]} for s in ok}
    stk = F.stack(pairs, members, {s: stations[s]["zone"] for s in ok})
    st_order = stk["stations"]
    it = members.index(truth)
    jf = [members.index(x) for x in far]
    n_true = n_far = n_all = 0
    rd = {k: 0 for k in F.READINGS}
    for i in range(n_rep):
        for k, s in enumerate(st_order):
            n = len(preps[s]["years"])
            stk["V"][0, k, :n] = obs[s][i]
        ex = F.screen_fast(stk, rng, B)
        n_true += not ex[it]
        n_far += int(ex[jf].sum()) if jf else 0
        n_all += bool(ex.all())
        surv = [mb for j, mb in enumerate(members) if not ex[j]]
        rd[F.reading(surv, cluster_far["cluster"], cluster_far["far"])] += 1
    return {"truth": truth, "true_retained": n_true / n_rep,
            "far_members": list(far),
            "far_excluded": (n_far / (n_rep * len(jf))) if jf else None,
            "p_all_excluded": n_all / n_rep,
            "readings": {k: v / n_rep for k, v in rd.items()},
            "n_rep": n_rep, "B": B, "n_stations": len(ok)}


def _task(a):
    stations, members, t, c, ss, n_rep, B, far, cf = a
    return run_cell(stations, members, t, c, np.random.default_rng(ss), n_rep, B, far, cf)


def run_g0(stations: dict, members: list[str], cut30: dict, seed: int, n_rep: int = K.N_REP,
           B: int = K.B_G0, cell_list: list[dict] | None = None,
           truths: list[str] | None = None, workers: int = 1) -> dict:
    """Every cell x truth; ``pass`` under G0_PASS (module doc). ``seed`` is required. Task k
    (cells in declared order, truths in member order) draws from SeedSequence(seed) child k,
    so the result does not depend on ``workers``."""
    K.check_seed(seed)
    fp = F.far_pairs(cut30, members)
    cf = F.cluster_far(cut30, members)
    tasks = [(c, t) for c in (cell_list if cell_list is not None else cells())
             for t in (truths if truths is not None else members)]
    kids = np.random.SeedSequence([seed, STREAM_G0]).spawn(len(tasks))
    args = [(stations, members, t, c, kids[k], n_rep, B, fp[t], cf)
            for k, (c, t) in enumerate(tasks)]
    if workers and workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_task, args)
    else:
        res = [_task(a) for a in args]
    recs = [{"cell": cell_key(c), **c, **r} for (c, _), r in zip(tasks, res, strict=True)]
    tr = [r["true_retained"] for r in recs]
    fx = [r["far_excluded"] for r in recs if r["far_excluded"] is not None]
    pa = [r["p_all_excluded"] for r in recs]
    summary = {"true_retained_min": min(tr), "far_excluded_pooled": float(np.mean(fx)) if fx
               else None, "far_excluded_worst": min(fx) if fx else None,
               "p_all_excluded_max": max(pa),
               "true_retained_mean": float(np.mean(tr)),
               "readings_by_truth": {t: {k: float(np.mean([r["readings"][k] for r in recs
                                                          if r["truth"] == t]))
                                         for k in F.READINGS}
                                     for t in sorted({r["truth"] for r in recs})}}
    P = K.G0_PASS
    checks = {"true_retained_every_cell": summary["true_retained_min"]
              >= P["true_retained_min_every_cell"],
              "far_excluded_pooled": fx != [] and summary["far_excluded_pooled"]
              >= P["far_excluded_min_pooled"],
              "far_excluded_worst_cell": fx != [] and summary["far_excluded_worst"]
              >= P["far_excluded_min_worst_cell"],
              "p_all_excluded_every_cell": summary["p_all_excluded_max"]
              <= P["p_all_excluded_max_every_cell"]}
    worst = sorted(recs, key=lambda r: r["true_retained"])[:5]
    return {"pass": all(checks.values()), "checks": checks, "criteria": P,
            "cell_definition": K.G0_CELL, "summary": summary,
            "worst_true_retained": [{k: w[k] for k in ("cell", "truth", "true_retained")}
                                    for w in worst],
            "cells": recs, "members": list(members), "cluster_far": cf, "far_pairs": fp,
            "cut30_cm": {m: cut30[m] for m in members}, "seed": seed, "n_rep": n_rep, "B": B}


def model_points(stations: dict, members: list[str]) -> dict:
    """Noise-free T1 and T2 of every member on its full model months."""
    E = _E()
    pairs = {}
    for s, st in stations.items():
        prep = E.prepare(np.asarray(st["m"]))
        if len(prep["years"]) < K.MIN_YEARS:
            continue
        mod = {mb: E.coefficients(prep, np.asarray(st["mod"][mb], float)) for mb in members}
        pairs[s] = {"years": prep["years"], "obs": mod[members[0]], "mod": mod}
    stk = F.stack(pairs, members, {s: stations[s]["zone"] for s in pairs})
    pt = F.point(stk)
    return {"T1": {mb: float(pt["T1"][1 + j]) for j, mb in enumerate(members)},
            "T2_m": {mb: float(pt["T2"][1 + j]) for j, mb in enumerate(members)},
            "n_stations": len(pairs)}


def run_gsep(stations: dict, members: list[str], cut30: dict, seed: int,
             n_rep: int = K.GSEP_NREP, B: int = K.B_G0) -> dict:
    """G-sep (module doc): far members vs every cluster member. ``seed`` is required."""
    K.check_seed(seed)
    if K.PAGE not in members:
        raise ValueError("the page column must be a screen member")
    kids = np.random.SeedSequence([seed, STREAM_GSEP]).spawn(len(K.GSEP_CELLS))
    E = _E()
    pts = model_points(stations, members)
    hw_cells = []
    for c, ss in zip(K.GSEP_CELLS, kids, strict=True):
        rng = np.random.default_rng(ss)
        cell = {"stress": "none", "loading_mm": c["loading_mm"],
                "thermal_mm": c["thermal_mm"], "sigma_mm": c["sigma_mm"],
                "jumps": c["jumps"], "model_err": c["model_err"]}
        keep = {s: rng.random(len(st["m"])) >= K.P_MISS for s, st in stations.items()}
        preps = {s: E.prepare(np.asarray(st["m"])[keep[s]]) for s, st in stations.items()}
        ok = [s for s in sorted(stations) if len(preps[s]["years"]) >= K.MIN_YEARS]
        keep = {s: keep[s] for s in ok}
        W = worlds(stations, K.PAGE, cell, rng, n_rep, keep, truth_phase(stations, K.PAGE))
        mod = {s: {K.PAGE: E.coefficients(preps[s], np.asarray(stations[s]["mod"][K.PAGE],
                                                               float)[keep[s]])} for s in ok}
        obs = {s: E.coefficients(preps[s], W[s]) for s in ok}
        pairs = {s: {"years": preps[s]["years"], "obs": obs[s][0], "mod": mod[s]} for s in ok}
        stk = F.stack(pairs, [K.PAGE], {s: stations[s]["zone"] for s in ok})
        h1, h2 = [], []
        for i in range(n_rep):
            for k, s in enumerate(stk["stations"]):
                stk["V"][0, k, :len(preps[s]["years"])] = obs[s][i]
            hw = F.half_widths(stk, rng, B, K.PAGE)
            h1.append(hw["T1"])
            h2.append(hw["T2"])
        hw_cells.append({**cell, "hw_T1": float(np.median(h1)), "hw_T2": float(np.median(h2))})
    hw1 = max(c["hw_T1"] for c in hw_cells)
    hw2 = max(c["hw_T2"] for c in hw_cells)
    cf = F.cluster_far(cut30, members)
    need1, need2 = hw1 + K.LOG_ALLOW, hw2 + K.LOG_ALLOW + K.LOG_FRAME
    sep = {}
    for f in cf["far"]:
        for c in cf["cluster"]:
            d1 = abs(math.log(pts["T1"][f]) - math.log(pts["T1"][c]))
            d2 = abs(math.log(pts["T2_m"][f]) - math.log(pts["T2_m"][c]))
            r = {"dlog_T1": d1, "dlog_T2": d2, "T1_separated": d1 > need1,
                 "T2_separated": d2 > need2}
            r["separated"] = r["T1_separated"] or r["T2_separated"]
            sep[f"{f}|{c}"] = r
    n = sum(v["separated"] for v in sep.values())
    ok = bool(cf["far"]) and bool(cf["cluster"]) and n == len(sep)
    return {"pass": ok, "n_separated": n, "n_pairs": len(sep),
            "rule": "every far member separated from every cluster member (lead, review B2)",
            "need_T1": need1, "need_T2": need2, "hw_T1": hw1, "hw_T2": hw2,
            "t1_route_pairs": sum(v["T1_separated"] for v in sep.values()),
            "cluster_far": cf, "allowance_log": K.LOG_ALLOW,
            "frame_margin_log": K.LOG_FRAME, "hw_cells": hw_cells, "separation": sep,
            "model_points": pts, "members": list(members), "seed": seed, "n_rep": n_rep,
            "B": B}


def standin_stations(seed: int, members=("page", "low", "near")) -> dict:
    """A FABRICATED stand-in primary unit (no model output, no observation): 11 distal and 8
    mid stations with v1's window shapes (NGL HUWE/DNAN 2014-08..2019-12, WRA 2020-01..
    2022-12) and members with declared seasonal amplitudes. Smoke runs and tests only."""
    rng = np.random.default_rng(seed)
    amp = {"page": {"distal": 0.020, "mid": 0.008}, "low": {"distal": 0.0065, "mid": 0.0036},
           "near": {"distal": 0.021, "mid": 0.0082}}
    out = {}
    names = [f"D{i:02d}" for i in range(11)] + [f"M{i:02d}" for i in range(8)]
    for i, s in enumerate(names):
        zone = "distal" if s.startswith("D") else "mid"
        ngl = i in (0, 11)
        m = (np.arange(12 * 2014 + 7, 12 * 2019 + 12) if ngl
             else np.arange(12 * 2020, 12 * 2022 + 12))
        ang = 2 * math.pi * ((m % 12) + 0.5) / 12.0
        ph = rng.uniform(-0.3, 0.3)
        trend = -0.01 * (m - m[0]) / 12.0
        out[s] = {"zone": zone, "family": "ngl" if ngl else "wra", "m": m,
                  "mod": {mb: trend + amp[mb][zone] * rng.uniform(0.8, 1.2)
                          * np.cos(ang - 2.0 - ph) for mb in members}}
    return out
