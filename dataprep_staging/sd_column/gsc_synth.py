"""G0-OC: synthetic operating characteristics of X5.gnss_seasonal_column.v1's decision rule
(model-only; run by the lead after the model side is frozen and BEFORE registration).

Worlds are built on the real primary stations' declared windows and the page column's own
monthly subsidence at their cells (the frozen model side), never on a GNSS value:

    obs_s(t) = r * mod_s(t) + loading + thermal + AR(1) noise [+ an uncorrected jump]

* r in {0.4, 1, 2.5}: the true ratio of observed to modelled seasonal (elastic) response;
* loading: a regional annual term of amplitude L in {0, 3, 6} mm at a random phase common to
  the stations of a world, of which a station-specific U(0.3, 1.0) share survives the
  off-fan common-mode removal;
* thermal: a monument term of amplitude T in {0, 2} mm phased to temperature (height
  maximum in July, i.e. subsidence minimum);
* noise: monthly AR(1), sigma in {2, 4} mm, rho 0.5 (a declared grid; sigma_hf_m of
  stations.csv is NOT read);
* jumps: off, or TKJS-size (each WRA station, with probability 0.25, carries one
  uncorrected step of U(30, 120) mm, random sign, at a random month of its window).

* missingness: each station-month is dropped with probability P_MISS (declared; the real
  rate after flags 3/9 and the >= 10 days rule is not read), once per station and grid
  cell; obs and model then use the kept months, as the scorer does (years with < 8 months
  and stations with < 2 years drop out).

The scored estimator runs on every world (two-stage bootstrap with B_G0 draws and the
declared mean-neutral nuisance propagation, gsc_estimator.NUISANCE) on the primary unit and,
from the same world and bootstrap seed, on the T1n stratum (the stations passed as
``strata['T1n']``).

PASS RULE (declared now; judge rank 10 thresholds, on the verdict that is SCORED, i.e. the
band reading required at both observation scales 1 and 1/1.42, gsc_estimator.scaled_reading):
P(consistent | r = 1) >= 0.80 and the 90 % interval covers r = 1 in >= 0.85 of worlds, both
over the declared grid with every cell weighted equally (the nuisance grid read as a prior,
POOLING); and in EVERY cell P(consistent | r = 0.4) <= 0.10 and P(consistent | r = 2.5) <=
0.10 (a false 'consistent' is controlled everywhere). The scale-1 band reading, the worst
cell of every statistic, the T1n stratum's readings (P(T1n 'larger' or 'smaller' | r = 1),
P(T1n inconclusive)) and P(S label | r = 1) = P(T1 consistent and T1n not opposite) are
reported beside the pass. A fail means the test is not run. Under the scored rule a
'consistent' reading needs the interval inside about [0.947, 1.5], so the pass is hard by
construction; that is the declared rule, not a defect of the worlds.
"""
from __future__ import annotations

import itertools
import math

import gsc_estimator as E
import numpy as np

R_GRID = (0.4, 1.0, 2.5)
LOADING_MM = (0.0, 3.0, 6.0)
THERMAL_MM = (0.0, 2.0)
SIGMA_MM = (2.0, 4.0)
RHO = 0.5
JUMPS = ("off", "tkjs")
P_JUMP = 0.25
JUMP_MM = (30.0, 120.0)
SURVIVE = (0.3, 1.0)
THERMAL_PEAK_MONTH = 6            # July (month index 6): height maximum
N_REP = 200
B_G0 = 500
SEED = 20261102
P_MISS = 0.10
PASS = {"p_consistent_r1_min": 0.80, "p_consistent_r04_max": 0.10,
        "p_consistent_r25_max": 0.10, "coverage_r1_min": 0.85}
PASS_RULE = ("the scored verdict: band [1/1.5, 1.5] at both observation scales 1 and 1/1.42 "
             "(gsc_estimator.scaled_reading); the scale-1 reading is reported only")
POOLING = ("P(consistent | r = 1) and coverage: mean over the declared grid cells (each cell "
           "weighted equally: the nuisance grid as a prior); false 'consistent' at r = 0.4 "
           "and r = 2.5: in every cell; worst cells reported beside")


def ar1(rng, n_rep: int, n: int, sigma: float, rho: float) -> np.ndarray:
    e = rng.normal(0.0, sigma * math.sqrt(1 - rho ** 2), size=(n_rep, n))
    x = np.empty_like(e)
    x[:, 0] = rng.normal(0.0, sigma, size=n_rep)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def worlds(stations: dict, r: float, L_mm: float, T_mm: float, sigma_mm: float, jumps: str,
           rng, n_rep: int) -> dict:
    """{station: (n_rep, n) synthetic observed subsidence (m)} for one grid cell."""
    phase = rng.uniform(0, 2 * math.pi, size=n_rep)
    out = {}
    for s, st in stations.items():
        m = np.asarray(st["m"])
        ang = 2 * math.pi * ((m % 12) + 0.5) / 12.0
        y = r * np.asarray(st["mod"], float)[None, :].repeat(n_rep, axis=0)
        surv = rng.uniform(*SURVIVE, size=n_rep)
        y += (L_mm / 1000.0) * surv[:, None] * np.cos(ang[None, :] - phase[:, None])
        tpk = 2 * math.pi * (THERMAL_PEAK_MONTH + 0.5) / 12.0
        y -= (T_mm / 1000.0) * np.cos(ang - tpk)[None, :]
        y += ar1(rng, n_rep, m.size, sigma_mm / 1000.0, RHO)
        if jumps == "tkjs" and st["family"] == "wra":
            has = rng.random(n_rep) < P_JUMP
            size = rng.uniform(*JUMP_MM, size=n_rep) / 1000.0 * rng.choice([-1, 1], n_rep)
            at = rng.integers(1, m.size, size=n_rep)
            step = (np.arange(m.size)[None, :] >= at[:, None]) * (has * size)[:, None]
            y += step
        out[s] = y
    return out


def run_cell(stations: dict, r: float, L_mm, T_mm, sigma_mm, jumps, rng, n_rep: int,
             B: int, strata: dict | None = None) -> dict:
    """One grid cell at one r: readings of the primary unit (scored rule and scale 1),
    coverage, and the readings of each stratum (scored rule) from the same worlds."""
    keepm = {s: rng.random(len(st["m"])) >= P_MISS for s, st in stations.items()}
    preps = {s: E.prepare(np.asarray(st["m"])[keepm[s]]) for s, st in stations.items()}
    fam = {s: st["family"] for s, st in stations.items()}
    mod = {s: E.coefficients(preps[s], np.asarray(st["mod"], float)[keepm[s]])
           for s, st in stations.items()}
    keep = [s for s in stations if len(preps[s]["years"]) >= E.MIN_YEARS]
    W = worlds({s: stations[s] for s in keep}, r, L_mm, T_mm, sigma_mm, jumps, rng, n_rep)
    obs = {s: E.coefficients(preps[s], W[s][:, keepm[s]]) for s in keep}  # (n_rep, Y, 2)
    strata = {k: [s for s in v if s in keep] for k, v in (strata or {}).items()}
    n_cons = n_cov = n_s = 0
    full = {k: 0 for k in E.READINGS}
    sc1 = {k: 0 for k in E.READINGS}
    st_counts = {k: {x: 0 for x in E.READINGS} for k in strata}
    for i in range(n_rep):
        pairs = {s: {"years": preps[s]["years"], "obs": obs[s][i], "mod": mod[s]}
                 for s in keep}
        seed_i = int(rng.integers(0, 2 ** 31))
        ci = E.bootstrap(pairs, fam, B=B, seed=seed_i)
        v = E.scaled_reading(ci["lo"], ci["hi"])["verdict"]
        full[v] += 1
        sc1[E.reading(ci["lo"], ci["hi"])] += 1
        n_cons += v == "consistent"
        n_cov += ci["lo"] <= r <= ci["hi"]
        clash = False
        for k, names in strata.items():
            if names:
                cn = E.bootstrap({s: pairs[s] for s in names}, fam, B=B, seed=seed_i)
                vn = E.scaled_reading(cn["lo"], cn["hi"])["verdict"]
            else:
                vn = "inconclusive"
            st_counts[k][vn] += 1
            clash = clash or vn in E.OPPOSITE[v]
        n_s += v == "consistent" and not clash
    return {"p_consistent": n_cons / n_rep, "coverage": n_cov / n_rep,
            "p_S_label": n_s / n_rep,
            "readings": {k: x / n_rep for k, x in full.items()},
            "p_consistent_scale1": sc1["consistent"] / n_rep,
            "readings_scale1": {k: x / n_rep for k, x in sc1.items()},
            "strata": {k: {"n_stations": len(strata[k]),
                           "readings": {x: c / n_rep for x, c in st_counts[k].items()}}
                       for k in strata},
            "n_rep": n_rep, "n_stations": len(keep)}


def _mean(cells, r, f) -> float:
    return float(np.mean([f(c[f"r={r:g}"]) for c in cells]))


def run(stations: dict, n_rep: int = N_REP, B: int = B_G0, seed: int = SEED,
        strata: dict | None = None) -> dict:
    """Every grid cell x r; ``pass`` under PASS / PASS_RULE / POOLING (module doc)."""
    rng = np.random.default_rng(seed)
    cells = []
    ok = True
    for L_mm, T_mm, sg, jp in itertools.product(LOADING_MM, THERMAL_MM, SIGMA_MM, JUMPS):
        rec = {"loading_mm": L_mm, "thermal_mm": T_mm, "sigma_mm": sg, "jumps": jp}
        for r in R_GRID:
            rec[f"r={r:g}"] = run_cell(stations, r, L_mm, T_mm, sg, jp, rng, n_rep, B,
                                       strata)
        rec["false_consistent_ok"] = (
            rec["r=0.4"]["p_consistent"] <= PASS["p_consistent_r04_max"]
            and rec["r=2.5"]["p_consistent"] <= PASS["p_consistent_r25_max"])
        ok = ok and rec["false_consistent_ok"]
        cells.append(rec)
    marginal = {"p_consistent_r1": _mean(cells, 1, lambda c: c["p_consistent"]),
                "coverage_r1": _mean(cells, 1, lambda c: c["coverage"]),
                "p_S_label_r1": _mean(cells, 1, lambda c: c["p_S_label"]),
                "p_consistent_r1_scale1": _mean(cells, 1, lambda c: c["p_consistent_scale1"]),
                "readings_r1": {k: _mean(cells, 1, lambda c, k=k: c["readings"][k])
                                for k in E.READINGS},
                "readings_r04": {k: _mean(cells, 0.4, lambda c, k=k: c["readings"][k])
                                 for k in E.READINGS},
                "readings_r25": {k: _mean(cells, 2.5, lambda c, k=k: c["readings"][k])
                                 for k in E.READINGS}}
    for k in strata or {}:
        def rd(c, x, k=k):
            return c["strata"][k]["readings"][x]
        marginal[f"{k}_r1"] = {
            "p_larger_or_smaller": _mean(cells, 1, lambda c: rd(c, "larger") + rd(c, "smaller")),
            "p_inconclusive": _mean(cells, 1, lambda c: rd(c, "inconclusive")),
            "p_consistent": _mean(cells, 1, lambda c: rd(c, "consistent"))}
        marginal[f"{k}_r04_p_larger"] = _mean(cells, 0.4, lambda c: rd(c, "larger"))
        marginal[f"{k}_r25_p_smaller"] = _mean(cells, 2.5, lambda c: rd(c, "smaller"))
    ok = (ok and marginal["p_consistent_r1"] >= PASS["p_consistent_r1_min"]
          and marginal["coverage_r1"] >= PASS["coverage_r1_min"])
    worst = {"p_consistent_r1": min(c["r=1"]["p_consistent"] for c in cells),
             "p_consistent_r04": max(c["r=0.4"]["p_consistent"] for c in cells),
             "p_consistent_r25": max(c["r=2.5"]["p_consistent"] for c in cells),
             "coverage_r1": min(c["r=1"]["coverage"] for c in cells),
             "p_S_label_r1": min(c["r=1"]["p_S_label"] for c in cells)}
    return {"pass": ok, "criteria": PASS, "pass_rule": PASS_RULE, "pooling": POOLING,
            "marginal": marginal, "worst_cell": worst, "cells": cells,
            "grid": {"r": R_GRID, "loading_mm": LOADING_MM, "thermal_mm": THERMAL_MM,
                     "sigma_mm": SIGMA_MM, "rho": RHO, "jumps": JUMPS, "p_jump": P_JUMP,
                     "jump_mm": JUMP_MM, "survive": SURVIVE, "p_miss": P_MISS},
            "n_rep": n_rep, "B": B, "seed": seed,
            "strata": {k: sorted(v) for k, v in (strata or {}).items()},
            "stations": {s: {"family": st["family"], "n_months": len(st["m"])}
                         for s, st in stations.items()}}
