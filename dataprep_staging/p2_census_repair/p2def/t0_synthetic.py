"""Synthetic checks of the P2d core (dlib.per_pump_index, ramp, synth_monthly, rates,
cell_weights) on a toy census whose cohorts follow the assumed model exactly. No real data.

Usage: python -I t0_synthetic.py <out_json>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402


def toy(seed: int = 7):
    rng = np.random.default_rng(seed)
    T = 192
    g_true = {"irrigation": np.array([0.55, 0.7, 0.8, 0.85, 0.9, 0.95]),
              "livestock": np.array([0.8, 0.9, 1.0, 1.0, 1.0, 1.0])}
    rows = []
    # old pumps: installed before 2007 (index < 0); new cohorts 2007-2012 at fixed rates
    for tw, n_old, r in (("A", 3000, 200), ("B", 2000, 120), ("C", 500, 30)):
        for c, frac in (("irrigation", 0.9), ("livestock", 0.1)):
            k = int(n_old * frac)
            rows += [(tw, c, -12 * rng.integers(1, 20)) for _ in range(k)]
            for y in range(2007, 2013):
                for m in range(12):
                    k = int(rng.poisson(r * frac / 12.0))
                    rows += [(tw, c, (y - 2007) * 12 + m)] * k
    df = pd.DataFrame(rows, columns=["town", "cls", "inst"])
    n = len(df)
    cell = np.where(df["town"] == "A", rng.integers(0, 3, n),
                    np.where(df["town"] == "B", rng.integers(3, 5, n), -1))
    season = 1.0 + 0.4 * np.sin(2 * np.pi * (np.arange(T) % 12) / 12.0)
    year = 1.0 + 0.1 * np.sin(np.arange(T) / 30.0)
    base = {"irrigation": 100.0 * season * year, "livestock": 800.0 * year}
    E = np.zeros((n, T))
    inst = df["inst"].to_numpy()
    cls = df["cls"].to_numpy()
    # energy of a pump of age a (in its a-th year) = g(a) * Q(c, t); old pumps (age > 6) = g(6)
    # then P(c, t) is a mixture; the ramp is defined relative to P, so recover it numerically
    for i in range(n):
        t = np.arange(T)
        age = np.where(t >= inst[i], (t - inst[i]) // 12 + 1, 0)
        gi = np.where(age > 0, g_true[cls[i]][np.minimum(age, 6) - 1], 0.0)
        E[i] = gi * base[cls[i]] * rng.gamma(20.0, 1 / 20.0, T)
    return df, E, cell, g_true, base, T


def main(out: str) -> None:
    df, E, cell, g_true, base, T = toy()
    town, cls, inst = df["town"].to_numpy(), df["cls"].to_numpy(), df["inst"].to_numpy()
    member = np.ones(len(df), bool)
    P = D.per_pump_index(E, cls, inst, member, T)
    # check 1: P equals a direct computation
    t = 100
    m = (cls == "irrigation") & (np.maximum(inst, 0) <= t)
    p_direct = E[m, t].sum() / m.sum()
    c1 = abs(P["irrigation"][t] - p_direct) / p_direct
    # check 2: the ramp relative to P; true relative ramp g_rel(a) = g(a) * Q / P
    cohort = (inst >= 0) & (inst < 72)
    R = D.ramp(E, cls, inst, cohort, P, data_end=T - 1)
    # expected relative ramp from the generating model (noise-free): sum over cohort pumps
    num = np.zeros(6)
    den = np.zeros(6)
    for i in np.flatnonzero(cohort & (cls == "irrigation")):
        for a in range(1, 7):
            lo = inst[i] + 12 * (a - 1)
            if lo + 12 - 1 > T - 1:
                continue
            num[a - 1] += (g_true["irrigation"][a - 1] * base["irrigation"][lo:lo + 12]).sum()
            den[a - 1] += P["irrigation"][lo:lo + 12].sum()
    g_exp = num / den
    c2 = float(np.nanmax(np.abs(R["g"]["irrigation"] - g_exp) / g_exp))
    # check 3: predicted energy of the 2010-2012 cohorts from rates of 2007-2009, ramp from
    # 2007-2009 cohorts with data to 2009-12, P from pumps installed before 2010 -> compared
    # with the actual energy of the 2010-2012 cohorts over 2010-2015 (rates equal by design)
    F = 36
    existing = inst < F
    Pf = D.per_pump_index(E, cls, inst, existing, T)
    coh = (inst >= 0) & (inst < F)
    Rf = D.ramp(E, cls, inst, coh, Pf, data_end=F - 1, min_class=50)
    rate = D.rates_by_town_class(town, cls, inst, (2007, 2008, 2009))
    w = D.cell_weights(town, cell, inst, (2007, 2008, 2009))
    S = D.synth_monthly(rate, w, Rf["g"], Pf, 36, 71, 36, 107)
    pred = sum(S[c]["county"].sum() for c in D.CLASSES)
    act_m = (inst >= 36) & (inst <= 71)
    act = E[act_m, 36:108].sum()
    c3 = pred / act - 1
    pred_cells = sum(S[c]["cells"].sum() for c in D.CLASSES)
    act_cells = E[act_m & (cell >= 0), 36:108].sum()
    c3b = pred_cells / act_cells - 1
    n_pred = sum(S[c]["n_county"][-1] for c in D.CLASSES)
    c4 = n_pred / act_m.sum() - 1
    res = {"P_direct_rel_err": c1, "ramp_rel_err_vs_generating": c2,
           "ramp_irrigation": R["g"]["irrigation"].tolist(), "ramp_expected": g_exp.tolist(),
           "backtest_county_kwh_err": c3, "backtest_ingrid_kwh_err": c3b,
           "backtest_count_err": c4,
           "pass": bool(c1 < 1e-12 and c2 < 0.05 and abs(c3) < 0.08 and abs(c3b) < 0.08
                        and abs(c4) < 0.08)}
    D.save_json(res, out)
    print(res)
    if not res["pass"]:
        raise SystemExit("synthetic checks failed")


if __name__ == "__main__":
    main(sys.argv[1])
