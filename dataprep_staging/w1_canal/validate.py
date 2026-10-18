"""Jan-May held-out benchmark (plan W1 validation): leave-one-year-out over 2015-2021.

For each held-out year k the crop-1 shares (s_chc, s_yl) and the winter weight are chosen on
the other six years' Jan-May supplies; beta, a_chc, a_yl are fitted to the annual district
totals of the variant's fit years (never to the Jan-May values); the Denton benchmark uses
the annual totals (known every year). Scored on year k only.

Baselines on the same seven values: v3 (its monthly timing x the annual total), the
climatological fraction (Jan-May share = mean of the other six years, x the annual total),
and the calendar without the supply limit (shares chosen the same way).

Comparison (pre-registered): RMSE per district; paired bootstrap over years of the RMSE
difference (method - baseline), 90 % percentile interval; paired two-sided sign test on the
absolute errors.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from w1_canal.config import (
    BENCH_YEARS,
    BOOT_B,
    BOOT_LEVEL,
    BOOT_SEED,
    DISTRICTS,
    JIJI_COL,
    SHARE_GRID,
    WINTER_GRID,
)
from w1_canal.diversion import annual_targets, denton_pfd, fit, janmay, simulate


def run_grid(dd, targets, fit_years, log=print) -> list[dict]:
    """Fit and benchmark every (cap, s_c, s_y, winter) combination once."""
    grid = []
    combos = list(itertools.product((True, False), SHARE_GRID, SHARE_GRID, WINTER_GRID))
    for k, (cap, s_c, s_y, w) in enumerate(combos):
        p, rms = fit(dd, targets, s_c, s_y, w, cap, fit_years)
        mc, my = simulate(dd, *p, s_c, s_y, w, cap)
        ny = len(dd.years)
        # Denton over the variant's years: annual totals exist for every year of the variant
        bc = denton_pfd(mc, targets["chc"])
        by = denton_pfd(my, targets["yl"])
        grid.append(dict(cap=cap, s_c=s_c, s_y=s_y, w=w, p=p, fit_rms=rms,
                         jm_c=janmay(bc, dd.years, BENCH_YEARS),
                         jm_y=janmay(by, dd.years, BENCH_YEARS),
                         jm_raw_c=janmay(mc, dd.years, BENCH_YEARS),
                         jm_raw_y=janmay(my, dd.years, BENCH_YEARS),
                         ann_c=mc.reshape(ny, 12).sum(1), ann_y=my.reshape(ny, 12).sum(1)))
        if (k + 1) % 216 == 0:
            log(f"  grid {k + 1}/{len(combos)}")
    return grid


def loo(grid, obs_c, obs_y, cap: bool, key=("jm_c", "jm_y")):
    cand = [g for g in grid if g["cap"] == cap]
    pc, py, chosen = [], [], []
    n = len(obs_c)
    for k in range(n):
        tr = [i for i in range(n) if i != k]
        g = min(cand, key=lambda g: ((g[key[0]][tr] - obs_c[tr]) ** 2).sum()
                + ((g[key[1]][tr] - obs_y[tr]) ** 2).sum())
        pc.append(g[key[0]][k])
        py.append(g[key[1]][k])
        chosen.append((BENCH_YEARS[k], g["s_c"], g["s_y"], g["w"]))
    return np.array(pc), np.array(py), chosen


def choose_all(grid, obs_c, obs_y, cap=True):
    cand = [g for g in grid if g["cap"] == cap]
    return min(cand, key=lambda g: ((g["jm_c"] - obs_c) ** 2).sum()
               + ((g["jm_y"] - obs_y) ** 2).sum())


def scores(pred: np.ndarray, obs: np.ndarray) -> dict:
    e = pred - obs
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()),
            "mae": float(np.abs(e).mean()), "r": float(np.corrcoef(pred, obs)[0, 1])}


def paired_compare(pred_m, pred_b, obs, B=BOOT_B, seed=BOOT_SEED, level=BOOT_LEVEL) -> dict:
    em, eb = pred_m - obs, pred_b - obs
    rng = np.random.default_rng(seed)
    n = len(obs)
    idx = rng.integers(0, n, size=(B, n))
    dm = np.sqrt((em[idx] ** 2).mean(1)) - np.sqrt((eb[idx] ** 2).mean(1))
    lo, hi = np.percentile(dm, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    wins = int((np.abs(em) < np.abs(eb)).sum())
    ties = int((np.abs(em) == np.abs(eb)).sum())
    n_eff = n - ties
    p = float(binomtest(wins, n_eff, 0.5).pvalue) if n_eff else 1.0
    d0 = float(np.sqrt(np.mean(em ** 2)) - np.sqrt(np.mean(eb ** 2)))
    return {"rmse_diff": d0, "boot_lo": float(lo), "boot_hi": float(hi),
            "excludes_zero": bool(hi < 0 or lo > 0), "method_better": bool(d0 < 0),
            "sign_wins": wins, "sign_n": n_eff, "sign_p_two_sided": p,
            "boot_B": B, "boot_seed": seed, "level": level}


def baselines(jiji, v3m, obs_c, obs_y):
    ann = {d: jiji[JIJI_COL[d]].reindex(list(BENCH_YEARS)).to_numpy(dtype=float)
           for d in DISTRICTS}
    obs = {"chc": obs_c, "yl": obs_y}
    out = {}
    clim = {}
    for d in DISTRICTS:
        f = obs[d] / ann[d]
        clim[d] = np.array([np.delete(f, k).mean() for k in range(len(f))]) * ann[d]
    out["clim_fraction_loo"] = clim
    v3 = {}
    for d in DISTRICTS:
        vals = []
        for i, y in enumerate(BENCH_YEARS):
            vv = v3m[d][v3m.index.year == y]
            vals.append(vv[vv.index.month <= 5].sum() / vv.sum() * ann[d][i])
        v3[d] = np.array(vals)
    out["v3"] = v3
    return out, ann


def evaluate(dd, jiji, v3m, bench, fit_years, log=print) -> dict:
    """Full benchmark for one variant; returns a JSON-able dict plus the grid."""
    targets = annual_targets(jiji, dd.years)
    obs_c = bench["chc"].reindex(list(BENCH_YEARS)).to_numpy(dtype=float)
    obs_y = bench["yl"].reindex(list(BENCH_YEARS)).to_numpy(dtype=float)
    grid = run_grid(dd, targets, fit_years, log)
    res = {"years": list(BENCH_YEARS), "obs": {"chc": obs_c.tolist(), "yl": obs_y.tolist()}}
    preds = {}
    pc, py, ch = loo(grid, obs_c, obs_y, True)
    preds["supply_limited_denton"] = {"chc": pc, "yl": py}
    res["chosen_supply_limited"] = ch
    pc2, py2, ch2 = loo(grid, obs_c, obs_y, False)
    preds["calendar_denton"] = {"chc": pc2, "yl": py2}
    res["chosen_calendar"] = ch2
    pc3, py3, ch3 = loo(grid, obs_c, obs_y, True, key=("jm_raw_c", "jm_raw_y"))
    preds["supply_limited_raw_indicator"] = {"chc": pc3, "yl": py3}
    res["chosen_raw_indicator"] = ch3
    base, ann = baselines(jiji, v3m, obs_c, obs_y)
    preds.update(base)
    obs = {"chc": obs_c, "yl": obs_y}
    res["scores"] = {m: {d: scores(p[d], obs[d]) for d in DISTRICTS} for m, p in preds.items()}
    res["pred"] = {m: {d: np.round(p[d], 4).tolist() for d in DISTRICTS}
                   for m, p in preds.items()}
    res["compare"] = {}
    for b in ("clim_fraction_loo", "v3"):
        res["compare"][b] = {d: paired_compare(preds["supply_limited_denton"][d], preds[b][d],
                                               obs[d]) for d in DISTRICTS}
    g = choose_all(grid, obs_c, obs_y, True)
    res["final_choice"] = {"s_c": g["s_c"], "s_y": g["s_y"], "winter": g["w"],
                           "beta_a": g["p"].tolist(), "annual_fit_rms": g["fit_rms"],
                           "janmay_insample": {"chc": np.round(g["jm_c"], 4).tolist(),
                                               "yl": np.round(g["jm_y"], 4).tolist()}}
    # acceptance (plan W1): beat both v3 and clim in both districts, with the 90 % paired
    # bootstrap interval excluding zero; otherwise "not distinguishable from climatology"
    beat = {b: {d: res["compare"][b][d]["method_better"] for d in DISTRICTS}
            for b in ("clim_fraction_loo", "v3")}
    excl = {b: {d: res["compare"][b][d]["excludes_zero"] and res["compare"][b][d]["method_better"]
                for d in DISTRICTS} for b in ("clim_fraction_loo", "v3")}
    res["acceptance"] = {
        "beats_point": beat, "interval_excludes_zero": excl,
        "passed": all(all(v.values()) for v in excl.values()),
        "label": None if all(all(v.values()) for v in excl.values()) else
        "not distinguishable from climatology on the benchmark"
        if not all(excl["clim_fraction_loo"].values()) else
        "not distinguishable from v3 on the benchmark"}
    return res, grid


def to_frame(res: dict) -> pd.DataFrame:
    rows = []
    for i, y in enumerate(res["years"]):
        r = {"year": y}
        for d in DISTRICTS:
            r[f"obs_{d}"] = res["obs"][d][i]
            for m, p in res["pred"].items():
                r[f"{m}_{d}"] = p[d][i]
        rows.append(r)
    return pd.DataFrame(rows)
