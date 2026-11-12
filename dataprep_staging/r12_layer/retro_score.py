"""Retrospective held-out-years test of the layer against a refitted column.

Label: post-failure, exploratory, retrospective. It follows the failed gating design study
(``gate_power``) as the one follow-up the retry rule allows. It is run once and cannot make
any claim 'supported'; leveling 2012-2021 is a spent unit for the layer's design choices.

Origin     : the spring 2016 survey. Nothing surveyed after 2016 enters either arm.
Column     : the page recipe (zonal, leveling target, hpc0 guard 365 d) refitted with every
             leveling survey and ring sample of 2017-01 onwards withheld
             (``retro_column``); its field at the benchmark's cell, re-zeroed at May 2016.
Layer  C0  : rate of 2014-2016 (3 km kernel, own weight 4) times the horizon.
Layer  HL  : the same with the head term; gamma refitted on leveling years 2007-2016 only.
Zero   Z   : no sinking.
Truth      : cumulative May-to-May leveling drop from 2016 over k = 1, 3, 5 years at
             benchmarks with every increment of the horizon observed and a prediction from
             every arm.
Primary    : k = 3 (2016 -> 2019), pooled over both counties, arm C0.
Rule       : 'layer ahead' iff RMSE(C0) < RMSE(column) and the 10 km spatial-block bootstrap
             95 % lower bound of MSE(column) - MSE(C0) is above 0; 'column ahead' iff the
             mirror holds; otherwise 'not separated'. k = 1, 5, the counties and HL are
             reported without a verdict.
Disclosed  : the flow model (the column's head driver) was fitted on heads of the whole
             record, which favours the column; the column has five surveys (2012-2016) to
             fit, which favours the layer; the layer's window, kernel and weight were
             chosen after looking at 2009-2021 leveling scores.

    python -m dataprep_staging.r12_layer.retro_score --field <dir>/field.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataprep_staging.r12_layer import model as M
from dataprep_staging.r12_leveling.paths import DATAPREP

ORIGIN = 2016
HORIZONS = (1, 3, 5)
PRIMARY = 3
BLOCK_M = 10_000.0
SEED = 20261114
N_BOOT = 4000
GAMMA_YEARS = (2007, 2016)


def column_cum(field_npz: str, xy: np.ndarray) -> dict:
    z = np.load(field_npz, allow_pickle=True)
    months = pd.PeriodIndex(pd.to_datetime([str(d) for d in z["dates"]]), freq="M")
    d, cell = cKDTree(z["centroids"]).query(xy)
    f = z["field"][cell]                                       # (n, T) metres, + = sinking
    f[d > 1000.0] = np.nan
    j0 = months.get_loc(pd.Period(f"{ORIGIN}-05", freq="M"))
    return {k: f[:, months.get_loc(pd.Period(f"{ORIGIN + k}-05", freq="M"))] - f[:, j0]
            for k in HORIZONS}


def block_bootstrap(d: np.ndarray, xy: np.ndarray, rng) -> tuple[float, float]:
    b = np.floor(xy / BLOCK_M).astype(int)
    _, g = np.unique(b, axis=0, return_inverse=True)
    sums = np.bincount(g, weights=d)
    cnts = np.bincount(g).astype(float)
    ix = rng.integers(0, len(sums), (N_BOOT, len(sums)))
    m = sums[ix].sum(1) / cnts[ix].sum(1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def run(field_npz: str) -> dict:
    rng = np.random.default_rng(SEED)
    L = M.Layer()
    yi = M.YEARS.index
    rate = M.rate_at(L.D, L.xy, (ORIGIN - 2, ORIGIN - 1, ORIGIN))
    gamma, _ = M.fit_gamma(L.D, L.F, L.xy, years=GAMMA_YEARS)
    ref = np.nanmean(L.F[:, [yi(y) for y in (ORIGIN - 2, ORIGIN - 1, ORIGIN)]], axis=1)
    col = column_cum(field_npz, L.xy)
    _, c = cKDTree(L.cxy).query(L.xy)
    county = L.cells.county.to_numpy()[c]
    out = {"origin": ORIGIN, "gamma_fit_to_2016": gamma, "seed": SEED, "horizons": {}}
    for k in HORIZONS:
        yrs = [ORIGIN + i for i in range(1, k + 1)]
        obs = L.D[:, [yi(y) for y in yrs]].sum(1)              # NaN if any increment missing
        c0 = k * rate
        hl = sum(rate + gamma * np.nan_to_num(L.F[:, yi(y)] - ref) * np.clip(rate, 0.002, None)
                 for y in yrs)
        ok = np.isfinite(obs) & np.isfinite(c0) & np.isfinite(col[k])
        res = {"n": int(ok.sum())}
        for name, sel in (("all", ok), ("雲林縣", ok & (county == "雲林縣")),
                          ("彰化縣", ok & (county == "彰化縣"))):
            e = {"C0": c0 - obs, "HL": hl - obs, "column": col[k] - obs, "zero": -obs}
            r = {a: float(100 * np.sqrt(np.mean(v[sel] ** 2))) for a, v in e.items()}
            r["bias_cm"] = {a: float(100 * np.mean(v[sel])) for a, v in e.items()}
            d = (e["column"][sel] ** 2 - e["C0"][sel] ** 2) * 1e4
            lo, hi = block_bootstrap(d, L.xy[sel], rng)
            r.update(n=int(sel.sum()), mse_gain_cm2=float(d.mean()), gain_lo=lo, gain_hi=hi)
            res[name] = r
        out["horizons"][k] = res
    p = out["horizons"][PRIMARY]["all"]
    out["verdict"] = ("layer ahead" if p["C0"] < p["column"] and p["gain_lo"] > 0 else
                      "column ahead" if p["column"] < p["C0"] and p["gain_hi"] < 0 else
                      "not separated")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    a = ap.parse_args()
    d = os.path.join(DATAPREP, "reports", "SL", "retro_2016")
    path = os.path.join(d, "result.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: the retrospective test is scored once")
    res = run(a.field)
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
