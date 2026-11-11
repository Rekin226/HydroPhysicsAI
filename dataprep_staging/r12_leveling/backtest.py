"""Rolling-origin backtest of multi-year subsidence at leveling benchmarks (round-12 exploratory).

For an origin year o (spring survey) and horizon h, the target is the elevation drop between
the surveys of o and o+h at one benchmark. Every predictor may use surveys up to o only;
tuning constants for origin o are learned from targets that closed at or before o.

Predictors
  r3 / r5     benchmark's mean increment over the last 3 / 5 years, continued
  sm          r5 averaged with the r5 of the 8 nearest benchmarks (noise reduction)
  eb          r5 shrunk toward ``sm`` with a weight learned on earlier origins
  ebd         ``eb`` times a fan-wide damping factor per horizon learned on earlier origins
  column      the page model's column at the benchmark's cell (origins >= 2012; calibrated on
              the 2012-2021 surveys themselves, so it is favoured, shown as a reference)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataprep_staging.r12_leveling.increments import site_table
from dataprep_staging.r12_leveling.nowcast import features
from dataprep_staging.r12_leveling.paths import FWD, POLY, out

YEARS = list(range(2004, 2022))
ORIGINS = list(range(2010, 2021))
HORIZONS = (1, 3, 5)


def column_at_sites(xy):
    from hydrophysics.twin.grid import build_grid
    g = build_grid(POLY)
    cent = g.centroids()
    d, cell = cKDTree(cent).query(xy)
    z = np.load(FWD, allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    may = {t.year: i for i, t in enumerate(dates) if t.month == 5}
    subs = z["subs_mean"][0]
    cum = {y: np.where(d <= g.dx, subs[cell, i], np.nan) for y, i in may.items()}
    return cum


def trailing(D, j, n, min_n):
    w = D[:, max(0, j - n + 1):j + 1]
    r = np.nanmean(w, axis=1)
    r[np.isfinite(w).sum(1) < min_n] = np.nan
    return r


def run():
    d, xy, sids, elev = site_table()
    E = elev.reindex(columns=YEARS).to_numpy("float64")
    D = d.reindex(columns=YEARS).to_numpy("float64")
    _, county = features(xy)
    tree = cKDTree(xy)
    _, nb = tree.query(xy, k=9)
    col = column_at_sites(xy)
    rows = []
    for o in ORIGINS:
        j = YEARS.index(o)
        r3, r5 = trailing(D, j, 3, 2), trailing(D, j, 5, 3)
        sm = np.nanmean(r5[nb], axis=1)
        sm[np.isfinite(r5[nb]).sum(1) < 3] = np.nan
        for h in HORIZONS:
            if o + h > YEARS[-1]:
                continue
            tgt = E[:, j] - E[:, j + h]
            tgt[np.abs(tgt) > 0.5 * h] = np.nan
            colp = (col[o + h] - col[o]) if o in col and (o + h) in col and o >= 2012 \
                else np.full(len(sids), np.nan)
            rows.append(pd.DataFrame({"sid": sids, "o": o, "h": h, "obs": tgt, "r3": r3 * h,
                                      "r5": r5 * h, "sm": sm * h, "column": colp,
                                      "county": county}))
    T = pd.concat(rows).dropna(subset=["obs", "r3", "r5", "sm"])
    # learned constants, forward in time only
    T["eb"] = np.nan
    T["ebd"] = np.nan
    for o in ORIGINS:
        for h in HORIZONS:
            te = (T.o == o) & (T.h == h)
            tr = (T.h == h) & (T.o + T.h <= o)
            if tr.sum() < 500 or te.sum() == 0:
                continue
            a = (T.r5 - T.sm)[tr].to_numpy()
            b = (T.obs - T.sm)[tr].to_numpy()
            w = float(np.clip((a * b).sum() / (a * a).sum(), 0, 1))
            eb_tr = T.sm[tr] + w * (T.r5 - T.sm)[tr]
            phi = float((eb_tr * T.obs[tr]).sum() / (eb_tr ** 2).sum())
            T.loc[te, "eb"] = T.sm[te] + w * (T.r5 - T.sm)[te]
            T.loc[te, "ebd"] = phi * T.loc[te, "eb"]
            T.loc[te, "w"] = w
            T.loc[te, "phi"] = phi
    return T


def table(T, preds=("r3", "r5", "sm", "eb", "ebd", "column"), need=("eb",)):
    S = T.dropna(subset=list(need))
    out = []
    for (cty, h), g in [((c, h), S[(S.h == h) & (S.county.isin(cs))])
                        for c, cs in (("all", S.county.unique()), ("Changhua", ["彰化縣"]),
                                      ("Yunlin", ["雲林縣"])) for h in HORIZONS]:
        row = {"county": cty, "h": h, "n": len(g), "origins": g.o.nunique(),
               "obs_sd_cm": 100 * g.obs.std()}
        for p in preds:
            k = g[p].notna()
            row[p] = 100 * float(np.sqrt(((g.obs - g[p])[k] ** 2).mean())) if k.any() else np.nan
        # column on its own common support
        k = g.column.notna()
        row["r3@col"] = 100 * float(np.sqrt(((g.obs - g.r3)[k] ** 2).mean())) if k.any() else np.nan
        row["ebd@col"] = 100 * float(np.sqrt(((g.obs - g.ebd)[k] ** 2).mean())) if k.any() else np.nan
        row["n_col"] = int(k.sum())
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    T = run()
    T.to_parquet(out("backtest_pairs.parquet"))
    pd.set_option("display.width", 250)
    print(table(T).round(2).to_string(index=False))
    print(T.dropna(subset=["eb"]).groupby(["h", "o"])[["w", "phi"]].first().round(2).T.to_string())
    print(json.dumps({"note": "RMSE in cm of the cumulative drop over the horizon"}))
