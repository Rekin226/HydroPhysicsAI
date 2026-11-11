"""Head-driven nowcast of annual leveling increments (round-12 exploratory, design-time).

Task: at a benchmark with a leveling history, predict the next May-to-May increment from
the observed heads of that year. Scored forward in time only: the coefficients for test
year y are fitted on increments of years < y. Competitors: the benchmark's trailing mean
increment (persistence), and the page model's column (fitted on these very surveys, so it
is favoured).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataprep_staging.r12_leveling.increments import (
    SHARED,
    site_table,
    to_sites,
    well_year_stats,
)

YEARS = list(range(2014, 2022))            # dmean / exceed need a previous year (2013)


def may_change(first_year=2013, last_year=2026):
    """wells x years: (Apr-Jun mean head of year y) - (same of y-1), the survey-time change."""
    H = pd.read_parquet(SHARED + "heads_monthly.parquet")
    H = H[(H.flag <= 1) & H.date.dt.month.isin([4, 5, 6])]
    t = H.groupby(["sid", H.date.dt.year]).value.mean().unstack()
    d = t.diff(axis=1)
    return d.loc[:, [y for y in d.columns if first_year <= y <= last_year]]


def features(xy, years=YEARS):
    stats, meta = well_year_stats()
    dm = may_change().reindex(meta.index)
    stats["dmay"] = dm
    F = {}
    for grp, layers in (("s", (1,)), ("d", (2, 3, 4)), ("a", (1, 2, 3, 4))):
        for name in ("dmay", "dmean", "exceed", "min", "mean"):
            tab = stats[name].reindex(columns=years)
            F[f"{name}_{grp}"] = to_sites(tab, meta, xy, layers)
    # county by nearest monitoring well
    wxy = meta[["x_twd97", "y_twd97"]].to_numpy("float64")
    ok = np.isfinite(wxy).all(1)
    _, idx = cKDTree(wxy[ok]).query(xy)
    county = meta.CountyName.to_numpy()[ok][idx]
    return F, county


def rmse(a):
    return float(np.sqrt(np.nanmean(a ** 2)))


def run(feats=("dmay_d", "exceed_d"), trail_n=3, scale_by_rate=False, min_train_years=3,
        ridge=1e-6):
    d, xy, sids, _ = site_table()
    F, county = features(xy)
    D = d.reindex(columns=list(range(2008, 2022))).to_numpy("float64")
    ycols = list(range(2008, 2022))
    rows = []
    for y in YEARS:
        j = ycols.index(y)
        trail = np.nanmean(D[:, j - trail_n:j], axis=1)
        nprev = np.isfinite(D[:, j - trail_n:j]).sum(1)
        trail[nprev < 2] = np.nan
        X = np.stack([F[f][:, YEARS.index(y)] for f in feats], 1)
        if scale_by_rate:
            X = X * np.clip(trail, 0.002, None)[:, None] / 0.02
        rows.append(pd.DataFrame({"sid": sids, "year": y, "obs": D[:, j], "trail": trail,
                                  "county": county,
                                  **{f: X[:, i] for i, f in enumerate(feats)}}))
    T = pd.concat(rows).dropna()
    T["res"] = T.obs - T.trail
    preds = []
    for y in YEARS:
        tr, te = T[T.year < y], T[T.year == y].copy()
        if tr.year.nunique() < min_train_years:
            continue
        A = tr[list(feats)].to_numpy()
        beta = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ tr.res.to_numpy())
        te["hd"] = te.trail + te[list(feats)].to_numpy() @ beta
        te["beta"] = [beta.round(5).tolist()] * len(te)
        # oracle reference: persistence + that year's true fan-wide mean departure
        te["oracle_year"] = te.trail + te.res.mean()
        preds.append(te)
    P = pd.concat(preds)
    out = {}
    for name, g in [("all", P), ("Changhua", P[P.county == "彰化縣"]),
                    ("Yunlin", P[P.county == "雲林縣"])]:
        e0, e1, e2 = g.obs - g.trail, g.obs - g.hd, g.obs - g.oracle_year
        out[name] = {"n": int(len(g)), "rmse_persist_cm": 100 * rmse(e0),
                     "rmse_headdriven_cm": 100 * rmse(e1),
                     "skill_vs_persist": 1 - (e1 ** 2).mean() / (e0 ** 2).mean(),
                     "oracle_year_skill": 1 - (e2 ** 2).mean() / (e0 ** 2).mean()}
    by_year = P.groupby("year").apply(
        lambda g: pd.Series({"n": len(g), "persist": 100 * rmse(g.obs - g.trail),
                             "hd": 100 * rmse(g.obs - g.hd),
                             "obs_mean": 100 * g.obs.mean(), "trail_mean": 100 * g.trail.mean(),
                             "hd_mean": 100 * g.hd.mean(), "beta": g.beta.iloc[0]}),
        include_groups=False)
    return out, by_year, P


if __name__ == "__main__":
    feats = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("dmay_d", "exceed_d")
    out, by_year, _ = run(feats, scale_by_rate="--scale" in sys.argv)
    print(json.dumps(out, indent=1))
    print(by_year.round(3).to_string())
