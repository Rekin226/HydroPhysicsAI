"""Head-driven rate modulation on the 2001-2026 head record (round-12 exploratory).

Heads: public hydrological-yearbook monthly means 2001-2011 (local extraction, validated on
overlap years) joined to L1_clean v3 from 2012. Driver: year-on-year change of each well's
hydrological-year mean, inverse-distance weighted to benchmarks, smoothed over 10 km and
centred on its mean over the trailing ``n`` years. Model: increment = layer rate x
(1 + gamma x driver); gamma for test year y is fitted on years < y only.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataprep_staging.r12_leveling.increments import SHARED, hydro_year, site_table
from dataprep_staging.r12_leveling.layer import kernel_rate
from dataprep_staging.r12_leveling.paths import YEARBOOK as YB
from dataprep_staging.r12_leveling.paths import out

YEARS = list(range(2002, 2027))


def monthly_heads(yb_flags=(0, 1, 2)) -> pd.DataFrame:
    y = pd.read_parquet(YB)
    y = y[(y.year <= 2011) & y.parse_flag.isin(yb_flags) & y.value_m.notna()]
    a = pd.DataFrame({"sid": y.sid, "date": pd.to_datetime(dict(year=y.year, month=y.month,
                                                                 day=1)), "value": y.value_m})
    h = pd.read_parquet(SHARED + "heads_monthly.parquet")
    h = h[h.flag <= 1][["sid", "date", "value"]]
    return pd.concat([a, h]).pivot_table(index="date", columns="sid", values="value")


def well_dmean(W: pd.DataFrame, min_months: int = 9) -> pd.DataFrame:
    g = W.groupby(hydro_year(W.index))
    m = g.mean().where(g.count() >= min_months)
    return m.diff().where(m.notna() & m.shift(1).notna()).reindex(YEARS).T   # wells x years


def to_benchmarks(tab: pd.DataFrame, xy: np.ndarray, L: float = 10_000.0) -> np.ndarray:
    meta = pd.read_parquet(SHARED + "station_meta_clean.parquet")
    meta = meta.set_index("sid").reindex(tab.index)
    wxy = meta[["x_twd97", "y_twd97"]].to_numpy("float64")
    V = tab.to_numpy("float64")
    out = np.full((len(xy), V.shape[1]), np.nan)
    for j in range(V.shape[1]):
        ok = np.isfinite(V[:, j]) & np.isfinite(wxy).all(1)
        if ok.sum() < 20:
            continue
        d, ix = cKDTree(wxy[ok]).query(xy, k=6)
        w = 1.0 / np.maximum(d, 500.0) ** 2
        f = (w * V[ok, j][ix]).sum(1) / w.sum(1)
        d2, i2 = cKDTree(xy).query(xy, k=200)
        w2 = np.exp(-0.5 * (d2 / L) ** 2)
        out[:, j] = (w2 * f[i2]).sum(1) / w2.sum(1)
    return out, meta


def run(n: int = 3, first_test: int = 2013):
    d, xy, sids, _ = site_table()
    D = d.reindex(columns=YEARS).to_numpy("float64")
    F, meta = to_benchmarks(well_dmean(monthly_heads()), xy)
    wxy = meta[["x_twd97", "y_twd97"]].to_numpy("float64")
    ok = np.isfinite(wxy).all(1)
    county = meta.CountyName.to_numpy()[ok][cKDTree(wxy[ok]).query(xy)[1]]
    rows = []
    for y in YEARS[n + 1:]:
        j = YEARS.index(y)
        if y > 2021:
            break
        w = D[:, j - n:j]
        tr = np.nanmean(w, axis=1)
        tr[np.isfinite(w).sum(1) < 2] = np.nan
        if np.isfinite(tr).sum() < 50:
            continue
        base = kernel_rate(tr, xy, None, 3.0, 4.0)
        ref = np.nanmean(F[:, j - n:j], axis=1)
        x = (F[:, j] - ref) * np.clip(base, 0.002, None)
        rows.append(pd.DataFrame({"sid": sids, "y": y, "obs": D[:, j], "base": base, "x": x,
                                  "drv": F[:, j] - ref, "county": county}))
    T = pd.concat(rows).dropna()
    out = []
    for y in sorted(T.y.unique()):
        a, b = T[T.y < y], T[T.y == y].copy()
        if y < first_test or a.y.nunique() < 2:
            continue
        g = float((a.x * (a.obs - a.base)).sum() / (a.x ** 2).sum())
        b["hd"], b["gamma"], b["n_train_years"] = b.base + g * b.x, g, a.y.nunique()
        out.append(b)
    return T, pd.concat(out)


def rm(e):
    return 100 * float(np.sqrt((e ** 2).mean()))


if __name__ == "__main__":
    T, P = run(first_test=int(sys.argv[1]) if len(sys.argv) > 1 else 2013)
    print("years with driver and leveling:", sorted(T.y.unique().tolist()))
    for nm, q in (("all", P), ("Changhua", P[P.county == "彰化縣"]),
                  ("Yunlin", P[P.county == "雲林縣"])):
        sk = 1 - ((q.obs - q.hd) ** 2).mean() / ((q.obs - q.base) ** 2).mean()
        print(f"{nm:9s} n={len(q):5d} layer {rm(q.obs - q.base):.3f} cm  head-driven "
              f"{rm(q.obs - q.hd):.3f} cm  skill {100 * sk:+.1f} %")
    by = P.groupby("y").apply(lambda q: pd.Series({
        "n": len(q), "train_years": q.n_train_years.iloc[0], "gamma": q.gamma.iloc[0],
        "layer": rm(q.obs - q.base), "hd": rm(q.obs - q.hd),
        "obs_mean": 100 * q.obs.mean(), "layer_mean": 100 * q.base.mean(),
        "hd_mean": 100 * q.hd.mean(), "driver_m": q.drv.mean()}), include_groups=False)
    print(by.round(3).to_string())
    P.to_parquet(out("longrecord_pairs.parquet"))
