"""Statistical subsidence layer, forward-selected (round-12 exploratory, design-time).

Family: benchmark rate = Gaussian-kernel average (length L km, self included with weight
``self_w``) of trailing-n-year mean increments of surrounding benchmarks. (n, L, self_w)
for origin o and horizon h are chosen on targets that closed at or before o; so is the
stacking weight of the page column and of the head-change term. Nothing from the scored
target enters its own prediction.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataprep_staging.r12_leveling.backtest import HORIZONS, YEARS, column_at_sites, trailing
from dataprep_staging.r12_leveling.increments import site_table
from dataprep_staging.r12_leveling.nowcast import features

ORIGINS = list(range(2009, 2021))
GRID = list(itertools.product((3, 5, 8), (1.5, 3.0, 6.0, 12.0), (0.0, 1.0, 4.0)))


def kernel_rate(r, xy, tree, L_km, self_w, kmax=60):
    ok = np.isfinite(r)
    if ok.sum() < 10:
        return np.full(len(r), np.nan)
    t = cKDTree(xy[ok])
    d, idx = t.query(xy, k=min(kmax, int(ok.sum())))
    w = np.exp(-0.5 * (d / (1000 * L_km)) ** 2)
    w[d < 1.0] *= self_w                      # the benchmark itself
    den = w.sum(1)
    out = (w * r[ok][idx]).sum(1) / np.where(den > 0, den, np.nan)
    nn = (w > 0.05).sum(1)
    out[nn < 2] = np.nan
    return out


def build():
    d, xy, sids, elev = site_table()
    E = elev.reindex(columns=YEARS).to_numpy("float64")
    D = d.reindex(columns=YEARS).to_numpy("float64")
    _, county = features(xy)
    col = column_at_sites(xy)
    tree = cKDTree(xy)
    rows = []
    for o in ORIGINS:
        j = YEARS.index(o)
        cand = {}
        for n, L, sw in GRID:
            r = trailing(D, j, n, min(n, 3) if n > 3 else 2)
            cand[f"k_{n}_{L}_{sw}"] = kernel_rate(r, xy, tree, L, sw)
        r3 = trailing(D, j, 3, 2)
        for h in HORIZONS:
            if o + h > YEARS[-1]:
                continue
            tgt = E[:, j] - E[:, j + h]
            tgt[np.abs(tgt) > 0.5 * h] = np.nan
            colp = (col[o + h] - col[o]) if o >= 2012 and (o + h) in col \
                else np.full(len(sids), np.nan)
            f = pd.DataFrame({"sid": sids, "o": o, "h": h, "obs": tgt, "r3": r3 * h,
                              "column": colp, "county": county})
            for k, v in cand.items():
                f[k] = v * h
            rows.append(f[np.isfinite(tgt)])
    return pd.concat(rows, ignore_index=True)


def forward_select(T):
    names = [c for c in T.columns if c.startswith("k_")]
    T["layer"] = np.nan
    T["rule"] = ""
    T["stack"] = np.nan
    for o in ORIGINS:
        for h in HORIZONS:
            te = (T.o == o) & (T.h == h)
            tr = (T.h == h) & (T.o + T.h <= o)
            if te.sum() == 0 or T[tr].o.nunique() < 2:
                continue
            A = T.loc[tr]
            mse = {k: float(((A.obs - A[k]) ** 2).mean()) for k in names
                   if A[k].notna().mean() > 0.9}
            best = min(mse, key=mse.get)
            T.loc[te, "layer"] = T.loc[te, best]
            T.loc[te, "rule"] = best
            # stack with the column where both exist in the training targets
            B = A.dropna(subset=["column", best, "obs"])
            if B.o.nunique() >= 2:
                X = np.stack([B[best].to_numpy(), B.column.to_numpy()], 1)
                beta = np.linalg.lstsq(X, B.obs.to_numpy(), rcond=None)[0]
                T.loc[te, "stack"] = T.loc[te, best] * beta[0] + T.loc[te, "column"] * beta[1]
                T.loc[te, "b_layer"], T.loc[te, "b_col"] = beta
    return T


def rm(e):
    e = e[np.isfinite(e)]
    return 100 * float(np.sqrt((e ** 2).mean())) if len(e) else np.nan


def table(T):
    S = T.dropna(subset=["layer", "r3"])
    out = []
    for cname, cs in (("all", None), ("Changhua", "彰化縣"), ("Yunlin", "雲林縣")):
        for h in HORIZONS:
            g = S[(S.h == h) & ((S.county == cs) if cs else True)]
            c = g.dropna(subset=["column"])
            s = g.dropna(subset=["stack"])
            out.append({"county": cname, "h": h, "n": len(g), "origins": g.o.nunique(),
                        "persist_r3": rm(g.obs - g.r3), "layer": rm(g.obs - g.layer),
                        "n_col": len(c), "column": rm(c.obs - c.column),
                        "layer@col": rm(c.obs - c.layer), "n_stack": len(s),
                        "stack": rm(s.obs - s["stack"]), "layer@stack": rm(s.obs - s.layer)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    T = forward_select(build())
    from dataprep_staging.r12_leveling.paths import out

    T.to_parquet(out("layer_pairs.parquet"))
    pd.set_option("display.width", 250)
    print(table(T).round(2).to_string(index=False))
    print(T.dropna(subset=["layer"]).groupby(["h", "o"]).agg(
        rule=("rule", "first"), b_layer=("b_layer", "first"), b_col=("b_col", "first")
    ).round(2).to_string())
