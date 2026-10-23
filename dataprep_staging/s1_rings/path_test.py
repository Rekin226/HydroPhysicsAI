"""Family-Y path test (plan 4.4 'Ring path'): masked calendar years 2017-2020, pre-origin.

Sets the path terms of the 2022 interior sd (declaration S1a.rings_bridged_rules.v1,
'path_sd'). For every ring site, every interval of the product (total, the three bands, and at
Xiutan the shallow pile band and its deep remainder) and every year y of
harness.gaps_guarded.ring_path_years(2017..2020): the months between the last survey month of
y-1 (dated >= Nov 1) and the first of y+1 (dated <= Mar 15) are masked; >= 8 masked months
with surveys are needed. Fills at the masked months' survey dates: linear, climatology (from
2014-05..2021-12 without the masked span), pile-shaped (unit slope; Tuku, Xizhou, Guangfu
totals) and Xiutan's hybrid total. Ring values <= 2021-12 and pile rows <= 2022-12 (the
pre-origin run, design_time_evidence slice) only.

    tools/cpu_slot.sh python -m s1_rings.path_test     (from dataprep_staging/)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import gaps_guarded  # noqa: E402

from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1io  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402


def post_win(site: str):
    return K.POST_WIN_BEICHEN if site == K.BEICHEN else K.POST_WIN


def intervals(site: str, gs: dict) -> dict[str, tuple[str, str]]:
    rings, zbar = gs["rings"], gs["zbar"]
    out = {"total": (rings[0], rings[-1])}
    for i, (a, b) in enumerate(L.band_indices(zbar)):
        out[f"band{i}"] = (rings[a], rings[b])
    if site in K.PILES and K.PILES[site][2] == "shallow":
        r70 = str((zbar - K.PILES[site][1]).abs().idxmin())
        out["shallow"] = (rings[0], r70)
        out["deep"] = (r70, rings[-1])
    return out


def endpoints(s: pd.Series, dates: pd.Series, y: int):
    v = s.dropna()
    left = v.loc[:pd.Period(f"{y - 1}-12", "M")]
    right = v.loc[pd.Period(f"{y + 1}-01", "M"):]
    if left.empty or right.empty:
        return None
    t0, t1 = left.index[-1], right.index[0]
    if dates[t0] < pd.Timestamp(f"{y - 1}-11-01") or dates[t1] > pd.Timestamp(f"{y + 1}-03-15"):
        return None
    inner = v.loc[(v.index > t0) & (v.index < t1)]
    if len(inner) < 8:
        return None
    return t0, t1, inner


def main() -> None:
    out_dir = K.EVIDENCE / "path_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    piles, notes = s1io.piles(preorigin=True)
    years_spec = gaps_guarded.ring_path_years(K.PATH_YEARS)
    rows = []
    for site in K.SITES:
        p = L.load_positions(site)
        gs = L.gap_stats(p, K.PRE_WIN, post_win(site))
        mpos, mdates = L.monthly(p.loc[:K.GAP_LAST_PRE])
        ivs = intervals(site, gs)
        ser = {k: L.interval_series(mpos, a, b) for k, (a, b) in ivs.items()}
        pile = K.PILES.get(site)
        q = piles[pile[0]] if pile else None
        for spec in years_spec:
            y = spec["year"]
            for name, s in ser.items():
                ep = endpoints(s, mdates, y)
                if ep is None:
                    continue
                t0, t1, inner = ep
                d0, d1, dd = mdates[t0], mdates[t1], mdates[inner.index]
                truth = inner.to_numpy(float)
                clim = L.climatology(s, exclude=(t0, t1))
                fills = {}
                fills["linear"], _ = L.fill(s[t0], s[t1], d0, d1, dd, "linear")
                fills["climatology"], _ = L.fill(s[t0], s[t1], d0, d1, dd, "climatology",
                                                 clim=clim)
                if name == "total" and pile and pile[2] == "total":
                    v, _ = L.fill(s[t0], s[t1], d0, d1, dd, "pile", q=q)
                    if np.all(np.isfinite(v)):
                        fills["pile"] = v
                if name == "total" and pile and pile[2] == "shallow":
                    sh, de = ser["shallow"], ser["deep"]
                    if all(np.isfinite([sh.get(t0, np.nan), sh.get(t1, np.nan),
                                        de.get(t0, np.nan), de.get(t1, np.nan)])):
                        fs, _ = L.fill(sh[t0], sh[t1], d0, d1, dd, "pile", q=q)
                        fl, _ = L.fill(sh[t0], sh[t1], d0, d1, dd, "linear")
                        fd, _ = L.fill(de[t0], de[t1], d0, d1, dd, "climatology",
                                       clim=L.climatology(de, exclude=(t0, t1)))
                        fdl, _ = L.fill(de[t0], de[t1], d0, d1, dd, "linear")
                        hyb = fills["linear"] + (fs - fl) + (fd - fdl)
                        if np.all(np.isfinite(hyb)):
                            fills["hybrid"] = hyb
                for meth, v in fills.items():
                    e = (v - truth) * 1000.0
                    rows.append({"site": site, "year": y, "interval": name, "method": meth,
                                 "n": len(truth), "t0": str(t0), "t1": str(t1),
                                 "rmse_mm": float(np.sqrt(np.mean(e ** 2))),
                                 "sse_mm2": float(np.sum(e ** 2)),
                                 "gap_compaction_mm": float((s[t1] - s[t0]) * 1000.0)})
    R = pd.DataFrame(rows)
    R.to_csv(out_dir / "path_rows.csv", index=False)

    def pooled(df):
        return float(np.sqrt(df["sse_mm2"].sum() / df["n"].sum())) if len(df) else float("nan")

    terms = {}
    tot = R[R.interval == "total"]
    for meth in ("linear", "climatology", "pile", "hybrid"):
        d = tot[tot.method == meth]
        terms[f"total.{meth}"] = {"rmse_mm": pooled(d), "site_years": int(len(d)),
                                  "months": int(d["n"].sum()),
                                  "sites": sorted(d.site.unique().tolist())}
    for b in ("band0", "band1", "band2"):
        for meth in ("linear", "climatology"):
            d = R[(R.interval == b) & (R.method == meth)]
            terms[f"{b}.{meth}"] = {"rmse_mm": pooled(d), "site_years": int(len(d)),
                                    "months": int(d["n"].sum())}
    # like-for-like comparison on the site-years where the pile/hybrid method ran
    comp = {}
    for meth in ("pile", "hybrid"):
        keys = tot[tot.method == meth][["site", "year"]]
        sub = tot.merge(keys, on=["site", "year"])
        comp[meth] = {m: pooled(sub[sub.method == m]) for m in ("linear", "climatology", meth)}
    res = {"years": list(K.PATH_YEARS), "terms": terms, "like_for_like": comp,
           "per_site": tot.groupby(["site", "method"]).apply(pooled).unstack().round(2)
           .to_dict(orient="index"),
           "guard_notes": notes,
           "rule": "path_term = pooled RMSE (sqrt(sum e^2 / n)) of the method over site-years "
                   "(declaration S1a.rings_bridged_rules.v1, path_sd)"}
    (out_dir / "path_terms.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v["rmse_mm"] for k, v in terms.items()}, indent=1))
    print(json.dumps(comp, indent=1))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)     # pyarrow thread teardown aborts at interpreter exit on this host
