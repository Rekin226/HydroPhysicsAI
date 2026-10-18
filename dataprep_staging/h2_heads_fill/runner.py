"""Artificial-gap runs for WP H2 (plan 4.4): one realization of one family per call.

Monthly families (N, S, E, H from ``harness.gaps_guarded.head_masks``; T from
``gaps_guarded.replica_test_period`` inside the last 45 pre-origin months) are applied to the
base matrix ``Ybase`` (observed C20 months plus the daily-bridged months as
pseudo-observations, exactly as in the product). Every method then fills the masked matrix,
and the masked cells whose truth is an observed C20 month are scored. Daily families
(d: random gaps; D: the 2026 outage window in every pre-origin year) mask the daily record of
one well at a time, bridge the touched months, and compare with the monthly methods applied
to the base matrix with those well-months masked.

Output: ``<workdir>/runs/<variant>/<family>_r<r>.parquet`` (one row per scored cell; columns
``pred_<method>`` and ``sd_<method>``) and ``.json`` (fit info, timing).

Usage: python -I runner.py VARIANT FAMILY R [R ...]      (VARIANT 'pre'; FAMILY N S E H T d D)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from h2_heads_fill import bridge, io_h2, kriging, methods  # noqa: E402
from h2_heads_fill import config as C  # noqa: E402
from harness import gaps_guarded  # noqa: E402

METHODS = ("router", "neigh", "softimpute_tb", "gp", "backfill", "clim", "clim_ar", "idw",
           "stk")


def base_matrix(h, years_allowed):
    """(Ybase, bridged table): C20 observations plus every bridgeable non-C20 month."""
    br = bridge.bridge_targets(h, years_allowed)
    Yb = h.Y.copy()
    si = {s: i for i, s in enumerate(h.sids)}
    ti = {d: j for j, d in enumerate(h.dates)}
    ok = br[br.ok] if len(br) else br
    for _, r in ok.iterrows():
        Yb[si[r.sid], ti[pd.Timestamp(r.month)]] = r.value
    return Yb, br


def load_base(variant: str):
    """Cached (h, Ybase, bridged) for a variant (the cache lives in the work directory)."""
    h = io_h2.load(variant)
    years = range(2012, 2023) if variant == "pre" else range(2012, 2027)
    cache = C.WORKDIR / f"base_{variant}.npz"
    brc = C.WORKDIR / f"bridged_{variant}.parquet"
    if cache.exists() and brc.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["sids"]) == h.sids and np.array_equal(z["Y"], h.Y, equal_nan=True):
            return h, z["Ybase"], pd.read_parquet(brc)
    Yb, br = base_matrix(h, years)
    C.WORKDIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, sids=np.array(h.sids), Y=h.Y, Ybase=Yb)
    br.to_parquet(brc)
    return h, Yb, br


def idw_defined(obs, xy, layers, max_km=C.IDW_MAX_KM):
    D = methods._distance_km(xy)
    same = layers[:, None] == layers[None, :]
    Wt = ((D <= max_km) & same & (D > 0.05)).astype(float)
    return (Wt @ obs.astype(float)) > 0


def fill_all(Ym, xy, layers, targets, gp_rows=None, run_gp=True):
    """Every method on the masked matrix; returns {method: (mean, sd)} and info."""
    info, comp, si_info, stk_info = {}, {}, {}, {}
    tic = time.time()
    m, s, meth = methods.router_native(Ym, xy, run_gp=run_gp, info=si_info, gp_rows=gp_rows,
                                       components=comp)
    info["t_router"] = time.time() - tic
    out = {"router": (m, s), "neigh": comp["neigh"], "softimpute_tb": comp["softimpute_tb"]}
    if comp["gp"][0] is not None:
        out["gp"] = comp["gp"]
    out["router_method"] = (meth, None)
    out["neigh_used"] = (comp["neigh_used"], None)
    out["backfill"] = methods.backfill(Ym)
    out["clim"] = methods.m0_clim(Ym)
    out["clim_ar"] = methods.m0_clim_ar(Ym)
    out["idw"] = methods.m6_idw_anom(Ym, xy=xy, layers=layers)
    tic = time.time()
    out["stk"] = kriging.stk_fill(Ym, xy=xy, targets=targets, info=stk_info)
    info["t_stk"] = time.time() - tic
    info["softimpute"] = si_info
    info["stk"] = stk_info
    return out, info


def _strata(family, cells):
    if family == "N":
        return [f"N_net{c[1]}" for c in cells]
    if family == "S":
        return [f"S{c[1]:02d}" for c in cells]
    if family == "E":
        return ["E_lead66" if c[1] == 66 else "E_trail48" for c in cells]
    if family == "H":
        return ["H_half12"] * len(cells)
    raise ValueError(family)


def monthly_masks(family, obs, months, r):
    T = obs.shape[1]
    if family == "T":
        lo = T - C.T_WINDOW_MONTHS
        targets = [i for i in range(obs.shape[0])
                   if obs[i, lo:].mean() >= 0.7 and obs[i].sum() >= 60]
        mask, cells = gaps_guarded.replica_test_period(obs, targets, r, (lo, T), months)
        # cells: (kind, g, i, t)
        rows = [{"stratum": "T_H06" if c[0].startswith("H") else "T_S", "gap": c[1],
                 "L": int(c[0][1:]), "w": c[2], "t": c[3]} for c in cells]
        return mask, rows
    targets = np.where(obs.mean(1) >= C.TARGET_MIN_COV)[0]
    last = None
    for rr, mask, cells in gaps_guarded.head_masks(family, obs, targets, months, nreal=r + 1):
        last = (rr, mask, cells)
    rr, mask, cells = last
    assert rr == r
    st = _strata(family, cells)
    rows = [{"stratum": s, "gap": c[0], "L": c[1], "w": c[2], "t": c[3], "edge_dist": c[4]}
            for s, c in zip(st, cells, strict=True)]
    return mask, rows


def cluster_of(family, r, row, sid):
    if family in ("S", "E") or (family == "T" and row["stratum"] == "T_S") or family in (
            "d", "D"):
        return sid
    if family == "N":
        return f"N{r}_{row['gap']}"
    return f"{family}{r}"                      # H, T_H06: the realization


def run_monthly(variant, family, r, h, Yb, clean):
    obs = np.isfinite(Yb)
    mask, rows = monthly_masks(family, obs, h.months, r)
    Ym = np.where(mask, np.nan, Yb)
    C20 = np.isfinite(h.Y)
    rows = [x for x in rows if C20[x["w"], x["t"]]]
    df = pd.DataFrame(rows)
    targets = np.zeros_like(obs)
    targets[df.w.to_numpy(), df.t.to_numpy()] = True
    xy, lay = h.xy(), h.meta.layer.to_numpy()
    gp_rows = sorted(set(df.w.tolist()))
    out, info = fill_all(Ym, xy, lay, targets, gp_rows=gp_rows)
    cls, runlen, edge = methods.classify_cells(np.isfinite(Ym))
    idd = idw_defined(np.isfinite(Ym), xy, lay)
    wi, ti = df.w.to_numpy(), df.t.to_numpy()
    df["family"], df["r"], df["variant"] = family, r, variant
    df["sid"] = [h.sids[i] for i in wi]
    df["month"] = [h.months[t] for t in ti]
    df["cls"] = cls[wi, ti]
    df["runlen"] = runlen[wi, ti]
    df["edge"] = edge[wi, ti]
    df["truth"] = h.Y[wi, ti]
    df["clean"] = clean[wi, ti]
    df["in_model"] = h.meta.in_model.to_numpy()[wi]
    df["idw_defined"] = idd[wi, ti]
    df["router_method"] = out["router_method"][0][wi, ti]
    df["neigh_used"] = out["neigh_used"][0][wi, ti]
    for k in METHODS:
        if k not in out:
            df[f"pred_{k}"] = np.nan
            df[f"sd_{k}"] = np.nan
            continue
        mm, ss = out[k]
        df[f"pred_{k}"] = mm[wi, ti]
        df[f"sd_{k}"] = ss[wi, ti] if ss is not None else np.nan
    df["cluster"] = [cluster_of(family, r, x, s) for x, s in
                     zip(df.to_dict("records"), df.sid, strict=True)]
    info.update({"n_cells": int(len(df)), "n_masked": int(mask.sum()),
                 "n_clusters": int(df.cluster.nunique())})
    return df, info


def run_daily(variant, family, r, h, Yb, clean):
    years = range(2012, 2023) if variant == "pre" else range(2012, 2027)
    end = C.PRE_END if variant == "pre" else C.FULL_END
    if family == "d":
        wins = bridge.d_windows(h, r, end)
    else:
        yrs = C.D_YEARS_PRE if variant == "pre" else C.D_YEARS_FULL
        wins = bridge.D_windows(h, yrs)
    ev = bridge.evaluate_windows(h, wins, years, family)
    if ev.empty:
        return ev, {"n_cells": 0}
    xy, lay = h.xy(), h.meta.layer.to_numpy()
    info = {"n_windows": len(wins)}
    # monthly comparators: mask the touched well-months (D: per year, network-wide)
    groups = [ev.index] if family == "d" else [g.index for _, g in ev.groupby(
        ev.month.dt.year)]
    idd = np.zeros(len(ev), dtype=bool)
    preds = {k: np.full(len(ev), np.nan) for k in METHODS}
    sds = {k: np.full(len(ev), np.nan) for k in METHODS}
    for gi in groups:
        sub = ev.loc[gi]
        mask = np.zeros_like(np.isfinite(Yb))
        if family == "D":
            for j in sorted(set(sub.t)):
                mask[:, j] = True
        mask[sub.w.to_numpy(), sub.t.to_numpy()] = True
        Ym = np.where(mask, np.nan, Yb)
        tg = np.zeros_like(mask)
        tg[sub.w.to_numpy(), sub.t.to_numpy()] = True
        out, inf = fill_all(Ym, xy, lay, tg, run_gp=False)
        info.setdefault("fits", []).append(inf)
        pos = ev.index.get_indexer(gi)
        idd[pos] = idw_defined(np.isfinite(Ym), xy, lay)[sub.w, sub.t]
        for k in METHODS:
            if k in out:
                mm, ss = out[k]
                preds[k][pos] = mm[sub.w, sub.t]
                if ss is not None:
                    sds[k][pos] = ss[sub.w, sub.t]
    for k in METHODS:
        ev[f"pred_{k}"] = preds[k]
        ev[f"sd_{k}"] = sds[k]
    ev["pred_bridge"] = ev["pred"]
    ev["idw_defined"] = idd
    ev["family"], ev["r"], ev["variant"] = family, r, variant
    ev["clean"] = clean[ev.w.to_numpy(), ev.t.to_numpy()]
    ev["in_model"] = h.meta.in_model.to_numpy()[ev.w.to_numpy()]
    ev["stratum"] = [f"{family}_f{bridge.frac_class(f)}" for f in ev.frac]
    ev["cluster"] = ev.sid
    ev["month"] = ev.month.dt.strftime("%Y-%m")
    info.update({"n_cells": int(len(ev)), "n_bridged": int(ev.ok.sum()),
                 "n_clusters": int(ev.cluster.nunique())})
    return ev, info


def main(argv):
    variant, family, rs = argv[0], argv[1], [int(x) for x in argv[2:]]
    if variant != "pre":
        raise SystemExit("artificial-gap runs use the pre-origin matrix (after-the-fact runs "
                         "of the full variant live in s7_after_fact.py)")
    h, Yb, _ = load_base(variant)
    clean = io_h2.clean_truth(h)
    outdir = C.WORKDIR / "runs" / variant
    outdir.mkdir(parents=True, exist_ok=True)
    for r in rs:
        tic = time.time()
        if family in ("d", "D"):
            df, info = run_daily(variant, family, r, h, Yb, clean)
        else:
            df, info = run_monthly(variant, family, r, h, Yb, clean)
        info["seconds"] = time.time() - tic
        df.to_parquet(outdir / f"{family}_r{r}.parquet")
        (outdir / f"{family}_r{r}.json").write_text(json.dumps(info, indent=1, default=float))
        print(f"{variant} {family} r{r}: {info.get('n_cells')} cells, "
              f"{info['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
