"""WP S5 data preparation: every instrument as an observation table on the twin grid.

Conventions: times in decimal years; coordinates in km (EPSG:3826 / TWD97 TM2 divided by
1000); displacement y in mm with the fusion's sign (positive = sinking for leveling, GNSS and
InSAR; positive = shortening for rings and piles, which see a fraction f of u).

Variants
--------
* ``pre``  - pre-origin data only (U7): leveling 2012-2021, gnss_clean/preorigin/v2, InSAR run P
  (LiCSBAS, pre-origin pairs only; cells built by ``s5insarP.py``), piles <= 2022-12, rings
  <= 2021-12 with a pre-origin ring-interval rule.
* ``full`` - the product: the same leveling, gnss_clean/full/v2, insar_069A/v2 (run F), piles to
  2025-11, rings to 2025-05 over the S1 interval (rings read on both sides of the 2022 gap).

Gate-failed inputs (gnss_clean, insar_069A, rings_bridged) are opened through the loader guard
for ``after_the_fact_analysis`` only (the S5 product inherits that restriction); the printed
piles (gate passed) are opened for the same use. Only flags 0-1 enter (U1-strict), except the
InSAR flag-4 values, which enter only through within-segment changes (free per-segment offsets).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
from common import guard  # noqa: E402

USE = "after_the_fact_analysis"


# ============================================================================ grid, time
def grid() -> dict:
    z = np.load(C.TWIN_GRID, allow_pickle=False)
    mask = np.asarray(z["mask"]).astype(bool)
    dx, x0, y0 = float(z["dx"]), float(z["x0"]), float(z["y0"])
    rows, cols = np.nonzero(mask)
    xy = np.column_stack([x0 + (cols + 0.5) * dx, y0 + (rows + 0.5) * dx]) / 1000.0
    lut = -np.ones(mask.shape, int)
    lut[rows, cols] = np.arange(rows.size)
    return {"mask": mask, "dx": dx, "x0": x0, "y0": y0, "rows": rows, "cols": cols,
            "cell_xy_km": xy, "lut": lut, "A": int(rows.size), "nx": int(z["nx"]),
            "ny": int(z["ny"])}


def cell_of(g: dict, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    x_m, y_m = np.asarray(x_m, float), np.asarray(y_m, float)
    col = np.floor((x_m - g["x0"]) / g["dx"]).astype(int)
    row = np.floor((y_m - g["y0"]) / g["dx"]).astype(int)
    ok = (col >= 0) & (col < g["nx"]) & (row >= 0) & (row < g["ny"])
    out = -np.ones(x_m.shape, int)
    out[ok] = g["lut"][row[ok], col[ok]]
    return out


def near_active(g: dict, xy_km: np.ndarray, reach_km: float) -> np.ndarray:
    from s5model import dist
    return dist(xy_km, g["cell_xy_km"]).min(1) <= reach_km


def dec_year(d) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(d))
    start = pd.to_datetime(d.year.astype(str) + "-01-01")
    end = pd.to_datetime((d.year + 1).astype(str) + "-01-01")
    return np.asarray(d.year + (d - start) / (end - start), float)


def month_mid(months) -> np.ndarray:
    p = pd.PeriodIndex(months, freq="M")
    a = p.to_timestamp(how="start")
    b = (p + 1).to_timestamp(how="start")
    return (dec_year(a) + dec_year(b)) / 2.0


def month_start(months) -> np.ndarray:
    return dec_year(pd.PeriodIndex(months, freq="M").to_timestamp(how="start"))


# ============================================================================ leveling (L0)
def leveling(g: dict, start: str = C.LEVELING_START, end: str = C.LEVELING_END
             ) -> tuple[pd.DataFrame, dict]:
    """Benchmark surveys in active cells: sid, date, t, y (mm, + = sinking, = -elevation),
    x_km, y_km, cell. QC: >= 2 surveys; a survey-to-survey |rate| > 0.5 m/yr (reset)
    drops the benchmark."""
    df = pd.read_parquet(C.LEVELING_PANEL, columns=["sid", "datetime", "elev_m", "x_3826",
                                                    "y_3826"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    first = df.sort_values("datetime").groupby("sid")[["x_3826", "y_3826"]].first()
    w = df[(df.datetime >= pd.Timestamp(start)) & (df.datetime <= pd.Timestamp(end))]
    w = w.groupby(["sid", "datetime"], as_index=False)["elev_m"].mean()
    w = w.join(first, on="sid")
    w["cell"] = cell_of(g, w["x_3826"].to_numpy(), w["y_3826"].to_numpy())
    qc = {"benchmarks_in_window": int(w.sid.nunique())}
    w = w[w.cell >= 0]
    qc["in_active_cells"] = int(w.sid.nunique())
    n = w.groupby("sid").size()
    w = w[w.sid.isin(n[n >= C.LEV_MIN_SURVEYS].index)]
    qc["with_min_surveys"] = int(w.sid.nunique())
    bad = []
    for sid, gg in w.sort_values("datetime").groupby("sid"):
        tt = dec_year(gg.datetime)
        dt = np.diff(tt)
        dz = np.abs(np.diff(gg.elev_m.to_numpy()))
        ok = dt > 1e-6
        if np.any(dz[ok] / dt[ok] > C.LEV_MAX_RATE_M_YR):
            bad.append(sid)
    w = w[~w.sid.isin(bad)]
    qc["dropped_reset"] = len(bad)
    qc["benchmarks"] = int(w.sid.nunique())
    qc["surveys"] = int(len(w))
    out = pd.DataFrame({"sid": w.sid.to_numpy(), "date": w.datetime.dt.strftime("%Y-%m-%d"),
                        "t": dec_year(w.datetime), "y": -1000.0 * w.elev_m.to_numpy(),
                        "x_km": w.x_3826.to_numpy() / 1000.0,
                        "y_km": w.y_3826.to_numpy() / 1000.0, "cell": w.cell.to_numpy()})
    out = out.sort_values(["sid", "t"]).reset_index(drop=True)
    return out, qc


# ============================================================================ GNSS (gnss_clean)
def gnss(g: dict, variant: str) -> tuple[pd.DataFrame, dict]:
    pdir = C.GNSS_PRE if variant == "pre" else C.GNSS_FULL
    p = guard.open_product(pdir, USE)
    st = p.read_table("stations.csv")
    st = st[st["role"].astype(str) != "dropped"].copy()
    st["x_km"] = st["x_twd97"] / 1000.0
    st["y_km"] = st["y_twd97"] / 1000.0
    st = st[np.isfinite(st.x_km) & np.isfinite(st.y_km)]
    st = st[near_active(g, st[["x_km", "y_km"]].to_numpy(), C.SITE_REACH_KM)]
    tabs = [("wra", "wra/monthly.parquet"), ("ngl", "ngl/monthly.parquet")]
    if variant == "full":
        tabs.append(("ncku", "wra/ncku_monthly.parquet"))
    rows = []
    for fam, name in tabs:
        if name not in p.outputs:
            continue
        m = p.read_table(name)
        m = m[m["flag"].isin([0, 1]) & (m["n_days"] >= 10) & np.isfinite(m["value"])]
        m = m[m["date"].astype(str) <= C.ORIGIN_END[:7]] if variant == "pre" else m
        m = m.merge(st[["family", "station", "x_km", "y_km"]], on=["family", "station"])
        m["fam"] = fam
        rows.append(m)
    d = pd.concat(rows, ignore_index=True)
    out = pd.DataFrame({"family": d["fam"].to_numpy(), "station": d["station"].to_numpy(),
                        "month": d["date"].astype(str).to_numpy(),
                        "t": month_mid(d["date"].astype(str)),
                        "y": -1000.0 * d["value"].to_numpy(),
                        "sd": 1000.0 * d["sd"].to_numpy(),
                        "x_km": d["x_km"].to_numpy(), "y_km": d["y_km"].to_numpy()})
    out = out[out.t >= C.T_REF].sort_values(["family", "station", "t"]).reset_index(drop=True)
    n = out.groupby(["family", "station"]).size()
    keep = n[n >= 6].index
    out = out.set_index(["family", "station"]).loc[keep].reset_index()
    qc = {"product": str(pdir.relative_to(C.REPO)), "stations": int(len(keep)),
          "months": int(len(out)),
          "by_family": {f: int((out.family == f).sum()) for f in out.family.unique()},
          "stations_by_family": {f: sorted(out[out.family == f].station.unique().tolist())
                                 for f in out.family.unique()}}
    return out, qc


# ============================================================================ InSAR
def insar_cells(variant: str) -> dict:
    """Cell values (A, T) mm positive up, flags, cell segment ids, network segment ids, dates."""
    if variant == "pre":
        z = np.load(C.WORK / "insar_runP_cells.npz", allow_pickle=False)
        return {k: z[k] for k in ("value", "flag", "segment", "network_segment", "dates")}
    p = guard.open_product(C.INSAR_V2, USE)
    z = np.load(p.path("vertical_cells.npz"), allow_pickle=False)
    return {k: z[k] for k in ("value", "flag", "segment", "network_segment", "dates")}


def insar_groups(g: dict, cells: dict, end: str | None = None) -> tuple[dict, dict]:
    """InSAR units (4 x 4-cell block x network segment) and month groups.

    A cell-epoch is usable when its flag is 0 or 4, its value is finite and its own segment id
    equals the network segment id (no extra cell gap so far). A cell enters a unit when usable
    in >= 90 % of the segment's epochs; a unit needs >= 4 cells; a unit-epoch value is the mean
    over the unit's cells when >= 80 % of them are usable. Month groups: calendar month x
    segment; the unit-month value is the mean over its epochs (nep), at the mean epoch time."""
    val = np.asarray(cells["value"], float)
    flag = np.asarray(cells["flag"])
    seg = np.asarray(cells["segment"])
    nseg = np.asarray(cells["network_segment"])
    dates = pd.to_datetime([str(d) for d in cells["dates"]])
    if end is not None:
        keep = dates <= pd.Timestamp(end)
        val, flag, seg, nseg, dates = val[:, keep], flag[:, keep], seg[:, keep], nseg[keep], \
            dates[keep]
    usable = np.isin(flag, [0, 4]) & np.isfinite(val) & (seg == nseg[None, :])
    blk = (g["rows"] // C.INSAR_BLOCK) * 1000 + (g["cols"] // C.INSAR_BLOCK)
    tdec = dec_year(dates)
    months = dates.to_period("M").astype(str)
    units, unit_cells = [], []
    unit_vals = {}
    for s in np.unique(nseg):
        ep = np.flatnonzero(nseg == s)
        cover = usable[:, ep].mean(1)
        for b in np.unique(blk):
            cs = np.flatnonzero((blk == b) & (cover >= C.INSAR_CELL_COVER))
            if cs.size < C.INSAR_BLOCK_MIN_CELLS:
                continue
            u = usable[np.ix_(cs, ep)]
            ok = u.mean(0) >= C.INSAR_EPOCH_COVER
            if ok.sum() < 2:
                continue
            v = np.where(u, val[np.ix_(cs, ep)], np.nan)
            series = np.where(ok, np.nanmean(np.where(u, v, np.nan), axis=0), np.nan)
            uid = len(units)
            units.append({"unit": uid, "block": int(b), "seg": int(s),
                          "x_km": float(g["cell_xy_km"][cs, 0].mean()),
                          "y_km": float(g["cell_xy_km"][cs, 1].mean()), "n_cells": int(cs.size)})
            unit_cells.append(cs)
            unit_vals[uid] = (ep, series)
    U = pd.DataFrame(units)
    groups = []
    for s in np.unique(nseg):
        ep_s = np.flatnonzero(nseg == s)
        for mth in pd.unique(months[ep_s]):
            ep = ep_s[months[ep_s] == mth]
            ids, ys, neps = [], [], []
            for uid in U.index[U.seg == s]:
                e_all, series = unit_vals[uid]
                pos = np.searchsorted(e_all, ep)
                vals = series[pos]
                okv = np.isfinite(vals)
                if okv.sum() == 0:
                    continue
                ids.append(uid)
                ys.append(-float(np.mean(vals[okv])))
                neps.append(int(okv.sum()))
            if len(ids) >= 6:
                groups.append({"seg": int(s), "month": str(mth), "t": float(tdec[ep].mean()),
                               "unit": np.array(ids), "y": np.array(ys),
                               "nep": np.array(neps)})
    qc = {"units": int(len(U)), "groups": len(groups),
          "obs": int(sum(len(gr["unit"]) for gr in groups)),
          "segments": sorted(int(s) for s in np.unique(nseg)),
          "first": str(dates.min().date()), "last": str(dates.max().date()),
          "usable_cell_epochs": float(usable.mean())}
    return {"units": U, "unit_cells": unit_cells, "groups": groups}, qc


# ============================================================================ rings (L0)
def _ring_sites() -> pd.DataFrame:
    st = pd.read_csv(C.RING_STATIONS)
    from s1_rings import s1const as K1
    st = st[st["sub_id"].isin(K1.SITES)].copy()
    st["x_km"] = st["x"] / 1000.0
    st["y_km"] = st["y"] / 1000.0
    return st[["sub_id", "x_km", "y_km"]].rename(columns={"sub_id": "site"})


def _segments(t: np.ndarray, gap_days: float) -> np.ndarray:
    seg = np.zeros(t.size, int)
    if t.size > 1:
        seg[1:] = np.cumsum(np.diff(t) * 365.25 > gap_days)
    return seg


def rings(g: dict, variant: str) -> tuple[pd.DataFrame, dict]:
    """Monthly ring-interval shortening (mm, + = shorter) from the L0 ring depths.

    pre: interval from the shallowest to the deepest ring read in >= 80 % of the 2019-2021
    surveys, data <= 2021-12-31. full: the S1 interval (rings_bridged/v2 ring_corrections.csv
    top_ring and bottom_ring, rings read on both sides of the 2022 gap), all data. Segments break
    at survey gaps > 180 days (free offsets: the 2022 instrument change never enters)."""
    from s1_rings import s1lib
    sites = _ring_sites()
    sites = sites[near_active(g, sites[["x_km", "y_km"]].to_numpy(), C.SITE_REACH_KM)]
    corr = None
    if variant == "full":
        p = guard.open_product(C.RINGS_BRIDGED, USE)
        corr = p.read_table("ring_corrections.csv").set_index("site")
    rows, info = [], {}
    for _, r in sites.iterrows():
        pos = s1lib.load_positions(r.site)
        if variant == "pre":
            w = pos.loc[C.RING_PRE_WIN[0]:C.RING_PRE_WIN[1]]
            cov = w.notna().mean()
            ok = [c for c in pos.columns if cov.get(c, 0.0) >= C.RING_COVER]
            if len(ok) < 2:
                continue
            top, bot = ok[0], ok[-1]
        else:
            top, bot = str(corr.loc[r.site, "top_ring"]), str(corr.loc[r.site, "bottom_ring"])
        c = s1lib.ring_comp_mm(pos, top, bot)
        per = c.index.to_period("M")
        mv = c.groupby(per).median()
        md = pd.Series(c.index, index=per).groupby(level=0).median()
        t = dec_year(pd.DatetimeIndex(md.to_numpy()))
        keep = t >= C.T_REF
        if variant == "pre":
            keep &= t <= dec_year([C.LEVELING_END])[0]
        t, y = t[keep], mv.to_numpy(float)[keep]
        if t.size < 3:
            continue
        seg = _segments(t, C.RING_SEGMENT_GAP_DAYS)
        info[r.site] = {"top": top, "bottom": bot, "n": int(t.size),
                        "segments": int(seg.max() + 1)}
        rows.append(pd.DataFrame({"sensor": f"ring:{r.site}", "kind": "ring", "t": t, "y": y,
                                  "seg": seg, "x_km": r.x_km, "y_km": r.y_km}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, {"sites": info}


# ============================================================================ piles (printed)
def piles(g: dict, variant: str) -> tuple[pd.DataFrame, dict]:
    """Printed monthly pile compression (mm/month, deep_benchmarks_printed/v2, flag 0) as
    cumulative series at month boundaries; a missing month starts a new segment."""
    p = guard.open_product(C.PILES, USE)
    d = p.read_table("printed_monthly.parquet")
    d = d[(d["flag"] == 0) & np.isfinite(d["value"])].copy()
    if variant == "pre":
        d = d[d["month"].astype(str) <= C.ORIGIN_END[:7]]
    gs = guard.open_product(C.GNSS_FULL, USE).read_table("stations.csv")
    gs = gs[gs.family == "wra"].set_index("station")
    rows, info = [], {}
    for sensor, s in d.groupby("sensor_id"):
        code = re.match(r"^([A-Z]+)", str(sensor)).group(1)
        if code not in gs.index:
            continue
        x_km, y_km = gs.loc[code, "x_twd97"] / 1000.0, gs.loc[code, "y_twd97"] / 1000.0
        s = s.sort_values("month")
        per = pd.PeriodIndex(s["month"].astype(str), freq="M")
        brk = np.r_[True, np.diff(per.asi8) != 1]
        seg_m = np.cumsum(brk) - 1
        for k in np.unique(seg_m):
            sel = seg_m == k
            pm = per[sel]
            v = s["value"].to_numpy(float)[sel]
            t = np.r_[month_start(pm[:1].astype(str)), month_start((pm + 1).astype(str))]
            y = np.r_[0.0, np.cumsum(v)]
            rows.append(pd.DataFrame({"sensor": f"pile:{sensor}", "kind": "pile", "t": t,
                                      "y": y, "seg": int(k), "x_km": x_km, "y_km": y_km,
                                      "depth_m": int(s["depth_m"].iloc[0])}))
        info[str(sensor)] = {"months": int(len(s)), "segments": int(seg_m.max() + 1),
                             "first": str(per.min()), "last": str(per.max())}
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, {"sensors": info}


def load_all(variant: str) -> tuple[dict, dict]:
    g = grid()
    lev, q_lev = leveling(g)
    gn, q_g = gnss(g, variant)
    cells = insar_cells(variant)
    ins, q_i = insar_groups(g, cells, end=C.ORIGIN_END if variant == "pre" else None)
    rg, q_r = rings(g, variant)
    pl, q_p = piles(g, variant)
    cmp_ = pd.concat([rg, pl], ignore_index=True)
    data = {"grid": g, "lev": lev, "gnss": gn, "insar": ins, "cmp": cmp_}
    qc = {"variant": variant, "leveling": q_lev, "gnss": q_g, "insar": q_i, "rings": q_r,
          "piles": q_p}
    return data, qc
