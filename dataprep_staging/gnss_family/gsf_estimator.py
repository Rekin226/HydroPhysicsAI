"""Estimator of X5.gnss_seasonal_family.v1. Pure numpy (pandas only in the two daily-table
helpers); no file is read here.

Series are monthly SUBSIDENCE (positive down), keyed by month index m = 12 * year + month - 1,
as in v1 (dataprep_staging/sd_column/gsc_estimator, imported read-only).

Per station: v1's per-year annual coefficients (one joint fit of constant, trend, annual and
semiannual harmonics over the window; its trend removed; per calendar year with >= 8 months a
fit of constant + annual + semiannual gives the year's (a, b)), then the component-wise
MEDIAN of the per-year (a, b) vectors (robust to one jump- or 2021-distorted year); the
station amplitude is the length of that median vector.

Zone statistics on the primary unit (distal and mid Yunlin stations):

* T2 = median over distal stations of the station amplitude (metres);
* T1 = T2 / median over mid stations of the station amplitude (a frame scale common to both
  zones cancels).

Interval: two-stage station-block bootstrap, stations resampled with replacement WITHIN their
zone, then years with replacement within each drawn station; observation and every member
resampled together (the model series sit on the observed months); percentile 90 % interval of
log(T_obs / T_m).

Screen of member m (declared before any GNSS value is read):
  T1: excluded if the interval of log(T1_obs / T1_m) lies wholly outside [-log 1.25, log 1.25];
  T2: excluded if the interval of log(T2_obs / T2_m) lies wholly outside
      [-log 1.25, log 1.42 + log 1.25] (GNSS over-reads by s in [1, 1.42]; true = obs / s);
  a member survives unless excluded by T1 or T2.

Readings by survivor set S, with C the cluster (cut30 within x1.5 of the page column) and F
the far members (the rest), both frozen before any look:
  (a) S meets C, misses F;  (b) S misses C, meets F;  (c) S meets both;  (d) S is empty.
"""
from __future__ import annotations

import math

import gsf_const as K
import numpy as np

READINGS = ("a", "b", "c", "d")
_FLOOR = 1e-12


# --------------------------------------------------------------------------- per station
def year_vectors(m, y, min_months: int = K.MIN_MONTHS) -> tuple[list[int], np.ndarray]:
    """(years, (Y, 2) annual coefficient vectors) of one series (v1's per-year fit)."""
    import gsc_estimator as E

    co = E.year_coefficients(np.asarray(m), np.asarray(y, float), min_months)
    yrs = sorted(co)
    return yrs, np.array([co[k] for k in yrs], float).reshape(len(yrs), 2)


def median_vector(V) -> np.ndarray:
    V = np.asarray(V, float)
    if V.size == 0:
        return np.array([np.nan, np.nan])
    return np.median(V.reshape(-1, 2), axis=0)


def amplitude(V) -> float:
    return float(np.hypot(*median_vector(V)))


def phase_months(V) -> float:
    """Month (0-12) of the annual subsidence maximum of the median vector."""
    a, b = median_vector(V)
    return (math.atan2(b, a) % (2 * math.pi)) / (2 * math.pi) * 12.0


# --------------------------------------------------------------------------- stacking
def stack(pairs: dict, members: list[str], zone_of: dict) -> dict:
    """pairs: {station: {"years": [...], "obs": (Y, 2), "mod": {member: (Y, 2)}}} ->
    {"V": (1 + n_members, n_st, Ymax, 2) NaN-padded, "ny": (n_st,), "zone": {z: idx}}.
    Row 0 of V is the observation."""
    st = sorted(pairs)
    ymax = max(len(pairs[s]["years"]) for s in st)
    V = np.full((1 + len(members), len(st), ymax, 2), np.nan)
    for i, s in enumerate(st):
        n = len(pairs[s]["years"])
        V[0, i, :n] = pairs[s]["obs"]
        for j, mb in enumerate(members):
            V[1 + j, i, :n] = pairs[s]["mod"][mb]
    ny = np.array([len(pairs[s]["years"]) for s in st])
    zone = {z: np.array([i for i, s in enumerate(st) if zone_of[s] == z], int) for z in K.ZONES}
    return {"V": V, "ny": ny, "zone": zone, "stations": st, "members": list(members)}


def _zone_median(V, ny, idx, P, Yk):
    """(K, B) median over drawn stations of the amplitude of the median of drawn years."""
    if idx.size == 0:
        return np.full((V.shape[0], P.shape[0]), np.nan)
    ymax = V.shape[2]
    g = idx[P]                                               # (B, n_z) global station ids
    n = ny[g]
    valid = np.arange(ymax)[None, None, :] < n[..., None]
    G = V[:, g[..., None], Yk]                               # (K, B, n_z, Ymax, 2)
    G = np.where(valid[None, ..., None], G, np.inf)
    G.sort(axis=3)                                           # the n drawn years first
    lo = np.broadcast_to(((n - 1) // 2)[None, ..., None, None], G.shape[:3] + (1, 2))
    hi = np.broadcast_to((n // 2)[None, ..., None, None], G.shape[:3] + (1, 2))
    with np.errstate(all="ignore"):
        med = 0.5 * (np.take_along_axis(G, lo, axis=3) + np.take_along_axis(G, hi, axis=3))
    med = med[..., 0, :]                                     # (K, B, n_z, 2) median of years
    amp = np.hypot(med[..., 0], med[..., 1])
    return np.median(amp, axis=2)


def _picks(rng, stk: dict, B: int):
    out = {}
    ymax = stk["V"].shape[2]
    for z, idx in stk["zone"].items():
        P = rng.integers(0, max(idx.size, 1), size=(B, idx.size))
        n = stk["ny"][idx[P]] if idx.size else np.zeros((B, 0), int)
        u = rng.random(size=(B, idx.size, ymax))
        Yk = np.minimum((u * n[..., None]).astype(int), np.maximum(n[..., None] - 1, 0))
        out[z] = (P, Yk)
    return out


def _identity(stk: dict):
    out = {}
    ymax = stk["V"].shape[2]
    for z, idx in stk["zone"].items():
        P = np.arange(idx.size)[None, :]
        Yk = np.broadcast_to(np.arange(ymax), (1, idx.size, ymax)).copy()
        out[z] = (P, Yk)
    return out


def zone_stats(stk: dict, picks: dict) -> dict:
    """{"T2": (K, B), "T1": (K, B)} for the given picks (row 0 the observation)."""
    V, ny = stk["V"], stk["ny"]
    d = _zone_median(V, ny, stk["zone"]["distal"], *picks["distal"])
    m = _zone_median(V, ny, stk["zone"]["mid"], *picks["mid"])
    with np.errstate(all="ignore"):
        return {"T2": d, "T1": d / m}


def point(stk: dict) -> dict:
    """Point T1 and T2 of the observation (row 0) and every member."""
    zs = zone_stats(stk, _identity(stk))
    return {"T1": zs["T1"][:, 0], "T2": zs["T2"][:, 0]}


def _log(x):
    return np.log(np.maximum(np.asarray(x, float), _FLOOR))


def screen(stk: dict, rng, B: int, level: float = K.LEVEL) -> dict:
    """Bootstrap intervals of log(T_obs / T_m) for every member and the screen decision."""
    pt = point(stk)
    zs = zone_stats(stk, _picks(rng, stk, B))
    a = (1.0 - level) / 2.0
    out = {}
    for j, mb in enumerate(stk["members"]):
        rec = {}
        for t, band in (("T1", K.T1_BAND), ("T2", K.T2_BAND)):
            d = _log(zs[t][0]) - _log(zs[t][1 + j])
            d = d[np.isfinite(d)]
            lo, hi = (np.quantile(d, [a, 1 - a]) if d.size else (np.nan, np.nan))
            est = float(_log(pt[t][0]) - _log(pt[t][1 + j]))
            excl = bool(np.isfinite(lo) and (hi < band[0] or lo > band[1]))
            rec[t] = {"log_ratio": est, "ci90": [float(lo), float(hi)], "excluded": excl}
        rec["excluded"] = rec["T1"]["excluded"] or rec["T2"]["excluded"]
        out[mb] = rec
    return {"members": out, "T1_obs": float(pt["T1"][0]), "T2_obs_m": float(pt["T2"][0]),
            "T1_model": {mb: float(pt["T1"][1 + j]) for j, mb in enumerate(stk["members"])},
            "T2_model_m": {mb: float(pt["T2"][1 + j]) for j, mb in enumerate(stk["members"])},
            "survivors": sorted(mb for mb, r in out.items() if not r["excluded"]),
            "B": int(B), "level": level}


def screen_fast(stk: dict, rng, B: int, level: float = K.LEVEL) -> np.ndarray:
    """Boolean (n_members,) exclusion vector (the screen's decision only; G0 speed path,
    identical rule to :func:`screen`)."""
    zs = zone_stats(stk, _picks(rng, stk, B))
    a = (1.0 - level) / 2.0
    ex = np.zeros(len(stk["members"]), bool)
    for t, band in (("T1", K.T1_BAND), ("T2", K.T2_BAND)):
        d = _log(zs[t][0])[None, :] - _log(zs[t][1:])
        lo, hi = np.nanquantile(d, [a, 1 - a], axis=1)
        ex |= np.isfinite(lo) & ((hi < band[0]) | (lo > band[1]))
    return ex


def half_widths(stk: dict, rng, B: int, member: str, level: float = K.LEVEL) -> dict:
    """90 % half-widths of log(T_obs / T_member) (G-sep's noise scale)."""
    j = stk["members"].index(member)
    zs = zone_stats(stk, _picks(rng, stk, B))
    a = (1.0 - level) / 2.0
    out = {}
    for t in ("T1", "T2"):
        d = _log(zs[t][0]) - _log(zs[t][1 + j])
        lo, hi = np.nanquantile(d, [a, 1 - a])
        out[t] = float((hi - lo) / 2.0)
    return out


def reading(survivors, cluster, far) -> str:
    """(a)-(d) from the survivor set (module doc)."""
    s = set(survivors)
    c, f = bool(s & set(cluster)), bool(s & set(far))
    if c and not f:
        return "a"
    if f and not c:
        return "b"
    if c and f:
        return "c"
    return "d"


def cluster_far(cut30: dict, members, page: str = K.PAGE, x: float = K.CLUSTER_X) -> dict:
    """{"cluster": [...], "far": [...]} by cut30 relative to the page column."""
    lp = math.log(cut30[page])
    cl = sorted(m for m in members if abs(math.log(cut30[m]) - lp) <= math.log(x))
    return {"cluster": cl, "far": sorted(set(members) - set(cl))}


def far_pairs(cut30: dict, members, x: float = K.FAR_X) -> dict:
    """{truth: [members more than x x off the truth in cut30]}."""
    return {t: sorted(m for m in members
                      if abs(math.log(cut30[m]) - math.log(cut30[t])) > math.log(x))
            for t in members}


def screened_range(cut30: dict, survivors) -> list | None:
    v = [cut30[m] for m in survivors]
    return [min(v), max(v)] if v else None


# --------------------------------------------------------------------------- U2
def non_spring(m) -> np.ndarray:
    """Boolean mask of the months outside SPRING_MONTHS (applied to pile AND model sides)."""
    return ~np.isin(np.asarray(m) % 12, K.SPRING_MONTHS)


def _site_ratio(v: dict, pk=None) -> float:
    o, m = np.asarray(v["obs"]), np.asarray(v["mod"])
    if pk is not None:
        o, m = o[pk], m[pk]
    am = amplitude(m)
    return amplitude(o) / am if am > 0 else float("nan")


def u2_site(gnss: dict, pile: dict, rng, B: int, level: float = K.LEVEL) -> dict:
    """One-sided bound on the GNSS frame scale s at ONE site (GNSS station and its pile).

    gnss = {"obs": (Y, 2), "mod": (Y, 2)} at the site's GNSS station (model = the page column
    at the GNSS cell); pile = {"obs", "mod"} at the pile (model = the page column's FULL-column
    subsidence at the pile cell, spring months masked on both sides). A pile measures a partial
    column, so p = pile/model <= r; g = GNSS/model = s r; hence s <= g / p. Years resampled
    within the station and within the sensor; the bound is the upper end of the 90 %
    percentile interval of g / p. No pooling across sites."""
    g, p = _site_ratio(gnss), _site_ratio(pile)
    draws = []
    for _ in range(B):
        gb = _site_ratio(gnss, rng.integers(0, len(gnss["obs"]), len(gnss["obs"])))
        pb = _site_ratio(pile, rng.integers(0, len(pile["obs"]), len(pile["obs"])))
        if pb > 0 and np.isfinite(gb):
            draws.append(gb / pb)
    a = (1.0 - level) / 2.0
    lo, hi = (np.quantile(draws, [a, 1 - a]) if draws else (np.nan, np.nan))
    return {"g_gnss_over_model": g, "p_pile_over_model": p,
            "s_point": g / p if p > 0 else float("nan"), "ci90": [float(lo), float(hi)],
            "s_max": float(hi), "B": int(B), "n_draws": len(draws)}


def u2_reading(s_max: float) -> str:
    if not np.isfinite(s_max):
        return "not evaluable"
    if s_max < 1.0:
        return "inconsistent with a pure frame over-read (bound below 1)"
    if s_max < K.FRAME_MAX:
        return f"frame over-read at most x{s_max:.2f} (tighter than x{K.FRAME_MAX})"
    return f"no tighter than x{K.FRAME_MAX}"


def u2_bound(sites: dict, rng, B: int, level: float = K.LEVEL) -> dict:
    """sites: {site: {"gnss": {...}, "pile": {...}, "gated": bool}}. One bound per site; the
    declared reading is the tightest bound over GATED sites (sensors with >= MIN_YEARS_U2
    years); report-only sites are computed and printed beside, never read."""
    if not sites:
        return {"evaluable": False, "reason": "no paired site"}
    per = {k: {**u2_site(v["gnss"], v["pile"], rng, B, level), "gated": bool(v["gated"])}
           for k, v in sorted(sites.items())}
    gated = {k: v for k, v in per.items() if v["gated"] and np.isfinite(v["s_max"])}
    if not gated:
        return {"evaluable": False, "reason": "no gated site", "sites": per}
    k = min(gated, key=lambda x: gated[x]["s_max"])
    return {"evaluable": True, "sites": per, "bound_site": k, "s_max": gated[k]["s_max"],
            "reading": u2_reading(gated[k]["s_max"])}


# --------------------------------------------------------------------------- observed side
def qc_jump_months(daily, jump_m: float = K.QC_JUMP_M,
                   per_month: int = K.QC_JUMPS_PER_MONTH) -> set:
    """REPORT-ONLY (U7, review S1). {(station, 'YYYY-MM')} months with >= per_month
    consecutive-calendar-day changes of value_raw larger than jump_m (rows with a finite
    value_raw, any flag; WRA family only: the caller passes WRA rows)."""
    import pandas as pd

    out = set()
    d = daily[np.isfinite(daily["value_raw"].astype(float))]
    for s, g in d.groupby("station"):
        g = g.sort_values("date")
        t = pd.to_datetime(g["date"]).to_numpy()
        v = g["value_raw"].to_numpy(float)
        if v.size < 2:
            continue
        dd = np.diff(t).astype("timedelta64[D]").astype(int)
        jump = (np.abs(np.diff(v)) > jump_m) & (dd == 1)
        months = pd.Series(jump.astype(int), index=g["date"].str[:7].to_numpy()[1:])
        cnt = months.groupby(level=0).sum()
        out |= {(str(s), str(k)) for k, c in cnt.items() if c >= per_month}
    return out


def drop_months(daily, months: set):
    """Daily rows minus the (station, month) pairs in ``months``."""
    if not months:
        return daily
    key = list(zip(daily["station"].astype(str), daily["date"].str[:7], strict=True))
    keep = np.array([k not in months for k in key])
    return daily[keep]


def pile_monthly(daily, sensor: str, window=K.PILE_WINDOW, flags=K.PILE_FLAGS,
                 feeds=K.PILE_FEEDS, min_days: int = K.MIN_DAYS,
                 variant: str = K.PILE_VARIANT):
    """(m, compaction in metres, feed) of one pile sensor, rows of the pre-origin run only
    (variant == 'preorigin'): ONE feed per sensor (the feed of
    ``feeds`` with the most flagged-in days in the window; ties to the first listed),
    monthly medians of >= min_days days; deep_benchmarks values are mm, compression
    positive (= subsidence positive)."""
    d = daily[(daily["sensor_id"] == sensor) & daily["flag"].isin(list(flags))
              & (daily["variant"] == variant)]
    d = d[(d["date"].str[:7] >= window[0]) & (d["date"].str[:7] <= window[1])]
    d = d[np.isfinite(d["value"].astype(float))]
    counts = {f: int((d["feed"] == f).sum()) for f in feeds}
    feed = max(feeds, key=lambda f: (counts[f], -feeds.index(f)))
    g = d[d["feed"] == feed]
    if g.empty:
        return np.array([], int), np.array([], float), feed
    agg = g.groupby(g["date"].str[:7])["value"].agg(["median", "count"])
    agg = agg[agg["count"] >= min_days]
    m = np.array([12 * int(k[:4]) + int(k[5:7]) - 1 for k in agg.index], dtype="int64")
    return m, agg["median"].to_numpy(float) / 1000.0, feed


# --------------------------------------------------------------------------- report-only F1
def deseason(m, y) -> np.ndarray:
    """Linear trend and a 12-month climatology (calendar-month means) removed."""
    m = np.asarray(m)
    y = np.asarray(y, float)
    t = (m - m.mean()) / 12.0
    X = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    for k in range(12):
        sel = (m % 12) == k
        if sel.any():
            r[sel] = r[sel] - r[sel].mean()
    return r


def f1_beta(series: dict, elec: dict, zone_of: dict, drop_year: int | None = None) -> dict:
    """Report-only F1 (C-R2): per zone and pooled, the slope of the deseasonalised monthly
    subsidence anomaly on the deseasonalised zone electricity anomaly (lag 0), pooled over
    stations. series: {station: (m, y)}; elec: {zone: {m: anomaly}} (deseasonalised on the
    electricity side only over the GNSS window); both sides are then deseasonalised again on
    each station's own months (trend and calendar-month means), so they share one filter."""
    num = {z: 0.0 for z in (*K.ZONES, "pooled")}
    den = {z: 0.0 for z in (*K.ZONES, "pooled")}
    for s, (m, y) in series.items():
        z = zone_of[s]
        if z not in elec:
            continue
        m = np.asarray(m)
        keep = np.array([int(k) in elec[z] for k in m])
        if drop_year is not None:
            keep &= (m // 12) != drop_year
        if keep.sum() < 12:
            continue
        r = deseason(m[keep], np.asarray(y, float)[keep])
        e = deseason(m[keep], np.array([elec[z][int(k)] for k in m[keep]], float))
        r = r - r.mean()
        for k in (z, "pooled"):
            num[k] += float(r @ e)
            den[k] += float(e @ e)
    return {k: (num[k] / den[k] if den[k] > 0 else float("nan")) for k in num}
