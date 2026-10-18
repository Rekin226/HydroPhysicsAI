"""H1a stage 2: evidence for each offset episode and step, and the flag rule (plan WP H1).

Flag rule (plan 2.4 and WP H1):
* dropouts (episodes whose displaced level is flat and > 3 m low, or below the well bottom)
  are invalid readings: flag 9, never corrected;
* an episode or step is corrected with flag 1 only when physical metadata or a co-located
  instrument backs it:
    - metadata: the displaced level is physically impossible (above WellElevation + 5 m or
      below WellElevation - WellDepth - 1 m) while the other side is not, or a station
      metadata date (ModificationDate, EstablishDate) lies within 7 days of the jump;
    - co-located: at least one nested well of the same site (``sid[:6]``) has samples within
      1 h of every jump of the event and none of them jumps by more than 0.3 |j| within 2 h;
* otherwise, a correction corroborated by a leave-out neighbour-regression residual above 4
  calibrated sd gets flag 3 (the corrected value is neighbour-derived; U2 keeps it out of
  calibration targets);
* otherwise flag 9 (kept, uncorrected).

Neighbour regression (daily, pre-origin fit): the 4 wells most correlated with the target
(r >= 0.6 on >= 365 common clean days, within 25 km) that are observed through the test
window; ridge regression on standardized neighbours fitted on pre-origin clean days outside
the test window +-30 days. Statistic: median residual over the displaced days minus the
median over the 15 reference days before (and after, for episodes); steps use 60 days each
side. Its null sd is calibrated on 200 placebo windows of the same length placed on the
target's pre-origin clean record (robust sd); corroborated when |delta| > 4 sd, the sign
agrees with the displacement and |delta - j| < 0.5 |j|. "Refuted" (the neighbours move with
the target: likely real hydraulics) when |delta| < 2 sd and |delta| < 0.25 |j|; it is
reported, the flag stays 9 (plan: flag by default).
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from . import config as C
from . import screens as S

DAY = 86400
SPLIT_DAY = pd.Timestamp(C.ORIGIN_SPLIT)


def daily_from_samples(t, v, mask):
    """Daily mean of hourly means of samples in ``mask``; returns (values, n_hours) Series."""
    if mask.sum() == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    s = pd.Series(v[mask], index=pd.to_datetime(t[mask], unit="s"))
    h = s.resample("h").mean()
    g = h.groupby(h.index.floor("D"))
    return g.mean(), g.count()


def well_daily(res: dict) -> dict:
    """Clean and clean-or-displaced daily series (loading removed) for one stage-1 result."""
    out = {}
    for part in ("pre", "post"):
        if part not in res:
            continue
        a = res[part]
        v = a["v_raw"] - a["L"]
        k = a["kind"]
        clean = k == 0
        disp = (k & ~np.uint16(S.K_EPISODE | S.K_STEP)) == 0
        disp &= (k & np.uint16(S.K_SENTINEL)) == 0
        dc, nc = daily_from_samples(a["t"], v, clean)
        dd, nd = daily_from_samples(a["t"], v, disp)
        out[part] = {"clean": dc[nc >= C.DAY_MIN_HOURS], "disp": dd[nd >= C.DAY_MIN_HOURS]}
    return out


def build_matrices(daily: dict[str, dict], kind: str):
    """(DataFrame days x wells) for pre and post parts."""
    pre = pd.DataFrame({s: d["pre"][kind] for s, d in daily.items() if "pre" in d})
    post = pd.DataFrame({s: d["post"][kind] for s, d in daily.items() if "post" in d})
    return pre.sort_index(), post.sort_index()


def sample_coverage(res: dict, t: int, half_s: int = 3600) -> bool:
    for part in ("pre", "post"):
        if part not in res:
            continue
        a = res[part]
        tt = a["t"][(a["kind"] & S.K_SENTINEL) == 0]
        i = np.searchsorted(tt, t - half_s)
        if i < len(tt) and tt[i] <= t + half_s:
            return True
    return False


def colocated(sid, times, j, site_members, results, jumps_by_sid) -> dict:
    sibs = [s for s in site_members if s != sid and s in results]
    covering, jumped = [], []
    for s in sibs:
        if not all(sample_coverage(results[s], t) for t in times):
            continue
        covering.append(s)
        J = jumps_by_sid.get(s, [])
        for t in times:
            if any(abs(x["t"] - t) <= C.COLOC_WINDOW_S and abs(x["j"]) > C.COLOC_SIB_FRAC * abs(j)
                   for x in J):
                jumped.append(s)
                break
    backed = bool(covering) and not jumped
    return {"siblings": sibs, "covering": covering, "jumped": sorted(set(jumped)),
            "backed": backed}


def metadata_dates(meta_row) -> list[pd.Timestamp]:
    out = []
    for key in ("ModificationDate", "EstablishDate"):
        v = meta_row.get(key)
        try:
            d = pd.to_datetime(str(int(v)), format="%Y%m%d")
            out.append(d)
        except (TypeError, ValueError):
            continue
    return out


def neighbour_test(sid, disp_days, ref_days, j_disp, pre_clean, post_clean, pre_disp,
                   post_disp, xy, exclude_days, use_post: bool = True) -> dict:
    """Leave-out neighbour regression test of a displacement ``j_disp`` over ``disp_days``
    (DatetimeIndex) against ``ref_days``. The regression is always fitted on pre-origin
    days; ``use_post=False`` (pre-origin events) also restricts the test windows, the
    neighbours' coverage and the predictions to pre-origin days, so that the flag of a
    pre-origin row never depends on 2023+ data."""
    if use_post:
        allc = pd.concat([pre_clean, post_clean]).sort_index()
        alld = pd.concat([pre_disp, post_disp]).sort_index()
    else:
        allc, alld = pre_clean, pre_disp
        disp_days = disp_days[disp_days < SPLIT_DAY]
        ref_days = ref_days[ref_days < SPLIT_DAY]
        if len(disp_days) == 0:
            return {"verdict": "untestable", "why": "no pre-origin displaced day"}
    if sid not in pre_clean.columns:
        return {"verdict": "untestable", "why": "no pre-origin daily record"}
    y_pre = pre_clean[sid]
    test_days = disp_days.union(ref_days)
    lo, hi = test_days.min() - pd.Timedelta(days=30), test_days.max() + pd.Timedelta(days=30)
    fit_mask = y_pre.notna() & ~((y_pre.index >= lo) & (y_pre.index <= hi))
    for (a, b) in exclude_days:
        fit_mask &= ~((y_pre.index >= a) & (y_pre.index <= b))
    if sid not in xy:
        return {"verdict": "untestable", "why": "no coordinates"}
    x0, y0 = xy[sid]
    cands = []
    for s in pre_clean.columns:
        if s == sid or s not in xy:
            continue
        if np.hypot(xy[s][0] - x0, xy[s][1] - y0) / 1e3 > C.NEIGH_MAX_KM:
            continue
        both = fit_mask & pre_clean[s].notna()
        if both.sum() < C.NEIGH_MIN_OVERLAP_D:
            continue
        r = np.corrcoef(y_pre[both], pre_clean[s][both])[0, 1]
        if not np.isfinite(r) or r < C.NEIGH_MIN_R:
            continue
        if s not in allc.columns:
            continue
        cov_d = allc[s].reindex(disp_days).notna().mean()
        cov_r = allc[s].reindex(ref_days).notna().sum()
        if cov_d < 0.5 or cov_r < 5:
            continue
        cands.append((r, s))
    if not cands:
        return {"verdict": "untestable", "why": "no qualifying neighbour"}
    nb = [s for _, s in sorted(cands, reverse=True)[:C.NEIGH_K]]
    Xf = pre_clean[nb][fit_mask]
    good = Xf.notna().all(1)
    Xf, yf = Xf[good], y_pre[fit_mask][good]
    if len(yf) < C.NEIGH_MIN_OVERLAP_D:
        return {"verdict": "untestable", "why": "too few common fit days", "neighbours": nb}
    mu, sd = Xf.mean(), Xf.std().replace(0, 1.0)
    Z = ((Xf - mu) / sd).to_numpy()
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    pen = C.NEIGH_RIDGE * len(Z) * np.eye(Z1.shape[1])
    pen[0, 0] = 0
    beta = np.linalg.solve(Z1.T @ Z1 + pen, Z1.T @ yf.to_numpy())

    def pred(days, src):
        X = src[nb].reindex(days)
        Zq = ((X - mu) / sd).to_numpy()
        return pd.Series(np.column_stack([np.ones(len(Zq)), Zq]) @ beta, index=days)

    # residuals: target displaced-or-clean daily values minus the neighbour prediction
    yt = alld[sid] if sid in alld.columns else pd.Series(dtype=float)
    r_d = (yt.reindex(disp_days) - pred(disp_days, allc)).dropna()
    yc = allc[sid] if sid in allc.columns else pd.Series(dtype=float)
    r_r = (yc.reindex(ref_days) - pred(ref_days, allc)).dropna()
    if len(r_d) < max(1, 0.3 * len(disp_days)) or len(r_r) < 5:
        return {"verdict": "untestable", "why": "residual coverage", "neighbours": nb}
    delta = float(np.median(r_d) - np.median(r_r))
    # placebo null on the target's pre-origin clean record (same lengths)
    resid_pre = (y_pre[fit_mask] - pred(y_pre[fit_mask].index, pre_clean)).dropna()
    if len(resid_pre) < 200:
        return {"verdict": "untestable", "why": "short placebo record", "neighbours": nb}
    rr = resid_pre.reindex(pd.date_range(resid_pre.index.min(), resid_pre.index.max(),
                                         freq="D"))
    L = max(1, min(len(disp_days), 365))
    R = C.NEIGH_REF_DAYS if len(ref_days) <= 2 * C.NEIGH_REF_DAYS + 1 else C.NEIGH_STEP_DAYS
    rng = np.random.default_rng(zlib.crc32(f"{sid}|{disp_days[0].date()}|{L}".encode()))
    vals = rr.to_numpy()
    n = len(vals)
    deltas = []
    for _ in range(C.NEIGH_PLACEBOS * 3):
        if n <= L + R + 2:
            break
        s0 = int(rng.integers(R, n - L))
        a = vals[s0:s0 + L]
        b = vals[max(0, s0 - R):s0]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < max(1, 0.3 * L) or len(b) < 5:
            continue
        deltas.append(np.median(a) - np.median(b))
        if len(deltas) >= C.NEIGH_PLACEBOS:
            break
    if len(deltas) < 50:
        return {"verdict": "untestable", "why": "too few placebos", "neighbours": nb}
    d = np.asarray(deltas)
    sdc = float(1.4826 * np.median(np.abs(d - np.median(d))))
    sdc = max(sdc, 1e-3)
    z = delta / sdc
    corro = (abs(delta) > C.NEIGH_SIG * sdc and np.sign(delta) == np.sign(j_disp)
             and abs(delta - j_disp) < C.NEIGH_SIZE_TOL * abs(j_disp))
    refuted = abs(delta) < 2 * sdc and abs(delta) < 0.25 * abs(j_disp)
    return {"verdict": "corroborated" if corro else ("refuted" if refuted else "not_corroborated"),
            "delta_m": delta, "sd_cal_m": sdc, "z": z, "neighbours": nb,
            "n_disp_days": int(len(r_d)), "n_ref_days": int(len(r_r))}


def interior_days(t1: int, t2: int) -> pd.DatetimeIndex:
    """Whole days strictly inside (t1, t2)."""
    a = pd.Timestamp(t1, unit="s").floor("D") + pd.Timedelta(days=1)
    b = pd.Timestamp(t2, unit="s").floor("D") - pd.Timedelta(days=1)
    return pd.date_range(a, b, freq="D") if b >= a else pd.DatetimeIndex([])


def days_before(t: int, n: int) -> pd.DatetimeIndex:
    b = pd.Timestamp(t, unit="s").floor("D") - pd.Timedelta(days=1)
    return pd.date_range(b - pd.Timedelta(days=n - 1), b, freq="D")


def days_after(t: int, n: int) -> pd.DatetimeIndex:
    a = pd.Timestamp(t, unit="s").floor("D") + pd.Timedelta(days=1)
    return pd.date_range(a, a + pd.Timedelta(days=n - 1), freq="D")
