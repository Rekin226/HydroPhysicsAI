"""Core computations of WP S1b (pure functions, no file access).

Sign conventions: every series is "down" in mm. Instruments: ``d(t)`` = subsidence (GNSS
``-1000 h``, InSAR ``-value``). Rings: ``c(t)`` = shortening of the top-to-bottom interval
(``-1000 (S - S0)``). Times are decimal years since 2000-01-01.

Instrument-implied ring gap compaction (plan WP S1 (c), the pilot's ``gnss_stepfix.py`` and
S1a's ``gnss_sensitivity.py``, with the seasonal term and the uncertainty made explicit):

* each window w (pre-gap, post-gap) gives an excess ``E_w`` over the gap: the instrument-minus-
  ring rate times the gap length, plus the instrument-minus-ring seasonal change between the
  last pre-gap survey ``t_lp`` and the first post-gap survey ``t_fp`` (harmonics from the
  pre-gap window, held in the post-gap window);
* ``C_w = G - E_w`` with ``G`` the instrument's change between ``t_lp`` and ``t_fp``;
* the product's bridged gap compaction passes at a site when it lies inside
  ``[min_w(C_w - k s_w), max_w(C_w + k s_w)]`` (the excess-rate bracket widened by k sigma),
  ``s_w^2 = var(E_w) + s_end^2 + 2 sd_s^2 (+ InSAR bridges) (+ se^2 of GNSS steps removed
  inside the gap)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

T0 = pd.Timestamp("2000-01-01")


def years(t) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(t))
    return np.asarray((t - T0).days, dtype=float) / 365.25


def ar1_factor(resid: np.ndarray) -> tuple[float, float]:
    """Variance inflation (1 + rho) / (1 - rho) for the lag-1 autocorrelation of time-ordered
    residuals, rho clipped to [0, 0.95]."""
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return 1.0, 0.0
    r = r - r.mean()
    den = float((r ** 2).sum())
    rho = float((r[1:] * r[:-1]).sum() / den) if den > 0 else 0.0
    rho = min(max(rho, 0.0), 0.95)
    return (1.0 + rho) / (1.0 - rho), rho


def harmonic(t) -> np.ndarray:
    t = np.asarray(t, float)
    return np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], -1)


@dataclass
class PreFit:
    """``y = a + r (t - t_ref) + h1 sin 2 pi t + h2 cos 2 pi t``; ``cov`` AR(1)-inflated."""
    beta: np.ndarray
    cov: np.ndarray
    t_ref: float
    n: int
    rho: float
    resid_sd: float
    seasonal: bool

    @property
    def rate(self) -> float:
        return float(self.beta[1])

    def season(self, t) -> np.ndarray:
        if not self.seasonal:
            return np.zeros_like(np.asarray(t, float))
        return harmonic(t) @ self.beta[2:4]

    @property
    def cov_h(self) -> np.ndarray:
        return self.cov[2:4, 2:4] if self.seasonal else np.zeros((2, 2))


@dataclass
class PostFit:
    """``y - s(t) = a_seg + r (t - t_ref)`` with one intercept per segment."""
    rate: float
    var_rate: float
    n: int
    rho: float
    n_seg: int


def fit_pre(t, y, t_ref: float, seasonal: bool = True) -> PreFit | None:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    p = 4 if seasonal else 2
    if len(y) < p + 2:
        return None
    o = np.argsort(t)
    t, y = t[o], y[o]
    cols = [np.ones_like(t), t - t_ref]
    if seasonal:
        cols += list(harmonic(t).T)
    X = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float((res ** 2).sum() / max(len(y) - p, 1))
    f, rho = ar1_factor(res)
    try:
        cov = f * s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    return PreFit(beta=beta, cov=cov, t_ref=float(t_ref), n=len(y), rho=rho,
                  resid_sd=math.sqrt(s2), seasonal=seasonal)


def fit_post(t, y, t_ref: float, season: PreFit, seg=None) -> PostFit | None:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    seg = np.zeros(len(t), int) if seg is None else np.asarray(seg)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y, seg = t[ok], y[ok], seg[ok]
    o = np.argsort(t)
    t, y, seg = t[o], y[o] - season.season(t[o]), seg[o]
    labs = sorted(set(seg.tolist()))
    if len(y) < len(labs) + 3:
        return None
    X = np.stack([(seg == s).astype(float) for s in labs] + [t - t_ref], 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float((res ** 2).sum() / max(len(y) - X.shape[1], 1))
    f, rho = ar1_factor(res)
    try:
        cov = f * s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    return PostFit(rate=float(beta[-1]), var_rate=float(cov[-1, -1]), n=len(y), rho=rho,
                   n_seg=len(labs))


def excess(inst_pre: PreFit, ring_pre: PreFit, inst_post: PostFit, ring_post: PostFit,
           y_lp: float, y_fp: float) -> dict:
    """Excess (instrument minus ring, mm) over the gap from each window, with its variance."""
    gap = y_fp - y_lp
    dh = harmonic(y_fp) - harmonic(y_lp)                 # (2,)
    ds = float(dh @ (inst_pre.beta[2:4] - ring_pre.beta[2:4])) if inst_pre.seasonal else 0.0
    g = np.array([0.0, gap, dh[0], dh[1]]) if inst_pre.seasonal else np.array([0.0, gap])
    cov_pre = inst_pre.cov + ring_pre.cov
    e_pre = (inst_pre.rate - ring_pre.rate) * gap + ds
    v_pre = float(g @ cov_pre @ g)
    e_post = (inst_post.rate - ring_post.rate) * gap + ds
    v_h = float(dh @ (inst_pre.cov_h + ring_pre.cov_h) @ dh) if inst_pre.seasonal else 0.0
    v_post = gap ** 2 * (inst_post.var_rate + ring_post.var_rate) + v_h
    return {"gap_yr": gap, "seasonal_mm": ds, "E_pre": e_pre, "E_post": e_post,
            "var_E_pre": v_pre, "var_E_post": v_post,
            "excess_rate_pre": inst_pre.rate - ring_pre.rate,
            "excess_rate_post": inst_post.rate - ring_post.rate}


def site_check(G: float, ex: dict, var_end_inst: float, sd_s: float, L_mm: float,
               C: dict, k: float = 2.0, n_bridges: int = 0, bridge_sd: float = 0.0,
               var_steps: float = 0.0) -> dict:
    """Bracket containment of the product's gap compaction values ``C`` (mm; keys 'primary',
    'A', 'B', 'raw', and any other candidate value) by the instrument-implied values
    ``C_w = G - E_w``. ``var_steps``: variance (mm^2) of the instrument's own corrections
    inside the gap (GNSS: the se^2 of every step the product removed between the two
    surveys)."""
    var_common = (var_end_inst + 2.0 * sd_s ** 2 + n_bridges * bridge_sd ** 2
                  + float(var_steps))
    out = {"G_mm": G, "var_common": var_common, "var_steps": float(var_steps)}
    lo, hi = math.inf, -math.inf
    for w in ("pre", "post"):
        cw = G - ex[f"E_{w}"]
        sw = math.sqrt(ex[f"var_E_{w}"] + var_common)
        out[f"C_{w}_mm"] = cw
        out[f"sigma_{w}_mm"] = sw
        out[f"b_{w}_ppm"] = 1e6 * (cw - C["raw"]) / L_mm
        lo, hi = min(lo, cw - k * sw), max(hi, cw + k * sw)
    out["lo_mm"], out["hi_mm"] = lo, hi
    out["C_mid_mm"] = 0.5 * (out["C_pre_mm"] + out["C_post_mm"])
    out["b_mid_ppm"] = 1e6 * (out["C_mid_mm"] - C["raw"]) / L_mm
    sd_line = math.sqrt(((out["C_pre_mm"] - out["C_post_mm"]) / 2.0) ** 2
                        + (out["sigma_pre_mm"] ** 2 + out["sigma_post_mm"] ** 2) / 2.0)
    out["b_sd_ppm"] = 1e6 * sd_line / L_mm
    for key, v in C.items():
        if v is None or not np.isfinite(v):
            out[f"contains_{key}"] = None
        else:
            out[f"contains_{key}"] = bool(lo <= v <= hi)
    ca, cb = out.get("contains_A"), out.get("contains_B")
    out["membership_evidence"] = (None if ca is None or cb is None else
                                  "both" if ca and cb else "A" if ca else "B" if cb else
                                  "neither")
    return out


def family_verdict(rows: list[dict], min_frac: float = 0.8, min_sites_gate: int = 3) -> dict:
    """Containment count of the primary value, RMSE of primary and raw against the bracket
    mid-point, and the declared pass rule."""
    ev = [r for r in rows if r.get("evaluable")]
    n = len(ev)
    n_pass = sum(bool(r["contains_primary"]) for r in ev)
    need = math.ceil(min_frac * n) if n else 0
    def rm(key: str) -> float:
        if not n:
            return float("nan")
        return float(np.sqrt(np.mean([(r[key] - r["C_mid_mm"]) ** 2 for r in ev])))

    rmse_b, rmse_r = rm("C_primary_mm"), rm("C_raw_mm")
    n_raw = sum(bool(r["contains_raw"]) for r in ev)
    ok = n >= 1 and n_pass >= need and rmse_b < rmse_r
    return {"n_evaluable": n, "n_contain_primary": n_pass, "n_needed": need,
            "n_contain_raw": n_raw, "rmse_primary_mm": rmse_b, "rmse_raw_mm": rmse_r,
            "pass": bool(ok), "in_gate": n >= min_sites_gate,
            "sites_failing": [r["site"] for r in ev if not r["contains_primary"]]}


def r_crit(n: int, alpha: float = 0.05) -> float:
    """One-sided critical Pearson r for H0 rho = 0 at level alpha with n pairs."""
    from scipy import stats
    if n < 3:
        return float("nan")
    t = stats.t.ppf(1 - alpha, n - 2)
    return float(t / math.sqrt(n - 2 + t ** 2))


def corr_check(x, y_bridged, y_raw, alpha: float = 0.05) -> dict:
    x, yb, yr = (np.asarray(v, float) for v in (x, y_bridged, y_raw))
    ok = np.isfinite(x) & np.isfinite(yb) & np.isfinite(yr)
    n = int(ok.sum())
    if n < 3:
        return {"n": n, "pass": False, "reason": "fewer than 3 sites"}
    rb = float(np.corrcoef(x[ok], yb[ok])[0, 1])
    rr = float(np.corrcoef(x[ok], yr[ok])[0, 1])
    rc = r_crit(n, alpha)
    return {"n": n, "r_bridged": rb, "r_raw": rr, "r_crit": rc,
            "pass": bool(rb >= rc and rb > rr)}


def endpoint_mean(dates: pd.DatetimeIndex, values: np.ndarray, centre, half_d: int,
                  min_n: int, seg: np.ndarray | None = None) -> dict:
    """Mean of the finite values within +-half_d days of ``centre``; with ``seg``, only the
    values in the segment of the finite sample nearest to the centre."""
    dates = pd.DatetimeIndex(dates)
    v = np.asarray(values, float)
    c = pd.Timestamp(centre)
    dd = np.abs((dates - c).days.to_numpy())
    sel = (dd <= half_d) & np.isfinite(v)
    if not sel.any():
        return {"mean": float("nan"), "n": 0, "seg": None}
    s0 = None
    if seg is not None:
        idx = np.flatnonzero(sel)
        s0 = int(np.asarray(seg)[idx[np.argmin(dd[idx])]])
        sel &= np.asarray(seg) == s0
    n = int(sel.sum())
    if n < min_n:
        return {"mean": float("nan"), "n": n, "seg": s0}
    return {"mean": float(v[sel].mean()), "n": n, "seg": s0}


def window_select(dates: pd.DatetimeIndex, lo, hi) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return np.asarray((d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi)))


def span_days(dates: pd.DatetimeIndex) -> int:
    d = pd.DatetimeIndex(dates)
    return int((d.max() - d.min()).days) if len(d) else 0


def instrument_vs_ring(*, ring_t: pd.DatetimeIndex, ring_c: np.ndarray,
                       inst_t: pd.DatetimeIndex, inst_d: np.ndarray, inst_seg=None,
                       t_lp, t_fp, pre_win, post_win, min_inst: int, min_ring: int,
                       min_span: int, end_half_d: int, end_min: int,
                       end_half_fallback: int | None = None, seasonal: bool = True,
                       segment_intercepts: bool = True) -> dict:
    """Window fits, endpoints and the excess for one instrument series against one ring
    series. Returns ``{"ok": bool, "reason": str, ...}``; ``G`` and the excess on success.
    Windows are clipped to the instrument's data and to the ring's gap; the ring is fitted on
    exactly the clipped instrument window."""
    ring_t, inst_t = pd.DatetimeIndex(ring_t), pd.DatetimeIndex(inst_t)
    ring_c, inst_d = np.asarray(ring_c, float), np.asarray(inst_d, float)
    seg = None if inst_seg is None else np.asarray(inst_seg)
    fin = np.isfinite(inst_d)
    if not fin.any():
        return {"ok": False, "reason": "no instrument data"}
    i_first, i_last = inst_t[fin].min(), inst_t[fin].max()
    pre_lo = max(pd.Timestamp(pre_win[0]), i_first)
    pre_hi = min(pd.Timestamp(pre_win[1]), pd.Timestamp(t_lp))
    post_lo = max(pd.Timestamp(post_win[0]), pd.Timestamp(t_fp))
    post_hi = min(pd.Timestamp(post_win[1]), i_last)
    t_mid = years([pd.Timestamp(t_lp) + (pd.Timestamp(t_fp) - pd.Timestamp(t_lp)) / 2])[0]
    out = {"pre_window": [str(pre_lo.date()), str(pre_hi.date())],
           "post_window": [str(post_lo.date()), str(post_hi.date())]}
    fits = {}
    for w, lo, hi in (("pre", pre_lo, pre_hi), ("post", post_lo, post_hi)):
        mi = window_select(inst_t, lo, hi) & fin
        mr = window_select(ring_t, lo, hi) & np.isfinite(ring_c)
        out[f"n_inst_{w}"], out[f"n_ring_{w}"] = int(mi.sum()), int(mr.sum())
        out[f"span_inst_{w}_d"] = span_days(inst_t[mi])
        if mi.sum() < min_inst or out[f"span_inst_{w}_d"] < min_span:
            return {**out, "ok": False, "reason": f"instrument {w} window: {int(mi.sum())} "
                    f"samples over {out[f'span_inst_{w}_d']} d"}
        if mr.sum() < min_ring:
            return {**out, "ok": False, "reason": f"ring {w} window: {int(mr.sum())} surveys"}
        fits[w] = (mi, mr)
    mi, mr = fits["pre"]
    ip = fit_pre(years(inst_t[mi]), inst_d[mi], t_mid, seasonal)
    rp = fit_pre(years(ring_t[mr]), ring_c[mr], t_mid, seasonal)
    mi, mr = fits["post"]
    s_post = seg[mi] if (seg is not None and segment_intercepts) else None
    iq = fit_post(years(inst_t[mi]), inst_d[mi], t_mid, ip, s_post) if ip else None
    rq = fit_post(years(ring_t[mr]), ring_c[mr], t_mid, rp) if rp else None
    if not (ip and rp and iq and rq):
        return {**out, "ok": False, "reason": "a window fit failed"}
    e0 = endpoint_mean(inst_t, inst_d, t_lp, end_half_d, end_min, seg)
    e1 = endpoint_mean(inst_t, inst_d, t_fp, end_half_d, end_min, seg)
    if end_half_fallback is not None:
        if not np.isfinite(e0["mean"]):
            e0 = endpoint_mean(inst_t, inst_d, t_lp, end_half_fallback, end_min, seg)
        if not np.isfinite(e1["mean"]):
            e1 = endpoint_mean(inst_t, inst_d, t_fp, end_half_fallback, end_min, seg)
    if not (np.isfinite(e0["mean"]) and np.isfinite(e1["mean"])):
        return {**out, "ok": False, "reason": f"endpoint values: {e0['n']} / {e1['n']}"}
    y_lp, y_fp = years([t_lp])[0], years([t_fp])[0]
    ex = excess(ip, rp, iq, rq, y_lp, y_fp)
    n_br = (int(e1["seg"]) - int(e0["seg"])) if seg is not None else 0
    out.update({"ok": True, "reason": "", "G_mm": e1["mean"] - e0["mean"], "n_end_lp": e0["n"],
                "n_end_fp": e1["n"], "n_bridges": n_br, "excess": ex,
                "rate_inst_pre": ip.rate, "rate_ring_pre": rp.rate, "rate_inst_post": iq.rate,
                "rate_ring_post": rq.rate, "rho_inst_pre": ip.rho, "rho_inst_post": iq.rho,
                "n_seg_post": iq.n_seg})
    return out
