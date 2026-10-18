"""Core of WP S1a: ring instrument-change model, group evidence, pooling, 2022 path.

Model (plan WP S1): after the 2022 hiatus every ring depth reads
``z_post = a_site + (1 + b_group) * z_true``, so ``z' = (z - a_site) / (1 + b_group)``.

* ``b`` of a group is pooled (inverse variance, DerSimonian-Laird between-site tau) from the
  per-site deep-interval scales ``b_deep`` of the sites ASSIGNED to the group, leaving the site
  being corrected out (pilot method C: deep, low-rate intervals; series restricted to rings
  read on both sides of the gap).
* ``a_site`` is the top-ring jump (window fits extrapolated to mid-gap) minus ``b * z_top``.
* The group is assigned from lines of evidence that do not use the whole-interval change
  across the gap: the offset cluster of the top-ring jump, the site's own ``b_deep``, and the
  pile-implied scale. A line votes for a group; a site is assigned when >= 2 lines agree and
  none disagrees and it is not on the plan's forced list; otherwise it is carried as both
  members, with a membership probability from the lines' likelihood ratios.

Units: depths and compaction in metres inside the library, mm in the products; scales are
dimensionless (ppm only in reports). Ring "compaction" is positive when an interval shortens.
Times are decimal years since 2000-01-01.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from s1_rings import s1const as K

T0 = pd.Timestamp("2000-01-01")


# =============================================================================== basics
def years(t) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(t))
    return np.asarray((t - T0).days, dtype=float) / 365.25


def load_positions(site: str, compaction_dir=None) -> pd.DataFrame:
    """Ring depths (m) of one site: index survey datetime, columns ring ids sorted shallow ->
    deep by their mean depth in PRE_WIN (overall mean for rings not read there)."""
    d = pd.read_parquet((compaction_dir or K.COMPACTION_DIR) / f"{site}.parquet",
                        columns=["datetime", "depth", "value"])
    p = d.pivot_table(index="datetime", columns="depth", values="value", aggfunc="mean")
    p.index = pd.DatetimeIndex(pd.to_datetime(p.index)).astype("datetime64[ns]")
    p = p.sort_index()
    pre = p.loc[K.PRE_WIN[0]:K.PRE_WIN[1]]
    key = pre.mean().fillna(p.mean())
    p = p[key.sort_values().index]
    p.columns = [str(c) for c in p.columns]
    return p


@dataclass
class Fit:
    beta: np.ndarray
    cov: np.ndarray
    t_ref: float
    resid_sd: float
    n: int

    @property
    def rate(self) -> float:
        return float(self.beta[1])

    @property
    def se_rate(self) -> float:
        return float(math.sqrt(max(self.cov[1, 1], 0.0)))

    def trend(self, t: float) -> float:
        """Deseasonalised level at decimal year ``t``."""
        return float(self.beta[0] + self.beta[1] * (t - self.t_ref))

    def se_trend(self, t: float) -> float:
        g = np.array([1.0, t - self.t_ref])
        c = self.cov[:2, :2]
        return float(math.sqrt(max(g @ c @ g, 0.0)))


def window_fit(t, y, t_ref: float, seasonal: bool = True, min_n: int = 6) -> Fit | None:
    """OLS ``y = c + r (t - t_ref) [+ annual harmonic]`` on finite samples."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < min_n:
        return None
    cols = [np.ones_like(t), t - t_ref]
    if seasonal:
        cols += [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
    X = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float((res ** 2).sum() / dof)
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
    return Fit(beta=beta, cov=cov, t_ref=float(t_ref), resid_sd=math.sqrt(s2), n=len(y))


def pool(vals, sds, floor: float = K.SD_FLOOR) -> dict:
    """Inverse-variance mean with a floor on each sd, its se and the DerSimonian-Laird
    between-site tau."""
    v = np.asarray(vals, float)
    s = np.maximum(np.asarray(sds, float), floor)
    ok = np.isfinite(v) & np.isfinite(s)
    n = int(ok.sum())
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "tau": float("nan"), "n": 0}
    w = 1.0 / s[ok] ** 2
    m = float((w * v[ok]).sum() / w.sum())
    se = float(math.sqrt(1.0 / w.sum()))
    tau2 = 0.0
    if n > 1:
        chi2 = float((w * (v[ok] - m) ** 2).sum())
        denom = float(w.sum() - (w ** 2).sum() / w.sum())
        tau2 = max(0.0, (chi2 - (n - 1)) / denom) if denom > 0 else 0.0
    return {"mean": m, "se": se, "tau": float(math.sqrt(tau2)), "n": n}


# =============================================================================== gap stats
def both_rings(p: pd.DataFrame, pre: pd.DataFrame, post: pd.DataFrame,
               cover: float = K.COVER) -> list[str]:
    return [c for c in p.columns
            if len(pre) and len(post) and pre[c].notna().mean() >= cover
            and post[c].notna().mean() >= cover]


def _value_at(s: pd.Series, t: pd.Timestamp, side: str) -> tuple[float, pd.Timestamp | None]:
    """Last valid value at or before ``t`` (side='le') or first at or after (side='ge')."""
    s = s.dropna()
    s = s.loc[:t] if side == "le" else s.loc[t:]
    if s.empty:
        return float("nan"), None
    return (float(s.iloc[-1]), s.index[-1]) if side == "le" else (float(s.iloc[0]), s.index[0])


def gap_stats(p: pd.DataFrame, pre_win, post_win) -> dict:
    """Per-site statistics across a survey hiatus (real or synthetic).

    Rings: read in >= COVER of the surveys of both windows; ordered by their mean depth in the
    pre window. Each ring gets an OLS trend (+ annual harmonic) in each window. ``J`` is the
    deseasonalised jump of each ring from the last pre survey ``t_lp`` (pre fit) to the first
    post survey ``t_fp`` (post fit); ``J_top_mid`` the top ring's jump between the two fits
    at mid-gap (the offset estimate, independent of b up to b * z_top).

    ``b_deep`` (pilot method C): over consecutive intervals k below the median ring depth whose
    shortening rate is below RATE_THR in both windows, ``b = sum(dS_k + C_k) / sum(L_k)``
    with ``dS_k`` the interval change from the jumps and ``C_k`` the expected physical
    shortening (mean rate x gap); jackknife sd over the intervals; NaN with < MIN_DEEP.

    Total interval (top ring -> deepest ring): ``dS_obs`` from the raw surveys at ``t_lp`` and
    ``t_fp``; shortening rates and their se from OLS fits of the total separation in each
    window; ``sd_s`` the residual sd of the pre-window fit (trend + annual)."""
    pre = p.loc[pre_win[0]:pre_win[1]]
    post = p.loc[post_win[0]:post_win[1]]
    rings = both_rings(p, pre, post)
    if len(rings) < 3:
        raise ValueError(f"only {len(rings)} rings read on both sides")
    zbar = pre[rings].mean().sort_values()
    rings = list(zbar.index)
    pre = pre[rings].dropna(how="all")
    post = post[rings].dropna(how="all")
    t_lp, t_fp = pre.index.max(), post.index.min()
    gap = (t_fp - t_lp).days / 365.25
    t_mid_ts = t_lp + (t_fp - t_lp) / 2
    t_m = float(years([t_mid_ts])[0])
    tp, tq = years(pre.index), years(post.index)
    y_lp, y_fp = float(years([t_lp])[0]), float(years([t_fp])[0])
    J, rp, rq, fits = [], [], [], {}
    for c in rings:
        fa = window_fit(tp, pre[c].to_numpy(), t_m)
        fb = window_fit(tq, post[c].to_numpy(), t_m)
        if fa is None or fb is None:
            raise ValueError(f"ring {c}: too few samples in a window")
        fits[c] = (fa, fb)
        J.append(fb.trend(y_fp) - fa.trend(y_lp))
        rp.append(fa.rate)
        rq.append(fb.rate)
    z = zbar.to_numpy()
    J, rp, rq = np.array(J), np.array(rp), np.array(rq)
    Lk, dSk = np.diff(z), np.diff(J)
    rk_pre, rk_post = -np.diff(rp), -np.diff(rq)
    Ck = 0.5 * (rk_pre + rk_post) * gap
    mid = 0.5 * (z[1:] + z[:-1])
    deep = (mid > np.median(z)) & (np.abs(rk_pre) < K.RATE_THR) & (np.abs(rk_post) < K.RATE_THR)
    b_deep = sd_deep = float("nan")
    if deep.sum() >= K.MIN_DEEP:
        yv, Lv = (dSk + Ck)[deep], Lk[deep]
        b_deep = float(yv.sum() / Lv.sum())
        loo = np.array([np.delete(yv, i).sum() / np.delete(Lv, i).sum() for i in range(len(yv))])
        nd = len(yv)
        sd_deep = float(math.sqrt((nd - 1) / nd * ((loo - loo.mean()) ** 2).sum()))
    top, bot = rings[0], rings[-1]
    fa_top, fb_top = fits[top]
    J_top_mid = fb_top.trend(t_m) - fa_top.trend(t_m)
    se_J_top = math.sqrt(fa_top.se_trend(t_m) ** 2 + fb_top.se_trend(t_m) ** 2)
    S = p[bot] - p[top]
    S_lp, d_lp = _value_at(S, t_lp, "le")
    S_fp, d_fp = _value_at(S, t_fp, "ge")
    fS_pre = window_fit(years(S.loc[pre_win[0]:pre_win[1]].index),
                        S.loc[pre_win[0]:pre_win[1]].to_numpy(), t_m)
    fS_post = window_fit(years(S.loc[post_win[0]:post_win[1]].index),
                         S.loc[post_win[0]:post_win[1]].to_numpy(), t_m)
    r_pre, r_post = -fS_pre.rate, -fS_post.rate           # shortening rates (m/yr)
    return {
        "rings": rings, "z": z, "zbar": zbar, "top": top, "bot": bot,
        "z_top": float(z[0]), "L": float(z[-1] - z[0]),
        "t_lp": t_lp, "t_fp": t_fp, "t_mid": t_mid_ts, "gap": gap,
        "n_pre": int(len(pre)), "n_post": int(len(post)),
        "J": J, "J_top_mid": float(J_top_mid), "se_J_top": float(se_J_top),
        "b_deep": b_deep, "sd_deep": sd_deep, "n_deep": int(deep.sum()),
        "deep_len": float(Lk[deep].sum()),
        "S_lp": S_lp, "S_fp": S_fp, "d_lp": d_lp, "d_fp": d_fp,
        "dS_obs": float(S_fp - S_lp),
        "r_pre": float(r_pre), "r_post": float(r_post),
        "se_r_pre": fS_pre.se_rate, "se_r_post": fS_post.se_rate,
        "sd_s": float(fS_pre.resid_sd),
        "C_lo": float(min(r_pre, r_post) * gap), "C_hi": float(max(r_pre, r_post) * gap),
        "C_mid": float(0.5 * (r_pre + r_post) * gap),
    }


# =============================================================================== piles
def pile_daily(daily: pd.DataFrame, sensor: str) -> pd.Series:
    """Vendor-feed daily compression (mm) of one deep benchmark, flags 0-2, as a daily
    Series on a ns DatetimeIndex (gaps left NaN)."""
    x = daily[(daily["sensor_id"] == sensor) & (daily["feed"] == "vendor")
              & (daily["flag"].isin([0, 1, 2]))]
    s = pd.Series(x["value"].to_numpy(float),
                  index=pd.DatetimeIndex(pd.to_datetime(x["date"])).astype("datetime64[ns]"))
    s = s[~s.index.duplicated()].sort_index()
    if s.empty:
        return s
    full = pd.date_range(s.index.min(), s.index.max(), freq="D").astype("datetime64[ns]")
    return s.reindex(full)


def pile_at(q: pd.Series, dates, tol_days: int = K.PILE_TOL_DAYS,
            limit: int = 5) -> np.ndarray:
    """Pile value at each date: the daily value (gaps of <= ``limit`` days interpolated inside
    the record), else the nearest valid day within ``tol_days``; NaN otherwise."""
    if q.empty:
        return np.full(len(dates), np.nan)
    qi = q.interpolate(limit=limit, limit_area="inside")
    v = qi.dropna()
    out = []
    for d in pd.DatetimeIndex(pd.to_datetime(dates)).normalize():
        if d in qi.index and np.isfinite(qi.loc[d]):
            out.append(float(qi.loc[d]))
            continue
        if v.empty:
            out.append(np.nan)
            continue
        j = v.index.get_indexer([d], method="nearest")[0]
        out.append(float(v.iloc[j]) if abs((v.index[j] - d).days) <= tol_days else np.nan)
    return np.asarray(out, float)


def ring_comp_mm(p: pd.DataFrame, top: str, bot: str) -> pd.Series:
    """Raw interval shortening (mm, + = shorter) relative to its first valid survey."""
    sep = (p[bot] - p[top]).dropna()
    return -(sep - sep.iloc[0]) * 1000.0


def _rate(c: pd.Series, q: np.ndarray) -> tuple[float, float]:
    tt = years(c.index)
    return float(np.polyfit(tt, c.to_numpy(float), 1)[0]), float(np.polyfit(tt, q, 1)[0])


def pile_evidence(p: pd.DataFrame, gs: dict, q: pd.Series, depth: float, pre_win,
                  post_win) -> dict:
    """Pile-implied scale (pilot dbm_vs_ring.py, with an explicit sd).

    Ring interval: top ring -> deepest both-sides ring when the pile is >= 290 m deep, else
    -> the both-sides ring nearest the pile depth. ratio = mean of the ring/pile rate ratios
    of the two windows (survey-matched samples); ``b = (ratio * pile_gap - ring_gap_raw) / L``.
    sd: half the pre/post ratio difference times the pile gap, and the RMS over the pre window
    of (ring change - ratio_pre x pile change) across 14-month spans (representativeness and
    survey noise), in quadrature, over L."""
    zbar = gs["zbar"]
    rings = gs["rings"]
    top = rings[0]
    first = q.first_valid_index()
    if first is not None and pd.Timestamp(pre_win[0]) < first:
        pre_win = (str(first.date()), pre_win[1])
    bot = rings[-1] if depth >= 290.0 else str((zbar - depth).abs().idxmin())
    c = ring_comp_mm(p[[top, bot]], top, bot)
    qa = pd.Series(pile_at(q, c.index), index=c.index)
    out = {"top": top, "bot": bot, "L": float(zbar[bot] - zbar[top]), "ok": False}

    def win(w):
        cc = c.loc[w[0]:w[1]]
        qq = qa.loc[cc.index]
        ok = qq.notna()
        return cc[ok], qq[ok].to_numpy(float), (float(ok.mean()) if len(ok) else 0.0)

    c0, q0, cov0 = win(pre_win)
    c1, q1, cov1 = win(post_win)
    out.update({"n_pre": len(c0), "n_post": len(c1), "cover_pre": cov0, "cover_post": cov1,
                "pre_win": list(pre_win), "post_win": list(post_win)})
    if len(c0) < 6 or len(c1) < 6 or cov0 < K.PILE_MIN_COVER or cov1 < K.PILE_MIN_COVER:
        out["reason"] = "pile coverage"
        return out
    rc0, rq0 = _rate(c0, q0)
    rc1, rq1 = _rate(c1, q1)
    if abs(rq0) < 0.5 or abs(rq1) < 0.5:
        out["reason"] = "pile rate below 0.5 mm/yr"
        return out
    ratio_pre, ratio_post = rc0 / rq0, rc1 / rq1
    ratio = 0.5 * (ratio_pre + ratio_post)
    t_lp, t_fp = gs["t_lp"], gs["t_fp"]
    ql, qf = pile_at(q, [t_lp, t_fp])
    cl, cf = c.get(t_lp, np.nan), c.get(t_fp, np.nan)
    if not all(np.isfinite([ql, qf, cl, cf])):
        out["reason"] = "pile or ring missing at the gap endpoints"
        return out
    pile_gap, ring_gap = float(qf - ql), float(cf - cl)
    L_mm = out["L"] * 1000.0
    # representativeness: 14-month spans inside the pre window
    d = []
    idx = c0.index
    for i, ti in enumerate(idx):
        target = ti + pd.DateOffset(months=K.PILE_WIN_MONTHS)
        j = idx.get_indexer([target], method="nearest")[0]
        if j <= i or abs((idx[j] - target).days) > 20:
            continue
        d.append((c0.iloc[j] - c0.iloc[i]) - ratio_pre * (q0[j] - q0[i]))
    term_win = float(np.sqrt(np.mean(np.square(d)))) if d else float("nan")
    term_ratio = abs(ratio_pre - ratio_post) / 2.0 * abs(pile_gap)
    sd = math.sqrt(term_ratio ** 2 + (term_win if np.isfinite(term_win) else 0.0) ** 2) / L_mm
    out.update({"ok": True, "ratio_pre": ratio_pre, "ratio_post": ratio_post, "ratio": ratio,
                "pile_gap_mm": pile_gap, "ring_gap_raw_mm": ring_gap,
                "b": (ratio * pile_gap - ring_gap) / L_mm, "sd": sd,
                "term_ratio_mm": term_ratio, "term_win_mm": term_win, "n_win": len(d)})
    return out


# =============================================================================== evidence
def _lognorm(x, m, s2):
    return -0.5 * math.log(2 * math.pi * s2) - 0.5 * (x - m) ** 2 / s2


def line_lr(value: float, sd: float, ref=K.REF) -> dict:
    """Likelihood ratio A:B of a scale estimate (Gaussian, sd floored, reference group
    spreads added), capped at LR_CAP; votes when LR >= LR_VOTE (or <= 1/LR_VOTE) and the
    value lies within Z_PLAUSIBLE sd of the favoured group."""
    if not (np.isfinite(value) and np.isfinite(sd)):
        return {"lr": 1.0, "vote": None, "zA": float("nan"), "zB": float("nan")}
    s = max(sd, K.SD_FLOOR)
    (mA, tA), (mB, tB) = ref["A"], ref["B"]
    s2A, s2B = s * s + tA * tA, s * s + tB * tB
    logl = _lognorm(value, mA, s2A) - _lognorm(value, mB, s2B)
    cap = math.log(K.LR_CAP)
    lr = math.exp(min(max(logl, -cap), cap))
    zA, zB = (value - mA) / math.sqrt(s2A), (value - mB) / math.sqrt(s2B)
    vote = None
    if lr >= K.LR_VOTE and abs(zA) <= K.Z_PLAUSIBLE:
        vote = "A"
    elif lr <= 1.0 / K.LR_VOTE and abs(zB) <= K.Z_PLAUSIBLE:
        vote = "B"
    return {"lr": lr, "vote": vote, "zA": zA, "zB": zB}


def offset_line(J_top: float, offset_lr: float = K.OFFSET_LR) -> dict:
    lo, hi = K.OFFSET_CLUSTERS["A"]
    if lo <= J_top <= hi:
        return {"lr": offset_lr, "vote": "A", "cluster": "A"}
    lo, hi = K.OFFSET_CLUSTERS["B"]
    if lo <= J_top <= hi:
        return {"lr": 1.0 / offset_lr, "vote": "B", "cluster": "B"}
    return {"lr": 1.0, "vote": None, "cluster": "none"}


def membership(lines: dict, prior: float = K.PRIOR_A) -> float:
    lr = 1.0
    for v in lines.values():
        lr *= v["lr"]
    return prior * lr / (prior * lr + (1.0 - prior))


def assign(lines: dict, forced: bool) -> dict:
    va = sum(1 for v in lines.values() if v["vote"] == "A")
    vb = sum(1 for v in lines.values() if v["vote"] == "B")
    grp = None
    if va >= 2 and vb == 0:
        grp = "A"
    elif vb >= 2 and va == 0:
        grp = "B"
    return {"votes_A": va, "votes_B": vb, "evidence_group": grp,
            "assigned": None if forced else grp, "forced_both": bool(forced)}


def group_scale(group: str, site: str, assigned: dict, b_deep: dict, sd_deep: dict) -> dict:
    """Pooled scale of ``group`` from the b_deep of the sites assigned to it, ``site`` left out.
    Falls back to the reference value (sd = its spread) when no site is left."""
    sites = [s for s, g in assigned.items() if g == group and s != site
             and np.isfinite(b_deep.get(s, np.nan))]
    if not sites:
        m, t = K.REF[group]
        return {"b": m, "sigma": t, "n": 0, "se": float("nan"), "tau": float("nan"),
                "sites": [], "fallback": True}
    pr = pool([b_deep[s] for s in sites], [sd_deep[s] for s in sites])
    sig = math.sqrt(pr["se"] ** 2 + pr["tau"] ** 2)
    return {"b": pr["mean"], "sigma": sig, "n": pr["n"], "se": pr["se"], "tau": pr["tau"],
            "sites": sites, "fallback": False}


# =============================================================================== evaluation
@dataclass
class SiteResult:
    site: str
    gs: dict
    lines: dict
    p_A: float
    assign: dict
    members: dict = field(default_factory=dict)   # group -> {b, sigma, a, C, sigma_C, pool}
    primary: dict = field(default_factory=dict)   # {group|'mix', C, sigma_C, b_eff}
    pile: dict | None = None


def member_gap(gs: dict, b: float, sigma_b: float) -> dict:
    """Bridged gap compaction (m, + = shortening) of a member with scale b."""
    a = gs["J_top_mid"] - b * gs["z_top"]
    C = gs["S_lp"] - gs["S_fp"] / (1.0 + b)
    sC = math.sqrt((sigma_b * gs["L"]) ** 2 + 2.0 * gs["sd_s"] ** 2)
    return {"b": b, "sigma": sigma_b, "a": a, "C": C, "sigma_C": sC}


def mixture(members: dict, p_A: float, report: str | None) -> dict:
    """Primary estimate: the assigned member (``report``) or the probability-weighted mixture;
    sd from the mixture's second moment about the reported value."""
    pg = {"A": p_A, "B": 1.0 - p_A}
    if report is None:
        C = sum(pg[g] * members[g]["C"] for g in "AB")
        b_eff = sum(pg[g] * members[g]["b"] for g in "AB")
        tag = "mix"
    else:
        C = members[report]["C"]
        b_eff = members[report]["b"]
        tag = report
    var = sum(pg[g] * (members[g]["sigma_C"] ** 2 + (members[g]["C"] - C) ** 2) for g in "AB")
    return {"tag": tag, "C": C, "sigma_C": math.sqrt(var), "b_eff": b_eff}


def evaluate(pos: dict, pre_win: dict, post_win: dict, piles: dict | None,
             forced: set, pile_wins: dict | None = None,
             offset_lr: float = K.OFFSET_LR, use_lines=("offset", "b_deep", "pile"),
             extra_lines: dict | None = None, gs_cache: dict | None = None) -> dict:
    """Run the evidence, assignment, pooling and member corrections for a set of sites.

    ``pos``: site -> positions; ``pre_win``/``post_win``: site -> (start, end);
    ``piles``: site -> (daily pile Series, depth) or None; ``pile_wins``: site -> (pre, post)
    windows for the rate ratio (default the gap windows); ``forced``: sites carried as both
    members in any case; ``extra_lines``: site -> {name: line dict} (sensitivity only)."""
    res: dict[str, SiteResult] = {}
    for s, p in pos.items():
        gs = gs_cache[s] if gs_cache and s in gs_cache else gap_stats(p, pre_win[s], post_win[s])
        lines = {}
        if "offset" in use_lines:
            lines["offset"] = offset_line(gs["J_top_mid"], offset_lr)
        if "b_deep" in use_lines:
            lines["b_deep"] = line_lr(gs["b_deep"], gs["sd_deep"])
        pe = None
        if "pile" in use_lines and piles and s in piles and piles[s] is not None:
            q, depth = piles[s]
            pw = (pile_wins or {}).get(s, (pre_win[s], post_win[s]))
            pe = pile_evidence(p, gs, q, depth, pw[0], pw[1])
            if pe.get("ok"):
                lines["pile"] = line_lr(pe["b"], pe["sd"])
        for k, v in ((extra_lines or {}).get(s) or {}).items():
            lines[k] = v
        res[s] = SiteResult(site=s, gs=gs, lines=lines, p_A=membership(lines),
                            assign=assign(lines, s in forced), pile=pe)
    assigned = {s: r.assign["assigned"] for s, r in res.items()}
    b_deep = {s: r.gs["b_deep"] for s, r in res.items()}
    sd_deep = {s: r.gs["sd_deep"] for s, r in res.items()}
    for s, r in res.items():
        for g in "AB":
            gsc = group_scale(g, s, assigned, b_deep, sd_deep)
            m = member_gap(r.gs, gsc["b"], gsc["sigma"])
            m["pool"] = gsc
            r.members[g] = m
        r.primary = mixture(r.members, r.p_A, r.assign["assigned"])
    return res


def corrected_positions(p: pd.DataFrame, a: float, b: float, first_post) -> pd.DataFrame:
    """Positions with every survey from ``first_post`` on corrected: (z - a) / (1 + b)."""
    q = p.copy()
    after = q.index >= pd.Timestamp(first_post)
    q.loc[after] = (q.loc[after] - a) / (1.0 + b)
    return q


def residual_offset(p_corr: pd.DataFrame, rings: list, t_lp, t_fp) -> dict:
    """OLS of the corrected survey-to-survey change across the gap on depth."""
    a0, a1 = p_corr.loc[t_lp, rings], p_corr.loc[t_fp, rings]
    ok = a0.notna() & a1.notna()
    z = a0[ok].to_numpy(float)
    dz = (a1 - a0)[ok].to_numpy(float)
    X = np.stack([np.ones_like(z), z], 1)
    beta, *_ = np.linalg.lstsq(X, dz, rcond=None)
    return {"offset_m": float(beta[0]), "scale": float(beta[1]), "n_rings": int(ok.sum())}


# =============================================================================== monthly path
def band_indices(zbar: pd.Series, edges=K.BAND_EDGES_M) -> list[tuple[int, int]]:
    """Ring index pairs of the depth bands (as mechanism.band_compaction: the ring nearest
    each edge below the top ring; the deepest ring closes the last band)."""
    depth = zbar.to_numpy(float)
    idx = sorted({int(np.argmin(np.abs(depth - (depth[0] + e)))) for e in edges}
                 | {len(depth) - 1})
    return list(zip(idx[:-1], idx[1:], strict=True))


def monthly(pos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Monthly median ring positions (PeriodIndex) and the median survey date per month."""
    per = pos.index.to_period("M")
    m = pos.groupby(per).median()
    dates = pd.Series(pos.index, index=per).groupby(level=0).median()
    return m, dates


def interval_series(mpos: pd.DataFrame, top: str, bot: str, zero_month=None) -> pd.Series:
    """Monthly interval shortening (m, + = shorter) re-zeroed to ``zero_month`` (default the
    first valid month)."""
    sep = mpos[bot] - mpos[top]
    z = sep.first_valid_index() if zero_month is None else zero_month
    if z is None or not np.isfinite(sep.get(z, np.nan)):
        return sep * np.nan
    return sep.loc[z] - sep


def climatology(series: pd.Series, win=K.CLIM_WIN, exclude=None) -> dict:
    """Mean monthly anomaly about a 13-month centred running mean (pilot), from months in
    ``win`` except ``exclude`` (a pair of periods, open interval)."""
    s = series.loc[pd.Period(win[0], "M"):pd.Period(win[1], "M")].copy()
    if exclude is not None:
        lo, hi = exclude
        s = s[(s.index <= lo) | (s.index >= hi)]
    if s.dropna().empty:
        return {m: 0.0 for m in range(1, 13)}
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    s = s.reindex(full)
    if exclude is not None:
        s.loc[(s.index > exclude[0]) & (s.index < exclude[1])] = np.nan
    si = s.interpolate(limit=3, limit_area="inside")
    if exclude is not None:      # never bridge the excluded span
        si.loc[(si.index > exclude[0]) & (si.index < exclude[1])] = np.nan
    an = si - si.rolling(13, center=True, min_periods=11).mean()
    clim = an.groupby(an.index.month).mean()
    return {m: float(clim.get(m, 0.0)) if np.isfinite(clim.get(m, np.nan)) else 0.0
            for m in range(1, 13)}


def virtual_dates(months: pd.PeriodIndex, day: int) -> pd.DatetimeIndex:
    out = []
    for m in months:
        d = min(day, m.days_in_month)
        out.append(pd.Timestamp(year=m.year, month=m.month, day=d))
    return pd.DatetimeIndex(out)


def fill(v0: float, v1: float, d0, d1, dates, method: str, clim: dict | None = None,
         q: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Interior path between two known values; returns (values, tau).

    linear: chord in time; climatology: chord + seasonal anomaly minus its own chord;
    pile: chord + the pile's deviation from its chord (unit slope), pile values at the
    dates (``q`` daily, compression mm -> converted to m)."""
    d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    dates = pd.DatetimeIndex(dates)
    tau = np.asarray((dates - d0).days, float) / max((d1 - d0).days, 1)
    lin = v0 + (v1 - v0) * tau
    if method == "linear":
        return lin, tau
    if method == "climatology":
        cm = np.array([clim[m] for m in dates.month])
        c0, c1 = clim[d0.month], clim[d1.month]
        return lin + cm - (c0 + (c1 - c0) * tau), tau
    if method == "pile":
        qv = pile_at(q, dates, limit=K.PILE_FILL_LIMIT) / 1000.0
        q0, q1 = pile_at(q, [d0, d1], limit=K.PILE_FILL_LIMIT) / 1000.0
        return lin + qv - (q0 + (q1 - q0) * tau), tau
    raise ValueError(method)
