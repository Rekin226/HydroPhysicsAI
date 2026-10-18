"""WP S3 core (v2 code, repair round 2026-10-12): per-station harmonic + step model, step
detection, local step estimates, outlier screens and the common-mode estimators.

v2 change against v1 (``dataprep_staging/s3_gnss/model.py``, kept unchanged for the v1
products): detected steps are re-tested after their final local size estimate (|size| >=
``delta_min``; see ``process_station``). Everything else is v1.

Model (plan WP S3): ``h(t) = a + b t + annual + semiannual + sum_k c_k H(t - t_k) + e``.

Detection is greedy on the residual of the global model (optionally minus a common mode):
a two-sample scan with ``win`` days on each side gives the candidate day; the change point
is then refined by the single-change-point likelihood inside ``[i - win, i + win)``; the
candidate is accepted if ``|D| >= delta_min`` and ``z >= z_min``; the global model is refit
and the scan repeated. Step sizes are estimated locally (``local_half`` days each side,
linear trend + step on the series minus the global seasonal term and the common mode), which
is robust to the non-linear subsidence the global linear trend cannot carry.

Every hyperparameter in :class:`Params` was fixed on pre-origin data only (series through
2022-12; ``tune.py``); nothing here reads a date after the origin to choose a value.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    win: int = 30                # scan half-window (days)
    min_obs: int = 15            # observations needed on each side of a scan point
    delta_min: float = 0.010     # m, smallest accepted detected step
    z_min: float = 6.0           # scan z threshold (noise sd inflated for serial correlation)
    local_half: int = 60         # days each side for the local step estimate
    gross_k: float = 8.0         # gross outlier screen: k * MAD about a 31-day running median
    gross_floor: float = 0.10    # m, and at least this much
    final_k: float = 5.0         # final outlier screen on the model residual (minus 31-day median)
    final_floor: float = 0.015   # m
    clip_k: float = 4.0          # robust-fit clip (k * MAD)
    max_steps: int = 12          # detected steps per station (catalogue/equipment extra)
    eq_delta_min: float = 0.005  # m, earthquake-candidate offsets kept when |c| >= this ...
    eq_z_min: float = 4.0        # ... and |c| / se >= this
    sharp_days: int = 7          # sharpness test: mean over 7 days after minus 7 days before ...
    sharp_min: float = 0.5       # ... must be >= this fraction of the scan D (a ramp is not a step)
    snap_days: int = 3           # a detected step this close to a metadata date takes that date
    exclusion_days: int = 5      # no second detected step this close to another one
    gap_days: int = 10           # a change point with no data this close is inside a gap
    tie_max_gap: int = 30        # offsets across longer gaps leave the segment untied (flag 9)
    transient_days: int = 120    # opposite steps this close with near-cancelling sizes ...
    transient_ratio: float = 0.3  # ... |c1 + c2| < ratio * max(|c1|, |c2|) -> transient
    hp_window: int = 31          # WRA high-frequency common mode: series minus 31-day mean

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Step:
    date: pd.Timestamp
    size: float = float("nan")      # m, height jump (after - before)
    se: float = float("nan")
    evidence: str = "detected"      # equipment | earthquake | catalogue | detected
    detail: str = ""
    z: float = float("nan")
    gap_lo: pd.Timestamp | None = None   # last observation before the step when in a gap
    gap_days: int = 0
    tied: bool = True
    fixed_date: bool = False         # metadata date, not refined

    def row(self) -> dict:
        return {"date": self.date, "size_m": self.size, "se_m": self.se,
                "evidence": self.evidence, "detail": self.detail, "z": self.z,
                "gap_lo": self.gap_lo, "gap_days": self.gap_days, "tied": self.tied}


# ----------------------------------------------------------------------------- helpers
def daily(s: pd.Series) -> pd.Series:
    """Series reindexed on a complete daily grid from first to last observation."""
    s = s.dropna().sort_index()
    if s.empty:
        return s
    idx = pd.date_range(s.index[0], s.index[-1], freq="D")
    return s.reindex(idx)


def dec_year(idx: pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(idx)
    return idx.year.to_numpy() + (idx.dayofyear.to_numpy() - 0.5) / np.where(
        idx.is_leap_year, 366.0, 365.0)


def mad_sd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def seasonal_matrix(t: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * t
    return np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])


def design(t: np.ndarray, dates: pd.DatetimeIndex, steps: list[Step], t_ref: float
           ) -> np.ndarray:
    cols = [np.ones_like(t), t - t_ref, seasonal_matrix(t)]
    for st in steps:
        cols.append((dates >= st.date).astype(float))
    return np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])


@dataclass
class Fit:
    beta: np.ndarray
    resid: np.ndarray          # y - model on the daily grid (NaN where y is NaN)
    used: np.ndarray           # bool, inside the robust fit
    sigma: float               # MAD sd of the used residuals
    t_ref: float
    seasonal: np.ndarray       # seasonal component on the daily grid
    steps: list[Step] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return float(self.beta[1])


def robust_fit(y: np.ndarray, dates: pd.DatetimeIndex, steps: list[Step], ok: np.ndarray,
               clip_k: float, t_ref: float | None = None) -> Fit:
    t = dec_year(dates)
    if t_ref is None:
        t_ref = float(np.round(np.nanmean(t[ok]), 0))
    X = design(t, dates, steps, t_ref)
    use = ok & np.isfinite(y)
    beta = np.zeros(X.shape[1])
    for _ in range(4):
        if use.sum() < X.shape[1] + 10:
            break
        beta, *_ = np.linalg.lstsq(X[use], y[use], rcond=None)
        r = y - X @ beta
        s = mad_sd(r[use])
        new = ok & np.isfinite(y) & (np.abs(r - np.median(r[use])) <= clip_k * max(s, 1e-4))
        if (new == use).all():
            break
        use = new
    r = y - X @ beta
    seas = seasonal_matrix(t) @ beta[2:6]
    return Fit(beta=beta, resid=np.where(np.isfinite(y), r, np.nan), used=use,
               sigma=mad_sd(r[use]), t_ref=t_ref, seasonal=seas, steps=list(steps))


def running(x: np.ndarray, w: int, fn: str = "median", min_periods: int | None = None
            ) -> np.ndarray:
    s = pd.Series(x)
    r = s.rolling(w, center=True, min_periods=min_periods or max(3, w // 3))
    return (r.median() if fn == "median" else r.mean()).to_numpy()


def hf_noise(r: np.ndarray, win: int) -> tuple[float, float]:
    """(sd, lag-1 autocorrelation) of the residual minus its running median (2 win + 1)."""
    hp = r - running(r, 2 * win + 1)
    s = mad_sd(hp)
    a, b = hp[:-1], hp[1:]
    m = np.isfinite(a) & np.isfinite(b)
    rho = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 30 else 0.0
    return s, float(np.clip(rho, 0.0, 0.95))


def gross_ok(y: np.ndarray, p: Params) -> np.ndarray:
    """False at gross outliers (|y - 31-day running median| > max(gross_k MAD, floor))."""
    d = y - running(y, 31)
    s = mad_sd(d)
    thr = max(p.gross_k * (s if np.isfinite(s) else 0.0), p.gross_floor)
    return np.isfinite(y) & ~(np.abs(d) > thr)


def final_ok(resid: np.ndarray, p: Params) -> np.ndarray:
    """False where the model residual departs from its 31-day running median by more than
    max(final_k MAD, final_floor)."""
    d = resid - running(resid, 31)
    s = mad_sd(d)
    thr = max(p.final_k * (s if np.isfinite(s) else 0.0), p.final_floor)
    return np.isfinite(resid) & ~(np.abs(d) > thr)


# ----------------------------------------------------------------------------- scanning
def _cs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.where(np.isfinite(x), x, 0.0)
    n = np.isfinite(x).astype(float)
    return np.concatenate([[0.0], np.cumsum(v)]), np.concatenate([[0.0], np.cumsum(n)])


def scan(r: np.ndarray, win: int, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """(c, v) for every index i: the step c of a local model ``a + b (t - i) + c H(t >= i)``
    fitted on [i - win, i + win) (one common slope, so a steeper local trend is not read as a
    step), and v = [(X'X)^-1]_cc (multiply by the noise variance for var c). NaN where a side
    has fewer than ``min_obs`` observations."""
    N = len(r)
    fin = np.isfinite(r)
    v = np.where(fin, r, 0.0)
    k = np.arange(N, dtype=float)
    w = fin.astype(float)

    def cs(x):
        return np.concatenate([[0.0], np.cumsum(x)])
    C0, C1, C2 = cs(w), cs(w * k), cs(w * k * k)
    Y0, Y1 = cs(v), cs(v * k)
    i = np.arange(N)
    lo = np.clip(i - win, 0, N)
    hi = np.clip(i + win, 0, N)
    # sums over the left [lo, i) and right [i, hi) parts, with t measured from i
    def part(a, b):
        n = C0[b] - C0[a]
        s1 = (C1[b] - C1[a]) - i * n
        s2 = (C2[b] - C2[a]) - 2 * i * (C1[b] - C1[a]) + i * i * n
        y0 = Y0[b] - Y0[a]
        y1 = (Y1[b] - Y1[a]) - i * y0
        return n, s1, s2, y0, y1
    nL, s1L, s2L, y0L, y1L = part(lo, i)
    nR, s1R, s2R, y0R, y1R = part(i, hi)
    n, s1, s2, y0, y1 = nL + nR, s1L + s1R, s2L + s2R, y0L + y0R, y1L + y1R
    A = np.empty((N, 3, 3))
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = n, s1, nR
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = s1, s2, s1R
    A[:, 2, 0], A[:, 2, 1], A[:, 2, 2] = nR, s1R, nR
    bvec = np.stack([y0, y1, y0R], axis=1)
    good = (nL >= min_obs) & (nR >= min_obs)
    c = np.full(N, np.nan)
    var = np.full(N, np.nan)
    if good.any():
        Ag = A[good]
        det = np.linalg.det(Ag)
        okd = np.abs(det) > 1e-9
        idx = np.flatnonzero(good)[okd]
        sol = np.linalg.solve(Ag[okd], bvec[good][okd][..., None])[..., 0]
        inv = np.linalg.inv(Ag[okd])
        c[idx] = sol[:, 2]
        var[idx] = inv[:, 2, 2]
    return c, var


def refine(r: np.ndarray, i0: int, win: int, min_obs: int,
           forbidden: np.ndarray | None = None, search: int = 10) -> int:
    """Change point within +-``search`` days of i0 minimising the residual sum of squares of
    ``a + b t + c H(t >= k)`` on [i0 - win, i0 + win); never on a ``forbidden`` index or an
    unobserved day."""
    lo, hi = max(0, i0 - win), min(len(r), i0 + win)
    seg = r[lo:hi]
    m = np.isfinite(seg)
    tt = np.arange(lo, hi, dtype=float) - i0
    best, bi = np.inf, i0
    for kk in range(max(lo + 1, i0 - search), min(hi - 1, i0 + search + 1)):
        if not np.isfinite(r[kk]) or (forbidden is not None and forbidden[kk]):
            continue
        H = (np.arange(lo, hi) >= kk).astype(float)
        n1, n2 = int((m & (H == 0)).sum()), int((m & (H == 1)).sum())
        if n1 < max(3, min_obs // 3) or n2 < max(3, min_obs // 3):
            continue
        X = np.column_stack([np.ones(m.sum()), tt[m], H[m]])
        beta, *_ = np.linalg.lstsq(X, seg[m], rcond=None)
        rr = seg[m] - X @ beta
        ssr = float(rr @ rr)
        if ssr < best:
            best, bi = ssr, kk
    return bi


def sharpness(r: np.ndarray, i_b: int, days: int) -> tuple[float, int, int]:
    """Mean of r on [i_b, i_b + days) minus mean on [i_b - days, i_b), with the counts."""
    a = r[max(0, i_b - days):i_b]
    b = r[i_b:i_b + days]
    na, nb = int(np.isfinite(a).sum()), int(np.isfinite(b).sum())
    if na < 2 or nb < 2:
        return float("nan"), na, nb
    return float(np.nanmean(b) - np.nanmean(a)), na, nb


def box_or_step(r: np.ndarray, ib: int, p: Params) -> int | None:
    """Excursion test for an accepted step at ``ib``: compare, on [ib - 60, ib + 180), the
    step model ``a + b t + c H(t >= ib)`` with a box ``a + b t + c [H(t >= ib) - H(t >= ie)]``
    for every return day ie in (ib + 3, ib + transient_days]. Returns ie when the box wins by
    BIC (one extra parameter) and its level returns to within ``transient_ratio`` of the
    pre-step level, else None (a permanent step)."""
    lo, hi = max(0, ib - 60), min(len(r), ib + 180)
    seg = r[lo:hi]
    m = np.isfinite(seg)
    if m.sum() < 60:
        return None
    k = np.arange(lo, hi)
    tt = (k - ib) / 365.25
    H1 = (k >= ib).astype(float)

    def ssr(cols):
        X = np.column_stack(cols)[m]
        beta, *_ = np.linalg.lstsq(X, seg[m], rcond=None)
        rr = seg[m] - X @ beta
        return float(rr @ rr), beta
    s_step, b_step = ssr([np.ones_like(tt), tt, H1])
    best, best_ie, best_beta = np.inf, None, None
    for ie in range(ib + 4, min(hi - 15, ib + p.transient_days + 1)):
        if not np.isfinite(r[ie]):
            continue
        box = H1 - (k >= ie).astype(float)
        if (m & (box == 1)).sum() < 3:
            continue
        s_box, bb = ssr([np.ones_like(tt), tt, box, (k >= ie).astype(float)])
        if s_box < best:
            best, best_ie, best_beta = s_box, ie, bb
    if best_ie is None:
        return None
    n = int(m.sum())
    bic_step = n * math.log(s_step / n) + 3 * math.log(n)
    bic_box = n * math.log(best / n) + 5 * math.log(n)
    c_in, c_after = best_beta[2], best_beta[3]      # level inside the box, level after return
    if not (bic_box < bic_step - 10 and abs(c_after) < p.transient_ratio * abs(c_in)):
        return None
    # the return must be as abrupt as an instrument step (a seasonal rebound is gradual)
    d_ret, _, _ = sharpness(r, best_ie, p.sharp_days)
    if not (np.isfinite(d_ret) and d_ret * c_in < 0 and abs(d_ret) >= p.sharp_min * abs(c_in)):
        return None
    return int(best_ie)


def local_step(y_adj: np.ndarray, dates: pd.DatetimeIndex, i_b: int, half: int,
               other_idx: list[int], rho: float) -> tuple[float, float, int]:
    """(size, se, n) of a step at index i_b from a local linear trend + step fit on
    [i_b - half, i_b + half); other steps inside the window get their own Heaviside term."""
    lo, hi = max(0, i_b - half), min(len(y_adj), i_b + half)
    yy = y_adj[lo:hi]
    m = np.isfinite(yy)
    if m.sum() < 20 or (~np.isfinite(y_adj[lo:i_b])).all() or (~np.isfinite(y_adj[i_b:hi])).all():
        return float("nan"), float("nan"), int(m.sum())
    tt = (np.arange(lo, hi) - i_b) / 365.25
    cols = [np.ones_like(tt), tt, (np.arange(lo, hi) >= i_b).astype(float)]
    for j in other_idx:
        if lo < j < hi and j != i_b:
            cols.append((np.arange(lo, hi) >= j).astype(float))
    X = np.column_stack(cols)[m]
    yv = yy[m]
    beta = np.zeros(X.shape[1])
    keep = np.ones(len(yv), bool)
    for _ in range(3):
        beta, *_ = np.linalg.lstsq(X[keep], yv[keep], rcond=None)
        rr = yv - X @ beta
        s = mad_sd(rr[keep])
        new = np.abs(rr - np.median(rr[keep])) <= 4 * max(s, 1e-4)
        if (new == keep).all():
            break
        keep = new
    rr = yv[keep] - X[keep] @ beta
    dof = max(int(keep.sum()) - X.shape[1], 1)
    s2 = float(rr @ rr) / dof
    try:
        cov = s2 * np.linalg.inv(X[keep].T @ X[keep])
        se = float(math.sqrt(max(cov[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        se = float("nan")
    infl = math.sqrt((1 + rho) / (1 - rho))
    return float(beta[2]), se * infl, int(keep.sum())


# ----------------------------------------------------------------------------- detection
@dataclass
class StationResult:
    station: str
    dates: pd.DatetimeIndex
    y: np.ndarray                 # raw height on the daily grid (NaN = no observation)
    ok: np.ndarray                # passes both outlier screens
    gross: np.ndarray             # True = gross outlier
    steps: list[Step]
    fit: Fit                      # global fit with the final steps
    correction: np.ndarray        # cumulative step correction (m) to subtract
    corr_se: np.ndarray           # its sd (m)
    untied: np.ndarray            # True on samples after an untied offset
    transient: np.ndarray         # True inside a flagged transient
    sigma_hf: float
    rho: float
    notes: list[str] = field(default_factory=list)

    @property
    def cleaned(self) -> np.ndarray:
        return self.y - self.correction


def _idx(dates: pd.DatetimeIndex, d: pd.Timestamp) -> int:
    return int(dates.searchsorted(pd.Timestamp(d)))


def _apply_steps(n: int, dates: pd.DatetimeIndex, steps: list[Step]) -> tuple[np.ndarray,
                                                                           np.ndarray]:
    corr = np.zeros(n)
    var = np.zeros(n)
    for st in steps:
        if not np.isfinite(st.size):
            continue
        i = _idx(dates, st.date)
        corr[i:] += st.size
        var[i:] += (st.se if np.isfinite(st.se) else 0.0) ** 2
    return corr, np.sqrt(var)


def process_station(s: pd.Series, p: Params, cm: pd.Series | None = None,
                    equipment: list[tuple[pd.Timestamp, str]] | None = None,
                    earthquakes: list[tuple[pd.Timestamp, str]] | None = None,
                    catalogue: list[tuple[pd.Timestamp, str]] | None = None,
                    t_ref: float | None = None, detect: bool = True) -> StationResult:
    """Clean one daily height series. ``cm`` (aligned by date, m) is subtracted only inside
    detection and the local step estimates; it is never removed from the output values."""
    s = daily(s)
    dates = pd.DatetimeIndex(s.index)
    y = s.to_numpy(dtype=float)
    n = len(y)
    notes: list[str] = []
    cmv = np.zeros(n)
    if cm is not None:
        c = cm.reindex(dates).to_numpy(dtype=float)
        cmv = np.where(np.isfinite(c), c, 0.0)
    g_ok = gross_ok(y, p)
    gross = np.isfinite(y) & ~g_ok
    yv = np.where(g_ok, y, np.nan)

    steps: list[Step] = []
    for d, det in equipment or []:
        if dates[0] < pd.Timestamp(d) <= dates[-1]:
            steps.append(Step(date=pd.Timestamp(d), evidence="equipment", detail=det,
                              fixed_date=True))
    ok = np.isfinite(yv)
    fit = robust_fit(yv, dates, steps, ok, p.clip_k, t_ref)
    sig_hf, rho = hf_noise(fit.resid - cmv, p.win)
    infl = math.sqrt((1 + rho) / (1 - rho))

    rejected = np.zeros(n, bool)
    transient_pre = np.zeros(n, bool)
    n_ramp = 0
    if detect:
        for _ in range(p.max_steps + 20):
            if sum(st.evidence == "detected" for st in steps) >= p.max_steps:
                break
            rc = fit.resid - cmv
            # scan every screened sample (gross outliers and flagged excursions are already
            # NaN in yv); the robust-fit clip mask is NOT applied here: with an undetected
            # step near a record end, the clip removes the short side of the step and the
            # scan could never see it (r2 fix, found on pre-origin injections)
            rc = np.where(np.isfinite(yv), rc, np.nan)
            D, vc = scan(rc, p.win, p.min_obs)
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.abs(D) / (sig_hf * infl * np.sqrt(vc))
            forbidden = np.zeros(n, bool)          # no second step this close to another
            for st in steps:
                j = _idx(dates, st.date)
                forbidden[max(0, j - p.exclusion_days): j + p.exclusion_days + 1] = True
            forbidden |= rejected
            z[forbidden] = np.nan
            if not np.isfinite(z).any():
                break
            i0 = int(np.nanargmax(z))
            if not (abs(D[i0]) >= p.delta_min and z[i0] >= p.z_min):
                break
            ib = refine(rc, i0, p.win, p.min_obs, forbidden)
            dsh, _, _ = sharpness(rc, ib, p.sharp_days)
            if not (np.isfinite(dsh) and dsh * D[i0] > 0 and abs(dsh) >= p.sharp_min * abs(D[i0])):
                # a ramp (deformation), not an instrument step: never re-test this stretch
                rejected[max(0, i0 - p.win // 2): i0 + p.win // 2] = True
                n_ramp += 1
                continue
            ie = box_or_step(rc, ib, p)
            if ie is not None:
                # an excursion that returns: flag it (9), never correct it as an offset
                transient_pre[ib:ie] = True
                notes.append(f"excursion {dates[ib].date()}..{dates[ie - 1].date()} "
                             f"({D[i0] * 1000:+.1f} mm) flagged 9, not corrected")
                yv = np.where(transient_pre, np.nan, yv)
                ok = np.isfinite(yv)
                fit = robust_fit(yv, dates, steps, ok, p.clip_k, fit.t_ref)
                continue
            steps.append(Step(date=dates[ib], evidence="detected", z=float(z[i0]),
                              detail=f"scan |D| {abs(D[i0]) * 1000:.1f} mm, 7-day "
                                     f"{abs(dsh) * 1000:.1f} mm"))
            steps.sort(key=lambda q: q.date)
            fit = robust_fit(yv, dates, steps, ok, p.clip_k, fit.t_ref)

    # catalogue steps the detection did not reproduce are forced (date refined +-10 days
    # at the largest day-to-day jump of the residual)
    for d, det in catalogue or []:
        d = pd.Timestamp(d)
        if not (dates[0] < d <= dates[-1]):
            continue
        if any(abs((st.date - d).days) <= 15 for st in steps):
            for st in steps:
                if abs((st.date - d).days) <= 15 and st.evidence == "detected":
                    st.evidence, st.detail = "detected+catalogue", det
            continue
        j = _idx(dates, d)
        lo, hi = max(1, j - 10), min(n, j + 11)
        rr = fit.resid - cmv
        dd = np.abs(np.diff(np.where(np.isfinite(rr), rr, np.nan)))[lo - 1:hi - 1]
        jb = lo + int(np.nanargmax(dd)) if np.isfinite(dd).any() else j
        steps.append(Step(date=dates[jb], evidence="catalogue", detail=det))
        notes.append(f"catalogue step {d.date()} forced at {dates[jb].date()}")
        steps.sort(key=lambda q: q.date)
        fit = robust_fit(yv, dates, steps, ok, p.clip_k, fit.t_ref)

    # snap detected steps to nearby metadata dates
    meta = [(pd.Timestamp(d), "equipment", t) for d, t in equipment or []] + \
        [(pd.Timestamp(d), "earthquake", t) for d, t in earthquakes or []]
    for st in steps:
        if st.evidence != "detected":
            continue
        near = [(abs((d - st.date).days), d, ev, t) for d, ev, t in meta
                if abs((d - st.date).days) <= p.snap_days and dates[0] < d <= dates[-1]]
        if near:
            _, d, ev, t = min(near, key=lambda q: q[0])
            st.detail += f"; snapped from {st.date.date()} to {ev} {d.date()} ({t})"
            st.date, st.evidence = d, f"detected+{ev}"
    steps = _dedupe(steps)

    # earthquake candidates not matched by a step: keep only clear local offsets
    if earthquakes:
        seas = fit.seasonal
        y_adj = yv - seas - cmv
        idxs = [_idx(dates, st.date) for st in steps]
        for d, det in earthquakes:
            d = pd.Timestamp(d)
            if not (dates[0] < d <= dates[-1]):
                continue
            if any(abs((st.date - d).days) <= p.snap_days for st in steps):
                continue
            j = _idx(dates, d)
            c, se, nn = local_step(y_adj, dates, j, p.local_half, idxs, rho)
            if np.isfinite(c) and abs(c) >= p.eq_delta_min and abs(c) / se >= p.eq_z_min:
                steps.append(Step(date=d, evidence="earthquake", detail=det,
                                  z=abs(c) / se, fixed_date=True))
                idxs.append(j)
        steps.sort(key=lambda q: q.date)
        fit = robust_fit(yv, dates, steps, ok, p.clip_k, fit.t_ref)

    # final outlier screen on the model residual, then sizes, gaps, transients
    f_ok = final_ok(fit.resid, p)
    ok = np.isfinite(yv) & f_ok
    fit = robust_fit(np.where(ok, yv, np.nan), dates, steps, ok, p.clip_k, fit.t_ref)
    sig_hf, rho = hf_noise(np.where(ok, fit.resid, np.nan) - cmv, p.win)
    _local_sizes(steps, np.where(ok, yv, np.nan) - fit.seasonal - cmv, dates, ok, rho, p)
    # v2 re-test (repair round 2026-10-12): a step accepted on the scan's |D| >= delta_min
    # must still have |size| >= delta_min once its size is re-estimated locally with the
    # final steps and outlier screen. Only steps with no metadata or catalogue backing
    # (evidence exactly "detected") are re-tested; the smallest failing one is dropped, the
    # global model refitted and every size re-estimated, until none fails.
    while True:
        small = [st for st in steps if st.evidence == "detected" and np.isfinite(st.size)
                 and abs(st.size) < p.delta_min]
        if not small:
            break
        worst = min(small, key=lambda q: abs(q.size))
        steps.remove(worst)
        notes.append(f"detected step {worst.date.date()} dropped by the re-test: final local "
                     f"size {worst.size * 1000:+.1f} mm < delta_min {p.delta_min * 1000:.0f} mm "
                     f"(scan z {worst.z:.1f})")
        fit = robust_fit(np.where(ok, yv, np.nan), dates, steps, ok, p.clip_k, fit.t_ref)
        sig_hf, rho = hf_noise(np.where(ok, fit.resid, np.nan) - cmv, p.win)
        _local_sizes(steps, np.where(ok, yv, np.nan) - fit.seasonal - cmv, dates, ok, rho, p)
    steps = [st for st in steps if np.isfinite(st.size) or st.evidence == "equipment"]
    for st in steps:
        if not np.isfinite(st.size):
            st.size, st.se = 0.0, float("nan")
            st.detail += "; size not estimable (no data on one side)"
    if n_ramp:
        notes.append(f"{n_ramp} scan candidate(s) rejected as ramps by the sharpness test")
    transient = transient_pre.copy()
    steps, transient = _transients(steps, dates, transient, p, notes)
    corr, corr_se = _apply_steps(n, dates, steps)
    untied = np.zeros(n, bool)
    for st in steps:
        if not st.tied:
            untied[_idx(dates, st.date):] = True
            notes.append(f"offset {st.size * 100:+.1f} cm across a {st.gap_days}-day gap ending "
                         f"{st.date.date()} left untied: later samples flag 9")
    fit = robust_fit(np.where(ok & ~transient, yv, np.nan) - corr, dates, [],
                     ok & ~transient, p.clip_k, fit.t_ref)
    fit.steps = steps
    return StationResult(station=str(s.name), dates=dates, y=y, ok=ok & ~transient,
                         gross=gross, steps=steps, fit=fit, correction=corr, corr_se=corr_se,
                         untied=untied, transient=transient, sigma_hf=sig_hf, rho=rho,
                         notes=notes)


def _local_sizes(steps: list[Step], y_adj: np.ndarray, dates: pd.DatetimeIndex,
                 ok: np.ndarray, rho: float, p: Params) -> None:
    """Local size and se of every step (``local_step``); a step whose date falls inside a gap
    longer than ``gap_days`` moves to the first observation after the gap and is left untied
    when the gap exceeds ``tie_max_gap`` (in place; idempotent for moved steps)."""
    idxs = [_idx(dates, st.date) for st in steps]
    obs_idx = np.flatnonzero(ok)
    for st, j in zip(steps, idxs, strict=True):
        c, se, _ = local_step(y_adj, dates, j, p.local_half, idxs, rho)
        st.size, st.se = c, se
        before = obs_idx[obs_idx < j]
        after = obs_idx[obs_idx >= j]
        if len(before) and len(after):
            gap = int(after[0] - before[-1])
            if gap > p.gap_days:
                st.gap_lo, st.gap_days = dates[before[-1]], gap
                st.date = dates[after[0]]
                st.tied = gap <= p.tie_max_gap


def _dedupe(steps: list[Step]) -> list[Step]:
    out: list[Step] = []
    for st in sorted(steps, key=lambda q: q.date):
        if out and (st.date - out[-1].date).days <= 1:
            if out[-1].evidence == "detected" and st.evidence != "detected":
                out[-1] = st
            continue
        out.append(st)
    return out


def _transients(steps: list[Step], dates: pd.DatetimeIndex, transient: np.ndarray,
                p: Params, notes: list[str]) -> tuple[list[Step], np.ndarray]:
    """Opposite detected steps closer than transient_days whose sizes nearly cancel are an
    excursion, not two offsets: the samples between them are flagged (9), not corrected."""
    keep = list(steps)
    changed = True
    while changed:
        changed = False
        for a in range(len(keep) - 1):
            s1, s2 = keep[a], keep[a + 1]
            if not (s1.evidence.startswith("detected") and s2.evidence.startswith("detected")):
                continue
            if (s2.date - s1.date).days > p.transient_days:
                continue
            if not (np.isfinite(s1.size) and np.isfinite(s2.size)) or s1.size * s2.size >= 0:
                continue
            if abs(s1.size + s2.size) < p.transient_ratio * max(abs(s1.size), abs(s2.size)):
                i1, i2 = _idx(dates, s1.date), _idx(dates, s2.date)
                transient[i1:i2] = True
                notes.append(f"transient {s1.date.date()}..{s2.date.date()} "
                             f"({s1.size * 100:+.1f} / {s2.size * 100:+.1f} cm): samples "
                             f"between flagged 9, no correction")
                del keep[a:a + 2]
                changed = True
                break
    return keep, transient


# ----------------------------------------------------------------------------- common mode
def median_cm(resid: pd.DataFrame, min_n: int = 2) -> pd.DataFrame:
    """Daily median of station residuals with its n and sd (1.2533 * spread / sqrt(n))."""
    n = resid.notna().sum(axis=1)
    med = resid.median(axis=1)
    spread = resid.sub(med, axis=0).abs().median(axis=1) * 1.4826
    sd = 1.2533 * spread / np.sqrt(n.clip(lower=1))
    med[n < min_n] = np.nan
    sd[n < min_n] = np.nan
    return pd.DataFrame({"cm": med, "n": n, "sd": sd})


def highpass_cm(resid: pd.DataFrame, window: int, min_n: int = 5) -> pd.DataFrame:
    """Network median residual minus its centred running mean (``window`` days): the
    high-frequency common mode (processing noise) without the slow fan-wide signal."""
    base = median_cm(resid, min_n=min_n)
    lp = base["cm"].rolling(window, center=True, min_periods=max(3, window // 3)).mean()
    out = base.copy()
    out["cm"] = base["cm"] - lp
    return out
