"""WP S3 core (v3 code, repair round 3 S3r, 2026-10-18): per-station harmonic + step model,
step detection, local step estimates, outlier screens and the common-mode estimators.

v3 changes against v2 (``dataprep_staging/s3_gnss/v2/model.py``, unchanged for the v2
products), each answering a defect diagnosed in ``results/dataprep/reports/S3.repair.md``
Part 1 before any re-test:

* R1' (D1; revised at design time, see the repair report Part 2): the local step size is a
  robust LEVEL DIFFERENCE (``size_order`` 0: level + step, no free local slope) on +-
  ``local_half`` days of the series minus the station's global model (linear trend and
  seasonal terms) and minus the detection common mode, other steps in the window fitted
  jointly. The first version of R1 (a quadratic local trend on a longer window) rested on a
  wrong premise: for a step at the centre of a symmetric window an even trend term is
  orthogonal to the odd step contrast and leaves the estimate unchanged, and the residuals'
  correlated noise makes longer windows worse. Design-time evidence at null positions: the
  free slope doubles the variance, and a +-10-15 day level difference has the smallest error.
  The scan, the change-point refinement and the excursion test keep a local trend
  (``trend_order``), so a ramp of deformation is still not read as a step.
* R2 (D2, D1): empirical-null calibration. The scan statistic u = D / sqrt(v) is scaled by its
  own robust spread, re-estimated at every iteration of the greedy loop (after each accepted
  step), so the noise scale includes the residuals' longer-memory noise and is never inflated
  by unmodelled real steps. The step se and the own-series z use the same calibration.
  R2b (design time, design run 2): the spread is TIME-LOCAL (rolling robust median and MAD of
  u over +-``NULL_HALF`` = 365 days, fixed a priori), because GNSS noise is not stationary: at
  ERLN the robust spread of u is 6.5-10 mm in 2007-11 and 15-36 mm in 2014-16, so a
  station-wide scale over-detects in the noisy years (spurious candidates filled the 12-step
  cap) and under-detects in the quiet ones (design seed 515: 0 of 12 ERLN steps found).
* R3 (D3): the sharpness test uses the nearest ``sharp_days`` observations on each side
  within ``sharp_days + gap_days`` calendar days, so a change point next to a short gap is no
  longer rejected as a ramp for want of data in one calendar week.
* R4 (D4): candidates are ranked only among positions with |D| >= ``delta_min``.
* R5 (D5): the excursion (box) test fits the already-accepted steps inside its window
  jointly (no artificial return at a modelled step), carries the quadratic trend in both the
  step and the box model, and divides the log-likelihood difference by the station's
  measured variance inflation of the scan statistic relative to independent noise.

* R6 (design time): the WRA detection common mode is the median of the 5 nearest member
  stations (``pipeline``), not of every member.

Everything else is v2 (re-test of detected steps after their final size, catalogue and
metadata handling, gaps, transient pairing, outlier screens, product common modes).

Model (plan WP S3): ``h(t) = a + b t + annual + semiannual + sum_k c_k H(t - t_k) + e``.
Every hyperparameter in :class:`Params` is fixed on pre-origin data only (``tune.py``).
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
    z_min: float = 6.0           # scan z threshold (empirical-null units in v3)
    local_half: int = 60         # days each side for the local step estimate
    gross_k: float = 8.0         # gross outlier screen: k * MAD about a 31-day running median
    gross_floor: float = 0.10    # m, and at least this much
    final_k: float = 5.0         # final outlier screen on the model residual (minus 31-day median)
    final_floor: float = 0.015   # m
    clip_k: float = 4.0          # robust-fit clip (k * MAD)
    max_steps: int = 12          # detected steps per station (catalogue/equipment extra)
    eq_delta_min: float = 0.005  # m, earthquake-candidate offsets kept when |c| >= this ...
    eq_z_min: float = 4.0        # ... and |c| / se >= this
    sharp_days: int = 7          # sharpness test: mean of 7 obs after minus 7 obs before ...
    sharp_min: float = 0.5       # ... must be >= this fraction of the scan D (a ramp is not a step)
    snap_days: int = 3           # a detected step this close to a metadata date takes that date
    exclusion_days: int = 5      # no second detected step this close to another one
    gap_days: int = 10           # a change point with no data this close is inside a gap
    tie_max_gap: int = 30        # offsets across longer gaps leave the segment untied (flag 9)
    transient_days: int = 120    # opposite steps this close with near-cancelling sizes ...
    transient_ratio: float = 0.3  # ... |c1 + c2| < ratio * max(|c1|, |c2|) -> transient
    hp_window: int = 31          # WRA high-frequency common mode: series minus N-day mean
    trend_order: int = 2         # v3: local trend order of the scan, refinement and box models
    size_order: int = 0          # v3 R1': local trend order of the step-size model (0 = level)

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
    """(sd, lag-1 autocorrelation) of the residual minus its running median (2 win + 1).
    v3: used only for the daily and monthly ``sd`` columns of the products, not for
    detection or step se (those use the empirical null, R2)."""
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
def _window_sums(x: np.ndarray, win: int, kernels: list[np.ndarray]) -> list[np.ndarray]:
    """For every index i: sum over d in [-win, win) of x[i + d] * g[d] for each kernel g
    (length 2 win, g[m] = value at d = m - win); x outside [0, N) counts as 0. Exact local
    coordinates (no cumulative-sum cancellation)."""
    n = len(x)
    xp = np.concatenate([np.zeros(win), x, np.zeros(win)])
    return [np.correlate(xp, g, mode="valid")[:n] for g in kernels]


def scan(r: np.ndarray, win: int, min_obs: int, order: int = 2
         ) -> tuple[np.ndarray, np.ndarray]:
    """(c, v) for every index i: the step c of the local model ``sum_p a_p ((t - i) / win)^p
    + c H(t >= i)`` (p = 0..order, one common trend, so a steeper or curved local trend is
    not read as a step) fitted on [i - win, i + win), and v = [(X'X)^-1]_cc (multiply by the
    noise variance for var c). NaN where a side has fewer than ``min_obs`` observations."""
    n = len(r)
    fin = np.isfinite(r)
    w = fin.astype(float)
    yv = np.where(fin, r, 0.0)
    d = (np.arange(2 * win) - win) / float(win)
    right = (d >= 0).astype(float)
    kp = order + 1
    full_k = [d ** q for q in range(2 * order + 1)]
    right_k = [right * d ** q for q in range(kp)]
    S = _window_sums(w, win, full_k)                 # sum w d^q, q = 0..2 order
    R = _window_sums(w, win, right_k)                # sum_right w d^q, q = 0..order
    Y = _window_sums(w * yv, win, full_k[:kp])       # sum w y d^q
    YR = _window_sums(w * yv, win, [right])[0]       # sum_right w y
    nR = R[0]
    nL = S[0] - nR
    A = np.empty((n, kp + 1, kp + 1))
    for a_ in range(kp):
        for b_ in range(kp):
            A[:, a_, b_] = S[a_ + b_]
        A[:, a_, kp] = R[a_]
        A[:, kp, a_] = R[a_]
    A[:, kp, kp] = nR
    bvec = np.stack(Y + [YR], axis=1)
    good = (nL >= min_obs) & (nR >= min_obs)
    c = np.full(n, np.nan)
    var = np.full(n, np.nan)
    if good.any():
        Ag = A[good]
        det = np.linalg.det(Ag)
        okd = np.abs(det) > 1e-12 * np.abs(Ag).max(axis=(1, 2)) ** (kp + 1)
        idx = np.flatnonzero(good)[okd]
        sol = np.linalg.solve(Ag[okd], bvec[good][okd][..., None])[..., 0]
        inv = np.linalg.inv(Ag[okd])
        c[idx] = sol[:, kp]
        var[idx] = inv[:, kp, kp]
    return c, var


def null_scale(u: np.ndarray, exclude: np.ndarray | None = None, min_n: int = 100
               ) -> tuple[float, float]:
    """(median, robust sd) of the scan statistic u = c / sqrt(v) over a station's positions
    (R2: the empirical null). Positions in ``exclude`` (near modelled steps) are left out
    when at least ``min_n`` others remain."""
    m = np.isfinite(u)
    if exclude is not None and (m & ~exclude).sum() >= min_n:
        m &= ~exclude
    if m.sum() < 20:
        return 0.0, float("nan")
    x = u[m]
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)))


#: v3 R2b: half-window (days) of the time-local empirical null (fixed a priori, not tuned)
NULL_HALF = 365


def null_scale_local(u: np.ndarray, exclude: np.ndarray | None = None, half: int = NULL_HALF,
                     min_n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Time-local empirical null (R2b): rolling median and robust sd (1.4826 x rolling median
    of |u - rolling median|) of the scan statistic over +-``half`` days, positions in
    ``exclude`` left out; where fewer than ``min_n`` positions remain, the station-wide values
    (``null_scale``) are used."""
    keep = np.isfinite(u)
    if exclude is not None and (keep & ~exclude).sum() >= min_n:
        keep &= ~exclude
    x = pd.Series(np.where(keep, u, np.nan))
    w = 2 * half + 1
    med = x.rolling(w, center=True, min_periods=min_n).median()
    sd = (x - med).abs().rolling(w, center=True, min_periods=min_n).median() * 1.4826
    gmed, gsd = null_scale(u, exclude)
    return med.fillna(gmed).to_numpy(), sd.fillna(gsd).to_numpy()


def near_steps_mask(n: int, idxs: list[int], half: int) -> np.ndarray:
    out = np.zeros(n, bool)
    for j in idxs:
        out[max(0, j - half): j + half] = True
    return out


def refine(r: np.ndarray, i0: int, win: int, min_obs: int,
           forbidden: np.ndarray | None = None, search: int = 10, order: int = 2) -> int:
    """Change point within +-``search`` days of i0 minimising the residual sum of squares of
    a local trend of ``order`` plus ``c H(t >= k)`` on [i0 - win, i0 + win); never on a
    ``forbidden`` index or an unobserved day."""
    lo, hi = max(0, i0 - win), min(len(r), i0 + win)
    seg = r[lo:hi]
    m = np.isfinite(seg)
    tt = (np.arange(lo, hi, dtype=float) - i0) / float(win)
    best, bi = np.inf, i0
    trend = [tt ** q for q in range(order + 1)]
    for kk in range(max(lo + 1, i0 - search), min(hi - 1, i0 + search + 1)):
        if not np.isfinite(r[kk]) or (forbidden is not None and forbidden[kk]):
            continue
        H = (np.arange(lo, hi) >= kk).astype(float)
        n1, n2 = int((m & (H == 0)).sum()), int((m & (H == 1)).sum())
        if n1 < max(3, min_obs // 3) or n2 < max(3, min_obs // 3):
            continue
        X = np.column_stack([q[m] for q in trend] + [H[m]])
        beta, *_ = np.linalg.lstsq(X, seg[m], rcond=None)
        rr = seg[m] - X @ beta
        ssr = float(rr @ rr)
        if ssr < best:
            best, bi = ssr, kk
    return bi


def sharpness(r: np.ndarray, i_b: int, days: int, gap_days: int = 10
              ) -> tuple[float, int, int]:
    """Mean of the first ``days`` observations at or after i_b minus the mean of the last
    ``days`` observations before i_b, each taken within ``days + gap_days`` calendar days of
    the change point (v3 R3: gap-aware), with the counts."""
    span = days + gap_days
    a = r[max(0, i_b - span):i_b]
    b = r[i_b:i_b + span]
    a = a[np.isfinite(a)][-days:]
    b = b[np.isfinite(b)][:days]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), len(a), len(b)
    return float(np.mean(b) - np.mean(a)), len(a), len(b)


def box_or_step(r: np.ndarray, ib: int, p: Params, other_idx: list[int] | None = None,
                kappa: float = 1.0) -> int | None:
    """Excursion test for an accepted step at ``ib`` (v3 R5): compare, on
    [ib - 60, ib + 180), the step model ``trend + c H(t >= ib)`` with a box model ``trend + c
    [H(t >= ib) - H(t >= ie)] + c' H(t >= ie)`` for every return day ie in (ib + 3, ib +
    transient_days]; the trend has order ``p.trend_order`` and every already-accepted step in
    the window (``other_idx``) is fitted jointly in both models. The log-likelihood
    difference is divided by ``kappa`` (>= 1), the station's variance inflation of the scan
    statistic relative to independent noise. Returns ie when the box wins by BIC (one extra
    parameter, margin 10) and its level returns to within ``transient_ratio`` of the
    pre-step level, else None (a permanent step)."""
    lo, hi = max(0, ib - 60), min(len(r), ib + 180)
    seg = r[lo:hi]
    m = np.isfinite(seg)
    if m.sum() < 60:
        return None
    k = np.arange(lo, hi)
    tt = (k - ib) / 120.0
    H1 = (k >= ib).astype(float)
    trend = [tt ** q for q in range(p.trend_order + 1)]
    others = [(k >= j).astype(float) for j in (other_idx or []) if lo < j < hi and j != ib]

    def ssr(cols):
        X = np.column_stack(cols)[m]
        beta, *_ = np.linalg.lstsq(X, seg[m], rcond=None)
        rr = seg[m] - X @ beta
        return float(rr @ rr), beta
    s_step, _ = ssr(trend + [H1] + others)
    nt = len(trend)
    best, best_ie, best_beta = np.inf, None, None
    for ie in range(ib + 4, min(hi - 15, ib + p.transient_days + 1)):
        if not np.isfinite(r[ie]):
            continue
        box = H1 - (k >= ie).astype(float)
        if (m & (box == 1)).sum() < 3:
            continue
        s_box, bb = ssr(trend + [box, (k >= ie).astype(float)] + others)
        if s_box < best:
            best, best_ie, best_beta = s_box, ie, bb
    if best_ie is None or best <= 0 or s_step <= 0:
        return None
    n = int(m.sum())
    lr = n * math.log(s_step / best) / max(kappa, 1.0)
    c_in, c_after = best_beta[nt], best_beta[nt + 1]   # level inside the box, after return
    if not (lr - 2 * math.log(n) > 10 and abs(c_after) < p.transient_ratio * abs(c_in)):
        return None
    # the return must be as abrupt as an instrument step (a seasonal rebound is gradual)
    d_ret, _, _ = sharpness(r, best_ie, p.sharp_days, p.gap_days)
    if not (np.isfinite(d_ret) and d_ret * c_in < 0 and abs(d_ret) >= p.sharp_min * abs(c_in)):
        return None
    return int(best_ie)


def local_step(y_adj: np.ndarray, dates: pd.DatetimeIndex, i_b: int, half: int,
               other_idx: list[int], sigma_u: float, order: int = 2
               ) -> tuple[float, float, int]:
    """(size, se, n) of a step at index i_b from a robust local fit of a trend of ``order``
    (0 = a level) plus the step on [i_b - half, i_b + half); other steps inside the window get
    their own Heaviside term. ``y_adj`` must already be detrended by the station's global model
    when ``order`` is 0 (``detrended``). se = sigma_u sqrt([(X'X)^-1]_cc) with ``sigma_u`` the
    station's empirical-null scale of the same estimator (v3 R2; ``local_null_sigma``). Needs
    >= 3 observations on each side and >= max(6, half) in the window."""
    lo, hi = max(0, i_b - half), min(len(y_adj), i_b + half)
    yy = y_adj[lo:hi]
    m = np.isfinite(yy)
    n_left = int(np.isfinite(y_adj[lo:i_b]).sum())
    n_right = int(np.isfinite(y_adj[i_b:hi]).sum())
    if m.sum() < max(6, half) or n_left < 3 or n_right < 3:
        return float("nan"), float("nan"), int(m.sum())
    tt = (np.arange(lo, hi) - i_b) / float(half)
    cols = [tt ** q for q in range(order + 1)] + [(np.arange(lo, hi) >= i_b).astype(float)]
    for j in other_idx:
        if lo < j < hi and j != i_b:
            cols.append((np.arange(lo, hi) >= j).astype(float))
    X = np.column_stack(cols)[m]
    yv = yy[m]
    ic = order + 1
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
    try:
        vcc = float(np.linalg.inv(X[keep].T @ X[keep])[ic, ic])
        se = float(sigma_u * math.sqrt(max(vcc, 0.0)))
    except np.linalg.LinAlgError:
        se = float("nan")
    return float(beta[ic]), se, int(keep.sum())


def local_null_sigma(resid: np.ndarray, half: int, min_obs: int, order: int,
                     step_idx: list[int]) -> np.ndarray:
    """Time-local empirical-null scale of the local size estimator on a station (R2/R2b):
    rolling robust sd of u = c / sqrt(v) of the OLS local step at every position of the
    step-free residual (``resid``: the global model's residual, every step modelled), leaving
    out positions within ``half`` days of a modelled step. One value per day."""
    c, v = scan(resid, half, min(min_obs, max(3, half // 2)), order)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = c / np.sqrt(v)
    _, s = null_scale_local(u, near_steps_mask(len(resid), step_idx, half))
    return s


def detrended(yv: np.ndarray, fit: Fit, cmv: np.ndarray | float, dates: pd.DatetimeIndex
              ) -> np.ndarray:
    """``yv`` minus the global model's linear trend and seasonal terms (not its steps) and minus
    the common mode ``cmv``: the input of the step-size model (v3 R1')."""
    t = dec_year(dates)
    return yv - fit.seasonal - fit.beta[1] * (t - fit.t_ref) - cmv


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
    sigma_u_local: float = float("nan")   # v3: median empirical-null scale of the size model
    cmv: np.ndarray | None = None         # v3: detection common mode used (aligned, m)

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


def _scan_z(rc: np.ndarray, steps: list[Step], dates: pd.DatetimeIndex, p: Params
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """(D, z, kappa) of the scan on ``rc`` with the empirical-null z (R2): z = |u - med(u)| /
    s_u, u = D / sqrt(v), med and s_u over the station's positions away from modelled steps;
    kappa = (s_u / s_w)^2 with s_w the independent-noise scale of rc (robust sd of day-to-day
    differences / sqrt 2): the variance inflation of the scan statistic used by the box test."""
    D, vc = scan(rc, p.win, p.min_obs, p.trend_order)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = D / np.sqrt(vc)
    excl = near_steps_mask(len(rc), [_idx(dates, s.date) for s in steps], p.win)
    _, su = null_scale(u, excl)
    if not np.isfinite(su) or su <= 0:
        return D, np.full(len(rc), np.nan), 1.0
    med_l, su_l = null_scale_local(u, excl)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(u - med_l) / np.where(su_l > 0, su_l, np.nan)
    sw = mad_sd(rc[1:] - rc[:-1]) / math.sqrt(2.0)
    kappa = float((su / sw) ** 2) if np.isfinite(sw) and sw > 0 else 1.0
    return D, z, max(kappa, 1.0)


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

    rejected = np.zeros(n, bool)
    transient_pre = np.zeros(n, bool)
    n_ramp = 0
    if detect:
        for _ in range(p.max_steps + 20):
            if sum(st.evidence == "detected" for st in steps) >= p.max_steps:
                break
            rc = fit.resid - cmv
            # scan every screened sample (gross outliers and flagged excursions are already
            # NaN in yv); the robust-fit clip mask is NOT applied here (v2/r2 fix)
            rc = np.where(np.isfinite(yv), rc, np.nan)
            D, z, kappa = _scan_z(rc, steps, dates, p)
            forbidden = np.zeros(n, bool)          # no second step this close to another
            for st in steps:
                j = _idx(dates, st.date)
                forbidden[max(0, j - p.exclusion_days): j + p.exclusion_days + 1] = True
            forbidden |= rejected
            z = np.where(forbidden | ~(np.abs(D) >= p.delta_min), np.nan, z)   # R4
            if not np.isfinite(z).any():
                break
            i0 = int(np.nanargmax(z))
            if not z[i0] >= p.z_min:
                break
            ib = refine(rc, i0, p.win, p.min_obs, forbidden, order=p.trend_order)
            dsh, _, _ = sharpness(rc, ib, p.sharp_days, p.gap_days)
            if not (np.isfinite(dsh) and dsh * D[i0] > 0 and abs(dsh) >= p.sharp_min * abs(D[i0])):
                # a ramp (deformation), not an instrument step: never re-test this stretch
                rejected[max(0, i0 - p.win // 2): i0 + p.win // 2] = True
                n_ramp += 1
                continue
            ie = box_or_step(rc, ib, p, [_idx(dates, st.date) for st in steps], kappa)
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
                              detail=f"scan |D| {abs(D[i0]) * 1000:.1f} mm, 7-obs "
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
        y_adj = detrended(yv, fit, cmv, dates)
        idxs = [_idx(dates, st.date) for st in steps]
        su = local_null_sigma(np.where(np.isfinite(yv), fit.resid - cmv, np.nan),
                              p.local_half, p.min_obs, p.size_order, idxs)
        for d, det in earthquakes:
            d = pd.Timestamp(d)
            if not (dates[0] < d <= dates[-1]):
                continue
            if any(abs((st.date - d).days) <= p.snap_days for st in steps):
                continue
            j = _idx(dates, d)
            c, se, nn = local_step(y_adj, dates, j, p.local_half, idxs, float(su[j]),
                                   p.size_order)
            if np.isfinite(c) and np.isfinite(se) and se > 0 and abs(c) >= p.eq_delta_min \
                    and abs(c) / se >= p.eq_z_min:
                steps.append(Step(date=d, evidence="earthquake", detail=det,
                                  z=abs(c) / se, fixed_date=True))
                idxs.append(j)
        steps.sort(key=lambda q: q.date)
        fit = robust_fit(yv, dates, steps, ok, p.clip_k, fit.t_ref)

    # final outlier screen on the model residual, then sizes, gaps, transients
    f_ok = final_ok(fit.resid, p)
    ok = np.isfinite(yv) & f_ok
    fit = robust_fit(np.where(ok, yv, np.nan), dates, steps, ok, p.clip_k, fit.t_ref)
    su = _local_sizes(steps, np.where(ok, yv, np.nan), fit, cmv, dates, ok, p)
    # v2 re-test: a step accepted on the scan's |D| >= delta_min must still have |size| >=
    # delta_min once its size is re-estimated locally with the final steps and outlier screen
    # (only steps with no metadata or catalogue backing; smallest failing one first)
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
        su = _local_sizes(steps, np.where(ok, yv, np.nan), fit, cmv, dates, ok, p)
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
    sig_hf, rho = hf_noise(np.where(ok & ~transient, fit.resid, np.nan) - cmv, p.win)
    return StationResult(station=str(s.name), dates=dates, y=y, ok=ok & ~transient,
                         gross=gross, steps=steps, fit=fit, correction=corr, corr_se=corr_se,
                         untied=untied, transient=transient, sigma_hf=sig_hf, rho=rho,
                         notes=notes, sigma_u_local=float(np.nanmedian(su)), cmv=cmv)


def _local_sizes(steps: list[Step], yv_ok: np.ndarray, fit: Fit, cmv: np.ndarray,
                 dates: pd.DatetimeIndex, ok: np.ndarray, p: Params) -> np.ndarray:
    """Local size and calibrated se of every step (``local_step`` on the series minus the
    global trend and seasonal terms and the detection common mode); a step whose date falls
    inside a gap
    longer than ``gap_days`` moves to the first observation after the gap and is left untied
    when the gap exceeds ``tie_max_gap`` (in place; idempotent for moved steps). Returns the
    time-local empirical-null scale used for the se (one value per day)."""
    idxs = [_idx(dates, st.date) for st in steps]
    y_adj = detrended(yv_ok, fit, cmv, dates)
    resid = np.where(np.isfinite(yv_ok), fit.resid - cmv, np.nan)
    su = local_null_sigma(resid, p.local_half, p.min_obs, p.size_order, idxs)
    obs_idx = np.flatnonzero(ok)
    for st, j in zip(steps, idxs, strict=True):
        c, se, _ = local_step(y_adj, dates, j, p.local_half, idxs, float(su[j]), p.size_order)
        st.size, st.se = c, se
        before = obs_idx[obs_idx < j]
        after = obs_idx[obs_idx >= j]
        if len(before) and len(after):
            gap = int(after[0] - before[-1])
            if gap > p.gap_days:
                st.gap_lo, st.gap_days = dates[before[-1]], gap
                st.date = dates[after[0]]
                st.tied = gap <= p.tie_max_gap
    return su


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
