"""H1a v2 (repair round): local level-consistency check of a detected jump.

The v1 step rule took the jump size ``j`` from the audit classifier (one sample-to-sample
difference, persistence judged on the 6-48 h median) and never checked that the levels on
the two sides of the jump differ by about ``j``. Returns from short dropouts or spikes whose
other edge was missed therefore became multi-year "steps" (07230332: -14.42 m with segment
medians -3.40 / -3.84 m; 09190132: +18.9 m with 10-day medians -21.74 / -21.89 m), and some
paired episodes were corrected although the level between their jumps is the normal level.

Check (fixed in the repair round before any re-scoring; threshold from first principles, no
2023+ observation used): around a jump at time ``t0``, take the valid samples of the
barometrically corrected series within ``LEVEL_WINDOW_D`` days on each side, truncated at
the neighbouring offset-class jumps of the same well, reduce them to hourly medians and fit
by least absolute deviations (median regression)

    v(tau) = a + b * tau + delta * 1[tau >= 0],     tau = (t - t0) in days.

``delta`` is the robust local level shift with a common linear trend removed. The jump is
level-consistent when ``sign(delta) == sign(j)`` and ``|delta| >= LEVEL_FRAC * |j|``; it is
"undetermined" when either side has fewer than ``LEVEL_MIN_HOURS`` hourly medians.

"Valid samples" exclude sentinels and samples that the stage-1 base screens mark as invalid
readings (spike, flat run, floor dropout run, below the well bottom). Samples flagged only by
the 15 x MAD or the 12 m rolling-median screens are kept, because the displaced side of a
large real step carries those flags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from . import config as C
from . import screens as S

LEVEL_WINDOW_D = 10.0          # days on each side of the jump (the verifier's 10-day medians)
LEVEL_MIN_HOURS = 12           # hourly medians needed on each side
LEVEL_FRAC = 0.5               # |delta| >= 0.5 |j| and same sign (first principles)
EXCLUDE_BITS = np.uint16(S.K_SENTINEL | S.K_SPIKE | S.K_FLAT | S.K_DROPFLAT | S.K_BOTTOM)
DAY = 86400.0


def lad_step(tau: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-absolute-deviation fit of y = a + b tau + delta [tau >= 0]; returns
    (delta, b, mean absolute residual)."""
    n = len(y)
    X = np.column_stack([np.ones(n), tau, (tau >= 0).astype(float)])
    p = X.shape[1]
    # variables: beta (free, p), u+ (n), u- (n); minimise sum(u+ + u-)
    c = np.r_[np.zeros(p), np.ones(2 * n)]
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    r = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if not r.success:
        return np.nan, np.nan, np.nan
    beta = r.x[:p]
    return float(beta[2]), float(beta[1]), float(r.fun / n)


def hourly_medians(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(t) == 0:
        return np.zeros(0), np.zeros(0)
    h = (np.asarray(t) // 3600).astype(np.int64)
    g = pd.Series(np.asarray(v, float)).groupby(h, sort=True).median()
    return g.index.to_numpy(np.int64) * 3600.0, g.to_numpy(float)


def level_shift(t: np.ndarray, v: np.ndarray, valid: np.ndarray, t0: int, j: float,
                lo_bound: int | None = None, hi_bound: int | None = None) -> dict:
    """Level-consistency check of a jump of size ``j`` whose first post-jump sample is at
    ``t0``. ``lo_bound``/``hi_bound``: times of the neighbouring offset jumps (the windows
    stop before them)."""
    w = int(LEVEL_WINDOW_D * DAY)
    a = t0 - w if lo_bound is None else max(t0 - w, int(lo_bound))
    b = t0 + w if hi_bound is None else min(t0 + w, int(hi_bound) - 1)
    i0, i1 = np.searchsorted(t, a, "left"), np.searchsorted(t, b, "right")
    tt, vv, ok = t[i0:i1], v[i0:i1], valid[i0:i1]
    tt, vv = tt[ok], vv[ok]
    pre, post = tt < t0, tt >= t0
    hp_t, hp_v = hourly_medians(tt[pre], vv[pre])
    ha_t, ha_v = hourly_medians(tt[post], vv[post])
    out = {"t0": int(t0), "j": float(j), "n_hours_before": int(len(hp_t)),
           "n_hours_after": int(len(ha_t)), "window_lo": int(a), "window_hi": int(b)}
    if len(hp_t) < LEVEL_MIN_HOURS or len(ha_t) < LEVEL_MIN_HOURS:
        return out | {"delta": np.nan, "slope_m_per_d": np.nan, "mad_resid": np.nan,
                      "ratio": np.nan, "status": "undetermined"}
    tau = (np.r_[hp_t, ha_t] - (t0 // 3600) * 3600.0) / DAY
    y = np.r_[hp_v, ha_v]
    delta, slope, mres = lad_step(tau, y)
    if not np.isfinite(delta):
        return out | {"delta": np.nan, "slope_m_per_d": np.nan, "mad_resid": np.nan,
                      "ratio": np.nan, "status": "undetermined"}
    ratio = delta / j if j != 0 else np.nan
    ok_ = bool(np.sign(delta) == np.sign(j) and abs(delta) >= LEVEL_FRAC * abs(j))
    return out | {"delta": delta, "slope_m_per_d": slope, "mad_resid": mres,
                  "ratio": float(ratio), "status": "consistent" if ok_ else "inconsistent"}


def part_arrays(part: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, barometrically corrected v, valid mask) of a stage-1 segment (``res['pre']`` or
    ``res['post']``)."""
    t = part["t"]
    v = part["v_raw"] - part["L"]
    kb = part["kind_base"]
    valid = np.isfinite(v) & (np.abs(part["v_raw"]) < C.SENTINEL_ABS) & ((kb & EXCLUDE_BITS) == 0)
    return t, v, valid


def offset_times(jumps: list[dict]) -> np.ndarray:
    """Sorted times of the offset-class jumps of >= 1 m (the window truncation points)."""
    return np.array(sorted(J["t"] for J in jumps
                           if J["cls"] == "offset" and abs(J["j"]) >= C.EPISODE_MIN_M), np.int64)


def check_jump(part: dict, jumps: list[dict], t0: int, j: float, other: int | None = None
               ) -> dict:
    """Check one jump of a stage-1 segment; ``other``: the paired jump of an episode (the
    window never crosses it; it is also one of the offset times)."""
    t, v, valid = part_arrays(part)
    ot = offset_times(jumps)
    lo = ot[ot < t0]
    hi = ot[ot > t0]
    lo_b = int(lo[-1]) if len(lo) else None
    hi_b = int(hi[0]) if len(hi) else None
    if other is not None:
        if other < t0:
            lo_b = max(lo_b, other) if lo_b is not None else other
        else:
            hi_b = min(hi_b, other) if hi_b is not None else other
    return level_shift(t, v, valid, int(t0), float(j), lo_b, hi_b)


# ----------------------------------------------------------------------------- segment check
SEG_MIN_HOURS = 12             # hourly medians needed on each side of a segment comparison
EP_OUT_MIN_D, EP_OUT_MAX_D = 15.0, 365.0   # outside windows of an episode: duration, clipped
RUN_STOP_HOURS = 6             # an excursion run ends after 6 hourly medians at the reference
RUN_MAX_D = 365.0              # and never runs longer than a year
EDGE_WINDOW_H = C.PERSIST_H[1]  # a rejected jump with no excursion run: flag [t0, t0 + 48 h]


def _hourly_in(t, v, valid, a, b):
    """Hourly medians of valid samples with a <= t < b."""
    i0, i1 = np.searchsorted(t, a, "left"), np.searchsorted(t, b, "left")
    ok = valid[i0:i1]
    return hourly_medians(t[i0:i1][ok], v[i0:i1][ok])


def segment_shift(t, v, valid, lo, t0, hi, j) -> dict:
    """Step persistence over the whole segments: median of the hourly medians of the valid
    samples in [lo, t0) and in [t0, hi); consistent when the difference has the sign of j and
    at least LEVEL_FRAC |j|."""
    _, hb = _hourly_in(t, v, valid, lo, t0)
    _, ha = _hourly_in(t, v, valid, t0, hi)
    out = {"seg_lo": int(lo), "seg_hi": int(hi), "seg_n_before": int(len(hb)),
           "seg_n_after": int(len(ha))}
    if len(hb) < SEG_MIN_HOURS or len(ha) < SEG_MIN_HOURS:
        return out | {"seg_median_before": np.nan, "seg_median_after": np.nan,
                      "seg_delta": np.nan, "seg_ratio": np.nan, "seg_status": "undetermined"}
    mb, ma = float(np.median(hb)), float(np.median(ha))
    d = ma - mb
    ok_ = bool(np.sign(d) == np.sign(j) and abs(d) >= LEVEL_FRAC * abs(j))
    return out | {"seg_median_before": mb, "seg_median_after": ma, "seg_delta": d,
                  "seg_ratio": d / j if j else np.nan,
                  "seg_status": "consistent" if ok_ else "inconsistent"}


SEG_WINDOW_D = 365.0           # segment check: up to a year on each side
SEG_MIN_DAYS = 5               # daily medians needed on each side


def daily_medians(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily medians of hourly medians (times: day start in seconds)."""
    ht, hv = hourly_medians(t, v)
    if len(ht) == 0:
        return np.zeros(0), np.zeros(0)
    d = (ht // DAY).astype(np.int64)
    g = pd.Series(hv).groupby(d, sort=True).median()
    return g.index.to_numpy(np.int64) * DAY, g.to_numpy(float)


def segment_shift_year(t, v, valid, lo, t0, hi, j) -> dict:
    """Step persistence over up to a year on each side: the median of the daily medians of
    the valid samples in [max(lo, t0 - 365 d), t0) and in [t0, min(hi, t0 + 365 d));
    consistent when the difference has the sign of j and at least LEVEL_FRAC |j|;
    undetermined with fewer than SEG_MIN_DAYS daily medians on a side. Round-2 design: the
    whole-segment medians of ``segment_shift`` are confounded by multi-year trends, and a
    least-absolute-deviation fit with a linear trend over the same windows is confounded by
    the seasonal swing (up to 13 m in Yunlin); a full year on each side spans one seasonal
    cycle. Chosen on pre-origin data and the design-time injections r = 0..9."""
    w = int(SEG_WINDOW_D * DAY)
    a, b = max(int(lo), int(t0) - w), min(int(hi), int(t0) + w)
    i0, i1 = np.searchsorted(t, a, "left"), np.searchsorted(t, b, "left")
    tt, vv, ok = t[i0:i1], v[i0:i1], valid[i0:i1]
    tt, vv = tt[ok], vv[ok]
    _, db = daily_medians(tt[tt < t0], vv[tt < t0])
    _, da = daily_medians(tt[tt >= t0], vv[tt >= t0])
    out = {"seg_lo": int(lo), "seg_hi": int(hi), "seg_win_lo": int(a), "seg_win_hi": int(b),
           "seg_n_before": int(len(db)), "seg_n_after": int(len(da))}
    if len(db) < SEG_MIN_DAYS or len(da) < SEG_MIN_DAYS:
        return out | {"seg_median_before": np.nan, "seg_median_after": np.nan,
                      "seg_delta": np.nan, "seg_ratio": np.nan, "seg_status": "undetermined"}
    mb, ma = float(np.median(db)), float(np.median(da))
    d = ma - mb
    ok_ = bool(np.sign(d) == np.sign(j) and abs(d) >= LEVEL_FRAC * abs(j))
    return out | {"seg_median_before": mb, "seg_median_after": ma, "seg_delta": d,
                  "seg_ratio": d / j if j else np.nan,
                  "seg_status": "consistent" if ok_ else "inconsistent"}


def episode_shift(t, v, valid, t1, t2, j1, j2, t_min, t_max) -> dict:
    """Episode persistence: median of the interior [t1, t2) against the median of the two
    outside windows [t1 - L, t1) and [t2, t2 + L), L = duration clipped to 15..365 days
    (within [t_min, t_max)); consistent when the difference has the sign of j1 and at least
    LEVEL_FRAC x mean(|j1|, |j2|)."""
    L = int(np.clip((t2 - t1) / DAY, EP_OUT_MIN_D, EP_OUT_MAX_D) * DAY)
    _, hi_ = _hourly_in(t, v, valid, t1, t2)
    _, hb = _hourly_in(t, v, valid, max(t1 - L, t_min), t1)
    _, ha = _hourly_in(t, v, valid, t2, min(t2 + L, t_max))
    ho = np.r_[hb, ha]
    jm = 0.5 * (abs(j1) + abs(j2))
    out = {"seg_n_inside": int(len(hi_)), "seg_n_outside": int(len(ho)), "seg_out_days": L / DAY}
    if len(hi_) < SEG_MIN_HOURS or len(ho) < SEG_MIN_HOURS:
        return out | {"seg_median_inside": np.nan, "seg_median_outside": np.nan,
                      "seg_delta": np.nan, "seg_ratio": np.nan, "seg_status": "undetermined"}
    mi, mo = float(np.median(hi_)), float(np.median(ho))
    d = mi - mo
    ok_ = bool(np.sign(d) == np.sign(j1) and abs(d) >= LEVEL_FRAC * jm)
    return out | {"seg_median_inside": mi, "seg_median_outside": mo, "seg_delta": d,
                  "seg_ratio": d / (np.sign(j1) * jm) if jm else np.nan,
                  "seg_status": "consistent" if ok_ else "inconsistent"}


def excursion_run(t, v, valid, t0, level_disp, level_ref, direction,
                  t_stop) -> tuple[int, int] | None:
    """Run of hourly medians adjacent to the jump at ``t0`` that sit closer to the displaced
    level than to the reference level. ``direction`` +1 searches forward from t0 (up to
    ``t_stop``), -1 backward from t0 (down to ``t_stop``); the run ends after
    RUN_STOP_HOURS consecutive hourly medians closer to the reference, and never exceeds
    RUN_MAX_D. Returns (t_first, t_last) in seconds (inclusive sample times) or None."""
    cap = int(RUN_MAX_D * DAY)
    if direction > 0:
        a, b = t0, min(t_stop, t0 + cap)
    else:
        a, b = max(t_stop, t0 - cap), t0
    ht, hv = _hourly_in(t, v, valid, a, b)
    if len(ht) == 0:
        return None
    if direction < 0:
        ht, hv = ht[::-1], hv[::-1]
    disp = np.abs(hv - level_disp) < np.abs(hv - level_ref)
    if not disp[0]:
        return None
    last, run_ref = 0, 0
    for k in range(len(hv)):
        if disp[k]:
            last, run_ref = k, 0
        else:
            run_ref += 1
            if run_ref >= RUN_STOP_HOURS:
                break
    if direction > 0:
        return int(max(ht[0], t0)), int(ht[last] + 3599)
    return int(ht[last]), int(min(ht[0] + 3599, t0 - 1))
