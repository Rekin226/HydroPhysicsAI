"""WP S1b repair round: pure functions for the after-the-fact readings (no file access).

Used by ``s1b_r2_readings.py``, ``s1b_r2_synth.py`` and ``s1b_r2_u1.py``. Nothing here changes
a registered rule: the registered test (``s1b_core``, ``s1b_run``) is untouched and its result
stands as registered. These functions compute the plan-literal and U1-strict readings the
verifier asked for, a dependent-correlation test for (d1), and helpers for the InSAR bridge
decomposition and GNSS data gaps.

Plan-literal readings of check (c) (plan WP S1, "within the +-30-40 ppm excess-rate bracket";
the pilot table ``gnss_vs_ring_stepfixed.csv`` gives that bracket as the pre-excess..post-excess
range ``C_post..C_pre``, whose half-width is 30-40 ppm of L at its sites):

* ``k0``: the unwidened range ``[min(C_pre, C_post), max(C_pre, C_post)]``;
* ``range_ppm:X``: that range extended by X ppm of L on each side;
* ``mid_ppm:X``: ``|C - C_mid| <= X ppm of L`` (scale within X ppm of the instrument mid scale);
* ``k:X``: the registered form with X sigma (``k:2`` is the registered rule).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

RULES = ("k:0", "k:0.5", "k:1", "k:1.5", "k:2", "k:2.5", "range_ppm:30", "range_ppm:40",
         "mid_ppm:30", "mid_ppm:40")
PLAN_LITERAL = ("k:0", "range_ppm:30", "range_ppm:40", "mid_ppm:30", "mid_ppm:40")


def contains(rule: str, c: float, c_pre: float, c_post: float, s_pre: float, s_post: float,
             L_mm: float) -> bool | None:
    """Containment of a candidate gap compaction ``c`` (mm) under one bracket rule."""
    vals = (c, c_pre, c_post, L_mm)
    if any(v is None or not np.isfinite(v) for v in vals):
        return None
    kind, x = rule.split(":")
    x = float(x)
    if kind == "k":
        lo = min(c_pre - x * s_pre, c_post - x * s_post)
        hi = max(c_pre + x * s_pre, c_post + x * s_post)
    elif kind == "range_ppm":
        d = x * 1e-6 * L_mm
        lo, hi = min(c_pre, c_post) - d, max(c_pre, c_post) + d
    elif kind == "mid_ppm":
        d = x * 1e-6 * L_mm
        mid = 0.5 * (c_pre + c_post)
        lo, hi = mid - d, mid + d
    else:
        raise ValueError(rule)
    return bool(lo <= c <= hi)


def min_k(c: float, c_pre: float, c_post: float, s_pre: float, s_post: float) -> float:
    """Smallest k for which ``[min_w(C_w - k s_w), max_w(C_w + k s_w)]`` contains ``c``."""
    lo0, hi0 = min(c_pre, c_post), max(c_pre, c_post)
    if lo0 <= c <= hi0:
        return 0.0
    if c < lo0:
        return float(min((c_pre - c) / s_pre, (c_post - c) / s_post))
    return float(min((c - c_pre) / s_pre, (c - c_post) / s_post))


def k_star(ks, min_frac: float = 0.8) -> float:
    """Smallest k at which at least ``ceil(min_frac n)`` sites contain the value."""
    ks = sorted(float(k) for k in ks if np.isfinite(k))
    if not ks:
        return float("nan")
    need = math.ceil(min_frac * len(ks))
    return ks[need - 1]


def family_counts(rows: list[dict], rule: str, key: str = "C_primary_mm",
                  min_frac: float = 0.8) -> dict:
    """Containment count of ``key`` under ``rule`` over evaluable rows (dicts with C_pre_mm,
    C_post_mm, sigma_pre_mm, sigma_post_mm, L_m), with the registered 80 % rule."""
    ev = [r for r in rows if r.get("evaluable")]
    got = [contains(rule, r[key], r["C_pre_mm"], r["C_post_mm"], r["sigma_pre_mm"],
                    r["sigma_post_mm"], 1000.0 * r["L_m"]) for r in ev]
    n = len(ev)
    n_in = sum(bool(g) for g in got)
    need = math.ceil(min_frac * n) if n else 0
    return {"rule": rule, "n": n, "n_contain": n_in, "n_needed": need,
            "frac": (n_in / n) if n else None, "containment_pass": bool(n and n_in >= need),
            "sites_outside": [r["site"] for r, g in zip(ev, got, strict=True) if not g]}


def rmse(rows: list[dict], key: str, ref: str) -> float:
    ev = [r for r in rows if r.get("evaluable")]
    if not ev:
        return float("nan")
    return float(np.sqrt(np.mean([(r[key] - r[ref]) ** 2 for r in ev])))


def williams_t(r12: float, r13: float, r23: float, n: int) -> dict:
    """Williams' t for H0 rho12 = rho13 with variable 1 shared (Steiger 1980, eq. 7)."""
    det = 1.0 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2.0 * r12 * r13 * r23
    rbar = 0.5 * (r12 + r13)
    den = 2.0 * (n - 1) / (n - 3) * det + rbar ** 2 * (1.0 - r23) ** 3
    t = (r12 - r13) * math.sqrt((n - 1) * (1.0 + r23) / den)
    from scipy import stats
    df = n - 3
    return {"t": float(t), "df": int(df), "p_one_sided": float(stats.t.sf(t, df)),
            "det_R": float(det)}


def corr_diff_bootstrap(x, yb, yr, n_boot: int = 10000, seed: int = 0) -> dict:
    """Site bootstrap of r(x, yb) - r(x, yr); resamples with fewer than 3 distinct sites or
    zero variance are skipped."""
    x, yb, yr = (np.asarray(v, float) for v in (x, yb, yr))
    n = len(x)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(i.tolist())) < 3:
            continue
        xs, bs, rs = x[i], yb[i], yr[i]
        if min(xs.std(), bs.std(), rs.std()) == 0:
            continue
        out.append(np.corrcoef(xs, bs)[0, 1] - np.corrcoef(xs, rs)[0, 1])
    d = np.asarray(out)
    return {"n_boot_used": int(len(d)), "diff_median": float(np.median(d)),
            "ci90": [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))],
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_diff_le_0": float(np.mean(d <= 0))}


def fisher_ci(r: float, n: int, level: float = 0.90) -> list[float]:
    from scipy import stats
    z = math.atanh(r)
    h = stats.norm.ppf(0.5 + level / 2) / math.sqrt(n - 3)
    return [float(math.tanh(z - h)), float(math.tanh(z + h))]


def bridge_increments(dates, d, seg, s_from: int, s_to: int) -> list[dict]:
    """Increments of a series across each network gap between segments ``s_from`` and
    ``s_to``: value at the first finite epoch of segment s+1 minus value at the last finite
    epoch of segment s (the part LiCSBAS set by its linear-velocity constraint plus the two
    epochs' noise)."""
    dates = pd.DatetimeIndex(dates)
    d = np.asarray(d, float)
    seg = np.asarray(seg)
    out = []
    for s in range(int(s_from), int(s_to)):
        a = np.flatnonzero((seg == s) & np.isfinite(d))
        b = np.flatnonzero((seg == s + 1) & np.isfinite(d))
        if not len(a) or not len(b):
            out.append({"from_seg": s, "inc_mm": float("nan"), "date_last": None,
                        "date_first": None, "days": None})
            continue
        i, j = a[-1], b[0]
        out.append({"from_seg": s, "inc_mm": float(d[j] - d[i]),
                    "date_last": str(dates[i].date()), "date_first": str(dates[j].date()),
                    "days": int((dates[j] - dates[i]).days)})
    return out


def longest_gap(dates, lo, hi) -> dict:
    """Longest run of days without a sample strictly inside (lo, hi), counting the interval
    ends as samples: returns its length in days and the bracketing sample dates."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    d = pd.DatetimeIndex(dates)
    d = d[(d > lo) & (d < hi)].sort_values()
    pts = pd.DatetimeIndex([lo]).append(d).append(pd.DatetimeIndex([hi]))
    gaps = np.diff(pts.asi8) / 86400e9
    k = int(np.argmax(gaps))
    return {"max_gap_days": float(gaps[k]) - 1.0, "after": str(pts[k].date()),
            "before": str(pts[k + 1].date()), "n_samples": int(len(d))}
