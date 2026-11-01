"""Pure functions of I1.CH_backcast.v1 (round 7 rank 18) and its governed local product
heads/L1_pre2012 v1 (user ruling R7.l1_pre2012: built from ls_cache, kept local, never
committed). No function here opens a file.

* ``parse_xy``: ``LocationByTWD97`` in either order (``mechanism._xy``'s rule);
* ``nest_wells``: wells within 1000 m of a ring site (``mechanism.NEST_RADIUS_M``): the
  wells whose monthly 2010-25 ls_cache anomalies ``mechanism._long_head`` may have read; a
  superset (no value is needed to form it), excluded from the primary and flagged;
* ``available_months``: from TIMESTAMPS only (no value): months of the window with at least
  ``BC_MONTH_MIN_DAYS`` days holding at least ``BC_DAY_MIN_HOURS`` distinct hours (H1a's C20
  rule applied to the record's own sampling, an upper bound on QC'd months);
* ``feasibility``: the registered stop rule (>= 10 Changhua wells with >= 18 months, nest
  wells excluded);
* ``qc_monthly``: L1_pre2012's QC of one well's 2010-11 raw record: H1a's pass-1 sample
  screens (``h1_heads_qc.screens.screen``: sentinels, jumps/spikes/episodes/steps, flat
  runs, 12 m rolling-median departures, dropouts, 15 x MAD around the window's own robust
  level, below-screen-bottom) with the robust reference fitted on the 2010-11 window itself
  (``pipeline.robust_ref``'s rule), clean samples -> hourly means -> daily means (>= 12 h)
  -> monthly means (>= 20 days), flag 0. No barometric correction (no 2010-11 pressure
  product is held; declared), no gap fill.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402

COUNTY_ZH = {"彰化縣": "changhua", "雲林縣": "yunlin"}


def parse_xy(s) -> tuple[float, float] | None:
    try:
        a, b = (float(v) for v in str(s).split()[:2])
    except (ValueError, TypeError):
        return None
    x, y = (a, b) if a < 1.0e6 else (b, a)
    return (x, y) if 140000 <= x <= 240000 and 2580000 <= y <= 2700000 else None


def nest_wells(xy: dict, ring_xy: np.ndarray, radius: float = K.BC_NEST_RADIUS_M) -> set:
    """Sids whose (x, y) lies within ``radius`` of any ring site."""
    r = np.asarray(ring_xy, float).reshape(-1, 2)
    out = set()
    for s, p in xy.items():
        if p is None:
            continue
        d = np.hypot(r[:, 0] - p[0], r[:, 1] - p[1])
        if (d <= radius).any():
            out.add(str(s))
    return out


def window_months(first: str = K.BC_WINDOW[0], last: str = K.BC_WINDOW[1]) -> list[str]:
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def available_months(ts, first: str = K.BC_WINDOW[0], last: str = K.BC_WINDOW[1],
                     day_min_hours: int = K.BC_DAY_MIN_HOURS,
                     month_min_days: int = K.BC_MONTH_MIN_DAYS) -> list[str]:
    """Months of [first, last] whose sampling (timestamps only) could make a monthly value."""
    t = pd.DatetimeIndex(pd.to_datetime(ts))
    lo = pd.Timestamp(f"{first}-01")
    hi = pd.Period(last, "M").end_time
    t = t[(t >= lo) & (t <= hi)]
    if len(t) == 0:
        return []
    hours = pd.Series(1, index=t.floor("h")).groupby(level=0).size().index
    per_day = pd.Series(1, index=hours).groupby(hours.floor("D")).size()
    good_days = per_day[per_day >= day_min_hours].index
    per_month = pd.Series(1, index=good_days).groupby(good_days.to_period("M")).size()
    return sorted(str(m) for m, n in per_month.items() if n >= month_min_days)


def feasibility(months_by_well: dict, county_of: dict, excluded: set,
                min_wells: int = K.BC_MIN_WELLS, min_months: int = K.BC_MIN_MONTHS) -> dict:
    """The registered stop rule on availability counts."""
    ch = {s: len(m) for s, m in months_by_well.items() if county_of.get(s) == "changhua"}
    ok_wells = sorted(s for s, n in ch.items() if n >= min_months and s not in excluded)
    return {"n_changhua_candidates": len(ch),
            "n_changhua_ge_min_months": sum(n >= min_months for n in ch.values()),
            "n_excluded_nest": len(set(ch) & set(excluded)),
            "eligible_wells": ok_wells, "n_eligible": len(ok_wells),
            "min_wells": min_wells, "min_months": min_months,
            "passed": len(ok_wells) >= min_wells}


# ----------------------------------------------------------------------------- QC
def robust_ref(v: np.ndarray, sentinel_abs: float = 1000.0) -> tuple[float, float]:
    """``h1_heads_qc.pipeline.robust_ref``'s rule (median, MAD; NaN with < 100 samples)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (np.abs(v) < sentinel_abs)]
    if len(v) < 100:
        return np.nan, np.nan
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad <= 0:
        mad = float(np.std(v)) or 1.0
    return med, mad


def qc_monthly(s: pd.Series, elev: float | None, depth: float | None,
               first: str = K.BC_WINDOW[0], last: str = K.BC_WINDOW[1]) -> dict:
    """One well's 2010-11 raw series (value indexed by datetime) -> monthly means and counts.

    Returns {"monthly": Series 'YYYY-MM' -> m, "n_days": Series, "n_raw", "n_clean",
    "kind_counts"}."""
    from h1_heads_qc import screens

    s = pd.Series(pd.to_numeric(s, errors="coerce"), index=pd.to_datetime(s.index))
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna()
    lo, hi = pd.Timestamp(f"{first}-01"), pd.Period(last, "M").end_time
    s = s[(s.index >= lo) & (s.index <= hi)]
    empty = pd.Series(dtype=float)
    if s.empty:
        return {"monthly": empty, "n_days": pd.Series(dtype=int), "n_raw": 0, "n_clean": 0,
                "kind_counts": {}}
    t, v = screens.to_arrays(s)
    med, mad = robust_ref(v)
    res = screens.screen(t, v, med_ref=med, mad_ref=mad, elev=elev, depth=depth)
    kind = res["kind"]
    good = kind == 0
    hourly = pd.Series(v[good], index=pd.to_datetime(t[good], unit="s")).resample("h").mean()
    hourly = hourly.dropna()
    if hourly.empty:
        return {"monthly": empty, "n_days": pd.Series(dtype=int), "n_raw": int(len(v)),
                "n_clean": 0, "kind_counts": _kinds(kind)}
    day_n = hourly.groupby(hourly.index.floor("D")).size()
    day_m = hourly.groupby(hourly.index.floor("D")).mean()
    day_m = day_m[day_n >= K.BC_DAY_MIN_HOURS]
    per = day_m.index.to_period("M")
    mon_n = day_m.groupby(per).size()
    mon_m = day_m.groupby(per).mean()
    keep = mon_n >= K.BC_MONTH_MIN_DAYS
    mon_m, mon_n = mon_m[keep], mon_n[keep]
    mon_m.index = mon_m.index.astype(str)
    mon_n.index = mon_n.index.astype(str)
    return {"monthly": mon_m, "n_days": mon_n, "n_raw": int(len(v)),
            "n_clean": int(good.sum()), "kind_counts": _kinds(kind)}


def _kinds(kind: np.ndarray) -> dict:
    from h1_heads_qc import screens

    bits = {n: getattr(screens, n) for n in dir(screens) if n.startswith("K_")}
    return {n: int(((kind & b) > 0).sum()) for n, b in sorted(bits.items())}
