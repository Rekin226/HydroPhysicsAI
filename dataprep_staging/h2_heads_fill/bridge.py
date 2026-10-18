"""Daily bridge of partial and outage months (plan WP H2 technique step 1; flag 2).

For a well's clean daily record (L1 flag 0/1 days with >= 12 h), every run of missing days
that touches a target month is bridged linearly between the 3-day means at its two edges
(each edge needs at least 2 observed days of its 3), plus the well's mean deviation curve
from the bridge over the same calendar window in other years (at least 3 years whose window
is more than 80 % observed; resampled on 37 points, as in the pilot's ``s4_daily.py``).
The monthly mean is then taken over the completed daily record. A month is bridged only if
every one of its days is observed or bridged and every run is at most
``BRIDGE_MAX_GAP_DAYS`` long (the tested envelope); otherwise it is left to the monthly
router. The deviation-curve years are restricted by the caller (pre-origin: <= 2022).

Bridge form (``config.BRIDGE_FORM``): ``shape`` (linear bridge plus the deviation curve, the
plan's form) or ``lin`` (the linear bridge alone). The pilot chose ``shape`` on the Feb-Mar
window in 2013-2025, which includes 2023-25 truth (U7); this WP re-decides the form on
pre-origin design realizations (``s1_bridge_form.py``, rule declared there before the run).
``evaluate_windows`` returns both forms (``pred_shape``, ``pred_lin``) and ``pred`` = the
configured one.

Families (artificial gaps on the daily record):
* ``D``: the 2026 outage window (11 Feb - 19 Mar) removed in every year (plan 4.4;
  ``harness.gaps.daily_outage_windows``); deterministic, design-time evidence (the pilot
  scored it on 2013-2025).
* ``d``: H2's own random daily gaps, one per eligible well per realization, length uniform in
  5..60 days, start uniform in the well's record (seed ``1000*ord('d') + r``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from h2_heads_fill import config as C

DAY = pd.Timedelta(days=1)


def _edge_mean(s: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float:
    v = s.reindex(pd.date_range(a, b)).to_numpy(float)
    ok = np.isfinite(v)
    return float(v[ok].mean()) if ok.sum() >= C.BRIDGE_EDGE_MIN else np.nan


def _lin(pre: float, post: float, L: int) -> np.ndarray:
    x = np.arange(L) + 1
    return pre + (post - pre) * x / (L + 1)


def _nan_runs(v: np.ndarray):
    m = np.concatenate([[0], (~np.isfinite(v)).astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True))


class WellBridge:
    """Bridge machinery for one well's daily series (clean days only, any index order)."""

    def __init__(self, s: pd.Series, years_allowed, shape: bool | None = None):
        s = s.sort_index()
        self.use_shape = (C.BRIDGE_FORM == "shape") if shape is None else bool(shape)
        self.years = set(int(y) for y in years_allowed)
        if s.empty:
            self.s = s
            self.idx = pd.DatetimeIndex([])
            return
        self.idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        self.s = s.reindex(self.idx)
        v = self.s.to_numpy(float)
        self.runs = [(self.idx[a], self.idx[b - 1], b - a) for a, b in _nan_runs(v)]
        self._shape_cache: dict = {}

    def _shape(self, g0: pd.Timestamp, L: int) -> tuple[np.ndarray, int]:
        key = (g0, L)
        if key in self._shape_cache:
            return self._shape_cache[key]
        if not self.use_shape:
            out = (np.zeros(L), 0)
            self._shape_cache[key] = out
            return out
        curves = []
        for y in sorted(self.years):
            if y == g0.year:
                continue
            g0y = g0 + pd.DateOffset(years=y - g0.year)
            g1y = g0y + (L - 1) * DAY
            pre = _edge_mean(self.s, g0y - C.BRIDGE_EDGE_DAYS * DAY, g0y - DAY)
            post = _edge_mean(self.s, g1y + DAY, g1y + C.BRIDGE_EDGE_DAYS * DAY)
            if g1y.year not in self.years:
                continue
            if not (np.isfinite(pre) and np.isfinite(post)):
                continue
            win = self.s.reindex(pd.date_range(g0y, g1y)).to_numpy(float)
            if np.isfinite(win).mean() <= C.BRIDGE_SHAPE_OBS_FRAC:
                continue
            dev = pd.Series(win - _lin(pre, post, L)).interpolate(
                limit_direction="both").to_numpy()
            curves.append(np.interp(np.linspace(0, 1, C.BRIDGE_SHAPE_GRID),
                                    np.linspace(0, 1, L), dev))
        if len(curves) >= C.BRIDGE_SHAPE_MIN_YEARS:
            shape = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, C.BRIDGE_SHAPE_GRID),
                              np.mean(curves, 0))
        else:
            shape = np.zeros(L)
        out = (shape, len(curves))
        self._shape_cache[key] = out
        return out

    def bridge_month(self, month: pd.Timestamp, max_gap: int = C.BRIDGE_MAX_GAP_DAYS) -> dict:
        """Bridge one month (its start date). Returns a dict with ``ok`` and, when ok,
        ``value`` (monthly mean of the completed record), ``n_bridged``, ``frac``,
        ``max_run``, ``n_shape_years`` (minimum over the month's runs), ``n_clean``."""
        days = pd.date_range(month, month + pd.offsets.MonthEnd(0), freq="D")
        rec = {"month": month, "ok": False, "reason": ""}
        if len(self.idx) == 0 or days[0] < self.idx[0] or days[-1] > self.idx[-1]:
            rec["reason"] = "outside_record"
            return rec
        vals = self.s.reindex(days).to_numpy(float).copy()
        n_clean = int(np.isfinite(vals).sum())
        if n_clean == len(days):
            rec["reason"] = "complete"
            return rec
        max_run, nshape = 0, 99
        for g0, g1, L in self.runs:
            if g1 < days[0] or g0 > days[-1]:
                continue
            if L > max_gap:
                rec["reason"] = f"run_{L}d_gt_{max_gap}"
                return rec
            pre = _edge_mean(self.s, g0 - C.BRIDGE_EDGE_DAYS * DAY, g0 - DAY)
            post = _edge_mean(self.s, g1 + DAY, g1 + C.BRIDGE_EDGE_DAYS * DAY)
            if not (np.isfinite(pre) and np.isfinite(post)):
                rec["reason"] = "no_edges"
                return rec
            shape, ny = self._shape(g0, L)
            b = pd.Series(_lin(pre, post, L) + shape, index=pd.date_range(g0, g1))
            sel = b.reindex(days).to_numpy(float)
            fill = ~np.isfinite(vals) & np.isfinite(sel)
            vals[fill] = sel[fill]
            max_run, nshape = max(max_run, L), min(nshape, ny)
        if not np.isfinite(vals).all():
            rec["reason"] = "uncovered_days"
            return rec
        nb = len(days) - n_clean
        rec.update({"ok": True, "value": float(vals.mean()), "n_bridged": int(nb),
                    "frac": nb / len(days), "max_run": int(max_run),
                    "n_shape_years": int(nshape), "n_clean": n_clean,
                    "method": "daily_bridge_partial" if n_clean > 0 else
                    "daily_bridge_outage"})
        return rec


def frac_class(frac: float) -> int:
    for k, (a, b) in enumerate(C.BRIDGE_FRAC_CLASSES):
        if a < frac <= b or (k == 0 and frac == 0):
            return k + 1
    return len(C.BRIDGE_FRAC_CLASSES)


def daily_series(daily: pd.DataFrame) -> dict[str, pd.Series]:
    return {sid: g.set_index("date")["value"] for sid, g in daily.groupby("sid")}


def bridge_targets(h, years_allowed, targets=None, shape: bool | None = None) -> pd.DataFrame:
    """Bridge every non-C20 month of every well inside its daily record (the real gaps) or
    the given ``targets`` [(sid, month)]. Returns one row per attempted month."""
    ser = daily_series(h.daily)
    rows = []
    if targets is None:
        targets = []
        obs = np.isfinite(h.Y)
        for i, sid in enumerate(h.sids):
            s = ser.get(sid)
            if s is None or s.empty:
                continue
            first, last = s.index.min(), s.index.max()
            for j, d in enumerate(h.dates):
                if obs[i, j]:
                    continue
                if d + pd.offsets.MonthEnd(0) < first or d > last:
                    continue
                targets.append((sid, d))
    by = {}
    for sid, d in targets:
        by.setdefault(sid, []).append(d)
    for sid, months in by.items():
        wb = WellBridge(ser.get(sid, pd.Series(dtype=float)), years_allowed, shape=shape)
        for d in months:
            r = wb.bridge_month(pd.Timestamp(d))
            r["sid"] = sid
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- families
def d_windows(h, r: int, end: str, rng_family: str = "d"):
    """Family d: one random daily gap per eligible well (seed 1000*ord('d') + r). A well is
    eligible if its clean record has >= 365 days; the gap starts on a uniformly drawn clean
    day at least 4 days after the record start and ends (length uniform 5..60) at least 4
    days before the record end and before ``end`` (last day of the variant's window).
    Returns [(sid, start, length)]."""
    from harness.gaps import rng_for
    rng = rng_for(rng_family, r)
    ser = daily_series(h.daily)
    last_ok = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
    out = []
    for sid in h.sids:
        s = ser.get(sid)
        if s is None or len(s) < 365:
            continue
        idx = s.index.sort_values()
        L = int(rng.integers(C.BRIDGE_D_LEN[0], C.BRIDGE_D_LEN[1] + 1))
        lo = idx[0] + 4 * DAY
        hi = min(idx[-1], last_ok) - (L + 4) * DAY
        cand = idx[(idx >= lo) & (idx <= hi)]
        if len(cand) == 0:
            continue
        g0 = cand[int(rng.integers(0, len(cand)))]
        out.append((sid, g0, L))
    return out


def evaluate_windows(h, windows, years_allowed, kind: str) -> pd.DataFrame:
    """Mask each window in its well's clean daily record and bridge every month it touches
    whose truth is C20 and that becomes non-C20 after the mask. Rows: sid, month, window
    start/length, truth, bridged value (NaN if not bridgeable), reason, n_clean_after,
    frac, partial mean of the surviving days; ``pred_shape`` / ``pred_lin`` are the two
    bridge forms and ``pred`` the configured one (``config.BRIDGE_FORM``)."""
    ser = daily_series(h.daily)
    si = {s: i for i, s in enumerate(h.sids)}
    ti = {d: j for j, d in enumerate(h.dates)}
    rows = []
    for sid, g0, L in windows:
        s = ser[sid]
        g1 = g0 + (L - 1) * DAY
        sm = s[(s.index < g0) | (s.index > g1)]
        wbs = {"shape": WellBridge(sm, years_allowed, shape=True),
               "lin": WellBridge(sm, years_allowed, shape=False)}
        m0 = g0.to_period("M").to_timestamp()
        m1 = g1.to_period("M").to_timestamp()
        for mth in pd.date_range(m0, m1, freq="MS"):
            j = ti.get(mth)
            if j is None:
                continue
            i = si[sid]
            truth = h.Y[i, j]
            if not np.isfinite(truth):
                continue
            mdays = pd.date_range(mth, mth + pd.offsets.MonthEnd(0))
            left = sm.reindex(mdays)
            k_after = int(left.notna().sum())
            if k_after >= 20:
                continue
            bf = {k: wb.bridge_month(mth) for k, wb in wbs.items()}
            b = bf[C.BRIDGE_FORM]
            rows.append({"kind": kind, "sid": sid, "month": mth, "w": i, "t": j,
                         "win_start": g0, "win_len": L, "truth": float(truth),
                         "pred": b.get("value", np.nan),
                         "pred_shape": bf["shape"].get("value", np.nan),
                         "pred_lin": bf["lin"].get("value", np.nan), "ok": bool(b["ok"]),
                         "reason": b.get("reason", ""), "n_clean_after": k_after,
                         "frac": b.get("frac", (len(mdays) - k_after) / len(mdays)),
                         "max_run": b.get("max_run", np.nan),
                         "n_shape_years": b.get("n_shape_years", np.nan),
                         "partial": float(left.mean()) if k_after else np.nan})
    return pd.DataFrame(rows)


def D_windows(h, years) -> list:
    """Family D: the 11 Feb - 19 Mar window in each year at every well with data there."""
    ser = daily_series(h.daily)
    out = []
    for sid in h.sids:
        s = ser.get(sid)
        if s is None or s.empty:
            continue
        for y in years:
            g0 = pd.Timestamp(f"{y}-{C.OUTAGE_2026[0][5:]}")
            g1 = pd.Timestamp(f"{y}-{C.OUTAGE_2026[1][5:]}")
            if s.index.min() < g0 - 3 * DAY and s.index.max() > g1 + 3 * DAY:
                out.append((sid, g0, int((g1 - g0).days) + 1))
    return out
