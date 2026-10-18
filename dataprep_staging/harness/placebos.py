"""Placebo generators (plan WP A1; used by P2a, P2f, P4.2, P4.3, V2).

A temporal regressor earns its place only if it beats placebos that keep the properties a
spurious fit could exploit:

* :func:`shift_months` - the regressor shifted by six months (keeps every value and the
  seasonal amplitude, breaks the phase);
* :func:`own_climatology` - its own 12-month climatology (isolates the seasonal part: beating
  it shows that the interannual timing is detected);
* :func:`year_shuffle` - a year-shuffled profile that keeps the seasonal shape (P2a: the
  Chow-Lin profile with its calendar years permuted, a derangement by default);
* :func:`distant_covariate` and :func:`regional_total` - kWh more than 15 km away and the
  Yunlin total, as regional covariates to regress out (P2a, P2f);
* :func:`rain_deficit_index` - a standardized rain-deficit anomaly (P4.2 drought-confounding
  placebo).

Anything fitted (a climatology, a standardization) is fitted on the months marked ``fit``
only, which must end by the origin for any pre-origin use (U3). Randomized placebos take an
explicit seed; the convention for downstream WPs is ``6000 + r`` for realization ``r``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from common.paths import ORIGIN

PLACEBO_SEED0 = 6000


def _dates(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(dates))


def _fit_mask(dates, fit_end: str | None):
    di = _dates(dates)
    if fit_end is None:
        return np.ones(len(di), bool)
    return np.asarray(di.strftime("%Y-%m") <= fit_end)


def shift_months(v, k: int = 6, mode: str = "circular") -> np.ndarray:
    """The series shifted forward by ``k`` months along its last axis: ``out[t] = v[t-k]``.

    ``mode="circular"`` wraps the last k months to the start (keeps the exact distribution
    and, for k = 6 on whole years, the month-of-year amplitude); ``mode="pad"`` leaves NaN at
    the start."""
    v = np.asarray(v, dtype=float)
    if mode == "circular":
        return np.roll(v, k, axis=-1)
    if mode == "pad":
        out = np.full_like(v, np.nan)
        if k >= 0:
            out[..., k:] = v[..., :v.shape[-1] - k]
        else:
            out[..., :k] = v[..., -k:]
        return out
    raise ValueError("mode must be 'circular' or 'pad'")


def own_climatology(v, dates, fit_end: str | None = ORIGIN) -> np.ndarray:
    """Month-of-year mean of ``v`` (last axis = time) over months up to ``fit_end``, spread
    over every month. ``fit_end=None`` uses all months (after-the-fact analysis only)."""
    v = np.asarray(v, dtype=float)
    di = _dates(dates)
    fit = _fit_mask(di, fit_end)
    out = np.full_like(v, np.nan)
    for m in range(1, 13):
        sel = np.asarray(di.month == m)
        if not sel.any():
            continue
        use = sel & fit
        if not use.any():
            raise ValueError(f"no fit months for calendar month {m}")
        out[..., sel] = np.nanmean(v[..., use], axis=-1, keepdims=True)
    return out


def year_shuffle(v, dates, seed: int, derange: bool = True,
                 years: list[int] | None = None) -> tuple[np.ndarray, dict]:
    """Permute whole calendar years of a monthly series (last axis = time): each year's
    12-month block moves to another year, so the seasonal shape is kept and the
    interannual timing is destroyed. Only complete years (or ``years``) are permuted;
    partial years stay in place. ``derange=True`` re-draws until no year keeps its place.
    Returns (shuffled, {target_year: source_year})."""
    v = np.asarray(v, dtype=float)
    di = _dates(dates)
    cand = sorted(y for y in set(di.year) if (di.year == y).sum() == 12)
    if years is not None:
        cand = [y for y in cand if y in set(years)]
    if len(cand) < 2:
        raise ValueError("need at least two complete years to shuffle")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = list(rng.permutation(cand))
        if not derange or all(a != b for a, b in zip(cand, perm, strict=True)):
            break
    else:  # pragma: no cover - 1000 failures is practically impossible for >= 2 years
        raise RuntimeError("could not draw a derangement")
    out = v.copy()
    mapping = {}
    for tgt, src in zip(cand, perm, strict=True):
        out[..., np.asarray(di.year == tgt)] = v[..., np.asarray(di.year == src)]
        mapping[int(tgt)] = int(src)
    return out, mapping


def distant_covariate(values, cell_xy, target_xy, min_km: float = 15.0) -> np.ndarray:
    """Sum over cells farther than ``min_km`` from each target of ``values`` (A, T).

    ``cell_xy`` (A, 2) and ``target_xy`` (n, 2) or (2,) in metres. Returns (n, T) (or (T,)
    for a single target): the regional covariate 'kWh more than 15 km away'."""
    val = np.asarray(values, dtype=float)
    cxy = np.asarray(cell_xy, dtype=float)
    txy = np.atleast_2d(np.asarray(target_xy, dtype=float))
    d = np.sqrt(((txy[:, None, :] - cxy[None, :, :]) ** 2).sum(-1)) / 1000.0
    far = (d > min_km).astype(float)
    out = far @ np.nan_to_num(val)
    return out[0] if np.asarray(target_xy).ndim == 1 else out


def regional_total(values, sel) -> np.ndarray:
    """Sum of ``values`` (A, T) over the cells in ``sel`` (A,) bool (e.g. all Yunlin cells)."""
    val = np.asarray(values, dtype=float)
    return np.nan_to_num(val[np.asarray(sel, bool)]).sum(0)


def rain_deficit_index(rain, dates, window: int = 3, fit_end: str | None = ORIGIN
                       ) -> np.ndarray:
    """Standardized rain-deficit anomaly: the trailing ``window``-month rain sum, minus its
    month-of-year mean, divided by its month-of-year sd (both over fit months up to
    ``fit_end``), sign flipped so that a drought is positive. NaN for the first
    ``window - 1`` months."""
    r = pd.Series(np.asarray(rain, dtype=float), index=_dates(dates))
    s = r.rolling(window, min_periods=window).sum()
    fit = _fit_mask(r.index, fit_end)
    mo = r.index.month
    out = np.full(len(s), np.nan)
    for m in range(1, 13):
        sel = np.asarray(mo == m)
        base = s[sel & fit].dropna()
        if len(base) < 2:
            continue
        mu, sd = base.mean(), base.std(ddof=1)
        out[sel] = -(s[sel].to_numpy() - mu) / (sd if sd > 0 else np.nan)
    return out
