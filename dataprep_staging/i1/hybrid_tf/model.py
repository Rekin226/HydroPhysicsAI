"""Transfer-function features, ridge fits and the forecast families of the hybrid layer.

A driver's effect on a well's head builds up and fades over months. The transfer-function
form used here passes each driver's anomaly through causal exponential filters with response
times ``TAUS`` (0 = same month) and regresses the head on all of them by ridge regression on
standardised features, with month-of-year dummies (the well's seasonal cycle). Two targets:

* ``tf``      the head itself (pure transfer-function model);
* ``phys_tf`` the head minus R0's head (a TF correction of the physics model).

Two forecast settings:

* **seasonal** (predictions frozen at the origin, no head observed after it): the TF model
  alone, or the TF model plus its residual at the origin decayed by the residual's lag-1
  coefficient (``*_ar``), as the existing statistical layer (``twin.hybrid`` ``stat``) decays
  the climatology anomaly;
* **one month ahead** (the previous month's head is known): the TF model plus the lag-1
  residual correction (``*_ar1``).

Drivers that are not yet recorded at the forecast time are replaced, from their cut-off month
on, by their month-of-year climatology over the fit window (or their last value for the
step-like public wells): :func:`cut_series`.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def efilter(x: np.ndarray, tau: float) -> np.ndarray:
    """Causal exponential filter with unit gain; ``tau = 0`` returns ``x``."""
    if tau == 0:
        return np.asarray(x, dtype=float).copy()
    a = np.exp(-1.0 / tau)
    out = np.empty(len(x), dtype=float)
    acc = 0.0
    for t, v in enumerate(x):
        acc = a * acc + (1 - a) * v
        out[t] = acc
    return out


def cut_series(s: np.ndarray, cut: int, cal: np.ndarray, fit: np.ndarray, hold: bool
               ) -> np.ndarray:
    """``s`` with every month from index ``cut`` on replaced by the month-of-year mean of
    ``s`` over the months flagged ``fit`` (or by ``s[cut - 1]`` when ``hold``)."""
    out = np.asarray(s, dtype=float).copy()
    if cut >= len(out):
        return out
    cut = max(cut, 1)
    if hold:
        out[cut:] = out[cut - 1]
        return out
    clim = np.array([s[fit & (cal == m)].mean() if (fit & (cal == m)).any() else s[fit].mean()
                     for m in range(12)])
    out[cut:] = clim[cal[cut:]]
    return out


def features(series: dict[str, np.ndarray], names, ref_mean: dict[str, float]) -> np.ndarray:
    """(T, len(names) * len(TAUS)) filtered anomalies; ``ref_mean`` is the fixed centring."""
    cols = []
    for n in names:
        s = np.asarray(series[n], dtype=float) - ref_mean[n]
        for tau in C.TAUS:
            cols.append(efilter(s, tau))
    return np.column_stack(cols)


def month_dummies(cal: np.ndarray) -> np.ndarray:
    return np.column_stack([(cal == k).astype(float) for k in range(11)])


class Ridge:
    """Ridge regression on standardised columns with an unpenalised intercept.

    Two guards against extrapolation (added after the first design look, 2026-10-20, when
    wells with short records produced errors of tens to thousands of metres): a column that
    is constant over the fit rows carries no information and is dropped; at prediction time
    each standardised column is clipped to the range it spans over the fit rows."""

    def __init__(self, alpha: float = C.ALPHA):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> Ridge:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.keep = sd > 1e-9 * (np.abs(mu) + 1.0)
        self.mu, self.sd = mu[self.keep], sd[self.keep]
        Xs = (X[:, self.keep] - self.mu) / self.sd
        self.lo, self.hi = Xs.min(axis=0), Xs.max(axis=0)
        self.c = float(y.mean())
        k = Xs.shape[1]
        self.beta = np.linalg.solve(Xs.T @ Xs + self.alpha * np.eye(k), Xs.T @ (y - self.c))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = np.clip((X[:, self.keep] - self.mu) / self.sd, self.lo, self.hi)
        return self.c + Xs @ self.beta


def lag1(r: np.ndarray, ok: np.ndarray) -> tuple[float, int]:
    """Lag-1 regression-through-origin coefficient of ``r`` over consecutive pairs both
    flagged ``ok``, clipped to [0, 0.99]; NaN below ``RHO_MIN_PAIRS`` pairs."""
    a, b = r[:-1], r[1:]
    m = ok[:-1] & ok[1:] & np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    den = float((a[m] ** 2).sum())
    if n < C.RHO_MIN_PAIRS or den <= 0:
        return float("nan"), n
    return float(np.clip((a[m] * b[m]).sum() / den, 0.0, 0.99)), n


def month_clim(y: np.ndarray, fit: np.ndarray, cal: np.ndarray) -> np.ndarray:
    """(T,) month-of-year mean of ``y`` over the fit months (fallback: their mean)."""
    m_all = float(np.nanmean(y[fit]))
    eff = np.array([np.nanmean(y[fit & (cal == k)]) if np.isfinite(y[fit & (cal == k)]).any()
                    else m_all for k in range(12)])
    return eff[cal]


def origin_index(y: np.ndarray, origin: int) -> int:
    """Index of the last observation at or before ``origin`` within ``MAX_ORIGIN_GAP``
    months, or -1."""
    if origin < 0:
        return -1
    lo = max(origin - C.MAX_ORIGIN_GAP, 0)
    idx = np.flatnonzero(np.isfinite(y[lo:origin + 1]))
    return int(lo + idx[-1]) if idx.size else -1
