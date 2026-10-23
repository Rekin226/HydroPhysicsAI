"""O1 phase coherence of hourly heads (barometric acceptance check, plan WP H1).

Copied from the head-signature pilot (``_pilots/head_pumping_signatures/scripts/
well_features.py``: ``astro``, ``nodal``, ``month_harmonics``; ``tide_coherence.py``: the
phase relative to the astronomical argument and the coherence across months), unchanged in
substance: per month with >= 20 days of hourly means, least squares of a cubic B-spline trend
(knots every 3 days) plus Q1, O1, S1, N2, M2, S2, S3, S4; the O1 phase relative to its
astronomical argument V_O1 + u; coherence = |mean over months of exp(i g_O1)| per well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

SPEED = {"Q1": 13.3986609, "O1": 13.9430356, "S1": 15.0, "N2": 28.4397295,
         "M2": 28.9841042, "S2": 30.0, "S3": 45.0, "S4": 60.0}
CONS = list(SPEED)


def astro(t: pd.Timestamp):
    d = (t - pd.Timestamp("2000-01-01 12:00")).total_seconds() / 86400.0
    s = (218.3164 + 13.17639648 * d) % 360
    h = (280.4661 + 0.98564736 * d) % 360
    N = (125.0445 - 0.05295377 * d) % 360
    return s, h, N


def nodal(N):
    n = np.radians(N)
    u = {"O1": 10.80 * np.sin(n) - 1.34 * np.sin(2 * n) + 0.19 * np.sin(3 * n),
         "M2": -2.14 * np.sin(n)}
    return u


def month_harmonics(hh: pd.Series, mstart: pd.Timestamp, mend: pd.Timestamp):
    t = (hh.index - mstart).total_seconds().to_numpy() / 3600.0
    T = (mend - mstart).total_seconds() / 3600.0
    tref = T / 2.0
    k = 3
    inner = np.arange(72.0, T - 36.0, 72.0)
    knots = np.r_[[0.0] * (k + 1), inner, [T] * (k + 1)]
    B = BSpline.design_matrix(np.clip(t, 0, T - 1e-6), knots, k).toarray()
    cols = [B]
    for c in CONS:
        w = np.radians(SPEED[c]) * (t - tref)
        cols += [np.cos(w)[:, None], np.sin(w)[:, None]]
    X = np.hstack(cols)
    y = hh.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    nb = B.shape[1]
    out = {}
    for i, c in enumerate(CONS):
        a, b = beta[nb + 2 * i], beta[nb + 2 * i + 1]
        out[c] = complex(a, -b)
    return out, mstart + pd.Timedelta(hours=tref)


def o1_phases(hh: pd.Series, months: pd.PeriodIndex) -> pd.DataFrame:
    """O1 (and M2) phase relative to the astronomical argument for every month with >= 20
    days of hourly values. ``hh``: hourly series (NaN = missing)."""
    hv = hh.dropna()
    tv = hv.index.to_numpy()
    rows = []
    for m in months:
        m0, m1 = m.start_time, (m + 1).start_time
        a, z = np.searchsorted(tv, np.datetime64(m0)), np.searchsorted(tv, np.datetime64(m1))
        if z - a < 24 * 20:
            continue
        c, tref = month_harmonics(hv.iloc[a:z], m0, m1)
        tu = tref - pd.Timedelta(hours=8)
        s, h, N = astro(tu)
        u = nodal(N)
        Th = 180.0 + 15.0 * (tu.hour + tu.minute / 60.0)
        VO1 = Th - 2 * s + h + 90.0 + u["O1"]
        VM2 = 2 * Th - 2 * s + 2 * h + u["M2"]
        rows.append({"month": str(m), "gO1": np.radians(VO1) - np.angle(c["O1"]),
                     "gM2": np.radians(VM2) - np.angle(c["M2"]),
                     "A_O1": abs(c["O1"]), "A_M2": abs(c["M2"])})
    return pd.DataFrame(rows)


def coherence(g: np.ndarray) -> float:
    return float(abs(np.exp(1j * np.asarray(g)).mean())) if len(g) else float("nan")
