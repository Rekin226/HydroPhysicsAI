"""Space-time kriging of standardized anomalies (WP H2 baseline; plan H2 'Validation').

Anomalies are the head minus the well's shrunk month-of-year climatology (the pilot's
``climatology``), divided by the well's anomaly sd. Their space-time correlation is modelled
as separable exponential with a nugget,

    rho(h, u) = (1 - nugget) * exp(-h / a_s) * exp(-|u| / a_t)   for (h, u) != (0, 0),

fitted by weighted least squares to the empirical correlation of anomaly pairs binned by
distance (km) and lag (months), on the matrix being filled (so on the masked matrix in every
artificial-gap test: nothing masked is seen). Each missing cell is predicted by simple
kriging (mean 0) from its ``STK_NEIGHBOURS`` most correlated observed space-time cells within
``STK_MAX_KM`` and ``STK_MAX_LAG_MONTHS``; the kriging variance gives the native sd. All
layers share one correlation model (a baseline, not a product method).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares

from h2_heads_fill import config as C
from h2_heads_fill.methods import _distance_km, climatology


def standardized_anomalies(Y):
    Cl = climatology(Y)
    A = Y - Cl
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = np.nanstd(A, 1)
    s = np.where(np.isfinite(s) & (s > 1e-6), s, np.nan)
    return Cl, A / s[:, None], s


def empirical_correlation(Z, D, km_bins=C.STK_VARIO_KM_BINS, lags=C.STK_VARIO_LAGS):
    """Rows (h_mid, u, corr, n_pairs) of the binned empirical anomaly correlation."""
    O = np.isfinite(Z)
    Zf = np.where(O, Z, 0.0)
    Of = O.astype(float)
    T = Z.shape[1]
    out = []
    edges = np.asarray(km_bins, float)
    for u in lags:
        if u >= T:
            continue
        S = Zf[:, :T - u] @ Zf[:, u:].T
        N = Of[:, :T - u] @ Of[:, u:].T
        if u == 0:
            np.fill_diagonal(N, 0.0)
            np.fill_diagonal(S, 0.0)
        same = D <= 0.05
        sel = same & (N > 0)
        if sel.any() and u > 0:
            out.append((0.0, u, float(S[sel].sum() / N[sel].sum()), float(N[sel].sum())))
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            sel = (D > max(a, 0.05)) & (D <= b) & (N > 0)
            if sel.any():
                out.append((0.5 * (a + b), u, float(S[sel].sum() / N[sel].sum()),
                            float(N[sel].sum())))
    return np.array(out)


def fit_correlation(E):
    """(a_s km, a_t months, nugget) by weighted least squares on the empirical table."""
    h, u, c, n = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
    w = np.sqrt(n / n.max())

    def f(p):
        a_s, a_t, nug = np.exp(p[0]), np.exp(p[1]), 1 / (1 + np.exp(-p[2]))
        return w * ((1 - nug) * np.exp(-h / a_s) * np.exp(-u / a_t) - c)

    r = least_squares(f, x0=[np.log(10.0), np.log(3.0), 0.0], method="lm")
    a_s, a_t, nug = np.exp(r.x[0]), np.exp(r.x[1]), 1 / (1 + np.exp(-r.x[2]))
    return float(a_s), float(a_t), float(nug)


def stk_fill(Y, xy=None, targets=None, params=None, info=None, **_):
    """Space-time kriging fill. ``targets`` restricts the predicted missing cells (a boolean
    (W, T) mask; default every missing cell). Returns (mean, sd) with observed cells kept."""
    W, T = Y.shape
    Cl, Z, s = standardized_anomalies(Y)
    D = _distance_km(xy)
    if params is None:
        E = empirical_correlation(Z, D)
        params = fit_correlation(E)
    a_s, a_t, nug = params
    O = np.isfinite(Z)
    mean = np.where(np.isfinite(Y), Y, Cl)
    sd = np.where(np.isfinite(Y), 0.0, np.broadcast_to(s[:, None], Y.shape))
    sd = np.where(np.isfinite(sd), sd, np.nan)
    tmask = (~np.isfinite(Y)) if targets is None else (targets & ~np.isfinite(Y))
    near = [np.where(D[i] <= C.STK_MAX_KM)[0] for i in range(W)]
    L = C.STK_MAX_LAG_MONTHS
    K = C.STK_NEIGHBOURS
    for i in range(W):
        ts = np.where(tmask[i])[0]
        if len(ts) == 0 or not np.isfinite(s[i]):
            continue
        nb = near[i]
        for t in ts:
            lo, hi = max(t - L, 0), min(t + L + 1, T)
            sub = O[np.ix_(nb, np.arange(lo, hi))]
            jj, tt = np.nonzero(sub)
            if len(jj) == 0:
                continue
            wells = nb[jj]
            months = tt + lo
            c0 = (1 - nug) * np.exp(-D[i, wells] / a_s) * np.exp(-np.abs(months - t) / a_t)
            if len(c0) > K:
                top = np.argpartition(-c0, K - 1)[:K]
                wells, months, c0 = wells[top], months[top], c0[top]
            Dk = D[np.ix_(wells, wells)]
            Uk = np.abs(months[:, None] - months[None, :])
            Sig = (1 - nug) * np.exp(-Dk / a_s) * np.exp(-Uk / a_t)
            np.fill_diagonal(Sig, 1.0)
            try:
                lam = np.linalg.solve(Sig, c0)
            except np.linalg.LinAlgError:
                lam = np.linalg.lstsq(Sig, c0, rcond=None)[0]
            z = Z[wells, months]
            mean[i, t] = Cl[i, t] + s[i] * float(lam @ z)
            v = max(1.0 - float(lam @ c0), 1e-6)
            sd[i, t] = s[i] * np.sqrt(v)
    if info is not None:
        info.update({"a_s_km": a_s, "a_t_months": a_t, "nugget": nug})
    return mean, sd
