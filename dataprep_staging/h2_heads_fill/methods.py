"""Monthly gap-filling methods and baselines for WP H2.

Ported from the head gap-fill pilot (``results/dataprep/_pilots/head_gapfill_pilot/lib.py``,
design-time evidence; plan WP H2 "the pilot's lib.py is a working reference"). The numerical
logic of ``climatology``, ``ar1_params``, ``ar1_bridge``, ``runs``, ``m0_clim``,
``m0_clim_ar``, ``m1_neighbour``, ``m6_idw_anom``, ``_standardize``, ``_soft_impute``,
``_block_cv_mask``, ``m2_softimpute_tb`` and ``m4_gp`` is the pilot's, unchanged except that
constants come from :mod:`config`, the GP runs in-process by default (CPU-slot budget), wells
without coordinates never become spatial neighbours, and ``m1_neighbour`` also returns the
donor ids. ``port_check.py`` verifies that the port reproduces the pilot functions on the L1
matrix. New here: ``backfill`` (the twin's ``calibrate_flow.prepare_series``), the gap
classes and the calibrated router (:func:`router_native`, :func:`classify_cells`).

Every method takes ``Y`` (W, T) with NaN for missing and returns (mean, sd); sd is None when
the method has no native uncertainty. Month index 0 must be a January (true for every matrix
of this WP: they start in 2012-01).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from h2_heads_fill import config as C


# ----------------------------------------------------------------------------- helpers
def climatology(Y, shrink=C.CLIM_SHRINK):
    """Well mean + month-of-year anomaly (shrunk toward 0 when few observations)."""
    T = Y.shape[1]
    moy = np.arange(T) % 12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(Y, 1)
    Cm = np.zeros((Y.shape[0], 12))
    for k in range(12):
        sub = Y[:, moy == k] - mu[:, None]
        n = np.isfinite(sub).sum(1)
        s = np.nansum(sub, 1)
        Cm[:, k] = s / (n + shrink)
    Cm -= Cm.mean(1, keepdims=True)
    return mu[:, None] + Cm[:, moy]


def ar1_params(a):
    """phi, stationary variance from a 1-D series with NaN (pairs of consecutive obs)."""
    x0, x1 = a[:-1], a[1:]
    ok = np.isfinite(x0) & np.isfinite(x1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanvar(a)
    if ok.sum() < 6 or not v > 0:
        return 0.0, (v if np.isfinite(v) else 0.0)
    phi = np.sum(x0[ok] * x1[ok]) / np.sqrt(np.sum(x0[ok] ** 2) * np.sum(x1[ok] ** 2))
    return float(np.clip(phi, 0.0, 0.995)), float(v)


def ar1_bridge(a, phi, var):
    """Fill NaN in anomaly series a with the AR(1) conditional mean/var given the nearest
    observed neighbours on each side (exact for a stationary AR(1))."""
    T = len(a)
    mean = a.copy()
    sd = np.zeros(T)
    obs = np.where(np.isfinite(a))[0]
    if len(obs) == 0:
        return np.zeros(T), np.full(T, np.sqrt(max(var, 0)))
    for t in np.where(~np.isfinite(a))[0]:
        left = obs[obs < t]
        right = obs[obs > t]
        if len(left) and len(right):
            lo, r = left[-1], right[0]
            d1, d2 = t - lo, r - t
            den = 1 - phi ** (2 * (d1 + d2))
            if den < 1e-9:
                w = d1 / (d1 + d2)
                mean[t] = (1 - w) * a[lo] + w * a[r]
                sd[t] = np.sqrt(var * (1 - phi**2) * d1 * d2 / (d1 + d2) + 1e-12)
                continue
            mean[t] = (phi**d1 * (1 - phi ** (2 * d2)) * a[lo]
                       + phi**d2 * (1 - phi ** (2 * d1)) * a[r]) / den
            v = var * (1 - phi ** (2 * d1)) * (1 - phi ** (2 * d2)) / den
            sd[t] = np.sqrt(max(v, 0))
        else:
            e = left[-1] if len(left) else right[0]
            d = abs(t - e)
            mean[t] = phi**d * a[e]
            sd[t] = np.sqrt(max(var * (1 - phi ** (2 * d)), 0))
    return mean, sd


def runs(mask):
    m = np.concatenate([[0], np.asarray(mask).astype(int), [0]])
    d = np.diff(m)
    st = np.where(d == 1)[0]
    en = np.where(d == -1)[0]
    return list(zip(st, en, strict=True))


# ----------------------------------------------------------------------------- baselines
def backfill(Y, **_):
    """The twin's backfill (``calibrate_flow.prepare_series``, backfill=True): linear
    interpolation across interior gaps and constant fill of the edges."""
    out = Y.copy()
    for i in range(Y.shape[0]):
        if np.isfinite(Y[i]).any() and not np.isfinite(Y[i]).all():
            out[i] = pd.Series(Y[i]).interpolate(limit_direction="both").to_numpy()
    return out, None


def m0_clim(Y, **_):
    Cl = climatology(Y)
    res = Y - Cl
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(res, 1)
    return np.where(np.isfinite(Y), Y, Cl), np.broadcast_to(sd[:, None], Y.shape).copy()


def m0_clim_ar(Y, **_):
    Cl = climatology(Y)
    A = Y - Cl
    mean = np.zeros_like(Y)
    sd = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        phi, v = ar1_params(A[i])
        m, s = ar1_bridge(A[i], phi, v)
        mean[i], sd[i] = Cl[i] + m, s
    return mean, sd


def _distance_km(xy):
    with np.errstate(invalid="ignore"):
        D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) / 1e3
    return np.where(np.isfinite(D), D, np.inf)


def m6_idw_anom(Y, xy=None, layers=None, max_km=C.IDW_MAX_KM, **_):
    """Spatial IDW of climatology anomalies from same-layer wells observed that month."""
    Cl = climatology(Y)
    A = Y - Cl
    D = _distance_km(xy)
    same = layers[:, None] == layers[None, :]
    Wt = np.where((D <= max_km) & same & (D > 0.05), 1.0 / np.maximum(D, 0.5) ** 2, 0.0)
    obs = np.isfinite(A)
    num = Wt @ np.where(obs, A, 0.0)
    den = Wt @ obs.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = np.where(den > 0, num / den, 0.0)
    return np.where(obs, Y, Cl + est), None


# ----------------------------------------------------------------------------- M1
def m1_neighbour(Y, xy=None, k=C.NEIGH_K, min_overlap=C.NEIGH_MIN_OVERLAP,
                 min_corr=C.NEIGH_MIN_CORR, max_km=C.NEIGH_MAX_KM, alpha=C.NEIGH_ALPHA, **_):
    """Per gap: ridge regression on the k most correlated wells observed at every gap month
    (within max_km), then AR(1) bridge of the regression residuals, with a
    parameter-uncertainty term. Falls back to the climatology+AR(1) bridge where no
    predictor qualifies (``used`` = 0 there)."""
    W, T = Y.shape
    base_m, base_s = m0_clim_ar(Y)
    mean, sd = base_m.copy(), base_s.copy()
    obs = np.isfinite(Y)
    Z = np.where(obs, Y, 0.0)
    n = obs.astype(float) @ obs.T.astype(float)
    sx = Z @ obs.T
    sxx = (Z**2) @ obs.T
    sxy = Z @ Z.T
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sx.T / n
        vx = sxx - sx**2 / n
        corr = cov / np.sqrt(vx * vx.T)
    corr[n < min_overlap] = np.nan
    np.fill_diagonal(corr, np.nan)
    if xy is not None:
        D = _distance_km(xy)
        corr[~(D <= max_km)] = np.nan
    used = np.zeros((W, T), dtype=np.int8)
    for i in range(W):
        if obs[i].sum() < min_overlap:
            continue
        for s0, e0 in runs(~obs[i]):
            gap = np.arange(s0, e0)
            cand = np.where(obs[:, gap].all(1) & (np.nan_to_num(corr[i], nan=-1) >= min_corr))[0]
            if len(cand) == 0:
                continue
            cand = cand[np.argsort(-corr[i, cand])][:k]
            rows = obs[i] & obs[cand].all(0)
            if rows.sum() < min_overlap:
                cand = cand[:1]
                rows = obs[i] & obs[cand].all(0)
                if rows.sum() < min_overlap:
                    continue
            X = Y[cand][:, rows].T
            mx, sx_ = X.mean(0), X.std(0) + 1e-9
            Xs = (X - mx) / sx_
            y = Y[i, rows]
            my = y.mean()
            A = Xs.T @ Xs + alpha * len(y) * np.eye(len(cand))
            beta = np.linalg.solve(A, Xs.T @ (y - my))
            fit_rows = my + Xs @ beta
            dof = max(len(y) - len(cand) - 1, 1)
            s2 = np.sum((y - fit_rows) ** 2) / dof
            avail = obs[cand].all(0)
            pred_all = np.full(T, np.nan)
            pred_all[avail] = my + ((Y[cand][:, avail].T - mx) / sx_) @ beta
            res = np.full(T, np.nan)
            okr = avail & obs[i]
            res[okr] = Y[i, okr] - pred_all[okr]
            phi, v = ar1_params(res)
            v = max(v, s2)
            rm, rs = ar1_bridge(res, phi, v)
            Ainv = np.linalg.inv(A)
            xg = (Y[cand][:, gap].T - mx) / sx_
            par = np.einsum("ij,jk,ik->i", xg, Ainv, xg) * s2 + s2 / len(y)
            mean[i, gap] = pred_all[gap] + rm[gap]
            sd[i, gap] = np.sqrt(rs[gap] ** 2 + par)
            used[i, gap] = 1
    return mean, sd, used


# ----------------------------------------------------------------------------- M2
def _standardize(Y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(Y, 1, keepdims=True)
        sd = np.nanstd(Y, 1, keepdims=True) + 1e-9
    return (Y - mu) / sd, mu, sd


def _soft_impute(X, lam, rank=None, iters=200, tol=1e-5):
    obs = np.isfinite(X)
    Z = np.zeros_like(X)
    Xf = np.where(obs, X, 0.0)
    for _ in range(iters):
        M = np.where(obs, Xf, Z)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        if rank is None:
            s = np.maximum(s - lam, 0)
        else:
            s[rank:] = 0
        Zn = (U * s) @ Vt
        if np.linalg.norm(Zn - Z) / (np.linalg.norm(Z) + 1e-9) < tol:
            Z = Zn
            break
        Z = Zn
    return Z


def _block_cv_mask(obs, frac=C.SI_CV_FRAC, block=C.SI_CV_BLOCK, rng=None):
    rng = rng or np.random.default_rng(C.SI_CV_SEED)
    W, T = obs.shape
    m = np.zeros_like(obs)
    target = frac * obs.sum()
    while m.sum() < target:
        i = rng.integers(W)
        t = rng.integers(0, T - block)
        m[i, t:t + block] = True
    return m & obs


def m2_softimpute_tb(Y, grid=C.SI_GRID, iters=C.SI_ITERS, info=None):
    """softImpute (lambda by 6-month block CV) + temporal bridge of latent factor scores for
    months with no observation anywhere + AR(1) residual bridge. sd from block-CV
    residuals (pilot)."""
    X, mu, sdw = _standardize(Y)
    o = np.isfinite(X)
    smax = np.linalg.svd(np.where(o, X, 0.0), compute_uv=False)[0]
    cvm = _block_cv_mask(o)
    Xtr = np.where(cvm, np.nan, X)
    errs = [np.mean((_soft_impute(Xtr, f * smax, iters=iters)[cvm] - X[cvm]) ** 2)
            for f in grid]
    best = int(np.argmin(errs))
    lam = grid[best] * smax
    cv_res = np.where(cvm, X - _soft_impute(Xtr, lam, iters=iters), np.nan)
    Z = _soft_impute(X, lam, iters=iters)
    U, s, Vt = np.linalg.svd(np.where(o, X, Z), full_matrices=False)
    k = min(int((s > 1e-6 * s[0]).sum()), 30)
    V = Vt[:k].T * s[:k]
    empty = ~o.any(0)
    if empty.any():
        Vn = V.copy()
        Vn[empty] = np.nan
        Vf, _ = m0_clim_ar(Vn.T)
        Z[:, empty] = (U[:, :k] @ Vf)[:, empty]
    pooled = np.nanvar(cv_res)
    res = np.where(o, X - Z, np.nan)
    mean = np.zeros_like(X)
    sd = np.zeros_like(X)
    for i in range(X.shape[0]):
        phi, v = ar1_params(res[i])
        vcv = np.nanmean(cv_res[i] ** 2) if np.isfinite(cv_res[i]).sum() >= 3 else pooled
        m, sdi = ar1_bridge(res[i], phi, max(v, vcv))
        mean[i], sd[i] = Z[i] + m, sdi
    if info is not None:
        info.update({"lambda_frac": float(grid[best]), "lambda": float(lam),
                     "cv_mse": dict(zip(map(float, grid), map(float, errs), strict=True))})
    return mu + mean * sdw, sd * sdw


# ----------------------------------------------------------------------------- M4 GP
def _gp_one(y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern, WhiteKernel
    from sklearn.gaussian_process.kernels import ConstantKernel as Ck
    t = np.arange(len(y), dtype=float)[:, None]
    ok = np.isfinite(y)
    k = (Ck(1.0, (1e-3, 1e2)) * RBF(60.0, (6.0, 2000.0))
         + Ck(0.3, (1e-4, 1e1)) * ExpSineSquared(1.0, 12.0, length_scale_bounds=(0.3, 10.0),
                                                 periodicity_bounds="fixed")
         * RBF(120.0, (24.0, 2000.0))
         + Ck(0.1, (1e-4, 1e1)) * Matern(3.0, (0.5, 36.0), nu=0.5)
         + WhiteKernel(0.01, (1e-5, 1.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1,
                                      random_state=0)
        gp.fit(t[ok], y[ok])
        m, s = gp.predict(t, return_std=True)
    return m, s


def m4_gp(Y, rows=None, n_jobs=None, **_):
    """Per-well GP (RBF + seasonal + exponential + white noise; pilot kernel) for wells with
    >= GP_MIN_OBS observed months; other rows keep the climatology fill."""
    mean, sd = m0_clim(Y)
    sd = sd.copy()
    rows = range(Y.shape[0]) if rows is None else rows
    rows = [i for i in rows if np.isfinite(Y[i]).sum() >= C.GP_MIN_OBS]
    n_jobs = int(os.environ.get("H2_GP_JOBS", "1")) if n_jobs is None else n_jobs
    if n_jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(_gp_one)(Y[i]) for i in rows)
    else:
        res = [_gp_one(Y[i]) for i in rows]
    for i, (m, s) in zip(rows, res, strict=True):
        mean[i], sd[i] = m, s
    return mean, sd


# ----------------------------------------------------------------------------- gap classes
def classify_cells(obs, thin_frac=C.THIN_FRAC):
    """Gap class of every missing cell of the input matrix ``obs`` (W, T), plus run length.

    Precedence: ``no_obs_month`` / ``no_obs_run2`` (no well observed in the month; a run of
    one such month / of two or more) > ``lead`` / ``trail`` (before the first / after the last
    observation of the well) > ``thin`` (fewer than ``thin_frac`` x the median number of
    observed wells per month) > internal run length (``int_le2`` 1-2, ``int_3_5``,
    ``int_6_11``, ``int_12_23``, ``int_ge24`` months). Observed cells get ''. Also returns
    the run length and the distance to the nearest observation of the well (edge distance).
    The split of the network-outage class by run length and of the plan's 3-23-month class
    at 6 and 12 months were made at design time (first design pass, realizations r <= 4: one
    pooled factor over-covered 1-month and under-covered 2-month outages, and over-covered
    3-month and under-covered 12-month gaps)."""
    W, T = obs.shape
    cls = np.full((W, T), "", dtype=object)
    runlen = np.zeros((W, T), dtype=int)
    edge = np.zeros((W, T), dtype=int)
    count = obs.sum(0)
    med = np.median(count[count > 0]) if (count > 0).any() else 0
    thin_m = (count > 0) & (count < thin_frac * med)
    empty = count == 0
    for i in range(W):
        idx = np.where(obs[i])[0]
        for s0, e0 in runs(~obs[i]):
            L = e0 - s0
            runlen[i, s0:e0] = L
            tt = np.arange(s0, e0)
            if len(idx) == 0:
                cls[i, s0:e0] = "never"
                continue
            if s0 == 0:
                cls[i, s0:e0] = "lead"
                edge[i, s0:e0] = idx[0] - tt
            elif e0 == T:
                cls[i, s0:e0] = "trail"
                edge[i, s0:e0] = tt - idx[-1]
            else:
                c = ("int_le2" if L <= 2 else "int_3_5" if L <= 5 else "int_6_11" if L <= 11
                     else "int_12_23" if L <= 23 else "int_ge24")
                cls[i, s0:e0] = c
                edge[i, s0:e0] = np.minimum(tt - s0 + 1, e0 - tt)
                th = thin_m[s0:e0]
                if th.any():
                    sub = cls[i, s0:e0]
                    sub[th] = "thin"
                    cls[i, s0:e0] = sub
    elen = np.zeros(T, dtype=int)
    for s0, e0 in runs(empty):
        elen[s0:e0] = e0 - s0
    for i in range(W):
        sel = empty & ~obs[i] & (cls[i] != "never")
        cls[i, sel & (elen == 1)] = "no_obs_month"
        cls[i, sel & (elen >= 2)] = "no_obs_run2"
    return cls, runlen, edge


def tested_envelope(obs_row, min_obs=C.MIN_OBS_MONTHS, max_lead=C.MAX_LEAD,
                    max_trail=C.MAX_TRAIL, max_internal=C.MAX_INTERNAL_RUN):
    """True where a missing cell lies inside the tested envelope (plan step 4; trailing
    edge tightened to the tested 48 months)."""
    T = len(obs_row)
    ok = np.zeros(T, bool)
    if obs_row.sum() < min_obs:
        return ok
    idx = np.where(obs_row)[0]
    for s0, e0 in runs(~obs_row):
        if s0 == 0:
            ok[s0:e0] = (idx[0] - np.arange(s0, e0)) <= max_lead
        elif e0 == T:
            ok[s0:e0] = (np.arange(s0, e0) - idx[-1]) <= max_trail
        else:
            ok[s0:e0] = (e0 - s0) <= max_internal
    return ok


# ----------------------------------------------------------------------------- router
def router_native(Y, xy, run_gp=True, info=None, gp_rows=None, components=None):
    """The recommended monthly fill before calibration (plan step 2): precision-weighted
    (1/sd^2) neighbour ridge + softImpute where any well is observed in the month (where no
    neighbour qualifies, softImpute alone); per-well GP in months with no observation
    anywhere. Returns ``(mean, native_sd, method)``; observed cells keep Y with sd 0.
    ``gp_rows`` restricts the GP to those wells (the GP of one well ignores the others, so
    this only saves time); ``components`` (a dict) receives the component fills."""
    mn, sn, used = m1_neighbour(Y, xy=xy)
    ms, ss = m2_softimpute_tb(Y, info=info)
    w1, w2 = 1 / np.maximum(sn, 1e-3) ** 2, 1 / np.maximum(ss, 1e-3) ** 2
    mean = np.where(used == 1, (w1 * mn + w2 * ms) / (w1 + w2), ms)
    sd = np.where(used == 1, (w1 * sn + w2 * ss) / (w1 + w2), ss)
    meth = np.where(used == 1, "ens_neigh_softimpute", "softimpute_tb").astype(object)
    obs = np.isfinite(Y)
    empty = ~obs.any(0)
    mg = sg = None
    if empty.any() and run_gp:
        rows = [i for i in (range(len(Y)) if gp_rows is None else gp_rows)
                if obs[i].sum() >= C.GP_MIN_OBS]
        mg, sg = m4_gp(Y, rows=rows)
        mean[:, empty] = mg[:, empty]
        sd[:, empty] = sg[:, empty]
        meth[:, empty] = "gp"
    if components is not None:
        components.update({"neigh": (mn, sn), "neigh_used": used, "softimpute_tb": (ms, ss),
                           "gp": (mg, sg)})
    mean = np.where(obs, Y, mean)
    sd = np.where(obs, 0.0, sd)
    meth = np.where(obs, "observed", meth)
    return mean, sd, meth
