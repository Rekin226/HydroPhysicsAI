"""Initial-condition head field (plan WP H2 technique step 7).

"The initial condition stays IDW unless kriging with an external drift or a GP field beats it
by leave-one-well-out on January 2012 heads." Candidates, on the January 2012 L1 monthly heads
(flag 0/1, any clean-day count, as the twin's month-0 field) of the 158 model wells:

* ``idw``  the twin's construction (``calibrate_flow._idw_initial_heads``): per-layer inverse
  distance squared over every well of the layer (all wells when the layer has none);
* ``ked``  kriging with an external drift: per layer (pooled across layers when a layer has
  fewer than ``MIN_LAYER`` wells) a drift ``a + b * dem`` (SRTM elevation of the well's model
  cell, the twin's ``--ground-elev dem`` field from ``results/twin/basemap.npz``) and an
  exponential residual covariance (sill, range, nugget by maximum likelihood on the training
  wells), universal-kriging equations;
* ``gp``   a GP field: the same per-layer drift fitted by least squares, and a Gaussian process
  (Matern 3/2 + white noise, scikit-learn, maximum likelihood) on the residuals of all layers.

Every parameter is refitted without the held-out well. The decision rule (registered) compares
the pooled leave-one-well-out RMSE.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

MIN_LAYER = 8


def idw_predict(xy_tr, v_tr, xy_te, power=2.0, eps=1e-6):
    d2 = ((xy_te[:, None, :] - xy_tr[None, :, :]) ** 2).sum(-1)
    w = 1.0 / (d2 ** (power / 2.0) + eps)
    return (w @ v_tr) / w.sum(1)


def idw_layered(xy, v, lay, xy_te, lay_te):
    out = np.empty(len(xy_te))
    for i, (p, k) in enumerate(zip(xy_te, lay_te, strict=True)):
        m = lay == k
        if m.sum() >= 1:
            out[i] = idw_predict(xy[m], v[m], p[None, :])[0]
        else:
            out[i] = idw_predict(xy, v, p[None, :])[0]
    return out


def _expcov(D, sill, rng, nug):
    return sill * np.exp(-D / rng) + nug * (D == 0)


def _nll(theta, D, r, X):
    sill, rng, nug = np.exp(theta)
    K = _expcov(D, sill, rng, nug) + 1e-8 * np.eye(len(r))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e12
    # restricted likelihood of the residuals after the drift (REML)
    Ki_X = np.linalg.solve(K, X)
    A = X.T @ Ki_X
    beta = np.linalg.solve(A, Ki_X.T @ r)
    e = r - X @ beta
    a = np.linalg.solve(L, e)
    sign, logdetA = np.linalg.slogdet(A)
    return 0.5 * (a @ a) + np.log(np.diag(L)).sum() + 0.5 * logdetA


def fit_ked(xy, v, X):
    D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) / 1e3
    var = np.var(v - X @ np.linalg.lstsq(X, v, rcond=None)[0]) + 1e-6
    best = None
    for r0 in (3.0, 10.0, 30.0):
        res = minimize(_nll, np.log([0.7 * var, r0, 0.3 * var]), args=(D, v, X),
                       method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(best.x)


def ked_predict(xy, v, X, xy_te, X_te, theta):
    sill, rng, nug = theta
    D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) / 1e3
    K = _expcov(D, sill, rng, nug) + 1e-8 * np.eye(len(v))
    D0 = np.hypot(xy_te[:, None, 0] - xy[None, :, 0], xy_te[:, None, 1] - xy[None, :, 1]) / 1e3
    k0 = sill * np.exp(-D0 / rng) + nug * (D0 == 0)
    n, p = X.shape
    A = np.zeros((n + p, n + p))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, X, X.T
    out, sd = np.empty(len(xy_te)), np.empty(len(xy_te))
    for i in range(len(xy_te)):
        b = np.concatenate([k0[i], X_te[i]])
        sol = np.linalg.solve(A, b)
        lam = sol[:n]
        out[i] = lam @ v
        sd[i] = np.sqrt(max(sill + nug - sol @ b, 1e-9))
    return out, sd


def ked_layered(xy, v, lay, dem, xy_te, lay_te, dem_te):
    out, sd = np.empty(len(xy_te)), np.empty(len(xy_te))
    for k in np.unique(lay_te):
        te = lay_te == k
        tr = lay == k
        if tr.sum() < MIN_LAYER:
            tr = np.ones_like(tr)
        X = np.column_stack([np.ones(tr.sum()), dem[tr]])
        th = fit_ked(xy[tr], v[tr], X)
        Xt = np.column_stack([np.ones(te.sum()), dem_te[te]])
        out[te], sd[te] = ked_predict(xy[tr], v[tr], X, xy_te[te], Xt, th)
    return out, sd


def gp_layered(xy, v, lay, dem, xy_te, lay_te, dem_te):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as Ck
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    drift_tr = np.empty(len(v))
    drift_te = np.empty(len(xy_te))
    for k in np.unique(np.concatenate([lay, lay_te])):
        tr = lay == k
        if tr.sum() < MIN_LAYER:
            tr = np.ones_like(tr)
        X = np.column_stack([np.ones(tr.sum()), dem[tr]])
        b = np.linalg.lstsq(X, v[tr], rcond=None)[0]
        drift_tr[lay == k] = b[0] + b[1] * dem[lay == k]
        drift_te[lay_te == k] = b[0] + b[1] * dem_te[lay_te == k]
    r = v - drift_tr
    kern = Ck(1.0, (1e-3, 1e3)) * Matern(10.0, (0.5, 200.0), nu=1.5) + WhiteKernel(
        0.1, (1e-4, 1e2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=2,
                                      random_state=0)
        gp.fit(xy / 1e3, r)
        m, s = gp.predict(xy_te / 1e3, return_std=True)
    return drift_te + m, s


def lowo(xy, v, lay, dem):
    """Leave-one-well-out predictions of the three candidates."""
    n = len(v)
    pred = {"idw": np.empty(n), "ked": np.empty(n), "gp": np.empty(n)}
    sd = {"ked": np.empty(n), "gp": np.empty(n)}
    for i in range(n):
        tr = np.arange(n) != i
        te = np.array([i])
        pred["idw"][i] = idw_layered(xy[tr], v[tr], lay[tr], xy[te], lay[te])[0]
        p, s = ked_layered(xy[tr], v[tr], lay[tr], dem[tr], xy[te], lay[te], dem[te])
        pred["ked"][i], sd["ked"][i] = p[0], s[0]
        p, s = gp_layered(xy[tr], v[tr], lay[tr], dem[tr], xy[te], lay[te], dem[te])
        pred["gp"][i], sd["gp"][i] = p[0], s[0]
    return pred, sd
