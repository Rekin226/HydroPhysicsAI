"""Downscaling models of WP X1: census in-grid county electricity from TPC county sales.

Notation: ``y_c(t)`` is the census series the hybrid layer uses for county c (GWh/month, the
in-grid sum of cells_total ``value_all``, recorded months only); ``x`` are features known
once TPC has published month t (d007012 county sectors, the Energy Administration national
agricultural total, fan-mean rain and ET0). No feature leads month t: month t can be
downscaled as soon as TPC posts it.

Candidates (fixed before any score was computed; hyperparameters copied, not tuned):

* ``clim``       month-of-year mean of y over the recorded training months from 2012 (the
                 fallback of the registered hybrid test);
* ``ratio``      TPC_ag_c(t) x r_c(moy), r = month-of-year mean of y / TPC_ag (P1's
                 transfer estimator);
* ``ridge``      log y(t) = mu_log(moy) + ridge on month-of-year anomalies of log features
                 (own-county ag at t, t-1, t-2; other-county ag at t; own-county
                 residential, services, industry at t; national ag at t), standardised,
                 alpha = 1 (the hybrid layer's convention);
* ``ridge_wx``   ``ridge`` plus rain (t, t-1) and ET0 (t) anomalies;
* ``gbm``        gradient-boosted trees (depth 2, 200 iterations, learning rate 0.05,
                 min 10 samples per leaf) on the ``ridge_wx`` anomaly features.

All TPC-based models train from 2016-01 (d007012's first month). Predictions are returned in
GWh; log models are back-transformed with the mean training residual variance (smearing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODELS = ("clim", "ratio", "ridge", "ridge_wx", "gbm", "ridge_wx_anchor12")
#: Post-hoc sensitivity (added after run 1 showed a 2026 Yunlin level drift of census / TPC):
#: ``ridge_wx`` rescaled by the geometric-mean ratio of observed to fitted values over the
#: last 12 recorded training months. Never a candidate for selection (gate_spec.CANDIDATES).
POST_HOC = ("ridge_wx_anchor12",)
TPC_START = pd.Timestamp("2016-01-01")
ALPHA = 1.0
OTHER = {"changhua": "yunlin", "yunlin": "changhua"}


def features(tpc: pd.DataFrame, nat: pd.Series, wx: pd.DataFrame, county: str,
             weather: bool) -> pd.DataFrame:
    o = OTHER[county]
    f = pd.DataFrame(index=tpc.index)
    lag = np.log(tpc[f"ag_{county}"])
    f["ag"] = lag
    f["ag_l1"] = lag.shift(1)
    f["ag_l2"] = lag.shift(2)
    f["ag_other"] = np.log(tpc[f"ag_{o}"])
    for s in ("res", "srv", "ind"):
        f[s] = np.log(tpc[f"{s}_{county}"])
    f["nat_ag"] = np.log(nat.reindex(f.index))
    if weather:
        w = wx.reindex(f.index)
        f["rain"] = w["rain"]
        f["rain_l1"] = w["rain"].shift(1)
        f["et0"] = w["et0"]
    return f


def _moy_mean(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.month).mean()


def _anom(df: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    m = means.reindex(df.index.month).to_numpy()
    return df - m


class Fitted:
    def __init__(self, name: str):
        self.name = name


def fit(name: str, y: pd.Series, tpc: pd.DataFrame, nat: pd.Series, wx: pd.DataFrame,
        county: str) -> Fitted:
    """Fit candidate ``name`` on the recorded months of ``y`` (index month starts)."""
    y = y.dropna()
    if name == "ridge_wx_anchor12":
        m = fit("ridge_wx", y, tpc, nat, wx, county)
        last = y[y.index >= TPC_START].iloc[-12:]
        p = predict(m, last.index, tpc, nat, wx)
        m.anchor = float(np.exp(np.mean(np.log(last / p))))
        m.name = name
        return m
    m = Fitted(name)
    m.county = county
    if name == "clim":
        m.clim = _moy_mean(y).reindex(range(1, 13))
        return m
    y = y[y.index >= TPC_START]
    if name == "ratio":
        t = tpc[f"ag_{county}"].reindex(y.index)
        m.r = _moy_mean(y / t).reindex(range(1, 13))
        return m
    f = features(tpc, nat, wx, county, weather=name in ("ridge_wx", "gbm"))
    X = f.reindex(y.index)
    ok = X.notna().all(axis=1)
    X, ly = X[ok], np.log(y[ok])
    m.cols = list(X.columns)
    m.mu_log = _moy_mean(ly).reindex(range(1, 13))
    m.fmeans = X.groupby(X.index.month).mean().reindex(range(1, 13))
    A = _anom(X, m.fmeans)
    a = ly - m.mu_log.reindex(ly.index.month).to_numpy()
    if name in ("ridge", "ridge_wx"):
        m.sd = A.std(ddof=0).replace(0, 1.0)
        Z = (A / m.sd).to_numpy()
        m.lo, m.hi = Z.min(0), Z.max(0)
        k = Z.shape[1]
        m.beta = np.linalg.solve(Z.T @ Z + ALPHA * np.eye(k), Z.T @ (a.to_numpy() - a.mean()))
        m.b0 = float(a.mean())
        res = a.to_numpy() - (m.b0 + Z @ m.beta)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        Z = A.copy()
        Z["moy"] = A.index.month
        m.gbm = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=10, random_state=0)
        m.gbm.fit(Z.to_numpy(), a.to_numpy())
        res = a.to_numpy() - m.gbm.predict(Z.to_numpy())
    m.smear = float(np.mean(res ** 2)) / 2.0
    m.train_resid_sd = float(np.std(res))
    return m


def predict(m: Fitted, idx: pd.DatetimeIndex, tpc: pd.DataFrame, nat: pd.Series,
            wx: pd.DataFrame) -> pd.Series:
    moy = idx.month
    if m.name == "clim":
        return pd.Series(m.clim.reindex(moy).to_numpy(), index=idx)
    if m.name == "ratio":
        t = tpc[f"ag_{m.county}"].reindex(idx).to_numpy()
        return pd.Series(t * m.r.reindex(moy).to_numpy(), index=idx)
    anchor = getattr(m, "anchor", 1.0)
    wxm = m.name in ("ridge_wx", "gbm", "ridge_wx_anchor12")
    f = features(tpc, nat, wx, m.county, weather=wxm)
    X = f.reindex(idx)[m.cols]
    A = _anom(X, m.fmeans)
    base = m.mu_log.reindex(moy).to_numpy()
    if m.name in ("ridge", "ridge_wx", "ridge_wx_anchor12"):
        Z = np.clip((A / m.sd).to_numpy(), m.lo, m.hi)
        a = m.b0 + Z @ m.beta
    else:
        Z = A.copy()
        Z["moy"] = moy
        a = m.gbm.predict(Z.to_numpy())
    return pd.Series(anchor * np.exp(base + a + m.smear), index=idx)


def shares(y_disc: pd.DataFrame, y_county: pd.Series) -> pd.DataFrame:
    """Month-of-year mean share of each disc in its county total (recorded months)."""
    ok = y_county.notna() & (y_county > 0)
    s = y_disc[ok].div(y_county[ok], axis=0)
    return s.groupby(s.index.month).mean().reindex(range(1, 13))
