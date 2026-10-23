"""Scoring rules of the N1 response tests (registered; hashed by the pre-registrations).

Event anomaly (tests T1, T2, G1-G3 and the consistency checks)
    For a well, fit ``h(t) = a + b (t - t0) + sum_k c_k cos(2 pi k m / 12) + d_k sin(...)``
    (k = 1..``harmonics``; t in months, m the calendar month) by least squares on the
    *reference months* where the observation exists; the observed anomaly of an event month
    is the observation minus that fit. A model series is treated identically, on the same
    months (the model's own fit), so a constant datum cancels. The climatology baseline CL
    predicts a zero anomaly; the zero-response baseline ZR is the model run with the
    intervention's forcing replaced by its reference-year mean.

Per-well error ``e_X(w) = sqrt(mean_m (a_X - a_obs)^2)`` over the observed event months;
equal-well score ``RMSE_X = sqrt(mean_w e_X(w)^2)``. Comparison ``D = RMSE_M - RMSE_X`` with a
paired site-cluster bootstrap (sites resampled with replacement): the verdict is
``model_better`` if the upper bound of the ``CI_LEVEL`` percentile interval is below 0,
``baseline_better`` if the lower bound is above 0, ``indistinguishable`` otherwise, and
``not_evaluable`` with fewer than ``min_sites`` sites.

Step change (test T3): ``delta = mean(post months) - mean(pre months)`` per well, the model on
the same months; baselines CL0 (``delta = 0``) and CL-trend (the trend slope of the well's
reference fit on its own pre-step reference years, times the difference of the mean times).
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------- design
def design(t: np.ndarray, month: np.ndarray, harmonics: int, t0: float) -> np.ndarray:
    cols = [np.ones_like(t, dtype=float), (t - t0) / 120.0]
    for k in range(1, harmonics + 1):
        w = 2.0 * math.pi * k * month / 12.0
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, axis=1)


def fit_reference(t: np.ndarray, month: np.ndarray, y: np.ndarray, ref: np.ndarray,
                  harmonics: int = 2) -> dict | None:
    """Least-squares trend + harmonics on ``ref`` months where ``y`` is finite. Returns
    coefficients and the t0 used, or None if fewer rows than 2x the parameters."""
    m = ref & np.isfinite(y)
    p = 2 + 2 * harmonics
    if m.sum() < 2 * p:
        return None
    t0 = float(np.mean(t[m]))
    X = design(t[m], month[m], harmonics, t0)
    beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    return {"beta": beta, "t0": t0, "harmonics": harmonics, "n": int(m.sum())}


def predict(fit: dict, t: np.ndarray, month: np.ndarray) -> np.ndarray:
    return design(t, month, fit["harmonics"], fit["t0"]) @ fit["beta"]


def anomalies(t, month, obs, model_series: dict, ref, event, harmonics=2) -> dict | None:
    """Event-month anomalies of the observation and of each model series (dict name -> array
    aligned with ``t``), each relative to its own reference fit on the months where the
    observation exists. Returns None when the observation has no reference fit."""
    fo = fit_reference(t, month, obs, ref, harmonics)
    if fo is None:
        return None
    ev = event & np.isfinite(obs)
    out = {"obs": obs[ev] - predict(fo, t[ev], month[ev]), "months": t[ev]}
    obs_mask = np.where(np.isfinite(obs), 1.0, np.nan)
    for name, s in model_series.items():
        fm = fit_reference(t, month, s * obs_mask, ref, harmonics)
        out[name] = s[ev] - predict(fm, t[ev], month[ev])
    return out


# ----------------------------------------------------------------------------- scores
def per_well_error(a_pred: np.ndarray, a_obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a_pred) - np.asarray(a_obs)) ** 2)))


def equal_well_rmse(errors: np.ndarray) -> float:
    e = np.asarray(errors, float)
    return float(np.sqrt(np.mean(e ** 2)))


def paired_site_bootstrap(e_model: np.ndarray, e_base: np.ndarray, sites: np.ndarray, *,
                          B: int, seed: int, level: float = 0.90,
                          min_sites: int = 5) -> dict:
    """Paired site-cluster bootstrap of ``D = RMSE(e_model) - RMSE(e_base)`` (equal-well)."""
    e_model, e_base = np.asarray(e_model, float), np.asarray(e_base, float)
    sites = np.asarray(sites)
    uniq = np.unique(sites)
    out = {"n_wells": int(len(e_model)), "n_sites": int(len(uniq)),
           "rmse_model": equal_well_rmse(e_model) if len(e_model) else float("nan"),
           "rmse_base": equal_well_rmse(e_base) if len(e_base) else float("nan")}
    out["diff"] = out["rmse_model"] - out["rmse_base"]
    out["ratio"] = out["rmse_model"] / out["rmse_base"] if out["rmse_base"] > 0 else float("nan")
    if len(uniq) < min_sites:
        out.update({"ci": [float("nan"), float("nan")], "verdict": "not_evaluable"})
        return out
    idx = {s: np.flatnonzero(sites == s) for s in uniq}
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for b in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[s] for s in pick])
        d[b] = equal_well_rmse(e_model[rows]) - equal_well_rmse(e_base[rows])
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    out["ci"] = [float(lo), float(hi)]
    out["verdict"] = ("model_better" if hi < 0 else "baseline_better" if lo > 0
                      else "indistinguishable")
    return out


def response_slope(mean_obs: np.ndarray, mean_mod: np.ndarray, sites: np.ndarray, *, B: int,
                   seed: int, level: float = 0.90) -> dict:
    """Least-squares slope through the origin of the per-well mean observed anomaly on the
    mean modelled anomaly, with a site bootstrap interval (reported, never gating)."""
    x, y, s = np.asarray(mean_mod, float), np.asarray(mean_obs, float), np.asarray(sites)
    if len(x) < 2 or float(np.sum(x * x)) == 0.0:
        return {"slope": float("nan"), "ci": [float("nan")] * 2, "n": int(len(x))}
    slope = float(np.sum(x * y) / np.sum(x * x))
    uniq = np.unique(s)
    idx = {k: np.flatnonzero(s == k) for k in uniq}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        rows = np.concatenate([idx[k] for k in rng.choice(uniq, size=len(uniq), replace=True)])
        den = float(np.sum(x[rows] ** 2))
        if den > 0:
            bs.append(float(np.sum(x[rows] * y[rows]) / den))
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2]) if bs else (np.nan,) * 2
    return {"slope": slope, "ci": [float(lo), float(hi)], "n": int(len(x)),
            "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
            "corr": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")}


# ----------------------------------------------------------------------------- step
def step_delta(t, values, pre, post, min_months: int) -> float:
    a, b = pre & np.isfinite(values), post & np.isfinite(values)
    if a.sum() < min_months or b.sum() < min_months:
        return float("nan")
    return float(np.mean(values[b]) - np.mean(values[a]))


def trend_delta(fit: dict, t, pre, post, obs) -> float:
    """CL-trend prediction of the step: slope x (mean post time - mean pre time) on the
    observed months."""
    a, b = pre & np.isfinite(obs), post & np.isfinite(obs)
    slope_per_month = float(fit["beta"][1]) / 120.0
    return slope_per_month * (float(np.mean(t[b])) - float(np.mean(t[a])))
