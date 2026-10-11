from __future__ import annotations

import numpy as np
import pandas as pd

from hydrophysics.twin.hybrid import (
    candidates,
    lag1_rho,
    month_of_year_mean,
    persisted_anomaly,
    score_candidates,
)


def _ar1_wells(W=6, T=96, fit=72, rho=0.85, seed=3):
    rng = np.random.default_rng(seed)
    moy = np.arange(T) % 12
    season = 2.0 * np.sin(2 * np.pi * moy / 12)
    obs = np.empty((W, T))
    for w in range(W):
        a = np.zeros(T)
        for t in range(1, T):
            a[t] = rho * a[t - 1] + rng.normal(0, 0.6)
        obs[w] = 10 + w + season + a
    return obs, moy, fit


def test_lag1_rho_recovers_ar1_coefficient():
    obs, moy, fit = _ar1_wells(T=600, fit=600)
    anom = obs - month_of_year_mean(obs, np.isfinite(obs), fit, moy)
    rho, n = lag1_rho(anom, fit)
    assert (n == 599).all()
    assert np.allclose(rho, 0.85, atol=0.08)


def test_lag1_rho_needs_pairs_and_clips():
    anom = np.array([[1.0, -2.0, 3.0, np.nan, 1.0]])
    rho, n = lag1_rho(anom, 5)
    assert n[0] == 2 and np.isnan(rho[0])
    rho, _ = lag1_rho(anom, 5, min_pairs=1)
    assert 0.0 <= rho[0] <= 0.99


def test_persisted_anomaly_decays_and_respects_gap():
    anom = np.full((2, 10), np.nan)
    anom[0, :6] = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]      # last fitted month = origin
    anom[1, :2] = [1.0, 1.0]                          # last observation far before origin
    rho = np.array([0.5, 0.5])
    a, last = persisted_anomaly(anom, fit=6, rho=rho)
    assert last.tolist() == [5, -1]
    assert np.allclose(a[0, 5:9], [2.0, 1.0, 0.5, 0.25])
    assert (a[0, :5] == 0).all() and (a[1] == 0).all()


def test_stat_hybrid_beats_climatology_in_the_first_year_only_when_memory_exists():
    obs, moy, fit = _ar1_wells()
    T = obs.shape[1]
    predicted = np.full_like(obs, 10.0) + np.sin(2 * np.pi * moy / 12)[None, :]   # a weak model
    hist = obs.copy()
    hist[:, fit:] = np.nan
    cands, info = candidates(hist, predicted, fit, moy)
    assert np.isfinite(info["rho"]).all() and (info["rho"] > 0.5).all()
    assert np.allclose(cands["mix"], cands["clim"] + 0.5 * (info["anomaly_stat"] + info["anomaly_phys"]))
    dates = pd.date_range("2015-01-01", periods=T, freq="MS")
    new = np.where(np.arange(T) >= fit, obs, np.nan)
    scores, valid, keep = score_candidates(cands, new, fit, dates, min_months=12)
    first = str(dates[fit].year)
    assert scores["per_year"][first]["stat"] < scores["per_year"][first]["clim"]
    # with the last anomaly gone, stat is climatology exactly
    hist2 = hist.copy()
    hist2[:, fit - 6:fit] = np.nan
    cands2, info2 = candidates(hist2, predicted, fit, moy)
    assert (info2["last_fitted_index"] == -1).all()
    assert np.allclose(cands2["stat"], cands2["clim"])


def test_score_candidates_rejects_empty():
    cands = {"clim": np.zeros((1, 4))}
    new = np.full((1, 4), np.nan)
    try:
        score_candidates(cands, new, 2, pd.date_range("2020-01-01", periods=4, freq="MS"), 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
