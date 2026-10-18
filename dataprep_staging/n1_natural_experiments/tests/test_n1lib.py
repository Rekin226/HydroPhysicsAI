"""CPU unit tests of the N1 scoring rules (synthetic data only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from n1_natural_experiments import n1lib as L  # noqa: E402


def _calendar():
    t = np.arange(12 * 11, dtype=float)              # 2012-01 .. 2022-12
    year = 2012 + (t // 12).astype(int)
    month = (t % 12).astype(int) + 1
    return t, year, month


def _series(t, month, a=10.0, b=0.3, c=1.5):
    return a + b * (t - 60) / 120.0 + c * np.cos(2 * np.pi * month / 12)


def test_fit_recovers_trend_and_season():
    t, year, month = _calendar()
    y = _series(t, month)
    ref = np.isin(year, [2013, 2014, 2016, 2017, 2018, 2019, 2022])
    f = L.fit_reference(t, month, y, ref, harmonics=2)
    assert f is not None
    assert np.allclose(L.predict(f, t, month), y, atol=1e-9)


def test_event_anomaly_and_datum_invariance():
    t, year, month = _calendar()
    base = _series(t, month)
    event = (year == 2021) & (month <= 5)
    obs = base.copy()
    obs[event] -= 2.0                                    # a 2 m drawdown in Jan-May 2021
    ref = np.isin(year, [2013, 2014, 2016, 2017, 2018, 2019, 2022])
    model = base + 7.0                                   # datum offset, no response
    model_resp = base - 50.0
    model_resp[event] -= 2.0                             # exact response, other datum
    a = L.anomalies(t, month, obs, {"M0": model, "M1": model_resp}, ref, event)
    assert np.allclose(a["obs"], -2.0)
    assert np.allclose(a["M0"], 0.0, atol=1e-9)
    assert np.allclose(a["M1"], -2.0, atol=1e-9)


def test_missing_months_follow_the_observation():
    t, year, month = _calendar()
    obs = _series(t, month)
    obs[year == 2016] = np.nan
    ref = np.isin(year, [2013, 2014, 2016, 2017, 2018, 2019, 2022])
    event = (year == 2015) & (month <= 5)
    a = L.anomalies(t, month, obs, {"M": _series(t, month) + 1.0}, ref, event)
    assert len(a["obs"]) == 5 and np.allclose(a["M"], 0.0, atol=1e-9)


def test_bootstrap_verdicts():
    rng = np.random.default_rng(0)
    sites = np.repeat(np.arange(12), 2)
    e_good = 0.2 + 0.05 * rng.random(24)
    e_bad = 1.0 + 0.05 * rng.random(24)
    r = L.paired_site_bootstrap(e_good, e_bad, sites, B=500, seed=1)
    assert r["verdict"] == "model_better" and r["ci"][1] < 0
    r = L.paired_site_bootstrap(e_bad, e_good, sites, B=500, seed=1)
    assert r["verdict"] == "baseline_better"
    r = L.paired_site_bootstrap(e_good, e_good + 1e-12, sites, B=200, seed=1)
    assert r["verdict"] in ("indistinguishable", "model_better")
    r = L.paired_site_bootstrap(e_good[:4], e_bad[:4], sites[:4], B=200, seed=1)
    assert r["verdict"] == "not_evaluable"


def test_step_and_trend():
    t, year, month = _calendar()
    y = 5.0 + 0.0 * t
    y[year >= 2019] += 1.0
    pre = np.isin(year, [2017, 2018]) & (month >= 6)
    post = np.isin(year, [2019, 2020]) & (month >= 6)
    assert abs(L.step_delta(t, y, pre, post, 8) - 1.0) < 1e-12
    trend = 3.0 + 0.6 * (t - 60) / 120.0
    f = L.fit_reference(t, month, trend, np.isin(year, [2013, 2014, 2016, 2017, 2018]), 2)
    d = L.trend_delta(f, t, pre, post, trend)
    assert abs(d - 0.6 * 24 / 120.0) < 1e-9


def test_response_slope():
    rng = np.random.default_rng(3)
    x = rng.normal(size=30)
    y = 0.5 * x
    r = L.response_slope(y, x, np.arange(30), B=200, seed=2)
    assert abs(r["slope"] - 0.5) < 1e-12 and r["sign_agreement"] == 1.0
