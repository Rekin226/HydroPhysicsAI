from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.ring_holdout import continued_rate, r2_rmse, window_stats


def _series(rate_per_yr=0.02, start="2015-01-31", months=130, offset=1.0):
    idx = pd.date_range(start, periods=months, freq="ME")
    years = (idx - idx[0]).days / 365.25
    return pd.Series(offset + rate_per_yr * years, index=idx)


def test_window_stats_measures_displacement_from_reference_and_slope():
    s = _series(0.02)
    cum, rate, n = window_stats(s, "2022-12-01", "2023-01-01", "2025-03-31")
    assert n == 27
    assert rate == pytest.approx(0.02, rel=1e-6)
    assert cum == pytest.approx(0.02 * (pd.Timestamp("2025-03-31") - pd.Timestamp("2022-12-31")).days / 365.25, rel=1e-3)


def test_window_stats_requires_reference_and_support():
    s = _series(0.02, start="2023-06-30", months=24)       # no Dec-2022 reference
    cum, rate, n = window_stats(s, "2022-12-01", "2023-01-01", "2025-03-31")
    assert np.isnan(cum) and np.isnan(rate)
    s = _series(0.02)
    cum, rate, n = window_stats(s, "2022-12-01", "2023-01-01", "2023-06-30")
    assert n == 6 and np.isnan(cum)                        # below min_months


def test_continued_rate_uses_only_the_prior_window():
    s = _series(0.02)
    s.loc["2023-01-01":] = 100.0                           # the future must not matter
    assert continued_rate(s, "2017-01-01", "2022-12-31") == pytest.approx(0.02, rel=1e-6)
    assert np.isnan(continued_rate(s.iloc[:10], "2015-01-01", "2015-12-31"))


def test_r2_rmse_scores_and_handles_small_samples():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_rmse(obs, obs)["r2"] == pytest.approx(1.0)
    m = r2_rmse(np.full(4, obs.mean()), obs)
    assert m["r2"] == pytest.approx(0.0) and m["bias"] == pytest.approx(0.0)
    assert np.isnan(r2_rmse(obs[:2], obs[:2])["r2"])
    worse = r2_rmse(obs + 5.0, obs)
    assert worse["r2"] < 0 and worse["bias"] == pytest.approx(5.0)
