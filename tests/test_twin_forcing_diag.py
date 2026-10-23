from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from hydrophysics.twin.forcing_diag import annual_anomaly, partial_corr  # noqa: E402


def test_partial_corr_removes_the_shared_driver():
    rng = np.random.default_rng(0)
    z = pd.Series(rng.normal(size=40))
    x = 2 * z + rng.normal(scale=0.1, size=40)
    y = -3 * z + rng.normal(scale=0.1, size=40)
    r_raw = np.corrcoef(x, y)[0, 1]
    r, p, n = partial_corr(pd.Series(x), pd.Series(y), z)
    assert n == 40 and r_raw < -0.9 and abs(r) < 0.4


def test_partial_corr_needs_six_years():
    s = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    r, p, n = partial_corr(s, s, s)
    assert n == 5 and np.isnan(r)


def test_annual_anomaly_groups_and_support():
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    obs = np.zeros((3, 36))
    obs[0, 12:24] = 1.0                     # well 0 sits one metre high in 2021
    clim = np.zeros_like(obs)
    groups = {"a": np.array([True, True, False]), "b": np.array([False, False, True])}
    table = annual_anomaly(obs, clim, dates, groups, min_values=10)
    assert list(table.index) == [2020, 2021, 2022]
    assert np.isclose(table.loc[2021, "a"], 0.5) and table.loc[2020, "a"] == 0
    assert table.loc[2021, "b"] == 0
    thin = annual_anomaly(obs, clim, dates, groups, min_values=30)
    assert np.isnan(thin.loc[2021, "b"])    # one well, twelve months: below support
