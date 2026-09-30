"""Causal scoring and rainfall time/unit checks for forcing attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from hydrophysics.twin.forcing_experiment import (
    advance_before_update,
    align_weather,
    paired_metrics,
    rain_daily,
)


def test_weather_alignment_does_not_learn_from_check_or_future_years():
    dates = pd.date_range("2012-01-01", periods=132, freq="MS")
    reference = np.full((2, 132), 2.)
    historical = np.full((2, 132), 4.)
    future_dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    future = np.full((2, 12), 8.)
    aligned, ratio = align_weather(reference, historical, future, dates, future_dates)
    np.testing.assert_allclose(aligned, 4.)
    reference[:, -12:] = 999.
    historical[:, -12:] = 9999.
    _, changed_ratio = align_weather(reference, historical, future*100, dates, future_dates)
    np.testing.assert_array_equal(changed_ratio, ratio)


def test_rain_uses_midnight_accumulation_once_on_previous_local_day():
    dates = pd.date_range("2023-06-01 00:10", periods=144, freq="10min",
                          tz="Asia/Taipei")
    frame = pd.DataFrame({"datetime": dates.tz_convert("UTC"),
                          "Past24hr": 30.0, "Past10Min": 1.0})
    daily = rain_daily(frame)
    assert daily.loc["2023-06-01"] == 30.0  # not 144*30 or 144*1
    assert "2023-06-02" not in daily.index


def test_rain_fallback_requires_complete_day_and_decodes_dry_codes():
    dates = pd.date_range("2023-06-01 00:10", periods=288, freq="10min", tz="Asia/Taipei")
    frame = pd.DataFrame({"datetime": dates, "Past24hr": -99., "Past10Min": -998.})
    frame.loc[0, "Past10Min"] = 12.
    frame.loc[200, "Past10Min"] = -99.
    daily = rain_daily(frame)
    assert daily.loc["2023-06-01"] == 12.
    assert np.isnan(daily.loc["2023-06-02"])


def test_future_observation_cannot_change_its_own_prediction():
    def simulate(observations):
        def advance(state, month):
            return state + 1

        def update(state, month):
            state.fill_(observations[month])  # even an in-place updater cannot leak
            return state

        return advance_before_update(torch.tensor([0.]), len(observations), advance, update)

    original = simulate([10., 20., 30.])
    changed = simulate([10., 999., 30.])
    torch.testing.assert_close(original, torch.tensor([[1., 11., 21.]]))
    torch.testing.assert_close(changed[..., :2], original[..., :2])
    assert changed[0, 2] == 1000.


def test_metrics_use_same_observations_and_equal_well_weights():
    observed = np.zeros((2, 4))
    predictions = {"a": np.array([[1., 1., 1., 1.], [3., 3., np.nan, np.nan]]),
                   "b": np.array([[2., 2., 2., np.nan], [0., 0., 0., 0.]])}
    report = paired_metrics(observed, predictions, minimum_months=2)
    assert report["well_months"] == 5
    assert report["metrics"]["a"]["rmse_m"] == pytest.approx(np.sqrt(5))
    assert report["metrics"]["b"]["rmse_m"] == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="enough"):
        paired_metrics(observed, predictions, minimum_months=4)
