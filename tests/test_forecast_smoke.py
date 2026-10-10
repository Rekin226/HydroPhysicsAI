"""Smoke test for the forecasting model. Skipped when torch is absent.

Checks that fit -> forecast produces a correctly-shaped cube and that the horizon-wise
eval harness runs and returns finite skill numbers on the synthetic sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics import Config, load_dataset
from hydrophysics.sample import write_sample

torch = pytest.importorskip("torch")


@pytest.fixture()
def data(tmp_path):
    d = write_sample(tmp_path / "data", n_wells=4, seed=3)
    return load_dataset(Config(data_dir=d))


def test_fit_forecast_and_eval(data):
    from hydrophysics.forecast_eval import horizon_table
    from hydrophysics.models.forecast_lstm import GlobalForecastLSTM

    horizons = [1, 7]
    model = GlobalForecastLSTM(lookback=30, horizon=max(horizons), hidden=16,
                               epochs=2, device="cpu")
    model.fit(data)
    cube = model.forecast(data)
    assert cube.shape == (data.n_wells, data.n_days, max(horizons))

    table = horizon_table(data, cube, horizons, period="val")
    assert ("forecast_lstm" in table.index.get_level_values("model"))
    # every reported aggregate should be a real number
    assert np.isfinite(table["rmse_median"].to_numpy()).all()


def test_amp_flag_fits(data):
    """The mixed-precision flag must not break fit/forecast (it is a no-op off CUDA)."""
    from hydrophysics.models.forecast_lstm import GlobalForecastLSTM

    model = GlobalForecastLSTM(lookback=30, horizon=7, hidden=16, epochs=2,
                               amp=True, device="cpu")
    model.fit(data)
    cube = model.forecast(data)
    assert cube.shape == (data.n_wells, data.n_days, 7)
    assert np.isfinite(cube[np.isfinite(cube)]).all()


def test_probabilistic_forecast(data):
    from hydrophysics.forecast_eval import probabilistic_horizon_table
    from hydrophysics.models.forecast_lstm import GlobalForecastLSTM

    horizons = [1, 7]
    model = GlobalForecastLSTM(lookback=30, horizon=max(horizons), hidden=16,
                               epochs=2, probabilistic=True, device="cpu")
    model.fit(data)
    mean, sigma = model.forecast_dist(data)
    assert mean.shape == sigma.shape == (data.n_wells, data.n_days, max(horizons))
    assert np.nanmin(sigma) > 0   # predictive std is positive where defined

    table = probabilistic_horizon_table(data, mean, sigma, horizons, period="val")
    lstm = table.xs("forecast_lstm", level="model")
    assert np.isfinite(lstm["crps_median"].to_numpy()).all()
    assert ((lstm["picp90"] >= 0) & (lstm["picp90"] <= 1)).all()


def _short_record():
    from hydrophysics.data import GWData

    dates = pd.date_range("2020-01-01", periods=120)
    t = np.arange(120)
    return GWData.from_arrays(
        well_ids=["synthetic"], dates=dates, target=(5 + np.sin(t / 7))[None],
        rainfall=(1 + np.cos(t / 9))[None], upstream=np.ones((1, 120)),
        split_date=str(dates[70].date()))


def test_heldout_heads_and_forcing_cannot_change_fitted_network():
    import copy

    from hydrophysics.models.forecast_lstm import GlobalForecastLSTM

    data = _short_record()
    altered = copy.deepcopy(data)
    altered.target[:, data.val_mask] = 10000
    altered.rainfall[:, data.val_mask] = 10000
    altered.upstream[:, data.val_mask] = -10000
    first = GlobalForecastLSTM(lookback=10, horizon=7, hidden=4, epochs=1, device="cpu")
    second = GlobalForecastLSTM(lookback=10, horizon=7, hidden=4, epochs=1, device="cpu")
    first.fit(data)
    second.fit(altered)
    for key, value in first.net.state_dict().items():
        torch.testing.assert_close(value, second.net.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("probabilistic", [False, True])
def test_live_origin_forecasts_without_future_observed_heads(probabilistic):
    from hydrophysics.models.forecast_lstm import GlobalForecastLSTM

    data = _short_record()
    model = GlobalForecastLSTM(lookback=10, horizon=7, hidden=4, epochs=1,
                               device="cpu", probabilistic=probabilistic).fit(data)
    origin = data.n_days - model.H - 1
    before, sigma = model._predict(data)
    data.target[:, origin + 1:] = np.nan
    after, live_sigma = model._predict(data)
    assert np.isfinite(after[:, origin]).all()
    np.testing.assert_array_equal(after[:, origin], before[:, origin])
    if probabilistic:
        np.testing.assert_array_equal(live_sigma[:, origin], sigma[:, origin])
    data.target[:] = np.nan
    assert np.isnan(model.forecast(data)).all()


def test_point_scores_use_common_support_and_report_missing_forecasts():
    from hydrophysics.forecast_eval import horizon_table

    data = _short_record()
    cube = np.full((1, data.n_days, 1), np.nan)
    origins = np.array([70, 73, 90])
    cube[0, origins, 0] = data.target[0, origins + 1] + .1
    table = horizon_table(data, cube, [1])
    assert (table.n_scored == 3).all()
    assert (table.n_observations == 50).all()
    assert (table.forecast_availability == 3 / 50).all()
    expected = np.sqrt(np.mean((data.target[0, origins + 1] - data.target[0, origins])**2))
    assert table.loc[(1, "persistence"), "rmse_median"] == pytest.approx(expected)


def test_probabilistic_scores_drop_missing_baseline_on_both_sides():
    from hydrophysics.forecast_eval import probabilistic_horizon_table

    data = _short_record()
    mean = np.full((1, data.n_days, 1), np.nan)
    sigma = np.full_like(mean, np.nan)
    origins = np.array([70, 73, 90])
    mean[0, origins, 0] = data.target[0, origins + 1]
    sigma[0, origins, 0] = 1
    data.target[0, 73] = np.nan
    table = probabilistic_horizon_table(data, mean, sigma, [1])
    assert (table.n_forecasts == 3).all()
    assert (table.n_scored == 2).all()
    assert np.isfinite(table.crps_median).all()
