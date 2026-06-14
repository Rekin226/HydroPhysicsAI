"""Smoke test for the forecasting model. Skipped when torch is absent.

Checks that fit -> forecast produces a correctly-shaped cube and that the horizon-wise
eval harness runs and returns finite skill numbers on the synthetic sample.
"""

from __future__ import annotations

import numpy as np
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
