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
