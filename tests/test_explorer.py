"""Tests for the Choushui head + subsidence explorer (subsidence.py + explorer.py)."""

from __future__ import annotations

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import pytest


def test_load_mlcw_stations():
    from hydrophysics.subsidence import load_mlcw_stations
    from hydrophysics.config import default_config

    cfg = default_config()
    path = cfg.data_dir / "mlcw_stations.csv"
    if not path.exists():
        pytest.skip("mlcw_stations.csv only present alongside the real data")
    df = load_mlcw_stations(path)
    assert list(df.columns[:3]) == ["sub_id", "x", "y"]
    assert len(df) == 14
    assert df["x"].between(170000, 210000).all()
    assert df["y"].between(2600000, 2660000).all()


@pytest.fixture()
def gwdata(tmp_path):
    from hydrophysics import Config, load_dataset
    from hydrophysics.sample import write_sample

    d = write_sample(tmp_path / "data", n_wells=5, seed=1)
    return load_dataset(Config(data_dir=d, baseline_results=d / "gw_fit_results.csv"))


def test_idw_interp_exact_and_blend():
    from hydrophysics.subsidence import idw_interp

    well_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    values = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2 wells, 2 timesteps)
    out = idw_interp(np.array([[0.0, 0.0]]), well_xy, values)
    assert np.allclose(out[0], [1.0, 2.0], atol=1e-3)
    mid = idw_interp(np.array([[5.0, 0.0]]), well_xy, values)[0]
    assert (mid >= values.min(0)).all() and (mid <= values.max(0)).all()


def test_monthly_heads_shapes(gwdata):
    from hydrophysics.subsidence import monthly_heads, well_xy

    H, dates = monthly_heads(gwdata)
    assert H.shape[0] == gwdata.n_wells
    assert H.shape[1] == len(dates)
    assert isinstance(dates, pd.DatetimeIndex)
    assert well_xy(gwdata).shape == (gwdata.n_wells, 2)
