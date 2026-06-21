"""Tests for the ET (evapotranspiration) driver. No network: ET0 is supplied or cached."""

from __future__ import annotations

import numpy as np
import pytest

from hydrophysics import Config, load_dataset
from hydrophysics.et import et0_for_data, net_recharge
from hydrophysics.sample import write_sample


@pytest.fixture()
def data(tmp_path):
    d = write_sample(tmp_path / "data", n_wells=4, seed=1)
    return load_dataset(Config(data_dir=d))


def test_net_recharge_transform(data):
    """net recharge = rain - coef*ET0; only the rainfall channel changes."""
    et0 = np.full(data.rainfall.shape, 2.0, dtype="float32")
    nd = net_recharge(data, et0=et0, coef=1.0)
    expected = np.nan_to_num(data.rainfall) - 2.0
    assert np.allclose(np.nan_to_num(nd.rainfall), expected)
    # other channels untouched, target identical
    assert np.array_equal(np.nan_to_num(nd.upstream), np.nan_to_num(data.upstream))
    assert np.array_equal(np.nan_to_num(nd.target), np.nan_to_num(data.target))
    # coef=0 is a no-op on rainfall
    assert np.allclose(np.nan_to_num(net_recharge(data, et0=et0, coef=0.0).rainfall),
                       np.nan_to_num(data.rainfall))


def test_et0_cache_roundtrip(data, tmp_path):
    """A matching cache is loaded without any fetch (offline)."""
    cache = tmp_path / "et.npz"
    et0 = np.random.default_rng(0).uniform(1, 5, data.target.shape).astype("float32")
    np.savez_compressed(cache, et0=et0,
                        well_ids=np.array([str(w) for w in data.well_ids]),
                        dates=np.array([str(d.date()) for d in data.dates]))
    loaded = et0_for_data(data, cache_path=str(cache))  # must not hit the network
    assert loaded.shape == data.target.shape
    assert np.allclose(loaded, et0)


def test_wells_lonlat_in_taiwan(data):
    """TWD97 -> WGS84 puts the (synthetic) wells in a plausible lon/lat range."""
    pytest.importorskip("pyproj")
    from hydrophysics.et import wells_lonlat
    lon, lat = wells_lonlat(data)
    assert lon.shape == (data.n_wells,) and lat.shape == (data.n_wells,)
    assert np.isfinite(lon).all() and np.isfinite(lat).all()
    assert (lon > 119).all() and (lon < 122).all()  # western Taiwan longitudes
