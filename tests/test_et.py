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


def test_wells_lonlat_native_wgs84_passthrough(data):
    """crs=EPSG:4326: tm_x/tm_y are taken as lon/lat directly, without pyproj."""
    from dataclasses import replace

    from hydrophysics.et import wells_lonlat
    attrs = data.attrs.copy()
    attrs["tm_x"] = np.linspace(120.1, 120.6, data.n_wells)   # already lon
    attrs["tm_y"] = np.linspace(23.5, 23.9, data.n_wells)     # already lat
    d = replace(data, attrs=attrs, crs="EPSG:4326")
    lon, lat = wells_lonlat(d)
    assert np.allclose(lon, attrs["tm_x"].to_numpy())
    assert np.allclose(lat, attrs["tm_y"].to_numpy())


def test_wells_lonlat_crs_mismatch_raises(data):
    """Declaring projected metres as EPSG:4326 must fail loudly, not fetch ET at (huge, huge)."""
    from dataclasses import replace

    from hydrophysics.et import wells_lonlat
    # The sample's tm_x/tm_y are projected metres (~1.9e5 / ~2.6e6); calling them lon/lat
    # is the exact silent-failure this guards against.
    d = replace(data, crs="EPSG:4326")
    with pytest.raises(ValueError, match="crs"):
        wells_lonlat(d)


def test_et_download_dates_are_aligned_without_shifting_gaps(monkeypatch):
    import sys
    from types import SimpleNamespace

    from hydrophysics.et import _fetch_one

    monkeypatch.setitem(sys.modules, "aquascope.collectors.openmeteo", None)
    payload = {"daily": {"time": ["2024-01-03", "2024-01-01"],
                         "et0_fao_evapotranspiration": [3.0, 1.0]}}
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(
        get=lambda *a, **kw: SimpleNamespace(json=lambda: payload)))
    values = _fetch_one(23.5, 120.5, "2024-01-01", "2024-01-03")
    np.testing.assert_allclose(values, [1.0, np.nan, 3.0])
    payload["daily"]["time"] = ["2024-01-01", "2024-01-01"]
    assert _fetch_one(23.5, 120.5, "2024-01-01", "2024-01-03") is None


def test_total_et_outage_cannot_produce_zero_weather_or_cache(data, tmp_path, monkeypatch):
    import hydrophysics.et as et

    monkeypatch.setattr(et, "wells_lonlat", lambda d: (np.zeros(d.n_wells),
                                                      np.zeros(d.n_wells)))
    monkeypatch.setattr(et, "_fetch_one", lambda *a: None)
    monkeypatch.setattr(et.time, "sleep", lambda *a: None)
    cache = tmp_path / "failed.npz"
    with pytest.raises(ValueError, match="No ET0 observations"):
        et0_for_data(data, str(cache), retries=1, pause=0)
    assert not cache.exists()


def test_spatial_et_fill_retains_imputation_mask(data, tmp_path, monkeypatch):
    import hydrophysics.et as et

    monkeypatch.setattr(et, "wells_lonlat", lambda d: (np.zeros(d.n_wells),
                                                      np.zeros(d.n_wells)))
    values = iter([None] + [np.full(data.n_days, 2.0)] * (data.n_wells - 1))
    monkeypatch.setattr(et, "_fetch_one", lambda *a: next(values))
    monkeypatch.setattr(et.time, "sleep", lambda *a: None)
    cache = tmp_path / "partial.npz"
    result = et0_for_data(data, str(cache), retries=1, pause=0)
    assert (result == 2.0).all()
    with np.load(cache) as saved:
        assert saved["spatially_imputed"][0].all()
        assert not saved["spatially_imputed"][1:].any()


def test_missing_weather_requires_explicit_repair(data):
    from dataclasses import replace

    rain = data.rainfall.copy()
    rain[0, 0] = np.nan
    with pytest.raises(ValueError, match="Rainfall contains missing"):
        net_recharge(replace(data, rainfall=rain), et0=np.ones_like(rain))
    with pytest.raises(ValueError, match="ET0 must be finite"):
        net_recharge(data, et0=np.full_like(rain, np.nan))
    with pytest.raises(ValueError, match="ET0 must be finite"):
        net_recharge(data, et0=np.ones((1, 1)))
