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


def test_mlcw_compaction_decodes_and_signs(tmp_path):
    from hydrophysics.subsidence import mlcw_compaction

    name = "僑義國小"
    enc = "".join(f"_{b:02X}" for b in name.encode("utf-8"))
    d = tmp_path / "ls_cache" / "clean"
    d.mkdir(parents=True)
    dates = pd.date_range("2014-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {"NO1": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
         "NO2": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
         "NO3": [100.0, 99.8, 99.6, 99.4, 99.2, 99.0]},
        index=pd.Index(dates, name="datetime"),
    )
    df.to_parquet(d / f"ls-wra-mlcw-obs__{enc}.parquet")

    series = mlcw_compaction(str(tmp_path))
    assert name in series
    s = series[name]
    assert len(s) == 6
    assert np.isclose(s.iloc[0], 0.0)
    assert s.iloc[-1] > 0
    assert np.isclose(s.iloc[-1], 1.0, atol=1e-6)


def test_calibrate_sk_recovers_slope():
    from hydrophysics.subsidence import calibrate_sk_from_pairs

    rng = np.random.default_rng(0)
    per_site = {}
    for s in ("A", "B"):
        D = np.linspace(0, 5, 12)
        C = 0.02 * D + rng.normal(0, 1e-4, size=D.size)
        per_site[s] = (D, C)
    res = calibrate_sk_from_pairs(per_site)
    assert abs(res["sk"] - 0.02) < 1e-3
    assert res["r2"] > 0.99


def test_head_and_subsidence_grid(gwdata):
    from shapely.geometry import box
    from hydrophysics.explorer import head_grid, subsidence_grid

    wx = gwdata.attrs["tm_x"].astype(float)
    wy = gwdata.attrs["tm_y"].astype(float)
    poly = box(wx.min(), wy.min(), wx.max(), wy.max())  # rectangle covering the wells
    XX, YY, HH, dates = head_grid(gwdata, poly, n=12)
    assert XX.shape == (12, 12)
    assert HH.shape == (len(dates), 12, 12)
    assert np.isfinite(HH).any()
    SS = subsidence_grid(HH, sk=0.01)
    assert SS.shape == HH.shape
    fin = np.isfinite(SS)
    assert (SS[fin] >= -1e-9).all()
    assert np.nanmax(SS[-1] - SS[0]) >= -1e-9


def test_build_explorer_writes_html(gwdata, tmp_path):
    from shapely.geometry import box
    from hydrophysics.explorer import build_explorer

    wx = gwdata.attrs["tm_x"].astype(float)
    wy = gwdata.attrs["tm_y"].astype(float)
    poly = box(wx.min(), wy.min(), wx.max(), wy.max())
    out = tmp_path / "explorer.html"
    path = build_explorer(gwdata, poly, stations=None, compaction=None,
                          out_html=str(out), n=10)
    assert path == str(out)
    assert out.exists() and out.stat().st_size > 1000
    assert "plotly" in out.read_text(errors="ignore").lower()


def test_fit_sk_regression_recovers_beta():
    from hydrophysics.subsidence import fit_sk_regression

    rng = np.random.default_rng(0)
    b0, b1 = -3.0, -1.5
    dist = {}
    per_site = {}
    for i in range(8):
        dc = 0.2 * i
        sk = np.exp(b0 + b1 * dc)
        D = np.linspace(0.5, 5.0, 20)
        C = sk * D + rng.normal(0, 1e-4, size=D.size)
        name = f"s{i}"
        dist[name] = dc
        per_site[name] = (D, C)
    fit = fit_sk_regression(per_site, dist)
    assert abs(fit["b0"] - b0) < 0.1
    assert abs(fit["b1"] - b1) < 0.1
    assert abs(fit["predict_sk"](0.0) - np.exp(b0)) < 1e-2
