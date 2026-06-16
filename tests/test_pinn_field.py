# tests/test_pinn_field.py
"""Tests for the spatial PINN head field. Torch tests skip when torch is absent."""

from __future__ import annotations

import numpy as np
import pytest

from hydrophysics import Config, load_dataset
from hydrophysics.sample import write_sample


@pytest.fixture()
def data(tmp_path):
    d = write_sample(tmp_path / "data", n_wells=4, seed=2)
    return load_dataset(Config(data_dir=d, baseline_results=d / "gw_fit_results.csv"))


def test_normalizer_roundtrips_coords_and_head(data):
    from hydrophysics.field_inputs import Normalizer

    norm = Normalizer.from_data(data)
    x = data.attrs["tm_x"].astype(float).to_numpy()
    y = data.attrs["tm_y"].astype(float).to_numpy()
    X, Y = norm.xy(x, y)
    # normalized coords land in [0, 1]
    assert X.min() >= -1e-6 and X.max() <= 1 + 1e-6
    assert Y.min() >= -1e-6 and Y.max() <= 1 + 1e-6
    # head standardize/unstandardize round-trips
    h = np.array([norm.h_mu - 2.0, norm.h_mu, norm.h_mu + 3.0])
    assert np.allclose(norm.h_from_norm(norm.h_to_norm(h)), h, atol=1e-5)
    # time in years
    assert np.isclose(norm.tau(np.array([365.25]))[0], 1.0)


def test_rainfall_field_idw(data):
    from hydrophysics.field_inputs import Normalizer, RainfallField, well_coords_norm

    norm = Normalizer.from_data(data)
    wc = well_coords_norm(data, norm)            # (W, 2)
    field = RainfallField(wc, np.nan_to_num(data.rainfall))

    # querying exactly at a well coordinate returns that well's rainfall that day
    day = int(np.flatnonzero(data.train_mask)[10])
    pts = wc[:1]                                  # first well
    val = field.at(pts, np.array([day]))
    assert np.isclose(val[0], np.nan_to_num(data.rainfall)[0, day], atol=1e-4)

    # interpolating between wells is finite and within the data range
    mid = wc.mean(axis=0, keepdims=True)
    v = field.at(mid, np.array([day]))
    assert np.isfinite(v).all()
    assert v[0] >= np.nan_to_num(data.rainfall)[:, day].min() - 1e-6
    assert v[0] <= np.nan_to_num(data.rainfall)[:, day].max() + 1e-6


torch = pytest.importorskip("torch")  # torch tests below skip if torch is absent


def test_positional_encoding_shape_and_grad():
    from hydrophysics.models.pinn_field import positional_encoding

    coords = torch.rand(5, 3, requires_grad=True)
    enc = positional_encoding(coords, n_bands=4)
    # original dims + sin/cos for each dim and band
    assert enc.shape == (5, 3 + 3 * 2 * 4)
    enc.sum().backward()
    assert coords.grad is not None and torch.isfinite(coords.grad).all()
