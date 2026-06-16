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


def test_pde_residual_matches_analytic_solution():
    """For h = sin(pi X) sin(pi Y) exp(-lam tau) with T=1, S=1, lam=2pi^2, the
    diffusion residual dh/dt - (h_xx + h_yy) is identically zero."""
    from hydrophysics.models.pinn_field import pde_residual

    lam = 2.0 * (np.pi ** 2)

    def h_fn(X, Y, tau):
        return torch.sin(np.pi * X) * torch.sin(np.pi * Y) * torch.exp(-lam * tau)

    n = 64
    X = torch.rand(n, 1, dtype=torch.float64, requires_grad=True)
    Y = torch.rand(n, 1, dtype=torch.float64, requires_grad=True)
    tau = torch.rand(n, 1, dtype=torch.float64, requires_grad=True)
    rain = torch.zeros(n, 1, dtype=torch.float64)

    res = pde_residual(
        h_fn, X, Y, tau, rain,
        T_fn=lambda X, Y: torch.ones_like(X),
        alpha_fn=lambda X, Y: torch.zeros_like(X),
        d_fn=lambda X, Y: torch.zeros_like(X),
        S=1.0,
    )
    assert res.shape == (n, 1)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-6)


def test_spatial_pinn_fit_simulate_shapes_and_benchmark(data):
    from hydrophysics import benchmark_table
    from hydrophysics.models.pinn_field import SpatialPINN

    model = SpatialPINN(device="cpu", epochs=3, n_collocation=128, seed=0)
    model.fit(data)
    pred = model.simulate(data)
    assert pred.shape == data.target.shape
    assert np.isfinite(pred).all()
    table = benchmark_table(data, {model.name: pred}, period="val")
    assert model.name in table.index


def test_spatial_pinn_lowo_no_leakage(data):
    """A held-out well contributes no observation rows to the data loss."""
    from hydrophysics.models.pinn_field import SpatialPINN

    held = np.zeros(data.n_wells, dtype=bool)
    held[0] = True
    model = SpatialPINN(device="cpu", epochs=1, n_collocation=32, seed=0)
    rows = model._obs_rows(data, train_wells=~held)
    # well index 0 must never appear among the observation-row well indices
    assert (rows["well"] != 0).all()


def test_build_model_registers_pinn():
    from hydrophysics.train import build_model
    from hydrophysics.models.pinn_field import SpatialPINN

    model = build_model("pinn", device="cpu", epochs=3)
    assert isinstance(model, SpatialPINN)
    assert model.name == "pinn"
