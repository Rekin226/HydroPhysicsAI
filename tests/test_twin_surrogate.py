"""FNO surrogate plumbing on a toy grid: rasterisation round-trip, dataset shapes, a
two-epoch training that runs end to end, and an autoregressive rollout of the right
shape. Skipped where PhysicsNeMo is not installed (CI on Python 3.10)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("physicsnemo")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.forward import load_members  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.inputs import TwinInputs  # noqa: E402
from hydrophysics.twin.surrogate import (  # noqa: E402
    IN_CHANNELS,
    N_LAYERS,
    Normalizer,
    build_dataset,
    load_surrogate,
    rasterize,
    surrogate_rollout,
    train,
    unrasterize,
)


def _inputs(T=24, nx=6, ny=4):
    mask = np.ones((ny, nx), dtype=bool)
    mask[0, 0] = False
    g = FanGrid(nx=nx, ny=ny, dx=1000.0, x0=200_000.0, y0=0.0, mask=mask)
    A = g.n_active
    dates = pd.date_range("2012-01-01", periods=T, freq="MS")
    well_xy = np.array([[201_500.0, 500.0], [204_500.0, 1500.0]])
    obs_idx = np.array([g.active_index(x, y) for x, y in well_xy])
    obs_h = np.tile(np.array([[3.0], [4.0]]), (1, T))
    return TwinInputs(grid=g, hf=None, dates=dates, obs_h=obs_h, obs_h_filled=obs_h,
                      obs_idx=obs_idx, obs_layer=np.array([0, 1]), well_xy=well_xy,
                      sids=["a", "b"], ground_elev=torch.full((A,), 12.0, dtype=torch.float64),
                      E_by_class={"irrigation": np.full((A, T), 100.0)},
                      recharge_field=torch.full((A, T), 1e-4, dtype=torch.float64))


def _theta(tmp_path):
    theta = {"log_T_proximal": [np.log(500.0)], "log_S_proximal": [np.log(1e-3)],
             "log_T_mid": [np.log(300.0)] * 4, "log_S_mid": [np.log(1e-3)] * 4,
             "log_T_distal": [np.log(100.0)] * 4, "log_S_distal": [np.log(1e-3)] * 4,
             "log_L_mid": [np.log(1e-4)] * 3, "log_L_distal": [np.log(1e-4)] * 3,
             "log_eta": np.log(0.3), "log_head_extra": np.log(30.0),
             "recharge_frac_logit": 0.0, "log_C_coast": [np.log(50.0)] * 4,
             "log_C_apex": [np.log(50.0)]}
    meta = {"param_mode": "zonal", "boundaries": "coast-apex", "zone_boundaries": "205,203",
            "meter_filter": "none"}
    p = tmp_path / "stage3_theta.json"
    p.write_text(json.dumps({"theta": theta, "meta": meta}))
    return str(p)


def test_rasterize_round_trips_and_fills_outside():
    inp = _inputs()
    v = np.arange(inp.grid.n_active, dtype="float32")
    r = rasterize(inp.grid, v, fill=-1.0)
    assert r.shape == inp.grid.mask.shape and r[0, 0] == -1.0
    assert np.array_equal(unrasterize(inp.grid, r), v)
    stacked = rasterize(inp.grid, np.stack([v, 2 * v]))
    assert stacked.shape == (2,) + inp.grid.mask.shape


def test_build_train_and_roll(tmp_path):
    inp = _inputs()
    member = load_members([_theta(tmp_path)])[0]
    data = build_dataset(inp, member, n_samples=3, horizon=2, seed=0, device="cpu",
                         log=lambda *_: None)
    assert data["X"].shape == (6, IN_CHANNELS, inp.grid.ny, inp.grid.nx)
    assert data["Y"].shape == (6, N_LAYERS, inp.grid.ny, inp.grid.nx)
    assert np.isfinite(data["X"]).all() and np.isfinite(data["Y"]).all()
    assert (data["X"][:, -1] == inp.grid.mask).all()          # mask channel

    norm = Normalizer.fit(data["X"], data["Y"], data["mask"])
    assert norm.x_std[-1] == 1.0 and norm.x_mean[-1] == 0.0
    out = str(tmp_path / "fno")
    res = train(data, out, epochs=2, batch=4, modes=2, width=4, layers=2, device="cpu",
                log=lambda *_: None)
    assert len(res["history"]) == 2 and np.isfinite(res["history"][-1]["val"])

    fno, norm2, meta = load_surrogate(out, device="cpu")
    assert meta["n_train"] + meta["n_val"] == 6
    A = inp.grid.n_active
    h0 = np.full((N_LAYERS, A), 5.0, dtype="float32")
    E = np.full((A, 3), 100.0)
    r = np.full((A, 3), 1e-4)
    heads = surrogate_rollout(fno, norm2, inp.grid, h0, E, r, inp.ground_elev.numpy(), "cpu")
    assert heads.shape == (N_LAYERS, A, 4)
    assert np.isfinite(heads).all()
    assert np.array_equal(heads[..., 0], h0)
