"""Stage-4 column recalibration internals on synthetic rings (the real run needs the MLCW
cache) and the Laplace posterior on a toy grid."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.calibrate_coupled import (  # noqa: E402
    _fit,
    _make,
    _predict,
    _WeightedColumn,
    column_json,
    loso,
)
from hydrophysics.twin.uncertainty import (  # noqa: E402
    clip_to_bounds,
    flatten,
    jacobian,
    laplace,
    unflatten,
)


def _rings(n=6, L=4, T=36, seed=0):
    rng = np.random.default_rng(seed)
    heads = 5.0 - 0.05 * np.arange(T)[None, None, :] + rng.normal(0, 0.2, (n, L, T))
    heads[:, 1] -= 1.0                                   # layer 2 lower, drives more
    obs = 0.004 * np.cumsum(np.clip(5.0 - heads.mean(1), 0, None), axis=1)
    mask = np.ones((n, T))
    mask[:, ::7] = 0.0
    zone = rng.integers(0, 3, n)
    t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt)  # noqa: E731
    return t(heads), t(obs), t(mask), torch.tensor(zone)


@pytest.mark.parametrize("config", ["shared", "weighted", "zonal"])
def test_each_column_configuration_fits_predicts_and_serialises(config):
    H, OBS, M, Z = _rings()
    model = _make(config, "cpu")
    loss = _fit(model, H, OBS, M, Z, epochs=15, lr=0.05)
    assert np.isfinite(loss)
    with torch.no_grad():
        pred = _predict(model, H, Z)
    assert pred.shape == OBS.shape and torch.isfinite(pred).all()
    js = column_json(model)
    json.dumps(js)
    if config == "weighted":
        assert isinstance(model, _WeightedColumn)
        assert len(js["layer_weights"]) == 4 and abs(sum(js["layer_weights"]) - 1) < 1e-6
    if config == "zonal":
        assert len(js["zonal"]) == 3
    r2 = loso(config, H, OBS, M, Z, epochs=5, lr=0.05, device="cpu")
    assert np.isfinite(r2)


def test_flatten_unflatten_and_clip_round_trip():
    theta = {"log_T_mid": [1.0, 2.0], "log_eta": 0.5, "eta": 1.65, "log_C_apex": [30.0]}
    vec, index = flatten(theta, fixed=("log_eta",))
    assert index == [("log_T_mid", 0), ("log_T_mid", 1), ("log_C_apex", 0)]
    back = unflatten(vec + 1.0, index, theta)
    assert back["log_T_mid"] == [2.0, 3.0] and back["log_eta"] == 0.5
    clipped = clip_to_bounds(vec, index)
    assert clipped[2] <= np.log(1e5)                     # apex conductance ceiling


def test_laplace_posterior_on_a_toy_model(tmp_path):
    from hydrophysics.twin.forward import load_members
    from tests.test_twin_forward import _inputs, _theta_file

    inp = _inputs()
    p, _ = _theta_file(tmp_path)
    member = load_members([p])[0]
    vec, index = flatten(member.theta)
    sub = [j for j, (k, _) in enumerate(index) if k in ("log_T_mid", "log_eta")][:3]
    index = [index[j] for j in sub]
    J, resid = jacobian(inp, member, index, "cpu", log=lambda *_: None)
    n_obs = inp.obs_h.shape[0] * (len(inp.dates) - 1)
    assert J.shape == (n_obs, 3) and resid.shape == (n_obs,)
    assert np.isfinite(J).all()
    cov, sigma2 = laplace(J, resid, prior_sd=2.0)
    assert cov.shape == (3, 3) and sigma2 > 0
    assert (np.diag(cov) > 0).all() and (np.sqrt(np.diag(cov)) <= 2.0 + 1e-9).all()


def test_leveling_target_kfold_and_rezero():
    from hydrophysics.twin.calibrate_coupled import _rezero, kfold_sites

    H, OBS, M, Z = _rings(n=8)
    p = torch.arange(1.0, 9.0)[:, None] + torch.arange(36.0)[None, :] * 0.1
    r = _rezero(p, M)
    first = int(torch.argmax((M[0] > 0).to(torch.int64)))
    assert torch.allclose(r[:, first], torch.zeros(8))
    r2 = kfold_sites("shared", H, OBS, M, Z, epochs=5, lr=0.05, device="cpu", n_folds=4)
    assert np.isfinite(r2)
