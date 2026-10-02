"""Physical invariants and leakage barriers for observation updates."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydrophysics.twin.assimilation import (  # noqa: E402
    HeadObservationOperator,
    historical_offsets,
    lagged_residual_forecast,
)


def operator(cells=(0, 1), layers=(0, 1)):
    return HeadObservationOperator([[0., 0.], [1000., 0.], [100000., 0.]], cells, layers,
                                   n_layers=2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_perfect_observations_preserve_nonuniform_state_and_offsets(dtype):
    op = operator()
    state = torch.tensor([[2., 40., 100.], [-7., 15., 80.]], dtype=dtype)
    offset = torch.tensor([9., -3.])
    changed = op.update(state, op.observe(state, offset), offset)
    torch.testing.assert_close(changed, state, rtol=0, atol=0)


def test_forward_update_accepts_cached_observation_operator():
    from types import SimpleNamespace

    from hydrophysics.twin.forward import nudge_to_observations

    op = operator()
    state = torch.tensor([[2., 40., 100.], [-7., 15., 80.]], dtype=torch.float64)
    inp = SimpleNamespace(obs_h=op.observe(state).numpy()[:, None])
    result = nudge_to_observations(inp, state, 0, 1., method="innovation", operator=op)
    torch.testing.assert_close(result, state, rtol=0, atol=0)


def test_missing_layers_locality_gain_and_no_mutation():
    op = operator()
    state = torch.zeros((2, 3), dtype=torch.float64)
    changed = op.update(state, [4., float("nan")], gain=.5, radius_km=2.)
    assert changed[0, 0] == 2.
    assert 0 < changed[0, 1] < 2.
    assert changed[0, 2] < 1e-100
    assert torch.count_nonzero(changed[1]) == 0
    assert torch.count_nonzero(state) == 0
    torch.testing.assert_close(op.update(state, [float("nan")]*2), state)


def test_shared_cell_wells_average_innovations_after_datum_removal():
    op = operator((0, 0), (0, 0))
    state = torch.zeros((2, 3), dtype=torch.float64)
    updated = op.update(state, [12., -6.], offset=[10., -10.])
    assert updated[0, 0] == 3.


def test_historical_datum_never_uses_future_observations():
    obs = np.array([[2., 4., 100.], [np.nan, np.nan, 100.]])
    predicted = np.ones_like(obs)
    first = historical_offsets(obs, predicted, 2, fallback=[0., 7.])
    obs[:, 2] = 99999
    np.testing.assert_array_equal(first, [2., 7.])
    np.testing.assert_array_equal(first, historical_offsets(obs, predicted, 2, [0., 7.]))


def test_bad_gain_or_geometry_fails():
    op = operator()
    with pytest.raises(ValueError, match="Gain"):
        op.update(torch.zeros((2, 3)), [1., 2.], gain=1.5)
    with pytest.raises(ValueError, match="geometry"):
        HeadObservationOperator([[0., 0.]], [5], [0])


def test_residual_forecast_is_causal_and_handles_missing_previous_observation():
    base = np.array([[2., 3., 4.]])
    obs = np.array([[5., np.nan, 99.]])
    first = lagged_residual_forecast(base, obs, [1.], [4.])
    np.testing.assert_allclose(first, [[5., 6., 4.]])
    obs[:, 0] = 1000
    changed = lagged_residual_forecast(base, obs, [1.], [4.])
    assert changed[0, 0] == first[0, 0]
    assert changed[0, 1] == 1001


def test_transient_storage_ceiling_applies_to_each_ensemble_member(monkeypatch):
    from types import SimpleNamespace

    from hydrophysics.twin import update_experiment

    model = SimpleNamespace(log_S=torch.nn.Parameter(torch.log(
        torch.tensor([.0102, .002], dtype=torch.float64))))
    monkeypatch.setattr(update_experiment, "build_model", lambda *args: (model, {}, None))
    scaled, _ = update_experiment.prepare_model(SimpleNamespace(grid=None), None, "cpu", {"S": 32})
    torch.testing.assert_close(scaled.log_S.exp(), torch.tensor([.3, .064], dtype=torch.float64))
