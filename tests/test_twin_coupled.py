"""Stage-4 coupling: flow heads must drive compaction, and the gradient must come back.

The point of ``CoupledTwin`` is that a loss on *subsidence* reaches the *flow* parameters.
If that path is broken the module is decorative -- joint calibration would silently
optimise nothing -- so it is tested directly rather than inferred from the forward pass.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydrophysics.twin.compaction import VEPColumn  # noqa: E402
from hydrophysics.twin.coupled import CoupledTwin, subsidence_loss  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402


def _grid(n=6):
    return FanGrid(nx=n, ny=n, dx=1000.0, x0=0.0, y0=0.0,
                   mask=np.ones((n, n), dtype=bool))


def _forcing(model, steps, pump=200.0, rech_rate=1e-6):
    """Forcing in which pumping dominates, so heads actually fall.

    ``recharge`` enters as ``rate * cell_area`` (1 km cells => 1e-6 m/d is 1 m3/d per
    cell) while pumping is already a volume, so a "small-looking" recharge rate can
    silently outweigh the pumping term and drive heads *up* -- which then leaves the
    preconsolidation gate shut and log_skv with no gradient at all.
    """
    A = model.flow.grid.n_active
    L = model.n_layers
    h0 = torch.zeros(L, A, dtype=torch.float64)
    rech = torch.full((L, A, steps), rech_rate, dtype=torch.float64)
    pumping = torch.zeros(L, A, steps, dtype=torch.float64)
    pumping[1] = pump
    return h0, rech, pumping


def test_shapes_and_sign():
    """Pumping lowers heads, and falling heads produce positive (sinking) subsidence."""
    m = CoupledTwin(_grid(), n_layers=4)
    steps = 6
    heads, subs = m(*_forcing(m, steps), steps)
    A = m.flow.grid.n_active
    assert heads.shape == (4, A, steps + 1)
    assert subs.shape == (A, steps + 1)
    assert torch.allclose(subs[:, 0], torch.zeros_like(subs[:, 0])), "must re-zero at t=0"
    # Pumped layer draws down, and the column responds by sinking.
    assert heads[1, :, -1].mean() < heads[1, :, 0].mean()
    assert subs[:, -1].mean() > 0


def test_gradient_reaches_flow_parameters():
    """A subsidence-only loss must produce non-zero grads on log_T / log_S / log_L.

    This is the property that makes Stage-4 joint calibration possible at all.
    """
    m = CoupledTwin(_grid(), n_layers=4)
    steps = 5
    h0, rech, pumping = _forcing(m, steps)
    _, subs = m(h0, rech, pumping, steps)

    obs = torch.zeros_like(subs)          # pull the prediction toward "no subsidence"
    mask = torch.ones_like(subs)
    subsidence_loss(subs, obs, mask).backward()

    for name in ("log_T", "log_S", "log_L"):
        g = getattr(m.flow, name).grad
        assert g is not None, f"{name} received no gradient from a subsidence loss"
        assert torch.isfinite(g).all(), f"{name} gradient not finite"
        assert g.abs().max() > 0, f"{name} gradient is identically zero"


def test_column_parameters_also_receive_gradient():
    m = CoupledTwin(_grid(), n_layers=4)
    steps = 5
    _, subs = m(*_forcing(m, steps), steps)
    subsidence_loss(subs, torch.zeros_like(subs), torch.ones_like(subs)).backward()
    for name in ("log_ske", "log_skv"):
        g = getattr(m.column, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0


def test_freeze_column_blocks_only_the_rheology():
    m = CoupledTwin(_grid(), n_layers=4).freeze_column()
    steps = 4
    _, subs = m(*_forcing(m, steps), steps)
    subsidence_loss(subs, torch.zeros_like(subs), torch.ones_like(subs)).backward()
    assert m.column.log_ske.grad is None
    assert m.flow.log_T.grad is not None and m.flow.log_T.grad.abs().max() > 0


def test_weighted_driver_starts_identical_to_mean():
    """Uniform logits => softmax 1/L => the weighted driver must equal the mean exactly."""
    steps = 4
    a = CoupledTwin(_grid(), n_layers=4, driver="mean")
    b = CoupledTwin(_grid(), n_layers=4, driver="weighted")
    b.flow.load_state_dict(a.flow.state_dict())
    b.load_column(a.column)
    args = _forcing(a, steps)
    _, sa = a(*args, steps)
    _, sb = b(*args, steps)
    assert torch.allclose(sa, sb, atol=1e-12)
    assert np.allclose(b.layer_weights(), 0.25)


def test_layer_driver_selects_that_layer():
    m = CoupledTwin(_grid(), n_layers=4, driver="layer", driver_layer=1)
    steps = 4
    heads, _ = m(*_forcing(m, steps), steps)
    assert torch.allclose(m.column_driver(heads), heads[1])
    assert np.allclose(m.layer_weights(), [0, 1, 0, 0])


def test_load_column_adopts_stage2_parameters():
    fitted = VEPColumn(n_sites=1)
    with torch.no_grad():
        fitted.log_ske.fill_(-5.0)
        fitted.log_skv.fill_(-3.0)
    m = CoupledTwin(_grid(), n_layers=2).load_column(fitted)
    assert float(m.column.log_ske) == pytest.approx(-5.0)
    assert float(m.column.log_skv) == pytest.approx(-3.0)


def test_per_site_column_is_pooled_on_load():
    """A per-site Stage-2 fit collapses to its mean, since the coupled column is global."""
    fitted = VEPColumn(n_sites=4)
    with torch.no_grad():
        fitted.log_ske.copy_(torch.tensor([-6.0, -4.0, -6.0, -4.0]))
    m = CoupledTwin(_grid(), n_layers=2).load_column(fitted)
    assert float(m.column.log_ske) == pytest.approx(-5.0)


def test_rejects_bad_driver():
    with pytest.raises(ValueError):
        CoupledTwin(_grid(), driver="nonsense")
    with pytest.raises(ValueError):
        CoupledTwin(_grid(), n_layers=4, driver="layer", driver_layer=9)
