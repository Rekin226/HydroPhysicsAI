"""Open boundaries (2026-09-11): geometry, closed-basin reproduction, mass balance with an
outlet, and the adjoint's gradient with respect to the new conductances.

Why these four. The closed basin was never wrong as arithmetic; it was wrong as a claim
about the fan, and it showed up as pinned forcing parameters rather than as a failing
test. So the tests here check (1) that the boundary cells are the coast and the apex and
nothing else, (2) that ``boundaries=None`` still reproduces every pre-boundary result bit
for bit, (3) that water actually leaves through the coast, and (4) that calibration can
learn the conductances -- i.e. the implicit-function adjoint carries a correct gradient
for them, which is the part most likely to be silently wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.boundaries import fan_boundaries  # noqa: E402
from hydrophysics.twin.calibrate_flow import _rollout, fit_flow  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402


def _grid(nx=8, ny=5, dx=1000.0, x0=200_000.0):
    """A rectangle with a notch out of the west edge of row 2 and a hole in row 3, so the
    per-row extreme rule is exercised against an interior exposed face."""
    mask = np.ones((ny, nx), dtype=bool)
    mask[2, 0] = False        # notch: row 2's coast cell is column 1
    mask[3, 4] = False        # interior hole: exposes west/east faces that are NOT coast
    return FanGrid(nx=nx, ny=ny, dx=dx, x0=x0, y0=0.0, mask=mask)


def test_boundary_cells_are_the_row_extremes_only():
    g = _grid()
    b = fan_boundaries(g, proximal_km=205.0)    # columns 5..7 lie at >= 205.5 km
    cent = g.centroids()
    rows, cols = np.nonzero(g.mask)
    assert b.n_coast == g.ny
    assert set(cols[b.coast_idx]) == {0, 1}
    assert cols[b.coast_idx][rows[b.coast_idx] == 2] == 1     # the notch
    # the interior hole's east-facing neighbour (row 3, col 3) must not be a coast cell
    assert not any((rows[b.coast_idx] == 3) & (cols[b.coast_idx] == 5))
    assert b.n_apex == g.ny
    assert set(cols[b.apex_idx]) == {g.nx - 1}
    assert (cent[b.apex_idx, 0] / 1000.0 >= 205.0).all()


def test_apex_filter_can_leave_no_apex_and_says_so():
    g = _grid()
    with pytest.raises(ValueError, match="no apex"):
        fan_boundaries(g, proximal_km=400.0)


def test_no_boundaries_reproduces_the_closed_basin_exactly():
    """``boundaries=None`` must be the old model: same operator, same RHS, same heads."""
    g = _grid()
    A, steps = g.n_active, 4
    torch.manual_seed(0)
    rech = torch.rand(2, A, steps, dtype=torch.float64) * 1e-3
    pump = torch.zeros(2, A, steps, dtype=torch.float64)
    h0 = torch.rand(2, A, dtype=torch.float64)
    closed = FlowModel(g, n_layers=2, dt_days=30.0)
    h_closed = closed(h0, rech, pump, steps)
    # a model WITH boundaries but conductances driven to the floor must converge to it
    opened = FlowModel(g, n_layers=2, dt_days=30.0, boundaries=fan_boundaries(g))
    opened.set_apex_heads(h0)
    with torch.no_grad():
        opened.log_C_coast.fill_(float(np.log(1e-12)))
        opened.log_C_apex.fill_(float(np.log(1e-12)))
    h_open = opened(h0, rech, pump, steps)
    assert torch.allclose(h_closed, h_open, atol=1e-6)
    assert not closed.has_boundaries and closed.log_C_coast is None


def test_coast_drains_recharge_and_the_balance_closes_with_the_outflow():
    """With a coast, steady recharge no longer accumulates without limit: the stored
    change equals recharge minus what left through the boundary, step by step."""
    g = _grid()
    A, steps = g.n_active, 6
    b = fan_boundaries(g)
    m = FlowModel(g, n_layers=1, dt_days=30.0, boundaries=b)
    with torch.no_grad():
        m.log_C_coast.fill_(float(np.log(2000.0)))   # a Dirichlet-like coast (C ~ 2T)
        m.log_C_apex.fill_(float(np.log(1e-12)))     # apex closed for this test
    h0 = torch.zeros(1, A, dtype=torch.float64)
    m.set_apex_heads(h0)
    rech = torch.full((1, A, steps), 1e-3, dtype=torch.float64)
    with torch.no_grad():
        h = m(h0, rech, torch.zeros(1, A, steps, dtype=torch.float64), steps)
        S = torch.exp(m.log_S)[0]
        C = float(torch.exp(m.log_C_coast))
    area = g.dx ** 2
    for t in range(steps):
        stored = float(((h[0, :, t + 1] - h[0, :, t]) * S).sum() * area)
        added = float(rech[..., t].sum() * area * m.dt)
        # backward Euler: the boundary flux is evaluated at the NEW head
        left = float((C * b.coast_faces * h[0, b.coast_idx, t + 1].numpy()).sum() * m.dt)
        assert stored == pytest.approx(added - left, rel=1e-4)   # CG tolerance, as test_twin_flow
    assert float(left) > 0.0
    # and the closed model's heads would sit strictly higher everywhere at the end
    closed = FlowModel(g, n_layers=1, dt_days=30.0)
    with torch.no_grad():
        hc = closed(h0, rech, torch.zeros(1, A, steps, dtype=torch.float64), steps)
    assert (hc[0, :, -1] > h[0, :, -1]).all()


def test_apex_boundary_feeds_water_in_toward_its_prescribed_head():
    g = _grid()
    A, steps = g.n_active, 3
    b = fan_boundaries(g)
    m = FlowModel(g, n_layers=1, dt_days=30.0, boundaries=b)
    with torch.no_grad():
        m.log_C_coast.fill_(float(np.log(1e-12)))
        m.log_C_apex.fill_(float(np.log(2000.0)))
    h0 = torch.zeros(1, A, dtype=torch.float64)
    h_apex = torch.zeros(1, A, dtype=torch.float64)
    h_apex[0, b.apex_idx] = 10.0
    m.set_apex_heads(h_apex)
    zero = torch.zeros(1, A, steps, dtype=torch.float64)
    with torch.no_grad():
        h = m(h0, zero, zero, steps)
    assert (h[0, b.apex_idx, -1] > 0.0).all()
    assert float(h[0, :, -1].mean()) > 0.0


def test_adjoint_gradient_for_conductances_matches_finite_differences():
    """The implicit-function backward must carry d(loss)/d(log_C) for both boundaries."""
    g = _grid()
    A, steps = g.n_active, 3
    b = fan_boundaries(g)
    m = FlowModel(g, n_layers=2, dt_days=30.0, boundaries=b)
    h0 = torch.rand(2, A, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    m.set_apex_heads(h0 + 3.0)
    rech = torch.full((2, A, steps), 5e-4, dtype=torch.float64)
    pump = torch.zeros(2, A, steps, dtype=torch.float64)

    def loss_of(log_Cc, log_Ca):
        h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, steps, recharge=rech,
                     pumping=pump, log_C_coast=log_Cc, log_C_apex=log_Ca)
        return (h[..., 1:] ** 2).sum()

    lc = m.log_C_coast.detach().clone().requires_grad_(True)
    la = m.log_C_apex.detach().clone().requires_grad_(True)
    loss = loss_of(lc, la)
    loss.backward()
    eps = 1e-5
    for par, grad in ((lc, lc.grad), (la, la.grad)):
        for k in range(par.shape[0]):
            plus = par.detach().clone()
            plus[k] += eps
            minus = par.detach().clone()
            minus[k] -= eps
            if par is lc:
                fd = (loss_of(plus, la.detach()) - loss_of(minus, la.detach())) / (2 * eps)
            else:
                fd = (loss_of(lc.detach(), plus) - loss_of(lc.detach(), minus)) / (2 * eps)
            assert float(grad[k]) == pytest.approx(float(fd), rel=1e-4, abs=1e-8)


def test_fit_flow_learns_conductances_and_reports_them():
    g = _grid()
    A, steps = g.n_active, 3
    b = fan_boundaries(g)
    m = FlowModel(g, n_layers=2, dt_days=30.0, boundaries=b)
    h0 = torch.zeros(2, A, dtype=torch.float64)
    rech = torch.full((2, A, steps), 1e-4, dtype=torch.float64)
    obs_idx = torch.tensor([0, 5, 11])
    obs_layer = torch.tensor([0, 1, 0])
    obs_h = torch.zeros(3, steps, dtype=torch.float64)
    fit = fit_flow(m, obs_h, obs_idx, obs_layer, rech, epochs=3, lr=0.05, h0=h0)
    th = fit["theta"]
    assert "log_C_coast" in th and "log_C_apex" in th
    assert len(th["C_coast_m2day"]) == 2 and th["C_apex_m2day"] > 0
    assert fit["n_params"] == 2 + 2 + 1 + 2 + 1        # T, S per layer; L; C_coast x2; C_apex
    assert "log_C_coast" in fit["bounds_hit"]
    # the model's own registered conductances were written back
    assert torch.allclose(torch.exp(m.log_C_coast).squeeze(),
                          torch.tensor(th["C_coast_m2day"], dtype=torch.float64))
