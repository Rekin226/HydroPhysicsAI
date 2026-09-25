"""Adversarial checks of the 2026-09-23 opt-in physics (delay bed, rivers, canal water).

Independent of the implementation's own tests: water budgets are closed from the heads the
rollout returns (not from the recurrence the code implements), signs are checked against
physical intuition, and the implicit adjoint is checked with ``torch.autograd.gradcheck``
on a tiny float64 grid with a mixed connected/disconnected RIV set and u0 != h0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin import flow as flow_mod  # noqa: E402
from hydrophysics.twin.calibrate_flow import _rollout  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.rivers import RiverSet  # noqa: E402

D = torch.float64


def _model(nx=5, ny=4, n_layers=3, seed=0):
    g = FanGrid(nx=nx, ny=ny, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((ny, nx), dtype=bool))
    m = FlowModel(g, n_layers=n_layers, dt_days=30.0)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        m.log_T.copy_(torch.tensor(np.log(rng.uniform(100, 800, m.log_T.shape)), dtype=D))
        m.log_S.copy_(torch.tensor(np.log(rng.uniform(1e-4, 1e-2, m.log_S.shape)), dtype=D))
        if n_layers > 1:
            m.log_L.copy_(torch.tensor(np.log(rng.uniform(1e-4, 1e-2, m.log_L.shape)),
                                       dtype=D))
    return m


def _storage(m, h):
    return float((torch.exp(m.log_S) * m.area * h).detach().sum())


def _delay_fields(m, seed=3):
    rng = np.random.default_rng(seed)
    L, A = m.n_layers, m.grid.n_active
    Sd = torch.tensor(np.log(rng.uniform(1e-3, 5e-2, (L, A))), dtype=D)
    tau = torch.tensor(np.log(rng.uniform(60.0, 3000.0, (L, A))), dtype=D)
    return Sd, tau


# ---------------------------------------------------------------------------------------
# mass balance
# ---------------------------------------------------------------------------------------
def test_delay_bed_closed_box_conserves_total_water():
    """No forcing, no boundaries, u0 != h0: aquifer + slow-store volume is constant."""
    m = _model()
    rng = np.random.default_rng(1)
    L, A = m.n_layers, m.grid.n_active
    h0 = torch.tensor(rng.normal(10.0, 3.0, (L, A)), dtype=D)
    u0 = torch.tensor(rng.normal(14.0, 3.0, (L, A)), dtype=D)
    lSd, ltau = _delay_fields(m)
    Sd = torch.exp(lSd)
    w0 = _storage(m, h0) + float((Sd * m.area * u0).sum())
    for n in (1, 5, 24):
        h, u = _rollout(m, m.log_T, m.log_S, m.log_L, h0, n, delay_Sd=lSd, delay_tau=ltau,
                        u0=u0, return_state=True)
        w = _storage(m, h[..., -1]) + float((Sd * m.area * u).sum())
        # the exchanged volume is ~ Sd*area*|u0-h0| ~ 1e6 m3; CG tol 1e-8 relative
        assert abs(w - w0) <= 1e-6 * abs(w0), (n, w - w0)
        # and water did move between the stores (the test is not vacuous)
        assert abs(_storage(m, h[..., -1]) - _storage(m, h0)) > 1e3


def test_delay_bed_equilibrium_is_steady():
    m = _model()
    L, A = m.n_layers, m.grid.n_active
    h0 = torch.full((L, A), 7.5, dtype=D)
    lSd, ltau = _delay_fields(m)
    h, u = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 12, delay_Sd=lSd, delay_tau=ltau,
                    return_state=True)
    assert torch.allclose(h, h0[..., None].expand_as(h), atol=1e-9, rtol=0)
    assert torch.allclose(u, h0, atol=1e-9, rtol=0)


def test_delay_bed_releases_water_under_pumping():
    """Pumping draws the aquifer below the store; the store must lose water (u falls)
    and the aquifer drawdown must be smaller than without the bed."""
    m = _model()
    L, A = m.n_layers, m.grid.n_active
    h0 = torch.full((L, A), 5.0, dtype=D)
    pump = torch.zeros(L, A, 6, dtype=D)
    pump[-1, A // 2, :] = 5000.0
    lSd, ltau = _delay_fields(m)
    h_ref = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 6, pumping=pump)
    h, u = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 6, pumping=pump, delay_Sd=lSd,
                    delay_tau=ltau, return_state=True)
    assert float(u.min()) < 5.0 and float(u.max()) <= 5.0 + 1e-9
    assert float(h[-1, A // 2, -1]) > float(h_ref[-1, A // 2, -1])
    # budget: pumped volume = aquifer loss + store loss
    pumped = float(pump.sum()) * m.dt
    lost = (_storage(m, h0) - _storage(m, h[..., -1])
            + float((torch.exp(lSd) * m.area * (h0 - u)).sum()))
    assert lost == pytest.approx(pumped, rel=1e-6)


def _rivers(m):
    A = m.grid.n_active
    idx = np.array([0, 1, 2, 7, A - 1], dtype="int64")
    return RiverSet(idx=idx, weight=np.array([0.4, 0.6, 1.0, 0.25, 1.0]),
                    group=np.array([0, 0, 0, 0, 1], dtype="int64"),
                    h_riv=np.array([12.0, 11.0, 10.0, 9.0, 3.0]),
                    rbot=np.array([10.0, 9.0, 8.0, 7.0, 1.0]), names=("main", "edge"))


@pytest.mark.parametrize("mode", ["ghb", "riv"])
def test_river_flux_closes_the_budget(mode):
    """One step at a time: storage change = dt * sum(river flux) with the RIV switch
    evaluated on the previous head, the MODFLOW convention the code claims."""
    m = _model(n_layers=2)
    m.set_rivers(_rivers(m), layer=0, mode=mode)
    L, A = m.n_layers, m.grid.n_active
    rng = np.random.default_rng(2)
    h = torch.tensor(rng.normal(9.0, 3.0, (L, A)), dtype=D)
    h[0, 0], h[0, 1], h[0, 2] = 2.0, 30.0, 8.5         # below rbot, above stage, between
    lC = torch.tensor([math.log(300.0), math.log(50.0)], dtype=D)
    rs = m.rivers
    c = np.exp(lC.numpy())[rs.group] * rs.weight
    for _ in range(4):
        h1 = _rollout(m, m.log_T, m.log_S, m.log_L, h, 1, log_C_riv=lC)[..., -1].detach()
        hr = h[0, rs.idx].numpy()
        hn = h1[0, rs.idx].numpy()
        conn = np.ones_like(hr, dtype=bool) if mode == "ghb" else hr > rs.rbot
        flux = np.where(conn, c * (rs.h_riv - hn), c * (rs.h_riv - rs.rbot))
        dS = _storage(m, h1) - _storage(m, h)
        assert dS == pytest.approx(m.dt * flux.sum(), rel=1e-6, abs=1.0)
        h = h1


def test_river_signs():
    """A river above the aquifer raises heads (losing river); one below lowers them."""
    m = _model(n_layers=2)
    m.set_rivers(_rivers(m), layer=0, mode="ghb")
    L, A = m.n_layers, m.grid.n_active
    lC = torch.tensor([math.log(500.0), math.log(500.0)], dtype=D)
    low = torch.full((L, A), 0.0, dtype=D)
    high = torch.full((L, A), 50.0, dtype=D)
    for h0, sgn in ((low, 1.0), (high, -1.0)):
        ref = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3)
        riv = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, log_C_riv=lC)
        d = (riv - ref)[0, 2, -1]
        assert sgn * float(d) > 0.0


def test_canal_water_budget_and_sign():
    m = _model(n_layers=2)
    L, A = m.n_layers, m.grid.n_active
    rng = np.random.default_rng(4)
    h0 = torch.tensor(rng.normal(5.0, 1.0, (L, A)), dtype=D)
    sw = torch.tensor(rng.uniform(0.0, 3e-3, (A, 5)), dtype=D)        # m/day delivered
    scale = 0.3
    ref = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 5)
    h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 5, sw_field=sw,
                 log_sw_scale=torch.tensor(math.log(scale), dtype=D), sw_layer=0)
    added = scale * float(sw.sum()) * m.area * m.dt
    gain = _storage(m, h[..., -1]) - _storage(m, ref[..., -1])
    assert gain == pytest.approx(added, rel=1e-6)
    assert bool((h[0, :, -1] >= ref[0, :, -1] - 1e-9).all())


# ---------------------------------------------------------------------------------------
# gradients: torch.autograd.gradcheck through the implicit adjoint
# ---------------------------------------------------------------------------------------
def test_gradcheck_delay_river_sw_together():
    m = _model(nx=3, ny=3, n_layers=2, seed=5)
    rs = RiverSet(idx=np.array([0, 4, 8], dtype="int64"),
                  weight=np.array([0.5, 1.0, 0.3]), group=np.array([0, 0, 1], dtype="int64"),
                  h_riv=np.array([11.0, 9.0, 6.0]), rbot=np.array([9.0, 7.0, 4.0]),
                  names=("a", "b"))
    m.set_rivers(rs, layer=0, mode="riv")
    L, A = m.n_layers, m.grid.n_active
    rng = np.random.default_rng(6)
    h0 = torch.tensor(rng.normal(8.0, 2.0, (L, A)), dtype=D)
    h0[0, 0] = 2.0                                         # disconnected at the start
    u0 = h0 + torch.tensor(rng.normal(0.0, 1.0, (L, A)), dtype=D)
    pump = torch.zeros(L, A, 3, dtype=D)
    pump[1, 4, :] = 3000.0
    sw = torch.tensor(rng.uniform(0.0, 2e-3, (A, 3)), dtype=D)
    wts = torch.tensor(rng.normal(size=(L, A, 4)), dtype=D)

    def f(lSd, ltau, lC, lsw, lT):
        h = _rollout(m, lT, m.log_S, m.log_L, h0, 3, pumping=pump,
                     delay_Sd=lSd.expand(L, A), delay_tau=ltau.expand(L, A), u0=u0,
                     log_C_riv=lC, sw_field=sw, log_sw_scale=lsw, sw_layer=0)
        return (h * wts).sum()

    args = (torch.tensor([[math.log(2e-2)], [math.log(5e-3)]], dtype=D, requires_grad=True),
            torch.tensor([[math.log(200.0)], [math.log(900.0)]], dtype=D, requires_grad=True),
            torch.tensor([math.log(200.0), math.log(40.0)], dtype=D, requires_grad=True),
            torch.tensor(math.log(0.25), dtype=D, requires_grad=True),
            m.log_T.detach().clone().requires_grad_(True))
    # tighten CG (its tol is a bound default argument) so FD noise is below gradcheck's
    old = flow_mod._cg.__defaults__
    try:
        flow_mod._cg.__defaults__ = (None, None, 1e-11, 5000)
        # full gradcheck's "backward multiplied by grad_output" probe feeds grad_y = 0,
        # which the warm-started adjoint CG (x0 = forward head, relative tol on a zero
        # rhs) cannot resolve -- a property of _ImplicitSolve predating these changes.
        # So compare the full analytic gradient with central differences directly.
        out = f(*args)
        grads = torch.autograd.grad(out, args)
        eps = 1e-6
        for j, (a, g) in enumerate(zip(args, grads, strict=True)):
            flat = a.detach().reshape(-1)
            for i in range(flat.numel()):
                lo = [x.detach().clone() for x in args]
                hi = [x.detach().clone() for x in args]
                lo[j].view(-1)[i] -= eps
                hi[j].view(-1)[i] += eps
                with torch.no_grad():
                    fd = (float(f(*hi)) - float(f(*lo))) / (2 * eps)
                an = float(g.reshape(-1)[i])
                assert an == pytest.approx(fd, rel=1e-4, abs=1e-5), (j, i, an, fd)
    finally:
        flow_mod._cg.__defaults__ = old
