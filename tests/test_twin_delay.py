"""Delay-bed storage, river cells and the surface-water hook in the flow rollout (2026-09-23).

Every term is opt-in; these tests pin (1) that the historical path is untouched, (2) the
limits the delay bed must reduce to, (3) an analytic single-cell recurrence, (4) that the
operator stays symmetric positive definite with the new diagonals, and (5) that the
implicit adjoint's gradients match central finite differences for every new parameter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    BOUNDS,
    _add_extension_params,
    _expand_zonal_delay,
    _extension_readouts,
    _rollout,
    delay_fields_from_theta,
    fit_flow,
)
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.rivers import RiverSet  # noqa: E402

D = torch.float64


def _grid(nx=6, ny=5, dx=1000.0):
    return FanGrid(nx=nx, ny=ny, dx=dx, x0=0.0, y0=0.0, mask=np.ones((ny, nx), dtype=bool))


def _model(n_layers=2, **kw):
    g = _grid(**kw)
    m = FlowModel(g, n_layers=n_layers, dt_days=30.0)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        m.log_T.copy_(torch.tensor(np.log(rng.uniform(100, 800, m.log_T.shape)), dtype=D))
        m.log_S.copy_(torch.tensor(np.log(rng.uniform(1e-4, 1e-2, m.log_S.shape)), dtype=D))
        if n_layers > 1:
            m.log_L.fill_(math.log(1e-3))
    return m


def _forcing(m, steps=3, seed=1):
    rng = np.random.default_rng(seed)
    A = m.grid.n_active
    h0 = torch.tensor(rng.normal(10.0, 2.0, (m.n_layers, A)), dtype=D)
    pump = torch.zeros(m.n_layers, A, steps, dtype=D)
    pump[-1, A // 2, :] = 2000.0
    rech = torch.tensor(rng.uniform(0, 2e-3, (m.n_layers, A, steps)), dtype=D)
    return h0, rech, pump


def _rivers(m):
    A = m.grid.n_active
    idx = np.array([1, 2, 3, A - 2], dtype="int64")
    return RiverSet(idx=idx, weight=np.array([0.3, 0.5, 1.0, 0.2]),
                    group=np.array([0, 0, 0, 1], dtype="int64"),
                    h_riv=np.array([12.0, 11.0, 10.5, 9.0]),
                    rbot=np.array([10.0, 9.0, 8.5, 7.0]), names=("a", "b"))


def _full(m, Sd, tau, **kw):
    L, A = m.n_layers, m.grid.n_active
    return (torch.full((L, A), math.log(Sd), dtype=D), torch.full((L, A), math.log(tau), dtype=D))


def test_no_new_arguments_is_the_historical_path():
    m = _model()
    h0, rech, pump = _forcing(m)
    a = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump)
    b = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                 delay_Sd=None, delay_tau=None, log_C_riv=None, sw_field=None)
    assert torch.equal(a, b)


def test_vanishing_delay_storage_matches_no_delay():
    m = _model()
    h0, rech, pump = _forcing(m)
    ref = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump)
    # S_d = 1e-12 against S ~ 1e-3 is a genuine 1e-9 relative perturbation of storage
    # (measured 4.5e-10 m); S_d = 1e-20 is below float64 resolution and reproduces exactly
    for sd, tol in ((1e-12, 1e-8), (1e-20, 1e-12)):
        Sd, tau = _full(m, sd, 365.0)
        h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                     delay_Sd=Sd, delay_tau=tau)
        assert torch.allclose(h, ref, atol=tol, rtol=0), sd


def test_instant_delay_bed_is_extra_elastic_storage():
    """tau -> 0: the store follows the aquifer, i.e. storage S + S_d. The lag is
    O(tau/dt) of a step's head change, so tau = 1e-6 d sits well inside 1e-6 m (at the
    design's 1e-3 d it is ~1e-4 m, which is physics, not error)."""
    m = _model()
    h0, rech, pump = _forcing(m)
    Sd_val = 5e-3
    Sd, tau = _full(m, Sd_val, 1e-6)
    h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                 delay_Sd=Sd, delay_tau=tau)
    log_S_tot = torch.log(torch.exp(m.log_S) + Sd_val)
    ref = _rollout(m, m.log_T, log_S_tot, m.log_L, h0, 3, recharge=rech, pumping=pump)
    assert torch.allclose(h, ref, atol=1e-6, rtol=0)


def test_single_cell_matches_the_analytic_recurrence():
    """1x1 grid, 1 layer, no faces: h and the slow store follow the condensed recurrence."""
    g = FanGrid(nx=1, ny=1, dx=100.0, x0=0.0, y0=0.0, mask=np.ones((1, 1), dtype=bool))
    m = FlowModel(g, n_layers=1, dt_days=30.0)
    S, Sd, tau, dt, area = 1e-3, 2e-2, 400.0, 30.0, 1e4
    with torch.no_grad():
        m.log_S.fill_(math.log(S))
    steps = 6
    pump = torch.zeros(1, 1, steps, dtype=D)
    pump[0, 0, :3] = 50.0                           # m3/day for three months, then off
    h0 = torch.zeros(1, 1, dtype=D)
    heads, u_end = _rollout(m, m.log_T, m.log_S, None, h0, steps, pumping=pump,
                            delay_Sd=torch.full((1, 1), math.log(Sd), dtype=D),
                            delay_tau=torch.full((1, 1), math.log(tau), dtype=D),
                            return_state=True)
    h, u = 0.0, 0.0
    beta = Sd / (tau + dt)
    for t in range(steps):
        q = -float(pump[0, 0, t])
        h = (S * area / dt * h + beta * area * u + q) / (S * area / dt + beta * area)
        u = (tau * u + dt * h) / (tau + dt)
        assert float(heads[0, 0, t + 1].detach()) == pytest.approx(h, rel=1e-9, abs=1e-12)
    assert float(u_end[0, 0]) == pytest.approx(u, rel=1e-9, abs=1e-12)
    # after pumping stops, the slow store keeps releasing: head recovers but stays
    # below the start while the store is still above it
    assert float(heads[0, 0, -1]) < 0.0 and float(u_end[0, 0]) < 0.0


def _dense(op, params, n_layers, A):
    n = n_layers * A
    M = torch.zeros(n, n, dtype=D)
    for i in range(n):
        e = torch.zeros(n, dtype=D)
        e[i] = 1.0
        M[:, i] = op(e.reshape(n_layers, A), *params).reshape(-1)
    return M


@pytest.mark.parametrize("mode", ["ghb", "riv"])
def test_operator_with_delay_and_river_diagonals_is_spd(mode):
    m = _model()
    m.set_rivers(_rivers(m), layer=0, mode=mode)
    L, A = m.n_layers, m.grid.n_active
    Sd, tau = _full(m, 1e-2, 200.0)
    lc = torch.tensor([math.log(500.0), math.log(50.0)], dtype=D)
    layout = m.operator_layout(delay=True, riv=True)
    mask = torch.tensor([True, False, True, False]) if mode == "riv" else None
    op = m.make_op(layout, riv_mask=mask)
    params = m.operator_params(m.log_T, m.log_S, m.log_L, delay=(Sd, tau), riv=(lc,))
    with torch.no_grad():
        M = _dense(op, params, L, A)
        base = _dense(m.make_op(()), m.operator_params(m.log_T, m.log_S, m.log_L), L, A)
    assert torch.allclose(M, M.T, atol=1e-9)
    assert float(torch.linalg.eigvalsh(M).min()) > 0
    extra = torch.diagonal(M - base)
    assert (extra >= -1e-12).all() and torch.allclose(M - base, torch.diag(extra), atol=1e-9)


def test_riv_mode_disconnected_cell_gets_constant_leakage_and_no_diagonal():
    m = _model()
    rs = _rivers(m)
    m.set_rivers(rs, layer=0, mode="riv")
    C = torch.tensor([1000.0, 10.0], dtype=D)
    h = torch.full((m.n_layers, m.grid.n_active), 0.0, dtype=D)   # far below every rbot
    mask = m.river_mask(h)
    assert not bool(mask.any())
    diag, rhs = m.river_terms(C, mask)
    assert float(diag.abs().sum()) == 0.0
    i = int(rs.idx[0])
    c = 1000.0 * rs.weight[0]
    assert float(rhs[0, i]) == pytest.approx(c * (rs.h_riv[0] - rs.rbot[0]))
    # connected: C w on the diagonal and C w h_riv on the right-hand side
    h_hi = torch.full_like(h, 50.0)
    diag, rhs = m.river_terms(C, m.river_mask(h_hi))
    assert float(diag[0, i]) == pytest.approx(c)
    assert float(rhs[0, i]) == pytest.approx(c * rs.h_riv[0])
    assert float(diag[1].abs().sum()) == 0.0                 # only the river layer


@pytest.mark.parametrize("mode", ["ghb", "riv"])
def test_vanishing_river_conductance_matches_no_rivers(mode):
    m = _model()
    h0, rech, pump = _forcing(m)
    ref = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump)
    m.set_rivers(_rivers(m), layer=0, mode=mode)
    h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                 log_C_riv=torch.full((2,), math.log(1e-12), dtype=D))
    assert torch.allclose(h, ref, atol=1e-7, rtol=0)


def test_zero_surface_water_matches_no_surface_water():
    m = _model()
    h0, rech, pump = _forcing(m)
    ref = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump)
    sw = torch.zeros(m.grid.n_active, 3, dtype=D)
    h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                 sw_field=sw, log_sw_scale=torch.tensor(math.log(0.01), dtype=D))
    assert torch.equal(h, ref)


def test_surface_water_adds_its_scaled_volume_to_the_chosen_layer():
    m = _model()
    h0, rech, pump = _forcing(m)
    sw = torch.full((m.grid.n_active, 3), 1e-3, dtype=D)
    lo = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, sw_field=sw,
                  log_sw_scale=torch.tensor(math.log(0.1), dtype=D))
    hi = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, sw_field=sw,
                  log_sw_scale=torch.tensor(math.log(0.5), dtype=D))
    assert (hi[..., -1] > lo[..., -1]).all()


def test_adjoint_matches_central_differences_for_every_new_parameter():
    """5x6 grid, 2 layers, 3 steps, delay + riv + sw together: relative error < 1e-5."""
    m = _model()
    m.set_rivers(_rivers(m), layer=0, mode="riv")
    h0, rech, pump = _forcing(m)
    A = m.grid.n_active
    sw = torch.tensor(np.random.default_rng(3).uniform(0, 3e-3, (A, 3)), dtype=D)
    w = torch.tensor(np.random.default_rng(4).normal(size=(2, A, 4)), dtype=D)
    base = {"Sd": math.log(3e-3), "tau": math.log(150.0),
            "criv": [math.log(300.0), math.log(40.0)], "sw": math.log(0.3)}

    def loss(p):
        L_, A_ = m.n_layers, A
        return (w * _rollout(
            m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
            delay_Sd=p["Sd"].reshape(1, 1).expand(L_, A_),
            delay_tau=p["tau"].reshape(1, 1).expand(L_, A_),
            log_C_riv=p["criv"], sw_field=sw, log_sw_scale=p["sw"])).sum()

    p = {k: torch.tensor(v, dtype=D, requires_grad=True) for k, v in base.items()}
    loss(p).backward()
    eps = 1e-5
    for k in base:
        g = p[k].grad.reshape(-1)
        for j in range(g.numel()):
            def at(delta, k=k, j=j):
                q = {kk: torch.tensor(vv, dtype=D) for kk, vv in base.items()}
                flat = q[k].reshape(-1).clone()
                flat[j] += delta
                q[k] = flat.reshape(q[k].shape)
                with torch.no_grad():
                    return float(loss(q))
            fd = (at(eps) - at(-eps)) / (2 * eps)
            assert abs(float(g[j]) - fd) <= 1e-5 * max(abs(fd), 1e-3), (k, j, float(g[j]), fd)


def test_extension_params_and_zonal_expansion():
    m = _model()
    theta = _add_extension_params({}, m, "zonal", n_riv=3, use_sw=True, zonal=True)
    assert [k for k in theta if k.startswith("log_Sd")] == [
        "log_Sd_proximal", "log_Sd_mid", "log_Sd_distal"]
    assert theta["log_C_riv"].shape == (3,)
    zone = torch.tensor(np.arange(m.grid.n_active) % 3)
    with torch.no_grad():
        theta["log_Sd_mid"].fill_(math.log(0.2))
    Sd, tau = _expand_zonal_delay(theta, zone, m.n_layers, m.grid.n_active)
    assert Sd.shape == (m.n_layers, m.grid.n_active)
    assert torch.allclose(torch.exp(Sd[:, zone == 1]), torch.tensor(0.2, dtype=D))
    with pytest.raises(ValueError):
        _add_extension_params({}, m, "zonal", zonal=False)
    assert "log_Sd" in BOUNDS and "log_tau" in BOUNDS and "log_C_riv" in BOUNDS
    ro = _extension_readouts({"log_Sd_mid": [math.log(0.1)], "log_tau_mid": [math.log(900.0)],
                              "log_C_riv": [0.0, math.log(10.0)], "log_sw_scale": 0.0})
    assert ro["Sd_mid"] == pytest.approx(0.1) and ro["tau_days_mid"] == pytest.approx(900.0)
    assert ro["C_riv_m2day"] == pytest.approx([1.0, 10.0]) and ro["sw_scale"] == 1.0
    Sd2, tau2 = delay_fields_from_theta({"log_Sd_proximal": [0.0], "log_Sd_mid": [-1.0],
                                         "log_Sd_distal": [-2.0], "log_tau_proximal": [5.0],
                                         "log_tau_mid": [6.0], "log_tau_distal": [7.0]},
                                        np.arange(m.grid.n_active) % 3, m.n_layers,
                                        m.grid.n_active)
    assert float(Sd2[0, 2]) == -2.0 and float(tau2[1, 1]) == 6.0


def test_fit_flow_runs_with_delay_rivers_and_surface_water_and_reports_them():
    m = _model()
    m.set_rivers(_rivers(m), layer=0, mode="ghb")
    h0, rech, _ = _forcing(m, steps=4)
    A = m.grid.n_active
    obs_idx = torch.tensor([0, 7, 14, 21, 28])
    obs_layer = torch.tensor([0, 1, 0, 1, 0])
    obs = h0[obs_layer, obs_idx][:, None].repeat(1, 4) + 0.1
    sw = torch.full((A, 4), 1e-3, dtype=D)
    zone = np.arange(A) % 3
    fit = fit_flow(m, obs, obs_idx, obs_layer, torch.zeros(2, A, 4, dtype=D),
                   recharge_field=rech[0], epochs=2, param_mode="zonal", h0=h0,
                   zone_of_cell=zone, delay_storage="zonal", sw_field=sw)
    th = fit["theta"]
    for k in ("tau_days_mid", "Sd_distal", "C_riv_m2day", "sw_scale"):
        assert k in th
    assert len(th["C_riv_m2day"]) == 2
    assert "log_Sd" in fit["bounds_hit"]["mid"] and "log_tau" in fit["bounds_hit"]["distal"]
    assert m.delay_log_Sd is not None and m.fit_log_C_riv is not None
