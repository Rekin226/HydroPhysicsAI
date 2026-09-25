"""Adversarial review of the second-round flow physics (2026-09-23): every new storage and
exchange term switched on at once, in a closed box, against an explicit water ledger; the
implicit adjoint against central differences with all groups in one operator (RIV switch
included); steady state preserved; and the canal-to-pump substitution checked in the
solver's own units (m3/day), not only in m3/month.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.calibrate_flow import _rollout  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.rivers import RiverSet  # noqa: E402

D = torch.float64


def _model(n_layers=3, nx=6, ny=5):
    g = FanGrid(nx=nx, ny=ny, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((ny, nx), dtype=bool))
    m = FlowModel(g, n_layers=n_layers, dt_days=30.0)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        m.log_T.copy_(torch.tensor(np.log(rng.uniform(100, 800, m.log_T.shape)), dtype=D))
        m.log_S.copy_(torch.tensor(np.log(rng.uniform(1e-4, 1e-2, m.log_S.shape)), dtype=D))
        m.log_L.copy_(torch.tensor(np.log(rng.uniform(1e-4, 1e-2, m.log_L.shape)), dtype=D))
    return m


def _rivers(m, h_riv=8.0, rbot=6.0):
    A = m.grid.n_active
    idx = np.array([0, 7, 13, A - 1], dtype="int64")
    return RiverSet(idx=idx, weight=np.array([0.3, 0.5, 1.0, 0.2]),
                    group=np.array([0, 0, 1, 1], dtype="int64"),
                    h_riv=np.full(4, h_riv), rbot=np.full(4, rbot),
                    names=("a", "b"))


def _setup(m, steps, mode="ghb", seed=3):
    rng = np.random.default_rng(seed)
    L, A = m.n_layers, m.grid.n_active
    m.set_rivers(_rivers(m), layer=0, mode=mode)
    ext = dict(
        delay_Sd=torch.tensor(np.log(rng.uniform(1e-3, 3e-2, (L, A))), dtype=D),
        delay_tau=torch.tensor(np.log(rng.uniform(200, 3000, (L, A))), dtype=D),
        aqt_Sa=torch.tensor(np.log(rng.uniform(1e-3, 3e-2, (L - 1, A))), dtype=D),
        aqt_G=torch.tensor(np.log(rng.uniform(1e-4, 1e-2, (L - 1, A))), dtype=D),
        log_C_riv=torch.tensor([math.log(300.0), math.log(40.0)], dtype=D),
        sw_field=torch.tensor(rng.uniform(0, 2e-3, (2, A, steps)), dtype=D),
        log_sw_scale=torch.tensor([math.log(0.15), math.log(0.3)], dtype=D),
    )
    h0 = torch.tensor(rng.normal(7.0, 1.5, (L, A)), dtype=D)
    u0 = h0 + torch.tensor(rng.uniform(0.0, 3.0, (L, A)), dtype=D)
    ua0 = 0.5 * (h0[:-1] + h0[1:]) + torch.tensor(rng.uniform(-1, 1, (L - 1, A)), dtype=D)
    pump = torch.zeros(L, A, steps, dtype=D)
    pump[-1, A // 2, :] = 3000.0
    pump[1, 3, :] = 1500.0
    return ext, h0, {"u": u0, "ua": ua0}, pump


def _ledger_step(m, ext, h, h_new, u, ua, q_step, mask):
    """Explicit water balance for one backward-Euler step: storage change of aquifer, delay
    bed and aquitard store vs dt x (sources + river exchange at the new head)."""
    S = torch.exp(m.log_S).detach()
    Sd = torch.exp(ext["delay_Sd"])
    tau = torch.exp(ext["delay_tau"])
    Sa = torch.exp(ext["aqt_Sa"])
    G = torch.exp(ext["aqt_G"])
    dt, area = m.dt, m.area
    u_new = (tau * u + dt * h_new) / (tau + dt)
    a = Sa * area / dt
    g = G * area
    ua_new = (a * ua + g * (h_new[:-1] + h_new[1:])) / (a + 2 * g)
    d_store = float((S * area * (h_new - h)).sum() + (Sd * area * (u_new - u)).sum()
                    + (Sa * area * (ua_new - ua)).sum())
    c = torch.exp(ext["log_C_riv"])[m.riv_group] * m.riv_w
    hr = h_new[m.river_layer, m.riv_idx]
    conn = torch.ones_like(c) if mask is None else mask.to(D)
    riv = float((c * conn * (m.riv_h - hr) + c * (1 - conn) * (m.riv_h - m.riv_rbot)).sum())
    return d_store, dt * (float(q_step.sum()) + riv), u_new, ua_new


@pytest.mark.parametrize("mode", ["ghb", "riv"])
def test_all_new_terms_together_close_the_water_ledger(mode):
    m = _model()
    steps = 6
    ext, h0, st0, pump = _setup(m, steps, mode=mode)
    h, st = _rollout(m, m.log_T.detach(), m.log_S.detach(), m.log_L.detach(), h0, steps,
                     pumping=pump, u0=st0, return_state=True, **ext)
    u, ua = st0["u"], st0["ua"]
    sw = (torch.exp(ext["log_sw_scale"]).reshape(-1, 1, 1) * ext["sw_field"]).sum(0)
    for t in range(steps):
        q = -pump[:, :, t].sum() + (sw[:, t] * m.area).sum()
        mask = m.river_mask(h[..., t])
        d_store, d_in, u, ua = _ledger_step(m, ext, h[..., t], h[..., t + 1], u, ua, q, mask)
        assert abs(d_store - d_in) <= 1e-7 * max(abs(d_in), 1.0), (t, d_store, d_in)
    assert torch.allclose(st["u"], u, atol=1e-10) and torch.allclose(st["ua"], ua, atol=1e-10)


def test_uniform_equilibrium_is_a_fixed_point_of_every_new_store():
    m = _model()
    L, A = m.n_layers, m.grid.n_active
    ext, _, _, _ = _setup(m, 4)
    m.set_rivers(None)
    for k in ("log_C_riv",):
        ext.pop(k)
    ext.pop("sw_field")
    ext.pop("log_sw_scale")
    h0 = torch.full((L, A), 4.0, dtype=D)
    h, st = _rollout(m, m.log_T.detach(), m.log_S.detach(), m.log_L.detach(), h0, 12,
                     return_state=True, **ext)
    assert float((h - 4.0).abs().max()) < 1e-8
    assert float((st["u"] - 4.0).abs().max()) < 1e-8
    assert float((st["ua"] - 4.0).abs().max()) < 1e-8


@pytest.mark.parametrize("mode", ["ghb", "riv"])
def test_adjoint_matches_central_differences_with_every_group_on(mode):
    m = _model(n_layers=2, nx=5, ny=4)
    steps = 4
    ext, h0, st0, pump = _setup(m, steps, mode=mode)
    # heads straddle rbot=6 so the RIV switch is exercised (mode 'riv')
    h0 = h0 - 1.0
    st0 = {"u": st0["u"] - 1.0, "ua": st0["ua"] - 1.0}
    names = ["delay_Sd", "delay_tau", "aqt_Sa", "aqt_G", "log_C_riv", "log_sw_scale"]
    base = {k: ext[k].clone() for k in names}
    lT, lS, lL = (m.log_T.detach().clone(), m.log_S.detach().clone(),
                  m.log_L.detach().clone())
    w = torch.tensor(np.random.default_rng(9).normal(size=(m.n_layers, m.grid.n_active,
                                                            steps + 1)), dtype=D)

    def loss(params, lT=lT, lS=lS, lL=lL):
        kw = dict(ext)
        kw.update(params)
        h = _rollout(m, lT, lS, lL, h0, steps, pumping=pump, u0=st0, **kw)
        return (w * h).sum()

    leaves = {k: base[k].clone().requires_grad_(True) for k in names}
    lTg, lSg, lLg = (x.clone().requires_grad_(True) for x in (lT, lS, lL))
    val = loss(leaves, lTg, lSg, lLg)
    grads = torch.autograd.grad(val, [leaves[k] for k in names] + [lTg, lSg, lLg])
    # directional derivative along a random direction, per group
    rng = np.random.default_rng(11)
    eps = 1e-5
    for k, gk in zip(names + ["log_T", "log_S", "log_L"], grads, strict=True):
        src = base[k] if k in base else {"log_T": lT, "log_S": lS, "log_L": lL}[k]
        v = torch.tensor(rng.normal(size=src.shape), dtype=D)

        def at(s, k=k, src=src, v=v):
            p = {q: base[q] for q in names}
            tt = {"log_T": lT, "log_S": lS, "log_L": lL}
            if k in p:
                p[k] = src + s * v
            else:
                tt[k] = src + s * v
            with torch.no_grad():
                return float(loss(p, tt["log_T"], tt["log_S"], tt["log_L"]))

        fd = (at(eps) - at(-eps)) / (2 * eps)
        ad = float((gk * v).sum())
        assert ad == pytest.approx(fd, rel=2e-4, abs=1e-6), (k, ad, fd)


def test_substituted_pumping_rate_equals_the_lost_canal_rate_in_solver_units():
    """sw=0, sw_sub=1: the solver applies canal recharge as m/day x area and pumping as
    volume/dt with dt = 30 days. The substituted pumping RATE should equal the lost
    delivery RATE (m3/day) in every month (review 2026-09-23: the volume used to be built
    with days_in_month, giving days/30 of it: +3.3 % in 31-day months, -6.7 % in Feb)."""
    from hydrophysics.twin.forward import sw_substitution_energy
    from hydrophysics.twin.pumping import energy_to_volume
    from hydrophysics.twin.scenario import PumpingScenario, climatology

    m = _model(n_layers=2)
    g = m.grid
    A = g.n_active
    dates = pd.date_range("2012-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(2)

    class Inp:
        grid = g
        sw_field = torch.tensor(rng.uniform(1e-4, 3e-3, (A, 24)), dtype=D)
        ground_elev = torch.tensor(rng.uniform(5, 30, A), dtype=D)

    Inp.dates = dates
    scen = PumpingScenario("dry", sw_factor=0.0, sw_sub=1.0)
    scalars = {"log_eta": torch.tensor(math.log(0.5), dtype=D),
               "log_head_extra": torch.tensor(math.log(40.0), dtype=D)}
    h_ref = torch.tensor(rng.normal(3.0, 2.0, (2, A)), dtype=D)
    kwh = sw_substitution_energy(Inp, scen, 12, scalars, h_ref, pump_layer=1)
    clim, _ = climatology(Inp.sw_field.numpy(), dates, 12)
    lost_rate = torch.tensor(clim * g.dx ** 2, dtype=D)                   # m3/day
    vol = energy_to_volume(kwh, (Inp.ground_elev - h_ref[1])[:, None], scalars["log_eta"],
                           head_extra=torch.exp(scalars["log_head_extra"]))
    pumped_rate = vol / m.dt                                               # as _rollout
    assert torch.allclose(pumped_rate, lost_rate, rtol=1e-9)
