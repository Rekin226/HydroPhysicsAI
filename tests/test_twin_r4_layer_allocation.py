"""Rung R4 (2026-10-21): per-layer shares of the pumped volume (--pump-layer-fractions, with a
Changhua copy --pump-fractions-county) and a lagged return flow (--return-lag).

- the shares are a softmax: every cell's column sums to one, and the Changhua copy applies to
  Changhua cells only;
- all of the volume in the pump layer reproduces the historical single-layer path bit for
  bit, and a vanishing lag reproduces the same-month return flow;
- the lag's store carries across segments: two half-runs chained through return_state equal
  one run;
- moving pumping to the shallow layer lowers the shallow head (the sign the data show);
- the parameters exist only when the flags are set."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_twin_well_datum import _problem  # noqa: E402

from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    _rollout,
    fit_flow,
    pump_layer_fractions,
)
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402

D = torch.float64


def _setup(steps=6):
    g = FanGrid(nx=5, ny=4, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((4, 5), dtype=bool))
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        m.log_T.copy_(torch.tensor(np.log(rng.uniform(100, 800, m.log_T.shape)), dtype=D))
        m.log_S.copy_(torch.tensor(np.log(rng.uniform(1e-4, 1e-2, m.log_S.shape)), dtype=D))
        m.log_L.fill_(math.log(1e-3))
    A = g.n_active
    h0 = torch.tensor(rng.normal(10.0, 1.0, (4, A)), dtype=D)
    E = torch.tensor(rng.uniform(1e3, 5e3, (A, steps)), dtype=D)
    kw = dict(E=E, log_eta=torch.tensor(math.log(0.5), dtype=D),
              ground_elev=torch.full((A,), 20.0, dtype=D), pump_layer=1,
              log_head_extra=torch.tensor(math.log(40.0), dtype=D),
              return_frac_logit=torch.tensor(0.0, dtype=D))
    return m, h0, steps, kw


def _run(m, h0, n, **kw):
    return _rollout(m, m.log_T, m.log_S, m.log_L, h0, n, **kw)


def test_shares_sum_to_one_and_the_changhua_copy_is_local():
    lg = torch.log(torch.tensor([0.1, 0.7, 0.15, 0.05], dtype=D))
    f = pump_layer_fractions(lg)
    assert f.shape == (4,) and float(f.sum()) == pytest.approx(1.0)
    assert f.tolist() == pytest.approx([0.1, 0.7, 0.15, 0.05])
    ch = torch.tensor([True, False, True])
    lc = torch.log(torch.tensor([0.6, 0.3, 0.05, 0.05], dtype=D))
    fc = pump_layer_fractions(lg, lc, ch)
    assert fc.shape == (4, 3)
    assert torch.allclose(fc.sum(dim=0), torch.ones(3, dtype=D))
    assert fc[:, 0].tolist() == pytest.approx([0.6, 0.3, 0.05, 0.05])
    assert fc[:, 1].tolist() == pytest.approx([0.1, 0.7, 0.15, 0.05])
    with pytest.raises(ValueError, match="county flags"):
        pump_layer_fractions(lg, lc, None)


def test_limits_reproduce_the_historical_path():
    m, h0, n, kw = _setup()
    base = _run(m, h0, n, **kw)
    one_layer = _run(m, h0, n, pump_layer_frac=torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=D),
                     **kw)
    assert torch.equal(base, one_layer)
    tiny_lag = _run(m, h0, n, ret_tau=torch.tensor(1e-6, dtype=D), **kw)
    assert torch.allclose(base, tiny_lag, rtol=0, atol=1e-12)


def test_lag_state_carries_across_segments():
    m, h0, n, kw = _setup(steps=8)
    tau = torch.tensor(3.0, dtype=D)
    full, st = _run(m, h0, n, ret_tau=tau, return_state=True, **kw)
    assert isinstance(st, dict) and st["ret"] is not None
    k1 = {**kw, "E": kw["E"][:, :4]}
    k2 = {**kw, "E": kw["E"][:, 4:]}
    a, sa = _run(m, h0, 4, ret_tau=tau, return_state=True, **k1)
    b, sb = _run(m, a[..., -1], 4, ret_tau=tau, return_state=True, u0=sa, **k2)
    assert torch.allclose(full, torch.cat([a, b[..., 1:]], dim=-1), rtol=0, atol=1e-9)
    assert torch.allclose(st["ret"], sb["ret"], rtol=0, atol=1e-9)


def test_shallow_pumping_lowers_the_shallow_head():
    m, h0, n, kw = _setup()
    kw = {k: v for k, v in kw.items() if k != "return_frac_logit"}
    deep = _run(m, h0, n, pump_layer_frac=torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=D), **kw)
    shallow = _run(m, h0, n, pump_layer_frac=torch.tensor([0.6, 0.4, 0.0, 0.0], dtype=D), **kw)
    assert float((shallow[0, :, -1] - deep[0, :, -1]).mean()) < 0.0
    assert float((shallow[1, :, -1] - deep[1, :, -1]).mean()) > 0.0


@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_parameters_exist_only_when_flagged():
    g, h0, rch, rf, idx, lay, _ = _problem()
    T = rch.shape[-1]
    obs = torch.full((4, T), 10.0, dtype=D)
    E = torch.full((g.n_active, T), 2e3, dtype=D)
    ge = torch.full((g.n_active,), 20.0, dtype=D)
    common = dict(epochs=1, lr=0.0, h0=h0, recharge_field=rf, E=E, ground_elev=ge,
                  return_flow=True)
    plain = fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, idx, lay, rch, **common)
    r4 = fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, idx, lay, rch, **common,
                  pump_layer_fractions_on=True, return_lag=True)
    assert "pump_layer_logits" not in plain["theta"] and "log_tau_ret" not in plain["theta"]
    assert r4["theta"]["pump_layer_shares"] == pytest.approx([0.1, 0.7, 0.15, 0.05])
    assert r4["theta"]["tau_ret_steps"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="replaces --pump-split"):
        fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, idx, lay, rch, **common,
                 pump_split=True, pump_layer_fractions_on=True)
    with pytest.raises(ValueError, match="needs --return-flow"):
        fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, idx, lay, rch,
                 **{**common, "return_flow": False}, return_lag=True)


@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_fixed_shares_are_held_and_recorded():
    """Rung R4p: --pump-layer-shares holds the shares at a measured prior; they are written
    into the theta (as a fixed parameter) so the forward twin rebuilds them."""
    g, h0, rch, rf, idx, lay, _ = _problem()
    T = rch.shape[-1]
    obs = torch.full((4, T), 10.0, dtype=D)
    E = torch.full((g.n_active, T), 2e3, dtype=D)
    ge = torch.full((g.n_active,), 20.0, dtype=D)
    sh = (0.84, 0.11, 0.025, 0.025)
    fit = fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, idx, lay, rch, epochs=2,
                   lr=0.05, h0=h0, recharge_field=rf, E=E, ground_elev=ge, return_flow=True,
                   pump_layer_fractions_on=True, return_lag=True, fix_pump_layer_shares=sh)
    assert fit["theta"]["pump_layer_shares"] == pytest.approx(list(sh))
    assert "pump_layer_logits" in fit["theta"]
    with pytest.raises(ValueError, match="summing to 1"):
        fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, idx, lay, rch, epochs=1,
                 lr=0.0, h0=h0, recharge_field=rf, E=E, ground_elev=ge, return_flow=True,
                 pump_layer_fractions_on=True, fix_pump_layer_shares=(0.5, 0.5, 0.5, 0.5))
