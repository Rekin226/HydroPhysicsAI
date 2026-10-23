"""Rung R* (2026-10-22): --eta-county, a learned Changhua factor on the pump efficiency."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from test_twin_well_datum import _problem  # noqa: E402

from hydrophysics.twin.calibrate_flow import eta_by_cell, fit_flow  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402

D = torch.float64


def test_factor_applies_to_changhua_cells_only():
    le = torch.tensor(math.log(0.5), dtype=D)
    ch = torch.tensor([True, False, True])
    out = eta_by_cell(le, torch.tensor(math.log(0.5), dtype=D), ch)
    assert torch.exp(out).tolist() == pytest.approx([0.25, 0.5, 0.25])
    assert eta_by_cell(le, None, None) is le
    with pytest.raises(ValueError, match="county flags"):
        eta_by_cell(le, torch.tensor(0.0, dtype=D), None)


@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_factor_exists_only_when_flagged_and_starts_neutral():
    g, h0, rch, rf, idx, lay, _ = _problem()
    T = rch.shape[-1]
    obs = torch.full((4, T), 10.0, dtype=D)
    E = torch.full((g.n_active, T), 2e3, dtype=D)
    ge = torch.full((g.n_active,), 20.0, dtype=D)
    ch = torch.zeros(g.n_active, dtype=torch.bool)
    ch[: g.n_active // 2] = True
    kw = dict(epochs=1, lr=0.0, h0=h0, recharge_field=rf, E=E, ground_elev=ge, fix_eta=0.5)
    plain = fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, idx, lay, rch, **kw)
    r = fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, idx, lay, rch, **kw,
                 eta_county=True, changhua_cells=ch)
    assert "log_eta_ratio_ch" not in plain["theta"]
    assert r["theta"]["eta_ratio_changhua"] == pytest.approx(1.0)
    assert r["loss"] == pytest.approx(plain["loss"])          # neutral at epoch 0
    with pytest.raises(ValueError, match="county flags"):
        fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, idx, lay, rch, **kw,
                 eta_county=True)
