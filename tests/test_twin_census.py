"""Census cleaning (2026-09-11) and per-purpose efficiency classes.

The raw TPC census attaches one meter's kWh series to every pump on that meter and
carries meters that draw many times their rated motor capacity. ``clean_census`` fixes
both; these tests pin the arithmetic on a hand-built census where the answer is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.calibrate_flow import fit_flow  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.pumping import (  # noqa: E402
    HOURS_PER_MONTH,
    KW_PER_HP,
    aggregate_pumps,
    clean_census,
    energy_to_volume,
)
from hydrophysics.twin.scenario import energy_by_class  # noqa: E402


def _census():
    """Four pumps on three meters. Meter M1 is shared by two irrigation pumps (3 HP and
    1 HP) and draws 400 kWh/month; M2 is a 10 HP aquaculture pump at 200 kWh/month; M3 is
    a 5 HP 'industry' meter drawing 50,000 kWh/month, 18x its rated capacity."""
    pumps = pd.DataFrame({
        "sid": ["p1", "p2", "p3", "p4"],
        "電號1": ["M1", "M1", "M2", "M3"],
        "PUMP_HP": [3.0, 1.0, 10.0, 5.0],
        "PURPOSE": ["農業用水(灌溉-旱作)", "農業用水(灌溉-旱作)", "農業用水(養殖-淡水)", "工業用水"],
        "TWD97_X": [500.0, 500.0, 1500.0, 1500.0],
        "TWD97_Y": [500.0, 500.0, 500.0, 1500.0],
    })
    months = pd.date_range("2012-01-01", periods=3, freq="MS")
    rows = []
    for sid, e in (("p1", 400.0), ("p2", 400.0), ("p3", 200.0), ("p4", 50_000.0)):
        for d in months:
            rows.append({"datetime": d, "electricity_kwh": e, "pump": sid})
    return pumps, pd.DataFrame(rows)


def test_shared_meter_is_counted_once_and_split_by_horsepower():
    pumps, kwh = _census()
    pc, kc, rep = clean_census(pumps, kwh, cap_duty=None)
    per_pump = kc.groupby("pump")["electricity_kwh"].sum()
    assert per_pump["p1"] == pytest.approx(3 * 300.0)     # 3/4 of 1200 kWh
    assert per_pump["p2"] == pytest.approx(1 * 300.0)
    assert per_pump["p1"] + per_pump["p2"] == pytest.approx(1200.0)
    assert per_pump["p3"] == pytest.approx(600.0)
    assert rep.loc["irrigation", "kwh_raw_GWh"] == pytest.approx(2400.0 / 1e6)
    assert rep.loc["irrigation", "kwh_dedup_GWh"] == pytest.approx(1200.0 / 1e6)
    assert len(pc) == 4 and rep["n_meters_dropped"].sum() == 0


def test_over_capacity_meter_is_dropped_whole():
    pumps, kwh = _census()
    cap = 5.0 * KW_PER_HP * HOURS_PER_MONTH
    assert 50_000.0 / cap > 1.0
    pc, kc, rep = clean_census(pumps, kwh, cap_duty=1.0)
    assert "p4" not in set(pc["sid"]) and "p4" not in set(kc["pump"])
    assert rep.loc["industry", "n_meters_dropped"] == 1
    assert rep.loc["industry", "kwh_kept_GWh"] == 0.0
    assert rep["kwh_kept_GWh"].sum() == pytest.approx(kc["electricity_kwh"].sum() / 1e6)
    # a generous cap keeps it
    _, kc2, _ = clean_census(pumps, kwh, cap_duty=100.0)
    assert "p4" in set(kc2["pump"])


def test_cleaned_tables_aggregate_like_the_raw_ones():
    pumps, kwh = _census()
    g = FanGrid(nx=2, ny=2, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((2, 2), dtype=bool))
    pc, kc, _ = clean_census(pumps, kwh, cap_duty=1.0)
    E, dates = aggregate_pumps(pc, kc, g, "2012-01-01", "2012-04-01")
    assert E.shape == (4, 3)
    assert E[0].sum() == pytest.approx(1200.0)       # both M1 pumps land in cell (0,0)
    assert E[1].sum() == pytest.approx(600.0)        # M2 at (0,1)
    assert E[3].sum() == 0.0                         # M3 dropped
    by_cls, _ = energy_by_class(pc, kc, g, "2012-01-01", "2012-04-01")
    assert set(by_cls) == {"irrigation", "aquaculture"}


def test_energy_to_volume_broadcasts_per_class_efficiencies():
    E = torch.tensor([[100.0, 200.0], [300.0, 400.0]], dtype=torch.float64)   # (C=2, A=2)
    lift = torch.tensor([10.0, 20.0], dtype=torch.float64)
    log_eta = torch.log(torch.tensor([0.5, 0.25], dtype=torch.float64)).reshape(-1, 1)
    v = energy_to_volume(E, lift, log_eta)
    assert v.shape == (2, 2)
    single = energy_to_volume(E[0], lift, log_eta[0, 0])
    assert torch.allclose(v[0], single)
    assert torch.allclose(v[1], energy_to_volume(E[1], lift, log_eta[1, 0]))


def test_fit_flow_with_class_axis_learns_one_eta_per_class():
    g = FanGrid(nx=4, ny=4, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((4, 4), dtype=bool))
    A, steps, C = g.n_active, 3, 3
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    h0 = torch.full((2, A), 5.0, dtype=torch.float64)
    rech = torch.zeros(2, A, steps, dtype=torch.float64)
    E = torch.full((C, A, steps), 50.0, dtype=torch.float64)
    ge = torch.full((A,), 10.0, dtype=torch.float64)
    obs_idx = torch.tensor([0, 7])
    obs_layer = torch.tensor([1, 1])
    obs_h = torch.full((2, steps), 4.9, dtype=torch.float64)
    fit = fit_flow(m, obs_h, obs_idx, obs_layer, rech, E=E, ground_elev=ge, h0=h0,
                   epochs=2, lr=0.01)
    assert len(fit["theta"]["log_eta"]) == C
    assert len(fit["theta"]["eta"]) == C
    assert fit["bounds_hit"]["log_eta"]["n"] == C


def test_fit_flow_can_hold_the_pump_conversion_fixed():
    g = FanGrid(nx=4, ny=4, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((4, 4), dtype=bool))
    A, steps = g.n_active, 3
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    h0 = torch.full((2, A), 5.0, dtype=torch.float64)
    rech = torch.zeros(2, A, steps, dtype=torch.float64)
    E = torch.full((A, steps), 50.0, dtype=torch.float64)
    ge = torch.full((A,), 10.0, dtype=torch.float64)
    obs_idx = torch.tensor([0, 7])
    obs_layer = torch.tensor([1, 1])
    obs_h = torch.full((2, steps), 4.9, dtype=torch.float64)
    fit = fit_flow(m, obs_h, obs_idx, obs_layer, rech, E=E, ground_elev=ge, h0=h0,
                   epochs=2, lr=0.01, fix_eta=0.5, fix_head_extra=40.0)
    assert fit["fixed"] == ["log_eta", "log_head_extra"]
    assert fit["theta"]["eta"] == pytest.approx(0.5)
    assert fit["theta"]["head_extra_m"] == pytest.approx(40.0)
    assert "log_eta" not in fit["bounds_hit"]
    assert fit["n_params"] == 2 + 2 + 1      # T, S per layer + L: nothing for the pumps


def test_pump_split_and_return_flow_move_stress_between_layers():
    """Splitting the abstraction toward layer 1 lowers layer 1's head and raises the
    production layer's; return flow raises layer 1 again. Both learnable, both reported."""
    from hydrophysics.twin.calibrate_flow import _rollout

    g = FanGrid(nx=4, ny=4, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((4, 4), dtype=bool))
    A, steps = g.n_active, 3
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    h0 = torch.full((2, A), 5.0, dtype=torch.float64)
    E = torch.full((A, steps), 500.0, dtype=torch.float64)
    ge = torch.full((A,), 10.0, dtype=torch.float64)
    log_eta = torch.tensor(np.log(0.5), dtype=torch.float64)
    hx = torch.tensor(np.log(40.0), dtype=torch.float64)
    with torch.no_grad():
        base = _rollout(m, m.log_T, m.log_S, m.log_L, h0, steps, E=E, log_eta=log_eta,
                        log_head_extra=hx, ground_elev=ge, pump_layer=1)
        split = _rollout(m, m.log_T, m.log_S, m.log_L, h0, steps, E=E, log_eta=log_eta,
                         log_head_extra=hx, ground_elev=ge, pump_layer=1,
                         pump_split_logit=torch.tensor(3.0, dtype=torch.float64))  # ~95% shallow
        ret = _rollout(m, m.log_T, m.log_S, m.log_L, h0, steps, E=E, log_eta=log_eta,
                       log_head_extra=hx, ground_elev=ge, pump_layer=1,
                       return_frac_logit=torch.tensor(5.0, dtype=torch.float64))   # ~0.7
    assert (split[0, :, -1] < base[0, :, -1]).all() and (split[1, :, -1] > base[1, :, -1]).all()
    assert (ret[0, :, -1] > base[0, :, -1]).all()

    rech = torch.zeros(2, A, steps, dtype=torch.float64)
    obs_idx = torch.tensor([0, 7])
    obs_layer = torch.tensor([1, 1])
    obs_h = torch.full((2, steps), 4.9, dtype=torch.float64)
    fit = fit_flow(m, obs_h, obs_idx, obs_layer, rech, E=E, ground_elev=ge, h0=h0,
                   epochs=2, lr=0.01, pump_split=True, return_flow=True)
    assert 0.0 < fit["theta"]["pump_frac_shallow"] < 1.0
    assert 0.0 < fit["theta"]["return_frac"] < 0.7
    assert fit["n_params"] == 2 + 2 + 1 + 1 + 1 + 1 + 1   # T,S,L, eta, head_extra, split, return


def test_l_min_raises_the_leakance_floor():
    from hydrophysics.twin.calibrate_flow import BOUNDS, set_l_min

    lo0 = BOUNDS["log_L"][0]
    try:
        set_l_min(1e-4)
        assert BOUNDS["log_L"][0] == pytest.approx(np.log(1e-4))
        assert BOUNDS["log_L"][1] == pytest.approx(np.log(1e-1))
    finally:
        BOUNDS["log_L"] = (lo0, BOUNDS["log_L"][1])


def test_spread_kernel_conserves_energy_and_widens_with_sigma():
    from hydrophysics.twin.spread import pairwise_d2_km, spread_energy, spread_matrix

    g = FanGrid(nx=6, ny=6, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((6, 6), dtype=bool))
    d2 = pairwise_d2_km(g)
    E = torch.zeros(g.n_active, 3, dtype=torch.float64)
    E[14, :] = 1000.0                                   # one hot cell
    for sigma in (0.5, 2.0):
        W = spread_matrix(d2, torch.tensor(np.log(sigma), dtype=torch.float64))
        assert torch.allclose(W.sum(dim=0), torch.ones(g.n_active, dtype=torch.float64))
        Es = spread_energy(E, W)
        assert torch.allclose(Es.sum(dim=0), E.sum(dim=0))          # mass conserved
    narrow = spread_energy(E, spread_matrix(d2, torch.tensor(np.log(0.5), dtype=torch.float64)))
    wide = spread_energy(E, spread_matrix(d2, torch.tensor(np.log(2.0), dtype=torch.float64)))
    assert narrow[14, 0] > wide[14, 0]                              # wider kernel = flatter
    E3 = torch.stack([E, 2 * E])
    assert spread_energy(E3, W).shape == E3.shape


def test_fit_flow_learns_or_fixes_the_spread_radius():
    g = FanGrid(nx=4, ny=4, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((4, 4), dtype=bool))
    A, steps = g.n_active, 3
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    h0 = torch.full((2, A), 5.0, dtype=torch.float64)
    rech = torch.zeros(2, A, steps, dtype=torch.float64)
    E = torch.zeros(A, steps, dtype=torch.float64)
    E[5, :] = 500.0
    ge = torch.full((A,), 10.0, dtype=torch.float64)
    obs_idx = torch.tensor([0, 7])
    obs_layer = torch.tensor([1, 1])
    obs_h = torch.full((2, steps), 4.9, dtype=torch.float64)
    fit = fit_flow(m, obs_h, obs_idx, obs_layer, rech, E=E, ground_elev=ge, h0=h0,
                   epochs=2, lr=0.01, learn_spread=True)
    assert 0.5 <= fit["theta"]["spread_km"] <= 10.0
    assert "log_spread_km" in fit["bounds_hit"]
    fixed = fit_flow(m, obs_h, obs_idx, obs_layer, rech, E=E, ground_elev=ge, h0=h0,
                     epochs=2, lr=0.01, spread_km=3.0)
    assert fixed["theta"]["spread_km"] == 3.0 and "log_spread_km" not in fixed["theta"]


def test_temporal_gate_scores_a_continuation_against_climatology():
    from hydrophysics.twin.calibrate_flow import temporal_gate

    g = FanGrid(nx=4, ny=4, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((4, 4), dtype=bool))
    A, T_full, T_fit = g.n_active, 30, 24
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    h0 = torch.full((2, A), 5.0, dtype=torch.float64)
    E = torch.full((A, T_full), 50.0, dtype=torch.float64)
    R = torch.full((A, T_full), 1e-4, dtype=torch.float64)
    ge = torch.full((A,), 10.0, dtype=torch.float64)
    obs_idx = torch.tensor([0, 7])
    obs_layer = torch.tensor([1, 1])
    obs_full = 5.0 + 0.3 * torch.sin(torch.arange(T_full, dtype=torch.float64) * 2 * np.pi / 12)
    obs_full = obs_full[None, :].repeat(2, 1)
    fit = fit_flow(m, obs_full[:, :T_fit], obs_idx, obs_layer,
                   torch.zeros(2, A, T_fit, dtype=torch.float64), E=E[:, :T_fit],
                   ground_elev=ge, h0=h0, epochs=2, lr=0.01, recharge_field=R[:, :T_fit])
    out = temporal_gate(m, fit, h0, obs_full, obs_idx, obs_layer, T_fit, E, R, ge)
    assert {"r2_model", "r2_clim", "r2_persist", "r2_anom_model", "r2_anom_clim",
            "r2_well_median_model", "n_months"} <= set(out)
    assert out["n_months"] == T_full - T_fit
    # a pure seasonal signal is reproduced by its own climatology
    assert out["r2_clim"] > 0.9
    assert np.isfinite(out["r2_model"]) and np.isfinite(out["r2_persist"])
