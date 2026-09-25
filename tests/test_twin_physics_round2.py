"""Second round of the opt-in flow physics (2026-09-23): learned delay-bed disequilibrium,
tau floor and S_eff, delay layers, the aquitard store, river zone split / apex skip /
season, multi-component surface water, canal-to-pump substitution, the v2 surface-water
construction, and the policy gate's non-inferiority rule and ranking key.

Every option defaults off; these tests pin the limits each must reduce to, conservation,
SPD-ness of the operator, and the implicit adjoint against central differences.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin import calibrate_flow as cf  # noqa: E402
from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    BOUNDS,
    _add_extension_params,
    _delay_diagnostics,
    _expand_aqt,
    _expand_delay_du0,
    _expand_zonal_delay,
    _rollout,
    fit_flow,
    parse_delay_layers,
    set_delay_tau_min,
)
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.rivers import RiverSet, apex_overlap, river_cells  # noqa: E402

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


def _full(m, val, rows=None):
    rows = m.n_layers if rows is None else rows
    return torch.full((rows, m.grid.n_active), math.log(val), dtype=D)


def _storage(m, h):
    return float((torch.exp(m.log_S) * m.area * h).sum().detach())


@pytest.fixture
def restore_bounds():
    saved = dict(BOUNDS)
    yield
    BOUNDS.clear()
    BOUNDS.update(saved)


# ---------------------------------------------------------------------------------------
# A1: learned initial disequilibrium of the delay bed
# ---------------------------------------------------------------------------------------
def test_zero_du0_reproduces_the_equilibrium_start_exactly():
    m = _model()
    h0, rech, pump = _forcing(m)
    Sd, tau = _full(m, 5e-3), _full(m, 400.0)
    eq = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                  delay_Sd=Sd, delay_tau=tau)
    same = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                    delay_Sd=Sd, delay_tau=tau, u0=h0 + 0.0 * h0)
    assert torch.equal(eq, same)


def test_positive_du0_in_a_closed_box_raises_heads_monotonically_and_conserves_water():
    m = _model()
    L, A = m.n_layers, m.grid.n_active
    h0 = torch.full((L, A), 5.0, dtype=D)
    Sd_val, du0 = 2e-2, 3.0
    Sd, tau = _full(m, Sd_val), _full(m, 600.0)
    u0 = h0 + du0
    w0 = _storage(m, h0) + Sd_val * m.area * float(u0.sum())
    h, u = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 36, delay_Sd=Sd, delay_tau=tau,
                    u0=u0, return_state=True)
    mean = h.mean(dim=(0, 1))
    assert (mean[1:] - mean[:-1] > -1e-10).all() and float(mean[-1]) > 5.0 + 0.1
    # toward the equilibrium: never past the store, and the gap closes
    assert float(h.max()) <= 5.0 + du0 + 1e-9
    assert float((u - h[..., -1]).abs().max()) < du0
    w = _storage(m, h[..., -1]) + Sd_val * m.area * float(u.sum())
    assert abs(w - w0) <= 1e-8 * abs(w0)


def test_du0_parameters_bounds_and_expansion():
    m = _model()
    th = _add_extension_params({}, m, "zonal", zonal=True, delay_u0="learned")
    assert [k for k in th if k.startswith("log_du0")] == [
        "log_du0_proximal", "log_du0_mid", "log_du0_distal"]
    assert BOUNDS["log_du0"] == pytest.approx((math.log(0.01), math.log(50.0)))
    zone = torch.tensor(np.arange(m.grid.n_active) % 3)
    with torch.no_grad():
        th["log_du0_mid"].fill_(math.log(4.0))
    du0 = _expand_delay_du0(th, zone, m.n_layers, m.grid.n_active)
    assert torch.allclose(du0[:, zone == 1], torch.tensor(4.0, dtype=D))
    assert torch.allclose(du0[:, zone == 0], torch.tensor(1.0, dtype=D))
    with pytest.raises(ValueError, match="needs --delay-storage"):
        _add_extension_params({}, m, "off", delay_u0="learned")
    # the aquitard store's pair expands to one row per interface
    th2 = _add_extension_params({}, m, aquitard="global")
    Sa, G = _expand_aqt(th2, None, m.n_layers, m.grid.n_active)
    assert Sa.shape == G.shape == (m.n_layers - 1, m.grid.n_active)
    assert torch.allclose(torch.exp(G), torch.tensor(cf.AQT_G_INIT, dtype=D))
    assert _expand_aqt({}, None, m.n_layers, m.grid.n_active) == (None, None)


# ---------------------------------------------------------------------------------------
# A2/A3: tau floor, S_eff, delay layers
# ---------------------------------------------------------------------------------------
def test_tau_floor_is_applied_and_s_eff_is_reported(restore_bounds):
    set_delay_tau_min(180.0)
    assert BOUNDS["log_tau"][0] == pytest.approx(math.log(180.0))
    diag = _delay_diagnostics({"log_S_mid": [math.log(0.1), math.log(0.2)],
                               "log_Sd_mid": [math.log(0.05)],
                               "log_tau_mid": [math.log(30.0)]}, 30.0, 2)
    assert diag["S_eff_mid"] == pytest.approx([0.1 + 0.025, 0.2 + 0.025])
    assert diag["delay_is_elastic_mid"] is True
    diag = _delay_diagnostics({"log_S": [math.log(0.1), math.log(0.2)],
                               "log_Sd": [[math.log(0.05)]], "log_tau": [[math.log(900.0)]]},
                              30.0, 2, layers=(1,))
    assert diag["S_eff"][0] == pytest.approx(0.1) and not diag["delay_is_elastic"]


def test_delay_layers_hold_excluded_layers_at_the_floor_and_the_adjoint_runs():
    m = _model(n_layers=3)
    A = m.grid.n_active
    assert parse_delay_layers("1,2", 3) == (1, 2) and parse_delay_layers(None) is None
    with pytest.raises(ValueError):
        parse_delay_layers("5", 3)
    th = _add_extension_params({}, m, "global")
    Sd, _ = _expand_zonal_delay(th, None, 3, A, layers=(1, 2))
    assert torch.all(Sd[0] == BOUNDS["log_Sd"][0])
    assert torch.allclose(Sd[1:], torch.tensor(math.log(cf.DELAY_SD_INIT), dtype=D))
    h0, rech, _ = _forcing(m, steps=4)
    obs_idx = torch.tensor([0, 7, 14, 21])
    obs_layer = torch.tensor([0, 1, 2, 1])
    obs = h0[obs_layer, obs_idx][:, None].repeat(1, 4) + torch.linspace(0, 1, 4)
    fit = fit_flow(m, obs, obs_idx, obs_layer, torch.zeros(3, A, 4, dtype=D),
                   recharge_field=rech[0], epochs=2, h0=h0, delay_storage="global",
                   delay_u0="learned", delay_layers=(1, 2))
    assert torch.all(m.delay_log_Sd[0] == BOUNDS["log_Sd"][0])
    assert "du0_m" in fit["theta"] and "S_eff" in fit["theta"]
    assert m.delay_log_du0 is not None


# ---------------------------------------------------------------------------------------
# A4: the aquitard store
# ---------------------------------------------------------------------------------------
def test_aquitard_with_vanishing_storage_is_plain_leakance():
    m = _model(n_layers=3)
    h0, rech, pump = _forcing(m, steps=4)
    G = 4e-4
    with_aqt = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 4, recharge=rech, pumping=pump,
                        aqt_Sa=_full(m, 1e-20, rows=2), aqt_G=_full(m, G, rows=2))
    log_L2 = torch.log(torch.exp(m.log_L) + G / 2.0)
    plain = _rollout(m, m.log_T, m.log_S, log_L2, h0, 4, recharge=rech, pumping=pump)
    assert torch.allclose(with_aqt, plain, atol=1e-10, rtol=0)


def test_aquitard_operator_is_symmetric_positive_definite():
    m = _model(n_layers=3, nx=4, ny=4)
    L, A = m.n_layers, m.grid.n_active
    _, g, c, _ = m.aqt_terms(_full(m, 0.05, rows=2), _full(m, 1e-2, rows=2))
    mv, diag = m._matvec_from(torch.exp(m.log_T), torch.exp(m.log_S), torch.exp(m.log_L),
                              aqt=(g, c))
    n = L * A
    M = torch.zeros(n, n, dtype=D)
    for j in range(n):
        e = torch.zeros(n, dtype=D)
        e[j] = 1.0
        M[:, j] = mv(e.reshape(L, A)).reshape(-1)
    assert torch.allclose(M, M.T, atol=1e-9 * float(M.abs().max()))
    assert float(torch.linalg.eigvalsh(M).min()) > 0.0
    assert torch.allclose(torch.diagonal(M), diag.reshape(-1))


def test_aquitard_store_conserves_water_in_a_closed_box():
    m = _model(n_layers=3)
    L, A = m.n_layers, m.grid.n_active
    rng = np.random.default_rng(3)
    h0 = torch.tensor(rng.normal(8.0, 2.0, (L, A)), dtype=D)
    Sa = 0.05
    ua0 = torch.tensor(rng.normal(12.0, 1.0, (L - 1, A)), dtype=D)
    w0 = _storage(m, h0) + Sa * m.area * float(ua0.sum())
    h, st = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 12, aqt_Sa=_full(m, Sa, rows=2),
                     aqt_G=_full(m, 1e-3, rows=2), u0={"u": None, "ua": ua0},
                     return_state=True)
    w = _storage(m, h[..., -1]) + Sa * m.area * float(st["ua"].sum())
    assert abs(w - w0) <= 1e-7 * abs(w0)
    assert abs(_storage(m, h[..., -1]) - _storage(m, h0)) > 1e3     # water moved


def test_adjoint_matches_central_differences_for_du0_sa_g():
    m = _model(n_layers=2, nx=4, ny=3)
    L, A = m.n_layers, m.grid.n_active
    h0, rech, pump = _forcing(m, steps=3)
    params = {"du0": torch.tensor(math.log(2.0), dtype=D),
              "Sa": torch.tensor(math.log(0.02), dtype=D),
              "G": torch.tensor(math.log(5e-3), dtype=D)}
    target = torch.linspace(0.0, 1.0, A, dtype=D)

    def loss(p):
        Sd = _full(m, 1e-2)
        tau = _full(m, 300.0)
        h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                     delay_Sd=Sd, delay_tau=tau, u0=h0 + torch.exp(p["du0"]),
                     aqt_Sa=p["Sa"].expand(L - 1, A), aqt_G=p["G"].expand(L - 1, A))
        return ((h[:, :, -1] - target) ** 2).sum()

    p = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    val = loss(p)
    val.backward()
    for k in params:
        eps = 1e-5

        def at(d, k=k):
            q = {kk: vv.clone() for kk, vv in params.items()}
            q[k] = q[k] + d
            with torch.no_grad():
                return float(loss(q))
        fd = (at(eps) - at(-eps)) / (2 * eps)
        g = float(p[k].grad)
        assert abs(g - fd) <= 1e-5 * max(abs(fd), 1e-3), (k, g, fd)


# ---------------------------------------------------------------------------------------
# B: rivers
# ---------------------------------------------------------------------------------------
def test_zone_split_with_equal_conductances_reproduces_the_group_result():
    from shapely.geometry import box

    g = _grid()
    poly = box(0.0, 1000.0, 6000.0, 1600.0)
    dem = np.linspace(20.0, 5.0, g.n_active)
    zone = (np.arange(g.n_active) % g.nx) // 2                  # 3 west-east bands
    base = river_cells(g, {"choushui": poly}, dem)
    split = river_cells(g, {"choushui": poly}, dem, zone_of_cell=zone, c_split="zone")
    assert split.names == ("choushui_proximal", "choushui_mid", "choushui_distal")
    assert np.array_equal(split.idx, base.idx)
    m1, m2 = _model(), _model()
    m1.set_rivers(base, mode="ghb")
    m2.set_rivers(split, mode="ghb")
    h0, rech, pump = _forcing(m1)
    C = math.log(250.0)
    a = _rollout(m1, m1.log_T, m1.log_S, m1.log_L, h0, 3, recharge=rech, pumping=pump,
                 log_C_riv=torch.full((1,), C, dtype=D))
    b = _rollout(m2, m2.log_T, m2.log_S, m2.log_L, h0, 3, recharge=rech, pumping=pump,
                 log_C_riv=torch.full((3,), C, dtype=D))
    assert torch.allclose(a, b, atol=1e-10, rtol=0)


def test_skip_apex_leaves_no_river_cell_on_an_apex_cell():
    from shapely.geometry import box

    g = _grid()
    poly = box(0.0, 1000.0, 6000.0, 1600.0)
    dem = np.full(g.n_active, 10.0)
    apex = np.array([g.active_index(5500.0, 1500.0), g.active_index(5500.0, 2500.0)])
    full = river_cells(g, {"choushui": poly}, dem)
    assert apex_overlap(full, apex) == 1
    skipped = river_cells(g, {"choushui": poly}, dem, exclude_idx=apex)
    assert apex_overlap(skipped, apex) == 0 and skipped.n_cells == full.n_cells - 1


def test_season_factor_zero_matches_no_rivers(tmp_path):
    from hydrophysics.twin.rivers import load_river_season

    m = _model()
    A = m.grid.n_active
    rs = RiverSet(idx=np.array([1, 2, A - 2]), weight=np.array([0.5, 1.0, 0.3]),
                  group=np.array([0, 0, 1]), h_riv=np.array([14.0, 13.0, 9.0]),
                  rbot=np.array([12.0, 11.0, 7.0]), names=("choushui_mid", "wu"))
    csv = tmp_path / "season.csv"
    pd.DataFrame({"month": range(1, 13), "choushui": [0.0] * 12,
                  "wu": [0.0] * 12}).to_csv(csv, index=False)
    season = load_river_season(str(csv), rs.names)
    assert season.shape == (2, 12) and (season == 0).all()
    import dataclasses

    m.set_rivers(dataclasses.replace(rs, season=season), mode="ghb")
    h0, rech, pump = _forcing(m)
    ref = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump)
    h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                 log_C_riv=torch.full((2,), math.log(1e3), dtype=D))
    assert torch.allclose(h, ref, atol=1e-10, rtol=0)
    # a season with one open month differs, and only from that month on
    pd.DataFrame({"month": range(1, 13), "choushui": [0.0] * 3 + [1.0] + [0.0] * 8,
                  "wu": [0.0] * 12}).to_csv(csv, index=False)
    m.set_rivers(dataclasses.replace(rs, season=load_river_season(str(csv), rs.names)),
                 mode="ghb")
    h2 = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, recharge=rech, pumping=pump,
                  log_C_riv=torch.full((2,), math.log(1e3), dtype=D), river_month0=1)
    assert torch.allclose(h2[..., :3], ref[..., :3], atol=1e-10)      # Feb, Mar closed
    assert not torch.allclose(h2[..., 3], ref[..., 3], atol=1e-6)      # April open


# ---------------------------------------------------------------------------------------
# C: surface water components and the canal-to-pump substitution
# ---------------------------------------------------------------------------------------
def test_two_sw_components_add_their_separately_scaled_volumes():
    m = _model()
    A = m.grid.n_active
    h0, rech, _ = _forcing(m)
    rng = np.random.default_rng(5)
    sw = torch.tensor(rng.uniform(0, 2e-3, (2, A, 3)), dtype=D)
    s = torch.tensor([math.log(0.1), math.log(0.4)], dtype=D)
    both = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, sw_field=sw, log_sw_scale=s)
    combined = 0.1 * sw[0] + 0.4 * sw[1]
    one = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3, sw_field=combined,
                   log_sw_scale=torch.tensor(0.0, dtype=D))
    assert torch.allclose(both, one, atol=1e-10, rtol=0)
    th = _add_extension_params({}, m, use_sw=2)
    assert th["log_sw_scale"].shape == (2,)
    assert torch.allclose(torch.exp(th["log_sw_scale"]),
                          torch.tensor(cf.SW_SCALE_INIT_2, dtype=D))


def test_load_sw_recharge_components(tmp_path):
    from hydrophysics.twin.inputs import load_sw_recharge

    g = _grid()
    A = g.n_active
    dates = pd.date_range("2012-01-01", periods=6, freq="MS")
    p = tmp_path / "sw.npz"
    np.savez(p, sw_m_per_day=np.full((A, 6), 1e-3), perc_m_per_day=np.full((A, 6), 2e-4),
             dates=np.array([d.strftime("%Y-%m-01") for d in dates]), dx=1000.0, n_active=A)
    one = load_sw_recharge(str(p), g, dates)
    both = load_sw_recharge(str(p), g, dates, components="both")
    perc = load_sw_recharge(str(p), g, dates, components="percolation")
    assert one.shape == (A, 6) and both.shape == (2, A, 6)
    assert torch.equal(both[0], one) and float(perc[0, 0]) == pytest.approx(2e-4)
    q = tmp_path / "v1.npz"
    np.savez(q, sw_m_per_day=np.full((A, 6), 1e-3),
             dates=np.array([d.strftime("%Y-%m-01") for d in dates]), dx=1000.0, n_active=A)
    with pytest.raises(ValueError, match="perc_m_per_day"):
        load_sw_recharge(str(q), g, dates, components="both")


def test_full_substitution_keeps_delivered_plus_pumped_volume():
    """sw = 0 with sw_sub = 1: every m3 of canal water the scenario removes comes back as
    pumped water at the reference head, cell by cell."""
    from hydrophysics.twin.forward import add_irrigation_energy, sw_substitution_energy
    from hydrophysics.twin.pumping import energy_to_volume
    from hydrophysics.twin.scenario import PumpingScenario

    g = _grid()
    A = g.n_active
    dates = pd.date_range("2012-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(2)

    class Inp:
        grid = g
        sw_field = torch.tensor(rng.uniform(0, 3e-3, (A, 24)), dtype=D)
        ground_elev = torch.tensor(rng.uniform(5, 30, A), dtype=D)

    Inp.dates = dates
    scen = PumpingScenario("dry", sw_factor=0.0, sw_sub=1.0)
    scalars = {"log_eta": torch.tensor(math.log(0.5), dtype=D),
               "log_head_extra": torch.tensor(math.log(40.0), dtype=D)}
    h_ref = torch.tensor(rng.normal(3.0, 2.0, (2, A)), dtype=D)
    kwh = sw_substitution_energy(Inp, scen, 12, scalars, h_ref, pump_layer=1)
    from hydrophysics.twin.scenario import climatology

    clim, fut = climatology(Inp.sw_field.numpy(), dates, 12)
    lost = clim * g.dx ** 2 * 30.0                     # m3 per 30-day solver step (dt)
    lift = Inp.ground_elev - h_ref[1]
    vol = energy_to_volume(kwh, lift[:, None], scalars["log_eta"],
                           head_extra=torch.exp(scalars["log_head_extra"]))
    assert torch.allclose(vol, torch.tensor(lost, dtype=D), rtol=1e-12)
    # nothing substituted without a cut or with sw_sub = 0
    assert sw_substitution_energy(Inp, PumpingScenario("b"), 12, scalars, h_ref) is None
    assert sw_substitution_energy(Inp, PumpingScenario("c", sw_factor=0.5), 12, scalars,
                                  h_ref) is None
    E = torch.ones(3, A, 12, dtype=D)
    E2 = add_irrigation_energy(E, kwh, ["aquaculture", "irrigation", "other"])
    assert torch.equal(E2[0], E[0]) and torch.allclose(E2[1], 1.0 + kwh)


def test_scenario_string_parses_sw_sub():
    from hydrophysics.twin.forward import parse_scenario

    s, _ = parse_scenario("dry:sw=0.6,sw_sub=0.8@2026-01")
    assert s.sw_factor == 0.6 and s.sw_sub == 0.8 and "replaced by pumping" in s.describe()
    with pytest.raises(ValueError):
        parse_scenario("bad:sw_sub=1.5")


# ---------------------------------------------------------------------------------------
# surface_water.py v2
# ---------------------------------------------------------------------------------------
def test_percolation_and_drought_duty_on_a_toy_fan():
    from hydrophysics.twin import surface_water as swm

    g = _grid(nx=3, ny=1)
    A = g.n_active
    dates = pd.date_range("2021-01-01", periods=12, freq="MS")
    farm = {d: {"paddy": np.full(A, 2.5e5), "dry": np.zeros(A),
                "paddy_total_m2": 2.5e5 * A, "dry_total_m2": 0.0}
            for d in swm.DISTRICTS}
    zone = np.array([0, 1, 2])
    perc = swm.build_percolation(g, farm, dates, None, zone)
    # April (flooded, no duty): i_tex x paddy share, summed over the two districts
    apr = 3
    assert perc[:, apr] == pytest.approx(2 * 0.25 * np.array(swm.I_TEX_MM_D) / 1000.0)
    assert (perc[:, [0, 11]] == 0).all()                            # Dec-Jan dry
    cut = swm.build_percolation(g, farm, dates, None, zone, duty=swm.DROUGHT_DUTY)
    assert cut[:, apr] == pytest.approx(perc[:, apr] * (0.4 + 0.5) / 2)
    assert cut[:, 6] == pytest.approx(perc[:, 6])                   # July untouched
    seasons = {d: pd.DataFrame({"crop1": [1e8], "crop2": [1e8], "split": ["share"]},
                               index=pd.Index([2021], name="year")) for d in swm.DISTRICTS}
    f0 = swm.build_field(g, seasons, farm, dates)
    f1 = swm.build_field(g, seasons, farm, dates, duty=swm.DROUGHT_DUTY)
    days = dates.days_in_month.to_numpy()
    v0 = (f0["sw_m_per_day"] * days).sum() * g.dx ** 2
    v1 = (f1["sw_m_per_day"] * days).sum() * g.dx ** 2
    assert v0 == pytest.approx(v1 + f1["deficit_m3"].sum())         # nothing lost
    assert f1["deficit_m3"][:, 6].sum() == 0 and f1["deficit_m3"][:, apr].sum() > 0


def test_rice_area_ratio_reads_associations_and_holds_past_the_table():
    from hydrophysics.twin import surface_water as swm

    rows = []
    for assoc, base in ((12, 30000.0), (13, 25000.0)):
        for y in range(107, 110):
            rows.append({"irrigationassociation": assoc, "year": y,
                         "firstphasericeirrigationarea": base * (1 + 0.1 * (y - 108)),
                         "secondphasericeirrigationarea": base})
    r = swm.rice_area_ratio(pd.DataFrame(rows), range(2018, 2023))
    assert r["changhua"].loc[2019, "crop1"] == pytest.approx(1.0)
    assert r["changhua"].loc[2020, "crop1"] == pytest.approx(1.1)
    assert r["yunlin"].loc[2022, "crop1"] == pytest.approx(1.1)
    assert r["yunlin"].loc[2022, "source"] == "held"


def test_forward_rebuilds_du0_aquitard_and_sw_components_from_theta(tmp_path):
    from test_twin_forward import _inputs, _theta_file

    from hydrophysics.twin.forward import (
        build_model,
        delay_state_path,
        load_members,
        rollout,
        sw_hist,
    )

    inp = _inputs(T=12)
    A = inp.grid.n_active
    p, _ = _theta_file(tmp_path, boundaries="none")
    with open(p) as fh:
        obj = json.load(fh)
    for z in ("proximal", "mid", "distal"):
        obj["theta"][f"log_Sd_{z}"] = [math.log(0.05)]
        obj["theta"][f"log_tau_{z}"] = [math.log(600.0)]
        obj["theta"][f"log_du0_{z}"] = [math.log(2.0)]
    obj["theta"]["log_sw_scale"] = [math.log(0.1), math.log(0.3)]
    obj["meta"].update(delay_layers=[1, 2, 3], sw_components="both")
    with open(p, "w") as fh:
        json.dump(obj, fh)
    mem = load_members([str(p)])[0]
    model, scalars, _ = build_model(inp.grid, mem, "cpu")
    assert "delay_du0" in scalars and scalars["log_sw_scale"].shape == (2,)
    assert torch.all(model.delay_log_Sd[0] == BOUNDS["log_Sd"][0])
    h0 = inp.initial_heads(0)
    E, R = inp.E_total[:, 1:], inp.recharge_field[:, 1:]
    inp.sw_field = torch.full((2, A, 12), 1e-3, dtype=D)
    sw = sw_hist(inp)
    assert sw.shape == (2, A, 11)
    h, u = rollout(model, scalars, h0, E[:, :5], R[:, :5], inp.ground_elev, sw_field=sw[..., :5],
                   return_state=True)
    h_eq, _ = rollout(model, scalars, h0, E[:, :5], R[:, :5], inp.ground_elev,
                      sw_field=sw[..., :5], u0=h0, return_state=True)
    assert float((h - h_eq)[1:].mean()) > 0          # the draining store lifts the heads
    path = delay_state_path(model, scalars, h)
    assert torch.allclose(path[..., 0], h0 + 2.0) and torch.allclose(path[..., -1], u)
    # the aquitard store rebuilds too, and its state round-trips as a dict
    for k, v in (("log_Sa", [math.log(0.02)]), ("log_G", [math.log(1e-3)])):
        obj["theta"][k] = v
    with open(p, "w") as fh:
        json.dump(obj, fh)
    model, scalars, _ = build_model(inp.grid, load_members([str(p)])[0], "cpu")
    assert "aqt_Sa" in scalars and model.aqt_log_Sa.shape == (3, A)
    h1, st = rollout(model, scalars, h0, E[:, :3], R[:, :3], inp.ground_elev,
                     sw_field=sw[..., :3], return_state=True)
    h2 = rollout(model, scalars, h1[..., -1], E[:, 3:5], R[:, 3:5], inp.ground_elev,
                 sw_field=sw[..., 3:5], u0=st)
    full = rollout(model, scalars, h0, E[:, :5], R[:, :5], inp.ground_elev,
                   sw_field=sw[..., :5])
    assert torch.allclose(h2[..., -1], full[..., -1], atol=1e-6)


# ---------------------------------------------------------------------------------------
# D: policy gate
# ---------------------------------------------------------------------------------------
def test_non_inferiority_rejects_the_21km_pair_and_accepts_the_deliverable():
    from hydrophysics.twin.policy_gate import policy_verdict

    ten = policy_verdict({0.85: 0.46, 0.7: 0.91}, {0.85: -0.0063, 0.7: -0.0125}, 0.0478,
                         min_response_frac=0.5)
    twenty = policy_verdict({0.85: 0.50, 0.7: 1.00}, {0.85: -0.0024, 0.7: -0.0044}, 0.0392,
                            min_response_frac=0.5)
    assert ten["verdict"] == "PASS" and twenty["verdict"] == "FAIL"
    assert not twenty["checks"]["non_inferior_to_reference"]
    # off by default: the pre-registered rule is unchanged
    assert "non_inferior_to_reference" not in policy_verdict(
        {0.85: 0.5, 0.7: 1.0}, {0.85: -0.0024, 0.7: -0.0044}, 0.0392)["checks"]


def test_rank_key_orders_scorecards():
    from hydrophysics.twin.policy_gate import rank_key

    def card(head, temp, pol, lev, ratio):
        return {"head_gate": {"verdict": head}, "temporal": {"verdict": temp,
                                                             "rmse_ratio": ratio},
                "policy_response": {"verdict": pol, "leveling_hindcast": {"r2": lev}}}

    a = card("PASS", "PASS", "FAIL", 0.60, 1.2)
    b = card("PASS", "PASS", "PASS", 0.50, 1.4)
    c = card("PASS", "FAIL", "PASS", 0.70, 0.9)
    d = card("PASS", "PASS", "PASS", 0.50, 1.1)
    order = sorted([a, b, c, d], key=rank_key, reverse=True)
    assert order == [d, b, a, c]
    assert json.loads(json.dumps(rank_key({})))[:3] == [0, 0, 0]


def _du0_member(tmp_path, aquitard=False, delay=True):
    from test_twin_forward import _theta_file

    from hydrophysics.twin.forward import load_members

    p, _ = _theta_file(tmp_path, boundaries="none")
    with open(p) as fh:
        obj = json.load(fh)
    if delay:
        for z in ("proximal", "mid", "distal"):
            obj["theta"][f"log_Sd_{z}"] = [math.log(0.05)]
            obj["theta"][f"log_tau_{z}"] = [math.log(600.0)]
            obj["theta"][f"log_du0_{z}"] = [math.log(2.0)]
    if aquitard:
        obj["theta"]["log_Sa"] = [math.log(0.02)]
        obj["theta"]["log_G"] = [math.log(1e-3)]
    with open(p, "w") as fh:
        json.dump(obj, fh)
    return load_members([str(p)])[0]


def test_delay_state_path_refuses_an_aquitard_only_model(tmp_path):
    """Review 2026-09-23 #2: an aquitard-only model has slow state (u_a); answering None
    let the surrogate silently restart it at equilibrium."""
    from test_twin_forward import _inputs

    from hydrophysics.twin.forward import build_model, delay_state_path

    inp = _inputs(T=6)
    model, scalars, _ = build_model(inp.grid, _du0_member(tmp_path, aquitard=True,
                                                          delay=False), "cpu")
    assert "aqt_Sa" in scalars and "delay_tau" not in scalars
    h = inp.initial_heads(0)[..., None].repeat(1, 1, 3)
    with pytest.raises(NotImplementedError, match="aquitard"):
        delay_state_path(model, scalars, h)


def test_nudged_hindcast_u_path_is_the_solver_state(tmp_path):
    """Review 2026-09-23 #3: with --hindcast-gain > 0 the dumped slow-store path must be
    the solver's own u (replayed from the un-nudged segments), ending exactly at u_end."""
    from test_twin_forward import _inputs

    from hydrophysics.twin.forward import (
        build_model,
        delay_state_path,
        hindcast_with_nudging,
        rollout,
    )

    inp = _inputs(T=12)
    model, scalars, _ = build_model(inp.grid, _du0_member(tmp_path), "cpu")
    h0 = inp.initial_heads(0)
    E, R = inp.E_total[:, 1:], inp.recharge_field[:, 1:]
    heads, u_end, u_path = hindcast_with_nudging(model, scalars, inp, h0, E, R, 0.7, 4,
                                                 return_u_path=True)
    assert u_path.shape == heads.shape
    assert torch.allclose(u_path[..., -1], u_end, atol=1e-12)
    # the first segment is un-nudged up to its last month: the solver's own state there
    _, u4 = rollout(model, scalars, h0, E[:, :4], R[:, :4], inp.ground_elev,
                    return_state=True, month0=1)
    assert torch.allclose(u_path[..., 4], u4, atol=1e-12)
    # replaying the nudged series is NOT the solver state (the old dump)
    assert not torch.allclose(delay_state_path(model, scalars, heads)[..., -1], u_end,
                              atol=1e-6)
    # gain 0: the path is the plain replay
    h_p, u_p, path_p = hindcast_with_nudging(model, scalars, inp, h0, E, R, 0.0, 0,
                                             return_u_path=True)
    assert torch.allclose(path_p, delay_state_path(model, scalars, h_p))
    assert torch.allclose(path_p[..., -1], u_p, atol=1e-12)


def test_policy_gate_projection_starts_in_its_own_calendar_month(tmp_path, monkeypatch):
    """Review 2026-09-23 #4: the policy projections pass month0 = the first projected
    month (river-stage season), as forward.run does, not the February default."""
    from test_twin_forward import _inputs, _theta_file

    import hydrophysics.twin.policy_gate as pg
    from hydrophysics.twin.forward import load_members
    from hydrophysics.twin.forward import rollout as real_rollout

    inp = _inputs(T=24)                  # 2012-01 .. 2013-12: the projection opens in January
    p, _ = _theta_file(tmp_path)
    vep = tmp_path / "vep.json"
    vep.write_text(json.dumps({"log_ske": math.log(1e-4), "log_skv": math.log(1e-3),
                               "log_tau": math.log(300.0), "h_pc0": 0.0}))
    seen = []

    def spy(*a, **kw):
        seen.append(kw.get("month0"))
        return real_rollout(*a, **kw)

    monkeypatch.setattr(pg, "rollout", spy)
    pg.policy_response(inp, load_members([p])[0], str(vep), horizon=3, device="cpu",
                       log=lambda *_: None)
    assert seen and all(m == 0 for m in seen)
