"""Round 3 (2026-09-23): opt-in compaction-column constraints and the proximal zone split.

Both are opt-in: the defaults must reproduce the historical clamp and the three-zone
map exactly, and the new options must travel from calibration to ``twin.forward``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

import hydrophysics.twin.calibrate_coupled as cc  # noqa: E402
from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    BOUNDS,
    _base_param_name,
    _expand_zonal,
    _parse_zone_boundaries,
    _zonal_bounds_hit,
    fit_flow,
    set_log_t_min_proximal,
)
from hydrophysics.twin.compaction import VEPColumn  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.forward import Member, build_model, load_or_fit_vep  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.zones import (  # noqa: E402
    MID,
    PROXIMAL,
    PROXIMAL_W,
    ZONE_NAMES_SPLIT,
    collapse_zones,
    fan_zones,
    zone_blend_weights,
    zone_names,
)

D = torch.float64


def _xy(*xs_km):
    return np.array([[x * 1000.0, 2_640_000.0] for x in xs_km], dtype="float64")


# ---- column constraints ----------------------------------------------------------------
@pytest.fixture
def constraints():
    old = (cc.TAU_MIN_DAYS, cc.SKE_MIN, cc.SKE_SKV_MAX, cc.TAU_MAX_YEARS)
    yield cc
    cc.TAU_MIN_DAYS, cc.SKE_MIN, cc.SKE_SKV_MAX, cc.TAU_MAX_YEARS = old


def _col(ske, skv, tau):
    c = VEPColumn(n_sites=1)
    with torch.no_grad():
        c.log_ske.fill_(math.log(ske))
        c.log_skv.fill_(math.log(skv))
        c.log_tau.fill_(math.log(tau))
    return c


def test_default_clamp_is_the_historical_one(constraints):
    constraints.TAU_MIN_DAYS = constraints.SKE_MIN = constraints.SKE_SKV_MAX = None
    constraints.TAU_MAX_YEARS = None
    c = _col(1e-9, 1e-7, 0.01)
    constraints._clamp_column(c, 132)
    assert float(c.log_ske.detach()) == pytest.approx(math.log(1e-6))
    assert float(c.log_skv.detach()) == pytest.approx(math.log(1e-5))
    assert float(c.log_tau.detach()) == pytest.approx(math.log(1.0), abs=1e-6)
    # an interior column is untouched, including a Ske above Skv
    c = _col(5e-2, 1e-3, 24.0)
    before = [float(getattr(c, k)) for k in ("log_ske", "log_skv", "log_tau")]
    constraints._clamp_column(c, 132)
    assert [float(getattr(c, k)) for k in ("log_ske", "log_skv", "log_tau")] == before
    assert constraints.column_constraints() == {"tau_min_days": None, "ske_min": None,
                                                "ske_skv_max": None}


def test_constraints_raise_floors_and_cap_the_ratio(constraints):
    constraints.TAU_MAX_YEARS = None
    constraints.TAU_MIN_DAYS, constraints.SKE_MIN, constraints.SKE_SKV_MAX = 180.0, 1e-3, 0.3
    # the round-2 proximal column: tau 24 d, Ske on its floor, Skv 0.257
    c = _col(1e-6, 0.257, 24.0)
    constraints._clamp_column(c, 132)
    assert math.exp(float(c.log_tau)) == pytest.approx(180.0)
    assert math.exp(float(c.log_ske)) == pytest.approx(1e-3)
    # a Ske/Skv above the cap is brought down to it, with Skv kept
    c = _col(5e-2, 1e-2, 500.0)
    constraints._clamp_column(c, 132)
    assert math.exp(float(c.log_skv)) == pytest.approx(1e-2)
    assert math.exp(float(c.log_ske - c.log_skv)) == pytest.approx(0.3)
    # the floor and the cap together: Skv is raised to ske_min / ratio
    c = _col(1e-4, 1e-4, 500.0)
    constraints._clamp_column(c, 132)
    assert math.exp(float(c.log_ske)) == pytest.approx(1e-3, rel=1e-5)
    assert math.exp(float(c.log_skv)) == pytest.approx(1e-3 / 0.3, rel=1e-5)
    assert constraints.column_constraints() == {"tau_min_days": 180.0, "ske_min": 1e-3,
                                                "ske_skv_max": 0.3}


def test_constraints_reject_inconsistent_values(constraints):
    constraints.TAU_MAX_YEARS = None
    constraints.SKE_MIN = constraints.SKE_SKV_MAX = None
    constraints.TAU_MIN_DAYS = 5000.0                       # above the 3960 d ceiling
    with pytest.raises(ValueError, match="tau-min-days"):
        constraints._clamp_column(VEPColumn(n_sites=1), 132)
    constraints.TAU_MIN_DAYS = None
    constraints.SKE_MIN, constraints.SKE_SKV_MAX = 0.05, 0.01   # needs Skv = 5
    with pytest.raises(ValueError, match="Skv"):
        constraints._clamp_column(VEPColumn(n_sites=1), 132)


def test_fit_keeps_every_column_inside_the_constraints(constraints):
    constraints.TAU_MAX_YEARS = None
    constraints.TAU_MIN_DAYS, constraints.SKE_MIN, constraints.SKE_SKV_MAX = 180.0, 1e-3, 0.3
    model = cc._make("zonal", "cpu")
    T = 24
    heads = torch.linspace(0.0, -4.0, T).repeat(3, 4, 1)
    obs = torch.linspace(0.0, 0.2, T).repeat(3, 1)
    cc._fit(model, heads, obs, torch.ones(3, T), torch.tensor([0, 1, 2]), 20, 0.3)
    for c in model:
        assert math.exp(float(c.log_tau)) >= 180.0 * (1 - 1e-6)
        assert math.exp(float(c.log_ske)) >= 1e-3 * (1 - 1e-6)
        assert float(c.log_ske - c.log_skv) <= math.log(0.3) + 1e-6


def test_forward_reads_a_constrained_column_json_unchanged(tmp_path, constraints):
    col = {"log_ske": math.log(1e-3), "log_skv": math.log(0.01), "log_tau": math.log(200.0),
           "h_pc0": 0.0}
    p = {"zonal": [col, col, col], "config": "zonal", "tau_max_years": None,
         "hpc0_guard_days": 365.0, "tau_min_days": 180.0, "ske_min": 1e-3,
         "ske_skv_max": 0.3, "zone_blend_km": 2.0}
    f = tmp_path / "vep_zonal_leveling.json"
    f.write_text(json.dumps(p))
    xy = _xy(210.0, 190.0, 170.0)
    zone = fan_zones(xy)
    column, back = load_or_fit_vep(str(f), None, None, "cpu", zone_of_cell=zone,
                                   zone_weights=lambda km: zone_blend_weights(xy, 205.0,
                                                                              182.0, km))
    assert back["tau_min_days"] == 180.0 and back["ske_skv_max"] == 0.3
    out = column(torch.linspace(0.0, -2.0, 10).repeat(3, 1))
    assert out.shape == (3, 10) and torch.isfinite(out).all()


# ---- the proximal split ----------------------------------------------------------------
def test_parse_zone_boundaries_split_is_opt_in():
    assert _parse_zone_boundaries("205,182") == (205.0, 182.0)
    assert _parse_zone_boundaries("205,182", allow_split=True) == (205.0, 182.0, None)
    assert _parse_zone_boundaries("205,182,211", allow_split=True) == (205.0, 182.0, 211.0)
    with pytest.raises(ValueError, match="third value"):
        _parse_zone_boundaries("205,182,211")               # a three-zone-only caller
    with pytest.raises(ValueError, match="east of the proximal"):
        _parse_zone_boundaries("205,182,200", allow_split=True)


def test_fan_zones_split_appends_an_id_and_collapses_back():
    xy = _xy(215.0, 211.0, 208.0, 205.0, 195.0, 170.0)
    z3 = fan_zones(xy)
    z4 = fan_zones(xy, split_km=211.0)
    assert z4.tolist() == [PROXIMAL, PROXIMAL, PROXIMAL_W, PROXIMAL_W, MID, 2]
    assert np.array_equal(collapse_zones(z4), z3)
    assert np.array_equal(fan_zones(xy, split_km=None), z3)
    assert zone_names(4) == ZONE_NAMES_SPLIT and ZONE_NAMES_SPLIT[PROXIMAL_W] == "proximal_w"
    with pytest.raises(ValueError):
        fan_zones(xy, split_km=204.0)
    w = zone_blend_weights(xy, 205.0, 182.0, 2.0, split_km=211.0)
    assert w.shape == (4, 6) and np.allclose(w.sum(0), 1.0)
    assert w[PROXIMAL_W, 2] == 1.0 and w[PROXIMAL, 0] == 1.0
    assert np.allclose(zone_blend_weights(xy, 205.0, 182.0, 0.0, split_km=211.0)[z4, range(6)],
                       1.0)


def _grid():
    # 8 columns of 2 km from x = 199 km: centroids at 200, 202, ..., 214 km
    return FanGrid(nx=8, ny=3, dx=2000.0, x0=199_000.0, y0=2_640_000.0,
                   mask=np.ones((3, 8), dtype=bool))


def test_expand_zonal_with_equal_parts_matches_three_zones():
    m = FlowModel(_grid(), n_layers=4, dt_days=30.0)
    from hydrophysics.twin.calibrate_flow import _make_zonal_params

    th3 = _make_zonal_params(m)
    th4 = _make_zonal_params(m, split=True)
    assert "log_T_proximal_w" in th4 and "log_T_proximal_w" not in th3
    assert _base_param_name("log_T_proximal_w") == "log_T"
    xy = m.grid.centroids()
    z3 = torch.tensor(fan_zones(xy, 205.0, 182.0))
    z4 = torch.tensor(fan_zones(xy, 205.0, 182.0, split_km=209.0))
    a = _expand_zonal(th3, z3, 4)
    b = _expand_zonal(th4, z4, 4)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)
    with torch.no_grad():
        th4["log_T_proximal_w"].fill_(math.log(3000.0))
    b = _expand_zonal(th4, z4, 4)
    sel = (z4 == PROXIMAL_W)
    assert sel.any() and torch.allclose(b[0][:, sel], torch.tensor(math.log(3000.0), dtype=D))
    assert torch.equal(b[0][:, ~sel], a[0][:, ~sel])


def test_log_t_min_proximal_raises_only_the_proximal_floor():
    th = {f"log_T_{z}": torch.full((1 if z.startswith("prox") else 4, 1), math.log(10.0),
                                   dtype=D) for z in ZONE_NAMES_SPLIT}
    try:
        hits = _zonal_bounds_hit(th)
        assert hits["proximal"]["log_T"]["lo"] == 1 and hits["mid"]["log_T"]["lo"] == 4
        set_log_t_min_proximal(58.0)
        hits = _zonal_bounds_hit(th)
        for z in ("proximal", "proximal_w"):
            assert math.exp(float(th[f"log_T_{z}"])) == pytest.approx(58.0)
            assert hits[z]["log_T"]["lo"] == 1
        assert math.exp(float(th["log_T_mid"][0])) == pytest.approx(10.0)
        assert hits["mid"]["log_T"]["lo"] == 4
        assert BOUNDS["log_T"][0] == pytest.approx(math.log(10.0))   # global untouched
    finally:
        set_log_t_min_proximal(None)
    assert not any(k[1].startswith("prox") for k in
                   __import__("hydrophysics.twin.calibrate_flow",
                              fromlist=["ZONE_LOWER_BOUNDS"]).ZONE_LOWER_BOUNDS)


def test_four_zone_fit_and_forward_rebuild_end_to_end():
    """fit_flow on four zone ids (zonal delay bed too), then twin.forward.build_model
    rebuilds the same field from the recorded 'P,D,S' string and hands its callers the
    three-zone ids."""
    g = _grid()
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    A = g.n_active
    zb = "205,182,209"
    zone = fan_zones(g.centroids(), 205.0, 182.0, split_km=209.0)
    assert set(zone.tolist()) == {PROXIMAL, MID, PROXIMAL_W}
    rng = np.random.default_rng(0)
    h0 = torch.tensor(rng.normal(10.0, 1.0, (4, A)), dtype=D)
    steps = 3
    obs_idx = torch.tensor([0, 5, 10, 15, 20])
    obs_layer = torch.tensor([0, 1, 2, 3, 0])
    obs = h0[obs_layer, obs_idx][:, None].repeat(1, steps) + 0.1
    rech = torch.full((A, steps), 1e-4, dtype=D)
    fit = fit_flow(m, obs, obs_idx, obs_layer, torch.zeros(4, A, steps, dtype=D),
                   recharge_field=rech, epochs=2, param_mode="zonal", h0=h0,
                   zone_of_cell=zone, delay_storage="zonal")
    th = fit["theta"]
    assert "log_T_proximal_w" in th and "log_Sd_proximal_w" in th
    assert "proximal_w" in fit["bounds_hit"]
    log_T_fit = m.log_T.detach().clone()
    rebuilt, _, zoc = build_model(g, Member("m", th, {"param_mode": "zonal",
                                                      "zone_boundaries": zb}), "cpu")
    assert torch.allclose(rebuilt.log_T, log_T_fit)
    assert zoc.max() <= 2 and np.array_equal(zoc, collapse_zones(zone))
    # a three-zone string would rebuild a different model: it must not load silently
    with pytest.raises(ValueError, match="proximal split"):
        build_model(g, Member("m", {k: v for k, v in th.items()
                                    if not k.endswith("_proximal_w")},
                                {"param_mode": "zonal", "zone_boundaries": zb}), "cpu")


def test_posterior_bounds_honour_recorded_proximal_floor():
    """A posterior drawn around a --log-t-min-proximal fit must keep the raised floor:
    ``uncertainty.effective_bounds`` reads it from the recorded meta (review, round 3)."""
    from hydrophysics.twin.uncertainty import effective_bounds

    index = [("log_T_proximal", 0), ("log_T_proximal_w", 0), ("log_T_mid", 0)]
    lo0, _ = effective_bounds(index, {})
    assert np.allclose(lo0, BOUNDS["log_T"][0])                 # default unchanged
    lo, _ = effective_bounds(index, {"log_t_min_proximal": 58.0})
    assert np.allclose(lo[:2], math.log(58.0))
    assert lo[2] == pytest.approx(BOUNDS["log_T"][0])            # other zones untouched
