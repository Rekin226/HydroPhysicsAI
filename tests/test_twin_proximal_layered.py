"""Opt-in layered proximal aquifer (2026-09-25): --proximal-layered and
--ic-layered-proximal. Defaults must reproduce the merged proximal aquifer exactly; the
layered form must start as the merged model and travel through the theta file."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    BOUNDS,
    _expand_zonal,
    _ic_zone_map,
    _idw_initial_heads,
    _make_zonal_params,
    _merged_proximal_heads,
    _zonal_bounds_hit,
    fit_flow,
    set_l_min,
    set_log_t_min_proximal,
)
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.forward import Member, build_model  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.inputs import TwinInputs, input_options  # noqa: E402
from hydrophysics.twin.uncertainty import effective_bounds  # noqa: E402
from hydrophysics.twin.zones import MID, PROXIMAL, PROXIMAL_W, fan_zones  # noqa: E402

D = torch.float64
X0, Y0 = 200000.0, 2600000.0


def _grid():
    # 8 columns of 2 km from x = 199 km: centroids at 200, 202, ..., 214 km
    return FanGrid(nx=8, ny=3, dx=2000.0, x0=199_000.0, y0=2_640_000.0,
                   mask=np.ones((3, 8), dtype=bool))


def _n_free(theta):
    return sum(int(p.numel()) for p in theta.values())


@pytest.mark.parametrize("split", [False, True])
def test_default_unchanged_and_layered_parameter_count(split):
    m = FlowModel(_grid(), n_layers=4, dt_days=30.0)
    base = _make_zonal_params(m, split=split)
    again = _make_zonal_params(m, split=split, proximal_layered=False)
    assert list(base) == list(again)
    assert all(torch.equal(base[k], again[k]) for k in base)
    assert "log_L_proximal" not in base and base["log_T_proximal"].shape == (1, 1)
    lay = _make_zonal_params(m, split=split, proximal_layered=True)
    zones = ("proximal", "proximal_w") if split else ("proximal",)
    # each proximal zone: 2 merged values -> 4 T + 4 S + 3 L = 11, the mid/distal form
    assert _n_free(lay) - _n_free(base) == 9 * len(zones)
    for z in zones:
        assert lay[f"log_T_{z}"].shape == (4, 1) and lay[f"log_S_{z}"].shape == (4, 1)
        assert lay[f"log_L_{z}"].shape == (3, 1)
        assert torch.equal(lay[f"log_T_{z}"], base[f"log_T_{z}"].expand(4, 1))
        assert torch.all(lay[f"log_L_{z}"] == BOUNDS["log_L"][1])
    # the historical keys keep their order; the leakances are appended
    assert [k for k in lay if k in base] == list(base)


@pytest.mark.parametrize("split", [False, True])
def test_layered_epoch0_equals_merged_model(split):
    m = FlowModel(_grid(), n_layers=4, dt_days=30.0)
    zt = torch.tensor(fan_zones(m.grid.centroids(), 205.0, 182.0,
                                split_km=209.0 if split else None))
    a = _expand_zonal(_make_zonal_params(m, split=split), zt, 4)
    b = _expand_zonal(_make_zonal_params(m, split=split, proximal_layered=True), zt, 4)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def test_layered_values_reach_only_their_proximal_layer():
    m = FlowModel(_grid(), n_layers=4, dt_days=30.0)
    zone = fan_zones(m.grid.centroids(), 205.0, 182.0)
    zt = torch.tensor(zone)
    th = _make_zonal_params(m, proximal_layered=True)
    base_T, _, base_L = _expand_zonal(th, zt, 4)
    with torch.no_grad():
        th["log_T_proximal"][2] = math.log(777.0)
        th["log_L_proximal"][1] = math.log(1e-5)
    T, _, L = _expand_zonal(th, zt, 4)
    prox = zone == PROXIMAL
    assert torch.allclose(T[2, prox], torch.tensor(math.log(777.0), dtype=D))
    assert torch.equal(T[[0, 1, 3]], base_T[[0, 1, 3]]) and torch.equal(T[:, ~prox],
                                                                        base_T[:, ~prox])
    assert torch.allclose(L[1, prox], torch.tensor(math.log(1e-5), dtype=D))
    assert torch.equal(L[:, ~prox], base_L[:, ~prox])


def test_layered_bounds_respect_proximal_floor_and_l_min():
    m = FlowModel(_grid(), n_layers=4, dt_days=30.0)
    th = _make_zonal_params(m, split=True, proximal_layered=True)
    old_L = BOUNDS["log_L"]
    try:
        set_log_t_min_proximal(58.0)
        set_l_min(1e-4)
        with torch.no_grad():
            for z in ("proximal", "proximal_w"):
                th[f"log_T_{z}"].fill_(math.log(10.0))
                th[f"log_L_{z}"].fill_(math.log(1e-8))
        hits = _zonal_bounds_hit(th)
        for z in ("proximal", "proximal_w"):
            assert torch.allclose(th[f"log_T_{z}"], torch.tensor(math.log(58.0), dtype=D))
            assert torch.allclose(th[f"log_L_{z}"], torch.tensor(math.log(1e-4), dtype=D))
            assert hits[z]["log_T"] == {"lo": 4, "hi": 0, "n": 4}
            assert hits[z]["log_L"]["lo"] == 3
    finally:
        set_log_t_min_proximal(None)
        BOUNDS["log_L"] = old_L
    lo, _ = effective_bounds([("log_T_proximal", 3), ("log_L_proximal", 2)],
                             {"log_t_min_proximal": 58.0, "l_min": 1e-4})
    assert lo[0] == pytest.approx(math.log(58.0)) and lo[1] == pytest.approx(math.log(1e-4))


def test_layered_fit_reloads_through_forward():
    g = _grid()
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    A = g.n_active
    zone = fan_zones(g.centroids(), 205.0, 182.0, split_km=209.0)
    rng = np.random.default_rng(0)
    h0 = torch.tensor(rng.normal(10.0, 1.0, (4, A)), dtype=D)
    steps = 3
    obs_idx = torch.tensor([0, 5, 7, 15, 23])
    obs_layer = torch.tensor([0, 1, 2, 3, 0])
    obs = h0[obs_layer, obs_idx][:, None].repeat(1, steps) + 0.1
    rech = torch.full((A, steps), 1e-4, dtype=D)
    kw = dict(recharge_field=rech, epochs=2, param_mode="zonal", h0=h0, zone_of_cell=zone)
    fit = fit_flow(m, obs, obs_idx, obs_layer, torch.zeros(4, A, steps, dtype=D),
                   proximal_layered=True, **kw)
    th = fit["theta"]
    assert len(th["log_T_proximal"]) == 4 and len(th["log_L_proximal_w"]) == 3
    m0 = FlowModel(g, n_layers=4, dt_days=30.0)
    fit0 = fit_flow(m0, obs, obs_idx, obs_layer, torch.zeros(4, A, steps, dtype=D), **kw)
    assert fit["n_params"] - fit0["n_params"] == 18
    rebuilt, _, _ = build_model(g, Member("m", th, {"param_mode": "zonal",
                                                    "zone_boundaries": "205,182,209"}), "cpu")
    assert torch.allclose(rebuilt.log_T, m.log_T.detach())
    assert torch.allclose(rebuilt.log_S, m.log_S.detach())
    assert torch.allclose(rebuilt.log_L, m.log_L.detach())
    with pytest.raises(ValueError, match="zonal"):
        fit_flow(FlowModel(g, n_layers=4, dt_days=30.0), obs, obs_idx, obs_layer,
                 torch.zeros(4, A, steps, dtype=D), epochs=1, h0=h0, proximal_layered=True)


# ---- --ic-layered-proximal ---------------------------------------------------------------
def _ic_grid():
    return FanGrid(nx=10, ny=3, dx=1000.0, x0=X0, y0=Y0, mask=np.ones((3, 10), dtype=bool))


def _wells(g):
    # one proximal nest with a vertical gradient (L1 60 m, L2 50 m) and far mid-fan wells
    xy = np.array([[X0 + 7500, Y0 + 1500], [X0 + 7500, Y0 + 1500],
                   [X0 + 500, Y0 + 500], [X0 + 500, Y0 + 2500]])
    h = np.array([60.0, 50.0, 5.0, 3.0])
    layer = np.array([0, 1, 2, 3])
    idx = np.array([g.active_index(*p) for p in xy])
    return xy, h, layer, idx


def test_ic_layered_keeps_the_vertical_gradient_and_falls_back_to_merged():
    g = _ic_grid()
    xy, h, layer, idx = _wells(g)
    h0 = _idw_initial_heads(g, xy, h, layer, 4)
    zc = _ic_zone_map(g, "205,182")
    merged, _ = _merged_proximal_heads(g, h0, xy, h, zc, zc[idx])
    out, rep = _merged_proximal_heads(g, h0, xy, h, zc, zc[idx], layer_of=layer)
    prox = zc == PROXIMAL
    assert torch.allclose(out[0, prox], torch.full((int(prox.sum()),), 60.0, dtype=D))
    assert torch.allclose(out[1, prox], torch.full((int(prox.sum()),), 50.0, dtype=D))
    # no proximal L3/L4 well: the merged value, never the mid-fan heads
    assert torch.equal(out[2:, prox], merged[2:, prox])
    assert torch.equal(out[:, ~prox], h0[:, ~prox])
    assert rep["proximal"]["n_wells_per_layer"] == [1, 1, 0, 0]
    assert (zc == MID).any()


def test_ic_layered_split_zone_uses_the_all_proximal_fallback_per_layer():
    g = _ic_grid()
    xy, h, layer, idx = _wells(g)
    h0 = _idw_initial_heads(g, xy, h, layer, 4)
    zc = _ic_zone_map(g, "205,182,207")
    out, rep = _merged_proximal_heads(g, h0, xy, h, zc, zc[idx], layer_of=layer)
    pw = zc == PROXIMAL_W
    assert rep["proximal_w"]["source"] == "all proximal"
    assert torch.allclose(out[0, pw], torch.full_like(out[0, pw], 60.0))
    assert torch.allclose(out[3, pw], torch.full_like(out[3, pw], 55.0))


def test_twin_inputs_and_input_options_carry_ic_layered():
    g = _ic_grid()
    xy, h, layer, idx = _wells(g)
    obs = np.stack([h, h + 1.0], axis=1)
    zc = _ic_zone_map(g, "205,182")

    def inp(layered):
        return TwinInputs(grid=g, hf=None,
                          dates=pd.date_range("2012-01-01", periods=2, freq="MS"),
                          obs_h=obs, obs_h_filled=obs, obs_idx=idx, obs_layer=layer,
                          well_xy=xy, sids=["a", "b", "c", "d"],
                          ground_elev=torch.zeros(g.n_active), E_by_class={},
                          recharge_field=torch.zeros(g.n_active, 2), ic_zone_of_cell=zc,
                          ic_layered=layered)

    base = _idw_initial_heads(g, xy, h, layer, 4)
    want_m, _ = _merged_proximal_heads(g, base, xy, h, zc, zc[idx])
    want_l, _ = _merged_proximal_heads(g, base, xy, h, zc, zc[idx], layer_of=layer)
    assert torch.equal(inp(False).initial_heads(0), want_m)
    assert torch.equal(inp(True).initial_heads(0), want_l)
    assert "ic_layered_proximal" not in input_options({"ic_merged_proximal": True})
    assert input_options({"ic_merged_proximal": True,
                          "ic_layered_proximal": True})["ic_layered_proximal"] is True


def test_members_from_gate_csv_carries_the_layered_flags(tmp_path):
    from hydrophysics.twin.forward import _members_from_gate_csv

    row = {"param_mode": "zonal", "boundaries": "coast-apex", "zone_proximal_km": 205.0,
           "zone_distal_km": 182.0, "zone_split_km": 208.0, "dx": 1000.0, "epochs": 1,
           "r2_insample": 0.5, "r2_kfold": 0.4, "r2_idw": 0.3, "n_folds": 10,
           "theta": "{'log_T_proximal': [1.0, 1.0, 1.0, 1.0]}", "ic_merged_proximal": True,
           "ic_layered_proximal": True, "proximal_layered": True}
    p = tmp_path / "stage3_flow.csv"
    pd.DataFrame([row]).to_csv(p, index=False)
    meta = _members_from_gate_csv(str(p))["meta"]
    assert meta["proximal_layered"] is True and meta["ic_layered_proximal"] is True
    assert meta["zone_boundaries"] == "205,182,208"
    assert input_options(meta)["ic_layered_proximal"] is True
