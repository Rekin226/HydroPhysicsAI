"""Opt-in fixes for the three model artefacts the app review found (2026-09-23), plus the
per-member yearly sidecar and the app's use of it. All CPU and synthetic.

A1: the fast proximal column's positive ``h_pc0`` is a start-up load that leveling cannot
see. A2: the whole-layer restart at the forecast origin. A3: the step at the mid/distal
line, which the zone blend smooths. Every default must reproduce the runs made before.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyproj")

from hydrophysics.twin.calibrate_flow import _expand_zonal, zone_tensor  # noqa: E402
from hydrophysics.twin.compaction import VEPColumn, vep_compaction  # noqa: E402
from hydrophysics.twin.forward import (  # noqa: E402
    ZonalColumn,
    build_model,
    compaction,
    load_members,
    load_members_sidecar,
    load_or_fit_vep,
    nudge_to_observations,
    parse_scenario,
    release_fast_startup,
    run,
    write_members_sidecar,
)
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.inputs import TwinInputs  # noqa: E402
from hydrophysics.twin.zones import (  # noqa: E402
    DISTAL,
    MID,
    PROXIMAL,
    fan_zones,
    zone_blend_weights,
)


def _xy(*x_km):
    return np.column_stack([np.asarray(x_km, dtype="float64") * 1000.0, np.zeros(len(x_km))])


def _inputs(T=24, nx=6, ny=3):
    g = FanGrid(nx=nx, ny=ny, dx=1000.0, x0=200_000.0, y0=0.0,
                mask=np.ones((ny, nx), dtype=bool))
    A = g.n_active
    dates = pd.date_range("2012-01-01", periods=T, freq="MS")
    well_xy = np.array([[200_500.0, 500.0], [204_500.0, 1500.0], [205_500.0, 2500.0]])
    obs_idx = np.array([g.active_index(x, y) for x, y in well_xy])
    obs_layer = np.array([0, 1, 1])
    obs_h = np.tile(np.array([[3.0], [4.0], [6.0]]), (1, T)) + 0.1 * np.arange(T)
    obs_h[0, -1] = np.nan
    E = {"irrigation": np.full((A, T), 100.0), "aquaculture": np.full((A, T), 10.0)}
    return TwinInputs(grid=g, hf=None, dates=dates, obs_h=obs_h,
                      obs_h_filled=np.nan_to_num(obs_h, nan=3.0), obs_idx=obs_idx,
                      obs_layer=obs_layer, well_xy=well_xy, sids=["a", "b", "c"],
                      ground_elev=torch.full((A,), 12.0, dtype=torch.float64),
                      E_by_class=E,
                      recharge_field=torch.full((A, T), 1e-4, dtype=torch.float64))


def _theta_file(tmp_path, blend_km=None):
    theta = {"log_T_proximal": [np.log(500.0)], "log_S_proximal": [np.log(1e-3)],
             "log_T_mid": [np.log(300.0)] * 4, "log_S_mid": [np.log(1e-3)] * 4,
             "log_T_distal": [np.log(100.0)] * 4, "log_S_distal": [np.log(1e-3)] * 4,
             "log_L_mid": [np.log(1e-4)] * 3, "log_L_distal": [np.log(1e-4)] * 3,
             "log_eta": np.log(0.3), "log_head_extra": np.log(30.0),
             "recharge_frac_logit": 0.0,
             "log_C_coast": [np.log(50.0)] * 4, "log_C_apex": [np.log(50.0)]}
    meta = {"param_mode": "zonal", "boundaries": "coast-apex", "zone_boundaries": "205,203",
            "meter_filter": "none", "eta_classes": None}
    if blend_km:
        meta["zone_blend_km"] = blend_km
    p = tmp_path / "stage3_theta.json"
    p.write_text(json.dumps({"theta": theta, "meta": meta}))
    return str(p)


def _col_params(h_pc0=(0.0, 0.0, 0.0), tau=(24.0, 3960.0, 821.0)):
    return [{"log_ske": float(np.log(1e-3)), "log_skv": float(np.log(0.2)),
             "log_tau": float(np.log(t)), "h_pc0": float(h)} for h, t in zip(h_pc0, tau,
                                                                              strict=True)]


# ---- zone blend (A3) -------------------------------------------------------------------
def test_zone_blend_zero_is_the_one_hot_of_the_sharp_zones():
    xy = _xy(170.0, 181.9, 182.0, 190.0, 205.0, 214.0)
    w = zone_blend_weights(xy, 205.0, 182.0, 0.0)
    z = fan_zones(xy, 205.0, 182.0)
    assert w.shape == (3, 6)
    assert np.array_equal(w.argmax(0), z) and np.array_equal(w.sum(0), np.ones(6))
    assert set(np.unique(w)) <= {0.0, 1.0}


def test_zone_blend_mixes_mid_and_distal_only_and_sums_to_one():
    xy = _xy(170.0, 180.0, 182.0, 184.0, 204.9, 205.0, 214.0)
    w = zone_blend_weights(xy, 205.0, 182.0, 2.0)
    assert np.allclose(w.sum(0), 1.0)
    assert np.isclose(w[DISTAL, 2], 0.5) and np.isclose(w[MID, 2], 0.5)   # on the line
    assert w[DISTAL, 0] > 0.99 and w[DISTAL, 1] > w[DISTAL, 3]            # monotone
    # the proximal line stays sharp: a cell just west of it carries no proximal weight
    assert w[PROXIMAL, 4] == 0.0 and w[PROXIMAL, 5] == 1.0 and w[PROXIMAL, 6] == 1.0
    assert w[DISTAL, 5] == 0.0


def test_expand_zonal_with_one_hot_weights_equals_the_gather_and_blend_reaches_both_zones():
    n_layers = 4
    th = {"log_T_proximal": torch.tensor([[1.0]], dtype=torch.float64),
          "log_S_proximal": torch.tensor([[-7.0]], dtype=torch.float64),
          "log_T_mid": torch.full((4, 1), 2.0, dtype=torch.float64, requires_grad=True),
          "log_S_mid": torch.full((4, 1), -6.0, dtype=torch.float64),
          "log_T_distal": torch.full((4, 1), 4.0, dtype=torch.float64, requires_grad=True),
          "log_S_distal": torch.full((4, 1), -5.0, dtype=torch.float64),
          "log_L_mid": torch.full((3, 1), -9.0, dtype=torch.float64),
          "log_L_distal": torch.full((3, 1), -8.0, dtype=torch.float64)}
    xy = _xy(175.0, 181.0, 183.0, 190.0, 210.0)
    z = fan_zones(xy, 205.0, 182.0)
    sharp = _expand_zonal(th, zone_tensor(z), n_layers)
    onehot = _expand_zonal(th, zone_tensor(z, zone_w=zone_blend_weights(xy, 205.0, 182.0, 0)),
                           n_layers)
    for a, b in zip(sharp, onehot, strict=True):
        assert torch.allclose(a, b)
    blended, _, _ = _expand_zonal(th, zone_tensor(None, zone_w=zone_blend_weights(
        xy, 205.0, 182.0, 2.0)), n_layers)
    v = blended.detach()
    assert 2.0 < float(v[0, 1]) < 4.0 and 2.0 < float(v[0, 2]) < 4.0
    blended[:, 1].sum().backward()          # a cell next to the line feeds both zones
    assert th["log_T_mid"].grad.abs().sum() > 0 and th["log_T_distal"].grad.abs().sum() > 0


def test_build_model_rebuilds_a_blended_flow_field_from_meta(tmp_path):
    inp = _inputs()
    m0, _, zone = build_model(inp.grid, load_members([_theta_file(tmp_path)])[0], "cpu")
    (tmp_path / "b").mkdir()
    m1, _, zone1 = build_model(inp.grid, load_members([_theta_file(tmp_path / "b", 1.0)])[0],
                               "cpu")
    assert np.array_equal(zone, zone1)            # the zone ids themselves are unchanged
    x = inp.grid.centroids()[:, 0] / 1000.0
    far = zone == PROXIMAL                         # untouched by the mid/distal blend
    assert torch.allclose(m0.log_T[:, far], m1.log_T[:, far], atol=1e-6)
    near = (np.abs(x - 203.0) < 1.0) & (zone != PROXIMAL)
    assert not torch.allclose(m0.log_T[:, near], m1.log_T[:, near])


# ---- column: blend and start-up release (A1, A3) ----------------------------------------
def test_vep_compaction_is_the_columns_recurrence():
    col = VEPColumn(n_sites=1)
    with torch.no_grad():
        col.h_pc0.fill_(1.5)
    h = torch.linspace(10.0, 6.0, 20).repeat(3, 1)
    assert torch.equal(col(h), vep_compaction(h, col.log_ske, col.log_skv, col.log_tau,
                                              col.h_pc0, col.dt_days))


def test_blended_column_one_hot_equals_the_sharp_zonal_column():
    zone = np.array([0, 1, 1, 2, 2])
    xy = _xy(210.0, 190.0, 183.0, 181.0, 170.0)
    params = _col_params(h_pc0=(0.0, 1.0, -2.0))
    sharp = ZonalColumn(params, zone)
    onehot = ZonalColumn(params, zone, zone_w=zone_blend_weights(xy, 205.0, 182.0, 0.0))
    heads = torch.linspace(0.0, -4.0, 30).repeat(4, 5, 1)
    assert np.allclose(compaction(sharp, heads), compaction(onehot, heads), atol=1e-6)
    blend = ZonalColumn(params, zone, zone_w=zone_blend_weights(xy, 205.0, 182.0, 2.0))
    s_b, s_s = compaction(blend, heads), compaction(sharp, heads)
    # the two cells either side of the line move toward each other
    assert abs(s_b[2, -1] - s_b[3, -1]) < abs(s_s[2, -1] - s_s[3, -1])


def test_release_fast_startup_zeroes_only_fast_positive_offsets_and_removes_the_step(tmp_path):
    params = _col_params(h_pc0=(4.7, 2.6, -5.4), tau=(24.0, 3960.0, 821.0))
    p, changed = release_fast_startup({"zonal": params}, 365.0)
    assert changed == [0]
    assert [c["h_pc0"] for c in p["zonal"]] == [0.0, 2.6, -5.4]
    assert release_fast_startup({"zonal": params}, None)[1] == []
    f = tmp_path / "vep.json"
    f.write_text(json.dumps({"zonal": params}))
    zone = np.array([0, 0, 1])
    heads = torch.full((4, 3, 12), 5.0)                   # constant heads: no real signal
    before = compaction(load_or_fit_vep(str(f), None, None, "cpu", zone_of_cell=zone)[0],
                        heads)
    col, meta = load_or_fit_vep(str(f), None, None, "cpu", zone_of_cell=zone,
                                hpc0_fast_days=365.0)
    after = compaction(col, heads)
    assert before[0, 3] > 0.9                    # 0.2 x 4.7 m released within months
    assert abs(after[0, -1]) < 1e-6              # nothing to release once h_pc0 = 0
    assert np.allclose(before[2], after[2])      # the long-tau mid column is untouched
    assert meta["hpc0_released"]["columns"] == [0]


def test_blended_column_json_needs_zone_weights_and_uses_them(tmp_path):
    f = tmp_path / "vep.json"
    f.write_text(json.dumps({"zonal": _col_params(), "zone_blend_km": 2.0}))
    zone = np.array([1, 2])
    with pytest.raises(ValueError, match="zone_weights"):
        load_or_fit_vep(str(f), None, None, "cpu", zone_of_cell=zone)
    xy = _xy(183.0, 181.0)
    col, _ = load_or_fit_vep(str(f), None, None, "cpu", zone_of_cell=zone,
                             zone_weights=lambda km: zone_blend_weights(xy, 205.0, 182.0, km))
    assert col.zone_w is not None and col.zone_w.shape == (3, 2)


def test_calibrate_coupled_guard_and_blended_predict():
    import hydrophysics.twin.calibrate_coupled as cc

    col = VEPColumn(n_sites=1)
    with torch.no_grad():
        col.log_tau.fill_(float(np.log(24.0)))
        col.h_pc0.fill_(4.0)
    old = cc.HPC0_GUARD_DAYS
    try:
        cc.HPC0_GUARD_DAYS = None
        with torch.no_grad():
            cc._guard_hpc0(col)
        assert float(col.h_pc0.detach()) == 4.0
        cc.HPC0_GUARD_DAYS = 365.0
        with torch.no_grad():
            cc._guard_hpc0(col)
        assert float(col.h_pc0.detach()) == 0.0
    finally:
        cc.HPC0_GUARD_DAYS = old
    model = cc._make("zonal", "cpu")
    with torch.no_grad():
        for i, c in enumerate(model):
            c.log_skv.fill_(float(np.log(0.05 * (i + 1))))
    heads = torch.linspace(0.0, -3.0, 16).repeat(3, 4, 1)          # (n, L, T)
    zid = torch.tensor([0, 1, 2])
    xy = _xy(210.0, 190.0, 170.0)
    zw = torch.tensor(zone_blend_weights(xy, 205.0, 182.0, 0.0).T, dtype=torch.float32)
    assert torch.allclose(cc._predict(model, heads, zid), cc._predict(model, heads, zw),
                          atol=1e-6)


# ---- restart taper and free-running column (A2) ------------------------------------------
def test_restart_taper_keeps_the_model_state_far_from_wells():
    inp = _inputs()
    A = inp.grid.n_active
    origin = len(inp.dates) - 1
    h_model = torch.zeros(4, A, dtype=torch.float64)
    full = nudge_to_observations(inp, h_model, origin, gain=1.0)
    assert torch.equal(nudge_to_observations(inp, h_model, origin, 1.0, taper_km=None), full)
    tap = nudge_to_observations(inp, h_model, origin, gain=1.0, taper_km=0.5)
    well = inp.obs_idx[1]                       # a layer-1 well cell: centre on the well
    assert torch.isclose(tap[1, well], full[1, well])
    cent = inp.grid.centroids()
    d = np.sqrt(((cent[:, None, :] - inp.well_xy[None, 1:]) ** 2).sum(-1)).min(1)
    # cells several radii from every layer-1 well keep (almost) the model state
    far = d > 2500.0
    assert far.any() and float(tap[1, far].abs().max()) < 1e-3 * float(full[1].abs().max())


def test_run_defaults_unchanged_free_column_and_member_sidecar(tmp_path):
    inp = _inputs()
    members = load_members([_theta_file(tmp_path)])
    scen = [(parse_scenario("base:irrigation=1")[0], 1.0),
            (parse_scenario("cut:irrigation=0.5")[0], 1.0)]
    col = VEPColumn(n_sites=1, dt_days=30.0)
    kw = dict(horizon=13, gain=1.0, ic_members=1, ic_sigma=0.3, seed=0, col=col,
              device="cpu", log=lambda *_: None)
    ref = run(inp, members, scen, **kw)
    same = run(inp, members, scen, **kw, column_heads="restart", restart_taper_km=None,
               save_members=None)
    assert np.array_equal(ref["subs_mean"], same["subs_mean"])
    free = run(inp, members, scen, **kw, column_heads="free", save_members="yearly")
    # displayed heads are still the restarted ones; only the column's driver changed
    assert np.array_equal(ref["heads_mean"], free["heads_mean"])
    assert np.array_equal(ref["subs_mean"][:, :, :24], free["subs_mean"][:, :, :24])
    assert not np.allclose(ref["subs_mean"][:, :, 24:], free["subs_mean"][:, :, 24:])
    # the ic perturbation still reaches the column under the free driver
    my = free["members_yearly"]
    assert my["subs"].shape == (2, 2, inp.grid.n_active, 3)       # Dec 2012, 2013, 2014
    assert my["years"] == [2012, 2013, 2014] and my["ic"] == [0, 1]
    assert not np.allclose(my["subs"][:, 0, :, -1], my["subs"][:, 1, :, -1])
    # the member mean of the yearly fields is the saved ensemble mean at those months
    assert np.allclose(my["subs"].mean(1), free["subs_mean"][..., my["ye_idx"]], atol=1e-6)
    path = str(tmp_path / "run.members.npz")
    sat = write_members_sidecar(path, my, free["scenario_names"], free["origin"],
                                free["dates"])
    assert sat == {"subs": 0, "head": 0}
    back = load_members_sidecar(path)
    assert np.abs(back["subs_cm"] - my["subs"] * 100.0).max() <= 0.05 + 1e-6    # 1 mm
    assert np.abs(back["headL2_m"] - my["headL2"]).max() <= 0.005 + 1e-6        # 1 cm
    assert back["scenario_names"] == ["base", "cut"]
    assert back["member"] == [members[0].label] * 2 and back["head_member"] == back["member"]


def test_run_rejects_unknown_options(tmp_path):
    inp = _inputs()
    members = load_members([_theta_file(tmp_path)])
    scen = [(parse_scenario("base:irrigation=1")[0], 1.0)]
    with pytest.raises(ValueError, match="column_heads"):
        run(inp, members, scen, 2, 1.0, 0, 0.1, 0, VEPColumn(1), "cpu", log=lambda *_: None,
            column_heads="nudged")


# ---- the app reads real per-member agreement ------------------------------------------------
def test_prep_agreement_bands_and_townships_from_member_fields(tmp_path):
    from hydrophysics.twin.app import prep

    S, M, A, Y = 2, 6, 4, 3
    rng = np.random.default_rng(0)
    base = rng.normal(10.0, 1.0, size=(M, A, Y))
    subs = np.stack([base, base.copy()])
    # scenario 1: cells 0-1 benefit in every member; cell 2 splits; cell 3 unchanged
    subs[1, :, 0, 1:] -= 1.0
    subs[1, :, 1, 1:] -= 0.5
    subs[1, :3, 2, 1:] -= 0.2
    subs[1, 3:, 2, 1:] += 0.2
    mf = {"subs": subs.astype("float32"), "headL2": np.zeros((S, M, A, Y), "float32"),
          "sets": ["a", "a", "b", "b", "c", "c"]}
    d = prep.member_delta(mf, 1, y_ref=0)
    assert d.shape == (M, A, Y) and np.allclose(d[..., 0], 0.0)
    ag = prep.cell_agreement(d, mf["sets"])
    assert ag.shape == (A, Y)
    assert ag[0, -1] == 1.0 and ag[1, -1] == 1.0
    assert ag[2, -1] < prep.AGREE_MIN            # sets a/b say benefit, c says harm
    assert ag[3, -1] == 0.0                       # nothing changed: nobody "agrees"
    band = prep.fan_member_band(d)
    assert band.shape == (5, Y) and np.allclose(band[:, 0], 0.0)
    assert (np.diff(band[:, -1]) >= -1e-12).all()
    towns = prep.township_agreement(d, mf["sets"], np.array([1, 1, 2, 0], "uint8"), 3)
    assert towns[1] == {"agree": 3, "n": 3} and towns[0]["n"] == 3
    # round trip through the forward sidecar format
    my = {"subs": subs / 100.0, "headL2": np.zeros((S, M, A, Y)), "years": [2012, 2013, 2014],
          "ye_idx": np.array([11, 23, 35]), "member": ["a", "a", "b", "b", "c", "c"],
          "ic": [0, 1] * 3, "rheology": ["column"] * M, "head_member": ["a", "a", "b", "b",
                                                                          "c", "c"],
          "head_ic": [0, 1] * 3}
    path = str(tmp_path / "x.members.npz")
    write_members_sidecar(path, my, ["baseline", "cut"], 23,
                          pd.date_range("2012-01-01", periods=36, freq="MS"))
    back = prep.load_member_fields(path)
    assert back["sets"] == mf["sets"] and back["years"] == [2012, 2013, 2014]
    assert np.abs(back["subs"] - subs).max() <= 0.05 + 1e-5
