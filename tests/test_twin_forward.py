"""Forward twin plumbing on synthetic inputs: scenario parsing, climatological forcing,
observation nudging, parameter-set loading and a tiny end-to-end run."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.compaction import VEPColumn  # noqa: E402
from hydrophysics.twin.forward import (  # noqa: E402
    build_model,
    future_forcing,
    load_members,
    nudge_to_observations,
    parse_scenario,
    run,
)
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.inputs import TwinInputs  # noqa: E402
from hydrophysics.twin.zones import fan_zones  # noqa: E402


def _inputs(T=24, nx=6, ny=3):
    g = FanGrid(nx=nx, ny=ny, dx=1000.0, x0=200_000.0, y0=0.0,
                mask=np.ones((ny, nx), dtype=bool))
    A = g.n_active
    dates = pd.date_range("2012-01-01", periods=T, freq="MS")
    well_xy = np.array([[200_500.0, 500.0], [204_500.0, 1500.0], [205_500.0, 2500.0]])
    obs_idx = np.array([g.active_index(x, y) for x, y in well_xy])
    obs_layer = np.array([0, 1, 1])
    obs_h = np.tile(np.array([[3.0], [4.0], [6.0]]), (1, T)) + 0.1 * np.arange(T)
    obs_h[0, -1] = np.nan                       # well 0 unobserved at the origin
    E = {"irrigation": np.full((A, T), 100.0), "aquaculture": np.full((A, T), 10.0)}
    return TwinInputs(grid=g, hf=None, dates=dates, obs_h=obs_h,
                      obs_h_filled=np.nan_to_num(obs_h, nan=3.0), obs_idx=obs_idx,
                      obs_layer=obs_layer, well_xy=well_xy, sids=["a", "b", "c"],
                      ground_elev=torch.full((A,), 12.0, dtype=torch.float64),
                      E_by_class=E,
                      recharge_field=torch.full((A, T), 1e-4, dtype=torch.float64))


def test_parse_scenario_reads_classes_rain_start_and_zones():
    s, rain = parse_scenario("cut:irrigation=0.7,aquaculture=0,rain=0.9@2026-01#proximal/mid")
    assert s.name == "cut" and s.factors == {"irrigation": 0.7, "aquaculture": 0.0}
    assert rain == 0.9 and s.start == "2026-01" and s.zones == ("proximal", "mid")
    with pytest.raises(ValueError, match="unknown class"):
        parse_scenario("x:golf=2")
    with pytest.raises(ValueError):
        parse_scenario("no-colon")


def test_future_forcing_is_climatology_under_the_policy():
    inp = _inputs()
    scen, rain = parse_scenario("half:irrigation=0.5")
    E, r, dates = future_forcing(inp, scen, horizon=4, rain_scale=rain)
    assert E.shape == (inp.grid.n_active, 4) and r.shape == (inp.grid.n_active, 4)
    assert len(dates) == 4 and dates[0] == inp.dates[-1] + pd.offsets.MonthBegin(1)
    assert torch.allclose(E, torch.full_like(E, 0.5 * 100.0 + 10.0))
    # per-class axis when the calibration used efficiency classes
    E3, _, _ = future_forcing(inp, scen, horizon=4, eta_classes=["irrigation", "aquaculture"])
    assert E3.shape == (2, inp.grid.n_active, 4)
    assert torch.allclose(E3[0], torch.full_like(E3[0], 50.0))
    assert torch.allclose(E3[1], torch.full_like(E3[1], 10.0))


def test_nudging_moves_observed_layers_only_and_gain_zero_is_a_no_op():
    inp = _inputs()
    A = inp.grid.n_active
    h_model = torch.zeros(4, A, dtype=torch.float64)
    origin = len(inp.dates) - 1
    same = nudge_to_observations(inp, h_model, origin, gain=0.0)
    assert torch.equal(same, h_model)
    out = nudge_to_observations(inp, h_model, origin, gain=1.0)
    # layer 0's only well is NaN at the origin -> untouched; layer 1 has two wells
    assert torch.equal(out[0], h_model[0])
    assert (out[1] > 0).all()
    assert torch.equal(out[2], h_model[2]) and torch.equal(out[3], h_model[3])
    half = nudge_to_observations(inp, h_model, origin, gain=0.5)
    assert torch.allclose(half[1], 0.5 * out[1])


def _theta_file(tmp_path, boundaries="coast-apex", eta_classes=None):
    log_eta = [np.log(0.3), np.log(0.2)] if eta_classes else np.log(0.3)
    theta = {"log_T_proximal": [np.log(500.0)], "log_S_proximal": [np.log(1e-3)],
             "log_T_mid": [np.log(300.0)] * 4, "log_S_mid": [np.log(1e-3)] * 4,
             "log_T_distal": [np.log(100.0)] * 4, "log_S_distal": [np.log(1e-3)] * 4,
             "log_L_mid": [np.log(1e-4)] * 3, "log_L_distal": [np.log(1e-4)] * 3,
             "log_eta": log_eta, "log_head_extra": np.log(30.0), "recharge_frac_logit": 0.0}
    if boundaries == "coast-apex":
        theta["log_C_coast"] = [np.log(50.0)] * 4
        theta["log_C_apex"] = [np.log(50.0)]
    meta = {"param_mode": "zonal", "boundaries": boundaries, "zone_boundaries": "205,203",
            "meter_filter": "none", "eta_classes": eta_classes}
    p = tmp_path / "stage3_theta.json"
    p.write_text(json.dumps({"theta": theta, "meta": meta}))
    folds = [{"fold": 0, "n_held": 1, "r2_kfold": 0.1, "r2_idw": 0.2, "theta": theta}]
    f = tmp_path / "stage3_fold_thetas.json"
    f.write_text(json.dumps(folds))
    return str(p), str(f)


def test_load_members_reads_in_sample_and_fold_files(tmp_path):
    p, f = _theta_file(tmp_path)
    members = load_members([p, f])
    assert [m.label for m in members] == [tmp_path.name, "fold0"]
    assert members[1].meta == members[0].meta
    with pytest.raises(ValueError, match="in-sample theta file before"):
        load_members([f])


def test_build_model_rebuilds_zones_and_boundaries(tmp_path):
    inp = _inputs()
    p, _ = _theta_file(tmp_path)
    m, scalars, zone = build_model(inp.grid, load_members([p])[0], "cpu")
    assert m.has_boundaries and m.boundaries.n_coast == inp.grid.ny
    expect = fan_zones(inp.grid.centroids(), proximal_km=205.0, distal_km=203.0)
    assert np.array_equal(zone, expect)
    # the proximal merged aquifer got log_T 500 in every layer, the distal 100
    prox = zone == 0
    assert torch.allclose(torch.exp(m.log_T[:, prox]), torch.full((4, prox.sum()), 500.0,
                                                                 dtype=torch.float64))
    assert set(scalars) == {"log_eta", "log_head_extra", "recharge_frac_logit"}


def test_end_to_end_run_returns_mean_and_spread_per_scenario(tmp_path):
    inp = _inputs()
    p, f = _theta_file(tmp_path)
    members = load_members([p, f])
    scen = [(parse_scenario("base:irrigation=1")[0], 1.0),
            (parse_scenario("cut:irrigation=0.5")[0], 1.0)]
    col = VEPColumn(n_sites=1, dt_days=30.0)
    res = run(inp, members, scen, horizon=3, gain=1.0, ic_members=1, ic_sigma=0.1,
              seed=0, col=col, device="cpu", log=lambda *_: None)
    T = len(inp.dates)
    assert res["heads_mean"].shape == (2, 4, inp.grid.n_active, T + 3)
    assert res["subs_mean"].shape == (2, inp.grid.n_active, T + 3)
    assert res["n_members"] == 4 and res["origin"] == T - 1
    assert np.isfinite(res["heads_mean"]).all() and (res["heads_std"] >= 0).all()
    # halving irrigation must leave heads no lower than the baseline at the horizon
    assert (res["heads_mean"][1, 1, :, -1] >= res["heads_mean"][0, 1, :, -1] - 1e-9).all()
    assert len(res["rows"]) == 2 * 4


def test_forward_figure_builds_from_a_run_npz(tmp_path):
    """The viewer's forward mode: one dropdown entry per scenario, a slider through the
    projection, and the gate verdict in the title."""
    pytest.importorskip("plotly")
    from hydrophysics.twin.explorer3d import _coarsen, build_forward_figure

    inp = _inputs()
    p, f = _theta_file(tmp_path)
    members = load_members([p, f])
    members[0].meta["gate"] = {"verdict": "FAIL", "r2_kfold": 0.4, "r2_idw": 0.7}
    scen = [(parse_scenario("base:irrigation=1")[0], 1.0),
            (parse_scenario("cut:irrigation=0.5")[0], 1.0)]
    res = run(inp, members, scen, horizon=3, gain=1.0, ic_members=0, ic_sigma=0.0,
              seed=0, col=VEPColumn(n_sites=1, dt_days=30.0), device="cpu",
              log=lambda *_: None)
    out = tmp_path / "run.npz"
    np.savez_compressed(
        out, dates=np.array([d.isoformat() for d in res["dates"]]), origin=res["origin"],
        heads_mean=res["heads_mean"], heads_std=res["heads_std"],
        subs_mean=res["subs_mean"], subs_std=res["subs_std"],
        scenario_names=np.array(res["scenario_names"]), scenarios=np.array(res["scenarios"]),
        mask=inp.grid.mask, nx=inp.grid.nx, ny=inp.grid.ny, dx=inp.grid.dx,
        x0=inp.grid.x0, y0=inp.grid.y0, n_members=res["n_members"],
        member_labels=np.array(res["member_labels"]),
        hindcast_r2=np.array(res["hindcast_r2"]),
        gate=json.dumps(members[0].meta), horizon=3, gain=1.0)
    fw = np.load(out, allow_pickle=False)
    fig = build_forward_figure(fw, stride=4, coarsen=1)
    n_scen, n_per = 2, 5
    assert len(fig.data) == n_scen * n_per
    assert [tr.visible for tr in fig.data] == [True] * n_per + [False] * n_per
    dropdown = fig.layout.updatemenus[1]
    assert len(dropdown.buttons) == n_scen
    assert dropdown.buttons[1].args[0]["visible"] == [False] * n_per + [True] * n_per
    labels = [s.label for s in fig.layout.sliders[0].steps]
    assert labels[-1].endswith("▸") and not labels[0].endswith("▸")
    assert "FAIL" in fig.layout.title.text
    # coarsening keeps NaN outside the fan and halves the raster
    r = _coarsen(np.where(inp.grid.mask, 1.0, np.nan), 2)
    assert r.shape == ((inp.grid.ny + 1) // 2, inp.grid.nx // 2)
    assert np.nanmax(r) == 1.0


def test_sequential_nudging_pulls_the_hindcast_toward_observations(tmp_path):
    from hydrophysics.twin.forward import hindcast_with_nudging

    inp = _inputs()
    p, _ = _theta_file(tmp_path)
    m, scalars, _ = build_model(inp.grid, load_members([p])[0], "cpu")
    h0 = inp.initial_heads(0)
    E = inp.E_total[:, 1:]
    R = inp.recharge_field[:, 1:]
    free = hindcast_with_nudging(m, scalars, inp, h0, E, R, gain=0.0, every=6)
    nudged = hindcast_with_nudging(m, scalars, inp, h0, E, R, gain=1.0, every=6)
    assert free.shape == nudged.shape == (4, inp.grid.n_active, E.shape[-1] + 1)
    obs_l1 = inp.obs_h[1:, :]                       # wells in layer 1 (index 1)
    idx = inp.obs_idx[1:]
    # at a nudge month (6) the nudged state sits on the observations; the free one need not
    err_free = np.abs(free[1, idx, 6].numpy() - obs_l1[:, 6]).max()
    err_nudge = np.abs(nudged[1, idx, 6].numpy() - obs_l1[:, 6]).max()
    assert err_nudge < err_free
    assert torch.equal(free[..., :6], nudged[..., :6])   # identical until the first nudge


def test_zonal_column_json_loads_and_applies_per_zone(tmp_path):
    from hydrophysics.twin.forward import ZonalColumn, compaction, load_or_fit_vep

    params = [{"log_ske": np.log(1e-3), "log_skv": np.log(2e-2), "log_tau": np.log(365.0),
               "h_pc0": 0.0},
              {"log_ske": np.log(5e-3), "log_skv": np.log(2e-2), "log_tau": np.log(365.0),
               "h_pc0": 0.0},
              {"log_ske": np.log(1e-3), "log_skv": np.log(2e-2), "log_tau": np.log(365.0),
               "h_pc0": 0.0}]
    p = tmp_path / "vep_zonal.json"
    p.write_text(json.dumps({"zonal": params, "config": "zonal"}))
    zone = np.array([0, 1, 1, 2])
    col, meta = load_or_fit_vep(str(p), None, None, "cpu", zone_of_cell=zone)
    assert isinstance(col, ZonalColumn) and meta["config"] == "zonal"
    heads = torch.full((4, 4, 6), 5.0, dtype=torch.float64)
    heads[..., 3:] = 4.0                                   # a 1 m drawdown from month 3
    subs = compaction(col, heads)
    assert subs.shape == (4, 6)
    # the mid zone's elastic coefficient is 5x the others -> more compaction there
    assert subs[1, -1] > subs[0, -1] and subs[1, -1] > subs[3, -1]
    with pytest.raises(ValueError, match="zone_of_cell"):
        load_or_fit_vep(str(p), None, None, "cpu")
