"""Rung R1c (2026-10-15): ``--zone-county-split`` gives every zone a Changhua copy.

Opt-in: without the flag every zone map, parameter dict, meta and recipe is the historical
one. With it, the copies start at their parents' values (epoch 0 is the unsplit model), the
split travels in the theta meta with the county map's hash, and every consumer
(``forward.build_model``, the zonal column, scenario zones, merge_folds) rebuilds exactly
the same zones or refuses.
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

import hydrophysics.twin.calibrate_coupled as cc  # noqa: E402
from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    BOUNDS,
    COUNTY_ARGS,
    _base_param_name,
    _expand_zonal,
    _flow_cfg,
    _flow_fit_cols,
    _make_zonal_params,
    _theta_zone_names,
    _write_theta,
    _zonal_bounds_hit,
    fit_flow,
    kfold_wells,
    main,
    set_log_t_min_proximal,
    write_fit_summary,
    write_fold_file,
)
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.forward import (  # noqa: E402
    Member,
    ZonalColumn,
    build_model,
    load_members,
    load_or_fit_vep,
)
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.inputs import input_options  # noqa: E402
from hydrophysics.twin.merge_folds import merge  # noqa: E402
from hydrophysics.twin.scenario import PumpingScenario  # noqa: E402
from hydrophysics.twin.zone_county import (  # noqa: E402
    column_zone_map,
    column_zone_names,
    column_zone_weights,
    county_flags_for_grid,
    county_flags_from_meta,
    county_meta,
    county_sha256,
    flow_zone_map,
    save_county_npz,
)
from hydrophysics.twin.zones import (  # noqa: E402
    COUNTY_SUFFIX,
    ZONE_NAMES,
    ZONE_NAMES_SPLIT,
    collapse_county_zones,
    collapse_zones,
    county_split,
    county_split_weights,
    fan_band_of,
    fan_zones,
    split_county,
    zone_blend_weights,
    zone_labels,
    zone_names,
)

D = torch.float64
ZB = "205,201"            # distal x < 201 km, mid 201-205, proximal >= 205
ZB_SPLIT = "205,201,209"  # proximal_w 205-209


def _grid():
    # 8 x 4 cells of 2 km: centroids at x = 200..214 km, y = 2641..2647 km
    return FanGrid(nx=8, ny=4, dx=2000.0, x0=199_000.0, y0=2_640_000.0,
                   mask=np.ones((4, 8), dtype=bool))


def _changhua(g):
    """Synthetic county: the northern half of the grid is 'Changhua'."""
    return g.centroids()[:, 1] > 2_644_000.0


@pytest.fixture
def county(tmp_path):
    """A county map npz for the synthetic grid and the meta a calibration records."""
    g = _grid()
    flags = _changhua(g)
    path = str(tmp_path / "zone_county_map.npz")
    save_county_npz(path, g, flags, polygon="synthetic", gpkg="synthetic.gpkg")
    return g, flags, path, county_meta(path, flags)


# ---------------------------------------------------------------------------------------
# zone ids
# ---------------------------------------------------------------------------------------
def test_zone_labels_and_ids_of_the_county_split():
    assert zone_labels() == ZONE_NAMES and zone_labels(True) == ZONE_NAMES_SPLIT
    six = zone_labels(county=True)
    assert six == ZONE_NAMES + tuple(f"{z}{COUNTY_SUFFIX}" for z in ZONE_NAMES)
    eight = zone_labels(True, True)
    assert eight[:4] == ZONE_NAMES_SPLIT and eight[4] == "proximal_changhua"
    assert eight[7] == "proximal_w_changhua"
    for n in (3, 4, 6, 8):
        assert len(zone_names(n)) == n
    with pytest.raises(ValueError):
        zone_names(5)
    z = np.array([0, 1, 2, 3, 0, 3])
    ch = np.array([False, True, True, True, False, False])
    zc = county_split(z, ch, 4)
    assert zc.tolist() == [0, 5, 6, 7, 0, 3]
    base, back = split_county(zc, 4)
    assert np.array_equal(base, z) and np.array_equal(back, ch)
    # the column's map: proximal_w back into proximal, the county kept
    assert collapse_county_zones(zc, 4).tolist() == [0, 4, 5, 3, 0, 0]
    assert fan_band_of(collapse_county_zones(zc, 4)).tolist() == collapse_zones(z).tolist()
    with pytest.raises(ValueError):
        county_split(np.array([3]), np.array([True]), 3)   # id outside the base zones
    with pytest.raises(ValueError):
        fan_band_of(np.array([0, 7]))                      # an uncollapsed flow map


def test_county_split_weights_are_the_one_hot_of_the_split_map():
    g = _grid()
    xy, ch = g.centroids(), _changhua(g)
    w = zone_blend_weights(xy, 205.0, 201.0, 0.0, split_km=209.0)
    wc = county_split_weights(w, ch)
    zc = county_split(fan_zones(xy, 205.0, 201.0, 209.0), ch, 4)
    assert wc.shape == (8, g.n_active) and np.allclose(wc.sum(0), 1.0)
    assert np.allclose(wc[zc, np.arange(g.n_active)], 1.0)
    wb = county_split_weights(zone_blend_weights(xy, 205.0, 201.0, 2.0), ch)
    assert wb.shape == (6, g.n_active) and np.allclose(wb.sum(0), 1.0)
    assert np.allclose(wb[:3][:, ch], 0.0) and np.allclose(wb[3:][:, ~ch], 0.0)


# ---------------------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------------------
HISTORICAL_SPLIT_KEYS = ["log_T_proximal", "log_S_proximal", "log_T_mid", "log_S_mid",
                         "log_T_distal", "log_S_distal", "log_L_mid", "log_L_distal",
                         "log_T_proximal_w", "log_S_proximal_w", "recharge_frac_logit",
                         "log_Sd_proximal", "log_Sd_mid", "log_Sd_distal",
                         "log_Sd_proximal_w", "log_tau_proximal", "log_tau_mid",
                         "log_tau_distal", "log_tau_proximal_w"]


def test_default_parameter_dict_is_the_historical_one():
    m = FlowModel(_grid(), n_layers=4, dt_days=30.0)
    th = _make_zonal_params(m, use_recharge=True, split=True, delay_storage="zonal")
    assert list(th) == HISTORICAL_SPLIT_KEYS
    assert not any(k.endswith(COUNTY_SUFFIX) for k in th)


def test_changhua_copies_start_at_their_parents_and_double_the_zonal_part():
    m = FlowModel(_grid(), n_layers=4, dt_days=30.0)
    base = _make_zonal_params(m, use_recharge=True, split=True, delay_storage="zonal",
                              proximal_layered=True)
    th = _make_zonal_params(m, use_recharge=True, split=True, delay_storage="zonal",
                            proximal_layered=True, county=True)
    assert list(th)[:len(base)] == list(base)          # the unsplit dict is a prefix
    copies = [k for k in th if k.endswith(COUNTY_SUFFIX)]
    zonal = [k for k in base if k != "recharge_frac_logit"]
    assert copies == [f"{k}{COUNTY_SUFFIX}" for k in zonal]
    for k in zonal:
        assert torch.equal(th[k], th[f"{k}{COUNTY_SUFFIX}"])
        assert th[k] is not th[f"{k}{COUNTY_SUFFIX}"]            # independent tensors
        assert _base_param_name(f"{k}{COUNTY_SUFFIX}") == _base_param_name(k)
    n = sum(p.numel() for p in th.values())
    n0 = sum(p.numel() for p in base.values())
    assert n - n0 == sum(base[k].numel() for k in zonal)
    assert _theta_zone_names(th) == zone_labels(True, True)
    assert _base_param_name("log_L_proximal_w_changhua") == "log_L"
    assert _base_param_name("recharge_frac_logit") == "recharge_frac_logit"


def test_expand_zonal_maps_each_cell_to_its_own_copy_and_sums_gradients():
    g = _grid()
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    ch = _changhua(g)
    z = fan_zones(g.centroids(), 205.0, 201.0, 209.0)
    zc = torch.tensor(county_split(z, ch, 4))
    th0 = _make_zonal_params(m, split=True)
    th = _make_zonal_params(m, split=True, county=True)
    a = _expand_zonal(th0, torch.tensor(z), 4)
    b = _expand_zonal(th, zc, 4)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)                       # epoch 0: the unsplit field
    # the gradient of a parent in the unsplit model is the sum of its two halves
    w = torch.randn(4, g.n_active, dtype=D, generator=torch.Generator().manual_seed(1))
    (a[0] * w).sum().backward()
    (b[0] * w).sum().backward()
    for k in ("log_T_mid", "log_T_proximal", "log_T_proximal_w"):
        assert torch.allclose(th0[k].grad, th[k].grad + th[f"{k}{COUNTY_SUFFIX}"].grad)
        assert torch.count_nonzero(th[f"{k}{COUNTY_SUFFIX}"].grad) > 0
    # a Changhua copy moves only the Changhua cells of its zone
    with torch.no_grad():
        th["log_T_mid_changhua"].fill_(math.log(1234.0))
    c = _expand_zonal(th, zc, 4)[0]
    sel = (zc == zone_labels(True, True).index("mid_changhua")).numpy()
    assert sel.any()
    assert torch.allclose(c[:, sel], torch.tensor(math.log(1234.0), dtype=D))
    assert torch.equal(c[:, ~sel], b[0][:, ~sel])


def test_proximal_floor_and_bounds_report_cover_the_copies():
    th = {f"log_T_{z}": torch.full((1 if z.startswith("prox") else 4, 1), math.log(10.0),
                                   dtype=D) for z in zone_labels(True, True)}
    try:
        set_log_t_min_proximal(58.0)
        hits = _zonal_bounds_hit(th)
        for z in ("proximal", "proximal_w", "proximal_changhua", "proximal_w_changhua"):
            assert math.exp(float(th[f"log_T_{z}"])) == pytest.approx(58.0)
            assert hits[z]["log_T"]["lo"] == 1
        assert hits["mid_changhua"]["log_T"]["lo"] == 4
        assert set(hits) == set(zone_labels(True, True)) | {"global"}
    finally:
        set_log_t_min_proximal(None)
    from hydrophysics.twin.uncertainty import effective_bounds

    idx = [("log_T_proximal_changhua", 0), ("log_T_proximal_w_changhua", 0),
           ("log_T_mid_changhua", 0)]
    lo, _ = effective_bounds(idx, {"log_t_min_proximal": 58.0})
    assert np.allclose(lo[:2], math.log(58.0)) and lo[2] == pytest.approx(BOUNDS["log_T"][0])


# ---------------------------------------------------------------------------------------
# the fit: epoch 0 is the unsplit model; the meta round trip
# ---------------------------------------------------------------------------------------
def _problem(g, T=4):
    A = g.n_active
    x = torch.tensor(g.centroids()[:, 0], dtype=D)
    h0 = (10.0 + (x - x.mean()) / 2000.0).expand(4, A).clone()
    rch = torch.zeros(4, A, T, dtype=D)
    rf = torch.full((A, T), 4e-4, dtype=D)
    idx = torch.tensor([0, 5, 10, 15, 20, 25, 30, 7])
    lay = torch.tensor([0, 1, 2, 3, 0, 1, 2, 1])
    # Changhua wells (north half) swing less than the rest: something to split
    north = torch.tensor(_changhua(g)[idx.numpy()], dtype=D)
    obs = (h0[lay, idx][:, None] + 0.2 * (1.0 - 0.7 * north)[:, None]
           * torch.arange(1, T + 1, dtype=D)[None, :])
    return h0, rch, rf, idx, lay, obs


def _fit(g, zone, zonation=None, epochs=1, lr=0.05, **kw):
    h0, rch, rf, idx, lay, obs = _problem(g)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    fit = fit_flow(m, obs, idx, lay, rch, recharge_field=rf, epochs=epochs, lr=lr,
                   param_mode="zonal", h0=h0, zone_of_cell=zone, zonation=zonation, **kw)
    return m, fit


@pytest.mark.parametrize("s,extra", [(None, {}), (209.0, {"delay_storage": "zonal",
                                                          "proximal_layered": True})])
def test_epoch0_loss_equals_the_unsplit_model(s, extra):
    g = _grid()
    z = fan_zones(g.centroids(), 205.0, 201.0, s)
    n_base = 3 + int(s is not None)
    zc = county_split(z, _changhua(g), n_base)
    _, f0 = _fit(g, z, epochs=1, **extra)
    _, f1 = _fit(g, zc, zonation=zone_labels(s is not None, True), epochs=1, **extra)
    assert f1["loss"] == f0["loss"]                    # bit for bit, before any step
    assert f1["n_params"] > f0["n_params"]
    assert set(f1["bounds_hit"]) == set(zone_labels(s is not None, True)) | {"global"}
    # with steps, the copies move away from their parents: the split is real
    _, f2 = _fit(g, zc, zonation=zone_labels(s is not None, True), epochs=6, **extra)
    th = f2["theta"]
    assert any(not np.allclose(th[k], th[f"{k}{COUNTY_SUFFIX}"]) for k in th
               if f"{k}{COUNTY_SUFFIX}" in th)


def test_fit_flow_refuses_a_map_outside_its_zonation():
    g = _grid()
    zc = county_split(fan_zones(g.centroids(), 205.0, 201.0), _changhua(g), 3)
    with pytest.raises(ValueError, match="fall outside"):
        _fit(g, zc)                                    # ids 3-5 but no zonation given
    with pytest.raises(ValueError, match="zone_labels"):
        _fit(g, zc, zonation=("proximal", "mid", "distal", "x", "y", "z"))


def test_theta_meta_round_trip_to_build_model_and_the_column_map(county):
    g, flags, path, cmeta = county
    zone, names = flow_zone_map(g, {"zone_boundaries": ZB_SPLIT, **cmeta})
    assert names == zone_labels(True, True)
    m, fit = _fit(g, zone, zonation=names, epochs=3, delay_storage="zonal")
    meta = {"param_mode": "zonal", "zone_boundaries": ZB_SPLIT, **cmeta}
    rebuilt, scalars, zoc = build_model(g, Member("m", fit["theta"], meta), "cpu")
    assert torch.allclose(rebuilt.log_T, m.log_T) and torch.allclose(rebuilt.log_S, m.log_S)
    assert torch.allclose(rebuilt.log_L, m.log_L)
    assert torch.allclose(scalars["delay_Sd"], m.delay_log_Sd)
    # callers get the column's map: six zones, the same one column_zone_map rebuilds
    assert zoc.max() == 5 and np.array_equal(zoc, collapse_county_zones(zone, 4))
    assert np.array_equal(zoc, column_zone_map(g, meta))
    assert np.array_equal(fan_band_of(zoc), collapse_zones(fan_zones(g.centroids(), 205.0,
                                                                     201.0, 209.0)))
    assert column_zone_names(meta) == zone_labels(county=True)
    # the split must be recorded, and recorded for this county map
    with pytest.raises(ValueError, match="county split"):
        build_model(g, Member("m", fit["theta"], {"param_mode": "zonal",
                                                  "zone_boundaries": ZB_SPLIT}), "cpu")
    with pytest.raises(ValueError, match="sha256"):
        build_model(g, Member("m", fit["theta"], {**meta, "zone_county_sha256": "0" * 64}),
                    "cpu")
    unsplit = {k: v for k, v in fit["theta"].items() if not k.endswith(COUNTY_SUFFIX)}
    with pytest.raises(ValueError, match="county split"):
        build_model(g, Member("m", unsplit, meta), "cpu")


def test_theta_file_meta_and_load_members_round_trip(county, tmp_path):
    g, flags, path, cmeta = county
    zone, names = flow_zone_map(g, {"zone_boundaries": ZB, **cmeta})
    _, fit = _fit(g, zone, zonation=names, epochs=2)
    p = tmp_path / "stage3_theta.json"
    _write_theta(str(p), fit["theta"], {"param_mode": "zonal", "zone_boundaries": ZB,
                                        **cmeta})
    (mem,) = load_members([str(p)])
    assert mem.meta["zone_county_sha256"] == county_sha256(flags)
    assert np.array_equal(county_flags_from_meta(g, mem.meta), flags)
    _, _, zoc = build_model(g, mem, "cpu")
    assert np.array_equal(zoc, zone)                   # no proximal split: the same map
    # the default meta reads no county map at all
    assert county_flags_from_meta(g, {"param_mode": "zonal"}) is None


def test_county_map_cache_is_used_only_on_its_own_grid(county):
    g, flags, path, cmeta = county
    got, src = county_flags_for_grid(g, path, expect_sha256=cmeta["zone_county_sha256"])
    assert src == path and np.array_equal(got, flags)
    other = FanGrid(nx=8, ny=4, dx=2000.0, x0=201_000.0, y0=2_640_000.0,
                    mask=np.ones((4, 8), dtype=bool))
    with pytest.raises(FileNotFoundError):             # not this grid: rebuilt from the gpkg
        county_flags_for_grid(other, path, gpkg="no/such/file.gpkg")


# ---------------------------------------------------------------------------------------
# consumers
# ---------------------------------------------------------------------------------------
def _col(tau_days, skv=1e-2):
    return {"log_ske": math.log(1e-4), "log_skv": math.log(skv),
            "log_tau": math.log(tau_days), "h_pc0": 0.0}


def test_zonal_column_must_be_split_like_the_flow_model(county, tmp_path):
    g, flags, path, cmeta = county
    meta = {"param_mode": "zonal", "zone_boundaries": ZB, **cmeta}
    zoc = column_zone_map(g, meta)
    three = tmp_path / "vep3.json"
    three.write_text(json.dumps({"zonal": [_col(100.0)] * 3, "config": "zonal"}))
    with pytest.raises(ValueError, match="county"):
        load_or_fit_vep(str(three), None, None, "cpu", zone_of_cell=zoc)
    six_p = {"zonal": [_col(100.0)] * 3 + [_col(100.0, skv=0.1)] * 3, "config": "zonal",
             "zone_county_split": True, "zone_names": list(zone_labels(county=True)),
             "zone_county_sha256": cmeta["zone_county_sha256"]}
    six = tmp_path / "vep6.json"
    six.write_text(json.dumps(six_p))
    with pytest.raises(ValueError, match="county"):
        load_or_fit_vep(str(six), None, None, "cpu",
                        zone_of_cell=column_zone_map(g, {"zone_boundaries": ZB}))
    with pytest.raises(ValueError, match="county map"):
        load_or_fit_vep(str(six), None, None, "cpu", zone_of_cell=zoc,
                        county_sha256="f" * 64)
    col, _ = load_or_fit_vep(str(six), None, None, "cpu", zone_of_cell=zoc,
                             county_sha256=cmeta["zone_county_sha256"])
    with torch.no_grad():
        out = col(torch.linspace(0.0, -3.0, 12, dtype=torch.float32).repeat(g.n_active, 1))
    # same band, same heads: a Changhua cell compacts with its own (softer) column
    mid = fan_band_of(zoc) == 1
    a = out[mid & flags][0, -1]
    b = out[mid & ~flags][0, -1]
    assert abs(float(a)) > abs(float(b)) > 0.0
    # blended over the mid/distal line, split by county
    six_b = tmp_path / "vep6b.json"
    six_b.write_text(json.dumps({**six_p, "zone_blend_km": 2.0}))
    colb, _ = load_or_fit_vep(str(six_b), None, None, "cpu", zone_of_cell=zoc,
                              zone_weights=column_zone_weights(g, meta))
    assert colb.zone_w.shape == (6, g.n_active)
    with pytest.raises(ValueError, match="parameter sets"):
        ZonalColumn([_col(100.0)] * 3, zoc)


def test_calibrate_coupled_zonal_column_has_a_set_per_county_zone(county):
    g, flags, path, cmeta = county
    meta = {"param_mode": "zonal", "zone_boundaries": ZB, **cmeta}
    zoc = column_zone_map(g, meta)
    n = len(column_zone_names(meta))
    model = cc._make("zonal", "cpu", n)
    assert len(model) == 6 and len(cc._make("zonal", "cpu")) == 3
    T = 18
    heads = torch.linspace(0.0, -4.0, T).repeat(g.n_active, 4, 1)
    obs = torch.where(torch.tensor(flags)[:, None], 0.02, 0.1) * torch.linspace(0, 1, T)
    zt = torch.tensor(zoc)
    cc._fit(model, heads, obs.float(), torch.ones(g.n_active, T), zt, 30, 0.3)
    pred = cc._predict(model, heads, zt)
    assert pred.shape == (g.n_active, T)
    js = cc.column_json(model)
    assert len(js["zonal"]) == 6
    # Changhua leveling rows draw on their own columns: the fitted sets differ
    assert js["zonal"][1] != js["zonal"][4]


def test_scenario_zones_select_a_band_including_its_changhua_copy(county):
    g, flags, path, cmeta = county
    zoc = column_zone_map(g, {"zone_boundaries": ZB, **cmeta})
    z3 = fan_zones(g.centroids(), 205.0, 201.0)
    dates = pd.date_range("2030-01-01", periods=3, freq="MS")
    E = {"irrigation": np.ones((g.n_active, 3))}
    s = PumpingScenario("cut", factors={"irrigation": 0.5}, zones=("proximal",))
    assert np.array_equal(s.apply(E, dates, zone_of_cell=zoc),
                          s.apply(E, dates, zone_of_cell=z3))
    s2 = PumpingScenario("sw", sw_factor=0.5, zones=("mid",))
    assert np.array_equal(s2.apply_sw(np.ones((g.n_active, 3)), dates, zoc),
                          s2.apply_sw(np.ones((g.n_active, 3)), dates, z3))


def test_initial_heads_are_not_split_by_county():
    meta = {"ic_merged_proximal": True, "zone_boundaries": "205,182,208"}
    with_county = {**meta, "zone_county_split": True, "zone_county_sha256": "a" * 64}
    assert input_options(with_county) == input_options(meta)


# ---------------------------------------------------------------------------------------
# calibrate_flow: the CLI, the recorded configuration and the parallel k-fold
# ---------------------------------------------------------------------------------------
def test_cli_refuses_a_county_split_it_cannot_honour():
    with pytest.raises(SystemExit, match="zonal"):
        main(["--zone-county-split"])
    with pytest.raises(SystemExit, match="river-c-split"):
        main(["--param-mode", "zonal", "--zone-county-split", "--river-c-split", "zone"])
    with pytest.raises(SystemExit, match="policy-response"):
        main(["--param-mode", "zonal", "--zone-county-split", "--policy-response"])
    with pytest.raises(SystemExit, match="zone-county-map"):
        main(["--zone-county-map", "x.npz"])


def _args(**kw):
    from types import SimpleNamespace

    a = dict(param_mode="zonal", log_t_min_proximal=None, zone_blend_km=0.0,
             no_forcing=False, boundaries="none", meter_filter="dedupe-cap", fix_eta=None,
             fix_head_extra=None, pump_split=False, return_flow=False, l_min=None,
             holdout_months=0, loss="level", delay_storage="off", rivers="none",
             river_set="choushui", sw_recharge=None, delay_u0="eq", aquitard_storage="off",
             sw_components="delivered", river_c_split="group", no_backfill=False, epochs=2,
             seed=0, compile_matvec=False, dx=2000.0, ic_merged_proximal=False,
             ground_elev="wells", strict_coverage=False, dem_npz="x.npz",
             ic_layered_proximal=False, proximal_layered=False, lr=0.05,
             zone_county_split=False, zone_county_map=None, county_tie=None,
             county_copy_sd=None, fold_site_tol_m=0.0)
    a.update(kw)
    return SimpleNamespace(**a)


def test_recorded_configuration_is_unchanged_by_default(county):
    g, flags, path, cmeta = county
    cfg = _flow_cfg(_args(), g.n_active, 205.0, 201.0, None, {}, None, "abc")
    assert not any(k in cfg["recipe"] for k in COUNTY_ARGS)
    assert "fold_site_tol_m" not in cfg["recipe"]          # opt-in, absent at its default
    assert not any(k.startswith("zone_county") for k in cfg)
    assert _flow_cfg(_args(fold_site_tol_m=200.0), g.n_active, 205.0, 201.0, None, {}, None,
                     "abc")["recipe"]["fold_site_tol_m"] == 200.0
    on = _flow_cfg(_args(zone_county_split=True, zone_county_map=path), g.n_active, 205.0,
                   201.0, None, {}, None, "abc", county_rec=cmeta)
    assert on["recipe"]["zone_county_split"] is True
    assert on["zone_county_sha256"] == cmeta["zone_county_sha256"]


def _seed_run(d, g, fit, cfg, cmeta):
    d.mkdir(parents=True, exist_ok=True)
    _write_theta(str(d / "stage3_theta.json"), fit["theta"],
                 {"param_mode": "zonal", "zone_boundaries": ZB, "boundaries": "none",
                  **cmeta})
    pd.DataFrame({"sid": SIDS}).to_csv(d / "stage3_wells.csv", index=False)
    write_fit_summary(str(d), cfg, _flow_fit_cols(fit, None, None), (0, 0.0), 1.0, SIDS)


SIDS = [f"w{i}" for i in range(8)]


@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_parallel_kfold_records_and_merges_the_county_split(county, tmp_path):
    g, flags, path, cmeta = county
    zone, names = flow_zone_map(g, {"zone_boundaries": ZB, **cmeta})
    h0, rch, rf, idx, lay, obs = _problem(g)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    fit = fit_flow(m, obs, idx, lay, rch, recharge_field=rf, epochs=2, lr=0.05,
                   param_mode="zonal", h0=h0, zone_of_cell=zone, zonation=names)
    well_xy = g.centroids()[idx.numpy()]
    kw = dict(n_layers=4, epochs=2, lr=0.05, n_folds=2, seed=0, param_mode="zonal",
              well_xy=well_xy, obs_h0=obs[:, 0].numpy(), recharge_field=rf,
              zone_of_cell=zone, zonation=names, sids=SIDS)
    args = _args(zone_county_split=True, zone_county_map=path)
    cfg = json.loads(json.dumps(_flow_cfg(args, g.n_active, 205.0, 201.0, None, {}, None,
                                          "abc", county_rec=cmeta)))
    run = tmp_path / "run"
    _seed_run(run, g, fit, cfg, cmeta)
    for k in (0, 1):
        gk = kfold_wells(g, obs, idx, lay, rch, only_fold=k, **kw)
        assert any(t.endswith(COUNTY_SUFFIX) for t in gk["per_fold"][0]["theta"])
        write_fold_file(str(run), gk, cfg, (0, 0.0), 1.0, SIDS)
    merged = merge(str(run), log=lambda *_: None)
    assert merged["n_folds"] == 2
    row = pd.read_csv(run / "stage3_flow.csv").iloc[0]
    assert bool(row["zone_county_split"]) and row["zone_county_sha256"] == \
        cmeta["zone_county_sha256"]
    # fold members inherit the split and rebuild on the same zones
    members = load_members([str(run / "stage3_theta.json"),
                            str(run / "stage3_fold_thetas.json")])
    for mem in members:
        _, _, zoc = build_model(g, mem, "cpu")
        assert np.array_equal(zoc, zone)
    # a gate CSV member carries the split too
    (csv_mem,) = load_members([str(run / "stage3_flow.csv")])
    assert csv_mem.meta["zone_county_split"] is True
    assert csv_mem.meta["zone_county_sha256"] == cmeta["zone_county_sha256"]
    # a fold run without the split is not the same configuration
    plain = _flow_cfg(_args(), g.n_active, 205.0, 201.0, None, {}, None, "abc")
    write_fold_file(str(run), gk, plain, (0, 0.0), 1.0, SIDS)
    with pytest.raises(ValueError, match="zone_county"):
        merge(str(run), log=lambda *_: None)


def test_real_county_map_when_the_data_is_present():
    """The rule on the real grid (skipped where the repository's data is not present): the
    product passed its independent-label gates (pump census, WRA station counties)."""
    import os

    from hydrophysics.twin.zone_county import COUNTY_GPKG, DEFAULT_MAP, load_county_npz

    if not (os.path.exists(COUNTY_GPKG) and os.path.exists(DEFAULT_MAP)):
        pytest.skip("county polygons / zone county map not present")
    z = load_county_npz(DEFAULT_MAP)
    assert 0 < int(z["changhua"].sum()) < int(z["n_active"])
    with open(os.path.join(os.path.dirname(DEFAULT_MAP), "validation.json"),
              encoding="utf-8") as fh:
        val = json.load(fh)
    assert val["passed"] and val["independent_labels"]["passed"]
    assert val["sha256_flags"] == str(z["sha256"])


# ---------------------------------------------------------------------------------------
# review round 2026-10-17: county rule, sparsity options, fold diagnostics
# ---------------------------------------------------------------------------------------
def _write_gpkg(path, polys: dict, epsg: int = 3828):
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    gdf = gpd.GeoDataFrame({"NL_NAME_2": list(polys)},
                           geometry=[shapely.box(*b) for b in polys.values()], crs=epsg)
    gdf.to_file(path, driver="GPKG")


def test_county_rule_reads_tm2_coordinates_and_takes_the_nearest_county(tmp_path):
    """The gpkg's coordinates are TM2 whatever its (wrong) EPSG:3828 tag says: a point
    300 m inside the Changhua polygon stays Changhua (a TWD67 -> TWD97 reprojection would
    move the line ~850 m), and a point in no polygon takes the nearest county."""
    from hydrophysics.twin.zone_county import changhua_flags, county_of_points

    gp = str(tmp_path / "counties.gpkg")
    # Changhua north of y = 2_640_000, Yunlin south of it; land ends at x = 180_000
    _write_gpkg(gp, {"彰化縣": (180_000, 2_640_000, 220_000, 2_680_000),
                     "雲林縣": (180_000, 2_600_000, 220_000, 2_640_000)})
    xy = np.array([[200_000.0, 2_640_300.0],     # 300 m inside Changhua
                   [200_000.0, 2_639_700.0],     # 300 m inside Yunlin
                   [179_500.0, 2_650_000.0],     # sea, nearest Changhua
                   [179_500.0, 2_620_000.0],     # sea, nearest Yunlin
                   [200_000.0, 2_640_000.0]])    # on the line: inside counts
    assert changhua_flags(xy, gp).tolist() == [True, False, True, False, True]
    county, d = county_of_points(xy[2:4], gp)
    assert county.tolist() == ["彰化縣", "雲林縣"] and np.allclose(d, 500.0)


def _tie_fit(g, epochs, **kw):
    zc = county_split(fan_zones(g.centroids(), 205.0, 201.0), _changhua(g), 3)
    return _fit(g, zc, zonation=zone_labels(False, True), epochs=epochs, **kw)


def test_tied_copies_follow_their_parents_and_rebuild_alike(county):
    g, flags, path, cmeta = county
    z = fan_zones(g.centroids(), 205.0, 201.0)
    _, f0 = _fit(g, z, epochs=1)
    _, f1 = _tie_fit(g, 1, county_tie=("proximal", "distal"))
    assert f1["loss"] == f0["loss"]                       # epoch 0 is still the unsplit model
    m, f = _tie_fit(g, 6, county_tie=("proximal", "distal"))
    th = f["theta"]
    for k in ("log_T_proximal", "log_S_proximal", "log_T_distal", "log_S_distal",
              "log_L_distal"):
        assert th[f"{k}{COUNTY_SUFFIX}"] == th[k]         # tied: exactly the parent
    assert not np.allclose(th["log_T_mid"], th[f"log_T_mid{COUNTY_SUFFIX}"])
    _, free = _tie_fit(g, 1)
    tied_n = sum(np.size(th[k]) for k in th if k.endswith(COUNTY_SUFFIX)
                 and ("_proximal_" in k or "_distal_" in k))
    assert f["n_params"] == free["n_params"] - tied_n
    assert f["county_tied"] and all(k.endswith(COUNTY_SUFFIX) for k in f["county_tied"])
    # the theta keeps its key order and rebuilds the same model; a perturbed parent moves
    # its tied copy with it (what uncertainty's Jacobian relies on)
    assert list(th)[: len(free["theta"])] == list(free["theta"])[: len(th)]
    meta = {"param_mode": "zonal", "zone_boundaries": ZB, **cmeta,
            "zone_county_tied": ["proximal", "distal"]}
    rebuilt, _, _ = build_model(g, Member("m", th, meta), "cpu")
    assert torch.allclose(rebuilt.log_T, m.log_T)
    bumped = {**th, "log_T_distal": [v + 0.5 for v in th["log_T_distal"]]}
    rb2, _, _ = build_model(g, Member("m", bumped, meta), "cpu")
    distal_ch = (county_split(z, flags, 3) == 5)
    assert distal_ch.any()
    assert torch.allclose(rb2.log_T[:, distal_ch], rebuilt.log_T[:, distal_ch] + 0.5)
    with pytest.raises(ValueError, match="base zones"):
        _tie_fit(g, 1, county_tie=("mid_changhua",))
    with pytest.raises(ValueError, match="county-split"):
        _fit(g, z, epochs=1, county_tie=("distal",))


def test_copy_prior_is_zero_at_start_and_shrinks_the_copies():
    g = _grid()
    z = fan_zones(g.centroids(), 205.0, 201.0)
    _, f0 = _fit(g, z, epochs=1)
    _, f1 = _tie_fit(g, 1, county_copy_sd=0.5)
    assert f1["loss"] == f0["loss"]
    _, free = _tie_fit(g, 8, lr=0.1)
    _, tight = _tie_fit(g, 8, lr=0.1, county_copy_sd=1e-4)

    def departure(th):
        return max(float(np.max(np.abs(np.asarray(th[k]) - np.asarray(th[k[:-9]]))))
                   for k in th if k.endswith(COUNTY_SUFFIX) and k[:-9] in th)

    assert departure(tight["theta"]) < departure(free["theta"])
    dep = tight["county_copy_departure"]
    assert dep and all(k.endswith(COUNTY_SUFFIX) and k in tight["theta"] for k in dep)
    assert f"log_T_mid{COUNTY_SUFFIX}" in dep
    with pytest.raises(ValueError, match="county_copy_sd"):
        _tie_fit(g, 1, county_copy_sd=0.0)


def test_init_scatter_gives_each_copy_its_parents_draw():
    g = _grid()
    z = fan_zones(g.centroids(), 205.0, 201.0)
    _, f0 = _fit(g, z, epochs=1, init_scatter=0.1, seed=3)
    _, f1 = _tie_fit(g, 1, init_scatter=0.1, seed=3)
    assert f1["loss"] == f0["loss"]                       # the scattered unsplit model


def test_expand_zonal_names_a_missing_key():
    g = _grid()
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    th = _make_zonal_params(m, county=True)
    del th[f"log_T_mid{COUNTY_SUFFIX}"]
    zc = county_split(fan_zones(g.centroids(), 205.0, 201.0), _changhua(g), 3)
    with pytest.raises(KeyError, match="log_T_mid_changhua"):
        _expand_zonal(th, torch.tensor(zc), 4)


def test_site_labels_with_a_tolerance_group_a_nest():
    from hydrophysics.twin.calibrate_flow import _fold_groups, _site_labels_tol

    xy = np.array([[0.0, 0.0], [17.0, 0.0], [17.0, 17.0], [300.0, 0.0], [300.0, 0.0]])
    assert len(np.unique(_fold_groups(xy))) == 4               # exact co-location
    lab = _site_labels_tol(xy, 100.0)
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] and lab[0] != lab[3]
    assert np.array_equal(_site_labels_tol(xy, 0.0), _fold_groups(xy))


def test_zone_support_reports_counts_thin_zones_and_shared_sites():
    from hydrophysics.twin.calibrate_flow import format_zone_support, zone_support

    zone = np.array([0, 1, 2, 3, 4, 5])
    idx = np.array([0, 1, 1, 4, 4, 4, 5])
    lay = np.array([0, 1, 2, 0, 1, 1, 3])
    xy = np.array([[0, 0], [1000, 0], [3000, 0], [5000, 0], [5010, 0], [5020, 0],
                   [9000, 0]], dtype=float)
    folds = [np.array([3]), np.array([0, 1, 2, 4, 5, 6])]
    sup = zone_support(zone, idx, lay, xy, zone_labels(False, True), folds)
    zm = sup["zones"]["mid_changhua"]
    assert zm["wells"] == 3 and zm["sites"] == 1 and zm["wells_per_layer"] == [1, 2, 0, 0]
    assert sup["zones"]["proximal_changhua"]["wells"] == 0
    # fold 0 holds out one screen of the mid_changhua nest: its siblings train the zone
    assert sup["per_fold"][0]["held_sharing_site_with_training_same_zone"] == {
        "mid_changhua": 1}
    # fold 1 leaves every zone but mid_changhua without a training well
    assert "proximal" in sup["per_fold"][1]["zones_without_training_well"]
    # a zone with no well at all is the thinnest of all
    assert "mid_changhua" in sup["thin"] and "proximal_changhua" in sup["thin"]
    lines = format_zone_support(sup)
    assert any("WARNING thin zones" in ln for ln in lines)


def test_posterior_tools_honour_ties_and_the_copy_prior():
    from hydrophysics.twin.uncertainty import copy_prior_precision, laplace
    from hydrophysics.twin.zone_county import (
        apply_county_ties,
        free_copy_pairs,
        tied_copy_keys,
    )

    th = {"log_T_mid": [1.0, 2.0], "log_T_mid_changhua": [1.5, 2.5],
          "log_T_distal": [3.0], "log_T_distal_changhua": [3.3], "log_eta": 0.1}
    meta = {"zone_county_split": True, "zone_county_tied": ["distal"]}
    assert tied_copy_keys(th, meta) == {"log_T_distal_changhua": "log_T_distal"}
    tied = apply_county_ties(th, meta)
    assert tied["log_T_distal_changhua"] == [3.0] and tied["log_T_mid_changhua"] == [1.5, 2.5]
    assert apply_county_ties(th, {}) is th
    assert free_copy_pairs(th, meta) == [("log_T_mid_changhua", "log_T_mid")]
    index = [("log_T_mid", 0), ("log_T_mid", 1), ("log_T_mid_changhua", 0),
             ("log_T_mid_changhua", 1)]
    P = copy_prior_precision(index, free_copy_pairs(th, meta), 0.5)
    assert P[0, 0] == 4.0 and P[0, 2] == -4.0 and P[1, 3] == -4.0 and P[0, 1] == 0.0
    J = np.zeros((10, 4))
    J[:, 0] = 1.0                                  # only the parent is informed
    cov0, _ = laplace(J, np.ones(10), prior_sd=2.0)
    cov1, _ = laplace(J, np.ones(10), prior_sd=2.0, extra_precision=P)
    assert cov1[2, 2] < cov0[2, 2]                 # the copy borrows its parent's information


def test_unsplit_column_relabelled_by_county_for_the_report():
    names = ("proximal", "mid", "distal")
    zone = np.array([0, 1, 1, 2, 2, 1])
    changhua = np.array([False, True, False, True, False, True])
    ids, split = cc.county_split_ids(zone, changhua, names)
    assert split == names + ("proximal_changhua", "mid_changhua", "distal_changhua")
    assert ids.tolist() == [0, 4, 1, 5, 2, 4]
    # blend weights relabel by their argmax
    w = np.eye(3)[zone]
    assert cc.county_split_ids(w, changhua, names)[0].tolist() == ids.tolist()
    # the report then scores the counties exactly as a split column's report does
    obs = np.arange(12, dtype=float).reshape(6, 2)
    rep = cc.oof_by_zone(obs + 0.1, obs, np.ones_like(obs, dtype=bool), ids, split)
    assert rep["county:changhua"]["benchmarks"] == 3
    assert rep["county:rest"]["benchmarks"] == 3
    assert rep["mid_changhua"]["benchmarks"] == 2
    with pytest.raises(ValueError):
        cc.county_split_ids(zone, changhua[:-1], names)
