"""Opt-in input constructions of 2026-09-23: --ic-merged-proximal, --ground-elev dem and
--strict-coverage. Every default must reproduce the historical construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from hydrophysics.twin.calibrate_flow import (
    _ic_zone_map,
    _idw_field,
    _idw_initial_heads,
    _load_ground_elev,
    _merged_proximal_heads,
)
from hydrophysics.twin.grid import FanGrid
from hydrophysics.twin.inputs import TwinInputs, input_options
from hydrophysics.twin.zones import MID, PROXIMAL, PROXIMAL_W

X0, Y0 = 200000.0, 2600000.0


def _grid():
    # x centroids 200.5 .. 209.5 km: cells >= 205 km are proximal at "205,182"
    return FanGrid(nx=10, ny=3, dx=1000.0, x0=X0, y0=Y0, mask=np.ones((3, 10), dtype=bool))


def _wells(g):
    # two proximal wells at one site with a vertical gradient, one far mid-fan well per
    # deep layer (the audit's situation: proximal L3-L4 filled from mid-fan wells)
    xy = np.array([[X0 + 7500, Y0 + 1500], [X0 + 7500, Y0 + 1500],
                   [X0 + 500, Y0 + 500], [X0 + 500, Y0 + 2500]])
    h = np.array([60.0, 50.0, 5.0, 3.0])
    layer = np.array([0, 1, 2, 3])
    idx = np.array([g.active_index(*p) for p in xy])
    return xy, h, layer, idx


def test_merged_proximal_overwrites_every_proximal_layer_from_proximal_wells_only():
    g = _grid()
    xy, h, layer, idx = _wells(g)
    h0 = _idw_initial_heads(g, xy, h, layer, 4)
    zc = _ic_zone_map(g, "205,182")
    out, rep = _merged_proximal_heads(g, h0, xy, h, zc, zc[idx])
    prox = zc == PROXIMAL
    assert prox.any() and (zc == MID).any()
    # every layer of the proximal cells is the same merged value, from the two proximal
    # wells only (co-located: their plain mean), never from the 3-5 m mid-fan wells
    assert torch.allclose(out[:, prox], out[0:1, prox].expand(4, -1))
    assert torch.allclose(out[0, prox], torch.full((int(prox.sum()),), 55.0,
                                                   dtype=torch.float64))
    # the per-layer field put the mid-fan heads into proximal L3-L4
    assert float(h0[2, prox].max()) < 10.0
    # other zones untouched, input not mutated
    assert torch.equal(out[:, ~prox], h0[:, ~prox])
    assert not torch.equal(out, h0)
    assert rep["proximal"]["n_wells"] == 2 and rep["proximal"]["source"] == "zone"


def test_merged_proximal_split_zone_without_wells_falls_back_to_all_proximal():
    g = _grid()
    xy, h, layer, idx = _wells(g)
    h0 = _idw_initial_heads(g, xy, h, layer, 4)
    zc = _ic_zone_map(g, "205,182,207")          # proximal_w = 205..207 km, no wells
    assert (zc == PROXIMAL_W).any()
    out, rep = _merged_proximal_heads(g, h0, xy, h, zc, zc[idx])
    pw = zc == PROXIMAL_W
    assert rep["proximal_w"]["source"] == "all proximal"
    assert torch.allclose(out[:, pw], torch.full_like(out[:, pw], 55.0))


def test_merged_proximal_without_any_proximal_well_keeps_h0():
    g = _grid()
    xy, h, layer, _ = _wells(g)
    xy, h, layer = xy[2:], h[2:], layer[2:]
    idx = np.array([g.active_index(*p) for p in xy])
    h0 = _idw_initial_heads(g, xy, h, layer, 4)
    zc = _ic_zone_map(g, "205,182")
    out, rep = _merged_proximal_heads(g, h0, xy, h, zc, zc[idx])
    assert torch.equal(out, h0)
    assert rep["proximal"]["n_wells"] == 0


def _inputs(g, zc=None):
    xy, h, layer, idx = _wells(g)
    obs = np.stack([h, h + 1.0], axis=1)
    return TwinInputs(grid=g, hf=None, dates=pd.date_range("2012-01-01", periods=2, freq="MS"),
                      obs_h=obs, obs_h_filled=obs, obs_idx=idx, obs_layer=layer, well_xy=xy,
                      sids=["a", "b", "c", "d"], ground_elev=torch.zeros(g.n_active),
                      E_by_class={}, recharge_field=torch.zeros(g.n_active, 2),
                      ic_zone_of_cell=zc)


def test_twin_inputs_initial_heads_default_unchanged_and_flag_matches_calibration():
    g = _grid()
    xy, h, layer, idx = _wells(g)
    base = _idw_initial_heads(g, xy, h, layer, 4)
    assert torch.equal(_inputs(g).initial_heads(0), base)
    zc = _ic_zone_map(g, "205,182")
    want, _ = _merged_proximal_heads(g, base, xy, h, zc, zc[idx])
    assert torch.equal(_inputs(g, zc).initial_heads(0), want)


def test_input_options_reads_meta_and_is_empty_for_old_runs():
    assert input_options({}) == {}
    assert input_options(None) == {}
    assert input_options({"ic_merged_proximal": False, "ground_elev": "wells",
                          "strict_coverage": False}) == {}
    got = input_options({"ic_merged_proximal": True, "zone_boundaries": "205,182,208",
                         "ground_elev": "dem", "strict_coverage": True})
    assert got == {"ic_merged_proximal": True, "zone_boundaries": "205,182,208",
                   "ic_month0_filled": True,
                   "ground_elev": "dem", "dem_npz": "results/twin/basemap.npz",
                   "strict_coverage": True}
    # a --no-backfill calibration built h0 from the raw month-0 heads
    assert "ic_month0_filled" not in input_options({"ic_merged_proximal": True,
                                                    "no_backfill": True})


def test_merged_initial_heads_use_backfilled_month0_like_the_calibration():
    # review 2026-09-23: calibrate_flow builds h0 (and the apex head) from back-filled
    # month-0 heads of every well; a proximal well first observed later (NaN in obs_h)
    # must still feed the forward path's merged proximal field
    g = _grid()
    xy, h, layer, idx = _wells(g)
    zc = _ic_zone_map(g, "205,182")
    inp = _inputs(g, zc)
    inp.obs_h = inp.obs_h.copy()
    inp.obs_h[0, 0] = np.nan                    # the 60 m proximal well starts in month 1
    base = _idw_initial_heads(g, xy, h, layer, 4)
    want, _ = _merged_proximal_heads(g, base, xy, h, zc, zc[idx])
    raw = inp.initial_heads(0)
    assert not torch.equal(raw, want)           # historical raw-month-0 construction
    inp.ic_month0_filled = True
    assert torch.equal(inp.initial_heads(0), want)
    # later months (restart assimilation) keep the raw observations
    assert torch.equal(inp.initial_heads(1), _inputs(g, zc).initial_heads(1))


def _stations():
    return pd.DataFrame({
        "sid": ["a", "b", "c", "d"],
        "GroundHeight": [0.0, 30.0, np.nan, 12.0],
        "LocationByTWD97": [f"{X0 + 500} {Y0 + 500}", f"{X0 + 8500} {Y0 + 1500}",
                            f"{X0 + 4500} {Y0 + 1500}", f"{X0 + 4500} {Y0 + 2500}"]})


def test_ground_elev_default_keeps_zero_codes_and_dem_mode_reads_the_npz(tmp_path, capsys):
    g = _grid()
    stn = _stations()
    ge = _load_ground_elev(g, stn)
    xy = np.array([[X0 + 500, Y0 + 500], [X0 + 8500, Y0 + 1500], [X0 + 4500, Y0 + 2500]])
    assert np.allclose(ge.numpy(), _idw_field(g, xy, np.array([0.0, 30.0, 12.0])))
    assert "1 of them exactly 0.0" in capsys.readouterr().out
    dem = np.linspace(1.0, 90.0, g.n_active).astype("float32")
    p = tmp_path / "basemap.npz"
    np.savez(p, dem=dem)
    got = _load_ground_elev(g, stn, mode="dem", dem_npz=str(p))
    assert np.allclose(got.numpy(), dem.astype("float64"))
    with pytest.raises(ValueError):
        _load_ground_elev(g, stn, mode="srtm")


def _write_well(d, sid, index):
    pd.DataFrame({"value": np.full(len(index), 10.0)}, index=index).to_parquet(
        d / f"{sid}.parquet")


def test_strict_coverage_rejects_late_starting_10_minute_wells(tmp_path):
    from hydrophysics.twin.heads import build_head_field

    t0, t1 = "2012-01-01", "2014-01-01"
    _write_well(tmp_path, "full", pd.date_range(t0, t1, freq="h", inclusive="left"))
    # starts 10 months late, but 10-minute sampling inflates len/n_hours to ~2.4
    _write_well(tmp_path, "late", pd.date_range("2012-11-01", t1, freq="10min",
                                                inclusive="left"))
    # hourly, starts 3 months late: 21/24 months observed, a 91-day leading gap
    _write_well(tmp_path, "short", pd.date_range("2012-04-01", t1, freq="h",
                                                 inclusive="left"))
    stn = pd.DataFrame({"sid": ["full", "late", "short"],
                        "GroundwaterLayerCode": ["1", "1", "2"],
                        "LocationByTWD97": [f"{X0 + 500} {Y0 + 500}"] * 3})
    legacy = build_head_field(str(tmp_path), stn, t0=t0, t1=t1)
    strict = build_head_field(str(tmp_path), stn, t0=t0, t1=t1, strict_coverage=True)
    assert legacy.sids == ["full", "late", "short"]
    assert strict.sids == ["full", "short"]         # 10/24 months observed for "late"


def test_members_from_gate_csv_carries_the_input_flags(tmp_path):
    from hydrophysics.twin.forward import _members_from_gate_csv

    row = {"param_mode": "zonal", "boundaries": "coast-apex", "zone_proximal_km": 205.0,
           "zone_distal_km": 182.0, "zone_split_km": "", "dx": 1000.0, "epochs": 1,
           "r2_insample": 0.5, "r2_kfold": 0.4, "r2_idw": 0.3, "n_folds": 10,
           "theta": "{'log_T_proximal': [1.0]}", "ic_merged_proximal": True,
           "ground_elev": "dem", "ground_elev_dem_npz": "x.npz", "strict_coverage": False,
           "no_backfill": True}
    p = tmp_path / "stage3_flow.csv"
    pd.DataFrame([row]).to_csv(p, index=False)
    meta = _members_from_gate_csv(str(p))["meta"]
    assert meta["ic_merged_proximal"] is True and meta["ground_elev"] == "dem"
    assert meta["ground_elev_dem_npz"] == "x.npz" and "strict_coverage" not in meta
    assert input_options(meta)["dem_npz"] == "x.npz"
    assert meta["no_backfill"] is True and "ic_month0_filled" not in input_options(meta)
