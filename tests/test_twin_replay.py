"""Weather-year replay engine (hydrophysics.twin.replay) on synthetic inputs: sequences,
the replayed forcing, the detrend, the windowed scenario, the column's parts, the capture,
and bit-identity of the climatology path through forward.run and forward.main."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin import forward  # noqa: E402
from hydrophysics.twin import replay as R  # noqa: E402
from hydrophysics.twin.compaction import VEPColumn, vep_compaction  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.inputs import TwinInputs  # noqa: E402
from hydrophysics.twin.scenario import PumpingScenario, climatology  # noqa: E402


def _inputs(T=36, nx=6, ny=3, seed=0):
    """Three whole years 2012-2014; energy with a trend and a dry-year bump in 2013."""
    rng = np.random.default_rng(seed)
    g = FanGrid(nx=nx, ny=ny, dx=1000.0, x0=200_000.0, y0=0.0,
                mask=np.ones((ny, nx), dtype=bool))
    A = g.n_active
    dates = pd.date_range("2012-01-01", periods=T, freq="MS")
    well_xy = np.array([[200_500.0, 500.0], [204_500.0, 1500.0], [205_500.0, 2500.0]])
    obs_idx = np.array([g.active_index(x, y) for x, y in well_xy])
    obs_h = np.tile(np.array([[3.0], [4.0], [6.0]]), (1, T)) + 0.01 * np.arange(T)
    yr = dates.year.to_numpy() - 2012
    season = 1.0 + 0.5 * np.sin(2 * np.pi * (dates.month.to_numpy() - 1) / 12.0)
    trend = 1.0 + 0.2 * yr
    dry = np.where(yr == 1, 1.6, 1.0)
    E = {"irrigation": 100.0 * season * trend * dry * (1 + 0.1 * rng.random((A, T))),
         "aquaculture": np.full((A, T), 10.0) * trend}
    rech = 2e-4 * season / dry * np.ones((A, 1))
    return TwinInputs(grid=g, hf=None, dates=dates, obs_h=obs_h,
                      obs_h_filled=obs_h.copy(), obs_idx=obs_idx,
                      obs_layer=np.array([0, 1, 1]), well_xy=well_xy, sids=["a", "b", "c"],
                      ground_elev=torch.full((A,), 12.0, dtype=torch.float64),
                      E_by_class=E, recharge_field=torch.as_tensor(rech, dtype=torch.float64))


def _theta_file(tmp_path):
    theta = {"log_T_proximal": [np.log(500.0)], "log_S_proximal": [np.log(1e-3)],
             "log_T_mid": [np.log(300.0)] * 4, "log_S_mid": [np.log(1e-3)] * 4,
             "log_T_distal": [np.log(100.0)] * 4, "log_S_distal": [np.log(1e-3)] * 4,
             "log_L_mid": [np.log(1e-4)] * 3, "log_L_distal": [np.log(1e-4)] * 3,
             "log_eta": np.log(0.3), "log_head_extra": np.log(30.0),
             "recharge_frac_logit": 0.0, "log_C_coast": [np.log(50.0)] * 4,
             "log_C_apex": [np.log(50.0)]}
    meta = {"param_mode": "zonal", "boundaries": "coast-apex", "zone_boundaries": "205,203",
            "meter_filter": "none"}
    p = tmp_path / "stage3_theta.json"
    p.write_text(json.dumps({"theta": theta, "meta": meta}))
    return str(p)


def _zonal(zone_of_cell):
    params = [{"log_ske": np.log(2e-3), "log_skv": np.log(3e-2), "log_tau": np.log(60.0),
               "h_pc0": 0.0},
              {"log_ske": np.log(1e-3), "log_skv": np.log(5e-2), "log_tau": np.log(900.0),
               "h_pc0": 0.0},
              {"log_ske": np.log(5e-4), "log_skv": np.log(1e-2), "log_tau": np.log(3000.0),
               "h_pc0": 0.2}]
    return forward.ZonalColumn(params, zone_of_cell)


# ------------------------------------------------------------------------- sequences
def test_years_parse_format_and_cyclic_sequences():
    assert R.parse_years("2015-2017,2012") == (2015, 2016, 2017, 2012)
    assert R.format_years((2015, 2016, 2017, 2012, 2013)) == "2015-2017,2012-2013"
    for bad in ("2015..2017", "15-17", "2017-2015", ""):
        with pytest.raises(ValueError):
            R.parse_years(bad)
    seqs = R.cyclic_sequences(2012, 2022)
    assert len(seqs) == 11 and seqs[0][0] == 2012 and seqs[10][0] == 2022
    assert all(sorted(s) == list(range(2012, 2023)) for s in seqs)
    assert seqs[3] == R.parse_years("2015-2022,2012-2014")
    fut = pd.date_range("2023-01-01", periods=24, freq="MS")
    src = R.source_years(fut, (2015, 2012))
    assert (src[:12] == 2015).all() and (src[12:] == 2012).all()


# ------------------------------------------------------------------------- forcing
def test_replay_from_the_first_month_is_the_record_bit_for_bit():
    inp = _inputs()
    T = len(inp.dates)
    for arr in (*inp.E_by_class.values(), inp.recharge_field.numpy()):
        ext, fut = R.replay_extend(arr, inp.dates, T, (2012, 2013, 2014), start=inp.dates[0])
        assert fut.equals(inp.dates) and np.array_equal(ext, arr)
    gate = R.forcing_identity_gate(inp)
    assert gate["passed"] and all(v["array_equal"] for v in gate["fields"].values())
    assert set(gate["fields"]) == {"energy:irrigation", "energy:aquaculture", "recharge"}
    pl = gate["pipeline"]
    assert pl["passed"] and pl["energy_by_class_equal"] and pl["energy_summed_equal"]
    assert pl["recharge_equal"] and pl["canal_equal"] and pl["dates_equal"]
    assert pl["detrended_by_class_times_factor_equal"]
    gate_sw = R.forcing_identity_gate(replace(inp, sw_field=inp.recharge_field * 3.0))
    assert gate_sw["passed"] and "canal" in gate_sw["fields"]
    assert gate_sw["pipeline"]["canal_equal"]
    assert all(getattr(forward, k) is R._FWD[k] for k in R._PATCHED)   # restored


def test_gate_pipeline_catches_a_broken_forward_path_and_reports_a_trend_refusal(
        monkeypatch):
    inp = _inputs()
    # a projection path that perturbs what the forward receives (here: the recharge's
    # torch conversion) passes the array layer but fails the pipeline layer
    orig = R.ReplayForcing.weather_field

    def off(self, arr, dates, horizon, end):
        ext, fut = orig(self, arr, dates, horizon, end)
        return ext * (1.0 + 1e-9), fut
    monkeypatch.setattr(R.ReplayForcing, "weather_field", off)
    g = R.forcing_identity_gate(inp)
    assert all(v["array_equal"] for v in g["fields"].values())
    assert not g["pipeline"]["recharge_equal"] and not g["passed"]
    monkeypatch.undo()
    # a collapsing class: the refusal is reported (False), not raised
    E = dict(inp.E_by_class)
    E["aquaculture"] = E["aquaculture"] * np.where(inp.dates.year == 2012, 10.0, 0.5)[None, :]
    g2 = R.forcing_identity_gate(replace(inp, E_by_class=E))
    assert not g2["trend_factors_positive"] and "trend factor" in g2["trend_refusal"]
    assert not g2["passed"]
    with pytest.raises(ValueError, match="start override"):
        R.ReplayForcing((2012,), pumping="climatology", start=inp.dates[0]).energy(
            "irrigation", E["irrigation"], inp.dates, 12, None)


def test_projection_months_come_from_the_declared_years_and_refusals():
    inp = _inputs()
    E = inp.E_by_class["irrigation"]
    ext, fut = R.replay_extend(E, inp.dates, 30, (2013, 2012))
    assert fut[0] == pd.Timestamp("2015-01-01")
    np.testing.assert_array_equal(ext[:, :12], E[:, 12:24])          # 2015 <- 2013
    np.testing.assert_array_equal(ext[:, 12:24], E[:, :12])          # 2016 <- 2012
    np.testing.assert_array_equal(ext[:, 24:], E[:, 12:18])          # 2017 <- 2013
    with pytest.raises(ValueError, match="not in the record"):
        R.replay_extend(E, inp.dates, 12, (2011,))
    with pytest.raises(ValueError, match="after the climatology end"):
        R.replay_extend(E, inp.dates, 12, (2014,), end="2013-12")
    bad = E.copy()
    bad[0, 12] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        R.replay_extend(bad, inp.dates, 12, (2013,))


def test_class_trend_factors_remove_a_linear_trend_and_refuse_a_collapse():
    inp = _inputs()
    f = R.class_trend_factors(inp.E_by_class, inp.dates, 2012, 2014)
    for g in f.values():
        assert np.isclose(np.mean(list(g.values())), 1.0)
    # aquaculture is exactly linear in the year: detrended annual totals are all equal
    aq = inp.E_by_class["aquaculture"]
    ext, _ = R.replay_extend(aq, inp.dates, 36, (2012, 2013, 2014), start=inp.dates[0],
                             divisor=f["aquaculture"])
    tots = [ext[:, 12 * k:12 * (k + 1)].sum() for k in range(3)]
    assert np.allclose(tots, tots[0], rtol=1e-12)
    zero = {"x": np.zeros_like(aq)}
    assert R.class_trend_factors(zero, inp.dates, 2012, 2014)["x"] == {2012: 1.0, 2013: 1.0,
                                                                       2014: 1.0}
    collapse = {"x": aq * np.where(inp.dates.year == 2012, 10.0, 0.5)[None, :]}
    with pytest.raises(ValueError, match="trend factor"):
        R.class_trend_factors(collapse, inp.dates, 2012, 2014)
    with pytest.raises(ValueError, match="complete year"):
        R.class_trend_factors(inp.E_by_class, inp.dates, 2012, 2015)
    # the recorded anomaly Y/L: exactly 1 for a linear class; the dry-year bump shows
    an = R.class_annual_anomalies(inp.E_by_class, inp.dates, 2012, 2014)
    assert np.allclose(list(an["aquaculture"].values()), 1.0, rtol=1e-12)
    assert an["irrigation"][2013] > 1.2 > an["irrigation"][2012]
    rf = R.ReplayForcing((2013,), pumping="raw", trend_years=(2012, 2014))
    rf.bind(inp)
    rec = rf.record()
    assert rec["trend_factors"] is None and rec["annual_anomaly_note"] is None
    assert set(rec["annual_anomaly_Y_over_L"]["irrigation"]) == {"2012", "2013", "2014"}
    rf2 = R.ReplayForcing((2013,), pumping="raw")              # trend years not in the record
    rf2.bind(inp)
    assert rf2.record()["annual_anomaly_Y_over_L"] is None
    assert "complete year" in rf2.record()["annual_anomaly_note"]


def test_patched_future_forcing_replays_every_field_from_the_same_year():
    inp = replace(_inputs(), sw_field=_inputs().recharge_field * 2.0)
    scen, _ = forward.parse_scenario("half:irrigation=0.5")
    E0, r0, d0 = forward.future_forcing(inp, scen, 24)
    rf = R.ReplayForcing((2013, 2012), pumping="raw")
    rf.bind(inp)
    with R.installed(rf):
        E, r, d = forward.future_forcing(inp, scen, 24, rain_scale=0.5)
        sw = forward.future_sw(inp, scen, 24)
    assert d.equals(d0)
    irr, aq = inp.E_by_class["irrigation"], inp.E_by_class["aquaculture"]
    want = 0.5 * np.concatenate([irr[:, 12:24], irr[:, :12]], axis=1) \
        + np.concatenate([aq[:, 12:24], aq[:, :12]], axis=1)
    np.testing.assert_allclose(E.numpy(), want, rtol=1e-15)
    rec = inp.recharge_field.numpy()
    np.testing.assert_array_equal(r.numpy(), 0.5 * np.concatenate([rec[:, 12:24],
                                                                   rec[:, :12]], axis=1))
    np.testing.assert_array_equal(sw.numpy(), 2.0 * np.concatenate([rec[:, 12:24],
                                                                     rec[:, :12]], axis=1))
    # pumping-only arm: recharge and canal stay at climatology
    rf2 = R.ReplayForcing((2013, 2012), pumping="raw", fields="energy")
    rf2.bind(inp)
    with R.installed(rf2):
        _, r2, _ = forward.future_forcing(inp, scen, 24)
        sw2 = forward.future_sw(inp, scen, 24)
    assert torch.equal(r2, r0) and torch.equal(sw2, forward.future_sw(inp, scen, 24))
    # detrended: the replayed energy is the record over its class factor
    rf3 = R.ReplayForcing((2013, 2012), pumping="detrended", trend_years=(2012, 2014))
    rf3.bind(inp)
    with R.installed(rf3):
        E3, _, _ = forward.future_forcing(inp, PumpingScenario("b"), 24)
    g = rf3.factors
    want3 = (np.concatenate([irr[:, 12:24] / g["irrigation"][2013],
                             irr[:, :12] / g["irrigation"][2012]], axis=1)
             + np.concatenate([aq[:, 12:24] / g["aquaculture"][2013],
                               aq[:, :12] / g["aquaculture"][2012]], axis=1))
    np.testing.assert_allclose(E3.numpy(), want3, rtol=1e-14)


def test_installed_restores_the_forward_and_refuses_nesting():
    names = {k: getattr(forward, k) for k in R._PATCHED}
    with R.installed(R.ReplayForcing((2012,)), R.Capture(county_flags=None)):
        assert forward.climatology is not climatology
        with pytest.raises(RuntimeError, match="already patched"), R.installed():
            pass
    assert all(getattr(forward, k) is v for k, v in names.items())
    assert forward.climatology is climatology


# ------------------------------------------------------------------------- scenarios
def test_windowed_scenario_lifts_the_cut_and_is_the_parent_without_an_end():
    inp = _inputs()
    dates = pd.date_range("2026-01-01", periods=36, freq="MS")
    E = {k: v[:, :36] for k, v in inp.E_by_class.items()}
    s0, _ = forward.parse_scenario("cut30:irrigation=0.7@2026-06")
    s1, _ = R.parse_scenario("cut30:irrigation=0.7@2026-06")
    assert type(s1) is PumpingScenario and s1 == s0
    w, rain = R.parse_scenario("cut30_lift:irrigation=0.7,rain=0.9@2026-06..2027-05#mid")
    assert isinstance(w, R.WindowedScenario) and w.end == "2027-05" and rain == 0.9
    assert w.describe() == "cut30_lift: irrigation x0.7 in mid from 2026-06 to 2027-05"
    w2, _ = R.parse_scenario("cut30_lift:irrigation=0.7@2026-06..2027-05")
    out = w2.apply(E, dates)
    base = sum(E.values())
    inside = (dates >= "2026-06-01") & (dates < "2027-06-01")
    np.testing.assert_allclose(out[:, inside], (0.7 * E["irrigation"] + E["aquaculture"])
                               [:, inside])
    np.testing.assert_array_equal(out[:, ~inside], base[:, ~inside])
    sw = np.ones((inp.grid.n_active, 36))
    ws = R.WindowedScenario("c", sw_factor=0.5, start="2026-06", end="2027-05")
    np.testing.assert_array_equal(ws.apply_sw(sw, dates)[0], np.where(inside, 0.5, 1.0))
    same = R.WindowedScenario("c", factors={"irrigation": 0.7}, start="2026-06")
    np.testing.assert_array_equal(same.apply(E, dates),
                                  PumpingScenario("c", factors={"irrigation": 0.7},
                                                  start="2026-06").apply(E, dates))
    for bad in ("x:irrigation=0.7@2027-01..2026-01", "x:irrigation=0.7@2026-01..2026"):
        with pytest.raises(ValueError):
            R.parse_scenario(bad)


# ------------------------------------------------------------------------- column parts
def test_vep_parts_total_is_the_column_and_the_parts_mean_what_they_say():
    t = torch.arange(60, dtype=torch.float32)
    h = torch.stack([5.0 - 0.05 * t,                         # falls: new lows every month
                     5.0 + 0.5 * (1.0 - torch.cos(t / 3.0)),  # oscillates, never below its start
                     5.0 - 2.0 * (t > 30).float()])          # one step down at month 31
    p = [torch.log(torch.tensor([1e-3, 2e-3, 1e-3])), torch.log(torch.tensor([2e-2] * 3)),
         torch.log(torch.tensor([90.0, 90.0, 900.0])), torch.tensor([0.0, 0.0, 0.0])]
    parts = R.vep_parts(h, *p)
    assert torch.equal(parts["total"], vep_compaction(h, *p))
    np.testing.assert_allclose(parts["elastic"].numpy(),
                               (torch.exp(p[0])[:, None] * (h[:, :1] - h)).numpy(), rtol=1e-6)
    assert parts["newlow"][0, 1:].all()
    assert not parts["newlow"][1].any() and float(parts["inelastic"][1].abs().max()) == 0.0
    assert int(parts["newlow"][2].sum()) == 1 and bool(parts["newlow"][2, 31])
    assert float(parts["inelastic"][2, 30]) == 0.0 and float(parts["inelastic"][2, -1]) > 0.0


def test_column_parts_match_zonal_blended_and_aquitard_columns_bit_for_bit():
    A, T = 18, 40
    rng = np.random.default_rng(1)
    drv = torch.as_tensor(5.0 - np.cumsum(rng.normal(0.02, 0.3, size=(A, T)), axis=1),
                          dtype=torch.float32)
    zone = np.repeat([0, 1, 2], 6)
    col = _zonal(zone)
    parts = R.column_parts(col, drv)
    assert torch.equal(parts["total"], col(drv))
    assert torch.equal(parts["elastic"] + parts["inelastic"], parts["total"])
    w = np.zeros((3, A))
    w[zone, np.arange(A)] = 0.6
    w[(zone + 1) % 3, np.arange(A)] = 0.4
    blend = forward.ZonalColumn([dict(log_ske=c.log_ske.item(), log_skv=c.log_skv.item(),
                                      log_tau=c.log_tau.item(), h_pc0=c.h_pc0.item())
                                 for c in col.cols], zone, zone_w=w)
    assert torch.equal(R.column_parts(blend, drv)["total"], blend(drv))
    np.testing.assert_array_equal(R.column_zone_ids(blend, A), zone)
    aq = VEPColumn(n_sites=1, creep="aquitard")
    pa = R.column_parts(aq, drv)
    assert torch.equal(pa["total"], aq(drv)) and float(pa["aquitard"].abs().max()) > 0.0


# ------------------------------------------------------------------------- the forward
def _run(inp, tmp_path, rf=None, cap=None, scen_text=("cut:irrigation=0.5@2015-01",)):
    from hydrophysics.twin.zone_county import column_zone_map

    members = forward.load_members([_theta_file(tmp_path)])
    col = _zonal(column_zone_map(inp.grid, members[0].meta))
    if cap is not None:
        cap.label(col, "page")
    with R.installed(rf, cap):
        scen = [(forward.BASELINE, 1.0)] + [forward.parse_scenario(s) for s in scen_text]
        return forward.run(inp, members, scen, horizon=24, gain=1.0, ic_members=1,
                           ic_sigma=0.1, seed=0, col=col, device="cpu",
                           log=lambda *_: None, column_heads="free",
                           update_method="innovation", restart_taper_km=5.0)


def test_climatology_path_is_bit_identical_and_the_capture_adds_up(tmp_path):
    inp = _inputs()
    from hydrophysics.twin.zone_county import column_zone_map

    members = forward.load_members([_theta_file(tmp_path)])
    col = _zonal(column_zone_map(inp.grid, members[0].meta))
    scen = [(forward.BASELINE, 1.0), (forward.parse_scenario("cut:irrigation=0.5@2015-01")[0],
                                      1.0)]
    kw = dict(horizon=24, gain=1.0, ic_members=1, ic_sigma=0.1, seed=0, col=col,
              device="cpu", log=lambda *_: None, column_heads="free")
    plain = forward.run(inp, members, scen, **kw)
    cap = R.Capture(county_flags=lambda g: np.arange(g.n_active) < 6)
    with R.installed(None, cap):
        same = forward.run(inp, members, scen, **kw)
    for k in ("heads_mean", "heads_std", "subs_mean", "subs_std"):
        np.testing.assert_array_equal(same[k], plain[k])
    arr = cap.arrays()
    assert arr["total_check_max_abs_m"] == 0.0 and int(arr["n_runs"]) == 2
    tot = arr["elastic_ye_mean"] + arr["inelastic_ye_mean"]           # (S, R, A, Y)
    ye = cap.ye_idx
    np.testing.assert_allclose(tot[:, 0], plain["subs_mean"][:, :, ye], atol=2e-7)
    assert list(arr["group_names"]) == ["fan", "zone0", "zone1", "zone2", "yunlin_rest",
                                        "changhua"]
    assert arr["run_group_elastic"].shape == (2, 1, 2, 6, len(ye))
    assert arr["newlow_runs"].shape == (2, 1, inp.grid.n_active, 2)
    assert arr["newlow_run_year"].shape == (2, 1, 2, inp.grid.n_active, 2)
    assert (arr["newlow_run_year"].sum(axis=2) <= arr["newlow_runs"]).all()
    np.testing.assert_allclose(arr["run_group_elastic"][:, :, :, 0].mean(axis=2),
                               arr["elastic_ye_mean"].mean(axis=2), rtol=1e-12)


def test_replay_changes_only_the_projection_and_a_dry_year_sets_new_lows(tmp_path):
    inp = _inputs()
    clim_cap, rep_cap = R.Capture(county_flags=None), R.Capture(county_flags=None)
    clim = _run(inp, tmp_path, None, clim_cap)
    rep = _run(inp, tmp_path, R.ReplayForcing((2013,), pumping="raw"), rep_cap)
    T = len(inp.dates)
    np.testing.assert_array_equal(rep["heads_mean"][..., :T], clim["heads_mean"][..., :T])
    assert not np.array_equal(rep["heads_mean"][..., T:], clim["heads_mean"][..., T:])
    # replaying the dry year (most pumping, least recharge) every year draws heads lower
    assert rep["heads_mean"][0, 1, :, -1].mean() < clim["heads_mean"][0, 1, :, -1].mean()
    lows_rep = rep_cap.arrays()["newlow_runs"][0, 0].sum()
    lows_clim = clim_cap.arrays()["newlow_runs"][0, 0].sum()
    assert lows_rep > lows_clim


def test_main_climatology_npz_equals_the_plain_forward_and_writes_the_capture(tmp_path,
                                                                              monkeypatch):
    inp = _inputs()
    monkeypatch.setattr(forward, "load_twin_inputs", lambda *a, **k: inp)
    monkeypatch.setattr(forward, "load_columns", lambda *a, **k: [
        ("page", VEPColumn(n_sites=1, dt_days=30.0), {"log_ske": -9.0, "log_skv": -7.0,
                                                      "log_tau": 5.0})])
    p = _theta_file(tmp_path)
    base = ["--theta", p, "--horizon", "24", "--device", "cpu", "--column-heads", "free",
            "--data", str(tmp_path / "no-data"), "--scenario", "cut:irrigation=0.7@2015-01",
            "--scenario", "lift:irrigation=0.7@2015-01..2015-12"]
    with pytest.raises(ValueError):
        forward.main(base + ["--out", str(tmp_path / "x")])   # the plain parser has no '..'
    plain = base[:-4] + ["--scenario", "cut:irrigation=0.7@2015-01"]
    forward.main(plain + ["--out", str(tmp_path / "plain")])
    assert R.main(["--projection-forcing", "climatology", "--capture", "none", "--"]
                  + plain + ["--out", str(tmp_path / "clim")]) == 0
    with np.load(tmp_path / "plain.npz") as a, np.load(tmp_path / "clim.npz") as b:
        assert set(a.files) == set(b.files)
        for k in a.files:
            assert np.array_equal(a[k], b[k]), k
    assert R.main(["--projection-forcing", "replay:2013,2012", "--replay-pumping", "raw",
                   "--"] + base + ["--out", str(tmp_path / "rep")]) == 0
    with np.load(tmp_path / "rep.fw.npz") as z:
        rec = json.loads(str(z["record"]))
        assert rec["replay"]["years"] == [2013, 2012] and rec["replay"]["pumping"] == "raw"
        assert set(rec["code_sha256"]) == {"replay.py", "forward.py", "scenario.py",
                                           "compaction.py"}
        # the county map's provenance is recorded (here: no cache on this grid/cwd)
        assert set(rec["county"]) >= {"source", "error"}
        assert list(z["scenario_names"]) == ["baseline", "cut", "lift"]
        assert list(z["rheology_labels"]) == ["page"]
    with np.load(tmp_path / "rep.npz") as z:
        assert str(z["scenarios"][2]).endswith("from 2015-01 to 2015-12")
    with pytest.raises(SystemExit):                       # never overwrites a capture
        R.main(["--projection-forcing", "replay:2013", "--"] + base
               + ["--out", str(tmp_path / "rep")])
    assert forward.parse_scenario is R._FWD["parse_scenario"]


def test_county_provenance_is_recorded_not_hidden(monkeypatch):
    inp = _inputs()
    import hydrophysics.twin.zone_county as zc

    def boom(grid, path):
        raise FileNotFoundError("no county polygons")
    monkeypatch.setattr(zc, "county_flags_for_grid", boom)
    flags, prov = R.default_county_flags(inp.grid)
    assert flags is None and prov["error"].startswith("FileNotFoundError")
    monkeypatch.setattr(zc, "county_flags_for_grid",
                        lambda grid, path: (np.arange(grid.n_active) < 4, "rebuilt from x"))
    flags, prov = R.default_county_flags(inp.grid)
    assert flags.sum() == 4 and prov["source"] == "rebuilt from x"
    assert prov["file_sha256"] is None and prov["county_sha256"] == zc.county_sha256(flags)
    cap = R.Capture(county_flags=lambda g: np.arange(g.n_active) < 4)
    cap.bind(inp, [(forward.BASELINE, 1.0)], 12)
    assert cap.county["source"] == "caller-supplied" and cap.changhua.sum() == 4
