"""The forward twin with the opt-in physics of 2026-09-23: delay-bed state carried from the
record into the projection, canal-delivery forcing and the ``sw=`` policy lever."""

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
    _members_from_gate_csv,
    attach_sw_recharge,
    build_model,
    delay_state_path,
    future_sw,
    hindcast_with_nudging,
    load_members,
    parse_scenario,
    rollout,
    run,
)
from hydrophysics.twin.inputs import load_sw_recharge  # noqa: E402
from hydrophysics.twin.uncertainty import (  # noqa: E402
    effective_bounds,
    flatten,
    unflatten,
)
from tests.test_twin_forward import _inputs, _theta_file  # noqa: E402


def _extend(p, **theta_extra):
    with open(p) as fh:
        obj = json.load(fh)
    obj["theta"].update(theta_extra)
    with open(p, "w") as fh:
        json.dump(obj, fh)
    return p


def _delay_theta(tmp_path):
    # closed basin: every forward rollout re-prescribes the apex boundary from its own
    # start field, which would otherwise mask the state-carry comparison below
    p, _ = _theta_file(tmp_path, boundaries="none")
    ext = {}
    for z, tau in (("proximal", 60.0), ("mid", 900.0), ("distal", 2000.0)):
        ext[f"log_Sd_{z}"] = [float(np.log(0.05))]
        ext[f"log_tau_{z}"] = [float(np.log(tau))]
    return _extend(p, **ext)


def test_sw_scenario_parses_and_scales_only_where_and_when_in_force():
    s, rain = parse_scenario("dry:sw=0.6,irrigation=1.1@2012-03")
    assert s.sw_factor == 0.6 and rain == 1.0 and s.factors == {"irrigation": 1.1}
    assert "canal water x0.6" in s.describe()
    dates = pd.date_range("2012-01-01", periods=4, freq="MS")
    out = s.apply_sw(np.ones((3, 4)), dates)
    assert np.allclose(out[:, :2], 1.0) and np.allclose(out[:, 2:], 0.6)
    with pytest.raises(ValueError):
        parse_scenario("bad:sw=-1")
    base, _ = parse_scenario("b:irrigation=1")
    assert base.sw_factor == 1.0


def test_delay_state_carries_from_the_record_into_the_projection(tmp_path):
    inp = _inputs(T=24)
    mem = load_members([_delay_theta(tmp_path)])[0]
    model, scalars, _ = build_model(inp.grid, mem, "cpu")
    assert "delay_Sd" in scalars and model.delay_log_Sd is not None
    h0 = inp.initial_heads(0)
    E, R = inp.E_total[:, 1:], inp.recharge_field[:, 1:]
    h_hist, u_end = hindcast_with_nudging(model, scalars, inp, h0, E, R, 0.0, 0,
                                          return_state=True)
    assert u_end is not None and u_end.shape == h0.shape
    # the store lags the aquifer, so it is NOT at the record's final head
    assert float((u_end - h_hist[..., -1]).abs().max()) > 1e-6
    # continuing from the carried state reproduces the uninterrupted run; restarting the
    # store at equilibrium (u0 = h) does not
    E2, R2 = E[:, :12], R[:, :12]
    full = rollout(model, scalars, h0, E2, R2, inp.ground_elev)
    first, u6 = rollout(model, scalars, h0, E2[:, :6], R2[:, :6], inp.ground_elev,
                        return_state=True)
    carried = rollout(model, scalars, first[..., -1], E2[:, 6:], R2[:, 6:], inp.ground_elev,
                      u0=u6)
    restarted = rollout(model, scalars, first[..., -1], E2[:, 6:], R2[:, 6:], inp.ground_elev)
    assert torch.allclose(carried[..., -1], full[..., -1], atol=1e-6)
    assert not torch.allclose(restarted[..., -1], full[..., -1], atol=1e-6)
    # and the whole ensemble runs with it
    res = run(inp, [mem], [(parse_scenario("b:irrigation=1")[0], 1.0)], horizon=3, gain=0.0,
              ic_members=0, ic_sigma=0.1, seed=0, col=VEPColumn(n_sites=1, dt_days=30.0),
              device="cpu", log=lambda *_: None)
    assert np.isfinite(res["heads_mean"]).all()


def test_vanishing_canal_recharge_on_zero_deliveries_matches_no_canal_term(tmp_path):
    inp = _inputs(T=24)
    A, T = inp.grid.n_active, len(inp.dates)
    p0, _ = _theta_file(tmp_path)
    ref_model, ref_scalars, _ = build_model(inp.grid, load_members([p0])[0], "cpu")
    h0 = inp.initial_heads(0)
    E, R = inp.E_total[:, 1:], inp.recharge_field[:, 1:]
    ref = rollout(ref_model, ref_scalars, h0, E, R, inp.ground_elev)
    d = tmp_path / "sw"
    d.mkdir()
    p1, _ = _theta_file(d)
    mem = load_members([_extend(p1, log_sw_scale=float(np.log(0.01)))])[0]
    model, scalars, _ = build_model(inp.grid, mem, "cpu")
    with pytest.raises(ValueError, match="sw_field"):
        rollout(model, scalars, h0, E, R, inp.ground_elev)
    zero = torch.zeros(A, T - 1, dtype=torch.float64)
    h = rollout(model, scalars, h0, E, R, inp.ground_elev, sw_field=zero)
    assert torch.equal(h, ref)
    wet = rollout(model, scalars, h0, E, R, inp.ground_elev, sw_field=zero + 1e-3)
    assert (wet[0, :, -1] > ref[0, :, -1]).all()
    # the policy lever: canal climatology x factor past the record
    inp.sw_field = torch.full((A, T), 2e-3, dtype=torch.float64)
    cut, _ = parse_scenario("c:sw=0.5")
    assert torch.allclose(future_sw(inp, cut, 4), torch.full((A, 4), 1e-3, dtype=torch.float64))


def test_load_sw_recharge_aligns_dates_and_checks_the_grid(tmp_path):
    inp = _inputs(T=24)
    A = inp.grid.n_active
    dates = pd.date_range("2011-12-01", periods=30, freq="MS")
    sw = np.arange(A * 30, dtype="float64").reshape(A, 30) * 1e-6
    p = tmp_path / "sw.npz"
    np.savez(p, sw_m_per_day=sw, dates=np.array([d.strftime("%Y-%m-01") for d in dates]),
             dx=1000.0, n_active=A)
    out = load_sw_recharge(str(p), inp.grid, inp.dates)
    assert out.shape == (A, 24) and np.allclose(out.numpy(), sw[:, 1:25])
    bad = tmp_path / "bad.npz"
    np.savez(bad, sw_m_per_day=sw, dates=np.array([d.strftime("%Y-%m-01") for d in dates]),
             dx=500.0, n_active=A)
    with pytest.raises(ValueError, match="dx"):
        load_sw_recharge(str(bad), inp.grid, inp.dates)
    short = tmp_path / "short.npz"
    np.savez(short, sw_m_per_day=sw[:, :10],
             dates=np.array([d.strftime("%Y-%m-01") for d in dates[:10]]), dx=1000.0,
             n_active=A)
    with pytest.raises(ValueError, match="missing"):
        load_sw_recharge(str(short), inp.grid, inp.dates)


def _gate_csv(dirpath, theta: dict, **cols):
    row = {"param_mode": "zonal", "boundaries": "none", "zone_proximal_km": 205.0,
           "zone_distal_km": 203.0, "dx": 1000.0, "epochs": 2, "r2_insample": 0.5,
           "r2_kfold": 0.8, "r2_idw": 0.7, "n_folds": 5, "l_min": 1e-4,
           "git_commit": "abc", "theta": str(theta)}
    row.update(cols)
    p = dirpath / "stage3_flow.csv"
    pd.DataFrame([row]).to_csv(p, index=False)
    return str(p)


def test_gate_csv_member_keeps_the_opt_in_physics_or_refuses(tmp_path):
    """Review 2026-09-23 defect 1: a CSV member used to drop rivers/sw/delay silently."""
    theta = {"log_T_mid": [0.0]}
    # historical run: the new columns are absent or at their defaults -> nothing needed
    old = tmp_path / "old"
    old.mkdir()
    meta = _members_from_gate_csv(_gate_csv(old, theta))["meta"]
    assert meta["l_min"] == pytest.approx(1e-4) and "rivers" not in meta
    meta = _members_from_gate_csv(_gate_csv(old, theta, rivers="none", delay_storage="off",
                                            river_set="", sw_recharge=""))["meta"]
    assert meta["rivers"] == "none" and meta["delay_storage"] == "off"
    assert "sw_recharge" not in meta
    # a river run without its stage3_theta.json: refused, not silently river-less
    riv = tmp_path / "riv"
    riv.mkdir()
    csv = _gate_csv(riv, theta, rivers="riv", river_set="choushui", delay_storage="off")
    with pytest.raises(ValueError, match="stage3_theta.json"):
        _members_from_gate_csv(csv)
    # with it: the full recipe comes along
    recipe = {"rivers": "riv", "river_set": "choushui", "river_layer": 0,
              "river_stage_depth": 1.0, "river_rbot_depth": 3.0, "dem_sha1": "f00",
              "delay_storage": "off", "sw_layer": 0}
    (riv / "stage3_theta.json").write_text(json.dumps({"theta": theta, "meta": recipe}))
    meta = _members_from_gate_csv(csv)["meta"]
    for k in ("rivers", "river_set", "river_layer", "river_stage_depth", "dem_sha1",
              "sw_layer"):
        assert meta[k] == recipe[k]
    # a sibling from a different run is caught
    (riv / "stage3_theta.json").write_text(json.dumps(
        {"theta": theta, "meta": dict(recipe, rivers="ghb")}))
    with pytest.raises(ValueError, match="disagrees"):
        _members_from_gate_csv(csv)


def test_delay_state_path_matches_the_solver_state(tmp_path):
    """Review defect 2: a projection started mid-record needs the slow store's state
    there; the replay must equal what the solver itself carries."""
    inp = _inputs(T=12)
    mem = load_members([_delay_theta(tmp_path)])[0]
    model, scalars, _ = build_model(inp.grid, mem, "cpu")
    h0 = inp.initial_heads(0)
    E, R = inp.E_total[:, 1:], inp.recharge_field[:, 1:]
    h, u_end = rollout(model, scalars, h0, E[:, :7], R[:, :7], inp.ground_elev,
                       return_state=True)
    u = delay_state_path(model, scalars, h)
    assert u.shape == h.shape and torch.equal(u[..., 0], h0)
    assert torch.allclose(u[..., -1], u_end, atol=1e-12)
    _, u4 = rollout(model, scalars, h0, E[:, :4], R[:, :4], inp.ground_elev,
                    return_state=True)
    assert torch.allclose(u[..., 4], u4, atol=1e-12)
    # a model without a delay bed has no state
    d = tmp_path / "plain"
    d.mkdir()
    p, _ = _theta_file(d)
    m2, s2, _ = build_model(inp.grid, load_members([p])[0], "cpu")
    assert delay_state_path(m2, s2, h) is None


def test_attach_sw_recharge_loads_what_the_member_was_calibrated_with(tmp_path):
    inp = _inputs(T=24)
    A = inp.grid.n_active
    attach_sw_recharge(inp, {}, log=lambda *_: None)
    assert inp.sw_field is None
    dates = pd.date_range("2012-01-01", periods=24, freq="MS")
    p = tmp_path / "sw.npz"
    np.savez(p, sw_m_per_day=np.full((A, 24), 1e-3),
             dates=np.array([d.strftime("%Y-%m-01") for d in dates]), dx=1000.0, n_active=A)
    attach_sw_recharge(inp, {"sw_recharge": str(p)}, log=lambda *_: None)
    assert inp.sw_field is not None and tuple(inp.sw_field.shape) == (A, 24)


def test_uncertainty_samples_the_opt_in_parameters_not_their_readouts():
    """Review defect 2: the log_* keys are parameters; Sd_*/tau_days_*/C_riv_m2day/sw_scale
    are readouts that must be refreshed from them, not perturbed as if independent."""
    theta = {"log_T_mid": [5.0], "log_Sd_mid": [np.log(1e-3)], "Sd_mid": 1e-3,
             "log_tau_mid": [np.log(365.0)], "tau_days_mid": 365.0,
             "log_C_riv": [np.log(1e3), np.log(2e3)], "C_riv_m2day": [1e3, 2e3],
             "log_sw_scale": float(np.log(0.25)), "sw_scale": 0.25}
    vec, index = flatten(theta)
    keys = {k for k, _ in index}
    assert keys == {"log_T_mid", "log_Sd_mid", "log_tau_mid", "log_C_riv", "log_sw_scale"}
    out = unflatten(vec + 0.1, index, theta)
    assert out["Sd_mid"] == pytest.approx(1e-3 * np.exp(0.1))
    assert out["tau_days_mid"] == pytest.approx(365.0 * np.exp(0.1))
    assert out["C_riv_m2day"] == pytest.approx([1e3 * np.exp(0.1), 2e3 * np.exp(0.1)])
    assert out["sw_scale"] == pytest.approx(0.25 * np.exp(0.1))
    lo, hi = effective_bounds(index, {"delay_tau_max_years": 11})
    j = [k for k, _ in index].index("log_tau_mid")
    assert hi[j] == pytest.approx(np.log(365.25 * 11))
    assert np.isfinite(lo[[k for k, _ in index].index("log_sw_scale")])
