"""Opt-in per-well datum in the observation operator (--well-datum fit, 2026-09-26).

pred_i(t) = h(cell_i, layer_i, t) + d_i with a N(0, sd^2) prior on d_i. The datum is not a
physical parameter: the default must be bit-identical, the datum must never reach theta,
the k-fold held-out wells or a rebuilt model, and the temporal gate must apply the
fitted-period datum unchanged to the held-out months."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    WELL_DATUM_MONTHS_PER_OBS,
    _predict_homogeneous,
    _profile_well_datum,
    fit_flow,
    kfold_wells,
    main,
    temporal_gate,
    well_datum_stats,
    well_datum_vector,
)
from hydrophysics.twin.drift_diag import fair_temporal_verdict  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402

D = torch.float64
# An exact fit (zero residual) hands the adjoint solve a zero right-hand side, whose
# "relative residual" is then undefined: the CG warning is expected there and harmless.
EXACT = pytest.mark.filterwarnings("ignore:_cg did not converge")
T = 24
OFFSETS = np.array([7.0, -3.0, 12.0, 0.5])


def _grid():
    return FanGrid(nx=6, ny=3, dx=1000.0, x0=200_000.0, y0=0.0,
                   mask=np.ones((3, 6), dtype=bool))


def _problem(T=T):
    """Tiny closed 4-layer problem: a sloping initial head relaxing under recharge."""
    g = _grid()
    A = g.n_active
    x = torch.tensor(g.centroids()[:, 0], dtype=D)
    h0 = (10.0 + (x - x.mean()) / 500.0).expand(4, A).clone()
    rch = torch.zeros(4, A, T, dtype=D)            # sets n_steps; the forcing is rf
    rf = torch.full((A, T), 4e-4, dtype=D)          # recharge field (learned fraction)
    well_xy = np.array([[200_500.0, 500.0], [202_500.0, 1500.0], [204_500.0, 2500.0],
                        [205_500.0, 500.0]])
    idx = torch.tensor([g.active_index(*p) for p in well_xy], dtype=torch.long)
    lay = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    return g, h0, rch, rf, idx, lay, well_xy


def _fit(obs, well_datum="off", epochs=1, lr=0.0, **kw):
    g, h0, rch, rf, idx, lay, _ = _problem(obs.shape[1])
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    fit = fit_flow(m, obs, idx, lay, rch, epochs=epochs, lr=lr, h0=h0, recharge_field=rf,
                   well_datum=well_datum, **kw)
    return m, fit


def _truth(T=T):
    """Physical heads at the wells for the initial (lr = 0) parameters."""
    g, h0, rch, rf, idx, lay, _ = _problem(T)
    m, fit = _fit(torch.zeros(4, T, dtype=D))
    h = _predict_homogeneous(m, fit, h0, T, recharge_field=rf)
    return h[lay, idx, 1:].clone()


def test_default_is_bit_identical_to_explicit_off():
    obs = _truth() + torch.tensor(OFFSETS, dtype=D)[:, None]
    _, a = _fit(obs, epochs=4, lr=0.05)
    _, b = _fit(obs, epochs=4, lr=0.05, well_datum="off", well_datum_sd=1.0)
    assert a["loss"] == b["loss"] and a["r2"] == b["r2"]
    assert json.dumps(a["theta"]) == json.dumps(b["theta"])
    assert not any(k.startswith(("well_datum", "r2_with")) for k in a)


@EXACT
def test_datum_recovers_a_constant_offset_per_well():
    truth = _truth()
    obs = truth + torch.tensor(OFFSETS, dtype=D)[:, None]
    _, fit = _fit(obs, well_datum="fit")
    # the heads fit exactly apart from the offsets: no residual scatter, no shrinkage
    assert fit["well_datum"] == pytest.approx(OFFSETS, abs=1e-8)
    assert fit["well_datum_kappa"] == pytest.approx(0.0, abs=1e-10)
    assert fit["r2_with_datum"] == pytest.approx(1.0, abs=1e-10)
    assert fit["r2"] < fit["r2_with_datum"] - 1e-3   # the physical heads alone miss them
    assert fit["loss"] == pytest.approx(0.0, abs=1e-12)
    # the datum is not a physical parameter: nothing of it reaches theta or bounds_hit
    assert not any("datum" in k for k in fit["theta"])
    assert not any("datum" in k for k in fit["bounds_hit"])
    # the same MAP datum under the anomaly loss
    _, an = _fit(obs, well_datum="fit", loss_mode="anomaly")
    assert an["well_datum"] == pytest.approx(OFFSETS, abs=1e-8)


def test_datum_fit_moves_the_physics_differently_but_stays_out_of_theta():
    obs = _truth() + torch.tensor(OFFSETS, dtype=D)[:, None]
    _, off = _fit(obs, epochs=5, lr=0.05)
    _, on = _fit(obs, well_datum="fit", epochs=5, lr=0.05)
    assert set(on["theta"]) == set(off["theta"])
    assert on["theta"] != off["theta"]              # the level misfit no longer drives it
    assert np.all(np.isfinite(on["well_datum"]))


def test_prior_shrinks_the_datum_of_wells_with_few_observations():
    rng = np.random.default_rng(0)
    truth = _truth()
    obs = (truth + 10.0 + torch.tensor(rng.normal(0, 1.0, truth.shape), dtype=D))
    obs[1, 2:] = float("nan")                       # well 1: two observed months only
    obs[3, 6:] = float("nan")                       # well 3: six
    _, fit = _fit(obs, well_datum="fit", well_datum_sd=5.0)
    d, n, kappa = fit["well_datum"], fit["well_datum_n"], fit["well_datum_kappa"]
    assert list(n) == [T, 2, T, 6]
    s2 = fit["well_datum_sigma_m"] ** 2
    m = WELL_DATUM_MONTHS_PER_OBS
    assert kappa == pytest.approx(m * s2 / 25.0)
    # d_i = n_i mean_i(obs - pred) / (n_i + kappa_i), kappa_i = min(n_i, m) s^2 / sd^2
    r = (obs - truth).numpy()
    rbar = np.nanmean(r, axis=1)
    k_i = np.minimum(n, m) * s2 / 25.0
    assert d == pytest.approx(n * rbar / (n + k_i), rel=1e-10)
    shrink = d / rbar
    # 2 and 6 correlated months are one look each: the same shrinkage, below the long wells
    assert shrink[1] == pytest.approx(shrink[3], rel=1e-12)
    assert shrink[1] == pytest.approx(1.0 / (1.0 + s2 / 25.0), rel=1e-10)
    assert shrink[1] < shrink[0]
    # a tighter prior bites harder, and hardest on the short well
    _, tight = _fit(obs, well_datum="fit", well_datum_sd=0.5)
    st = tight["well_datum"] / rbar
    assert st[1] < st[0] and np.all(st < shrink)
    stats = well_datum_stats(tight["well_datum"], tight["well_datum_n"],
                             tight["well_datum_kappa"])
    s2t = tight["well_datum_sigma_m"] ** 2
    assert stats["shrink_min"] == pytest.approx(1.0 / (1.0 + s2t / 0.25))


def test_prior_scaling_three_months_versus_a_long_record():
    """The review case: within-well sd ~3 m, prior sd 5 m. A well with 130 observed months
    carries ~11 independent looks and keeps ~96 % of its mean residual; one with 3 months
    is ONE look (not a quarter of one) and keeps 1 / (1 + s^2/sd^2), about 74 %."""
    W, Tm = 2, 130
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1.0, (W, Tm))
    noise -= noise.mean(axis=1, keepdims=True)
    obs = torch.tensor(10.0 + 3.0 * noise / noise.std(axis=1, keepdims=True), dtype=D)
    mask = torch.ones(W, Tm, dtype=torch.bool)
    mask[1, 3:] = False
    obs[1, :3] = torch.tensor([7.0, 10.0, 13.0], dtype=D)     # mean 10
    d, kappa, s2, n = _profile_well_datum(torch.zeros(W, Tm, dtype=D),
                                          torch.where(mask, obs, torch.zeros_like(obs)),
                                          mask, sd=5.0)
    shrink = (d / 10.0).squeeze(-1).numpy()
    q = float(s2) / 25.0
    assert list(n.squeeze(-1).numpy()) == [130, 3]
    assert shrink[0] == pytest.approx(130 / (130 + 12 * q))
    assert shrink[1] == pytest.approx(1 / (1 + q))
    assert 0.95 < shrink[0] < 0.97 and 0.70 < shrink[1] < 0.76
    stats = well_datum_stats(d.squeeze(-1).numpy(), n.squeeze(-1).numpy(),
                             12 * float(s2) / 25.0)
    assert stats["shrink_min"] == pytest.approx(shrink[1])


@EXACT
@pytest.mark.parametrize("loss_mode", ["level", "anomaly"])
def test_back_filled_months_never_inform_the_datum(loss_mode):
    """main passes the months actually observed as ``datum_mask``: a late-starting well
    back-filled from (held-out) values must not take its datum from those copies."""
    truth = _truth()
    obs = truth + torch.tensor(OFFSETS, dtype=D)[:, None]
    filled = obs.clone()
    filled[0, :20] = filled[0, 20] + 50.0      # well 0: 20 months of back-filled junk
    real = torch.ones(obs.shape, dtype=torch.bool)
    real[0, :20] = False
    real[2, :] = False                          # well 2: never actually observed
    _, fit = _fit(filled, well_datum="fit", datum_mask=real, loss_mode=loss_mode)
    assert list(fit["well_datum_n"]) == [4, T, 0, T]
    assert fit["well_datum"][0] == pytest.approx(OFFSETS[0], abs=1e-8)   # kappa = 0 here
    assert fit["well_datum"][2] == 0.0
    assert fit["well_datum"][[1, 3]] == pytest.approx(OFFSETS[[1, 3]], abs=1e-8)
    assert np.isfinite(fit["loss"])
    with pytest.raises(ValueError, match="datum_mask"):
        _fit(filled, well_datum="fit", datum_mask=real[:, :5])


@EXACT
def test_temporal_gate_applies_the_fitted_period_datum_unchanged():
    T_full, T_fit = 30, 24
    truth = _truth(T_full)
    obs = truth + torch.tensor(OFFSETS, dtype=D)[:, None]
    g, h0, rch, rf, idx, lay, _ = _problem(T_full)

    def _fit_period(o):
        m = FlowModel(g, n_layers=4, dt_days=30.0)
        return m, fit_flow(m, o[:, :T_fit], idx, lay, rch[..., :T_fit], epochs=1, lr=0.0,
                           h0=h0, recharge_field=rf[:, :T_fit], well_datum="fit")

    m, fit = _fit_period(obs)
    # the held-out months never enter the datum: corrupting them changes nothing
    wild = obs.clone()
    wild[:, T_fit:] += 100.0
    _, fit2 = _fit_period(wild)
    assert np.array_equal(fit["well_datum"], fit2["well_datum"])
    assert fit["well_datum"] == pytest.approx(OFFSETS, abs=1e-8)

    def _gate(datum):
        return temporal_gate(m, fit, h0, obs, idx, lay, T_fit, None, rf, None,
                             keep_arrays=True, datum=datum)

    plain, with_d = _gate(None), _gate(fit["well_datum"])
    pa, pd_ = plain["arrays"]["pred"], with_d["arrays"]["pred"]
    # the legacy metrics, the verdict policy_gate ranks on and the dumped pred stay the
    # PHYSICAL heads, comparable with every other run
    assert np.array_equal(pa, pd_)
    for k in ("verdict", "rmse_model_m", "rmse_ratio", "r2_shape_model", "bias_model_m"):
        assert with_d[k] == plain[k]
    assert not any(k.startswith("datum_") for k in plain)
    # the datum_* keys score the datum-shifted prediction: here it removes the whole error
    assert with_d["datum_rmse_model_m"] == pytest.approx(0.0, abs=1e-8)
    assert with_d["datum_bias_model_m"] == pytest.approx(0.0, abs=1e-8)
    assert plain["rmse_model_m"] > 1.0
    pd_ = pa + fit["well_datum"][:, None]
    # the fair verdict removes its own fitted-period datum, so it is invariant to ours
    fa = fair_temporal_verdict(pa, obs.numpy(), T_fit, min_fit_months=12)
    fb = fair_temporal_verdict(pd_, obs.numpy(), T_fit, min_fit_months=12)
    assert fa["rmse_datum_model_m"] == pytest.approx(fb["rmse_datum_model_m"], abs=1e-9)


def test_held_out_months_never_move_the_datum_or_the_physics_while_learning():
    """With a moving fit (lr > 0, several epochs, back-fill-shaped datum_mask), a +100 m
    shift of every held-out month leaves the datum AND theta bit-identical."""
    T_full, T_fit = 30, 24
    rng = np.random.default_rng(2)
    obs = (_truth(T_full) + torch.tensor(OFFSETS, dtype=D)[:, None]
           + torch.tensor(rng.normal(0, 0.5, (4, T_full)), dtype=D))
    g, h0, rch, rf, idx, lay, _ = _problem(T_full)
    real = torch.ones(4, T_fit, dtype=torch.bool)
    real[3, :10] = False

    def _run(o):
        m = FlowModel(g, n_layers=4, dt_days=30.0)
        return fit_flow(m, o[:, :T_fit], idx, lay, rch[..., :T_fit], epochs=4, lr=0.05,
                        h0=h0, recharge_field=rf[:, :T_fit], well_datum="fit",
                        datum_mask=real)

    a, wild = _run(obs), obs.clone()
    wild[:, T_fit:] += 100.0
    b = _run(wild)
    assert np.array_equal(a["well_datum"], b["well_datum"])
    assert json.dumps(a["theta"]) == json.dumps(b["theta"])
    assert np.all(a["well_datum"] != 0.0)


@EXACT
def test_kfold_held_out_wells_are_predicted_without_a_datum(tmp_path):
    truth = _truth()
    obs = truth + torch.tensor(OFFSETS, dtype=D)[:, None]
    g, h0, rch, rf, idx, lay, well_xy = _problem()
    res = {}
    for mode in ("off", "fit"):
        dump = tmp_path / f"{mode}.npz"
        res[mode] = kfold_wells(g, obs, idx, lay, rch, n_layers=4, epochs=1, lr=0.0,
                                n_folds=2, well_xy=well_xy, dump_path=str(dump),
                                recharge_field=rf, well_datum=mode)
        res[mode]["pred"] = np.load(dump)["pred"]
    # lr = 0: the physics is identical, so the held-out predictions are too -- the in-fold
    # datum (which would shift them by the offsets) never reaches a held-out well
    assert np.array_equal(res["off"]["pred"], res["fit"]["pred"])
    assert res["off"]["r2_kfold"] == res["fit"]["r2_kfold"]
    for f in res["fit"]["per_fold"]:
        assert not any("datum" in k for k in f["theta"])


def test_theta_with_well_datum_meta_loads_through_forward_and_policy_gate(tmp_path):
    from test_twin_forward import _inputs, _theta_file

    from hydrophysics.twin.forward import build_model, load_members
    from hydrophysics.twin.policy_gate import policy_response
    from hydrophysics.twin.uncertainty import flatten, jacobian

    inp = _inputs(T=24)
    p, f = _theta_file(tmp_path)
    with open(p) as fh:
        obj = json.load(fh)
    obj["meta"].update({"well_datum": {"a": 2.0, "c": -1.5}, "well_datum_mode": "fit",
                        "well_datum_sd": 5.0,
                        "well_datum_stats": well_datum_stats(np.array([2.0, -1.5]))})
    with open(p, "w") as fh:
        json.dump(obj, fh)
    members = load_members([p, f])
    assert members[1].meta["well_datum"] == {"a": 2.0, "c": -1.5}
    # the rebuilt model is the physical one: identical with and without the datum meta
    plain = load_members([p])[0]
    plain.meta = {k: v for k, v in plain.meta.items() if not k.startswith("well_datum")}
    m1, s1, _ = build_model(inp.grid, members[0], "cpu")
    m2, s2, _ = build_model(inp.grid, plain, "cpu")
    assert torch.equal(m1.log_T, m2.log_T) and torch.equal(m1.log_S, m2.log_S)
    assert well_datum_vector(members[0].meta, inp.sids) == pytest.approx([2.0, 0.0, -1.5])
    assert well_datum_vector(plain.meta, inp.sids) is None
    vep = tmp_path / "vep.json"
    vep.write_text(json.dumps({"log_ske": float(np.log(1e-4)), "log_skv": float(np.log(1e-3)),
                               "log_tau": float(np.log(300.0)), "h_pc0": 0.0}))
    out = policy_response(inp, members[0], str(vep), horizon=6, device="cpu",
                          log=lambda *_: None)
    assert out["verdict"] in ("PASS", "FAIL")
    # uncertainty: the datum is a held nuisance -- never a perturbed parameter, but it
    # shifts the residual by exactly d_i per well
    _, index = flatten(members[0].theta, fixed=())
    assert not any("datum" in k for k, _ in index)
    one = [index[0]]
    J1, r1 = jacobian(inp, members[0], one, "cpu", log=lambda *_: None)
    J2, r2 = jacobian(inp, plain, one, "cpu", log=lambda *_: None)
    W = len(inp.sids)
    # the datum is profiled out of the Jacobian: each well's rows centred over time
    J2w = J2.reshape(W, -1, J2.shape[1])
    centred = (J2w - J2w.mean(axis=1, keepdims=True)).reshape(J2.shape)
    assert np.allclose(J1, centred, rtol=0.0, atol=1e-12)
    assert (r1 - r2).reshape(W, -1) == pytest.approx(
        np.repeat(np.array([[2.0], [0.0], [-1.5]]), 23, axis=1))


def test_cli_refuses_the_datum_with_percell_and_a_non_positive_sd():
    with pytest.raises(SystemExit, match="homogeneous or zonal"):
        main(["--param-mode", "percell", "--well-datum", "fit"])
    with pytest.raises(SystemExit, match="must be > 0"):
        main(["--well-datum", "fit", "--well-datum-sd", "0"])
    with pytest.raises(ValueError, match="homogeneous/zonal"):
        _fit(torch.zeros(4, T, dtype=D), well_datum="fit", param_mode="percell")
