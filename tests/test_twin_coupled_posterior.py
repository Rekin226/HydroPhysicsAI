"""Stage-4 column recalibration internals on synthetic rings (the real run needs the MLCW
cache) and the Laplace posterior on a toy grid."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.calibrate_coupled import (  # noqa: E402
    _fit,
    _make,
    _predict,
    _WeightedColumn,
    column_json,
    loso,
)
from hydrophysics.twin.uncertainty import (  # noqa: E402
    clip_to_bounds,
    draw_posterior,
    effective_bounds,
    find_at_bound,
    flatten,
    jacobian,
    laplace,
    newton_shift,
    sample_truncated_mvn,
    unflatten,
)


def _rings(n=6, L=4, T=36, seed=0):
    rng = np.random.default_rng(seed)
    heads = 5.0 - 0.05 * np.arange(T)[None, None, :] + rng.normal(0, 0.2, (n, L, T))
    heads[:, 1] -= 1.0                                   # layer 2 lower, drives more
    obs = 0.004 * np.cumsum(np.clip(5.0 - heads.mean(1), 0, None), axis=1)
    mask = np.ones((n, T))
    mask[:, ::7] = 0.0
    zone = rng.integers(0, 3, n)
    t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt)  # noqa: E731
    return t(heads), t(obs), t(mask), torch.tensor(zone)


@pytest.mark.parametrize("config", ["shared", "weighted", "zonal"])
def test_each_column_configuration_fits_predicts_and_serialises(config):
    H, OBS, M, Z = _rings()
    model = _make(config, "cpu")
    loss = _fit(model, H, OBS, M, Z, epochs=15, lr=0.05)
    assert np.isfinite(loss)
    with torch.no_grad():
        pred = _predict(model, H, Z)
    assert pred.shape == OBS.shape and torch.isfinite(pred).all()
    js = column_json(model)
    json.dumps(js)
    if config == "weighted":
        assert isinstance(model, _WeightedColumn)
        assert len(js["layer_weights"]) == 4 and abs(sum(js["layer_weights"]) - 1) < 1e-6
    if config == "zonal":
        assert len(js["zonal"]) == 3
    r2 = loso(config, H, OBS, M, Z, epochs=5, lr=0.05, device="cpu")
    assert np.isfinite(r2)


def test_flatten_unflatten_and_clip_round_trip():
    theta = {"log_T_mid": [1.0, 2.0], "log_eta": 0.5, "eta": 1.65, "log_C_apex": [30.0]}
    vec, index = flatten(theta, fixed=("log_eta",))
    assert index == [("log_T_mid", 0), ("log_T_mid", 1), ("log_C_apex", 0)]
    back = unflatten(vec + 1.0, index, theta)
    assert back["log_T_mid"] == [2.0, 3.0] and back["log_eta"] == 0.5
    clipped = clip_to_bounds(vec, index)
    assert clipped[2] <= np.log(1e5)                     # apex conductance ceiling


def test_laplace_posterior_on_a_toy_model(tmp_path):
    from hydrophysics.twin.forward import load_members
    from tests.test_twin_forward import _inputs, _theta_file

    inp = _inputs()
    p, _ = _theta_file(tmp_path)
    member = load_members([p])[0]
    vec, index = flatten(member.theta)
    sub = [j for j, (k, _) in enumerate(index) if k in ("log_T_mid", "log_eta")][:3]
    index = [index[j] for j in sub]
    J, resid = jacobian(inp, member, index, "cpu", log=lambda *_: None)
    n_obs = inp.obs_h.shape[0] * (len(inp.dates) - 1)
    assert J.shape == (n_obs, 3) and resid.shape == (n_obs,)
    assert np.isfinite(J).all()
    cov, sigma2 = laplace(J, resid, prior_sd=2.0)
    assert cov.shape == (3, 3) and sigma2 > 0
    assert (np.diag(cov) > 0).all() and (np.sqrt(np.diag(cov)) <= 2.0 + 1e-9).all()


def test_leveling_target_kfold_and_rezero():
    from hydrophysics.twin.calibrate_coupled import _rezero, kfold_sites

    H, OBS, M, Z = _rings(n=8)
    p = torch.arange(1.0, 9.0)[:, None] + torch.arange(36.0)[None, :] * 0.1
    r = _rezero(p, M)
    first = int(torch.argmax((M[0] > 0).to(torch.int64)))
    assert torch.allclose(r[:, first], torch.zeros(8))
    r2 = kfold_sites("shared", H, OBS, M, Z, epochs=5, lr=0.05, device="cpu", n_folds=4)
    assert np.isfinite(r2)


# --------------------------------------------------------------------------------------
# bounded posterior sampling (G5, 2026-09-23)
# --------------------------------------------------------------------------------------
def test_unflatten_refreshes_derived_keys_the_model_reads():
    """build_model reads the stress radius from the derived spread_km: a sampled
    log_spread_km must reach it (before 2026-09-23 it never did)."""
    theta = {"log_spread_km": float(np.log(10.0)), "spread_km": 10.0,
             "return_frac_logit": 0.0, "return_frac": 0.35, "log_C_apex": [0.0],
             "C_apex_m2day": 1.0}
    vec, index = flatten(theta)
    assert [k for k, _ in index] == ["log_spread_km", "return_frac_logit", "log_C_apex"]
    back = unflatten(np.array([np.log(4.0), 50.0, np.log(7.0)]), index, theta)
    assert back["spread_km"] == pytest.approx(4.0)
    assert back["return_frac"] == pytest.approx(0.7)             # RETURN_FRAC_MAX
    assert back["C_apex_m2day"] == pytest.approx(7.0)


def test_effective_bounds_apply_the_calibration_floor_and_overrides():
    index = [("log_L_mid", 0), ("log_spread_km", 0), ("recharge_frac_logit", 0)]
    lo, hi = effective_bounds(index, {"l_min": 1e-4},
                              {"log_spread_km": (np.log(0.5), np.log(10.0))})
    assert lo[0] == pytest.approx(np.log(1e-4))
    assert hi[1] == pytest.approx(np.log(10.0))
    assert np.isinf(lo[2]) and np.isinf(hi[2])
    vec = np.array([np.log(1e-4), np.log(10.0), 0.3])
    assert find_at_bound(vec, lo, hi) == [0, 1]


@pytest.mark.parametrize("max_draws", [1_000_000, 0])          # rejection, then Gibbs
def test_truncated_normal_at_a_bound_is_a_half_normal(max_draws):
    rng = np.random.default_rng(0)
    n = 4000 if max_draws else 600
    x, method = sample_truncated_mvn(np.array([0.0]), np.array([[1.0]]), np.array([0.0]),
                                     np.array([np.inf]), n, rng, max_draws=max_draws,
                                     burn=20, thin=2)
    assert method == ("rejection" if max_draws else "gibbs")
    assert (x >= 0.0).all()
    assert x.mean() == pytest.approx(np.sqrt(2 / np.pi), abs=0.08)   # half-normal mean
    assert x.std() == pytest.approx(np.sqrt(1 - 2 / np.pi), abs=0.08)


def test_gibbs_matches_rejection_on_a_correlated_box():
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    lo, hi = np.array([0.0, -np.inf]), np.array([np.inf, 0.5])
    xr, mr = sample_truncated_mvn(mean, cov, lo, hi, 4000, np.random.default_rng(1))
    xg, mg = sample_truncated_mvn(mean, cov, lo, hi, 800, np.random.default_rng(2),
                                  max_draws=0, burn=50, thin=3)
    assert (mr, mg) == ("rejection", "gibbs")
    for x in (xr, xg):
        assert (x[:, 0] >= 0).all() and (x[:, 1] <= 0.5).all()
    assert np.allclose(xr.mean(0), xg.mean(0), atol=0.1)


def _toy_linear_posterior(seed=0):
    """y = J x + noise with x[0] constrained to <= 0 while the data want +1: the
    constrained fit sits on the bound, the free coordinate is interior."""
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(200, 2))
    x_true = np.array([1.0, -0.5])
    y = J @ x_true + rng.normal(0, 0.5, 200)
    lo, hi = np.array([-np.inf, -np.inf]), np.array([0.0, np.inf])
    # constrained least squares: x0 = 0, x1 solves the reduced problem
    x1 = float(np.linalg.lstsq(J[:, 1:], y, rcond=None)[0][0])
    x_hat = np.array([0.0, x1])
    resid = J @ x_hat - y
    cov, sigma2 = laplace(J, resid, prior_sd=1e3)
    return J, y, resid, cov, sigma2, x_hat, lo, hi


def test_newton_shift_points_to_the_unconstrained_optimum():
    J, y, resid, cov, sigma2, x_hat, _, _ = _toy_linear_posterior()
    x_ls = np.linalg.lstsq(J, y, rcond=None)[0]
    assert np.allclose(x_hat + newton_shift(J, resid, cov, sigma2), x_ls, atol=1e-3)


def test_draw_posterior_hold_keeps_default_and_truncnorm_samples_the_bound():
    J, y, resid, cov, sigma2, x_hat, lo, hi = _toy_linear_posterior()
    at_bound = find_at_bound(x_hat, lo, hi)
    assert at_bound == [0]
    held, m0 = draw_posterior(x_hat, cov, lo, hi, 50, np.random.default_rng(0), mode="hold")
    assert m0 == "hold" and (held[:, 0] == 0.0).all() and held[:, 1].std() > 0
    tn, m1 = draw_posterior(x_hat, cov, lo, hi, 400, np.random.default_rng(0),
                            mode="truncnorm")
    assert m1 in ("rejection", "gibbs")
    assert (tn[:, 0] <= 0.0).all() and tn[:, 0].std() > 0
    sd0 = np.sqrt(cov[0, 0])
    # a half-normal off the bound with the Laplace width (the joint truncation of a
    # correlated pair is still one-sided on x0; its mean offset is ~ sd*sqrt(2/pi))
    assert -tn[:, 0].mean() == pytest.approx(sd0 * np.sqrt(2 / np.pi), rel=0.25)
    # centred on the Gauss-Newton optimum past the bound the draws crowd the bound
    shift = np.zeros(2)
    shift[0] = newton_shift(J, resid, cov, sigma2)[0]
    tn2, _ = draw_posterior(x_hat, cov, lo, hi, 400, np.random.default_rng(0),
                            mode="truncnorm", shift=shift)
    assert (tn2[:, 0] <= 0.0).all() and -tn2[:, 0].mean() < -tn[:, 0].mean()
    # an uninformed coordinate listed in ``hold`` never moves
    tn3, _ = draw_posterior(x_hat, cov, lo, hi, 20, np.random.default_rng(0),
                            mode="truncnorm", hold=[1])
    assert (tn3[:, 1] == x_hat[1]).all()
    with pytest.raises(ValueError, match="unknown"):
        draw_posterior(x_hat, cov, lo, hi, 2, np.random.default_rng(0), mode="logit")


def test_cli_redraws_from_a_saved_cov_without_the_model(tmp_path):
    from hydrophysics.twin.uncertainty import main

    theta = {"log_S_mid": [float(np.log(0.3)), -5.0], "log_L_mid": [float(np.log(1e-4))],
             "log_spread_km": float(np.log(10.0)), "spread_km": 10.0,
             "recharge_frac_logit": -1.0, "recharge_frac": 0.27}
    meta = {"param_mode": "zonal", "l_min": 1e-4, "fix_eta": 0.5}
    p = tmp_path / "stage3_theta.json"
    p.write_text(json.dumps({"theta": theta, "meta": meta}))
    vec, index = flatten(theta)
    sd = np.array([0.1, 0.5, 0.1, 2.0, 0.2])          # spread: sd = prior -> uninformed
    cov = np.diag(sd ** 2)
    cov[0, 1] = cov[1, 0] = 0.02
    c = tmp_path / "saved_cov_in.json"
    c.write_text(json.dumps({"index": index, "mean": vec.tolist(), "sd": sd.tolist(),
                             "sigma2": 4.0, "at_bound": [], "cov": cov.tolist(),
                             "prior_sd": 2.0}))
    out = tmp_path / "post.json"
    main(["--theta", str(p), "--from-cov", str(c), "--bounded", "truncnorm",
          "--bound", "log_spread_km=-0.693147,2.302586", "--n-samples", "8",
          "--out", str(out)])
    samples = json.loads(out.read_text())
    assert len(samples) == 8 and all(s["posterior"] == "truncnorm" for s in samples)
    s_mid0 = [s["theta"]["log_S_mid"][0] for s in samples]
    l_mid = [s["theta"]["log_L_mid"][0] for s in samples]
    assert max(s_mid0) <= np.log(0.3) + 1e-12 and np.std(s_mid0) > 0
    assert min(l_mid) >= np.log(1e-4) - 1e-12 and np.std(l_mid) > 0      # l_min floor
    assert all(s["theta"]["spread_km"] == pytest.approx(10.0) for s in samples)  # held
    written = json.loads((tmp_path / "post_cov.json").read_text())
    assert written["bounded"] == "truncnorm" and written["uninformed"] == [3]
    assert written["from_cov"] == str(c) and written["grad"] is None
    with pytest.raises(SystemExit, match="gradient"):
        main(["--theta", str(p), "--from-cov", str(c), "--bounded", "truncnorm",
              "--bound-mean", "newton", "--out", str(tmp_path / "x.json")])


def test_spread_radius_reaches_the_model_in_the_jacobian(tmp_path):
    """Regression: the Jacobian column of log_spread_km was identically zero because
    build_model reads the derived spread_km."""
    from hydrophysics.twin.forward import load_members
    from tests.test_twin_forward import _inputs, _theta_file

    inp = _inputs()
    p, _ = _theta_file(tmp_path)
    from pathlib import Path

    obj = json.loads(Path(p).read_text())
    # a spatially varying pumping field, so smoothing it changes the heads
    inp.E_by_class["irrigation"][:3] *= 20.0
    obj["theta"]["log_spread_km"] = float(np.log(2.0))
    obj["theta"]["spread_km"] = 2.0
    Path(p).write_text(json.dumps(obj))
    member = load_members([p])[0]
    _, index = flatten(member.theta)
    index = [x for x in index if x[0] == "log_spread_km"]
    J, _ = jacobian(inp, member, index, "cpu", eps=1e-2, log=lambda *_: None)
    assert np.abs(J).max() > 0


# --------------------------------------------------------------------------------------
# creep time-constant ceiling (G4)
# --------------------------------------------------------------------------------------
def test_tau_ceiling_is_reported(monkeypatch):
    import hydrophysics.twin.calibrate_coupled as cc
    from hydrophysics.twin.compaction import VEPColumn

    monkeypatch.setattr(cc, "TAU_MAX_YEARS", None)
    assert cc.tau_ceiling_days(132) == pytest.approx(3960.0)
    monkeypatch.setattr(cc, "TAU_MAX_YEARS", 30.0)
    assert cc.tau_ceiling_days(132) == pytest.approx(30 * 365.25)
    col = VEPColumn(n_sites=1, dt_days=30.0)
    with torch.no_grad():
        col.log_tau.fill_(20.0)                                    # far past any ceiling
    cc._clamp_column(col, 132)
    js = cc.column_json(col)
    assert js["log_tau"] == pytest.approx(np.log(30 * 365.25), rel=1e-5)
    assert cc.tau_at_ceiling(js, 132) == [True]
    zonal = {"zonal": [js, {**js, "log_tau": float(np.log(100.0))}]}
    assert cc.tau_at_ceiling(zonal, 132) == [True, False]
