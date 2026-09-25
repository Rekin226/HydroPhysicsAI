"""CPU tests for the fair held-out-years verdict, its rescoring CLI and --no-backfill."""
from __future__ import annotations

import numpy as np
import pytest

from hydrophysics.twin import drift_diag
from hydrophysics.twin.drift_diag import fair_baselines, fair_temporal_verdict

T, T_FIT = 131, 95


def _record(W=12, seed=0, trend=0.0, noise=0.3):
    """Seasonal heads with per-well levels, noise (so no baseline is exact) and an
    optional linear trend."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    lvl = rng.uniform(-20, 40, size=(W, 1))
    amp = rng.uniform(0.5, 3.0, size=(W, 1))
    return (lvl + amp * np.sin(2 * np.pi * (t + 1) / 12) + trend * t / 12.0
            + noise * rng.standard_normal((W, T)))


def test_rule_is_preregistered():
    """The constants are the pre-registered ones; changing them is a new gate."""
    assert drift_diag.FAIR_RMSE_K == 1.25
    assert drift_diag.FAIR_SHAPE_TOL == 0.05
    assert drift_diag.FAIR_MIN_FIT_MONTHS == 24
    assert drift_diag.FAIR_BASELINES == ("clim", "clim_trend", "persist")


def test_pure_datum_offset_passes_and_level_error_is_reported():
    obs = _record()
    off = np.linspace(-8, 8, obs.shape[0])[:, None]
    pred = obs + off + 0.01 * np.random.default_rng(1).standard_normal(obs.shape)
    r = fair_temporal_verdict(pred, obs, T_FIT)
    assert r["verdict_fair"] == "PASS"
    assert r["rmse_datum_model_m"] < 0.05
    assert r["rmse_raw_model_m"] > 4.0
    assert r["level_err_mean_abs_m"] == pytest.approx(np.abs(off).mean(), abs=0.01)
    assert r["datum_share_of_mse"] > 0.99


def test_datum_uses_fitted_months_only_so_drift_is_charged():
    obs = _record()
    pred = obs.copy()
    pred[:, T_FIT:] += 5.0                       # a level jump after the fit
    r = fair_temporal_verdict(pred, obs, T_FIT)
    assert r["level_err_mean_abs_m"] == pytest.approx(0.0, abs=1e-12)
    assert r["rmse_datum_model_m"] == pytest.approx(5.0)
    assert r["bias_datum_model_m"] == pytest.approx(5.0)
    assert r["verdict_fair"] == "FAIL"


def test_never_observed_months_are_not_scored_and_short_wells_are_excluded():
    obs = _record(W=6)
    obs[0, :T_FIT - 10] = np.nan                 # late starter: 10 fitted months
    obs[1, T_FIT + 3:T_FIT + 6] = np.nan         # held-out gap
    pred = obs.copy()
    pred[1, T_FIT + 3:T_FIT + 6] = 1e6           # would dominate if the gap were scored
    pred[0] += 100.0                             # excluded well: never scored
    r = fair_temporal_verdict(np.nan_to_num(pred, nan=1e6), obs, T_FIT)
    assert r["n_wells_scored"] == 5
    assert r["n_wells_excluded_short_fit"] == 1
    assert r["n_cells"] == 5 * (T - T_FIT) - 3
    assert r["rmse_datum_model_m"] == pytest.approx(0.0, abs=1e-9)


def test_clim_trend_baseline_recovers_a_trend_plus_seasonal_record():
    obs = _record(trend=-0.8, noise=0.0)
    b = fair_baselines(obs, T_FIT)
    assert np.allclose(b["clim_trend"], obs, atol=1e-8)
    assert not np.allclose(b["clim"][:, T_FIT:], obs[:, T_FIT:], atol=0.5)
    assert np.allclose(b["persist"][:, T_FIT:], obs[:, T_FIT - 1:T_FIT])
    r = fair_temporal_verdict(b["clim"], obs, T_FIT)
    assert r["best_baseline"] == "clim_trend"
    assert r["verdict_fair"] == "FAIL"           # climatology alone cannot follow a trend


def test_flat_model_fails_on_shape():
    obs = _record()
    pred = np.repeat(obs[:, :T_FIT].mean(axis=1, keepdims=True), T, axis=1)
    r = fair_temporal_verdict(pred, obs, T_FIT)
    assert r["r2_shape_datum_model"] < r["r2_shape_clim"] - 0.05
    assert r["verdict_fair"] == "FAIL"


# --- rescoring CLI -------------------------------------------------------------------

def _dump(tmp_path, obs_raw, pred, with_raw, with_sids=True):
    from hydrophysics.twin.calibrate_flow import prepare_series

    filled = np.stack([prepare_series(r) for r in obs_raw])
    d = tmp_path / "run"
    d.mkdir()
    kw = {"pred": pred, "obs": filled, "clim": filled, "T_fit": np.array(T_FIT)}
    if with_sids:
        kw["sids"] = np.array([f"w{i}" for i in range(len(obs_raw))])
    if with_raw:
        kw["obs_raw"] = obs_raw
    np.savez_compressed(d / "stage3_temporal_pred.npz", **kw)
    return d


def test_rescore_reloads_raw_by_sid_and_matches_the_direct_score(tmp_path):
    from hydrophysics.twin.rescore_temporal import rescore

    obs = _record(W=8)
    obs[2, :80] = np.nan
    pred = obs + 3.0
    d = _dump(tmp_path, obs, pred, with_raw=False)
    raw_all = np.concatenate([np.full((8, 1), 1.0), obs], axis=1)[::-1]   # month 0 + shuffled
    sids = [f"w{i}" for i in range(8)][::-1]
    df = rescore([str(d)], raw_loader=lambda: (raw_all, sids))
    assert len(df) == 1 and df.raw_obs[0] == "reloaded"
    direct = fair_temporal_verdict(pred, obs, T_FIT)
    assert df.rmse_datum_model_m[0] == pytest.approx(direct["rmse_datum_model_m"])
    assert df.n_wells_scored[0] == 7


def test_rescore_content_match_and_mismatch_raises(tmp_path):
    from hydrophysics.twin.rescore_temporal import match_raw

    obs = _record(W=5)
    obs[1, :30] = np.nan
    filled = np.stack([np.nan_to_num(r, nan=r[30]) for r in obs])
    raw_all = np.concatenate([np.zeros((5, 1)), obs], axis=1)
    rows = match_raw(filled, raw_all[::-1], list("edcba"))
    assert np.array_equal(np.isnan(rows), np.isnan(obs))
    with pytest.raises(ValueError):
        match_raw(filled + 0.5, raw_all, list("abcde"), sids=np.array(list("abcde")))


def test_rescore_uses_obs_raw_when_the_dump_has_it(tmp_path):
    from hydrophysics.twin.rescore_temporal import rescore

    obs = _record(W=4)
    d = _dump(tmp_path, obs, obs + 1.0, with_raw=True)

    def _boom():
        raise AssertionError("must not reload")

    df = rescore([str(d)], raw_loader=_boom)
    assert df.raw_obs[0] == "dump" and df.verdict_fair[0] == "PASS"


# --- --no-backfill -------------------------------------------------------------------

def test_prepare_series_default_backfills_and_opt_out_keeps_nan():
    from hydrophysics.twin.calibrate_flow import prepare_series

    s = np.array([np.nan, np.nan, 1.0, np.nan, 3.0, np.nan])
    assert np.allclose(prepare_series(s), [1, 1, 1, 2, 3, 3])
    out = prepare_series(s, backfill=False)
    assert np.array_equal(np.isnan(out), np.isnan(s)) and out is not s


def test_fit_flow_masks_nan_target_months():
    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyproj")
    from hydrophysics.twin.calibrate_flow import _masked_mse, fit_flow
    from hydrophysics.twin.flow import FlowModel
    from hydrophysics.twin.grid import FanGrid

    g = FanGrid(nx=9, ny=9, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((9, 9), dtype=bool))
    A, steps = g.n_active, 10
    truth = FlowModel(g, n_layers=2, dt_days=30.0)
    rng = torch.Generator().manual_seed(0)
    rech = torch.rand(2, A, steps, generator=rng) * 1e-3
    rech[1] = 0.0
    with torch.no_grad():
        h = truth(torch.zeros(2, A), rech, torch.zeros(2, A, steps), steps)
    obs_idx = torch.arange(0, A, 7)
    obs_layer = torch.zeros_like(obs_idx)
    obs = h[obs_layer, obs_idx, 1:].clone()
    obs_nan = obs.clone()
    obs_nan[0, :4] = float("nan")                # late starter
    obs_nan[1, 5] = float("nan")                 # interior gap
    for mode in ("level", "anomaly"):
        m = FlowModel(g, n_layers=2, dt_days=30.0)
        out = fit_flow(m, obs_nan, obs_idx, obs_layer, rech, epochs=3, lr=0.05,
                       loss_mode=mode)
        assert np.isfinite(out["loss"]) and np.isfinite(out["r2"])
    mask = torch.isfinite(obs_nan)
    z = torch.where(mask, obs_nan, torch.zeros_like(obs_nan))
    p = obs + 1.0
    assert float(_masked_mse(p, z, mask)) == pytest.approx(1.0)
    # an all-finite target takes the unmasked path: same loss as before the change
    m1, m2 = FlowModel(g, n_layers=2, dt_days=30.0), FlowModel(g, n_layers=2, dt_days=30.0)
    a = fit_flow(m1, obs, obs_idx, obs_layer, rech, epochs=2, lr=0.05)
    b = fit_flow(m2, obs.clone(), obs_idx, obs_layer, rech, epochs=2, lr=0.05)
    assert a["loss"] == b["loss"]
