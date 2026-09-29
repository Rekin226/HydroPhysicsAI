"""Regression kriging at k-fold held-out wells (twin/residual_kriging.py): numpy only."""

from __future__ import annotations

import numpy as np
import pytest

from hydrophysics.twin import residual_kriging as rk


def _field(rng, n_wells=40, T=60):
    xy = rng.uniform(0, 40000, size=(n_wells, 2))
    t = np.arange(T)
    # a smooth regional signal plus a well-local one
    regional = np.sin(2 * np.pi * t / 12)[None] * (1 + xy[:, :1] / 40000)
    local = rng.normal(0, 0.3, size=(n_wells, T))
    return xy, t, regional, local


def test_residual_anomalies_cancel_a_per_well_constant():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(5, 30))
    model = obs + np.arange(5)[:, None] * 7.0            # a datum per well
    r = rk.residual_anomalies(model, obs)
    assert np.allclose(r, 0.0)
    obs[0, :4] = np.nan
    r = rk.residual_anomalies(model, obs)
    assert np.isnan(r[0, :4]).all() and np.allclose(r[0, 4:], 0.0)


def test_idw_correction_is_exact_at_a_source_and_layer_fallback():
    xy = np.array([[0.0, 0.0], [1000.0, 0.0], [5000.0, 0.0]])
    r = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]])
    out = rk.idw_correction(xy[:1], xy, r)
    assert np.allclose(out, r[:1], atol=1e-6)
    lay = np.array([0, 1, 1])
    tgt = np.array([[500.0, 0.0], [500.0, 0.0]])
    per_layer = rk.idw_correction(tgt, xy, r, src_layer=lay, tgt_layer=np.array([0, 2]))
    assert np.allclose(per_layer[0], r[0])                   # only the layer-0 source
    assert np.allclose(per_layer[1], rk.idw_correction(tgt[1:], xy, r)[0])   # fallback


def test_exponential_fit_recovers_a_known_variogram():
    import pandas as pd

    h = np.linspace(0, 30000, 25)
    g = 0.5 + 2.0 * (1 - np.exp(-h / 8000.0))
    vp = rk.fit_exponential(pd.DataFrame({"lag_m": h, "gamma": g, "n_pairs": 10}))
    assert vp["nugget"] == pytest.approx(0.5, abs=0.05)
    assert vp["psill"] == pytest.approx(2.0, rel=0.05)
    assert vp["range_m"] == pytest.approx(8000.0, rel=0.1)


def test_simple_kriging_honours_data_and_missing_months():
    xy = np.array([[0.0, 0.0], [3000.0, 0.0], [0.0, 4000.0]])
    r = np.array([[1.0, np.nan, 2.0], [0.5, 1.0, np.nan], [-1.0, 2.0, 0.0]])
    vp = {"nugget": 0.0, "psill": 1.0, "range_m": 5000.0}
    out = rk.simple_kriging(xy, xy, r, vp)
    fin = np.isfinite(r)
    assert np.allclose(out[fin], r[fin], atol=1e-6)
    assert np.isfinite(out).all()                            # a missing source is predicted


def test_simple_kriging_colocated_sources_share_psill_not_the_nugget():
    # two distinct wells in one cell (nested screens): the nugget sits on the diagonal
    # only, so the pair is not perfectly correlated and the target gets their mean
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [20000.0, 0.0]])
    C = rk._exp_cov(np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)),
                    {"nugget": 1.0, "psill": 2.0, "range_m": 10000.0})
    assert C[0, 1] == pytest.approx(2.0) and C[0, 0] == pytest.approx(3.0)
    assert np.linalg.cond(C) < 1e3
    r = np.array([[1.0], [3.0], [0.0]])
    out = rk.simple_kriging(np.array([[0.0, 0.0]]), xy, r,
                            {"nugget": 1.0, "psill": 2.0, "range_m": 10000.0})
    lam = np.linalg.solve(C, np.array([2.0, 2.0, 2.0 * np.exp(-2.0)]))
    assert out[0, 0] == pytest.approx(lam @ r[:, 0])
    assert lam[0] == pytest.approx(lam[1])


def test_error_decomposition_is_exact():
    rng = np.random.default_rng(1)
    obs = np.cumsum(rng.normal(size=(6, 50)), 1)
    pred = 0.5 * obs + rng.normal(size=(6, 50)) + np.linspace(0, 3, 50)
    mask = np.ones_like(obs, bool)
    mask[2, :10] = False
    d = rk.error_decomposition(pred, obs, mask)
    parts = d["mse_trend"] + d["mse_amplitude"] + d["mse_phase"]
    assert np.allclose(parts, d["mse_anom"])


def _synthetic_fm(rng, model_skill: float):
    xy, t, regional, local = _field(rng)
    W = xy.shape[0]
    obs = regional + local + rng.normal(0, 5, size=(W, 1))        # levels differ
    model = regional + model_skill * local + 3.0                   # a level bias
    folds = np.arange(W) % 4
    pe = {"fold": folds[np.argsort(folds, kind="stable")],
          "entry": np.argsort(folds, kind="stable"), "layer": np.zeros(W, "int64")}
    e = pe["entry"]
    pe["obs"] = obs[e]
    pe["pred"] = model[e]
    pe["idw"] = np.concatenate([
        rk.idw_interp(xy[e[pe["fold"] == f]], xy[folds != f], obs[folds != f])
        for f in range(4)])
    fm = {"model": np.stack([model] * 4), "obs": obs, "cell_xy": xy,
          "layer": np.zeros(W, "int64"), "fold_of": folds, "per_entry": pe}
    return fm


def test_evaluate_a_perfect_model_beats_idw_and_no_leakage():
    rng = np.random.default_rng(2)
    fm = _synthetic_fm(rng, model_skill=1.0)
    res = rk.evaluate(fm, kriging=True)
    s = res["scores"]
    assert s["model"]["r2_anom_kfold"] > 0.99
    assert s["rk_idw"]["r2_anom_kfold"] > 0.99            # zero residuals: model kept
    assert s["idw_anom"]["r2_anom_kfold"] == pytest.approx(s["idw_heads"]["r2_anom_kfold"])
    # a held-out well's own observation never enters its prediction
    fm2 = {**fm, "obs": fm["obs"].copy()}
    pe = fm["per_entry"]
    held0 = pe["entry"][pe["fold"] == 0]
    fm2["obs"][held0] += rng.normal(0, 10, size=fm2["obs"][held0].shape)
    res2 = rk.evaluate(fm2, kriging=True)
    rows = pe["fold"] == 0
    for k in ("idw_anom", "rk_idw", "rk_idw_layer", "rk_sk", "rk_beta"):
        assert np.allclose(res["preds"][k][rows], res2["preds"][k][rows]), k


def test_inner_beta_tracks_how_much_the_model_knows():
    rng = np.random.default_rng(3)
    xy, t, regional, local = _field(rng, n_wells=60)
    obs = regional + local
    assert rk.inner_beta(xy, obs + 2.0, obs) == pytest.approx(1.0, abs=1e-6)
    noise = regional + rng.normal(0, 0.3, size=obs.shape)
    assert rk.inner_beta(xy, noise, obs) < 0.3


def test_head_field_correction_reproduces_observed_anomalies_at_wells():
    rng = np.random.default_rng(4)
    cells = rng.uniform(0, 20000, size=(30, 2))
    idx = np.array([1, 5, 9, 20])
    lay = np.array([0, 0, 1, 1])
    T = 24
    heads = rng.normal(size=(3, 30, T))
    obs = rng.normal(size=(4, T)) + 50.0
    for method in rk.CORRECTION_METHODS:
        out, corr, info = rk.correct_head_field(heads, obs, lay, idx, cells, method=method)
        assert out.shape == heads.shape and np.isfinite(corr).all()
        if method == "sk":
            continue                                           # smooths when nugget > 0
        a_new = rk.row_anomaly(out[lay, idx])
        if method == "idw_layer":
            assert np.allclose(a_new, rk.row_anomaly(obs), atol=1e-5)
        # layer 2 (index 2) has no well: idw_layer falls back to every well
        assert np.allclose(corr[2], rk.idw_interp(cells, cells[idx],
                                                  rk.residual_anomalies(heads[lay, idx], obs)))
    # a model that already matches every observed anomaly gets no correction
    heads2 = heads.copy()
    heads2[lay, idx] = obs - 50.0
    _, corr, _ = rk.correct_head_field(heads2, obs, lay, idx, cells, method="idw_layer")
    assert np.allclose(corr, 0.0, atol=1e-9)


def test_bootstrap_margins_point_estimate_matches_scores():
    rng = np.random.default_rng(5)
    fm = _synthetic_fm(rng, model_skill=0.5)
    res = rk.evaluate(fm, kriging=False)
    b = rk.bootstrap_margins(res["preds"], fm["per_entry"], fm["cell_xy"][
        fm["per_entry"]["entry"]], n_boot=50)
    for _, row in b.iterrows():
        assert row["margin"] == pytest.approx(res["scores"][row["method"]][
            "margin_anom_kfold"], abs=1e-9)
