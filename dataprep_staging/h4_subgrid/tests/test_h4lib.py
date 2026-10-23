"""CPU unit tests of h4lib on synthetic data only (no product is opened)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import h4lib as L  # noqa: E402


def _data(seed=0, W=12, D=60, P=400):
    rng = np.random.default_rng(seed)
    dates = np.array([f"{2012 + (k + 1) // 12}-{(k + 1) % 12 + 1:02d}" for k in range(D)])
    dates = np.array(sorted(dates))
    xy = rng.uniform(0, 8000, (W, 2))
    pxy = rng.uniform(0, 8000, (P, 2))
    moy = np.array([int(s[5:7]) for s in dates])
    q = 20 + 15 * np.sin(2 * math.pi * (moy - 4) / 12)[None, :] * rng.uniform(0.5, 1.5, (P, 1))
    qbar = 20.0 + rng.normal(0, 1, (P, 1)) + np.cumsum(rng.normal(0, 1, (P, D)), axis=1)
    cfg = L.with_config(L.PRIMARY, fit_start=dates[0], fit_end=dates[39],
                        test_start=dates[40], test_end=dates[-1], min_train=12, min_test=6)
    return L.Data(cfg=cfg, dates=dates, sids=np.array([f"w{i}" for i in range(W)]),
                  county=np.array(["YL"] * W), layer=np.ones(W, int), cell=np.arange(W),
                  xy=xy, h_cell=rng.normal(0, 1, (W, D)), obs=np.full((W, D), np.nan),
                  T=np.full(W, 3000.0), S=np.full(W, 1.5e-4), w=np.full(W, 0.9),
                  t_at_bound=np.zeros(W, bool), pump_xy=pxy, pump_county=np.array(["yunlin"] * P),
                  pump_row=np.arange(P), Qbar=qbar, qs=q - qbar,
                  folds=[{"fold": 0, "train": [f"w{i}" for i in range(6)],
                          "test": [f"w{i}" for i in range(6, 12)]},
                         {"fold": 1, "train": [f"w{i}" for i in range(6, 12)],
                          "test": [f"w{i}" for i in range(6)]}])


def test_kernel_small_argument_is_log_like_and_zero_at_rmax():
    r = np.array([100.0, 500.0, 1999.999])
    gs, gl = L.kernel_values(r, 3000.0, 1.5e-4, 1.0, L.PRIMARY)
    assert np.all(np.abs(gs[:2] / gl[:2] - 1) < 0.05)
    assert abs(gs[-1]) < 1e-8 and abs(gl[-1]) < 1e-8
    # floor at r_min
    g0, _ = L.kernel_values(np.array([1.0]), 3000.0, 1.5e-4, 1.0, L.PRIMARY)
    assert math.isclose(g0[0], gs[0])


def test_fit_recovers_multipliers_and_scores():
    d = _data()
    Gs, Gl, n = L.operator_matrices(d)
    assert (n > 0).all()
    S, Lx = L.regressors(d, Gs, Gl)
    rng = np.random.default_rng(1)
    d.obs = d.h_cell - (6.0 * S + 2.0 * Lx) + rng.normal(0, 0.01, d.h_cell.shape) \
        + rng.normal(0, 3, (len(d.sids), 1))
    m = L.masks(d)
    a = L.fit_multipliers(d, [S, Lx], m, np.ones(len(d.sids), bool))
    assert np.allclose(a, [6.0, 2.0], rtol=0.05)
    q, sd, cuts = L.quartiles(d, m)
    clim = L.climatology(d, m)
    t1 = L.run_t1(d, [S, Lx], m, q, clim)
    assert np.nansum(t1["cor"]["mse"]) < np.nansum(t1["base"]["mse"])
    t2 = L.run_t2(d, [S, Lx], m, q, clim)
    assert set(t2["a_by_fold"]) == {0, 1}


def test_nonnegative_and_placebo_geometry():
    d = _data(2)
    Gs, Gl, n = L.operator_matrices(d)
    S, Lx = L.regressors(d, Gs, Gl)
    d.obs = d.h_cell + 5.0 * S       # wrong sign: rising heads with pumping
    m = L.masks(d)
    a = L.fit_multipliers(d, [S, Lx], m, np.ones(len(d.sids), bool))
    assert np.all(a >= 0)
    xy = L.shuffle_xy(d, np.random.default_rng(0))
    assert sorted(map(tuple, xy)) == sorted(map(tuple, d.pump_xy))
