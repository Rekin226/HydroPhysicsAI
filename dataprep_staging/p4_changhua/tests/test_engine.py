"""Synthetic CPU tests of the P4.0 engine (no repository data; small grids, < 30 s).

Run from dataprep_staging/: CUDA_VISIBLE_DEVICES= python -m pytest -q p4_changhua/tests
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from p4_changhua import engine  # noqa: E402

from hydrophysics.twin import calibrate_flow as cf  # noqa: E402
from hydrophysics.twin.boundaries import fan_boundaries  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.rivers import RiverSet  # noqa: E402
from hydrophysics.twin.spread import pairwise_d2_km, spread_matrix  # noqa: E402

torch.set_num_threads(1)
L = 4


def _model(seed=0, rivers=False):
    rng = np.random.default_rng(seed)
    mask = np.ones((6, 8), bool)
    mask[0, :2] = False
    mask[5, 6:] = False
    grid = FanGrid(nx=8, ny=6, dx=1000.0, x0=0.0, y0=0.0, mask=mask)
    A = grid.n_active
    m = FlowModel(grid, n_layers=L, dt_days=30.0, boundaries=fan_boundaries(grid, 5.0))
    with torch.no_grad():
        m.log_T.copy_(torch.as_tensor(rng.uniform(math.log(50), math.log(5000), (L, A))))
        m.log_S.copy_(torch.as_tensor(rng.uniform(math.log(1e-4), math.log(0.1), (L, A))))
        m.log_L.copy_(torch.as_tensor(rng.uniform(math.log(1e-4), math.log(1e-2),
                                                  (L - 1, A))))
        m.log_C_coast.copy_(torch.full((L, 1), math.log(500.0)))
        m.log_C_apex.copy_(torch.full((1, 1), math.log(800.0)))
    sc = {"log_eta": torch.tensor(math.log(0.5), dtype=torch.float64),
          "log_head_extra": torch.tensor(math.log(40.0), dtype=torch.float64),
          "pump_split_logit": torch.tensor(-1.0, dtype=torch.float64),
          "return_frac_logit": torch.tensor(0.3, dtype=torch.float64),
          "recharge_frac_logit": torch.tensor(-2.0, dtype=torch.float64),
          "spread_W": spread_matrix(pairwise_d2_km(grid),
                                    torch.tensor(math.log(0.8), dtype=torch.float64))}
    if rivers:
        idx = np.array([3, 10, 17, 24, 31])
        rs = RiverSet(idx=idx, weight=np.full(5, 0.3), group=np.array([0, 0, 1, 1, 1]),
                      h_riv=np.array([12.0, 11.0, 9.0, 8.0, 7.0]),
                      rbot=np.array([10.0, 9.0, 7.0, 6.0, 5.0]), names=("a", "b"))
        m.set_rivers(rs, layer=0, mode="riv")
        sc["log_C_riv"] = torch.tensor([math.log(2e4), math.log(5e3)], dtype=torch.float64)
    return grid, m, sc


def _forcing(A, T, seed=1):
    rng = np.random.default_rng(seed)
    E = rng.uniform(0, 4e4, (A, T)) * (1 + 0.8 * np.sin(np.arange(T) * 2 * np.pi / 12))
    R = rng.uniform(0, 3e-3, (A, T))
    ge = rng.uniform(8, 20, A)
    h0 = np.stack([ge - 2 - k for k in range(L)])
    return E, R, ge, h0


def test_constants_match_repository():
    assert engine.RETURN_FRAC_MAX == cf.RETURN_FRAC_MAX
    assert engine.WELL_DATUM_SD == cf.WELL_DATUM_SD_DEFAULT
    assert engine.WELL_DATUM_MONTHS_PER_OBS == cf.WELL_DATUM_MONTHS_PER_OBS


def test_operator_matches_matvec_with_multipliers():
    grid, m, _ = _model()
    A = grid.n_active
    cells = np.zeros(A, bool)
    cells[:9] = True
    Tm = engine.region_multiplier(cells, 3.0, L, rows=(1, 2))
    Lm = engine.region_multiplier(cells, 0.1, L - 1)
    op = engine.assemble_operator(m, T_mult=Tm, L_mult=Lm, S_mult=np.full(A, 2.0))
    chk = engine.verify_operator(op, m, T_mult=Tm, L_mult=Lm, S_mult=np.full(A, 2.0))
    assert chk["rel_max_matvec"] < 1e-13
    assert chk["max_asymmetry"] == 0.0


@pytest.mark.parametrize("rivers", [False, True])
def test_rollout_matches_torch(rivers):
    grid, m, sc = _model(rivers=rivers)
    A, T = grid.n_active, 24
    E, R, ge, h0 = _forcing(A, T)
    tw = engine.SparseTwin(m, sc, ge)
    Ex = np.zeros_like(E)
    Ex[5:12] = 3e4
    res = engine.verify_against_torch(tw, sc, h0, E, R, E_extra=Ex)
    assert res["max_abs_dh_m"] < 1e-6
    if rivers:
        H = res["H_torch"]
        conn = H[0, tw.riv_idx, :] > tw.riv_rbot[:, None]
        assert conn.any() and (~conn).any()       # the switch actually toggles


def test_adjoint_matches_forward_impulse_and_convolution():
    grid, m, sc = _model()
    A, NT = grid.n_active, 30
    tw = engine.SparseTwin(m, sc, np.zeros(A))
    oi = np.array([0, 7, 20, 33, 40])
    ol = np.array([0, 1, 2, 3, 1])
    G = engine.adjoint_greens(tw, oi, ol, NT, dtype=np.float64)
    rng = np.random.default_rng(3)
    for lay in range(L):
        q0 = np.zeros((L, A))
        c = int(rng.integers(A))
        q0[lay, c] = 1.0
        fwd = engine.forward_impulse(tw, q0, NT, oi, ol)
        assert np.abs(fwd - G[lay][:, c, :]).max() <= 1e-12 * np.abs(fwd).max()
    K = engine.pump_kernel(G, tw.pump_weights(), dtype=np.float64)
    rate = rng.uniform(0, 1e3, (A, NT))
    q = np.stack([tw.layer_sources(rate[:, t]) for t in range(NT)], axis=-1)
    H = tw.rollout_linear(q)[..., 1:][ol, oi, :]
    y = engine.convolve(K, rate)
    assert np.abs(-y - H).max() <= 1e-10 * np.abs(H).max()
    gain = engine.steady_gain(tw, oi, ol, layers=(1,))[1]
    G_long = engine.adjoint_greens(tw, oi, ol, 400, layers=(1,), dtype=np.float64)[1]
    assert np.abs(G_long.sum(axis=2) - gain).max() <= 1e-8 * np.abs(gain).max()


def test_frozen_lift_is_linear_and_matches_kernel():
    grid, m, sc = _model()
    A, T = grid.n_active, 18
    E, R, ge, h0 = _forcing(A, T)
    tw = engine.SparseTwin(m, sc, ge)
    H0 = tw.rollout(h0, E=E, R=R)
    Ex = np.zeros_like(E)
    Ex[4:9] = 5e4
    H1 = tw.rollout(h0, E=E, R=R, E_extra=Ex, frozen_head=H0[1, :, :-1])
    H0f = tw.rollout(h0, E=E, R=R, frozen_head=H0[1, :, :-1])
    oi, ol = np.arange(0, A, 5), np.full(len(range(0, A, 5)), 1)
    G = engine.adjoint_greens(tw, oi, ol, T, dtype=np.float64)
    K = engine.pump_kernel(G, tw.pump_weights(), dtype=np.float64)
    c = engine.energy_rate_jacobian(tw, H0)
    dq = c * tw.spread(Ex)
    dh = (H1 - H0f)[ol, oi, 1:]
    assert np.abs(dh + engine.convolve(K, dq)).max() <= 1e-10 * np.abs(dh).max()


def test_profile_datum_matches_fit_flow():
    rng = np.random.default_rng(5)
    W, T = 12, 40
    pred = rng.standard_normal((W, T))
    obs = pred + rng.standard_normal((W, 1)) * 4 + rng.standard_normal((W, T))
    mask = rng.uniform(size=(W, T)) > 0.3
    mask[3, :] = False
    mask[4, :5] = True
    mask[4, 5:] = False
    d, kappa, s2, n = engine.profile_datum(pred, np.where(mask, obs, np.nan), mask)
    dt, kt, s2t, nt = cf._profile_well_datum(torch.as_tensor(pred),
                                             torch.as_tensor(np.where(mask, obs, 0.0)),
                                             torch.as_tensor(mask), 5.0)
    assert np.abs(d - dt.numpy()[:, 0]).max() < 1e-12
    assert abs(s2 - float(s2t)) < 1e-12
    assert d[3] == 0.0


def test_solve_with_datum_profiles_exactly():
    rng = np.random.default_rng(7)
    W, T, P = 10, 60, 3
    X = rng.standard_normal((W, T, P))
    theta = np.array([0.5, -1.0, 2.0])
    obs = X @ theta + rng.standard_normal((W, 1)) * 3 + 0.5 * rng.standard_normal((W, T))
    mask = rng.uniform(size=(W, T)) > 0.2
    res = engine.solve_with_datum(X, np.where(mask, obs, np.nan))
    # joint augmented least squares with the converged kappa gives the same answer
    kap = res["kappa"]
    rows, ys = [], []
    for w in range(W):
        for t in np.flatnonzero(mask[w]):
            r = np.zeros(P + W)
            r[:P] = X[w, t]
            r[P + w] = 1.0
            rows.append(r)
            ys.append(obs[w, t])
    for w in range(W):
        r = np.zeros(P + W)
        r[P + w] = math.sqrt(kap[w])
        rows.append(r)
        ys.append(0.0)
    beta = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)[0]
    assert np.abs(beta[:P] - res["theta"]).max() < 1e-9
    assert np.abs(beta[P:] - res["datum"]).max() < 1e-9
    free = engine.solve_with_datum(X, np.where(mask, obs, np.nan), sd=np.inf)
    assert np.all(free["kappa"] == 0)
    fi = engine.fisher_with_datum(X, mask, res["s2"], kap, fixed=np.array([0, 1, 0], bool))
    assert np.isnan(fi["sd"][1]) and np.all(np.isfinite(fi["sd"][[0, 2]]))


def test_variance_formulas():
    rng = np.random.default_rng(11)
    W, A, NT, T = 4, 7, 9, 15
    K = rng.standard_normal((W, A, NT))
    var_q = rng.uniform(0.5, 2.0, (A, T))
    V = engine.head_var_independent(K, var_q)
    F = []
    for c in range(A):
        for t in range(T):
            f = np.zeros((A, T))
            f[c, t] = math.sqrt(var_q[c, t])
            F.append(f)
    U = engine.factor_responses(K, np.stack(F))
    assert np.abs(engine.head_var_factors(U) - V).max() < 1e-9 * V.max()
    Cs = np.cov(rng.standard_normal((A, 3 * A)))
    Ct = np.exp(-np.abs(np.subtract.outer(np.arange(T), np.arange(T))) / 3.0)
    Fs = engine.separable_factors(Cs, Ct, var_frac=1.0)
    S = sum(np.outer(f.ravel(), f.ravel()) for f in Fs)
    assert np.abs(S - np.kron(Cs, Ct)).max() < 1e-9


def test_bound_pinned_counts_match_meta_layout():
    theta = {"log_T_mid": [math.log(2e4), 5.0], "log_L_distal": [math.log(1e-4), -5.0],
             "log_spread_km": math.log(0.5), "pump_split_logit": -2.0}
    meta = {"l_min": 1e-4, "spread_max_km": 10.0}
    p = engine.bound_pinned(theta, meta)
    assert {(e.name, e.side) for e in p} == {("log_T_mid", "hi"), ("log_L_distal", "lo"),
                                            ("log_spread_km", "lo")}
    th2 = engine.perturb_theta(theta, "log_spread_km", 0, math.log(2.0))
    assert abs(th2["spread_km"] - 2.0) < 1e-12 and theta["log_spread_km"] == math.log(0.5)
