"""CPU unit tests of the S5 repair-round-2 model (synthetic values only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import s5model as M  # noqa: E402
import s5r2_model as RM  # noqa: E402

H = {"sig_v0": 7.0, "sig_be": 3.0, "sig_ep": 4.0, "sig_s": 2.0}


def test_temporal_prior_moments():
    tb = M.TimeBasis([2012.0, 2013.0, 2014.0, 2015.5])
    A = RM.temporal_cov_map(tb.knots)
    s2 = np.r_[H["sig_v0"] ** 2, np.full(tb.T, H["sig_be"] ** 2),
               np.full(tb.T - 1, H["sig_ep"] ** 2)]
    Ct = (A * s2) @ A.T
    assert np.all(np.linalg.eigvalsh(Ct) > 0)
    # Var(u_1) = d^2 (sig_v0^2 + sig_be^2 / 2) + sig_ep^2 with d = 1
    assert np.isclose(Ct[0, 0], 49.0 + 9.0 / 2 + 16.0)
    # Var(v_k) = sig_v0^2 + sig_be^2 ; Cov(v_j, v_k) = sig_v0^2
    iv = tb.T - 1
    assert np.isclose(Ct[iv, iv], 49.0 + 9.0)
    assert np.isclose(Ct[iv, iv + 1], 49.0)
    # increments of u over non-adjacent intervals share only b: Cov(u1-u0, u3-u2) = d1 d3 sig_v0^2
    e1 = np.zeros(tb.Kt)
    e1[0] = 1.0
    e3 = np.zeros(tb.Kt)
    e3[2], e3[1] = 1.0, -1.0
    assert np.isclose(e1 @ Ct @ e3, 1.0 * 1.5 * 49.0)
    J = RM.prior_precision2(tb, H)
    assert np.allclose(np.linalg.inv(J[:tb.Kt, :tb.Kt]), Ct, rtol=1e-8, atol=1e-8)
    assert np.isclose(J[tb.ic, tb.ic], 1.0 / 4.0)


def test_add_month2_matches_dense():
    rng = np.random.default_rng(3)
    K, m, ng, n = 3, 4, 7, 5
    asm = M.Assembler(K, m, ng)
    Phi = rng.normal(size=(n, m))
    h = np.array([0.3, 0.0, 0.7])
    B = rng.normal(size=(n, n))
    W = B @ B.T + n * np.eye(n)
    y = rng.normal(size=n)
    gidx = np.array([1, 2, 3, 4, 5, 6])
    G = rng.normal(size=(n, gidx.size))
    RM._add_month2(asm, Phi, h, W, y, gidx, G, 0.0, n)
    c = asm.result()
    X = np.zeros((n, K * m + ng))
    for k in range(K):
        X[:, k * m:(k + 1) * m] = h[k] * Phi
    for j, gi in enumerate(gidx):
        X[:, K * m + gi] += G[:, j]
    assert np.allclose(c.P, X.T @ W @ X)
    assert np.allclose(c.b, X.T @ W @ y)
    assert np.isclose(c.yWy, y @ W @ y)


def test_expcov_and_cmp_R_pd():
    t = np.linspace(2015, 2021, 40)
    h = {"sig_cmp": 1.0, "sig_rc": 0.5, "sig_cc": 3.0, "tau_c": 0.4}
    R = RM.cmp_R(t, h)
    assert np.all(np.linalg.eigvalsh(R) > 0)
    assert np.isclose(R[0, 0], 1.0 + 0.25 * (t[0] - t.mean()) ** 2 + 9.0)
