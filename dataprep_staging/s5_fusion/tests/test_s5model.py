"""CPU unit tests of the S5 model core (synthetic data only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import s5model as M  # noqa: E402


def _irw_cov(knots, sig_v0, q):
    """Covariance of [u_0..u_{T-1}, v_0..v_{T-1}] of the IRW by forward propagation."""
    T = len(knots)
    S = np.zeros((2 * T, 2 * T))
    # state transition per step, accumulate joint covariance
    A = np.zeros((2 * T, 2 * T))
    # express every state as a linear map of independent innovations
    n_innov = 1 + 2 * (T - 1)
    Lm = np.zeros((2 * T, n_innov))
    Lm[T, 0] = sig_v0                          # v_0
    for k in range(T - 1):
        d = knots[k + 1] - knots[k]
        Qd = q * np.array([[d ** 3 / 3, d ** 2 / 2], [d ** 2 / 2, d]])
        C = np.linalg.cholesky(Qd)
        Lm[k + 1] = Lm[k] + d * Lm[T + k]
        Lm[T + k + 1] = Lm[T + k]
        Lm[[k + 1, T + k + 1], 1 + 2 * k:3 + 2 * k] += C
    S = Lm @ Lm.T
    del A
    return S


def test_hermite_linear_exact():
    tb = M.TimeBasis([2012.0, 2013.0, 2014.5, 2016.0])
    a = 7.0
    state = np.zeros(tb.K)
    for k in range(tb.T):
        iu = tb.iu(k)
        if iu is not None:
            state[iu] = a * (tb.knots[k] - 2012.0)
        state[tb.iv(k)] = a
    t = np.array([2012.0, 2012.3, 2013.7, 2015.99, 2016.4])
    H = tb.rows(t, seasonal=False)
    assert np.allclose(H @ state, a * (t - 2012.0))


def test_prior_precision_matches_covariance():
    knots = [2012.0, 2013.0, 2014.0, 2015.5]
    tb = M.TimeBasis(knots)
    J = tb.prior_precision(sig_v0=3.0, q=2.0, sig_s=1.5)
    S = _irw_cov(np.array(knots), 3.0, 2.0)[1:, 1:]          # drop u_0 = 0
    assert np.allclose(np.linalg.inv(J[:tb.Kt, :tb.Kt]), S, rtol=1e-9, atol=1e-9)
    assert np.isclose(J[tb.ic, tb.ic], 1 / 1.5 ** 2)


def test_assembler_views():
    a = M.Assembler(5, 4, 3)
    assert np.shares_memory(a.P4, a.P) and np.shares_memory(a.Plg, a.P)


def _tiny_problem(seed=0):
    rng = np.random.default_rng(seed)
    nodes = np.array([[x, y] for x in (0.0, 5.0, 10.0) for y in (0.0, 5.0, 10.0)])
    lat = M.Lattice(nodes)
    lat.setup(8.0)
    tb = M.TimeBasis([2012.0, 2013.0, 2014.0, 2015.0])
    sites = rng.uniform(0, 10, size=(4, 2))
    return rng, lat, tb, sites


def _brute_m2ll(lat, tb, sites, obs, hyper, glob_var, V_off):
    """-2 log N(y; 0, Sigma) with every effect explicit (offsets with variance V_off)."""
    J = tb.prior_precision(hyper["sig_v0"], hyper["q"], hyper["sig_s"])
    Ct = np.linalg.inv(J)
    Kzz = lat.Kinv.copy()
    Kzz = np.linalg.inv(Kzz)
    Clat = np.kron(Ct, Kzz)
    rows, ys = [], []
    for t, y, site in obs:
        phi = lat.weights(sites[site][None])[0]
        H = tb.rows(t)
        for i in range(len(t)):
            g = np.outer(H[i], phi).ravel()
            rows.append((g, t[i], site))
            ys.append(y[i])
    n = len(ys)
    G = np.array([r[0] for r in rows])
    tt = np.array([r[1] for r in rows])
    st = np.array([r[2] for r in rows])
    Sig = G @ Clat @ G.T
    Sig += glob_var * np.outer(tt - 2012.0, tt - 2012.0)               # mu_v
    for s in np.unique(st):
        idx = np.flatnonzero(st == s)
        tau = tt[idx] - tt[idx].mean()
        Sig[np.ix_(idx, idx)] += V_off + hyper["sig_r"] ** 2 * np.outer(tau, tau)
    Sig += hyper["sig"] ** 2 * np.eye(n)
    y = np.array(ys)
    sgn, ld = np.linalg.slogdet(Sig)
    return float(y @ np.linalg.solve(Sig, y) + ld + n * np.log(2 * np.pi)), n


def _latent_m2ll(lat, tb, sites, obs, hyper, glob_var):
    K, m = tb.K, lat.m
    asm = M.Assembler(K, m, 1)
    for t, y, site in obs:
        phi = lat.weights(sites[site][None])[0]
        H = tb.rows(t)
        W, ld, ne = M.site_local(t, np.zeros(len(t), int), hyper["sig"], hyper["sig_r"])
        G = (t - 2012.0)[:, None]
        asm.add_site(phi, H, W, y, gidx=[0], G=G, logdet=ld, n_eff=ne)
    c = asm.result()
    J = tb.prior_precision(hyper["sig_v0"], hyper["q"], hyper["sig_s"])
    post = M.solve(c, J, lat, np.array([1.0 / glob_var]))
    return post.m2loglik, post


def test_marginal_likelihood_matches_bruteforce_differences():
    rng, lat, tb, sites = _tiny_problem()
    obs = []
    for s in range(len(sites)):
        t = np.sort(rng.uniform(2012.0, 2015.0, size=6))
        obs.append((t, rng.normal(0, 10, size=6) + 3.0 * (t - 2012), s))
    h1 = {"sig_v0": 4.0, "q": 3.0, "sig_s": 2.0, "sig": 2.0, "sig_r": 1.0}
    h2 = {"sig_v0": 6.0, "q": 1.0, "sig_s": 1.0, "sig": 3.0, "sig_r": 0.5}
    gv = 50.0 ** 2
    V = 1e8
    b1, n = _brute_m2ll(lat, tb, sites, obs, h1, gv, V)
    b2, _ = _brute_m2ll(lat, tb, sites, obs, h2, gv, V)
    l1, _ = _latent_m2ll(lat, tb, sites, obs, h1, gv)
    l2, _ = _latent_m2ll(lat, tb, sites, obs, h2, gv)
    # restricted vs explicit-diffuse differ by a constant n_sites * log(V) (+ 2pi terms)
    assert abs((b1 - b2) - (l1 - l2)) < 1e-4
    const = b1 - l1
    assert abs(const - len(sites) * (np.log(V) + np.log(2 * np.pi))) < 1e-3


def test_insar_month_projection_bruteforce():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 20, size=(8, 2))
    nep = rng.integers(1, 4, size=8)
    W, ld, ne = M.ramp_month(xy, nep, 2.0, 3.0, 6.0)
    # W annihilates planes
    xc = xy - xy.mean(0)
    X = np.column_stack([np.ones(8), xc])
    assert np.allclose(W @ X, 0, atol=1e-9)
    assert ne == 5
    # equals the large-variance limit of an explicit plane
    C = 4.0 * np.eye(8) + 9.0 * np.exp(-M.dist(xy, xy) / 6.0)
    s = 1 / np.sqrt(nep)
    R = C * np.outer(s, s)
    V = 1e9
    Rb = R + V * X @ X.T
    assert np.allclose(np.linalg.inv(Rb), W, atol=1e-6)


def test_posterior_coverage_synthetic():
    """Data simulated from the prior: 90 % intervals of held-out increments cover ~ 0.9."""
    rng = np.random.default_rng(11)
    nodes = np.array([[x, y] for x in np.arange(0, 21, 5.0) for y in np.arange(0, 21, 5.0)])
    lat = M.Lattice(nodes)
    lat.setup(10.0)
    tb = M.TimeBasis([2012.0, 2013.0, 2014.0, 2015.0, 2016.0])
    hyper = {"sig_v0": 10.0, "q": 4.0, "sig_s": 2.0, "sig": 2.0, "sig_r": 1.0}
    J = tb.prior_precision(hyper["sig_v0"], hyper["q"], hyper["sig_s"])
    Kzz = np.linalg.inv(lat.Kinv)
    hits = []
    for _rep in range(30):
        Ct = np.linalg.inv(J)
        w = (np.linalg.cholesky(np.kron(Ct, Kzz) + 1e-9 * np.eye(tb.K * lat.m))
             @ rng.normal(size=tb.K * lat.m))
        sites = rng.uniform(0, 20, size=(25, 2))
        phis = lat.weights(sites)
        asm = M.Assembler(tb.K, lat.m, 1)
        truth = []
        for s in range(len(sites)):
            t = np.sort(rng.uniform(2012.0, 2016.0, size=5))
            H = tb.rows(t)
            g = (H[:, :, None] * phis[s][None, None, :]).reshape(len(t), -1)
            r = rng.normal(0, hyper["sig_r"])
            y = g @ w + 50.0 + r * (t - t.mean()) + rng.normal(0, hyper["sig"], size=len(t))
            if s < 18:
                W, ld, ne = M.site_local(t, np.zeros(len(t), int), hyper["sig"],
                                         hyper["sig_r"])
                asm.add_site(phis[s], H, W, y, gidx=[0], G=(t - 2012.0)[:, None],
                             logdet=ld, n_eff=ne)
            else:
                truth.append((phis[s], g @ w))
        post = M.solve(asm.result(), J, lat, np.array([1e-6]))
        # held-out field values at the knots
        for phi, _ in truth:
            k = 4
            h = tb.knot_row(k)
            G = M.functional_rows(tb, phi[None], h[None], 1, mu_coef=np.array([4.0]))
            mu = float((G @ post.theta)[0])
            sd = float(np.sqrt(post.var_of(G))[0])
            true = float(np.outer(h, phi).ravel() @ w)
            hits.append(abs(true - mu) <= 1.645 * sd)
    cov = float(np.mean(hits))
    assert 0.8 <= cov <= 0.98, cov


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
