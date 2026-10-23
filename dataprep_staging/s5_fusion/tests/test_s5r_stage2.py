"""CPU unit tests of the S5 repair-round stage 2 (monthly anomaly; synthetic data only)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import s5model as M  # noqa: E402
import s5r_stage2 as S2  # noqa: E402


# ----------------------------------------------------------------------------- block tridiagonal
def _random_btd(rng, Mn=5, m=4):
    B = rng.normal(size=(m, m))
    B = 0.3 * (B + B.T)
    D = []
    for _ in range(Mn):
        X = rng.normal(size=(m, m))
        D.append(X @ X.T + 6.0 * np.eye(m))
    D = np.stack(D)
    P = np.zeros((Mn * m, Mn * m))
    for k in range(Mn):
        P[k * m:(k + 1) * m, k * m:(k + 1) * m] = D[k]
        if k:
            P[k * m:(k + 1) * m, (k - 1) * m:k * m] = B
            P[(k - 1) * m:k * m, k * m:(k + 1) * m] = B
    return D, B, P


def test_btd_matches_dense():
    rng = np.random.default_rng(1)
    Mn, m = 5, 4
    D, B, P = _random_btd(rng, Mn, m)
    bt = S2.BTD(D, B)
    assert abs(bt.logdet - np.linalg.slogdet(P)[1]) < 1e-9
    b = rng.normal(size=(Mn, m))
    assert np.allclose(bt.solve(b).ravel(), np.linalg.solve(P, b.ravel()))
    Pi = np.linalg.inv(P)
    marg = bt.marginals()
    for k in range(Mn):
        assert np.allclose(marg[k], Pi[k * m:(k + 1) * m, k * m:(k + 1) * m])
    month = np.array([0, 2, 2, 4, 1, 3])
    G = rng.normal(size=(6, m))
    Gd = np.zeros((6, Mn * m))
    for i, k in enumerate(month):
        Gd[i, k * m:(k + 1) * m] = G[i]
    assert np.allclose(bt.functional_cov(month, G), Gd @ Pi @ Gd.T)


# ----------------------------------------------------------------------------- stage 2
def _tiny_stage2(seed=0, Mn=6):
    rng = np.random.default_rng(seed)
    nodes = np.array([[x, y] for x in (0.0, 6.0, 12.0) for y in (0.0, 6.0, 12.0)])
    cells = np.array([[x + 0.5, y + 0.5] for x in range(12) for y in range(12)], float)
    unit_cells, unit_xy = [], []
    for bx in range(0, 12, 3):
        for by in range(0, 12, 3):
            c = [i for i, (x, y) in enumerate(cells) if bx <= x < bx + 3 and by <= y < by + 3]
            unit_cells.append(np.array(c))
            unit_xy.append(cells[c].mean(0))
    unit_xy = np.array(unit_xy)
    months = [f"2020-{k + 1:02d}" for k in range(Mn)]
    groups = []
    for k, mo in enumerate(months):
        if k == 2:
            continue                                   # a month without data
        u = np.sort(rng.choice(len(unit_cells), size=10, replace=False))
        groups.append({"month": mo, "unit": u, "r": rng.normal(0, 5, size=u.size),
                       "nep": rng.integers(1, 4, size=u.size).astype(float)})
    return rng, nodes, cells, unit_xy, unit_cells, months, groups


def _brute_stage2(s2: S2.Stage2, h: dict, V: float) -> float:
    """-2 log N(r; 0, Sigma) with the anomaly, the noise and explicit planes (variance V)."""
    s2.lattice(h["ell_a"])
    K = np.linalg.inv(s2.Kinv)
    Mn, m = s2.M, s2.m
    rho = h["rho"]
    Ct = np.array([[rho ** abs(i - j) for j in range(Mn)] for i in range(Mn)])
    Ca = np.kron(Ct, h["sig_a"] ** 2 * K)
    rows, r, blocks = [], [], []
    for g in s2.groups:
        Phi = s2.Phi_units[g["unit"]]
        for i in range(len(g["unit"])):
            row = np.zeros(Mn * m)
            row[g["mi"] * m:(g["mi"] + 1) * m] = Phi[i]
            rows.append(row)
        r.append(g["r"])
        xy = s2.unit_xy[g["unit"]]
        C = h["sig_ins2"] ** 2 * np.eye(len(xy)) + h["sig_aps2"] ** 2 * np.exp(
            -M.dist(xy, xy) / h["ell_aps2"])
        sc = 1 / np.sqrt(g["nep"])
        xc = xy - xy.mean(0)
        X = np.column_stack([np.ones(len(xy)), xc])
        blocks.append(C * np.outer(sc, sc) + V * X @ X.T)
    G = np.array(rows)
    r = np.concatenate(r)
    from scipy.linalg import block_diag
    Sig = G @ Ca @ G.T + block_diag(*blocks)
    sgn, ld = np.linalg.slogdet(Sig)
    return float(r @ np.linalg.solve(Sig, r) + ld + len(r) * math.log(2 * math.pi))


def test_stage2_m2ll_matches_bruteforce():
    rng, nodes, cells, unit_xy, unit_cells, months, groups = _tiny_stage2()
    s2 = S2.Stage2(nodes, cells, unit_xy, unit_cells, months, groups)
    h1 = {"sig_ins2": 1.5, "sig_aps2": 3.0, "ell_aps2": 8.0, "sig_a": 4.0, "ell_a": 7.0,
          "rho": 0.6}
    h2 = {"sig_ins2": 2.5, "sig_aps2": 2.0, "ell_aps2": 5.0, "sig_a": 2.0, "ell_a": 10.0,
          "rho": 0.2}
    V = 1e8
    b1, b2 = _brute_stage2(s2, h1, V), _brute_stage2(s2, h2, V)
    l1, l2 = s2.solve(h1)["m2ll"], s2.solve(h2)["m2ll"]
    assert abs((b1 - b2) - (l1 - l2)) < 1e-4
    p = 3 * len(s2.groups)
    assert abs((b1 - l1) - p * (math.log(V) + math.log(2 * math.pi))) < 1e-3


def test_stage2_cell_moments_and_support():
    rng, nodes, cells, unit_xy, unit_cells, months, groups = _tiny_stage2(2)
    s2 = S2.Stage2(nodes, cells, unit_xy, unit_cells, months, groups)
    h = {"sig_ins2": 1.0, "sig_aps2": 2.0, "ell_aps2": 8.0, "sig_a": 4.0, "ell_a": 7.0,
         "rho": 0.7}
    sol = s2.solve(h)
    mom = s2.cell_moments(h, sol)
    assert mom["mean"].shape == (s2.M, len(cells))
    assert np.all(mom["var"] <= mom["prior"][None, :] * (1 + 1e-9))
    obs = s2.observed_cells()
    assert obs.shape == mom["vr"].shape
    assert not obs[2].any()                            # month without data
    # a wider domain, units on the west half only, low noise: the west is resolved
    nodes = np.array([[x, y] for x in np.arange(0, 33, 4.0) for y in np.arange(0, 33, 4.0)])
    cells = np.array([[x + 0.5, y + 0.5] for x in range(32) for y in range(32)], float)
    unit_cells, unit_xy = [], []
    for bx in range(0, 32, 4):
        for by in range(0, 32, 4):
            c = [i for i, (x, y) in enumerate(cells) if bx <= x < bx + 4 and by <= y < by + 4]
            unit_cells.append(np.array(c))
            unit_xy.append(cells[c].mean(0))
    unit_xy = np.array(unit_xy)
    west = np.flatnonzero(unit_xy[:, 0] < 16)
    months = ["2020-01", "2020-02", "2020-03"]
    groups = [{"month": mo, "unit": west, "r": rng.normal(0, 3, size=west.size),
               "nep": np.ones(west.size)} for mo in months]
    s2 = S2.Stage2(nodes, cells, unit_xy, unit_cells, months, groups)
    h = {"sig_ins2": 0.3, "sig_aps2": 0.3, "ell_aps2": 8.0, "sig_a": 5.0, "ell_a": 5.0,
         "rho": 0.5}
    mom = s2.cell_moments(h, s2.solve(h))
    wc = cells[:, 0] < 14
    ec = cells[:, 0] > 22
    assert mom["vr"][1][wc].mean() > 0.5 > mom["vr"][1][ec].mean()


def test_stage2_recovers_simulated_anomaly():
    rng = np.random.default_rng(5)
    nodes = np.array([[x, y] for x in np.arange(0, 25, 6.0) for y in np.arange(0, 25, 6.0)])
    cells = np.array([[x + 0.5, y + 0.5] for x in range(24) for y in range(24)], float)
    unit_cells, unit_xy = [], []
    for bx in range(0, 24, 4):
        for by in range(0, 24, 4):
            c = [i for i, (x, y) in enumerate(cells) if bx <= x < bx + 4 and by <= y < by + 4]
            unit_cells.append(np.array(c))
            unit_xy.append(cells[c].mean(0))
    unit_xy = np.array(unit_xy)
    Mn = 24
    months = [str(p) for p in pd.period_range("2020-01", periods=Mn, freq="M")]
    truth_h = {"sig_ins2": 1.0, "sig_aps2": 2.0, "ell_aps2": 15.0, "sig_a": 5.0,
               "ell_a": 8.0, "rho": 0.8}
    s2 = S2.Stage2(nodes, cells, unit_xy, unit_cells, months, [])
    s2.lattice(truth_h["ell_a"])
    K = np.linalg.inv(s2.Kinv)
    L = np.linalg.cholesky(truth_h["sig_a"] ** 2 * K + 1e-9 * np.eye(len(nodes)))
    a = np.zeros((Mn, len(nodes)))
    a[0] = L @ rng.normal(size=len(nodes))
    for k in range(1, Mn):
        a[k] = truth_h["rho"] * a[k - 1] + math.sqrt(1 - truth_h["rho"] ** 2) * (
            L @ rng.normal(size=len(nodes)))
    groups = []
    for k, mo in enumerate(months):
        u = np.arange(len(unit_cells))
        xy = unit_xy[u]
        C = truth_h["sig_ins2"] ** 2 * np.eye(len(u)) + truth_h["sig_aps2"] ** 2 * np.exp(
            -M.dist(xy, xy) / truth_h["ell_aps2"])
        noise = np.linalg.cholesky(C) @ rng.normal(size=len(u))
        plane = rng.normal(0, 10) + rng.normal(0, 0.5) * xy[:, 0] + rng.normal(0, 0.5) * xy[:, 1]
        r = s2.Phi_units[u] @ a[k] + noise + plane
        groups.append({"month": mo, "unit": u, "r": r, "nep": np.ones(len(u))})
    s2 = S2.Stage2(nodes, cells, unit_xy, unit_cells, months, groups)
    h0 = {"sig_ins2": 2.0, "sig_aps2": 4.0, "ell_aps2": 8.0, "sig_a": 2.0, "ell_a": 12.0,
          "rho": 0.4}
    h, info = S2.fit_hyper(s2, h0, maxfev=250)
    assert 0.5 < h["rho"] < 0.97
    assert 2.0 < h["sig_a"] < 12.0
    sol = s2.solve(h)
    # compare the plane-free part of the anomaly at the units
    est = sol["x"] @ s2.Phi_units.T
    tru = a @ s2.Phi_units.T
    X = np.column_stack([np.ones(len(unit_xy)), unit_xy])
    Pp = np.eye(len(unit_xy)) - X @ np.linalg.pinv(X)
    e, t = (est @ Pp.T).ravel(), (tru @ Pp.T).ravel()
    assert np.corrcoef(e, t)[0, 1] > 0.7
    null = s2.m2ll_null(h["sig_ins2"], h["sig_aps2"], h["ell_aps2"])
    assert null - sol["m2ll"] > 10.0
