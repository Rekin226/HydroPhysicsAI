"""Synthetic checks of the pure functions behind N1.policy_response.v1 (no data read)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from n1_policy import policy_lib as PL  # noqa: E402


def test_fe_ols_recovers_the_slope_under_two_way_fixed_effects():
    rng = np.random.default_rng(0)
    n_i, n_t = 60, 4
    i = np.repeat(np.arange(n_i), n_t)
    t = np.tile(np.arange(n_t), n_i)
    x = rng.normal(size=n_i * n_t) + 0.5 * t + 0.3 * (i % 7)      # x correlated with the FE
    y = 2.5 * x + 3.0 * (i % 7) - 1.0 * t + rng.normal(0, 0.01, n_i * n_t)
    b = PL.fe_ols(y, x[:, None], [i, t])
    assert abs(b[0] - 2.5) < 0.01


def test_interval_exposure_sums_months_after_start_up_to_end():
    E = np.arange(12, dtype=float)[None, :].repeat(2, 0)          # (A=2, M=12)
    W = np.array([[1.0, 0.0], [1.0, 1.0]])
    out = PL.interval_exposure(E, W, np.array([2, 2]), np.array([5, 5]), np.array([1.0, 2.0]))
    assert out.tolist() == [3 + 4 + 5, 2 * (3 + 4 + 5) / 2.0]


def test_replace_months_uses_the_same_calendar_month_of_the_reference():
    mos = np.array([1, 2, 1, 2, 1, 2])
    E = np.array([[1.0, 10.0, 3.0, 30.0, 100.0, 200.0]])
    tgt = np.array([0, 0, 0, 0, 1, 1], bool)
    ref = np.array([1, 1, 1, 1, 0, 0], bool)
    Z = PL.replace_months(E, tgt, ref, mos)
    assert Z[0].tolist() == [1.0, 10.0, 3.0, 30.0, 2.0, 20.0]


def test_ratio_verdict_rules():
    assert PL.ratio_verdict(1.0, [0.9, 1.1])["verdict"] == "consistent"
    assert PL.ratio_verdict(1.0, [2.5, 4.0])["verdict"] == "inconsistent"
    assert PL.ratio_verdict(1.0, [0.5, 2.0])["verdict"] == "inconclusive"
    assert PL.ratio_verdict(1.0, [-0.1, 2.0])["verdict"] == "not_informative"


def test_partition_deviation_removes_the_mean_increment():
    cum = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 2.0, 5.0, 5.0]])
    d = PL.partition_deviation(cum)
    assert np.allclose(d[0], 0.0)
    assert np.allclose(d[1], [0.75, -1.25, 1.75, -1.25])


def test_cluster_bootstrap_interval_contains_the_full_statistic():
    rng = np.random.default_rng(1)
    v = rng.normal(5.0, 1.0, 200)
    cl = np.repeat(np.arange(20), 10)
    full, ci = PL.cluster_bootstrap(lambda r: np.array([v[r].mean()]), cl, B=300, seed=2)
    assert ci[0][0] < full[0] < ci[1][0]


def test_nearest_cell_refuses_points_off_the_grid():
    cent = np.array([[0.0, 0.0], [1000.0, 0.0]])
    c = PL.nearest_cell(cent, np.array([[900.0, 10.0], [5000.0, 0.0]]), 1000.0)
    assert c.tolist() == [1, -1]


def test_fe_ols_matches_the_dummy_regression_on_an_unbalanced_panel():
    rng = np.random.default_rng(3)
    n = 300
    i = rng.integers(0, 40, n)
    t = rng.integers(0, 5, n)
    x = rng.normal(size=(n, 2)) + i[:, None] * 0.1
    y = x @ np.array([1.5, -0.7]) + 0.2 * i + 0.5 * t + rng.normal(0, 0.1, n)
    D = np.zeros((n, 40))
    D[np.arange(n), i] = 1.0
    T_ = np.zeros((n, 5))
    T_[np.arange(n), t] = 1.0
    ref = np.linalg.lstsq(np.column_stack([x, D, T_[:, 1:]]), y, rcond=None)[0][:2]
    assert np.allclose(PL.fe_ols(y, x, [i, t]), ref, atol=1e-8)
