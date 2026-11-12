"""Round-12 gating design and retrospective scorer: unit checks on synthetic inputs."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from dataprep_staging.r12_layer import gate_power as G  # noqa: E402
from dataprep_staging.r12_layer import retro_score as R  # noqa: E402


def test_interval_error_is_the_distance_to_the_printed_interval():
    e = G.interval_error(np.array([1.0, 4.0, 7.0, 2.0]), np.array([3.0, 3.0, 3.0, -np.inf]),
                         np.array([6.0, 6.0, 6.0, 3.0]))
    assert e.tolist() == [2.0, 0.0, 1.0, 0.0]


def test_bounds_from_truth_follow_the_printed_statement():
    idx = pd.MultiIndex.from_tuples([("A", "t1"), ("A", "t2"), ("A", "t3"), ("B", "u1")])
    b = G.bounds_from_truth(pd.Series([5.0, 3.5, 1.0, 2.0], index=idx))
    assert b.loc[("A", "t1")].tolist() == [5.0, 5.0]          # the county maximum is exact
    assert b.loc[("A", "t2")].tolist() == [3.0, 5.0]          # listed: 3 to the maximum
    assert b.loc[("A", "t3")].tolist() == [-np.inf, 3.0]      # not listed: below 3
    assert b.loc[("B", "u1")].tolist() == [2.0, 2.0]          # a quiet county still has a max


def test_decide_needs_every_year_won_and_a_gain_above_zero():
    rng = np.random.default_rng(0)
    good, bad = np.full(40, 0.1), np.full(40, 1.0)
    assert G.decide([good, good], [bad, bad], [bad, bad], rng)["supported"]
    assert not G.decide([good, bad], [bad, good], [bad, bad], rng)["supported"]


def test_block_bootstrap_interval_brackets_a_constant_gain():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 50_000, size=(300, 2))
    lo, hi = R.block_bootstrap(np.full(300, 2.0), xy, rng)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)
