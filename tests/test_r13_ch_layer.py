"""Round-13 Changhua head-layer scorer: unit checks on synthetic inputs."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from dataprep_staging.r13_ch_layer import score as S  # noqa: E402


def _world(response: float, noise: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    W, T = 40, 36
    xy = rng.uniform(0, 60_000, (W, 2))
    c = rng.normal(0, 0.3, (W, T))                       # electricity-driven contrast
    base = rng.normal(0, 0.2, (W, T))
    obs = base + response * c + rng.normal(0, noise, (W, T))
    arms = {"PR": base + c, "ZR": base, "CL": np.zeros((W, T)),
            "Q0": base + rng.normal(0, 0.6, (W, T)), "R1c": base + rng.normal(0, 0.6, (W, T))}
    return arms, obs, np.ones((W, T), bool), S.block_ids(xy)


def test_rezero_removes_each_wells_fit_mean():
    a = np.array([[1.0, 3.0, 10.0], [2.0, np.nan, 4.0]])
    fit = np.array([[True, True, False], [True, False, False]])
    assert np.allclose(S.rezero(a, fit), [[-1.0, 1.0, 8.0], [0.0, np.nan, 2.0]], equal_nan=True)


def test_slope_is_per_well_demeaned():
    w = np.array([0, 0, 0, 1, 1, 1])
    c = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    e = 0.5 * c + np.array([5.0, 5.0, 5.0, -3.0, -3.0, -3.0])   # well offsets drop out
    assert S.slope_fe(e, c, w)[0] == pytest.approx(0.5)


def test_a_true_response_is_supported_with_a_consistent_size():
    arms, obs, test, g = _world(response=1.0, noise=0.1)
    r = S.decide(arms, obs, test, g, np.random.default_rng(1))
    assert r["skill"] == "layer ahead"
    assert r["response"]["verdict"] == "response supported, size consistent"


def test_no_response_is_not_supported_and_the_layer_is_not_ahead_of_zr():
    arms, obs, test, g = _world(response=0.0, noise=0.1)
    r = S.decide(arms, obs, test, g, np.random.default_rng(1))
    assert r["response"]["verdict"] == "response not supported"
    assert r["gain"]["ZR"]["mse_gain_m2"] < 0


def test_a_half_response_is_supported_without_a_consistent_size():
    arms, obs, test, g = _world(response=0.6, noise=0.05)
    r = S.decide(arms, obs, test, g, np.random.default_rng(1))
    assert r["response"]["verdict"] == "response supported"


def test_a_tiny_contrast_is_not_evaluable():
    arms, obs, test, g = _world(response=1.0, noise=0.1)
    arms["PR"] = arms["ZR"] + 0.01 * (arms["PR"] - arms["ZR"])
    r = S.decide(arms, obs, test, g, np.random.default_rng(1))
    assert r["response"]["verdict"] == "response not evaluable"
