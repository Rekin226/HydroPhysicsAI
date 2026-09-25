"""The policy-response gate and the first-class temporal verdict (G3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.calibrate_flow import temporal_verdict  # noqa: E402
from hydrophysics.twin.policy_gate import policy_response, policy_verdict  # noqa: E402


def test_monotone_sizeable_response_passes():
    v = policy_verdict({0.85: 0.4, 0.7: 0.9}, {0.85: -0.003, 0.7: -0.008}, s_base=0.10)
    assert v["verdict"] == "PASS" and v["rel_ds"] == pytest.approx(0.08)
    assert all(v["checks"].values())


@pytest.mark.parametrize("dh2, ds, s_base, failing", [
    ({0.85: 0.0, 0.7: 0.0}, {0.85: 0.0, 0.7: 0.0}, 0.1, "heads_recover_monotonically"),
    ({0.85: 1.0, 0.7: 0.6}, {0.85: -0.003, 0.7: -0.008}, 0.1, "heads_recover_monotonically"),
    ({0.85: 0.4, 0.7: 0.9}, {0.85: -0.009, 0.7: -0.004}, 0.1,
     "subsidence_falls_monotonically"),
    ({0.85: 0.1, 0.7: 0.3}, {0.85: -0.003, 0.7: -0.008}, 0.1, "head_response_large_enough"),
    ({0.85: 0.4, 0.7: 0.9}, {0.85: -0.0005, 0.7: -0.001}, 0.1,
     "subsidence_response_large_enough"),
])
def test_flat_reversed_or_tiny_responses_fail(dh2, ds, s_base, failing):
    v = policy_verdict(dh2, ds, s_base)
    assert v["verdict"] == "FAIL" and not v["checks"][failing]


def test_the_21km_model_fails_on_relative_subsidence():
    """STATE 2026-09-20: -0.18 cm on a 16.4 cm baseline (1.1 %)."""
    v = policy_verdict({0.85: 0.3, 0.7: 0.6}, {0.85: -0.0009, 0.7: -0.0018}, 0.164)
    assert v["verdict"] == "FAIL" and v["rel_ds"] == pytest.approx(0.011, abs=1e-3)


def test_optional_absolute_floor_separates_the_free_running_pair():
    """Free-running, measured 2026-09-23: 10 km -1.25 cm on 4.78; 21 km -0.44 on 3.92."""
    ten = policy_verdict({0.85: 0.46, 0.7: 0.91}, {0.85: -0.0063, 0.7: -0.0125}, 0.0478)
    twenty = policy_verdict({0.85: 0.50, 0.7: 1.00}, {0.85: -0.0024, 0.7: -0.0044}, 0.0392)
    assert ten["verdict"] == twenty["verdict"] == "PASS"      # the pre-registered rule
    ten = policy_verdict({0.85: 0.46, 0.7: 0.91}, {0.85: -0.0063, 0.7: -0.0125}, 0.0478,
                         min_abs_ds=0.005)
    twenty = policy_verdict({0.85: 0.50, 0.7: 1.00}, {0.85: -0.0024, 0.7: -0.0044}, 0.0392,
                            min_abs_ds=0.005)
    assert ten["verdict"] == "PASS" and twenty["verdict"] == "FAIL"


def _arrays(W=5, T=48, T_fit=36, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    obs = 10 * rng.random((W, 1)) + np.sin(2 * np.pi * t / 12)[None] + 0.02 * t[None]
    months = (t + 1) % 12
    clim = np.zeros_like(obs)
    for m in range(12):
        clim[:, months == m] = obs[:, :T_fit][:, months[:T_fit] == m].mean(axis=1,
                                                                          keepdims=True)
    return obs, clim, T_fit


def test_temporal_verdict_passes_a_good_continuation_and_fails_a_drifting_one():
    obs, clim, T_fit = _arrays()
    good = obs + 0.01
    v = temporal_verdict(good, obs, clim, T_fit, k=1.5)
    assert v["verdict"] == "PASS" and v["rmse_ratio"] < 0.1
    drift = obs.copy()
    drift[:, T_fit:] += np.linspace(0, 8, obs.shape[1] - T_fit)[None]
    v = temporal_verdict(drift, obs, clim, T_fit, k=1.5)
    assert v["verdict"] == "FAIL" and v["rmse_model_m"] > v["rmse_clim_m"]


def test_temporal_verdict_is_strict_at_the_ratio_boundary():
    obs, _, T_fit = _arrays()
    clim = obs.copy()
    clim[:, T_fit:] += 1.0                      # climatology: 1 m off, shape perfect
    at = obs + 2.0                              # model: exactly 2x climatology's RMSE
    v = temporal_verdict(at, obs, clim, T_fit, k=2.0)
    assert v["rmse_ratio"] == pytest.approx(2.0)
    assert v["verdict"] == "FAIL"               # strict: equal is not below
    # just inside the boundary, and with a shape that genuinely beats climatology's
    inside = obs + 1.999                        # a constant offset keeps the shape
    shape_bad_clim = clim.copy()
    shape_bad_clim[:, T_fit:] += np.linspace(0, 0.1, obs.shape[1] - T_fit)[None]
    v = temporal_verdict(inside, obs, shape_bad_clim, T_fit, k=2.0)
    assert v["rmse_ratio"] < 2.0 and v["r2_shape_model"] > v["r2_shape_clim"]
    assert v["verdict"] == "PASS"


def test_policy_response_runs_end_to_end_on_synthetic_inputs(tmp_path):
    from hydrophysics.twin.forward import load_members
    from tests.test_twin_forward import _inputs, _theta_file

    inp = _inputs(T=24)
    p, _ = _theta_file(tmp_path)
    vep = tmp_path / "vep.json"
    vep.write_text(json.dumps({"log_ske": float(np.log(1e-4)), "log_skv": float(np.log(1e-3)),
                               "log_tau": float(np.log(300.0)), "h_pc0": 0.0}))
    out = policy_response(inp, load_members([p])[0], str(vep), horizon=6, device="cpu",
                          log=lambda *_: None)
    assert set(out["dh2_m"]) == {"0.85", "0.7"} and out["column"] == "candidate"
    # cutting irrigation cannot lower heads relative to the baseline
    assert out["dh2_m"]["0.7"] >= out["dh2_m"]["0.85"] >= -1e-9
    assert out["verdict"] in ("PASS", "FAIL")
