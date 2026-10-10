from __future__ import annotations

import numpy as np
import pytest

from hydrophysics.twin.coverage import Z, _residual_sd, score_coverage


def _rows(seed=0, W=20, T=40, scale=1.0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(W, T))
    sigma = np.full((W, T), scale)
    valid = np.ones((W, T), dtype=bool)
    valid[:, :5] = False
    keep = np.ones(W, dtype=bool)
    return obs, sigma, valid, keep


def test_calibrated_gaussian_errors_cover_their_nominal_level():
    obs, sigma, valid, keep = _rows(W=200, T=400)
    pred = np.zeros_like(obs)                      # errors are N(0, 1) with sigma = 1
    out = score_coverage(obs, pred, sigma, valid, keep)
    for level in (50, 80, 90, 95):
        assert abs(out[f"picp_{level}"] - level / 100) < 0.02
        assert abs(out[f"picp_pooled_{level}"] - level / 100) < 0.02
    assert abs(out["inflation_90"] - 1.0) < 0.05
    assert out["mean_width_90_m"] == pytest.approx(2 * Z[90])


def test_underdispersed_spread_needs_inflation_and_covers_less():
    obs, sigma, valid, keep = _rows(W=200, T=400, scale=0.5)
    out = score_coverage(obs, np.zeros_like(obs), sigma, valid, keep)
    assert out["picp_90"] < 0.75
    assert abs(out["inflation_90"] - 2.0) < 0.15


def test_equal_well_weighting_differs_from_pooled():
    obs, sigma, valid, keep = _rows(W=2, T=10)
    obs[:] = 0.0
    pred = np.zeros_like(obs)
    pred[0, :] = 10.0                              # well 0 never covered
    valid[1, 5:8] = False                          # well 1 has fewer rows
    out = score_coverage(obs, pred, sigma, valid, keep)
    assert out["picp_90"] == pytest.approx(0.5)
    assert out["picp_pooled_90"] == pytest.approx(2 / 7)


def test_invalid_sigma_and_shapes_are_rejected():
    obs, sigma, valid, keep = _rows()
    with pytest.raises(ValueError):
        score_coverage(obs, obs, sigma[:, :-1], valid, keep)
    sigma[valid][0] = 0.0
    bad = sigma.copy()
    bad[valid] = 0.0
    with pytest.raises(ValueError):
        score_coverage(obs, obs, bad, valid, keep)


def test_residual_sd_uses_fitted_months_only():
    hist = np.zeros((2, 30))
    hist[0, :20] = np.arange(20) % 2              # sd 0.51 on the fitted months
    hist[0, 20:] = 100.0                          # held-out values must not matter
    hist[1, :20] = np.nan                         # no support
    sd = _residual_sd(hist, np.zeros_like(hist), fit_months=20)
    assert sd[0] == pytest.approx(np.std(np.arange(20) % 2, ddof=1))
    assert np.isnan(sd[1])
