"""CPU tests for the held-out-years error decomposition (hydrophysics.twin.drift_diag)."""
from __future__ import annotations

import numpy as np

from hydrophysics.twin.drift_diag import (
    COMPONENTS,
    decompose_series,
    decompose_wells,
    pooled_shares,
)


def test_components_sum_to_mse_and_recover_known_parts():
    t = np.arange(95, 131)                       # 36 held-out months, absolute index
    rng = np.random.default_rng(0)
    e = (2.0 + 0.5 * (t - t.mean()) / 12.0
         + 0.8 * np.cos(2 * np.pi * t / 12 + 0.3) + 0.05 * rng.standard_normal(t.size))
    d = decompose_series(e, t)
    assert np.isclose(sum(d[c] for c in COMPONENTS), d["mse"])
    assert np.isclose(d["offset_m"], e.mean())
    assert abs(d["trend_m_per_yr"] - 0.5) < 0.02
    assert abs(d["seas_amp_m"] - 0.8) < 0.03
    assert d["offset"] > d["seasonal"] > d["trend"] > d["residual"]


def test_pure_offset_is_all_offset_and_nan_months_are_skipped():
    e = np.full(36, -3.0)
    e[[4, 17]] = np.nan
    d = decompose_series(e)
    assert d["n"] == 34
    assert np.isclose(d["offset"] / d["mse"], 1.0)
    assert d["residual"] < 1e-12


def test_pooled_shares_match_gate_rmse():
    rng = np.random.default_rng(1)
    err = rng.standard_normal((5, 36)) + np.arange(5)[:, None]
    parts = decompose_wells(err)
    pooled = pooled_shares(parts)
    assert np.isclose(pooled["rmse"], np.sqrt(np.mean(err ** 2)))
    assert np.isclose(sum(pooled[f"{c}_frac"] for c in COMPONENTS), 1.0)
    sub = pooled_shares(parts, mask=np.array([0, 0, 0, 0, 1], bool))
    assert sub["n_wells"] == 1 and sub["offset_frac"] > 0.9


def test_short_series_returns_nan():
    d = decompose_series(np.ones(4))
    assert np.isnan(d["mse"]) and d["n"] == 4
