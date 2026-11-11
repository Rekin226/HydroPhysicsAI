"""Round-12 exploratory subsidence layer: unit checks on synthetic inputs (no agency data)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from dataprep_staging.r12_leveling.backtest import trailing  # noqa: E402
from dataprep_staging.r12_leveling.increments import hydro_year, increments  # noqa: E402
from dataprep_staging.r12_leveling.layer import kernel_rate  # noqa: E402
from dataprep_staging.r12_leveling.panel import random_folds, spatial_blocks  # noqa: E402


def test_increments_sign_and_artefact_screen():
    elev = pd.DataFrame({2019: [10.00, 5.0], 2020: [9.97, 5.0], 2021: [9.95, 4.2]},
                        index=["a", "b"])
    d = increments(elev)
    assert d.loc["a", 2020] == pytest.approx(0.03)       # a drop is positive sinking
    assert d.loc["a", 2021] == pytest.approx(0.02)
    assert np.isnan(d.loc["b", 2021])                    # 0.8 m step is not ground motion
    assert np.isnan(d.loc["a", 2019])


def test_hydro_year_matches_spring_survey():
    dates = pd.DatetimeIndex(["2020-05-01", "2020-06-01", "2021-05-01"])
    assert hydro_year(dates).tolist() == [2020, 2021, 2021]


def test_trailing_needs_enough_years_and_clips_window_start():
    D = np.array([[0.01, 0.03, np.nan, 0.02], [np.nan, np.nan, np.nan, 0.02]])
    r = trailing(D, 3, 3, 2)
    assert r[0] == pytest.approx(0.025)
    assert np.isnan(r[1])
    assert trailing(D, 1, 5, 2)[0] == pytest.approx(0.02)   # window starts at column 0


def test_kernel_rate_reduces_noise_and_keeps_gradient():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 30_000, size=(600, 2))
    truth = 0.01 + 0.03 * xy[:, 0] / 30_000
    noisy = truth + rng.normal(0, 0.01, size=600)
    sm = kernel_rate(noisy, xy, None, 3.0, 1.0)
    assert np.sqrt(np.nanmean((sm - truth) ** 2)) < 0.5 * np.sqrt(np.mean((noisy - truth) ** 2))
    assert np.corrcoef(sm[np.isfinite(sm)], truth[np.isfinite(sm)])[0, 1] > 0.95


def test_kernel_rate_self_weight_zero_is_leave_one_out():
    xy = np.array([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]] + [[50_000.0 + 1000 * i, 0.0]
                                                                  for i in range(10)])
    r = np.zeros(len(xy))
    r[0] = 1.0
    out = kernel_rate(r, xy, None, 3.0, 0.0)
    assert out[0] == pytest.approx(0.0, abs=1e-9)        # its own value carries no weight


def test_spatial_blocks_keep_a_block_in_one_fold():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 50_000, size=(500, 2))
    f = spatial_blocks(xy, size_m=10_000.0, n_folds=5, seed=3)
    key = np.floor(xy[:, 0] / 10_000).astype(int) * 1000 + np.floor(xy[:, 1] / 10_000).astype(int)
    assert all(len(set(f[key == k])) == 1 for k in np.unique(key))
    assert sorted(np.bincount(random_folds(11, 5, 0)).tolist()) == [2, 2, 2, 2, 3]
