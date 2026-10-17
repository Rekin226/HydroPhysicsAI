from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.mechanism import band_compaction, ring_hiatuses
from hydrophysics.twin.ring_holdout import (
    continued_rate,
    r2_rmse,
    score_sites,
    window_span,
    window_stats,
)


def _series(rate_per_yr=0.02, start="2015-01-31", months=130, offset=1.0):
    idx = pd.date_range(start, periods=months, freq="ME")
    years = (idx - idx[0]).days / 365.25
    return pd.Series(offset + rate_per_yr * years, index=idx)


def test_window_stats_measures_displacement_from_reference_and_slope():
    s = _series(0.02)
    cum, rate, n = window_stats(s, "2022-12-01", "2023-01-01", "2025-03-31")
    assert n == 27
    assert rate == pytest.approx(0.02, rel=1e-6)
    assert cum == pytest.approx(0.02 * (pd.Timestamp("2025-03-31") - pd.Timestamp("2022-12-31")).days / 365.25, rel=1e-3)


def test_window_stats_requires_reference_and_support():
    s = _series(0.02, start="2023-06-30", months=24)       # no Dec-2022 reference
    cum, rate, n = window_stats(s, "2022-12-01", "2023-01-01", "2025-03-31")
    assert np.isnan(cum) and np.isnan(rate)
    s = _series(0.02)
    cum, rate, n = window_stats(s, "2022-12-01", "2023-01-01", "2023-06-30")
    assert n == 6 and np.isnan(cum)                        # below min_months
    late = _series(0.02, start="2023-06-30", months=24)   # the age limit does not score it
    assert window_span(late, "2022-12-31", "2023-01-01", "2025-03-31", 62) is None
    assert np.isnan(window_stats(late, "2022-12-31", "2023-01-01", "2025-03-31",
                                 max_ref_age_days=62)[1])


def test_continued_rate_uses_only_the_prior_window():
    s = _series(0.02)
    s.loc["2023-01-01":] = 100.0                           # the future must not matter
    assert continued_rate(s, "2017-01-01", "2022-12-31") == pytest.approx(0.02, rel=1e-6)
    assert np.isnan(continued_rate(s.iloc[:10], "2015-01-01", "2015-12-31"))


def test_r2_rmse_scores_and_handles_small_samples():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_rmse(obs, obs)["r2"] == pytest.approx(1.0)
    m = r2_rmse(np.full(4, obs.mean()), obs)
    assert m["r2"] == pytest.approx(0.0) and m["bias"] == pytest.approx(0.0)
    assert np.isnan(r2_rmse(obs[:2], obs[:2])["r2"])
    worse = r2_rmse(obs + 5.0, obs)
    assert worse["r2"] < 0 and worse["bias"] == pytest.approx(5.0)


def _ring_site(rate=0.02, offset=0.5, scale_step=0.04, start="2015-01-15", end="2025-03-15"):
    """Monthly ring positions (m) of one site compacting ``rate`` m/yr over its interval,
    with no survey from 2022-01 to 2023-01 and, after the gap, every ring ``offset`` m
    deeper (new depth reference) and the deepest one ``scale_step`` m further down."""
    d = pd.date_range(start, end, freq="MS") + pd.Timedelta(days=14)
    d = d[(d < pd.Timestamp("2022-01-01")) | (d >= pd.Timestamp("2023-02-01"))]
    depth = np.array([2.0, 100.0, 200.0, 300.0])
    t = ((d - d[0]).days / 365.25).to_numpy()
    share = (depth - depth[0]) / (depth[-1] - depth[0])       # the top ring stays put
    pos = depth[None, :] - rate * share[None, :] * t[:, None]
    after = d >= pd.Timestamp("2023-02-01")
    pos[after] += offset
    pos[after, -1] += scale_step
    return pd.DataFrame(pos, index=d, columns=[f"NO{i + 1}" for i in range(len(depth))])


def test_ring_hiatus_flags_the_reference_change_and_the_scale_step():
    pos = _ring_site()
    tab = ring_hiatuses(pos)
    assert len(tab) == 1
    h = tab.iloc[0]
    assert (h.before.strftime("%Y-%m"), h.after.strftime("%Y-%m")) == ("2021-12", "2023-02")
    assert h.discontinuity and h.top_shift_m == pytest.approx(0.5, abs=1e-3)
    assert h.common_offset_m == pytest.approx(0.5, abs=0.01)
    assert h.expected_m == pytest.approx(0.02 * h.days / 365.25, rel=1e-6)
    assert h.unexplained_m == pytest.approx(-0.04, abs=1e-6)     # the interval lengthened
    # a gap with the depth reference kept (top ring fixed) is a hiatus, not a discontinuity
    clean = ring_hiatuses(_ring_site(offset=0.0, scale_step=0.0))
    assert len(clean) == 1 and not clean.iloc[0].discontinuity
    assert clean.iloc[0].unexplained_m == pytest.approx(0.0, abs=1e-9)


def test_window_stats_does_not_bridge_the_hiatus():
    total = band_compaction(_ring_site())["total"]
    # legacy rule: the reference is the December 2021 reading, the step enters the cumulative
    legacy, rate, n = window_stats(total, "2022-12-31", "2023-01-01", "2025-03-31")
    old = window_span(total, "2022-12-31", "2023-01-01", "2025-03-31")
    assert old[0].strftime("%Y-%m") == "2021-12" and n == 26
    assert legacy == pytest.approx(0.02 * (old[1] - old[0]).days / 365.25 - 0.04, abs=2e-4)
    span = window_span(total, "2022-12-31", "2023-01-01", "2025-03-31", max_ref_age_days=62)
    assert (span[0].strftime("%Y-%m"), span[1].strftime("%Y-%m")) == ("2023-02", "2025-03")
    cum, rate2, n2 = window_stats(total, "2022-12-31", "2023-01-01", "2025-03-31",
                                  max_ref_age_days=62)
    assert cum == pytest.approx(0.02 * (span[1] - span[0]).days / 365.25, abs=2e-4)
    assert rate2 == rate == pytest.approx(0.02, rel=1e-3) and n2 == n   # rates unchanged
    # a recent reference is still used as before
    s = _series(0.02)
    assert window_span(s, "2022-12-31", "2023-01-01", "2025-03-31", 62)[0] == pd.Timestamp(
        "2022-12-31")


def test_score_sites_differences_model_and_baselines_over_the_observed_span():
    rates = (0.01, 0.02, 0.04)
    obs = {f"s{i}": band_compaction(_ring_site(rate=r, offset=-0.15 if i else 0.5))["total"]
           for i, r in enumerate(rates)}
    xy = pd.DataFrame({"site": list(obs), "x": 0.0, "y": 0.0})
    dates = pd.date_range("2015-01-31", "2025-03-31", freq="ME")
    t = ((dates - dates[0]).days / 365.25).to_numpy()
    truth = np.array([r * t for r in rates])                     # the step-free compaction
    sims = {"climatology": np.stack([0.9 * truth, 1.1 * truth])}  # members bracket truth
    tab, scores = score_sites(obs, xy, np.arange(3), sims, dates, truth, "2022-12-31",
                              "2023-01-01", "2025-03-31", "2017-01-01", "2021-12-31")
    assert (tab.ref_month == "2023-02").all() and (tab.end_month == "2025-03").all()
    assert np.allclose(tab.climatology_cum_cm, tab.obs_cum_cm, atol=0.05)
    assert np.allclose(tab.archive_cum_cm, tab.obs_cum_cm, atol=0.05)
    assert np.allclose(tab.base_continue_cum_cm, tab.obs_cum_cm, atol=0.05)
    assert scores["cum"]["climatology"]["rmse"] < 0.05
    assert scores["cum"]["climatology"]["r2"] > 0.99
    # each coverage against its own member band (the observed rates of 1-4 cm/yr lie
    # outside the 1.9-9.2 cm cumulative bands they were once compared with)
    assert scores["cum"]["climatology"]["p10_p90_coverage"] == 1.0
    assert scores["rate"]["climatology"]["p10_p90_coverage"] == 1.0
    # the legacy reference (no age limit) puts the instrument step into the observations
    tab0, scores0 = score_sites(obs, xy, np.arange(3), sims, dates, truth, "2022-12-31",
                                "2023-01-01", "2025-03-31", "2017-01-01", "2021-12-31",
                                max_ref_age_days=None)
    assert (tab0.ref_month == "2021-12").all()
    assert scores0["cum"]["climatology"]["rmse"] > 3.0
