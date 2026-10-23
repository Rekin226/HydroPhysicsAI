from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from hydrophysics.twin.deep_pile_holdout import (  # noqa: E402
    printed_year_check,
    reconstruct_cumulative,
    score,
)


def _piles(months, vals, pile="土庫國中", verified=None):
    verified = [True] * len(months) if verified is None else verified
    return pd.DataFrame({"month": months, "pile": pile, "depth_m": 400, "month_cm": vals,
                         "year_cm": np.nan, "verified": verified})


def test_reconstruct_sums_monthly_values_and_leaves_an_unbridged_gap_undetermined():
    months = [str(p) for p in pd.period_range("2024-01", "2024-06", freq="M")]
    df = _piles(months, [0.5, 0.5, -0.2, 0.3, 0.1, 0.4], verified=[True, True, True, False, True, True])
    cum, resid = reconstruct_cumulative(df, "2023-12", "2024-06")
    P = lambda m: pd.Period(m, freq="M")  # noqa: E731
    assert np.isclose(cum.at[P("2023-12"), "土庫國中"], 0.0)
    assert np.isclose(cum.at[P("2024-03"), "土庫國中"], 0.8)
    assert cum.loc[P("2024-04"):, "土庫國中"].isna().all()           # nothing bridges the gap
    assert len(resid) == 5 and np.allclose(resid.resid_cm.dropna(), 0) and resid.resid_cm.isna().sum() == 2


def test_reconstruct_bridges_a_gap_with_the_printed_trailing_year():
    months = [str(p) for p in pd.period_range("2024-01", "2025-03", freq="M")]
    m = [0.2] * 15
    df = _piles(months, m)
    df.loc[df.month == "2024-12", ["month_cm", "year_cm"]] = [np.nan, np.nan]   # printed as —
    df.loc[df.month == "2025-01", "year_cm"] = 2.4                                # 2024-01-31 -> 2025-01-31
    cum, resid = reconstruct_cumulative(df, "2023-12", "2025-03")
    P = lambda m: pd.Period(m, freq="M")  # noqa: E731
    assert np.isclose(cum.at[P("2025-01"), "土庫國中"], 2.6, atol=1e-6)     # 0.2 + 2.4
    assert np.isclose(cum.at[P("2024-12"), "土庫國中"], 2.4, atol=1e-6)     # back-filled from both sides
    assert np.isclose(cum.at[P("2025-03"), "土庫國中"], 3.0, atol=1e-6)
    other = df.copy()
    other.loc[other.month == "2024-01", "depth_m"] = 300                       # a different sensor then
    cum2, _ = reconstruct_cumulative(other, "2023-12", "2025-03")
    assert cum2.loc[P("2024-12"):, "土庫國中"].isna().all()                    # the year value is not used


def test_printed_year_check_compares_sum_with_printed_value():
    months = [str(p) for p in pd.period_range("2024-01", "2024-12", freq="M")]
    df = _piles(months, [0.3] * 12)
    df.loc[df.month == "2024-12", "year_cm"] = 3.5
    cum, _ = reconstruct_cumulative(df, "2023-12", "2024-12")
    chk = printed_year_check(df, cum, "2024-12")
    assert np.isclose(chk["土庫國中"]["summed_cm"], 3.6) and np.isclose(chk["土庫國中"]["diff_cm"], 0.1)


def test_score_uses_mean_members_and_baselines():
    months = [str(p) for p in pd.period_range("2024-01", "2026-01", freq="M")]
    rates = {"A": 0.2, "B": 0.1, "C": 0.05}
    df = pd.concat([_piles(months, [v] * 25, k) for k, v in rates.items()])
    cum, _ = reconstruct_cumulative(df, "2023-12", "2026-01")
    sites = pd.DataFrame({"pile": list(rates), "depth_m": [400, 300, 300]})
    dates = pd.date_range("2012-01-01", "2032-12-01", freq="MS") + pd.offsets.MonthEnd(0)
    t = np.arange(len(dates)) / 12.0
    slopes = [0.03, 0.012, 0.006]                                # 3.0, 1.2, 0.6 cm/yr in metres
    mean = np.stack([sl * t for sl in slopes])
    years = list(range(2012, 2033))
    members = np.stack([np.stack([sl * (np.arange(21) + 1) * (1 + 0.1 * j) for j in range(-2, 3)])
                        for sl in slopes], 1)
    model = {"cells": np.array([5, 7, 9]), "dates": dates, "mean": mean, "members_yr": members,
             "years": years, "rheology": "test"}
    tab, sc = score(cum, sites, model, {"A": 2.4, "B": 1.2, "C": 0.6}, "2023-12", "2025-12",
                    "2025-05", "2026-01")
    tab = tab.set_index("pile")
    assert np.isclose(tab.at["A", "obs_cum_cm"], 4.8) and np.isclose(tab.at["B", "obs_cum_cm"], 2.4)
    assert np.isclose(tab.at["A", "model_cum_cm"], 6.0, atol=0.01)
    assert np.isclose(tab.at["B", "model_cum_cm"], 2.4, atol=0.01)
    assert np.isclose(tab.at["A", "base_continue_cum_cm"], 4.8)
    assert tab.at["B", "model_cum_p10_cm"] < 2.4 < tab.at["B", "model_cum_p90_cm"]
    assert sc["n_sites"] == 3 and sc["cum"]["base_continue"]["rmse"] < sc["cum"]["model"]["rmse"]
    assert np.isclose(tab.at["A", "obs_sub_cm"], 1.6) and sc["after_feeds"]["n_sites"] == 3
