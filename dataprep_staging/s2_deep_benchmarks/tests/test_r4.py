"""CPU unit tests of the round-4 S2 helpers on synthetic data (no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s2_deep_benchmarks import build_v2 as V  # noqa: E402
from s2_deep_benchmarks import label_confirm as LC  # noqa: E402
from s2_deep_benchmarks import printed_v2 as P  # noqa: E402


def _hourly(days: int, start: str = "2021-01-01", rate: float = 0.1) -> pd.Series:
    idx = pd.date_range(start, periods=24 * days, freq="h")
    return pd.Series(rate / 24.0 * np.arange(len(idx)), index=idx)


def test_monthly_mean_increments_hourly_and_daily_series():
    h = _hourly(120)
    inc = LC.monthly_mean_increments(h)
    assert list(inc.index[:2]) == ["2021-02", "2021-03"]
    # a linear 0.1 mm/day ramp: consecutive monthly means differ by about 0.1 x 30 days
    assert np.allclose(inc.iloc[:2].to_numpy(), [0.1 * (31 + 28) / 2, 0.1 * (28 + 31) / 2],
                       atol=0.06)
    d = h[h.index.hour == 0]                      # one sample a day
    inc_d = LC.monthly_mean_increments(d)
    assert len(inc_d) == len(inc)
    # an hourly series with only 6 valid hours a day yields no increment
    sparse = h[h.index.hour < 6]
    assert LC.monthly_mean_increments(sparse).empty


def _ct(rows):
    return pd.DataFrame(rows, columns=["vendor_key", "sensor_id", "depth_m", "n_months", "r",
                                       "median_abs_mm", "months"])


def test_decide_pair_confirmed_swapped_ambiguous():
    a, b = "NTUH(2)", "NTUH(1)"
    ct = _ct([["K1", a, 70, 20, 0.9, 1, ""], ["K1", b, 300, 20, 0.5, 2, ""],
              ["K2", a, 70, 20, 0.4, 2, ""], ["K2", b, 300, 20, 0.8, 1, ""]])
    out = LC.decide_pair(ct, "K1", "K2", a, b)
    assert out["status"] == "confirmed" and out["decision"] == {"K1": a, "K2": b}
    ct2 = _ct([["K1", a, 70, 20, 0.5, 1, ""], ["K1", b, 300, 20, 0.9, 2, ""],
               ["K2", a, 70, 20, 0.8, 2, ""], ["K2", b, 300, 20, 0.4, 1, ""]])
    out2 = LC.decide_pair(ct2, "K1", "K2", a, b)
    assert out2["status"] == "confirmed" and out2["decision"] == {"K1": b, "K2": a}
    ct3 = _ct([["K1", a, 70, 20, 0.9, 1, ""], ["K1", b, 300, 20, 0.5, 2, ""],
               ["K2", a, 70, 20, 0.8, 2, ""], ["K2", b, 300, 20, 0.4, 1, ""]])
    out3 = LC.decide_pair(ct3, "K1", "K2", a, b)
    assert out3["status"] == "ambiguous" and out3["decision"] == {"K1": None, "K2": None}
    ct4 = _ct([["K1", a, 70, 3, 0.9, 1, ""], ["K1", b, 300, 3, 0.5, 2, ""],
               ["K2", a, 70, 3, 0.4, 2, ""], ["K2", b, 300, 3, 0.8, 1, ""]])
    assert LC.decide_pair(ct4, "K1", "K2", a, b)["status"] == "ambiguous"   # < 5 months


def test_decide_tracking_uses_the_window_table():
    ct = _ct([["T3", "TKJS(3)", 400, 20, 0.7, 1, ""], ["T1", "TKJS(1)", 300, 40, 0.95, 1, ""]])
    win = _ct([["T1", "TKJS(1)", 300, 20, 0.6, 1, ""]])
    assert LC.decide_tracking(ct, "T3", "TKJS(3)", [("T1", "TKJS(1)")])["status"] == \
        "defective_preorigin"
    out = LC.decide_tracking(ct, "T3", "TKJS(3)", [("T1", "TKJS(1)")], ct_window=win)
    assert out["status"] == "tracks" and out["bar"] == 0.6
    neg = _ct([["T3", "TKJS(3)", 400, 20, -0.7, 1, ""]])
    assert LC.decide_tracking(neg, "T3", "TKJS(3)", [("T1", "TKJS(1)")],
                              ct_window=win)["status"] == "defective_preorigin"


def test_roc_months_and_periods():
    assert P.roc_to_month("113/11") == "2024-11"
    assert P.roc_to_month(" 109/4 ") == "2020-04"
    assert P.period_bounds("112/4~113/11") == ("2023-04", "2024-11")


def test_primary_printings_earliest_verified_vintage():
    d = pd.DataFrame({
        "sensor_id": ["S"] * 5, "month": ["2022-01", "2022-01", "2022-02", "2022-02",
                                           "2022-03"],
        "vintage_month": ["2022-02", "2023-02", "2022-03", "2023-03", "2023-04"],
        "flag": [0, 0, 9, 0, 8], "value": [1.0, 1.4, 5.0, 2.0, np.nan],
        "report": ["a", "b", "c", "d", "e"], "sd": [0.03] * 5})
    p = P.primary_printings(d, ["sensor_id", "month"]).set_index("month")
    assert p.loc["2022-01", "report"] == "a" and np.isclose(p.loc["2022-01", "reprint_diff_mm"],
                                                            0.4)
    assert p.loc["2022-02", "report"] == "d" and p.loc["2022-02", "flag"] == 0
    assert np.isnan(p.loc["2022-02", "reprint_diff_mm"])
    assert p.loc["2022-03", "flag"] == 8 and p.loc["2022-03", "n_printings"] == 1
    cov = P.coverage_rounding(p.reset_index())
    assert cov["n_pairs"] == 1 and cov["coverage"] == 0.0 and cov["calibrated"] is False


def test_take_splits_at_the_origin():
    df = pd.DataFrame({"month": ["2022-11", "2022-12", "2023-01"],
                       "date": ["2022-11-30", "2022-12-31", "2023-01-01"]})
    assert list(V._take(df, "month", True).month) == ["2022-11", "2022-12"]
    assert list(V._take(df, "month", False).month) == ["2023-01"]
    assert list(V._take(df, "date", True).date) == ["2022-11-30", "2022-12-31"]
    assert list(V._take(df, "date", False).date) == ["2023-01-01"]


def test_coverage_of_feed_sd():
    u = pd.DataFrame({"diff_mm": [0.1, -0.2, 3.0, 0.0], "sd_feed_mm": [0.1, 0.1, 0.1, 0.1],
                      "sd_round_mm": [0.0, 0.0, 0.0, 0.0]})
    c = V.coverage(u)
    assert c["n"] == 4 and c["coverage_95"] == 0.5 and c["calibrated"] is False
    assert c["sd_scale_for_95_after_the_fact"] > 1.0
