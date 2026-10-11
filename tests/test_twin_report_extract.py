from __future__ import annotations

import numpy as np
import pandas as pd

from hydrophysics.twin.report_extract import (
    chain_check,
    compare_transcriptions,
    feed_check,
    feed_monthly_cm,
    rows_from,
)


def _table(month_vals, year_vals, names=("溪州國小", "土庫國中")):
    return {"report_roc_year": 114, "report_month": 12,
            "piles": [{"row": i + 1, "county": "彰化", "township": "溪州鄉", "pile_name": n,
                       "pile_depth_m": 300, "month_change_cm": m, "year_change_cm": y,
                       "month_symbol": "▼", "year_symbol": "▼"}
                      for i, (n, m, y) in enumerate(zip(names, month_vals, year_vals, strict=True))]}


def test_rows_from_applies_the_alias_and_keeps_nulls():
    df = rows_from(_table([0.2, None], [0.9, 1.6], names=("溪州", "土庫國中")), "2025-12", "pdf")
    assert list(df.pile) == ["溪州國小", "土庫國中"]
    assert pd.isna(df.month_cm.iloc[1]) and df.year_cm.iloc[1] == 1.6


def test_compare_transcriptions_flags_any_difference():
    a = rows_from(_table([0.2, 0.9], [0.9, 3.3]), "2024-01", "pdf")
    b = rows_from(_table([0.2, 0.8], [0.9, 3.3]), "2024-01", "text")
    m = compare_transcriptions(a, b).set_index("pile")
    assert bool(m.loc["溪州國小", "agree"]) and not bool(m.loc["土庫國中", "agree"])
    c = rows_from(_table([0.2], [0.9], names=("溪州國小",)), "2024-01", "text")
    m2 = compare_transcriptions(a, c).set_index("pile")
    assert not bool(m2.loc["土庫國中", "agree"])          # missing in one transcription


def test_chain_check_uses_the_trailing_twelve_identity():
    months = pd.period_range("2024-01", "2025-02", freq="M")
    m = np.full(len(months), 0.3)
    m[12] = 0.8                                          # Jan 2025 compresses more than Jan 2024
    y = np.cumsum(m)                                     # a true trailing sum from a zero past
    y[12:] = y[12:] - np.cumsum(m)[:len(months) - 12]    # drop the month that left the window
    df = pd.DataFrame({"month": [str(p) for p in months], "pile": "土庫國中", "depth_m": 400,
                       "month_cm": m, "year_cm": np.round(y, 1)})
    out = chain_check(df)
    assert out.chain_ok.iloc[:12].isna().all()           # no m-12 yet
    assert bool(out.chain_ok.iloc[12]) and bool(out.chain_ok.iloc[13])
    bad = df.copy()
    bad.loc[13, "year_cm"] += 1.0                        # a transcription slip of 1 cm
    assert not chain_check(bad).chain_ok.iloc[13]
    other = df.copy()
    other.loc[13, "depth_m"] = 300                       # the report switched sensor that month
    assert chain_check(other).chain_ok.iloc[13] is None  # no same-sensor month to chain on


def test_feed_monthly_cm_is_month_end_difference_in_cm():
    idx = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h")
    v = pd.Series(np.linspace(0.0, 90.0, len(idx)), index=idx)   # mm, 1 mm/day
    tab = feed_monthly_cm(v).set_index("month")
    assert np.isclose(tab.loc["2024-02", "feed_cm"], 2.9, atol=0.05)   # 29 days -> 2.9 cm
    assert bool(tab.loc["2024-02", "feed_complete"]) and not bool(tab.loc["2024-01", "feed_complete"])
    v2 = v.drop(v["2024-02-29"].index)                                  # last day missing
    assert not bool(feed_monthly_cm(v2).set_index("month").loc["2024-02", "feed_complete"])


def test_feed_check_marks_only_complete_months():
    idx = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h")
    v = pd.Series(np.linspace(0.0, 90.0, len(idx)), index=idx)
    df = pd.DataFrame({"month": ["2024-02", "2024-03", "2024-04"], "pile": "溪州國小",
                       "month_cm": [2.9, 3.5, 3.1], "year_cm": [1, 1, 1]})
    out = feed_check(df, {"溪州國小": v})
    assert bool(out.feed_ok.iloc[0]) and out.feed_ok.iloc[1] is False or not out.feed_ok.iloc[1]
    assert pd.isna(out.feed_cm.iloc[2]) and out.feed_ok.iloc[2] is None
