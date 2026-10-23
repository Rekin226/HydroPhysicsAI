"""CPU unit tests of the S2 test-C helpers (synthetic tables only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s2_deep_benchmarks import c_test as T  # noqa: E402


def _daily(rate=0.1, feed="vendor", sid="X", start="2021-01-01", end="2021-04-30"):
    d = pd.date_range(start, end, freq="D")
    v = rate * np.arange(len(d))
    return pd.DataFrame({"sensor_id": sid, "feed": feed, "date": d.strftime("%Y-%m-%d"),
                         "month": d.strftime("%Y-%m"), "value": v, "flag": 0})


def _monthly(sid="X", months=("2021-01", "2021-02", "2021-03", "2021-04"), vflag=0):
    return pd.DataFrame({"sensor_id": sid, "month": list(months), "raw_flag": 8,
                         "vendor_flag": vflag})


def test_monthly_mean_increment_of_a_linear_trend():
    daily = _daily(rate=0.1)
    inc = T.monthly_mean_increments(daily, _monthly()).set_index("month")
    # Feb mean - Jan mean of a 0.1 mm/day trend = 0.1 * (days between month centres)
    jan = 0.1 * np.arange(31).mean()
    feb = 0.1 * (31 + np.arange(28)).mean()
    assert np.isclose(inc.at["2021-02", "inc_mm"], feb - jan)
    assert inc.at["2021-02", "feed"] == "vendor" and inc.at["2021-02", "flag"] == 0
    assert inc.at["2021-01", "flag"] == 8          # no previous month


def test_raw_preferred_and_suspect_flag():
    daily = pd.concat([_daily(rate=0.1, feed="vendor"), _daily(rate=0.2, feed="raw")])
    m = _monthly()
    m["raw_flag"] = 0
    m.loc[m.month == "2021-03", "raw_flag"] = 9
    inc = T.monthly_mean_increments(daily, m).set_index("month")
    assert inc.at["2021-02", "feed"] == "raw"
    assert inc.at["2021-03", "flag"] == 9 and inc.at["2021-04", "flag"] == 9


def test_short_months_untestable_and_score():
    daily = _daily(rate=0.1)
    daily = daily[~((daily.month == "2021-03") & (daily.date > "2021-03-10"))]
    inc = T.monthly_mean_increments(daily, _monthly()).set_index("month")
    assert inc.at["2021-03", "flag"] == 8 and inc.at["2021-04", "flag"] == 8
    units = pd.DataFrame({"sensor_id": "X", "month": ["2021-02"], "value": [3.0],
                          "printed_mm": [-3.0], "column": "report_month", "report": "r"})
    s = T.score(units, inc.reset_index())["summary"]
    assert s["n_testable"] == 1 and s["tolerance_pass"]
