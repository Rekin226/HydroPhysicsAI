"""CPU unit tests of the 007001 table parser (synthetic layout text only)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s2_deep_benchmarks import cy007001 as C  # noqa: E402

FOUR = """
           表 1 彰雲地區深層水準樁 111 年 12 月觀測成果
             設置深       單月變化量(mm)                累積變化量(mm)
號 市                 110/12       111/12   109/4~110/12   110/4~111/12
    彰
1     溪州國小    300    -0.3        -1.0↑       -20.9         -14.3↓
2     土庫國中    400    2.4          2.4           -            -34.3
5   林 中科虎尾    300    0.7         0.1 ↑        -52.2         -5↓
6     中科虎尾     70    0.9           -*         -3.2          -*
註：1.表中數值以正值代表回脹，負值代表壓縮
"""

SIX = """
                      表 2 彰雲地區深層水準樁 112 年 8 月觀測成果
號 市                            111/8    112/8   趨勢      110/4~111/8   111/4~112/8   趨勢
 1            溪州國小     300     0.7       -          -      9.4            -        -
 4          雲 土庫國中     400      2.5     -3.4        ↗      27.0          38.4      ↘
註：1.表中數值以正值代表下陷，負值代表回脹。
"""


def test_parse_four_column_layout():
    t = C.parse_layout(FOUR)
    assert (t["report_roc_year"], t["report_month"]) == (111, 12)
    assert t["positive_means"] == "rebound"
    assert (t["col_month_prev"], t["col_month_curr"]) == ("110/12", "111/12")
    assert t["col_cum_curr"] == "110/4~111/12"
    r = {(x["station"], x["depth_m"]): x for x in t["rows"]}
    assert r[("溪州國小", 300)]["month_curr_mm"] == -1.0
    assert r[("土庫國中", 400)]["cum_prev_mm"] is None
    assert r[("中科虎尾", 300)]["month_curr_mm"] == 0.1 and r[("中科虎尾", 300)]["cum_curr_mm"] == -5
    assert r[("中科虎尾", 70)]["month_curr_mm"] is None


def test_parse_trend_column_layout_and_sign():
    t = C.parse_layout(SIX)
    assert t["positive_means"] == "compression"
    r = {(x["station"], x["depth_m"]): x for x in t["rows"]}
    assert r[("溪州國小", 300)]["month_prev_mm"] == 0.7
    assert r[("溪州國小", 300)]["month_curr_mm"] is None
    assert r[("土庫國中", 400)]["month_curr_mm"] == -3.4 and r[("土庫國中", 400)]["cum_curr_mm"] == 38.4


def test_compare_flags_value_and_sign_disagreement():
    t = C.parse_layout(FOUR)
    a = C.rows_frame(t, "2022-12", "a")
    b = a.copy()
    b.loc[0, "month_curr_mm"] = -1.1
    b["positive_means"] = "rebound"
    m = C.compare(a, b)
    assert int((~m.agree).sum()) == 1
    b2 = a.copy()
    b2["positive_means"] = "compression"
    assert not C.compare(a, b2).agree.any()
    assert C.ym_label_to_month("111/12") == "2022-12"
    assert pd.isna(C.ym_label_to_month(None))
