"""CPU unit tests of the S2 library on synthetic series (no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s2_deep_benchmarks import s2lib as L  # noqa: E402


def _table(q_scale: float = 0.35) -> pd.DataFrame:
    rows = []
    for d in L.DURS_H:
        for est in ("rate", "glob"):
            q = q_scale * np.sqrt(d) * (1.0 if est == "rate" else 1.2)
            rows.append({"d_h": d, "estimator": est, "n": 1000, "q999_mm": q,
                         "sigma_fill_mm": q / 4.0, "median_mm": 0.0})
    return pd.DataFrame(rows)


def _series(n_hours: int = 24 * 120, rate_per_day: float = 0.2, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    x = rate_per_day / 24.0 * t + rng.normal(0, 0.03, n_hours)
    return pd.Series(x, index=pd.date_range("2020-01-01", periods=n_hours, freq="h"))


def test_detect_step_reset_spike_and_real_outage():
    h = _series()
    x = h.to_numpy().copy()
    x[24 * 20:] += 10.0                       # sensor step in continuous recording
    x[24 * 40:24 * 45] = np.nan               # 5-day outage, real movement only
    x[24 * 70:24 * 72] = np.nan               # 2-day outage with a 20 mm reset
    x[24 * 72:] -= 20.0
    x[24 * 90] += 5.0                         # one-hour spike
    h2 = pd.Series(x, index=h.index)
    det = L.Detector(_table())
    ev, xc, fl = L.detect(h2, det)
    kinds = ev.set_index("t_after")["kind"]
    assert kinds[h.index[24 * 20]] == "step"
    assert kinds[h.index[24 * 45]] == "outage"
    assert kinds[h.index[24 * 72]] == "step"
    assert (ev["kind"] == "spike").sum() == 1
    assert np.isnan(xc.iloc[24 * 90]) and fl.iloc[24 * 90] == 9
    # corrected series follows the trend within noise everywhere it is defined
    trend = 0.2 / 24.0 * np.arange(len(x))
    resid = (xc.to_numpy() - trend)
    ok = np.isfinite(resid)
    assert np.nanmax(np.abs(resid[ok] - np.median(resid[ok]))) < 0.6
    # flags: 0 before the first step, 1 after it
    assert fl.iloc[24 * 10] == 0 and fl.iloc[24 * 30] == 1


def test_month_end_convention_and_flags():
    days = pd.date_range("2021-01-01", "2021-04-30", freq="D")
    v = np.arange(len(days), dtype=float)       # +1 mm per day
    daily = pd.DataFrame({"date": days, "value": v, "n_hours": 24, "flag": 0})
    daily.loc[daily.date == "2021-03-31", "flag"] = 8
    daily.loc[daily.date == "2021-03-31", "value"] = np.nan
    det = L.Detector(_table())
    m = L.monthly_from_daily(daily, pd.DataFrame(columns=["kind"]), det, sigma_day=0.05)
    m = m.set_index("month")
    assert np.isclose(m.at["2021-02", "inc_cm"], 2.8)          # 28 days
    assert m.at["2021-02", "flag"] == 0
    assert m.at["2021-03", "flag"] == 2                       # bridged from 03-30
    assert np.isclose(m.at["2021-03", "inc_cm"], 3.0)
    assert m.at["2021-01", "flag"] == 8                       # no previous month end


def test_cumulative_bridges_gap_with_year_value():
    months = [str(p) for p in pd.period_range("2022-12", "2024-02", freq="M")]
    incs = pd.DataFrame({"month": months[1:], "value": 0.1, "sd": 0.03, "flag": 0})
    incs.loc[incs.month == "2023-06", "flag"] = 8               # missing month
    incs.loc[incs.month == "2023-06", "value"] = np.nan
    years = pd.DataFrame({"month": ["2023-12"], "value": [1.2], "sd": [0.03]})
    c = L.cumulative(months, incs, years).set_index("month")
    assert c.at["2022-12", "value"] == 0.0 and c.at["2022-12", "ref_month"] == "2022-12"
    assert np.isclose(c.at["2023-12", "value"], 1.2, atol=0.02)
    assert c.at["2024-02", "flag"] == 0
    assert np.isclose(c.at["2023-05", "value"], 0.5, atol=0.03)


def test_identity_offsets_and_sign_inversion():
    h = _series(24 * 30, rate_per_day=1.0)
    r = h.copy()
    v = r + 100.0
    v.iloc[24 * 10:] += 5.0                    # vendor stitch: offset changes
    v.iloc[24 * 20:] = -r.iloc[24 * 20:] + 300.0   # sign inversion afterwards
    out = L.identity(v, r)
    signs = [s["sign"] for s in out["segments"]]
    assert signs[0] == 1 and signs[-1] == -1
    kinds = [s["kind"] for s in out["stitches"]]
    assert "offset_change" in kinds


def test_artificial_gaps_and_envelope():
    h = _series(24 * 200)
    s = L.artificial_gaps(h.to_numpy(), durs=(1, 24, 72), n_per=200, seed=3)
    t = L.envelope_table(s)
    assert set(t.d_h) == {1, 24, 72}
    q = t.set_index(["d_h", "estimator"])["q999_mm"]
    assert q[(72, "rate")] < q[(72, "glob")]       # a steady trend: rate-adjusted is tighter


def test_as_ns_avoids_pandas_us_union_bug():
    # two regular microsecond hourly series: pandas 2.3.3 mis-aligns their union
    a = pd.Series(1.0, index=pd.date_range("2024-05-07", "2024-06-30", freq="h", unit="us"))
    b = pd.Series(2.0, index=pd.date_range("2024-05-20", "2024-07-31", freq="h", unit="us"))
    ha, hb = L.to_hourly(a), L.to_hourly(b)
    assert ha.index.unit == "ns" and hb.index.unit == "ns"
    j = pd.concat([ha, hb], axis=1)
    expect = pd.date_range("2024-05-07", "2024-07-31", freq="h")
    assert len(j) == len(expect) and j.index.max() == expect.max()


def test_identity_union_share_with_sign_inversion():
    h = _series(24 * 40, rate_per_day=1.0, seed=4)
    r = h.copy()
    v = r + 50.0
    v.iloc[24 * 25:] = -r.iloc[24 * 25:] + 400.0     # inverted after day 25
    out = L.identity(v, r)
    assert out["share_same"] < 0.98 and out["share_inverted"] < 0.98
    assert out["share_union"] >= 0.98
    sg = L.sign_series(r.index, out["segments"])
    assert sg[24 * 10] == 1 and sg[24 * 30] == -1


def test_sign_series_inversion_can_end():
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    segs = [{"sign": 1, "start": str(idx[0])}, {"sign": -1, "start": str(idx[30])},
            {"sign": 1, "start": str(idx[60])}]
    sg = L.sign_series(idx, segs)
    assert sg[10] == 1 and sg[40] == -1 and sg[80] == 1
