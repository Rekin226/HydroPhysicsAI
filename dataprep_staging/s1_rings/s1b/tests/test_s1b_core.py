"""CPU unit tests of the S1b core (synthetic series only, no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from s1_rings.s1b import s1b_core as C  # noqa: E402

T_LP, T_FP = pd.Timestamp("2021-12-10"), pd.Timestamp("2023-02-20")
PRE, POST = ("2020-01-01", "2021-12-31"), ("2023-02-01", "2025-05-31")


def _ring(rates=(30.0, 20.0, 15.0), amp=3.0, b_ppm=180.0, L_mm=285000.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DatetimeIndex(sorted(set(pd.date_range("2019-06-05", "2021-12-10", freq="MS")
                                    .tolist() + [T_LP] + [T_FP]
                                    + pd.date_range("2023-03-05", "2025-05-20", freq="MS")
                                    .tolist())))
    y = C.years(t)
    ylp, yfp = C.years([T_LP])[0], C.years([T_FP])[0]
    r1, rg, r2 = rates
    lin = np.where(y < ylp, r1 * (y - ylp), np.where(y < yfp, rg * (y - ylp),
                                                      rg * (yfp - ylp) + r2 * (y - yfp)))
    c = lin + amp * np.sin(2 * np.pi * y) + rng.normal(0, noise, len(y))
    c = c - (t >= T_FP) * b_ppm * 1e-6 * L_mm
    raw = float(c[t == T_FP][0] - c[t == T_LP][0])
    true_gap = rg * (yfp - ylp) + amp * (np.sin(2 * np.pi * yfp) - np.sin(2 * np.pi * ylp))
    return t, c, raw, true_gap


def _inst(ring_rates=(30.0, 20.0, 15.0), ex_rates=(10.0, 12.0, 14.0), amp=3.0, amp_e=2.0,
          freq="D", seg_offsets=None, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", "2025-03-31", freq=freq)
    y = C.years(t)
    ylp, yfp = C.years([T_LP])[0], C.years([T_FP])[0]

    def pw(r):
        return np.where(y < ylp, r[0] * (y - ylp), np.where(y < yfp, r[1] * (y - ylp),
                                                             r[1] * (yfp - ylp) + r[2] * (y - yfp)))
    d = (pw(ring_rates) + amp * np.sin(2 * np.pi * y) + pw(ex_rates)
         + amp_e * np.cos(2 * np.pi * y) + rng.normal(0, noise, len(y)))
    seg = None
    if seg_offsets is not None:
        bounds = [pd.Timestamp("2022-06-10"), pd.Timestamp("2022-08-20")]
        seg = np.searchsorted(np.array(bounds, dtype="datetime64[ns]"),
                              t.to_numpy(), side="right")
        d = d + np.asarray(seg_offsets)[seg]
    return t, d, seg


def test_fit_pre_recovers_rate_and_harmonic():
    t = np.linspace(20.0, 22.0, 200)
    y = 5 + 12.0 * (t - 21) + 3 * np.sin(2 * np.pi * t) - 1.5 * np.cos(2 * np.pi * t)
    f = C.fit_pre(t, y, 21.0)
    assert abs(f.rate - 12.0) < 1e-9
    assert np.allclose(f.beta[2:4], [3.0, -1.5])


def test_fit_post_segment_intercepts_remove_offsets():
    t = np.linspace(23.1, 25.3, 150)
    seg = (t > 23.6).astype(int) + (t > 24.6).astype(int)
    pre = C.fit_pre(np.linspace(20, 22, 50), np.zeros(50), 22.5)
    y = 7.0 * (t - 22.5) + np.array([0.0, 25.0, -40.0])[seg]
    q = C.fit_post(t, y, 22.5, pre, seg)
    assert abs(q.rate - 7.0) < 1e-9 and q.n_seg == 3
    q1 = C.fit_post(t, y, 22.5, pre, None)
    assert abs(q1.rate - 7.0) > 1.0          # without intercepts the offsets bias the rate


def test_bracket_contains_correct_member_and_excludes_wrong():
    L = 285000.0
    rt, rc, raw, true_gap = _ring(b_ppm=180.0, L_mm=L)
    it, idd, _ = _inst()
    res = C.instrument_vs_ring(ring_t=rt, ring_c=rc, inst_t=it, inst_d=idd, t_lp=T_LP,
                               t_fp=T_FP, pre_win=PRE, post_win=POST, min_inst=240,
                               min_ring=8, min_span=270, end_half_d=3, end_min=3)
    assert res["ok"], res
    Cv = {"primary": raw + 180e-6 * L, "A": raw + 180e-6 * L, "B": raw + 2e-6 * L, "raw": raw}
    chk = C.site_check(res["G_mm"], res["excess"], 1.0, 1.0, L, Cv, k=2.0)
    # the excess gap rate (12) lies between the pre (10) and post (14) rates
    assert chk["C_pre_mm"] - 1 <= true_gap <= chk["C_post_mm"] + 1 or \
        chk["C_post_mm"] - 1 <= true_gap <= chk["C_pre_mm"] + 1
    assert chk["contains_primary"] and chk["contains_A"]
    assert not chk["contains_B"] and not chk["contains_raw"]
    assert chk["membership_evidence"] == "A"
    assert abs(chk["b_mid_ppm"] - 180.0) < 15.0


def test_insar_like_series_with_bridges():
    L = 280000.0
    rt, rc, raw, _ = _ring(b_ppm=2.0, L_mm=L)
    it, idd, seg = _inst(freq="12D", seg_offsets=[0.0, 9.0, -6.0])
    res = C.instrument_vs_ring(ring_t=rt, ring_c=rc, inst_t=it, inst_d=idd, inst_seg=seg,
                               t_lp=T_LP, t_fp=T_FP, pre_win=PRE, post_win=POST, min_inst=20,
                               min_ring=8, min_span=270, end_half_d=18, end_min=2)
    assert res["ok"], res
    assert res["n_bridges"] == 2
    # the raw gap change contains the bridge offsets (+3 mm net): G is biased by the bridges
    Cv = {"primary": raw + 2e-6 * L, "A": raw + 182e-6 * L, "B": raw + 2e-6 * L, "raw": raw}
    chk = C.site_check(res["G_mm"], res["excess"], 2.0, 1.0, L, Cv, k=2.0, n_bridges=2,
                       bridge_sd=6.0)
    assert chk["contains_B"] and not chk["contains_A"]


def test_endpoint_mean_uses_segment_of_nearest_epoch():
    t = pd.date_range("2022-01-01", periods=10, freq="12D")
    v = np.arange(10.0)
    seg = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    e = C.endpoint_mean(t, v, t[4] + pd.Timedelta(days=2), 18, 2, seg)
    assert e["seg"] == 0 and e["n"] == 2 and e["mean"] == 3.5
    e = C.endpoint_mean(t, v, t[5], 18, 2, seg)
    assert e["seg"] == 1 and e["mean"] == 5.5


def test_family_verdict_rule():
    rows = [{"site": f"s{i}", "evaluable": True, "contains_primary": i < 8,
             "contains_raw": False, "C_primary_mm": 30.0, "C_raw_mm": -20.0, "C_mid_mm": 31.0}
            for i in range(10)]
    v = C.family_verdict(rows, 0.8, 3)
    assert v["n_needed"] == 8 and v["n_contain_primary"] == 8 and v["pass"]
    rows[7]["contains_primary"] = False
    assert not C.family_verdict(rows, 0.8, 3)["pass"]
    one = C.family_verdict(rows[:1], 0.8, 3)
    assert one["n_needed"] == 1 and not one["in_gate"]


def test_r_crit_and_corr():
    assert abs(C.r_crit(14) - 0.458) < 0.002
    x = np.arange(14.0)
    out = C.corr_check(x, x + 0.1 * np.sin(x), -x)
    assert out["pass"] and out["r_raw"] < 0


def test_ar1_factor():
    f, rho = C.ar1_factor(np.array([1.0, -1.0] * 50))
    assert rho == 0.0 and f == 1.0
    x = np.cumsum(np.ones(100))
    f, rho = C.ar1_factor(x)
    assert rho > 0.9 and f > 10


def test_site_check_step_variance_widens_bracket():
    ex = {"E_pre": 10.0, "E_post": 14.0, "var_E_pre": 4.0, "var_E_post": 4.0}
    Cv = {"primary": 40.0, "A": 40.0, "B": -10.0, "raw": -10.0}
    a = C.site_check(30.0, ex, 1.0, 1.0, 285000.0, Cv, k=2.0)
    b = C.site_check(30.0, ex, 1.0, 1.0, 285000.0, Cv, k=2.0, var_steps=49.0)
    assert b["var_common"] == a["var_common"] + 49.0
    assert b["hi_mm"] > a["hi_mm"] and b["lo_mm"] < a["lo_mm"]
    assert a["C_pre_mm"] == 20.0 and a["C_post_mm"] == 16.0


def test_gnss_series_steps_and_variance():
    from s1_rings.s1b import s1b_io as IO
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    g = {"daily": pd.DataFrame({"family": "wra", "station": "XX", "date": dates,
                                "value": np.zeros(10), "flag": [0] * 9 + [9]}),
         "steps": pd.DataFrame({"family": ["wra", "wra"], "station": ["XX", "XX"],
                                "date": ["2022-01-04", "2022-01-07"],
                                "size_m": [0.01, -0.02], "se_m": [0.003, 0.004],
                                "basis": ["own_series", "metadata_or_catalogue"]})}
    gap = (pd.Timestamp("2022-01-02"), pd.Timestamp("2022-01-09"))
    t, d, v = IO.gnss_series(g, "wra", "XX", gap=gap)
    assert len(t) == 9 and np.allclose(d, 0.0) and abs(v - (9.0 + 16.0)) < 1e-9
    t, d, v = IO.gnss_series(g, "wra", "XX", gap=gap, steps_mode="catalogue_in_gap")
    assert abs(v - 16.0) < 1e-9                      # the own-series step is added back
    assert np.allclose(d[:3], 0.0) and np.allclose(d[3:], -10.0)
