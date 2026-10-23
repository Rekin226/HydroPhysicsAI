"""CPU unit tests of the S1a library on synthetic ring sites (no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402


def fake_site(b=182e-6, a=-0.158, seed=0, noise=0.0005, gap=("2022-01-01", "2023-01-31")):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-05-05", "2025-05-05", freq="MS") + pd.Timedelta(days=4)
    dates = dates[(dates < gap[0]) | (dates > gap[1])]
    depths = np.array([2.0, 20, 45, 70, 100, 130, 160, 190, 220, 240, 260, 275, 290])
    # shortening rate of the column above each ring (m/yr): most compaction shallow
    rate_above = 0.03 * (1 - np.exp(-depths / 60.0))
    t = L.years(dates)
    t0 = t[0]
    z = depths[None, :] - (t[:, None] - t0) * rate_above[None, :] \
        - 0.004 * np.sin(2 * np.pi * t)[:, None] * (depths[None, :] < 150)
    z = z + rng.normal(0, noise, z.shape)
    truth = pd.DataFrame(z, index=dates, columns=[f"NO{i + 1}" for i in range(len(depths))])
    obs = truth.copy()
    after = obs.index >= pd.Timestamp("2023-01-01")
    obs.loc[after] = a + (1 + b) * obs.loc[after]
    return truth, obs


def test_window_fit_recovers_rate():
    t = np.linspace(19, 22, 37)
    y = 1.0 + 0.02 * (t - 20.5) + 0.003 * np.sin(2 * np.pi * t)
    f = L.window_fit(t, y, 20.5)
    assert abs(f.rate - 0.02) < 1e-9
    assert abs(f.trend(20.5) - 1.0) < 1e-9


def test_gap_stats_recovers_scale_and_offset():
    truth, obs = fake_site()
    gs = L.gap_stats(obs, K.PRE_WIN, K.POST_WIN)
    assert gs["n_deep"] >= 3
    assert abs(gs["b_deep"] - 182e-6) < 25e-6
    assert abs(gs["J_top_mid"] - (-0.158 + 182e-6 * 2.0)) < 0.003
    m = L.member_gap(gs, 182e-6, 10e-6)
    S = truth["NO13"] - truth["NO1"]
    C_true = S.loc[gs["t_lp"]] - S.loc[gs["t_fp"]]
    assert abs(m["C"] - C_true) < 0.002


def test_group_B_no_scale():
    truth, obs = fake_site(b=2e-6, a=0.52)
    gs = L.gap_stats(obs, K.PRE_WIN, K.POST_WIN)
    assert abs(gs["b_deep"]) < 25e-6
    assert L.offset_line(gs["J_top_mid"])["vote"] == "B"
    assert L.line_lr(gs["b_deep"], max(gs["sd_deep"], 10e-6))["vote"] == "B"


def test_votes_and_assignment():
    lines = {"offset": L.offset_line(-0.159), "b_deep": L.line_lr(190e-6, 15e-6),
             "pile": L.line_lr(150e-6, 30e-6)}
    a = L.assign(lines, forced=False)
    assert a["assigned"] == "A" and a["votes_A"] == 3
    assert L.assign(lines, forced=True)["assigned"] is None
    lines2 = {"offset": L.offset_line(-0.159), "b_deep": L.line_lr(np.nan, np.nan)}
    assert L.assign(lines2, forced=False)["assigned"] is None
    assert 0.5 < L.membership(lines2) < 1.0
    # implausible value: favours A by LR but lies > 3 sd from A -> no vote
    assert L.line_lr(600e-6, 20e-6)["vote"] is None


def test_pool_and_loo():
    pr = L.pool([180e-6, 190e-6, 170e-6], [10e-6, 10e-6, 10e-6])
    assert abs(pr["mean"] - 180e-6) < 1e-12 and pr["n"] == 3
    gsc = L.group_scale("A", "s1", {"s1": "A", "s2": "A", "s3": None},
                        {"s1": 100e-6, "s2": 180e-6, "s3": 0.0},
                        {"s1": 10e-6, "s2": 10e-6, "s3": 10e-6})
    assert gsc["sites"] == ["s2"] and abs(gsc["b"] - 180e-6) < 1e-12
    fb = L.group_scale("B", "s1", {"s1": "A"}, {"s1": 1e-4}, {"s1": 1e-5})
    assert fb["fallback"] and fb["b"] == K.REF["B"][0]


def test_mixture_second_moment():
    mem = {"A": {"C": 0.03, "sigma_C": 0.004, "b": 180e-6},
           "B": {"C": -0.02, "sigma_C": 0.004, "b": 2e-6}}
    mx = L.mixture(mem, 0.5, None)
    assert abs(mx["C"] - 0.005) < 1e-12
    assert abs(mx["sigma_C"] - np.sqrt(0.004 ** 2 + 0.025 ** 2)) < 1e-12
    asg = L.mixture(mem, 0.99, "A")
    assert asg["C"] == 0.03 and asg["sigma_C"] > 0.004


def test_fill_methods():
    d0, d1 = pd.Timestamp("2021-12-05"), pd.Timestamp("2023-02-20")
    dates = L.virtual_dates(pd.period_range("2022-01", "2023-01", freq="M"), 5)
    v, tau = L.fill(0.0, 0.03, d0, d1, dates, "linear")
    assert np.all(np.diff(v) > 0) and 0 < tau[0] < tau[-1] < 1
    clim = {m: (0.002 if m in (6, 7) else 0.0) for m in range(1, 13)}
    vc, _ = L.fill(0.0, 0.03, d0, d1, dates, "climatology", clim=clim)
    assert np.isclose(vc[5] - v[5], 0.002)
    q = pd.Series(np.linspace(0, 40, 600),
                  index=pd.date_range("2021-11-01", periods=600, freq="D"))
    vp, _ = L.fill(0.0, 0.03, d0, d1, dates, "pile", q=q)
    assert np.allclose(vp, v, atol=1e-4)      # a linear pile adds no shape


def test_bands_partition():
    zbar = pd.Series([2.0, 50, 98, 150, 205, 290], index=list("abcdef"))
    assert L.band_indices(zbar) == [(0, 2), (2, 4), (4, 5)]


def test_corrected_positions_roundtrip():
    truth, obs = fake_site()
    q = L.corrected_positions(obs, -0.158, 182e-6, "2023-01-01")
    assert np.allclose(q.to_numpy(), truth.to_numpy(), atol=1e-12)
    # with a compaction rate linear in depth the residual offset is zero (it is the intercept
    # of the corrected survey-to-survey change regressed on depth)
    lin = truth.copy()
    t = L.years(lin.index)
    zc = np.array([2.0, 20, 45, 70, 100, 130, 160, 190, 220, 240, 260, 275, 290])
    lin[:] = zc[None, :] - (t[:, None] - t[0]) * 1e-4 * zc[None, :]
    ob = lin.copy()
    after = ob.index >= pd.Timestamp("2023-01-01")
    ob.loc[after] = -0.158 + (1 + 182e-6) * ob.loc[after]
    qc = L.corrected_positions(ob, -0.158, 182e-6, "2023-01-01")
    ro = L.residual_offset(qc, list(qc.columns), qc.loc[:"2021-12-31"].index[-1],
                           qc.loc["2023-01-01":].index[0])
    assert abs(ro["offset_m"]) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
