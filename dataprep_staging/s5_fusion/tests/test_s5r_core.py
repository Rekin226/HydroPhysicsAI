"""CPU unit tests of the S5 repair-round v2 operators (GNSS segments, stepped benchmarks;
synthetic data only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import s5fit as F  # noqa: E402
import s5r_core as RC  # noqa: E402


# ----------------------------------------------------------------------------- GNSS segments
def test_gnss_breaks_and_segments():
    d1 = pd.date_range("2022-01-01", "2022-03-31")
    d2 = pd.date_range("2022-05-15", "2022-06-30")            # gap 45 d -> break
    d3 = pd.date_range("2022-07-21", "2022-09-30")            # gap 21 d -> no break
    dates = d1.append(d2).append(d3)
    daily = pd.DataFrame({"family": "wra", "station": "AAAA",
                          "date": dates.strftime("%Y-%m-%d"), "flag": 0})
    daily.loc[5:9, "flag"] = 9                               # flag 9 days do not count ...
    br = RC.gnss_breaks(daily)
    assert len(br[("wra", "AAAA")]) == 1
    (before, after, gap), = br[("wra", "AAAA")]
    assert str(before.date()) == "2022-03-31" and str(after.date()) == "2022-05-15"
    gn = pd.DataFrame({"family": "wra", "station": "AAAA",
                       "month": ["2022-01", "2022-03", "2022-05", "2022-06", "2022-09"]})
    assert list(RC.gnss_segments(gn, br)) == [0, 0, 1, 1, 1]


def test_step_scan():
    rng = np.random.default_rng(4)
    t = np.arange(2012.4, 2021.5, 1.0)
    r = rng.normal(0, 2.0, size=t.size) + 3.0 * (t - t.mean())
    r_step = r + 60.0 * (np.arange(t.size) >= 5)
    b = RC.step_scan(t, r_step, 2.0, 3.0)
    assert b["k"] == 5 and abs(b["delta_mm"] - 60.0) < 8.0 and abs(b["z"]) > 5
    b0 = RC.step_scan(t, r, 2.0, 3.0)
    assert abs(b0["z"]) < 5
    assert RC.step_scan(t[:3], r[:3], 2.0, 3.0) is None


# ----------------------------------------------------------------------------- v2 problem
def _tiny_data(seed=0, step_mm=0.0):
    rng = np.random.default_rng(seed)
    cells = np.array([[x + 0.5, y + 0.5] for x in range(8) for y in range(8)], float)
    g = {"cell_xy_km": cells}
    sids, rows = [f"B{i}" for i in range(6)], []
    for s in sids:
        xy = cells[rng.integers(0, len(cells))]
        for yr in range(2012, 2020):
            t = yr + 0.4
            rows.append({"sid": s, "date": f"{yr}-05-27", "t": t,
                         "y": 20.0 * (t - 2012) + rng.normal(0, 3), "x_km": xy[0],
                         "y_km": xy[1], "cell": 0})
    lev = pd.DataFrame(rows)
    months = [str(p) for p in pd.period_range("2015-01", "2019-12", freq="M")]
    from s5data import month_mid
    tt = month_mid(months)
    y = 20.0 * (tt - 2012) + rng.normal(0, 2, size=tt.size)
    y = y + step_mm * (np.arange(tt.size) >= 30)
    gn = pd.DataFrame({"family": "wra", "station": "GPS1", "month": months, "t": tt, "y": y,
                       "sd": 1.0, "x_km": 4.0, "y_km": 4.0})
    ins = {"units": pd.DataFrame(columns=["unit", "block", "seg", "x_km", "y_km", "n_cells"]),
           "unit_cells": [], "groups": []}
    cmp_ = pd.DataFrame(columns=["sensor", "kind", "t", "y", "seg", "x_km", "y_km"])
    return {"grid": g, "lev": lev, "gnss": gn, "insar": ins, "cmp": cmp_}


H_TINY = {"ell_km": 10.0, "sig_v0": 10.0, "q": 4.0, "sig_s": 2.0, "sig_lev": 3.0,
          "sig_rb": 2.0, "sig_gnss": 2.0, "sig_d": 3.0, "sig_cm": 1.0, "sig_ins": 1.0,
          "sig_aps": 1.0, "ell_aps_km": 10.0, "sig_cmp": 1.0, "sig_rc": 1.0}
KN = tuple(float(y) for y in range(2012, 2021))


def test_v2_without_segments_equals_round1():
    data = _tiny_data()
    fams = ("lev", "gnss")
    pr1 = F.make_problem(data, KN, fams)
    p1 = F.fit(pr1, H_TINY)
    pr2 = RC.make_problem_v2(data, KN, fams)
    p2 = RC.fit_v2(pr2, H_TINY)
    assert abs(p1.m2loglik - p2.m2loglik) < 1e-8
    assert np.allclose(p1.theta, p2.theta)


def test_v2_gnss_segment_absorbs_offset():
    data = _tiny_data(step_mm=-80.0)
    fams = ("lev", "gnss")
    data["gnss"]["seg"] = (np.arange(len(data["gnss"])) >= 30).astype(int)
    pr = RC.make_problem_v2(data, KN, fams)
    assert "goff:wra:GPS1:1" in pr.gindex
    post = RC.fit_v2(pr, H_TINY)
    off0 = post.theta[pr.L + pr.gindex["goff:wra:GPS1"]]
    off1 = post.theta[pr.L + pr.gindex["goff:wra:GPS1:1"]]
    assert abs((off1 - off0) + 80.0) < 6.0
    # leveling segments: a zero column changes nothing
    data["lev"]["seg"] = 0
    pr3 = RC.make_problem_v2(data, KN, fams)
    c3 = RC.contribution_v2(pr3, "lev", H_TINY)
    c1 = F.contribution(pr3, "lev", H_TINY)
    assert np.allclose(c3.P, c1.P) and abs(c3.logdetR - c1.logdetR) < 1e-9
