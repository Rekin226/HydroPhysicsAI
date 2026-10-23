"""CPU unit tests of the S1b repair round r3 helpers (synthetic values only, no real data)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from s1_rings.s1b import s1b_r3_after as AF  # noqa: E402
from s1_rings.s1b import s1b_r3_io as IO3  # noqa: E402
from s1_rings.s1b import s1b_r3_synth as SY  # noqa: E402


def _field(T: int = 4):
    v = np.zeros((T, 5, 5), float)
    for t in range(T):
        v[t] = 10.0 * t + np.arange(25).reshape(5, 5)       # up-positive mm
    src = np.ones((T, 5, 5), int)
    mask = np.ones((5, 5), bool)
    br = np.zeros((5, 5), bool)
    return v, src, mask, br


def test_pixel_rule_median_and_sign():
    v, src, mask, br = _field()
    px = IO3.pixel_rule(v, src, mask, br, 2, 2)
    # 3 x 3 window around (2, 2): values 6, 7, 8, 11, 12, 13, 16, 17, 18 (+ 10 t); median 12
    assert px["n_pix_ok"] == 9
    assert np.allclose(px["down_mm"], -(12.0 + 10.0 * np.arange(4)))
    assert (px["n_valid"] == 9).all()


def test_pixel_rule_excludes_bridged_masked_and_untied():
    v, src, mask, br = _field()
    br[1, 1] = True                  # a pixel with a bridged epoch is dropped at every epoch
    mask[1, 2] = False               # outside the primary LiCSBAS15 mask
    src[2, 3, 3] = 4                 # untied second-frame value at epoch 2 only
    v[3, 2, 1] = np.nan              # missing value at epoch 3
    px = IO3.pixel_rule(v, src, mask, br, 2, 2)
    assert px["n_pix_ok"] == 7
    assert list(px["n_valid"]) == [7, 7, 6, 6]
    # epoch 0 remaining values: 8, 11, 12, 13, 16, 17, 18 -> median 13
    assert math.isclose(px["down_mm"][0], -13.0)
    assert px["n_source4"] == 1


def test_pixel_rule_min_valid():
    v, src, mask, br = _field(2)
    mask[:] = False
    mask[1, 1] = mask[1, 2] = mask[1, 3] = True
    px = IO3.pixel_rule(v, src, mask, br, 2, 2, min_valid=4)
    assert np.isnan(px["down_mm"]).all()
    fin = IO3.pixel_rule(np.isfinite(v), src, mask, br, 2, 2, values=False)
    assert "down_mm" not in fin and list(fin["n_valid"]) == [3, 3]


def test_scenario_values():
    o = {"mem": {"A": 50.0, "B": 0.0}, "raw": -1.0}
    d = {"mem": {"A": 60.0, "B": 10.0}, "raw": 9.0}
    L = 280000.0
    dL = SY.S.ALT_PPM * 1e-6 * L
    sv = SY.scenario_values(o, d, "A", 0.6, L)
    assert sv["correct"][0] == 50.0 and sv["other"][0] == 0.0 and sv["raw"][0] == -1.0
    assert math.isclose(sv["up25"][0], 50.0 + dL) and math.isclose(sv["up25A"][0], 50.0 + dL)
    assert math.isclose(sv["mix"][0], 30.0) and math.isclose(sv["mix_drawn"][0], 40.0)
    assert sv["mix_drawn"][1] is d and sv["mix"][1] is o
    sb = SY.scenario_values(o, d, "B", 0.2, L)
    assert sb["up25A"][0] == 0.0 and math.isclose(sb["up25"][0], dL)


def test_family_eval_and_d1():
    def draw(c_pre, c_post, raw=0.0, g=30.0):
        return {"C_pre": c_pre, "C_post": c_post, "s_pre": 5.0, "s_post": 5.0, "raw": raw,
                "G": g, "gap_yr": 1.2}
    L = 250000.0
    items = [(30.0, draw(40.0, 20.0), L)] * 4 + [(60.0, draw(40.0, 20.0), L)]
    e = SY.family_eval(items, "k:0")
    assert e["count"] == 4 and e["n"] == 5 and e["contain_pass"]
    assert e["family_pass"]                  # rmse(value) 13.4 < rmse(raw) 30
    bad = [(30.0, draw(40.0, 20.0, raw=30.0), L)] * 5
    assert not SY.family_eval(bad, "k:0")["family_pass"]   # equal RMSE: clause fails
    d1 = SY.d1_eval([(10.0 * i, draw(0, 0, raw=(-1.0) ** i, g=10.0 * i + 3.0), L)
                     for i in range(6)])
    assert d1["crit"] and math.isclose(d1["r_b"], 1.0)


def test_probs_and_ratio():
    p = AF.probs([0, 2, 6, 2], 2)
    assert math.isclose(p["p_eq"], 0.6) and math.isclose(p["p_le"], 0.8) and p["n_real"] == 10
    assert AF.ratio(0.3, 0.0) is None and math.isclose(AF.ratio(0.3, 0.6), 0.5)


def test_install_restore():
    from s1_rings.s1b import s1b_run, s1b_synth
    before = (s1b_run.IO, s1b_synth.IO)
    ad = IO3.Adapter()
    prev = IO3.install(ad)
    try:
        assert s1b_run.IO is ad and s1b_synth.IO is ad
        assert ad.insar_pixel_series is IO3.insar_pixel_series
    finally:
        IO3.restore(prev)
    assert before == (s1b_run.IO, s1b_synth.IO)


def _fake_inputs():
    """Fake products shaped like rings_bridged/v3, gnss_clean/full/v4 and insar_069A/v3."""
    import pandas as pd
    from s1_rings.s1b import s1b_const as K
    rng = np.random.default_rng(5)
    xy = {s: (200000.0 + 3000.0 * i, 2600000.0) for i, s in enumerate(K.SITES)}
    t = pd.date_range("2019-01-15", "2025-05-15", freq="MS") + pd.Timedelta(days=14)
    t = t[(t < "2021-12-02") | (t >= "2023-02-20")]
    t = t.append(pd.DatetimeIndex(["2021-12-02", "2023-02-20"])).unique().sort_values()
    yr = (t - pd.Timestamp("2000-01-01")).days.to_numpy() / 365.25
    rings = {}
    for k, s in enumerate(K.SITES):
        c = (15.0 + k) * (yr - yr[0]) + rng.normal(0, 2, len(t))
        raw = float(c[t.get_loc(pd.Timestamp("2023-02-20"))] - c[t.get_loc(
            pd.Timestamp("2021-12-02"))])
        rings[s] = {"site": s, "t": t, "c": c, "t_lp": pd.Timestamp("2021-12-02"),
                    "t_fp": pd.Timestamp("2023-02-20"), "L_mm": 280000.0, "sd_s_mm": 3.0,
                    "gap_yr": 1.22, "p_A": 0.9, "primary": "mix", "assigned": "",
                    "members": "A,B", "C": {"primary": raw + (0.0 if k % 3 == 0
                                                              else 51.0), "A": raw + 51.0,
                                            "B": raw, "raw": raw},
                    "raw_check_mm": 0.0, "b_A_ppm": 182.0, "b_B_ppm": 0.0,
                    "sigma_C_primary_mm": 10.0}
    days = pd.date_range("2020-01-01", "2025-03-31", freq="D")
    dy = (days - pd.Timestamp("2000-01-01")).days.to_numpy() / 365.25
    names = [f"S{i:02d}" for i in range(10)]
    daily = []
    for n in names:
        h = -0.001 * (25.0 * (dy - dy[0]) + rng.normal(0, 3, len(days)))
        daily.append(pd.DataFrame({"family": "wra", "station": n, "date": days, "value": h,
                                   "value_raw": h, "correction_m": 0.0, "flag": 0}))
    daily = pd.concat(daily, ignore_index=True)
    stations = pd.DataFrame({"family": "wra", "station": names,
                             "x_twd97": [xy[K.SITES[i]][0] + 5.0 for i in range(10)],
                             "y_twd97": [xy[K.SITES[i]][1] for i in range(10)],
                             "first_day": "2020-01-01", "last_day": "2025-03-31",
                             "sigma_hf_m": 0.003, "rho_hf": 0.2})
    steps = pd.DataFrame({"family": ["wra"], "station": ["S03"], "date": ["2022-06-17"],
                          "size_m": [0.1], "se_m": [0.003], "basis": ["own_series"]})
    cm = pd.DataFrame({"date": days, "ngl_offfan": 0.0})
    g = {"daily": daily, "stations": stations, "steps": steps, "common_mode": cm, "notes": []}
    dates = pd.date_range("2020-01-12", "2025-12-11", freq="12D")
    ty = (dates - pd.Timestamp("2000-01-01")).days.to_numpy() / 365.25
    xs = 200000.0 + 1000.0 * np.arange(-2, 3 * len(K.SITES) + 2)
    X, Y = np.meshgrid(xs, 2600000.0 + 1000.0 * np.arange(-2, 3))
    val = np.broadcast_to((-27.0 * (ty - ty[0]))[:, None, None], (len(dates),) + X.shape)
    val = val + rng.normal(0, 2, val.shape)
    ins = {"x": X, "y": Y, "licsbas15_mask": np.ones(X.shape, bool),
           "any_bridged": np.zeros(X.shape, bool), "dates": dates,
           "network_segment": np.zeros(len(dates), int), "value": val,
           "source": np.ones(val.shape, int), "cell_dates": dates,
           "cell_segment": np.zeros((1, len(dates)), int),
           "cell_value": val[:, 2, 2][None, :].copy(),
           "cell_value_plane": val[:, 2, 3][None, :].copy(), "notes": []}
    return rings, xy, g, ins


def test_second_look_pipeline_on_fake_inputs():
    from s1_rings.s1b import s1b_r3_run as RR
    from s1_rings.s1b import s1b_run as RUN

    class Fake(IO3.Adapter):
        twin_cell = staticmethod(lambda x, y: 0)

    rings, xy, g, ins = _fake_inputs()
    prev = IO3.install(Fake())
    try:
        res = RUN.compute(RR.PRIMARY, rings, xy, g, ins)
        rb = RR.readings_block(res)
        assert rb["verdict_by_reading"]["k:2"]["c_wra_n"] == 10
        assert len(res["rows"]["d_insar"]) == 14
        assert all(r["evaluable"] for r in res["rows"]["c_wra"])
        assert all(r["n_bridges"] == 0 for r in res["rows"]["d_insar"])
        st = RR.d1_stats(res)
        assert st["rates"]["n"] == 14
        for over in RR.VARIANTS.values():
            RR.readings_block(RUN.compute({**RR.PRIMARY, **over}, rings, xy, g, ins))
        rp = RUN.compute({**RR.PRIMARY, "insar_source": "cells"}, rings, xy, g,
                         IO3.with_plane_cells(ins))
        assert rp["rows"]["d_insar"][0]["evaluable"]
    finally:
        IO3.restore(prev)
