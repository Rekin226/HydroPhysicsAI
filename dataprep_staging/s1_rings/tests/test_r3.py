"""CPU unit tests of the S1a r3 library on synthetic ring sites (no real data)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s1_rings import r3_lib as R3  # noqa: E402
from s1_rings import s1const as K  # noqa: E402
from s1_rings.tests.test_s1lib import fake_site  # noqa: E402

SX, SC = 1.27e-3, 1.35e-3


def _sites(groups, seed0=0):
    pos, truth = {}, {}
    for i, g in enumerate(groups):
        b, a = (182e-6, -0.158) if g == "A" else (2e-6, 0.52)
        tr, ob = fake_site(b=b, a=a, seed=seed0 + i)
        pos[f"s{i:02d}"], truth[f"s{i:02d}"] = ob, tr
    return pos, truth


def test_bdeep_sd_length_aware():
    gs = {"b_deep": 1e-4, "sd_deep": 20e-6, "deep_len": 25.0}
    s_short = R3.bdeep_sd(gs, SX)
    s_long = R3.bdeep_sd({**gs, "deep_len": 125.0}, SX)
    assert s_short > s_long > 20e-6
    assert abs(s_short - math.hypot(20e-6, SX / 25.0)) < 1e-12
    assert math.isnan(R3.bdeep_sd({**gs, "b_deep": float("nan")}, SX))


def test_em_prior_laplace_and_bounds():
    pi, _ = R3.em_prior([1.0] * 10)            # no information: stays 0.5
    assert abs(pi - 0.5) < 1e-9
    pi, _ = R3.em_prior([1e3] * 10)            # all strongly A: Laplace rule caps it
    assert 0.9 < pi <= 11 / 12 + 1e-9
    pi, _ = R3.em_prior([1e3] * 7 + [1e-3] * 3)
    assert abs(pi - 8 / 12) < 0.01


def test_pool_threshold_and_fallback():
    gss = {"a": {"b_deep": 180e-6, "deep_len": 100.0}, "b": {"b_deep": 190e-6, "deep_len": 50.0},
           "c": {"b_deep": 0.0, "deep_len": 80.0}, "i": {"b_deep": 170e-6, "deep_len": 60.0}}
    sds = {k: R3.bdeep_sd({**v, "sd_deep": 20e-6}, SX) for k, v in gss.items()}
    lrs = {"a": 1e3, "b": 1e3, "c": 1e-3, "i": 1e3}
    p = R3.pool_r3("A", "i", list(gss), lrs, 0.7, gss, sds, SC)
    assert p["sites"] == ["a", "b"] and not p["fallback"]       # i left out, c is B
    assert p["sigma"] > p["se"]                                 # common-mode term added
    q = R3.pool_r3("B", "c", list(gss), lrs, 0.7, gss, sds, SC)
    assert q["fallback"] and q["b"] == K.REF["B"][0]            # no other B site


def test_evaluate_r3_mixed_sites_recovers_members():
    groups = ["A"] * 7 + ["B"] * 3
    pos, truth = _sites(groups)
    pre = {s: K.PRE_WIN for s in pos}
    post = {s: K.POST_WIN for s in pos}
    res = R3.evaluate_r3(pos, pre, post, set(), SX, SC)
    for (s, r), g in zip(sorted(res.items()), groups, strict=True):
        assert "pile" not in r.lines
        assert (r.p_A > 0.5) == (g == "A")
        gs = r.gs
        S = truth[s][gs["bot"]] - truth[s][gs["top"]]
        C_true = S.loc[gs["t_lp"]] - S.loc[gs["t_fp"]]
        assert abs(r.members[g]["C"] - C_true) < 0.004
        assert 0.0 < r.assign["prior_A"] < 1.0


def test_forced_site_still_pools_others():
    groups = ["A"] * 6 + ["B"] * 2
    pos, _ = _sites(groups, seed0=20)
    pre = {s: K.PRE_WIN for s in pos}
    post = {s: K.POST_WIN for s in pos}
    forced = {"s00"}
    res = R3.evaluate_r3(pos, pre, post, forced, SX, SC)
    assert res["s00"].assign["assigned"] is None                # forced: carried as both
    # s00 still enters the other A sites' pools (pooling ignores forcing)
    assert any("s00" in res[s].members["A"]["pool"]["sites"] for s in res if s != "s00")


def test_corrected_a_trend_endpoints_close_to_survey_value():
    pos, truth = _sites(["A"] * 4 + ["B"] * 2, seed0=40)
    pre = {s: K.PRE_WIN for s in pos}
    post = {s: K.POST_WIN for s in pos}
    res = R3.evaluate_r3(pos, pre, post, set(), SX, SC)
    for s, r in res.items():
        te = R3.trend_endpoints(pos[s], r.gs, K.PRE_WIN, K.POST_WIN)
        ca = R3.corrected_a_site(r, te)
        assert te["pre"]["ar1_factor"] >= 1.0
        assert abs(ca["C"] - r.primary["C"]) < 0.01
        assert ca["sigma_C"] > 0


def test_build_helpers_on_toy_sites():
    from s1_rings import r3_build as R3B
    pos, _ = _sites(["A"] * 6 + ["B"] * 3, seed0=60)
    pre = {s: K.PRE_WIN for s in pos}
    post = {s: K.POST_WIN for s in pos}
    res = R3.evaluate_r3(pos, pre, post, set(), SX, SC)
    va = R3B.test_a_corrected(pos, res)
    assert va["n"] == 9 and np.isfinite(va["pooled_z"]) and 0 <= va["power_25pct"] <= 1
    vb = R3B.test_b(res, {"sigma_x_m": SX})
    assert vb["n_tested"] == 0 and vb["pools"]["A"]["n"] >= 5
    s = "s00"
    g = "A" if res[s].assign["assigned"] is None else res[s].assign["assigned"]
    f = R3B.corrected_file(s, res[s], g, pos, res[s].assign["assigned"])
    post_rows = f[f.flag > 0]
    assert len(post_rows) and post_rows.sd_m.notna().all() and (post_rows.sd_m > 0).all()
    assert f[f.flag == 0].sd_m.isna().all()
    allowed, forbidden, why = R3B.allowed_uses_rule(True, True, {})
    assert allowed == ["after_the_fact_analysis"] and "second look" in why
    allowed, _, _ = R3B.allowed_uses_rule(True, False, {}, u7_ruled=True)
    assert "validator" in allowed


def test_ar1_factor():
    rng = np.random.default_rng(0)
    assert abs(R3.ar1_factor(rng.normal(size=2000)) - 1.0) < 0.1
    x = np.cumsum(rng.normal(size=2000)) * 0.1 + rng.normal(size=2000)
    assert R3.ar1_factor(x) > 1.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
