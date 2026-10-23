"""CPU unit tests of the S1a repair-round modules (no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s1_rings import build_r2 as BR  # noqa: E402
from s1_rings import r2_diagnose as RD  # noqa: E402
from s1_rings import s1const as K  # noqa: E402


# ----------------------------------------------------------------------------- r2_diagnose
def _rows(n=400, seed=1):
    """Synthetic (e) rows with known members: C_A - C_B = D, mixture where unassigned."""
    rng = np.random.default_rng(seed)
    tg = np.where(rng.random(n) < 0.7, "A", "B")
    D = rng.uniform(3.0, 8.0, n)                       # C_A - C_B (cm)
    eo = rng.normal(-0.5, 0.8, n)                      # error of the true member
    p = np.clip(rng.beta(2, 2, n), 0, 1)
    asg = np.where(rng.random(n) < 0.4, tg, "")
    wrong = rng.random(n) < 0.05
    asg = np.where(wrong & (asg != ""), np.where(tg == "A", "B", "A"), asg)
    ta = (tg == "A").astype(float)
    w = np.where(asg == "A", 1.0, np.where(asg == "B", 0.0, p))
    err = eo + (w - ta) * D
    forced = (asg == "") & (rng.random(n) < 0.5)
    evg = np.where(forced & (rng.random(n) < 0.5), tg, "")
    x = pd.DataFrame({"variant": "primary", "window": rng.integers(0, 6, n), "true_group": tg,
                      "assigned": asg, "p_A": p, "err_cm": err, "err_oracle_cm": eo,
                      "forced": forced, "evidence_group": evg,
                      "lines": "offset:A,b_deep:-"})
    x["start"] = x["window"].map(lambda w: f"2016-{w + 1:02d}")
    x["end"] = x["window"].map(lambda w: f"2017-{w + 1:02d}")
    return x, D


def test_member_diff_reconstructs_D():
    x, D = _rows()
    d = RD.member_diff(x)
    ok = np.isfinite(d)
    assert ok.sum() > 200
    assert np.allclose(d[ok], D[ok], atol=1e-9)
    # assigned to the true group: not identified
    same = (x.assigned == x.true_group).to_numpy()
    assert not np.isfinite(d[same]).any()


def test_counterfactual_stored_and_oracle():
    x, D = _rows()
    w0 = RD.stored_weight(x)
    assert np.allclose(RD.counterfactual(x, w0), x.err_cm, atol=1e-9)
    ta = (x.true_group == "A").to_numpy().astype(float)
    assert np.allclose(RD.counterfactual(x, ta), x.err_oracle_cm, atol=1e-9)
    # hard assignment of mixture rows
    p = x.p_A.to_numpy()
    mix = (x.assigned == "").to_numpy()
    wh = np.where(mix, (p >= 0.5).astype(float), w0)
    expect = x.err_oracle_cm.to_numpy() + (wh - ta) * D
    got = RD.counterfactual(x, wh)
    assert np.allclose(got[mix], expect[mix], atol=1e-9)
    assert RD.n_fallback(x, wh) == 0


def test_decompose_e_runs_and_is_consistent():
    x, _ = _rows()
    out = RD.decompose_e(x, "primary", 0.7)
    assert out["reconstruction_max_abs_err_cm"] < 1e-9
    W = pd.DataFrame(out["per_window"])
    mse = W.rmse_cm ** 2
    assert np.allclose(mse, W.mse_scale + W.mse_membership + W.mse_cross, atol=1e-3)
    cf = out["counterfactuals_after_the_fact"]
    assert cf["true group known (scale component only)"]["n_windows_over_1p5cm"] <= \
        cf["stored (registered primary, check)"]["n_windows_over_1p5cm"] + 6


def test_line_vote_parser():
    assert RD.line_vote("offset:A,b_deep:B,pile:-", "b_deep") == "B"
    assert RD.line_vote("offset:A,b_deep:B", "pile") == ""


# ----------------------------------------------------------------------------- build_r2
def _daily(days_may=9):
    dates = pd.date_range("2022-03-01", "2022-07-31", freq="D")
    keep = [d for d in dates if not (d.month == 5 and d.day > days_may)]
    return pd.DataFrame({"sensor_id": "TKJS(1)", "feed": "vendor", "date": keep,
                         "value": np.arange(len(keep), dtype=float), "flag": 0,
                         "variant": "x"})


def test_vendor_days_counts_flag01_only():
    d = _daily(9)
    d.loc[d.index[:3], "flag"] = 2
    n = BR.vendor_days(d, "TKJS(1)")
    assert n[pd.Period("2022-05", "M")] == 9
    assert n[pd.Period("2022-03", "M")] == 28


def test_feed_check_r2_screens_short_months():
    d = _daily(9)
    q = pd.Series(d["value"].to_numpy(), index=pd.DatetimeIndex(d["date"]))
    q = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="D"))
    mm = q.groupby(q.index.to_period("M")).mean().diff()
    printed = pd.DataFrame({"sensor_id": "TKJS(1)", "month": [str(m) for m in mm.index[1:]],
                            "value": mm.to_numpy()[1:], "flag": 0, "depth_m": 300})
    printed.loc[printed.month == "2022-05", "value"] += 10.0   # a bad 9-day month
    piles = {"TKJS(1)": q, "SJES": q, "NTUH(1)": q, "STES": q}
    out = BR.feed_check_r2(piles, d.assign(sensor_id="TKJS(1)"), printed)
    s = out["sensors"]["TKJS(1)"]
    months = {r["month"]: r for r in s["months"]}
    assert not months["2022-05"]["comparable"] and not months["2022-06"]["comparable"]
    assert months["2022-04"]["comparable"] and months["2022-07"]["comparable"]
    assert s["ok"] and s["n_screened_out_2022"] == 2


def test_allowed_uses_rule():
    up_ok = {"x": {"gate_passed": True}}
    up_bad = {"x": {"gate_passed": False}}
    a, f, _ = BR.allowed_uses_rule(True, up_ok)
    assert set(a) == {"validator", "scoring_target", "after_the_fact_analysis"}
    assert not set(a) & set(f)
    for gp, up in ((True, up_bad), (False, up_ok), (False, up_bad)):
        a, f, why = BR.allowed_uses_rule(gp, up)
        assert a == ["after_the_fact_analysis"]
        assert "validator" in f and "scoring_target" in f and why


def test_offset_clusters_context_restores():
    before = dict(K.OFFSET_CLUSTERS)
    with BR.offset_clusters(BR.PLAN_CLUSTERS):
        assert K.OFFSET_CLUSTERS == BR.PLAN_CLUSTERS
    assert before == K.OFFSET_CLUSTERS
