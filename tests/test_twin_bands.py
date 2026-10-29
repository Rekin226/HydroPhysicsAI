"""Synthetic tests of the signed Mondrian conformal bands (uncertainty plan E1). No real data."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.bands import (
    apply_bands,
    calibrate_band,
    coverage_report,
    interval_censored_hits,
    loco_hits,
    min_units,
    mondrian_bands,
    signed_offsets,
)

DRAWS = 10_000        # the A1 coverage_sim rule: >= 0.89 of a true 0.90 in 10,000 draws


def _shock_table(rng, k, m=100, shock=2.0, noise=0.5, first=2016):
    """``k`` origins of ``m`` rows; every row of an origin shares its year shock."""
    s = rng.normal(0.0, shock, size=k)
    return pd.DataFrame({"residual": np.repeat(s, m) + rng.normal(0.0, noise, size=k * m),
                         "origin": np.repeat(np.arange(first, first + k), m)})


# ----------------------------------------------------------------------------- ranks
def test_min_units_matches_the_finite_sample_rank():
    assert min_units(0.90) == 19             # signed: alpha/2 = 0.05 per tail
    assert min_units(0.90, signed=False) == 9
    assert min_units(0.80) == 9
    assert min_units(0.95) == 39
    for level in (0.8, 0.9, 0.95):
        n = min_units(level)
        x = np.arange(n, dtype=float)
        assert np.isfinite(signed_offsets(x, level)[:2]).all()
        assert not np.isfinite(signed_offsets(x[:-1], level)[:2]).all()


def test_signed_offsets_are_order_statistics():
    x = np.arange(1.0, 51.0)                 # n = 50: hi rank ceil(51*0.95)=49, lo rank 2
    lo, hi, units = signed_offsets(x, 0.90)
    assert (lo, hi, units) == (2.0, 49.0, 50)
    lo, hi, _ = signed_offsets(np.append(x, np.nan), 0.90)      # NaN rows dropped
    assert (lo, hi) == (2.0, 49.0)


# ----------------------------------------------------------------------------- guarantee
def test_row_mode_guarantee_on_exchangeable_draws():
    rng = np.random.default_rng(11)
    for n in (19, 40):
        hits = np.empty(DRAWS, dtype=bool)
        for d in range(DRAWS):
            r = rng.standard_t(3, size=n + 1) - 2.5          # heavy tails and a bias
            lo, hi, _ = signed_offsets(r[:n], 0.90)
            hits[d] = lo <= r[n] <= hi
        assert hits.mean() >= 0.89, (n, hits.mean())


@pytest.mark.parametrize("shock,noise", [(1.0, 1.0), (3.0, 0.3)])
def test_cluster_mode_guarantee_for_a_new_cluster(shock, noise):
    """Exchangeable clusters with a common shock, k = 19 (the minimum) and unequal sizes:
    a random row of a new cluster is covered at >= 0.89."""
    rng = np.random.default_rng(int(shock * 10))
    k = 19
    hits = np.empty(DRAWS, dtype=bool)
    for d in range(DRAWS):
        m = rng.integers(1, 31, size=k)
        r = np.repeat(rng.normal(0, shock, size=k), m) + rng.normal(0, noise, size=m.sum())
        lo, hi, units = signed_offsets(r, 0.90, np.repeat(np.arange(k), m))
        new = rng.normal(0, shock) + rng.normal(0, noise)
        hits[d] = lo <= new <= hi
    assert units == k
    assert hits.mean() >= 0.89, hits.mean()


def test_signed_band_recovers_an_injected_bias():
    rng = np.random.default_rng(3)
    t = pd.DataFrame({"residual": rng.normal(-2.5, 1.0, size=4000)})
    b = calibrate_band(t["residual"])
    assert b.available
    assert abs(b.median - (-2.5)) < 0.1
    assert abs((b.lo + b.hi) / 2 - (-2.5)) < 0.1      # the band is centred on the bias
    new = rng.normal(-2.5, 1.0, size=20000)
    below, above = np.mean(new < b.lo), np.mean(new > b.hi)
    assert abs(below - 0.05) < 0.01 and abs(above - 0.05) < 0.01
    # the absolute-score band centred on the prediction cannot: it covers by widening,
    # and every miss is on one side
    q = np.quantile(np.abs(t["residual"]), 0.90)
    assert np.mean(new > q) < 0.001 and np.mean(new < -q) > 0.09


# ----------------------------------------------------------------------------- Mondrian
def _zoned(rng, n=400):
    zone = rng.choice(["distal", "mid"], size=n)
    lead = rng.choice([1, 2], size=n)
    bias = np.where(zone == "distal", -2.5, 0.0)
    scale = np.where(lead == 2, 2.0, 0.5)
    return pd.DataFrame({"zone": zone, "lead": lead,
                         "residual": bias + scale * rng.normal(size=n)})


def test_mondrian_bands_follow_each_stratum():
    t = _zoned(np.random.default_rng(5))
    b = mondrian_bands(t, ["zone", "lead"]).set_index(["zone", "lead"])
    assert b["available"].all()
    assert b.loc[("distal", 1), "hi"] < b.loc[("mid", 1), "lo"] + 1.0   # bias absorbed
    assert (b.loc[("mid", 2), "hi"] - b.loc[("mid", 2), "lo"]
            > 2.5 * (b.loc[("mid", 1), "hi"] - b.loc[("mid", 1), "lo"]))
    pred = pd.DataFrame({"zone": ["distal", "mid"], "lead": [1, 2], "predicted": [10.0, 0.0]})
    out = apply_bands(pred, b.reset_index(), ["zone", "lead"])
    assert out["band_available"].all()
    assert out.loc[0, "lower"] == pytest.approx(10.0 + b.loc[("distal", 1), "lo"])
    assert out.loc[1, "upper"] == pytest.approx(b.loc[("mid", 2), "hi"])
    unseen = pd.DataFrame({"zone": ["proximal"], "lead": [1], "predicted": [0.0]})
    out = apply_bands(unseen, b.reset_index(), ["zone", "lead"])
    assert not out.loc[0, "band_available"] and np.isnan(out.loc[0, "upper"])


def test_short_stratum_is_declared_not_available_or_takes_the_declared_fallback():
    rng = np.random.default_rng(6)
    t = _zoned(rng)
    t = pd.concat([t, pd.DataFrame({"zone": "distal", "lead": 5,
                                    "residual": rng.normal(-2.5, 2.0, size=10)})])
    b = mondrian_bands(t, ["zone", "lead"]).set_index(["zone", "lead"])
    short = b.loc[("distal", 5)]
    assert not short["available"] and np.isnan(short["lo"]) and np.isnan(short["hi"])
    assert short["units"] == 10 and short["n_min"] == 19 and "10 rows < 19" in short["reason"]
    pred = pd.DataFrame({"zone": ["distal"], "lead": [5], "predicted": [1.0]})
    out = apply_bands(pred, b.reset_index(), ["zone", "lead"])
    assert not out.loc[0, "band_available"] and np.isnan(out.loc[0, "lower"])
    fb = mondrian_bands(t, ["zone", "lead"], fallback=[["zone"]]).set_index(["zone", "lead"])
    assert fb.loc[("distal", 5), "available"]
    assert fb.loc[("distal", 5), "calibrated_on"] == "zone=distal"
    assert fb.loc[("distal", 1), "calibrated_on"] == "zone=distal,lead=1"
    with pytest.raises(ValueError, match="subset"):
        mondrian_bands(t, ["zone"], fallback=[["lead"]])


# ----------------------------------------------------------------------------- year shock
@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_common_year_shock_undercoverage_is_detected(k):
    """With 3-6 origins and a shared year shock, a band calibrated on rows looks fine under
    a random row split but under-covers a new origin. Leave-one-origin-out hits expose it;
    the cluster-aware band refuses to exist below 19 origins; the cluster-adjusted
    interval never certifies the naive band, and the iid interval that ignores the origins
    claims far more certainty than it has (design effect >> 1)."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(100 + k)
    reps = 60
    loo, true_cov, split_cov, flags, deffs = [], [], [], [], []
    for _ in range(reps):
        t = _shock_table(rng, k)
        naive = calibrate_band(t["residual"])
        assert naive.available                                   # k * 100 rows >= 19
        aware = calibrate_band(t["residual"], t["origin"])
        assert not aware.available and aware.units == k and "clusters" in aware.reason
        # the truth: one row from each of many fresh origins
        new = rng.normal(0, 2.0, size=4000) + rng.normal(0, 0.5, size=4000)
        true_cov.append(np.mean((new >= naive.lo) & (new <= naive.hi)))
        # a random row split ignores the clusters and reports nominal coverage
        idx = rng.permutation(len(t))
        half = calibrate_band(t["residual"].to_numpy()[idx[: len(t) // 2]])
        rest = t["residual"].to_numpy()[idx[len(t) // 2:]]
        split_cov.append(np.mean((rest >= half.lo) & (rest <= half.hi)))
        h = loco_hits(t, cluster="origin", cluster_aware=False)
        rep = coverage_report(h["hit"], h["origin"], h["side"])
        loo.append(rep["coverage"])
        flags.append(rep["point_flag"])
        assert rep["k_clusters"] == k and rep["share_unavailable"] == 0.0
        assert rep["iid_verdict"] in ("under", "consistent")
        assert rep["verdict"] != "over"
        deffs.append(rep["deff"])                 # misses come by origin, not by row
        assert rep["miss_below"] + rep["miss_above"] == pytest.approx(1 - rep["coverage"])
        # the cluster-aware leave-one-origin-out band is never available
        ha = loco_hits(t, cluster="origin", cluster_aware=True)
        ra = coverage_report(ha["hit"], ha["origin"])
        assert ra["share_unavailable"] == 1.0 and ra["verdict"] == "not_evaluable"
    assert abs(np.mean(split_cov) - 0.90) < 0.02                 # false reassurance
    assert np.mean(true_cov) < 0.85                              # the real under-coverage
    assert np.mean(loo) < 0.85                                   # ... which LOOO exposes
    # LOOO calibrates on k - 1 origins, so it reads at or below the k-origin band's truth
    assert np.mean(loo) <= np.mean(true_cov) + 0.02
    assert np.median(deffs) > 10
    assert np.mean(np.array(flags) == "under") >= 0.9


def test_cluster_adjusted_verdict_detects_undercoverage_with_many_clusters():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(8)
    t = _shock_table(rng, 60, m=20, shock=1.0, noise=1.0)
    h = loco_hits(t, cluster="origin", cluster_aware=True)
    assert h["band_available"].all()
    rep = coverage_report(h["hit"], h["origin"], h["side"])
    assert rep["verdict"] == "consistent" and rep["point_flag"] == "ok"
    # a band half as wide as calibrated, on the same rows
    mid = (h["lo"] + h["hi"]) / 2
    lo, hi = mid + (h["lo"] - mid) / 2, mid + (h["hi"] - mid) / 2
    r = h["residual"]
    narrow = ((r >= lo) & (r <= hi)).astype(float)
    rep2 = coverage_report(narrow, h["origin"])
    assert rep2["verdict"] == "under" and rep2["hi"] < 0.90
    assert coverage_report(narrow[:40], h["origin"][:40])["verdict"] == "not_evaluable"


def test_coverage_report_matches_the_registered_harness_interval():
    pytest.importorskip("scipy")
    path = Path(__file__).resolve().parents[1] / "dataprep_staging/harness/coverage.py"
    if not path.exists():
        pytest.skip("harness not present")
    spec = importlib.util.spec_from_file_location("_harness_coverage", path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    rng = np.random.default_rng(9)
    for k, rho in ((5, 0.4), (12, 0.1), (40, 0.0)):
        m = rng.integers(3, 25, size=k)
        p = np.clip(0.9 + rho * rng.normal(0, 0.3, size=k), 0, 1)
        hits = np.concatenate([rng.random(mi) < pi for mi, pi in zip(m, p, strict=True)])
        cl = np.repeat(np.arange(k), m)
        ours = coverage_report(hits.astype(float), cl)
        ref = harness.cluster_coverage_interval(hits, cl)
        for key in ("coverage", "rho", "deff", "n_eff", "lo", "hi"):
            assert ours[key] == pytest.approx(ref[key], rel=1e-12, abs=1e-12), key


# ----------------------------------------------------------------------------- censored
def test_interval_censored_hits_and_informative_count():
    floor = 3.0
    # band [lo, hi] against printed bounds: listed [3.5, 4.0], listed [6, inf), unlisted
    lower = np.array([3.0, 2.0, 0.0, np.nan])
    upper = np.array([3.6, 5.6, 2.0, np.nan])
    ol = np.array([3.5, 6.0, -np.inf, 4.0])
    ou = np.array([4.0, np.inf, floor, 5.0])
    r = interval_censored_hits(lower, upper, ol, ou, tol=0.5)
    np.testing.assert_array_equal(r["hits"][:3], [1.0, 1.0, 1.0])
    assert np.isnan(r["hits"][3])
    assert r["n_scored"] == 3 and r["n_informative"] == 2
    r0 = interval_censored_hits(lower, upper, ol, ou, tol=0.0)
    assert r0["hits"][1] == 0.0
    with pytest.raises(ValueError):
        interval_censored_hits([0.0], [1.0], [2.0], [1.0])


def test_missing_labels_and_column_clashes_are_refused():
    t = pd.DataFrame({"residual": [0.1, -0.2, 0.3], "origin": [1, None, 2], "zone": ["a"] * 3})
    with pytest.raises(ValueError, match="must not be missing"):
        mondrian_bands(t, ["zone"], cluster="origin")
    with pytest.raises(ValueError, match="must not be missing"):
        loco_hits(t, ["zone"], cluster="origin")
    good = t.assign(origin=[1, 1, 2])
    bands = mondrian_bands(good, ["zone"])
    with pytest.raises(ValueError, match="already have column"):
        apply_bands(pd.DataFrame({"predicted": [1.0], "zone": ["a"], "lo": [0.0]}), bands,
                    ["zone"])


def test_loco_hits_records_the_stratum_that_served():
    rng = np.random.default_rng(3)
    t = pd.DataFrame({"residual": rng.normal(size=400), "origin": np.repeat(range(20), 20),
                      "zone": np.tile(["a", "b"], 200)})
    h = loco_hits(t, ["zone"], cluster="origin")
    served = h.loc[h["band_available"], "calibrated_on"]
    assert len(served) and served.str.startswith("zone=").all()
    assert (h.loc[~h["band_available"], "calibrated_on"] == "").all()
