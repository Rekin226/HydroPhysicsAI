"""CPU unit tests for the WP A1 harness (synthetic data only; no repository data)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from harness import coverage, gaps, placebos, prereg, scorers, units

DATES = pd.date_range("2012-02-01", periods=131, freq="MS")


# ----------------------------------------------------------------------------- scorers
def _pilot_anom_rmse(o, p):
    r = o - p
    r = r - np.nanmean(r, axis=1, keepdims=True)
    return float(np.sqrt(np.nanmean(np.nanmean(r ** 2, axis=1))))


def test_anomaly_and_equal_well_rmse_match_pilot_formula():
    rng = np.random.default_rng(0)
    o = rng.normal(size=(6, 40))
    o[o > 1.5] = np.nan
    p = o + rng.normal(0, 0.3, size=o.shape) + np.arange(6)[:, None]
    assert scorers.anomaly_rmse(o, p)["rmse"] == pytest.approx(_pilot_anom_rmse(o, p), 1e-12)
    r = o - p
    pw = np.nanmean(r ** 2, axis=1)
    assert scorers.equal_well_rmse(o, p)["rmse"] == pytest.approx(np.sqrt(pw.mean()), 1e-12)
    assert scorers.equal_well_rmse(o, p, agg="mean_of_rmse")["rmse"] == \
        pytest.approx(np.sqrt(pw).mean(), 1e-12)
    # the datum-free score ignores constant offsets; the plain one does not
    assert scorers.anomaly_rmse(o, p + 5)["rmse"] == pytest.approx(
        scorers.anomaly_rmse(o, p)["rmse"], 1e-12)


def test_out_of_time_training_mean_datum_matches_pilot_and_beats_test_mean_leak():
    rng = np.random.default_rng(1)
    W, T, split = 5, 60, 40
    o = rng.normal(size=(W, T)) + np.arange(W)[:, None]
    p = o - 2.0 + rng.normal(0, 0.2, size=(W, T))
    o[0, :38] = np.nan                    # well 0: 2 training months only
    o[1, split:55] = np.nan               # well 1: 5 test months -> dropped (min_test 7)
    tr = np.arange(T) < split
    res = scorers.out_of_time_rmse(o, p, tr, ~tr)
    r = o - p
    dtrain = np.nanmean(r[:, :split], axis=1, keepdims=True)
    e = r[:, split:] - dtrain
    ok = np.isfinite(dtrain[:, 0]) & (np.isfinite(e).sum(1) > 6)
    ref = float(np.sqrt(np.nanmean(np.nanmean(e[ok] ** 2, axis=1))))
    assert res["rmse"] == pytest.approx(ref, 1e-12)
    assert res["n_wells"] == int(ok.sum()) == 4
    with pytest.raises(ValueError):
        scorers.out_of_time_rmse(o, p, tr, np.ones(T, bool))


def test_climatology_baseline_refuses_post_origin_fit():
    dates = pd.date_range("2012-01-01", "2024-12-01", freq="MS")
    o = np.tile(np.sin(np.arange(len(dates)) * 2 * np.pi / 12), (2, 1))
    b = scorers.climatology_baseline(o, dates, "2022-12")
    assert np.allclose(b, o)
    with pytest.raises(scorers.LeakageError):
        scorers.climatology_baseline(o, dates, "2024-12")


def test_coverage_native_and_conformal():
    rng = np.random.default_rng(2)
    n = 4000
    sd = rng.uniform(0.5, 2.0, n)
    err = rng.normal(0, 1, n) * sd * 1.3          # under-dispersed sd
    g = np.repeat(np.arange(400), 10)
    nat = scorers.coverage_native(err, sd)
    assert nat["coverage"] < 0.85
    cs = scorers.conformal_split(err, g, sd, seed=0)
    assert abs(cs["cov_oos"] - 0.90) < 0.02
    assert cs["sd_mult_full"] == pytest.approx(1.3, abs=0.08)
    hits, fold = scorers.conformal_crossfit_hits(err, g, sd, seed=0)
    assert abs(hits.mean() - 0.90) < 0.02 and set(fold) == set(range(5))
    assert np.isnan(scorers.coverage_native(err, np.where(np.arange(n) == 0, np.nan, sd))
                    ["coverage"])
    assert scorers.conformal_quantile([1, 2, 3], 0.9) == np.inf
    assert scorers.conformal_quantile(np.arange(1, 20), 0.9) == 18


def test_conformal_split_matches_pilot_recal():
    rng = np.random.default_rng(3)
    g = pd.Series(np.repeat([f"gap{i}" for i in range(30)], 7))
    err = pd.Series(rng.normal(size=len(g)))
    sd = pd.Series(rng.uniform(0.5, 1.5, len(g)))
    # the pilot's recal (s5c_ensemble.py), verbatim
    r2 = np.random.default_rng(1)
    gids = g.unique()
    z = np.abs(err / sd)
    cfull = np.quantile(z, 0.9) / 1.645
    cov = []
    for _ in range(20):
        cal = set(r2.choice(gids, max(len(gids) // 2, 1), replace=False))
        a = g.isin(cal)
        c = np.quantile(z[a], 0.9) / 1.645
        cov.append(np.mean(z[~a] <= 1.645 * c))
    res = scorers.conformal_split(err, g, sd, z=1.645, finite_sample=False, seed=1)
    assert res["cov_oos"] == pytest.approx(np.mean(cov), 1e-12)
    assert res["sd_mult_full"] == pytest.approx(cfull, 1e-12)
    assert res["width_median"] == pytest.approx(np.median(sd * cfull * 2 * 1.645), 1e-12)


def test_recovery_metrics_against_pilot_definitions():
    rng = np.random.default_rng(4)
    nb, NT = 3, 131
    truth = 100 + 20 * np.sin(np.arange(NT) * 2 * np.pi / 12)[None] + rng.normal(0, 2, (nb, NT))
    est = 0.8 * truth + rng.normal(0, 3, (nb, NT))
    n_k = np.array([10, 20, 30])
    m = scorers.recovery_metrics(est, truth, DATES, n_k)
    te, tt = est.sum(0), truth.sum(0)
    months = (np.arange(NT) + 1) % 12
    moy = lambda x: np.array([x[months == k].mean() for k in range(12)])  # noqa: E731
    assert m["bias_pct"] == pytest.approx(100 * (te.mean() / tt.mean() - 1), 1e-12)
    assert m["seasonal_amp_ratio"] == pytest.approx(np.ptp(moy(te)) / np.ptp(moy(tt)), 1e-12)
    assert m["annual_corr"] == pytest.approx(np.corrcoef(
        te[11:131].reshape(10, 12).mean(1), tt[11:131].reshape(10, 12).mean(1))[0, 1], 1e-12)
    flat = scorers.recovery_metrics(np.ones((1, NT)), truth[:1], DATES)
    assert flat["monthly_corr"] is None and flat["seasonal_amp_ratio"] == 0.0


# ----------------------------------------------------------------------------- coverage gate
def test_icc_and_design_effect():
    y = np.array([[10, 10, 0, 0]], float)      # perfectly clustered hits
    m = np.full(4, 10.0)
    assert coverage.icc_anova_batch(y, m)[0] == pytest.approx(1.0)
    y2 = np.array([[9, 9, 9, 9]], float)       # no between-cluster variance
    assert coverage.icc_anova_batch(y2, m)[0] == 0.0
    r = coverage.interval_from_totals_batch(y, m, 0.95, "z")
    assert r["n_eff"][0] == pytest.approx(4.0)  # deff = m = 10
    assert coverage.icc_anova_batch(np.array([[1, 0, 1]], float), np.ones(3))[0] == 0.0


def test_coverage_gate_rule_and_pooling():
    rng = np.random.default_rng(5)
    clusters = np.repeat(np.arange(200), 5)
    hits = rng.random(1000) < 0.90
    g = coverage.coverage_gate(hits, clusters)
    assert g["window_applies"] and g["pass"] == (g["contains_target"] and g["in_window"])
    bad = rng.random(1000) < 0.70
    assert not coverage.coverage_gate(bad, clusters)["pass"]
    small = {"a": (rng.random(10) < 0.9, np.arange(10)), "b": (rng.random(12) < 0.9,
                                                               np.arange(12) + 100),
             "c": (rng.random(400) < 0.9, np.arange(400) + 1000)}
    pooled = coverage.pool_small_strata(small, {"a": ["b", "c"], "b": ["a", "c"],
                                                "c": ["b"]})
    assert all(v["gate"]["n_eff"] >= 30 for v in pooled.values()) or len(pooled) == 1
    assert sorted(sum((v["members"] for v in pooled.values()), [])) == ["a", "b", "c"]


def test_cluster_interval_covers_true_rate_in_simulation():
    rng = np.random.default_rng(6)
    m = np.full(14, 13.0)
    ab = 1 / 0.2 - 1
    pi = rng.beta(0.9 * ab, 0.1 * ab, size=(2000, 14))
    Y = rng.binomial(13, pi).astype(float)
    r = coverage.interval_from_totals_batch(Y, m)
    assert np.mean((r["lo"] <= 0.9) & (r["hi"] >= 0.9)) >= 0.89


# ----------------------------------------------------------------------------- placebos
def test_placebos():
    v = np.sin(np.arange(131) * 2 * np.pi / 12) + np.linspace(0, 1, 131)
    s6 = placebos.shift_months(v, 6)
    assert np.allclose(np.sort(s6), np.sort(v)) and s6[6] == v[0]
    pad = placebos.shift_months(v, 6, "pad")
    assert np.isnan(pad[:6]).all() and pad[6] == v[0]
    clim = placebos.own_climatology(v, DATES, "2022-12")
    assert np.allclose(clim[DATES.month == 3], v[DATES.month == 3].mean())
    sh, mp = placebos.year_shuffle(v, DATES, seed=6000)
    assert all(a != b for a, b in mp.items()) and set(mp) == set(range(2013, 2023))
    assert np.allclose(placebos.own_climatology(sh, DATES), placebos.own_climatology(v, DATES))
    cxy = np.array([[0, 0], [20000, 0], [5000, 0]], float)
    val = np.ones((3, 4))
    assert np.allclose(placebos.distant_covariate(val, cxy, [0, 0]), 1.0)
    assert np.allclose(placebos.regional_total(val, [True, False, True]), 2.0)
    rain = 100 + 50 * np.sin(np.arange(131) * 2 * np.pi / 12)
    rain[60:63] = 0
    di = placebos.rain_deficit_index(rain, DATES, 3)
    assert np.isnan(di[:2]).all() and di[62] > 1


# ----------------------------------------------------------------------------- gaps
def test_head_masks_reproducible_and_seeded():
    rng = np.random.default_rng(7)
    obs = rng.random((60, 131)) < 0.95
    targets = list(range(60))
    a = list(gaps.head_masks("S", obs, targets))
    b = list(gaps.head_masks("S", obs, targets))
    assert len(a) == 4 and all(np.array_equal(x[1], y[1]) for x, y in zip(a, b, strict=True))
    assert gaps.seed_for("S", 2) == 1000 * ord("S") + 2 and gaps.seed_for("T", 1) == 7001
    for fam in ("N", "E", "H"):
        r, mask, cells = next(gaps.head_masks(fam, obs, targets))
        assert mask.any() and len(cells) > 0
    m, c = gaps.replica_test_period(obs, targets, 0, window=(86, 131))
    assert m[:, :86].sum() == 0 and m.any()
    assert all(obs[w, t] for _, _, w, t in c)


def test_post_origin_generators_refuse_without_flag():
    with pytest.raises(gaps.LeakageError):
        gaps.rolling_origin_july(range(2013, 2025))
    assert len(gaps.rolling_origin_july(range(2013, 2025), allow_post_origin=True)) == 12
    with pytest.raises(gaps.LeakageError):
        gaps.matched_lead_backcast((2020, 2021, 2022, 2023), (2016, 2017, 2018, 2019))
    k = gaps.matched_lead_backcast()
    assert k["leads"] == [4]
    with pytest.raises(gaps.LeakageError):
        gaps.daily_outage_windows(range(2013, 2026))
    assert len(gaps.daily_outage_windows(range(2013, 2023))) == 10
    assert gaps.daily_outage_windows([2016])[0]["days"] == 38   # leap year Feb


def test_injections_and_billing():
    Y = np.zeros((30, 400))
    Yi, ev = gaps.inject_offsets_steps(Y, 0, n_paired=5, n_steps=5)
    assert len(ev) == 10 and len({e["well"] for e in ev}) == 10
    for e in ev:
        if e["kind"] == "step":
            assert abs(Yi[e["well"], -1]) >= 3
        else:
            assert abs(Yi[e["well"], e["start"]]) >= 1 and Yi[e["well"], e["end"]] == 0
    s, steps = gaps.inject_gnss_steps(2000, 0)
    assert len(steps) == 3 and all(abs(x["size"]) >= 0.02 for x in steps)
    dates = pd.date_range("2013-01-01", periods=12, freq="MS")
    mon = np.arange(24, dtype=float).reshape(2, 12)
    tot, even = gaps.pair_sum(mon, dates, ["A", "B"])
    assert tot[0, 0] == tot[0, 1] == 1.0 and np.isnan(tot[1, 0]) and tot[1, 1] == 13 + 14
    assert len(gaps.ring_windows()) == 72
    inj = gaps.ring_instrument_injection([f"s{i}" for i in range(14)], 0)
    assert {x["group"] for x in inj} <= {"A", "B"}
    z = gaps.apply_ring_instrument([100.0], 0.5, 182e-6)
    assert z[0] == pytest.approx(100.5 + 0.0182)
    assert len(gaps.season_blocks([2016])) == 4 and len(gaps.canal_loyo()) == 7
    assert gaps.paired_bootstrap_indices(7, 10).shape == (10, 7)
    v = np.array([1.0, 3.0])
    assert np.allclose(gaps.cv_mismatch_rescale(v, 0.5), [1.5, 2.5])
    assert set(gaps.FAMILIES) >= {"D", "S", "E", "H", "N", "T", "O", "R", "Y", "G", "B", "J",
                                  "C", "X", "K", "W", "P", "L", "Q"}


# ----------------------------------------------------------------------------- units
def test_units_on_synthetic_grid():
    mask = np.ones((20, 30), bool)
    cen = units.centroids(mask, 0, 0, 1000)
    x = cen[:, 0] / 1000
    sel = x < 25
    ter = units.easting_terciles(x, sel)
    rot = units.rotated_terciles(x, sel)
    st = np.stack([rot["rot1"], rot["rot2"], rot["rot3"]])
    assert (st.sum(0) == sel).all()
    assert units.jaccard(rot["rot1"], ter["west"]) < 0.9
    km = units.kmeans_partition(cen / 1000, sel, 5, 2027)
    assert (np.stack(list(km.values())).sum(0) == sel).all()
    xy = np.array([[0, 0], [50, 0], [20000, 0], [20050, 0], [40000, 0]], float)
    site = units.well_sites(xy)
    assert site[0] == site[1] and site[2] == site[3] and len(set(site)) == 3
    f = units.spatial_block_folds(xy, list("abcde"), n_folds=2, buffer_km=5)
    assert f["fold_of"]["a"] == f["fold_of"]["b"]
    for fd in f["folds"]:
        assert not set(fd["test"]) & set(fd["train"])
    loco = units.spatial_block_folds(xy, list("abcde"), n_folds=None, buffer_km=5)
    assert loco["params"]["assign"] == "leave_one_cluster_out" and len(loco["folds"]) == 3
    assert units.temporal_block("2016-01", "2018-12")["n_months"] == 36


# ----------------------------------------------------------------------------- prereg
def _doc(pid, tests, kind="test"):
    na = prereg.not_applicable
    return {"id": pid, "wp": "TST", "title": "t", "kind": kind, "tests": tests,
            "zones": na("none"), "priors": na("none"), "burn_in": 12, "folds": na("none"),
            "fresh_units": ["fresh.u1"], "thresholds": {"bias": 0.3},
            "gate_scores": [{"name": "bias", "weight": 0.5}, {"name": "amp", "weight": 0.5}],
            "primary_model": "design 1", "inputs": na("none"), "interpretation": "x"}


def test_prereg_write_load_score_and_refusals(tmp_path):
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    reg = tmp_path / "units.json"
    reg.write_text(json.dumps({"units": [{"id": "fresh.u1", "role": "fresh"},
                                         {"id": "dt.u0", "role": "design_time"}]}))
    t = [{"test_id": "E1.a", "data_scope": "heads", "fresh_units": ["fresh.u1"]}]
    path, sha = prereg.write_prereg(_doc("P1", t), prereg_dir=pdir, rows_dir=rows,
                                    units_registry=reg)
    assert oct(os.stat(path).st_mode & 0o777) == "0o444"
    doc, sha2 = prereg.load_prereg("P1", prereg_dir=pdir, rows_dir=rows)
    assert sha2 == sha and doc["created"].endswith("Z")
    ok = prereg.assert_can_score("P1", "TST", prereg_dir=pdir, rows_dir=rows)
    assert ok["tests"] == ["E1.a"]
    # same test id cannot be re-registered (a changed design needs new ids)
    with pytest.raises(prereg.PreregistrationError):
        prereg.write_prereg(_doc("P2", t), prereg_dir=pdir, rows_dir=rows)
    # an unregistered scoring look taints the unit for later tests
    prereg.record_look(["fresh.u1"], "heads", "TST", "score", prereg_dir=pdir)
    t3 = [{"test_id": "E2.a", "data_scope": "heads", "fresh_units": ["fresh.u1"]}]
    with pytest.raises(prereg.PreregistrationError):
        prereg.write_prereg(_doc("P3", t3), prereg_dir=pdir, rows_dir=rows)
    # but a different data scope is not tainted
    t4 = [{"test_id": "E3.a", "data_scope": "leveling", "fresh_units": ["fresh.u1"]}]
    prereg.write_prereg(_doc("P4", t4), prereg_dir=pdir, rows_dir=rows)
    # design-time units cannot be fresh; weights must sum to 1; created cannot be supplied
    bad = _doc("P5", [{"test_id": "E4.a", "data_scope": "x", "fresh_units": ["dt.u0"]}])
    assert any("cannot be a fresh unit" in e for e in prereg.validate_prereg(bad, reg))
    bad2 = _doc("P6", t4)
    bad2["gate_scores"] = [{"name": "a", "weight": 0.7}]
    assert any("sum to" in e for e in prereg.validate_prereg(bad2))
    bad3 = dict(_doc("P7", t4), created="2020-01-01")
    assert any("created" in e for e in prereg.validate_prereg(bad3))
    # tampering is detected
    os.chmod(path, 0o644)
    path.write_text(path.read_text().replace('"t"', '"tampered"'))
    with pytest.raises(prereg.PreregistrationError):
        prereg.load_prereg("P1", prereg_dir=pdir, rows_dir=rows)
    # an edited look line breaks the chain
    lk = pdir / "looks.jsonl"
    lines = lk.read_text().splitlines()
    lines[0] = lines[0].replace("TST", "XXX")
    lk.write_text("\n".join(lines) + "\n")
    with pytest.raises(prereg.PreregistrationError):
        prereg.read_chained(lk)


def test_prereg_postdating_a_seeded_design_time_look_is_refused(tmp_path):
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    prereg.record_look(["fresh.u1"], "*", "A1", "design_time_seed",
                       seeded_time="2026-10-12T00:41:00.000000Z", prereg_dir=pdir)
    t = [{"test_id": "E1.b", "data_scope": "heads", "fresh_units": ["fresh.u1"]}]
    with pytest.raises(prereg.PreregistrationError):
        prereg.write_prereg(_doc("Q1", t), prereg_dir=pdir, rows_dir=rows)
    with pytest.raises(ValueError):
        prereg.record_look(["u"], "*", "A1", "score", seeded_time="2020-01-01T00:00:00Z",
                           prereg_dir=pdir)


# ----------------------------------------------------------------------------- additions
def test_sigma_obs_u2_preorigin_table():
    assert scorers.sigma_obs(20, 31) == pytest.approx(0.331143272089593, 1e-12)
    assert scorers.sigma_obs(31, 31) == pytest.approx(0.10)
    assert scorers.sigma_obs(28, 28) == pytest.approx(0.10)
    mid = scorers.sigma_obs(25, 30)
    assert mid == pytest.approx(0.331143272089593 + (0.10 - 0.331143272089593) * 0.5, 1e-12)
    assert scorers.sigma_obs(12.5, 30) == pytest.approx(
        (0.6505993940198698 + 0.4944701483871803) / 2, 1e-12)
    v = scorers.sigma_obs(np.array([3, 5, 10, 15, 20, 30]), 30)
    assert np.all(np.diff(v) < 0) and v[-1] == pytest.approx(0.10)
    for bad in ((2, 30), (31, 30), (20, 27)):
        with pytest.raises(ValueError):
            scorers.sigma_obs(*bad)
    assert scorers.u2_weight(0.0, 31, 31) == pytest.approx(1.0)
    so = scorers.sigma_obs(20, 30)
    assert scorers.u2_weight(so, 20, 30) == pytest.approx(0.5)


def test_replica_test_period_respects_avoid_window():
    rng = np.random.default_rng(8)
    obs = rng.random((80, 177)) < 0.97
    for r in range(3):
        m, c = gaps.replica_test_period(obs, list(range(80)), r, window=(132, 177),
                                        avoid=(169, 170))
        assert m[:, :132].sum() == 0
        single = [x for x in c if x[0].startswith("S")]
        assert not any(169 <= t <= 170 for _, _, _, t in single)
        assert {x[0] for x in c} <= {"S01", "S02", "S03", "S06", "S12", "H06"}


def test_coverage_confirm_mixture_and_sizes():
    from harness import coverage_confirm as cc
    assert cc.mix_icc() == pytest.approx(1 / 3)
    m = np.full(50, 20.0)
    Y = cc.draw_mix(m, 400, np.random.default_rng(0))
    assert Y.shape == (400, 50) and abs(Y.sum() / (400 * m.sum()) - 0.90) < 0.01
    s = cc.sizes("lognormal30_s1.5", 30, np.random.default_rng(1))
    assert s.size == 30 and s.min() >= 1
    assert len({c[0] for c in cc.CONFIGS}) == len(cc.CONFIGS)
    assert {c[5] for c in cc.CONFIGS} == {"primary", "secondary"}


def test_repro_rel_diff_semantics():
    from harness.repro_pilots import Tally, rel_diff
    assert rel_diff(1.0, 1.0) == 0.0 and rel_diff(None, float("nan")) == 0.0
    assert rel_diff(1.0, None) == np.inf and rel_diff(2e-7, 0.0) == pytest.approx(2e-7)
    t = Tally("x")
    t.add("a", 1.0 + 1e-9, 1.0)
    t.add("b", 1.1, 1.0)
    r = t.result()
    assert r["n_values"] == 2 and r["n_above_tol"] == 1 and not r["pass"]


def test_prereg_declaration_and_external_units(tmp_path):
    reg = tmp_path / "units.json"
    reg.write_text(json.dumps({"units": [{"id": "fresh.u1", "role": "fresh"}]}))
    na = prereg.not_applicable
    dec = {"id": "D1", "wp": "TST", "title": "d", "kind": "declaration",
           "tests": na("declaration"), "zones": na("z"), "priors": na("p"),
           "burn_in": na("b"), "folds": na("f"), "fresh_units": ["fresh.u1"],
           "thresholds": na("t"), "gate_scores": na("g"), "primary_model": na("m"),
           "inputs": na("i"), "interpretation": "x"}
    assert prereg.validate_prereg(dec, reg) == []
    t = _doc("T1", [{"test_id": "X.a", "data_scope": "sim", "fresh_units": ["sim.s1"]}])
    assert any("neither in the units registry" in e for e in prereg.validate_prereg(t, reg))
    t["tests"][0]["external_units"] = {"sim.s1": "seeds 1..10"}
    assert prereg.validate_prereg(t, reg) == []
    with pytest.raises(ValueError):
        prereg.not_applicable("")
