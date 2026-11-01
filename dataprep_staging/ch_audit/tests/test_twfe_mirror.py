"""CPU tests of X5.CH_twfe_mirror.v1: synthetic panels, fake forward files and tmp registries
only. No head, census or model output is read; nothing is written to the real registry.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/ch_audit/tests
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for p in (str(REPO), str(X), str(REPO / "dataprep_staging")):
    if p not in sys.path:
        sys.path.insert(0, p)
import twfe_lib as T  # noqa: E402
import twfe_mirror as M  # noqa: E402

DECL = REPO / f"results/dataprep/preregistration/{M.DECL_ID}.json"


# ----------------------------------------------------------------------------- helpers
def _world(seed=0, sites=(3, 2, 2, 1), W_layers=2, beta=-0.8, noise=0.0):
    rng = np.random.default_rng(seed)
    site, town = [], []
    s = 0
    for t, n in enumerate(sites):
        for _ in range(n):
            for _ in range(W_layers):
                site.append(s)
                town.append(t)
            s += 1
    site, town = np.array(site), np.array(town)
    W, Y = len(site), len(T.YEARS)
    Xs = rng.uniform(1, 5, (s, len(T.REG_YEARS))) + rng.normal(0, 0.2, (s, 1)) * np.arange(
        len(T.REG_YEARS))
    Xw = Xs[site]
    a = rng.normal(0, 2, (W, 1))
    g = rng.normal(0, 1, (town.max() + 1, Y))[town]
    H = a + g + beta * T.on_years(Xw) + noise * rng.standard_normal((W, Y))
    return site, town, Xw, H


# ----------------------------------------------------------------------------- estimator
def test_beta_exact_without_noise_and_residuals_orthogonal():
    site, town, Xw, H = _world()
    p = T.make_panel(site, town, np.ones(H.shape, bool))
    x = Xw[:, 5:][p.well, p.year]
    b, xtx = T.beta_fe(x, H[p.well, p.year], p)
    assert float(b) == pytest.approx(-0.8, abs=1e-9) and xtx > 0
    xt = T.resid_fe(x, p.well, p.group())
    assert np.abs(np.bincount(p.well, xt)).max() < 1e-9
    assert np.abs(np.bincount(np.unique(p.group(), return_inverse=True)[1], xt)).max() < 1e-9


def test_unbalanced_panel_converges_and_singletons_do_not_identify():
    site, town, Xw, H = _world(sites=(1, 1, 1))
    p = T.make_panel(site, town, np.ones(H.shape, bool))
    b, xtx = T.beta_fe(Xw[:, 5:][p.well, p.year], H[p.well, p.year], p)
    assert np.isnan(b) and xtx < 1e-9
    assert T.identifying(p)["n_identifying_townships"] == 0
    site, town, Xw, H = _world(sites=(3, 2))
    mask = np.random.default_rng(1).random(H.shape) > 0.3
    p = T.make_panel(site, town, mask)
    b, _ = T.beta_fe(Xw[:, 5:][p.well, p.year], H[p.well, p.year], p)
    assert float(b) == pytest.approx(-0.8, abs=1e-8)


def test_bootstrap_plans():
    site, town, Xw, H = _world()
    p = T.make_panel(site, town, np.ones(H.shape, bool))
    for d in T.bootstrap_plan(p, B=20, seed=1):
        assert len(d.rows) == p.n                       # each township keeps its site count
        for t in np.unique(town):
            assert (p.town[d.rows] == t).sum() == (p.town == t).sum()
    for d in T.bootstrap_plan(p, B=20, seed=1, kind="township"):
        assert len(np.unique(d.group // 10000 % 1000)) >= 1
    a = T.bootstrap_plan(p, B=3, seed=5)
    b = T.bootstrap_plan(p, B=3, seed=5)
    assert all((x.rows == y.rows).all() for x, y in zip(a, b, strict=True))


def test_boot_betas_shared_and_per_column_regressors_agree():
    site, town, Xw, H = _world(noise=0.3)
    p = T.make_panel(site, town, np.ones(H.shape, bool))
    x = Xw[:, 5:][p.well, p.year]
    Y = np.column_stack([H[p.well, p.year], H[p.well, p.year] + 1.0])
    plan = T.bootstrap_plan(p, B=30, seed=2)
    b1 = T.boot_betas(plan, x, Y)
    b2 = T.boot_betas(plan, np.column_stack([x, x]), Y)
    np.testing.assert_allclose(b1, b2, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(b1[:, 0], b1[:, 1], rtol=1e-8, equal_nan=True)


def test_kernel_energy():
    E = np.array([[2.0, 4.0]])
    out = T.kernel_energy([0.0, 3000.0, 9001.0], [0.0, 0.0, 0.0], [0.0], [0.0], E, 3.0)
    np.testing.assert_allclose(out[0], E[0])
    np.testing.assert_allclose(out[1], E[0] * np.exp(-0.5))
    np.testing.assert_allclose(out[2], 0.0)


def test_annual_from_monthly_flags_and_minimum():
    months = pd.date_range("2012-01-01", "2022-12-01", freq="MS")
    V = np.ones((2, len(months)))
    F = np.zeros((2, len(months)), int)
    F[0, :4] = 3                                        # 8 good months in 2012: missing
    F[1, :3] = 9                                        # 9 good months in 2012: kept
    V[1, 3:12] = 2.0
    a = T.annual_from_monthly(V, F, months.year.to_numpy())
    assert np.isnan(a[0, 0]) and a[1, 0] == pytest.approx(2.0)
    assert np.isfinite(a[:, 1:]).all()


def test_shifted_alignment():
    Xw = np.arange(len(T.REG_YEARS), dtype=float)[None, :] + np.zeros((2, 1))
    s = T.shifted(Xw, +2)
    assert s[0, 0] == T.REG_YEARS.index(2014) and np.isnan(s[0, -1]) and np.isnan(s[0, -2])
    assert T.shifted(Xw, -2)[0, 0] == T.REG_YEARS.index(2010)


def test_permutations_stay_within_township_and_move_sites_whole():
    site, town, Xw, H = _world(sites=(3, 2, 1))
    p = T.make_panel(site, town, np.ones(H.shape, bool))
    for src in T.permutations(p, len(site), P=50):
        assert (town[src] == town).all()
        for s in np.unique(site):
            assert len(set(site[src[site == s]])) == 1


def test_pooled_permutation_slope_is_degenerate_so_it_never_gates():
    """The review's finding: the pooled slope is ~0 whatever y is (strong response, pure noise,
    a nonlinear response), so it is report-only and the placebo verdict ignores it."""
    site, town, Xw, H = _world(sites=(3, 3, 2, 2, 4), beta=-5.0, noise=0.0)
    p = T.make_panel(site, town, np.ones(H.shape, bool))
    Xy = T.on_years(Xw)
    perms = T.permutations(p, len(site), P=400)
    rng = np.random.default_rng(9)
    x = Xy[p.well, p.year]
    b_true = float(T.beta_fe(x, H[p.well, p.year], p)[0])
    for y in (H[p.well, p.year], rng.standard_normal(p.n), x ** 2):
        _, pooled = T.placebo_perm(Xy, perms, y, p)
        b = float(T.beta_fe(x, y, p)[0])
        assert abs(pooled[0]) < 0.1 * max(abs(b), abs(b_true) * 0.05)
    plc = T.placebos(Xy, {2: T.shifted(Xw, 2), -2: T.shifted(Xw, -2)}, H, site, town,
                     halfwidth=1e-9, mode="literal")
    assert not plc["passed"] and "permutation_report_only" in plc
    assert "passed" not in plc["permutation_report_only"]


def test_ramp_hits_differ_within_every_identifying_township():
    site, town, _, _ = _world(sites=(3, 2, 4, 1, 2))
    rng = np.random.default_rng(3)
    single = []
    for _ in range(50):
        hit = T.ramp_hits(site, town, rng)
        for t in np.unique(town):
            ss = np.unique(site[town == t])
            hs = np.unique(site[(town == t) & hit])
            if len(ss) >= 2:
                assert 1 <= len(hs) == max(1, len(ss) // 2) < len(ss)
            else:
                single.append(len(hs))
        for s in np.unique(site):                     # screens of a site move together
            assert len(set(hit[site == s])) == 1
    assert 0 < np.mean(single) < 1
    r = T.ramp_profile()
    assert r[T.YEARS.index(2016)] == 0 and r[-1] == pytest.approx(T.RAMP_FRAC_2022)


def test_site_ramp_biases_the_slope_where_a_township_ramp_cannot():
    """With persistent site levels of local energy (as kernel energy has), a ramp at half the
    sites of a township leaves a site-specific trend the fixed effects cannot remove; the same
    ramp on whole townships only rescales the within-township deviations."""
    rng = np.random.default_rng(0)
    site, town, _, _ = _world(sites=(3, 3, 2, 2, 4))
    ns = site.max() + 1
    level = rng.uniform(2, 20, (ns, 1))
    Xs = level * (1 + 0.1 * rng.standard_normal((ns, len(T.REG_YEARS))))
    Xy = T.on_years(Xs[site])
    W, Y = Xy.shape
    p = T.make_panel(site, town, np.ones((W, Y), bool))
    a = rng.normal(0, 2, (W, 1))
    g = rng.normal(0, 1, (town.max() + 1, Y))[town]
    y = (a + g - 0.8 * Xy)[p.well, p.year]
    r = T.ramp_profile()
    b0 = float(T.beta_fe(Xy[p.well, p.year], y, p)[0])
    bias_site, bias_town = [], []
    for _ in range(20):
        hs = T.ramp_hits(site, town, rng)
        ht = np.isin(town, np.flatnonzero(rng.random(town.max() + 1) < 0.5))
        for hit, out in ((hs, bias_site), (ht, bias_town)):
            xr = (Xy * np.where(hit[:, None], 1 - r[None, :], 1.0))[p.well, p.year]
            out.append(abs(float(T.beta_fe(xr, y, p)[0]) - b0))
    assert np.mean(bias_site) > 2 * np.mean(bias_town)


def test_estimate_recovers_kappa_on_the_same_rows():
    site, town, Xw, H = _world(sites=(3, 3, 2, 2, 4), noise=0.0)
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, (len(site), 1))
    Yobs = a + 0.5 * (H - H.mean(axis=1, keepdims=True)) + 0.02 * rng.standard_normal(H.shape)
    Yobs[rng.random(H.shape) < 0.1] = np.nan
    est = T.estimate(T.on_years(Xw), Yobs, site, town, B=200, Hm={"R1c": H})
    m = est["models"]["R1c"]
    assert m["kappa"] == pytest.approx(0.5, abs=0.02)
    assert m["kappa_lo"] <= 0.5 <= m["kappa_hi"] and m["model_ci_excludes_0"]


DES = {"n_identifying_townships": 5, "n_identifying_sites": 15, "n_identifying_rows": 400}


def _est(excl0=True, lo=0.2, hi=0.61, design=None, beta=-0.3, deg=0.0):
    return {"beta": beta, "degenerate_frac": deg, "design": dict(design or DES),
            "models": {"R1c": {"model_ci_excludes_0": excl0, "kappa": (lo + hi) / 2,
                               "kappa_lo": lo, "kappa_hi": hi}}}


def test_reading_order_and_wording():
    fail = {"passed": False, "coverage_base": 0.7, "coverage_ramp": 0.9, "power": 0.5,
            "coverage_persistent": 0.9, "kappa_coverage_1": 0.8, "kappa_coverage_033": 0.9,
            "placebo_mode": None, "placebo_pass_rate": {}}
    r = T.reading(fail, None, None)
    assert r["key"] == "not_identifiable_g0" and "coverage_base 0.7" in r["text"]
    assert "kappa_coverage_1 0.8" in r["text"]
    ok = {"passed": True, "placebo_mode": "conditional", "design": dict(DES),
          "placebo_false_failure_rate": 0.22}
    assert T.reading(ok, _est(excl0=False), None)["key"] == "not_identifiable_model"
    plc = {"passed": False, "shifts": {"2": {"passed": False}, "-2": {"passed": True}}}
    r = T.reading(ok, _est(), plc)
    assert r["key"] == "not_identifiable_placebo" and "shift 2" in r["text"]
    assert "22%" in r["text"] and "not evidence against R1c" in r["text"]
    plc["passed"] = True
    r = T.reading(ok, _est(), plc)
    assert r["key"] == "narrowing" and "kappa [0.20, 0.61] of R1c's per GWh" in r["text"]
    assert "not a correction" in r["text"]
    assert T.reading(ok, _est(lo=-1.2, hi=-0.1), plc)["key"] == "opposite_sign"
    r = T.reading(ok, _est(lo=0.1, hi=1.4), plc)
    assert r["key"] == "uninformative" and "covers both 0.33 and 1" in r["text"]


def test_observed_design_check_precedes_the_model_step():
    ok = {"passed": True, "placebo_mode": "literal", "design": dict(DES)}
    lost = dict(DES, n_identifying_sites=14, n_identifying_rows=390)
    r = T.reading(ok, _est(excl0=False, design=lost), None)
    assert r["key"] == "not_identifiable_design" and "n_identifying_sites 14" in r["text"]
    thin = dict(DES, n_identifying_rows=290)
    assert T.reading(ok, _est(design=thin), None)["key"] == "not_identifiable_design"
    r = T.reading(ok, _est(beta=float("nan"), deg=1.0), None)
    assert r["key"] == "not_identifiable_design" and "not finite" in r["text"]
    assert T.design_check(ok, _est(design=dict(DES, n_identifying_rows=300)))["passed"]


def test_kappa_bands():
    assert T.kappa_band(-2.0, -0.01) == "opposite_sign"
    assert T.kappa_band(-2.0, 0.1) == "narrowing"
    assert T.kappa_band(0.3, 1.0) == "uninformative"
    assert T.kappa_band(0.34, 1.5) == "narrowing"
    assert T.kappa_band(float("nan"), 1.0) == "uninformative"


def test_g0_degenerate_design_fails_without_simulation():
    site, town, Xw, H = _world(sites=(1, 1, 1))
    noise = {"sigma_m": 1.0, "block_share": 0.4, "phi": 0.8}
    res = T.g0(H, T.on_years(Xw), site, town, np.zeros(len(site), int), noise,
               {2: T.shifted(Xw, 2), -2: T.shifted(Xw, -2)}, draws=5, B=10,
               report_arms=False)
    assert res["passed"] is False and "degenerate" in res["reason"]


def test_g0_small_run_structure_and_power_logic():
    site, town, Xw, H = _world(sites=(3, 3, 2, 2, 4), noise=0.0)
    noise = {"sigma_m": 0.05, "block_share": 0.4, "phi": 0.5}
    res = T.g0(H, T.on_years(Xw), site, town, town.copy(), noise,
               {2: T.shifted(Xw, 2), -2: T.shifted(Xw, -2)}, draws=40, B=150,
               report_arms=False)
    assert res["beta_model"] == pytest.approx(-0.8, abs=1e-8)
    assert res["power"] == pytest.approx(1.0)
    assert res["coverage_base"] >= 0.7
    assert res["kappa_coverage_1"] >= 0.7 and res["kappa_coverage_033"] >= 0.7
    assert res["kappa_power_report"]["excl_033_when_1"] == pytest.approx(1.0)
    assert res["median_kappa_width_base"] < 0.3
    for k in ("coverage_ramp", "coverage_persistent", "placebo_pass_rate", "placebo_mode",
              "placebo_false_failure_rate", "permutation_pass_rate_report_only", "passed"):
        assert k in res
    assert set(res["gates"]) == {"coverage_base", "coverage_ramp", "coverage_persistent",
                                 "kappa_coverage_1", "kappa_coverage_033", "power",
                                 "placebo", "degenerate"}
    assert res["passed"] == all(res["gates"].values())



def test_noise_model_moments():
    rng = np.random.default_rng(0)
    m = T.monthly_noise(rng, np.repeat(np.arange(20), 2), 1.3, 0.44, 0.86, 200)
    assert m.std() == pytest.approx(1.3, rel=0.05)
    c = np.corrcoef(m[0].ravel(), m[1].ravel())[0, 1]
    assert c == pytest.approx(0.44, abs=0.06)
    a = T.annual_noise(rng, np.repeat(np.arange(20), 2), 1.3, 0.44, 0.86, 200)
    assert a.std() == pytest.approx(T.annual_mean_sd(1.3, 0.86), rel=0.08)


def test_persistent_noise_moments():
    rng = np.random.default_rng(1)
    blk = np.repeat(np.arange(30), 2)
    e = T.persistent_noise(rng, blk, 1.3, 0.44, 0.86, 400)
    assert e.std() == pytest.approx(T.annual_mean_sd(1.3, 0.86), rel=0.05)
    lag1 = np.corrcoef(e[:, :-1].ravel(), e[:, 1:].ravel())[0, 1]
    assert lag1 == pytest.approx(T.PERSISTENT_PHI_ANNUAL, abs=0.05)
    c = np.corrcoef(e[0].ravel(), e[1].ravel())[0, 1]       # same block
    assert c == pytest.approx(0.44, abs=0.08)


# ----------------------------------------------------------------------------- driver
def test_model_annual_reads_baseline_hindcast_only(tmp_path):
    dates = np.array([d.strftime("%Y-%m") for d in pd.date_range("2012-01-01", "2032-12-01",
                                                                   freq="MS")])
    S, L, A = 2, 3, 5
    hm = np.zeros((S, L, A, len(dates)), np.float32)
    hm[0] = 1.0
    hm[0, :, :, 132:] = 99.0                            # forecast months must be ignored
    hm[1] = -7.0
    f = tmp_path / "fwd.npz"
    np.savez(f, dates=dates, origin=131, scenario_names=np.array(["baseline", "all30"]),
             heads_mean=hm, other=np.zeros(1))
    H, Hmon, info = M.model_annual(f, [0, 4], [1, 2])
    assert H.shape == (2, len(T.YEARS)) and np.allclose(H, 1.0)
    assert Hmon.shape == (2, 132) and np.allclose(Hmon, 1.0)
    assert info["n_months"] == 132
    np.savez(f, dates=dates, origin=130, scenario_names=np.array(["baseline"]),
             heads_mean=hm[:1])
    with pytest.raises(SystemExit, match="origin"):
        M.model_annual(f, [0], [1])


def test_build_design_and_observed_annual():
    wells = pd.DataFrame({"sid": ["a1", "a2", "b1"], "cell": [1, 1, 2], "layer": [1, 2, 1],
                          "county": ["changhua"] * 3})
    meta = pd.DataFrame({"sid": ["a1", "a2", "b1"], "site_id": ["A", "A", "B"],
                         "TownName": ["T1", "T1", None], "poly_town": ["T1", "T1", "T1"],
                         "x_twd97": [200000.0, 200000.0, 203000.0],
                         "y_twd97": [2640000.0] * 3})
    pumps = {"px": np.array([200000.0]), "py": np.array([2640000.0]),
             "E": np.ones((1, len(T.REG_YEARS)))}
    d = M.build_design(wells, meta, pumps)
    assert d["site"][0] == d["site"][1] != d["site"][2]
    assert len(set(d["town"])) == 1                     # poly_town fills the missing name
    assert d["X3"][0, 0] == pytest.approx(1.0) and d["X3"][2, 0] == pytest.approx(np.exp(-0.5))
    heads = pd.DataFrame({"sid": ["a1"] * 12 + ["b1"] * 12,
                          "date": list(pd.date_range("2015-01-01", periods=12, freq="MS")) * 2,
                          "value": [1.0] * 12 + [3.0] * 12, "flag": [0] * 12 + [2] * 12})
    Y = M.observed_annual(heads, ["a1", "a2", "b1"])
    assert Y[0, T.YEARS.index(2015)] == pytest.approx(1.0)
    assert np.isnan(Y[2]).all() and np.isnan(Y[1]).all()


def test_model_annual_mean_uses_the_observed_months():
    months = pd.date_range("2012-01-01", "2022-12-01", freq="MS")
    rows = []
    for i, d in enumerate(months[36:48]):              # 2015: drop Jan-Feb, keep 10 months
        if i >= 2:
            rows.append({"sid": "a", "date": d, "value": 5.0, "flag": 0})
    for d in months[48:56]:                            # 2016: 8 months only -> missing
        rows.append({"sid": "a", "date": d, "value": 5.0, "flag": 1})
    Y, okm = M.observed_annual(pd.DataFrame(rows), ["a"], return_mask=True)
    assert okm.shape == (1, 132) and okm[0].sum() == 10
    Hmon = np.arange(132, dtype=float)[None, :]
    Hm = M.masked_model_annual(Hmon, okm)
    assert Hm[0, T.YEARS.index(2015)] == pytest.approx(np.mean(np.arange(38, 48)))
    assert np.isnan(Hm[0, T.YEARS.index(2016)]) and np.isnan(Y[0, T.YEARS.index(2016)])
    assert np.isfinite(Hm) .sum() == np.isfinite(Y).sum() == 1


def test_g0_json_must_belong_to_the_registration():
    reg = {"sha256": "a" * 64, "doc": {"created": "2026-11-02T10:00:00+00:00"}}
    good = {"declaration_sha256": "a" * 64, "time": "2026-11-02T11:00:00+00:00"}
    M.check_g0(good, reg)
    with pytest.raises(SystemExit, match="registered declaration"):
        M.check_g0(dict(good, declaration_sha256=None), reg)
    with pytest.raises(SystemExit, match="postdate"):
        M.check_g0(dict(good, time="2026-11-02T09:00:00+00:00"), reg)


def test_smoke_end_to_end(tmp_path):
    out = M.cmd_smoke(tmp_path / "s", draws=20, B=60)
    assert "g0" in out and out["score"]["reading"]["key"] in T.READINGS
    assert "yunlin_parity" in out["score"]["report_only"]
    assert "may" not in out["score"]["report_only"]["note"]
    assert (tmp_path / "s" / "smoke_report.json").exists()


def test_runner_syntax_and_refusals():
    subprocess.run(["bash", "-n", str(X / "run_twfe.sh")], check=True)
    if not DECL.exists():
        for cmd in ("check", "g0", "score"):
            p = subprocess.run(["bash", str(X / "run_twfe.sh"), cmd], cwd=REPO,
                               capture_output=True, text=True, env={**os.environ})
            assert p.returncode == 1 and f"declare {M.DECL_ID} first" in p.stderr


def test_declaration_validates_and_registers_in_a_tmp_registry(tmp_path, monkeypatch):
    import prereg_twfe as P
    from harness import prereg3
    fz = {}
    for name in ("design.npz", "model_R1c.npz"):
        f = tmp_path / name
        f.write_bytes(name.encode())
        fz[str(f)] = M.sha256(f)
    monkeypatch.setattr(P, "frozen_inputs", lambda repo=REPO: fz)
    audit = {"path": M.AUDIT, "present": True, "sha256": "x", "D1": "model_census_v1_as_is",
             "D1b": "cap_immaterial",
             "annual_sum_max_rel_change": {"changhua": 0.001, "within": True},
             "forcing_parity": {"rel_diff": 0.003, "within": True}}
    monkeypatch.setattr(P, "audit_coupling", lambda repo=REPO: audit)
    d = P.doc()
    reg = tmp_path / "reg"
    assert prereg3.validate_prereg(d, prereg_dir=reg) == []
    assert d["kind"] == "declaration" and d["code_sha256"] == M.code_sha256()
    assert "kappa" in d["evidence_and_ledger_mapping"]["narrowing"]
    mp = d["evidence_and_ledger_mapping"]
    assert "C13a stays W" in mp["unchanged"]
    assert " may " not in json.dumps(mp) and "never README" in mp["yunlin_parity"]
    for k in T.READINGS:
        assert any(k in m or k.replace("not_identifiable", "") in m for m in mp), k
    assert "permutation_report_only" in d["design"]["placebos"]
    assert "persistent" in d["g0"]["arms"] and "kappa_coverage_1_min" in d["g0"]["gate"]
    for k in ("unlogged 2026-10-25", "I1_changhua", "N1.cpu_response", "step B"):
        assert any(k in s for s in d["rule_risks_disclosure"]), k
    pend = P.readiness(d, prereg_dir=reg)
    assert all("exists before" in s for s in pend) or pend == []
    for bad in ({"D1b": "cap_member_required"}, {"forcing_parity": {"within": False}}):
        d2 = json.loads(json.dumps(d))
        d2["census_audit_coupling"].update(bad)
        assert any(("cap-1.5" in s) or ("forcing" in s) for s in P.readiness(d2,
                                                                             prereg_dir=reg))
    d["inputs"] = [{"path": c, "sha256": h} for c, h in M.code_sha256().items()]
    path, sha = prereg3.write_prereg(d, prereg_dir=reg, rows_dir=reg / "_rows")
    assert path.exists() and len(sha) == 64
    r = M.registered(prereg_dir=reg)
    assert r["sha256"] == sha
    rec = M.log_look({"changhua": ["a"], "yunlin": ["b"]}, prereg_dir=reg)
    assert rec["purpose"] == "design_time_evidence" and rec["prereg_id"] == M.DECL_ID
    with pytest.raises(prereg3.PreregistrationError):
        prereg3.write_prereg(d, prereg_dir=reg, rows_dir=reg / "_rows")
    # a changed frozen input refuses
    Path(next(iter(fz))).write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differ"):
        M.registered(prereg_dir=reg)


def test_prereg_write_needs_the_self_sha(tmp_path, monkeypatch, capsys):
    import prereg_twfe as P
    monkeypatch.setattr(P, "frozen_inputs", lambda repo=REPO: {"x": "missing"})
    rc = P.main(["--write", "--quiet", "--expect-self-sha", "0" * 64],
                prereg_dir=tmp_path / "reg", rows_dir=tmp_path / "reg/_rows")
    assert rc == 1 and "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "reg" / f"{M.DECL_ID}.json").exists()


def test_json_of_the_declaration_is_serialisable(monkeypatch):
    import prereg_twfe as P
    monkeypatch.setattr(P, "frozen_inputs", lambda repo=REPO: {"x": "missing"})
    json.dumps(P.doc(), sort_keys=True, ensure_ascii=False)
