"""N2CS (round-8 rank #6): the one-county r builder, the two-contrast estimator wrapping the
pinned N2/N2C scorer, the readings and informative gate, the OSSE engine (worlds, G0, choice),
the OSSE declaration, the draft registration and the scorer's refusals and checks.

Everything runs on synthetic arrays and synthetic product directories in pytest's tmp dir; no
model output, census table or head is read. Checks that need the gitignored registry
(results/dataprep) skip when it is absent. numpy's BLAS is pinned to one thread (a loaded host
hung at default threading).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
X = REPO / "dataprep_staging" / "n2cs"
for _p in (str(X), str(REPO / "dataprep_staging" / "n2_policy"), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bootstrap_scorer as BS  # noqa: E402
import n2cs_const as K  # noqa: E402
import n2cs_lib as L  # noqa: E402

PREREG = REPO / "results" / "dataprep" / "preregistration"
NEED_REGISTRY = pytest.mark.skipif(not (PREREG / f"{K.N2C_ID}.json").exists(),
                                   reason="the pre-registration registry is gitignored, absent")
REGISTERED = (PREREG / f"{K.ID}.json").exists()
PARAMS = {"sigma_tot_m": 0.3, "block_share": 0.44, "phi": 0.86, "kappa_m": 0.1}


@pytest.fixture(autouse=True, scope="module")
def _one_blas_thread():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(1):
        yield


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- constants
def test_constants_and_window():
    assert len(K.FORMS) <= 3 and K.FORMS[0] == "twoway"
    assert K.K_GRID == (0.2, 1.0 / 3.0, 0.5, 1.0, 1.2)
    assert K.K_TRUE in K.K_GRID and K.K_OVER in K.K_GRID
    assert L.WINDOW_MONTHS[0] == "2026-12" and L.WINDOW_MONTHS[-1] == "2027-10"
    assert len(L.WINDOW_MONTHS) == K.N_WINDOW == 11
    axis = L.month_list("2012-01", "2032-12")
    assert len(axis) == K.N_T
    assert axis[K.FWD_WINDOW[0]:K.FWD_WINDOW[1]] == L.WINDOW_MONTHS
    assert set(K.GATE_VARIANTS) == {"uniform", "shaped", "shaped_thinned25", "shaped_drop15"}
    for v in K.GATE_VARIANTS.values():
        assert K.CONFOUNDERS[v["conf"]]["rho"] == -0.83
    assert K.CONFOUNDERS["shaped"]["load_rho"] == 0.5
    assert K.MAX_FILLED_FRACTION == 0.2
    assert "no correction factor follows" in K.WORDING
    assert not hasattr(K, "GOV_ENTRY_PROPOSED") and not hasattr(K, "GOV_KEY")


def test_pinned_files_unchanged():
    assert sha(REPO / K.SCORER) == K.SCORER_SHA
    assert sha(REPO / K.BUILD_R_BOTH) == K.BUILD_R_BOTH_SHA
    assert sha(REPO / K.BUILD_R) == K.BUILD_R_SHA
    assert sha(REPO / K.REPLAY) == K.REPLAY_SHA
    assert sha(REPO / K.SYNTH) == K.SYNTH_SHA
    assert sha(REPO / K.GOV) == K.GOV_SHA
    assert sha(REPO / K.E02C_SCORER) == K.E02C_SCORER_SHA


# ----------------------------------------------------------------------------- builder
def _builder():
    import build_r_county as BR

    return BR


def test_assemble_one_county_fields_are_complements(tmp_path):
    BR = _builder()
    rng = np.random.default_rng(3)
    flags = rng.random(50) < 0.4
    months = BR.target_months()
    assert months[0] == "2023-01" and months[-1] == "2027-10" and len(months) == 58
    rc, ry = rng.uniform(0.7, 1.3, 58), rng.uniform(0.7, 1.3, 58)
    cha = BR.assemble(flags, months, rc, ry, "changhua")
    yla = BR.assemble(flags, months, rc, ry, "yunlin")
    ch, yl = cha["multiplier"], yla["multiplier"]
    both = np.where(flags[None, :], rc[:, None], ry[:, None])
    np.testing.assert_array_equal(ch * yl, both)
    np.testing.assert_array_equal(ch[:, ~flags], 1.0)
    np.testing.assert_array_equal(yl[:, flags], 1.0)
    unit, rec = BR.build_unit(flags, "yunlin")
    np.testing.assert_array_equal(unit["multiplier"], 1.0)
    assert str(unit["start"]) == "2023-01" and rec["mode"] == "unit"
    with pytest.raises(ValueError):
        BR.assemble(flags, months, rc, ry, "both")
    # S7: the per-cell path check on the files as written
    p_ch, p_yl = tmp_path / "ch.npz", tmp_path / "yl.npz"
    np.savez(p_ch, **cha)
    np.savez(p_yl, **yla)
    assert BR.arm_problems(p_ch, flags, "changhua") == []
    assert BR.arm_problems(p_yl, flags, "yunlin") == []
    assert BR.arm_problems(p_ch, flags, "yunlin")          # the wrong county moves
    both_arr = dict(cha, multiplier=both)
    np.savez(tmp_path / "both.npz", **both_arr)
    assert BR.arm_problems(tmp_path / "both.npz", flags, "changhua")


def _x1_frame():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(5)
    rows = []
    for c in ("yunlin", "changhua"):
        last = "2027-03" if c == "changhua" else "2027-06"
        for m in L.month_list("2012-01", last):
            rows.append({"county": c, "scope": "in_grid_hybrid", "date": m,
                         "value": float(rng.uniform(80, 120)), "flag": 0,
                         "lo90": 70.0, "hi90": 130.0})
    return pd.DataFrame(rows)


def test_x1_changhua_arm_equals_n2c_declared_secondary():
    BR = _builder()
    from n2_policy import build_r_both as RB

    x1 = _x1_frame()
    flags = np.random.default_rng(1).random(40) < 0.5
    ch, rec = BR.build_x1(x1, "2023-01", flags, "changhua")
    ref, _ = RB.build(x1, "2023-01", flags, scope="changhua")
    np.testing.assert_array_equal(ch["multiplier"], ref["multiplier"])
    np.testing.assert_array_equal(ch["month"], ref["month"])
    yl, _ = BR.build_x1(x1, "2023-01", flags, "yunlin")
    both, _ = RB.build(x1, "2023-01", flags, scope="both")
    np.testing.assert_array_equal(ch["multiplier"] * yl["multiplier"], both["multiplier"])
    assert rec["window_months_recorded"] == {"changhua": 4, "yunlin": 7}


def test_max_filled_fraction_refuses():
    pd = pytest.importorskip("pandas")
    BR = _builder()
    sys.path.insert(0, str(REPO / "dataprep_staging" / "x5_changhua_lever"))
    import replay as RP

    idx = [str(p) for p in pd.period_range("2012-01", "2022-12", freq="M")]
    raw = pd.DataFrame({"changhua": 1.0, "yunlin": 1.0}, index=idx)
    assert BR.filled_problems(RP, raw) == []
    # rp2012's window replays 2015-12..2016-10: 3 of 11 filled (0.27 > 0.2) refuses
    raw.loc[["2015-12", "2016-01", "2016-02"], "yunlin"] = np.nan
    bad = BR.filled_problems(RP, raw)
    assert bad and bad[0].startswith("rp2012 yunlin: 3 of 11")
    raw.loc[["2016-01", "2016-02"], "yunlin"] = 1.0                  # 1 of 11: allowed
    assert not [b for b in BR.filled_problems(RP, raw) if b.startswith("rp2012")]


# ----------------------------------------------------------------------------- estimator
def _panel(seed=1):
    return L.synthetic_panel(seed)


def _rows(p, j, e):
    return L.long_rows(p["sid"], p["months"], e, p["c_ch"][j], p["c_yl"][j])


@pytest.mark.parametrize("form", K.FORMS)
def test_exact_recovery_and_frisch_waugh(form):
    p = _panel()
    blocks = dict(zip(p["sid"], p["block"], strict=True))
    e = 0.4 * p["c_ch"][2] + 1.3 * p["c_yl"][2]
    s = L.estimate(_rows(p, 2, e), blocks, form, B=200)
    assert abs(s["beta_ch"] - 0.4) < 1e-9
    assert abs(s["beta_yl_report_only"] - 1.3) < 1e-9
    rng = np.random.default_rng(0)
    e = e + rng.normal(0, 0.3, e.shape)
    rows = _rows(p, 2, e)
    s = L.estimate(rows, blocks, form, B=200)
    r, Q = L.prepare(rows, form)
    Xm = np.stack([L.resid(r["ch"], Q), L.resid(r["yl"], Q)], axis=1)
    full = np.linalg.lstsq(Xm, L.resid(r["e"], Q), rcond=None)[0]
    assert abs(s["beta_ch"] - full[0]) < 1e-9
    assert abs(L.refit_interval(rows, blocks, form, B=200)["beta_ch"] - full[0]) < 1e-9
    lo, hi = s["ci90_ch"]
    assert lo <= s["beta_ch"] <= hi


def test_wraps_the_pinned_bootstrap():
    """With c_YL = 0 the primary is exactly the pinned beta_response on the transformed rows."""
    p = _panel()
    blocks = dict(zip(p["sid"], p["block"], strict=True))
    rng = np.random.default_rng(2)
    e = 0.7 * p["c_ch"][0] + rng.normal(0, 0.2, p["c_ch"][0].shape)
    rows = L.long_rows(p["sid"], p["months"], e, p["c_ch"][0], np.zeros_like(e))
    s = L.estimate(rows, blocks, "twoway", B=500, yunlin=False)
    r, Q = L.prepare(rows, "twoway")
    ref = BS.beta_response(L.resid(r["e"], Q), L.resid(r["ch"], Q), r["well"], blocks, B=500)
    assert s["beta_ch"] == pytest.approx(ref["beta"], abs=1e-12)
    assert s["ci90_ch"] == pytest.approx(ref["ci90"], abs=1e-12)
    assert s["bootstrap_ch"]["seed"] == BS.SEED == 20261101


def test_twoway_absorbs_a_county_uniform_term():
    p = _panel()
    blocks = dict(zip(p["sid"], p["block"], strict=True))
    g = np.linspace(-1, 1, K.N_WINDOW) ** 3
    e = 0.5 * p["c_ch"][1] + 0.5 * p["c_yl"][1] + 3.0 * g[None, :]
    for f in K.FORMS:
        assert abs(L.estimate(_rows(p, 1, e), blocks, f, B=100)["beta_ch"] - 0.5) < 1e-9


def test_first_diff_and_min_months():
    p = _panel()
    e = np.ones_like(p["c_ch"][0])
    e[0, :6] = np.nan
    e[1, 5] = np.nan
    rows = _rows(p, 0, e)
    assert p["sid"][0] in rows["dropped_wells"]
    d = L.first_diff(rows)
    assert (d["well"] == p["sid"][1]).sum() == 8
    assert not np.isin(p["sid"][0], d["well"]).any()


def test_readings():
    assert L.reading(0.2, 0.1, 0.49, True) == "overstated_x2"
    assert L.reading(1.0, 0.51, 1.4, True) == "not_overstated_x2"
    assert L.reading(0.6, 0.3, 0.9, True) == "inconclusive"
    assert L.reading(0.5, 0.5, 0.5, True) == "inconclusive"
    assert L.reading(np.nan, np.nan, np.nan, True) == "inconclusive"
    assert L.reading(1.0, 0.9, 1.1, False) == "not_informative"
    assert set(K.READING_TEXT) == set(K.READINGS)


@pytest.mark.parametrize("form", K.FORMS)
def test_informative_gate_uses_the_form_and_the_partialled_regressor(form):
    p = _panel()
    blocks = dict(zip(p["sid"], p["block"], strict=True))
    args = (p["sid"], blocks, p["months"])
    g = L.informative_gate(p["c_ch"][0], p["c_yl"][0], *args, 11, form)
    assert g["form"] == form and g["passes"] and abs(g["corr_ch_yl"]) <= 0.8
    rows = L.long_rows(p["sid"], p["months"], np.zeros((74, 11)), p["c_ch"][0], p["c_yl"][0])
    r, Q = L.prepare(rows, form)
    chp, _ = L.partial(L.resid(r["ch"], Q), L.resid(r["yl"], Q))
    assert g["rms_well_demeaned_contrast_m"] == pytest.approx(np.sqrt(np.mean(chp ** 2)))
    same = L.informative_gate(p["c_ch"][0], 2 * p["c_ch"][0], *args, 11, form)
    assert not same["passes"]
    assert not L.informative_gate(1e-3 * p["c_ch"][0], p["c_yl"][0], *args, 11, form)["passes"]
    assert not L.informative_gate(p["c_ch"][0], p["c_yl"][0], *args, 5, form)["passes"]


# ----------------------------------------------------------------------------- OSSE
def test_world_is_seed_deterministic_and_confounder_exact():
    p = _panel()
    a = L.world(p, 0, 1.0, "shaped", np.random.default_rng(9), PARAMS)
    b = L.world(p, 0, 1.0, "shaped", np.random.default_rng(9), PARAMS)
    np.testing.assert_array_equal(a, b)
    xi = np.random.default_rng(99).standard_normal(11)
    g = L.confounder(xi, p["a_ch"][0], -0.83)
    assert np.corrcoef(g, p["a_ch"][0])[0, 1] == pytest.approx(-0.83, abs=1e-9)
    assert g.std() == pytest.approx(1.0, abs=1e-9)
    m = L.world(p, 0, 1.0, "shaped", np.random.default_rng(9), PARAMS, miss_frac=0.25)
    assert 0.1 < np.isnan(m).mean() < 0.4
    d = L.world(p, 0, 1.0, "shaped", np.random.default_rng(9), PARAMS, drop_wells=15)
    assert int(np.isnan(d).all(axis=1).sum()) == 15 and np.isfinite(d).any(axis=1).sum() == 59


def test_cells_and_plumbing():
    cl = L.cells()
    gate = [c for c in cl if c["role"] == "gate"]
    assert len(gate) == len(K.K_GRID) * len(K.GATE_VARIANTS)
    assert {c["variant"] for c in gate} == set(K.GATE_VARIANTS)
    assert len(cl) == len(gate) + len(K.REPORT_CELLS)
    assert all(c["conf"] == "none" or c.get("k_yl") or c.get("kappa_mult")
               for c in cl if c["role"] == "report_only")
    assert any(c.get("k_yl", 0) >= 3 for c in K.REPORT_CELLS)
    assert L.plumbing_exact(_panel())["ok"]


def test_run_osse_requires_a_seed_and_is_reproducible():
    p = _panel()
    with pytest.raises(TypeError):
        L.run_osse(p, PARAMS)
    with pytest.raises(TypeError):
        L.run_osse(p, PARAMS, seed=None)
    r1 = L.run_osse(p, PARAMS, seed=11, reps=7, B=50)
    r2 = L.run_osse(p, PARAMS, seed=11, reps=7, B=50)
    assert json.dumps(r1, default=float) == json.dumps(r2, default=float)
    assert [g["form"] for g in r1["gates"]] == list(K.FORMS)
    assert set(r1["gates"][0]["per_variant"]) == set(K.GATE_VARIANTS)
    assert r1["bootstrap_seed"] == BS.SEED and r1["seed"] == 11
    assert set(r1["choice"]) >= {"passed", "form", "osse_power_X"}


def _fake_results(power, false_no, false_ov, n_inf=100, reps=100):
    out = []
    for k in K.K_GRID:
        for var in K.GATE_VARIANTS:
            forms = {}
            for f in K.FORMS:
                no = power[f] if k == K.K_TRUE else (false_no[f] if k == K.K_OVER else 0.0)
                ov = false_ov[f] if k == K.K_TRUE else 0.0
                p = {r: 0.0 for r in K.READINGS}
                p.update(not_overstated_x2=no, overstated_x2=ov)
                pi = {r: p[r] * reps / n_inf for r in ("overstated_x2", "not_overstated_x2",
                                                       "inconclusive")}
                forms[f] = {"p": p, "p_given_informative": pi}
            out.append({"cell": {"k": k, "variant": var, "role": "gate"}, "forms": forms})
    return out


def test_gate_and_choice_rule():
    f1, f2, f3 = K.FORMS
    res = _fake_results({f1: 0.85, f2: 0.9, f3: 0.95}, {f1: 0.1, f2: 0.1, f3: 0.25},
                        {f1: 0.05, f2: 0.05, f3: 0.0})
    gates = [L.form_gate(res, f) for f in K.FORMS]
    assert [g["passes"] for g in gates] == [True, True, False]
    ch = L.choose(gates)
    assert ch["passed"] and ch["form"] == f2
    res = _fake_results({f1: 0.9, f2: 0.9, f3: 0.9}, {f1: 0.15, f2: 0.1, f3: 0.1},
                        {f1: 0.0, f2: 0.0, f3: 0.0})
    assert L.choose([L.form_gate(res, f) for f in K.FORMS])["form"] == f2
    res = _fake_results({f1: 0.9, f2: 0.9, f3: 0.9}, {f1: 0.15, f2: 0.15, f3: 0.15},
                        {f1: 0.0, f2: 0.0, f3: 0.0}, n_inf=50)
    assert not any(L.form_gate(res, f)["passes"] for f in K.FORMS)
    res = _fake_results({f1: 0.3, f2: 0.42, f3: 0.1}, {f1: 0.1, f2: 0.1, f3: 0.1},
                        {f1: 0.0, f2: 0.0, f3: 0.0})
    ch = L.choose([L.form_gate(res, f) for f in K.FORMS])
    assert not ch["passed"] and ch["form"] is None
    assert ch["n_line"] == "size class not decidable prospectively (OSSE power 0.42)"


def test_osse_design_and_declaration_need_a_seed():
    import osse_n2cs as O

    with pytest.raises(TypeError):
        O.draft_osse_record()                                   # no seed: refused
    with pytest.raises(TypeError):
        O.draft_osse_declaration(None, params=PARAMS)
    d = O.draft_osse_declaration(20261130, params=PARAMS)
    assert d["kind"] == "declaration" and d["id"] == K.OSSE_DECL_ID
    assert d["design"]["seeds"] == {"world_seed": 20261130, "bootstrap_seed": 20261101,
                                    "bootstrap_B": 2000}
    assert set(d["design"]["forms"]) == set(K.FORMS) and d["thresholds"]["g0"] == K.G0
    assert d["thresholds"]["max_filled_fraction"] == 0.2
    assert d["attempt_policy"]["sealing"] == O.SEALING
    assert f"{K.X}/n2cs_lib.py" in d["code_sha256"]
    for rec in d["inputs"]:
        path = rec["path"] if isinstance(rec, dict) else rec
        assert (REPO / path).exists()
    assert any("forcing_recorded_arm, which is not the use made here" in x
               for x in d["rule_risks_disclosure"])


@NEED_REGISTRY
def test_osse_declaration_validates():
    pytest.importorskip("pandas")
    import osse_n2cs as O

    assert O.validate_declaration(O.draft_osse_declaration(1, params=PARAMS)) == []


def test_osse_smoke(tmp_path):
    import osse_n2cs as O

    out = O.smoke(5, tmp_path, reps=7, B=50)
    assert out["smoke"] and (tmp_path / "smoke_result.json").exists()


def test_pregate_function():
    import osse_n2cs as O

    hm = np.zeros((3, 2, 4, 5))
    assert O.pregate(hm, ["baseline", "UCH", "UYL"])["ok"]
    hm[2, 1, 3, 4] = 2e-6
    pg = O.pregate(hm, ["baseline", "UCH", "UYL"])
    assert not pg["ok"] and pg["UYL"]["max_abs_m"] == pytest.approx(2e-6)


def test_run_refuses_a_panel_extract_did_not_log(tmp_path):
    import osse_n2cs as O

    att = tmp_path / "a.jsonl"
    assert O.last_extract_panel_sha(att) is None
    att.write_text(json.dumps({"event": "extract", "pregate_ok": True, "panel_sha256": "ab"})
                   + "\n" + json.dumps({"event": "extract", "pregate_ok": False}) + "\n")
    assert O.last_extract_panel_sha(att) == "ab"


# ----------------------------------------------------------------------------- draft
def _passing_result():
    gates = [{"form": f, "passes": f == "twoway", "power_min": 0.83,
              "false_not_overstated_max": 0.1, "per_variant": {}} for f in K.FORMS]
    return {"engineering_failure": False, "gates": gates, "seed": 1, "choice":
            {"passed": True, "form": "twoway", "osse_power_X": 0.83,
             "false_not_overstated_max": 0.1}}


def test_draft_refuses_a_failed_osse():
    import prereg_n2cs as P

    bad = {"engineering_failure": False, "gates": [],
           "choice": {"passed": False, "n_line": "size class not decidable prospectively "
                                                 "(OSSE power 0.31)"}}
    with pytest.raises(ValueError, match="OSSE power 0.31"):
        P.draft_prereg_n2cs(bad, osse_result_sha256="0", osse_declaration_sha256="0",
                            ids=[f"07{i:06d}" for i in range(74)], check_files=False)


def test_draft_fields():
    import prereg_n2cs as P

    ids = [f"07{i:06d}" for i in range(74)]
    d = P.draft_prereg_n2cs(_passing_result(), osse_result_sha256="r",
                            osse_declaration_sha256="s", ids=ids, check_files=True)
    t = d["tests"][0]
    assert d["id"] == K.ID and d["kind"] == "test" and d["primary_form"] == "twoway"
    assert t["data_scope"] == "heads:2026-12..2027-10" and t["fresh_units"] == [K.UNIT]
    assert t["external_units"][K.UNIT]["window"] == "2026-12..2027-10"
    assert t["heads_contract"] == K.HEADS_CONTRACT
    assert d["frozen_code"][K.GOV] == K.GOV_SHA
    assert d["frozen_code"][K.E02C_SCORER] == K.E02C_SCORER_SHA
    assert d["scoring"]["not_before"] == K.NOT_BEFORE
    assert K.N2C_REPORT in d["scoring"]["n2c_step2"]
    for key in ("multiplicity", "U7", "uniform_confounder", "winners_curse", "design_device",
                "census_source", "v3_exemption_vs_u7", "smoke_runs", "intervals"):
        assert key in d["disclosures"]
    assert "README.md" in d["page_mapping"]["before_scoring"]
    assert "MODEL_CARD.md" in d["page_mapping"]["before_scoring"]
    assert any("correction factor" in s for s in d["page_mapping"]["never"])
    assert "created" not in d and "GOV_KEY" not in json.dumps(d)


@NEED_REGISTRY
def test_draft_validates_against_prereg3():
    pytest.importorskip("pandas")
    import prereg_n2cs as P

    d = P.draft_prereg_n2cs(_passing_result(), osse_result_sha256="r",
                            osse_declaration_sha256="s")
    assert sorted(d["tests"][0]["external_units"][K.UNIT]["ids"]) == P.unit_ids()
    assert P.validate(d) == []


# ----------------------------------------------------------------------------- scorer
def _scorer():
    pytest.importorskip("pandas")
    import score_n2cs as S

    return S


NOW = dt.datetime(2028, 1, 20, tzinfo=dt.timezone.utc)


@pytest.mark.skipif(REGISTERED, reason="N2CS is registered")
def test_scorer_refuses_before_registration(tmp_path):
    S = _scorer()
    att = tmp_path / "att.jsonl"
    with pytest.raises(S.Refused, match="not registered"):
        S.score(tmp_path, tmp_path, "t", tmp_path / "n2c.json", out_dir=tmp_path / "o",
                attempts=att, now=NOW)
    ev = [json.loads(x)["event"] for x in att.read_text().splitlines()]
    assert ev == ["score_invoked", "score_refused"]


def _registered(monkeypatch, frozen=None):
    from harness import prereg3

    doc = {"frozen_code": frozen if frozen is not None else {}, "primary_form": "twoway"}
    monkeypatch.setattr(prereg3, "load_prereg", lambda *a, **k: (doc, "x"))


def test_scorer_refuses_before_not_before_and_twice(tmp_path, monkeypatch):
    S = _scorer()
    _registered(monkeypatch)
    att = tmp_path / "att.jsonl"
    early = dt.datetime(2028, 1, 14, 23, 0, tzinfo=dt.timezone.utc)
    with pytest.raises(S.Refused, match="before 2028-01-15"):
        S.score(tmp_path, tmp_path, "t", tmp_path / "n2c.json", out_dir=tmp_path / "o",
                attempts=att, now=early)
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "report.json").write_text("{}")
    with pytest.raises(S.Refused, match="scored once"):
        S.score(tmp_path, tmp_path, "t", tmp_path / "n2c.json", out_dir=tmp_path / "o",
                attempts=att, now=NOW)


def test_n2c_step2_by_id_and_verdict(tmp_path, monkeypatch):
    S = _scorer()
    real = tmp_path / "reports" / K.N2C_ID / "report.json"
    assert "missing" in S.n2c_step2_problem(real)
    real.parent.mkdir(parents=True)
    real.write_text(json.dumps({"verdict": "not_supported"}))     # verdict, but names no N2C
    assert "does not name" in S.n2c_step2_problem(real)
    real.write_text(json.dumps({"id": K.N2C_ID}))                 # names N2C, no verdict
    assert "no verdict" in S.n2c_step2_problem(real)
    real.write_text(json.dumps({"id": K.N2C_ID, "verdict": "direction_supported_size_not"}))
    assert S.n2c_step2_problem(real) is None
    other = tmp_path / "elsewhere.json"                           # any path, if it names N2C
    other.write_text(json.dumps({"prereg_id": K.N2C_ID, "verdict": "not_supported"}))
    assert S.n2c_step2_problem(other) is None


def test_scorer_refuses_without_the_bound_gov_and_e02c_files(tmp_path, monkeypatch):
    S = _scorer()
    real = tmp_path / "n2c" / "report.json"
    real.parent.mkdir()
    real.write_text(json.dumps({"id": K.N2C_ID, "verdict": "not_supported"}))
    monkeypatch.setattr(K, "N2C_REPORT", str(real))
    _registered(monkeypatch, frozen={K.E02C_SCORER: K.E02C_SCORER_SHA})
    with pytest.raises(S.Refused, match="heads_window.py is not bound"):
        S.score(tmp_path, tmp_path, "t", real, out_dir=tmp_path / "o",
                attempts=tmp_path / "a.jsonl", now=NOW)
    _registered(monkeypatch, frozen={K.E02C_SCORER: K.E02C_SCORER_SHA, K.GOV: K.GOV_SHA})
    monkeypatch.setattr(K, "FWD_A", str(tmp_path / "no_forward_a"))
    with pytest.raises(S.Refused, match="forward A"):
        S.score(tmp_path, tmp_path, "t", real, out_dir=tmp_path / "o",
                attempts=tmp_path / "a.jsonl", now=NOW)
    assert not (tmp_path / "o" / "report.json").exists()


def _window_product(root: Path, window) -> Path:
    spec = importlib.util.spec_from_file_location("gov_hw_test_n2cs", REPO / K.GOV)
    HW = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(HW)
    d = root / "heads_window" / "v1"
    d.mkdir(parents=True)
    (d / "validation.json").write_text("{}\n")
    rec = {"path": "heads.parquet", "sha256": "0" * 64}
    m = HW.instantiate(HW.template(), version="v1", created="2028-01-10", inputs=[],
                       code=[], git_commit="0" * 40, fit_window_end="2029-10", outputs=[rec],
                       validation_report="validation.json",
                       gate={"passed": True, "criteria": "synthetic"}, window=window)
    (d / "MANIFEST.json").write_text(json.dumps(m, indent=1))
    return d


def test_precondition_is_n2cs_own_covering_rule(tmp_path):
    S = _scorer()
    ok, why = S.precondition(_window_product(tmp_path / "a", ("2026-11", "2029-10")))
    assert ok and "covering_rule" in why
    ok, why = S.precondition(_window_product(tmp_path / "b", ("2027-11", "2029-10")))
    assert not ok and "does not cover" in why


class _FakeProduct:
    def __init__(self, path: Path):
        self.path = path
        self.outputs = {path.name: {"sha256": "0" * 64}}
        self.notes = []

    def _resolve(self, name):
        return self.path

    def read_table(self, name, columns=None):
        import pandas as pd

        df = pd.read_parquet(self.path)
        return df[[c for c in (columns or df.columns) if c in df.columns]
                  + [c for c in ("flag",) if c in df.columns and c not in (columns or [])]]


def _heads(tmp_path, **cols):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    p = tmp_path / "heads.parquet"
    pd.DataFrame(cols).to_parquet(p, index=False)
    return _FakeProduct(p)


def test_heads_contract_checks_before_the_look(tmp_path):
    S = _scorer()
    ids = [f"07{i:06d}" for i in range(74)]
    good = _heads(tmp_path, station_id=["07000001", "07000002"], date=["2026-12", "2027-01"],
                  observed_m=[1.0, 2.0], flag=[0, 1])
    assert S.contract_problems(good, "heads.parquet", ids) == []
    assert "not a parquet table listed" in S.contract_problems(good, "x.csv", ids)[0]
    (tmp_path / "l").mkdir()
    lost = _heads(tmp_path / "l", station_id=["7000001"], date=["2026-12"], observed_m=[1.0],
                  flag=[0])
    assert any("leading zeros" in x for x in S.contract_problems(lost, "heads.parquet", ids))
    (tmp_path / "n").mkdir()
    num = _heads(tmp_path / "n", station_id=[7000001], date=["2026-12"], observed_m=[1.0])
    assert "not a string" in S.contract_problems(num, "heads.parquet", ids)[0]
    (tmp_path / "w").mkdir()
    wrong = _heads(tmp_path / "w", station_id=["07000001"], date=["2026-12"], head_m=[1.0])
    assert "'observed_m' missing" in S.contract_problems(wrong, "heads.parquet", ids)[0]
    (tmp_path / "f").mkdir()
    flags = _heads(tmp_path / "f", station_id=["07000001"], date=["2026-12"], observed_m=[1.0],
                   flag=[3])
    assert "no scorable row" in S.contract_problems(flags, "heads.parquet", ids)[0]
    (tmp_path / "d").mkdir()
    dates = _heads(tmp_path / "d", station_id=["07000001"], date=["not a month"],
                   observed_m=[1.0])
    assert "does not parse" in S.contract_problems(dates, "heads.parquet", ids)[0]


def test_page_line_wording():
    S = _scorer()
    line = S.page_line("overstated_x2", 0.21, [0.05, 0.4])
    assert "overstated >= x2" in line and line.endswith(K.WORDING)
    assert "0.21 [0.05, 0.40]" in line


# ----------------------------------------------------------------------------- runner
@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_runner_dry_forward_commands():
    sh = X / "run_n2cs.sh"
    subprocess.run(["bash", "-n", str(sh)], check=True)
    env = {"DRY": "1", "HOME": "/nonexist", "PATH": "/usr/bin:/bin"}
    out = subprocess.run(["bash", str(sh), "forward-a"], env=env, capture_output=True,
                         text=True, check=True).stdout
    arms = [a.split("=")[0] for a in out.split("--recorded-pumping '")[1:]]
    assert arms == ["UCH", "UYL", *[f"RCH{y}" for y in K.REPLAY_YEARS]]
    assert "--device cpu" in out and K.FWD_FLAGS in out
    out = subprocess.run(["bash", str(sh), "forward-b"], env=env, capture_output=True,
                         text=True, check=True).stdout
    assert [a.split("=")[0] for a in out.split("--recorded-pumping '")[1:]] == \
        [f"RYL{y}" for y in K.REPLAY_YEARS]
    import osse_n2cs as O

    fc = O.forward_commands()               # the declaration's commands match the runner's
    for tag, sh_out in (("A", "forward-a"), ("B", "forward-b")):
        dry = subprocess.run(["bash", str(sh), sh_out], env=env, capture_output=True,
                             text=True, check=True).stdout.strip()
        assert dry == fc[tag]
