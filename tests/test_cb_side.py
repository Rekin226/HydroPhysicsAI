"""CB.coverage_side.v1 (round-8 rank 10), on synthetic and fabricated inputs only.

- seeds have no default; refused seeds raise; the CLI requires --seed;
- the ICC grid is U1.HB.power.v1's; cell (0, 0) gives FV v2's noise bit for bit; the shock
  decomposition keeps the marginal sd; the drift is group-centred with its variance kept and
  DR's 'origin' lead (2027-05 at 4.375 years);
- E02c's frozen side is read back from its own wording: over / under / pass / pooled-only /
  not-evaluable on fabricated coverage records;
- the E1 rule, the gate and the paired shock statistics on fabricated outcome codes;
- wording in both languages, never-words; no page line in any branch (LEAD, review B1); the
  datum-centre suffix appears exactly when that report-only arm misses 0.8; post-E1 failures
  are final not_evaluable;
- end to end on osse_fv2's SYNTHETIC inputs: E0 reproduces osse_fv2.score_worlds' own
  rates exactly; E0 failure raises; E1 failure writes not_evaluable; the pooled run's codes
  equal in-process codes exactly;
- the runner refuses before registration and re-runs (no report written), writes once, and
  logs every invocation in attempts.jsonl;
- the registration draft's key fields; the module never calls write_prereg.

Checks that need the gitignored results/ tree skip when it is absent (CI).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
CB_DIR = REPO / "dataprep_staging" / "cb_side"
HAVE_RESULTS = (REPO / "results" / "dataprep" / "preregistration").is_dir()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_saved = list(sys.path)
C = _load("cb_side", CB_DIR / "cb_side.py")
PR = _load("prereg_cb", CB_DIR / "prereg_cb.py")
sys.path[:] = _saved
if str(REPO / "dataprep_staging") not in sys.path:      # harness.prereg3 needs common
    sys.path.append(str(REPO / "dataprep_staging"))
S, O, V2, U = C.S, C.O, C.V2, C.U  # noqa: E741


# ----------------------------------------------------------------------------- seeds
def test_seed_has_no_default_and_refused_seeds_raise():
    for bad in (None, -1, 1.5, True, "7"):
        with pytest.raises(C.EngineeringFailure):
            C.check_seed(bad)
    for s in (O.SEED, U.SEED, C.SMOKE_SEED, 20261105, 20261102, 20261103):
        with pytest.raises(C.EngineeringFailure, match="refused"):
            C.check_seed(s)
    assert C.check_seed(123) == 123
    assert inspect.signature(PR.draft_prereg).parameters["seed"].default is \
        inspect.Parameter.empty
    assert inspect.signature(C.run_study).parameters["seed"].default is \
        inspect.Parameter.empty
    with pytest.raises(SystemExit):
        C.main(["run"])
    with pytest.raises(C.EngineeringFailure):
        PR.draft_prereg(O.SEED)


# ----------------------------------------------------------------------------- world parts
def test_icc_grid_is_u1hb_and_cell0_is_zero():
    assert C.ICC_YEAR == (0.0, 0.1, 0.2) == tuple(float(x) for x in U.ICC_YEAR)
    assert C.ICC_MONTH == (0.0, 0.05, 0.15, 0.3) == tuple(float(x) for x in U.ICC_MONTH)
    assert len(C.CELLS) == 12 and C.CELLS[0] == {"cell": 0, "icc_month": 0.0, "icc_year": 0.0}
    assert {(c["icc_month"], c["icc_year"]) for c in C.CELLS} == {
        (m, y) for m in C.ICC_MONTH for y in C.ICC_YEAR}


def test_shock_noise_identity_and_marginal_sd():
    rng = np.random.default_rng(0)
    sd, phi = rng.uniform(0.3, 1.0, 9), rng.uniform(0.2, 0.8, 9)
    e = O.simulate_ar1(sd, phi, 12, np.random.default_rng(1))
    n0 = C.shock_noise(e, 1.7, rng.standard_normal(12), sd, 0.0, 0.0)
    assert np.array_equal(n0, e)                       # bit for bit: FV v2's noise
    for ay, am in ((0.2, 0.3), (0.1, 0.0), (0.0, 0.15)):
        # coefficients of (C, B, e) in units of sd: their squares sum to 1
        assert math.isclose(ay + (1 - ay) * am + (1 - ay) * (1 - am), 1.0)
        n = C.shock_noise(np.zeros((9, 12)), 1.0, np.zeros(12), sd, ay, am)
        assert np.allclose(n, math.sqrt(ay) * sd[:, None])   # year part common to all


def test_drift_group_centred_and_lead():
    gidx = np.array([0] * 8 + [1] * 10)
    z = np.random.default_rng(3).standard_normal((2000, 18))
    b = np.stack([C.centre_rescale(r, gidx) for r in z])
    for g in (0, 1):
        assert np.allclose(b[:, gidx == g].sum(1), 0.0, atol=1e-12)
    assert abs(b.var() - 1.0) < 0.05
    dt = C.window_dt()
    assert dt.shape == (12,) and math.isclose(dt[0], 3.875) and math.isclose(dt[6], 4.375)
    ctx = type("Ctx", (), {"dt": {"none": None, "origin": np.broadcast_to(dt, (18, 12))},
                           "group": gidx, "tau2": np.full(18, 0.02)})()
    assert C.drift_field(ctx, z[0], "none") is None
    d = C.drift_field(ctx, z[0], "origin")
    assert np.allclose(d, (C.centre_rescale(z[0], gidx) * math.sqrt(0.02))[:, None] * dt)


# ----------------------------------------------------------------------------- E02c side
def _cc(cov, g1_pass, n=888, evaluable=True):
    return {"county": "Changhua", "evaluable": evaluable,
            "why": None if evaluable else "k = 3 wells < 20",
            "G1": {"coverage": cov, "n": n, "pass": g1_pass},
            "G2": {"miss_below": 0.01, "miss_above": 0.02, "balanced": True}}


def test_classify_reads_e02c_frozen_side():
    def code(g0, cc):
        return C.classify(S.cov_outcome(g0, cc, O.NEUTRAL_POWER), cc)

    assert code(True, _cc(0.97, False)) == C.OVER
    assert code(False, _cc(0.99, False)) == C.OVER
    assert code(True, _cc(0.80, False)) == C.UNDER
    assert code(True, _cc(0.90, True)) == C.PASS
    assert code(False, _cc(0.90, True)) == C.POOLED
    assert code(None, _cc(0.90, True)) == C.NOT_EVAL
    assert code(True, _cc(0.90, True, evaluable=False)) == C.NOT_EVAL
    cc = _cc(0.97, False)
    bad = {"verdict": "coverage_not_supported", "wording": "something else"}
    with pytest.raises(C.EngineeringFailure):
        C.classify(bad, cc)


# ----------------------------------------------------------------------------- statistics
def test_e1_rule():
    assert C.e1_check(0, 1, 250, 250)["pass"]
    assert C.e1_check(6, 1, 250, 250)["pass"]
    assert not C.e1_check(7, 1, 250, 250)["pass"]
    assert C.e1_check(0, 0, 250, 250)["pass"]          # SE 0: equality
    assert C.e1_check(250, 250, 250, 250)["pass"]          # pooled rate 1: SE 0


def _fab_codes(p_over: dict, k=40, seed=5):
    rng = np.random.default_rng(seed)
    A, J = len(C.DRIFT_ARMS), len(C.CELLS)
    codes, cov = {}, {}
    for t in C.TAGS:
        x = np.where(rng.uniform(size=(k, A, J)) < p_over[t], C.OVER, C.PASS).astype(np.int8)
        codes[t], cov[t] = x, rng.uniform(0.8, 1.0, (k, A, J))
    return codes, cov


def test_gate_and_paired_statistics_on_fabricated_codes():
    codes, cov = _fab_codes({"Q0": 1.0, "R0": 1.0, "R1c": 0.0, "X5R1": 0.0})
    st = C.summarize(codes, cov)
    g = C.gate(st)
    assert g["pass"] and g["page_family"]["worst"]["p"] == 1.0
    codes["R1c"][:20, C.DRIFT_ARMS.index("none"), 5] = C.OVER      # 0.5 in one cell
    g = C.gate(C.summarize(codes, cov))
    assert not g["pass"] and g["changhua_recipes"]["worst"]["cell"] == 5
    # the report-only arm never gates
    codes, cov = _fab_codes({"Q0": 1.0, "R0": 1.0, "R1c": 0.0, "X5R1": 0.0})
    codes["X5R1"][:, C.DRIFT_ARMS.index("datum_centre_report_only"), :] = C.OVER
    codes["Q0"][:, C.DRIFT_ARMS.index("none"), :] = C.PASS          # no-drift page: not gated
    assert C.gate(C.summarize(codes, cov))["pass"]
    # bias cancelled by shock: fail at (0, 0) and pass at the cell, same world and arm
    codes, cov = _fab_codes({"Q0": 0.0, "R0": 0.0, "R1c": 0.0, "X5R1": 0.0})
    a = C.DRIFT_ARMS.index("origin")
    codes["Q0"][:10, a, 0] = C.OVER
    codes["Q0"][:4, a, 3] = C.OVER
    st = C.summarize(codes, cov)
    assert math.isclose(st["Q0"]["origin"][3]["bias_cancelled_by_shock"]["p"], 6 / 40)
    assert st["Q0"]["origin"][0]["bias_cancelled_by_shock"]["p"] == 0.0
    bc = C.bias_cancelled_summary(st)
    assert math.isclose(bc["page_family_gated_arm_max"], 10 / 40)


# ----------------------------------------------------------------------------- wording
NEVER = ("calibrated", "pumping matters", "C20", "C21", "0.004", "0.992", "0.996", "0.012")


def test_wording_both_languages_and_never_words():
    for key in C.WORDING:
        assert set(C.WORDING[key]) == {"ledger"}                # no page wording anywhere
    for key in ("side_separates", "side_does_not_separate"):
        for lang in ("en", "zh"):
            s = C.WORDING[key]["ledger"][lang].format(a=0.85, b=0.1, c=0.05)
            assert "{" not in s and "0.850" in s and "calibrated" not in s
            assert "pumping matters" not in s
        assert "not on the page" in C.WORDING[key]["ledger"]["en"]
        assert "不上頁面" in C.WORDING[key]["ledger"]["zh"]
        assert "ICC_well not overlaid" in C.WORDING[key]["ledger"]["en"]
    for lang in ("en", "zh"):
        assert "{" not in C.WORDING["not_evaluable"]["ledger"][lang].format(why="x")
        assert "{" not in C.DATUM_CENTRE_SUFFIX[lang].format(d=0.5)
    assert "every year/month ICC cell" in C.WORDING["side_separates"]["ledger"]["en"]


def test_reading_has_no_page_line_and_datum_centre_suffix():
    for p_dc, suffix in ((1.0, False), (0.5, True)):
        codes, cov = _fab_codes({"Q0": 1.0, "R0": 1.0, "R1c": 0.0, "X5R1": 0.0})
        dc = C.DRIFT_ARMS.index("datum_centre_report_only")
        codes["R0"][:, dc, 7] = np.where(np.arange(40) < 40 * p_dc, C.OVER, C.PASS)
        st = C.summarize(codes, cov)
        g = C.gate(st)
        assert g["pass"]                                # the report-only arm never gates
        r = C.reading(g, st)
        assert r["page"] == C.PAGE_RULE and r["status_change"].startswith("none")
        assert r["datum_centre_gate_a_report_only"]["misses_0_8"] is suffix
        for lang in ("en", "zh"):
            tail = C.DATUM_CENTRE_SUFFIX[lang].format(d=0.5)
            assert r["wording"]["ledger"][lang].endswith(tail) is suffix
    ne = C.not_evaluable({}, "x")
    assert ne["reading"]["page"] == C.PAGE_RULE


# ----------------------------------------------------------------------------- synthetic run
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("cb_synth")
    plan = V2.Plan(k_truth=3, batch=2)
    P, rep = C.build_synthetic(out / "inputs", plan)
    ctx, info = C.load_ctx(P)
    truths, _ = C.load_truths(P, ctx, plan)
    return {"out": out, "plan": plan, "P": P, "v2": rep, "ctx": ctx, "info": info,
            "truths": truths, "pub": C.published_rates(P.fv2_report)}


def test_synthetic_end_to_end(synth, monkeypatch):
    monkeypatch.setattr(C, "E1_N_SE", 1e9)       # engineering only: always run the grid
    rep = C.run_study(synth["ctx"], synth["truths"], seed=C.SMOKE_SEED,
                      published=synth["pub"], k=3, k_published=3, workers=2,
                      log=lambda *_: None)
    assert rep["status"] == "scored"
    e0 = rep["engineering"]["E0"]
    assert all(v["pass"] for c in e0.values() for v in c.values())
    # E0 compares with osse_fv2.score_worlds' own output on the same synthetic inputs
    for c in S.COUNTIES_COV:
        for t in C.TAGS:
            assert e0[c][t]["published"] == \
                synth["v2"]["P"]["coverage"][c]["by_truth"][t]
    assert set(rep["stats"]) == set(C.TAGS)
    for t in C.TAGS:
        assert set(rep["stats"][t]) >= set(C.DRIFT_ARMS)
        assert all(len(rep["stats"][t][a]) == 12 for a in C.DRIFT_ARMS)
    assert rep["reading"]["reading"] in ("side_separates", "side_does_not_separate")
    assert "page_family_gated_arm_max" in rep["bias_cancelled_by_shock"]
    assert rep["reading"]["page"] == C.PAGE_RULE
    # worker count changes nothing: in-process codes equal the pooled (2-worker) run's exactly
    C._STATE.update({"ctx": synth["ctx"], "truths": synth["truths"], "seed": C.SMOKE_SEED})
    try:
        for tag, kk in (("R1c", 2), ("Q0", 0)):
            _, _, codes, _ = C._task_grid((C.TAGS.index(tag), kk))
            assert np.array_equal(codes, np.asarray(rep["codes_report_only"][tag][kk]))
    finally:
        C._STATE.clear()


def test_failure_after_e1_is_final_not_evaluable(synth, monkeypatch):
    monkeypatch.setattr(C, "E1_N_SE", 1e9)

    def boom(*a, **k):
        raise RuntimeError("fabricated grid crash")

    monkeypatch.setattr(C, "summarize", boom)
    rep = C.run_study(synth["ctx"], synth["truths"], seed=C.SMOKE_SEED,
                      published=synth["pub"], k=3, k_published=3, workers=1,
                      log=lambda *_: None)
    assert rep["status"] == "not_evaluable" and "after E1" in rep["why"]
    assert rep["engineering"]["E1"]["pass"] and "stats" not in rep


def test_e0_failure_raises_and_e1_failure_is_final(synth, monkeypatch):
    pub = json.loads(json.dumps(synth["pub"]))
    pub["Changhua"]["Q0"] = 0.5 if pub["Changhua"]["Q0"] != 0.5 else 0.0
    with pytest.raises(C.EngineeringFailure, match="E0"):
        C.run_study(synth["ctx"], synth["truths"], seed=C.SMOKE_SEED, published=pub, k=3,
                    k_published=3, workers=1, log=lambda *_: None)
    monkeypatch.setattr(C, "e1_check", lambda *a, **k: {"pass": False})
    rep = C.run_study(synth["ctx"], synth["truths"], seed=C.SMOKE_SEED,
                      published=synth["pub"], k=3, k_published=3, workers=1,
                      log=lambda *_: None)
    assert rep["status"] == "not_evaluable" and "stats" not in rep
    assert rep["reading"]["reading"] == "not_evaluable"


def test_dr_inputs_refuse_non_zone(synth, tmp_path):
    P = synth["P"]
    rep = json.loads(Path(P.dr_report).read_text())
    rep["decisions"]["Changhua"]["grouping"] = "layer"
    p = tmp_path / "dr.json"
    p.write_text(json.dumps(rep))
    ctx = synth["ctx"]
    with pytest.raises(C.EngineeringFailure, match="zone"):
        C.dr_inputs(p, P.dr_wells, ctx.ids, ctx.county)
    assert synth["info"]["tau2_by_county"] == {"Changhua": 0.02, "Yunlin": 0.0185}


def test_attempts_helpers(tmp_path):
    att = tmp_path / "attempts.jsonl"
    rows = [{"event": "forward", "truth": t, "batch": b, "rc": 0, "npz_sha256": f"{t}{b}"}
            for t in C.TAGS for b in range(5)]
    rows += [{"event": "forward", "truth": "Q0", "batch": 1, "rc": 1, "npz_sha256": "bad"},
             {"event": "score_worlds", "rc": 0}]
    att.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = C.fv2_batches_from_attempts(att)
    assert len(got) == 19 and "Q0_b1" not in got           # last attempt failed
    assert not C.seal_lapsed(att)["lapsed"]
    with open(att, "a") as f:
        f.write(json.dumps({"event": "score_ext", "rc": 0}) + "\n")
    assert C.seal_lapsed(att)["lapsed"]


# ----------------------------------------------------------------------------- runner
def test_runner_refuses_unregistered(synth, tmp_path):
    P = C.Paths(**{**synth["P"].__dict__, "report": str(tmp_path / "r.json"),
                   "attempts": str(tmp_path / "a.jsonl")})
    (tmp_path / "prereg").mkdir()
    with pytest.raises(C.EngineeringFailure, match="not registered"):
        C.cmd_run(123, 1, P=P, prereg_dir=tmp_path / "prereg", rows_dir=tmp_path / "rows")
    assert not (tmp_path / "r.json").exists()
    with pytest.raises(C.EngineeringFailure, match="refused"):
        C.cmd_run(C.SMOKE_SEED, 1, P=P, prereg_dir=tmp_path / "prereg")
    rows = [json.loads(s) for s in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert [r["rc"] for r in rows] == [1, 1] and [r["attempt"] for r in rows] == [1, 2]


def test_runner_writes_once_and_logs_attempts(synth, tmp_path, monkeypatch):
    P = C.Paths(**{**synth["P"].__dict__, "report": str(tmp_path / "r.json"),
                   "attempts": str(tmp_path / "a.jsonl")})
    monkeypatch.setattr(C, "load_registered", lambda **k: ({"frozen_code": {}}, "d" * 64))
    monkeypatch.setattr(C, "check_registered", lambda d, s, p: {"seal": {}, "fv2_batches": {}})
    monkeypatch.setattr(C, "verify_pins", lambda p: {})
    monkeypatch.setattr(C, "load_truths", lambda p, c, verify=None: ({}, {}))
    monkeypatch.setattr(C, "published_rates", lambda p: {"Changhua": {"Q0": 0.004}})

    def boom(*a, **k):
        raise C.EngineeringFailure("E0: fabricated")

    monkeypatch.setattr(C, "run_study", boom)
    with pytest.raises(C.EngineeringFailure):
        C.cmd_run(123, 1, P=P)
    assert not (tmp_path / "r.json").exists()
    monkeypatch.setattr(C, "run_study", lambda *a, **k: {"status": "scored"})
    assert C.cmd_run(123, 1, P=P) == 0
    with pytest.raises(C.EngineeringFailure, match="run once"):
        C.cmd_run(123, 1, P=P)
    rows = [json.loads(s) for s in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert [r["rc"] for r in rows] == [1, 0, 1] and rows[1]["report_sha256"]
    assert "run once" in rows[2]["error"]                  # refused re-run is logged
    monkeypatch.setattr(C, "load_registered", lambda **k: (_ for _ in ()).throw(
        KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        C.cmd_run(123, 1, P=P)
    last = json.loads((tmp_path / "a.jsonl").read_text().splitlines()[-1])
    assert last["rc"] == 1 and last["error"].startswith("KeyboardInterrupt")


# ----------------------------------------------------------------------------- draft
def test_draft_key_fields():
    d = PR.draft_prereg(424242)
    assert d["id"] == C.ID and d["wp"] == "CB" and d["kind"] == "declaration"
    assert d["tests"] == {"not_applicable": d["tests"]["not_applicable"]}
    assert d["design"]["seed"] == 424242 and d["design"]["streams"] == C.STREAMS
    assert set(d["frozen_code"]) == set(C.FROZEN)
    assert d["thresholds"]["p_over_min_page_family"] == 0.8
    assert d["thresholds"]["p_over_max_changhua_recipes"] == 0.2
    assert "C14a Changhua only" in d["readings"]["applies_to"]
    assert d["page_mapping"]["ruling"].startswith("R8.8")
    assert d["page_mapping"]["page"] == C.PAGE_RULE and "page_line" not in json.dumps(d)
    assert "N+D" not in json.dumps(d["page_mapping"]) + json.dumps(d["readings"])
    assert [u for u, _ in PR.BUILDER_LOOKS] == [
        "R8:cb_coverage_side_builder_reads_disclosure",
        "R8:cb_coverage_side_builder_reads_supplement"]
    assert all(u in d["disclosures"]["builder"] for u, _ in PR.BUILDER_LOOKS)
    assert "Common-mode drift" in d["disclosures"]["drift"]
    assert "median shift" in d["disclosures"]["drift"] and d["disclosures"]["icc_well"]
    assert "after_e1" in d["attempt_policy"]
    assert d["attempt_policy"]["exploratory_followups_max"] == 0
    assert "not FV v3" in d["label"]
    assert "R8:fe_encompass_fixer_reads_disclosure" in d["seal"]["status"]
    for k in ("builder", "optimism", "drift", "estimand_choice"):
        assert d["disclosures"][k]
    src = (CB_DIR / "prereg_cb.py").read_text() + (CB_DIR / "cb_side.py").read_text()
    assert "write_prereg(" not in src


@pytest.mark.skipif(not HAVE_RESULTS, reason="needs the local results/ registry")
def test_draft_validates_and_pins_match():
    from harness import prereg3

    d = PR.draft_prereg(424242)
    assert prereg3.validate_prereg(d) == []
    for k, v in d["binds"]["inputs"].items():
        assert v["sha256"] == v["expected_sha256"], k
    assert len(d["design"]["fv2_batches"]) == 20
    assert C.seal_lapsed(C.FV2_ATTEMPTS)["lapsed"]
    todo = PR.readiness(d)
    assert not any("builder look" in x for x in todo), todo      # both rows, real purposes
