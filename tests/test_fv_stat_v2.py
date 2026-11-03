"""FV.stat.v2 (round-8 rank 3): the engineering fixes of FV.stat.v1, on synthetic inputs only.

- the self-check fix: synthetic rows carry the sha of their own predictions file and pass
  ``twin.validation.evaluate`` against a synthetic protocol copy, identical to ``g0_e02``;
- the control: a protocol whose model_sha256 is not the rows' sha refuses them (and a control
  that would accept them is an engineering failure); v1's self-check fails the same way;
- the previewed seed (and both smoke seeds) are refused; the seed has no default;
- the wording map (P > 0.5 -> not_shown_weak, otherwise weak_by_design; MC failure ->
  not_evaluable), on FABRICATED records only: no test computes a pass rate from simulated
  worlds;
- the runner refuses an unregistered id, a refused seed, changed code and an existing report,
  and appends every attempt to attempts.jsonl; the scorer refuses without a registration;
- the smoke's shape check on a frame (E04's rule is never applied to a cell world there);
- the draft declaration's key fields.

The "real" protocol here is a synthetic stand-in written to tmp_path. Checks that need the
gitignored results/ tree skip when it is absent (CI).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")         # hydrophysics.twin.coverage imports it

REPO = Path(__file__).resolve().parents[1]
D = REPO / "dataprep_staging" / "fv_stat_v2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_saved = list(sys.path)
S = _load("stat_v2", D / "stat_v2.py")
PR = _load("prereg_v2", D / "prereg_v2.py")
sys.path[:] = _saved
OF, ST = S.O, S.ST

HAVE_RESULTS = (REPO / "results" / "dataprep" / "preregistration" / "INDEX.jsonl").is_file()
NEED_RESULTS = pytest.mark.skipif(not HAVE_RESULTS, reason="results/ (gitignored) is absent")
FAKE_MODEL_SHA = "ab" * 32
STATION = re.compile(r"\b(?:0[79]|10)[0-9]{6}\b")


@pytest.fixture(autouse=True)
def _staging_on_path():
    saved = list(sys.path)
    S._paths()
    yield
    sys.path[:] = saved


def _protocol(tmp_path: Path, model_sha: str = FAKE_MODEL_SHA) -> Path:
    """A stand-in for E04's protocol (same evaluate-read fields and rule; synthetic sha)."""
    p = tmp_path / "protocol_real_standin.json"
    p.write_text(json.dumps({
        "schema": 1, "task": "free-running-head-validation", "test_start": "2026-11-01",
        "test_end": "2027-10-31", "registered_before": "2026-10-11",
        "independent_holdout": True, "model_sha256": model_sha,
        "baseline": "month-of-year climatology", "model": "stand-in", "note": "stand-in",
        **S.E04_PROTOCOL}) + "\n")
    return p


def _frame():
    lay = ST.e04_layout()
    f, _ = ST.e04_frame(lay, S.world(lay, 0.86, 0.15, S.STREAM_SELFCHECK, 9, 0,
                                     seed=S.SMOKE_SEED))
    return lay, f


# ----------------------------------------------------------------------------- static
def test_new_files_conventions():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in [*sorted(D.glob("*.py")), Path(__file__)]:
        txt = p.read_text(encoding="utf-8")
        assert "from __future__ import annotations" in txt, p
        assert not pat.search(txt), p
        if p.parent == D:                     # the staged code never registers anything
            assert "write_prereg(" not in txt, p


def test_v1_design_copied_verbatim():
    assert S.RHO_GRID == ST.RHO_GRID == (0.80, 0.86, 0.93)
    assert S.SHARE_GRID == ST.SHARE_GRID == (0.0, 0.15, 0.30)
    assert S.K_STAT == ST.K_STAT == 500
    assert S.E04_PROTOCOL == ST.E04_PROTOCOL == {
        "max_rmse_ratio": 1.0, "min_stations": 100, "min_months": 6,
        "coverage_bounds": [0.85, 0.95]}
    assert S.cells() == ST.cells() and len(S.cells()) == 9
    assert [c["rho"] for c in S.null_cells()] == [0.0] * 3
    assert {S.STREAM_E04, S.STREAM_NULL, S.STREAM_SELFCHECK}.isdisjoint(
        {ST.STREAM_E04, ST.STREAM_T1, ST.STREAM_SELFCHECK})
    assert {ST.SELF, OF.SELF, S.SELF} <= set(S.FROZEN)
    # v1 froze neither osse_fv2.py nor score_e02c.py; v2 freezes both, and release.py
    assert {"dataprep_staging/fv_decidability_v2/osse_fv2.py",
            "dataprep_staging/e02c/score_e02c.py", "hydrophysics/twin/release.py"} <= set(
        S.FROZEN)
    assert all((REPO / p).is_file() for p in S.FROZEN)


# ----------------------------------------------------------------------------- seeds
def test_seed_has_no_default_and_previewed_seeds_are_refused():
    for fn in (S._score, S.run, S.verify, S.engineering, S.rng_for, PR.draft_prereg):
        p = inspect.signature(fn).parameters["seed"]
        assert p.default is inspect.Parameter.empty, fn.__name__
    for bad in (20261103, ST.SMOKE_SEED, S.SMOKE_SEED):
        with pytest.raises(S.EngineeringFailure, match="refused"):
            S.check_seed(bad)
        with pytest.raises(S.EngineeringFailure, match="refused"):
            PR.draft_prereg(bad)
    for bad in (True, -1, 1.5, "7"):
        with pytest.raises(S.EngineeringFailure):
            S.check_seed(bad)
    assert S.check_seed(123456789) == 123456789
    with pytest.raises(SystemExit):
        S.main(["run", "--prereg-id", S.ID])                 # --seed is required


# ----------------------------------------------------------------------------- self-check
def test_v1_selfcheck_fails_against_a_real_style_protocol(tmp_path):
    """The v1 bug: rows tagged 'synthetic' are refused by a protocol with a real sha."""
    lay = ST.e04_layout()
    with pytest.raises(ValueError, match=S.REFUSAL):
        ST.e04_selfcheck(lay, _protocol(tmp_path), tmp_path / "t", S.E04_PROTOCOL,
                         seed=S.SMOKE_SEED)


def test_v2_selfcheck_passes_on_the_synthetic_copy_and_control_refuses(tmp_path):
    lay, f = _frame()
    pp = _protocol(tmp_path)
    real = json.loads(pp.read_text())
    rec = S.selfcheck_world(f, S._origins(lay), pp, real, S.protocol_rule(real), tmp_path / "t")
    assert rec["identical"] and rec["control_refused"] and rec["control_message"] == S.REFUSAL
    assert not list((tmp_path / "t").iterdir())               # temporary files removed


def test_synthetic_copy_differs_from_real_only_in_model_sha(tmp_path):
    real = json.loads(_protocol(tmp_path).read_text())
    syn = S.synthetic_protocol(real, "cd" * 32)
    assert syn["model_sha256"] == "cd" * 32
    assert {k: v for k, v in syn.items() if k != "model_sha256"} == {
        k: real[k] for k in S.EVAL_KEYS}
    with pytest.raises(S.EngineeringFailure, match="lacks"):
        S.synthetic_protocol({k: v for k, v in real.items() if k != "test_end"}, "cd" * 32)


def test_mismatched_sha_is_refused_by_the_real_protocol(tmp_path):
    """Direct check of the control: rows whose sha differs from the protocol's are refused,
    and accepted once the protocol carries their sha."""
    from hydrophysics.twin.validation import evaluate

    lay, f = _frame()
    ev = tmp_path / "evaluation.csv"
    orig = S._origins(lay)
    f.assign(date=f["date"].dt.strftime("%Y-%m-%d"), prediction_origin=f["station_id"].map(orig),
             model_sha256="cd" * 32).to_csv(ev, index=False)
    with pytest.raises(ValueError, match=S.REFUSAL):
        evaluate(ev, _protocol(tmp_path))
    ok = tmp_path / "ok"
    ok.mkdir()
    assert evaluate(ev, _protocol(ok, "cd" * 32))["stations"] == len(lay.ids)


def test_a_control_that_accepts_is_an_engineering_failure(tmp_path, monkeypatch):
    lay, f = _frame()
    pp = _protocol(tmp_path)
    real = json.loads(pp.read_text())
    monkeypatch.setattr(S, "synthetic_sha", lambda path: FAKE_MODEL_SHA)
    with pytest.raises(S.EngineeringFailure, match="vacuous"):
        S.selfcheck_world(f, S._origins(lay), pp, real, S.protocol_rule(real), tmp_path / "t")


def test_control_refusing_for_another_reason_is_an_engineering_failure(tmp_path):
    lay, f = _frame()
    pp = _protocol(tmp_path)
    real = json.loads(pp.read_text())
    bad = json.loads(pp.read_text())
    bad["test_end"] = "2027-06-30"                     # rows fall outside the "real" window
    bp = tmp_path / "bad.json"
    bp.write_text(json.dumps(bad))
    with pytest.raises(S.EngineeringFailure, match="another reason"):
        S.selfcheck_world(f, S._origins(lay), bp, real, S.protocol_rule(real), tmp_path / "t")


def test_protocol_rule_must_equal_v1s_copy(tmp_path):
    real = json.loads(_protocol(tmp_path).read_text())
    assert S.protocol_rule(real) == S.E04_PROTOCOL
    with pytest.raises(S.EngineeringFailure, match="differs"):
        S.protocol_rule({**real, "max_rmse_ratio": 0.95})


# ----------------------------------------------------------------------------- wording map
def _fab(p_true: float, n_cells: int, k: int = 200) -> list[list[dict]]:
    n = round(p_true * k)
    ws = [{"pass": i < n, "ratio": 0.9, "coverage": 0.9} for i in range(k)]
    return [ws] * n_cells


@pytest.mark.parametrize("p,want", [(0.92, "not_shown_weak"), (0.505, "not_shown_weak"),
                                    (0.5, "weak_by_design"), (0.3, "weak_by_design")])
def test_wording_map_on_fabricated_records(p, want):
    P = S.pooled(S.summarise(S.cells(), _fab(p, 9)))
    P0 = S.pooled(S.summarise(S.null_cells(), _fab(0.45, 3)))
    rd = S.readings(P, P0)
    assert rd["e04"] == want
    en = rd["wording"]["en"]
    assert f"{P['p']:.2f}" in en and f"{P0['p']:.2f}" in en and "lag-1 0" in en
    assert ("weak by design" in en) == (want == "weak_by_design")
    clause = rd["what_a_2028_pass_shows"]
    assert "What it cannot show" in clause["en"] and clause["zh"]
    assert "no worse than climatology" in clause["en"] and "ties pass" in clause["en"]
    assert "beat" not in clause["en"] and "勝過" not in clause["zh"]
    assert "does not establish persistence" in clause["en"]
    assert f"{1 - P['p']:.2f}" in clause["en"] and f"{1 - P['p']:.2f}" in clause["zh"]
    assert "more informative" in clause["en"]


def test_mc_failure_reads_not_evaluable():
    P = {"p": 0.9, "mc_se": 0.1, "mc_half_width_90": 0.2}
    rd = S.readings(P, {"p": 0.45})
    assert rd["e04"] is None and rd["what_a_2028_pass_shows"] is None
    assert "not evaluable" in rd["wording"]["en"]


def test_wording_templates_are_clean():
    txt = json.dumps([S.WORDING, S.PASS_CLAUSE], ensure_ascii=False)
    assert not STATION.search(txt)
    for k in ("weak_by_design", "not_shown_weak"):
        for lang in ("en", "zh"):
            assert "{p:.2f}" in S.WORDING[k][lang] and "{p0:.2f}" in S.WORDING[k][lang]
            assert "K = 8" in S.WORDING[k][lang]


# ----------------------------------------------------------------------------- runner
def test_score_is_private_and_refuses_without_a_registration(tmp_path):
    assert not hasattr(S, "score")
    assert inspect.signature(S._score).parameters["decl_sha"].default is inspect.Parameter.empty
    for bad in (None, "", "f" * 10):
        with pytest.raises(S.EngineeringFailure, match="only under its registration"):
            S._score(seed=123456789, protocol_path=_protocol(tmp_path), tmp_root=tmp_path,
                     decl_sha=bad)


def _attempts(out: Path) -> list[dict]:
    p = out.parent / S.ATTEMPTS_NAME
    return [json.loads(x) for x in p.read_text().splitlines()]


def test_run_logs_every_attempt(tmp_path):
    reg = tmp_path / "prereg"
    reg.mkdir()
    out = tmp_path / "rep" / "report.json"
    assert S.attempts_path(str(out)) == out.parent / S.ATTEMPTS_NAME
    with pytest.raises(S.EngineeringFailure, match="refused"):
        S.run(20261103, S.ID, out=str(out), prereg_dir=reg)
    with pytest.raises(S.EngineeringFailure, match="not in INDEX.jsonl"):
        S.run(123456789, S.ID, out=str(out), prereg_dir=reg)
    rows = _attempts(out)
    assert [(r["event"], r["attempt"]) for r in rows] == [
        ("start", 1), ("engineering_failure", 1), ("start", 2), ("engineering_failure", 2)]
    assert "refused" in rows[1]["message"] and "INDEX.jsonl" in rows[3]["message"]
    assert all(r["id"] == S.ID and r["time"] for r in rows)
    assert rows[0]["seed"] == 20261103 and len(rows[0]["code_sha256"]) == 64
    assert not out.exists()


def test_run_logs_success_with_the_report_sha(monkeypatch, tmp_path):
    out = tmp_path / "rep" / "report.json"
    monkeypatch.setattr(S, "verify", lambda *a, **k: ({"frozen_code": {}}, "f" * 64,
                                                      tmp_path / "p.json"))
    monkeypatch.setattr(S, "_score", lambda **k: {"status": "scored"})   # no world is drawn
    S.run(123456789, S.ID, out=str(out))
    rows = _attempts(out)
    assert [r["event"] for r in rows] == ["start", "success"]
    assert rows[1]["report_sha256"] == OF.sha256(out)
    assert set(rows[1]) == {"id", "time", "event", "attempt", "report_sha256",
                            "declaration_sha256"}


def test_run_logs_other_errors(monkeypatch, tmp_path):
    out = tmp_path / "rep" / "report.json"

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(S, "verify", boom)
    with pytest.raises(RuntimeError):
        S.run(123456789, S.ID, out=str(out))
    rows = _attempts(out)
    assert rows[-1]["event"] == "error" and "RuntimeError: boom" in rows[-1]["message"]


def test_smoke_shape_check_is_shape_only():
    lay, f = _frame()
    assert S.frame_shape_ok(lay, f)
    assert not S.frame_shape_ok(lay, f.iloc[:-1])
    assert not S.frame_shape_ok(lay, f.drop(columns=["baseline_m"]))
    src = inspect.getsource(S.run_smoke)
    assert "cell_records" not in src and "e04_pass" not in src


def test_run_refuses_unregistered_id_and_refused_seed(tmp_path):
    reg = tmp_path / "prereg"
    reg.mkdir()
    out = tmp_path / "report.json"
    with pytest.raises(S.EngineeringFailure, match="not in INDEX.jsonl"):
        S.run(123456789, S.ID, out=str(out), prereg_dir=reg)
    with pytest.raises(S.EngineeringFailure, match="refused"):
        S.run(20261103, S.ID, out=str(out), prereg_dir=reg)
    with pytest.raises(S.EngineeringFailure, match="scores FV.stat.v2 only"):
        S.verify(123456789, "FV.stat.v1", prereg_dir=reg)
    out.write_text("{}")
    with pytest.raises(S.EngineeringFailure, match="scored once"):
        S.run(123456789, S.ID, out=str(out), prereg_dir=reg)
    assert S.main(["run", "--seed", "123456789", "--prereg-id", S.ID, "--out",
                   str(tmp_path / "r2.json")], prereg_dir=reg) == 1
    assert not (tmp_path / "r2.json").exists()


def _fake_registry(monkeypatch, doc):
    monkeypatch.setattr(S, "index_record", lambda *a, **k: {"id": S.ID})
    from harness import prereg3

    monkeypatch.setattr(prereg3, "load_prereg", lambda *a, **k: (doc, "f" * 64))
    monkeypatch.setattr(prereg3, "changed_inputs", lambda d: [])


def test_run_refuses_changed_code_and_a_different_seed(monkeypatch, tmp_path):
    fc = {p: OF.sha256(OF.rp(p)) for p in S.FROZEN}
    design = {"seed": 123456789, "K_STAT": S.K_STAT, "K_NULL": S.K_NULL,
              "rho_grid": list(S.RHO_GRID), "share_grid": list(S.SHARE_GRID),
              "null_rho": S.NULL_RHO, "streams": {"e04": S.STREAM_E04, "null": S.STREAM_NULL,
                                                  "selfcheck": S.STREAM_SELFCHECK}}
    doc = {"frozen_code": {**fc, S.SELF: "0" * 64}, "design": design,
           "e04": {"protocol_sha256": S.E04_PROTOCOL_SHA}}
    _fake_registry(monkeypatch, doc)
    with pytest.raises(S.EngineeringFailure, match="code sha256 differs"):
        S.verify(123456789, S.ID)
    doc["frozen_code"] = fc
    with pytest.raises(S.EngineeringFailure, match="seed"):
        S.verify(987654321, S.ID)
    doc["frozen_code"] = {k: v for k, v in fc.items() if k != ST.SELF}
    with pytest.raises(S.EngineeringFailure, match="exactly"):
        S.verify(123456789, S.ID)


def test_smoke_refuses_results_dir():
    with pytest.raises(S.EngineeringFailure, match="results/"):
        S.run_smoke(REPO / "results" / "fv_stat_v2_smoke_should_not_exist")
    assert not (REPO / "results" / "fv_stat_v2_smoke_should_not_exist").exists()


# ----------------------------------------------------------------------------- draft
def test_draft_declaration_key_fields():
    d = PR.draft_prereg(123456789)
    from harness import prereg

    assert set(prereg.REQUIRED) <= set(d)
    assert d["id"] == "FV.stat.v2" and d["kind"] == "declaration" and "created" not in d
    assert d["design"]["seed"] == 123456789 and "20261103" in d["design"]["refused_seeds"]
    assert d["attempt_policy"]["exploratory_followups_max"] == 0
    assert "scored once" in d["attempt_policy"]["scored_runs"]
    dis = json.dumps(d["disclosures"])
    assert "ac83bd46" in dis and "20261103" in dis
    assert "R8:e04_hybrid_parameters_disclosure" in dis
    assert "R8:fv_stat_v2_builder_reads_disclosure" in dis and "2026-11-03T01:39Z" in dis
    assert "FV2:builder_accidental_e04_rows" in dis
    assert "reroll" in d["disclosures"]["seed"] and "0.92" in d["disclosures"]["seed"]
    assert "41-43" in d["disclosures"]["seed"]
    for k in ("v1_frozen_code_incomplete", "t1_t2_dropped", "followups_tightening",
              "v1_builder_look", "r8_builder_reads_look"):
        assert d["disclosures"][k], k
    assert "osse_fv2.py" in dis and "score_e02c.py" in dis
    assert set(PR.LOGGED_LOOKS) <= set(d["looks"])
    assert S.ATTEMPTS_NAME in d["attempt_policy"]["attempts_log"]
    assert d["readings"]["null_reading"]["report_only"] is True
    assert set(d["frozen_code"]) == set(S.FROZEN)
    assert not STATION.search(json.dumps(d, ensure_ascii=False))


def test_readiness_checks_the_two_looks_only_when_looks_jsonl_exists(tmp_path):
    reg = tmp_path / "prereg"
    reg.mkdir()
    d = PR.draft_prereg(123456789, prereg_dir=reg)
    assert all(d["looks"][k] == PR.PENDING for k in PR.LOGGED_LOOKS)
    units = [u for u, _ in PR.LOGGED_LOOKS.values()]
    msgs = PR.readiness(d, prereg_dir=reg)
    assert not any(u in m for m in msgs for u in units)          # file absent: skipped
    (reg / "looks.jsonl").write_text("")
    msgs = PR.readiness(d, prereg_dir=reg)
    assert all(any(u in m for m in msgs) for u in units)


@NEED_RESULTS
def test_draft_validates_against_the_local_registry():
    from harness import prereg3

    d = PR.draft_prereg(123456789)
    assert prereg3.validate_prereg(d) == []
    assert d["e04"]["protocol_sha256"] == S.E04_PROTOCOL_SHA
    assert d["v1"]["engineering_failure"]["sha256"] == PR.V1_FAILURE_SHA
    assert d["looks"]["v1_stat_preview"] != PR.PENDING
    assert d["looks"]["r8_e04_parameters_disclosure"] != PR.PENDING
    for k in PR.LOGGED_LOOKS:
        assert d["looks"][k] != PR.PENDING, k
    assert not any("look" in m for m in PR.readiness(d))
