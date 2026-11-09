"""C14f.coverage_clause.v1 (round 11, user ruling R11.7): synthetic worlds only.

Nothing under results/ can be read here: a module fixture makes osse_fv.rp, osse_fv.sha256,
numpy.load, open/io.open, Path.read_text/read_bytes and pandas.read_csv raise on any path that
is, or resolves into, the results tree (run_c14f._under_results). Every input comes from
FV.ext.v1's own synthetic builder (ext_fv.build_synthetic).

- the reused FV.ext.v1 code is pinned and unchanged;
- R9.4's three outcomes on fabricated numbers (FV.ext.v1's whole-rule reading, P 0.592 with
  null 0.0, must read 'unchanged' as R10.5 graded it), and the EN/ZH wording of each;
- the recorder is a pass-through: the frozen scorer's report is identical with and without it;
- the call split and the clause: the clause passes whenever the whole rule passes;
- gate I refuses another reference; a synthetic end to end; the smoke never writes under
  results/;
- scoring is private: a real run needs this declaration registered, refuses before logging;
- the reading is model-only;
- the draft declaration's key fields (label, rule, attempt policy, what it can change).
"""
from __future__ import annotations

import builtins
import copy
import importlib.util
import io
import json
import pathlib
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")         # hydrophysics.twin imports it on the self-check path

REPO = Path(__file__).resolve().parents[1]
D = REPO / "dataprep_staging" / "c14f_clause"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_saved = list(sys.path)
R = _load("run_c14f", D / "run_c14f.py")
PR = _load("prereg_c14f", D / "prereg_c14f.py")
sys.path[:] = _saved
O = R.O  # noqa: E741  (osse_fv, frozen)
XF = R.XF
STATION = re.compile(r"\b(?:0[79]|10)[0-9]{6}\b")
K_NULL = 20


class ResultsRead(AssertionError):
    """A test tried to read a file under results/."""


@pytest.fixture(scope="module", autouse=True)
def no_results_reads():
    import numpy as np
    import pandas as pd

    def check(p):
        if isinstance(p, (str, pathlib.PurePath)) and str(p) and R._under_results(p):
            raise ResultsRead(f"a test read under results/: {p}")

    def wrap(fn, path_of=lambda a, k: a[0] if a else None):
        def inner(*a, **k):
            check(path_of(a, k))
            return fn(*a, **k)
        return inner

    mp = pytest.MonkeyPatch()
    mp.setattr(O, "rp", wrap(O.rp))
    mp.setattr(O, "sha256", wrap(O.sha256))
    mp.setattr(np, "load", wrap(np.load))
    mp.setattr(builtins, "open", wrap(builtins.open))
    mp.setattr(io, "open", wrap(io.open))
    mp.setattr(pd, "read_csv", wrap(pd.read_csv))
    for name in ("read_text", "read_bytes"):
        mp.setattr(pathlib.Path, name, wrap(getattr(pathlib.Path, name)))
    saved = list(sys.path)
    for p in (str(D), str(REPO / "dataprep_staging" / "fv_decidability_v2"),
              str(REPO / "dataprep_staging" / "fv_decidability"),
              str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"),
              str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    yield
    sys.path[:] = saved
    mp.undo()


def _quiet(*_):
    return None


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return R.build_synthetic(tmp_path_factory.mktemp("c14f") / "inputs")


@pytest.fixture(scope="module")
def reference(synth):
    rep = XF.score_ext(synth.ext, plan=R.SYNTH_PLAN, smoke=True, k_null=K_NULL, log=_quiet)
    return json.loads(json.dumps(rep, default=O._jsonable))


# ---- guards and pins --------------------------------------------------------------------------
def test_the_read_guard_holds():
    for p in ("results/twin_forward/fv2_truth_R0_b0.npz", R.FVEXT_REPORT, R.FVEXT_DECL):
        with pytest.raises(ResultsRead):
            O.rp(p)
        with pytest.raises(ResultsRead):
            open(R.REPO / p).close()  # noqa: SIM115
        with pytest.raises(ResultsRead):
            (R.REPO / p).read_text()


def test_new_files_conventions():
    for p in (D / "run_c14f.py", D / "prereg_c14f.py", Path(__file__)):
        src = p.read_text(encoding="utf-8")
        assert "from __future__ import annotations" in src, p
        assert "/" + "home/" not in src, p
        assert not STATION.search(src), p
        assert max(len(x) for x in src.splitlines()) <= 100, p


def test_reused_code_is_fvext_frozen_code_and_unchanged():
    assert set(R.REUSED) == set(XF.FROZEN)
    assert not R.reused_code_problems()
    assert (R.SELF, *R.REUSED) == R.FROZEN


def test_reused_code_problems_sees_a_changed_file(tmp_path):
    for p in R.REUSED:
        q = tmp_path / p
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes((R.REPO / p).read_bytes())
    assert not R.reused_code_problems(tmp_path)
    q = tmp_path / "dataprep_staging/fv_decidability_v2/ext_fv.py"
    q.write_text(q.read_text() + "\n# changed\n")
    assert R.reused_code_problems(tmp_path)


def test_design_is_fvext_seed_k_and_streams():
    d = R.design()
    assert d["seed"] == O.SEED == 20261101
    assert (d["K_TRUTH"], d["K_NULL"], d["batch"]) == (250, 500, R.V2.PLAN.batch)
    assert d["streams"] == {"truth_noise": 21, "null": 22, "selfcheck": 23}
    assert d["clause"] == {"coverage_bounds": [0.85, 0.95], "min_stations": 30,
                           "min_months": 6, "max_rmse_ratio": 1.0}


# ---- outcomes and wording ---------------------------------------------------------------------
@pytest.mark.parametrize("ns, n0, hp, hp0, want", [
    ((142, 136, 155, 159), 0, 0.025, 0.0, "unchanged"),   # FV.ext.v1's whole rule (0.592)
    ((150, 151, 250, 249), 0, 0.025, 0.0, "measured"),    # float mean 0.7999999999999999
    ((150, 151, 250, 249), 100, 0.025, 0.03, "measured"),  # P0 exactly 1/5: inclusive
    ((150, 151, 250, 248), 0, 0.025, 0.0, "unchanged"),   # 799/1000
    ((250, 250, 250, 250), 101, 0.025, 0.03, "unchanged"),
    ((250, 250, 250, 250), 251, 0.025, 0.03, "weak"),     # the null meets the band too often
    ((125, 125, 125, 124), 0, 0.025, 0.0, "weak"),        # 499/1000
    ((125, 125, 125, 125), 250, 0.025, 0.03, "unchanged"),  # exactly 1/2 on both sides
    ((225, 225, 225, 225), 0, 0.08, 0.0, "not_evaluable"),
    ((225, 225, 225, 225), 0, float("nan"), 0.0, "not_evaluable"),
    ((225, 225, 225, 251), 0, 0.02, 0.0, "not_evaluable"),  # a count above K
    ((225, 225, 225, 22.5), 0, 0.02, 0.0, "not_evaluable"),  # not an integer count
    ((225, 225, 225), 0, 0.02, 0.0, "not_evaluable"),        # a truth missing
])
def test_the_outcomes_follow_r9_4_on_integer_counts(ns, n0, hp, hp0, want):
    assert R.outcome_of(list(ns), 250, n0, 500, hp, hp0) == want


def test_the_float_mean_would_have_misread_the_boundary():
    ns = (150, 151, 250, 249)
    acc = 0.0                      # naive left-to-right float sum (Python >= 3.12's sum() is
    for n in ns:                   # compensated and would hide the rounding this test shows)
        acc += n / 250
    fmean = acc / 4
    assert fmean == 0.7999999999999999 and fmean < 0.8
    assert R.exact_p(list(ns), 250) == R.Fraction(4, 5) == R.P_MEASURED
    assert R.outcome_of(list(ns), 250, 0, 500, 0.02, 0.0) == "measured"
    assert isinstance(R.P_MEASURED, R.Fraction) and isinstance(R.P0_WEAK, R.Fraction)


def test_every_outcome_has_prewritten_wording_en_zh():
    for out in R.OUTCOMES:
        w = R.wording_for(out, p=0.45 if out == "weak" else 0.85, p0=0.6 if out == "weak"
                          else 0.05)
        assert set(w) == {"en", "zh"}
        assert "{" not in w["en"] + w["zh"]
        assert "0.850" in w["en"] or "0.450" in w["en"]
        assert "R11.7" in w["en"] and "R11.7" in w["zh"]
        assert "validation" in w["en"]
    w = R.wording_for("measured", p=R.Fraction(4, 5), p0=R.Fraction(0, 500))
    assert "0.800" in w["en"] and "0.000" in w["en"] and "0.800" in w["zh"]
    w = R.wording_for(R.NOT_EVALUABLE, why="x")
    assert "x" in w["en"] and "x" in w["zh"]
    assert set(R.PROPOSED_GRADE) == {*R.OUTCOMES, R.NOT_EVALUABLE}
    with pytest.raises(R.EngineeringFailure):
        R.wording_for("validated", p=1, p0=0)


# ---- recorder, split, clause ------------------------------------------------------------------
def test_the_recorder_is_a_pass_through(synth, reference):
    with R.recording() as calls:
        rep = XF.score_ext(synth.ext, plan=R.SYNTH_PLAN, smoke=True, k_null=K_NULL, log=_quiet)
    assert XF.rule is not None and XF.rule.__name__ == "rule"       # restored
    assert not R.identity_problems(rep, reference)
    parts = R.split_calls(calls, K_NULL, R.SYNTH_PLAN.k_truth)
    assert len(parts["null"]) == K_NULL
    ctx = XF.load_ext(synth.ext)
    ok = R.enough(ctx)
    assert ok and R.protocol_rule(ctx["protocol"]) == R.N3F_RULE
    R.check_calls(parts, ok, rep)
    for t in R.TAGS:
        for c in parts[t]["prim"] + parts[t]["reg"]:
            assert R.clause_pass(c["coverage"], ok) or not c["pass"]   # whole => clause
    with pytest.raises(R.EngineeringFailure):
        R.split_calls(calls[:-1], K_NULL, R.SYNTH_PLAN.k_truth)
    bad = copy.deepcopy(parts)
    bad[R.TAGS[0]]["reg"][0]["coverage"] += 0.01
    with pytest.raises(R.EngineeringFailure):
        R.check_calls(bad, ok, rep)


def test_clause_bounds_are_closed_and_need_enough():
    assert R.clause_pass(0.85, True) and R.clause_pass(0.95, True)
    assert not R.clause_pass(0.8499, True) and not R.clause_pass(0.9501, True)
    assert not R.clause_pass(0.9, False)


def test_protocol_rule_refuses_another_rule():
    with pytest.raises(R.EngineeringFailure):
        R.protocol_rule({"coverage_bounds": [0.8, 0.95], "min_stations": 30, "min_months": 6,
                         "max_rmse_ratio": 1.0})


def test_gate_i_refuses_another_reference(synth, reference):
    bad = copy.deepcopy(reference)
    bad["truth_worlds"][R.TAGS[0]]["passed"]["p"] += 0.004
    with pytest.raises(R.EngineeringFailure, match="gate I"):
        R._score(synth, smoke=True, plan=R.SYNTH_PLAN, k_null=K_NULL, reference=bad,
                 log=_quiet)
    assert not R.identity_problems(
        {**reference, "created": "x", "frozen_code": {"a": 1}}, reference)


def test_synthetic_end_to_end(synth, reference):
    rep = R._score(synth, smoke=True, plan=R.SYNTH_PLAN, k_null=K_NULL, reference=reference,
                   log=_quiet)
    assert rep["status"] in ("scored", "not_evaluable")
    assert rep["label"].startswith("post-failure, exploratory")
    for t in R.TAGS:
        tw = rep["truth_worlds"][t]
        assert tw["clause_passed"]["p"] >= tw["whole_rule_passed_fvext"]["p"]
        assert tw["clause_passed"]["k"] == R.SYNTH_PLAN.k_truth
        s = tw["fail_side_share"]
        assert abs(1 - s["below_0.85"] - s["above_0.95"] - tw["clause_passed"]["p"]) < 1e-12
    assert rep["P0"]["p"] == rep["null_world"]["clause_passed"]["p"]
    assert rep["P0"]["n"] == rep["null_world"]["clause_passed"]["n"]
    assert rep["P"]["n_total"] == sum(rep["truth_worlds"][t]["clause_passed"]["n"]
                                      for t in R.TAGS)
    assert rep["P"]["k_total"] == len(R.TAGS) * R.SYNTH_PLAN.k_truth
    assert rep["design_time_look"] == {"unit": XF.EXT_LOOK_UNIT, "new_look_logged": False,
                                       "ruling": "R11.7d", "logged_look": None}   # smoke
    assert set(rep["engineering"]["versions"]) == {"python", "numpy", "pandas"}
    assert "never change the outcome" in rep["report_only_note"]
    assert rep["report_only"]["midpoint_null"]["clause_passed"]["k"] == K_NULL
    assert rep["outcome"] == R.outcome_of([rep["P"]["n_by_truth"][t] for t in R.TAGS],
                                          R.SYNTH_PLAN.k_truth, rep["P0"]["n"], K_NULL,
                                          rep["P"]["mc_half_width_90"],
                                          rep["P0"]["mc_half_width_90"])
    assert rep["ledger_mapping"]["proposed_grade_key"] == R.PROPOSED_GRADE[rep["outcome"]]


def test_smoke_writes_once_and_never_under_results(tmp_path):
    rep = R.run_smoke(tmp_path / "s", k_null=K_NULL)
    assert (tmp_path / "s" / "smoke_report.json").is_file()
    assert rep["smoke"] is True
    with pytest.raises(R.EngineeringFailure):
        R.run_smoke(R.REPO / "results" / "c14f_smoke")
    with pytest.raises(R.EngineeringFailure):
        R.build_synthetic(R.REPO / "results" / "c14f_inputs")


def test_a_smoke_needs_synthetic_paths_and_a_reference(synth, reference):
    with pytest.raises(R.EngineeringFailure):
        R._score(R.ClausePaths(), smoke=True, plan=R.SYNTH_PLAN, k_null=K_NULL,
                 reference=reference, log=_quiet)
    with pytest.raises(R.EngineeringFailure):
        R._score(synth, smoke=True, plan=R.SYNTH_PLAN, k_null=K_NULL, log=_quiet)


# ---- scoring is private -----------------------------------------------------------------------
def test_a_real_run_refuses_another_design_or_a_given_reference(reference):
    with pytest.raises(R.EngineeringFailure):
        R._score(R.ClausePaths(), plan=R.SYNTH_PLAN, log=_quiet)
    with pytest.raises(R.EngineeringFailure):
        R._score(R.ClausePaths(), reference=reference, log=_quiet)


def test_run_refuses_before_logging_without_registration(tmp_path, synth):
    (tmp_path / "INDEX.jsonl").write_text("")
    with pytest.raises(R.EngineeringFailure, match="register it first"):
        R.run(synth, prereg_dir=tmp_path, rows_dir=tmp_path)
    assert not R.attempts_path(synth.report).exists()
    assert not Path(synth.report).exists()
    with pytest.raises(R.EngineeringFailure):
        R.load_registered("FV.ext.v1", prereg_dir=tmp_path)


# ---- model-only -------------------------------------------------------------------------------
def test_the_reading_is_model_only(tmp_path, monkeypatch, synth, reference):
    d = PR.draft_prereg(prereg_dir=tmp_path, P=synth)
    assert "never reads an observed head" in d["model_only"]["statement"]
    obs = re.compile(r"heads/L1|AMP_V2|wells\.parquet|ls_cache|observations?\.csv")
    assert not [p for p in d["inputs"] if obs.search(p)]
    code = re.sub(r'"""(?:.|\n)*?"""', "", (D / "run_c14f.py").read_text(encoding="utf-8"))
    assert not re.search(r"L1_clean|AMP_V2|read_parquet|open_product|heads/L1", code)
    opened: list[str] = []
    real_open, np_load = builtins.open, R.np.load

    def rec_open(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)

    def rec_load(f, *a, **k):
        opened.append(str(f))
        return np_load(f, *a, **k)

    monkeypatch.setattr(builtins, "open", rec_open)
    monkeypatch.setattr(R.np, "load", rec_load)
    R._score(synth, smoke=True, plan=R.SYNTH_PLAN, k_null=4, log=_quiet,
             reference=json.loads(json.dumps(XF.score_ext(
                 synth.ext, plan=R.SYNTH_PLAN, smoke=True, k_null=4, log=_quiet),
                 default=O._jsonable)))
    monkeypatch.undo()
    data = [f for f in opened if not f.endswith((".py", ".pyc")) and "site-packages" not in f]
    root = str(Path(synth.ext.ext_dir).resolve().parent)
    assert data and all(str(Path(f).resolve()).startswith(root) for f in data), data


# ---- the draft declaration --------------------------------------------------------------------
def test_draft_declaration_key_fields(tmp_path, synth):
    d = PR.draft_prereg(prereg_dir=tmp_path, P=synth)
    assert d["id"] == R.ID and d["kind"] == "declaration" and d["wp"] == "FV"
    assert d["label"].startswith("post-failure, exploratory")
    assert "post-failure, exploratory" in d["attempt_policy"]["label"]
    assert d["attempt_policy"]["exploratory_followups_max"] == 0
    assert d["attempt_policy"]["thresholds_frozen"] is True
    assert "one" in d["attempt_policy"]["scored_runs"]
    assert d["readings"]["outcomes"]["measured"] == "P >= 0.8 and P0 <= 0.2"
    assert d["readings"]["outcomes"]["weak"].startswith("P < 0.5 or P0 > 0.5")
    assert d["design"] == json.loads(json.dumps(R.design()))
    assert d["rule"]["declared"] == R.N3F_RULE
    assert d["rule"]["stated"] == R.N3F_RULE           # the synthetic export states it too
    assert any("later user ruling" in x for x in d["can_change"])
    assert "C14f's status (D)" in d["cannot_change"]
    assert d["fresh_units"] == {"not_applicable": "no observation is scored"}
    assert set(d["frozen_code"]) == set(R.FROZEN)
    assert set(d["rulings"]) == {"R11.7", "R11.7a", "R11.7b", "R11.7c", "R11.7d", "R10.5",
                                 "R9.4", "R9.4b"}
    assert R.RULINGS["R11.7d"] == (
        "results/dataprep/rulings/user_rulings_2026-11-09.json",
        "7c7219b85ef5ee83d7e1fd88b391367272c81f72b9edb985f2bd325052dc05d1")
    assert any(p.endswith("rulings_R11.7a.json") for p in d["inputs"])
    note = " ".join(d["readings"]["threshold_note"])
    assert "unregistered dataprep_staging/e02_decid/run_e02d1.py" in note
    assert "first registered statement of the weak pair" in note
    assert "51d7908" in d["status_note"] and "51d7908" in d["disclosures"]["code"]
    assert "same build checkout" in d["disclosures"]["registration_paths"]
    assert "DATAPREP_REPO is not set" in d["disclosures"]["registration_paths"]
    assert "R11.7d" in d["disclosures"]["design_time_look"]
    assert "0.592" in d["priors"]
    assert not any("/" + "home/" in p for p in d["inputs"])
    rd = PR.readiness(d, P=synth, prereg_dir=tmp_path)
    assert any("FV.ext.v1" in x for x in rd)            # synthetic: no FV.ext.v1 at its sha


def test_the_report_is_written_once_atomically(tmp_path):
    p = tmp_path / "r" / "report.json"
    R.write_once_atomic(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    with pytest.raises(FileExistsError):
        R.write_once_atomic(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}
    assert sorted(x.name for x in p.parent.iterdir()) == ["report.json"]   # no temp left
