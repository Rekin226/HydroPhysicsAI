"""E02.decidability.v1 (round 9, user ruling R9.4): synthetic worlds only.

Nothing under results/ can be read here: a module fixture makes osse_fv.rp, osse_fv.sha256,
numpy.load, open/io.open, Path.read_text/read_bytes and pandas.read_csv raise on any path that
is, or resolves into, the results tree (run_e02d1._under_results, which also sees a symlinked
results/). Every input, the draft declaration included, comes from run_e02d1.build_synthetic
(osse_fv2.build_synthetic underneath).

- the reused FV v2 / FV.stat.v2 code is pinned and unchanged; FV v2's own loaders build the
  right-kind worlds;
- the three pre-declared outcomes and their thresholds, on fabricated numbers (including
  FV.stat.v2's published P / P0, which must read 'unchanged' as ruling R8.17b graded E04);
- the pre-written EN/ZH wording of every outcome;
- the equal-skill null: midpoint truth, and its ratio half is a fair coin;
- a synthetic end-to-end score; gate E's checks (frame order, protocol rule, world files,
  FV v2's attempts log);
- scoring is private: a real run loads the registration itself; the runner refuses before it
  logs or reads anything, refuses changed code, another design and an existing report; a
  temporary registry never unlocks the real paths;
- the reading is model-only;
- the draft declaration's key fields and validation (synthetic sources), and the progress-map
  spec it declares is the committed one.
"""
from __future__ import annotations

import builtins
import dataclasses
import importlib.util
import io
import json
import pathlib
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")         # hydrophysics.twin.coverage imports it

REPO = Path(__file__).resolve().parents[1]
D = REPO / "dataprep_staging" / "e02_decid"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_saved = list(sys.path)
R = _load("run_e02d1", D / "run_e02d1.py")
PR = _load("prereg_e02d1", D / "prereg_e02d1.py")
sys.path[:] = _saved
O = R.O  # noqa: E741  (osse_fv, frozen)
STATION = re.compile(r"\b(?:0[79]|10)[0-9]{6}\b")


class ResultsRead(AssertionError):
    """A test tried to read a file under results/."""


@pytest.fixture(scope="module", autouse=True)
def no_results_reads():
    """Make every read of a results/ path raise, for the whole module."""
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
        mp.setattr(pathlib.Path, name, wrap(getattr(pathlib.Path, name),
                                            lambda a, k: a[0] if a else None))
    saved = list(sys.path)
    R._paths()
    yield
    sys.path[:] = saved
    mp.undo()


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return R.build_synthetic(tmp_path_factory.mktemp("e02d") / "inputs")


@pytest.fixture(scope="module")
def ctx(synth):
    return R.O.load_context(synth.v1)


def _quiet(*_):
    return None


# ---- guards and pins --------------------------------------------------------------------------
def test_the_read_guard_holds():
    for p in ("results/twin_forward/fv2_truth_R0_b0.npz",
              R.REPO / "results" / "dataprep" / "preregistration" / "INDEX.jsonl"):
        q = R.REPO / p if not Path(p).is_absolute() else p
        with pytest.raises(ResultsRead):
            O.rp(p)
        with pytest.raises(ResultsRead):
            open(q).close()  # noqa: SIM115
        with pytest.raises(ResultsRead):
            Path(q).read_text()
    with pytest.raises(ResultsRead):
        R.world_shas(R.DecidPaths())


def test_new_files_conventions():
    for p in (D / "run_e02d1.py", D / "prereg_e02d1.py", Path(__file__)):
        src = p.read_text(encoding="utf-8")
        assert "from __future__ import annotations" in src, p
        assert "/" + "home/" not in src, p
        assert not STATION.search(src), p


def test_reused_code_is_pinned_and_unchanged():
    assert R.reused_code_problems() == []
    assert set(R.FROZEN) == {R.SELF, *R.REUSED}
    for name in ("osse_fv.py", "osse_fv2.py", "score_e02c.py", "validation.py"):
        assert any(p.endswith(name) for p in R.REUSED), name
    # the FV v2 pins are the ones FW.rider.v1 pinned to FV v2's registered frozen_code
    rider = (REPO / "dataprep_staging" / "fw_rider" / "run_rider.py").read_text(encoding="utf-8")
    for p, h in R.REUSED.items():
        if R.REUSED_FROM[p] == R.FV2_ID:
            assert f'"{p}":\n        "{h}"' in rider, p
    stat = (REPO / "dataprep_staging" / "fv_stat_v2" / "stat_v2.py").read_text(encoding="utf-8")
    assert '"hydrophysics/twin/validation.py"' in stat and '"hydrophysics/twin/release.py"' in stat
    assert {p for p, d in R.REUSED_FROM.items() if d == R.FVS2_ID} == {
        "hydrophysics/twin/validation.py", "hydrophysics/twin/release.py"}
    assert R.K_TRUTH == 250 == R.V2.K_TRUTH and R.PLAN == R.V2.PLAN and R.TAGS == O.TAGS


def test_reused_code_problems_sees_a_changed_file(tmp_path):
    for p in R.REUSED:
        q = tmp_path / p
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes((REPO / p).read_bytes())
    assert R.reused_code_problems(tmp_path) == []
    first = next(iter(R.REUSED))
    (tmp_path / first).write_bytes((REPO / first).read_bytes() + b"\n# edited\n")
    assert R.reused_code_problems(tmp_path) == [
        f"{first} differs from {R.REUSED_FROM[first]}'s registered frozen_code"]


def test_seed_and_streams_are_fresh_and_fixed():
    assert R.SEED not in R.REFUSED_SEEDS and R.SEED != R.SMOKE_SEED
    assert R.SEED not in (O.SEED, 46190527, 31750093, 20261103, 52629095, 71120611)
    assert {R.STREAM_TRUTH, R.STREAM_NULL, R.STREAM_SELFCHECK}.isdisjoint(
        {1, 2, 3, 4, 11, 12, 13, 31, 32, 33, 41, 42, 43, 61, 62})
    a, b = (R.rng_for(R.STREAM_TRUTH, 2, 7, seed=R.SEED).standard_normal(4) for _ in range(2))
    assert (a == b).all()
    assert not (a == R.rng_for(R.STREAM_NULL, 2, 7, seed=R.SEED).standard_normal(4)).all()
    assert not (a == R.rng_for(R.STREAM_TRUTH, 2, 7, seed=R.SMOKE_SEED).standard_normal(4)).all()
    with pytest.raises(R.EngineeringFailure, match="non-negative int"):
        R.check_seed(True)


# ---- outcomes and wording -----------------------------------------------------------------------
@pytest.mark.parametrize("p, p0, hp, hp0, want", [
    (0.8, 0.2, 0.03, 0.02, "measured"),
    (0.99, 0.0, 0.01, 0.0, "measured"),
    (0.7999, 0.1, 0.03, 0.02, "unchanged"),
    (0.9, 0.2001, 0.03, 0.02, "unchanged"),
    (0.5, 0.5, 0.03, 0.02, "unchanged"),             # boundaries read on the no-change side
    (0.4999, 0.1, 0.03, 0.02, "weak"),
    (0.9, 0.5001, 0.03, 0.02, "weak"),
    (0.3, 0.7, 0.03, 0.02, "weak"),
    (0.930, 0.475, 0.012, 0.037, "unchanged"),       # FV.stat.v2's published P, P0 (R8.17b)
    (0.736, 0.45, 0.023, 0.018, "unchanged"),        # FV v2's report-only P with any P0 <= 0.5
    (0.9, 0.1, 0.0700001, 0.02, "not_evaluable"),
    (0.9, 0.1, 0.03, 0.0700001, "not_evaluable"),
    (float("nan"), 0.1, 0.03, 0.02, "not_evaluable"),
    (0.9, None, 0.03, 0.02, "not_evaluable"),
])
def test_the_outcomes_follow_the_declared_thresholds(p, p0, hp, hp0, want):
    assert R.outcome_of(p, p0, hp, hp0) == want


def test_thresholds_are_fixed_constants_and_the_grade_map_is_complete():
    assert (R.P_MEASURED, R.P0_MEASURED, R.P_WEAK, R.P0_WEAK) == (0.8, 0.2, 0.5, 0.5)
    assert R.MC_HALF_WIDTH == 0.07 and R.K_NULL == 2000
    assert set(R.LEDGER_GRADE) == {*R.OUTCOMES, R.NOT_EVALUABLE}
    assert R.LEDGER_GRADE == {"measured": "D-measured", "weak": "D-weak",
                              "unchanged": "D-unknown", "not_evaluable": "D-unknown"}
    d = R.design()["thresholds"]
    assert d == {"P_measured": 0.8, "P0_measured": 0.2, "P_weak": 0.5, "P0_weak": 0.5,
                 "mc_half_width_90": 0.07}


def test_every_outcome_has_prewritten_wording_en_zh():
    assert set(R.WORDING) == {*R.OUTCOMES, R.NOT_EVALUABLE}
    for out in R.OUTCOMES:
        w = R.wording_for(out, p=0.4321, p0=0.6789)
        assert set(w) == {"en", "zh"}
        for lang, t in w.items():
            assert "{" not in t and "0.43" in t and "0.68" in t, (out, lang)
            assert "E02.decidability.v1" in t and "FV.decidability.v2" in t
        assert "Model-only" in w["en"] and "僅模型" in w["zh"]
        assert "only the noise is fresh" in w["en"] and "0.736" in w["en"]
    weak = R.wording_for("weak", p=0.4, p0=0.6)
    assert R.WEAK_WHY["p"]["en"] in weak["en"] and R.WEAK_WHY["p0"]["en"] in weak["en"]
    assert R.WEAK_WHY["p"]["zh"] in weak["zh"] and R.WEAK_WHY["p0"]["zh"] in weak["zh"]
    only_p0 = R.wording_for("weak", p=0.9, p0=0.6)["en"]
    assert R.WEAK_WHY["p0"]["en"] in only_p0 and R.WEAK_WHY["p"]["en"] not in only_p0
    assert "can decide" in R.wording_for("measured", p=0.9, p0=0.1)["en"]
    assert "stays unknown" in R.wording_for("unchanged", p=0.7, p0=0.4)["en"]
    ne = R.wording_for(R.NOT_EVALUABLE, why="x")
    assert "not evaluable (x)" in ne["en"] and "無法評估（x）" in ne["zh"]
    for v in R.WORDING.values():
        for lang in ("en", "zh"):
            assert not re.search(r"\bsupport(ed|s)?\b|informative|calibrat|校準", v[lang]), v
    with pytest.raises(R.EngineeringFailure, match="no declared wording"):
        R.wording_for("decisive", p=0.5, p0=0.5)


# ---- worlds -----------------------------------------------------------------------------------
def test_the_null_is_the_midpoint_and_its_ratio_half_is_a_fair_coin(ctx):
    import numpy as np

    B = R.check_frame_order(ctx)
    obs = R.null_obs(ctx, B, 3, seed=R.SMOKE_SEED)
    e = O.simulate_ar1(ctx["sd"], ctx["phi"], O.N_WIN,
                       R.rng_for(R.STREAM_NULL, 0, 3, seed=R.SMOKE_SEED))
    mid = obs - e
    assert np.allclose(np.abs(mid - ctx["zr"]), np.abs(mid - B))       # equidistant
    assert np.allclose(mid, 0.5 * (ctx["zr"] + B))
    recs = [R.e02_rule(ctx, R.null_obs(ctx, B, k, seed=R.SMOKE_SEED)) for k in range(300)]
    share = float(np.mean([r["ratio_ok"] for r in recs]))
    assert 0.40 < share < 0.60, share                 # 300 draws: SE 0.029
    assert all(r["pass"] == (r["ratio_ok"] and r["in_bounds"]) for r in recs)


def test_e02_rule_is_fv2s_rate_and_evaluates_as_registered(ctx, synth, tmp_path):
    import numpy as np

    B = R.check_frame_order(ctx)
    for k in range(3):
        obs = R.null_obs(ctx, B, k, seed=R.SMOKE_SEED) if k == 2 else \
            ctx["zr"] + O.simulate_ar1(ctx["sd"], ctx["phi"], O.N_WIN,
                                       np.random.default_rng(k))
        mine = R.e02_rule(ctx, obs)
        assert mine["pass"] == bool(O.world_stats(ctx, obs)["e02_passed"])
        assert O.evaluate_selfcheck(ctx, obs, Path(synth.v1.protocol), tmp_path)["identical"]


def test_gate_e_checks_frame_order_and_protocol_rule(ctx):
    R.protocol_rule(ctx["protocol"])
    with pytest.raises(R.EngineeringFailure, match="another rule"):
        R.protocol_rule({**ctx["protocol"], "max_rmse_ratio": 1.05})
    with pytest.raises(R.EngineeringFailure, match="lacks"):
        R.protocol_rule({k: v for k, v in ctx["protocol"].items() if k != "min_months"})
    bad = dict(ctx, frame0=ctx["frame0"].iloc[::-1].reset_index(drop=True))
    with pytest.raises(R.EngineeringFailure, match="well x month order"):
        R.check_frame_order(bad)


def _scored(synth, k_null=200, monkeypatch=None):
    if monkeypatch is not None:                      # exercise the scored path at tiny K
        monkeypatch.setattr(R, "MC_HALF_WIDTH", 1.0)
    return R._score(synth, seed=R.SMOKE_SEED, plan=R.SYNTH_PLAN, k_null=k_null, smoke=True,
                    log=_quiet)


def test_synthetic_end_to_end(synth, monkeypatch):
    import numpy as np

    rep = _scored(synth, monkeypatch=monkeypatch)
    assert rep["status"] == "scored" and rep["outcome"] in R.OUTCOMES
    assert rep["model_only"] is True and rep["k_truth"] == R.SYNTH_PLAN.k_truth
    per = rep["truth_worlds"]
    assert set(per) == set(R.TAGS)
    want = float(np.mean([per[t]["passed"]["p"] for t in R.TAGS]))
    assert rep["P"]["p"] == pytest.approx(want)
    assert rep["P0"]["p"] == rep["null_world"]["passed"]["p"]
    assert rep["outcome"] == R.outcome_of(rep["P"]["p"], rep["P0"]["p"],
                                          rep["P"]["mc_half_width_90"],
                                          rep["P0"]["mc_half_width_90"])
    assert rep["ledger_mapping"]["grade_key"] == R.LEDGER_GRADE[rep["outcome"]]
    assert "{" not in rep["wording"]["en"] and f"{rep['P']['p']:.2f}" in rep["wording"]["en"]
    eng = rep["engineering"]
    assert [c["world"] for c in eng["selfcheck"]] == ["truth_R0", "truth_R0", "null"]
    assert all(c["evaluate_identical"] and c["fv2_world_stats_identical"]
               for c in eng["selfcheck"])
    assert set(eng["truths"]) == set(R.TAGS) and eng["repro"]["X5R1"]["pass"]
    assert all(b["hindcast_max_abs_vs_v1_m"] == 0.0 for t in R.TAGS for b in eng["truths"][t])
    assert set(rep["distance_in_mc_se"]) == {"P_minus_0.8", "P_minus_0.5", "P0_minus_0.2",
                                             "P0_minus_0.5"}
    again = _scored(synth, monkeypatch=monkeypatch)
    assert again["P"] == rep["P"] and again["P0"] == rep["P0"]           # deterministic


def test_smoke_writes_its_report_once_and_never_under_results(tmp_path):
    rep = R.run_smoke(tmp_path / "smoke", k_null=20)
    assert rep["smoke"] is True and (tmp_path / "smoke" / "smoke_report.json").is_file()
    assert rep["status"] == "not_evaluable" and rep["outcome"] == "not_evaluable"
    assert rep["ledger_mapping"]["grade_key"] == "D-unknown"
    with pytest.raises(FileExistsError):
        O._write_json_once(tmp_path / "smoke" / "smoke_report.json", rep)
    with pytest.raises(R.EngineeringFailure, match="results/"):
        R.run_smoke(REPO / "results" / "e02d_smoke_should_not_exist")
    assert not (REPO / "results" / "e02d_smoke_should_not_exist").exists()


def test_world_files_and_fv2_attempts_must_match(tmp_path):
    import numpy as np

    P = R.build_synthetic(tmp_path / "in")
    declared = R.world_shas(P)
    assert R.world_problems(P, declared) == []
    p = Path(P.fv2.batches["Q0"][1])
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    np.savez_compressed(p, **d, extra=np.zeros(1))              # same heads, new bytes
    assert R.world_problems(P, declared) == ["Q0_b1: sha256 differs from the declaration"]
    with pytest.raises(R.EngineeringFailure, match="last attempt"):
        R._score(P, seed=R.SMOKE_SEED, plan=R.SYNTH_PLAN, k_null=4, smoke=True, log=_quiet)
    assert "world files" not in str(R.world_problems(P, None))
    Path(P.fv2.v1.zr0).unlink()
    assert R.world_problems(P, None) == ["ZR0: missing"]


# ---- registration and the run ----------------------------------------------------------------
def test_scoring_is_private_and_needs_the_registration(synth, tmp_path):
    assert not hasattr(R, "score")
    for kw in ({}, {"k_null": 3}, {"seed": R.SEED + 1}, {"plan": R.SYNTH_PLAN}):
        with pytest.raises(R.EngineeringFailure, match="registered seed|INDEX.jsonl"):
            R._score(synth, **{"seed": R.SEED, "prereg_dir": tmp_path, **kw})
    with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
        R._score(synth, seed=R.SEED, smoke=True)
    with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):     # real paths, smoke flag
        R._score(R.DecidPaths(), seed=R.SMOKE_SEED, smoke=True)
    real = R.DecidPaths()
    for attr in ("fv2_decl", "fvs2_decl", "rulings", "dataprep", "report"):
        P = dataclasses.replace(synth, **{attr: getattr(real, attr)})
        with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
            R._score(P, seed=R.SMOKE_SEED, smoke=True)
    for attr in ("pred", "protocol", "baselines", "wells", "inputs_dir", "zr_forward", "zr0"):
        P = dataclasses.replace(synth, fv2=dataclasses.replace(synth.fv2, v1=dataclasses.replace(
            synth.v1, **{attr: getattr(real.v1, attr)})))
        with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
            R._score(P, seed=R.SMOKE_SEED, smoke=True)
    for attr in ("arms_dir", "attempts", "batches", "v1_truths"):
        P = dataclasses.replace(synth, fv2=dataclasses.replace(synth.fv2, **{
            attr: getattr(real.fv2, attr)}))
        with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
            R._score(P, seed=R.SMOKE_SEED, smoke=True)


def test_the_results_guard_sees_symlinked_and_nested_results(tmp_path):
    assert R._under_results("results/twin_forward/fv2_truth_R0_b0.npz")
    assert R._under_results(R.REPO / "results" / "x.json")
    assert R._under_results(tmp_path / "other" / "results" / "y.npz")
    assert not R._under_results(tmp_path / "synthetic" / "y.npz")
    real = tmp_path / "elsewhere"
    real.mkdir()
    link = tmp_path / "results_link"
    link.symlink_to(real)
    assert not R._under_results(link / "a.npz")
    tf = R.REPO / "results" / "twin_forward"
    if tf.is_symlink():
        assert R._under_results(tf / "fv2_truth_R0_b0.npz")


def _attempts(report: Path) -> list[dict]:
    return [json.loads(x) for x in (report.parent / R.ATTEMPTS_NAME).read_text().splitlines()]


def test_run_refuses_before_logging_or_reading(tmp_path, synth):
    reg = tmp_path / "prereg"
    reg.mkdir()
    P = dataclasses.replace(synth, report=str(tmp_path / "rep" / "r.json"))
    with pytest.raises(R.EngineeringFailure, match="not in INDEX.jsonl"):
        R.run(P, prereg_dir=reg)
    assert not (tmp_path / "rep").exists()
    with pytest.raises(R.EngineeringFailure, match="scores E02.decidability.v1 only"):
        R.load_registered("FV.stat.v2", prereg_dir=reg)


def _fake_registry(monkeypatch, doc, sha="f" * 64, index_sha=None):
    monkeypatch.setattr(R, "index_record", lambda *a, **k: {"id": R.ID,
                                                            "sha256": index_sha or sha})
    from harness import prereg3

    monkeypatch.setattr(prereg3, "load_prereg", lambda *a, **k: (doc, sha))
    monkeypatch.setattr(prereg3, "changed_inputs", lambda d: [])


def test_run_refuses_changed_code_another_design_or_index_sha(monkeypatch):
    fc = {p: O.sha256(O.rp(p)) for p in R.FROZEN}
    doc = {"frozen_code": {**fc, R.SELF: "0" * 64}, "design": R.design()}
    _fake_registry(monkeypatch, doc)
    with pytest.raises(R.EngineeringFailure, match="code sha256 differs"):
        R.load_registered()
    doc["frozen_code"] = {k: v for k, v in fc.items() if k != R.SELF}
    with pytest.raises(R.EngineeringFailure, match="exactly"):
        R.load_registered()
    doc["frozen_code"] = fc
    for k, v in (("seed", R.SEED + 1), ("K_NULL", 500),
                 ("thresholds", {**R.design()["thresholds"], "P_measured": 0.7})):
        doc["design"] = {**R.design(), k: v}
        with pytest.raises(R.EngineeringFailure, match=k):
            R.load_registered()
    doc["design"] = R.design()
    assert R.load_registered() == (doc, "f" * 64)
    _fake_registry(monkeypatch, doc, index_sha="e" * 64)
    with pytest.raises(R.EngineeringFailure, match="not INDEX.jsonl's"):
        R.load_registered()


def test_run_logs_start_and_success_with_the_report_sha(monkeypatch, tmp_path, synth):
    P = dataclasses.replace(synth, report=str(tmp_path / "rep" / "report.json"))
    monkeypatch.setattr(R, "load_registered", lambda *a, **k: ({"frozen_code": {}}, "f" * 64))
    monkeypatch.setattr(R, "_score", lambda *a, **k: {"status": "scored",
                                                      "declaration_sha256": "f" * 64})
    R.run(P)
    rows = _attempts(Path(P.report))
    assert [r["event"] for r in rows] == ["start", "success"]
    assert rows[1]["report_sha256"] == O.sha256(Path(P.report))
    with pytest.raises(R.EngineeringFailure, match="scored once"):
        R.run(P)
    assert [r["event"] for r in _attempts(Path(P.report))][-2:] == ["start",
                                                                   "engineering_failure"]


def _tmp_registry(tmp_path, P):
    """The draft written into a TEMPORARY registry and ledger (synthetic sources only)."""
    from harness import prereg3

    reg, rows = tmp_path / "prereg", tmp_path / "rows"
    reg.mkdir()
    rows.mkdir()
    d = PR.draft_prereg(prereg_dir=reg, P=P)
    prereg3.write_prereg(d, prereg_dir=reg, rows_dir=rows, units_registry=None, registries=[])
    return reg, rows


def test_a_temporary_registry_never_unlocks_the_real_worlds(tmp_path, synth):
    reg, rows = _tmp_registry(tmp_path, synth)
    doc, sha = R.load_registered(prereg_dir=reg, rows_dir=rows)   # the copy itself is intact
    assert {k: doc["design"][k] for k in R.design()} == R.design() and len(sha) == 64
    fake = tmp_path / "repo" / "results"
    P = dataclasses.replace(synth, report=str(fake / "rep" / "r.json"))
    for paths in (P, R.DecidPaths()):
        with pytest.raises(R.EngineeringFailure, match="real registry only"):
            R._score(paths, seed=R.SEED, prereg_dir=reg, rows_dir=rows)
        with pytest.raises(R.EngineeringFailure, match="real registry only"):
            R.run(paths, prereg_dir=reg, rows_dir=rows)
    assert not fake.exists()                                     # nothing written or logged
    R._registry_guard(synth, prereg_dir=reg, rows_dir=rows)      # synthetic: allowed
    from harness import prereg3
    R._registry_guard(R.DecidPaths(), prereg_dir=prereg3.PREREG_DIR,
                      rows_dir=prereg3.LEDGER_ROWS)              # the real registry: allowed


def test_a_registered_synthetic_run_scores_once_and_binds_the_declared_worlds(tmp_path,
                                                                             monkeypatch):
    """Registered on a temporary registry with SYNTHETIC worlds, the real branch of _score
    (the registered seed; load_registered; the declared world files) runs end to end. The plan,
    K and FV v1's export pins are shrunk or swapped to the synthetic ones by monkeypatching the
    module constants the design and gate E read."""
    import numpy as np

    P = R.build_synthetic(tmp_path / "in")
    monkeypatch.setattr(R, "PLAN", R.SYNTH_PLAN)
    monkeypatch.setattr(R, "K_TRUTH", R.SYNTH_PLAN.k_truth)
    monkeypatch.setattr(R, "K_NULL", 30)
    monkeypatch.setattr(R, "FV1_INPUTS", {n: O.sha256(Path(P.v1.inputs_dir) / n)
                                          for n in R.FV1_INPUTS})
    P = dataclasses.replace(P, report=str(tmp_path / "rep" / "report.json"))
    reg, rows = _tmp_registry(tmp_path, P)
    rep = R.run(P, prereg_dir=reg, rows_dir=rows, log=_quiet)
    assert len(rep["declaration_sha256"]) == 64 and rep["seed"] == R.SEED
    assert rep["status"] == "not_evaluable" and rep["k_null"] == 30     # tiny K: MC check
    assert [r["event"] for r in _attempts(Path(P.report))] == ["start", "success"]
    with pytest.raises(R.EngineeringFailure, match="scored once"):
        R.run(P, prereg_dir=reg, rows_dir=rows, log=_quiet)
    p = Path(P.fv2.batches["R1c"][0])           # a world file changed after registration
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    np.savez_compressed(p, **d, extra=np.zeros(1))
    Path(P.report).unlink()
    with pytest.raises(R.EngineeringFailure):
        R.run(P, prereg_dir=reg, rows_dir=rows, log=_quiet)
    assert not Path(P.report).exists()


# ---- model-only -------------------------------------------------------------------------------
def test_the_reading_is_model_only(tmp_path, monkeypatch, synth):
    """The declaration says so, its inputs hold no observation product, the code names none,
    and a synthetic score opens only its synthetic inputs."""
    d = PR.draft_prereg(prereg_dir=tmp_path, P=synth)
    assert "no observed head" in d["model_only"]["observations_read"]
    assert "never reads an observed head" in d["model_only"]["statement"]
    obs = re.compile(r"heads/L1|AMP_V2|wells\.parquet|ls_cache|observations?\.csv|"
                     r"evaluation\.csv")
    assert not [p for p in d["inputs"] if obs.search(p) and not p.endswith("/MANIFEST.json")]
    code = re.sub(r'"""(?:.|\n)*?"""', "", (D / "run_e02d1.py").read_text(encoding="utf-8"))
    code = code.split("SYNTH_PRODUCTS = ")[0] + code.split("def build_synthetic")[1]
    assert not re.search(r"L1_clean|AMP_V2|read_parquet|open_product|heads/L1", code)
    P = R.build_synthetic(tmp_path / "in")
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
    rep = R._score(P, seed=R.SMOKE_SEED, plan=R.SYNTH_PLAN, k_null=4, smoke=True, log=_quiet)
    monkeypatch.undo()
    assert rep["status"] in ("scored", "not_evaluable")
    data = [f for f in opened if not f.endswith((".py", ".pyc")) and "site-packages" not in f]
    root = str((tmp_path / "in").resolve())
    assert data and all(str(Path(f).resolve()).startswith(root) for f in data), data
    names = {Path(f).name for f in data}
    allowed = {"predictions.csv", "baselines_e02c.csv", "wells_158.csv", "noise_ar1.csv",
               "obs_mask.csv", "anomaly_pool.csv", "protocol.json", "arms.json",
               "attempts.jsonl", "zr.npz", "zr0.npz", "evaluation.csv"}
    extra = {n for n in names if n not in allowed
             and not re.fullmatch(r"(fv2_truth_\w+_b\d+|truth_\w+|batch27_\w+)\.npz|w\d{3}\.csv",
                                  n)}
    assert not extra, extra


# ---- the draft declaration --------------------------------------------------------------------
def test_draft_declaration_key_fields(tmp_path, synth):
    d = PR.draft_prereg(prereg_dir=tmp_path, P=synth)
    from harness import prereg, prereg3

    assert set(prereg.REQUIRED) <= set(d)
    assert prereg3.validate_prereg(d, units_registry=None, prereg_dir=tmp_path,
                                   registries=[]) == []
    assert d["id"] == "E02.decidability.v1" and d["kind"] == "declaration" and "created" not in d
    assert d["wp"] == "FV" and "R9.4" in d["label"]
    assert d["design"]["seed"] == R.SEED and d["design"]["K_TRUTH"] == 250
    assert d["design"]["K_NULL"] == 2000 and d["design"]["truth_order"] == list(R.TAGS)
    assert d["thresholds"]["P_measured"] == 0.8 and d["thresholds"]["P0_measured"] == 0.2
    assert d["thresholds"]["P_weak"] == 0.5 and d["thresholds"]["P0_weak"] == 0.5
    assert d["attempt_policy"]["exploratory_followups_max"] == 0
    assert "scored once" in d["attempt_policy"]["scored_runs"]
    assert d["attempt_policy"]["thresholds_frozen"] is True
    assert set(d["frozen_code"]) == set(R.FROZEN)
    assert d["reused_code"]["pins"] == R.REUSED and d["reused_code"]["current_files_match"]
    assert d["reused_code"]["pins_match_declarations"] is True
    assert d["rule"]["stated"] == d["rule"]["declared"] == R.E02_RULE
    assert set(d["readings"]["wording"]) == {*R.OUTCOMES, "not_evaluable"}
    assert d["readings"]["outcomes"]["measured"] == "P >= 0.8 and P0 <= 0.2"
    just = " ".join(d["readings"]["justification"])
    for k in ("Section 21", "R8.17b", "0.475", "FE.encompass", "FV.decidability.v1/v2",
              "FV.stat.v2", "unchanged"):
        assert k in just, k
    assert "(ZR(w, t) + B(w, t)) / 2" in d["null"]["formula"] and "FV.stat.v2" in d["null"]["as_fv_stat_v2"]
    assert "NEGATIVE skill" in d["null"]["not_fv_v1_null"]
    assert "Expected outcome: 'unchanged'" in d["priors"] and "0.736" in d["priors"]
    assert "No observation after 2022-12" in d["u7"] and "U7" in d["u7"]
    assert d["worlds"]["attempts_match"] is True
    assert set(d["worlds"]["files"]) == set(R.world_files(synth))
    uses = {(m["product"], m["use"]) for m in d["manifest_checks"] if m.get("manifest")}
    assert uses == {("results/dataprep/heads/L1_clean/v3", "design_time_evidence"),
                    ("results/dataprep/heads/L1_clean/v3", "baseline"),
                    ("results/dataprep/pump_electricity/preorigin/v3", "design_time_evidence")}
    assert all(m["allowed"] for m in d["manifest_checks"])
    assert "printed the header and first two rows" in d["disclosures"]["builder"]
    assert "did not open FV v2's report_worlds.json" in d["disclosures"]["builder"]
    lm = d["ledger_mapping"]
    assert lm["grade_keys"] == R.LEDGER_GRADE and "no status change from this reading" in \
        lm["never"]
    assert not STATION.search(json.dumps(d, ensure_ascii=False))
    assert "/" + "home/" not in json.dumps(d)
    msgs = PR.readiness(d, P=synth, prereg_dir=tmp_path)
    assert any(f"ruling {R.RULING_ID}" in m for m in msgs)            # not the real rulings sha
    assert any("protocol.json" in m for m in msgs)                    # not E02's real protocol
    assert any("is not in the registry" in m for m in msgs)           # temporary registry
    assert not any("MANIFEST" in m or "attempts" in m or "REUSED" in m or ": missing" in m
                   or m.startswith(("input ", "frozen code ")) for m in msgs), msgs
    late = PR.readiness(d, P=synth, prereg_dir=tmp_path, now=PR.TOO_LATE)
    assert any("too late" in m for m in late)


def test_not_registered_by_r9_4b_and_e02_stays_d_unknown(tmp_path, synth):
    """User ruling R9.4b: the reading is kept as unregistered code. The docstrings and the
    declaration say so, readiness never reads 'ready', and E02's committed progress entry stays
    D-unknown (the hook spec is declared here only)."""
    for p in (D / "run_e02d1.py", D / "prereg_e02d1.py"):
        assert "not registered by user ruling R9.4b" in p.read_text(encoding="utf-8"), p
    assert "bfeed6d6" in R.NOT_REGISTERED
    d = PR.draft_prereg(prereg_dir=tmp_path, P=synth)
    assert d["status_note"].startswith("Draft only, not registered by user ruling R9.4b")
    assert PR.readiness(d, P=synth, prereg_dir=tmp_path)[0].startswith(
        "E02.decidability.v1 is not registered by user ruling R9.4b")
    pmap = json.loads((REPO / "hydrophysics" / "twin" / "progress_map_v1.json").read_text())
    assert pmap["claims"]["E02"]["grade"] == {"D": {"D": "D-unknown"}}
    assert "R9.4b" in pmap["claims"]["E02"]["reason"]
    assert PR.PROGRESS_SPEC["reading_report"] == R.REPORT
    assert PR.PROGRESS_SPEC["reading_outcomes"] == R.LEDGER_GRADE
