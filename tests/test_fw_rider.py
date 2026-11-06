"""FW.rider.v1 (round-8 rank 13, rain-year rider): synthetic worlds only.

- the reused FV v2 code is pinned to FV v2's registered frozen_code and unchanged;
- the one-sided reading and its pre-written wording for every band, on fabricated numbers;
- a synthetic end-to-end score (reading 2's world construction, E02c's level rule);
- a problem in the content of FW's forwards maps to 'replay worlds unusable';
- the scorer refuses without the registration; the runner refuses an unregistered id, changed
  code, a different design and an existing report, and logs every attempt;
- the rider is model-only: it opens no observation file;
- the draft declaration's key fields (and, where the gitignored registry exists, validation).

No test reads an RD world. Checks that need the gitignored results/ tree skip in CI.
"""
from __future__ import annotations

import builtins
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
D = REPO / "dataprep_staging" / "fw_rider"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_saved = list(sys.path)
R = _load("run_rider", D / "run_rider.py")
PR = _load("prereg_rider", D / "prereg_rider.py")
sys.path[:] = _saved
O = R.O  # noqa: E741  (osse_fv, frozen)

FV2_DECL = REPO / "results" / "dataprep" / "preregistration" / f"{R.FV2_ID}.json"
HAVE_RESULTS = (REPO / "results" / "dataprep" / "preregistration" / "INDEX.jsonl").is_file()
NEED_RESULTS = pytest.mark.skipif(not HAVE_RESULTS, reason="results/ (gitignored) is absent")
STATION = re.compile(r"\b(?:0[79]|10)[0-9]{6}\b")


@pytest.fixture(autouse=True)
def _staging_on_path():
    saved = list(sys.path)
    R._paths()
    yield
    sys.path[:] = saved


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return R.build_synthetic(tmp_path_factory.mktemp("rider") / "inputs")


def test_new_files_conventions():
    for p in (D / "run_rider.py", D / "prereg_rider.py", Path(__file__)):
        src = p.read_text(encoding="utf-8")
        assert "from __future__ import annotations" in src, p
        assert max(len(ln) for ln in src.splitlines()) <= 100, p
        assert "/" + "home/" not in src, p


def test_reused_code_is_pinned_and_unchanged():
    assert R.reused_code_problems() == []
    assert set(R.FROZEN) == {R.SELF, *R.REUSED}
    for name in ("forcing_year.py", "osse_fv.py", "osse_fv2.py", "score_e02c.py"):
        assert any(p.endswith(name) for p in R.REUSED), name
    if FV2_DECL.is_file():                 # the pins are FV v2's registered frozen_code
        fc = json.loads(FV2_DECL.read_text(encoding="utf-8"))["frozen_code"]
        for p, h in R.REUSED.items():
            assert fc[p] == h, p
    # reading 2's constants are reused, not copied
    assert R.K_WORLD == R.FY.K_RC and R.N_SEQ == R.FY.N_SEQ and R.K_NULL == O.K_NULL


def test_reused_code_problems_sees_a_changed_file(tmp_path):
    for p in R.REUSED:
        q = tmp_path / p
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes((REPO / p).read_bytes())
    assert R.reused_code_problems(tmp_path) == []
    first = next(iter(R.REUSED))
    (tmp_path / first).write_bytes((REPO / first).read_bytes() + b"\n# edited\n")
    assert R.reused_code_problems(tmp_path) == [f"{first} differs from FV v2's registered "
                                                "frozen_code"]


@pytest.mark.parametrize("p, half, want", [
    (0.5, 0.03, "weak_in_replayed_years"),
    (0.2, 0.03, "weak_in_replayed_years"),
    (0.5000001, 0.03, "not_shown_weak_in_replayed_years"),
    (0.9, 0.07, "not_shown_weak_in_replayed_years"),
    (0.9, 0.0700001, None),
    (float("nan"), 0.03, None),
    (None, 0.03, None),
])
def test_the_reading_is_one_sided(p, half, want):
    assert R.reading_of(p, half) == want


def test_every_band_has_prewritten_wording_en_zh():
    assert set(R.WORDING) == {*R.READINGS, "not_evaluable"}
    for rd in R.READINGS:
        w = R.wording_for(rd, p=0.42, t_lo=0.3, t_hi=0.55, lo=0.1, hi=0.8)
        assert set(w) == {"en", "zh"}
        for lang, t in w.items():
            assert "{" not in t and "0.42" in t and "0.30-0.55" in t, (rd, lang)
            assert "FW.rider.v1" in t and "FW.weather_replay.v2" in t
        assert "One-sided" in w["en"] and "單向" in w["zh"]
    weak = R.wording_for("weak_in_replayed_years", p=0.4, t_lo=0, t_hi=1, lo=0, hi=1)["en"]
    assert "would not by itself refute" in weak and "every status are unchanged" in weak
    nsw = R.wording_for("not_shown_weak_in_replayed_years", p=0.8, t_lo=0, t_hi=1, lo=0,
                        hi=1)["en"]
    assert "not the chance of a 2028 pass" in nsw and "adds no other reading" in nsw
    ne = R.wording_for(None, why="x")
    assert "not evaluable (x)" in ne["en"] and "無法評估（x）" in ne["zh"]
    for lang in ("en", "zh"):                # a caveat only: never support, never informative
        for v in R.WORDING.values():
            assert not re.search(r"\bsupport(ed|s)?\b|informative|calibrat", v[lang]), v
    with pytest.raises(R.EngineeringFailure, match="no declared wording"):
        R.wording_for("decisive", p=0.5, t_lo=0, t_hi=1, lo=0, hi=1)


def test_summarise_reuses_reading_2s_summary():
    recs = [[{"level": "level_forecast_supported" if (s + j) % 3 else "x", "beat_cl": True,
              "beat_clt": False} for j in range(6)] for s in range(R.N_SEQ)]
    sm = R.summarise(recs)
    lev = [[r["level"] == "level_forecast_supported" for r in row] for row in recs]
    assert sm["p"] == pytest.approx(sum(map(sum, lev)) / (6 * R.N_SEQ))
    assert len(sm["by_sequence"]) == R.N_SEQ and sm["k_per_sequence"] == 6
    assert sm["report_only"]["beat_cl"] == 1.0 and sm["report_only"]["beat_clt"] == 0.0
    assert set(sm["sampling_11yr"]) >= {"t90", "bootstrap90_over_sequences"}


def test_synthetic_end_to_end(synth, tmp_path):
    rep = R.score(synth, seed=R.SMOKE_SEED, k=24, k_null=20, smoke=True, log=lambda *_: None)
    assert rep["status"] == "scored" and rep["reading"] in R.READINGS
    assert 0.0 <= rep["P"]["p"] <= 1.0 and rep["P"]["k_per_sequence"] == 24
    assert rep["null_world"]["pass"] and rep["mc_gate"]["pass"]
    assert rep["engineering"]["rd_hindcast_max_abs_m"] == 0.0
    assert rep["engineering"]["members_rd"] == list(R.RD_MEMBERS)
    assert rep["model_only"] is True and "{" not in rep["wording"]["en"]
    assert rep["decomposition_report_only"]["sequence_part_rms_m"] > 0
    assert f"{rep['P']['p']:.2f}" in rep["wording"]["en"]
    # the index is reading 2's own (forcing_year.build_index), metadata only
    ix = json.loads(Path(synth.index).read_text())
    assert ix["schema"] == R.FY.INDEX_SCHEMA and ix["rain"] == "replayed"
    assert ix["source_declaration"] == R.FW_ID
    # the same seed gives the same noise draws; another seed or stream does not
    a, b = (R.rng_for(R.STREAM_WORLD, 3, 7, seed=R.SEED).standard_normal(5) for _ in range(2))
    assert (a == b).all()
    assert not (a == R.rng_for(R.STREAM_WORLD, 3, 7, seed=R.SMOKE_SEED).standard_normal(5)).all()
    assert not (a == R.rng_for(R.STREAM_NULL, 3, 7, seed=R.SEED).standard_normal(5)).all()


def test_smoke_writes_its_report_once(tmp_path):
    rep = R.run_smoke(tmp_path / "smoke", k=2, k_null=4)
    assert rep["smoke"] is True and (tmp_path / "smoke" / "smoke_report.json").is_file()
    # K = 2 per sequence cannot meet the Monte Carlo check: not evaluable, P kept
    assert rep["status"] == "not_evaluable" and "Monte Carlo" in rep["why"]
    assert rep["reading"] is None and rep["P"]["k_per_sequence"] == 2
    with pytest.raises(FileExistsError):
        O._write_json_once(tmp_path / "smoke" / "smoke_report.json", rep)
    with pytest.raises(R.EngineeringFailure, match="results/"):
        R.run_smoke(REPO / "results" / "fw_rider_smoke_should_not_exist")
    assert not (REPO / "results" / "fw_rider_smoke_should_not_exist").exists()


def _rewrite(npz: Path, **over):
    import numpy as np

    with np.load(npz, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d.update(over)
    np.savez_compressed(npz, **d)


def test_content_problems_map_to_replay_worlds_unusable(tmp_path):
    import numpy as np

    P = R.build_synthetic(tmp_path / "in")
    _rewrite(Path(P.seqs()[3]), member_labels=np.array(["fold0"]))
    rep = R.score(P, seed=R.SMOKE_SEED, k=2, k_null=2, smoke=True, log=lambda *_: None)
    assert rep["status"] == "not_evaluable" and rep["why"].startswith(R.UNUSABLE)
    assert "members" in rep["why"] and "P" not in rep
    P2 = R.build_synthetic(tmp_path / "in2")
    p = Path(P2.seqs()[5])
    with np.load(p, allow_pickle=False) as z:
        h = z["heads_mean"].copy()
    h[..., 10] += 0.01                          # a hindcast month moved
    _rewrite(p, heads_mean=h)
    rep = R.score(P2, seed=R.SMOKE_SEED, k=2, k_null=2, smoke=True, log=lambda *_: None)
    assert rep["status"] == "not_evaluable" and "hindcasts differ" in rep["why"]


def _never_run(monkeypatch):
    """Any test that hands the scorer the REAL paths must not be able to read a world, even if
    a guard were broken: loading the context or the worlds raises."""
    def boom(*a, **k):
        raise AssertionError("a real input was about to be read")

    for mod, name in ((R.O, "load_context"), (R.FY, "build_index"), (R, "load_worlds"),
                      (R, "ensure_index")):
        monkeypatch.setattr(mod, name, boom)


def test_score_refuses_without_the_registration(synth, monkeypatch):
    for kw in ({}, {"decl": {}}, {"decl": {}, "decl_sha": "f" * 10},
               {"decl": {}, "decl_sha": "f" * 64, "seed": R.SEED + 1}):
        args = {"seed": R.SEED, **kw}
        with pytest.raises(R.EngineeringFailure, match="only under its registration"):
            R.score(synth, **args)
    with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
        R.score(synth, seed=R.SEED, smoke=True)
    _never_run(monkeypatch)
    with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):     # real worlds, smoke flag
        R.score(R.RiderPaths(), seed=R.SMOKE_SEED, smoke=True)
    import dataclasses

    real = R.RiderPaths()
    for attr in ("outd", "index", "report", "fw_report"):     # any one real path refuses
        P = dataclasses.replace(synth, **{attr: getattr(real, attr)})
        with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
            R.score(P, seed=R.SMOKE_SEED, smoke=True)
    for attr in ("pred", "protocol", "baselines", "wells", "inputs_dir"):
        P = dataclasses.replace(synth, v1=dataclasses.replace(synth.v1, **{
            attr: getattr(real.v1, attr)}))
        with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
            R.score(P, seed=R.SMOKE_SEED, smoke=True)


def test_the_results_guard_sees_symlinked_and_nested_results(tmp_path):
    assert R._under_results("results/twin_forward/fw_replay_v2")
    assert R._under_results(R.REPO / "results" / "x.json")
    assert R._under_results(tmp_path / "other" / "results" / "y.npz")
    assert not R._under_results(tmp_path / "synthetic" / "y.npz")
    real = tmp_path / "elsewhere"
    real.mkdir()
    link = tmp_path / "results_link"
    link.symlink_to(real)
    assert not R._under_results(link / "a.npz")      # no 'results' part once resolved
    tf = R.REPO / "results" / "twin_forward"
    if tf.is_symlink():                               # a build worktree: the symlinked tree
        assert R._under_results(tf / "fw_replay_v2" / "RD2012.npz")
    with pytest.raises(R.EngineeringFailure, match="smoke seed"):
        R.check_seed(R.SMOKE_SEED)


def _attempts(report: Path) -> list[dict]:
    return [json.loads(x) for x in (report.parent / R.ATTEMPTS_NAME).read_text().splitlines()]


def test_run_refuses_unregistered_and_logs_every_attempt(tmp_path, synth):
    reg = tmp_path / "prereg"
    reg.mkdir()
    P = R.RiderPaths(v1=synth.v1, outd=synth.outd, fw_report=synth.fw_report,
                     index=str(tmp_path / "ix.json"), report=str(tmp_path / "rep" / "r.json"))
    with pytest.raises(R.EngineeringFailure, match="not in INDEX.jsonl"):
        R.run(P, prereg_dir=reg)
    Path(P.report).write_text("{}")
    with pytest.raises(R.EngineeringFailure, match="scored once"):
        R.run(P, prereg_dir=reg)
    rows = _attempts(Path(P.report))
    assert [(r["event"], r["attempt"]) for r in rows] == [
        ("start", 1), ("engineering_failure", 1), ("start", 2), ("engineering_failure", 2)]
    assert all(r["id"] == R.ID and r["time"] for r in rows)
    with pytest.raises(R.EngineeringFailure, match="scores FW.rider.v1 only"):
        R.load_registered("FV.stat.v2", prereg_dir=reg)
    assert not Path(P.index).exists()             # refused before any index or world


def _fake_registry(monkeypatch, doc):
    monkeypatch.setattr(R, "index_record", lambda *a, **k: {"id": R.ID})
    from harness import prereg3

    monkeypatch.setattr(prereg3, "load_prereg", lambda *a, **k: (doc, "f" * 64))
    monkeypatch.setattr(prereg3, "changed_inputs", lambda d: [])


def _design() -> dict:
    return {"seed": R.SEED, "K_WORLD": R.K_WORLD, "K_NULL": R.K_NULL, "P_weak": R.P_WEAK,
            "streams": {"world": R.STREAM_WORLD, "null": R.STREAM_NULL},
            "hindcast_tol_m": R.HINDCAST_TOL_M}


def test_run_refuses_changed_code_and_a_different_design(monkeypatch):
    fc = {p: O.sha256(O.rp(p)) for p in R.FROZEN}
    doc = {"frozen_code": {**fc, R.SELF: "0" * 64}, "design": _design()}
    _fake_registry(monkeypatch, doc)
    with pytest.raises(R.EngineeringFailure, match="code sha256 differs"):
        R.load_registered()
    doc["frozen_code"] = {k: v for k, v in fc.items() if k != R.SELF}
    with pytest.raises(R.EngineeringFailure, match="exactly"):
        R.load_registered()
    doc["frozen_code"] = fc
    doc["design"] = {**_design(), "seed": R.SEED + 1}
    with pytest.raises(R.EngineeringFailure, match="seed"):
        R.load_registered()
    doc["design"] = _design()
    assert R.load_registered() == (doc, "f" * 64)


def test_run_logs_success_with_the_report_sha(monkeypatch, tmp_path):
    P = R.RiderPaths(report=str(tmp_path / "rep" / "report.json"))
    monkeypatch.setattr(R, "load_registered", lambda *a, **k: ({"frozen_code": {}}, "f" * 64))
    monkeypatch.setattr(R, "score", lambda *a, **k: {"status": "scored"})   # no world drawn
    R.run(P)
    rows = _attempts(Path(P.report))
    assert [r["event"] for r in rows] == ["start", "success"]
    assert rows[1]["report_sha256"] == O.sha256(Path(P.report))
    with pytest.raises(R.EngineeringFailure, match="scored once"):
        R.run(P)


def test_the_rider_is_model_only(tmp_path, monkeypatch):
    """The declaration says so, its inputs hold no observation product, the code names none,
    and a synthetic score opens only its synthetic inputs (worlds, frozen forecast, pre-origin
    baselines and noise export)."""
    d = PR.draft_prereg(prereg_dir=tmp_path)
    assert d["model_only"]["observations_read"] == "none"
    assert "never reads an observed head" in d["model_only"]["statement"]
    obs = re.compile(r"heads/L1|AMP_V2|wells\.parquet|ls_cache|observations?\.csv|"
                     r"evaluation\.csv")
    # a product's MANIFEST is metadata (allowed_uses), read for the use check only
    assert not [p for p in d["inputs"] if obs.search(p) and not p.endswith("/MANIFEST.json")]
    for src in (D / "run_rider.py",):
        code = re.sub(r'"""(?:.|\n)*?"""', "", src.read_text(encoding="utf-8"))
        assert not re.search(r"L1_clean|AMP_V2|read_parquet|open_product|heads/L1", code)
    P = R.build_synthetic(tmp_path / "in")
    opened: list[str] = []
    real_open = builtins.open
    np_load = R.np.load

    def rec_open(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)

    def rec_load(f, *a, **k):
        opened.append(str(f))
        return np_load(f, *a, **k)

    monkeypatch.setattr(builtins, "open", rec_open)
    monkeypatch.setattr(R.np, "load", rec_load)
    rep = R.score(P, seed=R.SMOKE_SEED, k=2, k_null=2, smoke=True, log=lambda *_: None)
    monkeypatch.undo()
    assert rep["status"] in ("scored", "not_evaluable")
    data = [f for f in opened if not f.endswith((".py", ".pyc")) and "site-packages" not in f]
    root = str((tmp_path / "in").resolve())
    assert data and all(str(Path(f).resolve()).startswith(root) for f in data), data
    names = {Path(f).name for f in data}
    assert names <= {"predictions.csv", "baselines_e02c.csv", "wells_158.csv", "noise_ar1.csv",
                     "obs_mask.csv", "anomaly_pool.csv", "protocol.json", "replay_index.json",
                     *[f"{a}.npz" for a in (*R.RD_ARMS, R.CLIM_ARM)],
                     *[f"{a}.fw.npz" for a in (*R.RD_ARMS, R.CLIM_ARM)]}, names


def test_draft_declaration_key_fields(tmp_path):
    d = PR.draft_prereg(prereg_dir=tmp_path)
    from harness import prereg

    assert set(prereg.REQUIRED) <= set(d)
    assert d["id"] == "FW.rider.v1" and d["kind"] == "declaration" and "created" not in d
    assert d["design"]["seed"] == R.SEED and d["design"]["K_WORLD"] == 250
    assert d["attempt_policy"]["exploratory_followups_max"] == 0
    assert "scored once" in d["attempt_policy"]["scored_runs"]
    assert set(d["frozen_code"]) == set(R.FROZEN)
    assert d["reused_code"]["pins"] == R.REUSED and d["reused_code"]["current_files_match"]
    assert set(d["readings"]["wording"]) == {*R.READINGS, "not_evaluable"}
    assert set(d["readings"]["bands"].values()) == {*R.READINGS, "not_evaluable"}
    cav = d["caveat_rule"]
    assert "never on the page" in cav["role"] and "shown nowhere before report.json" in cav["role"]
    assert any("no status change" in x for x in cav["never"])
    assert "No observation after 2022-12" in d["u7"]
    assert set(d["worlds"]["files"]) == {R.CLIM_ARM, *R.RD_ARMS}
    uses = {(m["product"], m["use"]) for m in d["manifest_checks"] if m.get("manifest")}
    assert ("results/dataprep/heads/L1_clean/v3", "baseline") in uses
    assert ("results/dataprep/twin_rain/preorigin/v2", "design_time_evidence") in uses
    assert all(m["allowed"] for m in d["manifest_checks"])
    assert d["gates"]["N"]["rule"] if d["gates"]["N"].get("status") == "ok" else True
    assert not STATION.search(json.dumps(d, ensure_ascii=False))
    assert "/" + "home/" not in json.dumps(d)


@NEED_RESULTS
def test_draft_validates_against_the_local_registry():
    from harness import prereg3

    d = PR.draft_prereg()
    assert prereg3.validate_prereg(d) == []
    assert d["gates"]["N"]["identical_to_fv_v1"] is True
    assert all(m["allowed"] is True for m in d["manifest_checks"] if m.get("manifest"))
    assert d["worlds"]["registration"] != PR.PENDING
    assert d["reused_code"]["registration"]["sha256"].startswith(R.FV2_DECL_SHA)
    probs = [m for m in PR.readiness(d) if "committed" not in m and "already" not in m]
    assert probs == [], probs
    assert not O.rp(R.REPORT).exists()            # never computed on the real worlds here
    assert "def score" in inspect.getsource(R)
