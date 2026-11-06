"""FW.rider.v1 (round-8 rank 13, rain-year rider): synthetic worlds only.

Nothing under results/ can be read here: a module fixture makes osse_fv.rp, osse_fv.sha256,
numpy.load, open/io.open, Path.read_text/read_bytes and pandas.read_csv raise on any path that
is, or resolves into, the results tree (run_rider._under_results). Every input, the draft
declaration included, comes from run_rider.build_synthetic.

- the reused FV v2 code is pinned and unchanged; reading 2's own loader builds the worlds;
- the one-sided reading and its pre-written wording for every band, on fabricated numbers;
- a synthetic end-to-end score (reading 2's F_s = h_W - h_C, E02c's level rule);
- a problem in the content of FW's forwards maps to 'replay worlds unusable';
- world files must match FW v2's report.json 'files';
- scoring is private: a real run loads the registration itself; the runner refuses before it
  logs or reads anything, and refuses changed code, another design and an existing report;
- the rider is model-only;
- the draft declaration's key fields and validation (synthetic sources).
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
    return R.build_synthetic(tmp_path_factory.mktemp("rider") / "inputs")


def test_the_read_guard_holds():
    for p in ("results/twin_forward/fw_replay_v2/W2012.npz",
              R.REPO / "results" / "dataprep" / "preregistration" / "INDEX.jsonl"):
        with pytest.raises(ResultsRead):
            O.rp(p)
        with pytest.raises(ResultsRead):
            open(R.REPO / p if not Path(p).is_absolute() else p).close()  # noqa: SIM115
        with pytest.raises(ResultsRead):
            Path(R.REPO / p if not Path(p).is_absolute() else p).read_text()
    with pytest.raises(ResultsRead):
        R.RiderPaths().clim() and R.world_shas(R.RiderPaths())


def test_new_files_conventions():
    for p in (D / "run_rider.py", D / "prereg_rider.py", Path(__file__)):
        src = p.read_text(encoding="utf-8")
        assert "from __future__ import annotations" in src, p
        assert "/" + "home/" not in src, p


def test_reused_code_is_pinned_and_unchanged():
    assert R.reused_code_problems() == []
    assert set(R.FROZEN) == {R.SELF, *R.REUSED}
    for name in ("forcing_year.py", "osse_fv.py", "osse_fv2.py", "score_e02c.py"):
        assert any(p.endswith(name) for p in R.REUSED), name
    assert R.K_WORLD == R.FY.K_RC and R.N_SEQ == R.FY.N_SEQ and R.K_NULL == O.K_NULL
    assert R.HINDCAST_TOL_M == 0.0 == R.V2.HINDCAST_TOL_M
    assert tuple(f"W{y}" for y in range(2012, 2023)) == R.W_ARMS and R.CLIM_ARM == "C"
    # the worlds are reading 2's own: load_replay, unchanged
    import inspect
    src = inspect.getsource(R.load_worlds)
    assert "FY.load_replay(" in src and "O.departure(" not in inspect.getsource(R)


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
        assert "One-sided" in w["en"] and "單向" in w["zh"] and "page ensemble" in w["en"]
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


def test_synthetic_end_to_end(synth):
    rep = R._score(synth, seed=R.SMOKE_SEED, k=24, k_null=20, smoke=True, log=lambda *_: None)
    assert rep["status"] == "scored" and rep["reading"] in R.READINGS
    assert 0.0 <= rep["P"]["p"] <= 1.0 and rep["P"]["k_per_sequence"] == 24
    assert rep["null_world"]["pass"] and rep["mc_gate"]["pass"]
    eng = rep["engineering"]
    assert eng["hindcast_tol_m"] == 0.0 and eng["rain"] == "replayed"
    assert [r["hindcast_max_abs_m"] for r in eng["replay"]] == [0.0] * R.N_SEQ
    assert rep["model_only"] is True and "{" not in rep["wording"]["en"]
    assert f"{rep['P']['p']:.2f}" in rep["wording"]["en"]
    ix = json.loads(Path(synth.index).read_text())
    assert ix["schema"] == R.FY.INDEX_SCHEMA and ix["rain"] == "replayed"
    assert ix["source_declaration"] == R.FW_ID
    a, b = (R.rng_for(R.STREAM_WORLD, 3, 7, seed=R.SEED).standard_normal(5) for _ in range(2))
    assert (a == b).all()
    assert not (a == R.rng_for(R.STREAM_WORLD, 3, 7, seed=R.SMOKE_SEED).standard_normal(5)).all()
    assert not (a == R.rng_for(R.STREAM_NULL, 3, 7, seed=R.SEED).standard_normal(5)).all()


def test_smoke_writes_its_report_once(tmp_path):
    rep = R.run_smoke(tmp_path / "smoke", k=2, k_null=4)
    assert rep["smoke"] is True and (tmp_path / "smoke" / "smoke_report.json").is_file()
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


def _rehash_report(P):
    """Keep FW's synthetic report.json files block in step with a rewritten world."""
    files = {a: {k: {"path": v, "sha256": O.sha256(Path(v))} for k, v in r.items()}
             for a, r in R.world_files(P).items()}
    Path(P.fw_report).write_text(json.dumps({"prereg_id": R.FW_ID, "files": files}))


def test_content_problems_map_to_replay_worlds_unusable(tmp_path):
    import numpy as np

    P = R.build_synthetic(tmp_path / "in")
    _rewrite(Path(P.seqs()[3]), member_labels=np.array(["fold0", "fold1", "fold2"]))
    _rehash_report(P)
    rep = R._score(P, seed=R.SMOKE_SEED, k=2, k_null=2, smoke=True, log=lambda *_: None)
    assert rep["status"] == "not_evaluable" and rep["why"].startswith(R.UNUSABLE)
    assert "members" in rep["why"] and "P" not in rep
    P2 = R.build_synthetic(tmp_path / "in2")
    p = Path(P2.seqs()[5])
    with np.load(p, allow_pickle=False) as z:
        h = z["heads_mean"].copy()
    h[..., 10] += 0.01                          # a hindcast month moved
    _rewrite(p, heads_mean=h)
    _rehash_report(P2)
    rep = R._score(P2, seed=R.SMOKE_SEED, k=2, k_null=2, smoke=True, log=lambda *_: None)
    assert rep["status"] == "not_evaluable" and "hindcast differs" in rep["why"]


def test_world_files_must_match_fw_v2s_report(tmp_path):
    import numpy as np

    P = R.build_synthetic(tmp_path / "in")
    assert R.world_problems(P) == []
    _rewrite(Path(P.seqs()[0]), extra=np.zeros(1))          # same heads, new bytes
    assert R.world_problems(P) == ["W2012: sha256 differs from FW v2's report.json files"]
    with pytest.raises(R.EngineeringFailure, match="report.json files"):
        R._score(P, seed=R.SMOKE_SEED, k=2, k_null=2, smoke=True, log=lambda *_: None)
    declared = R.world_shas(P)
    _rehash_report(P)
    assert R.world_problems(P, declared) == []
    declared["C"] = {"npz": "0" * 64, "capture": declared["C"]["capture"]}
    assert R.world_problems(P, declared) == ["C: sha256 differs from the declaration"]
    Path(P.fw_report).unlink()
    assert "missing" in R.world_problems(P)[0]


def test_scoring_is_private_and_needs_the_registration(synth, tmp_path):
    assert not hasattr(R, "score")
    for kw in ({}, {"k": 3}, {"seed": R.SEED + 1}):
        with pytest.raises(R.EngineeringFailure, match="registered seed and K|INDEX.jsonl"):
            R._score(synth, **{"seed": R.SEED, "prereg_dir": tmp_path, **kw})
    with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
        R._score(synth, seed=R.SEED, smoke=True)
    with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):     # real paths, smoke flag
        R._score(R.RiderPaths(), seed=R.SMOKE_SEED, smoke=True)
    real = R.RiderPaths()
    for attr in ("outd", "index", "report", "fw_report", "fw_decl", "fv2_decl", "rulings",
                 "dataprep"):
        P = dataclasses.replace(synth, **{attr: getattr(real, attr)})
        with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
            R._score(P, seed=R.SMOKE_SEED, smoke=True)
    for attr in ("pred", "protocol", "baselines", "wells", "inputs_dir"):
        P = dataclasses.replace(synth, v1=dataclasses.replace(synth.v1, **{
            attr: getattr(real.v1, attr)}))
        with pytest.raises(R.EngineeringFailure, match="SMOKE_SEED"):
            R._score(P, seed=R.SMOKE_SEED, smoke=True)
    with pytest.raises(R.EngineeringFailure, match="smoke seed"):
        R.check_seed(R.SMOKE_SEED)


def test_the_results_guard_sees_symlinked_and_nested_results(tmp_path):
    assert R._under_results("results/twin_forward/fw_replay_v2")
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
        assert R._under_results(tf / "fw_replay_v2" / "W2012.npz")


def _attempts(report: Path) -> list[dict]:
    return [json.loads(x) for x in (report.parent / R.ATTEMPTS_NAME).read_text().splitlines()]


def test_run_refuses_before_logging_or_reading(tmp_path, synth):
    reg = tmp_path / "prereg"
    reg.mkdir()
    P = dataclasses.replace(synth, index=str(tmp_path / "ix.json"),
                            report=str(tmp_path / "rep" / "r.json"))
    with pytest.raises(R.EngineeringFailure, match="not in INDEX.jsonl"):
        R.run(P, prereg_dir=reg)
    assert not (tmp_path / "rep").exists() and not Path(P.index).exists()
    with pytest.raises(R.EngineeringFailure, match="scores FW.rider.v1 only"):
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
    doc["design"] = {**R.design(), "seed": R.SEED + 1}
    with pytest.raises(R.EngineeringFailure, match="seed"):
        R.load_registered()
    doc["design"] = {**R.design(), "arms": ["RD2012"]}
    with pytest.raises(R.EngineeringFailure, match="arms"):
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


def test_the_rider_is_model_only(tmp_path, monkeypatch, synth):
    """The declaration says so, its inputs hold no observation product, the code names none,
    and a synthetic score opens only its synthetic inputs (worlds, frozen forecast, pre-origin
    baselines and noise export)."""
    d = PR.draft_prereg(prereg_dir=tmp_path, P=synth)
    assert "no observed head" in d["model_only"]["observations_read"]
    assert "pre-origin observation-derived statistics" in d["model_only"]["observations_read"]
    assert "never reads an observed head" in d["model_only"]["statement"]
    obs = re.compile(r"heads/L1|AMP_V2|wells\.parquet|ls_cache|observations?\.csv|"
                     r"evaluation\.csv")
    assert not [p for p in d["inputs"] if obs.search(p) and not p.endswith("/MANIFEST.json")]
    code = re.sub(r'"""(?:.|\n)*?"""', "", (D / "run_rider.py").read_text(encoding="utf-8"))
    # the synthetic MANIFEST stand-ins name products; they are written, never read
    code = code.split("SYNTH_PRODUCTS = ")[0] + code.split("def build_synthetic")[1]
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
    rep = R._score(P, seed=R.SMOKE_SEED, k=2, k_null=2, smoke=True, log=lambda *_: None)
    monkeypatch.undo()
    assert rep["status"] in ("scored", "not_evaluable")
    data = [f for f in opened if not f.endswith((".py", ".pyc")) and "site-packages" not in f]
    root = str((tmp_path / "in").resolve())
    assert data and all(str(Path(f).resolve()).startswith(root) for f in data), data
    names = {Path(f).name for f in data}
    assert names <= {"predictions.csv", "baselines_e02c.csv", "wells_158.csv", "noise_ar1.csv",
                     "obs_mask.csv", "anomaly_pool.csv", "protocol.json", "replay_index.json",
                     "fw_report.json",
                     *[f"{a}.npz" for a in (*R.W_ARMS, R.CLIM_ARM)],
                     *[f"{a}.fw.npz" for a in (*R.W_ARMS, R.CLIM_ARM)]}, names


def test_draft_declaration_key_fields(tmp_path, synth):
    d = PR.draft_prereg(prereg_dir=tmp_path, P=synth)
    from harness import prereg, prereg3

    assert set(prereg.REQUIRED) <= set(d)
    assert prereg3.validate_prereg(d, units_registry=None, prereg_dir=tmp_path,
                                   registries=[]) == []
    assert d["id"] == "FW.rider.v1" and d["kind"] == "declaration" and "created" not in d
    assert d["design"]["seed"] == R.SEED and d["design"]["K_WORLD"] == 250
    assert d["design"]["arms"] == list(R.W_ARMS) and d["design"]["hindcast_tol_m"] == 0.0
    assert d["attempt_policy"]["exploratory_followups_max"] == 0
    assert "scored once" in d["attempt_policy"]["scored_runs"]
    assert set(d["frozen_code"]) == set(R.FROZEN)
    assert d["reused_code"]["pins"] == R.REUSED and d["reused_code"]["current_files_match"]
    assert d["reused_code"]["pins_match_fv2_declaration"] is True
    assert set(d["readings"]["wording"]) == {*R.READINGS, "not_evaluable"}
    assert set(d["readings"]["bands"].values()) == {*R.READINGS, "not_evaluable"}
    cav = d["caveat_rule"]
    assert "never on the page" in cav["role"] and "shown nowhere before report.json" in cav["role"]
    assert any("no status change" in x for x in cav["never"])
    assert "No observation after 2022-12" in d["u7"]
    assert set(d["worlds"]["files"]) == {R.CLIM_ARM, *R.W_ARMS}
    assert d["worlds"]["report"]["files_match"] is True
    assert "W2012..W2022" in d["worlds"]["arms_note"] and "RD" in d["worlds"]["arms_note"]
    uses = {(m["product"], m["use"]) for m in d["manifest_checks"] if m.get("manifest")}
    assert ("results/dataprep/heads/L1_clean/v3", "baseline") in uses
    assert ("results/dataprep/twin_rain/preorigin/v2", "design_time_evidence") in uses
    assert all(m["allowed"] for m in d["manifest_checks"])
    assert d["gates"]["N"]["status"] == "ok" and d["gates"]["N"]["tolerance"] > 0
    acc = d["disclosures"]["accidental_compute"]
    for k in ("READ:", "WRITTEN:", "SEED:", "PRINTED:", "c8705388917735be", "31750093",
              "registered seed's draws were never made", "No P, pass rate",
              "'gate E passed (engineering)'", "'gate N passed'", "DID NOT RAISE",
              "noise_ar1.csv", "baselines_e02c.csv", "predictions.csv", "RD2012..RD2022"):
        assert k in acc, k
    look = d["disclosure_look"]
    assert look["units"] == ["FWR:accidental_smoke_compute_20261106"]
    assert look["purpose"] == "disclosure"
    assert "none" in look["members"]["FWR:accidental_smoke_compute_20261106"]
    assert not re.search(r"E02c?:|heads_20|prospective_heads", json.dumps(look))
    assert not STATION.search(json.dumps(d, ensure_ascii=False))
    assert "/" + "home/" not in json.dumps(d)
    # readiness on the synthetic draft: only what a synthetic set cannot have
    msgs = PR.readiness(d, P=synth, prereg_dir=tmp_path)
    assert any("ruling R8.7" in m for m in msgs)                 # not the real rulings sha
    assert not any("sha256 differs" in m or "MANIFEST" in m or "replay_fields" in m
                   or "REUSED" in m for m in msgs), msgs


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
    """NEW-1: with the declaration copied into a temporary registry, the real (or any
    results/) paths are refused before anything is read or logged; synthetic paths may use it."""
    reg, rows = _tmp_registry(tmp_path, synth)
    doc, sha = R.load_registered(prereg_dir=reg, rows_dir=rows)   # the copy itself is intact
    assert {k: doc["design"][k] for k in R.design()} == R.design() and len(sha) == 64
    fake = tmp_path / "repo" / "results"
    P = dataclasses.replace(synth, outd=str(fake / "twin_forward" / "fw_replay_v2"),
                            index=str(fake / "ix.json"), report=str(fake / "rep" / "r.json"))
    for paths in (P, R.RiderPaths()):
        with pytest.raises(R.EngineeringFailure, match="real registry only"):
            R._score(paths, seed=R.SEED, prereg_dir=reg, rows_dir=rows)
        with pytest.raises(R.EngineeringFailure, match="real registry only"):
            R.run(paths, prereg_dir=reg, rows_dir=rows)
        with pytest.raises(R.EngineeringFailure, match="real registry only"):
            R._registry_guard(paths, rows_dir=rows)
    assert not fake.exists()                                     # nothing written or logged
    R._registry_guard(synth, prereg_dir=reg, rows_dir=rows)      # synthetic: allowed
    from harness import prereg3
    R._registry_guard(R.RiderPaths(), prereg_dir=prereg3.PREREG_DIR,
                      rows_dir=prereg3.LEDGER_ROWS)              # the real registry: allowed


def test_the_disclosure_look_records_and_blocks_nothing(tmp_path):
    """NEW-2: DISCLOSURE_LOOK is accepted by prereg3.record_look as given, and its 'none'
    membership overlaps and blocks no head-window test."""
    from harness import prereg3

    reg = tmp_path / "prereg"
    reg.mkdir()
    L = PR.DISCLOSURE_LOOK
    assert set(L["members"]) == set(L["units"])
    prereg3.record_look(L["units"], L["data_scope"], L["wp"], L["purpose"],
                        members=L["members"], note="test", prereg_dir=reg, registries=[])
    lk = list(prereg3.read_chained(reg / "looks.jsonl"))[-1]
    assert lk["units"] == L["units"] and lk["purpose"] == "disclosure"
    idx = prereg3.membership_index(reg, registries=[])
    head = "I1:prospective_heads_2026-11_2027-10_R0wells"
    idx.members[head] = prereg3._declared_members(
        {"ids": ["w1", "w2"], "window": "2026-11..2027-10", "description": "stand-in"})
    test = {"test_id": "T", "data_scope": "heads:2026-11..2027-10", "fresh_units": [head]}
    assert idx.overlap(L["units"][0], head) is False
    assert prereg3.look_hits(lk, test, idx) == {"block": [], "overlap": [], "unknown": []}
    assert prereg3.blocking_looks([test], None, reg, idx) == []
