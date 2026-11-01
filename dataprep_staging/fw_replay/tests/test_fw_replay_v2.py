"""CPU tests of the FW.weather_replay.v2 scripts (prereg_fw2.py, score_fw2.py, run_fw2.sh,
manifest_rain2.py): no forward run, no GPU, nothing registered, no look logged, nothing
written outside pytest's tmp dirs. Forward outputs and captures are SYNTHETIC (the builders of
test_fw_replay.py, imported read-only). v1's files are checked to be byte-identical. Run from
the repository root:

    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        --basetemp=<scratch dir> dataprep_staging/fw_replay/tests
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(X))
sys.path.insert(0, str(REPO / "dataprep_staging"))
import manifest_rain2 as MR2  # noqa: E402
import prereg_fw2 as P2  # noqa: E402
import score_fw as S1  # noqa: E402
import score_fw2 as S2  # noqa: E402
import test_fw_replay as T1  # noqa: E402  (v1's synthetic builders, read-only)

DECL2 = REPO / f"results/dataprep/preregistration/{S2.PREREG_ID}.json"
V1_DECL = REPO / P2.V1_DECL
XR = "dataprep_staging/fw_replay"
V1_FILES = (f"{XR}/prereg_fw.py", f"{XR}/score_fw.py", f"{XR}/run_fw.sh",
            f"{XR}/manifest_rain.py", f"{XR}/tests/test_fw_replay.py",
            "hydrophysics/twin/replay.py")
CASES = ("inelastic_present", "mostly_elastic")
FILLS = (S2.PLACEHOLDER_FILL, {**S2.PLACEHOLDER_FILL, "years": ""},
         {**S2.PLACEHOLDER_FILL, "all_zero": True})


def run2(*cmd, env=None):
    return subprocess.run(["bash", str(X / "run_fw2.sh"), *cmd], cwd=REPO,
                          capture_output=True, text=True, env={**os.environ, **(env or {})})


# ----------------------------------------------------------------------------- v1 untouched
def test_v1_files_are_byte_identical():
    """v1 runs as registered: its frozen code and the files v2 imports keep v1's sha256."""
    want = {**P2.V1_FROZEN_CODE, f"{XR}/manifest_rain.py": P2.V1_MANIFEST_RAIN_SHA256}
    for p, h in want.items():
        assert S2.sha256(REPO / p) == h, p
    if shutil.which("git") and (REPO / ".git").exists():
        r = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--",
                            *V1_FILES], capture_output=True)
        assert r.returncode == 0, "a v1 file differs from HEAD"
    if V1_DECL.exists():                     # every v1 input under fw_replay / the engine
        assert S2.sha256(V1_DECL) == P2.V1_SHA256
        d = json.loads(V1_DECL.read_text())
        assert d["frozen_code"] == P2.V1_FROZEN_CODE
        assert d["recipe"]["run_code"] == P2.V1_RUN_CODE and d["created"] == P2.V1_CREATED
        for rec in d["inputs"]:
            p = rec["path"]
            if p.startswith((XR, "hydrophysics/twin/")):
                assert S2.sha256(REPO / p) == rec["sha256"], p
        assert P2.v1_problems(REPO) == []


def test_v1_scorer_is_imported_not_copied():
    assert S2.S1 is S1 and S2.arm_spec is S1.arm_spec and S2.ENGINE is S1.ENGINE
    assert f"{XR}/score_fw.py" in S2.FROZEN and S1.PREREG_ID == "FW.weather_replay.v1"
    assert (S1.OUT_STEM, S1.LOGDIR) == ("results/twin_forward/fw_replay/{arm}",
                                        "results/twin_runs/fw_replay")


# ----------------------------------------------------------------------------- separation
def test_ids_outputs_and_reports_are_separate():
    assert S2.PREREG_ID == P2.PREREG_ID == "FW.weather_replay.v2" != S1.PREREG_ID
    assert S2.OUT_STEM == "results/twin_forward/fw_replay_v2/{arm}"
    assert S2.LOGDIR == "results/twin_runs/fw_replay_v2"
    assert S2.REPORT == P2.REPORT == "results/dataprep/reports/FW.weather_replay.v2/report.json"
    assert P2.PREGATE_LOG == "results/twin_runs/fw_replay_v2/design_time/pregate.jsonl"
    assert S2.GATE_JSON.startswith(S2.LOGDIR + "/") and S2.DEFAULT_ATTEMPTS.startswith(
        S2.LOGDIR + "/")
    for v1 in ("results/twin_forward/fw_replay", "results/twin_runs/fw_replay/attempts.jsonl",
               "results/twin_runs/fw_replay/gate_forcing.json",
               "results/dataprep/reports/FW.weather_replay.v1/report.json"):
        assert S2.in_v1_dirs(v1), v1
    for v2 in (S2.OUT_STEM.format(arm="C") + ".npz", S2.DEFAULT_ATTEMPTS, S2.GATE_JSON,
               S2.REPORT, "results/twin_forward/fw_replay_v2"):
        assert not S2.in_v1_dirs(v2), v2
    assert P2.runner_problems() == [] and P2.scorer_problems() == []


def test_runner_lines_bind_v2_and_never_v1():
    lines = (X / "run_fw2.sh").read_text().splitlines()
    for must in ("ID=FW.weather_replay.v2", "LOG=$R/fw_replay_v2", "FIELDS=energy+weather",
                 "OUTD=results/twin_forward/fw_replay_v2", "TMUX_PREFIX=fwv2_",
                 "SCORER=dataprep_staging/fw_replay/score_fw2.py"):
        assert must in lines, must
    for never in ("ID=FW.weather_replay.v1", "LOG=$R/fw_replay",
                  "OUTD=results/twin_forward/fw_replay"):
        assert never not in lines, never


def test_tmux_session_names_never_collide_with_v1():
    p = run2("launch", "primary", "4", env={"DRY": "1"})
    assert p.returncode == 0, p.stderr
    got = p.stdout.split()
    assert got == ["fwv2_1", "fwv2_2", "fwv2_3", "fwv2_4"]
    assert not any(re.fullmatch(r"fw[0-9]+", s) for s in got)
    assert run2("launch", "primary", env={"DRY": "1"}).stdout.split() == [
        "fwv2_1", "fwv2_2", "fwv2_3"]
    p = run2("launch", "primary", "0", env={"DRY": "1"})
    assert p.returncode == 1 and "positive session count" in p.stderr
    sh = (X / "run_fw2.sh").read_text()
    assert '-s "fw$i"' not in sh and '-t "fw$i"' not in sh
    assert "fwv2_1..fwv2_n" in P2.doc()["recipe"]["tmux_sessions"]


def test_runner_dry_run_matches_the_declared_commands():
    subprocess.run(["bash", "-n", str(X / "run_fw2.sh")], check=True)
    for arm in S2.ARMS:
        p = run2("forward", arm, env={"DRY": "1"})
        assert p.returncode == 0, p.stderr
        assert p.stdout.strip() == P2.command(arm)
        assert f"--out results/twin_forward/fw_replay_v2/{arm}" in p.stdout
        assert "--out results/twin_forward/fw_replay/" not in p.stdout
        if arm != "C":
            assert "--replay-fields energy+weather" in p.stdout
    for group, want in (("weather", list(S2.RED_WEA)),
                        ("all", [*S2.PRIMARY, *S2.RED_DET, *S2.RED_RAW, *S2.RED_WEA])):
        p = run2("forward", group, env={"DRY": "1"})
        assert [ln.rsplit("/", 1)[-1] for ln in p.stdout.split("\n") if ln.strip()] == want
    d = P2.doc()
    for a in ("C", "W2015", "RD2015", "RR2015", "RW2015"):
        assert d["recipe"][f"forward_{a}"] == P2.command(a)


@pytest.mark.skipif(DECL2.exists(), reason="v2 registered; refusal test not applicable")
@pytest.mark.parametrize("cmd", [["check"], ["gate"], ["forward", "C"], ["repro"], ["score"],
                                 ["launch", "primary"], ["forward", "all"]])
def test_runner_refuses_without_registration(cmd):
    p = run2(*cmd)
    assert p.returncode == 1 and "declare FW.weather_replay.v2 first" in p.stderr, p.stderr


@pytest.mark.skipif(DECL2.exists(), reason="v2 registered; pregate refuses then")
def test_runner_pregate_refusals_write_nothing():
    for cmd, msg in ((["pregate"], "7-char run code"), (["pregate", "zzz"], "7-char run code"),
                     (["pregate", "0000000"], "worktree")):
        p = run2(*cmd, env={"HOME": "/nonexistent-home"})
        assert p.returncode == 1 and msg in p.stderr, p.stderr
    p = run2("forward", "nonsense", env={"DRY": "1"})
    assert p.returncode == 1 and "unknown arm or group" in p.stderr


# ----------------------------------------------------------------------------- weather mode
def test_scorer_refuses_any_mode_but_energy_weather():
    with pytest.raises(S2.EngineeringError, match="scores 'energy\\+weather' only"):
        S2.declared_fields({"recipe": {"replay_fields": "energy"}})
    assert S2.declared_fields({"recipe": {"replay_fields": "energy+weather"}}) == S2.FIELDS
    for f in (S2.arms, lambda m: S2.wording("mostly_elastic", S2.PLACEHOLDER_FILL, m),
              lambda m: S2.score({}, m)):
        with pytest.raises(ValueError, match="energy\\+weather"):
            f("energy")
    assert set(S2.arms()) == set(S1.ARMS) and set(S2.RED_WEA) <= set(S2.arms())


def test_weather_wordings_carry_the_gauge_caveat_and_v2_id():
    for must in ("rf11", "2021-22", "rf15", "2015 H2", "zero-fill"):
        assert must in S2.GAUGE_CAVEAT, must
    for case in CASES:
        for fill in FILLS:
            w = S2.wording(case, fill)
            w1 = S1.wording(case, fill, "energy+weather")
            assert set(w) == set(w1)
            for k, t in w.items():
                assert t.endswith(S2.GAUGE_CAVEAT), (case, k)
                # identical to v1's weather-mode text apart from the id and the caveat
                assert t == w1[k].replace(S1.PREREG_ID, S2.PREREG_ID) + " " + S2.GAUGE_CAVEAT
                assert "weather_replay.v1" not in t
                assert "pumping anomalies only" not in t
            for k in ("C15", "TWIN_EVIDENCE_s13"):
                assert "FW.weather_replay.v2" in w[k]
            assert "weather and detrended pumping" in w["C15"]


def test_declaration_is_weather_mode_and_renders_v2_wordings():
    from harness import prereg3

    d = P2.doc()
    assert prereg3.validate_prereg(d) == []
    assert d["id"] == "FW.weather_replay.v2" and d["recipe"]["replay_fields"] == "energy+weather"
    assert list(d["recipe"]["arms_run"]) == list(S1.ARMS)
    m = d["evidence_and_ledger_mapping"]
    assert m["case_mostly_elastic"] == S2.wording("mostly_elastic", S2.PLACEHOLDER_FILL)
    assert m["case_inelastic_present"] == S2.wording("inelastic_present", S2.PLACEHOLDER_FILL)
    for case in ("case_inelastic_present", "case_mostly_elastic",
                 "case_mostly_elastic_if_all_shares_zero"):
        assert all(S2.GAUGE_CAVEAT in t for t in m[case].values()), case
    assert S2.GAUGE_CAVEAT in m["case_inelastic_present_if_no_source_year"]["C27"]
    assert "weather_replay.v1" not in json.dumps({k: m[k] for k in m if k.startswith("case")})
    assert m["on_not_evaluable"].startswith("a ledger note 'FW.weather_replay.v2 not evaluable")
    assert d["frozen_code"].keys() == set(S2.FROZEN)
    assert d["recipe"]["outputs"]["never"] == [P2.V1_OUTD, P2.V1_LOG,
                                               "results/dataprep/reports/FW.weather_replay.v1"]


def test_declaration_discloses_v1_and_binds_its_inputs():
    d = P2.doc()
    w = d["disclosed"]["v1_concurrent"]
    for must in (P2.V1_CREATED, P2.V1_SHA256[:16], "pumping-only", "absolute-path bug",
                 "runs as registered", "not withdrawn", "FV.decidability.v2_postfailure",
                 "lower bound", "separate, concurrent declaration", "same recipe",
                 "except the replay mode", "fwv2_1..fwv2_n", P2.RULING_ID, P2.RULING_V1,
                 "byte-identical"):
        assert must in w, must
    assert "withdrawn unrun" not in json.dumps(d)
    g = d["disclosed"]["gauge_caveat"]
    assert "rf11 for all of 2021-22" in g and "rf15 in 2015 H2" in g and S2.GAUGE_CAVEAT in g
    assert P2.RAIN_MANIFEST.endswith("twin_rain/preorigin/v2/MANIFEST.json")
    assert P2.RAIN_MANIFEST in d["disclosed"]["rain_input"]
    paths = [r["path"] if isinstance(r, dict) else r for r in d["inputs"]]
    for must in (P2.RAIN_MANIFEST, P2.V1_DECL, P2.RULINGS_E, P2.RULINGS_F, P2.TESTS,
                 f"{XR}/score_fw.py", f"{XR}/prereg_fw.py", f"{XR}/manifest_rain2.py"):
        assert must in paths, must
    assert {"path": P2.V1_DECL, "sha256": P2.V1_SHA256} in d["inputs"]
    pa = d["attempt_policy"]["pre_declared_alternatives"][0]
    assert "none for the replay mode" in pa and P2.RULING_ID in pa


def test_readiness_refuses_outside_weather_mode(tmp_path):
    probs = P2.readiness(None, tmp_path)
    assert any("registers in 'energy+weather' mode only" in x for x in probs)
    assert any(P2.V1_DECL in x for x in probs)              # v1 bound at its sha256
    assert any(P2.RULINGS_F in x for x in probs)
    d = P2.doc(None, tmp_path)                               # still drafted in weather mode
    assert d["recipe"]["replay_fields"] == "energy+weather"
    assert "NOT READY" in d["recipe"]["replay_fields_reason"]
    rain_manifest(tmp_path)
    probs = P2.readiness(None, tmp_path)
    assert not any("mode only" in x for x in probs)
    assert P2.replay_fields(tmp_path)[0] == "energy+weather"
    assert "NOT READY" not in P2.doc(None, tmp_path)["recipe"]["replay_fields_reason"]
    rain_manifest(tmp_path, inputs=[{"path": "/srv/elsewhere/" + P2.RF,
                                     "sha256": P2.RF_SHA256}])
    probs = P2.readiness(None, tmp_path)
    assert any("mode only" in x and "not repo-relative" in x for x in probs)


def test_rulings_and_v1_checks(tmp_path):
    assert any(P2.V1_DECL in x for x in P2.v1_problems(tmp_path))
    rulings(tmp_path)
    assert P2.rulings_problems(tmp_path) == []
    rulings(tmp_path, applies=[P2.V1_ID])
    assert any("does not apply to FW.weather_replay.v2" in x
               for x in P2.rulings_problems(tmp_path))
    (tmp_path / P2.RULINGS_E).write_text(json.dumps({"rulings": []}))
    assert any(P2.RULING_V1 in x for x in P2.rulings_problems(tmp_path))
    if (REPO / P2.RULINGS_F).exists():
        assert P2.rulings_problems(REPO) == []


def test_pregate_record_is_v2s_own(tmp_path):
    assert P2.pregate_record("abc1234", tmp_path) is None
    v1 = tmp_path / S1.LOGDIR / "design_time/pregate.jsonl"   # v1's pass is not reused
    v1.parent.mkdir(parents=True)
    v1.write_text(json.dumps({"event": "pregate", "code": "abc1234", "rc": 0}) + "\n")
    assert P2.pregate_record("abc1234", tmp_path) is None
    f = tmp_path / P2.PREGATE_LOG
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"event": "pregate", "code": "abc1234", "rc": 0,
                             "gate_sha256": "e" * 64, "time": "t"}) + "\n")
    assert P2.pregate_record("abc1234", tmp_path)["rc"] == 0


def test_prereg_cli_validate_is_a_dry_run(capsys):
    reg = REPO / "results/dataprep/preregistration"
    before = sorted(reg.glob("FW.*")) if reg.exists() else []
    assert P2.main(["--validate", "--quiet"]) == 1
    out = capsys.readouterr().out
    assert "validate_prereg: valid" in out and "recipe/binding checks: ok" in out
    assert "--run-code not given" in out and "replay fields: energy+weather" in out
    assert (sorted(reg.glob("FW.*")) if reg.exists() else []) == before


# ----------------------------------------------------------------------------- rain manifest
def rain_manifest(repo: Path, **over) -> Path:
    """A rain product v2 manifest as manifest_rain2.py writes it (repo-relative paths)."""
    mf = repo / P2.RAIN_MANIFEST
    mf.parent.mkdir(parents=True, exist_ok=True)
    rec = {"product": P2.RAIN_PRODUCT, "version": P2.RAIN_VERSION, "gate": {"passed": True},
           "allowed_uses": ["forcing_frozen", "design_time_evidence"],
           "fit_window_end": "2022-12",
           "inputs": [{"path": P2.RF, "sha256": P2.RF_SHA256},
                      {"path": MR2.RF_STATIONS, "sha256": "d" * 64}]}
    mf.write_text(json.dumps({**rec, **over}))
    return mf


def rulings(repo: Path, applies=None) -> None:
    for f, rid in P2.RULINGS.items():
        p = repo / f
        p.parent.mkdir(parents=True, exist_ok=True)
        a = applies if (applies is not None and rid == P2.RULING_ID) else [P2.V1_ID,
                                                                          P2.PREREG_ID]
        p.write_text(json.dumps({"rulings": [{"id": rid, "applies_to": a}]}))


def test_rain_check_follows_the_manifest(tmp_path):
    assert P2.replay_fields(tmp_path)[0] == "energy"
    rain_manifest(tmp_path)
    assert P2.replay_fields(tmp_path)[0] == "energy+weather"
    rain_manifest(tmp_path, allowed_uses=["design_time_evidence"])
    assert P2.replay_fields(tmp_path)[0] == "energy+weather"
    for over, msg in (
            ({"allowed_uses": ["after_the_fact_analysis"]}, "allows neither"),
            ({"inputs": [{"path": P2.RF, "sha256": "0" * 64}]}, "not ef5a525f"),
            ({"inputs": [{"path": "chou-shui-data/data/other.csv", "sha256": P2.RF_SHA256}]},
             "does not name"),
            ({"version": "v1"}, "not 'twin_rain/preorigin' 'v2'"),
            ({"gate": {"passed": False}}, "gate did not pass"),
            ({"fit_window_end": "2023-01"}, "after the origin")):
        rain_manifest(tmp_path, **over)
        f, why = P2.replay_fields(tmp_path)
        assert f == "energy" and msg in why, (over, why)
    (tmp_path / P2.RAIN_MANIFEST).write_text("{not json")
    assert "unreadable" in P2.replay_fields(tmp_path)[1]


BAD_PATHS = ["/srv/elsewhere/chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv",
             "~/checkout/chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv",
             "chou-shui-data/../chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv",
             "C:/data/rf_timeseries_vendor_2012_2022.csv", ""]


@pytest.mark.parametrize("bad", BAD_PATHS)
def test_absolute_and_home_paths_are_refused_with_a_message(tmp_path, bad):
    """v1's bug: the manifest named the right file at the right sha256 by an absolute path
    and the comparison silently missed it. v2 refuses such a path on both sides and says
    why."""
    assert not P2.repo_relative(bad) and P2.repo_relative(P2.RF)
    with pytest.raises(MR2.M.ManifestError, match="not repo-relative"):
        MR2.repo_relative(bad)
    assert MR2.repo_relative(P2.RF) == P2.RF
    rain_manifest(tmp_path, inputs=[{"path": bad, "sha256": P2.RF_SHA256}])
    f, why = P2.replay_fields(tmp_path)
    assert f == "energy"
    assert "not repo-relative" in why and "refused" in why and "manifest_rain2.py" in why
    assert MR2.path_problems({"inputs": [{"path": bad}]})


def _cache(root: Path, empty=False) -> None:
    """A synthetic cache and gauge table at the repo-relative paths under ``root``."""
    days = [MR2.FIRST + dt.timedelta(days=i) for i in range((MR2.LAST - MR2.FIRST).days + 1)]
    rows = ["date time,rf1,rf2"] + [f"{d.isoformat()} 00:00,{'' if empty and i == 3 else 1.5},0"
                                    for i, d in enumerate(days)]
    rain, st = root / MR2.RF, root / MR2.RF_STATIONS
    rain.parent.mkdir(parents=True, exist_ok=True)
    rain.write_text("\ufeff" + "\n".join(rows) + "\n", encoding="utf-8")
    st.write_text("\ufeffrf_id,rf_num,TM_X97,TM_Y97\nrf1,1,0,0\nrf2,2,0,0\n", encoding="utf-8")


def test_rain_manifest_records_repo_relative_paths_through_a_symlink(tmp_path):
    """The v1 bug case: chou-shui-data/ is a symlink to another checkout. v2 records the
    paths as they sit in the repository, hashes the real file, and its manifest passes the
    declaration's check (sha256 aside: the cache here is synthetic)."""
    other = tmp_path / "other_checkout"
    _cache(other)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "chou-shui-data").symlink_to(other / "chou-shui-data")
    m = MR2.build(tmp_path / "prod" / MR2.VERSION, repo)
    assert m["version"] == "v2" and m["product"] == "twin_rain/preorigin"
    assert [r["path"] for r in m["inputs"]] == [MR2.RF, MR2.RF_STATIONS]
    assert m["inputs"][0]["sha256"] == S2.sha256(other / MR2.RF)
    assert [c["path"] for c in m["code"]] == list(MR2.CODE)
    assert m["allowed_uses"] == ["forcing_frozen", "design_time_evidence"]
    assert "scoring_target" in m["forbidden_uses"] and m["gate"]["passed"]
    assert any("rf11" in n and "rf15" in n for n in m["notes"])
    assert "FW.weather_replay.v2" in m["parameters"]["consumer"]
    txt = json.dumps(m)
    assert str(other) not in txt and str(tmp_path) not in txt
    assert not re.search(r"/home/\w", txt) and MR2.path_problems(m) == []
    assert "1.5" not in (tmp_path / "prod" / MR2.VERSION / "rain_record.json").read_text()
    good = {**m, "inputs": [{**m["inputs"][0], "sha256": P2.RF_SHA256}, m["inputs"][1]]}
    assert P2.rain_manifest_problems(good) == []
    # a v1-style manifest (absolute input paths) is flagged by both sides
    v1_like = {**good, "inputs": [{**r, "path": str(other / r["path"])}
                                  for r in good["inputs"]]}
    assert MR2.path_problems(v1_like) and P2.rain_manifest_problems(v1_like)


def test_rain_manifest_v2_never_rewrites_a_version_and_gates_structure(tmp_path):
    repo = tmp_path / "repo"
    _cache(repo)
    with pytest.raises(MR2.M.ManifestError, match="v1 is never rewritten"):
        MR2.build(tmp_path / "prod" / "v1", repo)
    d = tmp_path / "prod" / MR2.VERSION
    d.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        MR2.build(d, repo)
    bad = tmp_path / "bad"
    _cache(bad, empty=True)
    m = MR2.build(tmp_path / "prod2" / MR2.VERSION, bad)
    assert not m["gate"]["passed"]
    assert MR2.structure is MR2.MR1.structure and MR2.gate is MR2.MR1.gate


# ----------------------------------------------------------------------------- scoring
def register2(monkeypatch, fields="energy+weather"):
    def fake_load(prereg_dir=None, rows_dir=None):
        return {"frozen_code": {p: S2.sha256(REPO / p) for p in S2.FROZEN},
                "input_digests": P2.INPUT_DIGESTS,
                "recipe": {"replay_fields": fields, "engine_code_sha256": T1.ENGINE,
                           "run_code": T1.CODE, "county_map": T1.COUNTY}}, "f" * 64
    monkeypatch.setattr(S2, "load_registered", fake_load)
    d = P2.INPUT_DIGESTS[0]
    monkeypatch.setattr(S1, "dir_digest", lambda path, pattern: (d["n_files"], d["sha256"]))


@pytest.fixture
def any_ref(monkeypatch):
    orig = S1.load_forward

    def load(path, arm, *a, **k):
        d = orig(path, arm, *a, **k)
        if k.get("page"):
            monkeypatch.setattr(S1, "REF_SHA256", d["sha256"])
        return d
    monkeypatch.setattr(S1, "load_forward", load)


@pytest.fixture(scope="module")
def world_w(tmp_path_factory):
    return T1.build_world(tmp_path_factory.mktemp("fw2_high"), T1.SHARES_HIGH,
                          fields="energy+weather")


def test_v2_scores_like_v1_in_weather_mode_with_v2_wordings(world_w, monkeypatch, any_ref,
                                                            tmp_path, capsys):
    """Same synthetic weather-mode world, v1's scorer (registered in weather mode) and v2's:
    every number is identical; v2's wordings are v1's with v2's id and the gauge caveat."""
    w = world_w
    att = T1.attempts(tmp_path / "att.jsonl", w)
    T1.register(monkeypatch, "energy+weather")
    assert S1.main(T1.args(w, att, tmp_path / "r1.json")) == 0
    register2(monkeypatch)
    assert S2.main(T1.args(w, att, tmp_path / "r2.json")) == 0
    r1, r2 = (json.loads((tmp_path / f"r{i}.json").read_text()) for i in (1, 2))
    assert r2["prereg_id"] == "FW.weather_replay.v2" and r2["scorer_version"] == "2.0.0"
    assert r2["verdict"] == "scored" and r2["wording_case"] == "inelastic_present"
    for k in ("summary", "reported_not_gating", "metrics", "reproduction", "files",
              "verdict", "wording_case", "replay_fields"):
        assert r1[k] == r2[k], k
    assert r2["reported_not_gating"]["attribution"]["RW"]["share"]["n"] == 11
    assert r2["registration"]["frozen_code_sha256"].keys() == set(S2.FROZEN)
    for k, t in r2["wording"].items():
        assert t == r1["wording"][k].replace(S1.PREREG_ID, S2.PREREG_ID) + " " + \
            S2.GAUGE_CAVEAT, k
    assert "16.0 %" in r2["wording"]["C15"] and r2["rain_caveat"] == S2.GAUGE_CAVEAT
    assert "VERDICT: scored (wording case inelastic_present" in capsys.readouterr().out


def test_v2_mostly_elastic_and_not_evaluable(monkeypatch, any_ref, tmp_path):
    register2(monkeypatch)
    w = T1.build_world(tmp_path / "w", T1.SHARES_LOW, fields="energy+weather")
    out = tmp_path / "rep.json"
    assert S2.main(T1.args(w, T1.attempts(tmp_path / "att.jsonl", w), out)) == 0
    wd = json.loads(out.read_text())["wording"]
    assert wd["C15"].startswith("Mostly elastic (median 6.0 %, below the 10 % rule) also")
    assert "the mean-forcing caveat is closed." in wd["C15"]
    assert all(t.endswith(S2.GAUGE_CAVEAT) for t in wd.values())
    real = T1.parts

    def tiny(share, avoided_el=0.02, scale_tau30=0.5):
        return real(share, 1e-5 if share == 0.5 else avoided_el, scale_tau30)
    monkeypatch.setattr(T1, "parts", tiny)
    w = T1.build_world(tmp_path / "w2", {**T1.SHARES_HIGH, "W2012": 0.5, "W2013": 0.5,
                                         "W2014": 0.5}, fields="energy+weather")
    out = tmp_path / "rep2.json"
    assert S2.main(T1.args(w, T1.attempts(tmp_path / "att2.jsonl", w), out)) == 0
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "not_evaluable"
    note = rep["wording"]["ledger_note"]
    assert note.startswith("FW.weather_replay.v2 not evaluable (8 of 11")
    assert S2.GAUGE_CAVEAT not in note              # no result cited, no caveat


@pytest.mark.parametrize("what, msg", [
    ("energy", "scores 'energy+weather' only"), ("failedrepro", "under FW.weather_replay.v2"),
    ("v1dir", "lie in FW.weather_replay.v1's directories"), ("frozen", "frozen code differs")])
def test_v2_engineering_refusals(world_w, monkeypatch, any_ref, tmp_path, capsys, what, msg):
    register2(monkeypatch, "energy" if what == "energy" else "energy+weather")
    if what == "frozen":
        real = S2.load_registered

        def stale(prereg_dir=None, rows_dir=None):
            d, h = real()
            return {**d, "frozen_code": {**d["frozen_code"], f"{XR}/score_fw.py": "0" * 64}}, h
        monkeypatch.setattr(S2, "load_registered", stale)
    att = T1.attempts(tmp_path / "att.jsonl", world_w, failed_repro=what == "failedrepro")
    out = tmp_path / "rep.json"
    a = T1.args(world_w, att, out)
    if what == "v1dir":
        a = [x for x in a if not x.startswith("--out=")] + [
            "--out=results/dataprep/reports/FW.weather_replay.v1/report.json"]
    assert S2.main(a) == 1 and not out.exists()
    assert msg in capsys.readouterr().err


def test_v2_repro_mode_reads_c_only(world_w, monkeypatch, any_ref, tmp_path):
    register2(monkeypatch)
    att = T1.attempts(tmp_path / "att.jsonl", world_w, repro=False)
    assert S2.main(T1.args(world_w, att) + ["--repro"]) == 0


def test_no_home_paths_in_v2_files():
    for p in (X / "prereg_fw2.py", X / "score_fw2.py", X / "run_fw2.sh",
              X / "manifest_rain2.py", Path(__file__)):
        txt = p.read_text()
        assert not re.search(r"/home/\w", txt), p
        if p.suffix == ".py":
            assert "from __future__ import annotations" in txt, p
