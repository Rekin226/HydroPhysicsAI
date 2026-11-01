"""CPU tests of the FS.start_invariance.v1 scripts (no forward run, no GPU, nothing registered,
no look logged, nothing written outside pytest's tmp dirs). Every forward output here is
SYNTHETIC: fake npz files with the page run's layout whose levers are known multiples of each
other. The tmp-registry round trip and the reference check read the gitignored results/ tree
(registered model output and theta/column files only) and skip when it is absent, so these
tests live outside the repository's testpaths:

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        --basetemp=<scratch dir> dataprep_staging/fs_start/tests
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
sys.path.insert(0, str(X))
sys.path.insert(0, str(REPO / "dataprep_staging"))
import prereg_fs as P  # noqa: E402
import score_fs as S  # noqa: E402

DECL = REPO / f"results/dataprep/preregistration/{S.PREREG_ID}.json"
REF_REAL = REPO / S.REF
needs_results = pytest.mark.skipif(
    not all((REPO / p).exists() for p in P.INPUT_SHA256),
    reason="results/ inputs (page run, thetas, columns) not present")

LEVER = {"cut30": 1.19, "retire_aqua": 1.41, "cut30_2030": 0.90, "retire_aqua_2030": 1.05}
START = {"cut30": 168, "retire_aqua": 168, "cut30_2030": 216, "retire_aqua_2030": 216}


# ----------------------------------------------------------------------------- synthetic
def dates() -> np.ndarray:
    return np.array([f"{2012 + i // 12}-{i % 12 + 1:02d}-01T00:00:00" for i in range(S.N_T)])


def heads(names, k=1.0, cell_factor=None, level=0.0) -> np.ndarray:
    """(n_scen, 4, 2148, 252): a smooth baseline plus, per scenario, a step lever from its
    start month, scaled by ``k`` (and per cell by ``cell_factor``)."""
    t = np.arange(S.N_T, dtype=np.float32)
    base = (10.0 + level + 0.002 * np.arange(S.N_CELLS, dtype=np.float32)[None, :, None]
            + np.array([0.0, -1.0, -2.0, -3.0], dtype=np.float32)[:, None, None]
            + 0.3 * np.sin(2 * np.pi * t / 12.0)[None, None, :])
    out = np.empty((len(names), 4, S.N_CELLS, S.N_T), dtype=np.float32)
    cf = np.ones(S.N_CELLS, dtype=np.float32) if cell_factor is None else cell_factor
    for i, n in enumerate(names):
        out[i] = base
        if n != "baseline":
            step = (t >= START[n]).astype(np.float32)
            out[i] += (k * LEVER[n]) * cf[None, :, None] * step[None, None, :]
    return out


def fake_npz(path: Path, arm: str, k=1.0, cell_factor=None, level=0.0, desc_fix=None,
             gain=None) -> Path:
    spec = S.SPEC[arm]
    names, descs = (S.REF_NAMES, S.REF_DESC) if arm == "page" else (S.SCEN_NAMES, S.SCEN_DESC)
    descs = list(descs)
    if desc_fix:
        descs[-1] = desc_fix
    hm = heads(names, k, cell_factor, level)
    np.savez(path, dates=dates(), origin=S.ORIGIN, horizon=S.HORIZON, heads_mean=hm,
             scenario_names=np.array(names), scenarios=np.array(descs),
             member_labels=np.array(spec["members"]), n_members=spec["n_members"],
             gain=spec["gain"] if gain is None else gain, hindcast_gain=0.0,
             rheology_labels=np.array(S.RHEOLOGY),
             gate=json.dumps({"git_commit": S.THETA_GIT}),
             forward_options=json.dumps({**spec["fo"], "hpc0_guard_days": [365.0, 365.0]}))
    return path


def fake_members(path: Path, ks=(1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.03, 0.97, 1.0)) -> Path:
    years = list(range(2012, 2033))
    ye = [12 * (y - 2012) + 11 for y in years]
    q = np.empty((len(S.SCEN_NAMES), 9, S.N_CELLS, 21), dtype=np.int16)
    for m, k in enumerate(ks):
        h = heads(S.SCEN_NAMES, k)[:, 1][:, :, ye]                  # layer 2, Decembers
        q[:, m] = np.round(h / 0.01).astype(np.int16)
    np.savez(path, headL2_members_yr=q, head_scale_m=0.01, years=np.array(years),
             head_member=np.array(S.ONE_MEMBER * 9), head_ic=np.arange(9),
             scenario_names=np.array(S.SCEN_NAMES))
    return path


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """One synthetic set: the page run, S0 (= the page), S1 (x1.02, level -0.5 m), S2
    (x0.97), S3 (x1.01 mean) and S3ref."""
    t = tmp_path_factory.mktemp("fs_world")
    f = {"page": fake_npz(t / "page.npz", "page"),
         "S0": fake_npz(t / "S0.npz", "S0"),
         "S1": fake_npz(t / "S1.npz", "S1", k=1.02, level=-0.5),
         "S2": fake_npz(t / "S2.npz", "S2", k=0.97),
         "S3": fake_npz(t / "S3.npz", "S3", k=1.01),
         "S3ref": fake_npz(t / "S3ref.npz", "S3ref"),
         "S3m": fake_members(t / "S3.members.npz")}
    return t, f


def attempts(path: Path, files: dict, rc: int = 0, code: str = "4c2723e",
             repro: bool = True, codes: dict | None = None) -> Path:
    with path.open("w") as fh:
        for a in S.ARMS:
            if a not in files:
                continue
            rec = {"event": "forward", "attempt": 1, "arm": a, "rc": rc,
                   "code": (codes or {}).get(a, code), "npz_sha256": S.sha256(files[a])}
            if a == "S3" and "S3m" in files:
                rec["members_npz_sha256"] = S.sha256(files["S3m"])
            fh.write(json.dumps(rec) + "\n")
        if repro:
            fh.write(json.dumps({"event": "repro", "rc": 0, "code": code,
                                 "s0_npz_sha256": S.sha256(files["S0"])}) + "\n")
    return path


def args(f: dict, att: Path, out: Path | None = None, **over) -> list[str]:
    files = {**f, **over}
    a = [f"--{arm.lower()}={files[arm]}" for arm in S.ARMS]
    a += [f"--s3-members={files['S3m']}", f"--ref={files['page']}", f"--attempts={att}"]
    return a + ([f"--out={out}"] if out else [])


@pytest.fixture
def registered(monkeypatch):
    """Pretend the declaration is registered with the current frozen code (no registry); the
    synthetic page run stands in for the registered one."""
    def fake_load(prereg_dir=None, rows_dir=None):
        return {"frozen_code": {p: S.sha256(REPO / p) for p in S.FROZEN},
                "input_digests": P.INPUT_DIGESTS}, "f" * 64
    monkeypatch.setattr(S, "load_registered", fake_load)
    monkeypatch.setattr(S, "REF_SHA256", None)
    d = P.INPUT_DIGESTS[0]                         # the well directory is not hashed here
    monkeypatch.setattr(S, "dir_digest", lambda path, pattern: (d["n_files"], d["sha256"]))


@pytest.fixture
def any_ref(monkeypatch):
    """The scorer accepts the synthetic page run (its sha256 is checked against the real one)."""
    orig = S._load
    monkeypatch.setattr(S, "REF_SHA256", S.REF_SHA256)      # restored at teardown

    def load(path, arm, *a, **k):
        d = orig(path, arm, *a, **k)
        if arm == "page":
            S.REF_SHA256 = d["sha256"]
        return d
    monkeypatch.setattr(S, "_load", load)


# ----------------------------------------------------------------------------- static
def test_screens_frozen_and_match_x5():
    assert S._screens_sha() == S.SCREENS_SHA256
    assert len(S.CELLS) == 158 and "".join(S.COUNTY).count("Y") == 84
    assert P.scorer_problems() == []
    assert len(S.UNITS) == 10 and ("cut30", 2030, "2026-27") not in S.UNITS


def test_no_home_paths_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")     # a home directory with a user name
    for p in sorted(X.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh", ".json", ".md"):
            assert not pat.search(p.read_text()), p


def test_declaration_validates_and_binds():
    from harness import prereg3

    d = P.doc()
    assert prereg3.validate_prereg(d) == []
    assert P.runner_problems() == []
    assert set(d["frozen_code"]) == set(S.FROZEN) == set(P.FROZEN)
    assert d["frozen_code"] == {p: S.sha256(REPO / p) for p in S.FROZEN}
    declared = {r["path"]: r["sha256"] for r in d["inputs"] if isinstance(r, dict)}
    assert {p: declared[p] for p in S.FROZEN} == d["frozen_code"]
    assert declared[S.REF] == S.REF_SHA256
    assert d["kind"] == "declaration" and d["id"] == "FS.start_invariance.v1"
    th = d["thresholds"]
    assert "[0.95, 1.05]" in th["co_primary_1"] and "<= 0.1" in th["co_primary_2"]
    assert "column-heads free" in d["arms"]["no_column_heads_restart_arm"]
    assert all("--column-heads free" in d["recipe"][f"forward_{a}"] for a in S.ARMS)
    assert not any("--column-heads restart" in v for v in d["recipe"].values())
    assert "REPORT-ONLY" in d["arms"]["S3"] and "GATING" in d["arms"]["S1"]
    assert "moves no claim status" in d["title"]
    assert "SEALED" in d["attempt_policy"]["values_written_before_scoring"]
    assert "fa5cca2 cannot run the page recipe" in d["disclosed"]["code_provenance"]
    # the rain the page run read is pinned, and every forward input is declared
    assert P.RF == "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv"
    assert declared[P.RF] == P.RF_SHA256 and P.RF in d["disclosed"]["page_recipe"]
    assert {p: declared[p] for p in P.FORWARD_INPUTS} == P.FORWARD_INPUTS
    assert d["input_digests"] == P.INPUT_DIGESTS and d["input_digests"][0]["path"] == P.WELLS_DIR
    assert all(f"--rf-timeseries {P.RF}" in d["recipe"][f"forward_{a}"] for a in S.ARMS)
    # the pre-declared split wording when only the superseded field restart fails
    assert any("S2_only" in x for x in d["evidence_and_ledger_mapping"]["on_fail"])


def test_runner_dry_run_matches_declared_command():
    for a in S.ARMS:
        out = subprocess.run(["bash", str(X / "run_fs.sh"), "forward", a], cwd=REPO,
                             env={**os.environ, "DRY": "1"}, capture_output=True, text=True,
                             check=True).stdout.splitlines()
        assert out == [P.command(a)]
        assert f" --rf-timeseries {P.RF} " in out[0]
    out = subprocess.run(["bash", str(X / "run_fs.sh"), "forward", "arms"], cwd=REPO,
                         env={**os.environ, "DRY": "1"}, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert out == [P.command(a) for a in ("S1", "S2", "S3", "S3ref")]


def test_forward_parser_accepts_every_arm(monkeypatch):
    """Each declared command parses with the current forward CLI (stopped right after the
    parse, before any input is loaded)."""
    import argparse
    import shlex

    import hydrophysics.twin.forward as F

    class Stop(Exception):
        pass

    got = {}
    orig = argparse.ArgumentParser.parse_args

    def parse(self, a=None, ns=None):
        got["ns"] = orig(self, a, ns)
        return got["ns"]

    def stop(*_a, **_k):
        raise Stop

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    monkeypatch.setattr(F, "set_compile_matvec", stop)
    want = {"S0": (3, 1.0, "innovation", 5.0, 1), "S1": (3, 0.0, "innovation", 5.0, 1),
            "S2": (3, 1.0, "field", None, 1), "S3": (1, 1.0, "innovation", 5.0, 8),
            "S3ref": (1, 1.0, "innovation", 5.0, 0)}
    for a, w in want.items():
        toks = shlex.split(P.command(a))
        with pytest.raises(Stop):
            F.main(toks[toks.index("hydrophysics.twin.forward") + 1:])
        ns = got["ns"]
        assert (len(ns.theta), ns.gain, ns.update_method, ns.restart_taper_km,
                ns.ic_members) == w
        assert ns.column_heads == "free" and ns.apex_hold == "calibrated"
        assert ns.hindcast_gain == 0.0 and len(ns.scenario) == 4
        assert ns.rf_timeseries == P.RF


_LADDER_PARSE = r"""
import argparse, json, shlex, sys
import hydrophysics.twin.forward as F
assert F.__file__.startswith(sys.argv[1]), F.__file__
class Stop(Exception):
    pass
got = {}
orig = argparse.ArgumentParser.parse_args
def parse(self, a=None, ns=None):
    got["ns"] = orig(self, a, ns)
    return got["ns"]
def stop(*_a, **_k):
    raise Stop
argparse.ArgumentParser.parse_args = parse
F.set_compile_matvec = stop
out = {}
for arm, cmd in json.loads(sys.argv[2]).items():
    toks = shlex.split(cmd)
    try:
        F.main(toks[toks.index("hydrophysics.twin.forward") + 1:])
    except Stop:
        pass
    ns = got.pop("ns")
    out[arm] = [len(ns.theta), ns.gain, ns.update_method, ns.restart_taper_km, ns.ic_members,
                ns.column_heads, ns.apex_hold, ns.rf_timeseries, len(ns.scenario)]
print(json.dumps(out))
"""


def _has_commit(c: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{c}^{{commit}}"],
                          capture_output=True).returncode == 0


@pytest.mark.parametrize("code", P.CODES)
def test_ladder_parses_every_arm(code, tmp_path):
    """Every declared command parses with the forward CLI of each declared rung (the rung's
    hydrophysics package extracted with git archive into a tmp dir; each forward is stopped
    right after its argument parse, before any input is loaded)."""
    if not _has_commit(code):
        pytest.skip(f"commit {code} not in this clone")
    import shlex

    tar = tmp_path / "pkg.tar"
    subprocess.run(["git", "-C", str(REPO), "archive", "-o", str(tar), code, "hydrophysics"],
                   check=True)
    (tmp_path / "src").mkdir()
    subprocess.run(["tar", "-xf", str(tar), "-C", str(tmp_path / "src")], check=True)
    src = str(tmp_path / "src")
    cmds = {a: P.command(a) for a in S.ARMS}
    p = subprocess.run([sys.executable, "-P", "-c", _LADDER_PARSE, src, json.dumps(cmds)],
                       cwd=tmp_path, env={**os.environ, "PYTHONPATH": src},
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    got = json.loads(p.stdout.strip().splitlines()[-1])
    want = {"S0": [3, 1.0, "innovation", 5.0, 1], "S1": [3, 0.0, "innovation", 5.0, 1],
            "S2": [3, 1.0, "field", None, 1], "S3": [1, 1.0, "innovation", 5.0, 8],
            "S3ref": [1, 1.0, "innovation", 5.0, 0]}
    for a, w in want.items():
        assert got[a] == [*w, "free", "calibrated", P.RF, 4], (code, a, got[a])
        assert shlex.split(cmds[a]).count("--rf-timeseries") == 1


@pytest.mark.skipif(not (REPO / P.WELLS_DIR).exists() or not all(
    (REPO / p).exists() for p in P.FORWARD_INPUTS), reason="forward inputs not present")
def test_forward_inputs_unchanged_since_drafting():
    """The declared forward inputs (files and the well-directory digest) match the disk."""
    assert {p: P.sha256(REPO / p) for p in P.FORWARD_INPUTS} == P.FORWARD_INPUTS
    assert S.check_input_digests(P.doc()) == P.INPUT_DIGESTS


def test_runner_syntax():
    subprocess.run(["bash", "-n", str(X / "run_fs.sh")], check=True)


@pytest.mark.skipif(DECL.exists(), reason="declaration registered; refusal test not applicable")
@pytest.mark.parametrize("cmd", [["check"], ["forward", "S0"], ["repro"], ["score"]])
def test_runner_refuses_without_registration(cmd):
    p = subprocess.run(["bash", str(X / "run_fs.sh"), *cmd], cwd=REPO, capture_output=True,
                       text=True)
    assert p.returncode == 1 and "declare FS.start_invariance.v1 first" in p.stderr


def test_prereg_cli_validate_is_a_dry_run(capsys):
    reg = REPO / "results/dataprep/preregistration"
    before = sorted(reg.glob("FS.*")) if reg.exists() else []
    P.main(["--validate", "--quiet"])
    out = capsys.readouterr().out
    assert "validate_prereg: valid" in out and "recipe/binding checks: ok" in out
    assert (sorted(reg.glob("FS.*")) if reg.exists() else []) == before


@needs_results
def test_declaration_round_trips_in_tmp_registry(tmp_path):
    from harness import prereg3

    d = P.doc()
    path, h = prereg3.write_prereg(d, prereg_dir=tmp_path / "prereg",
                                   rows_dir=tmp_path / "rows", registries=[])
    got, sha = S.load_registered(prereg_dir=tmp_path / "prereg", rows_dir=tmp_path / "rows")
    assert sha == h and got["id"] == S.PREREG_ID
    assert S.check_frozen(got) == got["frozen_code"]
    assert S.check_input_digests(got) == P.INPUT_DIGESTS        # survives registration
    hashed = {r["path"]: r["sha256"] for r in got["inputs"] if isinstance(r, dict)}
    assert {p: hashed[p] for p in P.FORWARD_INPUTS} == P.FORWARD_INPUTS
    with pytest.raises(S.EngineeringError, match="frozen code differs"):
        S.check_frozen({**got, "frozen_code": {p: "0" * 64 for p in S.FROZEN}})


@pytest.mark.skipif(not REF_REAL.exists(), reason="page run not present")
def test_check_reference_on_the_page_run(capsys):
    """The scorer's N2 unit reproduces the page's published C01 numbers (+1.19 / +1.41)."""
    assert S.main(["--check-reference"]) == 0
    assert "(registered)" in capsys.readouterr().out


# ----------------------------------------------------------------------------- scoring
def test_synthetic_pass(world, registered, any_ref, tmp_path, capsys):
    _, f = world
    att = attempts(tmp_path / "att.jsonl", f)
    out = tmp_path / "rep.json"
    assert S.main(args(f, att, out)) == 0
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "pass"
    g = rep["table"]
    assert set(g) == set(S.GATING) and all(len(t) == 10 for t in g.values())
    r = g["S1"]["cut30@2026/2026-27"]
    assert abs(r["ratio"] - 1.02) < 1e-4 and abs(r["p90_rel_dev"] - 0.02) < 1e-4
    assert abs(g["S2"]["retire_aqua@2030/Dec-2032"]["ratio"] - 0.97) < 1e-4
    assert abs(rep["summary"]["max_abs_ratio_dev_pct"] - 3.0) < 1e-2
    rr = rep["reported_not_gating"]
    assert abs(rr["baseline_level_shift_m"]["S1"]["2026-12"]["n2_unit"] + 0.5) < 1e-4
    assert abs(rr["n2_unit_year_end_m"]["S0"]["cut30"][-1] - 1.19) < 1e-4
    assert abs(rr["S3_ensemble_mean_vs_S3ref"]["cut30@2026/2030-32"]["ratio"] - 1.01) < 1e-4
    mem = rr["S3_per_member_layer2"]["units"]["retire_aqua@2026/2030-32"]
    assert len(mem["ratios_ic1_8"]) == 8 and mem["share_in_band"] == 1.0
    assert rep["reproduction"]["ok"] and rep["registration"]["declaration_sha256"] == "f" * 64
    assert rep["summary"]["wording_case"] == "pass"
    assert rep["registration"]["input_digests"] == P.INPUT_DIGESTS
    assert "VERDICT: pass" in capsys.readouterr().out


def test_synthetic_fail_on_ratio(world, registered, any_ref, tmp_path, capsys):
    t, f = world
    s2 = fake_npz(tmp_path / "S2.npz", "S2", k=1.08)
    f2 = {**f, "S2": s2}
    out = tmp_path / "rep.json"
    assert S.main(args(f2, attempts(tmp_path / "a.jsonl", f2), out)) == 0
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "fail" and rep["summary"]["arms_failing"] == ["S2"]
    assert abs(rep["summary"]["max_abs_ratio_dev_pct"] - 8.0) < 1e-2
    assert rep["table"]["S2"]["cut30@2026/2026-27"]["ratio_in_report_band"] is True
    pa = rep["summary"]["per_arm"]
    assert rep["summary"]["wording_case"] == "S2_only" and pa["S1"]["pass"]
    assert abs(pa["S1"]["max_abs_ratio_dev_pct"] - 2.0) < 1e-2
    assert abs(pa["S2"]["max_abs_ratio_dev_pct"] - 8.0) < 1e-2
    assert pa["S2"]["ratio_band_fails"] == 10


def test_cancellation_in_the_mean_fails_co_primary_2(world, registered, any_ref, tmp_path):
    """Per-well start dependence that cancels in the 84-well mean is caught by the p90."""
    _, f = world
    cf = np.where(np.arange(S.N_CELLS) % 2 == 0, 1.2, 0.8).astype(np.float32)
    yc = np.asarray(S.CELLS)[S.masks()["yunlin"]]
    cf = cf * (1.0 / cf[yc].mean())                        # mean over the 84 wells exactly 1
    s1 = fake_npz(tmp_path / "S1.npz", "S1", cell_factor=cf)
    f2 = {**f, "S1": s1}
    out = tmp_path / "rep.json"
    assert S.main(args(f2, attempts(tmp_path / "a.jsonl", f2), out)) == 0
    rep = json.loads(out.read_text())
    r = rep["table"]["S1"]["cut30@2026/2030-32"]
    assert r["ratio_in_band"] and not r["p90_ok"]
    assert rep["verdict"] == "fail" and rep["summary"]["wording_case"] == "S1_only"
    assert rep["summary"]["per_arm"]["S1"]["ratio_band_fails"] == 0


def test_not_evaluable_when_the_lever_vanishes(world, registered, any_ref, tmp_path):
    _, f = world
    tiny = {**LEVER}
    try:
        for k in LEVER:
            LEVER[k] = 0.01
        s0 = fake_npz(tmp_path / "S0.npz", "S0")
        s1 = fake_npz(tmp_path / "S1.npz", "S1")
        s2 = fake_npz(tmp_path / "S2.npz", "S2")
    finally:
        LEVER.update(tiny)
    arms = {"S0": S._load(s0, "S0"), "S1": S._load(s1, "S1"), "S2": S._load(s2, "S2")}
    assert S.gated(arms)["verdict"] == "not_evaluable"


def test_s0_not_reproducing_is_engineering(world, registered, any_ref, tmp_path, capsys):
    _, f = world
    s0 = fake_npz(tmp_path / "S0.npz", "S0", k=1.02)            # cut30 off by 0.024 m
    f2 = {**f, "S0": s0}
    out = tmp_path / "rep.json"
    assert S.main(args(f2, attempts(tmp_path / "a.jsonl", f2), out)) == 3
    assert not out.exists()
    cap = capsys.readouterr()
    assert "VERDICT" not in cap.out and "S1 " not in cap.out
    # the repro subcommand gives the same exit, reading S0 only
    assert S.main(args(f2, attempts(tmp_path / "b.jsonl", {"S0": s0}, repro=False))
                  + ["--repro"]) == 3


def test_repro_mode_reads_s0_only(world, registered, any_ref, tmp_path, capsys):
    _, f = world
    att = attempts(tmp_path / "a.jsonl", {"S0": f["S0"]}, repro=False)
    missing = tmp_path / "nope.npz"
    assert S.main(args(f, att, S1=missing, S2=missing, S3=missing) + ["--repro"]) == 0
    assert "repro: ok" in capsys.readouterr().out


def test_wrong_scenario_string_is_engineering(world, registered, any_ref, tmp_path):
    _, f = world
    s2 = fake_npz(tmp_path / "S2.npz", "S2", desc_fix="retire_aqua_2030: aquaculture x0")
    f2 = {**f, "S2": s2}
    out = tmp_path / "rep.json"
    assert S.main(args(f2, attempts(tmp_path / "a.jsonl", f2), out)) == 2
    assert not out.exists()


def test_wrong_gain_is_engineering(world, registered, any_ref, tmp_path):
    _, f = world
    s1 = fake_npz(tmp_path / "S1.npz", "S1", gain=1.0)            # S1 must be gain 0
    f2 = {**f, "S1": s1}
    assert S.main(args(f2, attempts(tmp_path / "a.jsonl", f2), tmp_path / "r.json")) == 2


def test_attempt_log_binds_files_and_code(world, registered, any_ref, tmp_path):
    _, f = world
    out = tmp_path / "rep.json"
    wrong = attempts(tmp_path / "w.jsonl", {**f, "S2": f["S1"]})        # S2 sha of S1
    assert S.main(args(f, wrong, out)) == 1
    failed = attempts(tmp_path / "f.jsonl", f, rc=1)
    assert S.main(args(f, failed, out)) == 1
    mixed = attempts(tmp_path / "m.jsonl", f, codes={"S2": "730f65b"})
    assert S.main(args(f, mixed, out)) == 1
    norepro = attempts(tmp_path / "n.jsonl", f, repro=False)
    assert S.main(args(f, norepro, out)) == 1
    assert not out.exists()


def test_frozen_code_mismatch_refused(world, any_ref, monkeypatch, tmp_path):
    _, f = world
    monkeypatch.setattr(S, "load_registered", lambda *a, **k: (
        {"frozen_code": {p: "0" * 64 for p in S.FROZEN}}, "f" * 64))
    out = tmp_path / "rep.json"
    assert S.main(args(f, attempts(tmp_path / "a.jsonl", f), out)) == 1
    assert not out.exists()


def test_input_drift_refused_before_any_npz(world, registered, any_ref, monkeypatch,
                                            tmp_path):
    """A changed head-well directory refuses at step 0 (the runner's pre-forward check)."""
    _, f = world
    monkeypatch.setattr(S, "dir_digest", lambda path, pattern: (344, "0" * 64))
    out = tmp_path / "rep.json"
    assert S.main(args(f, attempts(tmp_path / "a.jsonl", f), out)) == 1
    assert S.main(["--verify-registration"]) == 1
    assert not out.exists()
    monkeypatch.setattr(S, "load_registered", lambda *a, **k: (
        {"frozen_code": {p: S.sha256(REPO / p) for p in S.FROZEN}}, "f" * 64))
    assert S.main(["--verify-registration"]) == 1          # no input_digests: refused


def test_scored_once(world, registered, any_ref, tmp_path):
    _, f = world
    out = tmp_path / "rep.json"
    att = attempts(tmp_path / "a.jsonl", f)
    assert S.main(args(f, att, out)) == 0
    before = out.read_bytes()
    assert S.main(args(f, att, out)) == 1                    # refused at step 0
    assert out.read_bytes() == before


def test_real_registration_absent_is_engineering(world, tmp_path):
    if DECL.exists():
        pytest.skip("declaration registered")
    _, f = world
    assert S.main(args(f, attempts(tmp_path / "a.jsonl", f), tmp_path / "r.json")) == 1
