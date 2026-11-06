"""The all30 scenario-forward runner (dataprep_staging/all30/run_all30.py; ruling R8.9), on a
SYNTHETIC repository tree: fake inputs, a fake bound forward run and a fake ``twin.forward`` that
writes known outputs. No GPU, no real file is read or written (every path is under tmp_path).

The refusals that matter: nothing runs unless the bound forward run is byte-identical to its
sha256 (cf2c7154... for the real file), a run that changes it fails (exit 3), and outputs are
never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from dataprep_staging.all30 import run_all30 as R

T, A, L = 252, 6, 4                     # Jan 2012 .. Dec 2032; a 6-cell, 4-layer grid
CELLS, LAYERS = np.array([0, 2, 5]), np.array([1, 1, 2])


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _dates():
    return np.array([f"{2012 + m // 12}-{m % 12 + 1:02d}-01T00:00:00" for m in range(T)],
                    dtype="U19")


def _ramp(start):
    t = np.arange(T, dtype="float64")
    return np.clip(t - start, 0, None) / 12.0


def _heads(levers: dict[str, float]) -> np.ndarray:
    base = np.full((L, A, T), 10.0) - 0.1 * _ramp(0)[None, None]
    out = [base]
    for amp in levers.values():
        out.append(base + amp * np.minimum(_ramp(168), 1.0)[None, None])
    return np.stack(out).astype("float32")


def _common(names, desc, heads):
    fo = {"column_heads": "free", "update_method": "innovation", "save_members": "yearly"}
    return dict(dates=_dates(), origin=np.int64(131), mask=np.ones((2, 3), dtype=bool),
                nx=np.int64(3), ny=np.int64(2), dx=np.float64(1000.0), x0=np.float64(0.0),
                y0=np.float64(0.0), n_members=np.int64(36),
                member_labels=np.array(["stage3_datum_gate", "fold0"]),
                rheology_labels=np.array(["coupled_leveling", "tau30y"]),
                horizon=np.int64(120), gain=np.float64(1.0), hindcast_gain=np.float64(0.0),
                hindcast_every=np.int64(12), gate=np.array(json.dumps({"git_commit": "fa5cca2"})),
                forward_options=np.array(json.dumps(fo)), scenario_names=np.array(names),
                scenarios=np.array(desc), heads_mean=heads,
                subs_mean=np.zeros((len(names), A, T), dtype="float32"))


REF_DESC = ["baseline: baseline (no change)", "cut30: irrigation x0.7 from 2026-01",
            "retire_aqua: aquaculture x0 from 2026-01"]
LEVERS = {"cut30": 1.19, "retire_aqua": 1.41}


def _write_ref(path: Path):
    np.savez(path, **_common(["baseline", "cut30", "retire_aqua"], REF_DESC, _heads(LEVERS)))


def _write_out(stem: Path, all30=2.05, perturb=0.0):
    h = _heads({**LEVERS, "all30": all30})
    h[0] += perturb
    np.savez(f"{stem}.npz", **_common(list(R.NAMES), [*REF_DESC, R.ALL30_DESC], h))
    for suf in (".summary.csv", ".members.csv"):
        Path(f"{stem}{suf}").write_text("scenario\nbaseline\n")
    np.savez(f"{stem}.members.npz", x=np.zeros(1))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repository tree with every declared input; the bound forward run is synthetic and
    its sha256 is patched in as the bound one."""
    root = _Root(tmp_path / "repo")
    for p in (*R.THETAS, *R.VEPS, R.RF, *R.DATA_FILES):
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(f"fake {p}\n")
    for d, _ in R.DATA_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / "w1.parquet").write_bytes(b"fake")
    (root / R.REF).parent.mkdir(parents=True, exist_ok=True)
    _write_ref(root / R.REF)
    monkeypatch.setattr(R, "REF_SHA256", _sha(root / R.REF))
    monkeypatch.setattr(R, "load_screens", lambda repo=None: (
        CELLS, LAYERS, {"source": "synthetic", "screens_sha256": "0" * 64, "n_yunlin": 3}))
    (root / R.FS_SCORER).parent.mkdir(parents=True, exist_ok=True)
    (root / R.FS_SCORER).write_text("# frozen screens (fake)\n")
    _write_fs_decl(root)
    rung = {"rung": "main"}
    monkeypatch.setattr(R, "check_code", lambda code, python: {
        "dir": "repo", "commit": ("4c2723e" if rung["rung"] == "4c2723e" else "f") + "0" * 33,
        "hydrophysics_clean": True, "forward_py_sha256": None, "rung": rung["rung"]})
    monkeypatch.setattr(R, "gpu_free_mib", lambda: 20000)
    root.rung = rung                      # tests switch the code's rung here
    return root


class _Root(type(Path())):
    """A Path that can carry the fixture's rung switch."""


def _write_fs_decl(root: Path, **override):
    """FS.start_invariance.v1's declaration in the fake tree: frozen scorer and input hashes."""
    ins = R.input_hashes(root)["sha256"]
    decl = {"frozen_code": {R.FS_SCORER: _sha(root / R.FS_SCORER)},
            "inputs": [{"path": k, "sha256": v} for k, v in ins.items() if isinstance(v, str)],
            "input_digests": [{"path": d, "glob": g, **ins[f"{d}/{g}"]} for d, g in R.DATA_DIRS]}
    decl.update(override)
    (root / R.FS_DECL).parent.mkdir(parents=True, exist_ok=True)
    (root / R.FS_DECL).write_text(json.dumps(decl))


def _fake_forward(tmp_path: Path, repo: Path, *, touch_ref=False, perturb=0.0, rc=0,
                  device="cuda") -> list[str]:
    """argv of a stand-in for twin.forward: prints its device line and copies outputs staged
    here into the run's output stem (and, with ``touch_ref``, appends a byte to the bound
    file)."""
    staged = tmp_path / "staged"
    staged.mkdir(exist_ok=True)
    _write_out(staged / "out", perturb=perturb)
    script = tmp_path / "fake_forward.py"
    dst = repo / R.OUT_STEM
    script.write_text(
        "import shutil, sys\n"
        f"print('device: {device}')\n"
        f"for suf in ('.npz', '.summary.csv', '.members.csv', '.members.npz'):\n"
        f"    shutil.copy({str(staged / 'out')!r} + suf, {str(dst)!r} + suf)\n"
        + (f"open({str(repo / R.REF)!r}, 'ab').write(b'x')\n" if touch_ref == "append" or
           touch_ref is True else "")
        + (f"import os; os.remove({str(repo / R.REF)!r})\n" if touch_ref == "remove" else "")
        + f"sys.exit({rc})\n")
    return [sys.executable, str(script)]


def test_the_command_is_the_page_recipe_with_all30_into_a_new_file():
    argv = R.forward_argv("PY")
    s = " ".join(argv)
    assert argv[:4] == ["PY", "-P", "-m", "hydrophysics.twin.forward"]
    assert argv.count("--theta") == 3 and "stage3_posterior_tn.json" in s
    assert argv.count("--scenario") == 3 and R.ALL30 in argv
    assert "cut30:irrigation=0.7@2026-01" in argv and "retire_aqua:aquaculture=0@2026-01" in argv
    for flag in ("--horizon 120", "--gain 1.0", "--update-method innovation",
                 "--restart-taper-km 5", "--apex-hold calibrated", "--column-heads free",
                 "--column-hpc0-fast-days 365", "--save-members yearly", "--device cuda",
                 "--rf-timeseries " + R.RF):
        assert flag in s, flag
    assert argv[argv.index("--out") + 1] == R.OUT_STEM != R.REF[:-4]
    assert "--data" not in argv                           # the forward's own default
    assert R.ALL30 == ("all30:irrigation=0.7,aquaculture=0.7,livestock=0.7,domestic=0.7,"
                       "industry=0.7,other=0.7@2026-01")


def test_the_bound_forward_run_must_be_byte_identical(repo):
    assert R.check_reference(repo) == R.REF_SHA256
    with open(repo / R.REF, "ab") as f:                       # one byte changed
        f.write(b"\0")
    with pytest.raises(R.Refused, match="changed"):
        R.check_reference(repo)
    with pytest.raises(R.Refused, match="changed"):
        R.preconditions(repo)
    (repo / R.REF).unlink()
    with pytest.raises(R.Refused, match="missing"):
        R.check_reference(repo)


def test_the_real_sha_is_cf2c7154_and_refuses_any_other_file(tmp_path):
    assert R.REF_SHA256.startswith("cf2c7154") and len(R.REF_SHA256) == 64
    (tmp_path / R.REF).parent.mkdir(parents=True)
    (tmp_path / R.REF).write_bytes(b"not the page run")
    with pytest.raises(R.Refused, match="cf2c7154"):
        R.check_reference(tmp_path)


def test_the_output_stem_cannot_be_the_bound_file():
    with pytest.raises(R.Refused, match="bound forward"):
        R.check_stem("results/twin_forward/datum_gate_innov")
    R.check_stem(R.OUT_STEM)


def test_dry_run_prints_the_command_and_writes_nothing(repo, capsys):
    before = sorted(p for p in repo.rglob("*"))
    assert R.cmd_run(repo, R.OUT_STEM, "PY", repo, dry=True) == 0
    out = capsys.readouterr().out
    assert R.ALL30 in out and "--out " + R.OUT_STEM in out
    assert sorted(p for p in repo.rglob("*")) == before


def test_a_run_writes_its_sidecar_and_reproduces_the_page_levers(repo, tmp_path, monkeypatch,
                                                                 capsys):
    monkeypatch.setattr(R, "forward_argv", lambda py, stem=R.OUT_STEM: _fake_forward(tmp_path,
                                                                                     repo))
    assert R.cmd_run(repo, R.OUT_STEM, "PY", repo, dry=False) == 0
    side = json.loads((repo / R.sidecar_path()).read_text())
    assert side["status"] == "ok" and side["returncode"] == 0
    assert side["reference"]["byte_identical"] and side["reference"]["sha256_before"] == \
        side["reference"]["sha256_after"] == R.REF_SHA256
    assert side["layout"]["ok"] and side["reproduction"]["ok"]
    assert side["reproduction"]["scenarios"]["cut30"]["heads_max_abs_diff_m"] == 0.0
    assert side["outputs_sha256"]["npz"] == _sha(repo / f"{R.OUT_STEM}.npz")
    assert set(side["inputs_sha256"]) >= {R.REF, *R.THETAS, *R.VEPS, R.RF}
    assert side["command"][0] == "$PYTHON" and side["model_only"] is True
    assert side["registration"] is None and "R8.9" in side["ruling"]
    u = side["n2_unit"]
    assert u["dec2032_m"]["all30"] == pytest.approx(2.05, abs=1e-6)
    assert u["published_check"]["ok"] and len(u["all30_series_m"]) == 21
    assert "all30 +2.050 m" in capsys.readouterr().out
    # a second run refuses before anything runs (it reproduced; and outputs are never
    # overwritten)
    with pytest.raises(R.Refused, match="reproduced; no rerun"):
        R.cmd_run(repo, R.OUT_STEM, "PY", repo, dry=False)
    assert R.cmd_verify(repo, R.OUT_STEM) == 0


def test_a_run_that_changes_the_bound_file_fails(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "forward_argv", lambda py, stem=R.OUT_STEM: _fake_forward(
        tmp_path, repo, touch_ref=True))
    assert R.cmd_run(repo, R.OUT_STEM, "PY", repo, dry=False) == 3
    side = json.loads((repo / R.sidecar_path()).read_text())
    assert side["status"] == "reference_changed" and not side["reference"]["byte_identical"]
    assert "reproduction" not in side                       # nothing is read past the change


def test_outputs_that_do_not_reproduce_the_page_run_fail(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "forward_argv", lambda py, stem=R.OUT_STEM: _fake_forward(
        tmp_path, repo, perturb=0.02))
    assert R.cmd_run(repo, R.OUT_STEM, "PY", repo, dry=False) == 3
    side = json.loads((repo / R.sidecar_path()).read_text())
    assert side["status"] == "does_not_reproduce" and not side["reproduction"]["ok"]
    assert side["reproduction"]["scenarios"]["baseline"]["heads_max_abs_diff_m"] == \
        pytest.approx(0.02, abs=1e-5)


def test_a_forward_off_the_gpu_or_failing_is_recorded_as_failed(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "forward_argv", lambda py, stem=R.OUT_STEM: _fake_forward(
        tmp_path, repo, device="cpu"))
    assert R.cmd_run(repo, R.OUT_STEM, "PY", repo, dry=False) == 2
    assert json.loads((repo / R.sidecar_path()).read_text())["status"] == "forward_failed"


def test_layout_check_names_what_differs(repo, tmp_path):
    ref = R._load(repo / R.REF)
    stem = tmp_path / "x"
    _write_out(stem)
    out = R._load(Path(f"{stem}.npz"))
    assert R.layout_check(out, ref)["ok"]
    out["dates"] = out["dates"][::-1].copy()
    out["scenarios"] = np.array([*REF_DESC, "all30: irrigation x0.7 from 2026-01"])
    bad = R.layout_check(out, ref)
    assert not bad["ok"] and "dates" in bad["mismatch"]
    assert any(str(m).startswith("scenarios") for m in bad["mismatch"])


def test_the_basis_cannot_give_all30_exactly_on_the_real_files():
    """On the real files (skipped without them): the response basis is one parameter set and
    misses its own irrigation x0.5 check, so all30 needs the forward (docs, section 24)."""
    note = R.superposition_note()
    if note is None:
        pytest.skip("results/ (page run, basis) not present")
    assert note["exact"] is False and note["basis_sets"] == 1 and note["forward_sets"] == 18
    assert note["nonlinearity_irr50"]["rel"] > 0.05


# ---- the declared attempt ladder ---------------------------------------------------------------
def _rows(repo):
    return R.read_attempts(repo)


def _run(repo, tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(R, "forward_argv", lambda py, stem=R.OUT_STEM: _fake_forward(
        tmp_path, repo, **kw))
    return R.cmd_run(repo, R.OUT_STEM, "PY", repo, dry=False)


def test_every_attempt_is_logged_and_an_ok_run_is_never_rerun(repo, tmp_path, monkeypatch):
    assert _run(repo, tmp_path, monkeypatch) == 0
    rows = _rows(repo)
    assert [r["event"] for r in rows] == ["start", "end"]
    assert rows[1]["status"] == "ok" and rows[1]["rung"] == "main" and rows[1]["rc"] == 0
    assert rows[1]["outputs_sha256"]["npz"] == _sha(repo / f"{R.OUT_STEM}.npz")
    with pytest.raises(R.Refused, match="not archived"):
        R.cmd_archive(repo, R.OUT_STEM, "try again")
    with pytest.raises(R.Refused, match="no rerun"):
        R.next_rung(_rows(repo))


def test_main_then_one_fallback_then_the_ladder_is_used_up(repo, tmp_path, monkeypatch):
    assert _run(repo, tmp_path, monkeypatch, perturb=0.02) == 3              # main spent
    assert R.next_rung(_rows(repo)) == "4c2723e"
    with pytest.raises(R.Refused, match="never overwritten"):
        _run(repo, tmp_path, monkeypatch)                                    # archive first
    R.cmd_archive(repo, R.OUT_STEM, "main did not reproduce")
    with pytest.raises(R.Refused, match="next rung is 4c2723e"):
        _run(repo, tmp_path, monkeypatch)                                    # main again: no
    repo.rung["rung"] = "4c2723e"
    assert _run(repo, tmp_path, monkeypatch, perturb=0.02) == 3
    side = json.loads((repo / R.sidecar_path()).read_text())
    assert side["rung"] == "4c2723e" and [a["rung"] for a in side["prior_attempts"]] == ["main"]
    R.cmd_archive(repo, R.OUT_STEM, "fallback did not reproduce")
    with pytest.raises(R.Refused, match="used up"):
        R.cmd_run(repo, R.OUT_STEM, "PY", repo, dry=True)


def test_engineering_failures_and_killed_runs_count_and_may_retry_after_archive(
        repo, tmp_path, monkeypatch):
    assert _run(repo, tmp_path, monkeypatch, rc=1) == 2
    assert _rows(repo)[-1]["status"] == "forward_failed"
    R.cmd_archive(repo, R.OUT_STEM, "forward crashed")
    # a killed run leaves a start row only: it counts as an attempt, killed_or_partial
    R.log_attempt(repo, {"event": "start", "attempt": 2, "rung": "main", "time": "t",
                         "code_commit": "f" * 40})
    summ = R.attempts_summary(_rows(repo))
    assert [a["status"] for a in summ] == ["forward_failed", "killed_or_partial"]
    assert R.next_rung(_rows(repo)) == "main"                    # engineering: same rung
    assert _run(repo, tmp_path, monkeypatch) == 0
    side = json.loads((repo / R.sidecar_path()).read_text())
    assert side["attempt"] == 3 and len(side["prior_attempts"]) == 2
    assert [r["event"] for r in _rows(repo)].count("archive") == 1


def test_a_removed_bound_file_after_the_run_is_recorded_not_a_crash(repo, tmp_path,
                                                                    monkeypatch):
    assert _run(repo, tmp_path, monkeypatch, touch_ref="remove") == 3
    side = json.loads((repo / R.sidecar_path()).read_text())
    assert side["status"] == "reference_changed" and side["reference"]["sha256_after"] is None
    assert _rows(repo)[-1]["status"] == "reference_changed"
    with pytest.raises(R.Refused):                       # nothing more runs
        R.next_rung(_rows(repo))


def test_preflight_refuses_a_changed_scorer_or_a_changed_declared_input(repo):
    R.preconditions(repo)
    (repo / R.FS_SCORER).write_text("# edited\n")
    with pytest.raises(R.Refused, match="frozen"):
        R.preconditions(repo)
    (repo / R.FS_SCORER).write_text("# frozen screens (fake)\n")
    (repo / R.THETAS[0]).write_text("another theta\n")
    with pytest.raises(R.Refused, match="inputs changed"):
        R.preconditions(repo)
    (repo / R.THETAS[0]).write_text(f"fake {R.THETAS[0]}\n")
    (repo / R.DATA_DIRS[0][0] / "w2.parquet").write_bytes(b"new well")
    with pytest.raises(R.Refused, match="inputs changed"):
        R.preconditions(repo)


def test_code_check_refuses_untracked_python_and_off_ladder_commits(tmp_path, monkeypatch):
    import subprocess

    code = tmp_path / "code"
    (code / "hydrophysics" / "twin").mkdir(parents=True)
    (code / "hydrophysics" / "twin" / "forward.py").write_text("x = 1\n")
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t",
                                                "commit", "-qm", "c"]):
        subprocess.run(["git", "-C", str(code), *cmd], check=True)
    info = R.code_info(code)
    assert info["hydrophysics_clean"] and info["rung"] in (None, "main")
    (code / "hydrophysics" / "twin" / "stray.py").write_text("y = 2\n")
    info = R.code_info(code)
    assert not info["hydrophysics_clean"] and info["untracked_py"]
    with pytest.raises(R.Refused, match="untracked"):
        R.check_code(code, sys.executable)
    assert R.rung_of("4c2723e" + "0" * 33, "a" * 40) == "4c2723e"
    assert R.rung_of("a" * 40, "a" * 40) == "main" and R.rung_of("b" * 40, "a" * 40) is None
