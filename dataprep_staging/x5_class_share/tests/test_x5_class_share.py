"""CPU tests of the X5.lever_class_share.v1 scripts (no forward run, no GPU, no write outside
pytest's tmp dirs, nothing registered, no look logged). They read the gitignored results/
tree (the batch-27 forward files datum_gate_{q0c,x5r1}.npz: model output, baseline and cut30
only), so they live here, outside the repository's testpaths:

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/x5_class_share/tests

Step 3 of the scorer is exercised on SYNTHETIC three-scenario copies of the batch-27 files,
whose fake 'all30' is a known multiple of the cut30 lever (no real all30 exists).
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
import prereg_class_share as P  # noqa: E402
import score_class_share as S  # noqa: E402

REF = {a: REPO / f"results/twin_forward/datum_gate_{a}.npz" for a in ("q0c", "x5r1")}
DECL = REPO / f"results/dataprep/preregistration/{S.PREREG_ID}.json"
needs_ref = pytest.mark.skipif(not all(p.exists() for p in REF.values()),
                               reason="batch-27 forward files not present")


# ----------------------------------------------------------------------------- static
def test_wells_frozen_matches_scorer():
    w = json.loads((X / "wells_frozen.json").read_text())
    assert (w["n"], w["n_changhua"], w["n_yunlin"]) == (158, 74, 84)
    assert w["rows_sha256"] == S.ROWS_SHA256 == S._rows_sha()
    rows = [(r["sid"], r["cell"], r["layer"], r["county"][0].upper()) for r in w["rows"]]
    assert rows == list(zip(S.SIDS, S.CELLS, S.LAYERS, "".join(S.COUNTY), strict=True))


def test_no_home_paths_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")     # a home directory with a user name
    for p in sorted(X.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh", ".json"):
            assert not pat.search(p.read_text()), p


def test_declaration_validates_and_binds():
    from harness import prereg3

    d = P.doc()
    assert prereg3.validate_prereg(d) == []
    assert P.runner_problems() == []
    assert set(d["frozen_code"]) == set(S.FROZEN) == set(P.FROZEN)
    assert d["frozen_code"] == {p: S.sha256(REPO / p) for p in S.FROZEN}
    declared = {r["path"]: r["sha256"] for r in d["inputs"] if isinstance(r, dict)}
    assert declared == d["frozen_code"]
    assert (S.BAND, S.STRICT_BAND) == (P.BAND, P.STRICT_BAND)
    m = d["evidence_and_ledger_mapping"]
    assert "adds no page lever" in m["scope"]
    assert "never compounded with R" in m["irrigation_only_lever_all_outcomes"]
    assert not any("conversion-invariant" in s and "never" not in s for s in m["on_pass"])
    assert "LOW-RISK" in d["priors"] and "[0.95, 1.05]" in d["priors"]
    assert "SEALED" in d["attempt_policy"]["values_written_before_scoring"]


def test_runner_dry_run_matches_declared_command():
    out = subprocess.run(["bash", str(X / "run_class_share.sh"), "forward"], cwd=REPO,
                         env={**os.environ, "DRY": "1"}, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert out == [P.command(a).split(" > ")[0] for a in ("q0c", "x5r1")]


def test_runner_syntax():
    subprocess.run(["bash", "-n", str(X / "run_class_share.sh")], check=True)


@pytest.mark.skipif(DECL.exists(), reason="declaration registered; refusal test not applicable")
@pytest.mark.parametrize("cmd", ["check", "forward", "score"])
def test_runner_refuses_without_registration(cmd):
    p = subprocess.run(["bash", str(X / "run_class_share.sh"), cmd], cwd=REPO,
                       capture_output=True, text=True)
    assert p.returncode == 1 and "declare X5.lever_class_share.v1 first" in p.stderr


# ----------------------------------------------------------------------------- plumbing
@needs_ref
def test_check_shapes_on_batch27(capsys):
    rc = S.main(["--check-shapes", "--q0c", str(REF["q0c"]), "--x5r1", str(REF["x5r1"])])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "all30 not present and not computed" in out


# ----------------------------------------------------------------------------- synthetic
def fake(src: Path, dst: Path, k: float, desc: str = S.SCEN_DESC[2],
         cut_bump: float = 0.0) -> None:
    f = np.load(src, allow_pickle=False)
    d = {n: f[n] for n in f.files}
    hm, sm = d["heads_mean"], d["subs_mean"]
    lev = hm[1] - hm[0]
    hm2 = np.concatenate([hm, (hm[0] + k * lev)[None]], 0)
    hm2[1] = hm2[1] + cut_bump
    sm2 = np.concatenate([sm, (sm[0] + k * (sm[1] - sm[0]))[None]], 0)
    d.update(heads_mean=hm2, heads_std=np.concatenate([d["heads_std"], d["heads_std"][:1]], 0),
             subs_mean=sm2, subs_std=np.concatenate([d["subs_std"], d["subs_std"][:1]], 0),
             scenario_names=np.array(list(S.SCEN_NAMES)),
             scenarios=np.array(list(d["scenarios"]) + [desc]))
    np.savez(dst, **d)


def attempts(path: Path, files: dict[str, Path], rc: int = 0) -> Path:
    with path.open("w") as f:
        for a, p in files.items():
            f.write(json.dumps({"event": "forward", "attempt": 1, "arm": a, "rc": rc,
                                "npz_sha256": S.sha256(p), "worktree_head": "x"}) + "\n")
    return path


@pytest.fixture
def registered(monkeypatch):
    """Pretend the declaration is registered with the current frozen code (no registry)."""
    def fake_load(prereg_dir=None, rows_dir=None):
        return {"frozen_code": {p: S.sha256(REPO / p) for p in S.FROZEN}}, "f" * 64
    monkeypatch.setattr(S, "load_registered", fake_load)


@pytest.fixture
def fakes(tmp_path):
    if not all(p.exists() for p in REF.values()):
        pytest.skip("batch-27 forward files not present")
    q, x = tmp_path / "q.npz", tmp_path / "x.npz"
    fake(REF["q0c"], q, 1.80)
    fake(REF["x5r1"], x, 1.80 / 1.3893)        # Changhua all30 ratio ~1.00 by construction
    return tmp_path, q, x


def run(t: Path, q: Path, x: Path, out: Path | None = None, att: Path | None = None):
    att = att or attempts(t / "att.jsonl", {"q0c": q, "x5r1": x})
    return S.main(["--q0c", str(q), "--x5r1", str(x), "--attempts", str(att),
                   "--out", str(out or t / "rep.json")])


def test_synthetic_pass(fakes, registered, capsys):
    t, q, x = fakes
    assert run(t, q, x) == 0
    rep = json.loads((t / "rep.json").read_text())
    g = rep["gated"]
    assert abs(g["changhua/2026-27"]["ratio"] - 1.0) < 1e-3
    assert abs(g["yunlin/2026-27"]["ratio"] - 1.2730 / 1.3893) < 1e-3      # 0.916
    assert rep["verdict"] == "pass"
    r = rep["reported_not_gating"]
    assert r["strict_band_report_only"]["all_four_inside"] is False        # Yunlin 0.916
    assert g["changhua/2026-27"]["inside_strict_band_report_only"] is True
    assert r["residual_factor_by_county"]["yunlin"][1] == 1.0
    assert abs(r["cut30_over_all30"]["q0c"]["changhua"]["2026-27"] - 1 / 1.8) < 1e-3
    assert rep["registration"]["declaration_sha256"] == "f" * 64
    assert rep["reproduction"]["ok"] and all(
        v["max_abs_diff_m"] == 0.0 for v in rep["reproduction"]["vs_batch27"].values())
    assert "VERDICT: pass" in capsys.readouterr().out


def test_synthetic_fail(fakes, registered, capsys):
    t, q, _ = fakes
    x2 = t / "x2.npz"
    fake(REF["x5r1"], x2, 1.80 / 1.20)
    assert run(t, q, x2) == 0
    assert "VERDICT: fail" in capsys.readouterr().out


def test_wrong_scenario_string_is_engineering(fakes, registered, capsys):
    t, q, _ = fakes
    x3 = t / "x3.npz"
    fake(REF["x5r1"], x3, 1.0, desc="all30: irrigation x0.7")
    assert run(t, q, x3) == 2
    assert "VERDICT" not in capsys.readouterr().out
    assert not (t / "rep.json").exists()


def test_perturbed_cut30_is_reproduction_failure(fakes, registered, capsys):
    t, q, _ = fakes
    x4 = t / "x4.npz"
    fake(REF["x5r1"], x4, 1.0, cut_bump=0.2)
    assert run(t, q, x4) == 3
    cap = capsys.readouterr()
    assert "all30 " not in (cap.out + cap.err).replace("all30 not read", "")
    assert not (t / "rep.json").exists()


def test_attempt_log_binds_files(fakes, registered):
    t, q, x = fakes
    other = attempts(t / "other.jsonl", {"q0c": q, "x5r1": q})          # wrong x5r1 sha
    assert run(t, q, x, att=other) == 1
    failed = attempts(t / "failed.jsonl", {"q0c": q, "x5r1": x}, rc=1)
    assert run(t, q, x, att=failed) == 1
    assert not (t / "rep.json").exists()


def test_frozen_code_mismatch_refused(fakes, monkeypatch):
    t, q, x = fakes
    monkeypatch.setattr(S, "load_registered", lambda *a, **k: (
        {"frozen_code": {p: "0" * 64 for p in S.FROZEN}}, "f" * 64))
    assert run(t, q, x) == 1
    assert not (t / "rep.json").exists()


def test_scored_once(fakes, registered):
    t, q, x = fakes
    assert run(t, q, x) == 0
    before = (t / "rep.json").read_bytes()
    assert run(t, q, x) == 1                    # refused at step 0: the report exists
    assert (t / "rep.json").read_bytes() == before


def test_real_registration_absent_is_engineering(fakes):
    if DECL.exists():
        pytest.skip("declaration registered")
    t, q, x = fakes
    assert run(t, q, x) == 1
