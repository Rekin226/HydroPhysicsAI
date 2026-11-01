"""CPU tests of SD.cut_response_spread.v2 (sd2lib, score_sd2, prereg_sd2, run_sd2.sh) and of
the X5.gnss_seasonal_column.v1 repoint to it. No forward runs, no GPU, nothing is registered
in the real registry and nothing is written outside pytest's tmp dirs. The forward npz and
column files here are SYNTHETIC; the tests marked needs_results read the real column JSONs,
v1's declaration, v1's attempts.jsonl (hashes and times) and the forward npz rheology_labels
and key list only (no sealed log, sd_* CSV, subsidence or head array), and skip when the
gitignored results/ tree is absent.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/sd_column/tests
"""
from __future__ import annotations

import json
import os
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

import gsc_model_side as M  # noqa: E402
import score_sd as S  # noqa: E402
import score_sd2 as S2  # noqa: E402
import sd2lib as L2  # noqa: E402
import sdlib as L  # noqa: E402
from common.paths import REPO as DATA  # noqa: E402  (DATAPREP_REPO may point elsewhere)

HPY = Path.home() / "miniforge3/envs/hydro/bin/python"
TAU30 = {"page_tau30", "cwa_tau30"}
FAILED_V1 = ("page", "cwa", "q0nh", "ccz", "q0c", "x5r1cs")
#: v1's registered inputs that are code v2 imports or reuses (prereg_sd.py, v1's tests)
V1_INPUT_CODE = {
    f"{L.X}/prereg_sd.py": "e1d187abe645ba3b94a3551b473acf13f5920d91be19afe6e4b3b4389bcd35cc",
    f"{L.X}/tests/test_sd_column.py":
        "fae544c2feab28713f5721b46c14082bbd96d181cacf97e34fc34a35bc272518",
}

needs_results = pytest.mark.skipif(
    not (DATA / L.column_path(L.PAGE_COLUMN)).exists() or not (DATA / L2.V1_DECL).exists(),
    reason="results/ (column JSONs, v1's declaration) not present in this checkout")
needs_forwards = pytest.mark.skipif(
    not all((DATA / f"{L.out_stem(k)}.npz").exists() for k in L.FORWARDS)
    or not (DATA / L2.V1_ATTEMPTS).exists(),
    reason="v1's forward npz files / attempts.jsonl not present in this checkout")


def col_params(c: str) -> dict:
    return {"tau_max_years": 30.0 if c in TAU30 else None, "zonal": []}


def tmp_columns(root: Path, cols=None, params=None) -> Path:
    """Synthetic column JSONs at the declared paths (only tau_max_years matters here)."""
    for c in cols or L.COLUMNS:
        p = root / L.column_path(c)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps((params or {}).get(c, col_params(c))))
    return root


def by_rheology(k: str) -> bool:
    return len(L.FORWARDS[k]["columns"]) > 1


# ----------------------------------------------------------------------------- the rule
def test_rule_reproduces_every_observed_label_set():
    for k, f in L.FORWARDS.items():
        paths = L2.vep_paths(L.forward_command(k))
        assert paths == [L.column_path(c) for c in f["columns"]]
        got = L2.forward_labels(paths, [col_params(c) for c in f["columns"]])
        assert got == L2.V1_OBSERVED_LABELS[k], k


def test_rule_details():
    assert L2.forward_labels(["a/x/v.json"], [{"tau_max_years": 30.0}]) == ["column"]
    assert L2.forward_labels(["a/x/v.json", "b/x/v.json"], [{}, {}]) == ["x", "x_1"]
    assert L2.forward_labels(["a/x/v.json", "b/y/v.json"],
                             [{"tau_max_years": 11}, {"tau_max_years": 30.5}]) \
        == ["tau11y", "tau30.5y"]
    assert L2.forward_labels(["v.json", "w.json"], [{}, {}]) == ["v.json", "w.json"]
    for bad in ("30", float("nan"), True, [30]):
        with pytest.raises(ValueError):
            L2.natural_label("a/x/v.json", {"tau_max_years": bad})


def test_mapping_maps_declared_order_on_observed_labels(tmp_path):
    tmp_columns(tmp_path)
    for k, f in L.FORWARDS.items():
        idx, e = L2.label_mapping(k, L2.V1_OBSERVED_LABELS[k], by_rheology(k), tmp_path)
        assert e == [], (k, e)
        assert idx == {c: i for i, c in enumerate(f["columns"])}


def test_v1_rule_is_exactly_what_failed():
    for k in L.FORWARDS:
        assert (L.expected_labels(k) == L2.V1_OBSERVED_LABELS[k]) == (k not in FAILED_V1), k
    text = L2.v1_failure_text()
    assert text.startswith("ENGINEERING FAILURE (exit 2): forward layout/rain: {")
    for k in L.FORWARDS:
        assert (f"'{k}': " in text) == (k in FAILED_V1), k
    assert "coupled_leveling_tau30" in L2.V1_FAILURE_REPORTED and "'column'" in text


# ----------------------------------------------------------------------------- refusals
def test_refusals_on_any_ambiguity(tmp_path):
    tmp_columns(tmp_path)
    obs = L2.V1_OBSERVED_LABELS

    def errs(k, labels, byr=None, cmd=None, root=tmp_path):
        idx, e = L2.label_mapping(k, labels, by_rheology(k) if byr is None else byr, root,
                                  cmd)
        assert (idx == {}) == bool(e)
        return " | ".join(e)

    assert "forward.py's rule" in errs("page", list(reversed(obs["page"])))     # permutation
    assert "not unique" in errs("cwa", ["tau30y", "tau30y"])
    assert "2 rheology labels for 3" in errs("page", obs["page"][:2])
    assert "forward.py's rule" in errs("q0nh", ["coupled_leveling"])            # v1's rule
    assert "subs_mean_by_rheology" in errs("ccz", ["column"], byr=True)
    assert "no subs_mean_by_rheology" in errs("q0", obs["q0"], byr=False)
    other = L.forward_command("q0").replace(L.column_path("q0_s29"), L.column_path("q0"))
    assert "not the declared columns'" in errs("q0", obs["q0"], cmd=other)
    twice = L.forward_command("ccz") + f" --vep-json {L.column_path('ccz')}"
    assert "2 --vep-json" in errs("ccz", ["column"], cmd=twice)
    assert "unreadable" in errs("page", obs["page"], root=tmp_path / "empty")
    tmp_columns(tmp_path, ["cwa_tau30"], {"cwa_tau30": {"tau_max_years": "30"}})
    assert "not a finite number" in errs("cwa", obs["cwa"])


def test_refusal_when_an_own_label_is_another_columns_final_label(tmp_path, monkeypatch):
    """Directories a, a, a_1 -> labels a, a_1, a_1_2: 'a_1' names two columns."""
    cols = {"c1": "a", "c2": "a", "c3": "a_1"}
    for i, (c, sub) in enumerate(cols.items()):
        monkeypatch.setitem(L.COLUMNS, c, (f"run{i}", sub, "B", None))
    monkeypatch.setitem(L.FORWARDS, "zz", {"run": "run0", "rf": L.RF_DEFAULT, "extra": "",
                                           "columns": tuple(cols)})
    tmp_columns(tmp_path, list(cols), {c: {} for c in cols})
    rule = L2.forward_labels(L2.vep_paths(L.forward_command("zz")), [{}, {}, {}])
    assert rule == ["a", "a_1", "a_1_2"]
    idx, e = L2.label_mapping("zz", rule, True, tmp_path)
    assert idx == {} and any("ambiguous" in x for x in e)
    # X_2, X, X -> the rule itself yields a duplicate label
    cols2 = {"d1": "x_2", "d2": "x", "d3": "x"}
    for i, (c, sub) in enumerate(cols2.items()):
        monkeypatch.setitem(L.COLUMNS, c, (f"r{i}", sub, "B", None))
    monkeypatch.setitem(L.FORWARDS, "yy", {"run": "r0", "rf": L.RF_DEFAULT, "extra": "",
                                           "columns": tuple(cols2)})
    tmp_columns(tmp_path, list(cols2), {c: {} for c in cols2})
    idx, e = L2.label_mapping("yy", ["x_2", "x", "x_2"], True, tmp_path)
    assert idx == {} and any("not unique" in x for x in e)


# ----------------------------------------------------------------------------- layout check
def fake_forward(fwd="page", gain=0.0, labels=None):
    T = L.N_T
    d = np.array([f"{2012 + i // 12}-{i % 12 + 1:02d}-01T00:00:00" for i in range(T)])
    z = {"scenario_names": np.array(["baseline", "cut30"]),
         "scenarios": np.array([L.BASE_DESC, L.CUT30_DESC]), "dates": d,
         "origin": np.array(L.ORIGIN), "n_members": np.array(1),
         "member_labels": np.array([L.FORWARDS[fwd]["run"]]), "gain": np.array(gain),
         "hindcast_gain": np.array(1.0 if fwd == "q0nh" else 0.0),
         "rheology_labels": np.array(labels or L2.V1_OBSERVED_LABELS[fwd]),
         "forward_options": np.array(json.dumps({
             "column_heads": "free", "update_method": "innovation",
             "apex_hold": "calibrated", "column_hpc0_fast_days": 365.0,
             "save_members": "none", "column_zone_blend_km": [0.0]})),
         "heads_mean": np.zeros((2, 4, L.N_CELLS, T), dtype="float32")}
    if by_rheology(fwd):
        z["subs_mean_by_rheology"] = np.zeros(1, dtype="float32")    # presence only
    return z


def test_check_forward_npz2_replaces_only_the_label_line(tmp_path):
    tmp_columns(tmp_path)
    for k in L.FORWARDS:
        z = fake_forward(k)
        v1 = L.check_forward_npz(z, k)
        assert (v1 != []) == (k in FAILED_V1)
        assert all(x.startswith("rheology labels ") for x in v1)
        assert L2.check_forward_npz2(z, k, tmp_path) == [], k
    e = L2.check_forward_npz2(fake_forward("page", gain=1.0), "page", tmp_path)
    assert e == ["gain 1.0"]                                   # v1's other checks still bite
    e = L2.check_forward_npz2(fake_forward("page", labels=L.expected_labels("page")), "page",
                              tmp_path)
    assert any("forward.py's rule" in x for x in e)


def test_scorer_mapping_must_be_declared_order():
    fw = {k: {"rheology_index": {c: i for i, c in enumerate(f["columns"])}}
          for k, f in L.FORWARDS.items()}
    assert S2.mapping_is_declared_order(fw) == []
    fw["page"]["rheology_index"] = {"page": 1, "page_tau30": 0, "page_aquitard": 2}
    assert S2.mapping_is_declared_order(fw)[0].startswith("page:")


# ----------------------------------------------------------------------------- reuse binding
def test_reuse_plan_names_only_the_differing_key():
    fw = {k: {"command": L.forward_command(k)} for k in L.FORWARDS}
    assert L2.reuse_plan(fw) == {"reuse": list(L.FORWARDS), "rerun": {}}
    fw["q0nh"] = {"command": L.forward_command("q0nh").replace("--seed 0", "--seed 1")}
    del fw["ccz"]
    fw["extra"] = {"command": "x"}
    plan = L2.reuse_plan(fw)
    assert plan["rerun"] == {"q0nh": "v1 command differs", "ccz": "not in v1's declaration",
                             "extra": "v1 key not declared by v2"}
    assert "q0nh" not in plan["reuse"] and "page" in plan["reuse"]


def synth_binding(root: Path):
    recs = {}
    rows = []
    for i, k in enumerate(L.FORWARDS):
        p = root / f"{L.out_stem(k)}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"synthetic {k}".encode())
        recs[k] = {"event": "forward", "fwd": k, "attempt": 1, "rc": 0,
                   "time": f"t{i}", "npz_sha256": L.sha(p)}
        rows.append(dict(recs[k], stdout_sealed=f"x/{k}"))
    rows.append({"event": "score", "attempt": 1, "rc": 2})
    att = root / "attempts.jsonl"
    att.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return att, recs


def test_bind_v1_forwards(tmp_path):
    att, recs = synth_binding(tmp_path)
    files, e = L2.bind_v1_forwards(att, tmp_path, recs)
    assert e == [] and sorted(files) == sorted(L.FORWARDS)
    assert L2.bind_v1_forwards(tmp_path / "none.jsonl", tmp_path, recs)[1]
    # a tampered npz
    (tmp_path / f"{L.out_stem('q0')}.npz").write_bytes(b"other")
    files, e = L2.bind_v1_forwards(att, tmp_path, recs)
    assert "q0" not in files and any(x.startswith("q0:") for x in e)


def test_bind_refuses_later_events_and_duplicates(tmp_path):
    att, recs = synth_binding(tmp_path)
    with open(att, "a") as fh:
        fh.write(json.dumps({"event": "forward", "fwd": "cwa", "attempt": 2, "rc": 1}) + "\n")
        fh.write(json.dumps({"event": "archive", "keys": "page ccz", "reason": "r"}) + "\n")
        fh.write(json.dumps(dict(recs["q0c"])) + "\n")
    files, e = L2.bind_v1_forwards(att, tmp_path, recs)
    assert sorted(files) == ["q0", "q0cs", "q0nh", "x5r1cs"]
    msg = " | ".join(e)
    assert "cwa: a later forward" in msg and "page: a later archive" in msg
    assert "ccz: a later archive" in msg and "q0c: 2 attempts.jsonl rows" in msg


def test_scorer_bound_npz_refuses_a_rerun_key(tmp_path):
    fw = {k: {"command": L.forward_command(k)} for k in L.FORWARDS}
    fw["cwa"] = {"command": "something else"}
    with pytest.raises(S2.Engineering) as ei:
        S2.bound_npz(tmp_path / "attempts.jsonl", {"forwards": fw})
    assert ei.value.code == 1 and "'cwa'" in str(ei.value)


def test_pinned_records_are_well_formed():
    assert sorted(L2.V1_FORWARDS) == sorted(L.FORWARDS)
    for k, r in L2.V1_FORWARDS.items():
        assert r["fwd"] == k and r["rc"] == 0 and r["declaration_sha256"] == L2.V1_SHA256
        assert all(len(r[f]) == 64 for f in ("npz_sha256", "summary_csv_sha256",
                                             "members_csv_sha256", "stdout_sealed_sha256"))
        assert r["worktree_head"].startswith(L.CODE)
    assert L2.V1_SCORE_ATTEMPT["rc"] == 2


@needs_forwards
def test_pinned_records_equal_v1_attempts_log():
    rows = [json.loads(s) for s in (DATA / L2.V1_ATTEMPTS).read_text().splitlines()
            if s.strip()]
    for k, want in L2.V1_FORWARDS.items():
        got = [r for r in rows if r.get("event") == "forward" and r.get("fwd") == k]
        assert len(got) == 1 and {f: got[0].get(f) for f in want} == want, k
    sc = [r for r in rows if r.get("event") == "score"]
    assert sc and {f: sc[0].get(f) for f in L2.V1_SCORE_ATTEMPT} == L2.V1_SCORE_ATTEMPT


@needs_forwards
@needs_results
def test_real_label_mapping_from_npz_metadata():
    """npz rheology_labels and key list only; the real column JSONs; v1's registered
    commands (the commands the runner passed)."""
    v1 = json.loads((DATA / L2.V1_DECL).read_text())
    assert L2.reuse_plan(v1["forwards"])["rerun"] == {}
    for k, f in L.FORWARDS.items():
        m = L2.npz_label_metadata(DATA / f"{L.out_stem(k)}.npz")
        assert m["labels"] == L2.V1_OBSERVED_LABELS[k]
        idx, e = L2.label_mapping(k, m["labels"], m["has_by_rheology"], DATA,
                                  v1["forwards"][k]["command"])
        assert e == [] and idx == {c: i for i, c in enumerate(f["columns"])}, (k, e)


# ----------------------------------------------------------------------------- v1 untouched
def test_v1_files_byte_identical_and_frozen_in_v2():
    for p, s in {**L2.V1_FROZEN, **V1_INPUT_CODE}.items():
        assert L.sha(REPO / p) == s, p
    assert set(L2.V1_FROZEN) <= set(S2.FROZEN) and tuple(S.FROZEN) == S2.FROZEN[3:]
    sh = (X / "run_sd2.sh").read_text()
    for p in S2.FROZEN:
        assert (REPO / p).exists() and p in sh, p
    assert L.SD_ID == L2.V1_ID == "SD.cut_response_spread.v1"


def test_forward_py_rule_source_pinned():
    g = subprocess.run(["git", "-C", str(REPO), "show", f"{L.CODE}:{L2.FORWARD_PY}"],
                       capture_output=True)
    if g.returncode != 0:
        pytest.skip(f"git object {L.CODE} not available")
    import hashlib

    assert hashlib.sha256(g.stdout).hexdigest() == L2.FORWARD_PY_SHA256
    text = g.stdout.decode()
    assert L2.rule_source_problems(text) == []
    cut = text.replace('            label = f"{label}_{len(out)}"\n', "")
    assert len(L2.rule_source_problems(cut)) == 1


def test_verify_v1_refuses_without_registration(tmp_path):
    with pytest.raises(S2.Engineering) as ei:
        S2.verify_v1(prereg_dir=tmp_path / "p", rows_dir=tmp_path / "rows")
    assert ei.value.code == 1


# ----------------------------------------------------------------------------- runner
def run_sh(script: str, *args, env_extra=None):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), **(env_extra or {})}
    return subprocess.run(["bash", str(X / script), *args], cwd=REPO, env=env,
                          capture_output=True, text=True)


def test_runner_v2_dry_lines_and_refusals(tmp_path):
    if HPY.exists():
        out = run_sh("run_sd2.sh", "forward", "all", env_extra={"DRY": "1"})
        assert out.returncode == 0
        assert out.stdout.splitlines() == [L.forward_command(k) for k in L.FORWARDS]
    r = run_sh("run_sd2.sh", "forward", "all")
    assert r.returncode == 1 and "no forward runs under SD.cut_response_spread.v2" in r.stderr
    r = run_sh("run_sd2.sh", "score", env_extra={"DATAPREP_REPO": str(tmp_path)})
    assert r.returncode == 1 and "declare SD.cut_response_spread.v2 first" in r.stderr
    assert os.access(X / "run_sd2.sh", os.X_OK)


# ----------------------------------------------------------------------------- declaration
@needs_results
def test_sd2_declaration_is_v1_but_for_the_declared_fields(tmp_path):
    import prereg_sd2 as P2
    from harness import prereg3

    d = P2.doc()
    v1 = P2.v1_registered()
    assert prereg3.validate_prereg(d) == []
    assert P2.identity_problems(d, v1) == []
    assert d["id"] == L2.SD2_ID and d["supersedes"]["sha256"] == L2.V1_SHA256
    assert sorted(d["frozen_code"]) == sorted(S2.FROZEN)
    assert all(r["v1_command_identical"] for r in d["reused_forwards"].values())
    assert d["columns"] == v1["columns"] and d["thresholds"]["readings"] == \
        v1["thresholds"]["readings"]
    eng = d["thresholds"]["engineering_checks_before_any_cut30_quantity_is_formed"]
    assert eng[0] == P2.STEP0_V2 and P2.LABELS_V2 in eng[2]
    assert "message_rebuilt" in d["v1_engineering_failure"]
    assert L2.V1_DECL in d["inputs"] and L2.V1_CLOSURE in d["inputs"]
    npz = [x for x in d["inputs"] if isinstance(x, dict) and x["path"].endswith(".npz")
           and "/sd_" in x["path"]]
    assert {x["sha256"] for x in npz} == {r["npz_sha256"] for r in L2.V1_FORWARDS.values()}
    if all((DATA / f"{L.out_stem(k)}.npz").exists() for k in L.FORWARDS):
        path, h = prereg3.write_prereg(d, prereg_dir=tmp_path / "prereg",
                                       rows_dir=tmp_path / "rows", registries=[])
        assert path.exists() and len(h) == 64


# ----------------------------------------------------------------------------- GSC repoint
def test_gsc_points_at_sd_v2():
    assert M.SD_ID == L2.SD2_ID and M.SD_REPORT == L2.REPORT
    sh = (X / "run_gsc.sh").read_text()
    assert f"SD_REPORT={L2.REPORT}" in sh.splitlines()
    assert "score SD.cut_response_spread.v2 first" in sh
    assert "reports/SD.cut_response_spread.v1" not in sh
    assert "SD.cut_response_spread.v2 is scored" in (X / "prereg_gsc.py").read_text()


def test_gsc_model_side_refuses_before_sd_v2_is_scored(tmp_path):
    r = run_sh("run_gsc.sh", "model-side", env_extra={"DATAPREP_REPO": str(tmp_path)})
    assert r.returncode == 1
    assert "REFUSED: score SD.cut_response_spread.v2 first" in r.stderr


def test_gsc_build_reads_only_an_sd_v2_report(tmp_path):
    p = tmp_path / M.SD_REPORT
    p.parent.mkdir(parents=True)
    checks = {c: {"pass": True, "rheology_index": L.FORWARDS[L.forward_of(c)]["columns"]
                  .index(c)} for c in L.COLUMNS}
    p.write_text(json.dumps({"id": L2.V1_ID, "columns_checks": checks}))
    with pytest.raises(SystemExit, match="is not SD.cut_response_spread.v2's report"):
        M.build(tmp_path)
    bad = json.loads(json.dumps(checks))
    bad["page_tau30"]["rheology_index"] = 2
    p.write_text(json.dumps({"id": L2.SD2_ID, "columns_checks": bad}))
    with pytest.raises(SystemExit, match="page_tau30: SD's rheology index"):
        M.build(tmp_path)
    no_page = dict(checks, page={"pass": False})
    p.write_text(json.dumps({"id": L2.SD2_ID, "columns_checks": no_page}))
    with pytest.raises(SystemExit, match="page column did not pass"):
        M.build(tmp_path)


def test_gsc_registration_lists_the_sd_v2_report(monkeypatch, tmp_path):
    import prereg_gsc as PG

    monkeypatch.setattr(PG, "repo", lambda: tmp_path)
    d = PG.doc()
    assert M.SD_REPORT in d["inputs"] and "SD.cut_response_spread.v1" not in M.SD_REPORT
