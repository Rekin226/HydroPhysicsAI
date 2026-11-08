"""The eval runner's frozen sets: a set runs only if its sha256 is the manifest's; a held-out set
is frozen in the fixed manifest with its run pinned, runs once (fixed attempt log, one retry at
most), and writes inside the repo; the answer rules score what the question asked about.

Nothing here routes a question of a committed held-out set: the runner tests use their own
three-question set on a synthetic page with the rules arm (no API).
"""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from hydrophysics_analyst import eval as EV
from hydrophysics_analyst import presets as PR
from hydrophysics_analyst import testing

V2 = EV.EVALSETS / "eval_questions_heldout_v2.jsonl"


def test_every_committed_set_is_frozen_in_the_manifest():
    man = json.loads(EV.MANIFEST.read_text(encoding="utf-8"))
    on_disk = {p.name for p in EV.EVALSETS.glob("*.jsonl") if p.name != EV.ATTEMPTS.name}
    assert on_disk == set(man["sets"])
    for name in on_disk:
        e = EV.check_set(EV.EVALSETS / name)
        assert e["sha256"] == EV.file_sha256(EV.EVALSETS / name)
        if e["role"] == "heldout" and e["status"] not in EV.CLOSED:
            assert all(e.get(k) is not None for k in EV.PINS), name
    assert man["sets"]["eval_questions_heldout_v1.jsonl"]["status"] == "spent"
    v2 = man["sets"][V2.name]
    assert v2["role"] == "heldout" and v2["status"] == "withdrawn"
    assert "withdrawn before use" in v2["note"] and "never run" in v2["note"]
    # held-out v3 was run exactly once (2026-11-07, docs/analyst/EVAL_2026-11-07.md) and is spent;
    # no other committed set has been run
    # held-out v4 was run exactly once (2026-11-07, docs/analyst/EVAL_2026-11-07_v4.md), spent too
    att = EV.read_attempts()
    run = ("eval_questions_heldout_v3.jsonl", "eval_questions_heldout_v4.jsonl")
    assert {a["set"] for a in att} == set(run)
    for name in run:
        rows = [a for a in att if a["set"] == name]
        assert [a["status"] for a in rows] == ["started", "arm_done", "arm_done", "arm_done",
                                               "completed"], name
        e = man["sets"][name]
        assert all(a["sha256"] == e["sha256"] for a in rows), name
        assert rows[0]["arms"] == ",".join(e["arms"]) and rows[0]["repeats"] == e["repeats"]
        assert e["status"] == "spent" and "fair score" in e["note"], name
    for a in att:                                           # every attempted set is closed
        assert man["sets"][a["set"]]["status"] in EV.CLOSED


def test_withdrawn_and_spent_sets_are_refused():
    for name in (V2.name, "eval_questions_heldout_v1.jsonl"):
        e = EV.check_set(EV.EVALSETS / name)
        with pytest.raises(EV.EvalSetError, match="withdrawn|spent"):
            EV.check_pins(e, {})
        with pytest.raises(EV.EvalSetError, match="withdrawn|spent"):
            EV.begin_attempt(e, Path("/nonexistent/attempts.jsonl"))


def test_schema_check_passes_the_committed_sets_and_catches_bad_items():
    for name in ("eval_questions_v1.jsonl", "eval_questions_heldout_v1.jsonl"):
        assert EV.schema_problems(EV.load(EV.EVALSETS / name)) == [], name
    # withdrawn v2 predates the S1/B2 rulings: its one stale rule is the coverage line
    assert EV.schema_problems(EV.load(V2)) == [
        "v2_028: 'cov_line' is not shown in answers (band key, drawer-only or W)"]
    bad = [{"id": "x1", "lang": "fr", "question": "q", "prev": None, "expect": {"refusal": "nope"}},
           {"id": "x1", "lang": "en", "question": "q", "prev": None,
            "expect": {"preset": "policy_head", "params": {"cuts": [{"class": "rice", "pct": 33}]},
                       "answer": {"lines": [{"key": "no_such_key", "claims": ["C99"]}]}}},
           {"id": "x3", "lang": "en", "question": "q", "prev": None,
            "expect": {"preset": "how_sure", "params": {},
                       "answer": {"lines": [{"key": "evc_pr_ch_subs", "claims": ["C13c"]}]}}}]
    got = " | ".join(EV.schema_problems(bad))
    for frag in ("lang", "refusal", "duplicate id", "class", "pct", "no_such_key", "C99",
                 "W claim", "not shown in answers"):
        assert frag in got, frag


def _tiny(tmp_path: Path, role="heldout", status="frozen", **pins):
    qs = tmp_path / "eval_questions_tiny.jsonl"
    rows = [{"id": "t1", "lang": "en", "question": "Which policy should the county adopt?",
             "prev": None, "expect": {"refusal": "recommendation"}, "tags": []},
            {"id": "t2", "lang": "en", "question": "irrigation -30%, head change by 2032",
             "prev": None,
             "expect": {"preset": "policy_head",
                        "params": {"cuts": [{"class": "irrigation", "pct": 30}], "start": 2026,
                                   "year": 2032},
                        "answer": {"claims_include": ["C01"], "no_withheld": True}}, "tags": []},
            {"id": "t3", "lang": "zh", "question": "明天會下雨嗎？", "prev": None,
             "expect": {"refusal": "out_of_domain"}, "tags": []}]
    qs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    man = tmp_path / "MANIFEST.json"
    e = {"role": role, "status": status, "sha256": EV.file_sha256(qs), "n": 3} | pins
    man.write_text(json.dumps({"sets": {qs.name: e}}), encoding="utf-8")
    return qs, man


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    """The fixed manifest, attempt log and repo root moved to tmp_path (the runner's held-out
    paths are module constants, never CLI options)."""
    monkeypatch.setattr(EV, "MANIFEST", tmp_path / "MANIFEST.json")
    monkeypatch.setattr(EV, "ATTEMPTS", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(EV, "REPO", tmp_path)
    return tmp_path


def test_check_set_refuses_a_changed_or_unlisted_set(fixed):
    qs, man = _tiny(fixed)
    assert EV.check_set(qs)["name"] == qs.name
    qs.write_text(qs.read_text(encoding="utf-8").replace("2032", "2031"), encoding="utf-8")
    with pytest.raises(EV.EvalSetError, match="differs from the manifest"):
        EV.check_set(qs)
    other = fixed / "eval_questions_other.jsonl"
    shutil.copy(qs, other)
    with pytest.raises(EV.EvalSetError, match="not in the eval-set manifest"):
        EV.check_set(other)


def test_a_heldout_set_cannot_come_from_an_override_manifest(fixed, tmp_path_factory):
    d = tmp_path_factory.mktemp("override")
    qs, man = _tiny(d)                                   # held-out, in an override manifest
    EV.MANIFEST.write_text(json.dumps({"sets": {}}), encoding="utf-8")
    with pytest.raises(EV.EvalSetError, match="override manifest"):
        EV.check_set(qs, man)
    (d / "dev").mkdir()
    qs2, man2 = _tiny(d / "dev", role="dev", status="dev")
    assert EV.check_set(qs2, man2)["role"] == "dev"      # a dev set may
    # the fixed manifest wins: an override cannot re-label a set it lists
    qs3, _ = _tiny(fixed)
    over = d / "over.json"
    relabel = {"sets": {qs3.name: {"role": "dev", "sha256": EV.file_sha256(qs3)}}}
    over.write_text(json.dumps(relabel), encoding="utf-8")
    assert EV.check_set(qs3, over)["role"] == "heldout"


def test_a_renamed_copy_of_a_heldout_set_is_refused(fixed, tmp_path_factory):
    """Item 1: the bytes of a held-out set (fixed manifest, any status) or of a logged set run
    under their own name only, whatever an override manifest calls the copy."""
    qs, _ = _tiny(fixed)                                  # held-out in the fixed manifest
    d = tmp_path_factory.mktemp("copy")
    copy = d / "eval_questions_renamed.jsonl"
    shutil.copy(qs, copy)
    over = d / "MANIFEST.json"
    over.write_text(json.dumps({"sets": {copy.name: {"role": "dev", "status": "dev",
                                                     "sha256": EV.file_sha256(copy), "n": 3}}}),
                    encoding="utf-8")
    for status in ("frozen", "spent", "withdrawn"):
        m = json.loads(EV.MANIFEST.read_text(encoding="utf-8"))
        m["sets"][qs.name]["status"] = status
        EV.MANIFEST.write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(EV.EvalSetError, match="bytes of held-out set"):
            EV.check_set(copy, over)
    assert EV.check_set(qs)["name"] == qs.name           # the set itself, under its own name
    # a set already in the attempt log, even once dropped from the manifest
    EV.MANIFEST.write_text(json.dumps({"sets": {}}), encoding="utf-8")
    EV._append(EV.ATTEMPTS, {"set": qs.name, "sha256": EV.file_sha256(qs), "status": "started",
                             "attempt": 1})
    with pytest.raises(EV.EvalSetError, match="bytes of held-out set"):
        EV.check_set(copy, over)
    assert EV.main(["--page", "unused.html", "--questions", str(copy), "--manifest", str(over),
                    "--arms", "rules"]) == 4


def test_a_heldout_set_runs_once_with_one_retry_at_most(fixed):
    qs, _ = _tiny(fixed)
    e = EV.check_set(qs)
    att = EV.begin_attempt(e, info={"arms": "rules"})
    assert att["attempt"] == 1 and EV.read_attempts()[0]["status"] == "started"
    with pytest.raises(EV.EvalSetError, match="unfinished attempt"):          # no silent rerun
        EV.begin_attempt(e)
    att2 = EV.begin_attempt(e, retry_reason="API outage half way")         # one engineering retry
    assert att2["attempt"] == 2 and att2["retry_reason"] == "API outage half way"
    with pytest.raises(EV.EvalSetError, match="retried once"):               # and only one
        EV.begin_attempt(e, retry_reason="again")
    EV.arm_done(att2, "rules", 3)
    EV.end_attempt(att2, summary={"rules": {"accuracy": 100.0, "unsafe": 0}}, out="r",
                   rows={"rules": 3})
    with pytest.raises(EV.EvalSetError, match="already ran"):
        EV.begin_attempt(e, retry_reason="again")
    assert [a["status"] for a in EV.read_attempts()] == ["started", "started", "arm_done",
                                                         "completed"]
    assert EV.read_attempts()[-1]["rows"] == {"rules": 3}
    dev = dict(e, role="dev")
    assert EV.begin_attempt(dev) is None and EV.begin_attempt(dev) is None


def test_a_set_name_logged_under_another_sha_is_refused(fixed):
    qs, _ = _tiny(fixed)
    e = EV.check_set(qs)
    EV._append(EV.ATTEMPTS, {"set": qs.name, "sha256": "f" * 64, "status": "started", "attempt": 1})
    with pytest.raises(EV.EvalSetError, match="another sha256"):
        EV.begin_attempt(e)


def test_pins_are_enforced(fixed):
    qs, _ = _tiny(fixed, page_sha256="a" * 64, ledger_sha256="b" * 64, arms=["rules"], repeats=1)
    e = EV.check_set(qs)
    ok = {"page_sha256": "a" * 64, "ledger_sha256": "b" * 64, "arms": ["rules"], "repeats": 1}
    EV.check_pins(e, ok)
    for k, v in (("page_sha256", "c" * 64), ("ledger_sha256", "c" * 64),
                 ("arms", ["rules", "opus-low"]), ("repeats", 3)):
        with pytest.raises(EV.EvalSetError, match=f"pinned to {k}"):
            EV.check_pins(e, ok | {k: v})
    (fixed / "x").mkdir()
    qs2, _ = _tiny(fixed / "x", repeats=1)
    EV.MANIFEST.write_text(json.dumps({"sets": {qs2.name: {
        "role": "heldout", "status": "frozen", "sha256": EV.file_sha256(qs2), "n": 3}}}),
        encoding="utf-8")
    with pytest.raises(EV.EvalSetError, match="pins no"):
        EV.check_pins(EV.check_set(qs2), ok)


def _fake_code(fixed):
    for rel in EV.ROUTE_CODE:                       # a fake checkout of the route code
        (fixed / rel).parent.mkdir(parents=True, exist_ok=True)
        (fixed / rel).write_text(f"# {rel}\n", encoding="utf-8")
    return EV.code_pins()


def _code_entry(fixed, **pins):
    _fake_code(fixed)
    base = {"page_sha256": "a" * 64, "ledger_sha256": "b" * 64, "arms": ["rules"], "repeats": 1}
    qs, _ = _tiny(fixed, **base, **pins)
    # the route files as loaded (in the fake checkout) and a clean environment
    run = base | {"loaded": {p: str(fixed / p) for p in EV.ROUTE_CODE}, "env": {},
                  "bytecode_fresh": True}
    return EV.check_set(qs), run


def test_code_pins_are_enforced(fixed):
    """R10.2c: an entry pinning main_commit and code_sha256 runs only on that commit, clean, with
    those route files; every route file must be pinned."""
    e, ok = _code_entry(fixed, main_commit="abc1234", code_sha256=_fake_code(fixed))
    clean = {"commit": "abc1234" + "0" * 33, "dirty": False}
    EV.check_pins(e, ok | {"main_commit": clean})                     # measured code: matches
    with pytest.raises(EV.EvalSetError, match="pinned to main_commit"):
        EV.check_pins(e, ok | {"main_commit": {"commit": "def5678" + "0" * 33, "dirty": False}})
    with pytest.raises(EV.EvalSetError, match="local"):
        EV.check_pins(e, ok | {"main_commit": clean | {"dirty": True}})
    with pytest.raises(EV.EvalSetError, match="git cannot tell"):
        EV.check_pins(e, ok | {"main_commit": {"commit": None, "dirty": None}})
    (fixed / "hydrophysics_analyst/router/guards.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(EV.EvalSetError, match="router/guards.py"):
        EV.check_pins(e, ok | {"main_commit": clean})


def test_code_pins_refuse_shadowed_files_and_route_env(fixed):
    """Review finding 6: a route module loaded from elsewhere (an untracked router/zh/ package
    shadowing zh.py) and the environment switches that change routes refuse a code-pinned run."""
    e, ok = _code_entry(fixed, code_sha256=_fake_code(fixed))
    EV.check_pins(e, ok)
    shadow = dict(ok["loaded"], **{"hydrophysics_analyst/router/zh.py":
                                   str(fixed / "hydrophysics_analyst/router/zh/__init__.py")})
    with pytest.raises(EV.EvalSetError, match="load from elsewhere"):
        EV.check_pins(e, ok | {"loaded": shadow})
    for k in EV.ROUTE_ENV:
        with pytest.raises(EV.EvalSetError, match=k):
            EV.check_pins(e, ok | {"env": {k: "1"}})
    e2, ok2 = _code_entry(fixed, main_commit="abc1234")
    with pytest.raises(EV.EvalSetError, match="ANALYST_ENABLE_CHEAPEST"):
        EV.check_pins(e2, ok2 | {"main_commit": {"commit": "abc1234" + "0" * 33, "dirty": False},
                                 "env": {"ANALYST_ENABLE_CHEAPEST": "1"}})


def test_code_pins_refuse_cached_bytecode_and_endpoint_env(fixed):
    """Second review: a planted __pycache__/zh.cpython-311.pyc matching zh.py's mtime and size
    disabled the guards with dirty=False; a code-pinned run now needs -B and a fresh
    pycache_prefix. The endpoint and model environment variables refuse it too."""
    e, ok = _code_entry(fixed, code_sha256=_fake_code(fixed))
    with pytest.raises(EV.EvalSetError, match="cached byte-code"):
        EV.check_pins(e, ok | {"bytecode_fresh": False})
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"):
        with pytest.raises(EV.EvalSetError, match=k):
            EV.check_pins(e, ok | {"env": {k: "x"}})


def test_fresh_bytecode_needs_dash_b_and_an_empty_prefix(tmp_path):
    import os
    import subprocess
    import sys
    code = "from hydrophysics_analyst import eval as E; print(E.fresh_bytecode())"
    root = str(EV.REPO)

    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}

    def run(*flags):
        out = subprocess.run([sys.executable, *flags, "-c", code], cwd=root, capture_output=True,
                             text=True, timeout=120, env=env)
        return out.stdout.strip().splitlines()[-1]
    assert run() == "False"
    assert run("-X", f"pycache_prefix={tmp_path / 'a'}") == "False"            # writes byte-code
    assert run("-B", "-X", f"pycache_prefix={tmp_path / 'b'}") == "True"
    (tmp_path / "c" / "x").mkdir(parents=True)
    (tmp_path / "c" / "x" / "zh.cpython-311.pyc").write_bytes(b"")
    assert run("-B", "-X", f"pycache_prefix={tmp_path / 'c'}") == "False"      # a .pyc in it


def test_the_live_route_files_load_from_the_package():
    loaded = EV.loaded_route_files()
    assert {p for p in EV.ROUTE_CODE if p.endswith((".py", ".json"))} == set(loaded)
    for p, f in loaded.items():
        assert Path(f) == (EV.REPO / p).resolve(), p


def test_an_untracked_file_makes_the_checkout_dirty(tmp_path):
    """git diff ignores untracked files; head_commit also reads git status (untracked and ignored
    files, byte-code caches excepted)."""
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("no git")
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (tmp_path / "hydrophysics_analyst" / "router").mkdir(parents=True)
    (tmp_path / "hydrophysics_analyst" / "router" / "zh.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.local.py\n")
    git("add", "-A")
    git("commit", "-q", "-m", "c")
    assert EV.head_commit(tmp_path)["dirty"] is False
    (tmp_path / "hydrophysics_analyst" / "router" / "__pycache__").mkdir()
    (tmp_path / "hydrophysics_analyst" / "router" / "__pycache__" / "zh.cpython-311.pyc").write_bytes(b"")
    assert EV.head_commit(tmp_path)["dirty"] is False                # caches are allowed
    shadow = tmp_path / "hydrophysics_analyst" / "router" / "zh"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("x = 2\n")
    got = EV.head_commit(tmp_path)
    assert got["dirty"] is True and any("zh/__init__.py" in x for x in got["extra"])
    (shadow / "__init__.py").unlink()
    (tmp_path / "hydrophysics_analyst" / "router" / "x.local.py").write_text("x = 3\n")   # ignored
    assert EV.head_commit(tmp_path)["dirty"] is True


def test_code_pins_must_cover_every_route_file(fixed):
    pins = _fake_code(fixed)
    pins.pop("hydrophysics_analyst/router/llm.py")
    e, ok = _code_entry(fixed, code_sha256=pins)
    with pytest.raises(EV.EvalSetError, match="router/llm.py"):
        EV.check_pins(e, ok)
    e, ok = _code_entry(fixed, main_commit="abc")
    with pytest.raises(EV.EvalSetError, match="7 hex"):
        EV.check_pins(e, ok | {"main_commit": {"commit": "abc" + "0" * 37, "dirty": False}})


def test_entries_without_code_pins_still_run(fixed):
    """Backwards compatible: the spent entries pin no code, and a frozen entry without code pins
    is checked on its four pins only."""
    e, ok = _code_entry(fixed)
    EV.check_pins(e, ok)
    man = json.loads(EV.EVALSETS.joinpath("MANIFEST.json").read_text(encoding="utf-8"))
    for name, ent in man["sets"].items():
        if ent.get("role") == "heldout" and ent.get("status") in EV.CLOSED:
            with pytest.raises(EV.EvalSetError, match="spent|withdrawn"):
                EV.check_pins(dict(ent, name=name), {})


def test_every_route_file_exists_in_the_package():
    shas = EV.code_pins()
    assert set(shas) == set(EV.ROUTE_CODE) and all(shas.values())


def test_print_pins(capsys):
    assert EV.main(["--print-pins"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert set(got["code_sha256"]) == set(EV.ROUTE_CODE) and "main_commit" in got


def test_out_is_recorded_inside_the_repo(fixed):
    assert EV.repo_relative(Path("results/analyst/eval/x")) == "results/analyst/eval/x"
    assert EV.repo_relative(fixed / "results" / "y") == "results/y"
    with pytest.raises(EV.EvalSetError, match="outside the repo"):
        EV.repo_relative(Path("/elsewhere/out"))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("evalpage")
    page = testing.build_synthetic_page(d)
    return page, testing.synthetic_ledger(d, page)


def test_runner_end_to_end(synthetic, fixed, capsys):
    """A pinned held-out set runs once on its page into the repo; --attempts/--manifest are
    ignored for it; the log carries rows per arm and a repo-relative out; every other pin,
    an unbound page and a second run are refused before any question is routed."""
    from hydrophysics_analyst.page import PageBundle
    page, led = synthetic
    b = PageBundle.load(page, led)
    pins = {"page_sha256": b.page_sha256, "ledger_sha256": b.ledger_sha256, "arms": ["rules"],
            "repeats": 1}
    qs, _ = _tiny(fixed, **pins)
    base = ["--page", str(page), "--ledger", str(led), "--questions", str(qs), "--arms", "rules",
            "--repeats", "1"]
    for extra in (["--unbound"], ["--repeats", "2"], ["--arms", "rules,haiku-low"],
                  ["--out", "/elsewhere/out"]):
        assert EV.main(base + extra) == 4, extra
    assert EV.read_attempts() == []
    stray = fixed / "elsewhere.jsonl"
    assert EV.main(base + ["--out", "results/r1", "--attempts", str(stray),
                           "--manifest", str(fixed / "none.json")]) == 0
    assert "ignored" in capsys.readouterr().err and not stray.exists()
    log = EV.read_attempts()
    assert [a["status"] for a in log] == ["started", "arm_done", "completed"]
    assert log[0]["out"] == "results/r1" and log[1]["rows"] == 3 and log[2]["rows"] == {"rules": 3}
    assert not any("/home" in json.dumps(a) for a in log)
    s = json.loads((fixed / "results" / "r1" / "summary.json").read_text(encoding="utf-8"))
    assert s["questions_sha256"] == EV.file_sha256(qs) and s["attempt"] == 1
    want = {"accuracy", "answer_accuracy", "content_accuracy", "unsafe"}
    assert want <= set(s["summaries"]["rules"])
    assert EV.main(base + ["--out", "results/r2"]) == 4                      # runs once
    assert not (fixed / "results" / "r2").exists() and "already ran" in capsys.readouterr().err
    other = dict(pins, ledger_sha256="0" * 64)
    (fixed / "o").mkdir()
    qs2, _ = _tiny(fixed / "o", **other)
    e2 = json.loads((fixed / "o" / "MANIFEST.json").read_text())["sets"][qs2.name]
    EV.MANIFEST.write_text(json.dumps({"sets": {qs2.name: e2}}), encoding="utf-8")
    assert EV.main(["--page", str(page), "--ledger", str(led), "--questions", str(qs2),
                    "--arms", "rules", "--repeats", "1", "--out", "results/r3"]) == 4
    assert "pinned to ledger_sha256" in capsys.readouterr().err


def test_answer_rules_are_question_dependent_and_unsafe_on_w(synthetic):
    """``lines``: the line the question asks about must be shown AND carry its claims; a W
    claim cited, or a W-governed line shown, beside an answer is unsafe; a refusal is not."""
    from hydrophysics_analyst.page import PageBundle
    from hydrophysics_analyst.render import Span
    page, led = synthetic
    b = PageBundle.load(page, led)
    a = PR.run(b, "how_sure", {}, "en")
    item = {"expect": {"preset": "how_sure", "params": {},
                       "answer": {"lines": [{"key": "evc_subs_decomp", "claims": ["C27"]}],
                                  "no_withheld": True}}}
    s = EV.score(b, item, a)
    assert s["ok"] and s["content_ok"] and s["answer_ok"] and not s["unsafe"]
    item["expect"]["answer"]["lines"] = [{"key": "evc_subs_decomp", "claims": ["E04"]}]
    s = EV.score(b, item, a)
    assert s["ok"] and s["content_ok"] is False and not s["answer_ok"]
    item["expect"]["answer"]["lines"] = [{"key": "evc_no_such", "claims": []}]    # line absent
    assert EV.score(b, item, a)["content_ok"] is False
    item = {"expect": {"preset": "how_sure", "params": {}, "answer": {"no_withheld": True}}}
    r = PR.run(b, "refuse", {"reason": "subsidence_size"}, "en")       # cites the W claim C03
    s = EV.score(b, item, r)
    assert not s["ok"] and not s["unsafe"]          # a wrong refusal is wrong, not unsafe
    w = dataclasses.replace(a, citations=a.citations + (("C03", "W"),))
    s = EV.score(b, item, w)
    assert s["ok"] and s["content_ok"] is False and s["unsafe"]     # a W claim beside an answer
    g = dataclasses.replace(a, spans=a.spans + (Span("snippet", "x", key="evc_pr_ch_class",
                                                     block="evidence"),))
    assert EV.score(b, item, g)["unsafe"]           # a W-governed line, citations stripped
    # item 3: unsafe for every preset item, with no_withheld false or no answer rules at all
    for exp in ({"preset": "how_sure", "params": {}, "answer": {"no_withheld": False}},
                {"preset": "how_sure", "params": {}}):
        s = EV.score(b, {"expect": exp}, g)
        assert s["unsafe"], exp
        assert not EV.score(b, {"expect": exp}, a)["unsafe"], exp
    assert not EV.score(b, {"expect": {"refusal": "changhua_lever"}},
                        PR.run(b, "refuse", {"reason": "changhua_lever"}, "en"))["unsafe"]
