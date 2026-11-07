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
    assert EV.read_attempts() == []                      # nothing committed has been run


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
