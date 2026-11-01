"""ask() end to end with a fake LLM, the audit log, and replay without the LLM."""
from __future__ import annotations

import json

import pytest

from hydrophysics_analyst import audit, testing
from hydrophysics_analyst.ask import ask
from hydrophysics_analyst.llm.backend import FakeBackend
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.router import routed_presets
from hydrophysics_analyst.router.llm import LLMRouter
from hydrophysics_analyst.session import Session


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("a")
    page = testing.build_synthetic_page(d)
    led = testing.synthetic_ledger(d, page)
    return page, led, PageBundle.load(page, led, allow_pending=True)


def test_ask_rules_path_logs_and_replays(built, tmp_path):
    page, led, b = built
    ans, v, rec = ask(b, "停止養殖抽水對2032年水位影響？", use_llm=False, audit_dir=tmp_path)
    assert v == [] and rec["stage"] == "rules" and ans.lang == "zh"
    again = audit.replay(rec["id"], b, tmp_path)
    assert again.text() == ans.text()


def test_session_follow_up(built, tmp_path):
    _, _, b = built
    s = Session()
    ask(b, "Cut aquaculture 30% from 2026", use_llm=False, audit_dir=tmp_path, session=s)
    ans, _, _ = ask(b, "same for 2030", use_llm=False, audit_dir=tmp_path, session=s)
    assert ans.params["start"] == 2030 and ans.params["cuts"][0]["class"] == "aquaculture"


def test_ask_llm_path_with_fake_backend(built, tmp_path):
    _, _, b = built
    out = {"preset": "policy_head", "cuts": [{"class": "irrigation", "pct": 35}], "start": 2026, "year": 2031,
           "town": None, "by": None, "target_m": None, "refusal": None, "clarify": None, "inherit": [],
           "answer_lang": "en"}
    llm = LLMRouter(b, FakeBackend([out]), presets=routed_presets())
    ans, v, rec = ask(b, "irrigation down by about a third, effect by 2031?", llm=llm, audit_dir=tmp_path)
    assert rec["stage"] == "llm" and v == [] and ans.params["year"] == 2031


def test_clarification_is_not_a_validator_failure(built, tmp_path):
    _, _, b = built
    out = {"preset": "policy_head", "cuts": None, "start": None, "year": 2031, "town": None, "by": None,
           "target_m": None, "refusal": None, "clarify": "out_of_grid", "inherit": [], "answer_lang": "en"}
    llm = LLMRouter(b, FakeBackend([out]), presets=routed_presets())
    ans, v, _ = ask(b, "irrigation by a third?", llm=llm, audit_dir=tmp_path)
    assert v == [] and ans.clarify["reason"] == "out_of_grid" and ans.fallback is None


def test_replay_refuses_another_page(built, tmp_path):
    page, led, b = built
    _, _, rec = ask(b, "Retire aquaculture", use_llm=False, audit_dir=tmp_path)
    other = PageBundle.load(page, led, allow_pending=True)
    other.page_sha256 = "x" * 64
    with pytest.raises(audit.ReplayMismatch):
        audit.replay(rec["id"], other, tmp_path)


def test_the_api_key_is_never_logged(built, tmp_path, monkeypatch):
    _, _, b = built
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test-SECRET-123")
    ask(b, "Retire aquaculture", use_llm=False, audit_dir=tmp_path)
    for f in tmp_path.glob("*.jsonl"):
        assert "SECRET-123" not in f.read_text(encoding="utf-8")
        for line in f.read_text(encoding="utf-8").splitlines():
            json.loads(line)
