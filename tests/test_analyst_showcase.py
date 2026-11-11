"""The shell's question library: every example answers cleanly, every try-a-refusal question is
refused by the fixed rules with the LLM off, and the shell's markup matches its script."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from hydrophysics_analyst import eval as E
from hydrophysics_analyst import presets as PR
from hydrophysics_analyst import showcase, testing
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.validate import validate

STATIC = Path(showcase.__file__).with_name("static")


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("sc")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_library_example_answers_cleanly(b, lang):
    exs = showcase.examples(b)
    assert len(exs) >= 20
    for ex in exs:
        assert ex["label"][lang] and ex["group"][lang] and ex["gid"], ex
        a = PR.run(b, ex["preset"], ex["params"], lang)
        assert a.clarify is None and a.refusal is None, ex
        assert validate(a, b) == [], ex


def test_library_covers_every_preset_the_old_chips_did(b):
    assert {e["preset"] for e in showcase.examples(b)} >= {e["preset"] for e in PR.examples(b)}


def test_groups_are_declared_and_nonempty(b):
    lib = showcase.library(b)
    ids = [g["id"] for g in lib["groups"]]
    assert len(ids) == len(set(ids))
    used = {e["gid"] for e in showcase.examples(b)} | {r["gid"] for r in lib["refusals"]}
    assert used <= set(ids)
    assert set(ids) - used == set(), "a declared group with no questions"
    for g in lib["groups"]:
        assert g["title"]["en"] and g["title"]["zh"]
        if g["desc"] or g["id"] in ("internals", "refusals"):   # these two must explain themselves
            assert g["desc"]["en"] and g["desc"]["zh"], g["id"]


def test_status_legend_quotes_the_ledger(b):
    st = showcase.library(b)["statuses"]
    assert [s["code"] for s in st] == ["S", "D", "N", "L", "W"]
    for s in st:
        assert s["en"] == b.ledger["status_codes"][s["code"]] and s["zh"]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_try_a_refusal_questions_are_refused_without_the_llm(b, lang, tmp_path):
    from hydrophysics_analyst.server import App
    app = App(b, use_llm=False, audit_dir=tmp_path)
    for r in showcase.library(b)["refusals"]:
        v = app.ask({"question": r["question"][lang], "session": "t"}, "t")
        assert v["preset"] == "refuse" and v["refusal"], (r["question"][lang], v["preset"], v["clarify"])


def test_meta_serves_the_library(b, tmp_path):
    from hydrophysics_analyst.server import App
    meta = App(b, use_llm=False, audit_dir=tmp_path).meta()
    assert meta["examples"] == showcase.examples(b)
    assert meta["library"]["groups"] and meta["library"]["refusals"] and meta["library"]["statuses"]


def test_library_is_display_only_and_not_route_code():
    assert "hydrophysics_analyst/showcase.py" not in E.ROUTE_CODE
    assert not any(p.startswith("hydrophysics_analyst/static/") for p in E.ROUTE_CODE)


def test_shell_script_only_uses_ids_the_markup_has():
    html = (STATIC / "shell.html").read_text(encoding="utf-8")
    js = (STATIC / "shell.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r'\$\("([A-Za-z][\w-]*)"\)', js))
    assert used and used <= ids, sorted(used - ids)
