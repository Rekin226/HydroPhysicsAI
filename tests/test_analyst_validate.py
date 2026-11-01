"""The validator rejects a red-team corpus and passes every preset render (0 false rejects)."""
from __future__ import annotations

import dataclasses as dc
import datetime as dt

import pytest

from hydrophysics_analyst import presets as PR
from hydrophysics_analyst import testing
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.render import Span
from hydrophysics_analyst.validate import validate

IRR30 = [{"class": "irrigation", "pct": 30}]


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("v")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


@pytest.fixture(scope="module")
def good(b):
    return PR.run(b, "rail_internals", {"cuts": IRR30, "start": 2026, "year": 2032}, "en")


def _with(a, extra=(), replace=None):
    spans = list(a.spans)
    if replace:
        i, s = replace
        spans[i] = s
    return dc.replace(a, spans=tuple(spans) + tuple(extra))


def _rules(a, b, **kw):
    return {v.rule for v in validate(a, b, reproduce=False, **kw)}


def test_preset_grid_has_no_false_rejects(b):
    n = 0
    for lang in ("en", "zh"):
        for st in (2026, 2030):
            for cuts in ([], IRR30, [{"class": "aquaculture", "pct": 100}]):
                for pr, p in (("policy_head", {"cuts": cuts, "start": st, "year": 2032}),
                              ("start_compare", {"cuts": cuts, "year": 2032}),
                              ("rail_internals", {"cuts": cuts, "start": st, "year": 2032}),
                              ("township_internals", {"cuts": cuts, "start": st, "year": 2032, "town": 0}),
                              ("solved_compare", {"year": 2030}),
                              ("lever_rank", {"year": 2032, "by": "gwh"})):
                    a = PR.run(b, pr, p, lang)
                    assert validate(a, b) == [], (pr, p, lang)
                    n += 1
        for r in PR.REFUSALS:
            assert validate(PR.run(b, "refuse", {"reason": r}, lang), b) == []
    assert n >= 60


def test_red_team(b, good):
    cases = {
        "numbers": [Span("frame", "Head rises 0.4 m."), Span("frame", "上升０.４公尺"),
                    Span("label", "−0.29 m")],
        "banned": [Span("frame", "This protects the rail."), Span("snippet", "pre-registered", key="tW_r"),
                   Span("snippet", "x", key="hl_yun"), Span("frame", "這對高鐵是安全的")],
        "changhua": [Span("frame", "Cut Changhua aquaculture 30% as a lever.")],
    }
    for rule, spans in cases.items():
        for s in spans:
            assert rule in _rules(_with(good, [s]), b), (rule, s)
    i = next(i for i, s in enumerate(good.spans) if s.kind == "value")
    bad = dc.replace(good.spans[i], text="9.99")
    assert "numbers" in _rules(_with(good, replace=(i, bad)), b)
    wh = dc.replace(good.spans[i], cls="withheld")
    assert "withheld" in _rules(_with(good, replace=(i, wh)), b)
    no_label = dc.replace(good, spans=tuple(s for s in good.spans if s.key != "rail_note")
                          + (dc.replace(good.spans[i], key="rail_base", text=next(
                              v.text for r in good.results for v in r.values if v.key == "rail_base"),
                              cls="internal"),))
    assert "labels" in _rules(no_label, b)
    assert "citations" in _rules(dc.replace(good, citations=good.citations + (("C16", "S"),)), b)


def test_stale_d_claim_needs_the_banner(b, good):
    late = dt.date(2028, 2, 1)
    assert "dates" in _rules(good, b, today=late)
    assert "dates" not in _rules(dc.replace(good, banners=("banner_stale_wording",)), b, today=late)


def test_zh_echo_with_a_pooled_town_passes_the_changhua_rule(b):
    a = PR.run(b, "township_internals",
               {"cuts": [{"class": "aquaculture", "pct": 30}], "start": 2026, "year": 2032, "town": 0}, "zh")
    assert "changhua" not in _rules(a, b)


def test_reproduction(b, good):
    assert validate(good, b) == []
    changed = dc.replace(good, params=good.params | {"year": 2031})
    assert "reproduce" in {v.rule for v in validate(changed, b)}
