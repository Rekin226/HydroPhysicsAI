"""Presets: page sentences only, the right claims, the right refusals; strings.json hygiene."""
from __future__ import annotations

import json
import re

import pytest

from hydrophysics_analyst import presets as PR
from hydrophysics_analyst import render as R
from hydrophysics_analyst import testing
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.validate import VERDICT

AQ100 = [{"class": "aquaculture", "pct": 100}]
IRR30 = [{"class": "irrigation", "pct": 30}]


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("p")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


def test_strings_json_is_complete_digit_free_and_verdict_free():
    st = json.loads(R.STRINGS.read_text(encoding="utf-8"))
    assert set(st["en"]) == set(st["zh"])
    for lang in ("en", "zh"):
        for k, v in st[lang].items():
            assert not re.search(r"\d", v), (lang, k)
            for w in VERDICT[lang]:
                assert w.lower() not in v.lower(), (lang, k, w)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_policy_head_cites_the_head_and_cost_claims(b, lang):
    a = PR.run(b, "policy_head", {"cuts": IRR30, "start": 2026, "year": 2032}, lang)
    assert dict(a.citations).keys() >= {"C01", "C08"}
    assert any(s.key == "hl_pol" for s in a.spans) and any(s.key == "tW_r" for s in a.spans)
    assert a.deep_link.startswith("#p=0.7,1,1,1,1,1&s=2026&y=2032")


def test_values_only_from_results(b):
    a = PR.run(b, "solved_compare", {"year": 2032}, "en")
    keys = {v.key for r in a.results for v in r.values}
    assert {s.key for s in a.spans if s.kind == "value"} <= keys


def test_lever_rank_is_labelled_internal_with_the_limit(b):
    a = PR.run(b, "lever_rank", {"year": 2032, "by": "head"}, "en")
    assert a.spans[0].text == "Model internals"
    assert any(s.key == "rank_limit" for s in a.spans) and any(s.key == "ev_decision" for s in a.spans)
    assert len(a.table) == 7


def test_pooled_township_carries_the_spillover_label(b):
    for lang, lab in (("en", "not a Changhua lever"), ("zh", "並非彰化槓桿")):
        a = PR.run(b, "township_internals", {"cuts": AQ100, "start": 2026, "year": 2032, "town": 0}, lang)
        assert lab in a.text()


def test_rail_never_gives_a_verdict(b):
    a = PR.run(b, "rail_internals", {"cuts": IRR30, "start": 2026, "year": 2032}, "en")
    assert any(s.key == "rail_note" for s in a.spans)
    for s in a.spans:
        if s.kind not in ("snippet", "value"):
            assert not any(w in s.text.lower() for w in VERDICT["en"])


def test_refusals_and_clarifications(b):
    a = PR.run(b, "refuse", {"reason": "subsidence_size"}, "en")
    assert a.refusal == "subsidence_size" and ("C03", "W") in a.citations and "l=dsubs" in a.deep_link
    a = PR.run(b, "policy_head", {"cuts": [{"class": "irrigation", "pct": 33}], "start": 2026, "year": 2032}, "en")
    assert a.clarify["reason"] == "out_of_grid"
    a = PR.run(b, "policy_head", {"cuts": IRR30, "start": 2026, "year": 2040}, "en")
    assert a.refusal == "beyond_horizon"
    a = PR.run(b, "township_internals", {"cuts": AQ100, "start": 2026, "year": 2032, "town": "x"}, "en")
    assert a.clarify["reason"] == "ambiguous_town"


def test_cheapest_is_off_until_ruled(b, monkeypatch):
    monkeypatch.delenv("ANALYST_ENABLE_CHEAPEST", raising=False)
    a = PR.run(b, "cheapest_for_head", {"target_m": 0.3, "start": 2026, "year": 2032}, "en")
    assert a.refusal == "disabled"
    monkeypatch.setenv("ANALYST_ENABLE_CHEAPEST", "1")
    a = PR.run(b, "cheapest_for_head", {"target_m": 0.3, "start": 2026, "year": 2032}, "en")
    assert a.refusal is None and a.spans[0].key == "frame_cheapest"


def test_schemas_use_only_the_structured_outputs_subset(b):
    def walk(x):
        if isinstance(x, dict):
            assert not {"oneOf", "minimum", "maximum", "multipleOf"} & set(x)
            if x.get("type") == "object":
                assert x.get("additionalProperties") is False
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(PR.schemas(b))
